import argparse
import os
import re
import sys
from time import sleep
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import RDConfig
from torch.nn import functional as F
from tqdm import tqdm
from molgpt.model import GPT, GPTConfig
from molgpt.utils import (
    set_seed,
    qed_func,
    ring_func,
    sa_func,
    sample,
    load_model_and_vocab
)
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
REGEX = re.compile(PATTERN)

def generate_smiles(model, stoi, itos, batch_size, max_len, device='cuda', 
                   prop=None, scaffold=None, temperature=1.0):
    model.eval()
    with torch.no_grad():
        start_token = stoi['<']
        x = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        y = sample(
            model, x, steps=max_len-1,
            temperature=temperature, sample=True, top_k=50,
            prop=prop, scaffold=scaffold
        )
        seq = []
        for gen_mol in y:
            completion = ''.join([itos.get(int(i), '') for i in gen_mol if int(i) in itos])
            completion = completion.replace('<', '')
            seq.append(completion)
        return seq

def smiles_to_tensor(smiles, stoi, max_len, device='cuda'):
    tokens = REGEX.findall(smiles.strip())
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += ['<'] * (max_len - len(tokens))
    dix = [stoi.get(t, stoi['<']) for t in tokens]
    return torch.tensor(dix, dtype=torch.long, device=device)

def compute_rewards(smiles_list):
    qed_vals = qed_func()(smiles_list)
    sa_vals = sa_func()(smiles_list)
    ring_vals = ring_func()(smiles_list)
    rewards = (
        qed_vals +
        (1.0 - np.clip(sa_vals, 1, 10) / 10.0) -
        0.1 * ring_vals
    )
    return np.clip(rewards, 0.0, 1.0).astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--scaffold', action='store_true', default=False)
    parser.add_argument('--lstm', action='store_true', default=False)
    parser.add_argument('--num_props', type=int, default=0)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--lstm_layers', type=int, default=0)
    parser.add_argument('--block_size', type=int, default=369)
    parser.add_argument('--props', nargs='+', default=['qed'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0)
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--run_name', type=str, default='rl_finetune')
    args = parser.parse_args()

    set_seed(42)
    os.makedirs("RL", exist_ok=True)

    model, config, stoi, itos = load_model_and_vocab(args.model_path, device='cpu')
    Agent = model
    Agent.train()
    Agent.to('cuda')

    from copy import deepcopy
    Prior = deepcopy(Agent)
    Prior.eval()
    for param in Prior.parameters():
        param.requires_grad_(False)

    from molgpt.trainer import TrainerConfig
    tconf = TrainerConfig(learning_rate=args.learning_rate)
    optimizer = Agent.configure_optimizers(tconf)

    f = open(f"RL/{args.run_name}_loss.txt", "w")

    for epoch in range(args.max_epochs):
        seqs = generate_smiles(
            Agent, stoi, itos, 
            batch_size=args.batch_size,
            max_len=args.block_size,
            device='cuda',
            temperature=1.0
        )

        valid_seqs = [s for s in seqs if Chem.MolFromSmiles(s) is not None]
        if len(valid_seqs) == 0:
            continue

        rewards = compute_rewards(valid_seqs)
        rewards = torch.tensor(rewards, device='cuda')

        log_probs_list = []
        for smiles in valid_seqs:
            x = smiles_to_tensor(smiles, stoi, args.block_size, device='cuda')
            x = x.unsqueeze(0)
            logits, _, _ = Agent(x, prop=None, scaffold=None)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, x.unsqueeze(-1)).squeeze(-1)
            seq_log_prob = token_log_probs.sum()
            log_probs_list.append(seq_log_prob)

        if not log_probs_list:
            continue
        log_probs = torch.stack(log_probs_list)

        with torch.no_grad():
            prior_log_probs_list = []
            for smiles in valid_seqs:
                x = smiles_to_tensor(smiles, stoi, args.block_size, device='cuda').unsqueeze(0)
                prior_logits, _, _ = Prior(x, prop=None, scaffold=None)
                prior_log_probs = F.log_softmax(prior_logits, dim=-1)
                token_log_probs = prior_log_probs.gather(2, x.unsqueeze(-1)).squeeze(-1)
                prior_log_probs_list.append(token_log_probs.sum())
            prior_log_probs = torch.stack(prior_log_probs_list)

        rl_loss = -(rewards * log_probs).mean()
        kl_loss = (log_probs - prior_log_probs).mean()
        total_loss = rl_loss + args.kl_coef * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(Agent.parameters(), args.grad_norm_clip)
        optimizer.step()

        avg_reward = rewards.mean().item()
        print(f"Epoch {epoch+1}/{args.max_epochs} | Loss: {total_loss.item():.4f} | Avg Reward: {avg_reward:.4f}")
        f.write(f"{epoch+1},{total_loss.item()},{avg_reward}\n")
        f.flush()

        if (epoch + 1) % 10 == 0:
            ckpt_path = f"RL/{args.run_name}_epoch{epoch+1}.pt"
            torch.save({
                'model_state_dict': Agent.state_dict(),
                'model_config': config.__dict__,
                'stoi': stoi,
                'itos': itos
            }, ckpt_path)

        sleep(1e-2)

    f.close()