import pandas as pd
import argparse
from utils import set_seed
import numpy as np
from utils import samplee,qed_func,sa_func,ring_func
import wandb
from time import sleep
from tqdm import tqdm
import torch
from rdkit import Chem
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from utils import Variable
from utils import get_mol
from molgpt import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
from utils import gen
import sys
from rdkit.Chem import RDConfig
import json

import os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer

from rdkit.Chem.rdMolDescriptors import CalcTPSA
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--num_props', type=int, default=0, help="number of properties to use for condition",
                        required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=60,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    parser.add_argument('--block_size', type=int, default=369, help="number of layers",
                        required=False)
    args = parser.parse_args()


    set_seed(42)

    data = pd.read_csv()

    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    train_data = data[data['source'] == 'train'].reset_index(
        drop=True)  

    # train_data = train_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    val_data = data[data['source'] == 'val'].reset_index(
        drop=True) 

    # val_data = val_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']
    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()
    num_props = args.num_props

    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
            for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('Max len: ', max_len)

    lens = [len(regex.findall(i.strip()))
            for i in (list(scaffold.values) + list(vscaffold.values))]
    scaffold_max_len = max(lens)
    print('Scaffold max len: ', scaffold_max_len)

    smiles = [i + str('<') * (max_len - len(regex.findall(i.strip())))
              for i in smiles]
    vsmiles = [i + str('<') * (max_len - len(regex.findall(i.strip())))
               for i in vsmiles]

    scaffold = [i + str('<') * (scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str('<') * (scaffold_max_len -
                                 len(regex.findall(i.strip()))) for i in vscaffold]

    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold,
                                 scaffold_maxlen=scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold,
                                 scaffold_maxlen=scaffold_max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,  
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold,
                      scaffold_maxlen=scaffold_max_len,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = torch.load()
    Prior = model
    Agent = model
    device = 'cuda'
    model.to('cuda')
    print('Model loaded')
    f = open(f"RL/loss", "wb")


    for epoch in range(args.max_epochs):
        seq = gen(args.batch_size)
        # qed = qed_func()(seq)
        # sa = np.array([float(x < 3.0) for x in sa_func()(seq)],
        #               dtype=np.float32)  # to keep all reward components between [0,1]
        # score =  qed + sa
        score = ring_func()(seq)
        scores =   Variable(score).unsqueeze(0)
        losses=[]
        smiles_save=[]
        scoreindex = list(np.where(success_score >3))
        success_smiles = np.array(smiles)[scoreindex]
        smiles_save.extend(success_smiles)
        is_train = 'train'
        model.train()
        data = train_dataset if is_train else valid_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=args.batch_size,
                            num_workers=0)

        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (x, y, p, scaffold) in pbar:
            # place data on the correct device
            x = x.to(device)
            y = y.to(device)
            p = p.to(device)
            scaffold = scaffold.to(device)

            # forward the model
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(True):
                    logits, loss, _ = model(x, y, p, scaffold)
                    logits = logits[:, -1, :]
                    # probs = F.softmax(logits, dim=-1)
                    # print(probs)
                    logprobs = -torch.log_softmax(logits, dim=-1)
                    # print(logprobs.size())
                    # print(scores.size())
                    # print(loss)
                    loss1 = -torch.mean(scores)
                    loss = 10*loss + 0.1*loss1 # collapse all losses if they are scattered on multiple gpus
                    loss = loss.mean()
                    losses.append(loss.item())
            optimizer = model.configure_optimizers(TrainerConfig)
            scaler = GradScaler()
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TrainerConfig.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
        ckpt_path=f'RL.pt'
        print(f"save model {ckpt_path.replace('.pt', f'-{epoch}.pt')}")
        torch.save(model, ckpt_path.replace('.pt', f"-{epoch}.pt"))
        print(
                f"{epoch + 1}/{args.max_epochs}   {round(loss.item(), 4)}        "
            )
        # wandb.log({"epoch": epoch + 1, "train_loss": loss.item(), "val_loss": loss_eval.item()})
        f.write(f"{epoch + 1},{loss}\n".encode("utf-8"))
        sleep(1e-2)
    f.close()
e
e

