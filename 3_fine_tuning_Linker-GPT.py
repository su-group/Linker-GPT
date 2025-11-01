import pandas as pd
import argparse
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from molgpt.utils import set_seed
from molgpt.model import GPT, GPTConfig
from molgpt.trainer import Trainer, TrainerConfig
from molgpt.dataset import SmileDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='finetune', help="Name for wandb run")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--scaffold', action='store_true', default=False, help='Condition on scaffold')
    parser.add_argument('--lstm', action='store_true', default=False, help='Use LSTM for scaffold')
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--lstm_layers', type=int, default=0)
    parser.add_argument('--data_path', type=str, required=True, help="Path to CSV data")
    parser.add_argument('--props', nargs='*', default=['qed'], help="Property columns to condition on (e.g., qed)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to pretrained .pt model")
    args = parser.parse_args()

    set_seed(42)
    wandb.init(project="linker-gpt", name=args.run_name, config=args)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    original_config_dict = checkpoint['model_config']
    stoi = checkpoint['stoi']
    itos = {int(k): v for k, v in checkpoint['itos'].items()}

    data = pd.read_csv(args.data_path)
    data.columns = data.columns.str.lower().str.strip()
    data = data.dropna(subset=['smiles']).reset_index(drop=True)

    if 'source' not in data.columns:
        raise ValueError("CSV must contain a 'source' column with values 'train'/'val'")
    train_data = data[data['source'] == 'train'].reset_index(drop=True)
    val_data = data[data['source'] == 'val'].reset_index(drop=True)

    num_props = len(args.props)
    if num_props > 0:
        missing = [p for p in args.props if p not in train_data.columns]
        if missing:
            raise ValueError(f"Missing property columns in CSV: {missing}")
        train_nan = train_data[args.props].isnull().sum()
        val_nan = val_data[args.props].isnull().sum()
        if train_nan.any() or val_nan.any():
            print("Warning: NaN values found in property columns. Dropping affected rows.")
            train_data = train_data.dropna(subset=args.props).reset_index(drop=True)
            val_data = val_data.dropna(subset=args.props).reset_index(drop=True)

    prop_train = train_data[args.props].values.astype(np.float32).tolist() if num_props > 0 else None
    prop_val = val_data[args.props].values.astype(np.float32).tolist() if num_props > 0 else None

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    all_smiles = train_data['smiles'].tolist() + val_data['smiles'].tolist()
    lens = [len(regex.findall(s.strip())) for s in all_smiles]
    max_len = min(max(lens), original_config_dict['block_size'])

    scaffold_max_len = original_config_dict.get('scaffold_maxlen', 0)
    if args.scaffold:
        if 'scaffold_smiles' not in train_data.columns:
            raise ValueError("scaffold_smiles column required when --scaffold is used")
        all_scaffolds = train_data['scaffold_smiles'].tolist() + val_data['scaffold_smiles'].tolist()
        scaf_lens = [len(regex.findall(s.strip())) for s in all_scaffolds]
        scaffold_max_len = min(max(scaf_lens), scaffold_max_len)

    def pad_smiles(smiles_list, length):
        padded = []
        for s in smiles_list:
            tokens = regex.findall(s.strip())
            if len(tokens) > length:
                tokens = tokens[:length]
            else:
                tokens += ['<'] * (length - len(tokens))
            padded.append(''.join(tokens))
        return padded

    train_smiles = pad_smiles(train_data['smiles'].tolist(), max_len)
    val_smiles = pad_smiles(val_data['smiles'].tolist(), max_len)
    train_scaf = pad_smiles(train_data['scaffold_smiles'].tolist(), scaffold_max_len) if args.scaffold else None
    val_scaf = pad_smiles(val_data['scaffold_smiles'].tolist(), scaffold_max_len) if args.scaffold else None

    whole_string = [token for token, idx in sorted(stoi.items(), key=lambda x: x[1])]
    train_dataset = SmileDataset(
        args, train_smiles, whole_string, max_len,
        prop=prop_train, aug_prob=0, scaffold=train_scaf, scaffold_maxlen=scaffold_max_len
    )
    valid_dataset = SmileDataset(
        args, val_smiles, whole_string, max_len,
        prop=prop_val, aug_prob=0, scaffold=val_scaf, scaffold_maxlen=scaffold_max_len
    )

    new_config_dict = original_config_dict.copy()
    new_config_dict['num_props'] = num_props
    mconf = GPTConfig(**new_config_dict)
    model = GPT(mconf)

    model_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in checkpoint['model_state_dict'].items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model.load_state_dict(filtered_state_dict, strict=False)

    tconf = TrainerConfig(
        max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
        final_tokens=args.max_epochs * len(train_data) * max_len, num_workers=0,
        ckpt_path=f"models/{args.run_name}.pt", block_size=max_len, generate=False
    )

    trainer = Trainer(model, train_dataset, valid_dataset, tconf, stoi, itos)
    trainer.train(wandb)

if __name__ == '__main__':
    main()