import pandas as pd
import argparse
from molgpt.utils import set_seed
import numpy as np
import wandb
import json
import torch
from torch.utils.data import DataLoader
from molgpt.model import GPT, GPTConfig
from molgpt.trainer import Trainer, TrainerConfig
from molgpt.dataset import SmileDataset
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="", help="name for wandb run")
    parser.add_argument("--data_path", type=str, default="data/QM9.csv", help="path of data csv")
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    parser.add_argument("--scaffold", action="store_true", default=False, help="condition on scaffold")
    parser.add_argument("--lstm", action="store_true", default=False, help="use lstm for transforming scaffold")
    parser.add_argument("--props", nargs="+", default=["a"], help="properties to be used for condition")
    parser.add_argument("--num_props", type=int, default=0, help="number of properties to use for condition")
    parser.add_argument("--n_layer", type=int, default=8, help="number of layers")
    parser.add_argument("--n_head", type=int, default=8, help="number of heads")
    parser.add_argument("--n_embd", type=int, default=256, help="embedding dimension")
    parser.add_argument("--max_epochs", type=int, default=50, help="total epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="learning rate")
    parser.add_argument("--lstm_layers", type=int, default=0, help="number of layers in lstm")
    args = parser.parse_args()

    set_seed(42)
    wandb.init(project="linker-gpt", name=args.run_name)

    data = pd.read_csv(args.data_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    train_data = data[data["source"] == "train"].reset_index(drop=True)
    val_data = data[data["source"] == "val"].reset_index(drop=True)

    smiles = train_data["smiles"]
    vsmiles = val_data["smiles"]
    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()
    num_props = args.num_props

    scaffold = train_data["scaffold_smiles"]
    vscaffold = val_data["scaffold_smiles"]

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    max_len = 400
    scaffold_max_len = 400

    smiles = [i + str("<") * (max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [i + str("<") * (max_len - len(regex.findall(i.strip()))) for i in vsmiles]
    scaffold = [i + str("<") * (scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str("<") * (scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold]

    with open(f"vocab/stoi.json", 'r') as f:
        stoi = json.load(f)
    whole_string = [token for token, idx in sorted(stoi.items(), key=lambda x: x[1])]

    train_dataset = SmileDataset(
        args, smiles, whole_string, max_len, prop=prop, aug_prob=0,
        scaffold=scaffold, scaffold_maxlen=scaffold_max_len
    )
    valid_dataset = SmileDataset(
        args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0,
        scaffold=vscaffold, scaffold_maxlen=scaffold_max_len
    )

    mconf = GPTConfig(
        train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len,
        lstm=args.lstm, lstm_layers=args.lstm_layers
    )
    model = GPT(mconf)

    tconf = TrainerConfig(
        max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
        final_tokens=args.max_epochs * len(train_data) * max_len, num_workers=10,
        ckpt_path=f"models/{args.run_name}.pt", block_size=train_dataset.max_len, generate=False
    )

    trainer = Trainer(
        model, train_dataset, valid_dataset, tconf, train_dataset.stoi, train_dataset.itos
    )
    trainer.train(wandb)