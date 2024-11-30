import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from molgpt import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,default='',
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=50,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)

    args = parser.parse_args()

    set_seed(42)

    wandb.init(project="linker-gpt", name=args.run_name)

    data = pd.read_csv('')
   
    data = data.dropna(axis=0).reset_index(drop=True)
   
    data.columns = data.columns.str.lower()

    train_data = data[data['source'] == 'train'].reset_index(
            drop=True)   # 'split' instead of 'source' in moses

    # train_data = train_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    val_data = data[data['source'] == 'val'].reset_index(
            drop=True)   

    # val_data = val_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    # prop = train_data[['qed']]
    # vprop = val_data[['qed']]

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

    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in vsmiles]

    scaffold = [i + str('<')*(scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str('<')*(scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in vscaffold]
#     whole_string = ['#', '%10', '%11', '%12', '%13', '%14', '%15', '%16', '%17', '%18', '%19', '%20', '%21', '%22', '%23', '%24', '%25',
#  '(', ')', '-', '.', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S',
#  '[11C@@H]', '[11CH2]', '[11CH3]', '[11CH]', '[11C]', '[11cH]', '[11c]', '[123I]', '[124I]', '[125I]', '[127I]', '[131I]', '[135I]',
#  '[13CH2]', '[13CH3]', '[13CH]', '[13C]', '[13cH]', '[13c]', '[14C@@H]', '[14C@H]', '[14CH2]', '[14CH3]', '[14CH]', '[14C]', '[14cH]',
#  '[14c]', '[15NH]', '[15nH]', '[15n]', '[17F]', '[18F]', '[18OH]', '[18O]', '[19F]', '[211At]', '[2H]', '[32P]', '[35S]', '[3H]', '[4H]',
# '[73Se]', '[75Se]', '[76Br]', '[Ag-4]', '[Ag]', '[AlH]', '[Al]', '[As+]', '[AsH+]', '[AsH2+]', '[AsH]', '[As]', '[At]', '[B-]', '[B@-]',
# '[B@@-]', '[BH-]', '[BH2-]', '[BH3-]', '[C+]', '[C-]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH+]', '[CH-]', '[CH2]', '[CH]', '[Cl-]',
# '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[N@+]', '[N@@+]', '[N@@]', '[N@]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[NH]', '[Na]',
# '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[P@+]', '[P@@+]', '[P@@]', '[P@]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[S@+]',
# '[S@@+]', '[S@@]', '[S@]', '[SH+]', '[SH2]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si@]', '[SiH2]', '[SiH]', '[Si]',
# '[TeH]', '[Te]', '[Zn+2]', '[Zn-2]', '[Zn]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]',
# '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', '[te+]', '[te]', '\\', 'b', 'c', 'n', 'o', 'p', 's']

    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold, scaffold_maxlen= scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold, scaffold_maxlen= scaffold_max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,  # args.num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len,
                        lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=10, ckpt_path=f'', block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos)

    trainer.train(wandb)

