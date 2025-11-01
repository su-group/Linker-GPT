import argparse
import json
import math
import os
import re
import sys
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import moses
from tqdm import tqdm
from molgpt.utils import check_novelty, sample, canonic_smiles, get_mol
from molgpt.model import GPT, GPTConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, required=True, help="path of model weights")
    parser.add_argument('--csv_name', type=str, default='generated.csv', help="output CSV name")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gen_size', type=int, default=1000)
    parser.add_argument('--props', nargs="+", default=[], help="properties for conditioning")
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--lstm', action='store_true', default=False)
    parser.add_argument('--lstm_layers', type=int, default=3)
    parser.add_argument('--data_path', type=str, default='data/QM9_with_scaffold_split.csv')
    parser.add_argument('--scaffold', action='store_true', default=False)
    parser.add_argument('--prop_condition', nargs="+", default=None)
    parser.add_argument('--vocab_dir', type=str, default='vocab')
    args = parser.parse_args()

    context = "C"
    checkpoint = torch.load(args.model_weight, map_location='cpu')
    original_config_dict = checkpoint['model_config']
    stoi = checkpoint['stoi']
    itos = {int(k): v for k, v in checkpoint['itos'].items()}

    target_num_props = len(args.props)
    new_config_dict = original_config_dict.copy()
    new_config_dict['num_props'] = target_num_props
    mconf = GPTConfig(**new_config_dict)
    model = GPT(mconf)

    state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to('cuda')

    data = pd.read_csv(args.data_path)
    data = data.dropna().reset_index(drop=True)
    data.columns = data.columns.str.lower()

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    gen_iter = math.ceil(args.gen_size / args.batch_size)
    all_dfs = []
    count = 0

    if not args.prop_condition:
        molecules = []
        count += 1
        for _ in tqdm(range(gen_iter)):
            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(
                args.batch_size, 1).to('cuda')
            y = sample(model, x, mconf.block_size, temperature=1, sample=True, top_k=None, prop=None, scaffold=None)
            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol]).replace('<', '')
                mol = get_mol(completion)
                if mol:
                    molecules.append(mol)
        results = pd.DataFrame([{'molecule': m, 'smiles': Chem.MolToSmiles(m)} for m in molecules])
        unique_smiles = list(set(canonic_smiles(s) for s in results['smiles']))
        novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))
        results['qed'] = results['molecule'].apply(QED.qed)
        results['sas'] = results['molecule'].apply(sascorer.calculateScore)
        results['logp'] = results['molecule'].apply(Crippen.MolLogP)
        results['tpsa'] = results['molecule'].apply(CalcTPSA)
        results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
        results['unique'] = np.round(len(unique_smiles) / len(results), 3)
        results['novelty'] = np.round(novel_ratio / 100, 3)
        all_dfs.append(results)
    elif args.prop_condition:
        for c in args.prop_condition:
            molecules = []
            count += 1
            for _ in tqdm(range(gen_iter)):
                x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(
                    args.batch_size, 1).to('cuda')
                if len(args.props) == 1:
                    p = torch.tensor([[float(c)]]).repeat(args.batch_size, 1).to('cuda')
                else:
                    p = torch.tensor([list(map(float, c))]).repeat(args.batch_size, 1).to('cuda')
                y = sample(model, x, mconf.block_size, temperature=1, sample=True, top_k=None, prop=p, scaffold=None)
                for gen_mol in y:
                    completion = ''.join([itos[int(i)] for i in gen_mol]).replace('<', '')
                    mol = get_mol(completion)
                    if mol:
                        molecules.append(mol)
            results = pd.DataFrame([{'molecule': m, 'smiles': Chem.MolToSmiles(m)} for m in molecules])
            unique_smiles = list(set(canonic_smiles(s) for s in results['smiles']))
            novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))
            results['condition'] = str(c)
            results['qed'] = results['molecule'].apply(QED.qed)
            results['sas'] = results['molecule'].apply(sascorer.calculateScore)
            results['logp'] = results['molecule'].apply(Crippen.MolLogP)
            results['tpsa'] = results['molecule'].apply(CalcTPSA)
            results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
            results['unique'] = np.round(len(unique_smiles) / len(results), 3)
            results['novelty'] = np.round(novel_ratio / 100, 3)
            all_dfs.append(results)

    results = pd.concat(all_dfs, ignore_index=True)
    results.to_csv(args.csv_name, index=False)