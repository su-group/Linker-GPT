from utils import check_novelty, sample, canonic_smiles
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
from molgpt import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_mol
import re
import moses
import json
from rdkit.Chem import RDConfig
import json

import os
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer

from rdkit.Chem.rdMolDescriptors import CalcTPSA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, help="path of model weights", default='',required=False)
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format",
                        default='pretrain',required=False)
    parser.add_argument('--batch_size', type=int, default=128, help="batch size", required=False)
    parser.add_argument('--gen_size', type=int, default=, help="number of times to generate from a batch",
                        required=False)
    parser.add_argument('--vocab_size', type=int, default=355, help="number of layers",
                        required=False)  #sca max len
    parser.add_argument('--block_size', type=int, default=369, help="number of layers",
                        required=False)  #smiles max len
    parser.add_argument('--props', nargs="+", default=[], help="properties to be used for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--lstm_layers', type=int, default=3, help="number of layers in lstm", required=False)

    args = parser.parse_args()

    context = "C"

    data = pd.read_csv()
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    smiles = data[data['source'] != 'test']['smiles']
    scaf = data[data['source'] != 'test']['scaffold_smiles']

    # scaffold = data[data['split']!='test_scaffolds']['scaffold_smiles']
    # lens = [len(i.strip()) for i in scaffold.values]
    # scaffold_max_len = max(lens)
    # ws={0: '#', 1: '%10', 2: '%11', 3: '%12', 4: '(', 5: ')', 6: '-', 7: '/', 8: '1', 9: '2', 10: '3', 11: '4', 12: '5',
    #     13: '6', 14: '7', 15: '8', 16: '9', 17: '<', 18: '=', 19: 'B', 20: 'Br', 21: 'C', 22: 'Cl', 23: 'F', 24: 'I', 25: 'N',
    #     26: 'O', 27: 'P', 28: 'S', 29: '[B-]', 30: '[BH-]', 31: '[BH2-]', 32: '[BH3-]', 33: '[B]', 34: '[C+]', 35: '[C-]', 36: '[C@@H]',
    #     37: '[C@H]', 38: '[CH+]', 39: '[CH-]', 40: '[CH2+]', 41: '[CH2]', 42: '[CH]', 43: '[F+]', 44: '[H]', 45: '[I+]', 46: '[IH2]',
    #     47: '[IH]', 48: '[N+]', 49: '[N-]', 50: '[NH+]', 51: '[NH-]', 52: '[NH2+]', 53: '[NH3+]', 54: '[N]', 55: '[O+]', 56: '[O-]',
    #     57: '[OH+]', 58: '[O]', 59: '[P+]', 60: '[PH+]', 61: '[PH2+]', 62: '[PH]', 63: '[S+]', 64: '[S-]', 65: '[SH+]', 66: '[SH]',
    #     67: '[Se+]', 68: '[SeH+]', 69: '[SeH]', 70: '[Se]', 71: '[Si-]', 72: '[SiH-]', 73: '[SiH2]', 74: '[SiH]', 75: '[Si]',
    #     76: '[c+]', 77: '[c-]', 78: '[cH+]', 79: '[cH-]', 80: '[n+]', 81: '[n-]', 82: '[nH+]', 83: '[nH]', 84: '[o+]', 85: '[s+]',
    #     86: '[sH+]', 87: '[se+]', 88: '[se]', 89: '\\', 90: 'c', 91: 'n', 92: 'o', 93: 'p', 94: 's'}

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    lens = [len(regex.findall(i)) for i in smiles]
    max_len = max(lens)
    smiles = [ i + str('<')*(max_len - len(regex.findall(i))) for i in smiles]

    lens = [len(regex.findall(i)) for i in scaf]
    scaffold_max_len = max(lens)

    scaf = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in scaf]

    scaffold_max_len = 355

    content = ' '.join(smiles + scaf)
    chars = sorted(list(set(regex.findall(content))))
    print(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    print(stoi)
    with open(f'', 'w') as f:
             json.dump(stoi, f)

    #stoi = json.load(open(f'data/gs.json', 'r'))
    #print(stoi)
    itos = { i:ch for i,ch in enumerate(chars) }
    #itos = {i: ch for ch, i in stoi.items()}

    print(itos)
    print(len(itos))
    num_props = len(args.props)
    mconf = GPTConfig(args.vocab_size, args.block_size, num_props=num_props,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold,
                      scaffold_maxlen=scaffold_max_len,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)


    model= torch.load()
    model.to('cuda')
    print('Model loaded')

    gen_iter = math.ceil(args.gen_size / args.batch_size)
    # gen_iter = 2
    scaf_condition = None

    # if args.scaffold:
    # scaf_condition = ['O=C(C=CC1=O)N1CC2CCC(C(ON3C(CCC3=O)=O)=O)CC2',
    #                     'O=C(CCCCCN1C(C=CC1=O)=O)O',
    #                     'CC([C@H](NC(CCCCCN1C(C=CC1=O)=O)=O)C(N[C@H](C(NC2=CC=C(CO)C=C2)=O)CCCNC(N)=O)=O)C',
    #                   'O=C(NCC1CCC(CC1)C(NCCN2C(C=CC2=O)=O)=O)CCCCN3C(C=CC3=O)=O']

    # scaf_condition = [i + str('<') * (scaffold_max_len - len(regex.findall(i))) for i in scaf_condition]

    all_dfs = []
    all_metrics = []
  
    count = 0
    if  scaf_condition is None:
        molecules = []
        count += 1
        for i in tqdm(range(gen_iter)):
            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(
                args.batch_size, 1).to('cuda')
            p = None
            # p = torch.tensor([[c]]).repeat(args.batch_size, 1).to('cuda')   # for single condition
            # p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to('cuda')    # for multiple conditions
            # sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
            sca = None
            y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p,
                       scaffold=sca)  # 0.7 for guacamol
            # print(y)

            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol])
                completion = completion.replace('<', '')
                # gen_smiles.append(completion)
                mol = get_mol(completion)
                if mol:
                    molecules.append(mol)

        "Valid molecules % = {}".format(len(molecules))

        mol_dict = []

        for i in molecules:
            mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})
        results = pd.DataFrame(mol_dict)

        canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        unique_smiles = list(set(canon_smiles))
        if 'moses' in args.data_name:
            novel_ratio = check_novelty(unique_smiles, set(
                data[data['split'] == 'train']['smiles']))  # replace 'source' with 'split' for moses
        else:
            novel_ratio = check_novelty(unique_smiles, set(
                data[data['source'] == 'train']['smiles']))  # replace 'source' with 'split' for moses

        print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
        print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
        print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

        results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
        results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
        results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
        results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
        # results['temperature'] = temp
        results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
        results['unique'] = np.round(len(unique_smiles) / len(results), 3)
        results['novelty'] = np.round(novel_ratio / 100, 3)
        all_dfs.append(results)


    elif (prop_condition is not None) and (scaf_condition is None):
        count = 0
        for c in prop_condition:
            molecules = []
            count += 1
            for i in tqdm(range(gen_iter)):
                x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(
                    args.batch_size, 1).to('cuda')
                p = None
                if len(args.props) == 1:
                    p = torch.tensor([[c]]).repeat(args.batch_size, 1).to('cuda')  # for single condition
                else:
                    p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to('cuda')  # for multiple conditions
                sca = None
                y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p,
                           scaffold=sca)  # 0.7 for guacamol
                for gen_mol in y:
                    completion = ''.join([itos[int(i)] for i in gen_mol])
                    completion = completion.replace('<', '')
                    # gen_smiles.append(completion)
                    mol = get_mol(completion)
                    if mol:
                        molecules.append(mol)

            "Valid molecules % = {}".format(len(molecules))

            mol_dict = []

            for i in molecules:
                mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})

            results = pd.DataFrame(mol_dict)
            canon_smiles = [canonic_smiles(s) for s in results['smiles']]
            unique_smiles = list(set(canon_smiles))
            if 'moses' in args.data_name:
                novel_ratio = check_novelty(unique_smiles, set(
                    data[data['split'] == 'train']['smiles']))  # replace 'source' with 'split' for moses
            else:
                novel_ratio = check_novelty(unique_smiles, set(
                    data[data['source'] == 'train']['smiles']))  # replace 'source' with 'split' for moses

            print(f'Condition: {c}')
            print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
            print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
            print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

            if len(args.props) == 1:
                results['condition'] = c
            elif len(args.props) == 2:
                results['condition'] = str((c[0], c[1]))
            else:
                results['condition'] = str((c[0], c[1], c[2]))

            results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
            results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
            results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
            results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
            # results['temperature'] = temp
            results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
            results['unique'] = np.round(len(unique_smiles) / len(results), 3)
            results['novelty'] = np.round(novel_ratio / 100, 3)
            all_dfs.append(results)


    elif  scaf_condition is not None:
        count = 0
        for j in scaf_condition:
            molecules = []
            count += 1
            for i in tqdm(range(gen_iter)):
                x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(
                    args.batch_size, 1).to('cuda')
                p = None
                sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None, ...].repeat(
                    args.batch_size, 1).to('cuda')
                y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p,
                           scaffold=sca)  # 0.7 for guacamol
                for gen_mol in y:
                    completion = ''.join([itos[int(i)] for i in gen_mol])
                    completion = completion.replace('<', '')
                    # gen_smiles.append(completion)
                    mol = get_mol(completion)
                    if mol:
                        molecules.append(mol)

            "Valid molecules % = {}".format(len(molecules))

            mol_dict = []

            for i in molecules:
                mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})

            results = pd.DataFrame(mol_dict)

            canon_smiles = [canonic_smiles(s) for s in results['smiles']]
            unique_smiles = list(set(canon_smiles))
            if 'moses' in args.data_name:
                novel_ratio = check_novelty(unique_smiles, set(
                    data[data['split'] == 'train']['smiles']))  # replace 'source' with 'split' for moses
            else:
                novel_ratio = check_novelty(unique_smiles, set(
                    data[data['source'] == 'train']['smiles']))  # replace 'source' with 'split' for moses

            print(f'Scaffold: {j}')
            print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
            print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
            print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

            results['scaffold_cond'] = j
            results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
            results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
            results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
            results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
            # results['temperature'] = temp
            results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
            results['unique'] = np.round(len(unique_smiles) / len(results), 3)
            results['novelty'] = np.round(novel_ratio / 100, 3)
            all_dfs.append(results)

                print(f'Condition: {c}')
                print(f'Scaffold: {j}')
                print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
                print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
                print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

                if len(args.props) == 1:
                    results['condition'] = c
                elif len(args.props) == 2:
                    results['condition'] = str((c[0], c[1]))
                else:
                    results['condition'] = str((c[0], c[1], c[2]))

                results['scaffold_cond'] = j
                results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
                results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
                results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
                results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
                # results['temperature'] = temp
                results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
                results['unique'] = np.round(len(unique_smiles) / len(results), 3)
                results['novelty'] = np.round(novel_ratio / 100, 3)
                all_dfs.append(results)

    results = pd.concat(all_dfs)
    results.to_csv( args.csv_name, index=False)

    unique_smiles = list(set(results['smiles']))
    canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    unique_smiles = list(set(canon_smiles))
 
    novel_ratio = check_novelty(unique_smiles, set(
    data[data['source'] == 'train']['smiles']))  

    print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter * count), 3))
    print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
    print('Novelty ratio: ', np.round(novel_ratio / 100, 3))
