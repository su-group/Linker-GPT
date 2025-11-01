# build_vocab_from_csvs.py
import pandas as pd
import re
import json
import argparse
import os

def build_vocab_from_csvs(csv_paths, smiles_col='smiles', scaffold_col='scaffold_smiles', output_dir='.'):
    """
    从多个 CSV 文件构建统一的 SMILES token 词汇表，并强制包含固定 token。
    """
    # 🔑 新增：固定必须包含的 tokens（即使数据中没有）
    fixed_tokens = {'<', '>', '[', ']', '#'}  # 你可以按需增删

    # 1. 定义与 molgpt 一致的 tokenization 正则表达式
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    all_tokens = set()
    total_smiles = 0

    for csv_path in csv_paths:
        print(f"Processing {csv_path}...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()

        smiles_candidates = [col for col in df.columns if 'smile' in col]
        if not smiles_candidates:
            raise ValueError(f"No SMILES column found in {csv_path}. Looked for columns containing 'smile'.")
        actual_smiles_col = smiles_candidates[0]
        print(f"  -> Using SMILES column: '{actual_smiles_col}'")

        smiles_series = df[actual_smiles_col].dropna().astype(str)
        total_smiles += len(smiles_series)

        for smi in smiles_series:
            tokens = regex.findall(smi)
            all_tokens.update(tokens)

        if scaffold_col and scaffold_col.lower() in df.columns:
            scaffold_series = df[scaffold_col.lower()].dropna().astype(str)
            for scaf in scaffold_series:
                tokens = regex.findall(scaf)
                all_tokens.update(tokens)
            print(f"  -> Also processed scaffold column: '{scaffold_col}'")

    # 🔑 关键修改：合并固定 token
    all_tokens = all_tokens.union(fixed_tokens)
    print(f"Added fixed tokens: {sorted(fixed_tokens)}")

    # 2. 构建词汇表（排序以确保可复现）
    chars = sorted(list(all_tokens))  # 排序保证 stoi 稳定
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 3. 保存
    os.makedirs(output_dir, exist_ok=True)
    stoi_path = os.path.join(output_dir, 'stoi.json')
    itos_path = os.path.join(output_dir, 'itos.json')

    with open(stoi_path, 'w', encoding='utf-8') as f:
        json.dump(stoi, f, indent=2)
    with open(itos_path, 'w', encoding='utf-8') as f:
        json.dump(itos, f, indent=2)

    print(f"\n✅ Vocabulary built successfully!")
    print(f"   Total SMILES processed: {total_smiles}")
    print(f"   Vocabulary size: {len(chars)} (including {len(fixed_tokens)} fixed tokens)")
    print(f"   stoi saved to: {stoi_path}")
    print(f"   itos saved to: {itos_path}")
    print(f"\nSample tokens: {list(chars)[:15]} ...")

    return stoi, itos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build vocabulary from multiple CSV files containing SMILES.")
    parser.add_argument('--csv_files', nargs='+', required=True,
                        help="List of CSV file paths (e.g., data/QM9.csv data/ZINC.csv)")
    parser.add_argument('--smiles_col', type=str, default='smiles',
                        help="Name of the SMILES column (case-insensitive, default: 'smiles')")
    parser.add_argument('--scaffold_col', type=str, default='scaffold_smiles',
                        help="Name of the scaffold SMILES column (optional; set to '' to skip)")
    parser.add_argument('--output_dir', type=str, default='.',
                        help="Directory to save stoi.json and itos.json (default: current dir)")

    args = parser.parse_args()

    scaffold_col = args.scaffold_col if args.scaffold_col.strip() != '' else None

    build_vocab_from_csvs(
        csv_paths=args.csv_files,
        smiles_col=args.smiles_col,
        scaffold_col=scaffold_col,
        output_dir=args.output_dir
    )