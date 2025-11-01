import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import functools
import os
import numpy as np
from tqdm import tqdm


def batch_process_csv(smiles_col='smiles',
                      output_col='scaffold_smiles',
                      id_col=None,
                      split_train_test_eval=True,
                      train_ratio=0.8,
                      test_ratio=0.1,
                      eval_ratio=0.1,
                      seed=42,
                      skip_invalid=True):
    """
    Decorator: Adds batch CSV processing capability to SMILES processing functions,
    with optional train/test/validation split.

    Args:
        skip_invalid (bool): If True, skip rows where scaffold generation fails (default: True).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(input_data, *args, **kwargs):
            if isinstance(input_data, str) and input_data.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(input_data)
                except Exception as e:
                    raise IOError(f"Failed to read CSV file: {e}")

                if smiles_col not in df.columns:
                    raise ValueError(f"Column '{smiles_col}' not found in CSV.")

                cols_to_save = []
                if id_col and id_col in df.columns:
                    cols_to_save.append(id_col)
                cols_to_save.append(smiles_col)

                smiles_list = df[smiles_col].astype(str).fillna('').tolist()

                scaffold_smiles_list = [
                    func(smi.strip()) if smi.strip() else None
                    for smi in tqdm(smiles_list, desc="Processing SMILES", unit="mol")
                ]

                df[output_col] = scaffold_smiles_list

                if skip_invalid:
                    valid_mask = df[output_col].notna()
                    df = df[valid_mask].copy().reset_index(drop=True)
                    if len(df) == 0:
                        raise ValueError("All SMILES are invalid; no valid data to output.")

                if split_train_test_eval:
                    total = train_ratio + test_ratio + eval_ratio
                    if abs(total - 1.0) > 1e-6:
                        raise ValueError("train_ratio + test_ratio + eval_ratio must sum to 1.0")

                    n = len(df)
                    indices = np.arange(n)
                    rng = np.random.default_rng(seed)
                    rng.shuffle(indices)

                    n_train = int(round(n * train_ratio))
                    n_test = int(round(n * test_ratio))

                    train_idx = indices[:n_train]
                    test_idx = indices[n_train:n_train + n_test]
                    eval_idx = indices[n_train + n_test:]

                    split_col = np.full(n, '', dtype=object)
                    split_col[train_idx] = 'train'
                    split_col[test_idx] = 'test'
                    split_col[eval_idx] = 'val'

                    df['source'] = split_col

                base, ext = os.path.splitext(input_data)
                suffix = '_with_scaffold'
                if skip_invalid:
                    suffix += '_valid_only'
                if split_train_test_eval:
                    suffix += '_split'
                output_file = f"{base}{suffix}{ext}"
                df.to_csv(output_file, index=False)
                print(f"Batch processing completed. Processed {len(scaffold_smiles_list)} rows, kept {len(df)} valid rows.")
                print(f"Results saved to: {output_file}")

                return df

            else:
                return func(input_data, *args, **kwargs)

        return wrapper
    return decorator


@batch_process_csv(
    smiles_col='smiles',
    output_col='scaffold_smiles',
    id_col='id',
    split_train_test_eval=True,
    skip_invalid=True
)
def get_scaffold_smiles(smiles):
    """
    Generate Murcko scaffold SMILES for a given molecule.
    Supports single SMILES string or batch CSV processing.
    """
    if not smiles or smiles.lower() in ['nan', 'none', 'null', '']:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold_mol, isomericSmiles=True) if scaffold_mol else None
    except Exception:
        return None


if __name__ == "__main__":
    get_scaffold_smiles("data/ChemBL.csv")
    get_scaffold_smiles("data/Zinc.csv")
    get_scaffold_smiles("data/QM9.csv")