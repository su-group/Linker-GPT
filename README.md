# Design of Antibody-Drug Conjugate Linkers with Molecular Generators and Reinforcement Learning
English | [中文版](README_zh.md) 
## ABSTRACT

The stability and release profile of antibody-drug conjugates (ADCs) are significantly influenced by the chemical linkers that connect the antibody and the cytotoxic drug. However, the limited diversity of available linkers leads to the repeated use of a small fraction of these in approved ADCs, highlighting the need for novel linker design. In this study, we trained an attention-based model, **Linker-GPT**, to design the linker portion of ADCs. The model consists of two stages: transfer learning and reinforcement learning.  

In the transfer learning stage, the pre-trained model is fine-tuned using an integrated dataset of linkers, achieving generated molecule validity and novelty scores of **0.894** and **0.997**, respectively. In the reinforcement learning stage, the objective is to generate molecules with good synthesizability and favorable drug-like properties. This ultimately increases the proportion of molecules with ideal properties to **98.7%**. The model can effectively be used for the exploration and screening of novel ADC linkers.

## Environment Setup

### 1. Install Dependencies via `requirements.txt`

Create and activate a Python 3.12+ virtual environment, then install core dependencies:

```bash
pip install -r requirements.txt
```

### 2. Install MOSES

This project uses [MOSES](https://github.com/molecularsets/moses) for molecular evaluation metrics (e.g., validity, novelty). Install it as follows:

```bash
git lfs install
git clone https://github.com/molecularsets/moses.git
python setup.py install
```

> **Note for PyTorch ≥ 2.0 users**:  
> MOSES contains legacy code incompatible with newer Python/RDKit versions. Please apply the following patch to `moses/metrics/SA_Score/sascorer.py`:
>
> - **Line 6**: Comment out or remove  
>   ```python
>   # from rdkit.six import iteritems
>   ```
> - **Line 63**: Replace  
>   ```python
>   for bitId, v in iteritems(fps):
>   ```
>   with  
>   ```python
>   for bitId, v in fps.items():
>   ```

This ensures compatibility with modern Python dictionaries.

## Data and Models

We provide the datasets used, the pre-trained Linker-GPT model, and the fine-tuned model after transfer learning.

- **Pre-training data**: Download from the following sources:
  - [ChEMBL](https://www.ebi.ac.uk/chembl/)
  - [ZINC](https://zinc.docking.org/)
  - [QM9](https://figshare.com/articles/dataset/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978905)

- **Fine-tuning dataset**: Located in the `data/` folder.

- **Pre-trained models**: Stored in `model/pretrain/`.

- **Fine-tuned models**: Stored in `model/fine-tuning/`.

### Pre-training Data Format

The pre-training data must be in CSV format with the following columns:

```
smiles,source,scaffold_smiles
CCC1=C(C(=NO1)C(=O)NC1CC1)C(=O)N1CCC(C1)NC(=O)C1CC1,train,O=C(NC1CC1)C1=C(C(=NO1)C(=O)N1CCC(C1)NC(=O)C1CC1)C
...
```

Where:
- `smiles`: SMILES representation of the full molecule.
- `source`: Data split (`train`/`val`/`test`).
- `scaffold_smiles`: SMILES of the molecular scaffold.

## Getting Started

Run the pipeline in sequence:

1. **Pre-training**
   ```bash
   python 1_pretrain_Linker-GPT.py --run_name <name> --data_path <pretrain.csv>
   ```

2. **Generation (for evaluation)**
   ```bash
   python 2_gen_Linker-GPT.py --gen_size 1000 --csv_name output.csv
   ```

3. **Fine-tuning (Transfer Learning)**
   ```bash
   python 3_fine_tuning_Linker-GPT.py --run_name <name> --data_path <finetune.csv> --model_path <pretrain.pt>
   ```

4. **Reinforcement Learning**
   ```bash
   python 4_RL_Linker-GPT.py --max_epochs 50 --batch_size 64 --model_path <finetune.pt>
   ```

## License

This project is licensed under the MIT License.