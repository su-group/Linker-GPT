# 基于分子生成与强化学习的抗体药物偶联物（ADC）连接子设计
[English](README.md) | 中文版
## 摘要

抗体药物偶联物（ADC）的稳定性与药物释放特性在很大程度上取决于连接抗体与细胞毒性药物的化学连接子。然而，现有连接子的结构多样性有限，导致已获批 ADC 中反复使用少数几种连接子，凸显了新型连接子设计的迫切需求。本研究开发了一种基于注意力机制的模型——**Linker-GPT**，用于生成 ADC 的连接子部分。该模型包含两个阶段：迁移学习与强化学习。

在迁移学习阶段，模型在整合的连接子数据集上进行微调，生成分子的有效性（validity）和新颖性（novelty）分别达到 **0.894** 和 **0.997**。在强化学习阶段，目标是生成具有良好可合成性与理想类药性质的分子，最终使具备理想性质的分子比例提升至 **98.7%**。该模型可有效用于新型 ADC 连接子的探索与筛选。

## 环境配置

### 1. 通过 `requirements.txt` 安装依赖

创建并激活 Python 3.12+ 虚拟环境，然后运行：

```bash
pip install -r requirements.txt
```


### 2. 安装 MOSES

本项目使用 [MOSES](https://github.com/molecularsets/moses) 计算分子生成指标（如有效性、新颖性）。安装命令：

```bash
git lfs install
git clone https://github.com/molecularsets/moses.git
python setup.py install

```

> **针对 PyTorch ≥ 2.0 用户的兼容性提示**：  
> MOSES 中的部分代码与新版 Python/RDKit 不兼容。请手动修改文件 `moses/metrics/SA_Score/sascorer.py`：
>
> - **第 6 行**：注释或删除  
>   ```python
>   # from rdkit.six import iteritems
>   ```
> - **第 63 行**：将  
>   ```python
>   for bitId, v in iteritems(fps):
>   ```
>   修改为  
>   ```python
>   for bitId, v in fps.items():
>   ```
>
> 此修改可确保代码在现代 Python 环境中正常运行。

## 数据与模型

我们提供了所用数据集、预训练的 Linker-GPT 模型以及迁移学习后的微调模型。

- **预训练数据**：可从以下来源获取：
  - [ChEMBL](https://www.ebi.ac.uk/chembl/)
  - [ZINC](https://zinc.docking.org/)
  - [QM9](https://figshare.com/articles/dataset/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978905)

- **微调数据集**：位于 `data/` 文件夹中。

- **预训练模型**：存放在 `model/pretrain/` 目录。

- **微调模型**：存放在 `model/fine-tuning/` 目录。

### 预训练数据格式

预训练数据需为 CSV 格式，包含以下三列：

```
smiles,source,scaffold_smiles
CCC1=C(C(=NO1)C(=O)NC1CC1)C(=O)N1CCC(C1)NC(=O)C1CC1,train,O=C(NC1CC1)C1=C(C(=NO1)C(=O)N1CCC(C1)NC(=O)C1CC1)C
...
```

字段说明：
- `smiles`：完整分子的 SMILES 表示。
- `source`：数据划分（`train`/`val`/`test`）。
- `scaffold_smiles`：分子骨架的 SMILES 表示。

## 快速开始

按顺序执行以下脚本：

1. **预训练**
   ```bash
   python 1_pretrain_Linker-GPT.py --run_name <名称> --data_path <预训练数据.csv>
   ```

2. **分子生成（用于评估）**
   ```bash
   python 2_gen_Linker-GPT.py --gen_size 1000 --csv_name output.csv
   ```

3. **微调（迁移学习）**
   ```bash
   python 3_fine_tuning_Linker-GPT.py --run_name <名称> --data_path <微调数据.csv> --model_path <预训练模型.pt>
   ```

4. **强化学习**
   ```bash
   python 4_RL_Linker-GPT.py --max_epochs 50 --batch_size 64 --model_path <微调模型.pt>
   ```

## 许可证

本项目采用 MIT 许可证。