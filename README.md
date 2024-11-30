<h1 id="V9U6j">Design of Antibody-drug Conjugates Linkers with Molecular Generators and Reinforcement Learning</h1>
<h2 id="R0uTF">ABSTRACT</h2>
<font style="color:black;background-color:#FFFFFF;">The stability and release profile of antibody-drug conjugates (ADCs) are significantly influenced by the chemical linkers that connect the antibody and the cytotoxic drug. However, the limited diversity of available linkers leads to the repeated use of a small fraction of these in approved ADCs, highlighting the need for novel linker design. In this study, we trained an attention-based model, Linker-GPT, to design the linker portion of ADCs. The model consists of two parts: transfer learning and reinforcement learning. In the transfer learning part, the pre-trained model is fine-tuned using an integrated dataset of linkers, achieving generated molecule validity and novelty scores of 0.894 and 0.997, respectively. In the reinforcement learning part, the goal is to generate molecules with good synthesizability and favorable drug-like properties. This ultimately increases the proportion of molecules with ideal properties to 98.7%. The model can effectively be used for the exploration and screening of novel ADC linkers.</font>

<h2 id="M5vDQ">Necessary package</h2>
Recommended installation under Linux

```plain
python = 3.7.16
pytorch = 1.13.1
numpy
pandas
wandb
RDKit = 2020.09.1.0
```

<h2 id="KPvBG">Data and Models</h2>
We provide the data used, the trained Linker-GPT pre-trained model, and the fine-tuned model after transfer learning.

Pre-training data can be downloaded from the link below.

[ChEMBL](https://www.ebi.ac.uk/chembl/)

[ZINC](https://zinc12.docking.org/)

[QM9](https://paperswithcode.com/dataset/qm9)

 Linkers dataset for fine-tuning  is placed under the `data` folder

pre-training models are placed under `model/pretrain` folder

Fine-tuning models are placed under `model/fine-tuning` folder

<h2 id="RiLtE">Getting Started</h2>
Users can customize their own tasks by modifying the code.  Users can run the Linker-GPT model by excuting the 1-4 .py files in sequence according to the following script.

`pretraining.py` is used for pre-training models. Pre-training datasets can be replaced by modifying read paths.

```plain
  python 1_pretrain_Linker-GPT.py --run_name{name_for_wandb_run} --data_path{your_pretrain_data}
```

`<font style="color:rgb(44, 44, 54);">generation.py</font>`<font style="color:rgb(44, 44, 54);"> is used for molecule generation and save the generated molecules in CSV format.</font>

```plain
python 2_gen_Linker-GPT.py --gen_size{number_of_times_to_generate_from_a_batch} --csv_path{save_path_for_generate_moleculars}
```

`fine_tuning.py` is used for transfer learning. Fine-tuning datasets can be replaced by modifying read paths.

```plain
python 3_fine_tuning_Linker-GPT.py --run_name{name_for_wandb_run} --data_path{your_fine_tuning_data}
```

`<font style="color:rgb(44, 44, 54);">RL.py</font>`<font style="color:rgb(44, 44, 54);"> is used for reinforcement learning. By default, it uses molecular properties such as QED (Quantitative Estimate of Drug-likeness), SAS (Synthetic Accessibility Score), and the number of rings.</font>

```plain
python 4_RL_Linker-GPT.py --max_epoch{total_epochs} --bach_size 128 --path{path_to_save_agent_model}
```

<h2 id="XdWqS">License</h2>
This project is licensed under the MIT License.

