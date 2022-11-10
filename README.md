# Multi-modal Protein-Drug Interaction Prediction Via Attention-based Network

A novel attention mechanism-driven deep model for estimating the binding affinity between protein and drug by exploiting multi-modal information including protein sequence, protein structure, and drug fingerprint.

* ✔ **Our pre-trained encoder is at `./files/pretrain/BertModel.pth`.**
* ✔ **Code of fine-tuning phase is released.**

## Table of Contents

- [Multi-modal Protein-Drug Interaction Prediction Via Attention-based Network](#multi-modal-protein-drug-interaction-prediction-via-attention-based-network)
  - [Table of Contents](#table-of-contents)
  - [Environment & Dependencies](#environment--dependencies)
  - [Usage](#usage)
  - [Addition](#addition)



## Environment & Dependencies

* `Python`==3.8.13

* `torch`==1.8.0

```bash
# create new environ
conda create -n pdi_fusionner python=3.8.13

# build dependencies
pip install -r requirements.txt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

1. Data preparation

* Place the processed data file, namely `pafnucy_total_rdkit-smiles-v1.csv`,  to `./files/data`. Notice that this file combines both the protein datasets, i.e., PDBbind, CASF-2013 and Astex datasets, and the drug SMILES dataset. An entry of this file should look like:
   
   |     | PDB-ID | seq        | SMILES     | rdkit_smiles | Affinity-Value | set   |
   |:---:| ------ | ---------- | ---------- | ------------ | -------------- | ----- |
   | 0   | 11gs   | PYTVVYF... | CCC(=O)... | CC[C@@H]...  | 5.82           | train |

* Place the protein contact map that processed from 3D crystal structure data, namely `pdbbind_2016`, to `./files/3d_pdb`

* Update the configurations `config.yaml` to make sure everything is matched
   
   

2. Fine-tuning
   
```python
python finetune_main.py --model_idx=0 --epochs=150 --gpu_start=0 --batch_size=64 \
                        --patience=10 --factor=0.5 --min_lr=1e-5  --tb
```

Specify `model_idx` to choose our proposed model or its ablation variants. More options please refer to `config.yaml` and the argument parser in `finetune_main.py`.


## Addition
**More detailed information will be released and updated soon!**
