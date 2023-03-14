# PyTorch DANS (Link Prediction)

PyTorch implementation of DANS (Diversified and Adaptive Negative Sampling
on Knowledge Graph). The code is sparsely optimized with torch_geometric library, which is builded based on PyTorch.

## Evironment Setting
This code is lastly tested with:
* pytorch==1.11.0
* torchvision==0.12.0
* torchaudio==0.11.0 
* cudatoolkit=11.3 -c pytorch
* pytorch-sparse -c pyg

## Data
We provide three datasets: WN18RR, NELL-995 and UMLS.

**The format of input training data**
Train/Validation/Test 
*Each line: source_node relation target_node

Entities & Relations Dictionary
*Each line: ID Name

## Basic Usage
python main.py
