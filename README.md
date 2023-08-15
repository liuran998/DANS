# PyTorch DANS (Link Prediction)

PyTorch implementation of DANS (Diversified and Adaptive Negative Sampling
on Knowledge Graph). The code is sparsely optimized with torch_geometric library, which is built based on PyTorch.

## Evironment Setting
This code is lastly tested with:
* pytorch==1.11.0
* torchvision==0.12.0
* torchaudio==0.11.0 
* cudatoolkit=11.3 -c pytorch
* pytorch-sparse -c pyg
* tqdm
* torch_geometric

## Data
We provide three datasets: WN18RR, NELL-995 and UMLS.

### The format of input training data
### Train/Validation/Test 
* Each line: source_node relation target_node

### Entities & Relations Dictionary
* Each line: ID Name

## Basic Usage
python main.py --dataset <dataset_name> --scoring_function <function_number> --node_embed_size <D> --dimension <D> <br /><br />
### As an example:
The following command trains and validates a DistMult model on wn18rr dataset with 100D in node_embed_size & dimension:<br />
``python main.py --dataset wn18rr 
--scoring_function DistMult
--node_embed_size 100 
--dimension 100``
The following command trains and validates a RotatE model on wn18rr dataset with 50D in node_embed_size & dimension:<br />
``python main.py --dataset wn18rr 
--scoring_function RotatE
--node_embed_size 50 
--dimension 50``
