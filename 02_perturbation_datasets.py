# Python built-in libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"

import pickle
import random
from datetime import datetime
from collections import Counter, defaultdict

# Third-party libraries
from datasets import Dataset
import numpy as np
import torch
from transformers import TrainingArguments, AutoModelForCausalLM
from tqdm import tqdm

# Single-cell libraries
import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.prompt_formatter import get_cell_sentence_str, PromptFormatter


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)


# Replace this with the actual path to your dataset, if using a custom dataset
# DATA_PATH = "/home/sr2464/scratch/C2S_API_Testing/Data/jurkat.h5ad"
DATA_PATH = "/home/shl003/work/ChatPathway_dev/Perturbation Response Prediction/CRISPR_GSE264667_Data/GSE264667_jurkat_raw_singlecell_01.h5ad"
adata = anndata.read_h5ad(DATA_PATH)

# print(adata)
# print(adata.obs.head())
# adata.obs['batch_var'] = adata.obs['gem_group'].apply(lambda x: f"jurkat{x}")
target_gene_counter = Counter(adata.obs['gene'])
# print(len(target_gene_counter))
# print(target_gene_counter.most_common(20))
# print(adata.var.head())
# print(adata.X.data[:10])
print(adata.X.max())
# target_gene_counter.most_common(20)