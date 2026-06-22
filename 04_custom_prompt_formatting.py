import os
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

DATA_PATH = "/root/autodl-tmp/GSE264667_jurkat_C2S_format.h5ad"
adata = anndata.read_h5ad(DATA_PATH)

print(adata)
print(adata.obs.head())

target_gene_counter = Counter(adata.obs['target_gene'])
print(len(target_gene_counter))

print(target_gene_counter.most_common(20))

print(adata.var.head())

# We'll keep all relevant columns for our new task
adata_obs_cols_to_keep = adata.obs.columns.tolist()

print(adata_obs_cols_to_keep)

arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata, 
    random_state=SEED, 
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)

print(arrow_ds)


# The input provides the control cell and the perturbation, asking for the perturbed result.
custom_input_prompt_template = """Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.
Control cell sentence: {control_cell_sentence}.

Perturbed cell sentence:"""

# The answer is simply the target cell sentence.
answer_template = "{perturbed_cell_sentence}."

from collections import defaultdict

class PerturbationPromptFormatter(PromptFormatter):
    def __init__(self,
        task_name,
        input_prompt,
        answer_template,
        top_k_genes, 
        perturbation_col='target_gene',
        control_label='non-targeting'
    ):
        """
        Initializes the custom prompt formatter.

        Args:
            task_name (str): The name for this task.
            input_prompt (str): The template for the model's input.
            answer_template (str): The template for the model's expected response.
            top_k_genes (int): The number of top genes to include in the cell sentence.
            perturbation_col (str): The column name in the dataset that contains perturbation info.
            control_label (str): The label used to identify control cells in the perturbation_col.
        """
        super().__init__()
        self.task_name = task_name
        self.input_prompt = input_prompt
        self.answer_template = answer_template
        self.top_k_genes = top_k_genes
        self.perturbation_col = perturbation_col
        self.control_label = control_label
        assert top_k_genes > 0, "'top_k_genes' must be an integer > 0"

    def format_hf_ds(self, hf_ds):
        """
        Custom formatting function for perturbation prediction. This function creates pairs of
        (control, perturbed) cells by sampling from a global pool of control cells.
        """
        model_inputs_list = []
        responses_list = []
        
        # 1. Separate all cells into a global control pool and a dict of perturbed cells
        control_indices = []
        pert_to_indices = defaultdict(list)

        print("Grouping cells by perturbation and identifying global controls...")
        for i, sample in enumerate(hf_ds):
            if sample[self.perturbation_col] == self.control_label:
                control_indices.append(i)
            else:
                pert_to_indices[sample[self.perturbation_col]].append(i)
        
        assert len(control_indices) > 0, "No control cells found. Cannot create pairs."
        print(f"Found {len(control_indices)} control cells.")
        print(f"Found {len(pert_to_indices)} unique perturbations.")

        # 2. Create prompt-response pairs by iterating through perturbed cells
        print("Creating control-perturbed pairs...")
        for pert_name, perturbed_indices in tqdm(pert_to_indices.items()):
            for perturbed_idx in perturbed_indices:
                # Pair each perturbed cell with a random control cell from the global pool
                control_idx = random.choice(control_indices)
                
                control_sample = hf_ds[control_idx]
                perturbed_sample = hf_ds[perturbed_idx]

                # Format control cell sentence
                control_sentence, num_genes_str = get_cell_sentence_str(
                    control_sample,
                    num_genes=self.top_k_genes
                )
                # Format perturbed cell sentence
                perturbed_sentence, _ = get_cell_sentence_str(
                    perturbed_sample,
                    num_genes=self.top_k_genes
                )

                # Format the model input string using the perturbation name
                model_input_str = self.input_prompt.format(
                    num_genes=num_genes_str,
                    perturbation_name=pert_name,
                    control_cell_sentence=control_sentence
                )
                
                # Format the response string
                response_str = self.answer_template.format(
                    perturbed_cell_sentence=perturbed_sentence
                )

                model_inputs_list.append(model_input_str)
                responses_list.append(response_str)

        # Create the final Hugging Face Dataset
        ds_split_dict = {
            "sample_type": [self.task_name] * len(model_inputs_list),
            "model_input": model_inputs_list,
            "response": responses_list,
        }
        ds = Dataset.from_dict(ds_split_dict)
        return ds

task_name = "perturbation_prediction"
prompt_formatter = PerturbationPromptFormatter(
    task_name=task_name,
    input_prompt=custom_input_prompt_template,
    answer_template=answer_template,
    top_k_genes=200 # Using top 200 genes for this example. For real applications, ideal to use all nonzero expressed genes if possible.
)

# Format the dataset
formatted_ds = prompt_formatter.format_hf_ds(arrow_ds)

print(formatted_ds)

# Inspect a formatted sample
print("--- Formatted Sample ---")
print("#----Model input:----#")
print(formatted_ds[0]["model_input"], "\n")
print("#----Response:----#")
print(formatted_ds[0]["response"])

# Save directory for Huggingface dataset
c2s_save_dir = "/root/CRISPR_GSE264667_Data"
c2s_save_name = "jurkat_perturbation_c2s"

csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=arrow_ds,  # Regular cell sentence dataset put here, finetune() function will repeat the formatting with the prompt formatter
    vocabulary=vocabulary,
    save_dir=c2s_save_dir,
    save_name=c2s_save_name,
    dataset_backend="arrow"
)
print(csdata)