import json
import os
import torch
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. Environment & Path Config =================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl"
test_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl"
base_model_path = "/root/autodl-tmp/models/C2S-Scale-Gemma-2-2B"
predictions_output_path = "/root/autodl-tmp/runs/c2s/jurkat_test_gemma_predictions_result_5percent_500.jsonl"
os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)

print(f"[*] Device: {device}")
print(f"[*] Base Model Path: {base_model_path}")

# ================= 2. Build Gene Reference Profile (HVGs) =================
print("[*] Scanning training set to build the global Highly Variable Genes (HVG) profile...")
gene_counts = {}

if not os.path.exists(train_jsonl_path):
    raise FileNotFoundError(f"[-] Training set not found at: {train_jsonl_path}. Cannot build evaluation profile!")

with open(train_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line.strip())
            for gene in data['output'].replace('.', '').split():
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
            if "Control cell sentence: " in data['instruction']:
                ctrl_part = data['instruction'].split("Control cell sentence: ")[1].split(".\n\nPerturbed")[0]
                for gene in ctrl_part.split():
                    gene_counts[gene] = gene_counts.get(gene, 0) + 1

top_genes_sorted = [g for g, c in sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)]
hvg_5000 = top_genes_sorted[:5000]
hvg_set = set(hvg_5000)

print(f"[+] Reference profile locked. Dimension: {len(hvg_5000)} global genes")
if len(hvg_5000) < 5000:
    print(f"[-] Warning: Unique gene count ({len(hvg_5000)}) is less than 5000. Falling back to full available gene mode.")

# ================= 3. Load Model and Tokenizer =================
print("[*] Loading pretrained model architecture and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,   
    attn_implementation="sdpa",   
    trust_remote_code=True
).to(device)
model.eval()
print("[+] Model loaded successfully. Initializing zero-shot batch inference...")

# ================= 4. C2S Text-to-Vector Utility =================
def c2s_text_to_vector(text, hvg_list):
    """Convert expression sequence text into a numerical profile vector based on ranks."""
    vector = np.zeros(len(hvg_list))
    clean_text = text.replace('<\\ctrl100>', '').replace('.', '')
    genes = clean_text.split()
    max_rank = len(genes)
    for rank, gene in enumerate(genes):
        if gene in hvg_set:
            idx = hvg_list.index(gene)
            vector[idx] = max_rank - rank
    return vector

# ================= 5. Batch Inference =================
print(f"[*] Starting inference on test set. Outputs will be saved to: {predictions_output_path}")
evaluated_records = []

if not os.path.exists(test_jsonl_path):
    raise FileNotFoundError(f"[-] Test set not found at: {test_jsonl_path}")

with open(test_jsonl_path, 'r', encoding='utf-8') as f:
    test_lines = [line.strip() for line in f if line.strip()]

# Fast validation mode (subsetting test set)
test_lines = test_lines[:500] 
print(f"[*] [Fast Validation] Subsampling first {len(test_lines)} records for evaluation...")

with open(predictions_output_path, 'w', encoding='utf-8') as out_f:
    for line in tqdm(test_lines, desc="Gemma-2B Generation"):
        data = json.loads(line)
        instruction = data['instruction']
        gt_text = data['output']
        
        try:
            control_text = instruction.split("Control cell sentence: ")[1].split(".\n\nPerturbed")[0]
        except Exception:
            control_text = ""
            
        # Standard raw text stream formatting for Gemma-2B
        input_ids_tensor = tokenizer.encode(instruction, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids_tensor,
                max_new_tokens=1200,          
                do_sample=False,              
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        gen_tokens = outputs[0][len(input_ids_tensor[0]):]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        save_item = {
            "instruction": instruction,
            "ground_truth": gt_text,
            "prediction": pred_text,
            "control_base": control_text
        }
        out_f.write(json.dumps(save_item, ensure_ascii=False) + "\n")
        evaluated_records.append(save_item)

# ================= 6. Metrics Calculation =================
print("\n[*] Inference completed. Calculating evaluation benchmarks (Table 2)...")

results_container = {
    "pearson": [], "spearman": [],
    "top20_de_pearson": [], "top20_de_spearman": [],
    "delta_pearson": [], "delta_spearman": [],
    "delta_top20_pearson": [], "delta_top20_spearman": []
}

for item in evaluated_records:
    v_pred = c2s_text_to_vector(item['prediction'], hvg_5000)
    v_gt = c2s_text_to_vector(item['ground_truth'], hvg_5000)
    v_ctrl = c2s_text_to_vector(item['control_base'], hvg_5000)
    
    if np.std(v_pred) == 0 or np.std(v_gt) == 0 or np.std(v_ctrl) == 0:
        continue
        
    delta_gt = v_gt - v_ctrl
    delta_pred = v_pred - v_ctrl
    
    # Identify top 20 Differentially Expressed Genes (DEGs) based on Ground Truth
    top20_de_indices = np.argsort(np.abs(delta_gt))[-20:]
    
    # 1. Absolute profile metrics
    r_p, _ = stats.pearsonr(v_pred, v_gt)
    r_s, _ = stats.spearmanr(v_pred, v_gt)
    results_container["pearson"].append(r_p)
    results_container["spearman"].append(r_s)
    
    # 2. Absolute profile Top-20 DE metrics
    if np.std(v_pred[top20_de_indices]) > 0 and np.std(v_gt[top20_de_indices]) > 0:
        top20_p, _ = stats.pearsonr(v_pred[top20_de_indices], v_gt[top20_de_indices])
        top20_s, _ = stats.spearmanr(v_pred[top20_de_indices], v_gt[top20_de_indices])
        results_container["top20_de_pearson"].append(top20_p)
        results_container["top20_de_spearman"].append(top20_s)
        
    # 3. Delta profile metrics
    if np.std(delta_pred) > 0 and np.std(delta_gt) > 0:
        dr_p, _ = stats.pearsonr(delta_pred, delta_gt)
        dr_s, _ = stats.spearmanr(delta_pred, delta_gt)
        results_container["delta_pearson"].append(dr_p)
        results_container["delta_spearman"].append(dr_s)
        
    # 4. Delta profile Top-20 DE metrics
    if np.std(delta_pred[top20_de_indices]) > 0 and np.std(delta_gt[top20_de_indices]) > 0:
        d_top20_p, _ = stats.pearsonr(delta_pred[top20_de_indices], delta_gt[top20_de_indices])
        d_top20_s, _ = stats.spearmanr(delta_pred[top20_de_indices], delta_gt[top20_de_indices])
        results_container["delta_top20_pearson"].append(d_top20_p)
        results_container["delta_top20_spearman"].append(d_top20_s)

# ================= 7. Final Report =================
print("\n" + "="*75)
print("="*75)
print(f" Evaluation Space: {len(hvg_5000)} Genes | Valid Test Rows: {len(results_container['pearson'])}")
print("-"*75)
print(f" [Absolute Profile] Pearson R:             {np.mean(results_container['pearson']):.4f} ± {np.std(results_container['pearson']):.4f}")
print(f" [Absolute Profile] Top-20 DE Pearson R:   {np.mean(results_container['top20_de_pearson']):.4f} ± {np.std(results_container['top20_de_pearson']):.4f}")
print(f" [Absolute Profile] Spearman R:            {np.mean(results_container['spearman']):.4f} ± {np.std(results_container['spearman']):.4f}")
print(f" [Absolute Profile] Top-20 DE Spearman R:  {np.mean(results_container['top20_de_spearman']):.4f} ± {np.std(results_container['top20_de_spearman']):.4f}")
print("-"*75)
print(f" [Delta Profile] Pearson R:               {np.mean(results_container['delta_pearson']):.4f} ± {np.std(results_container['delta_pearson']):.4f}")
print(f" [Delta Profile] Top-20 DE Pearson R:     {np.mean(results_container['delta_top20_pearson']):.4f} ± {np.std(results_container['delta_top20_pearson']):.4f}")
print(f" [Delta Profile] Spearman R:              {np.mean(results_container['delta_spearman']):.4f} ± {np.std(results_container['delta_spearman']):.4f}")
print(f" [Delta Profile] Top-20 DE Spearman R:    {np.mean(results_container['delta_top20_spearman']):.4f} ± {np.std(results_container['delta_top20_spearman']):.4f}")
print("="*75)
