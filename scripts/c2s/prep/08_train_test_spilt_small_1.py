import json
import os
import random
from datasets import load_from_disk
from tqdm import tqdm

# ================= Configuration =================
train_dataset_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_perturbation_c2s"
output_dir = "/root/autodl-tmp/data/CRISPR_GSE264667_Data"
top_k_genes = 200
perturbation_col = 'target_gene'
control_label = 'non-targeting'

# Downsampling strategy: Subsetting by perturbation types rather than total samples
train_ratio = 0.05
test_ratio = 0.01

# 1. Load original dataset
raw_dataset = load_from_disk(train_dataset_path)

# 2. First pass: Scan and group samples by perturbation type
control_samples = []
pert_samples_dict = {}

print("[*] Scanning dataset and aligning perturbation labels...")
for sample in tqdm(raw_dataset, desc="Scanning dataset"):
    cell_sent = " ".join(str(sample.get('cell_sentence', '')).split()[:top_k_genes])
    if not cell_sent.strip():
        continue
        
    p_name = sample[perturbation_col]
    
    if p_name == control_label:
        control_samples.append(cell_sent)
    else:
        if p_name not in pert_samples_dict:
            pert_samples_dict[p_name] = []
        pert_samples_dict[p_name].append(cell_sent)

all_pert_names = list(pert_samples_dict.keys())
print(f"\n[+] Scan completed. Control cells: {len(control_samples)}")
print(f"[+] Total unique perturbation types: {len(all_pert_names)}")

# 3. Split strategy: Splitting dataset strictly by perturbation types
random.seed(42)
random.shuffle(all_pert_names)

num_train_perts = int(len(all_pert_names) * train_ratio)
num_test_perts = int(len(all_pert_names) * test_ratio)

train_pert_names = set(all_pert_names[:num_train_perts])
test_pert_names = set(all_pert_names[num_train_perts: num_train_perts + num_test_perts])
excluded_pert_names = set(all_pert_names[num_train_perts + num_test_perts:])

print(f"[-] Split results:")
print(f"    -> Selected {len(train_pert_names)} perturbations for [Train Seen]")
print(f"    -> Selected {len(test_pert_names)} perturbations for [Test Unseen]")
print(f"    -> Remaining {len(excluded_pert_names)} perturbations for [Excluded Validation]")

# 4. Assemble QA pairs and write to JSONL
custom_input_prompt_template = "Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.\nControl cell sentence: {control_cell_sentence}.\n\nPerturbed cell sentence:"

train_output_path = os.path.join(output_dir, "jurkat_c2s_train_seen_small_5percent.jsonl")
test_output_path = os.path.join(output_dir, "jurkat_c2s_test_unseen_small_5percent.jsonl")
excluded_output_path = os.path.join(output_dir, "jurkat_c2s_test_excluded_5percent.jsonl")

def generate_qa_pairs(target_pert_names, output_path, global_seed_offset=0):
    """Pair control cells with perturbed cells and write to JSONL."""
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for p_idx, p_name in enumerate(target_pert_names):
            samples_under_this_pert = pert_samples_dict[p_name]
            
            for s_idx, pert_sentence in enumerate(samples_under_this_pert):
                random.seed(p_idx * 10000 + s_idx + global_seed_offset)
                control_cell_sentence = random.choice(control_samples)
                
                num_genes_str = str(len(control_cell_sentence.split()))
                prompt_text = custom_input_prompt_template.format(
                    num_genes=num_genes_str,
                    perturbation_name=p_name,
                    control_cell_sentence=control_cell_sentence
                )
                response_text = f"{pert_sentence}."
                
                if prompt_text.strip() and response_text.strip():
                    qa_pair = {
                        "instruction": prompt_text,
                        "output": response_text
                    }
                    f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                    count += 1
    return count

print("\n[*] Generating [Train Seen] dataset...")
train_count = generate_qa_pairs(train_pert_names, train_output_path, global_seed_offset=1234)

print("[*] Generating [Test Unseen] dataset...")
test_count = generate_qa_pairs(test_pert_names, test_output_path, global_seed_offset=5678)

print("[*] Generating [Excluded] dataset...")
excluded_count = generate_qa_pairs(excluded_pert_names, excluded_output_path, global_seed_offset=9999)

# ================= Summary =================
print("\n" + "="*50)
print(f"✨ Dataset splitting and downsampling completed successfully!")
print(f"📝 Train Path (Small): {train_output_path} | Total: {train_count} samples")
print(f"📝 Test Path (Small): {test_output_path} | Total: {test_count} samples")
print(f"📝 Excluded Path: {excluded_output_path} | Total: {excluded_count} samples")
print("="*50)
