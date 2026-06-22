import json
import os
import random
from datasets import load_from_disk
from tqdm import tqdm

# 配置
train_dataset_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_perturbation_c2s"
output_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_qa_datasets.jsonl"
top_k_genes = 200
perturbation_col = 'target_gene'
control_label = 'non-targeting'

# 1. 加载原始数据
raw_dataset = load_from_disk(train_dataset_path)

# 2. 筛选 control 和 perturbation
control_samples = []
pert_samples = []

for sample in raw_dataset:
    cell_sent = " ".join(str(sample.get('cell_sentence', '')).split()[:top_k_genes])
    if not cell_sent.strip():
        continue # 物理过滤掉原本就为空的脏数据
        
    if sample[perturbation_col] == control_label:
        control_samples.append(cell_sent)
    else:
        pert_samples.append({
            "pert_name": sample[perturbation_col],
            "pert_sentence": cell_sent
        })

print(f"Control 细胞数: {len(control_samples)}, 扰动对数: {len(pert_samples)}")

# 3. 配对并写入 JSONL
custom_input_prompt_template = "Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.\nControl cell sentence: {control_cell_sentence}.\n\nPerturbed cell sentence:"

with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    # 为了保证多样性，我们可以为每个扰动样本随机配对 1 个（或多个）Control 细胞
    for idx, pert in enumerate(tqdm(pert_samples, desc="Generating QA Dataset")):
        # 固定随机种子，保证可重复性
        random.seed(idx)
        control_cell_sentence = random.choice(control_samples)
        
        num_genes_str = str(len(control_cell_sentence.split()))
        
        # 组装成大模型通用的结构化 QA 格式
        prompt_text = custom_input_prompt_template.format(
            num_genes=num_genes_str,
            perturbation_name=pert["pert_name"],
            control_cell_sentence=control_cell_sentence
        )
        response_text = f"{pert['pert_sentence']}."
        
        # 检查防空兜底
        if prompt_text.strip() and response_text.strip():
            qa_pair = {
                "instruction": prompt_text,
                "output": response_text
            }
            f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")

print(f"✨ 标准 QA 数据集已成功写入: {output_jsonl_path}")
