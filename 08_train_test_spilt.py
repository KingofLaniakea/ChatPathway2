import json
import os
import random
from datasets import load_from_disk
from tqdm import tqdm

# ================= 配置 =================
train_dataset_path = "/root/CRISPR_GSE264667_Data/jurkat_perturbation_c2s"
output_dir = "/root/CRISPR_GSE264667_Data"
top_k_genes = 200
perturbation_col = 'target_gene'
control_label = 'non-targeting'
test_split_ratio = 0.15  # 15% 的扰动种类作为未见过的测试集（Held-out）

# 1. 加载原始数据
raw_dataset = load_from_disk(train_dataset_path)

# 2. 第一次遍历：筛选并统计所有的独立扰动种类
control_samples = []
pert_samples_dict = {}  # 用字典按扰动名归类样本

print("[*] 正在扫描原始数据集并对齐扰动标签...")
for sample in tqdm(raw_dataset, desc="Scanning dataset"):
    cell_sent = " ".join(str(sample.get('cell_sentence', '')).split()[:top_k_genes])
    if not cell_sent.strip():
        continue  # 过滤脏数据
        
    p_name = sample[perturbation_col]
    
    if p_name == control_label:
        control_samples.append(cell_sent)
    else:
        if p_name not in pert_samples_dict:
            pert_samples_dict[p_name] = []
        pert_samples_dict[p_name].append(cell_sent)

all_pert_names = list(pert_samples_dict.keys())
print(f"\n[+] 扫描完成。Control 细胞数: {len(control_samples)}")
print(f"[+] 总共包含不同的扰动种类数: {len(all_pert_names)}")

# 3. 核心策略：按扰动名字进行 Held-out 划分
random.seed(42)  # 固定种子保证划分可重复
random.shuffle(all_pert_names)

num_test_perts = int(len(all_pert_names) * test_split_ratio)
test_pert_names = set(all_pert_names[:num_test_perts])      # 🚨 Unseen 扰动
train_pert_names = set(all_pert_names[num_test_perts:])    # 🌟 Seen 扰动

print(f"[-] 划分结果：{len(train_pert_names)} 种扰动作为训练 Seen 集")
print(f"[-] 划分结果：{len(test_pert_names)} 种扰动作为测试 Unseen 集 (例如: {list(test_pert_names)[:3]}...)")

# 4. 组装并写入 JSONL
custom_input_prompt_template = "Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.\nControl cell sentence: {control_cell_sentence}.\n\nPerturbed cell sentence:"

train_output_path = os.path.join(output_dir, "jurkat_c2s_train_seen.jsonl")
test_output_path = os.path.join(output_dir, "jurkat_c2s_test_unseen.jsonl")

def generate_qa_pairs(target_pert_names, output_path, global_seed_offset=0):
    """根据指定的扰动集合，配对 Control 并写入文件"""
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        # 遍历目标扰动的种类
        for p_idx, p_name in enumerate(target_pert_names):
            samples_under_this_pert = pert_samples_dict[p_name]
            
            # 遍历该扰动下的每一个细胞样本
            for s_idx, pert_sentence in enumerate(samples_under_this_pert):
                # 确保每个样本对应的随机种子是唯一且可复现的
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

print("\n[*] 正在生成训练集 (Seen Perturbations)...")
train_count = generate_qa_pairs(train_pert_names, train_output_path, global_seed_offset=1234)

print("[*] 正在生成测试集 (Unseen Perturbations)...")
test_count = generate_qa_pairs(test_pert_names, test_output_path, global_seed_offset=5678)

# ================= 总结 =================
print("\n" + "="*50)
print(f"✨ 数据集切分与构建成功！")
print(f"📝 训练集 (Seen) 路径: {train_output_path} | 包含对话对: {train_count} 条")
print(f"📝 测试集 (Unseen) 路径: {test_output_path} | 包含对话对: {test_count} 条")
print("="*50)