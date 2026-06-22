import json
import os
import random
from datasets import load_from_disk
from tqdm import tqdm

# ================= 配置 =================
train_dataset_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_perturbation_c2s"
output_dir = "/root/autodl-tmp/data/CRISPR_GSE264667_Data"
top_k_genes = 200
perturbation_col = 'target_gene'
control_label = 'non-targeting'

# 🌟 核心调整：原先 21w 数据保留 15% 做测试，现在我们只拿 10% 的扰动种类来做小版训练
train_ratio = 0.05
test_ratio = 0.01   # 拿 2% 的种类做配套的测试集（足够评估用）
# 剩下的 ~88% 的扰动种类直接踢出去，不参与训练，用来当后续彻底没见过的测试盲区

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

# 3. 核心切分策略：按扰动名字进行严格的三分法切分
random.seed(42)  # 固定种子保证划分可重复
random.shuffle(all_pert_names)

num_train_perts = int(len(all_pert_names) * train_ratio)
num_test_perts = int(len(all_pert_names) * test_ratio)

# 👑 严格划分出小规模的 Seen 扰动和配套的 Unseen 扰动
train_pert_names = set(all_pert_names[:num_train_perts])
test_pert_names = set(all_pert_names[num_train_perts: num_train_perts + num_test_perts])
excluded_pert_names = set(all_pert_names[num_train_perts + num_test_perts:])  # 彻底被剔除的扰动（完全不参与训练）

print(f"[-] 👑 缩小规模划分结果：")
print(f"    -> 选取 {len(train_pert_names)} 种扰动作为【小训练集 Seen】")
print(f"    -> 选取 {len(test_pert_names)} 种扰动作为【配套测试集 Unseen】")
print(f"    -> 剩余 {len(excluded_pert_names)} 种扰动被踢出，可作为【终极盲区验证集 Excluded】")

# 4. 组装并写入 JSONL
custom_input_prompt_template = "Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.\nControl cell sentence: {control_cell_sentence}.\n\nPerturbed cell sentence:"

# 小版本的输出路径
train_output_path = os.path.join(output_dir, "jurkat_c2s_train_seen_small_5percent.jsonl")
test_output_path = os.path.join(output_dir, "jurkat_c2s_test_unseen_small_5percent.jsonl")
excluded_output_path = os.path.join(output_dir, "jurkat_c2s_test_excluded_5percent.jsonl")

def generate_qa_pairs(target_pert_names, output_path, global_seed_offset=0):
    """根据指定的扰动集合，配对 Control 并写入文件"""
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

print("\n[*] 正在向新文件注入【小训练集】数据...")
train_count = generate_qa_pairs(train_pert_names, train_output_path, global_seed_offset=1234)

print("[*] 正在向新文件注入【小测试集】数据...")
test_count = generate_qa_pairs(test_pert_names, test_output_path, global_seed_offset=5678)

print("[*] 正在生成【未参与训练的终极盲区集】数据（留作备用，可选跑）...")
excluded_count = generate_qa_pairs(excluded_pert_names, excluded_output_path, global_seed_offset=9999)

# ================= 总结 =================
print("\n" + "="*50)
print(f"✨ 小版本快捷数据集切分与精简成功！")
print(f"📝 训练集 (Small) 路径: {train_output_path} | 包含: {train_count} 条 (预计约 2 万条)")
print(f"📝 测试集 (Small) 路径: {test_output_path} | 包含: {test_count} 条")
print(f"📝 盲区测试集路径: {excluded_output_path} | 包含: {excluded_count} 条")
print("="*50)
