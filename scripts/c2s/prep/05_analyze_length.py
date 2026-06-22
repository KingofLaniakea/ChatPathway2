import os
import random
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

#官方的 get_cell_sentence_str 逻辑
def simulate_get_cell_sentence_str(sample, num_genes=200):
    raw_sentence = str(sample.get('cell_sentence', ''))
    # 细胞句子通常是以空格分隔的基因 Token
    words = raw_sentence.split()
    
    # 截取 top_k 的基因数量
    selected_words = words[:num_genes]
    actual_num = len(selected_words)
    
    # 重新拼接回文本
    cell_sentence_str = " ".join(selected_words)
    return cell_sentence_str, str(actual_num)

def run_analysis():
    base_model_path = "/root/autodl-tmp/models/qwen3_8B"
    dataset_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_perturbation_c2s"
    
    TOP_K_GENES = 200
    perturbation_col = 'target_gene'
    control_label = 'non-targeting'

    custom_input_prompt_template = """Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.
Control cell sentence: {control_cell_sentence}.

Perturbed cell sentence:"""

    answer_template = "{perturbed_cell_sentence}."

    print("正在加载 Qwen3 官方分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print("正在加载 26 万全量 C2S 原始数据集...")
    raw_dataset = load_from_disk(dataset_path)
    
    print("\n================================================================")
    print("[正在复刻官方 PerturbationPromptFormatter 动态配对总线]")
    
    # 3. 分离对照组与扰动组
    control_indices = []
    pert_to_indices = defaultdict(list)

    for i, sample in enumerate(tqdm(raw_dataset, desc="第一步：正在划分全球 Control 细胞池")):
        if sample[perturbation_col] == control_label:
            control_indices.append(i)
        else:
            pert_to_indices[sample[perturbation_col]].append(i)
            
    print(f"\n🔹 划分完毕: 成功入库 {len(control_indices)} 个对照组(Control)细胞")
    print(f"🔹 独特扰动靶点数: {len(pert_to_indices)}")
    print("================================================================")

    # 4. 严格按照官方双循环进行 1:1 配对采样，并用 Tokenizer 压测长度
    token_lengths = []
    random.seed(42) # 固定随机种子
    
    # 统计配对后的总样本量
    total_pairs = sum(len(indices) for indices in pert_to_indices.values())
    
    print(f"\n第二步：开始对全量 {total_pairs} 条配对样本进行在线 Token 长度分析...")
    
    for pert_name, perturbed_indices in tqdm(pert_to_indices.items(), desc="遍历扰动系"):
        for perturbed_idx in perturbed_indices:
            # 随机抽样一个对照组细胞合体
            control_idx = random.choice(control_indices)
            
            control_sample = raw_dataset[control_idx]
            perturbed_sample = raw_dataset[perturbed_idx]
            
            # 格式化 Control 细胞
            control_sentence, num_genes_str = simulate_get_cell_sentence_str(
                control_sample, num_genes=TOP_K_GENES
            )
            # 格式化 Perturbed 细胞
            perturbed_sentence, _ = simulate_get_cell_sentence_str(
                perturbed_sample, num_genes=TOP_K_GENES
            )
            
            # 填入官方模板
            model_input_str = custom_input_prompt_template.format(
                num_genes=num_genes_str,
                perturbation_name=pert_name,
                control_cell_sentence=control_sentence
            )
            response_str = answer_template.format(
                perturbed_cell_sentence=perturbed_sentence
            )
            
            # 组装成标准的 Qwen3 训练对对话格式
            full_text = f"<|im_start|>user\n{model_input_str}<|im_end|>\n<|im_start|>assistant\n{response_str}<|im_end|>"
            
            # 计算 Token 长度
            token_lengths.append(len(tokenizer.encode(full_text, add_special_tokens=False)))

    lengths = np.array(token_lengths)

    # 5. 生成精确的统计学报告
    print("\n================================================================")
    print("📈 [ChatPathway Stage 4 官方完全体配对长度报告]")
    print("================================================================")
    print(f"🔹 配对总样本数 (Total Pairs): {len(lengths)}")
    print(f"🔹 最短 Token 长度 (Min)        : {lengths.min()}")
    print(f"🔹 平均 Token 长度 (Mean)       : {lengths.mean():.2f}")
    print(f"🔹 95% 分位数长度  (95th Pct)   : {np.percentile(lengths, 95):.1f}")
    print(f"🔹 99% 分位数长度  (99th Pct)   : {np.percentile(lengths, 99):.1f}")
    print(f"🔹 最长样本长度    (Max)        : {lengths.max()}")
    print("================================================================")
    
    recommended_len = int(np.percentile(lengths, 99) + 32)
    recommended_len = ((recommended_len + 7) // 8) * 8
    print(f"💡 科学建议: 请将微调脚本 train_c2s.py 中的 max_length 修改为: {recommended_len}")
    print("================================================================")

if __name__ == "__main__":
    run_analysis()
