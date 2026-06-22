import json
import os
# 🚨 锁死在 GPU 1 上运行评测，绝对不干扰 GPU 0 上正在跑的训练
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 路径与环境配置 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 完美对接你的小数据集路径与官方原生的 Gemma 2B 预训练模型
train_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl"
test_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl"
base_model_path = "/root/autodl-tmp/models/C2S-Scale-Gemma-2-2B"

# 结果保存路径
predictions_output_path = "/root/autodl-tmp/runs/c2s/jurkat_test_gemma_predictions_result_5percent_500.jsonl"
os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)

print(f"[*] 使用设备: {device} (已通过系统环境变量强行锁死在物理 GPU 1)")
print(f"[*] 官方原生的 Gemma-2B 预训练模型路径: {base_model_path}")

# ================= 2. 从当前训练集动态构建全局基因表达量标尺 =================
print("[*] 严格遵循 C2S 论文标准：正在扫描当前小训练集以提取核心高变基因标尺...")
gene_counts = {}

if not os.path.exists(train_jsonl_path):
    raise FileNotFoundError(f"[-] 未能在指定路径找到小训练集文件: {train_jsonl_path}，无法构建评估标尺！")

# 扫描训练集文本，通过统计频次来模拟单细胞的高变基因 (HVG) 筛选
with open(train_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line.strip())
            # 统计受扰动后表达矩阵 (output) 中的基因出现频次
            for gene in data['output'].replace('.', '').split():
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
            # 统计对照组 (instruction) 中的基因出现频次
            if "Control cell sentence: " in data['instruction']:
                ctrl_part = data['instruction'].split("Control cell sentence: ")[1].split(".\n\nPerturbed")[0]
                for gene in ctrl_part.split():
                    gene_counts[gene] = gene_counts.get(gene, 0) + 1

# 按照全局出现频次从大到小排序，精准截取前 5000 个基因
top_genes_sorted = [g for g, c in sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)]
hvg_5000 = top_genes_sorted[:5000]
hvg_set = set(hvg_5000)

print(f"[+] 标尺锁定成功！主考场维度: {len(hvg_5000)} 个全局基因")
if len(hvg_5000) < 5000:
    print(f"⚠️ 提示：小训练集去重后的独特基因总数只有 {len(hvg_5000)} 个，已安全自动转为【当前规模全量基因】评估模式。")

# ================= 3. 加载官方模型与 Tokenizer =================
print("[*] 正在加载官方基础大模型结构与词表...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 纯粹的官方基础大模型，直接加载，不采用 PeftModel 封装
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,   # 与预训练精度对齐，显存减半且速度极快
    attn_implementation="sdpa",   # 开启原生注意力机制加速
    trust_remote_code=True
).to(device)
model.eval()
print("[+] 官方预训练模型完全就位，进入 Zero-shot 批量推理评估状态。")

# ================= 4. C2S 文本转连续数值向量工具 =================
def c2s_text_to_vector(text, hvg_list):
    """依照 C2S 论文 Rank 机制：靠前的词在文本中被视作高表达，赋予更高相对权重"""
    vector = np.zeros(len(hvg_list))
    # 彻底洗掉可能的标志符
    clean_text = text.replace('<\ctrl100>', '').replace('.', '')
    genes = clean_text.split()
    max_rank = len(genes)
    for rank, gene in enumerate(genes):
        if gene in hvg_set:
            idx = hvg_list.index(gene)
            vector[idx] = max_rank - rank
    return vector

# ================= 5. 批量推理与实时落盘 =================
print(f"[*] 开始对小测试集执行原生格式对接推理，结果将保存在: {predictions_output_path}")
evaluated_records = []

if not os.path.exists(test_jsonl_path):
    raise FileNotFoundError(f"[-] 未能在指定路径找到测试集文件: {test_jsonl_path}")

with open(test_jsonl_path, 'r', encoding='utf-8') as f:
    test_lines = [line.strip() for line in f if line.strip()]

test_lines = test_lines[:500] 

print(f"[*] 【快速验证模式】已截取前 {len(test_lines)} 条样本进行评估...")

# 💡 小版本提速建议：如果测试集很大，你可以给 test_lines 加切片，例如 test_lines[:100]，先看 100 条的整体学术水平
with open(predictions_output_path, 'w', encoding='utf-8') as out_f:
    for line in tqdm(test_lines, desc="Gemma-2B Generation"):
        data = json.loads(line)
        instruction = data['instruction']
        gt_text = data['output']
        
        # 解析 Control 基线细胞文本
        try:
            control_text = instruction.split("Control cell sentence: ")[1].split(".\n\nPerturbed")[0]
        except:
            control_text = ""
            
        # 🚨 核心对齐：Gemma-2B 预训练采用纯原生文本流，不需要 Qwen 专属的 `<|im_start|>` 节点符号
        input_ids_tensor = tokenizer.encode(instruction, return_tensors="pt").to(device)
        
        # 语言模型生成基因语句
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids_tensor,
                max_new_tokens=1200,          # 放宽上限，确保未微调模型完美吐出结束符 '.'
                do_sample=False,              # 使用 Greedy Search 保证确定性
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # 截断 Prompt，仅保留新生成的 Generated 基因片段
        gen_tokens = outputs[0][len(input_ids_tensor[0]):]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        # 格式化并实时落盘，防止中途异常导致数据丢失
        save_item = {
            "instruction": instruction,
            "ground_truth": gt_text,
            "prediction": pred_text,
            "control_base": control_text
        }
        out_f.write(json.dumps(save_item, ensure_ascii=False) + "\n")
        evaluated_records.append(save_item)

# ================= 6. 精细化论文指标矩阵计算 =================
print("\n[*] 推理落盘结束。正在计算 Table 2 同款论文评估指标矩阵...")

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
    
    # 基础数值有效性校验，防止分母为 0
    if np.std(v_pred) == 0 or np.std(v_gt) == 0 or np.std(v_ctrl) == 0:
        continue
        
    # 计算扰动引发的改变量 delta 
    delta_gt = v_gt - v_ctrl
    delta_pred = v_pred - v_ctrl
    
    # 依据 Ground Truth 锁定在当前维中变化最剧烈的前 20 个差异表达基因的索引（DEGs）
    top20_de_indices = np.argsort(np.abs(delta_gt))[-20:]
    
    # 1. 全量普通指标
    r_p, _ = stats.pearsonr(v_pred, v_gt)
    r_s, _ = stats.spearmanr(v_pred, v_gt)
    results_container["pearson"].append(r_p)
    results_container["spearman"].append(r_s)
    
    # 2. 全量 Top-20 DE 指标
    if np.std(v_pred[top20_de_indices]) > 0 and np.std(v_gt[top20_de_indices]) > 0:
        top20_p, _ = stats.pearsonr(v_pred[top20_de_indices], v_gt[top20_de_indices])
        top20_s, _ = stats.spearmanr(v_pred[top20_de_indices], v_gt[top20_de_indices])
        results_container["top20_de_pearson"].append(top20_p)
        results_container["top20_de_spearman"].append(top20_s)
        
    # 3. 剥离背景色的 Δ 指标（核心辨别是否是在抄写 Control）
    if np.std(delta_pred) > 0 and np.std(delta_gt) > 0:
        dr_p, _ = stats.pearsonr(delta_pred, delta_gt)
        dr_s, _ = stats.spearmanr(delta_pred, delta_gt)
        results_container["delta_pearson"].append(dr_p)
        results_container["delta_spearman"].append(dr_s)
        
    # 4. Δ 状态下的 Top-20 差异表达核心指标
    if np.std(delta_pred[top20_de_indices]) > 0 and np.std(delta_gt[top20_de_indices]) > 0:
        d_top20_p, _ = stats.pearsonr(delta_pred[top20_de_indices], delta_gt[top20_de_indices])
        d_top20_s, _ = stats.spearmanr(delta_pred[top20_de_indices], delta_gt[top20_de_indices])
        results_container["delta_top20_pearson"].append(d_top20_p)
        results_container["delta_top20_spearman"].append(d_top20_s)

# ================= 7. 标准汇报打印 =================
print("\n" + "="*75)
print("📊 FINAL C2S PERTURBATION ZERO-SHOT BENCHMARK REPORT (Table 2)")
print("="*75)
print(f" Evaluation Space: {len(hvg_5000)} Genes | Valid Test Rows: {len(results_container['pearson'])}")
print("-"*75)
print(f" [全量模式] Pearson R:             {np.mean(results_container['pearson']):.4f} ± {np.std(results_container['pearson']):.4f}")
print(f" [全量模式] Top-20 DE Pearson R:   {np.mean(results_container['top20_de_pearson']):.4f} ± {np.std(results_container['top20_de_pearson']):.4f}")
print(f" [全量模式] Spearman R:            {np.mean(results_container['spearman']):.4f} ± {np.std(results_container['spearman']):.4f}")
print(f" [全量模式] Top-20 DE Spearman R:  {np.mean(results_container['top20_de_spearman']):.4f} ± {np.std(results_container['top20_de_spearman']):.4f}")
print("-"*75)
print(f" [Δ 增量模式] Pearson R:            {np.mean(results_container['delta_pearson']):.4f} ± {np.std(results_container['delta_pearson']):.4f}")
print(f" [Δ 增量模式] Top-20 DE Pearson R:  {np.mean(results_container['delta_top20_pearson']):.4f} ± {np.std(results_container['delta_top20_pearson']):.4f}")
print(f" [Δ 增量模式] Spearman R:           {np.mean(results_container['delta_spearman']):.4f} ± {np.std(results_container['delta_spearman']):.4f}")
print(f" [Δ 增量模式] Top-20 DE Spearman R: {np.mean(results_container['delta_top20_spearman']):.4f} ± {np.std(results_container['delta_top20_spearman']):.4f}")
print("="*75)
