import json
import os
import torch
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 1. 路径与环境配置 =================
# 锁死在 GPU 1 上运行评测
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
base_model_path = "/root/autodl-tmp/models/qwen3_8B"
lora_checkpoint_path = "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_small_5percent/checkpoint_epoch_5"
train_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl"
test_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl"
predictions_output_path = "/root/autodl-tmp/runs/c2s/jurkat_ours_results_epoch5.jsonl"
os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)

print(f"[*] 使用设备: {device}")
print(f"[*] 基座模型: {base_model_path}")
print(f"[*] LoRA 权重: {lora_checkpoint_path}")

# ================= 2. 构建基因标尺 =================
print("[*] 正在构建全局基因标尺...")
gene_counts = {}
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

# ================= 3. 加载 Qwen + LoRA 模型 =================
print("[*] 正在载入 Qwen 基础大模型与 LoRA 适配器...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载基座
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True
).to(device)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
model.eval()
print("[+] 模型加载完毕，进入评估状态。")

# ================= 4. C2S 文本转向量工具 =================
def c2s_text_to_vector(text, hvg_list):
    vector = np.zeros(len(hvg_list))
    clean_text = text.replace('<\\ctrl100>', '').replace('.', '')
    genes = clean_text.split()
    max_rank = len(genes)
    for rank, gene in enumerate(genes):
        if gene in hvg_set:
            idx = hvg_list.index(gene)
            vector[idx] = max_rank - rank
    return vector

# ================= 5. 批量推理 =================
print(f"[*] 开始推理，结果保存至: {predictions_output_path}")
evaluated_records = []

# with open(test_jsonl_path, 'r', encoding='utf-8') as f:
#     test_lines = [line.strip() for line in f if line.strip()][:500]

with open(test_jsonl_path, 'r', encoding='utf-8') as f:
    test_lines = [line.strip() for line in f if line.strip()]

test_lines = test_lines[:100] 

print(f"[*] 【快速验证模式】已截取前 {len(test_lines)} 条样本进行评估...")

with open(predictions_output_path, 'w', encoding='utf-8') as out_f:
    for line in tqdm(test_lines, desc="Qwen-LoRA Generation"):
        data = json.loads(line)
        instruction = data['instruction']
        gt_text = data['output']
        
        control_text = instruction.split("Control cell sentence: ")[1].split(".\n\nPerturbed")[0]
        
        # ⚠️ 关键点：使用 Qwen 微调时定义的 Chat 模板
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=1200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        gen_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        save_item = {"instruction": instruction, "ground_truth": gt_text, "prediction": gen_text, "control_base": control_text}
        out_f.write(json.dumps(save_item, ensure_ascii=False) + "\n")
        evaluated_records.append(save_item)

# ================= 6. 指标矩阵计算 =================
print("\n[*] 推理结束，开始计算 Table 2 指标...")
results_container = {k: [] for k in ["pearson", "spearman", "top20_de_pearson", "top20_de_spearman", "delta_pearson", "delta_spearman", "delta_top20_pearson", "delta_top20_spearman"]}

for item in evaluated_records:
    v_pred = c2s_text_to_vector(item['prediction'], hvg_5000)
    v_gt = c2s_text_to_vector(item['ground_truth'], hvg_5000)
    v_ctrl = c2s_text_to_vector(item['control_base'], hvg_5000)
    
    if np.std(v_pred) == 0 or np.std(v_gt) == 0 or np.std(v_ctrl) == 0: continue
    
    delta_gt, delta_pred = v_gt - v_ctrl, v_pred - v_ctrl
    top20_indices = np.argsort(np.abs(delta_gt))[-20:]
    
    results_container["pearson"].append(stats.pearsonr(v_pred, v_gt)[0])
    results_container["spearman"].append(stats.spearmanr(v_pred, v_gt)[0])
    
    if np.std(v_pred[top20_indices]) > 0 and np.std(v_gt[top20_indices]) > 0:
        results_container["top20_de_pearson"].append(stats.pearsonr(v_pred[top20_indices], v_gt[top20_indices])[0])
        results_container["top20_de_spearman"].append(stats.spearmanr(v_pred[top20_indices], v_gt[top20_indices])[0])
        
    if np.std(delta_pred) > 0 and np.std(delta_gt) > 0:
        results_container["delta_pearson"].append(stats.pearsonr(delta_pred, delta_gt)[0])
        results_container["delta_spearman"].append(stats.spearmanr(delta_pred, delta_gt)[0])
        
    if np.std(delta_pred[top20_indices]) > 0 and np.std(delta_gt[top20_indices]) > 0:
        results_container["delta_top20_pearson"].append(stats.pearsonr(delta_pred[top20_indices], delta_gt[top20_indices])[0])
        results_container["delta_top20_spearman"].append(stats.spearmanr(delta_pred[top20_indices], delta_gt[top20_indices])[0])

# ================= 7. 最终报告 =================
# print("="*75 + "\n📊 QWEN3-LORA C2S BENCHMARK REPORT\n" + "="*75)
# print(f" [全量] Pearson: {np.mean(results_container['pearson']):.4f} | Top20 DE: {np.mean(results_container['top20_de_pearson']):.4f}")
# print(f" [Δ增量] Pearson: {np.mean(results_container['delta_pearson']):.4f} | Top20 DE: {np.mean(results_container['delta_top20_pearson']):.4f}")
# print("="*75)

print("\n" + "="*75)
print("📊 FINAL C2S PERTURBATION ZERO-SHOT BENCHMARK REPORT (Table 2)")
print("="*75)
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
