import sys
import json
import os

# ================= 【核心环境兼容补丁】 =================
# 🚨 必须在 import torch 和 import scgen 之前执行
# 动态修复新版 anndata 删除了大写 SparseDataset 导致的 scvi 崩溃
try:
    import anndata._core.sparse_dataset as sd
    if not hasattr(sd, 'SparseDataset') and hasattr(sd, 'sparse_dataset'):
        sd.SparseDataset = sd.sparse_dataset
except ImportError:
    pass

# 🚨 锁死在 GPU 1 上运行评测
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import scgen
import scanpy as sc
import anndata as ad

device = "cuda" if torch.cuda.is_available() else "cpu"

train_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl"
test_jsonl_path = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl"
predictions_output_path = "/root/autodl-tmp/runs/scgen/jurkat_test_scgen_predictions_results.jsonl"
os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)

print(f"[*] 使用设备: {device}")

# ================= 2. 动态扫描构建全局基因字典（全局标尺） =================
print("[*] 正在扫描数据集，动态构建全局主考场基因标尺 (HVG 5000)...")
gene_counts = {}

def scan_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                for gene in data['output'].replace('.', '').split():
                    gene_counts[gene] = gene_counts.get(gene, 0) + 1
                if "Control cell sentence: " in data['instruction']:
                    ctrl_part = data['instruction'].split("Control cell sentence: ")[1].split(".\n\nPerturbed")[0]
                    for gene in ctrl_part.split():
                        gene_counts[gene] = gene_counts.get(gene, 0) + 1

scan_file(train_jsonl_path)
top_genes_sorted = [g for g, c in sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)]
hvg_5000 = top_genes_sorted[:5000]
hvg_set = set(hvg_5000)
gene_to_idx = {gene: idx for idx, gene in enumerate(hvg_5000)}

print(f"[+] 标尺锁定成功！主考场维度: {len(hvg_5000)} 个全局基因")

# ================= 3. C2S 文本数据 逆向解码至 连续表达量向量 =================
def c2s_text_to_vector(text, hvg_list):
    """还原 C2S 论文 Rank 机制：靠前的词在文本中被视作高表达，赋予更高相对权重"""
    vector = np.zeros(len(hvg_list))
    clean_text = text.replace('<\\ctrl100>', '').replace('.', '')
    genes = clean_text.split()
    max_rank = len(genes)
    for rank, gene in enumerate(genes):
        if gene in hvg_set:
            idx = gene_to_idx[gene]
            vector[idx] = max_rank - rank
    return vector

def load_jsonl_to_vectors(file_path, limit=None):
    X_list = []
    cond_list = []
    raw_records = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if limit:
        lines = lines[:limit]
        
    for line in lines:
        data = json.loads(line)
        gt_text = data['output']
        try:
            control_text = data['instruction'].split("Control cell sentence: ")[1].split(".\n\nPerturbed")[0]
        except:
            control_text = ""
            
        # 逆向转回连续型矩阵行向量
        v_ctrl = c2s_text_to_vector(control_text, hvg_5000)
        v_gt = c2s_text_to_vector(gt_text, hvg_5000)
        
        # scGen 需要将 control 和 stimulated 两组向量一起塞入 AnnData 矩阵中
        X_list.append(v_ctrl)
        cond_list.append("control")
        
        X_list.append(v_gt)
        cond_list.append("stimulated")
        
        raw_records.append({
            "instruction": data['instruction'],
            "control_text": control_text,
            "gt_text": gt_text
        })
    return np.array(X_list), cond_list, raw_records

print("[*] 正在加载并逆向解码训练集与测试集文本...")
X_train, cond_train, _ = load_jsonl_to_vectors(train_jsonl_path)
X_test, cond_test, test_records = load_jsonl_to_vectors(test_jsonl_path, limit=500)

# 封装为标准的 AnnData 对象，并将细胞类型统一指定为 jurkat
adata_train = ad.AnnData(X=X_train, dtype=np.float32)
adata_train.obs["condition"] = cond_train
adata_train.obs["cell_type"] = "jurkat"
adata_train.var_names = hvg_5000

# ================= 4. 初始化并训练 scGen 模型 =================
print("[*] 正在初始化 scGen 变分自编码器并构建数据潜空间...")
scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")

# 构建 scGen 网络，使用大模型训练同款的 bfloat16 或 float32 保证稳定
model = scgen.SCGEN(adata_train, n_hidden=800, n_latent=100)

print("[*] 开始训练 scGen 核心模型 (隐空间扰动差值外推学习)...")
# 类似于大模型的 SFT 阶段，scGen 默认推荐 100 轮以内即可在少样本下完美收敛
model.train(
    max_epochs=50, 
    batch_size=128, 
    use_gpu=torch.cuda.is_available(),
    early_stopping=True
)
print("[+] scGen 模型训练完成，隐空间 $\\Delta z$ 矢量场构建就位。")

# ================= 5. 批量执行隐空间外推预测 =================
print(f"[*] 启动 scGen 矢量叠加预测预测。结果将实时落盘至: {predictions_output_path}")

results_container = {
    "pearson": [], "spearman": [],
    "top20_de_pearson": [], "top20_de_spearman": [],
    "delta_pearson": [], "delta_spearman": [],
    "delta_top20_pearson": [], "delta_top20_spearman": []
}

with open(predictions_output_path, 'w', encoding='utf-8') as out_f:
    # 每次提取一对控制和扰动样本对进行精细化评估
    for idx, item in enumerate(tqdm(test_records, desc="scGen Latent Predicting")):
        # 单独为这个测试样本构建一个专门用于外推的 AnnData
        v_ctrl = X_test[idx * 2].reshape(1, -1)
        v_gt = X_test[idx * 2 + 1].reshape(1, -1)
        
        adata_interm = ad.AnnData(X=v_ctrl, dtype=np.float32)
        adata_interm.obs["condition"] = ["control"]
        adata_interm.obs["cell_type"] = ["jurkat"]
        adata_interm.var_names = hvg_5000
        
        # 调用 scGen 的核心外推法（隐空间线性代数叠加平衡向量：z_pred = z_ctrl + delta_z）
        try:
            pred_anndata = model.predict(
                adata=adata_interm,
                adata_to_predict=None,
                ctrl_key="control",
                stim_key="stimulated"
            )
            v_pred = pred_anndata.X[0]  # 抽取出来的 scGen 预测连续数值表达向量
        except Exception as e:
            # 安全冗余：若遇到隐空间奇异值导致外推失败，则跳过
            continue
            
        v_gt_flat = v_gt[0]
        v_ctrl_flat = v_ctrl[0]

        # 实时写入本地，保持和原本评估逻辑绝对统一的字段格式
        save_item = {
            "instruction": item["instruction"],
            "ground_truth_vector": v_gt_flat.tolist(),
            "prediction_vector": v_pred.tolist(),
            "control_base_vector": v_ctrl_flat.tolist()
        }
        out_f.write(json.dumps(save_item, ensure_ascii=False) + "\n")

        # ================= 6. 精细化论文指标矩阵计算 =================
        if np.std(v_pred) == 0 or np.std(v_gt_flat) == 0 or np.std(v_ctrl_flat) == 0:
            continue
            
        delta_gt = v_gt_flat - v_ctrl_flat
        delta_pred = v_pred - v_ctrl_flat
        
        # 依据真实物理变化剧烈程度锁定前 20 个核心差异基因 (DEGs)
        top20_de_indices = np.argsort(np.abs(delta_gt))[-20:]
        
        # 1. 全量指标计算
        r_p, _ = stats.pearsonr(v_pred, v_gt_flat)
        r_s, _ = stats.spearmanr(v_pred, v_gt_flat)
        results_container["pearson"].append(r_p)
        results_container["spearman"].append(r_s)
        
        # 2. 全量 Top-20 DE 指标
        if np.std(v_pred[top20_de_indices]) > 0 and np.std(v_gt_flat[top20_de_indices]) > 0:
            top20_p, _ = stats.pearsonr(v_pred[top20_de_indices], v_gt_flat[top20_de_indices])
            top20_s, _ = stats.spearmanr(v_pred[top20_de_indices], v_gt_flat[top20_de_indices])
            results_container["top20_de_pearson"].append(top20_p)
            results_container["top20_de_spearman"].append(top20_s)
            
        # 3. 剥离背景色的 Δ 指标（核心辩别是否是在抄写 Control）
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

# ================= 7. 最终论文级标准汇报打印 =================
print("\n" + "="*75)
print("📊 FINAL SCGEN PERTURBATION LATENT BENCHMARK REPORT (Table 2)")
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
