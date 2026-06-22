import json
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Academic Style Configuration =================
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# File Paths
ours_jsonl = "/root/autodl-tmp/runs/c2s/legacy/jurkat_ours_results_epoch5.jsonl"
gemma_jsonl = "/root/autodl-tmp/runs/c2s/legacy/jurkat_test_gemma_predictions_result_5percent_100.jsonl"
# output_pdf = "./fig3_model_comparison_scatter.pdf"
output_png = "/root/autodl-tmp/runs/figures/fig3_model_comparison_scatter.png"
os.makedirs(os.path.dirname(output_png), exist_ok=True)

# ================= 1. Helper Function to Parse First Sample =================
def load_first_sample_ranks(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                # Capture the very first available sample
                return item
    return None

ours_sample = load_first_sample_ranks(ours_jsonl)
gemma_sample = load_first_sample_ranks(gemma_jsonl)

# ================= 2. Text to Rank Converters =================
def get_rank_metrics(sample_dict):
    # Standardizing clean strings
    gt_clean = sample_dict["ground_truth"].replace('<\ctrl100>', '').replace('.', '')
    pred_clean = sample_dict["prediction"].replace('<\ctrl100>', '').replace('.', '')
    
    gt_genes = gt_clean.split()
    pred_genes = pred_clean.split()
    
    # 建立预测词的 Rank 字典
    pred_max = len(pred_genes)
    r_pred = {gene: pred_max - idx for idx, gene in enumerate(pred_genes)}
    
    # 严格遍历 Ground Truth 里的全量 200 个词
    gt_max = len(gt_genes)
    x_gt, y_pred = [], []
    
    for idx, gene in enumerate(gt_genes):
        gt_rank_score = gt_max - idx # 200 down to 1
        pred_rank_score = r_pred.get(gene, 0) # 未预测出来的基因赋予零分位置
        
        x_gt.append(gt_rank_score)
        y_pred.append(pred_rank_score)
        
    x_gt = np.array(x_gt)
    y_pred = np.array(y_pred)
    
    # Calculate exact correlations for this specific single cell text profile
    pearson_r, _ = stats.pearsonr(x_gt, y_pred)
    spearman_r, _ = stats.spearmanr(x_gt, y_pred)
    
    return x_gt, y_pred, pearson_r, spearman_r

# Process both datasets
x_gt_gemma, y_pred_gemma, p_gemma, s_gemma = get_rank_metrics(gemma_sample)
x_gt_ours, y_pred_ours, p_ours, s_ours = get_rank_metrics(ours_sample)

# ================= 3. Plotting 1x2 Subplots =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

# Common settings for beautiful charts
for ax in [ax1, ax2]:
    ax.plot([0, 205], [0, 205], color="#94a3b8", linestyle="--", linewidth=1.5, alpha=0.7, label="Ideal Line ($y=x$)")
    ax.set_xlim(-10, 210)
    ax.set_ylim(-10, 210)
    ax.set_xlabel("Ground Truth Gene Rank", fontsize=11, fontweight='bold', labelpad=6)
    ax.grid(True, linestyle=":", alpha=0.3, color="#cbd5e1")

# ---------------- Left Panel: C2S Baseline ----------------
ax1.scatter(x_gt_gemma, y_pred_gemma, color="#e67e22", alpha=0.6, s=35, edgecolors='none', label="C2S Preds")
ax1.set_ylabel("Predicted Gene Rank", fontsize=11, fontweight='bold', labelpad=6)
ax1.set_title("C2S-Scale-Gemma-2-2B", fontsize=12, fontweight='bold', pad=10)

# Annotation for Left Model Performance
gemma_text = f"Pearson $R$: {p_gemma:.4f}\nSpearman $R$: {s_gemma:.4f}"
ax1.text(15, 160, gemma_text, fontsize=10.5, fontweight='bold', color="#d35400",
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e2e8f0', boxstyle='round,pad=0.5'))
ax1.legend(loc="lower right", fontsize=9.5)

# ---------------- Right Panel: Our Model ----------------
ax2.scatter(x_gt_ours, y_pred_ours, color="#2b7bba", alpha=0.6, s=35, edgecolors='none', label="Ours Preds")
ax2.set_title("Our Proposed Model", fontsize=12, fontweight='bold', pad=10)

# Annotation for Our Model Performance
ours_text = f"Pearson $R$: {p_ours:.4f}\nSpearman $R$: {s_ours:.4f}"
ax2.text(15, 160, ours_text, fontsize=10.5, fontweight='bold', color="#1f618d",
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e2e8f0', boxstyle='round,pad=0.5'))
ax2.legend(loc="lower right", fontsize=9.5)

# Diagnostic Labels for Generative Model Dropouts
ax1.text(100, -7, "← Missed Genes (Predicted Rank = 0) →", color="#94a3b8", fontsize=8.5, ha="center", style='italic')
ax2.text(100, -7, "← Missed Genes (Predicted Rank = 0) →", color="#94a3b8", fontsize=8.5, ha="center", style='italic')

plt.suptitle("Predicted vs. True Gene Expression Ranks Under RCOR1 Perturbation", fontsize=13, fontweight='bold', y=0.98)
sns.despine(trim=False)
plt.tight_layout()

# Save figures
# plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
plt.savefig(output_png, bbox_inches='tight', dpi=300)

print(f"[+] 1x2 comparison figures successfully generated and saved to PDF and PNG!")
