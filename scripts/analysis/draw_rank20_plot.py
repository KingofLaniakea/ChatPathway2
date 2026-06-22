import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Academic Style Configuration =================
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# File Paths
ours_jsonl = "/root/autodl-tmp/runs/c2s/legacy/jurkat_ours_results_epoch5.jsonl"
gemma_jsonl = "/root/autodl-tmp/runs/c2s/legacy/jurkat_test_gemma_predictions_result_5percent_100.jsonl"
output_png = "/root/autodl-tmp/runs/figures/fig4_deg_trajectories_grid.png"
os.makedirs(os.path.dirname(output_png), exist_ok=True)

# ================= 1. Load Data =================
def load_first_sample(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                return json.loads(line.strip())
    return None

ours_sample = load_first_sample(ours_jsonl)
gemma_sample = load_first_sample(gemma_jsonl)

# ================= 2. Convert Text to Rank Maps =================
def text_to_rank_dict(text):
    clean_text = text.replace('<\ctrl100>', '').replace('.', '')
    genes = clean_text.split()
    max_rank = len(genes)
    return {gene: max_rank - i for i, gene in enumerate(genes)}

r_ctrl = text_to_rank_dict(ours_sample["control_base"])
r_gt = text_to_rank_dict(ours_sample["ground_truth"])
r_pred_ours = text_to_rank_dict(ours_sample["prediction"])
r_pred_gemma = text_to_rank_dict(gemma_sample["prediction"])

# ================= 3. Identify Top-20 Shifting DEGs (Gold Standard) =================
deg_candidates = []
for g in r_gt.keys():
    gt_score = r_gt[g]
    ctrl_score = r_ctrl.get(g, 0)
    deg_candidates.append({"gene": g, "shift": abs(gt_score - ctrl_score)})

# Sort by absolute biology response shift and pick top 20
deg_candidates = sorted(deg_candidates, key=lambda x: x["shift"])
top20_degs = [d["gene"] for d in deg_candidates[-20:]]

# ================= 4. Plotting 4x5 Facet Grid Maps =================
fig, axes = plt.subplots(4, 5, figsize=(15, 11), sharex=True, sharey=True)
axes = axes.flatten()

x_states = [0, 1, 2]
state_labels = ["Ctrl", "Pred", "True"]

for idx, gene in enumerate(top20_degs):
    ax = axes[idx]
    
    y_ctrl = r_ctrl.get(gene, 0)
    y_gt = r_gt.get(gene, 0)
    y_ours = r_pred_ours.get(gene, 0)
    y_gemma = r_pred_gemma.get(gene, 0)
    
    # ---- Line Drawing (Base Trajectories) ----
    gemma_points = [y_ctrl, y_gemma, y_gt]
    ax.plot(x_states, gemma_points, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.5)
    
    ours_points = [y_ctrl, y_ours, y_gt]
    ax.plot(x_states, ours_points, color="#2b7bba", linestyle="-", linewidth=1.5, alpha=0.6)
    
    # ---- Distinct Node Shifting Representation ----
    # 1. State: Ctrl (Universal Biological Baseline -> Grey Circle)
    ax.scatter(0, y_ctrl, color="#7f8c8d", marker="o", s=45, zorder=5, 
               label="Ctrl Base (Basal State)" if idx == 0 else "")
    
    # 2. State: Pred (Model Modeling Hub -> Distinct Shapes)
    ax.scatter(1, y_gemma, color="#e67e22", marker="d", s=50, zorder=5, 
               label="C2S Prediction" if idx == 0 else "")
    ax.scatter(1, y_ours, color="#2b7bba", marker="s", s=45, zorder=5, 
               label="Our Prediction" if idx == 0 else "")
    
    # 3. State: True (Gold Standard Observation -> Black Star)
    ax.scatter(2, y_gt, color="#2c3e50", marker="*", s=75, zorder=5, 
               label="Ground Truth (Observed)" if idx == 0 else "")
    
    # ---- Text Alert for Generative Dropouts ----
    if y_gemma == 0:
        ax.text(1.0, 15, "C2S\nMissing", color="#c0392b", fontsize=7.5, ha="center", fontweight='bold')
    if y_ours == 0:
        ax.text(1.0, 45, "Ours\nMissing", color="#c0392b", fontsize=7.5, ha="center", fontweight='bold')
        
    # Title Configuration
    ax.set_title(gene, fontsize=10, fontweight='bold', color="#2c3e50", pad=4)
    
    # Background Layout Structure
    ax.set_xticks(x_states)
    ax.set_xticklabels(state_labels, fontsize=9.5)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-10, 210)
    ax.grid(True, axis='y', linestyle=':', alpha=0.3, color='#cbd5e1')

# Add global axis labels
fig.text(0.01, 0.5, 'Gene Expression Rank Score (0 to 200 Scale)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
fig.text(0.5, 0.01, 'Experimental & Modeling Alignment States', ha='center', fontsize=12, fontweight='bold')

# Main Figure Title (Pure academic phrasing)
plt.suptitle("State-Transition Trajectory Analysis of Top-20 Shifting DEGs Under RCOR1 Perturbation", 
             fontsize=14, fontweight='bold', y=0.98)

# Unified explicit legend capturing both states and markers
fig.legend(loc="upper right", bbox_to_anchor=(0.99, 0.97), frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9.5)

sns.despine()
plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.95])

# Save Output
plt.savefig(output_png, bbox_inches='tight', dpi=300)
print(f"[+] Facet Grid Graph with distinctive markers successfully generated!")
