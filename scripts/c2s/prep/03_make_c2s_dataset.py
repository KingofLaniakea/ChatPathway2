import os
import anndata
import pandas as pd
from collections import Counter

# 1. 原始文件路径
ORIGINAL_DATA_PATH = "/root/autodl-tmp/data/GSE264667_jurkat_raw_singlecell_01.h5ad"

adata_raw = anndata.read_h5ad(ORIGINAL_DATA_PATH)
print("原始矩阵规模:", adata_raw.shape)

print("\n正在克隆新数据对象...")
adata_c2s = adata_raw.copy()

# 1. 映射 batch_var (数字批次加前缀，如 27 -> jurkat27)
adata_c2s.obs['batch_var'] = adata_c2s.obs['gem_group'].apply(lambda x: f"jurkat{x}")

# 2. 补齐 cell_type 静态标签
adata_c2s.obs['cell_type'] = 'jurkat'

# 3. 映射 target_gene (克隆原始敲除基因)
adata_c2s.obs['target_gene'] = adata_c2s.obs['gene'].astype(str)

# 4. 映射 gene_id
adata_c2s.obs['gene_id'] = adata_c2s.obs['gene_id'].astype(str)

# 5. 将原始的对照组标签 "NC" 统一替换为 C2S 官方默认的 "non-targeting"
# adata_c2s.obs['target_gene'] = adata_c2s.obs['target_gene'].replace({'NC': 'non-targeting'})
# adata_c2s.obs['gene_id'] = adata_c2s.obs['gene_id'].replace({'NC': 'non-targeting'})

# 6. 只切出 C2S 案例一模一样的 6 列，把多余的批次 Z-score 和转录本标签丢掉
c2s_obs_cols = ['batch_var', 'cell_type', 'target_gene', 'gene_id', 'mitopercent', 'UMI_count']
adata_c2s.obs = adata_c2s.obs[c2s_obs_cols]

# =====================================================================
# 第二步：改造 var (基因元数据) ── 提升 gene_name 并去除多余列
# =====================================================================
print("\n[步骤 4] 正在转换 var_index 并精简基因为官方 Symbol...")

# 1. 确保基因名是字符串
adata_c2s.var['gene_name'] = adata_c2s.var['gene_name'].astype(str)

# 2. 防止潜在的重名冲突
adata_c2s.var_names_make_unique()
adata_c2s.var['gene_name'] = pd.Series(adata_c2s.var['gene_name']).mask(
    adata_c2s.var['gene_name'].duplicated(), 
    adata_c2s.var['gene_name'] + "-dup"
)

# 3. 将行名（Index）从 ENSG 身份证号彻底替换为大模型认识的基因名字（Gene Symbol）
adata_c2s.var.index = adata_c2s.var['gene_name']

# 4. 剥离所有多余的生信列（如 chr, start, mean 等），使其长相与官网完全一致
adata_c2s.var = adata_c2s.var[[]]  # 传入空列表可以直接清空所有附加列，只留 Index 行名本身

print("\n正在导出全新的 C2S 规范版数据集...")
NEW_DATA_PATH = "/root/autodl-tmp/data/GSE264667_jurkat_C2S_format.h5ad"
adata_c2s.write_h5ad(NEW_DATA_PATH)

print("\n" + "="*50)
print(f"新文件保存路径: {NEW_DATA_PATH}")
print("最终转换后的矩阵规模:", adata_c2s.shape)
print("\n--- 新数据的 adata.obs.head() 验证 ---")
print(adata_c2s.obs.head())
print("\n--- 新数据的 adata.var.head() 验证 ---")
print(adata_c2s.var.head())
print("="*50)
