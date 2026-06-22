import os
import anndata
import pandas as pd
from collections import Counter

# 1. Original File Path
ORIGINAL_DATA_PATH = "/root/autodl-tmp/data/GSE264667_jurkat_raw_singlecell_01.h5ad"

adata_raw = anndata.read_h5ad(ORIGINAL_DATA_PATH)
print("Original matrix shape:", adata_raw.shape)

print("\nCloning data object for format conversion...")
adata_c2s = adata_raw.copy()

# 1. Map batch_var (add prefix to numeric batch, e.g., 27 -> jurkat27)
adata_c2s.obs['batch_var'] = adata_c2s.obs['gem_group'].apply(lambda x: f"jurkat{x}")

# 2. Add static cell_type label
adata_c2s.obs['cell_type'] = 'jurkat'

# 3. Map target_gene (clone original knockout gene)
adata_c2s.obs['target_gene'] = adata_c2s.obs['gene'].astype(str)

# 4. Map gene_id
adata_c2s.obs['gene_id'] = adata_c2s.obs['gene_id'].astype(str)

# 5. Keep only required columns, discarding extra metadata
c2s_obs_cols = ['batch_var', 'cell_type', 'target_gene', 'gene_id', 'mitopercent', 'UMI_count']
adata_c2s.obs = adata_c2s.obs[c2s_obs_cols]

# =====================================================================
# Step 2: Transform var (gene metadata) & streamline columns
# =====================================================================
print("\nConverting var_index to official Gene Symbols...")

# 1. Ensure gene names are strings
adata_c2s.var['gene_name'] = adata_c2s.var['gene_name'].astype(str)

# 2. Prevent potential duplicate name conflicts
adata_c2s.var_names_make_unique()
adata_c2s.var['gene_name'] = pd.Series(adata_c2s.var['gene_name']).mask(
    adata_c2s.var['gene_name'].duplicated(), 
    adata_c2s.var['gene_name'] + "-dup"
)

# 3. Replace index from ENSG ID to Gene Symbol
adata_c2s.var.index = adata_c2s.var['gene_name']

# 4. Remove all redundant metadata columns, keeping only the index
adata_c2s.var = adata_c2s.var[[]]

print("\nExporting formatted dataset...")
NEW_DATA_PATH = "/root/autodl-tmp/data/GSE264667_jurkat_C2S_format.h5ad"
adata_c2s.write_h5ad(NEW_DATA_PATH)

print("\n" + "="*50)
print(f"Saved to: {NEW_DATA_PATH}")
print("Final matrix shape:", adata_c2s.shape)
print("\n--- Validation: adata.obs.head() ---")
print(adata_c2s.obs.head())
print("\n--- Validation: adata.var.head() ---")
print(adata_c2s.var.head())
print("="*50)
