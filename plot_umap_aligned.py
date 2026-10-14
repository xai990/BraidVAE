"""
Generate UMAP plot from MNN-Procrustes aligned latent space using scanpy.
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "results_multistage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

latent_cols = [f'z{i}' for i in range(1, 33)]

df = pd.read_csv("latent_all_stages_aligned.tsv", sep='\t', index_col=0)
adata = sc.AnnData(df[latent_cols].values)
adata.obs['label'] = pd.Categorical(df['label'].values,
    categories=['normal_GTEX', 'normal_TCGA', 'Stage_I', 'Stage_II', 'Stage_III'])
adata.obs_names = df.index.tolist()

sc.pp.neighbors(adata, use_rep='X')
sc.tl.umap(adata, random_state=42)

fig, ax = plt.subplots(figsize=(8, 7))
sc.pl.umap(adata, color='label', ax=ax, show=False,
           title='UMAP: Aligned Latent Space (MNN + Procrustes)', frameon=True)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "global_umap_aligned.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
