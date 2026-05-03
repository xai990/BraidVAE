"""
Supervised Pseudotime from MNN-Procrustes Aligned Latent Space.

Uses the aligned latent space where cross-stage Euclidean distances are valid.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import scanpy as sc
import matplotlib.pyplot as plt

df = pd.read_csv('latent_all_stages_aligned.tsv', sep='\t', index_col=0)
latent_cols = [f'z{i}' for i in range(1, 33)]
X = df[latent_cols].values

print(f"Data shape: {df.shape}")
print(f"Unique labels: {df['label'].unique()}")

stage_order = ['normal_GTEX', 'normal_TCGA', 'Stage_I', 'Stage_II', 'Stage_III']

stage_ranges = {
    'normal_GTEX': (0.00, 0.10),
    'normal_TCGA': (0.10, 0.20),
    'Stage_I':     (0.20, 0.50),
    'Stage_II':    (0.50, 0.75),
    'Stage_III':   (0.75, 1.00),
}


def compute_supervised_pseudotime(df, X, stage_order, stage_ranges):
    pseudotime = np.zeros(len(df))
    prev_centroid = None

    for stage in stage_order:
        mask = df['label'] == stage
        indices = np.where(mask)[0]

        if len(indices) == 0:
            print(f"Warning: No samples found for {stage}")
            continue

        stage_X = X[indices]
        t_min, t_max = stage_ranges[stage]

        if prev_centroid is None:
            centroid = stage_X.mean(axis=0)
            distances = cdist(stage_X, centroid.reshape(1, -1)).flatten()
        else:
            distances = cdist(stage_X, prev_centroid.reshape(1, -1)).flatten()

        ranks = np.argsort(np.argsort(distances))

        if len(indices) > 1:
            normalized_ranks = ranks / (len(indices) - 1)
        else:
            normalized_ranks = np.array([0.5])

        stage_pseudotime = t_min + normalized_ranks * (t_max - t_min)
        pseudotime[indices] = stage_pseudotime
        prev_centroid = stage_X.mean(axis=0)

        print(f"{stage}: {len(indices)} samples, pseudotime [{stage_pseudotime.min():.3f}, {stage_pseudotime.max():.3f}]")

    return pseudotime


print("\nComputing supervised pseudotime on aligned latent space...")
df['supervised_pseudotime'] = compute_supervised_pseudotime(df, X, stage_order, stage_ranges)

# Unsupervised DPT for comparison
adata = sc.AnnData(X)
adata.obs['label'] = df['label'].values
sc.pp.neighbors(adata, use_rep='X')
sc.tl.diffmap(adata)
root_idx = np.where(df['label'].values == 'normal_GTEX')[0][0]
adata.uns['iroot'] = root_idx
sc.tl.dpt(adata)
df['unsupervised_pseudotime'] = adata.obs['dpt_pseudotime'].values

sc.tl.umap(adata, random_state=42)
umap_coords = adata.obsm['X_umap']

output_file = 'latent_all_stages_aligned_supervised_pseudotime.tsv'
df.to_csv(output_file, sep='\t')
print(f"\nSaved to: {output_file}")

# Comparison figure (2x2)
colors = {
    'normal_GTEX': '#1f77b4', 'normal_TCGA': '#ff7f0e',
    'Stage_I': '#2ca02c', 'Stage_II': '#d62728', 'Stage_III': '#9467bd',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
sc1 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                 c=df['unsupervised_pseudotime'], cmap='viridis', s=15, alpha=0.8)
plt.colorbar(sc1, ax=ax, label='Pseudotime')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('Unsupervised DPT Pseudotime')

ax = axes[0, 1]
sc2 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                 c=df['supervised_pseudotime'], cmap='viridis', s=15, alpha=0.8)
plt.colorbar(sc2, ax=ax, label='Pseudotime')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('Supervised Pseudotime (Aligned Latent)')

ax = axes[1, 0]
for label in stage_order:
    mask = df['label'] == label
    ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
               c=colors[label], label=label, s=15, alpha=0.8)
ax.legend(loc='upper right')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('Sample Labels (Aligned Latent)')

ax = axes[1, 1]
stage_data = [df[df['label'] == s]['supervised_pseudotime'].values for s in stage_order]
bp = ax.boxplot(stage_data, labels=['GTEx', 'TCGA', 'Stage I', 'Stage II', 'Stage III'],
                patch_artist=True)
for patch, s in zip(bp['boxes'], stage_order):
    patch.set_facecolor(colors[s])
    patch.set_alpha(0.7)
ax.set_ylabel('Supervised Pseudotime')
ax.set_xlabel('Stage')
ax.set_title('Pseudotime Distribution by Stage')

plt.tight_layout()
plt.savefig('figures_aligned/supervised_pseudotime_comparison_aligned.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('figures_aligned/supervised_pseudotime_comparison_aligned.pdf', bbox_inches='tight', facecolor='white')
print("Saved: figures_aligned/supervised_pseudotime_comparison_aligned.png/pdf")

# Also save a single UMAP colored by pseudotime
fig, ax = plt.subplots(figsize=(8, 7))
sc3 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                 c=df['supervised_pseudotime'], cmap='magma', s=15, alpha=0.8)
plt.colorbar(sc3, ax=ax, label='Supervised Pseudotime')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('Pseudotime Trajectory (MNN-Aligned Latent)')
ax.set_facecolor('#f5f5f5')
plt.tight_layout()
plt.savefig('figures_aligned/umap_pseudotime_aligned.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: figures_aligned/umap_pseudotime_aligned.png")

print("\nSUPERVISED PSEUDOTIME SUMMARY (Aligned Latent)")
print("=" * 60)
for stage in stage_order:
    mask = df['label'] == stage
    pt = df.loc[mask, 'supervised_pseudotime']
    print(f"{stage:15s}: mean={pt.mean():.3f}, min={pt.min():.3f}, max={pt.max():.3f}")
