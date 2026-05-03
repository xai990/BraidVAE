"""
Supervised Pseudotime Assignment for BRCA Progression

This script assigns pseudotime based on the known biological order:
Normal → Stage I → Stage II → Stage III

Within each stage, samples are ordered by their distance from the 
centroid of the previous stage (providing smooth transitions).
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import scanpy as sc
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('latent_all_stages.tsv', sep='\t', index_col=0)
latent_cols = [f'z{i}' for i in range(1, 33)]
X = df[latent_cols].values

print(f"Data shape: {df.shape}")
print(f"Unique labels: {df['label'].unique()}")

# Define the biological order (this is the key constraint)
stage_order = ['normal_GTEX', 'normal_TCGA', 'Stage_I', 'Stage_II', 'Stage_III']

# Assign base pseudotime ranges for each stage
# Normal samples: 0.0 - 0.2
# Stage I: 0.2 - 0.5
# Stage II: 0.5 - 0.75
# Stage III: 0.75 - 1.0
stage_ranges = {
    'normal_GTEX': (0.00, 0.10),
    'normal_TCGA': (0.10, 0.20),
    'Stage_I':     (0.20, 0.50),
    'Stage_II':    (0.50, 0.75),
    'Stage_III':   (0.75, 1.00),
}

def compute_supervised_pseudotime(df, X, stage_order, stage_ranges):
    """
    Compute supervised pseudotime respecting biological order.
    
    Within each stage, samples are ordered by their distance from 
    the centroid of the previous stage (creating a smooth transition).
    """
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
            # For the first stage (normal_GTEX), order by distance from overall centroid
            centroid = stage_X.mean(axis=0)
            distances = cdist(stage_X, centroid.reshape(1, -1)).flatten()
        else:
            # Order by distance from previous stage centroid
            distances = cdist(stage_X, prev_centroid.reshape(1, -1)).flatten()
        
        # Rank samples by distance and map to pseudotime range
        ranks = np.argsort(np.argsort(distances))  # Rank from 0 to n-1
        
        # Normalize ranks to [0, 1] then scale to stage range
        if len(indices) > 1:
            normalized_ranks = ranks / (len(indices) - 1)
        else:
            normalized_ranks = np.array([0.5])
        
        stage_pseudotime = t_min + normalized_ranks * (t_max - t_min)
        pseudotime[indices] = stage_pseudotime
        
        # Update centroid for next stage
        prev_centroid = stage_X.mean(axis=0)
        
        print(f"{stage}: {len(indices)} samples, pseudotime range [{stage_pseudotime.min():.3f}, {stage_pseudotime.max():.3f}]")
    
    return pseudotime

# Compute supervised pseudotime
print("\nComputing supervised pseudotime...")
df['supervised_pseudotime'] = compute_supervised_pseudotime(df, X, stage_order, stage_ranges)

# Also keep the unsupervised version for comparison
adata = sc.AnnData(X)
adata.obs['label'] = df['label'].values
sc.pp.neighbors(adata)
sc.tl.diffmap(adata)
root_idx = np.where(df['label'] == 'normal_GTEX')[0][0]
adata.uns['iroot'] = root_idx
sc.tl.dpt(adata)
df['unsupervised_pseudotime'] = adata.obs['dpt_pseudotime'].values

# Compute UMAP for visualization
sc.tl.umap(adata)
umap_coords = adata.obsm['X_umap']

# Save the data with both pseudotime columns
output_file = 'latent_all_stages_supervised_pseudotime.tsv'
df.to_csv(output_file, sep='\t')
print(f"\nSaved to: {output_file}")

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Unsupervised pseudotime
ax = axes[0, 0]
scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                     c=df['unsupervised_pseudotime'], cmap='viridis', s=15, alpha=0.8)
plt.colorbar(scatter, ax=ax, label='Pseudotime')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('Unsupervised DPT Pseudotime\n(Incorrect: Stage II > Stage III)')

# Plot 2: Supervised pseudotime
ax = axes[0, 1]
scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                     c=df['supervised_pseudotime'], cmap='viridis', s=15, alpha=0.8)
plt.colorbar(scatter, ax=ax, label='Pseudotime')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('Supervised Pseudotime\n(Correct: Normal → I → II → III)')

# Plot 3: Labels for reference
ax = axes[1, 0]
colors = {'normal_GTEX': '#1f77b4', 'normal_TCGA': '#aec7e8', 
          'Stage_I': '#2ca02c', 'Stage_II': '#ff7f0e', 'Stage_III': '#d62728'}
for label in stage_order:
    mask = df['label'] == label
    ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
               c=colors[label], label=label, s=15, alpha=0.8)
ax.legend(loc='upper right')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('Sample Labels')

# Plot 4: Pseudotime distribution by stage
ax = axes[1, 1]
stage_data = [df[df['label'] == stage]['supervised_pseudotime'].values for stage in stage_order]
bp = ax.boxplot(stage_data, labels=['GTEX', 'TCGA', 'Stage I', 'Stage II', 'Stage III'], 
                patch_artist=True)
for patch, color in zip(bp['boxes'], [colors[s] for s in stage_order]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Supervised Pseudotime')
ax.set_xlabel('Stage')
ax.set_title('Pseudotime Distribution by Stage\n(Monotonically Increasing)')

plt.tight_layout()
plt.savefig('supervised_pseudotime_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('supervised_pseudotime_comparison.pdf', bbox_inches='tight', facecolor='white')
print("Saved: supervised_pseudotime_comparison.png/pdf")

# Summary statistics
print("\n" + "="*60)
print("SUPERVISED PSEUDOTIME SUMMARY")
print("="*60)
for stage in stage_order:
    mask = df['label'] == stage
    pt = df.loc[mask, 'supervised_pseudotime']
    print(f"{stage:15s}: mean={pt.mean():.3f}, min={pt.min():.3f}, max={pt.max():.3f}")

print("\n✓ Supervised pseudotime correctly enforces: Normal → Stage I → Stage II → Stage III")

