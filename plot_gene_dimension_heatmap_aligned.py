"""
Generate heatmap of genes vs. latent dimensions from genes_per_transition_all.csv.
One subplot per transition (no averaging across models).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DIM_SIG_PATH = "results_cascaded_aligned_genes/dimension_significance.csv"
GENES_PATH = "results_cascaded_aligned_genes/genes_per_transition_all.csv"
OUTPUT_DIR = "results_cascaded_aligned_genes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Layout
TRANSITIONS = [
    ('normal_GTEX_to_normal_TCGA', 'GTEx → TCGA'),
    ('normal_GTEX_to_Stage_I', 'GTEx → Stage I'),
    ('normal_TCGA_to_Stage_I', 'TCGA → Stage I'),
    ('Stage_I_to_Stage_II', 'Stage I → Stage II'),
    ('Stage_II_to_Stage_III', 'Stage II → Stage III'),
]
TOP_GENES = 15  # Per transition
MAX_DIMS = 8    # Per transition


def get_sig_dims_for_transition(df_sig, trans_key):
    """Get dimension names significant for this transition only."""
    sig_col = f'{trans_key}_significant'
    if sig_col not in df_sig.columns:
        return []
    return df_sig[df_sig[sig_col].fillna(False).astype(bool)]['dimension'].tolist()


def main():
    df_sig = pd.read_csv(DIM_SIG_PATH)
    df_genes = pd.read_csv(GENES_PATH)

    # Create 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (trans_key, trans_label) in enumerate(TRANSITIONS):
        ax = axes[idx]

        sig_dims = get_sig_dims_for_transition(df_sig, trans_key)
        trans_genes = df_genes[df_genes['transition'] == trans_key]

        if len(sig_dims) == 0 or len(trans_genes) == 0:
            ax.set_visible(False)
            continue

        sig_dims = sig_dims[:MAX_DIMS]

        # Build gene x dimension matrix for THIS transition only
        gene_dim = {}
        for _, row in trans_genes.iterrows():
            if row['dimension'] not in sig_dims:
                continue
            key = (row['gene'], row['dimension'])
            gene_dim[key] = max(gene_dim.get(key, 0), row['importance'])

        gene_total = {}
        for (gene, dim), imp in gene_dim.items():
            gene_total[gene] = gene_total.get(gene, 0) + imp

        top_genes = sorted(gene_total.keys(), key=lambda g: -gene_total[g])[:TOP_GENES]

        matrix = np.zeros((len(top_genes), len(sig_dims)))
        for i, gene in enumerate(top_genes):
            for j, dim in enumerate(sig_dims):
                matrix[i, j] = gene_dim.get((gene, dim), 0)

        vmax = matrix.max() if matrix.max() > 0 else 0.01
        sns.heatmap(matrix, cmap='YlOrRd', ax=ax,
                    xticklabels=sig_dims,
                    yticklabels=top_genes,
                    cbar_kws={'label': 'Importance'},
                    vmin=0, vmax=vmax)
        ax.set_xlabel('Latent Dimension', fontsize=10)
        ax.set_ylabel('Gene', fontsize=10)
        ax.set_title(trans_label, fontsize=12, fontweight='bold')

    axes[5].set_visible(False)

    fig.suptitle('Top Genes per Transition (Integrated Gradients)\nNo averaging across models', fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'gene_importance_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")

    # PDF
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.flatten()
    for idx, (trans_key, trans_label) in enumerate(TRANSITIONS):
        ax = axes[idx]
        sig_dims = get_sig_dims_for_transition(df_sig, trans_key)
        trans_genes = df_genes[df_genes['transition'] == trans_key]
        if len(sig_dims) == 0 or len(trans_genes) == 0:
            ax.set_visible(False)
            continue
        sig_dims = sig_dims[:MAX_DIMS]
        gene_dim = {}
        for _, row in trans_genes.iterrows():
            if row['dimension'] not in sig_dims:
                continue
            key = (row['gene'], row['dimension'])
            gene_dim[key] = max(gene_dim.get(key, 0), row['importance'])
        gene_total = {}
        for (gene, dim), imp in gene_dim.items():
            gene_total[gene] = gene_total.get(gene, 0) + imp
        top_genes = sorted(gene_total.keys(), key=lambda g: -gene_total[g])[:TOP_GENES]
        matrix = np.zeros((len(top_genes), len(sig_dims)))
        for i, gene in enumerate(top_genes):
            for j, dim in enumerate(sig_dims):
                matrix[i, j] = gene_dim.get((gene, dim), 0)
        vmax = matrix.max() if matrix.max() > 0 else 0.01
        sns.heatmap(matrix, cmap='YlOrRd', ax=ax,
                    xticklabels=sig_dims, yticklabels=top_genes,
                    cbar_kws={'label': 'Importance'}, vmin=0, vmax=vmax)
        ax.set_xlabel('Latent Dimension', fontsize=10)
        ax.set_ylabel('Gene', fontsize=10)
        ax.set_title(trans_label, fontsize=12, fontweight='bold')
    axes[5].set_visible(False)
    fig.suptitle('Top Genes per Transition (Integrated Gradients)\nNo averaging across models', fontsize=14, y=1.02)
    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'gene_importance_heatmap.pdf')
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
