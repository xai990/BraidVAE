"""
Generate histogram/bar plot of latent dimension importance from the Informer model.
One subplot per transition (not averaged).
Uses dimension_significance.csv from the cascaded genes analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
DIM_SIG_PATH = "results_cascaded_aligned_genes/dimension_significance.csv"
OUTPUT_DIR = "results_cascaded_aligned_genes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Transition columns for importance (short labels for titles)
TRANSITIONS = [
    ('normal_GTEX_to_normal_TCGA', 'GTEx → TCGA'),
    ('normal_GTEX_to_Stage_I', 'GTEx → Stage I'),
    ('normal_TCGA_to_Stage_I', 'TCGA → Stage I'),
    ('Stage_I_to_Stage_II', 'Stage I → Stage II'),
    ('Stage_II_to_Stage_III', 'Stage II → Stage III'),
]

def main():
    df = pd.read_csv(DIM_SIG_PATH)
    dim_labels = df['dimension'].tolist()
    x = np.arange(len(dim_labels))

    # Create 2x3 subplots, one per transition (5 transitions)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (trans_key, trans_label) in enumerate(TRANSITIONS):
        ax = axes[idx]
        imp_col = f'{trans_key}_importance'
        sig_col = f'{trans_key}_significant'

        if imp_col not in df.columns:
            ax.set_visible(False)
            continue

        importance = df[imp_col].values
        is_sig = df[sig_col].values if sig_col in df.columns else np.zeros(len(df), dtype=bool)

        # Threshold: mean + 1 std for THIS transition
        mean_imp = np.mean(importance)
        std_imp = np.std(importance)
        threshold = mean_imp + 1.0 * std_imp

        colors = ['#d62728' if s else 'steelblue' for s in is_sig]
        ax.bar(x, importance, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(y=threshold, color='green', linestyle='--', linewidth=1.5,
                   label=f'mean+1σ = {threshold:.3f}')
        ax.set_xlabel('Latent Dimension', fontsize=10)
        ax.set_ylabel('Importance', fontsize=10)
        ax.set_title(trans_label, fontsize=12)
        ax.set_xticks(x[::4])  # Every 4th tick to avoid crowding
        ax.set_xticklabels([dim_labels[i] for i in range(0, len(dim_labels), 4)], rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplot (6th in 2x3 grid)
    if len(TRANSITIONS) < len(axes):
        axes[len(TRANSITIONS)].set_visible(False)

    fig.suptitle('Informer Model: Dimension Importance per Transition\n(Red = significant)', fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'informer_dimension_histogram.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")

    pdf_path = os.path.join(OUTPUT_DIR, 'informer_dimension_histogram.pdf')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (trans_key, trans_label) in enumerate(TRANSITIONS):
        ax = axes[idx]
        imp_col = f'{trans_key}_importance'
        sig_col = f'{trans_key}_significant'
        if imp_col not in df.columns:
            ax.set_visible(False)
            continue
        importance = df[imp_col].values
        is_sig = df[sig_col].values if sig_col in df.columns else np.zeros(len(df), dtype=bool)
        mean_imp = np.mean(importance)
        std_imp = np.std(importance)
        threshold = mean_imp + 1.0 * std_imp
        colors = ['#d62728' if s else 'steelblue' for s in is_sig]
        ax.bar(x, importance, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(y=threshold, color='green', linestyle='--', linewidth=1.5, label=f'mean+1σ = {threshold:.3f}')
        ax.set_xlabel('Latent Dimension', fontsize=10)
        ax.set_ylabel('Importance', fontsize=10)
        ax.set_title(trans_label, fontsize=12)
        ax.set_xticks(x[::4])
        ax.set_xticklabels([dim_labels[i] for i in range(0, len(dim_labels), 4)], rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    if len(TRANSITIONS) < len(axes):
        axes[len(TRANSITIONS)].set_visible(False)
    fig.suptitle('Informer Model: Dimension Importance per Transition\n(Red = significant)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {pdf_path}")

if __name__ == "__main__":
    main()
