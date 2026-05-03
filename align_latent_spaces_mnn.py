"""
Align stage-specific VAE latent spaces using MNN (Mutual Nearest Neighbors) in gene
expression space + Procrustes alignment. Produces a unified latent space for valid
cross-stage comparisons.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths
LATENT_PATH = "latent_all_stages.tsv"
TRAIN_EXPR = "datasets_breast/breast_train_GEM_transpose.txt"
TRAIN_LABELS = "datasets_breast/train_stage_labels_new_updated.txt"
TEST_EXPR = "datasets_breast/breast_test_GEM_transpose.txt"
TEST_LABELS = "datasets_breast/test_stage_labels_new_updated.txt"
HALLMARK_GENE_LIST = "datasets_breast/hallmark_allgene_transposed.txt"  # Use only these genes for MNN
OUTPUT_PATH = "latent_all_stages_aligned.tsv"
STAGE_ORDER = ['normal_GTEX', 'normal_TCGA', 'Stage_I', 'Stage_II', 'Stage_III']
REFERENCE_STAGE = 'normal_TCGA'  # Align all to TCGA
N_PCA =20  # PCA dims for MNN in gene space
K_MNN = 5  # k for mutual nearest neighbors


def load_hallmark_genes(path=HALLMARK_GENE_LIST):
    """Load hallmark gene names from first line of hallmark file (tab-separated, first col is header)."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        first_line = f.readline().strip()
    genes = [g.strip() for g in first_line.split('\t') if g.strip()]
    if genes and genes[0].lower().startswith('hallmark'):
        genes = genes[1:]
    return set(genes) if genes else None


def load_gene_expr_and_labels(restrict_to_hallmark=True):
    """Load gene expression and labels, merge train+test. Optionally restrict to hallmark genes."""
    tr_expr = pd.read_csv(TRAIN_EXPR, sep='\t', index_col=0)
    te_expr = pd.read_csv(TEST_EXPR, sep='\t', index_col=0)
    tr_lab = pd.read_csv(TRAIN_LABELS, sep='\t', header=None, names=['sample_id', 'label'])
    te_lab = pd.read_csv(TEST_LABELS, sep='\t', header=None, names=['sample_id', 'label'])
    tr_lab = tr_lab.set_index('sample_id')
    te_lab = te_lab.set_index('sample_id')
    common_genes = tr_expr.columns.intersection(te_expr.columns)
    if restrict_to_hallmark:
        hallmark = load_hallmark_genes()
        if hallmark:
            common_genes = common_genes.intersection(hallmark)
            print(f"  Restricting to {len(common_genes)} hallmark genes (from {HALLMARK_GENE_LIST})")
        else:
            print("  Warning: Hallmark list not found or empty; using all common genes")
    tr_expr = tr_expr.loc[:, common_genes]
    te_expr = te_expr.loc[:, common_genes]
    expr = pd.concat([tr_expr, te_expr], axis=0)
    labels = pd.concat([tr_lab, te_lab], axis=0)
    return expr, labels


def find_mnn_pairs(X_src, X_tgt, k=20):
    """Find mutual nearest neighbor pairs between source and target."""
    # For each src point, find k nearest in tgt
    d_s2t = cdist(X_src, X_tgt, metric='euclidean')
    nn_s2t = np.argsort(d_s2t, axis=1)[:, :k]  # [n_src, k]
    # For each tgt point, find k nearest in src
    d_t2s = cdist(X_tgt, X_src, metric='euclidean')
    nn_t2s = np.argsort(d_t2s, axis=1)[:, :k]  # [n_tgt, k]
    pairs = []
    for i in range(len(X_src)):
        for j in nn_s2t[i]:
            if i in nn_t2s[j]:
                pairs.append((i, j))
    return pairs


def procrustes_align(Z_src, Z_tgt, pairs):
    """Align Z_src to Z_tgt using Procrustes with given correspondences."""
    if len(pairs) < 2:
        return np.eye(Z_src.shape[1])
    src_idx = [p[0] for p in pairs]
    tgt_idx = [p[1] for p in pairs]
    A = Z_src[src_idx]
    B = Z_tgt[tgt_idx]
    R, scale = orthogonal_procrustes(A, B)
    return R


def main():
    print("Loading latent embeddings...")
    df_latent = pd.read_csv(LATENT_PATH, sep='\t', index_col=0)
    latent_cols = [c for c in df_latent.columns if c.startswith('z')]
    dim = len(latent_cols)

    print("Loading gene expression...")
    expr, labels = load_gene_expr_and_labels()
    common_samples = df_latent.index.intersection(expr.index).tolist()
    df_latent = df_latent.loc[common_samples]
    expr = expr.reindex(common_samples).fillna(0)
    labels = labels.reindex(common_samples)
    print(f"  {len(common_samples)} samples in common")

    # Fill NaN with 0, log1p - X_gene row order matches common_samples
    X_gene = np.nan_to_num(expr.values, nan=0.0, posinf=0.0, neginf=0.0)
    X_gene = np.log1p(np.maximum(X_gene, 0))
    scaler = StandardScaler()
    X_gene = scaler.fit_transform(X_gene)
    if X_gene.shape[1] > N_PCA:
        pca = PCA(n_components=N_PCA, random_state=42)
        X_pca = pca.fit_transform(X_gene)
        print(f"  PCA: {X_gene.shape[1]} genes -> {N_PCA} components")
    else:
        X_pca = X_gene

    # X_pca[i] corresponds to common_samples[i] which is df_latent.index[i]
    ref_mask = df_latent['label'].values == REFERENCE_STAGE
    Z_ref = df_latent.loc[ref_mask, latent_cols].values
    X_ref = X_pca[ref_mask]

    aligned_blocks = []
    for s in STAGE_ORDER:
        mask = (df_latent['label'].values == s)
        Z_s = df_latent.loc[mask, latent_cols].values
        X_s = X_pca[mask]
        if s == REFERENCE_STAGE:
            aligned_blocks.append((df_latent.index[mask].tolist(), Z_s))
            continue
        pairs = find_mnn_pairs(X_s, X_ref, k=K_MNN)
        print(f"  {s} -> {REFERENCE_STAGE}: {len(pairs)} MNN pairs")
        R = procrustes_align(Z_s, Z_ref, pairs)
        Z_aligned = Z_s @ R
        aligned_blocks.append((df_latent.index[mask].tolist(), Z_aligned))

    # Reconstruct dataframe
    all_idx = []
    all_Z = []
    for idx_list, Z_block in aligned_blocks:
        all_idx.extend(idx_list)
        all_Z.append(Z_block)
    Z_all = np.vstack(all_Z)
    df_out = pd.DataFrame(Z_all, index=all_idx, columns=latent_cols)
    df_out['split'] = df_latent.loc[all_idx, 'split'].values
    df_out['label'] = df_latent.loc[all_idx, 'label'].values
    df_out.to_csv(OUTPUT_PATH, sep='\t')
    print(f"\nSaved aligned latent: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
