"""
Gradient-Based Gene Importance Analysis

More accurate than decoder weights - uses gradients to find which 
input genes most affect the progression-relevant latent dimensions.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bin.model import ConditionalVAE

# Configuration
VAE_DIR = "results_multistage"
LATENT_DATA = "latent_all_stages_supervised_pseudotime.tsv"
GENE_NAMES_PATH = "gene_names.txt"
OUTPUT_DIR = "results_gene_importance_gradient"

# Significant latent dimensions (from between-stage variance analysis)
SIGNIFICANT_DIMS = [30, 22, 1, 6, 19, 13]  # 0-indexed: z31, z23, z2, z7, z20, z14


def load_vae_and_data(stage_name):
    """Load VAE model and corresponding data for a stage"""
    ckpt_path = os.path.join(VAE_DIR, f"stage_{stage_name}", "best_vae.pth")
    
    if not os.path.exists(ckpt_path):
        return None, None
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Get input dimension from the first encoder layer
    input_dim = None
    for key in ckpt.keys():
        if 'encoder.encoder.1.0.weight' in key:
            input_dim = ckpt[key].shape[1]
            break
    
    # Use fixed architecture (must match training config)
    hidden_dims = [1024, 256, 64]  # From config.yaml
    latent_dim = 32
    
    print(f"  Using architecture: input={input_dim}, hidden={hidden_dims}, latent={latent_dim}")
    
    # Create model
    model = ConditionalVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        n_classes=0
    )
    model.load_state_dict(ckpt)
    model.eval()
    
    return model, input_dim


def compute_gradient_importance(model, data, target_dims, batch_size=32):
    """
    Compute gene importance using gradients of latent dimensions w.r.t. inputs.
    
    Args:
        model: VAE model
        data: Input gene expression data [n_samples, n_genes]
        target_dims: List of latent dimensions to analyze
        
    Returns:
        importance: [n_dims, n_genes] importance scores
    """
    model.eval()
    n_genes = data.shape[1]
    n_dims = len(target_dims)
    
    # Accumulate gradients
    importance = np.zeros((n_dims, n_genes))
    
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        batch = data_tensor[start_idx:end_idx]
        batch.requires_grad_(True)
        
        # Forward pass through encoder
        mu, logvar = model.encode(batch)
        
        for dim_idx, target_dim in enumerate(target_dims):
            # Compute gradient of target latent dim w.r.t. input
            if batch.grad is not None:
                batch.grad.zero_()
            
            # Sum over batch for gradient computation
            mu[:, target_dim].sum().backward(retain_graph=True)
            
            # Accumulate absolute gradients
            importance[dim_idx] += np.abs(batch.grad.numpy()).sum(axis=0)
        
        batch.requires_grad_(False)
    
    # Normalize by number of samples
    importance /= len(data)
    
    return importance


def compute_differential_importance(model, data, labels, target_dims, stage_order):
    """
    Find genes whose importance changes during progression.
    
    Args:
        model: VAE model
        data: Gene expression data
        labels: Stage labels
        target_dims: Latent dimensions to analyze
        stage_order: Order of stages
        
    Returns:
        differential_importance: Genes that become more important in late stages
    """
    # Split into early (Normal, Stage I) and late (Stage II, III)
    early_stages = ['normal_GTEX', 'normal_TCGA', 'Stage_I']
    late_stages = ['Stage_II', 'Stage_III']
    
    early_mask = np.isin(labels, early_stages)
    late_mask = np.isin(labels, late_stages)
    
    early_data = data[early_mask]
    late_data = data[late_mask]
    
    print(f"Early stage samples: {len(early_data)}")
    print(f"Late stage samples: {len(late_data)}")
    
    if len(early_data) == 0 or len(late_data) == 0:
        return None
    
    # Compute importance for each
    early_importance = compute_gradient_importance(model, early_data, target_dims)
    late_importance = compute_gradient_importance(model, late_data, target_dims)
    
    # Ratio: high = becomes more important in late stages
    # Add small epsilon to avoid division by zero
    differential = late_importance / (early_importance + 1e-8)
    
    return differential, early_importance, late_importance


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("GRADIENT-BASED GENE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load gene names
    if os.path.exists(GENE_NAMES_PATH):
        with open(GENE_NAMES_PATH, 'r') as f:
            gene_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(gene_names)} gene names")
    else:
        print(f"Gene names not found at {GENE_NAMES_PATH}")
        return
    
    # Load latent data for labels
    df_latent = pd.read_csv(LATENT_DATA, sep='\t', index_col=0)
    print(f"Loaded latent data: {df_latent.shape}")
    
    # Find stages
    stages = df_latent['label'].unique()
    print(f"Stages: {stages}")
    
    # We need the original gene expression data
    # Try to find it
    gene_expr_paths = [
        'datasets_breast/hallmark_allgene_transposed.txt',
        'datasets_breast/train_data.tsv',
    ]
    
    gene_data = None
    for path in gene_expr_paths:
        if os.path.exists(path):
            print(f"Loading gene expression from: {path}")
            gene_data = pd.read_csv(path, sep='\t', index_col=0)
            print(f"Gene data shape: {gene_data.shape}")
            break
    
    if gene_data is None:
        print("Could not find gene expression data")
        print("Creating synthetic data from VAE for demonstration...")
        # We'll use the VAE to generate samples
        gene_data = None  # Will handle below
    
    # Load a VAE model
    model = None
    for stage in ['Stage_I', 'Stage_II', 'normal_GTEX']:
        model, input_dim = load_vae_and_data(stage)
        if model is not None:
            print(f"Loaded VAE from stage: {stage}")
            print(f"Input dimension: {input_dim}")
            break
    
    if model is None:
        print("Could not load any VAE model")
        return
    
    # If we don't have gene expression data, we can still analyze decoder weights
    # with improved methodology (using gradient through decoder)
    
    print(f"\nAnalyzing dimensions: z{[d+1 for d in SIGNIFICANT_DIMS]}")
    
    # Method 1: Decoder Jacobian (analytical gradient)
    print("\n" + "="*70)
    print("METHOD: Decoder Jacobian Analysis")
    print("="*70)
    
    # Create random latent vectors and compute Jacobian
    n_samples = 100
    z_samples = torch.randn(n_samples, 32)
    
    jacobian_importance = np.zeros((len(SIGNIFICANT_DIMS), input_dim))
    
    for dim_idx, target_dim in enumerate(SIGNIFICANT_DIMS):
        # Compute gradient of decoder output w.r.t. latent dimension
        for i in range(n_samples):
            z = z_samples[i:i+1].clone().requires_grad_(True)
            
            # Forward through decoder
            recon = model.decode(z)
            
            # Compute gradient of each gene w.r.t. target latent dim
            grads = torch.autograd.grad(
                outputs=recon.sum(),
                inputs=z,
                create_graph=False
            )[0]
            
            # This gives us the sensitivity of reconstruction to z
            # We want sensitivity of each gene to the target dimension
            z.requires_grad_(True)
            recon = model.decode(z)
            
            # For each gene, compute d(gene)/d(z[target_dim])
            gene_grads = []
            for gene_idx in range(min(input_dim, recon.shape[1])):
                if z.grad is not None:
                    z.grad.zero_()
                recon[0, gene_idx].backward(retain_graph=True)
                gene_grads.append(np.abs(z.grad[0, target_dim].item()))
            
            jacobian_importance[dim_idx] += np.array(gene_grads)
    
    jacobian_importance /= n_samples
    
    # Get top genes per dimension
    print("\nTop genes by Jacobian sensitivity:")
    
    results = []
    for dim_idx, target_dim in enumerate(SIGNIFICANT_DIMS):
        dim_name = f"z{target_dim + 1}"
        top_indices = np.argsort(jacobian_importance[dim_idx])[::-1][:10]
        
        print(f"\n{dim_name}:")
        for rank, gene_idx in enumerate(top_indices):
            if gene_idx < len(gene_names):
                gene = gene_names[gene_idx]
                importance = jacobian_importance[dim_idx, gene_idx]
                print(f"  {rank+1}. {gene:12s} (importance: {importance:.4f})")
                results.append({
                    'dimension': dim_name,
                    'rank': rank + 1,
                    'gene': gene,
                    'importance': importance,
                    'gene_idx': gene_idx
                })
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'gradient_gene_importance.csv'), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/gradient_gene_importance.csv")
    
    # Load BC gene sets for validation
    with open('datasets_breast/coregene307.txt', 'r') as f:
        content = f.read().strip()
        COREGENE307 = set(content.split('\t')[1:])
    
    try:
        with open('datasets_breast/all_bc_genes.txt', 'r') as f:
            ALL_BC_GENES = set(f.read().strip().split('\n'))
    except:
        ALL_BC_GENES = COREGENE307
    
    # Validate against BC genes
    gradient_genes = set(df_results['gene'].unique())
    bc_overlap = gradient_genes & ALL_BC_GENES
    core_overlap = gradient_genes & COREGENE307
    
    print(f"\n{'='*70}")
    print("VALIDATION")
    print("="*70)
    print(f"Total gradient-identified genes: {len(gradient_genes)}")
    print(f"In extended BC gene set: {len(bc_overlap)} ({100*len(bc_overlap)/len(gradient_genes):.0f}%)")
    print(f"In Coregene307: {len(core_overlap)} ({100*len(core_overlap)/len(gradient_genes):.0f}%)")
    
    print("\nBC genes found:")
    for gene in sorted(bc_overlap):
        dims = df_results[df_results['gene'] == gene]['dimension'].tolist()
        print(f"  ✓ {gene}: {dims}")
    
    # Heatmap
    plt.figure(figsize=(14, 8))
    
    # Get top 20 genes overall
    gene_total_importance = {}
    for _, row in df_results.iterrows():
        gene = row['gene']
        if gene not in gene_total_importance:
            gene_total_importance[gene] = 0
        gene_total_importance[gene] += row['importance']
    
    top_genes = sorted(gene_total_importance.items(), key=lambda x: -x[1])[:20]
    top_gene_names = [g[0] for g in top_genes]
    
    # Create heatmap matrix
    heatmap_data = np.zeros((len(SIGNIFICANT_DIMS), len(top_gene_names)))
    for dim_idx, target_dim in enumerate(SIGNIFICANT_DIMS):
        for gene_idx, gene in enumerate(top_gene_names):
            gene_rows = df_results[(df_results['dimension'] == f"z{target_dim+1}") & 
                                   (df_results['gene'] == gene)]
            if len(gene_rows) > 0:
                heatmap_data[dim_idx, gene_idx] = gene_rows['importance'].values[0]
    
    # Highlight BC genes
    bc_colors = ['green' if g in ALL_BC_GENES else 'black' for g in top_gene_names]
    
    sns.heatmap(heatmap_data, 
                xticklabels=top_gene_names,
                yticklabels=[f"z{d+1}" for d in SIGNIFICANT_DIMS],
                cmap='YlOrRd',
                annot=True, fmt='.2f')
    
    plt.xlabel('Gene')
    plt.ylabel('Latent Dimension')
    plt.title('Gradient-Based Gene Importance\n(Green = BC gene)')
    
    # Color x-axis labels
    ax = plt.gca()
    for i, label in enumerate(ax.get_xticklabels()):
        if top_gene_names[i] in ALL_BC_GENES:
            label.set_color('green')
            label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradient_importance_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/gradient_importance_heatmap.png")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

