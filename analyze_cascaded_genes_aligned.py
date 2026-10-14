"""
Analyze Cascaded Informer Models to Identify Significant Latent Dimensions
and Trace Back to Important Genes

This script:
1. Analyzes each stage transition model to find significant latent dimensions
2. Uses VAE decoder weights to trace back to genes
3. Outputs gene importance rankings for each transition
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Captum for Integrated Gradients (Gold Standard for attribution)
try:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: captum not installed. Install with: pip install captum")

# Kneed for Elbow method (gene selection)
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
    print("Warning: kneed not installed. Install with: pip install kneed")

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Informer2020'))

from models.model import Informer
from bin.model import ConditionalVAE
from bin.utils import seed_everything

# ============================================================================
# CONFIGURATION
# ============================================================================
CASCADED_DIR = "results_cascaded_aligned"
VAE_DIR = "results_multistage"
DATA_PATH = "latent_all_stages_aligned.tsv"
GENE_LIST_PATH = "gene_names.txt"
OUTPUT_DIR = "results_cascaded_aligned_genes"

# VAE architecture (must match training)
LATENT_DIM = 16  # Will be detected from data
HIDDEN_DIMS = [1024, 256, 64]

# Gene selection method: Choose 'std' or 'elbow'
# - 'std': Statistical threshold (> mean + N*std) - rigorous, interpretable
# - 'elbow': Elbow/Kneedle method - data-driven, finds natural cutoff
GENE_SELECTION_METHOD = 'std'  # Options: 'std' or 'elbow'

# Parameters for statistical threshold method (when GENE_SELECTION_METHOD = 'std')
# Using >2 std for more stringent selection (fewer genes)
# Can adjust: 1.0 std (more genes) to 3.0 std (very few genes)
GENE_SELECTION_STD_THRESHOLD = 2.0  # Select genes with importance > mean + 2*std

# Parameters for elbow method (when GENE_SELECTION_METHOD = 'elbow')
# Higher S = less sensitive = finds elbow earlier = fewer genes selected
# Lower S = more sensitive = finds elbow later = more genes selected
# Recommended range: 0.5 (more genes) to 5.0 (fewer genes)
ELBOW_SENSITIVITY = 3  # Adjust this to control number of genes selected

# Stage transitions
TRANSITIONS = [
    ('normal_GTEX', 'normal_TCGA'),
    ('normal_GTEX', 'Stage_I'),  # Direct transition from GTEX to Stage I
    ('normal_TCGA', 'Stage_I'),
    ('Stage_I', 'Stage_II'),
    ('Stage_II', 'Stage_III'),
]

STAGE_ORDER = ['normal_GTEX', 'normal_TCGA', 'Stage_I', 'Stage_II', 'Stage_III']

TOP_K_GENES = 30

seed_everything(42)


def load_cascaded_models(cascaded_dir, dim, device):
    """Load all cascaded Informer models and their scalers."""
    models = []
    scalers = []

    for src, tgt in TRANSITIONS:
        name = f"{src}_to_{tgt}"
        model_path = os.path.join(cascaded_dir, f"informer_{name}.pth")
        scaler_path = os.path.join(cascaded_dir, f"scaler_{name}.pkl")

        if not os.path.exists(model_path):
            print(f"Warning: Model not found: {model_path}")
            continue

        # Create model
        model = Informer(
            enc_in=dim,
            dec_in=dim,
            c_out=dim,
            seq_len=1,
            label_len=1,
            out_len=1,
            factor=5,
            d_model=64,
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=256,
            dropout=0.1,
            attn='full',
            embed='timeF',
            freq='t',
            distil=False,
            device=device
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append((name, model))

        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        scalers.append(scaler)

        print(f"  Loaded: {name}")

    return models, scalers


def find_vae_models(vae_dir):
    """Find all VAE model checkpoints."""
    pattern = os.path.join(vae_dir, "stage_*", "best_vae.pth")
    models = glob.glob(pattern)

    result = {}
    for path in models:
        stage_dir = os.path.dirname(path)
        stage_name = os.path.basename(stage_dir).replace("stage_", "")
        result[stage_name] = path

    return result


class DecoderWrapper(nn.Module):
    """Wrapper to make VAE decoder compatible with Captum IntegratedGradients."""
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model

    def forward(self, z):
        return self.vae.decode(z)


def compute_gradient_gene_importance(model, device='cpu', n_samples=10):
    """
    Compute gene importance using INTEGRATED GRADIENTS (Captum).

    This is the GOLD STANDARD for attribution in deep learning.
    - Published in top venues (Nature Machine Intelligence, etc.)
    - Satisfies axioms: Sensitivity, Implementation Invariance
    - Computes gradient at multiple steps from baseline to input

    IG(x) = (x - baseline) × ∫₀¹ ∂F/∂x(baseline + α(x - baseline)) dα

    Returns: [n_genes, latent_dim] importance matrix
    """
    model.eval()
    model.to(device)

    n_genes = model.input_dim
    latent_dim = model.latent_dim

    print(f"    Computing INTEGRATED GRADIENTS (Captum): {n_genes} genes × {latent_dim} latent dims")

    if not CAPTUM_AVAILABLE:
        print("    ERROR: captum not available. Falling back to simple gradients.")
        return compute_simple_gradient_importance(model, device, n_samples)

    # Wrap decoder for Captum
    decoder_wrapper = DecoderWrapper(model).to(device)
    decoder_wrapper.eval()

    # Baseline: zero latent vector (represents "no signal")
    baseline = torch.zeros(1, latent_dim, device=device)

    # Multiple input points to average over (representing typical latent encodings)
    print(f"    Using {n_samples} reference points and 50 integration steps per point...")

    # Sample reference points from standard normal (typical VAE latent space)
    reference_points = []
    reference_points.append(torch.zeros(1, latent_dim, device=device))  # Zero point
    for _ in range(n_samples - 1):
        # Sample from prior (standard normal for VAE)
        reference_points.append(torch.randn(1, latent_dim, device=device) * 0.5)

    all_attributions = []

    for ref_idx, z_input in enumerate(reference_points):
        # Compute IG for each gene output
        # Attribution shape will be [1, latent_dim] for each gene

        gene_attributions = np.zeros((n_genes, latent_dim))

        # Process genes in batches for efficiency
        batch_size = 100
        n_batches = (n_genes + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_genes)

            for gene_idx in range(start_idx, end_idx):
                # Create IG for this specific gene output
                def forward_gene(z):
                    output = decoder_wrapper(z)
                    return output[:, gene_idx:gene_idx+1]

                ig = IntegratedGradients(forward_gene)

                # Compute attributions
                attributions = ig.attribute(
                    z_input,
                    baselines=baseline,
                    n_steps=50,  # Integration steps
                    return_convergence_delta=False
                )

                # Store absolute attributions
                gene_attributions[gene_idx, :] = attributions.abs().cpu().numpy().flatten()

        all_attributions.append(gene_attributions)

        if (ref_idx + 1) % 2 == 0 or ref_idx == 0:
            print(f"    Processed {ref_idx + 1}/{n_samples} reference points...")

    # Average attributions across all reference points
    avg_importance = np.mean(all_attributions, axis=0)

    print(f"    Integrated Gradients complete. Shape: {avg_importance.shape}")

    return avg_importance


def compute_simple_gradient_importance(model, device='cpu', n_samples=10):
    """
    Fallback: Simple gradient computation if captum is not available.
    """
    model.eval()
    model.to(device)

    n_genes = model.input_dim
    latent_dim = model.latent_dim

    print(f"    Computing simple gradients (fallback): {n_genes} genes × {latent_dim} latent dims")

    reference_points = [torch.zeros(1, latent_dim, device=device)]
    for _ in range(n_samples - 1):
        reference_points.append(torch.randn(1, latent_dim, device=device) * 0.1)

    all_importance = []

    for ref_idx, z_ref in enumerate(reference_points):
        importance_matrix = np.zeros((n_genes, latent_dim))

        z = z_ref.clone().detach().requires_grad_(True)
        recon = model.decode(z)

        for gene_idx in range(n_genes):
            grad = torch.autograd.grad(
                recon[0, gene_idx],
                z,
                retain_graph=True,
                create_graph=False
            )[0]
            importance_matrix[gene_idx, :] = grad.abs().cpu().numpy().flatten()

        all_importance.append(importance_matrix)

        if (ref_idx + 1) % 3 == 0:
            print(f"    Processed {ref_idx + 1}/{n_samples} reference points...")

    avg_importance = np.mean(all_importance, axis=0)
    return avg_importance


def analyze_transition_importance(model, scaler, df, src_stage, tgt_stage, latent_cols, device):
    """
    Analyze which latent dimensions are most important for a transition.
    """
    model.eval()
    dim = len(latent_cols)

    # Get data
    src_data = df[df['label'] == src_stage][latent_cols].values.astype(np.float32)
    tgt_data = df[df['label'] == tgt_stage][latent_cols].values.astype(np.float32)

    if len(src_data) == 0 or len(tgt_data) == 0:
        return None

    # Compute stage means (raw)
    src_mean = src_data.mean(axis=0)
    tgt_mean = tgt_data.mean(axis=0)

    # Transition magnitude per dimension
    transition_magnitude = np.abs(tgt_mean - src_mean)

    # Prediction error per dimension
    n_samples = min(100, len(src_data))
    dim_errors = []

    with torch.no_grad():
        for i in range(n_samples):
            # Get sample
            src_vec = src_data[i]

            # Find nearest target (ground truth)
            dists = np.linalg.norm(tgt_data - src_vec, axis=1)
            nearest_idx = np.argmin(dists)
            tgt_vec = tgt_data[nearest_idx]

            # Scale and predict
            src_scaled = scaler.transform(src_vec.reshape(1, -1)).astype(np.float32)
            src_t = torch.tensor(src_scaled).unsqueeze(1).to(device)

            def get_dummy_time_features(batch_size, seq_len, device):
                x_mark = torch.zeros(batch_size, seq_len, 5).to(device)
                return x_mark

            x_enc = src_t
            x_dec = src_t.clone()
            x_mark_enc = get_dummy_time_features(1, 1, device)
            x_mark_dec = get_dummy_time_features(1, 1, device)

            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred_scaled = output.squeeze().cpu().numpy()
            pred = scaler.inverse_transform(pred_scaled.reshape(1, -1)).flatten()

            # Per-dimension error
            error = (pred - tgt_vec) ** 2
            dim_errors.append(error)

    dim_errors = np.stack(dim_errors).mean(axis=0)

    # Combined importance: high transition + low error = important & well-modeled
    # Normalize
    trans_norm = transition_magnitude / (transition_magnitude.max() + 1e-8)
    error_norm = dim_errors / (dim_errors.max() + 1e-8)

    # High transition, low error is best
    importance = trans_norm * (1 - error_norm * 0.3)

    return {
        'transition_magnitude': transition_magnitude,
        'prediction_error': dim_errors,
        'combined_importance': importance
    }


def trace_genes_for_dimension(vae_importance_matrix, dim_idx, gene_names, method=None, std_threshold=None, elbow_sensitivity=None):
    """
    Get significant genes for a specific latent dimension.

    Supports two methods:
    1. 'std': Statistical threshold (> mean + N*std) - rigorous, interpretable
    2. 'elbow': Elbow/Kneedle method - data-driven, finds natural cutoff in power-law distribution

    Args:
        vae_importance_matrix: [n_genes, n_dims] importance matrix from Integrated Gradients
        dim_idx: which latent dimension to analyze
        gene_names: list of gene names
        method: 'std' or 'elbow' (if None, uses global GENE_SELECTION_METHOD)
        std_threshold: Number of standard deviations above mean (if None, uses GENE_SELECTION_STD_THRESHOLD)
        elbow_sensitivity: Sensitivity parameter S for Kneedle (if None, uses ELBOW_SENSITIVITY)

    Returns:
        Tuple: (list of dicts with gene info, n_selected, threshold_value, method_used)
    """
    if method is None:
        method = GENE_SELECTION_METHOD

    dim_importance = vae_importance_matrix[:, dim_idx]

    # Sort indices by importance (descending)
    sorted_indices = np.argsort(dim_importance)[::-1]
    sorted_importance = dim_importance[sorted_indices]

    n_genes_selected = 0
    threshold_value = None
    method_used = method

    if method == 'std':
        # Statistical threshold method
        if std_threshold is None:
            std_threshold = GENE_SELECTION_STD_THRESHOLD

        mean_imp = np.mean(dim_importance)
        std_imp = np.std(dim_importance)
        threshold_value = mean_imp + std_threshold * std_imp

        # Select genes above threshold
        significant_mask = dim_importance > threshold_value
        selected_indices_all = np.where(significant_mask)[0]

        # Sort by importance (descending)
        selected_indices_all = selected_indices_all[np.argsort(dim_importance[selected_indices_all])[::-1]]

        n_genes_selected = len(selected_indices_all)
        selected_indices = selected_indices_all

    elif method == 'elbow':
        # Elbow/Kneedle method
        if not KNEED_AVAILABLE:
            print(f"    Warning: kneed not available, falling back to std method")
            return trace_genes_for_dimension(vae_importance_matrix, dim_idx, gene_names, 'std', std_threshold, None)

        if elbow_sensitivity is None:
            elbow_sensitivity = ELBOW_SENSITIVITY

        try:
            # X-axis: gene rank (0, 1, 2, ...)
            x = np.arange(len(sorted_importance))

            # Find the elbow point using Kneedle algorithm
            kneedle = KneeLocator(
                x,
                sorted_importance,
                curve='convex',
                direction='decreasing',
                S=elbow_sensitivity
            )

            if kneedle.elbow is not None and kneedle.elbow > 0:
                n_genes_selected = kneedle.elbow
                threshold_value = sorted_importance[kneedle.elbow]
                selected_indices = sorted_indices[:n_genes_selected]
            else:
                # Fallback: use > mean + 1 std
                mean_imp = np.mean(dim_importance)
                std_imp = np.std(dim_importance)
                threshold_value = mean_imp + 1.0 * std_imp
                significant_mask = dim_importance > threshold_value
                selected_indices_all = np.where(significant_mask)[0]
                selected_indices_all = selected_indices_all[np.argsort(dim_importance[selected_indices_all])[::-1]]
                n_genes_selected = len(selected_indices_all)
                selected_indices = selected_indices_all
                method_used = 'std_fallback'

        except Exception as e:
            print(f"    Warning: Elbow detection failed ({e}), falling back to std method")
            return trace_genes_for_dimension(vae_importance_matrix, dim_idx, gene_names, 'std', std_threshold, None)

    else:
        # Invalid method - fallback to std
        print(f"    Warning: Unknown method '{method}', using std method")
        return trace_genes_for_dimension(vae_importance_matrix, dim_idx, gene_names, 'std', std_threshold, None)

    results = []
    for rank, gene_idx in enumerate(selected_indices):
        gene_name = gene_names[gene_idx] if gene_names else f"Gene_{gene_idx}"
        results.append({
            'rank': rank + 1,
            'gene': gene_name,
            'gene_idx': gene_idx,
            'importance': dim_importance[gene_idx]
        })

    return results, n_genes_selected, threshold_value, method_used


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("CASCADED INFORMER: SIGNIFICANT DIMENSIONS & GENE TRACING")
    print("="*70)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
    latent_cols = [c for c in df.columns if c.startswith('z')]
    dim = len(latent_cols)
    print(f"Data shape: {df.shape}, Latent dims: {dim}")

    # Load gene names
    gene_names = None
    if os.path.exists(GENE_LIST_PATH):
        with open(GENE_LIST_PATH, 'r') as f:
            gene_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(gene_names)} gene names")
    else:
        print("No gene names file - will use indices")

    # Load cascaded models
    print(f"\nLoading cascaded Informer models from {CASCADED_DIR}...")
    models, scalers = load_cascaded_models(CASCADED_DIR, dim, device)

    # Load VAE models for gene tracing
    print(f"\nLoading VAE models from {VAE_DIR}...")
    vae_models = find_vae_models(VAE_DIR)
    print(f"Found {len(vae_models)} VAE models")

    # Analyze each transition
    print("\n" + "="*70)
    print("ANALYZING STAGE TRANSITIONS")
    print("="*70)

    all_results = {}
    transition_significant_dims = {}

    for i, ((trans_name, model), scaler) in enumerate(zip(models, scalers)):
        src, tgt = TRANSITIONS[i]
        print(f"\n[{i+1}/{len(models)}] {src} → {tgt}")

        analysis = analyze_transition_importance(model, scaler, df, src, tgt, latent_cols, device)

        if analysis is None:
            continue

        all_results[trans_name] = analysis

        # Statistical selection: dimensions with importance > mean + 1 std
        importance = analysis['combined_importance']
        mean_imp = np.mean(importance)
        std_imp = np.std(importance)
        threshold = mean_imp + 1.0 * std_imp

        # Select dimensions above threshold
        significant_mask = importance > threshold
        top_dims = np.where(significant_mask)[0]

        # Sort by importance (descending)
        top_dims = top_dims[np.argsort(importance[top_dims])[::-1]]
        transition_significant_dims[trans_name] = top_dims

        print(f"  Threshold (mean + 1 std): {threshold:.4f}")
        print(f"  Significant dimensions (n={len(top_dims)}): {['z'+str(d+1) for d in top_dims]}")
        trans_mags = [f'{analysis["transition_magnitude"][d]:.4f}' for d in top_dims]
        print(f"  Transition magnitudes: {trans_mags}")

    # Analyze significant dimensions SEPARATELY for each transition
    print("\n" + "="*70)
    print("DIMENSION SIGNIFICANCE PER TRANSITION (SEPARATE RANKING)")
    print("="*70)

    # Save separate dimension significance for each transition
    for trans_name, analysis in all_results.items():
        print(f"\n--- {trans_name} ---")

        # Rank dimensions by combined importance for THIS transition only
        trans_importance = analysis['combined_importance']
        trans_ranks = np.argsort(trans_importance)[::-1]

        # Statistical threshold for this transition
        mean_imp = np.mean(trans_importance)
        std_imp = np.std(trans_importance)
        threshold = mean_imp + 1.0 * std_imp

        # Create results for this transition
        trans_results = []
        for d in range(dim):
            rank_in_list = list(trans_ranks).index(d) + 1
            is_significant = trans_importance[d] > threshold
            trans_results.append({
                'dimension': f'z{d+1}',
                'dim_idx': d,
                'importance_score': trans_importance[d],
                'rank': rank_in_list,
                'is_significant': is_significant,
                'threshold': threshold,
                'transition_magnitude': analysis['transition_magnitude'][d],
                'prediction_error': analysis['prediction_error'][d]
            })

        df_trans = pd.DataFrame(trans_results).sort_values('rank')
        trans_file = os.path.join(OUTPUT_DIR, f'dimension_significance_{trans_name}.csv')
        df_trans.to_csv(trans_file, index=False)

        sig_dims = df_trans[df_trans['is_significant']]['dimension'].tolist()
        print(f"  Threshold (mean + 1 std): {threshold:.4f}")
        print(f"  Significant dimensions (n={len(sig_dims)}): {sig_dims}")
        print(f"  Saved: {trans_file}")

    # Also save a combined summary file with per-transition ranks and significance
    print("\n--- Combined Summary (with per-transition ranks) ---")
    dim_results = []
    for d in range(dim):
        row = {
            'dimension': f'z{d+1}',
            'dim_idx': d,
        }
        for trans_name, analysis in all_results.items():
            trans_importance = analysis['combined_importance']
            trans_ranks = np.argsort(trans_importance)[::-1]
            rank_in_list = list(trans_ranks).index(d) + 1

            # Statistical threshold
            mean_imp = np.mean(trans_importance)
            std_imp = np.std(trans_importance)
            threshold = mean_imp + 1.0 * std_imp
            is_sig = trans_importance[d] > threshold

            row[f'{trans_name}_importance'] = trans_importance[d]
            row[f'{trans_name}_rank'] = rank_in_list
            row[f'{trans_name}_significant'] = is_sig
            row[f'{trans_name}_magnitude'] = analysis['transition_magnitude'][d]
            row[f'{trans_name}_error'] = analysis['prediction_error'][d]
        dim_results.append(row)

    df_dims = pd.DataFrame(dim_results)
    df_dims.to_csv(os.path.join(OUTPUT_DIR, 'dimension_significance.csv'), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/dimension_significance.csv")

    # Load breast cancer gene lists for validation
    bc_genes = set()
    core_genes = set()

    bc_path = "datasets_breast/bc_gene_list.txt"
    core_path = "datasets_breast/coregene307.txt"

    if os.path.exists(bc_path):
        with open(bc_path, 'r') as f:
            for line in f:
                gene = line.strip().split('\t')[0] if '\t' in line else line.strip()
                if gene:
                    bc_genes.add(gene.upper())
        print(f"Loaded {len(bc_genes)} breast cancer genes")

    if os.path.exists(core_path):
        with open(core_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                core_genes.update([g.strip().upper() for g in parts if g.strip()])
        print(f"Loaded {len(core_genes)} core genes")

    # Trace genes using VAE decoder with GRADIENT method
    if vae_models:
        print("\n" + "="*70)
        print("TRACING LATENT DIMENSIONS TO GENES (Gradient-Based Method)")
        print("="*70)
        print("\nThis properly accounts for non-linear activations in the VAE decoder.")

        # Compute gradient-based importance for each stage's VAE
        vae_importance_matrices = []
        input_dim = None

        for stage, model_path in vae_models.items():
            print(f"\n  Processing VAE for stage: {stage}")
            state_dict = torch.load(model_path, map_location=device, weights_only=True)

            # Get input dim
            decoder_keys = [k for k in state_dict.keys() if 'decoder' in k and 'weight' in k]
            if decoder_keys:
                dims_list = [(k, state_dict[k].shape[0]) for k in decoder_keys]
                output_layer = max(dims_list, key=lambda x: x[1])
                if input_dim is None:
                    input_dim = output_layer[1]

            # Load model
            model = ConditionalVAE(
                input_dim=input_dim,
                latent_dim=dim,
                hidden_dims=HIDDEN_DIMS,
                n_classes=0
            )
            model.load_state_dict(state_dict)
            model.to(device)

            # Compute gradient-based importance
            importance = compute_gradient_gene_importance(model, device, n_samples=5)
            vae_importance_matrices.append(importance)

        # Average across stages
        avg_vae_importance = np.mean(vae_importance_matrices, axis=0)
        print(f"\nVAE importance matrix shape: {avg_vae_importance.shape}")

        # ====================================================================
        # TRACE GENES PER TRANSITION (Not Aggregated)
        # ====================================================================
        print("\n" + "="*70)
        print("TRACING GENES PER TRANSITION")
        print("="*70)

        if GENE_SELECTION_METHOD == 'std':
            print(f"Gene selection method: Statistical threshold (> mean + {GENE_SELECTION_STD_THRESHOLD}*std)")
        elif GENE_SELECTION_METHOD == 'elbow':
            print(f"Gene selection method: Elbow/Kneedle (S={ELBOW_SENSITIVITY})")
        else:
            print(f"Gene selection method: {GENE_SELECTION_METHOD}")

        all_transition_genes = {}

        for trans_name, analysis in all_results.items():
            print(f"\n{'─'*60}")
            print(f"TRANSITION: {trans_name.replace('_to_', ' → ')}")
            print(f"{'─'*60}")

            # Statistical selection: dimensions with importance > mean + 1 std
            importance = analysis['combined_importance']
            mean_imp = np.mean(importance)
            std_imp = np.std(importance)
            threshold = mean_imp + 1.0 * std_imp

            significant_mask = importance > threshold
            top_dims_for_trans = np.where(significant_mask)[0]
            top_dims_for_trans = top_dims_for_trans[np.argsort(importance[top_dims_for_trans])[::-1]]

            print(f"Top dimensions for this transition (>{mean_imp:.3f}+{std_imp:.3f}): {['z'+str(d+1) for d in top_dims_for_trans]}")

            transition_genes = []

            for dim_idx in top_dims_for_trans:
                print(f"\n  Dimension z{dim_idx+1}:")
                genes, n_selected, threshold_val, method_used = trace_genes_for_dimension(
                    avg_vae_importance, dim_idx, gene_names, method=GENE_SELECTION_METHOD
                )

                if method_used == 'std':
                    threshold_info = f" (threshold={threshold_val:.4f}, n={n_selected} genes)" if threshold_val else f" (n={n_selected} genes)"
                    print(f"    Selected {n_selected} genes via statistical method (> mean + {GENE_SELECTION_STD_THRESHOLD}*std){threshold_info}")
                elif method_used == 'elbow':
                    threshold_info = f" (elbow at rank {n_selected}, cutoff={threshold_val:.4f})" if threshold_val else f" (elbow at rank {n_selected})"
                    print(f"    Selected {n_selected} genes via elbow method (S={ELBOW_SENSITIVITY}){threshold_info}")
                else:
                    print(f"    Selected {n_selected} genes via {method_used} method")

                for g in genes[:10]:  # Print top 10
                    gene_upper = g['gene'].upper() if g['gene'] else ''
                    in_bc = gene_upper in bc_genes
                    in_core = gene_upper in core_genes
                    marker = " ★" if in_bc else (" ●" if in_core else "")
                    print(f"    {g['rank']:2d}. {g['gene']:15s} ({g['importance']:.4f}){marker}")

                # Add ALL selected genes (not just top 10) to results
                for g in genes:
                    gene_upper = g['gene'].upper() if g['gene'] else ''
                    in_bc = gene_upper in bc_genes
                    in_core = gene_upper in core_genes

                    transition_genes.append({
                        'transition': trans_name,
                        'dimension': f'z{dim_idx+1}',
                        'rank': g['rank'],
                        'gene': g['gene'],
                        'importance': g['importance'],
                        'in_bc': in_bc,
                        'in_core': in_core,
                        'n_selected': n_selected
                    })

            all_transition_genes[trans_name] = transition_genes

        # Save per-transition gene results
        all_trans_df = []
        for trans_name, genes in all_transition_genes.items():
            df_trans = pd.DataFrame(genes)
            all_trans_df.append(df_trans)

            # Save individual transition file
            safe_name = trans_name.replace(' ', '_')
            df_trans.to_csv(os.path.join(OUTPUT_DIR, f'genes_{safe_name}.csv'), index=False)
            print(f"\nSaved: {OUTPUT_DIR}/genes_{safe_name}.csv")

        # Combined file
        df_all_trans = pd.concat(all_trans_df, ignore_index=True)
        df_all_trans.to_csv(os.path.join(OUTPUT_DIR, 'genes_per_transition_all.csv'), index=False)
        print(f"Saved: {OUTPUT_DIR}/genes_per_transition_all.csv")

        # BC genes per transition
        df_bc_trans = df_all_trans[df_all_trans['in_bc'] | df_all_trans['in_core']]
        if len(df_bc_trans) > 0:
            df_bc_trans.to_csv(os.path.join(OUTPUT_DIR, 'bc_genes_per_transition.csv'), index=False)
            print(f"Saved: {OUTPUT_DIR}/bc_genes_per_transition.csv ({len(df_bc_trans)} BC-related genes)")

        # ====================================================================
        # ALSO TRACE FOR OVERALL TOP DIMENSIONS (for comparison)
        # ====================================================================
        print("\n" + "="*70)
        print("OVERALL AGGREGATED DIMENSIONS (for comparison)")
        print("="*70)

        # Compute overall top dims by summing importance across all transitions
        dim_scores = np.zeros(dim)
        for trans_name, analysis in all_results.items():
            dim_scores += analysis['combined_importance']

        # Statistical selection: dimensions with aggregated score > mean + 1 std
        mean_score = np.mean(dim_scores)
        std_score = np.std(dim_scores)
        threshold = mean_score + 1.0 * std_score

        significant_mask = dim_scores > threshold
        overall_top_dims = np.where(significant_mask)[0]
        overall_top_dims = overall_top_dims[np.argsort(dim_scores[overall_top_dims])[::-1]]

        print(f"Threshold (mean + 1 std): {threshold:.4f}")
        print(f"Significant dimensions (n={len(overall_top_dims)}): {['z'+str(d+1) for d in overall_top_dims]}")

        # Trace genes for significant dimensions
        all_gene_results = []

        for dim_idx in overall_top_dims:  # All significant dimensions
            print(f"\nDimension z{dim_idx+1}:")
            genes, n_selected, threshold_val, method_used = trace_genes_for_dimension(
                avg_vae_importance, dim_idx, gene_names, method=GENE_SELECTION_METHOD
            )

            if method_used == 'std':
                threshold_info = f" (threshold={threshold_val:.4f})" if threshold_val else ""
                print(f"  Statistical method (> mean + {GENE_SELECTION_STD_THRESHOLD}*std) selected {n_selected} genes{threshold_info}")
            elif method_used == 'elbow':
                threshold_info = f" (elbow at rank {n_selected}, cutoff={threshold_val:.4f})" if threshold_val else f" (elbow at rank {n_selected})"
                print(f"  Elbow method (S={ELBOW_SENSITIVITY}) selected {n_selected} genes{threshold_info}")
            else:
                print(f"  {method_used} method selected {n_selected} genes")

            for g in genes[:10]:  # Print top 10
                gene_upper = g['gene'].upper() if g['gene'] else ''
                in_bc = gene_upper in bc_genes
                in_core = gene_upper in core_genes
                marker = " ★" if in_bc else (" ●" if in_core else "")
                print(f"  {g['rank']:2d}. {g['gene']:15s} ({g['importance']:.4f}){marker}")

            # Add ALL selected genes (not just top 10) to results
            for g in genes:
                gene_upper = g['gene'].upper() if g['gene'] else ''
                in_bc = gene_upper in bc_genes
                in_core = gene_upper in core_genes

                all_gene_results.append({
                    'dimension': f'z{dim_idx+1}',
                    'rank': g['rank'],
                    'gene': g['gene'],
                    'importance': g['importance'],
                    'in_bc': in_bc,
                    'in_core': in_core,
                    'elbow_n': n_selected
                })

        # Save gene results
        df_genes = pd.DataFrame(all_gene_results)
        df_genes.to_csv(os.path.join(OUTPUT_DIR, 'cascaded_top_genes.csv'), index=False)
        print(f"\nSaved: {OUTPUT_DIR}/cascaded_top_genes.csv")

        # Save BC-related genes only
        df_bc_genes = df_genes[df_genes['in_bc'] | df_genes['in_core']]
        if len(df_bc_genes) > 0:
            df_bc_genes.to_csv(os.path.join(OUTPUT_DIR, 'cascaded_bc_genes.csv'), index=False)
            print(f"Saved: {OUTPUT_DIR}/cascaded_bc_genes.csv ({len(df_bc_genes)} BC-related genes)")

        # Create heatmap (use all significant dimensions, or limit to 10 if too many)
        dims_for_heatmap = overall_top_dims[:10] if len(overall_top_dims) > 10 else overall_top_dims
        plot_gene_heatmap(avg_vae_importance, dims_for_heatmap, gene_names, OUTPUT_DIR)

        # Create per-transition heatmap
        plot_per_transition_heatmap(avg_vae_importance, all_results, gene_names, OUTPUT_DIR)

    # Visualization
    plot_transition_analysis(all_results, dim, OUTPUT_DIR)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")

    # Compute final summary of overall significant dimensions
    dim_scores_final = np.zeros(dim)
    for trans_name, analysis in all_results.items():
        dim_scores_final += analysis['combined_importance']

    mean_score_final = np.mean(dim_scores_final)
    std_score_final = np.std(dim_scores_final)
    threshold_final = mean_score_final + 1.0 * std_score_final

    significant_mask_final = dim_scores_final > threshold_final
    overall_top_dims_final = np.where(significant_mask_final)[0]
    overall_top_dims_final = overall_top_dims_final[np.argsort(dim_scores_final[overall_top_dims_final])[::-1]]

    # Print summary
    print("\n*** STATISTICALLY SIGNIFICANT DIMENSIONS (> mean + 1 std) ***")
    print(f"Selection method: importance > {mean_score_final:.4f} + {std_score_final:.4f} = {threshold_final:.4f}")
    print(f"Number of significant dimensions: {len(overall_top_dims_final)}")
    for i, d in enumerate(overall_top_dims_final):
        print(f"  {i+1}. z{d+1} (score: {dim_scores_final[d]:.4f})")


def plot_gene_heatmap(importance_matrix, significant_dims, gene_names, output_dir):
    """Create heatmap of gene importance for significant dimensions."""

    # Get top genes across significant dimensions
    combined_importance = importance_matrix[:, significant_dims].sum(axis=1)
    top_gene_indices = np.argsort(combined_importance)[::-1][:25]

    subset = importance_matrix[top_gene_indices][:, significant_dims]

    if gene_names:
        gene_labels = [gene_names[i] for i in top_gene_indices]
    else:
        gene_labels = [f"Gene_{i}" for i in top_gene_indices]

    dim_labels = [f"z{d+1}" for d in significant_dims]

    plt.figure(figsize=(10, 12))
    sns.heatmap(subset, cmap='YlOrRd',
                xticklabels=dim_labels,
                yticklabels=gene_labels,
                cbar_kws={'label': 'Gradient Importance'})
    plt.xlabel('Latent Dimension', fontsize=12)
    plt.ylabel('Gene', fontsize=12)
    plt.title('Top Genes Contributing to Significant Latent Dimensions\n(Gradient-Based Analysis)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_importance_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/gene_importance_heatmap.png")


def plot_per_transition_heatmap(importance_matrix, all_results, gene_names, output_dir):
    """Create a heatmap showing top genes per transition."""

    n_trans = len(all_results)
    n_cols = 3
    n_rows = (n_trans + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)
    axes = axes.flatten()

    for idx, (trans_name, analysis) in enumerate(all_results.items()):
        ax = axes[idx]

        # Get top 5 dimensions for this transition
        top_dims = np.argsort(analysis['combined_importance'])[::-1][:5]

        # Get top 15 genes across these dimensions
        combined_importance = importance_matrix[:, top_dims].sum(axis=1)
        top_gene_indices = np.argsort(combined_importance)[::-1][:15]

        subset = importance_matrix[top_gene_indices][:, top_dims]

        if gene_names:
            gene_labels = [gene_names[i] for i in top_gene_indices]
        else:
            gene_labels = [f"Gene_{i}" for i in top_gene_indices]

        dim_labels = [f"z{d+1}" for d in top_dims]

        sns.heatmap(subset, cmap='YlOrRd', ax=ax,
                    xticklabels=dim_labels,
                    yticklabels=gene_labels,
                    cbar_kws={'label': 'Importance'})
        ax.set_xlabel('Latent Dimension', fontsize=10)
        ax.set_ylabel('Gene', fontsize=10)
        ax.set_title(f'{trans_name.replace("_to_", " → ")}', fontsize=12, fontweight='bold')

    for idx in range(n_trans, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle('Top Genes per Stage Transition\n(Gradient-Based Analysis)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genes_per_transition_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/genes_per_transition_heatmap.png")


def plot_transition_analysis(all_results, dim, output_dir):
    """Plot transition analysis results - one plot per transition."""

    n_trans = len(all_results)

    # Create separate plots for each transition
    for trans_idx, (trans_name, analysis) in enumerate(all_results.items()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Transition Analysis: {trans_name.replace("_to_", " → ")}',
                     fontsize=16, fontweight='bold')

        # Get significant dimensions for this transition
        importance = analysis['combined_importance']
        mean_imp = np.mean(importance)
        std_imp = np.std(importance)
        threshold = mean_imp + 1.0 * std_imp
        significant_mask = importance > threshold
        sig_dims = np.where(significant_mask)[0]

        # Plot 1: Transition magnitudes
        ax = axes[0, 0]
        bar_colors = ['red' if i in sig_dims else 'steelblue' for i in range(dim)]
        ax.bar(range(dim), analysis['transition_magnitude'], color=bar_colors, alpha=0.7)
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Transition Magnitude')
        ax.set_title('Transition Magnitude per Dimension\n(Red = Significant)')
        ax.set_xticks(range(0, dim, 4))
        ax.set_xticklabels([f'z{i+1}' for i in range(0, dim, 4)])
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Prediction errors
        ax = axes[0, 1]
        ax.plot(analysis['prediction_error'], marker='o', markersize=4, color='orange', linewidth=2)
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Prediction Error (MSE)')
        ax.set_title('Prediction Error per Dimension')
        ax.set_xticks(range(0, dim, 4))
        ax.set_xticklabels([f'z{i+1}' for i in range(0, dim, 4)])
        ax.grid(True, alpha=0.3)

        # Plot 3: Combined importance for THIS transition
        ax = axes[1, 0]
        bar_colors = ['red' if i in sig_dims else 'steelblue' for i in range(dim)]
        ax.bar(range(dim), analysis['combined_importance'], color=bar_colors, alpha=0.7)
        ax.axhline(y=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold (mean+1σ={threshold:.3f})')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Combined Importance')
        ax.set_title(f'Combined Importance (n={len(sig_dims)} significant dims)')
        ax.set_xticks(range(0, dim, 4))
        ax.set_xticklabels([f'z{i+1}' for i in range(0, dim, 4)])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Importance vs Error scatter
        ax = axes[1, 1]
        scatter_colors = ['red' if i in sig_dims else 'steelblue' for i in range(dim)]
        ax.scatter(analysis['prediction_error'], analysis['combined_importance'],
                  c=scatter_colors, s=100, alpha=0.6, edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Prediction Error (MSE)')
        ax.set_ylabel('Combined Importance')
        ax.set_title('Importance vs Error\n(Red = Significant dimensions)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with transition name
        safe_name = trans_name.replace(' ', '_')
        output_path = os.path.join(output_dir, f'transition_analysis_{safe_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    print(f"Saved {n_trans} transition analysis plots")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Historical decoder IG analysis; explicit gene-selection rule")
    parser.add_argument("--gene-selection", choices=["std", "elbow"], default="std")
    parser.add_argument("--elbow-sensitivity", type=float, default=3.0)
    options = parser.parse_args()
    GENE_SELECTION_METHOD = options.gene_selection
    ELBOW_SENSITIVITY = options.elbow_sensitivity
    if not CAPTUM_AVAILABLE:
        raise RuntimeError("Install captum: refusing to silently replace IG with ordinary gradients")
    if GENE_SELECTION_METHOD == "elbow" and not KNEED_AVAILABLE:
        raise RuntimeError("Install kneed: refusing to silently replace Kneedle with a threshold")
    main()
