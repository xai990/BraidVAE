import torch as th 
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import numpy as np
from . import logger
from umap import UMAP
import pandas as pd 
import scanpy as sc
import os 
import random
# ===================== Embedding Extraction and Visualization =====================
def extract_embeddings(model, 
                      data_loader,
                      device='cuda' if th.cuda.is_available() else 'cpu',
                      dataset=None):
    """
    Extract latent embeddings from trained VAE

    Returns:
        embeddings: np.ndarray (n_samples, latent_dim)
        names: list of sample names
        labels: np.ndarray (encoded labels)
    """
    model = model.to(device)
    model.eval()
    
    embeddings = []
    all_labels = []
    all_names = []

    # try to fetch dataset indices from the DataLoader
    if hasattr(data_loader, 'dataset'):
        dataset = data_loader.dataset

    with th.no_grad():
        # iterate over batches
        sample_counter = 0
        for batch in data_loader:
            if len(batch) == 2:
                data, labels = batch
                data = data.to(device)
                all_labels.append(labels.cpu().numpy())
            else:
                data = batch.to(device)
                labels = None

            # get latent representation
            if model.is_conditional and len(batch) == 2:
                labels = labels.to(device)
                c_onehot = F.one_hot(labels, num_classes=model.n_classes).float()
                mu, _ = model.encode(data, c_onehot)
            else:
                mu, _ = model.encode(data)

            embeddings.append(mu.cpu().numpy())

            # assign names from AnnData obs if available
            n_batch = data.shape[0]
            if dataset is not None and hasattr(dataset, "obs_names"):
                names = dataset.obs_names[sample_counter : sample_counter + n_batch]
                all_names.extend(names)
            else:
                all_names.extend([f"sample_{sample_counter + j}" for j in range(n_batch)])
            sample_counter += n_batch

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else None
    return embeddings, np.array(all_names), labels


def visualize_embeddings(embeddings, 
                        labels= None,
                        label_names= None,
                        save_path = None,
                        coord_save_path = None):
    """
    Visualize embeddings using UMAP or t-SNE
    
    Args:
        embeddings: Latent embeddings
        labels: Class labels
        label_names: Names of classes
        method: Visualization method ('umap' or 'tsne')
        save_path: Path to save figure
        coord_save_path: Path to save the umap coordinates
    """
   
    
    
    # Reduce to 2D

    reducer = UMAP(n_neighbors=30, min_dist=0.2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    if coord_save_path:
        df = pd.DataFrame(embeddings_2d, columns=['UMAP_1', 'UMAP_2'])
        if labels is not None:
            df['label'] = labels
        df.to_csv(coord_save_path, index=False)
        print(f"Coordinates saved to {coord_save_path}")
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Define consistent stage color mapping
    STAGE_COLORS = {
        'Normal': '#1f77b4',      # blue
        'Stage I': '#ff7f0e',     # orange
        'Stage IIA': '#2ca02c',   # green
        'Stage IIB': '#d62728',   # red
        'Stage III': '#9467bd',   # purple
        'Stage IIIA': '#8c564b',  # brown
        'Unknown': '#7f7f7f',     # gray
    }
    
    if labels is not None:
        unique_labels = np.unique(labels)
        
        # Use consistent stage colors if label names match stage patterns
        use_stage_colors = False
        if label_names is not None:
            use_stage_colors = any(name in STAGE_COLORS for name in label_names)
        
        if use_stage_colors and label_names is not None:
            for i, label in enumerate(unique_labels):
                mask = labels == label
                label_name = label_names[i] if label_names is not None else str(label)
                color = STAGE_COLORS.get(label_name, '#7f7f7f')
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[color], label=label_name, alpha=0.6, s=20)
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                label_name = label_names[i] if label_names is not None else str(label)
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=label_name, alpha=0.6, s=20)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
    
    plt.title(f'VAE Latent Space Visualization (UMAP)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.axis('off')
    plt.tight_layout()
    logger.log("The plot process is finished")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



def get_active_latent_dims(
    df_all: pd.DataFrame,
    outdir: str,
    activity_threshold: float = 0.4,
    plot_threshold_line: float = 0.4,
    plot: bool = True,
):
    """
    Compute latent dimension activity and return the active latent dims.

    Args:
        df_all: DataFrame containing latent dimensions as columns z1, z2, ...
        outdir: Directory where outputs (TSV + PNG) will be saved.
        activity_threshold: Variance cutoff to count a latent dim as "active".
        plot_threshold_line: Horizontal line (variance) shown on the plot.
        plot: Whether to generate a PNG barplot of variance per latent dim.

    Returns:
        List of active latent dimension names, e.g. ["z1", "z3", ...].
        If no latent columns found, returns [].
    """
    # Only keep latent columns z1, z2, ...
    latent_values = df_all.filter(regex=r"^z\d+").values
    if latent_values.size == 0:
        logger.log("[warn] No latent columns (z1, z2, ...) found in df_all.")
        return []

    latent_var = np.var(latent_values, axis=0)
    mean_var = float(np.mean(latent_var))

    all_latent_names = [f"z{i+1}" for i in range(len(latent_var))]
    active_mask = latent_var > activity_threshold
    active_latent_dims = [name for name, flag in zip(all_latent_names, active_mask) if flag]

    logger.log(f"[Latent Summary] Mean variance across dims: {mean_var:.6f}")
    logger.log(
        f"[Latent Summary] Active latent dims (>{activity_threshold} var): "
        f"{len(active_latent_dims)} / {len(all_latent_names)}"
    )
    logger.log(f"[Latent Summary] Active dims: {active_latent_dims}")

    # Save variance profile
    # var_df = pd.DataFrame({
    #     "latent_dim": all_latent_names,
    #     "variance": latent_var,
    #     "active": active_mask.astype(int),
    # })
    # var_path = os.path.join(outdir, "latent_activity.tsv")
    # var_df.to_csv(var_path, sep="\t", index=False)
    # logger.log(f"Saved latent dimension activity to {var_path}")

    # Optional plot
    if plot:
        try:
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(latent_var)), latent_var)
            plt.axhline(
                y=plot_threshold_line,
                color="red",
                linestyle="--",
                label="activity threshold"
            )
            plt.xlabel("Latent Dimension Index")
            plt.ylabel("Variance of μ(x)")
            plt.title("Latent Dimension Activity (VAE)")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(outdir, "latent_activity.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.log(f"Saved latent activity plot to {plot_path}")
        except Exception as e:
            logger.log(f"[warn] Could not generate latent activity plot: {e}")

    return active_latent_dims


def origin_plot(
    adata,
    color= "label",
    save_path: str = "originplot.png",
    n_pcs: int = 50,
    n_neighbors: int = 30,
    min_dist: float = 0.2,
):
    """
    Compute PCA + UMAP on `adata` and save a UMAP plot.

    Args:
        adata: AnnData object.
        color: One or more keys from `adata.obs` to color by (e.g. "label").
               If None, no color is used.
        save_path: Output path for the PNG file.
        n_pcs: Number of principal components to use.
        n_neighbors: Number of neighbors for the KNN graph.
        min_dist: UMAP min_dist parameter.
    """
    # PCA
    sc.tl.pca(adata, n_comps=n_pcs)

    # Neighbors
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")

    # UMAP embedding
    sc.tl.umap(adata, min_dist=min_dist)

    # UMAP plot (no blocking, so we can save with matplotlib)
    sc.pl.umap(
        adata,
        color=color,
        show=False,
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close()


def seed_everything(seed=42):
    """Locks all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    
    # These two lines force predictable (but slower) algorithms on the GPU
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed}")