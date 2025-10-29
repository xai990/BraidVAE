import torch as th 
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import numpy as np
from . import logger

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
                        save_path = None):
    """
    Visualize embeddings using UMAP or t-SNE
    
    Args:
        embeddings: Latent embeddings
        labels: Class labels
        label_names: Names of classes
        method: Visualization method ('umap' or 'tsne')
        save_path: Path to save figure
    """
   
    from umap import UMAP
    
    # Reduce to 2D

    reducer = UMAP(n_neighbors=50, min_dist=0.2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
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


