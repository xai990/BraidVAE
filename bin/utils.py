import torch as th 
import matplotlib.pyplot as plt
import seaborn as sns
# ===================== Embedding Extraction and Visualization =====================
def extract_embeddings(model, 
                      data_loader,
                      device= 'cuda' if th.cuda.is_available() else 'cpu'):
    """
    Extract latent embeddings from trained VAE
    
    Args:
        model: Trained VAE model
        data_loader: Data loader
        device: Device to use
    
    Returns:
        embeddings: Latent embeddings
        labels: Corresponding labels
    """
    model = model.to(device)
    model.eval()
    
    embeddings = []
    all_labels = []
    
    with th.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                data, labels = batch
                data = data.to(device)
                all_labels.append(labels.numpy())
            else:
                data = batch.to(device)
            
            # Get latent representation
            if model.is_conditional and len(batch) == 2:
                labels = labels.to(device)
                c_onehot = F.one_hot(labels, num_classes=model.n_classes).float()
                mu, _ = model.encode(data, c_onehot)
            else:
                mu, _ = model.encode(data)
            
            embeddings.append(mu.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    if all_labels:
        labels = np.concatenate(all_labels, axis=0)
    else:
        labels = None
    
    return embeddings, labels

def visualize_embeddings(embeddings, 
                        labels= None,
                        label_names= None,
                        method = 'umap',
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
    from sklearn.manifold import TSNE
    from umap import UMAP
    
    # Reduce to 2D
    if method == 'umap':
        reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names[label] if label_names is not None else f"Stage {label}"
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label_name, alpha=0.6, s=20)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
    
    plt.title(f'VAE Latent Space Visualization ({method.upper()})')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


