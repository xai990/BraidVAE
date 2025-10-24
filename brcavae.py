import pandas as pd 
import torch as th 
import scanpy as sc
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from bin.utils import extract_embeddings,visualize_embeddings
from bin.model import ConditionalVAE, VAELoss,train_vae_with_test_set

class GeneExpressionDataset(Dataset):
    """Dataset wrapper for gene expression data with labels"""
    
    def __init__(self, adata, use_labels= True):
        """
        Args:
            adata: AnnData object containing expression data and labels
            use_labels: Whether to return labels for conditional VAE
        """
        self.data = th.FloatTensor(adata.X.copy())
        self.use_labels = use_labels
        
        if use_labels and 'label' in adata.obs.columns:
            # Encode labels as integers
            le = LabelEncoder()
            self.labels = th.LongTensor(le.fit_transform(adata.obs['label'].values))
            self.n_classes = len(le.classes_)
            self.label_encoder = le
        else:
            self.labels = None
            self.n_classes = 0
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.use_labels and self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]



def load_and_clean_data(
    data_path,
    label_path,
    unknown_pattern = "unknown_tumor_type",
    verbose= True
):
    """
    Load and clean a single dataset (train or test).
    
    Args:
        data_path: Path to gene expression data
        label_path: Path to labels file
        unknown_pattern: Pattern to identify unknown tumor types
        verbose: Whether to print statistics
    
    Returns:
        Tuple of (cleaned_expression_df, cleaned_labels_df)
    """
    # Read the data
    df = pd.read_csv(data_path, sep='\t', index_col=0)
    df_labels = pd.read_csv(label_path, sep='\t', header=None)
    df_labels.columns = ['sample_id', 'label']
    
    if verbose:
        print(f"Original data shape: {df.shape}")
        print(f"Original samples: {len(df_labels)}")
        print(f"Label distribution:\n{df_labels['label'].value_counts()}\n")
    
    # Filter unknown tumor types
    valid_mask = ~df_labels['label'].str.lower().str.contains(unknown_pattern, na=False)
    valid_mask = valid_mask & df_labels['label'].notna()  # Also remove NaN
    
    # Apply filtering to labels
    df_labels_clean = df_labels[valid_mask].reset_index(drop=True)
    
    # Filter expression data using sample IDs
    valid_sample_ids = df_labels_clean['sample_id'].values
    # Use iloc with positional indexing since sample_ids are the index
    df_clean = df.loc[df.index.isin(valid_sample_ids)]
    
    # Reset index to match labels
    df_clean = df_clean.reset_index(drop=True)
    
    # Drop NaN columns
    df_clean = df_clean.dropna(axis=1)
    
    if verbose:
        print(f"After filtering '{unknown_pattern}':")
        print(f"  Cleaned data shape: {df_clean.shape}")
        print(f"  Remaining samples: {len(df_labels_clean)}")
        print(f"  Removed samples: {len(df_labels) - len(df_labels_clean)}")
        print(f"  Label distribution:\n{df_labels_clean['label'].value_counts()}\n")
    
    return df_clean, df_labels_clean



def create_aligned_datasets(
    train_data_path,
    train_label_path,
    test_data_path,
    test_label_path,
    unknown_pattern = "unknown_tumor_type",
    n_top_genes = 2000,
    scale_max_value = 12,
    gene_list = None  # <-- ADDED: New parameter for custom gene list
):
    """
    Load and preprocess both train and test datasets with aligned features.
    
    Args:
        train_data_path: Path to training gene expression data
        train_label_path: Path to training labels
        test_data_path: Path to test gene expression data
        test_label_path: Path to test labels
        unknown_pattern: Pattern to identify unknown tumor types
        n_top_genes: Number of highly variable genes to select (used if gene_list is None)
        scale_max_value: Maximum value for scaling
        gene_list: Optional list of genes to use. If None, HVGs are calculated.
    
    Returns:
        Tuple of (train_adata, test_adata, train_loader, test_loader, train_dataset)
    """
    
    print("=" * 70)
    print("LOADING AND PREPROCESSING TRAINING DATA")
    print("=" * 70)
    
    # Load training data
    df_train, df_labels_train = load_and_clean_data(
        train_data_path, 
        train_label_path, 
        unknown_pattern
    )
    
    print("=" * 70)
    print("LOADING AND PREPROCESSING TEST DATA")
    print("=" * 70)
    
    # Load test data
    df_test, df_labels_test = load_and_clean_data(
        test_data_path, 
        test_label_path, 
        unknown_pattern
    )
    
    # Ensure both datasets have the same genes (intersection)
    common_genes = df_train.columns.intersection(df_test.columns)
    print(f"\nGene alignment:")
    print(f"  Training genes: {len(df_train.columns)}")
    print(f"  Test genes: {len(df_test.columns)}")
    print(f"  Common genes: {len(common_genes)}")
    
    # Filter to common genes
    df_train = df_train[common_genes]
    df_test = df_test[common_genes]
    
    # Create AnnData objects
    train_adata = ad.AnnData(
        X=df_train.values,
        obs=df_labels_train.copy(),
        var=pd.DataFrame(index=common_genes)
    )
    
    test_adata = ad.AnnData(
        X=df_test.values,
        obs=df_labels_test.copy(),
        var=pd.DataFrame(index=common_genes)
    )
    
    print("\n" + "=" * 70)
    print("PREPROCESSING WITH SCANPY")
    print("=" * 70)
    
    # Preprocessing for training data
    print("\nTraining data preprocessing:")
    print(f"  Initial shape: {train_adata.shape}")
    
    sc.pp.filter_cells(train_adata, min_genes=200)
    sc.pp.filter_genes(train_adata, min_cells=3)
    print(f"  After filtering: {train_adata.shape}")
    
    # Store raw counts for later
    train_adata.raw = train_adata.copy()
    # --- MODIFIED GENE SELECTION LOGIC ---
    if gene_list is None:
        print("  No gene list provided, finding highly variable genes...")
        # Identify highly variable genes on training data
        sc.pp.highly_variable_genes(train_adata, n_top_genes=n_top_genes, flavor="seurat")
        
        # Get the highly variable genes
        selected_genes = train_adata.var_names[train_adata.var["highly_variable"]].tolist()
        print(f"  Highly variable genes selected: {len(selected_genes)}")
        
        # Apply HVG selection to training data
        train_adata = train_adata[:, train_adata.var["highly_variable"]]
    
    else:
        print(f"  Using provided gene list ({len(gene_list)} genes)...")
        # Filter the provided list to genes present in the data
        available_genes = [g for g in gene_list if g in train_adata.var_names]
        print(f"  {len(available_genes)} genes from list are present in the data.")
        
        if len(available_genes) == 0:
            raise ValueError("No genes from the provided gene_list are in the dataset!")
        
        selected_genes = available_genes
        
        # Apply gene list selection to training data
        train_adata = train_adata[:, selected_genes]
    sc.pp.normalize_total(train_adata, target_sum=1e4)
    sc.pp.log1p(train_adata)  

    # Scale training data
    sc.pp.scale(train_adata, max_value=scale_max_value)
    print(f"  Final training shape: {train_adata.shape}")
    
    # Preprocessing for test data (using training data statistics)
    print("\nTest data preprocessing:")
    print(f"  Initial shape: {test_adata.shape}")
    
    # Apply same cell filtering
    sc.pp.filter_cells(test_adata, min_genes=200)
    
    # --- MODIFIED: Use 'selected_genes' (which is either HVGs or gene_list) ---
    # Filter to genes that exist in test data
    genes_in_test = [g for g in selected_genes if g in test_adata.var_names]
    print(f"  Selected genes present in test: {len(genes_in_test)}/{len(selected_genes)}")
    
    # Store raw counts
    test_adata.raw = test_adata.copy()
    
    # Normalize and log transform
    sc.pp.normalize_total(test_adata, target_sum=1e4)
    sc.pp.log1p(test_adata)
    
    # Select the same genes as training
    test_adata = test_adata[:, genes_in_test]
    # --- END MODIFICATION ---

    
    # Scale test data
    sc.pp.scale(test_adata, max_value=scale_max_value)
    print(f"  Final test shape: {test_adata.shape}")
    
    # Ensure both have same genes after all preprocessing
    if train_adata.shape[1] != test_adata.shape[1]:
        print(f"\nAligning final gene sets...")
        common_final_genes = train_adata.var_names.intersection(test_adata.var_names)
        train_adata = train_adata[:, common_final_genes]
        test_adata = test_adata[:, common_final_genes]
        print(f"  Final aligned shape - Train: {train_adata.shape}, Test: {test_adata.shape}")
    
    print("\n" + "=" * 70)
    print("CREATING DATASETS AND DATALOADERS")
    print("=" * 70)
    
    # Create datasets
    train_dataset = GeneExpressionDataset(train_adata, use_labels=True)
    test_dataset = GeneExpressionDataset(test_adata, use_labels=True)
    
    print(f"\nDataset information:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of features: {train_adata.shape[1]}")
    print(f"  Number of classes: {train_dataset.n_classes}")
    
    # Check if classes are consistent
    train_classes = set(train_adata.obs['label'].unique())
    test_classes = set(test_adata.obs['label'].unique())
    
    if train_classes != test_classes:
        print(f"\nWarning: Class mismatch between train and test!")
        print(f"  Train classes: {train_classes}")
        print(f"  Test classes: {test_classes}")
        print(f"  Train only: {train_classes - test_classes}")
        print(f"  Test only: {test_classes - train_classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if th.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if th.cuda.is_available() else False
    )
    
    print("\nDataLoaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_adata, test_adata, train_loader, test_loader, train_dataset

def main():
    """Main function to run the complete pipeline"""
    
    print("Loading and preprocessing data...")
    
    # read the data from the txt file 
    # Define file paths
    train_data = "datasets_breast/breast_train_GEM_transpose.txt"
    train_label = "datasets_breast/train_stage_labels_new_updated.txt"
    test_data = "datasets_breast/breast_test_GEM_transpose.txt"
    test_label = "datasets_breast/test_stage_labels_new_updated.txt"
    gene_list = "datasets_breast/coregene307.txt"
    with open(gene_list, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_set_name = parts[0]
            genes = parts[1:]
            gene_sets = {'gene_set':gene_set_name, 'genes': genes}
    
    # Load and preprocess data
    train_adata, test_adata, train_loader, test_loader, train_dataset = create_aligned_datasets(
        train_data_path=train_data,
        train_label_path=train_label,
        test_data_path=test_data,
        test_label_path=test_label,
        unknown_pattern="unknown_tumor_type",
        n_top_genes=2000,
        scale_max_value=12,
        gene_list = gene_sets
    )
    
    print("\n" + "=" * 70)
    print("DATA LOADING COMPLETE - READY FOR VAE TRAINING")
    print("=" * 70)
    
    print(f"\nFinal dataset summary:")
    print(f"  Training: {train_adata.shape[0]} samples × {train_adata.shape[1]} genes")
    print(f"  Test: {test_adata.shape[0]} samples × {test_adata.shape[1]} genes")
    
    
    # Initialize model
    input_dim = train_adata.shape[1]
    latent_dim = 8
    n_classes = len(train_adata.obs['label'].unique())
    
    print(f"\nInitializing Conditional VAE:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Number of classes: {n_classes}")
    
    model = ConditionalVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[512, 256, 128],
        # n_classes=n_classes,
        dropout=0.1
    )
    
    # Train model
    print("\nTraining VAE...")
    history = train_vae_with_test_set(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=200,
        learning_rate=1e-3,
        early_stopping_patience=20
    )
    
    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].legend()
    axes[0].set_title('Total Loss')

    axes[1].plot(history['train_recon'], label='Train')
    axes[1].plot(history['test_recon'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].legend()
    axes[1].set_title('Reconstruction Loss')

    axes[2].plot(history['train_kl'], label='Train')
    axes[2].plot(history['test_kl'], label='Test')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Loss')
    axes[2].legend()
    axes[2].set_title('KL Divergence Loss')
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300)
    plt.show()
    
    # Extract and visualize embeddings
    print("\nExtracting embeddings...")
    # full_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    embeddings, labels = extract_embeddings(model, train_loader)
    label_names = train_dataset.label_encoder.classes_
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Visualize using UMAP
    print("\nVisualizing embeddings with UMAP...")
    visualize_embeddings(
        embeddings, 
        labels, 
        label_names,
        method='umap',
        save_path='results/vae_embeddings_umap.png'
    )
    
    # Calculate separation metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    if labels is not None:
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        
        print(f"\nEmbedding Quality Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    
    # Save the trained model
    print("\nSaving final model...")
    th.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'n_classes': n_classes,
    }, 'results/final_vae_model.pth')
    
    print("\nTraining complete! Model and visualizations saved.")
    
    return model, embeddings, labels

if __name__ == "__main__":
    # Run the main pipeline
    model, embeddings, labels = main()