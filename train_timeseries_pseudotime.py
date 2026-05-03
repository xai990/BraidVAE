"""
Time Series Model using Supervised Pseudotime

This version uses the continuous supervised pseudotime as the temporal ordering,
rather than discrete stage labels. The pseudotime provides a smooth, biologically-
meaningful progression: Normal → Stage I → Stage II → Stage III.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import joblib
from sklearn.preprocessing import StandardScaler

# Add Informer2020 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Informer2020'))

try:
    from models.model import Informer
except ImportError:
    print("Error: Could not import Informer. Make sure Informer2020 folder exists.")
    sys.exit(1)

from bin.utils import seed_everything


class PseudotimeTrajectoryDataset(Dataset):
    """
    Constructs trajectories ordered by supervised pseudotime.
    
    Instead of discrete stages, we bin the pseudotime continuum and 
    sample one point from each bin to create smooth trajectories.
    """
    def __init__(self, df, num_bins=10, samples_per_epoch=1000, scaler=None):
        self.samples_per_epoch = samples_per_epoch
        self.latent_cols = [c for c in df.columns if c.startswith('z')]
        self.dim = len(self.latent_cols)
        self.num_bins = num_bins
        self.scaler = scaler
        
        # Extract latent vectors and pseudotime
        self.X = df[self.latent_cols].values.astype(np.float32)
        
        # Apply scaling if scaler is provided
        if self.scaler is not None:
            self.X = self.scaler.transform(self.X).astype(np.float32)
            
        self.pseudotime = df['supervised_pseudotime'].values.astype(np.float32)
        self.labels = df['label'].values
        
        # Create bins based on pseudotime
        self.bin_edges = np.linspace(0, 1, num_bins + 1)
        self.bin_indices = []
        
        for i in range(num_bins):
            low, high = self.bin_edges[i], self.bin_edges[i + 1]
            if i == num_bins - 1:  # Include right edge for last bin
                mask = (self.pseudotime >= low) & (self.pseudotime <= high)
            else:
                mask = (self.pseudotime >= low) & (self.pseudotime < high)
            indices = np.where(mask)[0]
            self.bin_indices.append(indices)
            
        # Print bin statistics
        print(f"\nPseudotime bins ({num_bins} bins):")
        for i, indices in enumerate(self.bin_indices):
            if len(indices) > 0:
                pt_range = f"[{self.pseudotime[indices].min():.3f}, {self.pseudotime[indices].max():.3f}]"
            else:
                pt_range = "empty"
            print(f"  Bin {i}: {len(indices)} samples, pseudotime {pt_range}")
            
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        """
        Construct a trajectory by sampling one point from each pseudotime bin.
        Uses nearest neighbor matching for smooth transitions.
        """
        trajectory = []
        pseudotimes = []
        
        # Start from a random sample in the first bin (lowest pseudotime)
        first_bin = self.bin_indices[0]
        if len(first_bin) > 0:
            start_idx = np.random.choice(first_bin)
            curr_vec = self.X[start_idx]
            curr_pt = self.pseudotime[start_idx]
        else:
            curr_vec = np.zeros(self.dim, dtype=np.float32)
            curr_pt = 0.0
            
        trajectory.append(curr_vec)
        pseudotimes.append(curr_pt)
        
        # Walk through remaining bins using nearest neighbor
        for bin_idx in range(1, self.num_bins):
            bin_samples = self.bin_indices[bin_idx]
            
            if len(bin_samples) > 0:
                # Find K nearest neighbors in this bin
                bin_X = self.X[bin_samples]
                dists = np.linalg.norm(bin_X - curr_vec, axis=1)
                
                K = min(len(bin_samples), 5)
                top_k = np.argpartition(dists, K-1)[:K] if K < len(bin_samples) else np.arange(len(bin_samples))
                
                # Sample from top-K
                choice = np.random.choice(top_k)
                next_idx = bin_samples[choice]
                next_vec = self.X[next_idx]
                next_pt = self.pseudotime[next_idx]
            else:
                # Interpolate if bin is empty
                next_vec = curr_vec
                next_pt = (self.bin_edges[bin_idx] + self.bin_edges[bin_idx + 1]) / 2
                
            trajectory.append(next_vec)
            pseudotimes.append(next_pt)
            curr_vec = next_vec
            
        trajectory = np.stack(trajectory)  # [num_bins, dim]
        pseudotimes = np.array(pseudotimes, dtype=np.float32)  # [num_bins]
        
        return torch.tensor(trajectory), torch.tensor(pseudotimes)


def get_pseudotime_features(pseudotimes, device):
    """
    Convert pseudotime values to time features for Informer.
    
    We encode pseudotime using multiple representations:
    - Raw pseudotime (normalized 0-1)
    - Sine/cosine encoding for periodicity awareness
    - Squared term for non-linear progression
    """
    batch_size, seq_len = pseudotimes.shape
    
    # Create feature tensor [batch, seq_len, 5]
    x_mark = torch.zeros(batch_size, seq_len, 5).to(device)
    
    pt = pseudotimes.to(device)
    
    x_mark[:, :, 0] = pt                          # Raw pseudotime
    x_mark[:, :, 1] = torch.sin(pt * np.pi)       # Sine encoding
    x_mark[:, :, 2] = torch.cos(pt * np.pi)       # Cosine encoding  
    x_mark[:, :, 3] = pt ** 2                     # Quadratic term
    x_mark[:, :, 4] = torch.sqrt(pt + 1e-8)       # Square root term
    
    return x_mark


def visualize_pseudotime_predictions(model, dataset, device, outdir, args):
    """
    Visualize model predictions along pseudotime trajectory.
    """
    model.eval()
    
    num_simulations = 200
    predicted_trajectories = []
    actual_trajectories = []
    pseudotime_trajectories = []
    
    seq_len = args.seq_len
    label_len = args.label_len
    pred_len = args.pred_len
    
    with torch.no_grad():
        for _ in range(num_simulations):
            # Get a trajectory from dataset
            traj, pt = dataset[0]  # Random due to sampling
            traj = traj.unsqueeze(0).to(device)  # [1, num_bins, dim]
            pt = pt.unsqueeze(0).to(device)      # [1, num_bins]
            
            # Encoder input: first seq_len steps
            x_enc = traj[:, :seq_len, :]
            pt_enc = pt[:, :seq_len]
            
            # Decoder input: label_len overlap + zeros for prediction
            dec_input = traj[:, seq_len-label_len:seq_len, :]
            dec_zeros = torch.zeros(1, pred_len, dataset.dim).to(device)
            x_dec = torch.cat([dec_input, dec_zeros], dim=1)
            
            pt_dec = pt[:, seq_len-label_len:seq_len+pred_len]
            
            # Time features
            x_mark_enc = get_pseudotime_features(pt_enc, device)
            x_mark_dec = get_pseudotime_features(pt_dec, device)
            
            # Forward pass
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Full predicted trajectory
            full_pred = torch.cat([x_enc, output], dim=1).squeeze(0).cpu().numpy()
            actual_full = traj.squeeze(0).cpu().numpy()
            pt_full = pt.squeeze(0).cpu().numpy()
            
            predicted_trajectories.append(full_pred)
            actual_trajectories.append(actual_full)
            pseudotime_trajectories.append(pt_full)
            
    predicted_trajectories = np.stack(predicted_trajectories)
    actual_trajectories = np.stack(actual_trajectories)
    pseudotime_trajectories = np.stack(pseudotime_trajectories)
    
    # Mean pseudotime for x-axis
    mean_pseudotime = pseudotime_trajectories.mean(axis=0)
    
    pred_means = predicted_trajectories.mean(axis=0)
    pred_stds = predicted_trajectories.std(axis=0)
    actual_means = actual_trajectories.mean(axis=0)
    actual_stds = actual_trajectories.std(axis=0)
    
    # Select dimensions with highest BETWEEN-STAGE variance
    # This captures dimensions that change most across cancer progression,
    # not just dimensions with high variance due to normal vs cancer difference
    stage_order = ['normal_GTEX', 'normal_TCGA', 'Stage_I', 'Stage_II', 'Stage_III']
    stage_means = []
    for stage in stage_order:
        mask = dataset.labels == stage
        if mask.sum() > 0:
            stage_means.append(dataset.X[mask].mean(axis=0))
    stage_means = np.stack(stage_means)  # [5 stages, 32 dims]
    
    # Compute variance of stage means (between-stage variance)
    between_stage_var = np.var(stage_means, axis=0)
    top_dims = np.argsort(between_stage_var)[::-1][:6]
    
    print(f"\nTop 6 dimensions by between-stage variance:")
    for i, dim in enumerate(top_dims):
        print(f"  {i+1}. z{dim+1} (between-stage var: {between_stage_var[dim]:.4f})")
    
    # Plot individual dimensions
    for dim in top_dims:
        plt.figure(figsize=(10, 6))
        
        x = mean_pseudotime
        
        # Ground truth
        plt.plot(x, actual_means[:, dim], 'o--', label='Ground Truth', color='blue', alpha=0.7)
        plt.fill_between(x, actual_means[:, dim] - actual_stds[:, dim], 
                        actual_means[:, dim] + actual_stds[:, dim], 
                        color='blue', alpha=0.1)
        
        # Predictions
        plt.plot(x[:seq_len], pred_means[:seq_len, dim], 's-', 
                color='green', label='Input (Encoder)', linewidth=2)
        plt.plot(x[seq_len:], pred_means[seq_len:, dim], 'x-', 
                color='orange', label='Predicted (Decoder)', linewidth=2)
        plt.fill_between(x[seq_len:], 
                        pred_means[seq_len:, dim] - pred_stds[seq_len:, dim],
                        pred_means[seq_len:, dim] + pred_stds[seq_len:, dim],
                        color='orange', alpha=0.2)
        
        # Add vertical line at prediction boundary
        plt.axvline(x=x[seq_len-1], color='gray', linestyle='--', alpha=0.5, label='Prediction Start')
        
        plt.xlabel('Pseudotime')
        plt.ylabel(f'Latent Dimension z{dim+1}')
        plt.title(f'Trajectory Prediction - Dimension z{dim+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        fname = os.path.join(outdir, f"pseudotime_traj_dim_{dim+1}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")
    
    # Heatmap of all dimensions
    plt.figure(figsize=(14, 8))
    
    # Normalize for visualization
    pred_norm = (pred_means - pred_means.min(axis=0)) / (pred_means.max(axis=0) - pred_means.min(axis=0) + 1e-8)
    
    plt.subplot(1, 2, 1)
    sns.heatmap(actual_means.T, cmap='viridis', 
                xticklabels=[f'{pt:.2f}' for pt in mean_pseudotime],
                yticklabels=[f'z{i+1}' for i in range(dataset.dim)])
    plt.xlabel('Pseudotime')
    plt.ylabel('Latent Dimension')
    plt.title('Ground Truth Trajectories')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(pred_means.T, cmap='viridis',
                xticklabels=[f'{pt:.2f}' for pt in mean_pseudotime],
                yticklabels=[f'z{i+1}' for i in range(dataset.dim)])
    plt.xlabel('Pseudotime')
    plt.ylabel('Latent Dimension')
    plt.title('Predicted Trajectories')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pseudotime_trajectory_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(outdir, 'pseudotime_trajectory_heatmap.png')}")


def train(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data with supervised pseudotime
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, sep='\t', index_col=0)
    
    if 'supervised_pseudotime' not in df.columns:
        raise ValueError("Data must contain 'supervised_pseudotime' column. "
                        "Run calc_supervised_pseudotime.py first.")
    
    print(f"Data shape: {df.shape}")
    print(f"Pseudotime range: [{df['supervised_pseudotime'].min():.3f}, {df['supervised_pseudotime'].max():.3f}]")
    
    # Split data
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'test']
    else:
        # Random split
        mask = np.random.rand(len(df)) < 0.8
        train_df = df[mask]
        val_df = df[~mask]
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Fit StandardScaler on training data
    latent_cols = [c for c in train_df.columns if c.startswith('z')]
    scaler = StandardScaler()
    scaler.fit(train_df[latent_cols].values)
    
    print(f"\nApplying StandardScaler:")
    print(f"  Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Std range:  [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    
    # Save scaler for inference
    scaler_path = os.path.join(args.outdir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to: {scaler_path}")
    
    # Create datasets with scaler
    train_ds = PseudotimeTrajectoryDataset(train_df, num_bins=args.num_bins, 
                                           samples_per_epoch=args.samples_per_epoch,
                                           scaler=scaler)
    val_ds = PseudotimeTrajectoryDataset(val_df, num_bins=args.num_bins,
                                         samples_per_epoch=args.samples_per_epoch // 5,
                                         scaler=scaler)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Configure model
    # seq_len: number of input time steps (encoder)
    # pred_len: number of steps to predict (decoder output)
    args.seq_len = args.num_bins // 2      # First half as input
    args.label_len = min(2, args.seq_len)  # Overlap between encoder and decoder
    args.pred_len = args.num_bins - args.seq_len  # Predict second half
    
    print(f"\nModel configuration:")
    print(f"  seq_len (encoder input): {args.seq_len}")
    print(f"  label_len (decoder overlap): {args.label_len}")
    print(f"  pred_len (prediction steps): {args.pred_len}")
    print(f"  Total trajectory length: {args.num_bins}")
    
    model = Informer(
        enc_in=train_ds.dim,
        dec_in=train_ds.dim,
        c_out=train_ds.dim,
        seq_len=args.seq_len,
        label_len=args.label_len,
        out_len=args.pred_len,
        factor=5,
        d_model=args.d_model,
        n_heads=args.nhead,
        e_layers=args.num_layers,
        d_layers=max(1, args.num_layers // 2),
        d_ff=args.d_model * 4,
        dropout=0.1,
        attn='full',
        embed='timeF',
        freq='t',
        distil=False,
        device=device
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_traj, batch_pt in train_loader:
            batch_traj = batch_traj.to(device)  # [B, num_bins, dim]
            batch_pt = batch_pt.to(device)      # [B, num_bins]
            
            # Encoder input
            x_enc = batch_traj[:, :args.seq_len, :]
            pt_enc = batch_pt[:, :args.seq_len]
            
            # Target: predict remaining steps
            y = batch_traj[:, args.seq_len:, :]
            
            # Decoder input: overlap + zeros
            dec_overlap = batch_traj[:, args.seq_len-args.label_len:args.seq_len, :]
            dec_zeros = torch.zeros(batch_traj.size(0), args.pred_len, train_ds.dim).to(device)
            x_dec = torch.cat([dec_overlap, dec_zeros], dim=1)
            
            pt_dec = batch_pt[:, args.seq_len-args.label_len:]
            
            # Time features from pseudotime
            x_mark_enc = get_pseudotime_features(pt_enc, device)
            x_mark_dec = get_pseudotime_features(pt_dec, device)
            
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_traj, batch_pt in val_loader:
                batch_traj = batch_traj.to(device)
                batch_pt = batch_pt.to(device)
                
                x_enc = batch_traj[:, :args.seq_len, :]
                pt_enc = batch_pt[:, :args.seq_len]
                y = batch_traj[:, args.seq_len:, :]
                
                dec_overlap = batch_traj[:, args.seq_len-args.label_len:args.seq_len, :]
                dec_zeros = torch.zeros(batch_traj.size(0), args.pred_len, train_ds.dim).to(device)
                x_dec = torch.cat([dec_overlap, dec_zeros], dim=1)
                
                pt_dec = batch_pt[:, args.seq_len-args.label_len:]
                
                x_mark_enc = get_pseudotime_features(pt_enc, device)
                x_mark_dec = get_pseudotime_features(pt_dec, device)
                
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                val_loss += criterion(output, y).item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_informer_pseudotime.pth"))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save final model and training curves
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.outdir, "informer_pseudotime.pth"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Informer Training with Supervised Pseudotime')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "pseudotime_loss_curve.png"), dpi=150)
    plt.close()
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Load best model for visualization
    model.load_state_dict(torch.load(os.path.join(args.outdir, "best_informer_pseudotime.pth")))
    
    # Visualize predictions
    visualize_pseudotime_predictions(model, train_ds, device, args.outdir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Informer with Supervised Pseudotime")
    parser.add_argument("--data_path", type=str, default="latent_all_stages_supervised_pseudotime.tsv")
    parser.add_argument("--outdir", type=str, default="results_pseudotime")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_bins", type=int, default=10, 
                       help="Number of pseudotime bins (trajectory length)")
    parser.add_argument("--samples_per_epoch", type=int, default=2000)
    
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    train(args)

