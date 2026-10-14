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
import pickle
from sklearn.preprocessing import StandardScaler

# Add Informer2020 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Informer2020'))

try:
    from models.model import Informer
except ImportError:
    print("Error: Could not import Informer. Make sure the Informer2020 folder is in the same directory.")
    sys.exit(1)

from bin import logger
from bin.utils import seed_everything

# Define the order of stages
STAGE_ORDER = [
    'normal_GTEX',
    'normal_TCGA',
    'Stage_I',
    'Stage_II',
    'Stage_III'
]

# Define the transitions we want to model
# Each model predicts the NEXT stage given the CURRENT stage
TRANSITIONS = [
    ('normal_GTEX', 'normal_TCGA'),
    ('normal_GTEX', 'Stage_I'),  # Direct transition from GTEX to Stage I
    ('normal_TCGA', 'Stage_I'),
    ('Stage_I', 'Stage_II'),
    ('Stage_II', 'Stage_III'),
]

class StageTransitionDataset(Dataset):
    """
    Dataset for a single stage transition.
    Uses Nearest Neighbor matching to create paired samples.
    """
    def __init__(self, df, source_stage, target_stage, samples_per_epoch=1000, scaler=None, fit_scaler=False):
        self.samples_per_epoch = samples_per_epoch
        self.latent_cols = [c for c in df.columns if c.startswith('z')]
        self.dim = len(self.latent_cols)

        # Get data for source and target stages
        source_df = df[df['label'] == source_stage]
        target_df = df[df['label'] == target_stage]

        self.source_data = source_df[self.latent_cols].values.astype(np.float32)
        self.target_data = target_df[self.latent_cols].values.astype(np.float32)

        # Scaling
        self.scaler = scaler
        if fit_scaler and scaler is not None:
            # Fit scaler on combined source and target data
            combined = np.concatenate([self.source_data, self.target_data], axis=0)
            self.scaler.fit(combined)

        if self.scaler is not None:
            self.source_data = self.scaler.transform(self.source_data).astype(np.float32)
            self.target_data = self.scaler.transform(self.target_data).astype(np.float32)

        self.source_stage = source_stage
        self.target_stage = target_stage

        print(f"  Transition {source_stage} -> {target_stage}: {len(self.source_data)} source, {len(self.target_data)} target samples")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 1. Sample a random source
        if len(self.source_data) > 0:
            src_idx = np.random.randint(len(self.source_data))
            src_vec = self.source_data[src_idx]
        else:
            src_vec = np.zeros(self.dim, dtype=np.float32)

        # 2. Find nearest neighbor in target stage (with some randomness)
        if len(self.target_data) > 0:
            dists = np.linalg.norm(self.target_data - src_vec, axis=1)
            K = min(len(self.target_data), 5)
            if K < len(self.target_data):
                top_k_indices = np.argpartition(dists, K)[:K]
            else:
                top_k_indices = np.arange(len(self.target_data))
            choice = np.random.choice(top_k_indices)
            tgt_vec = self.target_data[choice]
        else:
            tgt_vec = np.zeros(self.dim, dtype=np.float32)

        # Return source and target as tensors
        return (
            torch.tensor(src_vec, dtype=torch.float32),
            torch.tensor(tgt_vec, dtype=torch.float32)
        )

def get_dummy_time_features(batch_size, seq_len, device):
    """Generates dummy time features for Informer."""
    steps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    x_mark = torch.zeros(batch_size, seq_len, 5).to(device)
    x_mark[:, :, 3] = steps  # Hour
    x_mark[:, :, 4] = steps  # Minute
    x_mark[:, :, 0] = 1  # Month
    x_mark[:, :, 1] = 1  # Day
    x_mark[:, :, 2] = 1  # Weekday
    return x_mark

def create_informer_model(input_dim, d_model, nhead, num_layers, device):
    """Creates an Informer model for single-step prediction."""
    model = Informer(
        enc_in=input_dim,
        dec_in=input_dim,
        c_out=input_dim,
        seq_len=1,      # Input is single time step
        label_len=1,    # Label overlap
        out_len=1,      # Predict single next step
        factor=5,
        d_model=d_model,
        n_heads=nhead,
        e_layers=num_layers,
        d_layers=1,
        d_ff=d_model * 4,
        dropout=0.1,
        attn='full',
        embed='timeF',
        freq='t',
        distil=False,
        device=device
    ).to(device)
    return model

def train_transition_model(model, train_loader, val_loader, epochs, lr, device, transition_name):
    """Train a single transition model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for src, tgt in train_loader:
            src = src.to(device).unsqueeze(1)  # [B, 1, D]
            tgt = tgt.to(device).unsqueeze(1)  # [B, 1, D]

            # Prepare Informer inputs
            x_enc = src
            x_dec = src.clone()  # Use source as decoder start token

            x_mark_enc = get_dummy_time_features(src.size(0), 1, device)
            x_mark_dec = get_dummy_time_features(src.size(0), 1, device)

            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device).unsqueeze(1)
                tgt = tgt.to(device).unsqueeze(1)

                x_enc = src
                x_dec = src.clone()
                x_mark_enc = get_dummy_time_features(src.size(0), 1, device)
                x_mark_dec = get_dummy_time_features(src.size(0), 1, device)

                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                val_loss += criterion(output, tgt).item()

        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

    return history, best_val_loss

def visualize_real_data_predictions(models, scalers, df, device, outdir, args):
    """
    Generate visualization using REAL data at each stage for prediction.
    No cascading - each model uses actual data from its source stage.
    """

    latent_cols = [c for c in df.columns if c.startswith('z')]
    dim = len(latent_cols)

    # Calculate Ground Truth Stats (unscaled)
    stage_data_raw = {}
    stage_means = []
    stage_stds = []

    for stage in STAGE_ORDER:
        data = df[df['label'] == stage][latent_cols].values.astype(np.float32)
        stage_data_raw[stage] = data
        if len(data) > 0:
            stage_means.append(np.mean(data, axis=0))
            stage_stds.append(np.std(data, axis=0))
        else:
            stage_means.append(np.zeros(dim))
            stage_stds.append(np.zeros(dim))

    stage_means = np.stack(stage_means)
    stage_stds = np.stack(stage_stds)

    # Select dimensions to plot
    all_data = np.concatenate([stage_data_raw[s] for s in STAGE_ORDER if len(stage_data_raw[s]) > 0], axis=0)
    variances = np.var(all_data, axis=0)
    top_dims = np.argsort(variances)[::-1][:4]
    dims_to_plot = sorted(list(set([0, 1] + list(top_dims))))

    # For each transition, use REAL source data to predict target
    # Then compare predictions to REAL target data
    num_simulations = 200

    # Store predictions for each stage (predictions[stage_idx] = predictions for that stage)
    # Stage 0 (normal_GTEX) has no prediction - it's the start
    all_predictions = {0: None}  # No prediction for first stage

    for trans_idx, (src_stage, tgt_stage) in enumerate(TRANSITIONS):
        model = models[trans_idx]
        scaler = scalers[trans_idx]
        model.eval()

        # Get real source data
        src_data_raw = stage_data_raw[src_stage]

        predictions = []

        with torch.no_grad():
            for _ in range(num_simulations):
                # Sample a random real source sample
                if len(src_data_raw) > 0:
                    idx = np.random.randint(len(src_data_raw))
                    src_vec_raw = src_data_raw[idx]
                else:
                    src_vec_raw = np.zeros(dim, dtype=np.float32)

                # Scale the input
                src_vec_scaled = scaler.transform(src_vec_raw.reshape(1, -1)).astype(np.float32)
                src_tensor = torch.tensor(src_vec_scaled).unsqueeze(1).to(device)  # [1, 1, D]

                # Predict
                x_enc = src_tensor
                x_dec = src_tensor.clone()
                x_mark_enc = get_dummy_time_features(1, 1, device)
                x_mark_dec = get_dummy_time_features(1, 1, device)

                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                pred_scaled = output.squeeze(0).squeeze(0).cpu().numpy()

                # Inverse scale to get back to original space
                pred_raw = scaler.inverse_transform(pred_scaled.reshape(1, -1)).flatten()
                predictions.append(pred_raw)

        predictions = np.stack(predictions, axis=0)
        tgt_stage_idx = STAGE_ORDER.index(tgt_stage)
        all_predictions[tgt_stage_idx] = predictions

    # Calculate prediction stats
    pred_means = []
    pred_stds = []
    for stage_idx in range(len(STAGE_ORDER)):
        if all_predictions[stage_idx] is None:
            # First stage - use ground truth (no prediction)
            pred_means.append(stage_means[0])
            pred_stds.append(np.zeros(dim))  # No std for seed
        else:
            pred_means.append(np.mean(all_predictions[stage_idx], axis=0))
            pred_stds.append(np.std(all_predictions[stage_idx], axis=0))

    pred_means = np.stack(pred_means)
    pred_stds = np.stack(pred_stds)

    # Plot each dimension
    for dim_idx in dims_to_plot:
        plt.figure(figsize=(12, 7))

        gt_m = stage_means[:, dim_idx]
        gt_s = stage_stds[:, dim_idx]
        x = np.arange(len(STAGE_ORDER))

        # Ground Truth
        plt.plot(x, gt_m, 'o--', label='GroundTruth Mean', color='blue', alpha=0.8, markersize=10)
        plt.fill_between(x, gt_m - gt_s, gt_m + gt_s, color='blue', alpha=0.15, label='GroundTruth Std Dev')

        # Predictions (starting from stage 1)
        pr_m = pred_means[:, dim_idx]
        pr_s = pred_stds[:, dim_idx]

        # Plot seed (first stage - no prediction)
        plt.plot(x[0], gt_m[0], 's', color='green', markersize=12, zorder=5, label='Input (Real Data)')

        # Plot predictions for remaining stages
        plt.plot(x[1:], pr_m[1:], 'x-', color='orange', linewidth=2, markersize=10, label='Model Predictions')
        plt.fill_between(x[1:], pr_m[1:] - pr_s[1:], pr_m[1:] + pr_s[1:], color='orange', alpha=0.25, label='Prediction Std Dev')

        # Add transition arrows (from source stage index to target stage index)
        for src, tgt in TRANSITIONS:
            src_idx = STAGE_ORDER.index(src)
            tgt_idx = STAGE_ORDER.index(tgt)
            plt.annotate('', xy=(tgt_idx, gt_m[tgt_idx]), xytext=(src_idx, gt_m[src_idx]),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3))

        plt.xticks(x, STAGE_ORDER, rotation=15, fontsize=10)
        plt.xlabel('Stage', fontsize=12)
        plt.ylabel(f'Latent Dimension {dim_idx+1} Value', fontsize=12)
        plt.title(f'Stage Prediction (Real Data Input) - Dimension {dim_idx+1}', fontsize=14)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(outdir, f"real_data_pred_dim_{dim_idx+1}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved plot: {fname}")

def train(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, sep='\t', index_col=0)
    df = df[df['label'].isin(STAGE_ORDER)]
    print(f"Data shape: {df.shape}")

    latent_cols = [c for c in df.columns if c.startswith('z')]
    dim = len(latent_cols)

    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'test']
    else:
        train_df = df
        val_df = df

    os.makedirs(args.outdir, exist_ok=True)

    # 2. Train a model for each transition
    print("\n" + "="*60)
    print("TRAINING CASCADED INFORMER MODELS (with proper scaling)")
    print("="*60)

    trained_models = []
    trained_scalers = []
    all_histories = {}

    for i, (src_stage, tgt_stage) in enumerate(TRANSITIONS):
        transition_name = f"{src_stage}_to_{tgt_stage}"
        print(f"\n[Model {i+1}/{len(TRANSITIONS)}] Training: {src_stage} -> {tgt_stage}")

        # Create scaler for this transition
        scaler = StandardScaler()

        # Create datasets (fit scaler on training data)
        train_ds = StageTransitionDataset(
            train_df, src_stage, tgt_stage,
            samples_per_epoch=args.samples_per_epoch,
            scaler=scaler, fit_scaler=True
        )
        val_ds = StageTransitionDataset(
            val_df, src_stage, tgt_stage,
            samples_per_epoch=args.samples_per_epoch // 5,
            scaler=scaler, fit_scaler=False  # Use same scaler, don't refit
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        # Create model
        model = create_informer_model(dim, args.d_model, args.nhead, args.num_layers, device)

        # Train
        history, best_loss = train_transition_model(
            model, train_loader, val_loader,
            args.epochs, args.lr, device, transition_name
        )

        # Save model
        model_path = os.path.join(args.outdir, f"informer_{transition_name}.pth")
        torch.save(model.state_dict(), model_path)

        # Save scaler
        scaler_path = os.path.join(args.outdir, f"scaler_{transition_name}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"    Saved model to {model_path}")
        print(f"    Saved scaler to {scaler_path}")

        trained_models.append(model)
        trained_scalers.append(scaler)
        all_histories[transition_name] = history

    # 3. Plot Training Curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, hist in all_histories.items():
        plt.plot(hist['train_loss'], label=name.replace('_to_', '→'))
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for name, hist in all_histories.items():
        plt.plot(hist['val_loss'], label=name.replace('_to_', '→'))
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "training_curves.png"))
    plt.close()

    # 4. Generate Visualizations (using REAL data, not cascaded)
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS (Real Data Predictions)")
    print("="*60)
    visualize_real_data_predictions(trained_models, trained_scalers, train_df, device, args.outdir, args)

    print("\nDone! Results saved to:", args.outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="latent_all_stages.tsv")
    parser.add_argument("--outdir", type=str, default="results_cascaded")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--samples_per_epoch", type=int, default=2000)

    args = parser.parse_args()
    train(args)
