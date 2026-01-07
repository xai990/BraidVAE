import os
import math
import json
import shutil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Local imports
from . import logger 
from .model import ConditionalVAE, VAELoss
from .transformer_model import GeneInformer 
from .dataset import PerGeneMinMaxScaler, _to_numpy, make_loader, PerGeneZeroOneScaler, LatentDataset
from .utils import extract_embeddings, visualize_embeddings

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pearson_correlation(x, y):
    """Differentiable Pearson Correlation for Loss Function"""
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return cost

# ===================== 1. Helper: Standard VAE Training Loop =====================

def train_vae_with_test_set(
    model: ConditionalVAE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    early_stopping_patience: int = 10,
    warmup_epochs: int = 50,
    ckpt_path: str = "results/best_vae_model.pth",
) -> Dict[str, List[float]]:
    
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
    criterion = VAELoss()

    history = {k: [] for k in ["train_loss", "test_loss", "train_recon", "train_kl", "test_recon", "test_kl"]}
    best_loss = float("inf")
    patience = 0

    for epoch in range(n_epochs):
        # KL warmup
        criterion.beta = min(1.0, epoch / max(1, warmup_epochs))

        # ---- train ----
        model.train()
        tr_loss = tr_rec = tr_kl = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
                y = y.to(device) if y is not None else None
            else:
                x = batch
                y = None
            x = x.to(device)

            opt.zero_grad()
            recon, inp, mu, logvar, z, _ = model(x, y)
            loss, dct = criterion(recon, inp, mu, logvar)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_rec += dct["recon"]
            tr_kl  += dct["kl"]

        ntr = max(1, len(train_loader))
        tr_loss /= ntr
        tr_rec /= ntr
        tr_kl /= ntr

        # ---- test ----
        model.eval()
        te_loss = te_rec = te_kl = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                    y = y.to(device) if y is not None else None
                else:
                    x = batch
                    y = None
                x = x.to(device)
                
                recon, inp, mu, logvar, z, _ = model(x, y)
                loss, dct = criterion(recon, inp, mu, logvar)
                te_loss += loss.item()
                te_rec += dct["recon"]
                te_kl += dct["kl"]

        nte = max(1, len(test_loader))
        te_loss /= nte
        te_rec /= nte
        te_kl /= nte
        
        # Scheduler & Early Stopping
        finite_te = np.isfinite(te_loss)
        sched_metric = te_loss if finite_te else float("inf")
        sched.step(sched_metric)

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_recon"].append(tr_rec)
        history["train_kl"].append(tr_kl)
        history["test_recon"].append(te_rec)
        history["test_kl"].append(te_kl)

        improved = finite_te and (te_loss < best_loss)
        if improved:
            best_loss = te_loss
            patience = 0
            _ensure_dir(os.path.dirname(ckpt_path))
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
        
        if patience >= early_stopping_patience:
            logger.log(f"Early stop at epoch {epoch}")
            break
            
        if (epoch + 1) % 10 == 0:
            logger.log(f"Epoch {epoch+1:03d} | train {tr_loss:.4f} | test {te_loss:.4f}")

    # Ensure we load best weights before returning
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        
    return history

# ===================== 2. Multi-Stage VAE (Step 1) =====================

def train_multistage_vaes(
    train_adata: ad.AnnData,
    test_adata: ad.AnnData,
    label_key: Optional[str] = "label",
    latent_dim: int = 32,
    hidden_dims: List[int] = (512, 256, 128),
    dropout: float = 0.1,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 1e-3,
    early_stop: int = 10,
    warmup_epochs: int = 50,
    outdir: str = "results_multistage",
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    
    _ensure_dir(outdir)
    stages = list(pd.unique(train_adata.obs[label_key]))
    if test_adata is not None:
        for s in pd.unique(test_adata.obs[label_key]):
            if s not in stages:
                stages.append(s)
    stages = sorted([str(s) for s in stages])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary: Dict[str, Dict] = {}
    all_latent_dfs = []

    for s in stages:
        logger.log(f"\n=== Training VAE for Stage: {s} ===")
        stage_dir = os.path.join(outdir, f"stage_{s}")
        _ensure_dir(stage_dir)

        tr_s = train_adata[train_adata.obs[label_key].astype(str) == s].copy()
        if test_adata is not None:
            te_s = test_adata[test_adata.obs[label_key].astype(str) == s].copy()
        else:
            te_s = tr_s[0:0]

        if tr_s.n_obs < 5:
            logger.log(f"  [skip] Too few training samples ({tr_s.n_obs}) for stage={s}")
            continue

        scaler = PerGeneMinMaxScaler()
        scaler.fit(_to_numpy(tr_s.X))
        
        tr_loader, tr_ds = make_loader(tr_s, batch_size, shuffle=True, label_key=label_key, scaler=scaler)
        te_loader, te_ds = make_loader(te_s if te_s.n_obs > 0 else tr_s, batch_size, shuffle=False, label_key=label_key, scaler=scaler)

        # Unconditional VAE for this specific stage (n_classes=0)
        model = ConditionalVAE(
            input_dim=tr_s.n_vars, latent_dim=latent_dim, hidden_dims=list(hidden_dims),
            n_classes=0, dropout=dropout,
        )

        ckpt = os.path.join(stage_dir, "best_vae.pth")
        hist = train_vae_with_test_set(
            model, tr_loader, te_loader, n_epochs=epochs, learning_rate=lr, device=device,
            early_stopping_patience=early_stop, warmup_epochs=warmup_epochs, ckpt_path=ckpt,
        )

        # Extract Embeddings
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device).eval()

        ext_tr_loader, _ = make_loader(tr_s, batch_size, shuffle=False, label_key=label_key, scaler=scaler)
        Z_tr, names_tr, _ = extract_embeddings(model, ext_tr_loader, device)
        
        df_tr = pd.DataFrame(Z_tr, index=names_tr, columns=[f"z{i+1}" for i in range(latent_dim)])
        df_tr["split"] = "train"
        df_tr[label_key] = s
        all_latent_dfs.append(df_tr)

        if te_s.n_obs > 0:
            Z_te, names_te, _ = extract_embeddings(model, te_loader, device)
            df_te = pd.DataFrame(Z_te, index=names_te, columns=[f"z{i+1}" for i in range(latent_dim)])
            df_te["split"] = "test"
            df_te[label_key] = s
            all_latent_dfs.append(df_te)

        with open(os.path.join(stage_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        summary[str(s)] = {"history": hist, "model_path": ckpt}

    if not all_latent_dfs:
        raise ValueError("No VAEs were trained successfully.")

    df_all = pd.concat(all_latent_dfs, axis=0)
    master_path = os.path.join(outdir, "latent_all_stages.tsv")
    df_all.to_csv(master_path, sep="\t")
    return df_all, summary

# ===================== 3. Single VAE (Original Option) =====================

def train_single_vae(
    train_adata: ad.AnnData,
    test_adata: Optional[ad.AnnData] = None,
    label_key: Optional[str] = "label",
    latent_dim: int = 32,
    hidden_dims: List[int] = (512, 256, 128),
    dropout: float = 0.1,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 1e-3,
    early_stop: int = 10,
    warmup_epochs: int = 50,
    outdir: str = "results_single_vae",
    run_name: str = "all",
) -> Dict[str, Dict]:
    
    run_dir = os.path.join(outdir, run_name) if run_name else outdir
    _ensure_dir(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if test_adata is None or test_adata.n_obs == 0:
        test_adata = train_adata

    scaler = PerGeneMinMaxScaler()
    scaler.fit(_to_numpy(train_adata.X))

    tr_loader, tr_ds = make_loader(train_adata, batch_size, shuffle=True, label_key=label_key, scaler=scaler)
    te_loader, te_ds = make_loader(test_adata, batch_size, shuffle=False, label_key=label_key, scaler=scaler)

    n_classes = 0
    if label_key is not None and label_key in train_adata.obs.columns:
        n_classes = len(pd.unique(train_adata.obs[label_key].astype(str)))

    model = ConditionalVAE(
        input_dim=train_adata.n_vars, latent_dim=latent_dim, hidden_dims=list(hidden_dims),
        n_classes=n_classes, dropout=dropout,
    )

    ckpt = os.path.join(run_dir, "best_vae.pth")
    history = train_vae_with_test_set(
        model, tr_loader, te_loader, n_epochs=epochs, learning_rate=lr, device=device,
        early_stopping_patience=early_stop, warmup_epochs=warmup_epochs, ckpt_path=ckpt,
    )

    # Extract
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    ext_tr_loader, _ = make_loader(train_adata, batch_size, shuffle=False, label_key=label_key, scaler=scaler)
    Z_tr, names_tr, _ = extract_embeddings(model, ext_tr_loader, device)
    df_tr = pd.DataFrame(Z_tr, index=names_tr, columns=[f"z{i+1}" for i in range(Z_tr.shape[1])])
    df_tr["split"] = "train"
    if label_key: df_tr[label_key] = train_adata.obs.loc[names_tr, label_key].values

    if test_adata is not None:
        Z_te, names_te, _ = extract_embeddings(model, te_loader, device)
        df_te = pd.DataFrame(Z_te, index=names_te, columns=[f"z{i+1}" for i in range(Z_te.shape[1])])
        df_te["split"] = "test"
        if label_key: df_te[label_key] = test_adata.obs.loc[names_te, label_key].values
        df_all = pd.concat([df_tr, df_te], axis=0)
    else:
        df_all = df_tr

    latent_path = os.path.join(run_dir, "latent_all.tsv")
    df_all.to_csv(latent_path, sep="\t")

    return {"history": history, "model_path": ckpt, "latent_path": latent_path}

# ===================== 4. Transformer on Latent (Step 4) =====================

def train_latent_transformer(
    latent_df: pd.DataFrame,
    label_key: str = "label",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    keep_ratio: float = 0.5, 
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    outdir: str = "results_transformer_on_latent"
):
    _ensure_dir(outdir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Prepare Data
    # Identify input features (z1, z2, ...)
    feature_cols = [c for c in latent_df.columns if c.startswith("z")]
    if not feature_cols:
        logger.log("[Error] No 'z' columns found in latent DataFrame!")
        return pd.DataFrame()

    X = latent_df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(latent_df[label_key].astype(str))
    n_classes = len(le.classes_)

    # Log initial state
    logger.log(f"Transformer Input: {X.shape[1]} VAE Latents (z1...z{X.shape[1]})")
    
    # Train/Test Split
    if "split" in latent_df.columns:
        mask = latent_df["split"] == "train"
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[~mask], y[~mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Model (GeneInformer)
    model = GeneInformer(
        num_genes=len(feature_cols), 
        num_classes=n_classes, 
        d_model=d_model,
        nhead=nhead, 
        num_encoder_layers=num_layers, 
        keep_ratio=keep_ratio, 
        dropout=0.1
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0
    model_path = os.path.join(outdir, "best_latent_transformer.pth")

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device).long()
            opt.zero_grad()
            logits, _ = model(bx)
            loss = crit(logits, by)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in te_loader:
                bx, by = bx.to(device), by.to(device).long()
                logits, _ = model(bx)
                correct += (torch.argmax(logits, 1) == by).sum().item()
                total += by.size(0)
        acc = correct / total if total > 0 else 0
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            
        if (epoch+1) % 10 == 0:
            logger.log(f"Epoch {epoch+1:03d} | Loss: {tr_loss/len(tr_loader):.4f} | Acc: {acc:.4f}")

    # 4. Feature Selection Analysis (The "Show me" part)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Pass one batch to get the indices of selected features
    dummy_x = torch.tensor(X[:5]).to(device) # Take 5 samples
    with torch.no_grad():
        _, _, selected_indices = model(dummy_x, return_indices=True)
    
    # Calculate stats
    # selected_indices shape is [Batch, k]
    # We take the most frequent indices across the batch to be robust
    if selected_indices is not None:
        flat_indices = selected_indices.flatten().cpu().numpy()
        counts = pd.Series(flat_indices).value_counts()
        # Just take the indices that appear in majority of samples
        common_indices = counts.index.sort_values().tolist()
        
        selected_vae_dims = [feature_cols[i] for i in common_indices]
        logger.log("\n" + "="*40)
        logger.log(f"LATENT SELECTION REPORT")
        logger.log(f"Original VAE Dims: {len(feature_cols)}")
        logger.log(f"Selected VAE Dims: {len(selected_vae_dims)} (Ratio: {keep_ratio})")
        logger.log(f"Top Dimensions Kept: {selected_vae_dims}")
        logger.log("="*40 + "\n")
    else:
        logger.log("[Warn] Model did not return indices (Mixing used instead of Pruning?)")

    # 5. Extract Distilled Embeddings for Output
    full_ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False)
    
    dist_list = []
    with torch.no_grad():
        for bx, _ in full_loader:
            _, dist_vec = model(bx.to(device))
            dist_list.append(dist_vec.cpu().numpy())
    
    dist_matrix = np.concatenate(dist_list, axis=0)
    
    # Create DataFrame with "dist_dim" columns
    df_dist = pd.DataFrame(
        dist_matrix, 
        index=latent_df.index, 
        columns=[f"dist_dim{i}" for i in range(dist_matrix.shape[1])]
    )
    df_dist[label_key] = latent_df[label_key].values
    
    # Save
    df_dist.to_csv(os.path.join(outdir, "distilled_embeddings.tsv"), sep="\t")

    # Internal Viz
    try:
        visualize_embeddings(
            dist_matrix, labels=df_dist[label_key].values,
            label_names=np.unique(df_dist[label_key].values),
            save_path=os.path.join(outdir, "final_distilled_umap.png")
        )
    except Exception as e:
        logger.log(f"Viz failed in Step 4: {e}")
    
    return df_dist
# ===================== 5. Trajectory VAE (Step 5) =====================

def train_trajectory_vae(
    df_all: pd.DataFrame,
    label_col: str = "stage",
    latent_dim: int = 8,
    hidden_dims: List[int] = [64, 32],
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    outdir: str = "results_trajectory",
    time_reg_weight: float = 5.0 
):
    _ensure_dir(outdir)
    ds = LatentDataset(df_all, label_col=label_col)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalVAE(
        input_dim=ds.X.shape[1], latent_dim=latent_dim, hidden_dims=hidden_dims,
        n_classes=0 
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = VAELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_corr = 0
        for x, y_time in loader:
            x, y_time = x.to(device), y_time.to(device)
            opt.zero_grad()
            recon, inp, mu, logvar, z, _ = model(x)
            vae_loss, _ = criterion(recon, inp, mu, logvar)
            
            # Time Regularization: Force z[0] to correlate with Stage Order
            corr = pearson_correlation(z[:, 0], y_time)
            time_loss = 1.0 - (corr ** 2)
            loss = vae_loss + (time_reg_weight * time_loss)
            
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_corr += corr.item()
            
        if (epoch+1) % 10 == 0:
            logger.log(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Corr: {total_corr/len(loader):.4f}")

    # Extract
    model.eval()
    all_z = []
    ext_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, _ in ext_loader:
            _, _, _, _, z, _ = model(x.to(device))
            all_z.append(z.cpu().numpy())
    
    final_z = np.concatenate(all_z, axis=0)
    df_traj = pd.DataFrame(final_z, index=df_all.index, columns=[f"traj_z{i+1}" for i in range(latent_dim)])
    for col in df_all.columns:
        if not col.startswith("z"):
            df_traj[col] = df_all[col]
            
    df_traj.to_csv(os.path.join(outdir, "trajectory_embeddings.tsv"), sep="\t")
    return df_traj

# ===================== 6. Gene Transformer (Raw Input) =====================

def train_gene_transformer(
    train_adata: ad.AnnData,
    test_adata: Optional[ad.AnnData] = None,
    label_key: str = "label",
    top_k: int = 512,
    d_model: int = 256,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 1e-3,
    early_stop: int = 10,
    outdir: str = "results_transformer",
) -> Dict[str, Dict]:
    
    _ensure_dir(outdir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if test_adata is None or test_adata.n_obs == 0: test_adata = train_adata

    scaler = PerGeneMinMaxScaler()
    scaler.fit(_to_numpy(train_adata.X))
    
    tr_loader, tr_ds = make_loader(train_adata, batch_size, shuffle=True, label_key=label_key, scaler=scaler)
    te_loader, te_ds = make_loader(test_adata, batch_size, shuffle=False, label_key=label_key, scaler=scaler)
    
    n_classes = len(pd.unique(train_adata.obs[label_key]))
    model = GeneInformer(num_genes=train_adata.n_vars, num_classes=n_classes, d_model=d_model, keep_ratio=0.5).to(device) # using keep_ratio logic
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_loss = float('inf')
    ckpt_path = os.path.join(outdir, "best_transformer.pth")
    patience = 0
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device).long()
            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval()
        te_loss = 0
        with torch.no_grad():
            for x, y in te_loader:
                x, y = x.to(device), y.to(device).long()
                logits, _ = model(x)
                te_loss += crit(logits, y).item()
        te_loss /= len(te_loader)
        
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)

        if te_loss < best_loss:
            best_loss = te_loss
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
        
        if patience >= early_stop: break
        if (epoch+1) % 10 == 0: logger.log(f"Epoch {epoch+1} | Train: {tr_loss:.4f} | Test: {te_loss:.4f}")

    return {"model_path": ckpt_path, "history": history}