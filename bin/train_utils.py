
import os
import math
import json
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from . import logger 
from .model import ConditionalVAE, VAELoss
from .dataset import PerGeneMinMaxScaler, _to_numpy, make_loader 
from .utils import extract_embeddings,visualize_embeddings

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ===================== Multi-Stage Orchestrator =====================

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
) -> Dict[str, Dict]:
    """
    Trains one VAE per stage value in train_adata.obs[label_key].
    Returns dict:
      stage -> {
         "history": ...,
         "model_path": ...,
         "latent_train_path": ...,
         "latent_test_path": ...
      }
    """
    _ensure_dir(outdir)

    # stages present in train; (optionally include those present only in test)
    stages = list(pd.unique(train_adata.obs[label_key]))
    # add any test-only stages
    for s in pd.unique(test_adata.obs[label_key]):
        if s not in stages:
            stages.append(s)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary: Dict[str, Dict] = {}

    for s in stages:
        logger.log(f"\n=== Stage: {s} ===")
        stage_dir = os.path.join(outdir, f"stage_{str(s)}")
        _ensure_dir(stage_dir)

        tr_s = train_adata[train_adata.obs[label_key] == s].copy()
        te_s = test_adata[test_adata.obs[label_key] == s].copy()

        if tr_s.n_obs == 0:
            logger.log(f"  [skip] No training samples for stage={s}")
            continue

        # Fit scaler on train of this stage; apply to both
        scaler = PerGeneMinMaxScaler()
        scaler.fit(_to_numpy(tr_s.X))
        tr_loader, tr_ds = make_loader(tr_s, batch_size, shuffle=True,  label_key=label_key, scaler=scaler)
        if te_s.n_obs > 0:
            te_loader, te_ds = make_loader(te_s, batch_size, shuffle=False, label_key=label_key, scaler=scaler)
        else:
            logger.log(f"  [warn] No test samples for stage={s}; using train as test for eval")
            te_loader, te_ds = tr_loader, tr_ds

        n_classes = 0
        if label_key is not None and label_key in tr_s.obs.columns:
            n_classes = len(pd.unique(tr_s.obs[label_key].astype(str)))

        model = ConditionalVAE(
            input_dim=tr_s.n_vars,
            latent_dim=latent_dim,
            hidden_dims=list(hidden_dims),
            n_classes=n_classes,
            dropout=dropout,
        )

        ckpt = os.path.join(stage_dir, "best_vae.pth")
        hist = train_vae_with_test_set(
            model,
            tr_loader,
            te_loader,
            n_epochs=epochs,
            learning_rate=lr,
            device=device,
            early_stopping_patience=early_stop,
            warmup_epochs=warmup_epochs,
            ckpt_path=ckpt,
        )

        # Extract latent embeddings for this stage (train & test separately)
        # reload best weights to be safe
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()

        # --- Extract latent embeddings for train/test ---
        Z_tr, names_tr,_ = extract_embeddings(model, tr_loader, device)
        df_tr = pd.DataFrame(Z_tr, index=names_tr, columns=[f"z{i+1}" for i in range(Z_tr.shape[1])])
        logger.log(f"The information of df_tr:{df_tr.head()}")
        df_tr["split"] = "train"
        if label_key and label_key in tr_s.obs.columns:
            df_tr[label_key] = tr_s.obs.loc[names_tr, label_key].values

        if te_s.n_obs > 0:
            Z_te, names_te,_ = extract_embeddings(model, te_loader, device)
            df_te = pd.DataFrame(Z_te, index=names_te, columns=[f"z{i+1}" for i in range(Z_te.shape[1])])
            df_te["split"] = "test"
            if label_key and label_key in te_s.obs.columns:
                df_te[label_key] = te_s.obs.loc[names_te, label_key].values
            df_all = pd.concat([df_tr, df_te], axis=0)
        else:
            df_all = df_tr
        latent_path = os.path.join(stage_dir, "latent_all.tsv") 
        df_all.to_csv(latent_path, sep="\t")                    
        logger.log(f"[save] Latent embeddings saved to {latent_path}") 
        # visualize embeddings
        try:
            
            label_encoder = tr_ds.label_encoder if hasattr(tr_ds, "label_encoder") else None
            label_names = label_encoder.classes_ if label_encoder is not None else None
            visualize_embeddings(
                df_all.iloc[:, :11].values,
                labels=df_all[label_key].values if label_key in df_all.columns else None,
                label_names=label_names,
                save_path=os.path.join(stage_dir, f"latent_umap_{s}.png")
            )
        except Exception as e:
            logger.log(f"[warn] Failed to visualize embeddings for {s}: {e}")
        # Save history
        with open(os.path.join(stage_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)

        summary[str(s)] = {
            "history": hist,
            "model_path": ckpt,
            # "latent_train_path": tr_lat_path,
            # "latent_test_path": te_lat_path,
        }

    return summary


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
    best = None
    patience = 0

    for epoch in range(n_epochs):
        # KL warmup
        criterion.beta = min(1.0, epoch / max(1, warmup_epochs))

        # ---- train ----
        model.train()
        tr_loss = tr_rec = tr_kl = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                y = y.to(device)
            else:
                x = batch
                y = None
            x = x.to(device)

            opt.zero_grad()
            recon, inp, mu, logvar, z, _= model(x, y)
            loss, dct = criterion(recon, inp, mu, logvar)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_rec += dct["recon"]
            tr_kl  += dct["kl"]

        ntr = max(1, len(train_loader))
        tr_loss /= ntr
        tr_rec  /= ntr
        tr_kl   /= ntr

        # ---- test ----
        model.eval()
        te_loss = te_rec = te_kl = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                    y = y.to(device)
                else:
                    x = batch
                    y = None
                x = x.to(device)
                recon, inp, mu, logvar, z,_ = model(x, y)
                loss, dct = criterion(recon, inp, mu, logvar)
                te_loss += loss.item()
                te_rec  += dct["recon"]
                te_kl   += dct["kl"]

        nte = max(1, len(test_loader))
        te_loss /= nte
        te_rec  /= nte
        te_kl   /= nte
        # Replace non-finite test loss with +inf for logic/scheduler
        finite_te = np.isfinite(te_loss)
        te_loss_for_sched = te_loss if finite_te else float("inf")

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_recon"].append(tr_rec)
        history["train_kl"].append(tr_kl)
        history["test_recon"].append(te_rec)
        history["test_kl"].append(te_kl)

        sched.step(te_loss_for_sched)

        # Save on first epoch, or any improvement with finite test loss
        improved = (best is None) or (finite_te and te_loss < best)
        if improved:
            best = te_loss if finite_te else best
            patience = 0
            _ensure_dir(os.path.dirname(ckpt_path) or ".")
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                logger.log(f"[Early stop] epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            logger.log(
                f"Epoch {epoch+1:03d} | train {tr_loss:.4f} (R {tr_rec:.4f} KL {tr_kl:.4f}) | "
                f"test {te_loss:.4f} (R {te_rec:.4f} KL {te_kl:.4f})"
            )
    # load best
    # ---- ensure a checkpoint exists; if not, save current ----
    if not os.path.exists(ckpt_path):
        _ensure_dir(os.path.dirname(ckpt_path) or ".")
        torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return history


