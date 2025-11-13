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
from sklearn.preprocessing import LabelEncoder


def load_txt_to_anndata(
    data_path: str,
    label_path: str,
    label_col: str = "label",
    sample_col_in_labels: Optional[str] = None,
    sep: str = "\t",
    drop_unknown: bool = True,
    has_header_labels: bool = False,
    gene_list_path: Optional[str] = None,
    min_class_size: int = 20,
) -> ad.AnnData:
    """
    Reads a samples×genes TSV and a label TSV, merges on sample IDs, returns AnnData.

    data_path: TSV with index=sample_id, columns=genes
    label_path: TSV with sample_id and at least one label column (e.g., label, stage)
                If has_header_labels=False, infers columns as:
                col0=sample_id, col1=label, (optional) col2=stage
    """
    # gene matrix
    df = pd.read_csv(data_path, sep=sep, index_col=0)
    logger.info(f"The size of the df:{df.shape}")
    # labels
    if has_header_labels:
        df_lab = pd.read_csv(label_path, sep=sep)
        if sample_col_in_labels is None:
            # best guess: first column is sample id if not specified
            sample_col_in_labels = df_lab.columns[0]
    else:
        df_lab = pd.read_csv(label_path, sep=sep, header=None)
        # infer columns
        cols = ["sample_id", label_col]
        df_lab.columns = cols
        sample_col_in_labels = "sample_id"
    if gene_list_path is not None and os.path.exists(gene_list_path):
        try:
            gene_set_genes = set()
            with open(gene_list_path, 'r') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    gene_set_name = parts[0]
                    genes = [g.strip() for g in parts[1:] if g.strip()]
                    gene_set_genes.update(genes)
            logger.info(f" genes set : has {len(gene_set_genes)}")
            common_genes = [g for g in df.columns if g in gene_set_genes]
            logger.info(f"Filtering genes: kept {len(common_genes)} / {len(df.columns)} from list")

            if len(common_genes) == 0:
                logger.log("[warn] No overlap between gene set and data columns")
            else:
                df = df.loc[:, common_genes]
        except Exception as e:
            logger.log(f"[warn] Failed to apply gene list filter: {e}")

    
    if drop_unknown:
        # build a normalized helper column for robust matching
        unknown_markers = {"unknown_tumor_type"}
        norm = (
            df_lab[label_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        keep_mask = ~norm.isin(unknown_markers)
        df_lab = df_lab.loc[keep_mask].copy()
    # drop label classes with fewer than `min_class_size` samples
    if min_class_size is not None and min_class_size > 0:
        counts = df_lab[label_col].value_counts()
        keep_labels = counts[counts >= min_class_size].index
        dropped_labels = counts[counts < min_class_size].index.tolist()

        if len(dropped_labels) > 0:
            logger.info(
                f"Dropping classes with < {min_class_size} samples: "
                f"{dropped_labels}"
            )

        df_lab = df_lab[df_lab[label_col].isin(keep_labels)].copy()
    # merge by sample IDs
    df_lab = df_lab.set_index(sample_col_in_labels)
    df = df.loc[df.index.intersection(df_lab.index)].copy()
    obs = pd.DataFrame(index=df.index)
    if label_col in df_lab.columns:
        obs[label_col] = df_lab.loc[obs.index, label_col].astype(str).values

    adata = ad.AnnData(X=df.values.astype(np.float32))
    adata.obs = obs
    adata.var = pd.DataFrame(index=df.columns)
    return adata

def align_genes(train_adata: ad.AnnData, test_adata: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData]:
    """Intersect genes and order identically for train/test."""
    common = train_adata.var_names.intersection(test_adata.var_names)
    train_adata = train_adata[:, common].copy()
    test_adata = test_adata[:, common].copy()
    return train_adata, test_adata


# ===================== Preprocess =====================

class PerGeneMinMaxScaler:
    """
    Fit on train (per stage) and transform train/test with the same params.
    Scales each gene to [-1, 1].
    """
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray):
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        # avoid zero range
        eps = 1e-12
        self.max_ = np.where(np.isfinite(self.max_), self.max_, 0.0)
        self.min_ = np.where(np.isfinite(self.min_), self.min_, 0.0)
        self.range_ = np.maximum(self.max_ - self.min_, eps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.min_) / self.range_
        return (Z * 2.0 - 1.0).astype(np.float32)



def preprocess_anndata(
    adata: ad.AnnData,
    nan_policy: str = "drop"   # "replace" or "drop"
) -> ad.AnnData:
    X = _to_numpy(adata.X).astype(np.float32)
    if nan_policy == "drop":
        keep = ~np.isnan(X).any(axis=0)
        adata = adata[:, keep].copy()
        X = X[:, keep]  # ensure same shape
    else:
        X = np.nan_to_num(X, nan=0)

    adata.X = X.astype(np.float32)
    return adata



# ===================== Dataset / Loader =====================

class AnnDataGeneDataset(Dataset):
    def __init__(
        self,
        adata: ad.AnnData,
        label_key: Optional[str] = None,
        scaler: Optional[PerGeneMinMaxScaler] = None,
        label_encoder: Optional[LabelEncoder] = None,   # <-- new
    ):
        self.obs_names = adata.obs_names
        self.X = _to_numpy(adata.X).astype(np.float32)
        self.label_key = label_key
        self.y = None
        self.label_encoder = label_encoder

        if label_key is not None and label_key in adata.obs.columns:
            labels = adata.obs[label_key].astype(str).values
            if self.label_encoder is None:
                # fit if not provided (e.g., for train)
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(labels)
            self.y = self.label_encoder.transform(labels).astype(np.int64)

        if scaler is not None:
            self.X = scaler.transform(self.X)

        self.X = self.X.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is None:
            return x
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def make_loader(adata: ad.AnnData, batch_size: int, shuffle: bool, label_key: Optional[str], scaler: Optional[PerGeneMinMaxScaler]):
    ds = AnnDataGeneDataset(adata, label_key=label_key, scaler=scaler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0), ds


def _to_numpy(x):
    # AnnData X can be dense or scipy sparse
    try:
        import scipy.sparse as sp
        if sp.issparse(x):
            return x.A
    except Exception:
        pass
    return np.asarray(x)