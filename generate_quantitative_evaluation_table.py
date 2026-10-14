#!/usr/bin/env python3
"""
Generate reviewer-ready quantitative evaluation tables for:
1) VAE quality (reconstruction/KL/ELBO-style metrics)
2) Latent stage-separation and alignment quality
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STAGE_ORDER = ["normal_GTEX", "normal_TCGA", "Stage_I", "Stage_II", "Stage_III"]
NORMAL_LABELS = ["normal_GTEX", "normal_TCGA"]

LATENT_SPACES = {
    "Raw VAE latent": "latent_all_stages.tsv",
    "CCA-aligned latent": "latent_all_stages_aligned_cca.tsv",
    "MNN+Procrustes aligned latent": "latent_all_stages_aligned.tsv",
}

HISTORY_ROOT = Path("results_multistage")
OUTDIR = Path("results_multistage")


def latent_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("z")]


def stage_centroid_order_spearman(X: np.ndarray, labels: np.ndarray) -> float:
    centroids = []
    stages_present = []
    for s in STAGE_ORDER:
        mask = labels == s
        if np.any(mask):
            centroids.append(X[mask].mean(axis=0))
            stages_present.append(s)

    if len(centroids) < 3:
        return float("nan")

    C = np.vstack(centroids)
    Cc = C - C.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Cc, full_matrices=False)
    axis = vt[0]
    proj = Cc @ axis
    stage_idx = np.array([STAGE_ORDER.index(s) for s in stages_present], dtype=float)
    rho, _ = spearmanr(stage_idx, proj)
    return float(abs(rho))


def normal_alignment_metrics(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    mask = np.isin(labels, NORMAL_LABELS)
    Xn = X[mask]
    yn = labels[mask]

    if Xn.shape[0] < 3 or len(np.unique(yn)) < 2:
        return float("nan"), float("nan"), float("nan")

    sil = float(silhouette_score(Xn, yn))

    gtex = Xn[yn == "normal_GTEX"]
    tcga = Xn[yn == "normal_TCGA"]
    if len(gtex) == 0 or len(tcga) == 0:
        centroid_dist = float("nan")
    else:
        centroid_dist = float(np.linalg.norm(gtex.mean(axis=0) - tcga.mean(axis=0)))

    d = cdist(Xn, Xn, metric="euclidean")
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argmin(d, axis=1)
    cross_rate = float(np.mean(yn[nn_idx] != yn))

    return sil, centroid_dist, cross_rate


def evaluate_latent_space(df: pd.DataFrame) -> Dict[str, float]:
    zcols = latent_columns(df)
    X = df[zcols].to_numpy(dtype=float)
    y = df["label"].astype(str).to_numpy()
    split = df["split"].astype(str).to_numpy()

    metrics: Dict[str, float] = {}
    metrics["n_samples"] = int(len(df))

    metrics["stage_silhouette"] = float(silhouette_score(X, y))
    metrics["centroid_order_spearman_abs"] = stage_centroid_order_spearman(X, y)

    n_sil, n_dist, n_mix = normal_alignment_metrics(X, y)
    metrics["normal_silhouette"] = n_sil
    metrics["normal_centroid_distance"] = n_dist
    metrics["normal_cross_cohort_1nn_rate"] = n_mix

    train_mask = split == "train"
    test_mask = split == "test"
    if np.any(train_mask) and np.any(test_mask):
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        multi_class="multinomial",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        clf.fit(X[train_mask], y[train_mask])
        pred = clf.predict(X[test_mask])
        y_true = y[test_mask]
        metrics["test_accuracy"] = float(accuracy_score(y_true, pred))
        metrics["test_balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
        metrics["test_macro_f1"] = float(f1_score(y_true, pred, average="macro"))
    else:
        metrics["test_accuracy"] = float("nan")
        metrics["test_balanced_accuracy"] = float("nan")
        metrics["test_macro_f1"] = float("nan")

    return metrics


def collect_vae_metrics() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    base_latent = pd.read_csv("results_multistage/latent_all_stages.tsv", sep="\t", index_col=0)
    test_counts = (
        base_latent[base_latent["split"] == "test"]["label"].value_counts().to_dict()
    )

    for hist_path in sorted(HISTORY_ROOT.glob("stage_*/history.json")):
        stage_name = hist_path.parent.name.replace("stage_", "")
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        test_loss = np.array(hist["test_loss"], dtype=float)
        test_recon = np.array(hist["test_recon"], dtype=float)
        test_kl = np.array(hist["test_kl"], dtype=float)

        best_loss_idx = int(np.argmin(test_loss))
        elbo_proxy = test_recon + test_kl
        best_elbo_idx = int(np.argmin(elbo_proxy))

        rows.append(
            {
                "stage": stage_name,
                "n_test": int(test_counts.get(stage_name, 0)),
                "epochs_trained": int(len(test_loss)),
                "best_epoch_by_test_loss": int(best_loss_idx + 1),
                "best_test_loss": float(test_loss[best_loss_idx]),
                "best_epoch_by_elbo_proxy": int(best_elbo_idx + 1),
                "best_elbo_proxy_recon_plus_kl": float(elbo_proxy[best_elbo_idx]),
                "best_elbo_proxy_neg_recon_plus_kl": float(-elbo_proxy[best_elbo_idx]),
                "best_elbo_test_recon": float(test_recon[best_elbo_idx]),
                "best_elbo_test_kl": float(test_kl[best_elbo_idx]),
            }
        )

    per_stage = pd.DataFrame(rows).sort_values(
        by="stage", key=lambda s: s.map({name: i for i, name in enumerate(STAGE_ORDER)})
    )

    weights = per_stage["n_test"].to_numpy(dtype=float)
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)

    summary = pd.DataFrame(
        [
            {
                "metric": "weighted_mean_best_test_loss",
                "value": float(np.average(per_stage["best_test_loss"], weights=weights)),
            },
            {
                "metric": "weighted_mean_best_elbo_test_recon",
                "value": float(np.average(per_stage["best_elbo_test_recon"], weights=weights)),
            },
            {
                "metric": "weighted_mean_best_elbo_test_kl",
                "value": float(np.average(per_stage["best_elbo_test_kl"], weights=weights)),
            },
            {
                "metric": "weighted_mean_best_elbo_proxy_recon_plus_kl",
                "value": float(np.average(per_stage["best_elbo_proxy_recon_plus_kl"], weights=weights)),
            },
            {
                "metric": "weighted_mean_best_elbo_proxy_neg_recon_plus_kl",
                "value": float(np.average(per_stage["best_elbo_proxy_neg_recon_plus_kl"], weights=weights)),
            },
        ]
    )

    return per_stage, summary


def render_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else float_fmt.format(x))
        else:
            out[col] = out[col].astype(str)

    headers = list(out.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    per_stage_vae, vae_summary = collect_vae_metrics()
    per_stage_path = OUTDIR / "supp_table_vae_quality_by_stage.csv"
    summary_path = OUTDIR / "supp_table_vae_quality_summary.csv"
    per_stage_vae.to_csv(per_stage_path, index=False)
    vae_summary.to_csv(summary_path, index=False)

    latent_rows = []
    for method_name, path in LATENT_SPACES.items():
        df = pd.read_csv(path, sep="\t", index_col=0)
        metrics = evaluate_latent_space(df)
        metrics["latent_space"] = method_name
        latent_rows.append(metrics)

    latent_df = pd.DataFrame(latent_rows)
    cols = [
        "latent_space",
        "n_samples",
        "stage_silhouette",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "centroid_order_spearman_abs",
        "normal_silhouette",
        "normal_centroid_distance",
        "normal_cross_cohort_1nn_rate",
    ]
    latent_df = latent_df[cols]
    latent_path = OUTDIR / "supp_table_latent_alignment_and_stage_separation.csv"
    latent_df.to_csv(latent_path, index=False)

    md_path = OUTDIR / "supplementary_quantitative_tables.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Supplementary Quantitative Evaluation Tables\n\n")
        f.write("## Table S1. VAE Quality by Stage\n\n")
        f.write(render_markdown_table(per_stage_vae))
        f.write("\n\n")
        f.write("## Table S2. Latent Alignment and Stage-Separation Metrics\n\n")
        f.write(render_markdown_table(latent_df))
        f.write("\n")

    print("Wrote:")
    print(f"  - {per_stage_path}")
    print(f"  - {summary_path}")
    print(f"  - {latent_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
