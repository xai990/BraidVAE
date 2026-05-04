# BRAID-VAE: Breast Cancer Stage Integration
and Dynamics with an Interpretable Multi-VAE
Framework

A Python-based computational pipeline for multi-modal bulk RNA-seq analysis, pseudotime inference, sequence prediction, and gradient attribution.

---

## Pipeline Overview

```
MultiVAE  ──►  MNNAlign  ──►  PseudotimeAssign  ──►  Informer  ──►  Gradient
   │             │                  │                  │               │
Embed &      Align latent       Order cells        Predict       Attribute
compress     spaces             along time         dynamics      importance
```

| Stage | Script | Description |
|---|---|---|
| 1. MultiVAE | `brcavae.py` | Variational autoencoder workflow that learns stage-aware latent embeddings from transcriptomic data |
| 2. MNNAlign | `align_latent_spaces_mnn.py` | Aligns stage-specific latent spaces with mutual nearest neighbors (MNN) + Procrustes alignment |
| 3. PseudotimeAssign | `calc_supervised_pseudotime_aligned.py` | Assigns supervised pseudotime on the aligned latent space |
| 4. Informer | `train_timeseries_pseudotime.py` | Informer-based time-series model that predicts latent dynamics along pseudotime |
| 5. Gradient | `trace_genes_gradient.py` | Computes gradient-based attribution scores to identify genes driving progression-relevant latent dimensions |

---

## Requirements

- Python >= 3.8
- conda (recommended) or pip

### Install with conda

```bash
conda env create -f Informer2020/environment.yml
conda activate informer2020
pip install -e .
```

### Install with pip

```bash
pip install -r Informer2020/requirements.txt
```

---

## Usage

```bash
# Stage 1: Train MultiVAE and export latent embeddings
python brcavae.py --config configs/config.yaml

# Stage 2: Align latent spaces using MNN + Procrustes
python align_latent_spaces_mnn.py

# Stage 3: Assign Supervised Pseudotime on aligned latent space
python calc_supervised_pseudotime_aligned.py

# Stage 4: Train Informer on Pseudotime Trajectories
python train_timeseries_pseudotime.py --data_path latent_all_stages_aligned_supervised_pseudotime.tsv --outdir results_pseudotime

# Stage 5: Compute Gradient-Based Gene Attribution
python trace_genes_gradient.py
```

> Adjust paths and hyperparameters as needed for your local data layout.

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── setup.py
├── .gitignore
├── .gitmodules
├── brcavae.py                         # Stage 1: MultiVAE workflow
├── align_latent_spaces_mnn.py         # Stage 2: MNN latent alignment
├── calc_supervised_pseudotime_aligned.py  # Stage 3: Aligned pseudotime assignment
├── train_timeseries_pseudotime.py     # Stage 4: Informer training
├── trace_genes_gradient.py            # Stage 5: Gradient attribution
├── configs/
│   └── config.yaml                    # Shared pipeline configuration
├── bin/                               # Shared modules imported by pipeline scripts
└── Informer2020/                      # Informer implementation (submodule)
```

> Note: Results, data files, notebooks, and QC scripts are intentionally excluded from version control. See `.gitignore` for details.

---

## Configuration

Key parameters can be adjusted in `configs/config.yaml`:

```yaml
data:
  train_path: "datasets_breast/breast_train_GEM_transpose.txt"
  train_label_path: "datasets_breast/train_stage_labels_new_updated.txt"
  test_path: "datasets_breast/breast_test_GEM_transpose.txt"
  test_label_path: "datasets_breast/test_stage_labels_new_updated.txt"
  gene_list: "datasets_breast/hallmark_allgene_transposed.txt"
  filter: "replace"

train:
  hidden_dims: [1024, 256, 64]
  latent_dim: 32
```

---

## .gitignore Summary

The following are excluded from version control:

- `results*`, `outputs/`, `figures/`, `plots/` - generated outputs
- `data/`, `raw/`, `processed/` - input data folders
- `*.ipynb`, `.ipynb_checkpoints/` - notebooks
- `*.pkl`, `*.pt`, `*.h5ad`, `*.npy`, `*.npz`, `*.csv`, `*.tsv` - binary/data artifacts
- QC and validation scripts such as `lint_*.py`, `check_*.py`, `qc_*.py`, `validate_*.py`
- `venv/`, `env/`, `__pycache__/`, `*.pyc` - environments and cache files

---

## Citation

If you use this pipeline in your research, please cite accordingly.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
