# BRAID-VAE

An interpretable workflow for **bulk breast RNA-seq**: stage-specific VAEs,
MNN/Procrustes alignment, descriptive supervised pseudotime, five source–target
Informer models, and decoder-based Integrated Gradients (IG).

This code-and-documentation release includes recovered analysis scripts and
execution instructions. Expression matrices, sample manifests, gene lists, latent
representations, and model checkpoints are not distributed. **It is not yet a
verified reproduction of every manuscript result from the Figshare source**:
see [data provenance and outstanding gaps](docs/DATA.md) and
[verification status](docs/REPRODUCIBILITY.md). No full retraining is claimed by this release.

## 1. Install

Use Python 3.10 or 3.11, with this repository as the working directory.

```bash
git clone --recurse-submodules https://github.com/xai990/BraidVAE.git
cd BraidVAE
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively: `conda env create -f environment.yml` then `conda activate braid-vae`.
Use the **root** requirements/environment, not the legacy Informer environment.
If already cloned, run `git submodule update --init --recursive`.
The pinned Informer source is included as an HTTPS submodule. CPU execution is
supported; CUDA is selected where available. MPI is not needed for this workflow.
Run scripts from the repository root; installation supplies the shared `bin` package,
while top-level analysis scripts and the submodule are used from the checkout.

## 2. Obtain and prepare data

Primary source: [Figshare Data record 3](https://figshare.com/articles/dataset/Data_record_3/5330593),
[version 2 DOI](https://doi.org/10.6084/m9.figshare.5330593.v2), Wang, Gao and Schultz (2017), CC BY 4.0.

Download source data and prepare inputs after supplying your own sample manifest
and ordered gene list (formats and provenance requirements in [DATA.md](docs/DATA.md)):

```bash
python scripts/download_figshare.py
python scripts/prepare_figshare.py --manifest /path/to/sample_manifest.tsv --genes /path/to/model_genes.txt --outdir datasets_breast
```

The script applies the author-reported `log2(x + 1)` transformation and writes
matrices, labels, and the gene-list format used by `configs/config.yaml`.
Exact recovery of historical model inputs from source has not been fully verified.
Clinical stage labels and the historical split cannot be recovered from the
expression download alone. These normalized data are not raw counts for DESeq2.

## 3. Execute the analysis

Print commands without training:

```bash
python scripts/run_pipeline.py
```

Execute all stages, including expensive decoder IG:

```bash
python scripts/run_pipeline.py --execute --gene-selection elbow
```

Individual commands:

```bash
python brcavae.py --config configs/config.yaml
python align_latent_spaces_mnn.py
python plot_umap_aligned.py
python calc_supervised_pseudotime_aligned.py
python train_cascaded.py --data_path latent_all_stages_aligned.tsv --outdir results_cascaded_aligned
python analyze_cascaded_genes_aligned.py --gene-selection elbow --elbow-sensitivity 3
python plot_informer_dim_histogram_aligned.py
python plot_gene_dimension_heatmap_aligned.py
```

The former README's `train_timeseries_pseudotime.py` and `trace_genes_gradient.py`
remain as historical alternatives, but are **not** the stage-transition/decoder-IG
route above. Pseudotime is descriptive and does not construct Informer training pairs.

### Corrected VAE objective (2026-11-09)

The VAE now uses reconstruction squared error **summed over genes and averaged
across samples**, plus beta times KL divergence summed over latent dimensions and
averaged across samples. This replaces the historical per-gene mean reconstruction
loss, changing the relative weighting of reconstruction and KL.

Checkpoint selection, early stopping and learning-rate scheduling start only when
beta reaches 1: epoch 51 with the default 50-epoch warmup. Training must run for
more epochs than the warmup; `warmup_epochs=0` uses beta=1 from the first epoch.
Both VAE training entry points use the same corrected loop.

This changes new training runs; it does not update existing checkpoints or gene
lists. Outputs from the historical objective and this corrected objective are
separate model versions. To adopt newly trained VAEs downstream, regenerate their
latent coordinates, alignment, transition models, attribution scores and gene
lists together. The shared `VAELoss` also changes the loss scale in historical
alternative training routes; any auxiliary loss weights there require reassessment.

### Principal settings

- VAE: 4,098 features; hidden widths 1024/256/64; 32 latent dimensions;
  batch 16; Adam 0.001; maximum 200 epochs; early-stop patience 10;
  KL warmup 50 epochs; seed 42. Actual settings are in config and function defaults.
- MNN: PCA 20, k=5, TCGA normal reference; orthogonal Procrustes alignment.
- Pseudotime intervals: GTEx [0,.1], TCGA normal [.1,.2], I [.2,.5],
  II [.5,.75], III [.75,1]. These do not estimate elapsed biological time.
- Informer: five predefined comparisons, 100 epochs, batch 32, Adam 0.001,
  d_model=64, heads=4, encoder layers=2; nearest-neighbor pairs. The recovered
  transition implementation uses **full attention** (`attn='full'`), not ProbSparse.
- Decoder IG: zero baseline; five evaluation vectors (one zero and four
  0.5-scaled normal random vectors), 50 integration steps; absolute attributions
  averaged across evaluation vectors and stage VAEs in the recovered code.
- Latent selection: inspect the importance formula in the analysis script;
  mean + one SD threshold. Gene selection is separately configurable.
  The recovered local default is `std` (mean + 2 SD); the manuscript describes
  Kneedle, so the documented manuscript-oriented command explicitly selects
  `elbow` with sensitivity 3. This discrepancy is recorded, not silently erased.

## 4. Figures and outputs

| Analysis / figure | Entry point | Main output |
|---|---|---|
| Stage-specific VAE learning | `brcavae.py` | `results_multistage/stage_*/best_vae.pth`, histories, latent table |
| Aligned representation / Fig. 3 | `plot_umap_aligned.py` | `results_multistage/global_umap_aligned.png` |
| Assigned pseudotime / Fig. 4 | `calc_supervised_pseudotime_aligned.py` | `figures_aligned/` and pseudotime TSV |
| Transition models | `train_cascaded.py` | `results_cascaded_aligned/` checkpoints and scalers |
| IG gene candidates | `analyze_cascaded_genes_aligned.py` | `results_cascaded_aligned_genes/` CSVs |
| Dimension scores / Fig. 5 | `plot_informer_dim_histogram_aligned.py` | same analysis directory |
| Gene heatmaps / Fig. 6 | `plot_gene_dimension_heatmap_aligned.py` | same analysis directory |

For the revised three-panel Figure 2, the optional scripts require locally supplied
historical latents (`resources/latent_all_stages.tsv` and
`resources/latent_all_stages_aligned.tsv`) and prepared model inputs. These data are
not included in this release. Once available, train **only** the pooled unconditional VAE:

```bash
python scripts/pooled/run_pooled.py
python scripts/pooled/plot_figure2.py
python scripts/pooled/orient_figure2.py
```

This creates `Figures/Fig2_original_dataplot.png` and a vector PDF. All three panels
use the same samples and plotting settings; orientation is a distance-preserving
rotation for display, and tick numbers are hidden. No new stage-specific training
is performed by these three commands. Figure 1 is a schematic, not an analysis output.
Downstream enrichment, DESeq2, mutation and PPI table workflows are not fully recovered.

## 5. Verification and interpretation

```bash
python -m unittest discover -s tests -v
```

The original `test` split is used for VAE early stopping. Alignment uses the cohort
jointly. Thus these outputs must not be described as a wholly untouched external test.
Stage labels determine group-specific model assignment; visual separation alone is
not evidence of universally superior representations. IG scores are decoder model
associations, not causal gene effects. Coordinate consistency between aligned latent
selection and native decoder coordinates remains unaudited in this historical code.

Code: MIT (LICENSE). Downloaded Figshare expression data: CC BY 4.0 with the
attribution above. No expression data are bundled here. The Informer submodule retains its own license. Cite the Figshare
record and relevant methods; a manuscript DOI can be added once available.
