# Data provenance and preparation

## Primary expression source

Wang, Qingguo; Gao, Jianjiong; Schultz, Nikolaus (2017).
**Unified RNA-seq datasets in human cancers and normal tissues – normalized data**.
Figshare, version 2. DOI: https://doi.org/10.6084/m9.figshare.5330593.v2
Record: https://figshare.com/articles/dataset/Data_record_3/5330593
License of the source data: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

| Reference group | File | Figshare file ID |
|---|---|---|
| TCGA breast tumors | brca-rsem-fpkm-tcga-t.txt.gz | 9150571 |
| TCGA normal breast | brca-rsem-fpkm-tcga.txt.gz | 9150574 |
| GTEx breast | breast-rsem-fpkm-gtex.txt.gz | 9150577 |

Download approximately 45 MB and verify source MD5 checksums:

```bash
python scripts/download_figshare.py
```

The downloaded files have genes in rows, `Hugo_Symbol` and `Entrez_Gene_Id`
annotation columns, and complete sample identifiers in the remaining columns.
They are normalized expression, not raw integer read counts. Do not pass these
values or the model-input snapshot to DESeq2 as raw counts.

## Prepare local inputs

This repository publishes code and documentation only. Expression matrices,
sample-level labels/splits, gene lists, latent representations, and checkpoints
are not included. Download expression data using the command above, then provide:

- A tab-separated manifest with header `sample_id`, `label`, `split` (in that
  order). IDs must match full sample column names in the source files. Supported
  labels are `normal_GTEX`, `normal_TCGA`, `Stage_I`, `Stage_II`, `Stage_III`;
  split values are `train` and `test` (the latter is used as VAE validation).
- An ordered text file containing one selected gene symbol per line. Symbols must
  match `Hugo_Symbol` in the source matrices. The historical analysis used 4,098
  selected genes; the original gene-set database release has not been recovered.

Obtain tumor stage labels from clinical annotations and record the annotation
release, matching rules, inclusion criteria, and split procedure. Do not infer
clinical stage from expression values. The historical clinical annotation version,
label-curation script, and original split seed have not been recovered, so the
expression download alone does not recreate the historical sample manifest.

```bash
python scripts/prepare_figshare.py --manifest /path/to/sample_manifest.tsv --genes /path/to/model_genes.txt --outdir datasets_breast
```

The preparation script:

1. Reads the three normalized expression matrices and transposes them to samples by genes.
2. Combines the source groups and selects samples from your manifest.
3. Selects genes in the order of your gene list.
4. Applies the author-reported `log2(x + 1)` transformation, preserving missing values.
5. Writes training and validation matrices, headerless sample-label files, and
   `hallmark_allgene_transposed.txt` in the format expected by the VAE loader.
   `preprocessing.json` records source, transformation, and output dimensions.

With `--outdir datasets_breast`, the output paths match `configs/config.yaml`.
Exact recovery of the historical model inputs from source has not been fully
verified. The historical `test` split was used for checkpoint selection and is
therefore validation data, not an untouched test set.

## Optional historical-input utilities

`scripts/prepare_inputs.py` is an optional utility for users who already possess
all historical input files. It expects a local `resources/` directory containing
`model_input_train.tsv.gz`, `model_input_test.tsv.gz`, `sample_manifest.tsv`,
`model_genes.txt`, `hallmark_allgene_transposed.txt`, and `input_checksums.json`.
The checksum file is a JSON list of objects with `file` and `sha256` fields.
These files are not distributed, and this utility is not part of the default runner.
It restores historical inputs without applying another log transformation.

## Model-side preprocessing

After loading model inputs, the existing VAE code replaces missing entries through
its `replace` policy, applies `log1p(exp(x))`, and fits per-gene min–max scaling on
training data separately for each stage. Do not apply an extra log transform to
already transformed historical inputs. MNN has its own preprocessing in `align_latent_spaces_mnn.py`.

The DESeq2 raw-count accession and preparation scripts, GDC query scripts, STRING
query settings, and full ToppFun export workflow are not present in this source
snapshot. They remain required to reproduce all downstream biological tables.
