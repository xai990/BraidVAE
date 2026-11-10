# Verification and remaining limitations

Checked on 2026-10-14, using Python 3.10 and the direct dependency versions pinned
in root requirements.txt. This is a recovered workflow release, not a claim of a
complete historical raw-data-to-paper rerun.

The public release contains code and documentation only. Local expression data,
labels, gene lists, latent representations, and checkpoints are excluded. Checks
using historical inputs below were performed locally; these files are not bundled.

## Completed checks

- Editable installation in a temporary environment sharing the existing dependency stack (not a fresh dependency resolution).
- Wheel build with `pip wheel . --no-deps --no-build-isolation`.
- Syntax parsing of 27 Python source files in the main pipeline and new scripts.
- Imports of VAE entry point, alignment, transition training, and decoder IG analysis.
- Two data-independent model tests: nonlinear decoder IG output/completeness
  smoke test and Informer forward interface/finite output.
- Optional local input checksums/order/counts test (skipped without historical files).
- `prepare_inputs.py`: 759 training and 379 validation samples, 4,098 ordered genes.
- Pipeline dry run: commands and file handoffs reviewed.
- All three downloaded Figshare breast archives match source MD5 checksums.
- Source preparation implements the author-reported `log2(x + 1)` transformation,
  selected genes, and saved sample split. Exact historical input recovery is not established.

## Not established by these checks

- A clean-room full dependency installation on all operating systems, GPU
  reproducibility, or full retraining of the VAE/Informer/IG workflow.
- Exact numerical recovery of historical figures, gene sets, or downstream tables.
- Complete provenance of historical expression transformation and clinical labels.
- Original clinical data version, stage-label curation procedure and split seed.
- Gene-set database release used to obtain the recovered feature list.
- Decoder-native versus aligned-coordinate attribution consistency. The recovered
  code is exposed without changing the scientific method to resolve this issue.
- External validation, patient-level longitudinal progression or causal gene effects.

## Historical differences to reconcile before submission

1. Local aligned gene-analysis script defaulted to mean + 2 SD gene selection;
   the paper states Kneedle. Both options are explicit; the runner requests Kneedle.
2. The recovered transition model uses full attention, despite ProbSparse wording
   in the manuscript. Review the manuscript against actual training code.
3. The decoder IG samples latent vectors rather than patient posterior means.
4. VAE checkpoint selection uses the split historically called test. Alignment
   is not isolated to a training-only cohort.
5. DESeq2 requires a separately documented raw-count source. Figshare Data record 3
   supplies normalized expression and cannot support the raw-count claim by itself.
6. Complete GDC, STRING, ToppFun, and DESeq2 reproduction scripts were not found
   in the supplied repository. The manual/document-analysis workflow is not bundled.

Public scripts and explicit limitations are preferable to asserting these steps
were verified. New method changes should be validated separately before updating
scientific conclusions.

## Corrected VAE objective update (2026-11-09)

The shared VAE loss now sums reconstruction squared error over genes and averages
it over samples, matching the per-sample scale of the KL term. The shared training
loop defers checkpoint selection, early stopping and learning-rate scheduling
until the KL weight reaches one (epoch 51 for the default 50-epoch warmup).
The legacy training entry point delegates to this loop. Empty loaders, insufficient
epoch budgets and runs with no finite post-warmup validation loss are rejected;
a stale checkpoint cannot silently be loaded from a previous run.

Validation for this update: nine unittest checks passed locally (including the
optional historical-input check), and `pip wheel . --no-deps --no-build-isolation`
succeeded. The new regression checks cover loss and gradient scaling, batch-size
invariance, warmup selection and scheduler timing, stale-checkpoint rejection,
real-model training and best-weight restoration, and the legacy entry point.
These checks validate software behavior, not end-to-end scientific results.

This is a new training objective, not a retroactive change to existing model
weights or manuscript results. Historical checkpoints and gene lists remain
separate artifacts; adopting new checkpoints requires regenerating downstream
alignment, transition models and attribution results consistently.
