# Reviewer Evidence Direct-Run Guide - 2026-07-11

## Purpose

This is the current execution order for the reviewer-evidence branch. The
notebooks now use `auto` controls: they reuse an artifact only when its
contract is compatible; otherwise they either resume the missing fold/cache or
print a precise deferred dependency. No result should be moved into the
manuscript until Notebook 07 completes its strict validation gate.

The evaluated morphology transform is a **fixed-seed ROCKET-family random
convolution MAX+PPV transform**, not canonical MiniRocket. The compact
Transformer is a trained patch Transformer, not a pretrained foundation model.

## Evidence Boundaries

- PTB-XL and Georgia are separate mapped, record-level external tasks. Never
  pool their metrics.
- CPSC2021 is a separate annotation-aligned 10-second AF/AFL mapped-window
  task. Do not pool it with PTB-XL or Georgia, and do not call it the official
  episode-boundary challenge score.
- The group-safe score-calibration runner changes frozen output scores only;
  it does not change model weights.
- The true few-shot runner fits new linear heads on frozen encoder embeddings;
  it is not end-to-end fine-tuning.
- The frozen-transform MLP is a head-capacity sensitivity control. It does not
  make random-convolution kernels learnable or prove determinism versus
  regularization.
- Representation UMAP/probe/CKA is an audit. It cannot be phrased as proven
  morphology-rhythm disentanglement.

## Runtime Selection

| Work | Hardware | What `auto` does |
| --- | --- | --- |
| Notebook 02 canonical OOF or missing external Full-model prediction | A100 High-RAM preferred | Reuses frozen OOF/external artifacts when manifest checks pass; otherwise runs inference. |
| Notebook 03 calibration and presentation assets | CPU | Reuses matching CI/table/figure outputs. Presentation assets defer until current paired artifacts exist. |
| Notebook 04 Raw Mamba, Transformer, frozen-transform MLP | A100 High-RAM | Trains only missing or stale folds; checkpoints and fold prediction caches are mirrored after each fold. |
| Notebook 04 paired comparisons and ledger | CPU | Bootstrap is CPU-bound; no model inference. |
| Notebook 05 comparator stress prediction | A100 High-RAM | Generates or reuses stress prediction files per comparator, stress, and fold. |
| Notebook 05 multi-comparator bootstrap ledger | CPU | Reuses metric cache entries keyed by prediction provenance. |
| Notebook 06 external pooling / probes | CPU for pooling and probe; GPU + Mamba only for missing embeddings | Reuses v3 probe/figure only when its manifest protocol matches. |
| Notebook 07 final matrix and claim gates | CPU | Fails on stale required OOF/freeze/paired contracts rather than reusing them. |

Use a GPU runtime only while a cell reports that inference or training is
needed. Once all model-level artifacts exist, the paired, bootstrap, final
matrix, copy, and publication cells can run on CPU.

## Recommended Sequence

1. Run Notebook 00 and Notebook 01 once in a fresh Colab runtime. They mount
   Drive, pull the current repository, and verify the base protocol.
2. Run Notebook 02 through the canonical OOF, external Full-model export, and
   external protocol-gate cells. This can run before Notebook 04. It will also
   export PTB-XL fold 9 when needed for group-safe adaptation.
3. Run Notebook 03 through calibration CI. Its reviewer-presentation cell may
   print `Deferred` at this point; that is expected until Notebook 04 paired
   artifacts are current.
4. Run Notebook 04. Leave the heavy runner controls as `auto`. For any missing
   Raw Mamba, Transformer, or frozen-transform MLP folds, the notebook runs
   one fold at a time and publishes the Drive mirror immediately. Then run the
   paired-comparison cells and the completion ledger.
5. Return to the new end-of-Notebook-02 cells. Their first pass after Notebook
   04 produces or verifies zero-target-label PTB-XL/Georgia ResNet1D/CNN and
   Raw Mamba outputs. A Transformer is added only if its in-domain OOF and
   paired gate passed. The remaining cells then run external paired bootstrap,
   PTB-XL group-safe score calibration, optional frozen-encoder representations,
   and true linear-head adaptation. Before Notebook 04 these cells defer
   instead of failing.
6. Rerun the reviewer-presentation cell in Notebook 03. It creates the
   reliability figure, compact calibration/paired-CI tables, Q=3 sensitivity
   table, fold-specific PCA variance table, and training-configuration table.
7. Run Notebook 05. Start with its reviewer-minimal stress profile if a full
   six-stress/five-metric multi-comparator bootstrap would exceed the runtime
   budget. Treat a reviewer-minimal profile as named screening evidence; do not
   silently substitute it for the canonical six-stress ledger.
8. Run Notebook 06. External pooling keeps PTB-XL and Georgia separate. Run
   representation extraction only if missing; it needs the same Mamba runtime
   as Full ECG-RAMBA. The v3 probe runs afterwards and produces the audit
   figure plus fold-level probe table.
9. Run Notebook 07. It first runs claim-readiness gates, then writes final
   evidence tables only when calibration and paired OOF contracts match the
   active frozen OOF. It mirrors all artifacts and copies the current source
   tables to `final_evidence_tables`.
10. Build the marked manuscript only in an environment with both `latexdiff`
    and `latexmk`/LaTeX. A `blocked_missing_tool` manifest is deliberate and
    must not be called a marked PDF.

## Restart And Resume Rules

- After a Colab disconnect, rerun the current notebook from **Setup**. Do not
  delete local artifacts. Targeted Drive restore will copy only missing files.
- A fresh Colab clone can use these controls only after the current notebook
  and runner changes are committed and pushed. Pulling an older `main` branch
  will correctly fail the compatibility preflight rather than hot-patching a
  stale runner.
- For model runners, leave `ONLY_FOLDS='auto'`, `SAVE_CHECKPOINTS=True`, and
  `REUSE_CHECKPOINTS=True`. The notebook detects existing per-fold caches and
  skips them.
- The external learned-comparator runner reads checkpoints directly from
  `revision_artifacts/reports/revision/experimental/` on Drive. Its final
  manifest includes frozen OOF/freeze SHA256, checkpoint SHA256, archive SHA256,
  mapping SHA256 where relevant, and runner SHA256. Old manifests are
  regenerated from compatible fold caches rather than accepted blindly.
- Metric caches use input/protocol-derived cache keys. A changed prediction,
  group split, threshold, bootstrap count, or seed produces a new cache entry.
- Notebook 02 `auto` cells re-enter the lightweight paired/calibration/head
  runners so they can validate canonical OOF/freeze and runner SHA contracts;
  compatible metric/head caches are reused and model inference is skipped.
- Every heavy cell writes an ordinary command log under
  `reports/revision/logs/` and publishes to the Drive mirror after successful
  fold or aggregate completion.
- Mirror publication is merge-only and checksum-verified: a partial Colab
  runtime updates artifacts it owns but preserves previously verified fold
  checkpoints/caches in the manifest. It does not prune artifacts that are
  absent from the current runtime.

## Required Completion Artifacts

The following outputs prove a completed item; file presence alone is not
sufficient because Notebook 07 and claim gates also inspect manifests and
checksums.

| Reviewer item | Primary artifacts |
| --- | --- |
| R1-C1/R1-C5 reliability and paired CI presentation | `figure_calibration_audit.png`, `table_calibration_ci_compact.csv`, `table_paired_baseline_ci_compact.csv`, `reviewer_completion_input_contract.json` |
| R1-C2 head-capacity sensitivity | `hybrid_morphology_baseline_summary.json`, paired Hybrid comparison, five checkpoint files |
| R1-C4 Transformer comparator | `transformer_ecg_baseline_summary.json`, paired Transformer comparison, five checkpoint files |
| R1-C6 pooling sensitivity | `pooling_sensitivity.csv`, `pooling_sensitivity_external.csv`, Q=3 paired bootstrap output |
| R1-C7 PCA/config appendix | `table_fold_pca_provenance.csv`, `table_training_configuration.csv`, morphology transform contract |
| R2-C2 representation audit | v3 `representation_probe_manifest.json`, fold probe table, CKA table, audit figure |
| R2-C4 adaptation | group-safe score-calibration manifest and true frozen-encoder linear-head manifest, both PTB-XL fold 9/10 only |
| Editorial marked manuscript | `marked_manuscript_manifest.json` with `editorial_ready=true` and a nonempty marked PDF |

## Final Safe Wording Gate

The only source of truth for manuscript/rebuttal wording is the regenerated
pair:

- `reports/revision/tables/table_final_evidence_matrix.csv`
- `reports/revision/tables/table_final_safe_wording.csv`

The convenience copies in Drive are for handoff only. Before submission,
rerun Notebook 07, compile the manuscript, create fresh PDF text, and run the
forbidden-claim scan. Do not claim SOTA, broad in-domain superiority,
zero-shot superiority, general external superiority, end-to-end few-shot
fine-tuning, proven disentanglement, general robustness superiority, complete
HRV, or clinical readiness unless a future protocol-specific artifact makes
that wording valid.
