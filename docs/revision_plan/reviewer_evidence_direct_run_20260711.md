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

## Canonical Drive Layout

Use exactly one operational source for revision artifacts:

```text
/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision
```

The related paths have different roles:

- `/content/ECG-RAMBA`: ephemeral active source checkout and runtime outputs.
- `ECG-Ramba/revision_artifacts/reports/revision`: canonical mirror for every
  artifact restore, reuse decision, and publish.
- `ECG-Ramba/model_runs/<run_id>`: immutable core Full ECG-RAMBA checkpoints;
  the current run pointer and OOF run manifest freeze their exact paths/SHA256.
- `ECG-Ramba/ECG-RAMBA`: legacy Drive checkout; source/archive only, never an
  automatic artifact restore source.
- `ECG-Ramba/final_evidence_tables`: manuscript convenience export, not a
  training or evaluation cache. Notebook 07 recreates it and writes
  `final_evidence_export_manifest.json`; no notebook restores inputs from it.
- `revision_feature_cache`, `revision_external_cache`, and
  `revision_pca_models`: large keyed feature caches with separate contracts.

Within the canonical mirror, resumable fold caches live in
`predictions/folds/`, learned-comparator checkpoints live in
`experimental/*_checkpoints/`, external source-bound representation folds live
in `predictions/external_representation_folds/`, group-safe adaptation caches
live in `predictions/fewshot_head_adaptation_cache/` and
`metrics/*fewshot*metric_cache/`, and live command traces live in `logs/`.
Baseline and representation runners write fold state directly to these Drive
paths using atomic file replacement. The `/content/ECG-RAMBA` copies of final
metrics/tables are disposable staging outputs.

A partially downloaded Windows folder is not an authority on cloud presence.
For example, a local sync may contain only `revision_artifacts` and
`final_evidence_tables` while omitting `model_runs`, `revision_pca_models`, or
multi-GB checkpoint directories. Use the Colab storage audit against the
mounted canonical root before concluding that a cache/checkpoint is absent.
The local audit maps manifest paths beginning with
`/content/drive/MyDrive/ECG-Ramba/` into the local Drive root when those tiers
have actually been synchronized.

Notebook 00 runs `37_artifact_source_audit.py`. Leave
`MIGRATE_LEGACY_ONLY_ARTIFACTS=False` during normal runs. Set it to `True` once
only after reviewing `table_artifact_source_audit.csv`; migration imports
legacy-only paths and never overwrites a conflicting canonical artifact.

Notebook 00 then runs `38_pipeline_storage_audit.py` in non-strict mode. Its
JSON/CSV outputs check each expected fold and each of the six named stresses,
including manifest coverage, rather than relying on aggregate file counts. It
also checks the exact PTB-XL fold-9/fold-10 representation slots and expected
group-safe calibration/true-head cache grid.
Before final packaging, run the same command with `--strict --full-sha`; this
is intentionally deferred because hashing all learned checkpoints on every
Colab bootstrap is expensive.

## Evidence Boundaries

- PTB-XL and Georgia are separate mapped, record-level external tasks. Never
  pool their metrics.
- CPSC2021 is a separate annotation-aligned 10-second AF/AFL mapped-window
  task. Do not pool it with PTB-XL or Georgia, and do not call it the official
  episode-boundary challenge score.
- The group-safe score-calibration runner changes frozen output scores only;
  it does not change model weights.
- The true few-shot runner fits new fold-specific linear heads on train-only
  standardized, mean-pooled record embeddings from frozen encoders. Fractions
  are nested fractions of independent target groups, not k examples per class.
  The pre-specified 10% budget is the primary endpoint; 1% and 5% are
  sensitivity points, not test-selected alternatives. It is not end-to-end
  fine-tuning. `table_true_fewshot_head_ptbxl_primary.csv` uses one shared
  patient-group bootstrap grid across all five adaptation seeds and matched
  models, so the reported primary and paired CIs are directly comparable.
- The frozen-transform MLP remains a head-capacity sensitivity control. The
  separate controlled morphology-learnability experiment compares an
  identically initialized reduced random-convolution bank with 0% versus 25%
  trainable kernel channels. It isolates kernel learnability within that
  reduced control, but it is not the evaluated 10,000-kernel branch and does
  not prove a causal mechanism for Full ECG-RAMBA.
- Representation UMAP/probe/CKA is an audit. It cannot be phrased as proven
  morphology-rhythm disentanglement.

## Runtime Selection

| Work | Hardware | What `auto` does |
| --- | --- | --- |
| Notebook 02 canonical OOF or missing external Full-model prediction | A100 High-RAM preferred | Reuses frozen OOF/external artifacts when manifest checks pass; otherwise runs inference. |
| Notebook 03 calibration and presentation assets | CPU | Reuses matching CI/table/figure outputs. Presentation assets defer until current paired artifacts exist. |
| Notebook 04 Raw Mamba, Transformer, frozen-transform MLP, controlled kernel-learnability bank | A100 High-RAM | Trains only missing or stale folds; checkpoints and fold prediction caches are mirrored after each fold. |
| Notebook 04 paired comparisons and ledger | CPU | Bootstrap is CPU-bound; no model inference. |
| Notebook 05 comparator stress prediction | A100 High-RAM | Generates or reuses stress prediction files per comparator, stress, and fold. |
| Notebook 05 multi-comparator bootstrap ledger | CPU | Reuses metric cache entries keyed by prediction provenance. |
| Notebook 06 external pooling / probes | CPU for pooling and probe; GPU + Mamba only for missing embeddings | Reuses v3 probe/figure only when its manifest protocol matches. |
| Notebook 07 final matrix and claim gates | CPU | Fails on stale required OOF/freeze/paired contracts rather than reusing them. |

## Hypothesis-Testing Extension (2026-07-18)

The P0--P2 extension is implemented as four fail-closed runners. It does not
replace the frozen source-of-truth package until their artifacts are regenerated
and Notebook 07 passes in strict mode.

| Question | Notebook / runner | Durable outputs | Claim boundary |
| --- | --- | --- | --- |
| Does post-hoc calibration change operating-point behavior under matched records and folds? | Notebook 03 / `42_matched_oof_calibration.py` | raw and cross-fitted monotone-Platt table, coefficients, paired bootstrap JSON, pooled reliability figure, manifest | Positive-slope mappings cannot reverse within-fold score order. This remains a score-level sensitivity only: calibrators and base models are not refitted inside bootstrap, and it is not a nested deployment estimate or clinical threshold validation. |
| Do the explicit morphology, rhythm, and context/fusion interfaces contribute inside the architecture? | Notebook 04 / `43_structured_ablation_5fold.py` | 20 fold checkpoints, four OOF prediction sets, paired table, TeX table, checkpoint/PCA/initialization audit, manifest | Fresh matched Full is compared with removal of the fixed-transform morphology stream and its dependent cross-attention interaction, removal of the checkpoint-compatible five-RR-plus-six-global-statistics conditioning interface, and joint removal of the context/fusion stack. Raw ECG remains in the first two controls, and Raw Mamba is a separate architecture control. Any conclusion remains endpoint-specific and internal to this training protocol; the no-morphology control does not isolate the fixed transform from its fusion interaction. |
| Does target-label adaptation form a reproducible learning curve? | Notebook 02 / `35_true_fewshot_head_adaptation.py` | 0/1/5/10% tables and figure for Full, ResNet, Raw Mamba, and Transformer; group-bootstrap caches and manifest | Frozen-encoder linear-head adaptation, not end-to-end fine-tuning. The 10% budget is primary; encoder/head training uncertainty is not resampled by the reported conditional bootstrap. |
| Do branch embeddings predict measured physiological intervals selectively? | Notebook 06 / `44_physiological_interval_probe.py` | target audit, fold-held-out probe CSV and manuscript TeX table, branch contrasts, summary and manifest | Runs only with reviewed, record-aligned HR/PR/QRS/QT/QTc measurements independent of model outputs. Missing metadata produces a blocker, removes stale TeX output, and generates no proxy targets. |

Notebook 07 then runs `45_hypothesis_control_claim_boundary.py --strict` and
requires exactly seven rows: morphology, rhythm, context/fusion, calibration,
robustness, external transfer, and adaptation. The generated
`table_hypothesis_control_finding_claim_boundary.tex` is the compact
Hypothesis--Control--Finding--Claim-boundary table used by the manuscript.

### P1 structured-ablation resume contract

- Use A100 High-RAM. Leave the notebook controls in `auto` mode.
- Every invocation selects the first fold still missing across the four matched
  variants. Completed folds are authenticated and skipped.
- A scalar seed alone is not accepted as a matched initialization. For each
  fold, the trainer first creates one fold-seeded Full reference and copies the
  exact overlapping state into every removal variant. Checkpoints record
  per-module-group initialization hashes; the gate invalidates the whole fold
  family if any retained group differs from Full. A dedicated DataLoader
  generator also fixes the minibatch order independently of constructor RNG use.
- Checkpoints are written directly to
  `revision_artifacts/reports/revision/experimental/structured_ablation_checkpoints/<variant>/`.
- Fold PCA files are written directly to
  `revision_artifacts/reports/revision/experimental/structured_ablation_pca_models/`.
  All four variants for one fold must declare the same PCA SHA, training-index
  hash, output dimension, and explained-variance value.
- OOF fold caches are written to
  `revision_artifacts/reports/revision/predictions/structured_ablation_folds/<variant>/`.
- The fixed training batch size, epoch budget, loss schedule, EMA rule, folds,
  threshold, and Q=3 aggregation remain inherited from the canonical training
  configuration. `STRUCTURED_ABLATION_OOF_BATCH_SIZE` controls export only.
- Disconnect only after the cell reports a successful canonical mirror publish.
  A disconnect during a fold requires that fold to restart; already completed
  folds remain reusable.
- Checkpoints created before protocol
  `matched_retrained_structured_ablation_5fold_v3` lack the common-state hash
  contract and are intentionally retrained once. This prevents an apparently
  matched comparison whose retained layers started from different weights.

Use a GPU runtime only while a cell reports that inference or training is
needed. Once all model-level artifacts exist, the paired, bootstrap, final
matrix, copy, and publication cells can run on CPU.

## Recommended Sequence

1. Run Notebook 00 and Notebook 01 once in a fresh Colab runtime. They mount
   Drive, pull the current repository, and verify the base protocol.
2. Run Notebook 02 through the canonical OOF, external Full-model export, and
   external protocol-gate cells. This can run before Notebook 04. It will also
   export PTB-XL fold 9 when needed for group-safe adaptation.
3. Run Notebook 03 through the canonical calibration CI. The **Matched
   Cross-Fitted Calibration Audit** and reviewer-presentation cell may print
   `Deferred` at this point; that is expected until Notebook 04 has published
   the required learned-baseline OOF artifacts. A missing optional
   frozen-transform MLP does not block the five-model matched audit. A present
   artifact with an invalid checksum still fails immediately.
4. Run Notebook 04. Leave the heavy runner controls as `auto`. For any missing
   Raw Mamba, Transformer, frozen-transform MLP, or controlled
   frozen-versus-partially-learnable morphology folds, the notebook runs one
   fold at a time and publishes the Drive mirror immediately. Then run every
   paired-comparison cell, including the controlled-kernel paired bootstrap,
   and the completion ledger. Then run **Matched Five-Fold Structured Ablation
   Runner** repeatedly on A100 until all four variants have five authenticated
   folds. Its final invocation exports OOF predictions and computes the paired
   record-bootstrap table.
5. Return to the new end-of-Notebook-02 cells. Their first pass after Notebook
   04 produces or verifies zero-target-label PTB-XL/Georgia ResNet1D/CNN and
   Raw Mamba outputs. A Transformer is added only if its in-domain OOF and
   paired gate passed. The remaining cells then run external paired bootstrap,
   PTB-XL group-safe score calibration, optional frozen-encoder representations,
   and true linear-head adaptation. Before Notebook 04 these cells defer
   instead of failing.
6. Return to Notebook 03 and run **Matched Cross-Fitted Calibration Audit**,
   followed by the reviewer-presentation cell. The calibration cell is CPU-only,
   uses positive-slope monotone Platt mappings, and reuses bootstrap entries only
   when probability hashes, protocol v3, and the frozen OOF contract match. The
   presentation cell creates the reliability figure, compact calibration/paired-CI
   tables, Q=3 sensitivity table, fold-specific PCA variance table, and
   training-configuration table.
7. Run Notebook 05 with `ROBUSTNESS_MULTI_RUN_PROFILE='canonical_resume'`.
   The six stresses are processed separately and publish after each completed
   stage; the shared metric cache resumes interrupted 1,000-bootstrap work.
   Notebook 07 rejects reviewer-minimal screening output as final R2-C3
   evidence.
8. Run Notebook 06. External pooling keeps PTB-XL, Georgia, and CPSC2021
   separate and requires six methods plus 1,000 group-bootstrap replicates.
   Run
   representation extraction only if missing; it needs the same Mamba runtime
   as Full ECG-RAMBA. The v3 probe runs afterwards and produces the audit
   figure plus fold-level probe table. The measured physiological interval probe
   is optional and CPU-only after embeddings exist; a blocker is the correct
   outcome when reviewed measurements are unavailable.

### Measured physiological-interval metadata gate

Notebook 06 searches for
`/content/drive/MyDrive/ECG-Ramba/physiological_interval_metadata.csv` and then
`/content/drive/MyDrive/ECG-Ramba/metadata/physiological_interval_metadata.csv`.
The table must contain one unique `record_id` row and at least one genuinely
measured target among `heart_rate_bpm`, `pr_ms`, `qrs_ms`, `qt_ms`, and
`qtc_ms`. Do not derive these columns from ECG-RAMBA predictions or from the
branch embeddings being probed.

Place a reviewed sidecar next to the CSV as
`physiological_interval_metadata.csv.provenance.json`, or pass its path
explicitly. Start from the checked-in templates
`docs/revision_plan/physiological_interval_metadata_template.csv` and
`docs/revision_plan/physiological_interval_metadata_provenance_template.json`.
The template deliberately fails closed. Compute the exact SHA256 of the CSV and
set `metadata_sha256` to that value; record `reviewed_by`, a timezone-aware
`reviewed_utc`, and the original `source_description`. Change `status` to
`reviewed` and set both `independent_of_model_outputs=true` and
`independent_of_ecg_ramba_feature_cache=true` only after verifying that the
targets were not copied from model outputs, branch representations, or the
audited feature cache. Replace every target's source column, unit, and
measurement kind only after this source audit. Accepted measurement kinds are
`measured`, `device_measured`, and `expert_annotated`. Protocol v3 also requires
at least two measured target values in every evaluated fold and all requested
pointwise bootstrap replicates to be finite before it emits a complete status.
9. Run Notebook 07. It first runs claim-readiness gates and the strict
   Hypothesis--Control--Finding--Claim-boundary ledger, then writes final
   evidence tables only when calibration and paired OOF contracts match the
   active frozen OOF. It mirrors all artifacts and copies the current source
   tables to the output-only `final_evidence_tables` snapshot with an export
   checksum manifest.
10. Build the marked manuscript only in an environment with both `latexdiff`
    and `latexmk`/LaTeX. A `blocked_missing_tool` manifest is deliberate and
    must not be called a marked PDF.

## Direct Colab Cell Checklist

Each active notebook has a self-contained **Setup** section. After a Colab
disconnect, open the notebook that was interrupted and rerun from its Setup
section; Notebook 00 does not need to be rerun first. The exception is a
session-local Mamba installation: a fresh runtime must run the Mamba installer
in Notebook 00, 02, 05, or 06 before the first Full/Raw-Mamba inference cell.

Use the checked-in defaults unless this table explicitly says otherwise:

| Notebook | Run sections | Runtime and controls | Skip/defer rule |
| --- | --- | --- | --- |
| 00 | All sections, top to bottom | CPU is enough for storage/audit; GPU only if immediately continuing to Mamba inference. Keep migration disabled and storage audit non-strict. | Do not use legacy migration unless the source-audit table was reviewed and conflict-free. |
| 01 | All sections | CPU. | None. This is a protocol audit, not model training. |
| 02 first pass | Setup through **External Protocol Gate**, then **PTB-XL Fold 9 Adaptation-Pool Export** | A100 High-RAM when an OOF/external export is missing. Keep OOF/external force flags false and reuse enabled. | The learned-comparator, representation, and true-head sections may defer before Notebook 04; this is expected. |
| 03 first pass | Setup through canonical calibration CI | CPU High-RAM. | Matched calibration and presentation assets defer until Notebook 04 OOF/paired artifacts exist. |
| 04 | All sections | A100 High-RAM through **Controlled Frozen-vs-Partially-Learnable Morphology Bank**; CPU is enough for paired comparisons and ledgers. Keep runner mode `auto`, force-rerun false, checkpoint saving/reuse true. | Skip Notebook 02a unless the frozen Full final-EMA checkpoint contract itself is invalid and a paper-level retrain was explicitly chosen. |
| 02 second pass | **External Learned-Comparator Zero-Target-Label Inference** through **True Few-Shot Frozen-Encoder Head Adaptation**, then inventory/mirror | A100 for external comparator inference and source-bound representation extraction. CPU High-RAM for paired bootstrap, group-safe score calibration, and linear heads. Keep `TRUE_FEWSHOT_PRIMARY_FRACTION=0.10`, seeds 42--46, fractions 0/1/5/10%, reuse true. | Old representation protocol-v1 caches are intentionally rejected once; protocol-v2 fold caches resume thereafter. |
| 03 second pass | **Matched Cross-Fitted Calibration Audit**, then **Build Reviewer Presentation Assets**, then mirror | CPU High-RAM. Keep 1,000 bootstraps and reuse enabled. | None after the five required baseline OOF artifacts are current; frozen-transform MLP is included when authenticated but is not a blocker. |
| 05 GPU pass | Setup through **Comparator Stress Prediction Generation** | A100 High-RAM, batch 512; lower to 256 only on OOM. Canonical mode requires all six stresses for ResNet, Raw Mamba, and Transformer. Each stress is published immediately. | Do not rerun completed stress files; the exact checkpoint/prediction contract is checked before reuse. |
| 05 CPU pass | **Multi-Comparator Robustness Ledger** through mirror/output | CPU High-RAM. Keep `canonical_resume`, `n_boot=1000`, five metrics, and eight bootstrap workers unless RAM pressure requires fewer workers. | Do not present reviewer-minimal output as the canonical six-stress ledger. |
| 06 | All sections | CPU for pooling/probe; A100 plus Mamba only if representation embeddings are missing/stale. | Existing compatible embeddings/probe artifacts are reused; final pooling requires PTB-XL, Georgia, and CPSC2021. |
| 07 | All sections | CPU High-RAM. | Run only after the preceding mirror publishes. Any strict failure means the source-of-truth tables must not be used yet. |

For Notebook 02, the direct-run heavy controls should remain:

```python
RUN_EXTERNAL_LEARNED_COMPARATORS = 'auto'
RUN_PTBXL_FOLD9_COMPARATORS = 'auto'
RUN_GROUP_SAFE_SCORE_CALIBRATION = 'auto'
GROUP_SAFE_CALIBRATION_PRIMARY_FRACTION = 0.10
RUN_EXTERNAL_REPRESENTATION_EXTRACTION = 'auto'
RUN_TRUE_FEWSHOT_HEAD_ADAPTATION = 'auto'
TRUE_FEWSHOT_PRIMARY_FRACTION = 0.10
TRUE_FEWSHOT_N_BOOT = 1000
```

For Notebook 05 final reviewer evidence, keep:

```python
ROBUSTNESS_MULTI_RUN_PROFILE = 'canonical_resume'
ROBUSTNESS_MULTI_N_BOOT = 1000
ROBUSTNESS_MULTI_BOOTSTRAP_JOBS = 8
```

The recommended disconnect boundary is immediately after a cell prints a
successful canonical mirror publish. Do not disconnect while a checkpoint,
stress prediction, representation fold, or atomic NPZ is still being written.

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
- A missing fold cache does not imply retraining when a compatible checkpoint
  exists. ResNet, Raw Mamba, Transformer, and the frozen-transform MLP first
  validate checkpoint protocol, fold, frozen OOF/freeze provenance, raw or
  feature cache provenance, and training parameters, then regenerate only the
  validation predictions.
- The external learned-comparator runner reads checkpoints directly from
  `revision_artifacts/reports/revision/experimental/` on Drive. Its final
  manifest includes frozen OOF/freeze SHA256, checkpoint SHA256, archive SHA256,
  mapping SHA256 where relevant, and runner SHA256. Old manifests are
  regenerated from compatible fold caches rather than accepted blindly.
- Full external inference authenticates all five files against
  `oof_final_ema_prediction_run_manifest.json` before `torch.load`. Learned
  comparators authenticate all five files against their baseline checkpoint
  contracts. External representation protocol v2 additionally binds every
  fold cache to the source prediction manifest, current archive/loader
  provenance, canonical OOF contract, and exact checkpoint SHA256. True
  few-shot adaptation requires that v2 manifest and NPZ pair, so a stale or
  cross-archive representation cannot enter adaptation.
- Metric caches use input/protocol-derived cache keys. A changed prediction,
  group split, threshold, bootstrap count, or seed produces a new cache entry.
- Notebook 02 `auto` cells re-enter the lightweight paired/calibration/head
  runners so they can validate canonical OOF/freeze and runner SHA contracts;
  compatible metric/head caches are reused and model inference is skipped.
- Every heavy cell streams output and dual-writes its command log to local
  `reports/revision/logs/` and canonical Drive `logs/`. Logs are deliberately
  excluded from the immutable artifact manifest so rerunning a command can
  safely update the same trace path.
- Mirror publication is merge-only and checksum-verified: a partial Colab
  runtime updates artifacts it owns but preserves previously verified fold
  checkpoints/caches in the manifest. It does not prune artifacts that are
  absent from the current runtime.
- A file that merely exists below the mirror path is not evidence until it has
  a mirror-manifest row. Notebook restores reject unmanifested active files,
  and mirror discovery ignores interrupted atomic-write names such as
  `.partial`, `.tmp`, and `.lock`. Run the publish cell after a completed fold
  or stress so its final file becomes checksum-addressable.
- Intermediate publishes use `--verify-existing size` to avoid repeatedly
  hashing multi-GB preserved checkpoints; every new or overwritten artifact is
  still SHA256-verified. Notebook 07 uses the default full verification pass
  before final evidence export.

## Required Completion Artifacts

The following outputs prove a completed item; file presence alone is not
sufficient because Notebook 07 and claim gates also inspect manifests and
checksums.

| Reviewer item | Primary artifacts |
| --- | --- |
| R1-C1/R1-C5 reliability and paired CI presentation | `figure_calibration_audit.png`, `table_calibration_ci_compact.csv`, `table_paired_baseline_ci_compact.csv`, `reviewer_completion_input_contract.json` |
| R1-C2 determinism/regularization sensitivity | Existing Hybrid MLP package plus `morphology_learnability_summary.json`, frozen/partial OOF predictions, ten controlled checkpoints, `table_paired_morphology_learnability.csv`, and its paired manifest |
| R1-C4 Transformer comparator | `transformer_ecg_baseline_summary.json`, paired Transformer comparison, five checkpoint files |
| R1-C5 external zero-target-label uncertainty | Exact 3-dataset x 3-comparator x 5-metric group-paired grid, `table_external_zero_target_ci_compact.csv`, and authenticated manifest |
| R1-C6 pooling sensitivity | Chapman sensitivity plus exact PTB-XL/Georgia/CPSC2021 x six-method table, 75 Q=3 paired group-bootstrap items, and `table_pooling_cross_dataset_compact.csv` |
| R1-C7 PCA/config appendix | `table_fold_pca_provenance.csv`, `table_training_configuration.csv`, morphology transform contract |
| R2-C2 representation audit | v3 `representation_probe_manifest.json`, fold probe table, CKA table, audit figure |
| R2-C3 robustness | Canonical six-stress x four-comparator x five-metric paired-degradation ledger, 1,000 valid bootstrap replicates per row, and `table_robustness_six_stress_compact.csv` |
| R2-C4 adaptation | group-safe score-calibration manifest and true frozen-encoder linear-head manifest, both PTB-XL fold 9/10 only |
| Editorial marked manuscript | `marked_manuscript_manifest.json` with `editorial_ready=true` and a nonempty marked PDF |

Notebook 07 runs `41_reviewer_gap_closure.py --strict`. All four rows
(`R1-C2`, `R1-C5`, `R1-C6`, and `R2-C3`) must be `complete` and
`manuscript_ready=true` before the compact tables are promoted into the final
evidence package.

## Final Safe Wording Gate

The only source of truth for manuscript/rebuttal wording is the regenerated
pair:

- `reports/revision/tables/table_final_evidence_matrix.csv`
- `reports/revision/tables/table_final_safe_wording.csv`

The convenience copies in Drive are for handoff only. Before submission,
rerun Notebook 07, then run:

```bash
python -u scripts/revision/38_pipeline_storage_audit.py \
  --canonical-root "/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision" \
  --strict --full-sha
```

Compile the manuscript, create fresh PDF text, and run the forbidden-claim
scan. Do not claim SOTA, broad in-domain superiority,
zero-shot superiority, general external superiority, end-to-end few-shot
fine-tuning, proven disentanglement, general robustness superiority, complete
HRV, or clinical readiness unless a future protocol-specific artifact makes
that wording valid.
