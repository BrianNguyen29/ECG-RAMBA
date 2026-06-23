# Baseline Training Pipeline Audit

Date: 2026-06-23

Scope: MiniRocket-only, ResNet1D/CNN, and Raw Mamba fair-comparator pipelines used in Notebook 04.

This document is a technical audit note, not a manuscript claim source. Manuscript/rebuttal wording must still follow:

```text
reports/revision/metrics/final_evidence_matrix.json
reports/revision/tables/table_final_safe_wording.csv
docs/revision_plan/manuscript_rebuttal_drive_source_update_20260622.md
```

## Audit Result

The source-level pipeline audit passes with warnings:

- no failing source-contract checks were found;
- MiniRocket-only and ResNet1D/CNN artifacts exist in the local Drive mirror;
- Raw Mamba runner is implemented but does not yet have a manuscript-ready completed artifact in the mirror;
- MiniRocket-only uses un-clipped fold-train `pos_weight` in the torch-linear head, so its strong PR-AUC/ROC-AUC should be interpreted with its poor calibration and high-recall operating behavior.

Machine-readable audit outputs:

```text
reports/revision/metrics/baseline_pipeline_audit.json
reports/revision/tables/table_baseline_pipeline_audit.csv
```

## MiniRocket-Only Pipeline

Status: technically acceptable as a feature baseline with explicit calibration limitations.

Validated properties:

- uses frozen `oof_final_ema_predictions.npz` for labels, `record_id`, `fold_id`, and `class_names`;
- validates the frozen OOF manifest and prediction SHA before training/reuse;
- requires a record-fingerprinted RAW MiniRocket feature cache by default;
- rejects legacy shape-only cache unless explicitly allowed;
- computes feature standardization on each training fold only;
- trains a fold-safe linear multilabel head and writes full provenance into the prediction NPZ and summary JSON;
- recomputes metrics from the prediction NPZ under the same threshold and bootstrap protocol.

Important limitation:

- the torch-linear MiniRocket head uses fold-train class imbalance weights without clipping. This is defensible as a strong feature baseline, but it explains the observed tradeoff: high PR-AUC/ROC-AUC, high recall, poor Brier/ECE. Do not describe it as well calibrated.

## ResNet1D/CNN Pipeline

Status: technically acceptable and manuscript-ready after completed artifact validation.

Validated properties:

- loads raw ECG cache only when labels and record-order fingerprint match the frozen OOF contract;
- trains from scratch on frozen folds;
- does not use MiniRocket, HRV, PCA, or ECG-RAMBA checkpoints;
- uses fold-train `pos_weight` with clipping for multilabel imbalance;
- uses fixed final epoch rather than best-validation epoch selection;
- uses the shared Power Mean Q=3 aggregation helper;
- writes record-level and slice-level prediction artifacts, class/fold tables, summary JSON, and manifest;
- reuse path now validates both record-level and slice-level artifacts, including protocol, model parameters, SHA, fold mapping, finite probabilities, and reconstructed `slice_count`.

Interpretation:

- ResNet1D/CNN is currently stronger than ECG-RAMBA on frozen Chapman OOF PR-AUC, ROC-AUC, F1, Brier, and ECE. This blocks any in-domain superiority or SOTA-style claim for ECG-RAMBA.

## Raw Mamba Pipeline

Status: implementation is ready for rerun; artifact completion is still pending.

Validated properties:

- instantiates `ECGRambaV7Advanced` with structural ablation:

```python
{"no_rocket": True, "no_hrv": True, "no_fusion": True}
```

- passes zero auxiliary tensors only to satisfy the model signature;
- uses raw ECG slices and the same frozen OOF folds;
- uses shared Power Mean Q=3 aggregation;
- uses EMA final weights for evaluation when configured;
- writes prediction diagnostics (`Pmean`, `Pstd`, `P>=thr`) so all-negative collapse is visible early;
- records the weighted-BCE protocol explicitly:

```text
raw_mamba_retrained_weighted_bce_same_folds_power_mean_v2_q3_threshold_0.5
```

Reason for the weighted-BCE change:

- the first Raw Mamba run with unweighted BCE collapsed to near-constant predictions (`F1=0`, `ROC=0.5`, PR near prevalence);
- the current runner uses fold-train `pos_weight` during BCE warm-up to make the comparator trainable under the same severe multilabel imbalance;
- this should be reported as a retrained fair-comparator protocol, not as the exact original full ECG-RAMBA training protocol.

Required next validation:

- rerun Notebook 04 Raw Mamba cell after pulling the commit with weighted BCE;
- confirm logs show `BCE pos_weight enabled`;
- by epoch 2-5, confirm `Pstd > 0`, `P>=thr > 0`, and ROC moves above 0.5;
- after completion, rerun Notebook 04 matrix and Notebook 07 final evidence.

## Aggregation And Metrics

Validated properties:

- all three baseline runners use the frozen OOF labels/folds;
- raw-signal baselines use shared `src/aggregation.py` Power Mean Q=3 for slice-to-record aggregation;
- metric computation uses shared revision helpers for multilabel metrics, Brier/ECE, and bootstrap CI;
- fixed threshold remains 0.5.

## Manuscript Safety

Safe after current audit:

- MiniRocket-only can be cited as a deterministic feature baseline with rank-vs-calibration tradeoff.
- ResNet1D/CNN can be cited as a strong in-domain raw ECG baseline that outperforms ECG-RAMBA on frozen Chapman OOF.
- Raw Mamba can be described only as implemented/pending until artifacts complete.

Unsafe:

- Do not claim ECG-RAMBA in-domain superiority.
- Do not claim ECG-RAMBA superiority over all fair baselines until Raw Mamba artifacts complete, and even then only if the result supports it.
- Do not claim MiniRocket calibration quality.
- Do not treat Raw Mamba weighted-BCE results as identical to the original full ECG-RAMBA training recipe.
