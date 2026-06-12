# Revision Artifact Contract

All reviewer-facing numbers must be traceable to files under
`reports/revision/`. Large outputs remain ignored by Git and should be kept in
Google Drive.

## Directory Layout

```text
reports/revision/
  audit_protocol.json
  hrv36_schema.csv
  predictions/
    oof_final_ema_predictions.npz
    oof_final_ema_slice_predictions.npz
    oof_best_ema_predictions.npz          # diagnostic only
    oof_best_ema_slice_predictions.npz    # diagnostic only
    oof_full_predictions.npz
    oof_full_slice_predictions.npz
    hrv_only_oof_predictions.npz
    baseline_<name>_<dataset>_predictions.npz
  metrics/
    oof_final_ema_prediction_summary.json
    calibration_ci_oof_final_ema_predictions.json
    oof_full_prediction_summary.json
    calibration_ci_oof_full_predictions.json
    baseline_summary.csv
    hrv_only_baseline_summary.json
    hrv_domain_classifier_summary.json
    hrv_domain_summary.csv
    robustness_summary.csv
    pooling_sensitivity.csv
    pooling_sensitivity.json
  figures/
    reliability_<dataset>.png
    robustness_<dataset>.png
    representation_umap.png
  tables/
    oof_final_ema_class_summary.csv
    oof_full_class_summary.csv
    table_baselines.csv
    table_calibration.csv
    table_bootstrap_ci.csv
    table_hrv_domain_status.csv
    table_hrv_only_class_metrics.csv
    reliability_bins_<dataset>.csv
    table_robustness.csv
  logs/
    <notebook_or_script>_<timestamp>.log
  manifests/
    oof_final_ema_prediction_run_manifest.json
    oof_final_ema_freeze_manifest.json
    oof_full_prediction_run_manifest.json
    oof_freeze_manifest.json
    mirror_manifest.json
    artifacts_manifest.json
    artifacts_manifest.csv
  experimental/
    external/
      external_summary_experimental.csv
      <dataset>/
        <dataset>_full_predictions.npz
        <dataset>_full_slice_predictions.npz
        <dataset>_full_prediction_summary.json
```

## Prediction NPZ Contract

Every prediction file used for metrics must contain:

```text
y_true: shape (N, C), binary labels
y_prob: shape (N, C), probabilities in [0, 1]
```

Optional fields:

```text
record_id: shape (N,)
class_names: shape (C,)
dataset: scalar/string
protocol: scalar/string
config_hash: scalar/string
source_config_hash: scalar/string
evaluation_config_hash: scalar/string
git_commit: scalar/string
created_utc: scalar/string
fold_id: shape (N,)
slice_count: shape (N,)
valid_record_mask: shape (N,)
checkpoint_kind: scalar/string
batch_size: scalar/integer
aggregation_method: scalar/string
aggregation_q: scalar/float
aggregation_implementation: scalar/string
cache_schema_version: scalar/integer
checkpoint_fingerprints_json: scalar/JSON string
slice_prob: shape (total_slices, C), paired with record_id and slice_index
slice_index: shape (total_slices,)
```

Legacy OOF artifacts without `aggregation_implementation=power_mean_v2` and
`cache_schema_version>=2` are invalid. Georgia and CPSC2021 are distinct
datasets and must never share a dataset label.

After the checkpoint/EMA blocker is resolved, the canonical manuscript OOF
artifact is `oof_final_ema_predictions.npz` with
`oof_final_ema_freeze_manifest.json`. It uses EMA weights at the
pre-specified final epoch. It is reusable only when the freeze
manifest confirms 44,186 records, all five folds, Q=3 re-aggregation
equivalence, matching `fold*_final_ema.pt` SHA256 fingerprints, and checksums for
the prediction, slice, summary, class table, run manifest, and OOF logs.
Checkpoint, feature-cache, and prediction metadata must also carry the same
ordered Chapman record-ID fingerprint; shape-only cache matching is
insufficient.

Validation-selected `oof_best_ema_*` artifacts are diagnostic because the
selection fold is also represented in OOF. Historical `oof_full_*` and raw
`oof_final_*` artifacts produced from raw
`fold*_best.pt` or `fold*_final.pt` checkpoints are diagnostic only unless their
checkpoint metadata proves they are explicit EMA aliases from a post-fix
retrain.

External outputs are stored under `reports/revision/experimental/` with
`manuscript_ready=false` until dataset-specific protocol review, fold-specific
PCA provenance, fair baselines, and uncertainty analysis are complete.

HRV-domain artifacts are manuscript-usable only for the analyses marked
`complete` in `metrics/hrv_domain_summary.csv`. Duration/noise HRV sensitivity
and robustness stress tests remain blocked until dedicated perturbation runners
produce metric artifacts rather than status ledgers.

## Minimum Metric Contract

Each dataset/model result should report:

- Macro ROC-AUC
- Macro PR-AUC
- Macro F1 at threshold 0.5
- Sensitivity, specificity, PPV, NPV at threshold 0.5 where applicable
- ECE, MCE, and Brier score for calibration
- Record-level bootstrap 95% CI for ROC-AUC, PR-AUC, and F1

## Freeze Rule

Before using a number in the manuscript or rebuttal, run:

```bash
python scripts/revision/05_artifact_inventory.py
```

Then record the manifest path and Git commit in the response letter working
notes.
