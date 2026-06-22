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
    minirocket_only_oof_predictions.npz
    resnet1d_cnn_oof_predictions.npz
    resnet1d_cnn_slice_predictions.npz
    baseline_<name>_<dataset>_predictions.npz
  metrics/
    oof_final_ema_prediction_summary.json
    calibration_ci_oof_final_ema_predictions.json
    oof_full_prediction_summary.json
    calibration_ci_oof_full_predictions.json
    baseline_summary.csv
    hrv_only_baseline_summary.json
    minirocket_only_baseline_summary.json
    resnet1d_cnn_baseline_summary.json
    paired_full_vs_minirocket_comparison.json
    paired_full_vs_resnet_comparison.json
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
    table_minirocket_only_class_metrics.csv
    table_minirocket_only_fold_summary.csv
    table_resnet1d_cnn_class_metrics.csv
    table_resnet1d_cnn_fold_summary.csv
    table_paired_full_vs_minirocket.csv
    table_paired_full_vs_resnet.csv
    table_hrv_domain_status.csv
    table_hrv_only_class_metrics.csv
    reliability_bins_<dataset>.csv
    table_robustness.csv
  logs/
    <notebook_or_script>_<timestamp>.log
  manifests/
    oof_final_ema_prediction_run_manifest.json
    oof_final_ema_freeze_manifest.json
    minirocket_only_baseline_manifest.json
    resnet1d_cnn_baseline_manifest.json
    paired_full_vs_minirocket_manifest.json
    paired_full_vs_resnet_manifest.json
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
remains separate. Robustness stress-test claims are manuscript-usable only after
`scripts/revision/12_robustness_stress.py` produces
`metrics/robustness_summary.csv`,
`metrics/robustness_full_vs_minirocket_comparison.json`, and
`manifests/robustness_stress_manifest.json`; a status ledger alone is not
robustness evidence.

MiniRocket-only baseline artifacts are manuscript-usable only when
`metrics/minirocket_only_baseline_summary.json` reports
`protocol=minirocket_raw_standardized_torch_linear_same_folds_threshold_0.5`,
`feature_preprocessing=fold_train_standardization`, `manuscript_ready=true`,
44,186 records, 27 classes, the canonical `oof_final_ema` prediction SHA256,
and a record-fingerprinted RAW MiniRocket cache. The classifier is a fixed-epoch
linear logistic head trained only on each fold's training records; it is a
feature baseline, not ECG-RAMBA checkpoint inference.

ResNet1D/CNN baseline artifacts are manuscript-usable only when
`metrics/resnet1d_cnn_baseline_summary.json` reports
`protocol=resnet1d_cnn_raw_same_folds_power_mean_v2_q3_threshold_0.5`,
`feature_contract=raw_ecg_12lead`, `manuscript_ready=true`, 44,186 records,
27 classes, the canonical `oof_final_ema` prediction SHA256, and the frozen
OOF fold contract. Its paired comparison against Full ECG-RAMBA is
manuscript-usable only after `metrics/paired_full_vs_resnet_comparison.json`
and `tables/table_paired_full_vs_resnet.csv` validate identical `y_true`,
`record_id`, `fold_id`, and `class_names`. Current final evidence shows
ResNet1D/CNN is significantly stronger than Full ECG-RAMBA on the frozen
Chapman OOF PR-AUC, ROC-AUC, F1, Brier, and ECE metrics; do not use
ECG-RAMBA in-domain fair-baseline superiority wording.

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
