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
    oof_full_predictions.npz
    ptbxl_full_predictions.npz
    georgia_full_predictions.npz
    cpsc2021_full_predictions.npz
    baseline_<name>_<dataset>_predictions.npz
  metrics/
    oof_full_prediction_summary.json
    calibration_ci_<dataset>.json
    baseline_summary.csv
    hrv_domain_summary.csv
    robustness_summary.csv
    pooling_sensitivity.csv
  figures/
    reliability_<dataset>.png
    robustness_<dataset>.png
    representation_umap.png
  tables/
    oof_full_class_summary.csv
    table_baselines.csv
    table_calibration.csv
    table_bootstrap_ci.csv
    reliability_bins_<dataset>.csv
    table_robustness.csv
  logs/
    <notebook_or_script>_<timestamp>.log
  manifests/
    oof_full_prediction_run_manifest.json
    artifacts_manifest.json
    artifacts_manifest.csv
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
slice_prob: shape (total_slices, C), paired with record_id and slice_index
slice_index: shape (total_slices,)
```

Legacy OOF artifacts without `aggregation_implementation=power_mean_v2` and
`cache_schema_version>=2` are invalid. Georgia and CPSC2021 are distinct
datasets and must never share a dataset label.

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
