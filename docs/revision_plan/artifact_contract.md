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
    cpsc_full_predictions.npz
    baseline_<name>_<dataset>_predictions.npz
  metrics/
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
    table_baselines.csv
    table_calibration.csv
    table_bootstrap_ci.csv
    table_robustness.csv
  logs/
    <notebook_or_script>_<timestamp>.log
  manifests/
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
slice_prob: shape (N, S, C), if pooling sensitivity is needed
```

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
