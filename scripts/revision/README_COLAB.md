# ECG-RAMBA Revision Pipeline on Colab

This folder contains lightweight scripts for the JBHI resubmission workflow.
Keep `notebooks/01_exploratory.ipynb` as an archive. Use the numbered
revision notebooks as the main Colab workflow.

## Recommended Drive Layout

```text
/content/drive/MyDrive/ECG-Ramba/
  WFDB-ChapmanShaoxing.zip
  PTB-XL.zip
  Georgia.zip
  cpsc2021.zip
  model/
    fold1_best.pt
    ...
    fold5_best.pt
    ptbxl_hrv36.npz              # optional, enables HRV domain classifier
    cpsc2021_hrv36.npz           # optional, enables HRV domain classifier
  ECG-RAMBA/
    configs/
    src/
    scripts/
    notebooks/
    reports/
```

The current `configs/config.py` detects `/content/drive/MyDrive/ECG-Ramba`
or the `ECG_RAMBA_DRIVE_ROOT` environment variable as the cache root when
running in Colab. It also detects the uploaded dataset names
`WFDB-ChapmanShaoxing.zip`, `PTB-XL.zip`, `Georgia.zip`, and `cpsc2021.zip`.
The repo may also
live directly in `MyDrive/ECG-Ramba` if that folder contains `configs/`,
`src/`, `scripts/`, and `notebooks/`.

## Upload Options

Option A - GitHub, preferred:

```bash
cd /content/drive/MyDrive/ECG-Ramba
git clone -b main https://github.com/BrianNguyen29/ECG-RAMBA.git
cd ECG-RAMBA
git pull --ff-only
```

For later updates in the same Colab/Drive folder:

```bash
cd /content/drive/MyDrive/ECG-Ramba/ECG-RAMBA
git fetch origin
git pull --ff-only
```

Keep large artifacts outside Git. Checkpoints can stay at
`MyDrive/ECG-Ramba/model`; `configs/config.py` detects that Drive-root folder
without copying it into the cloned repo.

Option B - Manual Drive upload:

1. Zip the local `ECG-RAMBA` folder.
2. Upload it to `MyDrive/ECG-Ramba`.
3. Extract it in Colab to `/content/drive/MyDrive/ECG-Ramba/ECG-RAMBA`.

Do not use the older patch-zip workflow for new runs. It is easy to extract
files into the wrong folder and bypass the Git/manifest checks. If GitHub is
temporarily unavailable, upload the whole `ECG-RAMBA` repo folder and start
from `notebooks/00_colab_bootstrap.ipynb`.

## First Colab Run

Run the bootstrap notebook:

```text
notebooks/00_colab_bootstrap.ipynb
```

Start with:

```bash
python scripts/revision/00_audit_protocol.py
```

Do not launch full training/evaluation until the audit warnings are resolved.

## Notebook Workflow

```text
notebooks/00_colab_bootstrap.ipynb
notebooks/01_a0_protocol_audit.ipynb
notebooks/02a_retrain_best_ema.ipynb
notebooks/02_predictions_and_external_eval.ipynb
notebooks/03_calibration_and_ci.ipynb
notebooks/06_pooling_and_representation.ipynb  # pooling sensitivity
notebooks/04_baselines_and_component_checks.ipynb
notebooks/05_hrv_domain_and_robustness.ipynb
notebooks/06_pooling_and_representation.ipynb  # representation follow-up
notebooks/07_results_freeze.ipynb
```

Colab package installations are scoped to the current runtime session. Do not
manually disconnect between notebooks unless you need a fresh session. Restart
only when a dependency cell explicitly requests it, then rerun the current
notebook from the first cell. Notebook 02a replays the canonical base and Mamba
installers when it detects a fresh runtime.

Planning metadata lives in:

```text
docs/revision_plan/
```

Final/generated experiment artifacts are written to:

```text
reports/revision/
```

## Minimum Artifact Contract

Revision scripts should write artifacts to:

```text
  reports/revision/
  audit_protocol.json
  hrv36_schema.csv
  predictions/oof_final_ema_predictions.npz
  predictions/oof_final_ema_slice_predictions.npz
  predictions/hrv_only_oof_predictions.npz
  manifests/oof_final_ema_freeze_manifest.json
  metrics/oof_final_ema_prediction_summary.json
  metrics/calibration_ci_oof_final_ema_predictions.json
  metrics/baseline_summary.csv
  metrics/hrv_domain_summary.csv
  metrics/robustness_summary.csv
  metrics/pooling_sensitivity.csv
  tables/table_calibration.csv
  tables/table_bootstrap_ci.csv
  tables/table_hrv_domain_status.csv
  tables/table_robustness.csv
  experimental/external/<dataset>/<dataset>_full_predictions.npz
  figures/
```

Prediction files used by `04_calibration_ci.py` must contain:

```text
y_true: shape (N, C)
y_prob: shape (N, C)
```

## Blockers to Resolve Before Manuscript Numbers

- HRV36 currently contains 5 RR summary slots, 20 reserved zero slots,
  5 intended amplitude slots, and 6 global signal-stat slots. The current
  Chapman checkpoints received zero amplitude slots because of a training
  feature-input mismatch. Do not describe it as
  full HRV with RMSSD/SDNN/LF-HF unless those features are implemented and
  the model is retrained.
- Georgia and CPSC2021 are distinct datasets. Never label `Georgia.zip`
  results as CPSC2021.
- Legacy OOF artifacts without `aggregation_implementation=power_mean_v2`
  and `cache_schema_version>=2` must be re-aggregated from complete saved
  slice probabilities or regenerated.
- OOF reuse additionally requires `oof_freeze_manifest.json` to verify all
  44,186 records, five folds, current checkpoint SHA256 fingerprints, and
  artifact checksums.
- External evaluation requires five fold-specific PCA objects generated by
  `08_build_fold_pca.py --checkpoint-kind final_ema`. The PCA manifest is
  rejected unless its source config hash matches the selected checkpoints.
  External files remain under
  `reports/revision/experimental/` with `manuscript_ready=false`.
- Use the PTB superclass mapping in `scripts.revision.common.PTB_SUPERCLASS_MAPPING`.
- Treat inference ablation as diagnostic. Reviewer-facing component claims need
  retrained or protocol-fair baselines where possible.
- Add calibration, bootstrap CI, learned CNN/ResNet baseline, HRV domain check,
  and minimal robustness tests before final response-letter claims.
