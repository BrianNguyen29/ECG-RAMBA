# ECG-RAMBA Revision Pipeline on Colab

This folder contains lightweight scripts for the JBHI resubmission workflow.
Keep `notebooks/01_exploratory.ipynb` as an archive. Use
`notebooks/00_revision_runner.ipynb` to run reproducible steps from Colab.

## Recommended Drive Layout

```text
/content/drive/MyDrive/ECG-Ramba/
  WFDB-ChapmanShaoxing.zip
  PTB-XL.zip
  Georgia.zip
  model/
    fold1_best.pt
    ...
    fold5_best.pt
    global_pca_zeroshot.pkl
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
`WFDB-ChapmanShaoxing.zip`, `PTB-XL.zip`, and `Georgia.zip`. The repo may also
live directly in `MyDrive/ECG-Ramba` if that folder contains `configs/`,
`src/`, `scripts/`, and `notebooks/`.

## Upload Options

Option A - GitHub, preferred:

```bash
cd /content/drive/MyDrive/ECG-Ramba
git clone -b revision/colab-pipeline https://github.com/BrianNguyen29/ECG-RAMBA.git
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

Option C - Apply the revision patch zip:

Use this only when Git push/pull is not available.

1. Upload `ECG_RAMBA_colab_revision_patch_v2.zip` to
   `MyDrive/ECG-Ramba/notebook`.
2. In Colab, extract it into the repo root, not into `notebooks/`.
3. Open `notebooks/00_revision_runner.ipynb` and run top to bottom.

## First Colab Run

Run the lightweight notebook:

```text
notebooks/00_revision_runner.ipynb
```

Start with:

```bash
python scripts/revision/00_audit_protocol.py
```

Do not launch full training/evaluation until the audit warnings are resolved.

## Minimum Artifact Contract

Revision scripts should write artifacts to:

```text
reports/revision/
  audit_protocol.json
  hrv36_schema.csv
  oof_predictions.npz
  ptbxl_predictions.npz
  cpsc_predictions.npz
  calibration_ci.json
  ablation_results.csv
  robustness_results.csv
  figures/
```

Prediction files used by `04_calibration_ci.py` must contain:

```text
y_true: shape (N, C)
y_prob: shape (N, C)
```

## Blockers to Resolve Before Manuscript Numbers

- HRV36 currently contains 5 RR summary slots, 20 reserved zero slots,
  5 amplitude slots, and 6 global signal-stat slots. Do not describe it as
  full HRV with RMSSD/SDNN/LF-HF unless those features are implemented and
  the model is retrained.
- CPSC cells in the exploratory notebook use different sources/protocols.
  Build one canonical manifest and use it everywhere.
- Use the PTB superclass mapping in `scripts.revision.common.PTB_SUPERCLASS_MAPPING`.
- Treat inference ablation as diagnostic. Reviewer-facing component claims need
  retrained or protocol-fair baselines where possible.
- Add calibration, bootstrap CI, learned CNN/ResNet baseline, HRV domain check,
  and minimal robustness tests before final response-letter claims.
