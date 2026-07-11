# Revision Notebook Suite

Use these notebooks on Colab as the execution layer for the reviewer-revision
plan. Large data and model files stay on Google Drive; GitHub only transports
code, notebooks, and planning metadata.

## Drive Artifact Source

All notebooks restore and publish revision artifacts through one canonical
mirror:

```text
/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision
```

`/content/drive/MyDrive/ECG-Ramba/ECG-RAMBA` is a legacy source checkout and
must not be used to decide whether an experiment cache exists. Notebook 00
audits that legacy tree and can import legacy-only files without overwriting
canonical conflicts.

## Recommended Order

1. `00_colab_bootstrap.ipynb`: mount Drive, pull the current branch, validate
   source compatibility, install the numeric/Mamba runtime, and print cache,
   checkpoint, stress-prediction, and log counts.
2. `01_a0_protocol_audit.ipynb`: run the protocol/A0 gates and publish their
   artifacts to the verified Drive mirror.
3. `02_predictions_and_external_eval.ipynb`: freeze/reuse canonical OOF,
   external Full-model outputs, protocol gates, and PTB-XL fold-9 inputs.
4. `03_calibration_and_ci.ipynb`: regenerate/reuse calibration and bootstrap
   CI for the active OOF. Reviewer presentation assets can defer on this first
   pass until paired baseline artifacts are current.
5. `04_baselines_and_component_checks.ipynb`: on A100, leave baseline controls
   on `auto`; only missing/stale folds or checkpoints are trained. Run paired
   comparisons and the completion ledger afterwards.
6. Return to the end of `02_predictions_and_external_eval.ipynb` for external
   learned comparators, paired external audits, group-safe score calibration,
   representation export, and true frozen-encoder head adaptation.
7. Rerun the reviewer-presentation cell in `03_calibration_and_ci.ipynb`.
8. `05_hrv_domain_and_robustness.ipynb`: use A100 only for missing comparator
   stress predictions; HRV, aggregation, and bootstrap ledgers are CPU-bound.
9. `06_pooling_and_representation.ipynb`: external pooling/probes are CPU;
   missing ECG-RAMBA representation embeddings require A100 plus Mamba.
10. `07_results_freeze.ipynb`: CPU-only strict final claim/evidence gate.

`02a_retrain_best_ema.ipynb` is intentionally excluded from the normal
revision sequence. Run it only when creating a new five-fold core ECG-RAMBA
checkpoint set. Its default `RUN_RETRAIN` is false and existing folds are
resume-audited rather than overwritten.

## Resume Rules

- Rerun the current notebook from **Setup** after a Colab disconnect.
- Keep `FORCE_RERUN_* = False`, fold selectors on `auto`, checkpoint saving
  enabled, and reuse controls enabled unless an artifact is known to be bad.
- A completed heavy fold is reusable only after its checkpoint/fold prediction
  and the subsequent `Published and verified ... artifacts` message appear.
- Normal command output is streamed to the notebook and written under
  `reports/revision/logs/`. The verified mirror is
  `/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision`.
- Fresh Colab clones can run this suite only after the current notebooks and
  revision scripts are committed and pushed to the selected branch.

Original exploratory/demo notebooks are retained under `notebooks/archive/`.
Avoid adding reviewer-revision work to those legacy notebooks.

Legacy direct-runner notebooks were removed to avoid running the wrong setup
path. Use the numbered suite above for revision work.
