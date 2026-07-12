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

Long-running state is written directly below the canonical mirror:

- `predictions/folds/`: resumable OOF, baseline, and representation fold caches.
- `predictions/external_representation_folds/`: source-bound PTB-XL fold-9/
  fold-10 record-embedding caches for matched frozen-encoder adaptation.
- `predictions/fewshot_head_adaptation_cache/` and
  `metrics/true_fewshot_head_metric_cache/`: group-safe head predictions,
  coefficient sidecars, and paired group-bootstrap caches.
- `experimental/*_checkpoints/`: ResNet, Raw Mamba, Transformer, and optional
  frozen-transform head checkpoints.
- `logs/`: live command traces, flushed while a cell is running.

The five core Full ECG-RAMBA checkpoints remain under the immutable
`model_runs/<run_id>/` directory selected by
`current_final_ema_model_dir.txt`; their paths and SHA256 values are frozen in
`oof_final_ema_prediction_run_manifest.json`. They are not duplicated into the
revision mirror.

`revision_pca_models`, `revision_feature_cache`, and
`revision_external_cache` are separate manifest-keyed storage tiers, not
alternative evidence sources. `final_evidence_tables` is output-only and has
its own export checksum manifest. A partial Windows Drive download may omit
these large tiers; verify cloud presence from Colab before scheduling a rerun.

Logs intentionally stay outside the immutable checksum manifest because a
rerun may rewrite the same log path. Fold caches, checkpoints, predictions,
tables, metrics, and manifests remain checksum-tracked evidence artifacts.
Existence alone is never a reuse contract: restore cells require a manifest
row plus matching size and SHA256. Interrupted `.partial`, `.tmp`, and `.lock`
files are excluded from mirror discovery. A completed direct-to-Drive fold is
durable immediately, but run its publish step before a later notebook consumes
it as evidence.

Notebook 00 also writes `metrics/pipeline_storage_audit.json` and
`tables/table_pipeline_storage_audit.csv`. The audit verifies each required
fold and each named stress separately, so duplicate cache variants or a
MiniRocket clean-reference file cannot hide a missing fold/stress. An
complete full-reviewer audit also requires the exact external representation
and adaptation cache slots. An incomplete bootstrap audit is informational; use
`38_pipeline_storage_audit.py --strict --full-sha` only for the final package.

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
- A completed heavy fold survives immediately after its atomic checkpoint and
  fold-cache write to the canonical Drive path. The following publish step
  registers their SHA256 values in the mirror manifest.
- If a fold cache is missing but a compatible checkpoint remains, ResNet,
  Raw Mamba, Transformer, and the frozen-transform MLP regenerate validation
  predictions without training. Contract mismatches are rejected.
- Full external inference authenticates its five checkpoints against the OOF
  run manifest before loading them. Learned external comparators use their
  baseline checkpoint manifests. External representation protocol v2 also
  binds fold caches to the source prediction manifest, dataset archive/loader
  provenance, canonical OOF, and checkpoint SHA256. True few-shot adaptation
  requires the matching v2 embedding manifest and NPZ; its 1/5/10% budgets are
  nested fractions of independent target groups, with 10% pre-specified as the
  primary endpoint rather than selected on the test set.
- Normal command output is streamed to the notebook and written under
  local `reports/revision/logs/` and canonical Drive `logs/` simultaneously.
- Frequent intermediate publishes use size verification for unchanged
  manifest rows while always SHA-verifying new/overwritten files. Notebook 07
  performs the final full-manifest checksum pass.
- Fresh Colab clones can run this suite only after the current notebooks and
  revision scripts are committed and pushed to the selected branch.

Original exploratory/demo notebooks are retained under `notebooks/archive/`.
Avoid adding reviewer-revision work to those legacy notebooks.

Legacy direct-runner notebooks were removed to avoid running the wrong setup
path. Use the numbered suite above for revision work.
