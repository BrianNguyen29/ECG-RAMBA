# Full Reviewer Experiment Runbook - 2026-07-07

> Superseded for direct execution by
> [reviewer_evidence_direct_run_20260711.md](reviewer_evidence_direct_run_20260711.md).
> This historical runbook remains for provenance only; its MiniRocket wording
> and step order predate the current method-identity and auto-defer guards.

This runbook opens the remaining optional reviewer experiments while preserving the frozen evidence contract. It is intentionally conservative: every new experiment must either produce auditable artifacts or remain explicitly blocked.

## Scope

Run these additional experiments if the revision should answer reviewer requests as fully as possible:

1. Transformer/foundation-style raw ECG comparator.
2. Hybrid/partially learnable MiniRocket morphology head.
3. Learned-comparator robustness against ResNet1D/CNN and Raw Mamba.
4. Full HRV feature set only as a separate retrain branch.

Do not use these experiments to revive broad superiority claims. The current fair-baseline evidence already shows ResNet1D/CNN and Raw Mamba outperform ECG-RAMBA on several principal in-domain metrics.

## Hardware

- Notebook 04 heavy baselines: A100 High-RAM strongly preferred.
- Notebook 05 learned-comparator stress inference: A100 High-RAM strongly preferred.
- Notebook 07 final evidence regeneration: CPU is sufficient.
- Full HRV retrain branch: A100 High-RAM for all model-training notebooks.

## Prerequisites

Run or restore these first:

1. Notebook 00 setup/bootstrap.
2. Notebook 01 protocol audit.
3. Notebook 02 OOF/external artifacts, including final EMA OOF.
4. Notebook 03 calibration CI.
5. Notebook 04 required MiniRocket/ResNet/Raw Mamba artifacts if not already present.

The active repo and Drive mirror should contain:

- `reports/revision/predictions/oof_final_ema_predictions.npz`
- `reports/revision/manifests/oof_final_ema_freeze_manifest.json`
- `reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json`

## Track 1 - Transformer ECG Comparator

Notebook: `notebooks/04_baselines_and_component_checks.ipynb`

Cell: `Transformer ECG Baseline Runner`

Current direct-run configuration:

```python
RUN_TRANSFORMER_ECG_BASELINE = True
TRANSFORMER_EPOCHS = 20
TRANSFORMER_BATCH_SIZE = 256
TRANSFORMER_N_BOOT = 1000
TRANSFORMER_EMBED_DIM = 96
TRANSFORMER_FORCE_RERUN = False
TRANSFORMER_SAVE_CHECKPOINTS = True
```

Expected outputs:

- `reports/revision/predictions/transformer_ecg_oof_predictions.npz`
- `reports/revision/predictions/transformer_ecg_slice_predictions.npz`
- `reports/revision/metrics/transformer_ecg_baseline_summary.json`
- `reports/revision/tables/table_transformer_ecg_class_metrics.csv`
- `reports/revision/tables/table_transformer_ecg_fold_summary.csv`
- `reports/revision/manifests/transformer_ecg_baseline_manifest.json`
- Optional checkpoints under `reports/revision/experimental/transformer_ecg_checkpoints/`

Then run the paired Transformer cell in Notebook 04. Auto mode runs when all inputs exist:

- `reports/revision/metrics/paired_full_vs_transformer_comparison.json`
- `reports/revision/tables/table_paired_full_vs_transformer.csv`
- `reports/revision/manifests/paired_full_vs_transformer_manifest.json`

Allowed wording after completion:

- Comparator-specific Transformer ECG result only.
- No broad superiority claim.

## Track 2 - Hybrid MiniRocket Morphology MLP

Notebook: `notebooks/04_baselines_and_component_checks.ipynb`

Cell: `Hybrid MiniRocket Morphology MLP Runner`

Current direct-run configuration:

```python
RUN_HYBRID_MORPHOLOGY_BASELINE = True
HYBRID_MORPHOLOGY_EPOCHS = 20
HYBRID_MORPHOLOGY_BATCH_SIZE = 4096
HYBRID_MORPHOLOGY_STATS_BATCH_SIZE = 1024
HYBRID_MORPHOLOGY_N_BOOT = 1000
HYBRID_MORPHOLOGY_FORCE_RERUN = False
```

Expected outputs:

- `reports/revision/predictions/hybrid_morphology_oof_predictions.npz`
- `reports/revision/metrics/hybrid_morphology_baseline_summary.json`
- `reports/revision/tables/table_hybrid_morphology_class_metrics.csv`
- `reports/revision/tables/table_hybrid_morphology_fold_summary.csv`
- `reports/revision/manifests/hybrid_morphology_baseline_manifest.json`

Then run the paired Hybrid cell in Notebook 04. Auto mode runs when all inputs exist:

- `reports/revision/metrics/paired_full_vs_hybrid_morphology_comparison.json`
- `reports/revision/tables/table_paired_full_vs_hybrid_morphology.csv`
- `reports/revision/manifests/paired_full_vs_hybrid_morphology_manifest.json`

Allowed wording after completion:

- Controlled morphology-head sensitivity.
- Do not claim deterministic MiniRocket morphology is causally superior.

## Track 3 - Learned-Comparator Robustness

This track requires ResNet and Raw Mamba checkpoints, not just clean OOF prediction caches.

### Step 3.1 Save ResNet checkpoints

Notebook: `notebooks/04_baselines_and_component_checks.ipynb`

Cell: `ResNet1D/CNN Baseline Runner`

Use:

```python
RUN_RESNET1D_CNN_BASELINE = True
RESNET_FORCE_RERUN = 'auto'
RESNET_SAVE_CHECKPOINTS = True
RESNET_RETRAIN_FOR_MISSING_CHECKPOINTS = False
```

`RESNET_FORCE_RERUN='auto'` reruns the ResNet baseline only when baseline
prediction/summary artifacts are missing. Missing checkpoint files alone do not
trigger retraining while `RESNET_RETRAIN_FOR_MISSING_CHECKPOINTS=False`, so
completed baseline outputs are reused by default. Set
`RESNET_RETRAIN_FOR_MISSING_CHECKPOINTS=True` only if Notebook 05 learned-
comparator stress inference requires ResNet checkpoint files.

Expected checkpoint outputs:

- `reports/revision/experimental/resnet1d_cnn_checkpoints/fold1_resnet1d_cnn_final.pt`
- `reports/revision/experimental/resnet1d_cnn_checkpoints/fold2_resnet1d_cnn_final.pt`
- `reports/revision/experimental/resnet1d_cnn_checkpoints/fold3_resnet1d_cnn_final.pt`
- `reports/revision/experimental/resnet1d_cnn_checkpoints/fold4_resnet1d_cnn_final.pt`
- `reports/revision/experimental/resnet1d_cnn_checkpoints/fold5_resnet1d_cnn_final.pt`

### Step 3.2 Save Raw Mamba checkpoints

Notebook: `notebooks/04_baselines_and_component_checks.ipynb`

Cell: `Raw Mamba Baseline Runner`

Use the direct-run auto preset:

```python
RUN_RAW_MAMBA_BASELINE = True
RAW_MAMBA_SAVE_CHECKPOINTS = True
RAW_MAMBA_FORCE_RERUN = 'auto'
RAW_MAMBA_ONLY_FOLDS = 'auto'
RAW_MAMBA_RETRAIN_FOR_MISSING_CHECKPOINTS = False
```

`RAW_MAMBA_ONLY_FOLDS='auto'` selects only missing fold prediction caches. If
all fold caches and canonical baseline outputs are already present, the cell
runs in aggregate/reuse mode and does not retrain. Missing checkpoint files
alone do not trigger retraining while
`RAW_MAMBA_RETRAIN_FOR_MISSING_CHECKPOINTS=False`. Set that flag to `True`
only if Notebook 05 learned-comparator stress inference requires Raw Mamba
checkpoint files.

Expected checkpoint outputs:

- `reports/revision/experimental/raw_mamba_checkpoints/fold1_raw_mamba_final_ema.pt`
- `reports/revision/experimental/raw_mamba_checkpoints/fold2_raw_mamba_final_ema.pt`
- `reports/revision/experimental/raw_mamba_checkpoints/fold3_raw_mamba_final_ema.pt`
- `reports/revision/experimental/raw_mamba_checkpoints/fold4_raw_mamba_final_ema.pt`
- `reports/revision/experimental/raw_mamba_checkpoints/fold5_raw_mamba_final_ema.pt`

### Step 3.3 Generate stressed comparator predictions

Notebook: `notebooks/05_hrv_domain_and_robustness.ipynb`

Cell: `Comparator Stress Prediction Generation`

Use:

```python
RUN_COMPARATOR_STRESS_PREDICTIONS = True
COMPARATOR_STRESS_BATCH_SIZE = 256
COMPARATOR_STRESS_STRICT = False
```

Expected prediction outputs include:

- `reports/revision/predictions/robustness_resnet1d_cnn_snr20db_predictions.npz`
- `reports/revision/predictions/robustness_raw_mamba_snr20db_predictions.npz`
- and corresponding files for `snr10db`, `snr5db`, `random_3_lead_dropout`, `precordial_dropout`, and `resample_250hz`.

### Step 3.4 Aggregate multi-comparator robustness

Notebook: `notebooks/05_hrv_domain_and_robustness.ipynb`

Cell: `Multi-Comparator Robustness Ledger`

Use:

```python
RUN_ROBUSTNESS_MULTICOMPARATOR = True
ROBUSTNESS_MULTI_N_BOOT = 1000
ROBUSTNESS_MULTI_STRICT = False
```

Expected outputs:

- `reports/revision/metrics/robustness_full_vs_resnet_comparison.json`
- `reports/revision/metrics/robustness_full_vs_raw_mamba_comparison.json`
- `reports/revision/metrics/robustness_multicomparator_pairwise.json`
- `reports/revision/metrics/robustness_multicomparator_summary.csv`
- `reports/revision/tables/table_robustness_multicomparator.csv`
- `reports/revision/manifests/robustness_multicomparator_manifest.json`

Allowed wording after completion:

- Metric-specific, comparator-specific robustness only.
- No general robustness superiority claim unless every relevant metric/comparator supports it.

## Track 4 - Full HRV Feature Set

This is not a lightweight add-on and must not be mixed with the current final-EMA evidence.

Current reason:

- `src/features.py` currently extracts five RR summary statistics, amplitude features, and global record statistics into a fixed 36-dimensional checkpoint-compatible vector.
- Current final-EMA checkpoints were trained under that feature contract.
- Replacing reserved/zero or limited HRV slots with RMSSD/SDNN/LF-HF would silently change feature semantics and invalidate the current frozen checkpoint contract.

Required full-HRV branch:

1. Define a new HRV schema, e.g. `hrv_full36_schema_v1`, including RMSSD, SDNN, pNN50/pNN20, HR, robust RR summaries, and frequency-domain LF/HF only if defensible for the record length and sampling.
2. Add a new cache name and manifest so old `hrv36` caches cannot be reused accidentally.
3. Retrain all five folds with the new HRV schema.
4. Regenerate OOF, freeze manifest, calibration CI, baselines, paired comparisons, external gates, robustness, and final evidence.
5. Keep current final-EMA evidence separate from the full-HRV branch.

Allowed wording before this branch is complete:

- Do not claim full HRV features, RMSSD, SDNN, LF/HF, or HRV domain invariance.

## Finalization After Additional Experiments

After any new optional artifact is created:

1. Publish the mirror:

```bash
python -u scripts/revision/artifact_mirror.py publish --mirror-root "/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision"
```

2. Rerun Notebook 07.
3. Use regenerated:

- `table_final_evidence_matrix.csv`
- `table_final_safe_wording.csv`
- `table_claim_readiness_gates.csv`

4. Update manuscript/rebuttal from regenerated tables only.
5. Compile PDF.
6. Run forbidden-claim scan again.
