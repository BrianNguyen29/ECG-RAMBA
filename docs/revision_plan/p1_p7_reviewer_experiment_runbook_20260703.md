# P1-P7 Reviewer Experiment Runbook

Updated: 2026-07-06

This runbook converts the reviewer-driven deferred workstreams into explicit,
cache-aware execution steps. It does not relax the current safe-claim boundary:
new claims are allowed only after the named gate artifacts pass.

## P1 - Georgia External Gate

Status: implemented as a gated exporter path; not manuscript-ready until a
reviewed mapping yields usable labels and the gate passes.

Implementation:

- Exporter: `scripts/revision/03_generate_external_predictions.py`
- Gate: `scripts/revision/18_external_protocol_gate.py`
- Review file: `docs/revision_plan/georgia_label_mapping_review_20260703.csv`
- Inventory output: `reports/revision/tables/table_georgia_snomed_code_inventory.csv`

Notebook 02 settings:

```python
RUN_GEORGIA_EXPORT = True
RUN_EXTERNAL_PROTOCOL_GATE = True
EXTERNAL_GATE_DATASETS = 'ptbxl,georgia'
EXTERNAL_GATE_STRICT = False
```

Pass criteria:

- The review file maps only clinically reviewed SNOMED codes to frozen
  Chapman classes.
- Unmapped Georgia labels are skipped and reported; they are never coerced to
  negative labels.
- `external_georgia_protocol_gate.json` has `protocol_gate_passed=true`.

Allowed wording after pass: Georgia mapped-task evaluation only. No
unqualified zero-shot or external superiority claim.

## P2 - CPSC2021 Annotation-Aligned Gate

Status: complete as a protocol-gated mapped-window external evaluation. It is
not the official episode-boundary CPSC2021 challenge score.

Implementation:

- Exporter: `scripts/revision/03_generate_external_predictions.py`
- Gate: `scripts/revision/18_external_protocol_gate.py`
- Audit output: `reports/revision/tables/table_cpsc2021_annotation_audit.csv`

Notebook 02 settings, if regeneration is needed:

```python
RUN_CPSC2021_EXPORT = True
RUN_EXTERNAL_PROTOCOL_GATE = True
EXTERNAL_GATE_DATASETS = 'ptbxl,cpsc2021'
EXTERNAL_GATE_STRICT = False
```

Pass criteria:

- Signals and annotations are WFDB-readable.
- Windows are positive only when AF/AFL occupies the majority of the window.
- Windows are negative only when recognized normal rhythm occupies the
  majority of the window.
- Ambiguous windows are skipped and counted.
- `external_cpsc2021_protocol_gate.json` has `protocol_gate_passed=true`.

Allowed wording after pass: annotation-aligned CPSC2021 mapped rhythm-window
evaluation only.

## P3 - Few-Shot After A Passed External Gate

Status: complete for PTB-XL score calibration. CPSC2021 is now runnable because
its mapped-window external protocol gate has passed. Georgia stays excluded
until its gate passes.

Notebook 02 settings:

```python
RUN_FEWSHOT_ADAPTATION = True
FORCE_RERUN_FEWSHOT_ADAPTATION = False
FEWSHOT_DATASETS = 'ptbxl,cpsc2021'
FEWSHOT_FRACTIONS = '0,0.01,0.05,0.10'
FEWSHOT_SEEDS = '42,43,44,45,46'
FEWSHOT_TEST_FRACTION = 0.50
FEWSHOT_N_BOOT = 1000
FEWSHOT_STRICT = True
```

Pass criteria:

- External gate for the dataset passed.
- Split file is saved.
- The method is described as score calibration, not weight fine-tuning.

Current completed evidence: PTB-XL score calibration. CPSC2021 score
calibration should be run next if broader external-adaptation evidence is
needed; it must remain described as dataset-specific frozen-score calibration
on annotation-aligned AF/AFL mapped windows.

## P4 - Robustness Vs ResNet1D/CNN And Raw Mamba

Status: aggregation gate and stressed-prediction generator exist; comparator
checkpoints must be saved first.

Notebook 04 checkpoint rerun:

```python
RUN_RESNET1D_CNN_BASELINE = True
RESNET_SAVE_CHECKPOINTS = True
RESNET_FORCE_RERUN = True

RUN_RAW_MAMBA_BASELINE = True
RAW_MAMBA_SAVE_CHECKPOINTS = True
RAW_MAMBA_FORCE_RERUN = True
RAW_MAMBA_ONLY_FOLDS = '1'  # run fold by fold on Colab, then aggregate with ''
```

Notebook 05 stressed predictions and aggregation:

```python
RUN_COMPARATOR_STRESS_PREDICTIONS = True
RUN_ROBUSTNESS_MULTICOMPARATOR = True
```

Pass criteria:

- `reports/revision/experimental/resnet1d_cnn_checkpoints/fold*_resnet1d_cnn_final.pt`
  exist.
- `reports/revision/experimental/raw_mamba_checkpoints/fold*_raw_mamba_final_ema.pt`
  exist.
- `robustness_resnet1d_cnn_*_predictions.npz` and
  `robustness_raw_mamba_*_predictions.npz` exist for each stress.
- Multi-comparator summary records completed rows, not missing-artifact rows.

Allowed wording: metric-specific paired degradation only.

## P5 - Transformer ECG Comparator

Status: optional runner plus paired bootstrap gate implemented; not part of
current final evidence until run and reviewed.

Runner:

```text
scripts/revision/24_transformer_ecg_baseline.py
scripts/revision/25_paired_full_vs_transformer.py
```

Notebook 04 settings:

```python
RUN_TRANSFORMER_ECG_BASELINE = True
TRANSFORMER_BATCH_SIZE = 256
TRANSFORMER_EMBED_DIM = 96
```

Outputs:

- `reports/revision/predictions/transformer_ecg_oof_predictions.npz`
- `reports/revision/metrics/transformer_ecg_baseline_summary.json`
- `reports/revision/tables/table_transformer_ecg_class_metrics.csv`
- `reports/revision/manifests/transformer_ecg_baseline_manifest.json`
- `reports/revision/metrics/paired_full_vs_transformer_comparison.json`
- `reports/revision/tables/table_paired_full_vs_transformer.csv`
- `reports/revision/manifests/paired_full_vs_transformer_manifest.json`

Allowed wording: comparator-specific transformer baseline. A paired comparison
must be generated before using significant delta language.

## P6 - Hybrid / Partially Learnable Morphology

Status: not run by default. This remains mechanism-oriented optional evidence.

Required before implementation:

- Define whether the hybrid tests learnability, determinism, or regularization.
- Use the same frozen folds, threshold, and Q=3 aggregation.
- Compare against MiniRocket-only, ResNet1D/CNN, Raw Mamba, and Full.

Allowed wording after completion: mechanism sensitivity only. Do not infer
causal disentanglement.

## P7 - Full HRV Feature Set

Status: retrain-level only. The current final EMA checkpoints cannot support
RMSSD/SDNN/LF-HF claims.

Required before implementation:

- Replace reserved zero-filled HRV slots with real features.
- Update `hrv36_schema.csv`.
- Retrain all five final EMA folds.
- Regenerate OOF, calibration, baselines, HRV/domain evidence, robustness,
  pooling, representation, and final evidence.

Allowed wording only after full retrain: full HRV feature-set evidence under a
new frozen protocol. Do not mix this with the current final EMA evidence.

## Recommended Execution Order

1. Run Notebook 02 in reuse/auto mode to keep PTB-XL and CPSC2021 gates current.
2. Run P3 CPSC2021 few-shot score calibration if broader external-adaptation
   evidence is needed.
3. Rerun Notebook 07 to regenerate final evidence tables.
4. Run P1 Georgia gate only if the mapping review produces usable export
   artifacts; keep it deferred otherwise.
5. Run P4 comparator checkpoint saving and multi-comparator robustness if broad
   robustness is still needed.
6. Run P5 transformer only if a transformer-specific comparator is required.
7. Keep P6/P7 deferred unless the rebuttal explicitly needs mechanism or full
   HRV retrain evidence.
