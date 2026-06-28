# ECG-RAMBA Deferred Evidence Implementation Plan

Date: 2026-06-22

Purpose: define a technically correct plan for evidence items that remain incomplete or must stay deferred in the current rebuttal package. This document is not a claim source. The current claim source remains:

```text
docs/revision_plan/manuscript_rebuttal_drive_source_update_20260622.md
docs/revision_plan/discussion_final_evidence_review_20260622.md
reports/revision/metrics/final_evidence_matrix.json
reports/revision/tables/table_final_safe_wording.csv
```

## Current Boundary

The current manuscript/rebuttal package is internally consistent only if these restrictions are preserved:

- Do not claim global or in-domain superiority for ECG-RAMBA. ResNet1D/CNN is stronger on frozen Chapman OOF PR-AUC, ROC-AUC, F1, Brier, and ECE; Raw Mamba is stronger on PR-AUC, ROC-AUC, and F1.
- Do not claim superiority over all fair baselines. The Raw Mamba fair comparator has completed successfully and Notebook 04/07 have been rerun, but the resulting evidence still supports only comparator-specific, metric-specific statements.
- Do not claim external or zero-shot superiority because PTB-XL, Georgia, and CPSC remain experimental.
- Do not claim few-shot results because no few-shot adaptation package is complete.
- Do not claim proven morphology-rhythm disentanglement because no representation probe, UMAP, CKA, or probing artifact is complete.
- Do not claim a full HRV feature set because the current final EMA checkpoints do not implement true RMSSD, SDNN, LF/HF, or full amplitude HRV slots.
- Do not claim general robustness superiority. Current robustness evidence is metric-specific: ECG-RAMBA is better than MiniRocket-only for stressed F1/Brier/ECE, while MiniRocket-only is better for stressed PR-AUC/ROC-AUC and is less degraded in most stress-metric rows.

## Evidence Inventory

| Area | Current state | Existing support | Claim status |
|---|---|---|---|
| Frozen Chapman OOF | Complete | final EMA checkpoints, Q=3 aggregation, calibration CI | Manuscript-ready |
| ResNet1D/CNN fair baseline | Complete | `scripts/revision/14_resnet1d_cnn_baseline.py`, paired Full-vs-ResNet | Manuscript-ready, favors ResNet |
| MiniRocket-only baseline | Complete | `scripts/revision/10_minirocket_only_baseline.py`, paired Full-vs-MiniRocket | Manuscript-ready, metric-specific |
| HRV-only/domain | Complete with limitation | `scripts/revision/09_hrv_domain_analysis.py` | Manuscript-ready limitation |
| Robustness vs MiniRocket | Complete with limitation | `scripts/revision/12_robustness_stress.py`, 6 stresses x 5 metrics | Manuscript-ready, metric-specific |
| Raw Mamba fair comparator | Complete | `scripts/revision/16_raw_mamba_baseline.py`, `scripts/revision/17_paired_full_vs_raw_mamba.py`, paired Full-vs-Raw-Mamba artifacts | Manuscript-ready, metric-specific |
| External PTB-XL/Georgia/CPSC | Scaffolded with gate runner, but not yet manuscript-ready until dataset gates pass | `scripts/revision/03_generate_external_predictions.py`, `scripts/revision/18_external_protocol_gate.py`, fold PCA builder | Deferred unless gate passes |
| Few-shot adaptation | Missing | no runner/artifacts | Deferred |
| Representation probe | Missing | Notebook 06 records blocked status | Deferred |
| Full HRV feature set | Not implemented in current checkpoints | HRV36 schema audit | Deferred/retrain-only |
| General robustness superiority | Not supported | current robustness table contradicts broad wording | Must remain blocked |

## Recommended Priority

### Completed - Raw Mamba fair comparator

Raw Mamba is now complete and closes the previous fair-baseline-row gap. It does not restore global superiority: Raw Mamba is stronger than Full ECG-RAMBA on PR-AUC, ROC-AUC, and F1, while Full ECG-RAMBA has lower Brier score and ECE. Use only comparator-specific, metric-specific wording.

### P2 - External protocol gate

Run only if a transfer/generalization claim is required. The gate runner now exists, but external evidence still needs dataset-specific label/window/PCA checks before it can be cited. Until a dataset passes the gate, it must remain experimental.

### P2 - Representation probe

Run only if the rebuttal needs stronger support for morphology-rhythm architecture interpretation. This should support cautious representation analysis, not causal disentanglement.

### P3 - Few-shot adaptation

Run only after external label/window protocols are locked. Few-shot without an external protocol gate is high leakage risk and easy to overclaim.

### P4 - Full HRV feature set

Do not open this for the current rebuttal unless the paper must make HRV-feature claims. It changes model input semantics and requires full 5-fold retraining.

### P4 - General robustness superiority

Do not pursue as a blanket claim. If needed, extend robustness to ResNet1D/CNN and Raw Mamba, then report only metric-specific conclusions.

## Workstream A: Raw Mamba Fair Comparator - Completed

### Goal

Train a raw-signal Mamba-only comparator under the same frozen Chapman OOF protocol. This is a fair architecture baseline, not an inference ablation.

### Technical Design

Implemented:

```text
scripts/revision/16_raw_mamba_baseline.py
scripts/revision/17_paired_full_vs_raw_mamba.py
```

Use `scripts/revision/14_resnet1d_cnn_baseline.py` as the template because it already implements:

- frozen OOF contract validation;
- record-order and label equality checks;
- subject-aware fold reuse from frozen OOF;
- slice protocol compatibility;
- Q=3 power-mean record aggregation;
- fixed threshold 0.5;
- summary JSON, class table, fold table, prediction NPZ, manifest;
- per-fold prediction caches and `--reuse-predictions`.

The model should instantiate the existing ECG-RAMBA model in a raw-only mode:

```python
ECGRambaV7Advanced(
    cfg=CONFIG,
    ablation={"no_rocket": True, "no_hrv": True, "no_fusion": True},
)
```

The runner must not pass MiniRocket PCA features or HRV information into training. It uses zero tensors only to satisfy the forward signature while structurally disabling Rocket, HRV, and fusion modules. The manifest must declare:

```text
feature_contract = raw_ecg_12lead_mamba_only
uses_minirocket = false
uses_hrv = false
uses_pca = false
checkpoint_kind = raw_mamba_retrained_weighted_bce_same_folds_power_mean_v2_q3_threshold_0.5
```

### Outputs

Required outputs:

```text
reports/revision/predictions/raw_mamba_oof_predictions.npz
reports/revision/predictions/raw_mamba_slice_predictions.npz
reports/revision/metrics/raw_mamba_baseline_summary.json
reports/revision/tables/table_raw_mamba_class_metrics.csv
reports/revision/tables/table_raw_mamba_fold_summary.csv
reports/revision/manifests/raw_mamba_baseline_manifest.json
reports/revision/logs/baseline_raw_mamba.log
```

Add paired comparison:

```text
scripts/revision/17_paired_full_vs_raw_mamba.py
reports/revision/metrics/paired_full_vs_raw_mamba_comparison.json
reports/revision/tables/table_paired_full_vs_raw_mamba.csv
reports/revision/manifests/paired_full_vs_raw_mamba_manifest.json
```

### Notebook Updates

Update:

```text
notebooks/04_baselines_and_component_checks.ipynb
scripts/revision/13_final_evidence_matrix.py
docs/revision_plan/experiment_registry.csv
docs/revision_plan/task_board.csv
```

Notebook 04 should change the Raw Mamba row only when:

- prediction NPZ exists;
- manifest exists;
- OOF contract passes;
- same `record_id`, `fold_id`, `class_names`, and `y_true` as frozen OOF;
- `feature_contract == raw_ecg_12lead_mamba_only`;
- paired comparison exists.

### Compute

Recommended runtime:

```text
A100 High-RAM
```

T4 is not recommended because Mamba training is GPU-bound and Colab timeout risk is high.

### Claim Gate

The completed result supports:

> Raw Mamba was evaluated as an additional fair comparator under the same frozen OOF protocol.

It cannot support:

> ECG-RAMBA is globally superior.

Raw Mamba is stronger than Full ECG-RAMBA on PR-AUC, ROC-AUC, and F1, while Full ECG-RAMBA has lower Brier score and ECE. The manuscript must therefore report a comparator-specific tradeoff and must not state that the structured ECG-RAMBA design improves in-domain performance over Raw Mamba.

## Workstream B: External PTB-XL / Georgia / CPSC Protocol Gate

### Goal

Convert current external exporters from experimental artifacts into protocol-gated evidence, only if external claims are needed.

### Existing Code

Current scaffold:

```text
scripts/revision/03_generate_external_predictions.py
scripts/revision/08_build_fold_pca.py
scripts/revision/18_external_protocol_gate.py
notebooks/02_predictions_and_external_eval.ipynb
```

The exporter already includes important safeguards:

- `--acknowledge-experimental`;
- official PTB diagnostic superclass mapping;
- CPSC annotation-window handling;
- checkpoint-compatible HRV36 construction;
- fold-specific PCA manifest requirement.

### Required Gate Before Claiming

Run the external protocol gate script after external predictions are generated:

```text
scripts/revision/18_external_protocol_gate.py
```

Operational controls now implemented:

- Notebook 02 writes the gate run log to `reports/revision/logs/external_protocol_gate.log`.
- The gate writes a per-dataset `gate_cache_key` from source artifact SHA256 values, OOF manifest SHA256, metric parameters, and gate schema version.
- `--reuse-existing` reuses prior gate outputs when the cache key and required output artifacts match, avoiding repeated bootstrap work.
- The summary CSV exposes `reused_existing`, `gate_cache_key`, source prediction SHA256, source slice-prediction SHA256, and gate artifact paths for monitoring.
- Gate outputs are ordinary `reports/revision` artifacts and should be published/restored through `scripts/revision/artifact_mirror.py` after a successful run.

The gate should validate:

- dataset archive SHA256;
- record count;
- label source;
- mapped/unmapped label counts;
- windowing rule;
- class set used for evaluation;
- fold PCA manifest completeness and checksum;
- checkpoint fingerprint match;
- no missing or silently negative labels;
- metrics are reported only for mapped labels.

### Outputs

For each dataset:

```text
reports/revision/metrics/external_<dataset>_protocol_gate.json
reports/revision/metrics/external_protocol_gate_summary.csv
reports/revision/tables/table_external_<dataset>_label_mapping.csv
reports/revision/tables/table_external_<dataset>_metrics.csv
reports/revision/manifests/external_<dataset>_protocol_gate_manifest.json
```

### Claim Gate

Until all gates are complete:

> External outputs remain experimental and are not manuscript-ready.

If complete, the safe claim is still narrow:

> We report protocol-gated external evaluation under mapped label/task definitions.

Avoid:

> zero-shot superiority
> cross-dataset robustness superiority
> external SOTA

## Workstream C: Few-Shot Adaptation

### Goal

Evaluate whether ECG-RAMBA adapts efficiently on external datasets after the external protocol gate is complete.

### Preconditions

Do not start until Workstream B is complete for at least one target dataset.

### Technical Design

Create:

```text
scripts/revision/19_fewshot_adaptation.py
```

Protocol:

- target dataset split: train/validation/test or repeated stratified train/test splits;
- shot fractions: for example 1%, 5%, 10%;
- repeated seeds: at least 5;
- adaptation modes:
  - linear/head-only adaptation;
  - optional last-block fine-tune;
- same target splits for all comparators;
- no test-set threshold tuning;
- report mean and CI over seeds.

Comparators should include at least:

- ECG-RAMBA final EMA feature extractor;
- ResNet1D/CNN fine-tune or head-only baseline;
- MiniRocket-only linear baseline if feature extraction is available.

### Outputs

```text
reports/revision/metrics/fewshot_<dataset>_summary.csv
reports/revision/tables/table_fewshot_<dataset>.csv
reports/revision/manifests/fewshot_<dataset>_manifest.json
```

### Claim Gate

Only after completion:

> We evaluated few-shot adaptation under fixed target-domain splits.

Avoid:

> few-shot experiments were added

until the artifacts exist and the final evidence matrix includes them.

## Workstream D: Representation Probe

### Goal

Provide representation-level evidence for architecture analysis without claiming causal disentanglement.

### Technical Design

Create:

```text
scripts/revision/20_representation_probe.py
```

Required model instrumentation:

- expose or hook raw Mamba backbone embeddings;
- expose or hook MiniRocket/Perceiver morphology embeddings;
- expose or hook HRV/rhythm embeddings;
- expose final fused embedding.

Suggested analyses:

- linear probes for rhythm-heavy labels vs morphology-heavy labels;
- CKA similarity between branch embeddings;
- optional UMAP/t-SNE visualizations as qualitative figures;
- subject/fold-aware probe training to avoid leakage.

### Outputs

```text
reports/revision/metrics/representation_probe_summary.json
reports/revision/tables/table_representation_probe.csv
reports/revision/tables/table_representation_cka.csv
reports/revision/figures/representation_umap_<view>.png
reports/revision/manifests/representation_probe_manifest.json
```

### Claim Gate

Safe wording if complete:

> Representation probes suggest branch-specific information differences.

Unsafe wording:

> The model proves morphology-rhythm disentanglement.

## Workstream E: Full HRV Feature Set

### Goal

Implement a true HRV feature schema only if the manuscript must claim RMSSD/SDNN/LF-HF or similar clinical HRV descriptors.

### Technical Reality

This cannot be added to existing final EMA checkpoints. Changing HRV features changes the model input contract and requires retraining all folds.

### Technical Design

Create a versioned schema:

```text
hrv_schema = hrv_full_v2
hrv_dim = new_dim
```

Candidate features:

- time domain: mean RR, SDNN, RMSSD, pNN50, median RR, IQR RR;
- morphology-related amplitude summaries only if validated;
- frequency domain: LF, HF, LF/HF only if the 10-second window supports a defensible interpolation/spectral protocol;
- quality flags: valid beat count, missing-lead count, invalid spectrum flag.

Required outputs:

```text
reports/revision/hrv_full_v2_schema.csv
reports/revision/metrics/hrv_full_v2_validation.json
reports/revision/manifests/hrv_full_v2_manifest.json
```

Then retrain:

```text
notebooks/02a_retrain_best_ema.ipynb
```

with a new protocol name such as:

```text
ema_protocol_hrv_full_v2
```

### Recommendation

Do not run this before resubmission unless HRV claims are central. Current corrected manuscript already avoids full-HRV wording.

## Workstream F: Robustness Beyond MiniRocket

### Goal

Test whether robustness conclusions hold against ResNet1D/CNN and Raw Mamba, not just MiniRocket-only.

### Technical Design

Extend:

```text
scripts/revision/12_robustness_stress.py
```

into a comparator-registry design:

```text
scripts/revision/21_robustness_multicomparator.py
```

Comparators:

- Full ECG-RAMBA;
- MiniRocket-only;
- ResNet1D/CNN;
- Raw Mamba, if Workstream A is complete.

Stress tests should reuse the same six perturbations:

- `snr20db`
- `snr10db`
- `snr5db`
- `random_3_lead_dropout`
- `precordial_dropout`
- `resample_250hz`

### Claim Gate

Even if complete, report by metric and comparator. Do not write:

> ECG-RAMBA is generally more robust.

Write only if supported:

> Under specific stress settings and metrics, ECG-RAMBA shows lower error/calibration degradation than comparator X.

## Execution Order

Recommended order if new evidence is required after the current manuscript/rebuttal package:

1. Keep the current manuscript/rebuttal wording fixed to the regenerated final evidence tables.
2. If transfer claims are needed, implement Workstream B external protocol gate.
3. Only after Workstream B, consider Workstream C few-shot.
4. If architecture interpretation is challenged, implement Workstream D representation probe.
5. Avoid Workstream E unless full HRV claims are unavoidable.
6. Avoid Workstream F unless a reviewer explicitly asks for general robustness against CNN/ResNet.

## Notebook/Script Update Map

| Workstream | Scripts to add/update | Notebooks to update | Final evidence update |
|---|---|---|---|
| Raw Mamba | completed `16_raw_mamba_baseline.py` and `17_paired_full_vs_raw_mamba.py`; included in regenerated final evidence tables | Notebook 04 | `13_final_evidence_matrix.py`, task board, registry |
| External gate | implemented `18_external_protocol_gate.py`; update `03_generate_external_predictions.py` only if a gate issue reveals an exporter defect | Notebook 02 | `13_final_evidence_matrix.py`, claim map |
| Few-shot | add `19_fewshot_adaptation.py` | new optional notebook or Notebook 02 extension | `13_final_evidence_matrix.py` |
| Representation | add `20_representation_probe.py`, minor model hook support | Notebook 06 | `13_final_evidence_matrix.py` |
| Full HRV | update feature extraction/model config/training | Notebook 00/02a/03/04/05/07 | new evidence protocol, not current final EMA |
| Multi-comparator robustness | add `21_robustness_multicomparator.py` | Notebook 05 | `13_final_evidence_matrix.py` |

## Current Recommendation

For the current resubmission, do not open all remaining deferred workstreams unless a reviewer explicitly requires additional evidence. Workstream A is complete and narrows the claims further: Raw Mamba is stronger on discrimination/F1 metrics, while ECG-RAMBA has lower Brier/ECE.

The current response is scientifically defensible because it explicitly acknowledges the stronger ResNet1D/CNN baseline, reports the Raw Mamba tradeoff, and removes unsupported global-superiority, external-transfer, full-HRV, and disentanglement claims.
