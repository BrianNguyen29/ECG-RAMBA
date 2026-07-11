# External / Adaptation Reviewer Evidence Execution Plan - 2026-07-03

Scope:

- Source reviewer letter: `D:\WorkSpace\ECG\ECG-Ramba\Decision letter (Initial Submission.txt`
- Current reviewer discussion: `D:\WorkSpace\ECG\ECG-Ramba\ECG_RAMBA_Reviewer_Discussion.md`
- Current evidence source of truth:
  - `reports/revision/tables/table_final_evidence_matrix.csv`
  - `reports/revision/tables/table_final_safe_wording.csv`
  - `reports/revision/metrics/external_protocol_gate_summary.csv`

This plan is for opening additional external/adaptation evidence while preserving scientific integrity. It must not reintroduce broad zero-shot, SOTA, clinical-safety, disentanglement-proof, or global-superiority claims.

## 1. Reviewer Requirements Mapped To Execution Workstreams

| Reviewer requirement | Current status | New execution needed? | Priority |
|---|---:|---:|---:|
| Calibration / conservative operating point | Complete with downgraded wording | No | Done |
| Fair baselines including CNN/raw Mamba | Complete for MiniRocket, ResNet1D/CNN, Raw Mamba | Optional transformer only if needed | P2 |
| HRV-only/domain sensitivity | Complete with limitation | No unless full HRV claim is needed | Done/P3 |
| Confidence intervals / significance testing | Complete for OOF and paired baseline comparisons | Extend to new external/few-shot work | P0/P1 |
| Q=3 pooling sensitivity | Complete with limitation | No | Done |
| Reproducibility / Algorithm 1 / provenance | Complete for current protocol | Add manifests for new work | P0/P1 |
| Robustness under noise/missing leads/sampling | Complete only Full vs MiniRocket | Multi-comparator robustness if broader robustness is needed | P2 |
| Morphology-rhythm visualization/probing | Complete as conservative audit | Stronger controlled probes only if needed | P2 |
| Few-shot vs zero-shot external scenarios | Complete for PTB-XL, Georgia, and CPSC2021 score calibration | No for current mapped-task scope; add new datasets only after their own passed gate | Done |
| External Georgia/CPSC evidence | Complete with strict mapped-task restrictions | No for current mapped-task scope | Done |

## 2. Completed P0 Workstream: External Few-Shot Sensitivity

Rationale: PTB-XL, Georgia, and CPSC2021 now have passed dataset-specific protocol gates. Few-shot/adaptation evidence has therefore been run as leakage-audited frozen-score calibration for all three mapped external tasks.

Current gate artifacts include:

```text
reports/revision/metrics/external_ptbxl_protocol_gate.json
reports/revision/metrics/external_georgia_protocol_gate.json
reports/revision/metrics/external_cpsc2021_protocol_gate.json
reports/revision/tables/table_external_ptbxl_label_mapping.csv
reports/revision/tables/table_external_georgia_label_mapping.csv
reports/revision/tables/table_external_cpsc2021_label_mapping.csv
reports/revision/tables/table_external_ptbxl_metrics.csv
reports/revision/tables/table_external_georgia_metrics.csv
reports/revision/tables/table_external_cpsc2021_metrics.csv
reports/revision/manifests/external_ptbxl_protocol_gate_manifest.json
reports/revision/manifests/external_georgia_protocol_gate_manifest.json
reports/revision/manifests/external_cpsc2021_protocol_gate_manifest.json
```

Implemented runner:

```text
scripts/revision/19_fewshot_adaptation.py
```

Important limitation:

- The current few-shot runner performs per-class score calibration on frozen external predictions.
- It is not ECG-RAMBA weight fine-tuning.
- Safe wording: "few-shot score-calibration sensitivity under a protocol-gated PTB-XL mapped task."
- Unsafe wording: "few-shot transfer learning", "few-shot ECG-RAMBA adaptation proves external superiority".

### Completed Result

Current final evidence reports:

```text
zero-shot mapped-task PR-AUC = 0.6116
zero-shot mapped-task ROC-AUC = 0.8025
zero-shot mapped-task F1     = 0.4684
10% score-calibrated F1      = 0.6160
10% F1 gain vs zero-shot     = +0.1477
```

Interpretation:

- This is score calibration on frozen ECG-RAMBA external predictions.
- ECG-RAMBA model weights are unchanged.
- Ranking metrics do not improve because score calibration does not change within-class ranking.
- This supports a PTB-XL mapped-task score-calibration sensitivity statement only.

### Reproduction Command

In `notebooks/02_predictions_and_external_eval.ipynb`, after Setup and external gate restore/validation, keep external exporters disabled and set:

```python
RUN_PTBXL_EXPORT = False
RUN_GEORGIA_EXPORT = False
RUN_CPSC2021_EXPORT = False
RUN_EXTERNAL_PROTOCOL_GATE = True
EXTERNAL_GATE_DATASETS = 'ptbxl'
EXTERNAL_GATE_REUSE_EXISTING = True
RUN_FEWSHOT_ADAPTATION = True
FEWSHOT_DATASET = 'ptbxl'
FEWSHOT_FRACTIONS = '0,0.01,0.05,0.10'
FEWSHOT_SEEDS = '42,43,44,45,46'
FEWSHOT_TEST_FRACTION = 0.50
FEWSHOT_N_BOOT = 1000
FEWSHOT_STRICT = True
```

Equivalent CLI:

```bash
python -u scripts/revision/19_fewshot_adaptation.py \
  --dataset ptbxl \
  --fractions 0,0.01,0.05,0.10 \
  --seeds 42,43,44,45,46 \
  --test-fraction 0.50 \
  --threshold 0.5 \
  --n-bins 15 \
  --n-boot 1000 \
  --strict
```

### Expected Outputs

```text
reports/revision/metrics/fewshot_ptbxl_summary.csv
reports/revision/tables/table_fewshot_ptbxl.csv
reports/revision/metrics/fewshot_ptbxl_bootstrap.json
reports/revision/manifests/fewshot_ptbxl_splits.npz
reports/revision/manifests/fewshot_ptbxl_run_manifest.json
reports/revision/logs/fewshot_ptbxl.log
```

### Acceptance Criteria

- Gate JSON has `protocol_gate_passed=true` and `manuscript_ready=true`.
- Few-shot manifest has `status=complete`.
- Split NPZ records fixed test IDs and nested train IDs for all seeds/fractions.
- 0 percent row is treated as no-target-label baseline.
- 1/5/10 percent rows use only adaptation-pool labels.
- Bootstrap CI exists for PR-AUC, ROC-AUC, F1, Brier, and ECE.
- No target test labels are used for model selection or threshold tuning.

### After Running Or Rerunning

1. Rerun `notebooks/07_results_freeze.ipynb`.
2. Confirm final evidence matrix includes few-shot PTB-XL.
3. Update manuscript/rebuttal only with conservative wording.

## 3. P1 Workstream: Georgia External Gate

Rationale: Reviewer pressure on external generalization can be addressed more strongly only if another dataset passes a dataset-specific gate. Georgia currently cannot be cited because the current mapping produced no usable mapped labels.

Known blocker:

- Georgia headers were readable.
- Existing diagnosis codes did not map under the current frozen Chapman/SNOMED taxonomy.
- Unmapped labels must not be coerced to negative labels.

### Implementation Steps

1. Generate a Georgia code inventory:
   - all observed SNOMED/diagnosis codes;
   - counts;
   - examples of records per code;
   - current mapping status.
2. Create a reviewed mapping file:

```text
docs/revision_plan/georgia_label_mapping_review_20260630.csv
```

Required columns:

```text
source_code, source_label, count, mapped_target, mapping_scope, action, rationale, reviewer
```

3. Update `scripts/revision/03_generate_external_predictions.py` to load this reviewed mapping.
4. Refuse records with only unsupported/unmapped labels.
5. Rerun Georgia exporter:

```bash
python -u scripts/revision/03_generate_external_predictions.py \
  --dataset georgia \
  --checkpoint-kind final_ema \
  --batch-size 128 \
  --allow-experimental
```

6. Rerun gate:

```bash
python -u scripts/revision/18_external_protocol_gate.py \
  --dataset georgia \
  --expected-checkpoint-kind final_ema \
  --threshold 0.5 \
  --n-bins 15 \
  --n-boot 1000 \
  --reuse-existing \
  --strict
```

### Expected Outputs

```text
reports/revision/experimental/external/georgia/georgia_full_predictions.npz
reports/revision/experimental/external/georgia/georgia_full_slice_predictions.npz
reports/revision/experimental/external/georgia/georgia_full_prediction_summary.json
reports/revision/experimental/external/georgia/georgia_full_prediction_run_manifest.json
reports/revision/metrics/external_georgia_protocol_gate.json
reports/revision/tables/table_external_georgia_label_mapping.csv
reports/revision/tables/table_external_georgia_metrics.csv
reports/revision/manifests/external_georgia_protocol_gate_manifest.json
```

### Acceptance Criteria

- Gate status is `protocol_gate_passed`.
- Mapping table explicitly lists mapped and unsupported codes.
- No unmapped-only record is counted as a negative sample.
- Bootstrap CI exists.
- Archive/checkpoint/PCA/prediction SHA values are in the manifest.

### Allowed Claim If Passed

"Georgia was evaluated as a reviewed mapped-task external dataset." This still does not support unqualified zero-shot superiority.

## 4. P1 Workstream: CPSC2021 External Gate

Rationale: CPSC2021 is rhythm/AF-focused and directly relevant to external rhythm adaptation, but it requires annotation-aligned windows.

Known blocker:

- Prior runs reached WFDB loading but failed or were slow at annotation/window processing.
- Annotation failures must not be converted into negative labels.

### Implementation Steps

1. Harden annotation parsing in `scripts/revision/03_generate_external_predictions.py`.
2. Add a separate CPSC annotation audit output:

```text
reports/revision/tables/table_cpsc2021_annotation_audit.csv
```

3. Define a fixed window protocol:
   - 10-second windows;
   - AF/AFL positive if majority overlap with AF/AFL interval;
   - normal negative only if a recognized normal rhythm interval dominates;
   - ambiguous or transition windows reported separately.
4. Rerun CPSC exporter:

```bash
python -u scripts/revision/03_generate_external_predictions.py \
  --dataset cpsc2021 \
  --checkpoint-kind final_ema \
  --batch-size 128 \
  --allow-experimental
```

5. Rerun gate:

```bash
python -u scripts/revision/18_external_protocol_gate.py \
  --dataset cpsc2021 \
  --expected-checkpoint-kind final_ema \
  --threshold 0.5 \
  --n-bins 15 \
  --n-boot 1000 \
  --reuse-existing \
  --strict
```

### Expected Outputs

```text
reports/revision/experimental/external/cpsc2021/cpsc2021_full_predictions.npz
reports/revision/experimental/external/cpsc2021/cpsc2021_full_slice_predictions.npz
reports/revision/experimental/external/cpsc2021/cpsc2021_full_prediction_summary.json
reports/revision/experimental/external/cpsc2021/cpsc2021_full_prediction_run_manifest.json
reports/revision/metrics/external_cpsc2021_protocol_gate.json
reports/revision/tables/table_external_cpsc2021_label_mapping.csv
reports/revision/tables/table_external_cpsc2021_metrics.csv
reports/revision/manifests/external_cpsc2021_protocol_gate_manifest.json
```

### Acceptance Criteria

- Positive and negative windows are both present.
- Loaded window count matches prediction rows.
- Annotation failures are counted and reported.
- No parse failure is used as a negative label.
- Bootstrap CI exists.

### Allowed Claim If Passed

"CPSC2021 was evaluated as an annotation-aligned AF/AFL mapped-task dataset." This still does not support broad zero-shot superiority.

## 5. P2 Workstream: Few-Shot Beyond PTB-XL

Run this only after Georgia or CPSC2021 gate passes.

Protocol:

```bash
python -u scripts/revision/19_fewshot_adaptation.py \
  --dataset <georgia_or_cpsc2021> \
  --fractions 0,0.01,0.05,0.10 \
  --seeds 42,43,44,45,46 \
  --test-fraction 0.50 \
  --threshold 0.5 \
  --n-bins 15 \
  --n-boot 1000 \
  --strict
```

Acceptance criteria are identical to PTB-XL few-shot, with a dataset-specific gate precondition.

## 6. P2 Workstream: Multi-Comparator Robustness

Current evidence supports only Full ECG-RAMBA vs MiniRocket-only robustness. Reviewer 2 asked about robustness of the proposed framework; the current manuscript can answer metric-specific robustness only. To broaden this, generate ResNet1D/CNN and Raw Mamba stress predictions, then run the existing aggregator.

Existing aggregator:

```text
scripts/revision/21_robustness_multicomparator.py
```

Current gap:

- It aggregates existing predictions.
- It does not generate ResNet/Raw-Mamba stress predictions.

Required new work:

1. Add stress prediction generation for ResNet1D/CNN.
2. Add stress prediction generation for Raw Mamba.
3. Reuse the same six stress definitions:
   - `snr20db`
   - `snr10db`
   - `snr5db`
   - `random_3_lead_dropout`
   - `precordial_dropout`
   - `resample_250hz`
4. Run:

```bash
python -u scripts/revision/21_robustness_multicomparator.py \
  --comparators full,minirocket,resnet,raw_mamba \
  --stress-tests snr20db,snr10db,snr5db,random_3_lead_dropout,precordial_dropout,resample_250hz \
  --threshold 0.5 \
  --n-bins 15 \
  --n-boot 1000 \
  --reuse-metric-cache
```

Expected outputs:

```text
reports/revision/metrics/robustness_multicomparator_summary.csv
reports/revision/metrics/robustness_multicomparator_pairwise.json
reports/revision/tables/table_robustness_multicomparator.csv
reports/revision/manifests/robustness_multicomparator_manifest.json
```

Allowed claim if complete:

- Metric-specific robustness against each comparator.
- No broad robustness superiority unless all paired degradation CIs support it, which is unlikely.

## 7. P2 Workstream: Transformer / Foundation ECG Comparator

Only implement if reviewer pressure specifically requires transformer/foundation baseline.

Minimum fair protocol:

- same Chapman folds;
- same raw ECG cache;
- same slice length/stride;
- no threshold tuning on validation/test;
- same `threshold=0.5`;
- same `Q=3` record aggregation;
- bootstrap CI and paired comparison.

Suggested runner:

```text
scripts/revision/24_transformer_ecg_baseline.py
```

Expected outputs:

```text
reports/revision/predictions/transformer_ecg_oof_predictions.npz
reports/revision/metrics/transformer_ecg_baseline_summary.json
reports/revision/tables/table_transformer_ecg_class_metrics.csv
reports/revision/manifests/transformer_ecg_baseline_manifest.json
reports/revision/tables/table_paired_full_vs_transformer.csv
```

Important: This baseline is unlikely to restore ECG-RAMBA broad superiority because ResNet1D/CNN and Raw Mamba already outperform it on several key in-domain metrics.

## 8. P3 Workstream: Full HRV Feature Set

This is a retrain-level change and should not be opened unless full HRV semantics are essential.

Required:

1. Implement true HRV feature schema with RMSSD/SDNN and defensible frequency-domain policy.
2. Retrain five folds under a new protocol.
3. Regenerate OOF, calibration, baselines, HRV/domain, robustness as needed.

Do not mix new HRV checkpoints with current final EMA evidence.

## 9. Recommended Execution Order

1. Treat PTB-XL, Georgia, and CPSC2021 few-shot score calibration as complete and keep conservative wording.
2. If adding any new external dataset, repeat reviewed mapping/window protocol, prediction export, protocol gate, then score calibration.
3. If robustness pressure remains, implement or run stress prediction generation for ResNet/Raw Mamba and then run multi-comparator aggregation.
4. Only if directly requested, run transformer/foundation baseline.
5. Optional frozen fixed-seed ROCKET-family transform MLP-head sensitivity may be run for head-capacity questions; it does not isolate determinism from regularization.
6. Avoid full HRV retraining unless manuscript needs true full-HRV claims.

## 10. Compute Guidance

| Workstream | GPU needed? | Recommended runtime |
|---|---:|---|
| PTB-XL few-shot score calibration | No | CPU/T4 is enough; CPU-bound bootstrap |
| Georgia/CPSC export inference | Yes if predictions must be generated | A100 High-RAM preferred |
| External gate only | No | CPU/T4 enough |
| Few-shot after gate | No | CPU/T4 enough; bootstrap can take time |
| Multi-comparator robustness generation | Yes | A100 High-RAM preferred |
| Multi-comparator robustness aggregation only | No | CPU/T4 enough if predictions exist |
| Transformer baseline training | Yes | A100 High-RAM |
| Full HRV retraining | Yes | A100 High-RAM |

## 11. Immediate Next Command

For the current project state, PTB-XL few-shot is already complete. The next scientifically valid commands depend on the desired reviewer gap:

```bash
# External breadth: first attempt a reviewed Georgia mapped-task gate.
python -u scripts/revision/03_generate_external_predictions.py \
  --dataset georgia \
  --checkpoint-kind final_ema \
  --batch-size 128 \
  --allow-experimental
```

or, for robustness breadth after comparator stress predictions exist:

```bash
python -u scripts/revision/21_robustness_multicomparator.py \
  --comparators full,minirocket,resnet,raw_mamba \
  --stress-tests snr20db,snr10db,snr5db,random_3_lead_dropout,precordial_dropout,resample_250hz \
  --threshold 0.5 \
  --n-bins 15 \
  --n-boot 1000 \
  --reuse-metric-cache
```

After any new evidence, rerun `notebooks/07_results_freeze.ipynb` and use the resulting final evidence tables as the only manuscript/rebuttal source of truth.
