# Post-P0 Full Execution Roadmap

Date: 2026-06-29

This roadmap translates the remaining reviewer-driven workstreams into an executable engineering plan. It assumes the current manuscript/rebuttal package is already claim-bounded and P0 is complete:

- IEEE PDF compiles.
- Algorithm 1 documents fold-aware PCA/training/inference/Q=3 aggregation.
- Figures/captions/tables were visually checked.
- Positive forbidden-claim scan passed.

The current manuscript should remain stable unless a workstream below is intentionally opened.

## Current Claim Boundary

The source-of-truth evidence supports:

- Frozen Chapman OOF as the primary manuscript-ready evidence.
- PTB-XL, Georgia, and CPSC2021 only as protocol-gated mapped-task external evaluations.
- Fair-baseline results as comparator-specific and metric-specific.
- HRV as useful but domain-sensitive.
- Robustness only as metric-specific against MiniRocket-only.
- Q=3 as a frozen/sensitivity-tested operating point, not a global optimum.

The evidence does not support:

- ECG-RAMBA as the leading in-domain model or broadly superior model.
- Unqualified cross-dataset advantage.
- Proven morphology-rhythm disentanglement.
- HRV invariance or full HRV feature semantics.
- General robustness superiority.
- Few-shot results.

## P0: Submission Packaging - Complete

### Purpose

Keep the resubmission internally consistent and prevent stale claims from reappearing in abstract, captions, tables, response letter, or appendix.

### Current Artifacts

```text
docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/main.tex
docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/build/main.pdf
docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/response_to_reviewers_revised_20260622.md
ECG-RAMBA/docs/revision_plan/p0_pdf_compile_claim_scan_20260629.md
```

### If Any Text Changes

Rerun:

1. PDF compile from `main.tex`.
2. Render all pages to PNG/contact sheet.
3. Active graphics existence check.
4. Missing label/reference check.
5. Positive forbidden-claim scan on `main.tex` and response letter.
6. Manual review of figure/table captions.

### Acceptance Criteria

- PDF compiles with no LaTeX errors.
- Undefined references/citations: 0.
- Positive forbidden-claim scan: pass.
- Algorithm 1 still present and referenced.
- Captions do not imply benchmark-leading performance, unqualified cross-dataset advantage, strict mechanism proof, or general robustness.

## P1: Georgia External Protocol Gate

### Trigger

Open only if additional external mapped-task evidence is needed beyond PTB-XL.

### Current State

Georgia is deferred. The current exporter found readable headers, but available diagnostic codes did not map to the frozen 27-class Chapman/SNOMED taxonomy under the current reviewed mapping. The correct behavior is to defer, not coerce unmapped records to negative labels.

### Existing Starting Points

```text
scripts/revision/03_generate_external_predictions.py
scripts/revision/18_external_protocol_gate.py
notebooks/02_predictions_and_external_eval.ipynb
```

### Required Implementation

1. Build a reviewed Georgia label mapping table:
   - source SNOMED code;
   - human-readable label;
   - mapped frozen class or external mapped-task class;
   - mapping rationale;
   - action for unmapped labels.
2. Add the mapping to the external exporter without silent negative conversion.
3. Report:
   - total headers;
   - loaded records;
   - skipped without mapped label;
   - top unmapped codes;
   - records with multiple mapped labels.
4. Generate predictions only for records with at least one reviewed mapped label.
5. Run external protocol gate with bootstrap CI and cache key.
6. Publish artifacts through `artifact_mirror.py`.

### Required Outputs

```text
reports/revision/experimental/external/georgia/georgia_full_predictions.npz
reports/revision/experimental/external/georgia/georgia_full_slice_predictions.npz
reports/revision/experimental/external/georgia/georgia_full_prediction_summary.json
reports/revision/experimental/external/georgia/georgia_full_prediction_run_manifest.json
reports/revision/metrics/external_georgia_protocol_gate.json
reports/revision/tables/table_external_georgia_label_mapping.csv
reports/revision/tables/table_external_georgia_metrics.csv
reports/revision/manifests/external_georgia_protocol_gate_manifest.json
reports/revision/logs/georgia_generate_predictions.log
reports/revision/logs/external_protocol_gate.log
```

### Acceptance Criteria

- Gate status: `protocol_gate_passed`.
- Artifact SHA256 and archive fingerprint recorded.
- Fold PCA manifest complete for all five folds.
- Checkpoint fingerprints match frozen OOF final EMA.
- Label table documents mapped and unmapped classes.
- No record is treated as negative solely because annotation is missing/unmapped.

### Safe Claim If Passed

Georgia can be cited only as a dataset-specific mapped-task external evaluation. It still does not support an unqualified cross-dataset advantage.

## P1: CPSC2021 External Protocol Gate

### Trigger

Open only if AF/AFL episode/window external evidence is needed.

### Current State

CPSC2021 is deferred. The correct target is annotation-aligned window evaluation. Annotation parsing must not turn annotation errors into negative windows.

### Existing Starting Points

```text
scripts/revision/03_generate_external_predictions.py
scripts/revision/18_external_protocol_gate.py
notebooks/02_predictions_and_external_eval.ipynb
```

### Required Implementation

1. Harden CPSC annotation parsing:
   - support the archive's annotation format;
   - handle uint8/string parsing safely;
   - record parse failures separately.
2. Define a window protocol:
   - window length and stride;
   - positive AF/AFL overlap rule;
   - normal/negative window rule;
   - exclusion windows around ambiguous boundaries.
3. Emit a load summary:
   - headers checked;
   - signal records skipped;
   - annotation records skipped;
   - loaded windows;
   - positive/negative counts;
   - parse examples.
4. Generate predictions for annotation-aligned windows.
5. Run external protocol gate and bootstrap CI.
6. Publish artifacts through `artifact_mirror.py`.

### Required Outputs

```text
reports/revision/experimental/external/cpsc2021/cpsc2021_full_predictions.npz
reports/revision/experimental/external/cpsc2021/cpsc2021_full_slice_predictions.npz
reports/revision/experimental/external/cpsc2021/cpsc2021_full_prediction_summary.json
reports/revision/experimental/external/cpsc2021/cpsc2021_full_prediction_run_manifest.json
reports/revision/metrics/external_cpsc2021_protocol_gate.json
reports/revision/tables/table_external_cpsc2021_label_mapping.csv
reports/revision/tables/table_external_cpsc2021_metrics.csv
reports/revision/manifests/external_cpsc2021_protocol_gate_manifest.json
reports/revision/logs/cpsc2021_generate_predictions.log
reports/revision/logs/external_protocol_gate.log
```

### Acceptance Criteria

- Gate status: `protocol_gate_passed`.
- Both positive and negative annotation windows exist.
- Loaded window count equals prediction rows.
- Annotation skip counts are explicit.
- No annotation failure is converted to a negative label.

### Safe Claim If Passed

CPSC2021 can be cited only as annotation-aligned mapped-task evidence. It still does not support broad cross-dataset advantage.

## P2: Few-Shot Adaptation

### Trigger

Open only after at least one external dataset gate passes. Do not run few-shot on an ungated dataset.

### Required New Runner

```text
scripts/revision/19_fewshot_adaptation.py
```

### Current Implementation Status

`scripts/revision/19_fewshot_adaptation.py` is implemented as a gated, conservative score-calibration runner on frozen external predictions. Notebook 02 now exposes it as an optional disabled-by-default cell after the external protocol gate. The runner refuses to produce completion evidence unless the dataset-specific external gate has passed. For each seed, the target-domain test split is frozen before any labeled fraction is selected, and the 1/5/10 percent training subsets are nested prefixes of the remaining target-domain pool. This is suitable for a leakage-audited few-shot sensitivity analysis, but it is not model-weight fine-tuning and must not be described as full few-shot transfer.

### Protocol

1. Load a protocol-gated external dataset.
2. Freeze target-domain test split before selecting few-shot labels.
3. Use repeated seeds and fixed labeled fractions:
   - 0 percent baseline;
   - 1 percent;
   - 5 percent;
   - 10 percent.
4. Current implemented mode:
   - score calibration on frozen external predictions.
5. Optional future modes, only if reviewer pressure justifies the compute:
   - head-only adaptation;
   - optional last-block fine-tune;
   - optional full fine-tune only if compute allows.
6. Use the same splits for all comparators:
   - ECG-RAMBA;
   - ResNet1D/CNN;
   - Raw Mamba;
   - MiniRocket-only if features are available.
7. Do not tune threshold on test labels.
8. Report record-level bootstrap CI and seed variability.

### Required Outputs

```text
reports/revision/metrics/fewshot_<dataset>_summary.csv
reports/revision/metrics/fewshot_<dataset>_bootstrap.json
reports/revision/tables/table_fewshot_<dataset>.csv
reports/revision/manifests/fewshot_<dataset>_splits.npz
reports/revision/manifests/fewshot_<dataset>_run_manifest.json
reports/revision/logs/fewshot_<dataset>.log
```

### Acceptance Criteria

- Dataset gate passed before few-shot run.
- Split manifest proves no test leakage.
- Same splits/seeds across comparators.
- Results include CI and failure/deferred status if any comparator cannot run.

### Safe Claim If Complete

Few-shot adaptation can be reported as a sensitivity analysis under a gated dataset protocol. Do not claim general transfer superiority unless all paired comparisons support it.

## P2: Representation Probe / UMAP / CKA

### Trigger

Open only if the rebuttal needs stronger evidence for architecture interpretation.

### Required New Runner

```text
scripts/revision/22_extract_representations.py
scripts/revision/20_representation_probe.py
```

### Current Implementation Status

`scripts/revision/22_extract_representations.py` is implemented as a checkpoint-fingerprinted embedding extractor for the frozen final EMA model. It reuses the existing OOF data/fold/PCA/checkpoint contract, writes fold-level caches for Colab resume, validates the frozen OOF checksum/checkpoint kind, and emits a record-level embedding NPZ plus manifest.

`scripts/revision/20_representation_probe.py` is implemented as an embedding-artifact consumer. It validates the embedding NPZ, runs fold-safe linear probes, computes branch CKA, and writes UMAP/PCA figures. `notebooks/06_pooling_and_representation.ipynb` now wires both steps behind explicit flags.

This workstream is still not manuscript evidence until the extractor has produced all five fold caches/final embedding NPZ and the probe runner has completed. Even then, the safe claim remains suggestive branch-specific information only.

### Required Instrumentation

Expose or hook:

- MiniRocket/morphology branch embedding;
- HRV/rhythm descriptor vector;
- Mamba contextual embedding;
- final fused embedding.

### Analyses

1. Fold-safe linear probes:
   - morphology-heavy labels from morphology/fused/context embeddings;
   - rhythm-heavy labels from rhythm/fused/context embeddings.
2. CKA or similarity analysis between branches.
3. UMAP/t-SNE visualizations as qualitative figures only.
4. Subject/fold-aware train/test split for probes.

### Required Outputs

```text
reports/revision/predictions/representation_embeddings_final_ema.npz
reports/revision/manifests/representation_embedding_manifest.json
reports/revision/metrics/representation_probe_summary.json
reports/revision/tables/table_representation_probe.csv
reports/revision/tables/table_representation_cka.csv
reports/revision/figures/representation_<umap_or_pca>_<view>.png
reports/revision/manifests/representation_probe_manifest.json
reports/revision/logs/representation_probe.log
```

### Acceptance Criteria

- All five representation fold caches are generated or restored.
- Final embedding NPZ has matching `record_id`, `fold_id`, `class_names`, and `y_true` with frozen final EMA OOF.
- Embedding manifest includes OOF/freeze/checkpoint SHA256 values and dataset record-order fingerprint.
- Probe splits are fold-safe.
- UMAP is not used as quantitative proof.
- Summary wording explicitly blocks strict mechanism-proof language.

### Safe Claim If Complete

Representation probes suggest branch-specific information differences. They do not prove strict disentanglement.

## P3: Full HRV Feature Set

### Trigger

Open only if the manuscript must claim real RMSSD, SDNN, LF/HF, or full HRV semantics.

### Technical Constraint

This is a retrain-level change. The current final EMA checkpoints are compatible with the existing HRV36 schema that contains reserved zero-filled slots. Full HRV features cannot be retrofitted into current checkpoints.

### Required Implementation

1. Define a versioned schema:

```text
hrv_full_v2
```

2. Implement feature extraction with quality flags:
   - time-domain features such as mean RR, SDNN, RMSSD, pNN50;
   - frequency-domain LF/HF only if the short ECG window supports a defensible spectral protocol;
   - invalid-spectrum and valid-beat-count flags.
3. Update config and model input dimension.
4. Retrain all five folds under a new protocol name.
5. Regenerate:
   - OOF;
   - calibration;
   - baselines if model comparisons are affected;
   - final evidence matrix.

### Required Outputs

```text
reports/revision/hrv_full_v2_schema.csv
reports/revision/metrics/hrv_full_v2_validation.json
reports/revision/manifests/hrv_full_v2_manifest.json
reports/revision/manifests/oof_<new_protocol>_freeze_manifest.json
reports/revision/metrics/calibration_ci_oof_<new_protocol>_predictions.json
```

### Acceptance Criteria

- All five folds retrained.
- New HRV schema is in manifest and freeze contract.
- No old HRV36 artifacts are mixed with the new protocol.

### Safe Claim If Complete

Only then can the manuscript claim a true HRV feature set. Otherwise keep the current HRV36 restriction.

## P3: Robustness Beyond MiniRocket

### Trigger

Open only if the authors want robustness claims beyond MiniRocket-only.

### Current Constraint

`scripts/revision/12_robustness_stress.py` is designed for Full ECG-RAMBA vs MiniRocket-only. ResNet1D/CNN and Raw Mamba stress comparisons are not yet implemented.

### Required New Runner

```text
scripts/revision/21_robustness_multicomparator.py
```

or extend `12_robustness_stress.py` into a comparator registry.

### Current Implementation Status

`scripts/revision/21_robustness_multicomparator.py` is implemented as a low-memory aggregation/gating runner and is now exposed in Notebook 05 after the Full-vs-MiniRocket robustness ledger. It validates existing clean/stressed prediction artifacts for each comparator, records missing comparator-stress artifacts as blocked rows, and computes metric-specific paired degradation CIs when inputs are complete. It does not generate ResNet1D/CNN or Raw Mamba stress predictions; those prediction-generation steps remain required before broad multi-comparator robustness evidence exists.

### Comparators

- Full ECG-RAMBA;
- MiniRocket-only;
- ResNet1D/CNN;
- Raw Mamba.

### Stress Tests

Reuse the same stress definitions:

- `snr20db`
- `snr10db`
- `snr5db`
- `random_3_lead_dropout`
- `precordial_dropout`
- `resample_250hz`

### Required Outputs

```text
reports/revision/metrics/robustness_multicomparator_summary.csv
reports/revision/metrics/robustness_multicomparator_pairwise.json
reports/revision/tables/table_robustness_multicomparator.csv
reports/revision/manifests/robustness_multicomparator_manifest.json
reports/revision/logs/robustness_multicomparator.log
```

### Acceptance Criteria

- Clean and stressed predictions exist for every comparator.
- Same `record_id`, `fold_id`, `class_names`, and `y_true`.
- Paired degradation CI is computed metric-by-metric.
- No blanket robustness wording is emitted.

### Safe Claim If Complete

Report only metric-specific and comparator-specific robustness behavior.

## Recommended Execution Order

1. Do not modify the current manuscript unless a new workstream is opened.
2. If manuscript text changes, rerun P0 compile/render/scan immediately.
3. If additional external evidence is required, implement Georgia and/or CPSC2021 gates first.
4. Only after an external gate passes, implement few-shot on that gated dataset.
5. If architecture interpretation is challenged, implement representation probe.
6. Avoid full HRV retraining unless HRV-feature claims are essential.
7. Avoid broad robustness expansion unless reviewers explicitly ask for multi-comparator robustness.

## Compute Guidance

| Workstream | Recommended runtime | Notes |
|---|---|---|
| P0 compile/scan | CPU/local | No GPU needed. |
| Georgia/CPSC external inference | A100 High-RAM preferred | Mamba runtime required; caches should be reused. |
| External gate aggregation | CPU or T4 | If predictions already exist, GPU is not needed. |
| Few-shot | A100 High-RAM | Use small pilot first; repeated seeds can be expensive. |
| Representation probe | A100 High-RAM for embedding extraction; CPU for UMAP/CKA | Cache embeddings before analysis. |
| Full HRV retrain | A100 High-RAM | Full 5-fold retraining and all downstream evidence regeneration. |
| Multi-comparator robustness | A100 High-RAM | Use prediction reuse and per-stress resume to avoid Colab disconnect losses. |
