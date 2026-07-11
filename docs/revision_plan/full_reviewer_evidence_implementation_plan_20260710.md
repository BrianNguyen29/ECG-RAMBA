# Full Reviewer Evidence Implementation Plan

Date: 2026-07-10

## Objective

Close the remaining Reviewer 1 and Reviewer 2 evidence gaps without mixing incompatible protocols or converting exploratory results into broad performance claims. Every new result must be generated from a versioned runner, have reusable caches/checkpoints, write a manifest and log, pass an explicit readiness gate, and be ingested by Notebook 07 before it can enter the manuscript or response letter.

## Critical Findings From The Pre-Implementation Audit

1. The active local `reports/revision` tree is stale relative to the Drive mirror. The authoritative Drive OOF SHA256 is `99c3dea401e9ca9c21b4735e6a2b2adcf3f3977f3ac36f88eb57d954ceec6162`; the active local tree currently contains the older `375e5f...` container. All work must start with verified mirror restore and checksum guards.
2. The current few-shot score-calibration v1 script splits prediction rows, not patients/source records. This is unsafe for repeated observations. CPSC2021 window IDs such as `data_10_1:0:5000` show that windows from one source ECG can enter both adaptation and test sets. PTB-XL predictions also omit `patient_id`, so patient disjointness cannot currently be proved. Existing few-shot v1 numbers are provisional until a group-safe v2 rerun.
3. The manuscript states that fold-aware PCA retains more than 99% variance, but the five-fold manifest reports 97.4729-97.4774%. The manuscript and response must be corrected before final compilation.
4. The current frozen-transform morphology MLP changes only the learnable head over fixed-seed ROCKET-family MAX+PPV features. It tests head capacity, not learnability of the random-convolution kernels and not deterministic-versus-learned morphology by itself.
5. Existing representation UMAP figures are colored by fold. They audit fold/domain structure but do not visualize morphology-versus-rhythm label organization as requested by Reviewer 2.
6. The evaluated `MiniRocketNative` branch is not canonical MiniRocket. The archived training code and current `src/features.py` both use 10,000 fixed-seed random `{-1,0,1}` kernels, Gaussian biases, and concatenate MAX+PPV into 20,000 outputs. Canonical MiniRocket uses 84 fixed kernels, training-derived quantile biases, and PPV features. This identity mismatch must be resolved in wording before any new morphology experiment is interpreted.

## Non-Negotiable Evidence Contract

- Canonical source model: five Chapman subject-aware folds, `final_ema`, fixed threshold 0.5, frozen class order, and versioned aggregation.
- No target labels may influence zero-shot predictions, mappings, thresholds, model selection, or preprocessing.
- Any adaptation analysis must use patient/source-record-disjoint adaptation, validation, and test groups.
- Bootstrap resampling must use the independent sampling unit: subject for Chapman/PTB-XL when repeated records exist, source ECG record for CPSC2021 windows, and documented record/patient unit for Georgia.
- All model comparisons use exactly the same evaluable records/windows and label mapping.
- Reviewer-minimal/screening outputs and canonical 1000-bootstrap outputs must have different names and may not be silently merged.
- Notebook 07 is the only gateway from experimental artifacts to final manuscript evidence.
- An artifact cannot be called MiniRocket evidence unless its transform contract matches the cited implementation; otherwise use the exact custom-transform name.

## Phase 0 - Restore, Version, And Gate Provenance

### Work

1. Restore the Drive mirror into the active repo using `artifact_mirror.py restore --replace-mismatched`.
2. Verify OOF, freeze, calibration, baseline, representation, pooling, external, and few-shot hashes before any runner starts.
3. Commit and push current notebook/generator changes before Colab execution. A clean Colab clone must contain every required runner token.
4. Add a preflight report that records local path, mirror path, SHA256, size, and selected authority for every required input.
5. Update `create_revision_notebooks.py` or formally retire it as a generator; it must not regenerate older notebook cells over newer manual changes.

### Outputs

- `reports/revision/manifests/reviewer_completion_input_contract.json`
- `reports/revision/tables/table_reviewer_completion_input_contract.csv`
- `reports/revision/logs/reviewer_completion_preflight.log`

### Gate

- OOF prediction SHA equals the current freeze/calibration contract.
- No required artifact is selected from a lower-priority stale source.
- Git commit and working-tree state are recorded in every new manifest.
- The evaluated morphology transform identity is recorded and no checkpoint is paired with a different transform implementation.

### Hardware

CPU only.

## Phase 0A - Morphology Transform Identity Gate

### Recommended decision for the current checkpoint

Preserve the evaluated final-EMA checkpoint and correct the terminology. Describe the branch as a **fixed-seed ROCKET-family random-convolution transform** with:

- 10,000 length-9 random convolution kernels with weights sampled from `{-1,0,1}` using seed 42.
- Dyadic dilations determined from the slice length.
- Fixed Gaussian random biases.
- MAX and PPV statistics per kernel, producing 20,000 outputs.
- Fold-train-only PCA to 3,072 dimensions.

Do not call this canonical MiniRocket or claim that the implementation inherits MiniRocket's exact determinism/efficiency properties. The archived training notebook confirms that the final checkpoints used this same custom transform, so changing the transform now would invalidate checkpoint compatibility.

### Alternative requiring full retraining

If the paper must retain the MiniRocket name, implement the official multivariate MiniRocket contract, fit training-derived biases independently inside each OOF fold, rebuild all caches/PCA objects, and retrain all five ECG-RAMBA folds and every dependent ablation/external export. This creates a new evidence package and may not be mixed with the current final-EMA results.

### Outputs

- `tables/table_morphology_transform_contract.csv` and `.tex`.
- `manifests/morphology_transform_identity_gate.json`.
- A readiness gate with `evaluated_transform_name`, source hash, seed, kernel/bias/pooling contract, output dimension, and checkpoint compatibility.

### Gate

- Manuscript, response, captions, tables, and code use one accurate transform name.
- The 10,000-kernel versus 20,000-output distinction is explicit.
- Algorithm 1 says "apply the fixed transform" rather than "fit MiniRocket" for the evaluated checkpoint.

## Phase 1 - R1-C1 And R1-C5: Calibration, Reliability, And Confidence Intervals

### Statistical corrections

1. Confirm the independent bootstrap unit. If subjects have repeated Chapman records, add `subject_id` to the frozen prediction contract and rerun calibration and paired CIs with subject-cluster bootstrap.
2. Keep the existing micro-flattened reliability curve, but label its scope explicitly: all record-class probability pairs, 15 bins.
3. Add class-level calibration evidence because a micro curve can be dominated by common negatives in a 27-label problem.
4. Generate paired delta CIs for PR-AUC, ROC-AUC, F1, Brier, and ECE for each completed fair comparator.
5. For external mapped tasks, use patient/source-record cluster bootstrap and report the task unit in the table caption.

### Implementation

Extend `04_calibration_ci.py` and the paired comparison helpers, or add a single presentation generator:

- Implemented as `scripts/revision/29_reviewer_presentation_assets.py`.

The generator reads only validated artifacts and writes:

- `figures/figure_calibration_audit.png`: micro reliability plus class-ECE distribution or selected class panels.
- `tables/table_calibration_ci_compact.csv` and `.tex`.
- `tables/table_paired_baseline_ci_compact.csv` and `.tex`.
- `tables/table_external_ci_compact.csv` and `.tex`.
- `manifests/reviewer_presentation_assets_manifest.json`.

### Manuscript integration

- Main text: ECE/Brier point estimates and concise 95% CI statement.
- Appendix/supplement: reliability figure and complete paired-delta table.
- Remove all clinical safety interpretation; calibration is operating-point evidence only.

Reliability diagrams are standard diagnostic tools for confidence calibration; ECE must be interpreted together with the binning scope rather than as a standalone safety measure.

### Gate

- Every CI row reports point estimate, lower bound, upper bound, valid bootstrap count, seed, and cluster unit.
- Reliability figure source SHA matches canonical predictions.
- Manuscript values are generated from the compact tables, not typed independently.

### Hardware

CPU; bootstrap can be long but requires no GPU.

## Phase 2 - R1-C6 And R1-C7: Pooling, PCA, And Training Provenance

### Q=3 pooling

Current Chapman sensitivity shows Q=3 is not the best point estimate for all metrics. Q=3 ranks fifth for F1, third for PR-AUC, fourth for ROC-AUC, and is therefore only a frozen operating point.

Implement:

1. Compact table for mean, Q=2, Q=3, Q=4, Q=8, and max pooling.
2. Paired bootstrap deltas between Q=3 and each alternative using the same records.
3. Post-hoc sensitivity on PTB-XL, Georgia, and CPSC2021 from existing slice predictions. This is a sensitivity audit, not a new rule-selection exercise.
4. Keep CPSC2021 in a separate mapped-window panel.

Suggested runner:

- `30_pooling_sensitivity_external.py`

Outputs:

- `metrics/pooling_sensitivity_external.csv`
- `tables/table_pooling_sensitivity_across_datasets.csv`
- `metrics/pooling_q3_paired_bootstrap.json`
- `manifests/pooling_sensitivity_external_manifest.json`

### PCA and model provenance

Generate an appendix table directly from the fold PCA manifest and training config:

- Fold 1: 97.4754%
- Fold 2: 97.4773%
- Fold 3: 97.4774%
- Fold 4: 97.4729%
- Fold 5: 97.4730%

Include:

- 3,072 PCA components from 20,000 raw morphology features.
- 10,000 fixed-seed random kernels, each contributing MAX and PPV; do not call the 20,000 outputs 10,000 MiniRocket features.
- Training-fold-only PCA fitting and fold-specific validation/test transformation.
- Instance-wise signal normalization.
- BCE warm-up for 8 epochs, ASL with gamma-minus 2.5 and gamma-plus 0.
- AdamW, peak LR `9e-4`, minimum LR `1e-6`, weight decay `0.05`.
- EMA decay `0.999`, model dimension 384, and 16 Mamba layers.
- Any dropout/norm values must come from the run/checkpoint manifest, not memory or prose.

Outputs:

- `tables/table_fold_pca_provenance.csv` and `.tex`.
- `tables/table_training_configuration.csv` and `.tex`.

### Gate

- Remove the incorrect “more than 99%” statement everywhere.
- No claim that Q=3 is optimal.
- Every training value has a config/manifest source path and SHA.

### Hardware

CPU only; no inference is required when slice predictions are reusable.

## Phase 3 - R1-C2: Determinism, Head Capacity, And Learned Morphology

### Experiment A: Head-capacity sensitivity

Run the existing `26_hybrid_morphology_baseline.py` and `27_paired_full_vs_hybrid_morphology.py`, but label it accurately:

- Frozen custom ROCKET-family transform plus learnable MLP head.
- Tests nonlinear head capacity and optimizer/loss sensitivity.
- Does not make MiniRocket kernels learnable.

Add a matched neural linear-head control using the same BCE, class weights, optimizer, standardization, epochs, and seed as the MLP. This separates MLP nonlinearity from a change in training objective relative to the existing scikit-learn logistic baseline. Rename output files and manifests so they do not imply that the kernels themselves are partially learnable.

### Experiment B: Direct learned morphology encoder

Add a morphology-only raw-ECG CNN baseline under the same OOF slices, folds, threshold, and Q=3 aggregation:

- Not yet implemented. Reserve a new runner ID after `36_build_marked_manuscript.py`
  for a matched learned morphology-only Conv1D control if this mechanism-level
  experiment is required.
- Small Conv1D morphology encoder, global pooling, linear 27-label head.
- Fixed architecture and epochs; no validation-based checkpoint/threshold selection.
- Same slice and record IDs as MiniRocket-only.

This is the direct deterministic-feature versus learned-morphology comparison requested by Reviewer 1. ResNet remains a broader architecture comparator and cannot by itself isolate this mechanism.

### Required artifacts

- OOF and slice predictions for neural-linear, frozen-transform morphology MLP, and learned morphology CNN.
- Per-fold training logs and fold caches.
- Model/checkpoint config and seeds.
- Summary/class/fold tables.
- Paired bootstrap tables against MiniRocket-only and Full ECG-RAMBA.
- Manifests with raw feature/cache/checkpoint hashes.

### Allowed conclusion

Only morphology-encoder/head sensitivity. Do not claim a causal benefit of determinism unless all controlled contrasts consistently support it.

### Hardware

A100 recommended. MiniRocket head training is lighter; learned raw-ECG CNN requires GPU.

## Phase 4 - R1-C4: Compact Transformer Baseline

### Audit and hardening before run

The current `24_transformer_ecg_baseline.py` is a compact patch Transformer, not a pretrained foundation model. Keep that name in the manuscript.

If the response specifically requires a foundation model, treat it as a
separate workstream. An open model such as ECG-FM is technically feasible, but
its published pretraining uses large PhysioNet-derived corpora. Before any
Chapman/PTB-XL/Georgia comparison, document pretraining-dataset overlap and
exclude a model whose pretraining contains the evaluation cohort; otherwise
the comparison is not a clean zero-target-label generalization test.

Before execution:

1. Expose and record `embed_dim`, `n_heads`, `depth`, `patch_size`, `patch_stride`, feed-forward multiplier, positional encoding, pooling, norm order, and dropout in CLI and manifest.
2. Ensure cache validation includes every architecture field.
3. Preserve fixed 20 epochs, fold-train positive weights, AdamW, cosine schedule, threshold 0.5, and Q=3.
4. Reuse completed fold checkpoints/caches only when their full config hash matches.

### Run sequence

1. Restore available Transformer fold checkpoints and fold prediction caches.
2. Run only missing folds with `--only-folds` and immediate mirror publish after each fold.
3. Aggregate all five folds without retraining.
4. Compute 1000-bootstrap CI on CPU.
5. Run `25_paired_full_vs_transformer.py`.
6. Ingest through Notebook 04 and readiness gate before external inference.

### Gate

- 44,186 records, 27 classes, five fold IDs, canonical record fingerprint.
- Summary, OOF prediction, slice prediction, class/fold tables, manifest, and paired artifacts all present.
- No “foundation model” wording unless a separately audited pretrained model is actually used.
- The complete Transformer architecture/config hash covers patch size/stride, embedding width, heads, layers, feed-forward width, positional embeddings, pooling, norm order, dropout, optimizer, and schedule.

### Hardware

A100 High-RAM for missing training/inference; CPU for aggregation/bootstrap.

## Phase 5 - Matched Zero-Shot External Baselines

### Scope and order

1. PTB-XL: Full, ResNet1D/CNN, Raw Mamba, then compact Transformer after its in-domain gate passes.
2. Georgia: same model set under the reviewed 27-class SNOMED-to-Chapman mapping.
3. CPSC2021: same model set only as a separate AF/AFL 10-second mapped-window task.

PTB-XL provides recommended patient-respecting folds and a well-documented benchmark structure. Use fold 10 as the fixed external test set. For later adaptation, use fold 9 as the target adaptation/validation pool rather than randomly splitting fold 10.

### Implementation

Use the implemented comparator exporter rather than overloading the Full ECG-RAMBA exporter:

- `scripts/revision/31_generate_external_comparator_predictions.py`

Arguments:

- `--dataset {ptbxl,georgia,cpsc2021}`
- `--comparator {resnet,raw_mamba,transformer}`
- checkpoint directory and expected architecture/config hash
- `--reuse-existing`, `--only-folds`, batch size, device, AMP

Reuse from `03_generate_external_predictions.py`:

- Dataset loading, reviewed mappings, record/window order, perturbation-free signal normalization, and task definitions.
- PTB-XL mapping: max over the same source Chapman codes for each supported superclass.
- CPSC2021 mapping: max of AF and AFL probabilities.

Do not reuse Full-model-only Hydra/PCA/HRV features for raw comparator models.

### Prediction schema

Every external model NPZ must include:

- `y_true`, `y_prob`, `record_id`, `group_id`, `class_names`.
- Dataset/task protocol version and mapping hash.
- Source checkpoint fingerprints for all five folds.
- Model architecture/config hash.
- Archive/data fingerprint and ordered record fingerprint.
- Threshold and aggregation metadata.

### Group definitions

- PTB-XL: `patient_id` from metadata; official fold ID retained.
- Georgia: audited patient/group identifier if available; otherwise assert and document one ECG per independent unit.
- CPSC2021: source record prefix before the window start/end suffix.

### Comparison and gate

Use:

- existing dataset gate `scripts/revision/18_external_protocol_gate.py`
- implemented paired comparator audit `scripts/revision/32_paired_external_comparators.py`

Use group-cluster paired bootstrap on the common evaluable units. Report absolute metrics and paired deltas, not a pooled average across datasets. CPSC2021 is always a separate panel.

### Hardware

A100 for inference, particularly Georgia and CPSC2021. CPU for gates and bootstrap.

## Phase 6 - R2-C4: Group-Safe Score Calibration And True Few-Shot Adaptation

### Step 1: invalidate v1 as claim-ready

Update the readiness gate so `fewshot_score_calibration_v1_gated_external` is not manuscript-ready when no group IDs and disjointness audit are present. Preserve the old files for provenance; do not overwrite them.

Also remove the phrase "leakage-audited" from current manuscript/rebuttal text until v2 passes. A fixed row-level test set is not sufficient when rows are repeated windows or multiple ECGs from one patient.

### Step 2: score-calibration v2

Replace row-level random splitting with group-safe splitting:

- PTB-XL: adaptation from a nested 1/5/10% subset of fold 9 patient groups; fixed fold 10 patient groups for test.
- Georgia: frozen group-stratified adaptation/test split, nested adaptation subsets, five seeds.
- CPSC2021: source-record-disjoint groups; all windows from one record remain in one split.

Bootstrap groups, not windows. Store and validate:

- adaptation/test group IDs and SHA.
- zero group overlap.
- class coverage and unsupported/skipped classes.
- records/windows per group and per split.
- group-level seed/fraction definitions.

New protocol name:

- `group_safe_score_calibration_v2_gated_external`, implemented in
  `scripts/revision/33_group_safe_score_calibration.py`

### Step 3: true parameter adaptation

Use the implemented representation/adaptation runners:

- `scripts/revision/34_extract_external_representations.py`
- `scripts/revision/35_true_fewshot_head_adaptation.py`

Recommended first claim-ready mode:

- Frozen source encoder plus a newly trained target-task linear head.
- Optional bottleneck adapter as a second mode.
- Full fine-tuning only as an explicitly separate high-variance sensitivity, preferably at 10% labels.

For each model, expose the penultimate representation:

- Full ECG-RAMBA: fused/context representation.
- ResNet1D/CNN: pooled residual representation.
- Raw Mamba: pooled Mamba representation.
- Transformer: normalized pooled token representation.

Use fixed hyperparameters or an adaptation-only validation group; never select epochs, thresholds, or modes on the held-out test groups.

### Required comparisons

- Zero-shot mapped score.
- Group-safe score calibration.
- Frozen-encoder linear-head adaptation.
- Optional adapter/full fine-tuning sensitivity.
- Full, ResNet, Raw Mamba, and Transformer on identical splits where available.

### Gate

- Train/validation/test group intersections are empty.
- Fractions refer to target groups/patients, not windows.
- CPSC train counts are reported as source records and windows separately.
- Ranking and threshold metrics are reported separately.
- No general few-shot superiority claim without matched comparator paired CIs across more than one task.

### Hardware

A100 for one-time embedding extraction and any adapter/full fine-tuning. Frozen-head training and bootstrap can run on CPU.

## Phase 7 - R2-C2: Representation Audit Presentation And Null Tests

### Existing evidence

- Fold-safe linear probes are complete but weak.
- Linear CKA shows non-identical embeddings but does not prove semantic disentanglement.
- Existing UMAP plots are colored by fold and therefore do not answer the requested morphology/rhythm visualization directly.

### Implemented extension

`scripts/revision/20_representation_probe.py` now provides:

1. Unsupervised UMAP/PCA coordinates colored after fitting by morphology-only,
   rhythm-only, overlapping, and neither label groups.
2. Fold-colored panels retained as a leakage/domain audit.
3. Fold-safe out-of-fold linear probes and fold-level metric rows.
4. All-OOF and held-out-fold linear CKA rows.

### Remaining optional controls for mechanism-level closure

1. Label-permutation null distributions for each probe/view pair.
2. Dimension-matched random-embedding baselines.
3. Confidence intervals and paired branch-view probe comparisons against null
   and against other views.

These controls are not required for the conservative audit wording, but they
are required before using the stronger mechanism-level acceptance row below.

### Manuscript output

- One appendix figure with morphology, rhythm, context, and fused panels.
- One compact probe/CKA/null table.
- Caption: representation audit only; weak probes do not establish morphology-rhythm disentanglement.

CKA is a representation-similarity statistic; it can identify correspondence/non-identity but is not a semantic separation test by itself.

### Hardware

CPU for probes/CKA; GPU only if embeddings need re-extraction.

## Phase 8 - Editorial Marked Manuscript And Final Freeze

### Marked manuscript

Use `BACKUP.tex` as the original source only after confirming it matches the initially submitted manuscript. Generate a flattened diff with `latexdiff`:

- `main_marked.tex`
- `main_marked.pdf`
- `main_clean.pdf`

The build/status wrapper is implemented as
`scripts/revision/36_build_marked_manuscript.py`. It must report
`editorial_ready=true`; a `blocked_missing_tool` manifest is not a PDF.

If automatic diff is visually unreadable, create a curated marked version that highlights substantive changes by section while preserving equations/tables.

### Final sequence

1. Rerun affected notebooks in dependency order.
2. Run claim-readiness gates.
3. Regenerate Notebook 07 final tables.
4. Update manuscript/response only from generated tables.
5. Compile clean and marked PDFs.
6. Render and visually inspect every page.
7. Extract PDF text and run forbidden-claim/numeric consistency scans.
8. Publish mirror and submission package with checksums.

### Final package

- Clean manuscript PDF.
- Marked manuscript PDF.
- Point-by-point response with exact section/table/figure references.
- Final evidence tables and claim-readiness gates.
- Artifact provenance and checksum manifest.
- Optional supplementary evidence figures/tables.

## Notebook Integration Plan

### Notebook 02

- Group IDs and split IDs are implemented in external exports.
- Matched external comparator exports and paired gates are integrated in auto mode.
- Few-shot v1 is blocked from claim ingestion; group-safe v2 and true frozen-head adaptation are integrated.
- Publish after each dataset/model stage.

### Notebook 03

- Regenerate canonical calibration/reliability from the current OOF SHA.
- Add cluster-bootstrap mode and presentation assets.

### Notebook 04

- Run the morphology-transform identity preflight, then complete the compact Transformer and frozen-transform MLP head control.
- A separate learned morphology-only CNN remains an optional mechanism-level experiment and is not currently implemented.
- Run paired comparisons only after all SHA/config checks pass.
- Save/publish fold caches and checkpoints immediately.

### Notebook 05

- Keep Full-vs-MiniRocket stress evidence canonical.
- Ingest learned-comparator robustness only after named profile completion.

### Notebook 06

- Label/fold-colored representation panels and fold-safe probes are integrated.
- Null/permutation and dimension-matched random baselines remain optional mechanism-level work.
- Reuse embeddings when their manifest matches.

### Notebook 07

- Refuse stale few-shot v1 artifacts.
- Ingest only group-safe v2/adaptation and matched external baseline artifacts that pass gates.
- Export all paper-facing figures/tables and readiness status.

## Recommended Run Order

1. Phase 0 provenance restore, morphology-transform identity decision, and code commit.
2. Phase 1/2 presentation assets, PCA correction, pooling tables.
3. Phase 3 frozen-transform linear/MLP head controls.
4. Phase 4 Transformer OOF plus paired comparison.
5. Phase 5 PTB-XL external ResNet/Raw Mamba/Transformer.
6. Phase 5 Georgia external comparators.
7. Phase 5 CPSC2021 external comparators as a separate mapped-window track.
8. Phase 6 group-safe calibration v2, then true few-shot adaptation.
9. Phase 7 current representation audit and appendix figure.
10. Notebook 07 final freeze, manuscript update, marked PDF, and final scan.
11. Only if mechanism-level reviewer pressure remains: implement the learned
    morphology-only CNN and representation null/random-baseline extensions,
    rerun Notebook 07, and revise wording again.

## Stop Conditions

- Stop and repair provenance if any canonical SHA differs from the selected Drive authority.
- Stop manuscript ingestion if the current custom random-convolution transform is labeled as canonical MiniRocket.
- Stop few-shot/adaptation if any group appears in more than one split.
- Stop a comparator if OOF record/class/fold coverage is incomplete.
- Stop manuscript ingestion if a screening artifact is being treated as canonical.
- Do not pursue broad superiority, clinical readiness, full-HRV, or mechanistic-disentanglement wording regardless of favorable isolated results.

## Acceptance Matrix

| Workstream | Minimum complete artifact set | Statistical unit | Notebook 07 gate |
|---|---|---|---|
| Reliability/CI | reliability figure, compact CI table, paired delta table, manifest | Chapman subject/independent record as audited | Canonical OOF/freeze SHA and 1000 valid resamples |
| Morphology identity | transform contract table and identity manifest | Not applicable | Exact transform name and checkpoint/source hash match |
| Frozen-transform heads | OOF NPZ, fold/summary tables, paired JSON/CSV, manifest | Chapman subject/record | Five-fold complete, same features/splits, config hash match |
| Learned morphology CNN | OOF/slice NPZ, checkpoints, logs, paired artifacts, manifest | Chapman subject/record | Five-fold complete, fixed epoch/threshold/Q=3 |
| Compact Transformer | OOF/slice NPZ, five fold checkpoints, summary/class/fold tables, paired artifacts | Chapman subject/record | Architecture hash complete and paired contract passed |
| External comparators | per-dataset/model predictions, mapping/config manifests, paired tables | PTB patient; Georgia audited group; CPSC source ECG | Common evaluable units and group-cluster paired CI |
| Score calibration v2 | group split NPZ, overlap audit, summary/bootstrap, manifest | Target group | Zero overlap and nested group fractions |
| Parameter adaptation | embeddings/checkpoints, fixed split logs, paired tables, manifest | Target group | No test-based selection; matched models/splits |
| Conservative representation audit | UMAP/PCA panels, fold probe table, CKA table, manifest | Chapman subject/record | Fold-safe coordinates/probes; wording explicitly disclaims disentanglement |
| Mechanism-level representation extension | permutation-null and dimension-matched random baselines plus paired CIs | Chapman subject/record | Null contracts and paired comparisons complete; still no causal claim |
| Editorial | clean PDF, marked PDF, PDF text, scan report, checksums | Not applicable | Visual QA and forbidden-claim scan passed |

## External Methodological References

- PTB-XL provides patient-respecting recommended folds and explicit diagnostic superclass metadata: https://physionet.org/content/ptb-xl/1.0.2/
- The PTB-XL data descriptor documents 21,837 ten-second ECGs, diagnostic superclasses, and recommended stratified folds: https://pmc.ncbi.nlm.nih.gov/articles/PMC7248071/
- The PhysioNet/CinC 2020 source documents the Georgia records and SNOMED-CT diagnosis labels; mappings must be reviewed rather than treating unmapped labels as negatives: https://physionet.org/content/challenge-2020/1.0.2/
- CPSC2021 is an AF-event localization challenge; a 10-second AF/AFL window audit is a derived task and not the official onset/offset score: https://physionet.org/content/cpsc2021/1.0.0/
- Calibration and reliability diagrams: Guo et al., 2017, https://proceedings.mlr.press/v70/guo17a.html
- The original ROCKET transform uses random convolutional kernels and MAX+PPV features: https://arxiv.org/abs/1910.13051
- MiniRocket is almost deterministic rather than universally deterministic: https://arxiv.org/abs/2012.08791
- CKA measures representation similarity/correspondence, not semantic disentanglement: https://arxiv.org/abs/1905.00414
- UMAP is a descriptive nonlinear projection method and does not by itself establish class separation: https://arxiv.org/abs/1802.03426
- ECG-FM is an open pretrained ECG foundation model, but its pretraining sources must be audited for overlap before it can be used as a fair external comparator: https://arxiv.org/abs/2408.05178
