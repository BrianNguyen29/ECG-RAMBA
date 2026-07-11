# Reviewer Requirement Gap Assessment

*Reviewed: 2026-07-11*
*Sources: `Decision letter (Initial Submission).txt`, `ECG_RAMBA_Reviewer_Discussion.md`, current `main.tex`, response letter, and final-evidence artifacts.*

## Bottom Line

The revision pipeline is substantially stronger and its claim discipline is appropriate, but it does **not** yet fully close every item raised in the original decision letter. The new runners/notebook cells are implemented and pass static/contract tests; the current local artifact tree is not submission-ready because required OOF-dependent artifacts are stale and the new optional experiments have not all been rerun. Notebook 07 strict validation, not file presence, is the completion criterion.

Two verified morphology/PCA description errors were corrected in the working `main.tex`: the frozen manifest records 0.974729-0.974774 explained variance across the five folds (97.4729-97.4774%), and the training code creates 10,000 fixed-seed random kernels whose MAX and PPV statistics yield 20,000 outputs before PCA. The corrected source still requires a clean/marked PDF build and visual verification before submission.

The local `MiniRocketNative` implementation is also not the canonical MiniRocket transform described by Dempster et al. Canonical MiniRocket uses 84 fixed length-9 kernels, training-derived quantile biases, and PPV features. The evaluated checkpoint instead uses fixed-seed `{-1,0,1}` random kernels, Gaussian random biases, and MAX+PPV. The current model must therefore be described as a fixed-seed random-convolution transform inspired by the ROCKET family unless the complete five-fold model is retrained with a canonical multivariate MiniRocket implementation.

## Requirement Map

| ID | Reviewer request | Current status | Evidence and current limitation | Required action for full closure |
|---|---|---|---|---|
| R1-C1 | Calibration/safety interpretation | Implemented, pending current-SHA rerun and manuscript insertion | The presentation generator now validates the frozen OOF/calibration contract and produces a reliability figure plus compact CI tables. Current local calibration/paired inputs are stale, and `main.tex` still does not include the generated figure/table. | Rerun Notebook 03 after current calibration/paired artifacts exist, insert the outputs, and keep all wording non-clinical. |
| R1-C2 | Deterministic morphology versus learned/hybrid alternatives | Frozen-head control implemented; causal question remains open | The existing branch is a fixed-seed random-convolution MAX+PPV transform, not canonical MiniRocket. The five-fold frozen-transform MLP runner and paired gate now exist, but it changes only the head and does not make kernels learnable. Full ResNet does not isolate determinism versus regularization. | Run the linear-head/MLP-head control and report head-capacity sensitivity only. Add a learned morphology-only Conv1D or genuinely learnable-kernel control only if the rebuttal must directly isolate the mechanism. |
| R1-C3 | HRV domain bias | Addressed with limitation | HRV-only baseline and near-perfect domain classifier directly show domain sensitivity. The manuscript correctly disclaims full HRV features. | Preserve limitation. Do not retrofit RMSSD/SDNN/LF/HF claims. |
| R1-C4 | Strong CNN and Transformer baselines | Compact Transformer pipeline implemented, evidence pending | ResNet1D/CNN and Raw Mamba are fair frozen-OOF comparators. A compact patch-Transformer five-fold runner, checkpoint cache, paired gate, and external eligibility gate are implemented, but current final OOF/paired artifacts are absent or stale. It is not a pretrained foundation model. | Complete five folds and paired comparison before citing it. If a foundation-model comparison is required, define a separate pretrained model and preprocessing/fine-tuning protocol. |
| R1-C5 | Significance testing / confidence intervals | Partial | OOF and external bootstrap CI artifacts exist and paired bootstrap is complete for MiniRocket, ResNet, and Raw Mamba. The manuscript table displays only point estimates and no CI/delta table. | Add a compact CI/paired-delta appendix table and cite it from the main baseline/external sections. |
| R1-C6 | Q=3 rationale and consistency | Generators implemented, presentation pending | Chapman sensitivity exists and a group-bootstrap external pooling runner now evaluates PTB-XL/Georgia separately while keeping CPSC2021 separate. The manuscript still gives no visible sensitivity table/range. | Regenerate current-contract tables, insert them in the appendix, and describe Q=3 only as frozen and sensitivity-tested. |
| R1-C7 | HRV, PCA, Mamba reproducibility details | Source correction complete; generated table/PDF pending | HRV schema and Algorithm 1 are explicit. `main.tex` now states the verified PCA range and 10,000-kernel/20,000-output contract and says to apply the fixed transform before fitting training-fold PCA. The generated per-fold table is not yet inserted. | Insert the manifest-derived fold table, compile, and visually verify the method/appendix pages. |
| R2-C1 | Step-by-step mathematical pipeline and variable definitions | Substantially addressed, pending PCA correction | Algorithm 1 and symbols around attention/aggregation are clearer. | Correct PCA variance; do a final equation/notation pass before rebuild. |
| R2-C2 | Quantitative and qualitative branch-separation evidence | Audit pipeline implemented, current artifact rerun/presentation pending | Fold-safe probes, held-out-fold CKA, and fold/label-colored UMAP/PCA audit figure are implemented with canonical provenance gates. The current local embedding is stale for the active OOF, probes are weak, and the manuscript has no figure. | Restore/re-extract matching embeddings, rerun the v3 audit, and add the figure/table with explicit descriptive-audit/no-disentanglement wording. |
| R2-C3 | Noise, missing leads, sampling-rate robustness | Addressed within stated comparator scope | Six stresses and 30 Full-vs-MiniRocket metric rows are complete. | Keep metric-specific Full-vs-MiniRocket wording. Add learned-comparator robustness only after canonical aggregate outputs exist. |
| R2-C4 | Few-shot versus no-target-label classification | Group-safe and true-head runners implemented, evidence pending | v1 row-split score calibration is retained only for provenance and must not be called leakage-audited. Group-safe v2 score calibration and matched frozen-encoder linear-head adaptation now enforce PTB-XL fold 9/10 or source-group-disjoint splits and cluster bootstrap, but their current artifacts are absent or stale. | Run the v2/head pipelines, validate zero overlap and matched models/splits, then report score calibration separately from parameter adaptation. |
| Editorial | Highlighted revised manuscript and point-by-point response | Build wrapper implemented, PDF incomplete | The marked-manuscript runner records a reproducible block when `latexdiff`/LaTeX is missing; no visually verified marked PDF exists yet. | Run it in TeX Live/Overleaf with `latexdiff` and `latexmk`, visually inspect clean/marked PDFs, refresh PDF text, and scan forbidden claims. |

## Mandatory Presentation Corrections Before Resubmission

1. Preserve the corrected morphology/PCA description in `main.tex`: the evaluated transform uses 10,000 fixed-seed random kernels and emits 20,000 MAX+PPV features; PCA retains `97.4729-97.4774%`. Verify this wording in the compiled clean and marked PDFs.
2. Add the existing reliability diagram and a compact bootstrap CI/paired-delta table. Existing artifacts are sufficient; this does not require retraining.
3. Add the existing UMAP/PCA and probe/CKA evidence in an appendix/supplement. Its caption must state that it is an audit and does not establish morphology-rhythm disentanglement.
4. Add a visible Q=3 sensitivity table/appendix summary. Keep the language “frozen operating point” and remove any suggestion of universal optimality.
5. Generate a marked/highlighted manuscript PDF and rerun the PDF/forbidden-claim checks.
6. Remove "leakage-audited" from the current v1 score-calibration wording until group-safe v2 split and overlap audits pass.

## Experimental Work Needed To Fully Answer Remaining Reviewer Pressure

### Priority 0: Method identity and group-split audit

1. Freeze and document the exact evaluated morphology transform: seed, kernel distribution, kernel count, dilation set, bias generation, MAX/PPV pooling, and 20,000-dimensional output.
2. Decide between a claim-only correction (recommended for this checkpoint) and a complete five-fold retrain using canonical multivariate MiniRocket. Do not silently substitute a new transform into the old checkpoint contract.
3. Add `group_id` to external prediction artifacts and make the v1 few-shot readiness gate fail manuscript ingestion.

### Priority 1: Controlled morphology and transformer baselines

1. Rename the current Hybrid runner as a frozen-transform MLP-head control; add a matched neural linear-head control and a learned morphology-only Conv1D encoder.
2. Run the frozen-transform controls and their paired comparisons under the canonical OOF contract.
3. Complete `24_transformer_ecg_baseline.py`, ensure OOF predictions/summary/manifest are complete, then run `25_paired_full_vs_transformer.py`.
4. Regenerate Notebook 04 and Notebook 07 only after each paired artifact passes the freeze and full-prediction provenance checks.

### Priority 2: True few-shot adaptation, if the authors wish to claim it

1. Retain the current score-calibration analyses as a separate, useful operating-point sensitivity.
2. Define fixed per-dataset target-domain adaptation/test splits before training.
3. Compare zero-shot, linear-head/adaptor adaptation, and limited full fine-tuning at 1%, 5%, and 10% labels.
4. Use the same labels, split manifests, thresholds, and bootstrap protocol for ECG-RAMBA and all selected fair comparators.
5. Report ranking and fixed-threshold metrics separately. Never infer broad transfer superiority from a calibration-only gain.

### Priority 3: Learned-comparator robustness

1. Finish a named multi-comparator profile from the already generated stress predictions.
2. Publish the complete summary, pairwise JSON, table, bootstrap cache/manifest, and profile label to the mirror.
3. Extend Notebook 07 to ingest exactly that profile. Do not merge a reviewer-minimal 200-bootstrap screen with canonical 1000-bootstrap claims.
4. Update the manuscript only if the regenerated final evidence explicitly permits metric/comparator-specific wording.

## Work That Should Remain Explicitly Deferred

- Full HRV (RMSSD/SDNN/LF/HF) requires a new schema and five-fold retraining. It cannot be added to the current final-EMA contract.
- Clinical deployment or safety readiness requires prospective/clinical utility validation and a pre-specified decision target.
- Broad in-domain, SOTA, global fairness, or unqualified external-transfer advantage is contradicted by the current ResNet1D/CNN and Raw Mamba evidence.

## Recommended Submission Decision

Do not submit the current PDF unchanged. First complete the method-identity correction, PCA/output-dimension correction, v1 few-shot wording downgrade, presentation assets, and marked manuscript. After that, the paper can be resubmitted with a claim-bounded response. Run the Priority 1 experiments before resubmission only if the team can complete and integrate them cleanly; otherwise do not imply that hybrid or Transformer comparisons were performed.
