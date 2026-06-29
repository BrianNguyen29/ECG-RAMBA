# Reviewer Requirements Gap Matrix

Date: 2026-06-29

Source documents checked:

- `D:\WorkSpace\ECG\ECG-Ramba\Decision letter (Initial Submission.txt`
- `D:\WorkSpace\ECG\ECG-Ramba\ECG_RAMBA_Reviewer_Discussion.md`
- Current manuscript: `D:\WorkSpace\ECG\ECG-Ramba\docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\main.tex`
- Current response letter: `D:\WorkSpace\ECG\ECG-Ramba\docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\response_to_reviewers_revised_20260622.md`
- Final evidence tables under `D:\WorkSpace\ECG\ECG-Ramba\drive\final_evidence_tables`
- Revision artifacts under `D:\WorkSpace\ECG\ECG-Ramba\drive\revision_artifacts\reports\revision`

This file maps the original reviewer requirements to the current evidence package. It separates manuscript-ready evidence from items that still require technical implementation. It is a planning and audit document, not a replacement for the final evidence matrix.

## Executive Status

The current manuscript is defensible if it remains claim-bounded:

- ECG-RAMBA must not be presented as SOTA, globally superior, or the best in-domain model.
- ResNet1D/CNN is the strongest completed in-domain fair comparator across the principal metrics.
- Raw Mamba is also stronger than Full ECG-RAMBA on PR-AUC, ROC-AUC, and F1.
- ECG-RAMBA has comparator-specific operating-point advantages only against MiniRocket-only and Raw Mamba for calibration/error metrics, not against ResNet1D/CNN.
- PTB-XL is manuscript-ready only as a protocol-gated mapped-task external evaluation.
- Georgia/CPSC2021, few-shot adaptation, representation probes, full HRV, and broad robustness superiority remain deferred.
- P0 manuscript packaging is now complete: the IEEE PDF compiles, Algorithm 1 is present, figures/captions/tables were visually checked, and the positive forbidden-claim scan passed. See `ECG-RAMBA/docs/revision_plan/p0_pdf_compile_claim_scan_20260629.md`.

The strongest rebuttal strategy is therefore not to defend the original "zero-shot/disentanglement/SOTA" narrative. The revised response should emphasize that the authors accepted the reviewers' concerns, added fair evidence, and narrowed the claims.

## Reviewer 1 Matrix

| Reviewer item | Original concern | Current evidence status | Current manuscript/rebuttal action | Remaining technical work |
|---|---|---|---|---|
| R1-C1 Safety/ranking-decision gap | ROC-AUC vs F1 gap cannot justify "safety-oriented" behavior without calibration. | Complete. `calibration_ci_oof_final_ema_predictions.json` reports ECE, MCE, Brier, reliability artifacts, threshold metrics, and bootstrap CI. | Current text reframes this as fixed-threshold/calibration operating-point behavior, not clinical safety. | None required for current manuscript. Do not reintroduce "safety-oriented" language. |
| R1-C2 Deterministic MiniRocket benefit | Need comparison against learned morphology encoder; unclear determinism vs regularization. | Partially/mostly complete. MiniRocket-only and ResNet1D/CNN baselines are complete. Paired Full-vs-MiniRocket and Full-vs-ResNet are available. ResNet is stronger than Full ECG-RAMBA. | Use this as model-characterization evidence. Do not claim deterministic morphology is superior. | Optional: implement hybrid CNN+HRV+Mamba or No-MiniRocket variants only if reviewer demands fine-grained component attribution. Not required if claim is narrowed. |
| R1-C3 HRV domain bias | Full-record HRV may encode recording length, preprocessing, noise, or dataset identity. | Complete as limitation evidence. HRV-only and HRV domain classifier are complete. HRV domain classifier nearly separates dataset source perfectly. HRV36 schema documents reserved zero-filled slots. | Current text correctly says HRV is useful but domain-sensitive, not invariant. | Optional: duration-normalized HRV and HRV noise-sensitivity analysis if stronger HRV-control evidence is needed. Do not claim HRV invariance. |
| R1-C4 Baselines too weak | Raw Mamba alone is not representative; need CNN/Transformer/strong ECG baselines. | Mostly complete for fair in-domain baselines. ResNet1D/CNN, Raw Mamba, MiniRocket-only, HRV-only are complete under frozen OOF. | Current manuscript acknowledges ResNet and Raw Mamba are stronger and removes best-model claims. | Optional: add Transformer/InceptionTime/foundation baseline only if reviewer explicitly requires broader benchmark coverage. Current ResNet/CNN baseline already addresses the most direct learned-architecture gap. |
| R1-C5 Statistical significance/CI | Need CI/significance, especially for zero-shot. | Complete for frozen OOF and fair paired comparisons. PTB-XL mapped-task gate includes bootstrap CI. | Current response can cite record-level bootstrap with 1000 resamples and paired bootstrap tables. | Do not claim zero-shot superiority. If adding Georgia/CPSC/few-shot later, each must have its own bootstrap CI/gate. |
| R1-C6 Power Mean Q=3 | Q=3 appears heuristic. | Complete with limitation. Pooling sensitivity artifacts exist; Q=3 is frozen/sensitivity-tested, not globally optimal. | Current text says Q=3 is a pre-specified/frozen operating point and tradeoff. | None required. Avoid "optimal Q" wording. |
| R1-C7 Reproducibility | Need HRV list, PCA, Mamba config, normalization, regularization details. | Complete for the current claim scope. Provenance table, HRV schema, artifact manifests, final EMA contract, fixed threshold, aggregation, baseline scripts, logs, and Algorithm 1 exist. | Current manuscript includes HRV schema restriction, protocol, model config, artifact provenance, and fold-aware Algorithm 1. | None required before submission except rerunning compile/scan if the manuscript changes. Full HRV RMSSD/SDNN/LF-HF claims remain forbidden unless retrained with real features. |

## Reviewer 2 Matrix

| Reviewer item | Original concern | Current evidence status | Current manuscript/rebuttal action | Remaining technical work |
|---|---|---|---|---|
| R2-C1 Mathematical pipeline and variable definitions | Method is superficial; define equations and training/PCA/testing steps. | Complete for the revised manuscript. Method now describes preprocessing, fold-aware PCA, Mamba equations, Power Mean aggregation, training protocol, and Algorithm 1. Variables including `T`, `d_k`, `M_i`, `Q`, `A`, `B`, `x_t`, and `h_t` are defined near first use. | Current text directly addresses the requested step-by-step pipeline and variable definitions. | None required before submission except rerunning PDF compile/scan if the manuscript changes. |
| R2-C2 Morphology-rhythm separation/disentanglement | Lead dropout alone is insufficient; need qualitative/quantitative representation evidence. | Not complete as manuscript evidence. `22_extract_representations.py` now exists as a checkpoint-fingerprinted embedding extractor, and `20_representation_probe.py` exists as a gated embedding-artifact consumer. Notebook 06 wires both steps, but the full embedding/probe artifact package has not yet been run. | Current manuscript correctly avoids proven disentanglement claims and presents lead dropout/saliency/ablation as component-sensitivity evidence only. | If reviewer demands stronger architecture evidence, run Notebook 06 representation extraction for all five folds, then run `20_representation_probe.py` for fold-safe linear probes, UMAP/PCA figures, and CKA. Safe claim: "suggestive branch-specific information", not disentanglement. |
| R2-C3 Robustness under noise/missing leads/sampling rates | Need simulations for noise corruptions, missing leads, and sampling-rate shifts. | Complete with limitation. Robustness artifacts cover 6 stresses x 5 metrics vs MiniRocket-only: SNR20/10/5, random 3-lead dropout, precordial dropout, resample 250 Hz. | Current text says robustness is metric-specific vs MiniRocket-only: ECG-RAMBA better for stressed F1/Brier/ECE, MiniRocket better for PR-AUC/ROC-AUC and less degraded in most rows. | Optional: extend robustness to ResNet1D/CNN and Raw Mamba if a reviewer asks for general robustness. Do not claim broad robustness superiority now. |
| R2-C4 Few-shot vs zero-shot comparison | Need comparative simulation between few-shot and zero-shot ECG classification. | Not complete as manuscript evidence. `19_fewshot_adaptation.py` now exists as a gated score-calibration runner with fixed target-domain test splits and nested few-shot pools, but it has not been run on a completed external gate package. | Current response defers few-shot and says no zero-shot superiority claim is made. | Run only after an external dataset gate passes. Current runner supports leakage-audited score calibration on frozen predictions; model-weight fine-tuning and matched comparator few-shot runs remain optional future extensions if required. |

## Associate Editor Requirements

| AE requirement | Current status | Action |
|---|---|---|
| Major revision with detailed point-by-point response | Mostly complete. Response letter exists and is aligned with final evidence. | Finalize after PDF compile; ensure every reviewer item has section/table references. |
| Highlight changes in manuscript | Unknown from current audit. | Prepare highlighted manuscript or marked PDF before portal upload. |
| One major revision only; avoid another major gap | High risk if unsupported claims return. | Keep claim boundaries strict and do not open new experiments unless they are protocol-gated. |
| PDF compile, figure/caption/table sanity, forbidden-claim scan | Complete as of 2026-06-29. | Preserve the audit file and rerun only after any text/table/caption edits. |

## Manuscript-Ready Evidence That Can Be Added Or Cited

Use these as rebuttal evidence:

| Evidence | Artifact/source | Key values | Safe interpretation |
|---|---|---|---|
| Frozen Chapman OOF Full ECG-RAMBA | `oof_final_ema_predictions.npz`, `calibration_ci_oof_final_ema_predictions.json` | ROC-AUC 0.8373, PR-AUC 0.3432, F1 0.3998, Brier 0.0355, ECE 0.0552 | Primary protocol-faithful OOF evidence, not best-in-domain. |
| ResNet1D/CNN fair baseline | `resnet1d_cnn_baseline_summary.json`, `table_paired_full_vs_resnet.csv` | ROC-AUC 0.9391, PR-AUC 0.4899, F1 0.4906, Brier 0.0342, ECE 0.0414 | ResNet is stronger in-domain; ECG-RAMBA cannot claim best model. |
| Raw Mamba fair baseline | `raw_mamba_baseline_summary.json`, `table_paired_full_vs_raw_mamba.csv` | ROC-AUC 0.9284, PR-AUC 0.4377, F1 0.4465, Brier 0.0401, ECE 0.1137 | Raw Mamba stronger on discrimination/F1; ECG-RAMBA lower Brier/ECE. |
| MiniRocket-only baseline | `minirocket_only_baseline_summary.json`, `table_paired_full_vs_minirocket.csv` | ROC-AUC 0.9169, PR-AUC 0.4508, F1 0.2477, Brier 0.1906, ECE 0.3759 | MiniRocket-only stronger for ranking; ECG-RAMBA stronger for threshold/calibration metrics. |
| HRV-only/domain evidence | `hrv_only_baseline_summary.json`, `hrv_domain_classifier_summary.json` | HRV-only ROC-AUC 0.8113; domain classifier near-perfect | HRV has signal but is domain-sensitive; no invariance claim. |
| Robustness vs MiniRocket | `table_final_robustness_claims.csv`, `robustness_summary.csv` | 30 rows over 6 stresses and 5 metrics | Metric-specific robustness only. |
| Pooling sensitivity | `pooling_sensitivity.csv`, `pooling_decision_summary.json` | Q grid available | Q=3 is frozen/sensitivity-tested, not globally optimal. |
| PTB-XL external gate | `external_ptbxl_protocol_gate.json` | 2198 records, 4 classes; ROC-AUC 0.8007, PR-AUC 0.6097, F1 0.4623; bootstrap CI present | Protocol-gated mapped-task external evaluation only; not zero-shot superiority. |

## Items Still Requiring Technical Implementation

| Priority | Item | Why it matters | Minimal correct implementation |
|---|---|---|---|
| P0 complete | Compile PDF and forbidden-claim scan | Prevent stale caption/table text from contradicting final evidence. | Completed. IEEE PDF compiled from `main.tex`; active figures exist; no missing refs/citations; positive forbidden-claim scan passed. Audit: `p0_pdf_compile_claim_scan_20260629.md`. |
| P0 complete | Algorithm/protocol clarity check | Reviewer 2 explicitly requested step-by-step math/pipeline. | Completed. Algorithm 1 documents fold-aware PCA/training/inference/Q=3 aggregation; variables are defined close to first use. |
| P1 only if more external evidence is required | Georgia protocol gate | Georgia cannot be used without reviewed label mapping. | Build reviewed SNOMED/Chapman mapping, report unmapped labels, do not coerce unmapped labels to negative, run exporter/gate/bootstrap. |
| P1 only if more external evidence is required | CPSC2021 protocol gate | Requires annotation-aligned episode/window evaluation. | Fix annotation parsing/windowing, validate AF/AFL/normal boundaries, run exporter/gate/bootstrap. |
| P2 if reviewer challenges representation | Representation probe/CKA/UMAP | Current disentanglement evidence is insufficient for strong claims. | Extractor and probe runners exist (`22_extract_representations.py`, `20_representation_probe.py`) and Notebook 06 has explicit flags/cache-aware wiring. Real evidence still requires a completed run and generated artifacts. Use cautious "suggestive" wording only. |
| P2 after external gate | Few-shot adaptation | Reviewer requested zero-shot vs few-shot, but it is high leakage risk without target gate. | Runner exists (`19_fewshot_adaptation.py`) for gated score calibration with fixed test splits and nested few-shot pools. Full model fine-tuning/matched comparator few-shot remains optional future work. |
| P3 optional | Transformer/foundation baseline | Reviewer mentioned transformer-based ECG models; ResNet already addresses learned baseline, but transformer would broaden comparison. | Implement a simple transformer/attention ECG baseline under same frozen OOF, or justify omission in limitation. |
| P3 optional | Robustness vs ResNet/Raw Mamba | Needed only for broad robustness claims. | Aggregation/gating runner exists (`21_robustness_multicomparator.py`), but ResNet/Raw Mamba stress prediction generation is still required. Report only metric-specific paired degradation; do not chase blanket robustness wording. |
| Retrain-level only | Full HRV feature set | Current checkpoint has reserved zero-filled slots; cannot retrofit RMSSD/SDNN/LF-HF. | Implement real HRV slots, update schema, retrain all folds, regenerate OOF/baselines if full-HRV claims are required. |

## Response Strategy By Claim Boundary

### What to say

- "We removed claims of best in-domain performance and global superiority."
- "We added fair baselines under the same frozen OOF protocol."
- "ResNet1D/CNN is stronger than ECG-RAMBA on the main in-domain metrics."
- "Raw Mamba is stronger on PR-AUC/ROC-AUC/F1, while ECG-RAMBA has lower Brier/ECE."
- "PTB-XL is reported only as protocol-gated mapped-task external evidence."
- "Georgia/CPSC2021 and few-shot transfer are deferred until dataset-specific gates are completed."
- "Representation separation remains future work; current analyses support component sensitivity only."

### What not to say

- "ECG-RAMBA is SOTA."
- "ECG-RAMBA is the best in-domain model."
- "ECG-RAMBA is globally robust."
- "ECG-RAMBA proves morphology-rhythm disentanglement."
- "HRV is an invariant anchor."
- "PTB-XL/Georgia/CPSC prove zero-shot generalization."
- "Few-shot experiments were added."
- "Full HRV features including RMSSD/SDNN/LF-HF were implemented."

## Recommended Execution Order

1. Keep the current manuscript/rebuttal wording frozen around the final evidence package.
2. If any text, caption, table, or response-letter edit is made, rerun PDF compile, visual page render, reference check, and positive forbidden-claim scan.
3. If and only if a new external-transfer claim is required, work on Georgia/CPSC2021 gates before any few-shot adaptation.
4. If and only if architecture interpretation is challenged, implement representation probing with cautious wording.
5. Do not start full HRV retraining or broad multi-comparator robustness unless reviewers explicitly request those extensions.

## Post-P0 Full Execution Plan

This is the current execution plan after the PDF/Algorithm/claim-scan update.

Detailed implementation roadmap: `ECG-RAMBA/docs/revision_plan/post_p0_full_execution_roadmap_20260629.md`.

### Submission-critical tasks already complete

1. Calibration and uncertainty evidence for the ranking-vs-decision gap.
2. Fair in-domain baseline matrix: MiniRocket-only, ResNet1D/CNN, Raw Mamba, HRV-only.
3. Paired bootstrap comparisons for Full ECG-RAMBA vs MiniRocket-only, ResNet1D/CNN, and Raw Mamba.
4. HRV-only and HRV domain-sensitivity evidence.
5. Pooling sensitivity for Power Mean `Q=3`.
6. Perturbation robustness vs MiniRocket-only across SNR, lead-dropout, and resampling stresses.
7. PTB-XL protocol-gated mapped-task external evaluation.
8. Algorithm 1 and mathematical pipeline clarification.
9. PDF compile, visual figure/table/caption check, and positive forbidden-claim scan.

### Remaining work only if stronger claims are desired

| Workstream | Trigger | Required implementation | Allowed claim if completed |
|---|---|---|---|
| Georgia gate | Need additional external mapped-task evidence. | Reviewed SNOMED/Chapman mapping; explicit unmapped-label table; no coercion to negative labels; exporter; protocol gate; bootstrap CI; manifest/checksum. | Dataset-specific mapped-task result only. No zero-shot superiority. |
| CPSC2021 gate | Need episode/window AF evidence. | Robust annotation parser; AF/AFL/normal window construction; overlap policy; no annotation-error-to-negative conversion; exporter; protocol gate; bootstrap CI; manifest/checksum. | Annotation-aligned CPSC2021 mapped-task result only. |
| Few-shot adaptation | Reviewer explicitly requires zero-shot vs few-shot after at least one external gate passes. | Frozen external splits; 0/1/5/10% labeled target-domain train subsets; head-only and optional full fine-tune; matched baselines; leakage audit; bootstrap CI. | Few-shot sensitivity under a gated dataset protocol. No global transfer claim. |
| Representation probe | Reviewer requires stronger support for branch separation. | Run Notebook 06 representation extraction/probe or directly run `22_extract_representations.py` then `20_representation_probe.py`; verify fold caches, manifest/checksums, probe table, CKA table, and figures. | Suggestive branch-specific information, not proven disentanglement. |
| Full HRV feature set | Authors want RMSSD/SDNN/LF-HF claims. | Implement real HRV slots, update schema, retrain all folds, regenerate frozen OOF and all dependent evidence. | Full HRV claim only after retrain and artifact freeze. |
| Robustness vs ResNet/Raw Mamba | Authors want broader robustness claims. | Extend stress runner to ResNet1D/CNN and Raw Mamba, regenerate paired degradation CIs for all comparators. | Metric-specific multi-comparator robustness only. No blanket robustness superiority unless every metric/comparator supports it. |
