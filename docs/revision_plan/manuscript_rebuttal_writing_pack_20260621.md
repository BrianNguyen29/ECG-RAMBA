# ECG-RAMBA Manuscript/Rebuttal Writing Pack

Date: 2026-06-21

Purpose: convert the final evidence package into manuscript and rebuttal text that is defensible under reviewer scrutiny. This pack intentionally narrows claims where the evidence is incomplete.

## Authoritative Inputs

- Final evidence folder: `D:\WorkSpace\ECG\ECG-Ramba\final_evidence_tables-20260620T053650Z-3-001\final_evidence_tables`
- Final evidence manifest commit: `32f9d83c66ed1711375ddd4df2a411bc05d6402c`
- `final_ready_for_rebuttal`: `True`
- `all_claims_supported`: `False`
- Updated discussion analysis: `docs/revision_plan/discussion_update_from_final_evidence_20260620.md`
- Claim rewrite table: `docs/revision_plan/claim_rewrite_matrix_20260621.csv`

## Non-Negotiable Writing Rules

1. Do not claim global superiority over fair baselines.
2. Do not claim ECG-RAMBA is generally more robust than MiniRocket-only.
3. Do not claim strict morphology-rhythm disentanglement.
4. Do not describe HRV as an invariant anchor.
5. Do not describe reserved HRV slots as implemented RMSSD/SDNN/LF-HF features.
6. Do not claim external PTB/Georgia/CPSC or few-shot results are manuscript-ready.
7. Do not cite local `model/fold*_final.pt` as canonical manuscript checkpoints; the manuscript evidence is the frozen final_ema OOF artifact package.

## Supported Core Story

The revised paper should be framed as a protocol-faithful evaluation of a structured ECG model with complementary morphology-rhythm streams. The strongest supported result is not global ranking superiority. The supported result is a calibrated fixed-threshold operating-point advantage for Full ECG-RAMBA over MiniRocket-only, while MiniRocket-only remains stronger for rank-based PR-AUC/ROC-AUC.

Use this one-sentence thesis:

> Under a frozen Chapman OOF protocol, ECG-RAMBA provides better calibrated fixed-threshold operating behavior than a MiniRocket-only feature baseline, while MiniRocket-only remains the stronger rank-based discriminator; therefore, the contribution is best presented as a physiologically structured operating-point tradeoff rather than global model superiority.

## Key Numbers To Carry Into The Manuscript

- Frozen Chapman OOF records: 44,186; classes: 27.
- Full ECG-RAMBA: ROC-AUC=0.8373, PR-AUC=0.3432, F1=0.3998, Brier=0.0355, ECE=0.0552.
- MiniRocket-only: ROC-AUC=0.9169, PR-AUC=0.4508, F1=0.2477, Brier=0.1906, ECE=0.3759.
- HRV-only: ROC-AUC=0.8113, PR-AUC=0.1771, F1=0.2077.
- HRV domain classifier: domain AUC approximately 1.0000, so HRV is domain-sensitive.
- Robustness: Full less degraded in 4/30 stress-metric rows; MiniRocket-only less degraded in 26/30 rows. Full is better under stress in 18/30 rows, specifically the fixed-threshold/calibration family.

## Response Letter Draft Pack

### Associate Editor

We thank the Associate Editor and Editor-in-Chief for coordinating the review. We have revised the response strategy by narrowing unsupported claims and tying each reviewer-facing statement to a frozen evidence artifact. The current evidence package supports a cautious rebuttal, not a claim of global superiority.

### Reviewer 1 Comment 1 - Ranking-decision gap and safety interpretation

We agree that the original safety-oriented interpretation was stronger than warranted. The revised wording should describe a calibrated fixed-threshold operating behavior. Evidence: frozen Chapman OOF F1=0.3998, PR-AUC=0.3432, Brier=0.0355, ECE=0.0552. Do not claim clinical safety.

### Reviewer 1 Comment 2 - Deterministic morphology / MiniRocket

The evidence does not support deterministic morphology superiority. MiniRocket-only is stronger for PR-AUC/ROC-AUC, while Full ECG-RAMBA is stronger for F1/Brier/ECE. The rebuttal should present this as a metric-dependent tradeoff and avoid saying ECG-RAMBA outperforms MiniRocket overall.

### Reviewer 1 Comment 3 - HRV domain bias

This concern is supported by the new evidence. HRV-only has non-trivial signal, but HRV domain classification is near-perfect. The manuscript should say HRV is useful but domain-sensitive, and should remove invariant-anchor wording.

### Reviewer 1 Comment 4 - Baseline strength

Only MiniRocket-only and HRV-only feature baselines are complete under the frozen evidence package. Raw Mamba and ResNet1D/CNN fair runners remain deferred. The rebuttal must not claim broad fair-baseline superiority.

### Reviewer 1 Comment 5 - Confidence intervals and significance

The final package includes calibration/bootstrap artifacts, paired Full-vs-MiniRocket evidence, and paired robustness CI rows. The response should say uncertainty analysis was added where final artifacts support it, then report only metric-specific conclusions.

### Reviewer 1 Comment 6 - Power Mean Q=3

Q=3 should be described as the frozen operating point and a sensitivity-tested tradeoff, not a theoretical or global optimum.

### Reviewer 1 Comment 7 - Implementation details

The Methods/Appendix should be rewritten around the artifact contract, A0 blockers, HRV schema, final EMA checkpoint contract, subject-aware OOF protocol, fixed threshold, Q=3 aggregation, and artifact checksums. Do not describe RMSSD/SDNN/LF-HF or amplitude contribution for current checkpoints.

### Reviewer 2 Comment 1 - Mathematical and method clarity

This is primarily a writing task now. Add a concise pipeline description: data -> subject folds -> feature/cache provenance -> fold-aware PCA -> EMA final checkpoints -> slice probabilities -> Q=3 record aggregation -> calibration/baseline/robustness metrics.

### Reviewer 2 Comment 2 - Morphology-rhythm separation

Representation separation remains blocked. Replace disentanglement language with complementary morphology-rhythm stream wording. Do not state that UMAP, probing, or CKA were added.

### Reviewer 2 Comment 3 - Robustness under noise, missing leads, sampling shift

Stress tests are complete, but conclusions are metric-specific. Full ECG-RAMBA is better under stress for F1/Brier/ECE; MiniRocket-only remains better for PR-AUC/ROC-AUC and is less degraded in most rows.

### Reviewer 2 Comment 4 - Few-shot comparison

Few-shot remains optional/deferred. Do not claim few-shot experiments were added. If discussed, frame as future work after the frozen OOF manuscript is locked.

## Abstract Draft

ECG-RAMBA combines morphology- and rhythm-oriented streams for ECG classification and is evaluated under a frozen, traceable Chapman OOF protocol. The revised analysis reports calibration, threshold-dependent operating metrics, paired MiniRocket-only and HRV-only feature baselines, pooling sensitivity, HRV domain-sensitivity analysis, and perturbation stress tests. The results show a calibrated fixed-threshold operating-point advantage for the full model, while also revealing that MiniRocket-only remains stronger for rank-based discrimination. These findings narrow the contribution to a metric-dependent, protocol-faithful characterization rather than a claim of global superiority or clinical safety.

## Contribution Draft

Use these contributions instead of the stronger original claims:

1. A structured ECG modeling pipeline combining morphology- and rhythm-oriented inputs under a subject-aware frozen OOF protocol.
2. A calibration-aware evaluation showing improved fixed-threshold operating behavior for Full ECG-RAMBA relative to MiniRocket-only.
3. A feature-baseline analysis showing that HRV contains rhythm signal but is highly domain-sensitive.
4. A pooling sensitivity analysis supporting Q=3 as a frozen tradeoff operating point, not as a global optimum.
5. A perturbation stress-test analysis showing metric-specific robustness behavior across noise, missing-lead, and sampling-rate shifts.
6. A transparent blocker ledger identifying deferred external, fair-baseline, HRV-schema, and representation-separation limitations.

## Results/Discussion Structure

Recommended order for the manuscript Results and Discussion:

1. Frozen Chapman OOF and calibration metrics.
2. Full ECG-RAMBA vs MiniRocket-only: metric-dependent tradeoff.
3. HRV-only and HRV domain sensitivity.
4. Pooling sensitivity and Q=3 wording.
5. Robustness stress tests with metric-family interpretation.
6. Limitations and deferred claims: external datasets, representation separation, fair baselines, HRV schema/amplitude contribution.

## Safe Discussion Paragraphs

### Full vs MiniRocket-only

The comparison with MiniRocket-only reveals a metric-dependent tradeoff. MiniRocket-only achieves stronger rank-based discrimination, with higher macro PR-AUC and macro ROC-AUC. In contrast, the full ECG-RAMBA model provides better fixed-threshold and calibration-sensitive behavior, including higher macro F1 and lower Brier/ECE at the frozen threshold. This distinction is important because the proposed model should not be interpreted as uniformly superior; rather, it changes the operating-point behavior in a way that is favorable for the calibrated threshold used in this protocol.

### HRV domain sensitivity

The HRV-only analysis confirms that the rhythm descriptor branch contains useful predictive signal. However, the HRV domain classifier separates dataset source almost perfectly, indicating that these descriptors are domain-sensitive. We therefore no longer describe HRV as an invariant anchor. Instead, HRV is treated as a useful but potentially domain-coupled source of rhythm information, and this limitation is explicitly reported.

### Robustness

The perturbation experiments show that robustness depends on the metric family. ECG-RAMBA remains better than MiniRocket-only for stressed fixed-threshold/calibration metrics such as F1, Brier, and ECE. MiniRocket-only, however, remains stronger for stressed PR-AUC/ROC-AUC and is less degraded in most stress-metric rows. The revised manuscript should therefore report metric-specific robustness behavior, not a general robustness advantage.

### Representation separation

The current evidence does not prove strict morphology-rhythm disentanglement. The architecture is designed to combine complementary morphology- and rhythm-oriented streams, and the available behavior is consistent with complementary stream usage. Representation-level probing, UMAP/t-SNE, or CKA analyses would be required before making a stronger disentanglement claim.

### External datasets

The current manuscript-ready evidence is the frozen Chapman OOF protocol. PTB/Georgia/CPSC outputs remain experimental until dataset-specific label mapping, annotation/windowing, fold-PCA provenance, and uncertainty gates are completed. External results should not be used as primary manuscript claims in the current writing pass.

## Old Claim To Revised Claim Matrix

| Area | Old/risky claim | Revised claim | Evidence status | Manuscript action |
|---|---|---|---|---|
| R1C1 safety / ranking-decision gap | safety-oriented behavior; clinically appropriate conservatism | calibrated fixed-threshold operating behavior under a frozen Chapman OOF protocol | `supported_with_limitations` | Replace safety language in Abstract, Results, and Discussion; report calibration metrics and threshold-0.5 metrics. |
| R1C2 MiniRocket / morphology branch | deterministic morphology improves robustness; ECG-RAMBA outperforms MiniRocket | MiniRocket-only is stronger for rank-based discrimination; Full ECG-RAMBA is stronger for fixed-threshold and calibration-sensitive metrics | `blocked_fair_baselines_missing` | Report metric-dependent tradeoff. Do not claim deterministic morphology superiority or global baseline superiority. |
| R1C3 HRV domain bias | HRV serves as an invariant anchor | HRV provides complementary rhythm descriptors but is domain-sensitive | `partially_supported_with_domain_limitation` | Move HRV invariance to limitations. State HRV domain classifier result directly. |
| R1C4 baseline strength | ECG-RAMBA is superior to fair ECG baselines | Only feature baselines MiniRocket-only and HRV-only are complete; Raw Mamba and ResNet1D/CNN fair runners remain deferred | `blocked_fair_baselines_missing` | Do not present broad fair-baseline superiority. If space allows, add ResNet1D/CNN later as a separate risk-managed experiment. |
| R1C5 uncertainty | point estimates alone establish superiority | record-level bootstrap and paired comparisons support only metric-specific conclusions | `supported_with_limitations` | Add uncertainty table text and cite bootstrap method. For robustness, quote paired CI rows only where present. |
| R1C6 Power Mean Q=3 | Q=3 is optimal | Q=3 is the pre-specified frozen operating point and a sensitivity-tested tradeoff | `supported_as_tradeoff_not_optimality_claim` | Keep Q=3 wording as tradeoff, not optimality. Cite pooling sensitivity table. |
| R2C2 morphology-rhythm separation | validating morphology-rhythm disentanglement | architecture combines complementary morphology-rhythm streams; strict representation separation remains future work | `blocked_representation_probe_missing` | Remove proven-disentanglement wording. Do not say UMAP/probing/CKA were added unless those artifacts are generated later. |
| R2C3 robustness stress tests | ECG-RAMBA is more robust than MiniRocket | Full ECG-RAMBA is better under stress for F1/Brier/ECE; MiniRocket-only is better for PR-AUC/ROC-AUC and less degraded in most rows | `supported_metric_specific_with_limitations` | Add robustness subsection with two columns: ranking metrics and operating-point/calibration metrics. |
| R2C4 few-shot / external transfer | external zero-shot transfer is manuscript-ready; few-shot experiments were added | current manuscript-ready evidence is frozen Chapman OOF; PTB/Georgia/CPSC and few-shot remain experimental/deferred | `oof_supported_external_experimental` | Do not claim external or few-shot results unless a separate protocol-specific evidence package is completed. |
| Implementation / HRV schema | 36 HRV features include RMSSD/SDNN/LF-HF and amplitude contribution | current checkpoint input has 5 RR statistics, reserved zero slots, intended amplitude slots, and global statistics; no causal amplitude claim | `manuscript_corrected_with_deferred_feature_claims` | Update Methods and Appendix HRV feature list before any manuscript text leaves the repo. |

## Robustness Summary Table

| Stress | Full less degraded | MiniRocket less degraded | Full better under stress | MiniRocket better under stress |
|---|---:|---:|---:|---:|
| `precordial_dropout` | 0 | 5 | 3 | 2 |
| `random_3_lead_dropout` | 0 | 5 | 3 | 2 |
| `resample_250hz` | 0 | 5 | 3 | 2 |
| `snr10db` | 2 | 3 | 3 | 2 |
| `snr20db` | 0 | 5 | 3 | 2 |
| `snr5db` | 2 | 3 | 3 | 2 |

## Experiment Decision Gate

Do not open new experiments until the manuscript and response letter are locked to the safe wording above.

If an additional experiment is needed after writing lock, the highest-ROI option is a ResNet1D/CNN fair baseline under the same frozen OOF protocol. This would address the strongest remaining baseline blocker, but it has risk: it may not outperform MiniRocket, it requires implementation and compute, and it can delay the resubmission. Do not start Transformer, few-shot, or new external experiments before deciding whether the ResNet1D/CNN baseline is worth that risk.

## Immediate Execution Checklist

1. Replace unsafe wording in Abstract, Introduction, Results, and Discussion.
2. Build the response letter from the comment-by-comment sections above.
3. Insert the claim rewrite matrix as an internal author checklist or rebuttal appendix table.
4. Update task status metadata so it reflects current final evidence rather than the original plan.
5. After text lock, run a final pass searching for forbidden terms: `safety-oriented`, `invariant anchor`, `validates disentanglement`, `global superiority`, `external zero-shot`, `few-shot added`, `RMSSD`, `SDNN`, `LF/HF`.
