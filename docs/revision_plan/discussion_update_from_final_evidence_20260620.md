# ECG-RAMBA Discussion Update From Final Evidence

Date: 2026-06-20

Purpose: consolidate the downloaded final evidence tables and align them with `docs/ECG_RAMBA_Reviewer_Discussion_Revised.docx` so the manuscript Discussion and rebuttal can be rewritten without overclaiming.

## Source Files

Downloaded final evidence folder:

```text
D:\WorkSpace\ECG\ECG-Ramba\final_evidence_tables-20260620T053650Z-3-001\final_evidence_tables
```

Primary evidence artifacts:

```text
final_evidence_matrix.json
table_final_evidence_matrix.csv
table_final_safe_wording.csv
table_final_blocker_status.csv
table_final_robustness_claims.csv
final_evidence_matrix_manifest.json
```

Discussion plan checked against:

```text
D:\WorkSpace\ECG\ECG-Ramba\docs\ECG_RAMBA_Reviewer_Discussion_Revised.docx
```

Final evidence manifest:

- `status`: true
- `final_ready_for_rebuttal`: true
- `all_claims_supported`: false
- `missing_inputs`: none
- `contract_issues`: none
- `unresolved_blockers`: none
- source commit: `32f9d83c66ed1711375ddd4df2a411bc05d6402c`

Interpretation: the evidence package is ready to support a cautious rebuttal, but not every original or planned claim is supported. The manuscript must be revised around the safe wording, not around the stronger language in the old plan.

## High-Level Decision

The revised discussion should not claim global superiority, strict disentanglement, HRV invariance, or external-dataset readiness.

The strongest supported position is:

> ECG-RAMBA provides a protocol-faithful frozen Chapman OOF evaluation with a calibrated fixed-threshold operating-point advantage over the MiniRocket-only feature baseline, while MiniRocket-only remains stronger for rank-based discrimination. Robustness claims must be metric-specific: ECG-RAMBA is usually better at the stressed fixed-threshold operating point for F1, Brier, and ECE, but MiniRocket-only often degrades less and remains stronger for PR-AUC/ROC-AUC.

This is scientifically cleaner than the older broad claim that ECG-RAMBA is simply more robust or more generalizable.

## Key Numbers To Use

### Frozen Chapman OOF, Full ECG-RAMBA

- Records: 44,186
- Classes: 27
- Protocol: `fold_final_ema_power_mean_v2_q3_threshold_0.5`
- ROC-AUC macro: 0.8373
- PR-AUC macro: 0.3432
- F1 macro: 0.3998
- Brier macro: 0.0355
- ECE macro: 0.0552
- ECE micro: 0.0538

Use for: calibrated fixed-threshold operating-point evidence and protocol-faithful OOF reporting.

Do not use for: proving clinical safety or global ranking superiority.

### Full ECG-RAMBA vs MiniRocket-only

MiniRocket-only baseline:

- ROC-AUC macro: 0.9169
- PR-AUC macro: 0.4508
- F1 macro: 0.2477
- Brier macro: 0.1906
- ECE macro: 0.3759

Paired comparison interpretation:

- MiniRocket-only is significantly better for PR-AUC and ROC-AUC.
- Full ECG-RAMBA is significantly better for F1 macro, Brier macro, and ECE macro.

Manuscript implication:

- Do not write that ECG-RAMBA outperforms MiniRocket overall.
- Write that ECG-RAMBA improves fixed-threshold and calibration-sensitive operating metrics, whereas MiniRocket-only gives stronger rank-based discrimination.

### HRV-only and HRV Domain Sensitivity

HRV-only baseline:

- ROC-AUC macro: 0.8113
- PR-AUC macro: 0.1771
- F1 macro: 0.2077

HRV domain classifier:

- ROC-AUC OVR macro: 0.9999999
- PR-AUC macro: 0.9999998
- Balanced accuracy: 0.9996

Manuscript implication:

- HRV has useful rhythm signal but is highly domain-sensitive.
- Do not describe HRV as an invariant anchor.
- Do not describe the current HRV36 vector as full RMSSD/SDNN/LF-HF HRV. The A0 blocker says current checkpoint input has 5 RR statistics, 20 reserved zero slots, 5 intended amplitude slots, and 6 global statistics; amplitude contribution cannot be claimed for current checkpoints.

### Robustness Stress Tests

Stresses completed:

- `snr20db`
- `snr10db`
- `snr5db`
- `random_3_lead_dropout`
- `precordial_dropout`
- `resample_250hz`

Rows: 30 rows, 6 stresses x 5 metrics.

Metric-specific counts:

- Degradation comparison: MiniRocket-only is significantly less degraded in 26/30 rows; Full ECG-RAMBA is significantly less degraded in 4/30 rows.
- Stressed operating-point comparison: Full ECG-RAMBA is significantly better under stress in 18/30 rows; MiniRocket-only is significantly better under stress in 12/30 rows.

Pattern:

- Full ECG-RAMBA is better under stress for F1, Brier, and ECE in all six stress tests.
- MiniRocket-only is better under stress for PR-AUC and ROC-AUC in all six stress tests.
- Full ECG-RAMBA degrades less only for Brier/ECE under `snr10db` and `snr5db`.

Manuscript implication:

- Do not claim that ECG-RAMBA is generally more robust.
- It is valid to claim a stress-test operating-point advantage for fixed-threshold/calibration metrics.
- It is also necessary to acknowledge that MiniRocket-only is less degraded for most ranking metrics and remains better for PR-AUC/ROC-AUC under stress.

### Power Mean Pooling

Q=3 frozen operating point:

- PR-AUC macro: 0.3432
- ROC-AUC macro: 0.8373
- F1 macro: 0.3998

Manuscript implication:

- Present Q=3 as the pre-specified/frozen operating point and sensitivity-tested tradeoff.
- Do not claim Q=3 is globally optimal.

## Alignment With Reviewer Discussion Plan

| Discussion item | Original plan in revised discussion | Final evidence status | Required update |
|---|---|---|---|
| Safety / ranking-decision gap | Add ECE, MCE, Brier, reliability diagram, operating-point metrics. | Supported with limitations. Full has ECE 0.0552 and Brier 0.0355; paired comparison supports Full for F1/Brier/ECE but not PR-AUC/ROC-AUC. | Replace safety wording with calibrated fixed-threshold operating-point wording. |
| MiniRocket / morphology baseline | Add MiniRocket-only, learned morphology, CNN/ResNet1D, CNN+HRV+Mamba. | MiniRocket-only and HRV-only are complete. Raw Mamba and ResNet1D/CNN fair runners remain TBD. MiniRocket beats Full on ranking metrics. | Do not claim fair-baseline superiority. Report the mixed result and keep Raw Mamba/ResNet/CNN as pending or future work unless completed later. |
| HRV domain bias | Add HRV-only and HRV domain classifier. | Complete, but domain classifier is near-perfect. | Use this as a limitation and domain-sensitivity finding. Remove HRV-invariance language. |
| Statistical CI | Add bootstrap CI and paired bootstrap. | Complete for calibration and MiniRocket paired comparison; robustness also has paired degradation CIs. | Use record-level bootstrap wording and cite tables. |
| Power Mean Q=3 | Add pooling sensitivity. | Supported as tradeoff, not optimality. | Write Q=3 as frozen/sensitivity-tested operating point only. |
| Robustness under noise/missing leads/sampling | Add stress tests. | Complete across six stresses. Results are mixed by metric family. | Use metric-specific robustness claims only. Full wins stressed F1/Brier/ECE; MiniRocket wins stressed PR-AUC/ROC-AUC and less-degradation in most rows. |
| Morphology-rhythm disentanglement | Add UMAP/t-SNE, probing, CKA/similarity. | Blocked. No completed representation artifact. | Do not claim proven disentanglement. Say architecture is designed to combine complementary streams; representation separation remains future work. |
| External transfer PTB/Georgia/CPSC | Improve mapping and protocol. | OOF supported; external remains experimental. | Keep external results out of manuscript-ready claims unless dataset-specific protocol gates are completed. |

## Safe Manuscript / Rebuttal Wording

### Calibration and fixed-threshold behavior

Use:

> We agree that a ranking-decision gap alone is insufficient to justify safety-oriented claims. We therefore replaced the original safety wording with a calibration-aware operating-point analysis. In the frozen Chapman OOF protocol, ECG-RAMBA achieved macro ROC-AUC 0.8373, macro PR-AUC 0.3432, macro F1 0.3998, macro Brier score 0.0355, and macro ECE 0.0552 at threshold 0.5. These results support a calibrated fixed-threshold operating-point advantage, but not a broad clinical safety claim.

Avoid:

> ECG-RAMBA demonstrates clinically safe or safety-oriented behavior.

### Full ECG-RAMBA vs MiniRocket-only

Use:

> The MiniRocket-only baseline achieved stronger rank-based discrimination than the full model, with higher macro PR-AUC and ROC-AUC. However, ECG-RAMBA achieved substantially better fixed-threshold and calibration-sensitive metrics, including macro F1, Brier score, and ECE. We therefore report the comparison as a metric-dependent tradeoff rather than as global superiority.

Avoid:

> ECG-RAMBA outperforms MiniRocket-only.

### HRV domain sensitivity

Use:

> HRV-only features retained non-trivial rhythm-discrimination signal, but the HRV domain classifier also separated dataset sources almost perfectly. We therefore revised the discussion to treat HRV as a useful but domain-sensitive descriptor, not as an invariant anchor.

Avoid:

> HRV serves as an invariant physiological anchor.

### Robustness

Use:

> Across six perturbation settings, ECG-RAMBA was consistently better than MiniRocket-only for stressed fixed-threshold/calibration metrics such as F1, Brier score, and ECE. In contrast, MiniRocket-only remained stronger for stressed PR-AUC and ROC-AUC and was less degraded in most ranking-based comparisons. We therefore restrict robustness claims to metric-specific operating-point robustness rather than general robustness.

Avoid:

> ECG-RAMBA is more robust than MiniRocket.

### Morphology-rhythm separation

Use:

> The current experiments support complementary behavior of morphology and rhythm streams but do not prove strict morphology-rhythm disentanglement. Representation-level probing or embedding analyses are required before making a stronger claim.

Avoid:

> The experiments validate morphology-rhythm disentanglement.

### External datasets

Use:

> The current revision establishes a frozen, traceable Chapman OOF protocol. External PTB/Georgia/CPSC outputs are treated as experimental until dataset-specific label, window, annotation, and fold-PCA protocol checks are completed.

Avoid:

> The current external results are manuscript-ready evidence of zero-shot transfer.

## Blockers That Must Stay Visible

The final evidence package records these restrictions:

- `A0-HRV-01`: no full-HRV or invariant-anchor claim is allowed.
- `A0-HRV-02`: current checkpoints cannot support a causal amplitude-feature claim.
- `A0-PTB-01`: PTB outputs remain experimental because predictions are Chapman-class proxies and HYP is unsupported.
- `A0-BASE-01`: fair-baseline superiority remains blocked until Raw Mamba and ResNet1D/CNN runners are implemented and completed under the same frozen protocol.
- `A0-PCA-01`: external outputs stay experimental until fold PCA artifacts and external protocol gates are reviewed.
- `A0-EXT-01`: all external outputs remain isolated as experimental with `manuscript_ready=false`.

## Recommended Manuscript Edits

### Abstract

Replace broad claims such as robust zero-shot generalization, safety-oriented behavior, and disentanglement validation with:

> ECG-RAMBA combines morphology and rhythm-oriented streams and is evaluated under a frozen, traceable Chapman OOF protocol. The revised analysis reports calibration, fixed-threshold operating metrics, paired baseline comparisons, pooling sensitivity, HRV-domain sensitivity, and perturbation stress tests.

### Results

Add or update result subsections:

1. Frozen OOF calibration and operating-point metrics.
2. Metric-dependent MiniRocket-only comparison.
3. HRV-only baseline and HRV domain sensitivity.
4. Q=3 pooling sensitivity.
5. Perturbation robustness with metric-specific claims.
6. Blocked/deferred external protocol status.

### Discussion

Add a paragraph explicitly acknowledging the mixed MiniRocket comparison:

> The comparison with MiniRocket-only reveals an important tradeoff. MiniRocket-only provides stronger rank-based discrimination, whereas the full ECG-RAMBA model provides better calibrated fixed-threshold operating behavior. This suggests that the added sequence and rhythm pathways do not simply improve all metrics, but reshape the decision behavior at the operating point used in the protocol.

Add a limitation paragraph:

> Several claims from the original manuscript have been narrowed. The HRV branch is domain-sensitive rather than invariant; external results are not treated as manuscript-ready until dataset-specific protocol checks are complete; and representation-level disentanglement remains future work.

### Rebuttal

Use the final evidence matrix as the source of truth. If a reviewer asks whether all planned baselines are complete, answer directly:

> We completed MiniRocket-only and HRV-only feature baselines under the frozen OOF protocol and added paired comparisons against MiniRocket-only. We do not claim broad fair-baseline superiority because Raw Mamba and ResNet1D/CNN fair runners remain deferred. The revised manuscript has been adjusted accordingly.

## Final Writing Rule

Every strong sentence in the revised manuscript should fall into one of these categories:

1. Supported: calibration/fixed-threshold OOF, Q=3 as frozen tradeoff, HRV-only feature baseline, metric-specific robustness.
2. Supported with limitation: HRV as useful but domain-sensitive, Full vs MiniRocket as metric-dependent.
3. Not supported yet: strict disentanglement, global superiority, external zero-shot manuscript-ready claims, full HRV/RMSSD/SDNN/LF-HF implementation, causal amplitude contribution.

If a claim is in category 3, either remove it, move it to limitations/future work, or explicitly mark it as deferred.

