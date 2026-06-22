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

## Re-Review Findings

This file was rechecked against the final evidence CSV/JSON artifacts on 2026-06-20. The source package is internally consistent:

- `final_ready_for_rebuttal=true`, so the package can support a rebuttal draft.
- `all_claims_supported=false`, so the manuscript must explicitly narrow or defer unsupported claims.
- `missing_inputs=[]`, `contract_issues=[]`, and `unresolved_blockers=[]`, so the final matrix was generated from a complete artifact set.
- The final evidence package is the source of truth. The lightweight local `reports/revision` folder in the repository does not contain all large Colab-generated artifacts.
- Numeric CI values should only be quoted when they are present in the final evidence tables. In this downloaded package, robustness paired CIs are available in `table_final_robustness_claims.csv`; the clean Full-vs-MiniRocket paired comparison is summarized by status and key numbers in `table_final_evidence_matrix.csv`.

The most important technical distinction is between:

1. **Rank-based discrimination**: PR-AUC and ROC-AUC. MiniRocket-only is stronger here.
2. **Fixed-threshold/calibration operating behavior**: F1 at threshold 0.5, Brier, and ECE. Full ECG-RAMBA is stronger than MiniRocket-only here, but the completed ResNet1D/CNN paired comparison is stronger than Full on F1, Brier, and ECE.
3. **Relative degradation under perturbation**: which model loses less from clean to stressed conditions. MiniRocket-only is less degraded in most rows, so the manuscript cannot claim general robustness superiority for ECG-RAMBA.

## High-Level Decision

The revised discussion should not claim global superiority, strict disentanglement, HRV invariance, or external-dataset readiness.

The strongest supported position is:

> ECG-RAMBA provides a protocol-faithful frozen Chapman OOF evaluation with metric-specific behavior. It has a calibrated fixed-threshold operating-point advantage over the MiniRocket-only feature baseline, while MiniRocket-only remains stronger for rank-based discrimination. However, a completed ResNet1D/CNN baseline significantly outperforms ECG-RAMBA on PR-AUC, ROC-AUC, F1, Brier, and ECE, so the paper must not claim in-domain fair-baseline superiority. Robustness claims must remain metric-specific: ECG-RAMBA is usually better than MiniRocket-only at the stressed fixed-threshold operating point for F1, Brier, and ECE, but MiniRocket-only often degrades less and remains stronger for PR-AUC/ROC-AUC.

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

### Full ECG-RAMBA vs ResNet1D/CNN

ResNet1D/CNN baseline:

- ROC-AUC macro: 0.9391
- PR-AUC macro: 0.4899
- F1 macro: 0.4906
- Brier macro: 0.0342
- ECE macro: 0.0414

Full ECG-RAMBA:

- ROC-AUC macro: 0.8373
- PR-AUC macro: 0.3432
- F1 macro: 0.3998
- Brier macro: 0.0355
- ECE macro: 0.0552

Paired Full-vs-ResNet result:

- ResNet1D/CNN is significantly better for PR-AUC, ROC-AUC, F1, Brier, and ECE.
- Full-minus-ResNet improvement CIs are negative for all five metrics: PR-AUC `[-0.1607, -0.1392]`, ROC-AUC `[-0.1127, -0.0909]`, F1 `[-0.1016, -0.0796]`, Brier-improvement `[-0.0017, -0.0010]`, and ECE-improvement `[-0.0143, -0.0134]`.
- This invalidates any broad in-domain claim that ECG-RAMBA is stronger than fair architecture baselines.

Manuscript consequence:

- Do not claim ECG-RAMBA has global or in-domain fair-baseline superiority.
- Do not generalize the MiniRocket-only fixed-threshold/calibration advantage to ResNet1D/CNN.
- If preserving ECG-RAMBA as a contribution, frame it as a structured model characterization and as a hypothesis for external/few-shot transfer, which still requires separate evidence.

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
| MiniRocket / morphology baseline | Add MiniRocket-only, learned morphology, CNN/ResNet1D, CNN+HRV+Mamba. | MiniRocket-only, HRV-only, and ResNet1D/CNN are complete. MiniRocket beats Full on ranking metrics; ResNet1D/CNN beats Full on PR-AUC, ROC-AUC, F1, Brier, and ECE. Raw Mamba remains TBD. | Do not claim fair-baseline superiority. Report the MiniRocket mixed result and the ResNet in-domain loss explicitly. |
| HRV domain bias | Add HRV-only and HRV domain classifier. | Complete, but domain classifier is near-perfect. | Use this as a limitation and domain-sensitivity finding. Remove HRV-invariance language. |
| Statistical CI | Add bootstrap CI and paired bootstrap. | Complete for calibration and MiniRocket paired comparison; robustness also has paired degradation CIs. | Use record-level bootstrap wording and cite tables. |
| Power Mean Q=3 | Add pooling sensitivity. | Supported as tradeoff, not optimality. | Write Q=3 as frozen/sensitivity-tested operating point only. |
| Robustness under noise/missing leads/sampling | Add stress tests. | Complete across six stresses. Results are mixed by metric family. | Use metric-specific robustness claims only. Full wins stressed F1/Brier/ECE; MiniRocket wins stressed PR-AUC/ROC-AUC and less-degradation in most rows. |
| Morphology-rhythm disentanglement | Add UMAP/t-SNE, probing, CKA/similarity. | Blocked. No completed representation artifact. | Do not claim proven disentanglement. Say architecture is designed to combine complementary streams; representation separation remains future work. |
| External transfer PTB/Georgia/CPSC | Improve mapping and protocol. | OOF supported; external remains experimental. | Keep external results out of manuscript-ready claims unless dataset-specific protocol gates are completed. |

## Safe Manuscript / Rebuttal Wording

### Calibration and fixed-threshold behavior

Use:

> We agree that a ranking-decision gap alone is insufficient to justify safety-oriented claims. We therefore replaced the original safety wording with a calibration-aware operating-point analysis. In the frozen Chapman OOF protocol, ECG-RAMBA achieved macro ROC-AUC 0.8373, macro PR-AUC 0.3432, macro F1 0.3998, macro Brier score 0.0355, and macro ECE 0.0552 at threshold 0.5. These results support reporting ECG-RAMBA's operating behavior, but not a broad clinical safety claim and not superiority over ResNet1D/CNN, which is better on the paired in-domain metrics.

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
- `A0-BASE-01`: fair-baseline superiority remains blocked. ResNet1D/CNN is now completed/validated and significantly outperforms Full ECG-RAMBA in-domain; Raw Mamba remains deferred.
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

> We completed MiniRocket-only, HRV-only, and ResNet1D/CNN baselines under the frozen OOF protocol and added paired comparisons against MiniRocket-only and ResNet1D/CNN. ResNet1D/CNN significantly outperforms Full ECG-RAMBA on the in-domain Chapman OOF metrics. We therefore do not claim broad fair-baseline superiority; Raw Mamba also remains deferred.

## Writing Rule

Every strong sentence in the revised manuscript should fall into one of these categories:

1. Supported: calibration/fixed-threshold OOF, Q=3 as frozen tradeoff, HRV-only feature baseline, metric-specific robustness.
2. Supported with limitation: HRV as useful but domain-sensitive, Full vs MiniRocket as metric-dependent, and Full vs ResNet as an in-domain negative result for ECG-RAMBA superiority.
3. Not supported yet: strict disentanglement, global superiority, external zero-shot manuscript-ready claims, full HRV/RMSSD/SDNN/LF-HF implementation, causal amplitude contribution.

If a claim is in category 3, either remove it, move it to limitations/future work, or explicitly mark it as deferred.

## Verified Claim Matrix

This table is copied from `table_final_evidence_matrix.csv` and should drive the manuscript/rebuttal wording.

| Claim | Evidence status | Key numbers | Allowed use | Blocker |
|---|---|---|---|---|
| `C01` Fair baseline superiority / external transfer | `blocked_fair_baselines_missing` | Full PR-AUC=0.3432, F1=0.3998; MiniRocket PR-AUC=0.4508, F1=0.2477; ResNet1D/CNN PR-AUC=0.4899, F1=0.4906 | Do not claim superiority over all fair baselines. Report comparator-specific, metric-specific paired deltas; ResNet1D/CNN is stronger than Full on the completed in-domain architecture comparison. | Raw Mamba remains TBD; ResNet1D/CNN is complete and unfavorable to ECG-RAMBA in-domain superiority. |
| `C02` Fixed-threshold ranking-decision gap | `supported_with_limitations` | OOF F1=0.3998, PR-AUC=0.3432, ECE=0.0552, Brier=0.0355; paired MiniRocket F1/Brier/ECE favor Full; paired ResNet F1/Brier/ECE favor ResNet. | Frozen OOF supports only metric-specific operating-point statements. Do not say ECG-RAMBA has a general fixed-threshold/calibration advantage once ResNet is included. | None. |
| `C03` HRV feature evidence and domain sensitivity | `partially_supported_with_domain_limitation` | HRV-only ROC-AUC=0.8113, PR-AUC=0.1771, F1=0.2077; domain status=complete, domain AUC=1.0000. | Report HRV-only as a feature baseline. Present HRV as domain-sensitive and avoid domain-invariance wording. | Current HRV36 schema still contains reserved zero slots and no full RMSSD/SDNN/LF-HF claim. |
| `C04` Morphology-rhythm separation | `blocked_representation_probe_missing` | Robustness rows=30; Full less-degraded metrics=4; MiniRocket less-degraded metrics=26. | Do not claim proven morphology-rhythm disentanglement. State that the architecture is designed to combine complementary streams and that representation separation remains future work. | No completed UMAP/probing/CKA representation artifact. |
| `C05` Q=3 pooling operating point | `supported_as_tradeoff_not_optimality_claim` | Q=3 PR-AUC=0.3432, ROC-AUC=0.8373, F1=0.3998. | Present Q=3 as the pre-specified/frozen operating point and a sensitivity-tested tradeoff, not as globally optimal. | None. |
| `C06` Protocol-faithful OOF evaluation | `oof_supported_external_experimental` | A0 audit_complete=True; blockers=8; calibration n=44,186; micro ECE=0.0538. | Claim protocol-faithful frozen Chapman OOF evaluation. Keep PTB/Georgia/CPSC outputs experimental unless their dataset-specific protocols are separately completed. | Deferred blockers remain documented; protocol_ready is distinct from audit_complete. |

## Baseline Interpretation

The Full-vs-MiniRocket result is mixed, not uniformly positive for either side.

| Metric family | Winner / interpretation | Manuscript consequence |
|---|---|---|
| PR-AUC, ROC-AUC | MiniRocket-only is stronger for rank-based discrimination. | Do not claim ECG-RAMBA is the better ranker. |
| F1 at threshold 0.5 | Full ECG-RAMBA is stronger than MiniRocket-only but weaker than ResNet1D/CNN. | It is acceptable to discuss MiniRocket-specific fixed-threshold tradeoff, not general fair-baseline benefit. |
| Brier, ECE | Full ECG-RAMBA is substantially better calibrated. | It is acceptable to discuss calibration-sensitive operating behavior. |
| Broad fair-baseline superiority | Blocked. Raw Mamba remains TBD, and ResNet1D/CNN is completed but beats Full ECG-RAMBA on all five paired in-domain metrics. | Do not claim superiority over all fair baselines. |

Reviewer-facing wording should explicitly mention this tradeoff. Hiding MiniRocket's better PR-AUC/ROC-AUC would be scientifically weak and likely to be flagged.

## Robustness Detail With Paired CIs

Definitions used below:

- `Full less-degradation advantage`: positive means Full ECG-RAMBA degraded less than MiniRocket-only from clean to stressed condition; negative means MiniRocket-only degraded less.
- `Full stressed-performance advantage`: positive means Full ECG-RAMBA had the better stressed metric value; negative means MiniRocket-only had the better stressed metric value.
- For Brier and ECE, lower is better, but the final table has already converted the direction so positive stressed-performance advantage favors Full.

Summary by stress:

| Stress | Full less degraded | MiniRocket less degraded | Full better under stress | MiniRocket better under stress |
|---|---:|---:|---:|---:|
| `precordial_dropout` | 0 | 5 | 3 | 2 |
| `random_3_lead_dropout` | 0 | 5 | 3 | 2 |
| `resample_250hz` | 0 | 5 | 3 | 2 |
| `snr10db` | 2 | 3 | 3 | 2 |
| `snr20db` | 0 | 5 | 3 | 2 |
| `snr5db` | 2 | 3 | 3 | 2 |

Detailed rows:

| Stress | Metric | Full stress | Mini stress | Full less-degradation advantage, 95% CI | Full stressed-performance advantage, 95% CI | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `precordial_dropout` | `brier_macro` | 0.0692 | 0.2117 | -0.0126 [-0.0132, -0.0121] | 0.1425 [0.1420, 0.1430] | MiniRocket less degraded; Full better under stress. |
| `precordial_dropout` | `ece_macro` | 0.0829 | 0.3820 | -0.0217 [-0.0223, -0.0211] | 0.2991 [0.2985, 0.2996] | MiniRocket less degraded; Full better under stress. |
| `precordial_dropout` | `f1_macro` | 0.2103 | 0.2000 | -0.1418 [-0.1472, -0.1361] | 0.0103 [0.0067, 0.0141] | MiniRocket less degraded; Full better under stress. |
| `precordial_dropout` | `pr_auc_macro` | 0.1782 | 0.3055 | -0.0197 [-0.0284, -0.0117] | -0.1273 [-0.1399, -0.1227] | MiniRocket less degraded; MiniRocket better under stress. |
| `precordial_dropout` | `roc_auc_macro` | 0.7238 | 0.8187 | -0.0153 [-0.0322, -0.0007] | -0.0949 [-0.1126, -0.0781] | MiniRocket less degraded; MiniRocket better under stress. |
| `random_3_lead_dropout` | `brier_macro` | 0.0689 | 0.2138 | -0.0103 [-0.0108, -0.0098] | 0.1449 [0.1444, 0.1454] | MiniRocket less degraded; Full better under stress. |
| `random_3_lead_dropout` | `ece_macro` | 0.0877 | 0.3877 | -0.0207 [-0.0213, -0.0201] | 0.3000 [0.2995, 0.3006] | MiniRocket less degraded; Full better under stress. |
| `random_3_lead_dropout` | `f1_macro` | 0.2380 | 0.2141 | -0.1283 [-0.1338, -0.1228] | 0.0239 [0.0202, 0.0273] | MiniRocket less degraded; Full better under stress. |
| `random_3_lead_dropout` | `pr_auc_macro` | 0.1857 | 0.3511 | -0.0579 [-0.0675, -0.0476] | -0.1655 [-0.1732, -0.1612] | MiniRocket less degraded; MiniRocket better under stress. |
| `random_3_lead_dropout` | `roc_auc_macro` | 0.7304 | 0.8718 | -0.0618 [-0.0777, -0.0477] | -0.1414 [-0.1536, -0.1301] | MiniRocket less degraded; MiniRocket better under stress. |
| `resample_250hz` | `brier_macro` | 0.0403 | 0.1907 | -0.0047 [-0.0049, -0.0045] | 0.1504 [0.1500, 0.1509] | MiniRocket less degraded; Full better under stress. |
| `resample_250hz` | `ece_macro` | 0.0587 | 0.3760 | -0.0034 [-0.0036, -0.0032] | 0.3173 [0.3169, 0.3177] | MiniRocket less degraded; Full better under stress. |
| `resample_250hz` | `f1_macro` | 0.3740 | 0.2476 | -0.0258 [-0.0294, -0.0227] | 0.1264 [0.1220, 0.1308] | MiniRocket less degraded; Full better under stress. |
| `resample_250hz` | `pr_auc_macro` | 0.3171 | 0.4508 | -0.0261 [-0.0282, -0.0238] | -0.1337 [-0.1472, -0.1243] | MiniRocket less degraded; MiniRocket better under stress. |
| `resample_250hz` | `roc_auc_macro` | 0.8254 | 0.9169 | -0.0118 [-0.0180, -0.0049] | -0.0915 [-0.1029, -0.0801] | MiniRocket less degraded; MiniRocket better under stress. |
| `snr10db` | `brier_macro` | 0.0435 | 0.2067 | 0.0081 [0.0078, 0.0084] | 0.1632 [0.1628, 0.1636] | Full less degraded; Full better under stress. |
| `snr10db` | `ece_macro` | 0.0619 | 0.3963 | 0.0137 [0.0133, 0.0140] | 0.3344 [0.3339, 0.3348] | Full less degraded; Full better under stress. |
| `snr10db` | `f1_macro` | 0.3610 | 0.2362 | -0.0273 [-0.0320, -0.0231] | 0.1248 [0.1202, 0.1291] | MiniRocket less degraded; Full better under stress. |
| `snr10db` | `pr_auc_macro` | 0.3049 | 0.4353 | -0.0228 [-0.0304, -0.0155] | -0.1304 [-0.1400, -0.1250] | MiniRocket less degraded; MiniRocket better under stress. |
| `snr10db` | `roc_auc_macro` | 0.8186 | 0.9091 | -0.0109 [-0.0210, -0.0008] | -0.0905 [-0.1035, -0.0776] | MiniRocket less degraded; MiniRocket better under stress. |
| `snr20db` | `brier_macro` | 0.0409 | 0.1913 | -0.0047 [-0.0049, -0.0045] | 0.1504 [0.1500, 0.1508] | MiniRocket less degraded; Full better under stress. |
| `snr20db` | `ece_macro` | 0.0590 | 0.3778 | -0.0019 [-0.0022, -0.0017] | 0.3188 [0.3183, 0.3192] | MiniRocket less degraded; Full better under stress. |
| `snr20db` | `f1_macro` | 0.3708 | 0.2488 | -0.0302 [-0.0340, -0.0267] | 0.1220 [0.1175, 0.1265] | MiniRocket less degraded; Full better under stress. |
| `snr20db` | `pr_auc_macro` | 0.3149 | 0.4501 | -0.0276 [-0.0302, -0.0250] | -0.1352 [-0.1485, -0.1263] | MiniRocket less degraded; MiniRocket better under stress. |
| `snr20db` | `roc_auc_macro` | 0.8266 | 0.9165 | -0.0102 [-0.0175, -0.0024] | -0.0898 [-0.1020, -0.0780] | MiniRocket less degraded; MiniRocket better under stress. |
| `snr5db` | `brier_macro` | 0.0505 | 0.2368 | 0.0312 [0.0308, 0.0316] | 0.1863 [0.1859, 0.1868] | Full less degraded; Full better under stress. |
| `snr5db` | `ece_macro` | 0.0692 | 0.4235 | 0.0336 [0.0332, 0.0341] | 0.3543 [0.3538, 0.3548] | Full less degraded; Full better under stress. |
| `snr5db` | `f1_macro` | 0.3301 | 0.2071 | -0.0291 [-0.0340, -0.0240] | 0.1231 [0.1190, 0.1269] | MiniRocket less degraded; Full better under stress. |
| `snr5db` | `pr_auc_macro` | 0.2764 | 0.4137 | -0.0298 [-0.0389, -0.0171] | -0.1374 [-0.1436, -0.1325] | MiniRocket less degraded; MiniRocket better under stress. |
| `snr5db` | `roc_auc_macro` | 0.7962 | 0.8950 | -0.0191 [-0.0316, -0.0072] | -0.0987 [-0.1111, -0.0867] | MiniRocket less degraded; MiniRocket better under stress. |

Discussion consequence:

- The only defensible robustness claim is metric-specific: **Full ECG-RAMBA is better at the stressed fixed threshold and remains better calibrated under all six stresses.**
- The manuscript must also say that **MiniRocket-only is less degraded in most stress/metric rows and remains the better ranker under stress.**
- Avoid compressing this into "ECG-RAMBA is more robust"; that would contradict 26/30 degradation rows.

## Final Blocker Table

| Blocker | Status | Restriction |
|---|---|---|
| `A0-HRV-01` HRV feature names exceed implemented RR statistics | `manuscript-corrected` | No full-HRV or invariant-anchor claim is allowed. |
| `A0-HRV-02` Amplitude slots were zeroed by the Chapman training feature pipeline | `deferred` | Current checkpoints cannot support a causal amplitude-feature claim. |
| `A0-PTB-01` PTB superclass mapping differed between notebook and script | `resolved` | PTB outputs remain experimental because predictions are Chapman-class proxies and HYP is unsupported. |
| `A0-AGG-01` Mean and incorrect geometric-mean implementations conflicted with Power Mean Q=3 | `resolved` | Legacy record and fold caches without implementation/schema metadata are invalid. |
| `A0-DATA-01` Georgia archive was aliased as CPSC2021 | `resolved` | Never label Georgia-derived results as CPSC2021. |
| `A0-BASE-01` Baseline preprocessing and aggregation fairness | `deferred` | A0 completion records the deferral; baseline claims remain blocked until A3 completes. |
| `A0-PCA-01` Fold PCA provenance and external PCA mismatch | `deferred` | External outputs stay experimental/not_manuscript_ready until fold PCA artifacts and the remaining external protocol gates are reviewed. |
| `A0-EXT-01` External dataset label and window protocols require dataset-specific validation | `deferred` | All external outputs are isolated under reports/revision/experimental with `manuscript_ready=false`. |
