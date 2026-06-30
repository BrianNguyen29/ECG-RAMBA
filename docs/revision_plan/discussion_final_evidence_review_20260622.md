# ECG-RAMBA Discussion Final Evidence Review

Date: 2026-06-22

This note records the current review of `D:\WorkSpace\ECG\ECG-Ramba\docs\ECG_RAMBA_Reviewer_Discussion_Revised.docx` against the final evidence package under:

```text
D:\WorkSpace\ECG\ECG-Ramba\drive\final_evidence_tables
D:\WorkSpace\ECG\ECG-Ramba\drive\revision_artifacts\reports\revision
```

Generated updated discussion document:

```text
D:\WorkSpace\ECG\ECG-Ramba\docs\ECG_RAMBA_Reviewer_Discussion_Final_Evidence_20260622.docx
```

## Current Evidence Position

The old discussion document is a useful planning artifact, but it contains planned responses and strong wording that no longer match the completed evidence. The current source of truth is the final evidence package plus:

```text
docs/revision_plan/manuscript_rebuttal_drive_source_update_20260622.md
```

Deferred evidence planning is tracked separately in:

```text
docs/revision_plan/deferred_evidence_implementation_plan_20260622.md
```

The central decision is:

- ResNet1D/CNN is the stronger in-domain architecture baseline.
- ECG-RAMBA must not be claimed as benchmark-leading, broadly superior, or the leading in-domain classifier.
- MiniRocket-only shows a metric-specific tradeoff.
- HRV is useful but domain-sensitive.
- Robustness is metric-specific.
- Unqualified external-transfer and few-shot claims remain deferred. PTB-XL is usable only as protocol-gated mapped-task external evidence; Georgia/CPSC2021 remain deferred.

## Key Numbers

| Model | ROC-AUC | PR-AUC | F1 | Brier | ECE | Status |
|---|---:|---:|---:|---:|---:|---|
| Full ECG-RAMBA | 0.8373 | 0.3432 | 0.3998 | 0.0355 | 0.0552 | Complete frozen OOF |
| ResNet1D/CNN | 0.9391 | 0.4899 | 0.4906 | 0.0342 | 0.0414 | Complete fair architecture baseline |
| MiniRocket-only | 0.9169 | 0.4508 | 0.2477 | 0.1906 | 0.3759 | Complete feature baseline |
| HRV-only | 0.8113 | 0.1771 | 0.2077 | 0.1613 | 0.2844 | Complete feature baseline |
| Raw Mamba | 0.9284 | 0.4377 | 0.4465 | 0.0401 | 0.1137 | Complete fair comparator; stronger than Full on PR-AUC/ROC-AUC/F1, weaker on Brier/ECE |

## Discussion Mismatches Found

| Old discussion topic | Final evidence | Required update |
|---|---|---|
| External generalization as the central story | Unqualified external-transfer and few-shot outputs are not manuscript-ready. PTB-XL is only protocol-gated mapped-task evidence. | Reframe as protocol-faithful Chapman OOF structured model analysis. |
| Morphology-rhythm disentanglement | Representation probe/CKA is complete as a conservative audit, but linear probes are weak and do not establish label-aligned separation. | Use structured morphology-rhythm design, component-sensitivity language, and limited branch-embedding audit wording only. |
| Safety-oriented ranking-decision gap | Calibration exists, but ResNet is better on F1/Brier/ECE. | Use fixed-threshold operating-point wording only. |
| MiniRocket/deterministic morphology improves robustness | MiniRocket wins PR-AUC/ROC-AUC; Full wins F1/Brier/ECE only against MiniRocket. | State metric-specific MiniRocket tradeoff. |
| Learned morphology baseline as planned item | ResNet1D/CNN is complete and beats ECG-RAMBA. | Explicitly acknowledge ResNet is stronger in-domain. |
| HRV invariant anchor | HRV domain classifier is near-perfect. | State HRV is useful but domain-sensitive. |
| Robustness as general claim | Full wins stressed F1/Brier/ECE; MiniRocket wins stressed PR-AUC/ROC-AUC and degrades less in most rows. | Use metric-family robustness only. |
| Few-shot/external-transfer response | Deferred. | Move to future work unless separate protocol gates are completed. |

## Updated Reviewer Response Direction

### Reviewer 1

- R1C1: agree, remove safety language, report calibration and operating-point metrics.
- R1C2: report MiniRocket as a metric-specific baseline; do not generalize to ResNet/CNN.
- R1C3: HRV-only has signal, but HRV is highly domain-sensitive.
- R1C4: acknowledge ResNet1D/CNN is stronger in-domain across all five paired metrics; Raw Mamba is also stronger on PR-AUC/ROC-AUC/F1 but weaker on Brier/ECE.
- R1C5: use bootstrap/paired bootstrap only for metric-specific conclusions.
- R1C6: Q=3 is a frozen operating point and sensitivity-tested tradeoff.
- R1C7: use final EMA, artifact provenance, fixed threshold, Q=3, and corrected HRV36 schema.

### Reviewer 2

- R2C1: present the pipeline as data -> folds -> final EMA checkpoints -> slice probabilities -> Q=3 aggregation -> calibration/baseline/robustness evidence.
- R2C2: representation-level probe/CKA artifacts are complete, but they support only a conservative audit; do not claim established morphology-rhythm separation.
- R2C3: robustness claims must be metric-specific.
- R2C4: PTB-XL is limited to mapped-task external evidence; Georgia/CPSC2021 and few-shot transfer remain deferred.

## Next Items

| Priority | Item | Concrete action |
|---|---|---|
| P0 | Finalize manuscript text from current evidence | Completed in `D:\WorkSpace\ECG\ECG-Ramba\docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\main.tex`. The manuscript now states that ResNet1D/CNN is stronger in-domain, removes benchmark-leading/broad-advantage wording, and reframes ECG-RAMBA as structured model analysis. |
| P0 | Finalize response letter | Completed in `D:\WorkSpace\ECG\ECG-Ramba\docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\response_to_reviewers_revised_20260622.md`. Each response now includes manuscript section/table/figure references. |
| P0 | Compile and visually inspect IEEE PDF | Completed with Tectonic portable. Output: `D:\WorkSpace\ECG\ECG-Ramba\docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\build\main.pdf`; 10 pages; compile exit code 0. |
| P1 | Update or remove old external-transfer figures | Completed in current manuscript wording. PTB-XL is described only as protocol-gated mapped-task evidence; Georgia/CPSC/few-shot outputs remain deferred and are not used as primary manuscript claims. |
| P1 | Add artifact provenance appendix/table | Completed. Appendix Table `\ref{tab:provenance}` includes final EMA, Q=3, threshold 0.5, HRV schema restriction, artifact paths, and checksums. |
| P1 | Align figures with final claims | Completed for current manuscript captions and discussion. Per-class, lead-dropout, saliency, and ablation wording no longer imply safety, broad advantage, or established morphology-rhythm separation. |
| P2 | Run Raw Mamba runner | Completed. Canonical artifacts and paired comparison are available; use only comparator-specific wording. Do not claim superiority over all fair baselines. |
| P2 | Optional external/few-shot package | Deferred. Only run if adding a new transfer claim; it needs separate label/window/PCA/CI gates. |
| P2 | Representation probe | Complete as conservative evidence. Use only the limited branch-embedding audit wording; do not upgrade to a mechanistic separation claim. |

Detailed implementation planning for these deferred items is tracked in `docs/revision_plan/deferred_evidence_implementation_plan_20260622.md`.

## Rebuttal Completion Status

### Completed And Usable In The Response Letter

| Reviewer issue | Current status | Evidence / manuscript location |
|---|---|---|
| Overclaim and safety wording | Complete. Safety-oriented language was removed and replaced by fixed-threshold operating-point wording. | Abstract; Table `\ref{tab:in_distribution}`; Discussion subsection `Operating-Point and Calibration Behavior`; calibration artifact `calibration_ci_oof_final_ema_predictions.json`. |
| Calibration and uncertainty | Complete. Final OOF calibration reports Brier 0.0355, ECE 0.0552, MCE 0.5995, plus bootstrap CIs. | Table `\ref{tab:in_distribution}`; Appendix Table `\ref{tab:provenance}`; `reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json`. |
| ResNet1D/CNN fair baseline | Complete. ResNet1D/CNN is stronger than ECG-RAMBA on PR-AUC, ROC-AUC, F1, Brier, and ECE under the frozen Chapman OOF protocol. | Table `\ref{tab:in_distribution}`; Discussion subsection `Fair Baselines and Claim Boundaries`; `table_paired_full_vs_resnet.csv`. |
| MiniRocket-only fair feature baseline | Complete. MiniRocket-only is stronger for PR-AUC/ROC-AUC; ECG-RAMBA is stronger than MiniRocket-only for F1/Brier/ECE at the frozen threshold. | Table `\ref{tab:in_distribution}`; `table_paired_full_vs_minirocket.csv`. |
| HRV-only and HRV domain sensitivity | Complete with limitation. HRV-only has signal, but HRV domain classifier is near-perfect; therefore HRV must be described as domain-sensitive. | Discussion subsection `HRV as Useful but Domain-Sensitive`; `hrv_only_baseline_summary.json`; `hrv_domain_classifier_summary.json`. |
| Power Mean Q=3 | Complete with limitation. Q=3 is a frozen/sensitivity-tested operating point, not globally optimal. | Methods/aggregation section; Discussion subsection `Operating-Point and Calibration Behavior`; `pooling_decision_summary.json`. |
| Perturbation robustness | Complete with limitation. Six stress settings and 30 metric rows are available; claims are metric-specific only. | Discussion subsection `Metric-Specific Robustness`; `table_final_robustness_claims.csv`; `robustness_summary.csv`. |
| Figure/caption overclaim audit | Complete for current manuscript. Lead dropout, saliency, and ablation are now component-sensitivity or qualitative checks, not proof of disentanglement. | Section `Spatial and Temporal Behavior Checks`; Figures `\ref{fig:lead_dropout}`, `\ref{fig:saliency}`, `\ref{fig:ablation}`. |
| Artifact provenance | Complete. The appendix records final EMA, Q=3, threshold 0.5, HRV schema restriction, paths, and checksums. | Appendix Table `\ref{tab:provenance}`. |
| IEEE PDF build | Complete. PDF compiled successfully after switching the IEEE class logo include from EPS to PDF for local Tectonic compatibility. | `build\main.pdf`; `build\main.log`; `build\tectonic_final_stdout.log`. |

### Not Completed / Deferred And Must Not Be Claimed

| Item | Status | Rebuttal restriction |
|---|---|---|
| Raw Mamba fair comparator | Complete. | Do not claim superiority over all fair baselines. State that Raw Mamba is stronger on PR-AUC/ROC-AUC/F1, while ECG-RAMBA has lower Brier/ECE. |
| PTB-XL mapped-task external evaluation | Complete with restriction. | Use only "protocol-gated mapped-task external evaluation" wording. Do not claim zero-shot generalization, external superiority, or cross-dataset robustness. |
| Georgia / CPSC2021 manuscript-ready transfer | Deferred. | Keep Georgia/CPSC2021 external outputs deferred until dataset-specific mapping/annotation gates pass. Do not coerce unmapped labels or incomplete annotations into negatives. |
| Few-shot adaptation | Deferred. | Do not say few-shot experiments were added. Present few-shot as future protocol-specific work unless a separate evidence package is completed. |
| Mechanistic morphology-rhythm separation | Not established. | Probe/CKA artifacts are complete, but weak linear probes mean the evidence supports only a conservative branch-embedding audit plus component-sensitivity and qualitative behavior. |
| Full HRV feature set | Not supported by current checkpoints. | Do not claim RMSSD, SDNN, LF/HF, complete HRV spectrum, or amplitude-feature contribution. Use the HRV36 schema restriction from Appendix Table `\ref{tab:provenance}`. |
| General robustness superiority | Not supported. | Use only metric-specific robustness wording: ECG-RAMBA has stressed fixed-threshold/calibration advantages against MiniRocket-only, while MiniRocket remains stronger for stressed ranking metrics and is less degraded in most rows. |

## Current Recommendation

Proceed with a conservative resubmission package centered on protocol-faithful Chapman OOF evidence and honest baseline interpretation. Do not reopen broad experiments until the current manuscript and response letter are internally consistent.
