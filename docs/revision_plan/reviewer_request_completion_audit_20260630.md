# Reviewer Request Completion Audit - 2026-07-09

Scope: original decision letter and reviewer discussion file:

- `D:\WorkSpace\ECG\ECG-Ramba\Decision letter (Initial Submission.txt`
- `D:\WorkSpace\ECG\ECG-Ramba\ECG_RAMBA_Reviewer_Discussion.md`

Current source of truth:

- `D:\WorkSpace\ECG\ECG-Ramba\drive\final_evidence_tables\table_final_evidence_matrix.csv`
- `D:\WorkSpace\ECG\ECG-Ramba\drive\final_evidence_tables\table_final_safe_wording.csv`
- `D:\WorkSpace\ECG\ECG-Ramba\docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\main.tex`
- `D:\WorkSpace\ECG\ECG-Ramba\docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\response_to_reviewers_revised_20260622.md`

## Executive Status

The revised package now satisfies the most important reviewer requirements only after narrowing the original claims. The evidence supports a protocol-faithful structured model analysis, not a broad in-domain advantage, unqualified external-transfer advantage, clinical safety, full-HRV, or established morphology-rhythm separation claim.

## Completed Or Sufficiently Addressed

| Reviewer item | Current status | Evidence / manuscript basis | Safe interpretation |
|---|---|---|---|
| R1-C1 calibration and safety interpretation | Complete with claim downgrade | `calibration_ci_oof_final_ema_predictions.json`; reliability figure; Table `\ref{tab:in_distribution}`; Discussion `Operating-Point and Calibration Behavior` | Report fixed-threshold/calibration behavior only; no clinical safety claim. |
| R1-C2 deterministic morphology evidence | Substantially addressed | MiniRocket-only, ResNet1D/CNN, Raw Mamba, paired bootstrap tables | MiniRocket gives a metric-specific tradeoff; do not claim deterministic morphology is generally superior. |
| R1-C3 HRV domain bias | Complete with limitation | HRV-only baseline; HRV domain classifier; final evidence C03 | HRV contains useful signal but is highly domain-sensitive. |
| R1-C4 stronger baselines | Substantially addressed | ResNet1D/CNN and Raw Mamba complete; paired Full-vs-ResNet and Full-vs-Raw-Mamba comparisons | ResNet1D/CNN and Raw Mamba narrow the ECG-RAMBA claim; no broad fair-baseline advantage. |
| R1-C5 confidence intervals/statistical testing | Complete for frozen OOF and paired fair comparisons | Bootstrap CI and paired bootstrap artifacts for calibration, MiniRocket, ResNet, Raw Mamba | Use metric-specific paired intervals; avoid broad superiority wording. |
| R1-C6 Power Mean Q=3 | Complete with limitation | `pooling_sensitivity.csv`; `pooling_decision_summary.json` | Q=3 is a frozen, sensitivity-tested tradeoff, not a globally optimal exponent. |
| R1-C7 implementation details/reproducibility | Complete for current claim scope | Algorithm 1; Appendix provenance table; HRV schema restriction; artifact manifests | Current checkpoint protocol is reproducible, but full HRV feature claims are disallowed. |
| R2-C1 mathematical/pipeline clarity | Complete | Method equations and Algorithm 1 define fold-aware PCA/training/inference/aggregation | Pipeline clarity requirement is addressed. |
| R2-C2 morphology-rhythm representation evidence | Complete as conservative audit, not proof | Representation embeddings, fold-safe probes, CKA table, UMAP/PCA figures | Evidence suggests non-identical branch embeddings but does not establish label-aligned disentanglement. |
| R2-C3 robustness stress tests | Complete for Full vs MiniRocket-only | Six stresses x five metrics; `robustness_summary.csv`; `table_final_robustness_claims.csv` | Robustness claims are metric-specific vs MiniRocket-only; no general robustness superiority. |
| PTB-XL external evidence | Complete only as protocol-gated mapped task | `external_ptbxl_protocol_gate.json`; label mapping and metrics tables | PTB-XL can be cited only as mapped-task external evidence, not zero-shot/general external transfer. |
| Georgia external evidence | Complete only as protocol-gated mapped task | `external_georgia_protocol_gate.json`; reviewed mapping inventory; label mapping and metrics tables | Georgia can be cited only as mapped-task external evidence, not broad external-transfer superiority. |
| CPSC2021 external evidence | Complete only as annotation-aligned mapped-window task | `external_cpsc2021_protocol_gate.json`; annotation audit; label mapping and metrics tables | CPSC2021 can be cited only as 10-second AF/AFL mapped-window evidence, not official episode-boundary challenge scoring. |
| R2-C4 few-shot vs no-target-label external scenario | Complete for PTB-XL, Georgia, and CPSC2021 score calibration | `fewshot_ptbxl_summary.csv`; `fewshot_georgia_summary.csv`; `fewshot_cpsc2021_summary.csv`; bootstrap JSONs; split/run manifests; final evidence C06 | Report as leakage-audited score calibration on frozen mapped-task predictions. It is not model-weight fine-tuning and not general few-shot superiority. |

## Not Completed / Must Not Be Claimed

| Reviewer item | Current status | What is missing | Required action if authors want to complete it |
|---|---|---|---|
| Additional non-mapped external superiority claim | Not supported | PTB-XL/Georgia/CPSC2021 are mapped-task evaluations only | Add a new reviewed protocol and matched baselines before any broader external-transfer claim. |
| Few-shot model-weight adaptation | Not complete / intentionally not claimed | Current few-shot packages calibrate frozen scores only | Implement a separate leakage-audited head/fine-tuning protocol with matched baselines before claiming adaptation beyond score calibration. |
| Transformer/foundation ECG baseline | Runner and paired gate implemented / optional evidence not run by default | `scripts/revision/24_transformer_ecg_baseline.py`; `scripts/revision/25_paired_full_vs_transformer.py`; no completed Transformer artifacts yet | Run only if reviewer insists; otherwise state ResNet1D/CNN is the learned architecture baseline and avoid benchmark-leading claims. |
| Hybrid/partially learnable MiniRocket morphology configurations | Runner and paired gate implemented / optional evidence not run by default | `scripts/revision/26_hybrid_morphology_baseline.py`; `scripts/revision/27_paired_full_vs_hybrid_morphology.py` | Run only if pursuing the determinism-vs-regularization mechanism question further; interpret as sensitivity evidence only. |
| Robustness vs ResNet1D/CNN and Raw Mamba | Aggregation gate implemented / stress predictions deferred | `scripts/revision/21_robustness_multicomparator.py` and Notebook 05 ledger exist; missing ResNet/Raw-Mamba stress predictions remain blocked rows | Generate comparator-specific stressed predictions before making broader robustness claims; otherwise report only the existing metric-specific MiniRocket comparison. |
| Full HRV feature set | Not complete / retrain-level change | Current checkpoints use reserved zero-filled slots and do not implement RMSSD/SDNN/LF-HF | Implement true HRV slots and retrain all folds before claiming a full HRV feature set. |
| Mechanistic morphology-rhythm disentanglement | Not established | Probes are weak; CKA shows non-identical embeddings but not label-aligned separation | Do not claim this. More probing/controlled labels may support only suggestive branch-specific information. |
| Clinical deployment/safety readiness | Not complete | No clinical deployment validation, prospective study, or threshold calibration study | Keep out of claims; present as future work. |
| Broad in-domain or fair-baseline superiority | Contradicted by evidence | ResNet1D/CNN wins all five paired metrics; Raw Mamba wins PR-AUC/ROC-AUC/F1 | Do not claim superiority; report comparator-specific tradeoffs. |

## Reviewer Discussion Alignment

`ECG_RAMBA_Reviewer_Discussion.md` was updated on 2026-07-09 to align with the current final evidence package and the P0/P1/P2 execution split. Use it together with the final evidence tables, not as a substitute for regenerated Notebook 07 artifacts.

Current alignment checks:

- Completed PTB-XL/Georgia/CPSC2021 score-calibration packages are distinguished from broader few-shot transfer learning.
- Manuscript-facing wording uses no-target-label mapped-task/window audits and target-label score calibration, not zero-shot superiority.
- Transformer/foundation and Hybrid MiniRocket morphology runs are P1/P2 pending until artifacts exist.
- Learned-comparator robustness remains pending until Notebook 05 multi-comparator outputs exist and Notebook 07 regenerates the final tables.
- Representation evidence remains an audit/limitation, not a morphology-rhythm disentanglement proof.

Use `table_final_evidence_matrix.csv` and `table_final_safe_wording.csv` as the final source of truth.

## Practical Next Steps To Fully Address Remaining Reviewer Pressure

1. Before submission, do not add new broad claims. The current manuscript is claim-safe after PDF compile and forbidden-claim scan.
2. If reviewer pressure is mainly on few-shot/external transfer, use the completed PTB-XL/Georgia/CPSC2021 score-calibration results with mapped-task restrictions.
3. If reviewer pressure is mainly on baselines, consider a transformer/foundation baseline only if a fair, same-protocol runner can be implemented without changing the claim scope.
4. If reviewer pressure is mainly on robustness, extend stress tests to ResNet1D/CNN and Raw Mamba, then keep conclusions metric-specific.
5. If reviewer pressure is mainly on mechanism, do not try to upgrade the current representation audit into a disentanglement proof. Add a controlled probing package only as suggestive branch-specific evidence.
