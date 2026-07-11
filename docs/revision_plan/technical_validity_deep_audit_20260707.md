# Technical Validity Deep Audit - 2026-07-07

This memo audits whether the current reviewer-response evidence package is technically valid, reproducible, and appropriately worded. It cross-checks local artifacts against peer-reviewed papers, official dataset pages, and clinical-AI reporting guidance.

## Source Of Truth

Use these generated tables as the current source of truth:

- `drive/final_evidence_tables/table_final_evidence_matrix.csv`
- `drive/final_evidence_tables/table_final_safe_wording.csv`
- `drive/final_evidence_tables/table_claim_readiness_gates.csv`
- `drive/final_evidence_tables/external_protocol_gate_summary.csv`

Current final-evidence state:

- Frozen Chapman OOF evidence is complete and protocol-traceable.
- Fair-baseline evidence is complete for MiniRocket-only, ResNet1D/CNN, and Raw Mamba, but it is comparator-specific and does not support broad superiority.
- External mapped-task gates pass for PTB-XL, Georgia, and CPSC2021.
- Few-shot artifacts exist for PTB-XL, Georgia, and CPSC2021, but they are score-calibration-only on frozen external predictions.
- Representation probe/CKA artifacts exist, but they support only a conservative branch-embedding audit, not mechanistic disentanglement.
- Claim-readiness gates still block transformer/foundation baseline, hybrid morphology sensitivity, learned-comparator robustness, full HRV feature-set claims, clinical deployment readiness, and broad in-domain/global superiority.

## External Dataset Protocols

### PTB-XL

Official PhysioNet and the PTB-XL paper describe PTB-XL as a large clinical 12-lead, 10-second ECG dataset with SCP-ECG diagnostic/form/rhythm statements and recommended splits. This supports using PTB-XL only under explicit mapped-task definitions, not as an unqualified external superiority test.

Implementation check:

- `scripts/revision/03_generate_external_predictions.py` records unsupported PTB-XL superclass handling and keeps the output under `reports/revision/experimental/external`.
- `scripts/revision/18_external_protocol_gate.py` validates the external artifacts, label mapping, metrics, bootstrap CI, and `evidence_status=experimental`.
- Final evidence correctly says PTB-XL is a protocol-gated mapped-task external evaluation, not zero-shot superiority.

### Georgia / PhysioNet Challenge 2020

The PhysioNet/CinC 2020 challenge used multi-source 12-lead ECGs with SNOMED-CT labels. Therefore, Georgia evaluation must be label-mapping-reviewed. Unmapped labels must not be silently treated as negative labels.

Implementation check:

- `03_generate_external_predictions.py` has `--georgia-mapping-review` and `--georgia-code-inventory-out`.
- The loader counts unmapped SNOMED codes and explicitly blocks the "coerce unmapped records to negative" failure mode.
- `18_external_protocol_gate.py` includes `georgia_mapping_inventory` in schema-v3 gate artifacts.
- Current source-of-truth shows Georgia gate passed with `n_records=9458`, PR-AUC `0.3220`, ROC-AUC `0.8363`, F1 `0.2427`, Brier `0.0614`, ECE `0.0736`.

Verdict: Georgia is now usable as protocol-gated mapped-task evidence, but not as a general external superiority claim because there is no matched external baseline superiority package.

### CPSC2021

Official CPSC2021 materials define a PAF event-detection challenge on dynamic ECG recordings with episode-oriented scoring. Our current CPSC2021 evaluation is not the official challenge score; it is an annotation-aligned 10-second majority-rhythm window task.

Implementation check:

- `03_generate_external_predictions.py` records `annotation_aligned_nonoverlapping_10s_windows_majority_af_or_normal`.
- The output summary explicitly states that CPSC2021 is evaluated as annotation-aligned windows, not official episode-boundary scoring.
- `18_external_protocol_gate.py` requires the CPSC annotation audit table and schema-v3 gate metadata.
- Current source-of-truth shows CPSC2021 gate passed with `n_records=66775` windows, PR-AUC `0.9855`, ROC-AUC `0.9128`, F1 `0.3443`, Brier `0.3304`, ECE `0.5015`.

Verdict: CPSC2021 is usable only as protocol-gated mapped-window evidence. It must not be described as official CPSC2021 challenge scoring.

## Few-Shot / Adaptation Validity

The few-shot runner is technically a post-hoc score-calibration gate, not weight fine-tuning:

- `scripts/revision/19_fewshot_adaptation.py` declares `fewshot_score_calibration_v1_gated_external`.
- It loads frozen external prediction NPZ files and validates finite probabilities in `[0, 1]`.
- It uses per-class logistic score calibration where target-domain labels are available.
- The split policy freezes one test split per seed and uses nested prefixes of the remaining target-domain pool.
- The manifest records `adaptation_kind=score_calibration_only` and states that ECG-RAMBA weights are unchanged.

Current few-shot evidence:

- PTB-XL: zero-shot PR-AUC `0.6116`, ROC-AUC `0.8025`, F1 `0.4684`; best fixed-threshold F1 `0.6160` at 10 percent target labels; F1 gain `+0.1477`.
- Georgia: zero-shot PR-AUC `0.3298`, F1 `0.2425`; best fixed-threshold F1 `0.2847` at 1 percent target labels; F1 gain `+0.0423`.
- CPSC2021: zero-shot PR-AUC `0.9855`, F1 `0.3437`; best fixed-threshold F1 `0.7256` at 1 percent target labels; F1 gain `+0.3818`.

Important interpretation:

- Score calibration can improve fixed-threshold F1 while leaving ranking metrics unchanged or nearly unchanged.
- Therefore this is evidence for dataset-specific operating-point calibration, not general few-shot transfer and not fine-tuning.

## Baseline And In-Domain Claims

MiniRocket is an appropriate deterministic/feature baseline because the original MiniRocket work is an almost deterministic ROCKET reformulation for fast time-series classification. However, the local evidence shows MiniRocket-only is stronger than ECG-RAMBA on rank-based discrimination, while ECG-RAMBA is stronger on MiniRocket fixed-threshold/calibration metrics.

ResNet1D/CNN and Raw Mamba are stronger fair in-domain comparators:

- Full ECG-RAMBA: PR-AUC `0.3432`, F1 `0.3998`.
- MiniRocket-only: PR-AUC `0.4508`, F1 `0.2477`.
- ResNet1D/CNN: PR-AUC `0.4899`, F1 `0.4906`.
- Raw Mamba: PR-AUC `0.4377`, F1 `0.4465`.

Verdict:

- Do not claim SOTA, best in-domain performance, or broad/global superiority.
- Use comparator-specific and metric-specific wording only.
- The scientific contribution should be framed as structured model analysis, calibration/operating-point tradeoffs, and protocol-audited evidence generation.

## Calibration, Brier, And Bootstrap

The calibration evidence is methodologically appropriate for probability outputs:

- Brier score is a standard probabilistic forecast error measure.
- ECE/Brier reporting is motivated by calibration literature showing modern neural networks can be miscalibrated.
- Bootstrap CI is appropriate for empirical uncertainty when analytic variance formulas are inconvenient.
- Paired bootstrap comparisons are the correct direction for same-record OOF comparisons because the two models are evaluated on the same records.

Implementation check:

- `18_external_protocol_gate.py` computes bootstrap CIs for PR-AUC, ROC-AUC, F1, Brier, and ECE.
- Paired comparison scripts validate input contracts: same labels, record IDs, folds/classes, and frozen OOF SHA.
- Current final evidence separates rank metrics from fixed-threshold/calibration metrics, which is technically important.

## Robustness

Current robustness evidence supports only metric-specific Full ECG-RAMBA vs MiniRocket-only claims. It does not establish general robustness:

- MiniRocket remains stronger on PR-AUC/ROC-AUC under many stresses.
- ECG-RAMBA is stronger against MiniRocket-only on fixed-threshold/calibration-style stressed metrics.
- Learned-comparator robustness is still blocked because ResNet/Raw Mamba stress prediction comparisons are missing.

Implementation status:

- `scripts/revision/23_generate_comparator_stress_predictions.py` and `21_robustness_multicomparator.py` exist.
- `table_claim_readiness_gates.csv` still reports `robustness_learned_comparators=blocked_missing_learned_comparator_stress_evidence`.

Verdict: do not claim general robustness superiority. Run learned-comparator stress predictions before any broader robustness claim.

## Representation / CKA

CKA is a valid representation-similarity audit method, but it is not mechanistic proof. The local probes are weak:

- Best probe ROC-AUC: `0.6491` for fused/rhythm labels.
- Morphology-to-morphology ROC-AUC: `0.4707`, PR-AUC `0.0324`.
- Rhythm-to-rhythm ROC-AUC: `0.4853`, PR-AUC `0.0934`.
- CKA morphology/rhythm: `0.2293`; max CKA: `0.4586` for context/rhythm.

Verdict:

- It is valid to say the branches are not identical and were audited.
- It is not valid to say morphology-rhythm disentanglement is proven.

## HRV

Current HRV evidence is useful but limited:

- HRV-only ROC-AUC `0.8113`, PR-AUC `0.1771`, F1 `0.2077`.
- HRV domain classifier ROC-AUC is near 1.0, indicating strong domain sensitivity.
- Current checkpoint contains reserved/zero HRV slots rather than a complete RMSSD/SDNN/LF-HF schema.

Verdict:

- Do not claim a full HRV feature set.
- Do not claim HRV domain invariance.
- A true HRV feature-set claim requires retraining all folds with a revised feature schema.

## Clinical Readiness

TRIPOD+AI and PROBAST+AI emphasize transparent reporting, validation, risk-of-bias, and applicability for prediction models. CONSORT-AI applies to prospective clinical trials/interventions. The current ECG-RAMBA evidence is retrospective model-evaluation evidence only.

Verdict:

- Do not claim clinical deployment readiness, clinical safety readiness, or prospective utility.
- If needed, the next evidence package must define a clinical threshold target, prospective/external clinical validation, and utility/decision-curve style analysis.

## Implementation Quality Checks Completed

Local syntax checks passed for the current revision scripts:

- `03_generate_external_predictions.py`
- `18_external_protocol_gate.py`
- `19_fewshot_adaptation.py`
- `20_representation_probe.py`
- `21_robustness_multicomparator.py`
- `23_generate_comparator_stress_predictions.py`
- `24_transformer_ecg_baseline.py`
- `25_paired_full_vs_transformer.py`
- `26_hybrid_morphology_baseline.py`
- `27_paired_full_vs_hybrid_morphology.py`
- `28_claim_readiness_gates.py`
- `13_final_evidence_matrix.py`

Forbidden-claim scan on `main.tex`, the response letter, and final evidence tables found no literal hits for:

- `SOTA`
- `state-of-the-art`
- `zero-shot superiority`
- `global superiority`
- `proven disentanglement`
- `clinical readiness`
- `best in-domain`
- `general robustness superiority`
- `few-shot fine-tuning`
- `external superiority`

## Final Technical Verdict

The current implementation is technically sound for conservative reviewer-response claims:

- frozen Chapman OOF evaluation;
- protocol-gated mapped-task external evaluation for PTB-XL, Georgia, and CPSC2021;
- dataset-specific score calibration on frozen external predictions;
- metric-specific fair-baseline comparisons;
- HRV/domain-sensitivity limitation;
- representation audit with disentanglement limitation;
- explicit blocker status for missing or contradicted claims.

The current implementation is not sufficient for:

- broad in-domain/global superiority;
- SOTA claims;
- general zero-shot/few-shot superiority;
- official CPSC2021 challenge performance;
- general robustness across learned comparators;
- full HRV feature semantics;
- mechanistic morphology-rhythm disentanglement;
- clinical deployment readiness.

## References

- PTB-XL official PhysioNet page: https://physionet.org/content/ptb-xl/
- PTB-XL Scientific Data paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7248071/
- PhysioNet/CinC Challenge 2020 official page: https://moody-challenge.physionet.org/2020/
- PhysioNet Challenge 2020 paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8015789/
- CPSC2021 official PhysioNet page: https://physionet.org/content/cpsc2021/1.0.0/
- CPSC2021 official challenge page: https://icbeb2021.pastconf.com/CPSC2021
- MiniRocket paper: https://arxiv.org/abs/2012.08791
- Mamba paper: https://arxiv.org/abs/2312.00752
- Calibration paper: https://arxiv.org/abs/1706.04599
- CKA paper: https://arxiv.org/abs/1905.00414
- Bootstrap reference: https://www.hms.harvard.edu/bss/neuro/bornlab/nb204/statistics/bootstrap.pdf
- Brier score original paper: https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml
- TRIPOD+AI statement: https://www.bmj.com/content/385/bmj-2023-078378
- PROBAST+AI statement: https://www.bmj.com/content/388/bmj-2024-082505
- CONSORT-AI extension: https://www.nature.com/articles/s41591-020-1034-x
