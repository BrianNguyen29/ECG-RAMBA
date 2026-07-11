# ECG-RAMBA Manuscript/Rebuttal Update From Drive Final Evidence

Date: 2026-06-22; updated 2026-06-27 after Raw Mamba completion and final evidence regeneration

Purpose: use the verified Drive evidence package as the source of truth for rewriting the manuscript and rebuttal. This file supersedes any broader wording that predates the completed ResNet1D/CNN and Raw Mamba fair baselines.

## Source Of Truth

Primary Drive evidence folder:

```text
D:\WorkSpace\ECG\ECG-Ramba\drive\final_evidence_tables
```

Canonical mirrored artifacts:

```text
D:\WorkSpace\ECG\ECG-Ramba\drive\revision_artifacts\reports\revision
```

Local downloaded ZIP extraction note: the current local copy under `D:\WorkSpace\ECG\ECG-Ramba\drive\revision_artifacts` is nested one extra level. The verified local artifact root is:

```text
D:\WorkSpace\ECG\ECG-Ramba\drive\revision_artifacts\revision_artifacts\reports\revision
```

Required source files:

- `final_evidence_matrix.json`
- `final_evidence_matrix_manifest.json`
- `table_final_evidence_matrix.csv`
- `table_final_safe_wording.csv`
- `table_final_blocker_status.csv`
- `table_final_robustness_claims.csv`
- `metrics/baseline_summary.csv`
- `tables/table_paired_full_vs_resnet.csv`
- `tables/table_paired_full_vs_minirocket.csv`
- `tables/table_paired_full_vs_raw_mamba.csv`

Downloaded verification snapshot used for this manuscript/rebuttal pass:

- `D:\Download\Duong Nguyen\Downloads\table_final_evidence_matrix (2).csv`
  - SHA256: `689effcdf61a4f5e73fbe20a87a5f0d27934f8fc1ce6ce4a39c9d949f0ea3733`
  - shape: 6 rows x 8 columns
- `D:\Download\Duong Nguyen\Downloads\table_final_safe_wording (2).csv`
  - SHA256: `b5227ee0772774fb92f6a8cb7872ebb11d2d3fb83235afc5ce558be0db885a3b`
  - shape: 6 rows x 5 columns

Final evidence manifest:

- `final_ready_for_rebuttal`: `true`
- `all_claims_supported`: `false`
- `missing_inputs`: none
- `contract_issues`: none
- source commit: `4deb9f077a7635a480ceca88d62f7d2b9bf8d2fd`
- protocol: `final_reviewer_evidence_matrix_v1`

Interpretation: the package is ready for rebuttal writing, but it explicitly does not support all original claims. Manuscript wording must be narrowed to the supported evidence below.

## Main Decision

The completed ResNet1D/CNN and Raw Mamba fair baselines change the manuscript posture.

Do not claim:

- broad SOTA-style performance for ECG-RAMBA.
- ECG-RAMBA has in-domain superiority on Chapman OOF.
- ECG-RAMBA is generally superior to fair CNN/ResNet/Mamba baselines.
- ECG-RAMBA is generally more robust than MiniRocket-only.
- HRV is domain-invariant.
- The current HRV36 input implements full RMSSD/SDNN/LF-HF HRV.
- External PTB/Georgia/CPSC or few-shot transfer is manuscript-ready.

Supported contribution:

> ECG-RAMBA should be presented as a physiologically structured ECG model and evidence package, not as the best in-domain classifier. Under the frozen Chapman OOF protocol, ResNet1D/CNN and Raw Mamba are stronger fair comparators for discrimination/F1 metrics, while ECG-RAMBA contributes transparent morphology-rhythm design analysis, comparator-specific calibration tradeoffs, HRV/domain-sensitivity evidence, and metric-specific robustness behavior.

## Key Results To Use

### Frozen Chapman OOF

| Model | ROC-AUC macro | PR-AUC macro | F1 macro | Brier macro | ECE macro |
|---|---:|---:|---:|---:|---:|
| Full ECG-RAMBA | 0.8373 | 0.3432 | 0.3998 | 0.0355 | 0.0552 |
| ResNet1D/CNN | 0.9391 | 0.4899 | 0.4906 | 0.0342 | 0.0414 |
| Raw Mamba | 0.9284 | 0.4377 | 0.4465 | 0.0401 | 0.1137 |
| MiniRocket-only | 0.9169 | 0.4508 | 0.2477 | 0.1906 | 0.3759 |
| HRV-only | 0.8113 | 0.1771 | 0.2077 | 0.1613 | 0.2844 |

### Full ECG-RAMBA vs ResNet1D/CNN

The paired Full-minus-ResNet comparison favors ResNet1D/CNN on all five in-domain metrics:

| Metric | Full | ResNet1D/CNN | Full minus ResNet | 95% CI | Interpretation |
|---|---:|---:|---:|---|---|
| PR-AUC macro | 0.3432 | 0.4899 | -0.1467 | [-0.1607, -0.1392] | ResNet significantly better |
| ROC-AUC macro | 0.8373 | 0.9391 | -0.1019 | [-0.1127, -0.0909] | ResNet significantly better |
| F1 macro | 0.3998 | 0.4906 | -0.0908 | [-0.1016, -0.0796] | ResNet significantly better |
| Brier macro | 0.0355 | 0.0342 | -0.0014 | [-0.0017, -0.0010] | ResNet significantly better |
| ECE macro | 0.0552 | 0.0414 | -0.0139 | [-0.0143, -0.0134] | ResNet significantly better |

Manuscript implication:

- ResNet1D/CNN must be described as the stronger in-domain architecture baseline.
- ECG-RAMBA must not be described as an in-domain winner.
- Any MiniRocket-only advantage for ECG-RAMBA must be explicitly comparator-specific and cannot be generalized to ResNet/CNN.

### Full ECG-RAMBA vs Raw Mamba

The paired Full-minus-Raw-Mamba comparison is comparator-specific. Raw Mamba is stronger for discrimination/F1, while ECG-RAMBA has lower Brier score and ECE:

| Metric | Full | Raw Mamba | Full minus Raw Mamba | Interpretation |
|---|---:|---:|---:|---|
| PR-AUC macro | 0.3432 | 0.4377 | negative | Raw Mamba significantly better |
| ROC-AUC macro | 0.8373 | 0.9284 | negative | Raw Mamba significantly better |
| F1 macro | 0.3998 | 0.4465 | negative | Raw Mamba significantly better |
| Brier macro | 0.0355 | 0.0401 | Full lower | Full significantly better |
| ECE macro | 0.0552 | 0.1137 | Full lower | Full significantly better |

Manuscript implication:

- Raw Mamba must be treated as a completed fair comparator, not a deferred item.
- ECG-RAMBA must not claim general superiority over Mamba baselines.
- ECG-RAMBA may only claim Raw-Mamba-specific calibration/error advantages for Brier and ECE.

### Full ECG-RAMBA vs MiniRocket-only

The MiniRocket-only comparison is a metric-dependent tradeoff:

| Metric | Full | MiniRocket-only | Full minus MiniRocket | 95% CI | Interpretation |
|---|---:|---:|---:|---|---|
| PR-AUC macro | 0.3432 | 0.4508 | -0.1076 | [-0.1221, -0.0982] | MiniRocket significantly better |
| ROC-AUC macro | 0.8373 | 0.9169 | -0.0796 | [-0.0911, -0.0685] | MiniRocket significantly better |
| F1 macro | 0.3998 | 0.2477 | 0.1522 | [0.1477, 0.1569] | Full significantly better |
| Brier macro | 0.0355 | 0.1906 | 0.1551 | [0.1547, 0.1555] | Full significantly better |
| ECE macro | 0.0552 | 0.3759 | 0.3207 | [0.3203, 0.3211] | Full significantly better |

Manuscript implication:

- MiniRocket-only is stronger for rank-based discrimination.
- ECG-RAMBA is stronger than MiniRocket-only for the fixed threshold and calibration-sensitive operating point.
- This is not a global superiority claim.

### HRV And Domain Sensitivity

HRV-only:

- ROC-AUC macro: 0.8113
- PR-AUC macro: 0.1771
- F1 macro: 0.2077

HRV domain classifier:

- ROC-AUC OVR macro: approximately 1.0000
- PR-AUC macro: approximately 1.0000
- balanced accuracy: 0.9996

Manuscript implication:

- HRV contains useful rhythm signal.
- HRV is highly domain-sensitive.
- Do not describe HRV as an invariant anchor.
- Do not claim full HRV features such as RMSSD/SDNN/LF-HF unless those features are implemented and retrained.

### Robustness

Stress tests completed:

- `snr20db`
- `snr10db`
- `snr5db`
- `random_3_lead_dropout`
- `precordial_dropout`
- `resample_250hz`

Final robustness table contains 30 rows: 6 stresses x 5 metrics.

Metric-family pattern:

- Full ECG-RAMBA is better under stress for F1, Brier, and ECE in all six stress tests.
- MiniRocket-only is better under stress for PR-AUC and ROC-AUC in all six stress tests.
- MiniRocket-only is less degraded in 26/30 rows.
- Full ECG-RAMBA is less degraded in only 4/30 rows, limited to Brier/ECE under `snr10db` and `snr5db`.

Manuscript implication:

- Do not write that ECG-RAMBA is generally more robust.
- It is valid to write that ECG-RAMBA maintains a better stressed fixed-threshold/calibration operating point than MiniRocket-only.
- It is also necessary to state that MiniRocket-only degrades less in most stress-metric rows and remains stronger for ranking metrics under stress.

## Manuscript Text To Use

### Abstract Replacement

Use:

> We evaluate ECG-RAMBA under a frozen subject-aware Chapman OOF protocol with calibration, baseline, HRV/domain, pooling, and perturbation analyses. The completed fair-baseline evidence does not support an in-domain superiority claim: ResNet1D/CNN outperforms ECG-RAMBA on macro PR-AUC, macro ROC-AUC, macro F1, Brier score, and ECE, and Raw Mamba outperforms ECG-RAMBA on discrimination/F1 metrics. ECG-RAMBA is therefore framed as a physiologically structured morphology-rhythm model whose value lies in transparent component analysis and metric-specific operating behavior rather than global classification superiority.

Avoid:

> Any sentence stating broad state-of-the-art performance for ECG-RAMBA.

Avoid:

> ECG-RAMBA outperforms CNN/ResNet baselines.

### Contribution Replacement

Use these contributions:

1. A physiologically structured ECG classification architecture that combines morphology- and rhythm-oriented streams.
2. A frozen, traceable Chapman OOF evaluation protocol with final EMA checkpoints, Q=3 record aggregation, calibration metrics, and artifact provenance.
3. A baseline analysis showing that ResNet1D/CNN is the stronger in-domain architecture baseline, while ECG-RAMBA shows a MiniRocket-specific fixed-threshold/calibration tradeoff.
4. An HRV/domain analysis showing useful rhythm signal but high domain sensitivity.
5. A perturbation analysis showing metric-specific robustness behavior, not general robustness superiority.

### Results Baseline Paragraph

Use:

> Under the frozen Chapman OOF protocol, Full ECG-RAMBA achieved macro ROC-AUC 0.8373, macro PR-AUC 0.3432, macro F1 0.3998, Brier score 0.0355, and macro ECE 0.0552. The completed ResNet1D/CNN baseline achieved higher in-domain performance across all paired metrics, with macro ROC-AUC 0.9391, macro PR-AUC 0.4899, macro F1 0.4906, Brier score 0.0342, and macro ECE 0.0414. Raw Mamba also achieved higher discrimination/F1 metrics, with macro ROC-AUC 0.9284, macro PR-AUC 0.4377, and macro F1 0.4465, although ECG-RAMBA had lower Brier and ECE than Raw Mamba. We therefore do not claim in-domain superiority for ECG-RAMBA over fair CNN/ResNet/Mamba baselines.

### MiniRocket Tradeoff Paragraph

Use:

> The MiniRocket-only feature baseline reveals a different, metric-specific tradeoff. MiniRocket-only is stronger for rank-based metrics, with higher macro PR-AUC and macro ROC-AUC. Full ECG-RAMBA is stronger at the frozen threshold and calibration-sensitive operating point, with higher macro F1 and lower Brier/ECE. This result should be interpreted as a MiniRocket-specific operating-point tradeoff, not as evidence that ECG-RAMBA is globally superior to feature or architecture baselines.

### Discussion Baseline Paragraph

Use:

> The revised baseline evidence changes the interpretation of the model. ECG-RAMBA should not be presented as the strongest in-domain Chapman classifier. Instead, the fair-baseline results indicate that ResNet1D/CNN is stronger across all principal in-domain metrics and Raw Mamba is stronger on discrimination/F1 metrics. The contribution of ECG-RAMBA is therefore better framed as structured model analysis: it tests a morphology-rhythm design, exposes comparator-specific tradeoffs between ranking and calibrated threshold behavior, and provides explicit evidence about HRV domain sensitivity and perturbation behavior.

### Robustness Paragraph

Use:

> Robustness findings are metric-specific. Across six perturbations, ECG-RAMBA is better than MiniRocket-only under stress for fixed-threshold and calibration-sensitive metrics, including F1, Brier score, and ECE. MiniRocket-only remains better under stress for PR-AUC and ROC-AUC and is less degraded in most stress-metric comparisons. We therefore report robustness by metric family rather than claiming a general robustness advantage.

### HRV/Domain Paragraph

Use:

> HRV-only features provide non-trivial signal, but the HRV domain classifier separates dataset source almost perfectly. This supports reporting HRV as useful but domain-sensitive. We removed wording that framed HRV as an invariant anchor and corrected the HRV36 description to match the implemented checkpoint input schema.

### External/Few-Shot Paragraph

Use:

> External and few-shot transfer remain outside the current manuscript-ready evidence package. The present results support frozen Chapman OOF conclusions only. PTB-XL, Georgia, CPSC2021, and few-shot analyses should be reported only after separate dataset-specific label, windowing, PCA, and uncertainty gates are completed.

## Rebuttal Text To Use

### Baseline Criticism

Use:

> We agree that stronger in-domain baselines are necessary. We therefore added ResNet1D/CNN and Raw Mamba baselines under the same frozen subject-aware OOF protocol, threshold, and Q=3 aggregation. ResNet1D/CNN significantly outperformed ECG-RAMBA on PR-AUC, ROC-AUC, F1, Brier, and ECE. Raw Mamba outperformed ECG-RAMBA on PR-AUC, ROC-AUC, and F1, while ECG-RAMBA had lower Brier and ECE. We revised the manuscript accordingly and removed any claim of in-domain superiority over fair CNN/ResNet/Mamba baselines.

### MiniRocket Criticism

Use:

> The MiniRocket-only comparison shows a metric-dependent result rather than a uniform win for ECG-RAMBA. MiniRocket-only is better for PR-AUC and ROC-AUC, while ECG-RAMBA is better at the frozen threshold for F1 and calibration metrics. We now present this as an operating-point tradeoff and not as evidence of global superiority.

### Robustness Criticism

Use:

> We added stress tests for noise, lead dropout, and sampling-rate perturbation. The resulting claims are metric-specific. ECG-RAMBA is stronger under stress for fixed-threshold/calibration metrics, whereas MiniRocket-only remains stronger for ranking metrics and is less degraded in most comparisons. The revised manuscript therefore avoids a general robustness claim.

### HRV Criticism

Use:

> We agree that HRV can encode domain-specific acquisition characteristics. The added HRV domain classifier shows near-perfect domain separability, so we revised the manuscript to describe HRV as useful but domain-sensitive rather than invariant. We also corrected the HRV36 description to avoid claiming unimplemented RMSSD/SDNN/LF-HF slots.

### External/Few-Shot Criticism

Use:

> We separated manuscript-ready Chapman OOF evidence from external and few-shot hypotheses. External PTB/Georgia/CPSC and few-shot results are not used for primary claims until dataset-specific mapping, windowing, PCA, and uncertainty checks are completed.

## Do-Not-Write List

Remove or rewrite any sentence containing these ideas:

- broad state-of-the-art performance
- "in-domain superiority"
- "outperforms CNN/ResNet baselines"
- "globally superior"
- "more robust" without a metric family and comparator
- "HRV invariant"
- validated morphology-rhythm disentanglement
- "RMSSD/SDNN/LF-HF" for current checkpoint features
- "external zero-shot/few-shot" as completed manuscript evidence
- "clinical safety" or safety-oriented claims based only on OOF metrics

## Claim Boundaries

Allowed:

- Protocol-faithful frozen Chapman OOF evaluation.
- ResNet1D/CNN is stronger in-domain across the principal metrics.
- Raw Mamba is stronger on discrimination/F1 metrics.
- MiniRocket-only shows stronger ranking metrics.
- ECG-RAMBA shows comparator-specific calibration/error advantages only where paired evidence supports them.
- HRV is useful but domain-sensitive.
- Robustness is metric-specific.
- Q=3 is a frozen/sensitivity-tested operating point.

Not allowed:

- best-in-domain wording for ECG-RAMBA.
- SOTA-style wording for ECG-RAMBA.
- ECG-RAMBA is generally robust.
- ECG-RAMBA generalizes better externally without a separate completed protocol.

## Next Writing Step

Update the manuscript in this order:

1. Abstract and Contributions: remove superiority wording.
2. Results: include ResNet1D/CNN, Raw Mamba, MiniRocket-only, and HRV-only in the fair-baseline table before tradeoff interpretation.
3. Discussion: reframe ECG-RAMBA as structured model analysis.
4. Limitations: add HRV domain sensitivity, external protocol deferral, representation-probe blocker, and comparator-specific baseline limits.
5. Rebuttal: use the response snippets above and cite `table_final_evidence_matrix.csv`, `table_final_safe_wording.csv`, `table_paired_full_vs_resnet.csv`, `table_paired_full_vs_minirocket.csv`, and `table_final_robustness_claims.csv`.
