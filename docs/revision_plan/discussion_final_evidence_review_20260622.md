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

The central decision is:

- ResNet1D/CNN is the stronger in-domain architecture baseline.
- ECG-RAMBA must not be claimed as SOTA, globally superior, or the best in-domain classifier.
- MiniRocket-only shows a metric-specific tradeoff.
- HRV is useful but domain-sensitive.
- Robustness is metric-specific.
- External and few-shot claims remain experimental/deferred.

## Key Numbers

| Model | ROC-AUC | PR-AUC | F1 | Brier | ECE | Status |
|---|---:|---:|---:|---:|---:|---|
| Full ECG-RAMBA | 0.8373 | 0.3432 | 0.3998 | 0.0355 | 0.0552 | Complete frozen OOF |
| ResNet1D/CNN | 0.9391 | 0.4899 | 0.4906 | 0.0342 | 0.0414 | Complete fair architecture baseline |
| MiniRocket-only | 0.9169 | 0.4508 | 0.2477 | 0.1906 | 0.3759 | Complete feature baseline |
| HRV-only | 0.8113 | 0.1771 | 0.2077 | 0.1613 | 0.2844 | Complete feature baseline |
| Raw Mamba | NA | NA | NA | NA | NA | Runner TBD / incomplete |

## Discussion Mismatches Found

| Old discussion topic | Final evidence | Required update |
|---|---|---|
| Zero-shot generalization as the central story | External and few-shot outputs are not manuscript-ready. | Reframe as protocol-faithful Chapman OOF structured model analysis. |
| Morphology-rhythm disentanglement | Representation probe remains blocked. | Use structured morphology-rhythm design and component-sensitivity language only. |
| Safety-oriented ranking-decision gap | Calibration exists, but ResNet is better on F1/Brier/ECE. | Use fixed-threshold operating-point wording only. |
| MiniRocket/deterministic morphology improves robustness | MiniRocket wins PR-AUC/ROC-AUC; Full wins F1/Brier/ECE only against MiniRocket. | State metric-specific MiniRocket tradeoff. |
| Learned morphology baseline as planned item | ResNet1D/CNN is complete and beats ECG-RAMBA. | Explicitly acknowledge ResNet is stronger in-domain. |
| HRV invariant anchor | HRV domain classifier is near-perfect. | State HRV is useful but domain-sensitive. |
| Robustness as general claim | Full wins stressed F1/Brier/ECE; MiniRocket wins stressed PR-AUC/ROC-AUC and degrades less in most rows. | Use metric-family robustness only. |
| Few-shot/zero-shot response | Deferred. | Move to future work unless separate protocol gates are completed. |

## Updated Reviewer Response Direction

### Reviewer 1

- R1C1: agree, remove safety language, report calibration and operating-point metrics.
- R1C2: report MiniRocket as a metric-specific baseline; do not generalize to ResNet/CNN.
- R1C3: HRV-only has signal, but HRV is highly domain-sensitive.
- R1C4: acknowledge ResNet1D/CNN is stronger in-domain; Raw Mamba remains incomplete.
- R1C5: use bootstrap/paired bootstrap only for metric-specific conclusions.
- R1C6: Q=3 is a frozen operating point and sensitivity-tested tradeoff.
- R1C7: use final EMA, artifact provenance, fixed threshold, Q=3, and corrected HRV36 schema.

### Reviewer 2

- R2C1: present the pipeline as data -> folds -> final EMA checkpoints -> slice probabilities -> Q=3 aggregation -> calibration/baseline/robustness evidence.
- R2C2: do not claim proven disentanglement; representation-level probing/CKA/UMAP remains future work.
- R2C3: robustness claims must be metric-specific.
- R2C4: external and few-shot transfer remain deferred.

## Next Items

| Priority | Item | Concrete action |
|---|---|---|
| P0 | Finalize manuscript text from current evidence | Continue from `docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/main.tex`; remove any remaining old figure/table captions that imply unsupported claims. |
| P0 | Finalize response letter | Use `docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/response_to_reviewers_revised_20260622.md`, then add exact manuscript section/table references. |
| P0 | Compile and visually inspect IEEE PDF | Run `pdflatex`/BibTeX or Overleaf; inspect table widths, figures, captions, references, and page layout. |
| P1 | Update or remove old external zero-shot figures | External figures must be labeled experimental or moved out of primary results. |
| P1 | Add artifact provenance appendix/table | Include frozen OOF protocol, final EMA checkpoints, threshold 0.5, Q=3, HRV schema restriction, and source artifact paths/checksums. |
| P1 | Align figures with final claims | Per-class, lead-dropout, ablation, and robustness figures must not imply superiority, safety, or proven disentanglement. |
| P2 | Decide on Raw Mamba runner | Optional. It remains the only incomplete fair comparator row, but ResNet and MiniRocket already answer the main baseline concern. |
| P2 | Optional external/few-shot package | Only run if making a new transfer claim; it needs separate label/window/PCA/CI gates. |
| P2 | Optional representation probe | Only run if stronger morphology-rhythm separation claims are necessary. |

## Current Recommendation

Proceed with a conservative resubmission package centered on protocol-faithful Chapman OOF evidence and honest baseline interpretation. Do not reopen broad experiments until the current manuscript and response letter are internally consistent.
