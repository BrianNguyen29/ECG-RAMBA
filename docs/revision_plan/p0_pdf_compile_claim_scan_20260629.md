# P0 PDF Compile And Claim Scan Audit - 2026-06-29

## Scope

This audit covers the IEEE manuscript source at:

- `D:/WorkSpace/ECG/ECG-Ramba/docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/main.tex`
- `D:/WorkSpace/ECG/ECG-Ramba/docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/response_to_reviewers_revised_20260622.md`

## Incremental Update - 2026-07-09

The manuscript and response letter were updated after the latest final evidence package:

- `table_final_evidence_matrix.csv` and `table_final_safe_wording.csv` now include PTB-XL, Georgia, and CPSC2021 as protocol-gated mapped-task external evidence.
- Few-shot wording is restricted to leakage-audited score calibration on frozen external prediction scores; ECG-RAMBA weights remain unchanged.
- Manuscript/response wording was changed from `zero-shot mapped-task/window` to `no-target-label mapped-task/window` to avoid implying general zero-shot superiority.
- P1/P2 items, including Transformer ECG, Hybrid MiniRocket morphology MLP, and learned-comparator robustness, remain pending until their artifacts are completed and Notebook 07 regenerates the final tables.

Local Windows validation on 2026-07-09:

- `latexmk`, `pdflatex`, and `pdftotext` were not available in PATH, but portable Tectonic 0.16.9 was available at `D:/WorkSpace/ECG/ECG-Ramba/.codex_tools/tectonic-0.16.9/tectonic.exe`.
- Recompiled successfully with Tectonic into `docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/build/main.pdf`, then synchronized the generated PDF/text to `main.pdf` and `main.pdf.txt`.
- PDF info: 11 pages, letter paper, PDF 1.5, not encrypted, file size 1,593,190 bytes.
- Text extraction used a local `pdftotext` helper at `D:/WorkSpace/ECG/ECG-Ramba/.codex_tools/bin/pdftotext.cmd`, backed by `pypdf`; rendering used Poppler `pdftoppm.exe` from the Codex runtime.
- Rendered pages are in `docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/build/render_check_20260709/`.
- Source-level and PDF-text scans found zero remaining hits for the main positive-claim red flags: `zero-shot`, `SOTA`, `best in-domain`, `global superiority`, `broad superiority`, `proven disentanglement`, `general robustness superiority`, `zero-shot superiority`, `few-shot superiority`, `fine-tune weights`, and `clinical deployment readiness`.
- Literal forbidden-term matches in final evidence tables remain in negated/restrictive readiness-gate wording, e.g., `Do not claim ...`.

## Build Result

- Compiler: portable Tectonic 0.16.9
- Output PDF: `D:/WorkSpace/ECG/ECG-Ramba/docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/build/main.pdf`
- Render check: `D:/WorkSpace/ECG/ECG-Ramba/docs/IEEE_JBHI___ECG_RAMBA___XT_Reviewed/build/render_check_20260629_final/`
- Rendered pages: 10

LaTeX log summary:

- Undefined references: 0
- Undefined citations: 0
- LaTeX errors: 0
- Emergency/fatal errors: 0
- Overfull hbox warnings: 3
- Overfull vbox warnings: 10
- Underfull hbox warnings: 18

The remaining warnings are layout/font warnings from the IEEE template and float placement. Visual rendering of pages with Algorithm 1, figures, and tables did not show missing content or broken captions.

## Algorithm 1 Check

Algorithm 1 was added to document the reviewer-requested protocol:

- Fold-aware subject split.
- Training-fold-only MiniRocket/PCA fitting.
- Validation/test transformation with fold-specific PCA.
- Checkpoint-compatible rhythm/global vector extraction.
- BCE warm-up, ASL, AdamW, cosine learning-rate schedule, and EMA weights.
- Frozen `final_ema` checkpoint contract.
- Fixed threshold `tau = 0.5`.
- Power Mean aggregation with `Q = 3`.
- OOF prediction, slice-probability, checksum, fold, and configuration manifests.

PDF text extraction confirms:

- `Algorithm 1` present.
- `Fold-aware ECG-RAMBA training and inference` present.
- `Power Mean` and `Q = 3` present.

## Figure, Caption, And Table Check

Active graphics found:

- `architecture.png`
- `perc.png`
- `fig_lead_dropout_clean.png`
- `Saliency.png`
- `fig_ablation.pdf`

Reference check:

- Missing `\ref` / `\eqref` / `\autoref` labels: 0
- Algorithm label `alg:fold_protocol`: present and referenced

Caption safety:

- Lead-dropout caption states sensitivity analysis, not proven disentanglement.
- Saliency caption states descriptive temporal sensitivity, not causal attribution.
- Ablation caption states component sensitivity under the evaluated protocol.
- External evidence table states PTB-XL mapped-task status and Georgia/CPSC2021/few-shot/probe deferral.

## Forbidden-Claim Scan

Positive forbidden-claim scan passed for:

- `main.tex`
- `response_to_reviewers_revised_20260622.md`

The scan found no positive claims of:

- ECG-RAMBA as SOTA / best-in-domain / global-superiority model.
- Zero-shot superiority.
- General robustness superiority.
- Proven morphology-rhythm disentanglement.
- HRV domain invariance.
- Completed few-shot experiments.

Broad literal term searches still find safe negation or restriction statements such as "does not support a zero-shot superiority claim" and "does not present ECG-RAMBA as the best in-domain model"; these are intentional claim-boundary wording.

## Remaining Claim Boundaries

Keep the following restrictions in manuscript/rebuttal wording:

- Use Chapman frozen OOF as the main manuscript-ready evidence.
- Use PTB-XL, Georgia, and CPSC2021 only as protocol-gated mapped-task external evaluations.
- Georgia and CPSC2021 are no longer deferred under the current mapped-task gates, but neither supports broad external-transfer, official CPSC episode scoring, or clinical deployment claims.
- Do not claim zero-shot or external superiority.
- Do not claim global or in-domain superiority over fair baselines.
- Do not claim strict morphology-rhythm disentanglement.
- Do not describe HRV36 reserved zero-filled slots as implemented RMSSD/SDNN/LF-HF features.
- Use robustness claims only metric-by-metric and comparator-by-comparator.
