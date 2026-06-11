# Checkpoint / EMA Audit

## Finding

The historical training loop evaluated validation metrics with EMA weights after
the asymmetric-loss warmup, then restored raw weights before writing
`fold*_best.pt`. As a result, the saved "best" checkpoints did not necessarily
match the weights that produced the training-log best metrics.

## Diagnostic Evidence

Training logs reported mean best-fold performance around:

- macro F1: 0.435951
- macro AUPRC: 0.429900

Colab OOF audits of the available raw checkpoints produced much lower numbers:

- `fold*_best.pt` raw OOF: macro F1 0.133118, macro PR-AUC 0.098396,
  macro ROC-AUC 0.641011
- `fold*_final.pt` raw OOF: macro F1 0.125474, macro PR-AUC 0.091182,
  macro ROC-AUC 0.626931

This gap is consistent with a checkpoint/EMA provenance mismatch. These raw
OOF artifacts are diagnostic only and must not be used as manuscript results.

## Required Resolution

Retrain the five Chapman folds after updating `scripts/train.py` so that it
writes explicit checkpoint variants:

- `fold*_best_ema.pt`: selected and saved with EMA weights when EMA validation
  wins.
- `fold*_best_raw.pt`: raw companion saved for diagnostics.
- `fold*_final_ema.pt` and `fold*_final_raw.pt`: final-epoch diagnostic
  variants.
- `fold*_best.pt`: compatibility alias only, with metadata indicating the
  underlying weights kind.

Manuscript OOF must be generated from `--checkpoint-kind best_ema` and frozen
only after checksum, fold coverage, Q=3 re-aggregation, and checkpoint
fingerprint validation pass.
