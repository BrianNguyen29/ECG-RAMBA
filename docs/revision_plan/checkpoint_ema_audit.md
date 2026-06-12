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
  wins. This is a diagnostic checkpoint because selection uses the same held-out
  fold later represented in OOF.
- `fold*_best_raw.pt`: raw companion saved for diagnostics.
- `fold*_final_ema.pt` and `fold*_final_raw.pt`: fixed final-epoch variants.
- Generic `fold*_best.pt` and `fold*_final.pt` aliases are not written for new
  runs because they duplicate multi-GB checkpoint data and obscure weights
  provenance. Revision scripts must request an explicit checkpoint kind.

Manuscript OOF must be generated from `--checkpoint-kind final_ema` at the
pre-specified epoch and frozen only after checksum, fold coverage, Q=3
re-aggregation, and checkpoint fingerprint validation pass. Using
`best_ema` for manuscript OOF would select an epoch on the same fold being
reported and is not allowed without nested CV.

New checkpoints also record the fixed loss scaling contract: BCE averages over
the batch and classes, while asymmetric loss sums classes and averages the
batch. This legacy scale change is not an LR warmup and must not be described
as one.
