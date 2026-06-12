# Training Epoch And Checkpoint Policy

## Canonical protocol

- The manuscript Chapman OOF run uses `fold*_final_ema.pt` at the
  pre-specified training horizon.
- The current pre-specified horizon is 20 epochs.
- `fold*_best_ema.pt` is retained as a convergence and checkpoint diagnostic.
  It is selected on the same validation fold used to form OOF predictions, so
  it must not be promoted to manuscript-ready OOF without nested CV.
- The fixed decision threshold remains 0.5. Epoch selection must not use
  threshold tuning.

## Why a 30-epoch run is separate

`CosineAnnealingLR` uses `T_max=epochs`. Changing 20 to 30 changes the learning
rate at every epoch, not only after epoch 20. A 30-epoch run is therefore a new
training protocol with a new config hash and model-run directory. It is not a
valid continuation of the 20-epoch run.

The training log records the learning rate used during each epoch. The
scheduler step occurs after the epoch, so the logged value must be captured
before `scheduler.step()`.

## Decision rule

1. Complete the pre-specified 20-epoch five-fold run.
2. Inspect `best_ema_epoch` and both final and best-EMA metrics for convergence.
3. If at least three folds have their diagnostic best EMA epoch at 19 or 20,
   an independent 30-epoch sensitivity run is justified.
4. Do not choose between e20 and e30 by reporting whichever performs better on
   the same OOF folds. Either keep e20 as the primary pre-specified result and
   report e30 as sensitivity, or use a nested/independent model-selection
   design.

## Runtime and resume policy

- Notebook 02a resumes completed folds by default.
- A partially completed fold restarts from epoch 1 because optimizer/scheduler
  state is not checkpointed mid-fold.
- Completed folds are reused only when all explicit checkpoints and logs exist
  and `final_ema` metadata matches the current config, fixed epoch, EMA weights,
  and Chapman record-order fingerprint.
- Fold PCA objects are reused by config hash, training-index hash, and output
  dimension. A five-minute heartbeat is printed while a new CPU PCA fit runs.
- MiniRocket and HRV caches are keyed by the ordered Chapman record IDs. Legacy
  shape-only caches are not manuscript-safe and are regenerated once into the
  fingerprinted cache format.

## Required artifacts

- `training_log_epochs.csv`
- `cv_results_clean_core.csv`
- `fold_split_audit.json`
- `fold_label_prevalence.csv`
- `fold*_final_ema.pt`
- `fold*_best_ema.pt`
- checkpoint manifest with config hash, epoch, selection rule, weights kind,
  metric weights kind, aggregation metadata, and PCA provenance
