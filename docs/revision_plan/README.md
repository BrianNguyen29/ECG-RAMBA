# ECG-RAMBA Reviewer Revision Plan

This folder is the repo-tracked source of truth for the JBHI major-revision
work. It summarizes the action items extracted from:

```text
D:/WorkSpace/ECG/ECG-Ramba/docs/ECG_RAMBA_Reviewer_Discussion_Revised.docx
```

The notebooks in `notebooks/` are the execution layer. Generated predictions,
metrics, figures, tables, and manifests must be written under
`reports/revision/` and are ignored by Git except for `.gitkeep` files.

## Operating Model

1. Keep GitHub for code, notebooks, and planning metadata.
2. Keep data, checkpoints, caches, and large experiment outputs on Drive.
3. Run experiments from numbered notebooks.
4. Write every reusable result to `reports/revision/`.
5. Freeze final numbers with `scripts/revision/05_artifact_inventory.py`.

## File Map

- `task_board.csv`: reviewer-response work board with priority, output, owner
  notebook, and status.
- `claim_evidence_map.csv`: allowed claims, required evidence, and fallback
  wording if results do not support the claim.
- `experiment_registry.csv`: experiment groups, expected scripts, input/output
  contracts, and completion gates.
- `artifact_contract.md`: folder layout and naming rules for reproducible
  outputs.
- `training_epoch_policy.md`: fixed-epoch manuscript policy, EMA checkpoint
  roles, and the e20/e30 decision rule.
- `manuscript_rebuttal_drive_source_update_20260622.md`: current
  manuscript/rebuttal wording source generated from
  `D:/WorkSpace/ECG/ECG-Ramba/drive/final_evidence_tables` after the completed
  ResNet1D/CNN baseline. Use this file before reusing older discussion wording.
- `discussion_final_evidence_review_20260622.md`: current audit of the
  reviewer discussion plan against the final evidence package, including the
  updated next-item checklist.

## Notebook Sequence

Run in this order on Colab:

1. `notebooks/00_colab_bootstrap.ipynb`
2. `notebooks/01_a0_protocol_audit.ipynb`
3. `notebooks/02a_retrain_best_ema.ipynb`
4. `notebooks/02_predictions_and_external_eval.ipynb`
5. `notebooks/03_calibration_and_ci.ipynb`
6. `notebooks/06_pooling_and_representation.ipynb` (pooling sensitivity)
7. `notebooks/04_baselines_and_component_checks.ipynb`
8. `notebooks/05_hrv_domain_and_robustness.ipynb`
9. Revisit `notebooks/06_pooling_and_representation.ipynb` for representation work.
10. `notebooks/07_results_freeze.ipynb`

Colab package installations are runtime-scoped. Keep the same runtime between
notebooks when practical. If Colab restarts or you disconnect, rerun the current
notebook from its first cell; Notebook 02a will replay the canonical base and
Mamba bootstrap before retraining.

Notebook 02 must first produce and freeze the post-fix
`oof_final_ema_*` artifacts from explicit fixed-epoch
`fold*_final_ema.pt` checkpoints.
Notebook 02a writes those retrained checkpoints to a versioned Drive model-run
directory under `Drive/ECG-Ramba/model_runs/` and writes
`model_runs/current_final_ema_model_dir.txt`. Notebook 02 reads that pointer
before falling back to the historical `Drive/ECG-Ramba/model` directory.
Notebook 03 and pooling sensitivity consume only the checksum-verified
`oof_final_ema_freeze_manifest.json`. Validation-selected `oof_best_ema_*`
and historical raw `oof_full_*` and `oof_final_*` outputs are diagnostic.
External outputs remain experimental until their separate readiness restrictions
are resolved.

The current priority is A-level work: consistency audit, calibration,
baseline expansion, bootstrap CI, method clarity, and minimum robustness.
