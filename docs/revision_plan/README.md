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

## Notebook Sequence

Run in this order on Colab:

1. `notebooks/00_colab_bootstrap.ipynb`
2. `notebooks/01_a0_protocol_audit.ipynb`
3. `notebooks/02_predictions_and_external_eval.ipynb`
4. `notebooks/03_calibration_and_ci.ipynb`
5. `notebooks/04_baselines_and_component_checks.ipynb`
6. `notebooks/05_hrv_domain_and_robustness.ipynb`
7. `notebooks/06_pooling_and_representation.ipynb`
8. `notebooks/07_results_freeze.ipynb`

The current priority is A-level work: consistency audit, calibration,
baseline expansion, bootstrap CI, method clarity, and minimum robustness.
