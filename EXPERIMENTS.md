# Reproducing Reviewer Evidence

The revision notebooks are the supported experiment entry points. Do not use historical metric values in this file as evidence. The generated `table_final_evidence_matrix.csv` and `table_final_safe_wording.csv` are the only manuscript/rebuttal source of truth.

See [`docs/revision_plan/notebook_forensic_audit_runbook.md`](docs/revision_plan/notebook_forensic_audit_runbook.md) for the complete hardware, cache, resume, provenance, and statistical protocol.

## Canonical storage

- Canonical artifacts: `/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision`
- Convenience output only: `/content/drive/MyDrive/ECG-Ramba/final_evidence_tables`
- Legacy audit-only checkout: `/content/drive/MyDrive/ECG-Ramba/ECG-RAMBA/reports/revision`

File presence is not a reuse contract. A reusable artifact must match the canonical manifest SHA, current input/source/config hashes, record order, and producer authority.

## Notebook order

1. `00_colab_bootstrap.ipynb` on CPU.
2. `01_a0_protocol_audit.ipynb` on CPU.
   Run `02a_retrain_best_ema.ipynb` on A100 only if the fixed checkpoint protocol itself must be retrained; skip it for a normal evidence refresh.
3. `02_predictions_and_external_eval.ipynb` on A100 only for missing inference.
4. `03_calibration_and_ci.ipynb` on CPU High-RAM.
5. `04_baselines_and_component_checks.ipynb` on A100 for missing folds, then CPU for paired ledgers.
6. Run Notebook 02 again for missing PTB-XL fold 9/external representations; adaptation statistics are CPU-only after predictions exist.
7. `05_hrv_domain_and_robustness.ipynb` on A100 for missing stress predictions, then CPU for aggregation.
8. `06_pooling_and_representation.ipynb` on A100 only when checkpoint-local embeddings are missing.
9. `07_results_freeze.ipynb` on CPU High-RAM. Run all cells in order.

## Experiment identity

- The morphology transform is a fixed-seed ROCKET-family ternary-kernel MAX+PPV transform, not canonical MiniRocket.
- ResNet1D/CNN, Raw Mamba, compact Transformer ECG, and morphology MLP are same-fold comparators unless a complete budget-matching contract proves otherwise.
- External PTB-XL and Georgia results are separate mapped record-level tasks.
- CPSC2021 is a separate annotation-aligned 10-second AF/AFL mapped-window task.
- Score calibration and frozen-encoder head adaptation are separate protocols.
- Checkpoint-local probes and fold CKA are representation audits, not proof of mechanistic separation.

## Final verification

Notebook 07 must pass the strict full-SHA storage audit and forensic notebook audit before creating `final_evidence_tables`. Submission assets are generated only after those gates pass and the forbidden-claim scan is clean.
