# Revision Notebook Suite

Use these notebooks on Colab as the execution layer for the reviewer-revision
plan. Large data and model files stay on Google Drive; GitHub only transports
code, notebooks, and planning metadata.

## Recommended Order

1. `00_colab_bootstrap.ipynb`
2. `01_a0_protocol_audit.ipynb`
3. `02_predictions_and_external_eval.ipynb`
4. `03_calibration_and_ci.ipynb`
5. `04_baselines_and_component_checks.ipynb`
6. `05_hrv_domain_and_robustness.ipynb`
7. `06_pooling_and_representation.ipynb`
8. `07_results_freeze.ipynb`

Original exploratory/demo notebooks are retained under `notebooks/archive/`.
Avoid adding reviewer-revision work to those legacy notebooks.

Legacy direct-runner notebooks were removed to avoid running the wrong setup
path. Use the numbered suite above for revision work.
