# Local Artifact Sync Audit

Date: 2026-06-28

Scope: local workspace copy under `D:\WorkSpace\ECG\ECG-Ramba` after manuscript/rebuttal updates from the regenerated final evidence tables.

The downloaded artifact bundle is extracted with one extra nested directory. The actual local revision artifact root is:

```text
D:\WorkSpace\ECG\ECG-Ramba\drive\revision_artifacts\revision_artifacts\reports\revision
```

## Verified Complete

- `docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\main.tex` contains the final evidence interpretation:
  - ResNet1D/CNN is stronger than ECG-RAMBA on PR-AUC, ROC-AUC, F1, Brier, and ECE.
  - Raw Mamba is stronger than ECG-RAMBA on PR-AUC, ROC-AUC, and F1.
  - ECG-RAMBA has lower Brier score and ECE than Raw Mamba.
  - ECG-RAMBA is framed as structured model analysis, not as SOTA, best in-domain, global robustness, external-transfer, or disentanglement evidence.
- `docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\response_to_reviewers_revised_20260622.md` includes ResNet1D/CNN and Raw Mamba responses with section/table references.
- The IEEE PDF exists and was checked:
  - `docs\IEEE_JBHI___ECG_RAMBA___XT_Reviewed\build\main.pdf`
  - 10 pages
  - contains Raw Mamba and ResNet1D/CNN rows and values.

## Final Evidence Tables Synced Locally

The downloaded regenerated tables are available in the convenience table folder and in the verified artifact mirror:

- `drive\final_evidence_tables\final_evidence_matrix.json`
- `drive\final_evidence_tables\final_evidence_matrix_manifest.json`
- `drive\final_evidence_tables\table_final_evidence_matrix.csv`
- `drive\final_evidence_tables\table_final_safe_wording.csv`
- `drive\revision_artifacts\revision_artifacts\reports\revision\metrics\final_evidence_matrix.json`
- `drive\revision_artifacts\revision_artifacts\reports\revision\manifests\final_evidence_matrix_manifest.json`
- `drive\revision_artifacts\revision_artifacts\reports\revision\tables\table_final_evidence_matrix.csv`
- `drive\revision_artifacts\revision_artifacts\reports\revision\tables\table_final_safe_wording.csv`

Checksums:

- `final_evidence_matrix.json`
  - SHA256: `ba06cf74a06bc9b38a03338246bf6e009dc19fe807543c471fbaf85224942573`
  - size: 56572 bytes
- `final_evidence_matrix_manifest.json`
  - SHA256: `59fa3016b90a7e05a491611381fb0836b5aa94ea558d398680ce23742e4e1c3d`
  - size: 1902 bytes
- `table_final_evidence_matrix.csv`
  - SHA256: `689effcdf61a4f5e73fbe20a87a5f0d27934f8fc1ce6ce4a39c9d949f0ea3733`
  - size: 4504 bytes
- `table_final_safe_wording.csv`
  - SHA256: `b5227ee0772774fb92f6a8cb7872ebb11d2d3fb83235afc5ce558be0db885a3b`
  - size: 2185 bytes

## Local Mirror Status

The local downloaded `drive\revision_artifacts` folder contains a complete latest artifact bundle under the nested root above. Manifest verification passed:

- `manifests\mirror_manifest.json`
- artifact count: 212
- checked files: 212
- missing files: 0
- checksum/size mismatches: 0

Raw Mamba artifact-level files are present:

- `reports/revision/metrics/raw_mamba_baseline_summary.json`
- `reports/revision/tables/table_raw_mamba_class_metrics.csv`
- `reports/revision/tables/table_raw_mamba_fold_summary.csv`
- `reports/revision/manifests/raw_mamba_baseline_manifest.json`
- `reports/revision/metrics/paired_full_vs_raw_mamba_comparison.json`
- `reports/revision/tables/table_paired_full_vs_raw_mamba.csv`
- `reports/revision/manifests/paired_full_vs_raw_mamba_manifest.json`

When referencing artifacts locally, use the nested path above. If a script expects `drive\revision_artifacts\reports\revision`, either point it to the nested root or flatten the extracted archive before running the script.

## Search Checks

The current manuscript and revision-plan docs were scanned for stale high-risk wording. The active sources no longer contain exact unsupported SOTA, best-in-domain, state-of-the-art, Raw-Mamba-incomplete, or stale fair-baseline-blocker statements.
