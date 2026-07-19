# ECG-RAMBA Forensic Notebook Runbook

The audit uses TRIPOD+AI to organize reporting/traceability checks and
PROBAST+AI to organize risk-of-bias, leakage, analysis, and applicability
checks. Passing the automated pipeline gates is not a clinical validation.

## Authority and storage

1. Commit and push every forensic change before opening Colab. The strict audit rejects a dirty worktree.
2. Use the clean commit reported by `git rev-parse HEAD` as the only code authority for the run.
3. Use only `/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision` as the canonical artifact root.
4. Treat `/content/drive/MyDrive/ECG-Ramba/ECG-RAMBA/reports/revision` as legacy audit-only data.
5. Treat `final_evidence_tables` as a generated output snapshot. Never restore pipeline inputs from it.
6. Do not publish debug, record-limited, or smoke-test outputs into the canonical root.

The starting authority for this audit was commit `055587f2b6a2f8715cdd51782daedfa683a196b1`. After the forensic fixes are committed, the new clean commit containing those fixes becomes the authority. Artifact producer commits must match it.

Notebook 00 stores that authority in
`revision_artifacts/reports/revision/manifests/notebook_code_authority.json`.
The first clean bootstrap pins the fetched `main` HEAD (or the full SHA supplied
through `ECG_RAMBA_AUTHORITY_COMMIT`) and writes the manifest atomically. Every
downstream setup, including Notebook 07's embedded direct-run guard, requires
the manifest and checks out its 40-character SHA in detached mode. It fails
closed on a missing/invalid manifest, a conflicting environment SHA, or tracked
local edits; it never silently follows a newer branch after a disconnect.

To rotate authority after committing and pushing a reviewed change, run
Notebook 00 once with both variables set before its setup cell:

```python
import os
os.environ['ECG_RAMBA_AUTHORITY_COMMIT'] = '<full-40-character-reviewed-commit>'
os.environ['ECG_RAMBA_RESET_CODE_AUTHORITY'] = '1'
```

Do not set `ECG_RAMBA_RESET_CODE_AUTHORITY` in Notebooks 01--07. They are
consumers and deliberately cannot establish or rotate authority.

## Runtime order

| Order | Notebook | Runtime | Run rule |
|---|---|---|---|
| 1 | `00_colab_bootstrap.ipynb` | CPU | Mount Drive, clone the clean authority commit, run source/storage audit. CUDA is not required. |
| 2 | `01_a0_protocol_audit.ipynb` | CPU | Run protocol audit and resolve any A0 blocker. |
| Optional | `02a_retrain_best_ema.ipynb` | A100 High-RAM | Use only when the fixed checkpoint protocol itself cannot be authenticated or a paper-level retrain is intentionally opened. It is not part of a normal cached evidence refresh. |
| 3 | `02_predictions_and_external_eval.ipynb` | A100 only when inference is missing | Restore authenticated caches first. Generate only missing OOF/external folds. Regenerate the SHA-bound Chapman patient/group sidecar and run strict freeze and external gates. |
| 4 | `03_calibration_and_ci.ipynb` | CPU High-RAM | Recompute calibration and the independent statistical oracle against the frozen OOF and authenticated patient/group contract. |
| 5 | `04_baselines_and_component_checks.ipynb` | A100 for missing folds; CPU for paired ledgers | Reuse authenticated fold predictions/checkpoints. Train only missing folds. Regenerate paired effect-size ledgers. |
| 6 | `02_predictions_and_external_eval.ipynb` second pass | A100 for missing representations; CPU for adaptation statistics | Export PTB-XL fold 9 adaptation pool, external representations, score calibration, and frozen-encoder-head learning curves. |
| 7 | `05_hrv_domain_and_robustness.ipynb` | A100 for missing stress predictions; CPU for ledgers | Reuse completed stress NPZ files. Recompute metric caches after statistical-schema changes. |
| 8 | `06_pooling_and_representation.ipynb` | A100 only for missing checkpoint-local embeddings | Run pooling sensitivity and checkpoint-local train/probe/validation audits. UMAP remains descriptive. |
| 9 | `07_results_freeze.ipynb` | CPU High-RAM | Run all cells in order. Cell 14 must pass strict full-SHA storage and forensic gates before cell 16 exports `final_evidence_tables`. |

## Required monitoring

Each command must print and persist:

- a unique `run_id`;
- start and end timestamps;
- return code;
- active git commit and source contract where supported;
- input/output paths and SHA256 values;
- cache reuse or regeneration decision;
- a durable log under the canonical mirror.

After every expensive fold or stress prediction, require a successful mirror publish before disconnecting.

Notebook 02 discovers the base and Mamba installers through one exact
capability/schema marker pair per installer. Notebook 00, 02a, 05, and 06 reject
zero or multiple Mamba installer candidates; token-based fallback discovery is
not allowed.

Notebook 02a routes `scripts/train.py` through the forensic streaming wrapper.
The live output is persisted under
`revision_artifacts/reports/revision/logs/history/retrain_best_ema_train/<run_id>.log`,
with start/end timestamps and return code. The model directory receives only a
latest-run convenience copy; the run-ID log is the durable monitoring record.

## Resume test

Perform this once for every expensive runner family:

1. Run exactly one fold or one stress condition.
2. Confirm its prediction/checkpoint and durable log are in the canonical manifest.
3. Disconnect and start a fresh runtime.
4. Run Notebook 00 setup.
5. Reopen the producing notebook and confirm the completed unit is authenticated and reused.
6. Confirm only missing units are scheduled.

Any `.partial.*` file must be ignored or quarantined. A final filename without a matching canonical manifest SHA is not reusable.

If Colab disconnects during the short mirror-commit window, the canonical root
may retain `.artifact_mirror.publish.lock`. Do not delete it manually. Inspect
the lock metadata and confirm that no earlier runtime is still publishing; only
then use `artifact_mirror.py publish --recover-stale-lock` after the configured
stale interval (six hours by default). A different-host lock is deliberately
not stolen automatically.

## Statistical rules

- Manuscript-ready record bootstrap requires the freeze-bound Chapman patient/group sidecar: exact record-order SHA, OOF SHA, sidecar SHA, reviewed source semantics/counts, and one record per authenticated group must all pass. A row bootstrap without this contract is exploratory only.
- If Notebook 02 regenerates OOF or the freeze manifest, rerun Notebook 03 calibration and every Notebook 05 robustness ledger whose manifest binds the previous OOF, freeze, or group-sidecar SHA.
- Independently recomputed F1, PR-AUC, ROC-AUC, Brier, NLL, ECE, calibration slope, and intercept must agree within `1e-8`.
- Q=3 record reconstruction must agree within `1e-6`.
- Percentile bootstrap intervals are pointwise effect-size uncertainty summaries.
- Do not report bootstrap-tail proportions as p-values.
- Use the word `significant` only for a pre-declared paired sign-flip or label-swap permutation test with at least 10,000 permutations and Holm correction for the declared family.
- Call ResNet1D/CNN, Raw Mamba, Transformer ECG, and morphology MLP results same-fold comparators unless the comparator contract proves matched training budgets.

## Claim boundaries

- Use `fixed-seed ROCKET-family MAX+PPV transform`, not canonical MiniRocket.
- Representation probes and fold CKA are selectivity/similarity audits, not proof of disentanglement.
- Score calibration is not model-weight adaptation.
- Frozen-encoder head training is not end-to-end fine-tuning.
- CPSC2021 is an annotation-aligned 10-second AF/AFL mapped-window task and is not pooled with record-level PTB-XL or Georgia results.
- Reserved rhythm/amplitude slots do not establish a full RMSSD/SDNN/LF-HF implementation.
- Do not claim clinical readiness, broad superiority, or general robustness superiority.

## Final gate and submission

Inspect `reports/revision/tables/table_forensic_rerun_dependencies.csv`. Rerun by dependency, not by file presence.

Notebook 07 first publishes outputs produced by the detached authority commit
with `source-conflict-policy newer`, runs the strict full-SHA storage gate, runs
the forensic gate, publishes the resulting audit outputs, and then runs the
strict full-SHA storage gate again. The second storage pass binds the updated
audit files into the final canonical state. `newer` is permitted only in this
sequence because the setup has already verified the canonical authority SHA and
a clean tracked worktree; downstream notebooks cannot rotate that SHA.

Notebook 07 may export the final snapshot only after both commands pass (the
storage command is also repeated after the forensic audit publish):

```bash
python -u scripts/revision/38_pipeline_storage_audit.py \
  --canonical-root /content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision \
  --strict --full-sha

python -u scripts/revision/47_forensic_notebook_audit.py \
  --canonical-root /content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision \
  --strict
```

Then regenerate manuscript tables from `table_final_evidence_matrix.csv` and `table_final_safe_wording.csv`, compile clean and marked PDFs, visually inspect both, and run the forbidden-claim scan on manuscript source, response, extracted PDF text, and final evidence tables.

Run the final scan from the repository root after PDF text extraction:

```bash
python -u scripts/revision/46_submission_claim_scan.py --strict \
  --path "/path/to/main.tex" \
  --path "/path/to/response_letter.md" \
  --path "/path/to/main.pdf.txt" \
  --path "/path/to/final_evidence_tables"
```

The scanner treats explicit boundaries such as “we do not claim zero-shot
superiority” as safe, while the corresponding positive assertion fails the
strict gate. Inspect every row in `table_submission_forbidden_claim_scan.csv`;
the automated result complements rather than replaces manuscript review.
