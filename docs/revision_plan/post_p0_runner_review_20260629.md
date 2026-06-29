# Post-P0 Runner Review

Date: 2026-06-29

Scope:

- `scripts/revision/19_fewshot_adaptation.py`
- `scripts/revision/20_representation_probe.py`
- `scripts/revision/21_robustness_multicomparator.py`
- `docs/revision_plan/post_p0_full_execution_roadmap_20260629.md`
- `docs/revision_plan/reviewer_requirements_gap_matrix_20260629.md`

## Review Result

The newly added post-P0 runners are implemented as gated/deferred-evidence tooling. They do not create manuscript-ready evidence unless the required upstream artifacts exist and pass their protocol checks.

## Fixes Applied During Review

1. Few-shot split integrity:
   - Before review, `19_fewshot_adaptation.py` created a fresh train/test split for each labeled fraction.
   - This could make 0/1/5/10 percent few-shot rows use different test sets, which is not a valid few-shot comparison.
   - The runner now freezes one test split per seed before selecting any labeled fraction.
   - The few-shot train subsets are nested prefixes of the remaining shuffled target-domain pool.
   - The split NPZ records fixed test indices/record IDs and candidate pool order for reuse/audit.

2. Few-shot input validation:
   - Fractions are validated to be within `[0, 1]`.
   - Seed list must be non-empty.
   - `--test-fraction` must be strictly between 0 and 1.
   - Fractions are normalized to sorted unique values in the manifest.

3. Robustness multi-comparator contract handling:
   - `21_robustness_multicomparator.py` no longer deletes comparator entries while iterating over a live dictionary view.
   - Contract-failed comparators are removed from a snapshot iteration, avoiding `RuntimeError: dictionary changed size during iteration`.

4. Planning documents:
   - The roadmap now states that `19_fewshot_adaptation.py` is score-calibration only, not model-weight few-shot transfer.
   - The roadmap now states that `20_representation_probe.py` consumes an embedding NPZ and still needs a separate embedding extraction hook.
   - The roadmap now states that `21_robustness_multicomparator.py` aggregates existing stress predictions and does not generate ResNet/Raw-Mamba stress predictions.
   - The gap matrix now distinguishes implemented runners from completed manuscript evidence.

## Verification Commands

Syntax check:

```text
python -m py_compile scripts/revision/19_fewshot_adaptation.py scripts/revision/20_representation_probe.py scripts/revision/21_robustness_multicomparator.py
```

Result: passed.

CLI help checks:

```text
python scripts/revision/19_fewshot_adaptation.py --help
python scripts/revision/20_representation_probe.py --help
python scripts/revision/21_robustness_multicomparator.py --help
```

Result: passed.

Few-shot valid smoke test:

- Created a small temporary gated prediction NPZ.
- Ran `19_fewshot_adaptation.py` with fractions `0.4,0,0.2`, seed `7`, and `n_boot=3`.
- Verified:
  - fixed test set across all fractions: `True`;
  - nested train sets: `True`;
  - train sets match prefixes of the recorded pool order: `True`;
  - manifest fractions normalized to `[0.0, 0.2, 0.4]`.

Representation blocked-mode smoke test:

- Ran `20_representation_probe.py` without `--embedding-npz`.
- Result: wrote `blocked_missing_embeddings` artifacts and did not fabricate representation evidence.

Robustness blocked-mode smoke test:

- Ran `21_robustness_multicomparator.py` with missing local frozen Full prediction artifacts.
- Result: wrote `blocked_missing_full_clean_predictions` artifacts and did not fabricate robustness evidence.

## Current Evidence Boundary

The current implementation is technically suitable as controlled infrastructure, but it does not change manuscript claims by itself.

Allowed:

- Use these scripts to run future gated workstreams with manifests/logs/cacheable outputs.
- Use blocked outputs to document that evidence remains deferred.

Not allowed:

- Do not claim few-shot experiments are complete until a protocol-gated external dataset is run and artifacts are generated.
- Do not claim morphology-rhythm disentanglement until checkpoint-fingerprinted embeddings are extracted and `20_representation_probe.py` completes.
- Do not claim broad robustness superiority until ResNet1D/CNN and Raw Mamba stress predictions exist and `21_robustness_multicomparator.py` produces complete paired degradation CIs.

## Remaining Technical Work

1. Georgia/CPSC2021 gates remain deferred unless their mapping/annotation protocol is completed.
2. Few-shot evidence requires a passed external gate and then a real run of `19_fewshot_adaptation.py`.
3. Representation evidence requires an embedding extraction hook and a real embedding NPZ.
4. Multi-comparator robustness requires stress prediction generation for ResNet1D/CNN and Raw Mamba.
5. Full HRV claims still require retraining with a new HRV schema; they cannot be retrofitted into current final EMA checkpoints.
