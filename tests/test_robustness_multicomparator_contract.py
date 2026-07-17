import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.revision.common import sha256_file


ROBUSTNESS = importlib.import_module("scripts.revision.21_robustness_multicomparator")
READINESS = importlib.import_module("scripts.revision.28_claim_readiness_gates")


class RobustnessMulticomparatorContractTests(unittest.TestCase):
    def test_minirocket_uses_matching_robustness_clean_reference(self):
        self.assertEqual(
            ROBUSTNESS.COMPARATORS["minirocket"]["clean"],
            "robustness_minirocket_clean_ref_predictions.npz",
        )

    def test_sidecar_names_are_profile_scoped(self):
        canonical = Path("reports/revision/metrics/robustness_multicomparator_pairwise.json")
        screening = Path(
            "reports/revision/metrics/robustness_multicomparator_reviewer_minimal_pairwise.json"
        )
        self.assertEqual(
            ROBUSTNESS.comparator_sidecar_path(canonical, "resnet").name,
            "robustness_full_vs_resnet_comparison.json",
        )
        self.assertEqual(
            ROBUSTNESS.comparator_sidecar_path(screening, "resnet").name,
            "robustness_full_vs_resnet_reviewer_minimal_comparison.json",
        )

    def test_full_stress_checkpoint_sha_must_match_clean_oof_contract(self):
        hashes = [f"sha-{fold}" for fold in range(1, 6)]
        clean = {"checkpoint_sha256": np.asarray(hashes)}
        metadata = {
            "fold_rows": [
                {"fold": fold, "checkpoint_sha256": hashes[fold - 1]}
                for fold in range(1, 6)
            ]
        }
        stress = {
            "protocol": np.asarray("robustness_full_vs_minirocket_perturbation_v1"),
            "stress_name": np.asarray("snr20db"),
            "model_label": np.asarray("Full ECG-RAMBA"),
            "metadata_json": np.asarray(json.dumps(metadata)),
        }
        ROBUSTNESS.validate_stress_provenance("full", "snr20db", clean, stress)

        metadata["fold_rows"][2]["checkpoint_sha256"] = "wrong"
        stress["metadata_json"] = np.asarray(json.dumps(metadata))
        with self.assertRaisesRegex(RuntimeError, "frozen Full OOF contract"):
            ROBUSTNESS.validate_stress_provenance("full", "snr20db", clean, stress)

    def test_npz_loader_requires_safe_well_formed_prediction_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid.npz"
            np.savez_compressed(
                path,
                y_true=np.asarray([[0.0, 1.0]], dtype=np.float32),
                y_prob=np.asarray([[0.2, 0.8]], dtype=np.float32),
                record_id=np.asarray(["record-1"]),
                fold_id=np.asarray([1], dtype=np.int16),
                class_names=np.asarray(["A", "B"]),
            )
            loaded = ROBUSTNESS.load_npz(path)
            self.assertEqual(loaded["y_prob"].shape, (1, 2))

            unsafe = Path(tmp) / "unsafe.npz"
            np.savez_compressed(
                unsafe,
                y_true=np.asarray([[0.0]], dtype=np.float32),
                y_prob=np.asarray([[1.2]], dtype=np.float32),
                record_id=np.asarray([{"id": "record-1"}], dtype=object),
                fold_id=np.asarray([1], dtype=np.int16),
                class_names=np.asarray(["A"]),
            )
            with self.assertRaises(ValueError):
                ROBUSTNESS.load_npz(unsafe)

    def test_readiness_csv_loader_reads_present_rows(self):
        with tempfile.TemporaryDirectory(dir=READINESS.PROJECT_ROOT) as tmp:
            path = Path(tmp) / "rows.csv"
            path.write_text("status,n_boot\ncomplete,1000\n", encoding="utf-8")
            self.assertEqual(
                READINESS.read_csv_if_present(path),
                [{"status": "complete", "n_boot": "1000"}],
            )

    def test_readiness_accepts_extended_canonical_contract_only_when_core_matches(self):
        expected = {"oof_sha256": "oof", "freeze_sha256": "freeze"}
        self.assertTrue(
            READINESS.contract_matches(
                {**expected, "checkpoint_kind": "final_ema", "validated_records": 44186},
                expected,
            )
        )
        self.assertFalse(
            READINESS.contract_matches(
                {**expected, "freeze_sha256": "stale", "checkpoint_kind": "final_ema"},
                expected,
            )
        )

    def test_cached_bootstrap_row_resolves_interpretation_without_rebootstrap(self):
        cached = {
            "status": "complete",
            "interpretation": "comparator_significantly_less_degraded",
            "degradation_adv_ci_low": -0.2,
            "degradation_adv_ci_high": -0.1,
        }
        self.assertEqual(
            ROBUSTNESS.row_interpretation(cached),
            "comparator_significantly_less_degraded",
        )

    def test_threaded_bootstrap_preserves_sequential_samples_and_results(self):
        y_true = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)

        def data(probabilities):
            return {"y_true": y_true, "y_prob": np.asarray(probabilities, dtype=np.float32)[:, None]}

        spec = {
            "name": "mean_probability",
            "direction": "higher",
            "fn": lambda _y, probability: float(np.mean(probability)),
        }
        arguments = (
            spec,
            data([0.1, 0.9, 0.2, 0.8]),
            data([0.2, 0.8, 0.3, 0.7]),
            data([0.15, 0.85, 0.25, 0.75]),
            data([0.25, 0.75, 0.35, 0.65]),
            25,
            123,
        )
        sequential = ROBUSTNESS.paired_bootstrap(*arguments, n_jobs=1)
        threaded = ROBUSTNESS.paired_bootstrap(*arguments, n_jobs=4)
        self.assertEqual(sequential, threaded)

        shared = {}
        first = ROBUSTNESS.paired_bootstrap(
            *arguments,
            n_jobs=2,
            shared_full_cache=shared,
            shared_full_cache_key=("stress", "metric", 123),
        )
        second = ROBUSTNESS.paired_bootstrap(
            *arguments,
            n_jobs=4,
            shared_full_cache=shared,
            shared_full_cache_key=("stress", "metric", 123),
        )
        self.assertEqual(sequential, first)
        self.assertEqual(first, second)
        self.assertEqual(len(shared[("stress", "metric", 123)]), 25)

    def test_fast_weighted_engine_matches_explicit_record_resampling(self):
        rng = np.random.default_rng(20260717)
        y_true = rng.integers(0, 2, size=(240, 5)).astype(np.float32)
        # Quantization deliberately creates score ties, exercising the cached
        # threshold boundaries used by the exact PR/ROC implementation.
        y_prob = np.round(rng.uniform(0.0, 1.0, size=y_true.shape), 2).astype(np.float32)
        data = {"y_true": y_true, "y_prob": y_prob}
        specs = ROBUSTNESS.metric_specs(threshold=0.5, n_bins=15)

        for _ in range(20):
            indices = rng.integers(0, len(y_true), size=len(y_true))
            counts = np.bincount(indices, minlength=len(y_true)).astype(np.float64)
            for spec in specs:
                expected = ROBUSTNESS.metric_value(spec, data, indices)
                observed = ROBUSTNESS.weighted_resample_metric(spec, data, counts)
                tolerance = 1e-7 if spec["name"] in {"brier_macro", "ece_macro"} else 1e-12
                self.assertAlmostEqual(expected, observed, delta=tolerance, msg=spec["name"])

    def test_fast_standard_metric_bootstrap_is_thread_deterministic(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=(120, 3)).astype(np.float32)

        def data(offset: float):
            probability = np.clip(
                0.15 + 0.65 * y_true + offset + rng.normal(0.0, 0.08, size=y_true.shape),
                0.0,
                1.0,
            ).astype(np.float32)
            return {"y_true": y_true, "y_prob": probability}

        datasets = (data(0.00), data(-0.02), data(0.01), data(-0.01))
        for spec in ROBUSTNESS.metric_specs(threshold=0.5, n_bins=15):
            sequential = ROBUSTNESS.paired_bootstrap(spec, *datasets, 30, 123, n_jobs=1)
            threaded = ROBUSTNESS.paired_bootstrap(spec, *datasets, 30, 123, n_jobs=4)
            self.assertEqual(sequential, threaded, spec["name"])
            self.assertEqual(sequential["bootstrap_engine"], ROBUSTNESS.BOOTSTRAP_ENGINE)

    def test_seeded_record_count_cache_matches_explicit_bootstrap_draws(self):
        n_records, n_boot, seed = 123, 7, 20260717
        observed = ROBUSTNESS.bootstrap_record_counts(n_records, n_boot, seed)
        expected_rng = np.random.default_rng(seed)
        expected = tuple(
            np.bincount(
                expected_rng.integers(0, n_records, size=n_records),
                minlength=n_records,
            )
            for _ in range(n_boot)
        )
        self.assertIs(observed, ROBUSTNESS.bootstrap_record_counts(n_records, n_boot, seed))
        for observed_counts, expected_counts in zip(observed, expected):
            np.testing.assert_array_equal(observed_counts, expected_counts)

    def test_canonical_readiness_rejects_screening_profile(self):
        with tempfile.TemporaryDirectory(dir=READINESS.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            metrics = root / "metrics"
            tables = root / "tables"
            manifests = root / "manifests"
            for directory in (metrics, tables, manifests):
                directory.mkdir()

            summary = metrics / "robustness_multicomparator_summary.csv"
            table = tables / "table_robustness_multicomparator.csv"
            pairwise = metrics / "robustness_multicomparator_pairwise.json"
            manifest = manifests / "robustness_multicomparator_manifest.json"
            sidecars = {
                comparator: metrics / f"robustness_full_vs_{comparator}_comparison.json"
                for comparator in READINESS.ROBUSTNESS_LEARNED_COMPARATORS
            }
            canonical = {"oof_sha256": "oof-sha", "freeze_sha256": "freeze-sha"}
            runner_sha = sha256_file(
                READINESS.PROJECT_ROOT
                / "scripts"
                / "revision"
                / "21_robustness_multicomparator.py"
            )

            rows = []
            for stress in READINESS.ROBUSTNESS_STRESSES:
                for comparator in READINESS.ROBUSTNESS_PAIRED_COMPARATORS:
                    for metric in READINESS.ROBUSTNESS_METRICS:
                        rows.append(
                            {
                                "stress": stress,
                                "comparator": comparator,
                                "metric": metric,
                                "status": "complete",
                                "output_profile": "canonical",
                                "n_boot": str(READINESS.ROBUSTNESS_N_BOOT),
                            }
                        )
            for path in (summary, table):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            expected_rows = len(rows)
            pairwise_payload = {
                "status": "complete",
                "protocol": READINESS.ROBUSTNESS_PROTOCOL,
                "output_profile": "canonical",
                "canonical_contract": canonical,
                "runner_sha256": runner_sha,
                "comparators": READINESS.ROBUSTNESS_COMPARATORS,
                "stress_tests": READINESS.ROBUSTNESS_STRESSES,
                "metrics": READINESS.ROBUSTNESS_METRICS,
                "n_boot": READINESS.ROBUSTNESS_N_BOOT,
                "blocked_rows": 0,
                "completed_rows": expected_rows,
                "items": {f"row-{idx}": row for idx, row in enumerate(rows)},
            }
            pairwise.write_text(json.dumps(pairwise_payload), encoding="utf-8")
            pairwise_sha = sha256_file(pairwise)
            pairwise_rel = pairwise.relative_to(READINESS.PROJECT_ROOT).as_posix()

            for comparator, path in sidecars.items():
                comparator_rows = [row for row in rows if row["comparator"] == comparator]
                path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "protocol": READINESS.ROBUSTNESS_PROTOCOL,
                            "comparator": comparator,
                            "output_profile": "canonical",
                            "canonical_contract": canonical,
                            "runner_sha256": runner_sha,
                            "source_pairwise": pairwise_rel,
                            "source_pairwise_sha256": pairwise_sha,
                            "stress_tests": READINESS.ROBUSTNESS_STRESSES,
                            "metrics": READINESS.ROBUSTNESS_METRICS,
                            "n_boot": READINESS.ROBUSTNESS_N_BOOT,
                            "blocked_rows": 0,
                            "completed_rows": len(comparator_rows),
                            "rows": comparator_rows,
                        }
                    ),
                    encoding="utf-8",
                )

            manifest_payload = {
                **pairwise_payload,
                "outputs": {
                    "summary": summary.relative_to(READINESS.PROJECT_ROOT).as_posix(),
                    "table": table.relative_to(READINESS.PROJECT_ROOT).as_posix(),
                    "pairwise": pairwise_rel,
                    "manifest": manifest.relative_to(READINESS.PROJECT_ROOT).as_posix(),
                    "comparator_sidecars": {
                        comparator: path.relative_to(READINESS.PROJECT_ROOT).as_posix()
                        for comparator, path in sidecars.items()
                    },
                },
                "artifact_sha256": {
                    "summary": sha256_file(summary),
                    "table": sha256_file(table),
                    "pairwise": pairwise_sha,
                    "comparator_sidecars": {
                        comparator: sha256_file(path) for comparator, path in sidecars.items()
                    },
                },
            }
            manifest_payload.pop("items")
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

            issues = READINESS.robustness_contract_issues(
                manifest_path=manifest,
                pairwise_path=pairwise,
                summary_path=summary,
                table_path=table,
                sidecar_paths=sidecars,
                canonical=canonical,
            )
            self.assertEqual(issues, [])

            manifest_payload["output_profile"] = "reviewer_minimal"
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            issues = READINESS.robustness_contract_issues(
                manifest_path=manifest,
                pairwise_path=pairwise,
                summary_path=summary,
                table_path=table,
                sidecar_paths=sidecars,
                canonical=canonical,
            )
            self.assertIn("manifest.output_profile='reviewer_minimal'", issues)


if __name__ == "__main__":
    unittest.main()
