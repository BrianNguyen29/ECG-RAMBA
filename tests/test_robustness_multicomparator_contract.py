import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            "stress_json": np.asarray(
                json.dumps(ROBUSTNESS.expected_stress_spec("snr20db", 42), sort_keys=True)
            ),
            "model_label": np.asarray("Full ECG-RAMBA"),
            "metadata_json": np.asarray(json.dumps(metadata)),
        }
        expected_spec = ROBUSTNESS.expected_stress_spec("snr20db", 42)
        ROBUSTNESS.validate_stress_provenance("full", "snr20db", clean, stress, expected_spec)

        metadata["fold_rows"][2]["checkpoint_sha256"] = "wrong"
        stress["metadata_json"] = np.asarray(json.dumps(metadata))
        with self.assertRaisesRegex(RuntimeError, "frozen Full OOF contract"):
            ROBUSTNESS.validate_stress_provenance("full", "snr20db", clean, stress, expected_spec)

    def test_learned_stress_requires_exact_spec_and_clean_input_contract(self):
        expected_spec = ROBUSTNESS.expected_stress_spec("snr20db", 42)
        hashes = np.asarray([f"sha-{fold}" for fold in range(1, 6)])
        clean = {
            "checkpoint_sha256": hashes,
            "raw_cache_sha256": np.asarray("raw-sha"),
            "oof_predictions_sha256": np.asarray("oof-sha"),
            "freeze_manifest_sha256": np.asarray("freeze-sha"),
            "aggregation_implementation": np.asarray("power_mean_v2"),
            "power_mean_q": np.asarray(3.0),
        }
        stress = {
            "y_true": np.zeros((3, 2), dtype=np.float32),
            "checkpoint_sha256": hashes.copy(),
            "protocol": np.asarray("comparator_stress_predictions_v1_same_folds_power_mean_v2_q3"),
            "comparator": np.asarray("resnet"),
            "stress_test": np.asarray("snr20db"),
            "stress_metadata_json": np.asarray(json.dumps({"spec": expected_spec}, sort_keys=True)),
            "raw_cache_sha256": np.asarray("raw-sha"),
            "oof_predictions_sha256": np.asarray("oof-sha"),
            "freeze_manifest_sha256": np.asarray("freeze-sha"),
            "aggregation_implementation": np.asarray("power_mean_v2"),
            "power_mean_q": np.asarray(3.0),
            "slice_count": np.ones(3, dtype=np.int16),
        }
        ROBUSTNESS.validate_stress_provenance(
            "resnet", "snr20db", clean, stress, expected_spec
        )

        stress["raw_cache_sha256"] = np.asarray("stale-raw")
        with self.assertRaisesRegex(RuntimeError, "raw_cache_sha256"):
            ROBUSTNESS.validate_stress_provenance(
                "resnet", "snr20db", clean, stress, expected_spec
            )
        stress["raw_cache_sha256"] = np.asarray("raw-sha")
        stress["stress_metadata_json"] = np.asarray(
            json.dumps({"spec": {**expected_spec, "seed": 999}}, sort_keys=True)
        )
        with self.assertRaisesRegex(RuntimeError, "perturbation specification mismatch"):
            ROBUSTNESS.validate_stress_provenance(
                "resnet", "snr20db", clean, stress, expected_spec
            )

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
            "comparator_nominal_95ci_more_favorable_change",
        )

    def test_legacy_metric_cache_requires_exact_numeric_contract_attestation(self):
        expected_metadata = {
            "protocol": ROBUSTNESS.PROTOCOL,
            "stress": "snr5db",
            "comparator": "resnet",
            "metric": "pr_auc_macro",
            "metric_cache_schema_version": ROBUSTNESS.METRIC_CACHE_SCHEMA_VERSION,
            "bootstrap_engine_contract": ROBUSTNESS.BOOTSTRAP_ENGINE,
            "macro_class_support_policy": ROBUSTNESS.MACRO_CLASS_SUPPORT_POLICY,
        }
        legacy_metadata = {
            key: value
            for key, value in expected_metadata.items()
            if key
            not in {
                "metric_cache_schema_version",
                "bootstrap_engine_contract",
                "macro_class_support_policy",
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.json"
            path.write_text(
                json.dumps(
                    {
                        "metadata": legacy_metadata,
                        "row": {
                            "bootstrap_engine": "record_index_resample_v1_cached_exact",
                            "degradation_adv_ci_low": -0.1,
                            "degradation_adv_ci_high": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = ROBUSTNESS.read_metric_cache(path, expected_metadata)
            self.assertIsNotNone(row)
            self.assertEqual(
                row["metric_cache_compatibility_attestation"],
                "legacy_schema_exact_regression_parity_verified",
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["row"]["bootstrap_engine"] = "unknown_buggy_engine"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(ROBUSTNESS.read_metric_cache(path, expected_metadata))

    def test_bootstrap_unit_contract_is_bound_to_current_oof_and_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            canonical = {"oof_sha256": "oof", "freeze_sha256": "freeze"}
            path.write_text(
                json.dumps(
                    {
                        "bootstrap": {
                            "unit": "chapman_record_subject",
                            "independence_contract": "one_chapman_record_per_subject",
                        },
                        "predictions_sha256": "oof",
                        "freeze_manifest_sha256": "freeze",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(ROBUSTNESS, "CALIBRATION_CI", path):
                contract = ROBUSTNESS.load_bootstrap_independence_contract(canonical)
                self.assertEqual(contract["unit"], ROBUSTNESS.BOOTSTRAP_UNIT)
                self.assertEqual(
                    contract["training_variability_scope"],
                    ROBUSTNESS.TRAINING_VARIABILITY_SCOPE,
                )
                with self.assertRaisesRegex(RuntimeError, "different canonical OOF"):
                    ROBUSTNESS.load_bootstrap_independence_contract(
                        {"oof_sha256": "new", "freeze_sha256": "freeze"}
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

    def test_fast_engine_matches_bin_edges_and_degenerate_labels(self):
        n_records = 180
        y_true = np.column_stack(
            [
                np.zeros(n_records),
                np.ones(n_records),
                np.arange(n_records) % 2,
                (np.arange(n_records) % 3) == 0,
            ]
        ).astype(np.float32)
        bin_edges = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        y_prob = np.column_stack(
            [np.roll(np.resize(bin_edges, n_records), shift) for shift in range(y_true.shape[1])]
        ).astype(np.float32)
        data = {"y_true": y_true, "y_prob": y_prob}
        rng = np.random.default_rng(91)
        indices = rng.integers(0, n_records, size=n_records)
        counts = np.bincount(indices, minlength=n_records)

        for spec in ROBUSTNESS.metric_specs(threshold=0.5, n_bins=15):
            expected = ROBUSTNESS.metric_value(spec, data, indices)
            observed = ROBUSTNESS.weighted_resample_metric(spec, data, counts)
            tolerance = 1e-7 if spec["name"] in {"brier_macro", "ece_macro"} else 1e-12
            self.assertAlmostEqual(expected, observed, delta=tolerance, msg=spec["name"])

    def test_fast_paired_intervals_match_legacy_record_resampling(self):
        rng = np.random.default_rng(121)
        y_true = rng.integers(0, 2, size=(80, 4)).astype(np.float32)

        def data(offset: float):
            y_prob = np.clip(
                0.1 + 0.7 * y_true + offset + rng.normal(0.0, 0.15, y_true.shape),
                0.0,
                1.0,
            ).astype(np.float32)
            return {"y_true": y_true, "y_prob": y_prob}

        datasets = (data(0.0), data(-0.04), data(0.02), data(-0.01))
        n_boot, seed = 25, 72
        for spec in ROBUSTNESS.metric_specs(threshold=0.5, n_bins=15):
            observed = ROBUSTNESS.paired_bootstrap(spec, *datasets, n_boot, seed, n_jobs=3)
            legacy_rng = np.random.default_rng(seed)
            degradation_values = []
            stressed_values = []
            for _ in range(n_boot):
                indices = legacy_rng.integers(0, len(y_true), size=len(y_true))
                fc, fs, cc, cs = (
                    ROBUSTNESS.metric_value(spec, dataset, indices) for dataset in datasets
                )
                full_degradation = ROBUSTNESS.benefit(fs, spec["direction"]) - ROBUSTNESS.benefit(
                    fc,
                    spec["direction"],
                )
                comparator_degradation = ROBUSTNESS.benefit(
                    cs,
                    spec["direction"],
                ) - ROBUSTNESS.benefit(cc, spec["direction"])
                degradation_values.append(full_degradation - comparator_degradation)
                stressed_values.append(
                    ROBUSTNESS.benefit(fs, spec["direction"])
                    - ROBUSTNESS.benefit(cs, spec["direction"])
                )

            degradation_ci = np.quantile(degradation_values, [0.025, 0.975])
            stressed_ci = np.quantile(stressed_values, [0.025, 0.975])
            tolerance = 1e-7 if spec["name"] in {"brier_macro", "ece_macro"} else 1e-12
            self.assertAlmostEqual(
                observed["degradation_adv_mean"],
                float(np.mean(degradation_values)),
                delta=tolerance,
                msg=spec["name"],
            )
            self.assertAlmostEqual(observed["degradation_adv_ci_low"], degradation_ci[0], delta=tolerance)
            self.assertAlmostEqual(observed["degradation_adv_ci_high"], degradation_ci[1], delta=tolerance)
            self.assertAlmostEqual(observed["stressed_adv_ci_low"], stressed_ci[0], delta=tolerance)
            self.assertAlmostEqual(observed["stressed_adv_ci_high"], stressed_ci[1], delta=tolerance)

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
            bootstrap_source = metrics / "calibration_bootstrap_contract.json"
            bootstrap_source.write_text(
                json.dumps(
                    {
                        "bootstrap": {
                            "unit": "chapman_record_subject",
                            "independence_contract": "one_chapman_record_per_subject",
                        }
                    }
                ),
                encoding="utf-8",
            )
            bootstrap_contract = {
                "unit": READINESS.ROBUSTNESS_BOOTSTRAP_UNIT,
                "independence_contract": "one_chapman_record_per_subject",
                "source": bootstrap_source.relative_to(READINESS.PROJECT_ROOT).as_posix(),
                "source_sha256": sha256_file(bootstrap_source),
                "training_variability_scope": READINESS.ROBUSTNESS_TRAINING_VARIABILITY_SCOPE,
            }
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
                                "bootstrap_unit": READINESS.ROBUSTNESS_BOOTSTRAP_UNIT,
                                "training_variability_scope": (
                                    READINESS.ROBUSTNESS_TRAINING_VARIABILITY_SCOPE
                                ),
                                "ci_scope": READINESS.ROBUSTNESS_CI_SCOPE,
                                "macro_class_support_policy": (
                                    READINESS.ROBUSTNESS_MACRO_CLASS_SUPPORT_POLICY
                                ),
                                "interpretation": "nominal_95ci_inconclusive_change_difference",
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
                "bootstrap_unit": READINESS.ROBUSTNESS_BOOTSTRAP_UNIT,
                "bootstrap_independence_contract": bootstrap_contract,
                "training_variability_scope": READINESS.ROBUSTNESS_TRAINING_VARIABILITY_SCOPE,
                "ci_scope": READINESS.ROBUSTNESS_CI_SCOPE,
                "metric_cache_schema_version": READINESS.ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION,
                "macro_class_support_policy": READINESS.ROBUSTNESS_MACRO_CLASS_SUPPORT_POLICY,
                "stress_contracts": {
                    stress: {"spec": ROBUSTNESS.expected_stress_spec(stress, 42)}
                    for stress in READINESS.ROBUSTNESS_STRESSES
                },
                "blocked_rows": 0,
                "completed_rows": expected_rows,
                "artifact_status": [
                    {"comparator": comparator, "kind": kind, "status": "ready"}
                    for comparator in READINESS.ROBUSTNESS_COMPARATORS
                    for kind in (
                        "clean",
                        *(f"stress:{stress}" for stress in READINESS.ROBUSTNESS_STRESSES),
                    )
                ],
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
                            "bootstrap_unit": READINESS.ROBUSTNESS_BOOTSTRAP_UNIT,
                            "bootstrap_independence_contract": bootstrap_contract,
                            "training_variability_scope": (
                                READINESS.ROBUSTNESS_TRAINING_VARIABILITY_SCOPE
                            ),
                            "ci_scope": READINESS.ROBUSTNESS_CI_SCOPE,
                            "metric_cache_schema_version": (
                                READINESS.ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION
                            ),
                            "macro_class_support_policy": (
                                READINESS.ROBUSTNESS_MACRO_CLASS_SUPPORT_POLICY
                            ),
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
