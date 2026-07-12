import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


common = importlib.import_module("scripts.revision.common")
fewshot = importlib.import_module("scripts.revision.35_true_fewshot_head_adaptation")
representations = importlib.import_module("scripts.revision.34_extract_external_representations")
final_evidence = importlib.import_module("scripts.revision.13_final_evidence_matrix")
claim_readiness = importlib.import_module("scripts.revision.28_claim_readiness_gates")


class GroupSafeExternalStatisticsTests(unittest.TestCase):
    def test_balanced_group_split_is_deterministic_disjoint_and_group_complete(self):
        groups = np.repeat(np.asarray([f"g{idx:02d}" for idx in range(12)]), 2)
        y_true = np.tile(np.asarray([[1, 0], [0, 1]], dtype=np.float32), (12, 1))

        train_a, test_a, audit_a = common.balanced_group_train_test_split(
            y_true,
            groups,
            test_fraction=0.5,
            seed=42,
            n_candidates=64,
        )
        train_b, test_b, audit_b = common.balanced_group_train_test_split(
            y_true,
            groups,
            test_fraction=0.5,
            seed=42,
            n_candidates=64,
        )

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        self.assertEqual(audit_a, audit_b)
        self.assertFalse(set(train_a) & set(test_a))
        self.assertEqual(set(train_a) | set(test_a), set(np.unique(groups)))
        self.assertEqual(audit_a["coverage_failures"], 0)
        for group in np.unique(groups):
            rows = np.where(groups == group)[0]
            self.assertTrue(np.all(np.isin(groups[rows], train_a)) or np.all(np.isin(groups[rows], test_a)))

    def test_cluster_bootstrap_preserves_pairing_and_resamples_whole_groups(self):
        groups = np.repeat(np.asarray(["a", "b", "c", "d"]), 2)
        y_true = np.zeros((len(groups), 1), dtype=np.float32)
        comparator = np.linspace(0.1, 0.6, len(groups), dtype=np.float64)[:, None]
        full = comparator + 0.2
        metric = lambda _y, probability: float(np.mean(probability[:, 0]))

        interval = common.cluster_bootstrap_ci(
            y_true,
            full,
            groups,
            metric,
            n_boot=100,
            seed=7,
        )
        paired = common.paired_cluster_bootstrap_delta(
            y_true,
            full,
            comparator,
            groups,
            metric,
            n_boot=100,
            seed=7,
        )

        self.assertEqual(interval["n_groups"], 4)
        self.assertEqual(interval["n_boot_valid"], 100)
        self.assertEqual(interval["sample_unit"], "group")
        self.assertEqual(paired["n_groups"], 4)
        self.assertEqual(paired["n_boot_valid"], 100)
        self.assertAlmostEqual(paired["point_delta_a_minus_b"], 0.2, places=12)
        self.assertAlmostEqual(paired["lo"], 0.2, places=12)
        self.assertAlmostEqual(paired["hi"], 0.2, places=12)

    def test_fewshot_fractions_are_nested_group_prefixes(self):
        pool = np.asarray([f"p{idx}" for idx in range(10)])
        one_percent = fewshot.nested_groups(pool, 0.01)
        ten_percent = fewshot.nested_groups(pool, 0.10)
        half = fewshot.nested_groups(pool, 0.50)

        self.assertEqual(len(one_percent), 1)
        np.testing.assert_array_equal(one_percent, ten_percent)
        np.testing.assert_array_equal(half, pool[:5])
        self.assertTrue(set(ten_percent).issubset(set(half)))

    def test_representation_fingerprint_changes_with_source_provenance(self):
        contract = {
            "record_id": np.asarray(["r1", "r2"]),
            "group_id": np.asarray(["g1", "g2"]),
            "split_id": np.asarray(["test", "test"]),
            "class_names": np.asarray(["a"]),
            "y_true": np.asarray([[0], [1]], dtype=np.float32),
            "sha256": "a" * 64,
        }
        source_a = {"manifest_sha256": "b" * 64, "archive": {"sha256": "c" * 64}}
        source_b = {"manifest_sha256": "b" * 64, "archive": {"sha256": "d" * 64}}

        fingerprint_a = representations.input_fingerprint(contract, source_a)
        fingerprint_b = representations.input_fingerprint(contract, source_b)

        self.assertEqual(len(fingerprint_a), 64)
        self.assertNotEqual(fingerprint_a, fingerprint_b)

    def test_fold9_full_representation_source_is_bound_to_prediction_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external_root = root / "external"
            dataset_root = external_root / "ptbxl"
            dataset_root.mkdir(parents=True)
            prediction = dataset_root / "ptbxl_full_fold9_predictions.npz"
            source = {"path": prediction, "sha256": "a" * 64}
            archive = {
                "path": str(root / "PTB-XL.zip"),
                "name": "PTB-XL.zip",
                "size_bytes": 123,
                "fingerprint": "f" * 16,
                "sha256": "b" * 64,
            }
            manifest_path = dataset_root / "ptbxl_full_fold9_prediction_run_manifest.json"
            canonical = {"oof_sha256": "c" * 64, "freeze_sha256": "d" * 64}
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "ptbxl",
                        "runner_sha256": representations.sha256_file(
                            representations.PROJECT_ROOT
                            / "scripts"
                            / "revision"
                            / "03_generate_external_predictions.py"
                        ),
                        "canonical_contract": canonical,
                        "outputs": {prediction.name: {"sha256": source["sha256"]}},
                        "archive": {
                            "size_bytes": archive["size_bytes"],
                            "fingerprint": archive["fingerprint"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                dataset="ptbxl",
                external_root=external_root,
                output_tag="fold9",
                ptbxl_folds="9",
            )

            provenance = representations.validate_source_provenance(
                args,
                "full",
                source,
                archive,
                canonical,
            )
            self.assertEqual(provenance["manifest_sha256"], representations.sha256_file(manifest_path))
            self.assertEqual(provenance["archive"]["sha256"], archive["sha256"])

            with self.assertRaises(RuntimeError):
                representations.validate_source_provenance(
                    args,
                    "full",
                    source,
                    archive,
                    {"oof_sha256": "0" * 64, "freeze_sha256": "d" * 64},
                )

            archive["fingerprint"] = "e" * 16
            with self.assertRaises(RuntimeError):
                representations.validate_source_provenance(
                    args,
                    "full",
                    source,
                    archive,
                    canonical,
                )

    def test_final_adaptation_summary_filters_model_and_uses_prespecified_budget(self):
        fractions = [0.0, 0.01, 0.05, 0.10]
        full_f1 = {0.0: 0.2, 0.01: 0.3, 0.05: 0.9, 0.10: 0.4}
        rows = []
        for model in ("full", "resnet"):
            for seed in (1, 2):
                for fraction in fractions:
                    rows.append(
                        {
                            "model": model,
                            "seed": str(seed),
                            "fraction": str(fraction),
                            "mode": "zero_target_label" if fraction == 0 else "adapted",
                            "train_records_or_windows": str(int(100 * fraction)),
                            "test_records_or_windows": "50",
                            "f1_macro": str(full_f1[fraction] if model == "full" else 0.99),
                            "pr_auc_macro": "0.5",
                            "roc_auc_macro": "0.6",
                            "brier_macro": "0.1",
                            "ece_macro": "0.1",
                            "fold_heads": "0" if fraction == 0 else "5",
                        }
                    )
        canonical = {"oof_sha256": "a" * 64, "freeze_sha256": "b" * 64}
        runner = final_evidence.PROJECT_ROOT / "scripts" / "revision" / "35_true_fewshot_head_adaptation.py"
        manifest = {
            "status": "complete_true_classifier_head_adaptation",
            "protocol": "frozen_encoder_true_linear_head_adaptation_v2_group_safe_gated",
            "zero_group_overlap_all_splits": True,
            "canonical_contract": canonical,
            "runner_sha256": final_evidence.sha256_file(runner),
            "seeds": [1, 2],
        }

        summary = final_evidence.summarize_external_adaptation(
            rows,
            manifest,
            "ptbxl",
            expected_status="complete_true_classifier_head_adaptation",
            expected_protocol="frozen_encoder_true_linear_head_adaptation_v2_group_safe_gated",
            adaptation_label="true_fewshot",
            safe_wording="safe",
            canonical_contract=canonical,
            runner_name="35_true_fewshot_head_adaptation.py",
            model_filter="full",
            primary_fraction=0.10,
        )

        self.assertTrue(summary["complete"])
        self.assertAlmostEqual(summary["primary_fraction"]["f1_macro_mean"], 0.4)
        self.assertEqual(summary["primary_fraction"]["n_seeds"], 2)
        self.assertIn("pre-specified primary fraction=0.10", summary["key_numbers"])
        self.assertNotIn("F1-best", summary["key_numbers"])

    def test_claim_gate_requires_exact_adaptation_grid_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            rows = ["model,seed,fraction"]
            for model in ("full", "resnet"):
                for seed in claim_readiness.FEWSHOT_SEEDS:
                    for fraction in (0.0, 0.01, 0.05, 0.10):
                        rows.append(f"{model},{seed},{fraction}")
            summary.write_text("\n".join(rows) + "\n", encoding="utf-8")
            self.assertEqual(
                claim_readiness.adaptation_grid_issues(summary, model_filter="full"),
                [],
            )

            output = root / "artifact.csv"
            output.write_text("value\n1\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "outputs": [
                            {
                                "path": str(output),
                                "size_bytes": output.stat().st_size,
                                "sha256": claim_readiness.sha256_file(output),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(claim_readiness.manifest_output_issues(manifest, [output]), [])
            output.write_text("value\n2\n", encoding="utf-8")
            issues = claim_readiness.manifest_output_issues(manifest, [output])
            self.assertTrue(any("sha256_mismatch" in issue for issue in issues))

    def test_score_calibration_requires_prespecified_group_budget_contract(self):
        rows = []
        for seed in (1, 2):
            for fraction in (0.0, 0.01, 0.05, 0.10):
                rows.append(
                    {
                        "seed": str(seed),
                        "fraction": str(fraction),
                        "mode": "zero" if fraction == 0 else "calibrated",
                        "fraction_unit": "independent_target_groups_from_adaptation_pool",
                        "fraction_sampling": "nested_random_group_prefix_per_seed",
                        "f1_macro": "0.4",
                        "pr_auc_macro": "0.5",
                        "roc_auc_macro": "0.6",
                        "brier_macro": "0.1",
                        "ece_macro": "0.1",
                    }
                )
        canonical = {"oof_sha256": "a" * 64, "freeze_sha256": "b" * 64}
        runner = final_evidence.PROJECT_ROOT / "scripts" / "revision" / "33_group_safe_score_calibration.py"
        manifest = {
            "status": "complete_group_safe_score_calibration",
            "protocol": "group_safe_score_calibration_v2_gated_external",
            "zero_group_overlap_all_splits": True,
            "canonical_contract": canonical,
            "runner_sha256": final_evidence.sha256_file(runner),
            "seeds": [1, 2],
            "primary_fraction": 0.10,
            "primary_fraction_policy": "pre_specified_before_test_metric_evaluation",
            "fraction_unit": "independent_target_groups_from_adaptation_pool",
            "fraction_sampling": "nested_random_group_prefix_per_seed",
        }
        required_manifest = {
            "primary_fraction": 0.10,
            "primary_fraction_policy": "pre_specified_before_test_metric_evaluation",
            "fraction_unit": "independent_target_groups_from_adaptation_pool",
            "fraction_sampling": "nested_random_group_prefix_per_seed",
        }
        required_rows = {
            "fraction_unit": "independent_target_groups_from_adaptation_pool",
            "fraction_sampling": "nested_random_group_prefix_per_seed",
        }
        summary = final_evidence.summarize_external_adaptation(
            rows,
            manifest,
            "ptbxl",
            expected_status="complete_group_safe_score_calibration",
            expected_protocol="group_safe_score_calibration_v2_gated_external",
            adaptation_label="score_calibration",
            safe_wording="safe",
            canonical_contract=canonical,
            runner_name="33_group_safe_score_calibration.py",
            required_manifest_fields=required_manifest,
            required_row_fields=required_rows,
        )
        self.assertTrue(summary["complete"])
        rows[0].pop("fraction_unit")
        rejected = final_evidence.summarize_external_adaptation(
            rows,
            manifest,
            "ptbxl",
            expected_status="complete_group_safe_score_calibration",
            expected_protocol="group_safe_score_calibration_v2_gated_external",
            adaptation_label="score_calibration",
            safe_wording="safe",
            canonical_contract=canonical,
            runner_name="33_group_safe_score_calibration.py",
            required_manifest_fields=required_manifest,
            required_row_fields=required_rows,
        )
        self.assertFalse(rejected["complete"])

    def test_primary_endpoint_uses_shared_group_bootstrap_across_seeds_and_models(self):
        groups = np.repeat(np.asarray(["g1", "g2", "g3", "g4"]), 2)
        y_true = np.tile(np.asarray([[0], [1]], dtype=np.float32), (4, 1))
        zero = np.tile(np.asarray([[0.2], [0.8]], dtype=np.float32), (4, 1))
        full_seed_1 = np.tile(np.asarray([[0.1], [0.9]], dtype=np.float32), (4, 1))
        full_seed_2 = np.tile(np.asarray([[0.15], [0.85]], dtype=np.float32), (4, 1))
        resnet_seed_1 = np.tile(np.asarray([[0.25], [0.75]], dtype=np.float32), (4, 1))
        resnet_seed_2 = np.tile(np.asarray([[0.3], [0.7]], dtype=np.float32), (4, 1))

        rows, payload = fewshot.primary_endpoint_rows(
            dataset="ptbxl",
            y_true=y_true,
            groups=groups,
            predictions={
                "full": {1: full_seed_1, 2: full_seed_2},
                "resnet": {1: resnet_seed_1, 2: resnet_seed_2},
            },
            zero_probabilities={"full": zero, "resnet": zero},
            threshold=0.5,
            n_bins=5,
            n_boot=50,
            primary_fraction=0.10,
            bootstrap_seed=9,
        )

        self.assertEqual(len(rows), 15)
        self.assertEqual(payload["n_boot"], 50)
        self.assertEqual(payload["n_groups"], 4)
        paired_f1 = next(
            row
            for row in rows
            if row["comparison_type"] == "full_vs_comparator_at_primary_fraction"
            and row["metric"] == "f1_macro"
        )
        self.assertEqual(paired_f1["n_seeds"], 2)
        self.assertEqual(paired_f1["n_boot_valid"], 50)


if __name__ == "__main__":
    unittest.main()
