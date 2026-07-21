from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "revision" / "41_reviewer_gap_closure.py"


def load_module():
    spec = importlib.util.spec_from_file_location("reviewer_gap_closure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ReviewerGapClosureContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.args = argparse.Namespace(
            external_table=root / "external.csv",
            external_summary=root / "external.json",
            external_manifest=root / "external_manifest.json",
            pooling_table=root / "pooling.csv",
            pooling_bootstrap=root / "pooling_bootstrap.json",
            pooling_manifest=root / "pooling_manifest.json",
            morphology_table=root / "morphology.csv",
            morphology_json=root / "morphology.json",
            morphology_manifest=root / "morphology_manifest.json",
            robustness_table=root / "robustness.csv",
            robustness_pairwise=root / "robustness_pairwise.json",
            robustness_manifest=root / "robustness_manifest.json",
        )
        self.robustness_bootstrap_contract = root / "calibration_ci.json"
        write_json(self.robustness_bootstrap_contract, {"status": "complete"})
        self._write_external()
        self._write_pooling()
        self._write_morphology()
        self._write_robustness()

    def tearDown(self):
        self.temp.cleanup()

    def runner_sha(self, name: str) -> str:
        return self.module.sha256_file(ROOT / "scripts" / "revision" / name)

    def output_row(self, path: Path) -> dict:
        return {"path": str(path), "sha256": self.module.sha256_file(path)}

    def _write_external(self):
        rows = []
        for dataset in self.module.DATASETS:
            for comparator in self.module.EXTERNAL_COMPARATORS:
                for metric in self.module.METRICS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "comparator": comparator,
                            "comparator_label": comparator,
                            "metric": metric,
                            "n_boot_valid": 1000,
                            "n_groups": 20,
                            "group_unit": "patient/source-record group",
                            "full_value": 0.5,
                            "comparator_value": 0.4,
                            "improvement_full_over_comparator": 0.1,
                            "improvement_ci_low": 0.05,
                            "improvement_ci_high": 0.15,
                            "inference_scope": "pointwise_percentile_ci_effect_size_only",
                            "null_test": "not_run",
                            "interpretation": "full_nominal_95ci_better",
                        }
                    )
        write_csv(self.args.external_table, rows)
        write_json(self.args.external_summary, {"status": "complete"})
        write_json(
            self.args.external_manifest,
            {
                "status": "complete",
                "failures": [],
                "runner_sha256": self.runner_sha("32_paired_external_comparators.py"),
                "outputs": [
                    self.output_row(self.args.external_table),
                    self.output_row(self.args.external_summary),
                ],
            },
        )

    def _write_pooling(self):
        rows = []
        for dataset in self.module.DATASETS:
            for method in self.module.POOLING_METHODS:
                rows.append(
                    {
                        "dataset": dataset,
                        "pooling": method,
                        "n_groups": 20,
                        "group_unit": "patient/source group",
                        "group_safe": True,
                        **{metric: 0.5 for metric in self.module.METRICS},
                    }
                )
        write_csv(self.args.pooling_table, rows)
        items = {}
        for dataset in self.module.DATASETS:
            for method in self.module.POOLING_METHODS:
                if method == "power_mean_q3":
                    continue
                for metric in self.module.METRICS:
                    items[f"{dataset}__q3_vs_{method}__{metric}"] = {
                        "point_delta_a_minus_b": 0.0,
                        "lo": -0.01,
                        "hi": 0.01,
                        "n_boot_valid": 1000,
                        "n_groups": 20,
                        "group_safe": True,
                    }
        write_json(self.args.pooling_bootstrap, {"n_boot": 1000, "items": items})
        write_json(
            self.args.pooling_manifest,
            {
                "status": True,
                "protocol": "external_pooling_sensitivity_v2_group_bootstrap",
                "strict_group_bootstrap": True,
                "datasets": list(self.module.DATASETS),
                "n_boot": 1000,
                "runner_sha256": self.runner_sha("30_pooling_sensitivity_external.py"),
                "outputs": [
                    self.output_row(self.args.pooling_table),
                    self.output_row(self.args.pooling_bootstrap),
                ],
            },
        )

    def _write_morphology(self):
        rows = []
        for comparison in ("partial_vs_frozen", "full_vs_partial"):
            for metric in self.module.METRICS:
                rows.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "n_boot_valid": 1000,
                        "first_value": 0.5,
                        "second_value": 0.4,
                        "improvement_first_over_second": 0.1,
                        "improvement_ci_low": 0.05,
                        "improvement_ci_high": 0.15,
                        "inference_scope": "pointwise_percentile_ci_effect_size_only",
                        "null_test": "not_run",
                        "interpretation": "full_nominal_95ci_better",
                    }
                )
        write_csv(self.args.morphology_table, rows)
        write_json(self.args.morphology_json, {"status": True})
        write_json(
            self.args.morphology_manifest,
            {
                "status": "complete",
                "runner_sha256": self.runner_sha("40_paired_morphology_learnability.py"),
                "outputs": [
                    self.output_row(self.args.morphology_table),
                    self.output_row(self.args.morphology_json),
                ],
            },
        )

    def _write_robustness(self):
        independence = {
            "unit": self.module.ROBUSTNESS_BOOTSTRAP_UNIT,
            "independence_contract": self.module.CHAPMAN_GROUP_SEMANTICS,
            "group_semantics_reference": self.module.CHAPMAN_GROUP_REFERENCE,
            "group_sidecar": "manifests/test_group_sidecar.npz",
            "group_sidecar_sha256": "1" * 64,
            "source": str(self.robustness_bootstrap_contract),
            "source_sha256": self.module.sha256_file(self.robustness_bootstrap_contract),
            "training_variability_scope": self.module.ROBUSTNESS_TRAINING_VARIABILITY_SCOPE,
        }
        stress_contracts = {stress: {"spec": {"name": stress}} for stress in self.module.STRESSES}
        artifact_status = []
        for comparator in ("full", *self.module.ROBUSTNESS_COMPARATORS):
            artifact_status.append({"comparator": comparator, "kind": "clean", "status": "ready"})
            for stress in self.module.STRESSES:
                artifact_status.append(
                    {"comparator": comparator, "kind": f"stress:{stress}", "status": "ready"}
                )
        rows = []
        for stress in self.module.STRESSES:
            for comparator in self.module.ROBUSTNESS_COMPARATORS:
                for metric in self.module.METRICS:
                    rows.append(
                        {
                            "stress": stress,
                            "comparator": comparator,
                            "comparator_label": comparator,
                            "metric": metric,
                            "status": "complete",
                            "n_boot_valid": 1000,
                            "clean_full": 0.5,
                            "stress_full": 0.4,
                            "degradation_full_benefit": -0.1,
                            "clean_comparator": 0.45,
                            "stress_comparator": 0.34,
                            "degradation_comparator_benefit": -0.11,
                            "degradation_advantage_full": 0.01,
                            "stressed_advantage_full": 0.06,
                            "degradation_adv_ci_low": -0.01,
                            "degradation_adv_ci_high": 0.02,
                            "stressed_adv_ci_low": 0.02,
                            "stressed_adv_ci_high": 0.10,
                            "ci_scope": self.module.ROBUSTNESS_CI_SCOPE,
                            "bootstrap_unit": self.module.ROBUSTNESS_BOOTSTRAP_UNIT,
                            "training_variability_scope": self.module.ROBUSTNESS_TRAINING_VARIABILITY_SCOPE,
                            "macro_class_support_policy": (
                                self.module.ROBUSTNESS_MACRO_CLASS_SUPPORT_POLICY
                            ),
                            "interpretation": "nominal_95ci_inconclusive_change_difference",
                        }
                    )
        write_csv(self.args.robustness_table, rows)
        write_json(
            self.args.robustness_pairwise,
            {
                "status": "complete",
                "n_boot": 1000,
                "output_profile": "canonical",
                "runner_sha256": self.runner_sha("21_robustness_multicomparator.py"),
                "ci_scope": self.module.ROBUSTNESS_CI_SCOPE,
                "bootstrap_unit": self.module.ROBUSTNESS_BOOTSTRAP_UNIT,
                "training_variability_scope": self.module.ROBUSTNESS_TRAINING_VARIABILITY_SCOPE,
                "metric_cache_schema_version": self.module.ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION,
                "macro_class_support_policy": self.module.ROBUSTNESS_MACRO_CLASS_SUPPORT_POLICY,
                "bootstrap_independence_contract": independence,
                "stress_contracts": stress_contracts,
            },
        )
        write_json(
            self.args.robustness_manifest,
            {
                "status": "complete",
                "n_boot": 1000,
                "output_profile": "canonical",
                "runner_sha256": self.runner_sha("21_robustness_multicomparator.py"),
                "ci_scope": self.module.ROBUSTNESS_CI_SCOPE,
                "bootstrap_unit": self.module.ROBUSTNESS_BOOTSTRAP_UNIT,
                "training_variability_scope": self.module.ROBUSTNESS_TRAINING_VARIABILITY_SCOPE,
                "metric_cache_schema_version": self.module.ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION,
                "macro_class_support_policy": self.module.ROBUSTNESS_MACRO_CLASS_SUPPORT_POLICY,
                "bootstrap_independence_contract": independence,
                "stress_contracts": stress_contracts,
                "artifact_status": artifact_status,
                "artifact_sha256": {
                    "table": self.module.sha256_file(self.args.robustness_table),
                    "pairwise": self.module.sha256_file(self.args.robustness_pairwise),
                },
            },
        )

    def test_all_exact_grids_are_accepted(self):
        for validator in (
            self.module.validate_morphology,
            self.module.validate_external,
            self.module.validate_pooling,
            self.module.validate_robustness,
        ):
            status, compact = validator(self.args)
            self.assertEqual(status["status"], "complete", status["issues"])
            self.assertTrue(status["manuscript_ready"])
            self.assertTrue(compact)

        external_status, external_compact = self.module.validate_external(self.args)
        self.assertIn("record-level resampling", external_status["safe_wording"])
        self.assertTrue(all("Inference scope" in row for row in external_compact))
        robustness_status, robustness_compact = self.module.validate_robustness(self.args)
        self.assertIn("pointwise", robustness_status["safe_wording"])
        self.assertTrue(all("Nominal pointwise CI overlaps zero" in row for row in robustness_compact))

    def test_short_pooling_bootstrap_is_rejected(self):
        payload = json.loads(self.args.pooling_bootstrap.read_text(encoding="utf-8"))
        first = next(iter(payload["items"].values()))
        first["n_boot_valid"] = 999
        write_json(self.args.pooling_bootstrap, payload)
        manifest = json.loads(self.args.pooling_manifest.read_text(encoding="utf-8"))
        manifest["outputs"][1] = self.output_row(self.args.pooling_bootstrap)
        write_json(self.args.pooling_manifest, manifest)
        status, _ = self.module.validate_pooling(self.args)
        self.assertEqual(status["status"], "incomplete")
        self.assertTrue(any("short" in issue for issue in status["issues"]))

    def test_pointwise_interval_controls_reviewer_facing_morphology_conclusion(self):
        with self.args.morphology_table.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["improvement_ci_low"] = "-0.01"
        rows[0]["interpretation"] = "full_nominal_95ci_better"
        write_csv(self.args.morphology_table, rows)
        manifest = json.loads(self.args.morphology_manifest.read_text(encoding="utf-8"))
        manifest["outputs"][0] = self.output_row(self.args.morphology_table)
        write_json(self.args.morphology_manifest, manifest)

        status, compact = self.module.validate_morphology(self.args)
        self.assertEqual(status["status"], "complete", status["issues"])
        first = next(row for row in compact if row["Comparison"] == "partial_vs_frozen")
        self.assertEqual(first["Inference scope"], "Pointwise 95% CI includes zero")

    def test_nonfinite_robustness_interval_is_rejected(self):
        with self.args.robustness_table.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["degradation_adv_ci_low"] = "nan"
        write_csv(self.args.robustness_table, rows)
        manifest = json.loads(self.args.robustness_manifest.read_text(encoding="utf-8"))
        manifest["artifact_sha256"]["table"] = self.module.sha256_file(self.args.robustness_table)
        write_json(self.args.robustness_manifest, manifest)

        status, _ = self.module.validate_robustness(self.args)
        self.assertEqual(status["status"], "incomplete")
        self.assertTrue(any("Non-finite" in issue for issue in status["issues"]))


if __name__ == "__main__":
    unittest.main()
