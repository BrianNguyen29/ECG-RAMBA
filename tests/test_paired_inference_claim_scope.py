from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_paired_helpers():
    path = ROOT / "scripts" / "revision" / "11_paired_full_vs_minirocket.py"
    spec = importlib.util.spec_from_file_location("paired_helpers_claim_scope", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_revision_script(module_name: str, filename: str):
    path = ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PairedInferenceClaimScopeTests(unittest.TestCase):
    def test_paired_loader_rejects_prediction_without_fold_id(self) -> None:
        helpers = load_paired_helpers()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing_fold_id.npz"
            np.savez_compressed(
                path,
                y_true=np.zeros((2, 1), dtype=np.float32),
                y_prob=np.full((2, 1), 0.5, dtype=np.float32),
                record_id=np.asarray([1, 2], dtype=np.int64),
                class_names=np.asarray(["A"]),
            )
            with self.assertRaisesRegex(KeyError, "fold_id"):
                helpers.load_prediction_set(path, "missing-fold")

    def test_paired_freeze_requires_live_authenticated_group_sidecar(self) -> None:
        helpers = load_paired_helpers()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            root = Path(temp_dir)
            prediction_path = root / "full.npz"
            np.savez_compressed(
                prediction_path,
                y_true=np.zeros((5, 1), dtype=np.float32),
                y_prob=np.full((5, 1), 0.5, dtype=np.float32),
                record_id=np.arange(5, dtype=np.int64),
                fold_id=np.arange(1, 6, dtype=np.int16),
                class_names=np.asarray(["A"]),
            )
            prediction = helpers.load_prediction_set(prediction_path, "full")
            sidecar = root / "groups.npz"
            sidecar.write_bytes(b"authenticated-groups")
            freeze_path = root / "freeze.json"
            relative_prediction = prediction_path.resolve().relative_to(ROOT.resolve()).as_posix()
            relative_sidecar = sidecar.resolve().relative_to(ROOT.resolve()).as_posix()
            freeze_path.write_text(
                helpers.json.dumps(
                    {
                        "status": "frozen",
                        "checkpoint_kind": "final_ema",
                        "validated_records": 5,
                        "n_classes": 1,
                        "group_contract": {
                            "status": "verified",
                            "group_semantics": helpers.CHAPMAN_GROUP_SEMANTICS,
                            "group_semantics_reference": helpers.CHAPMAN_GROUP_REFERENCE,
                            "bootstrap_unit": helpers.AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
                            "one_record_per_group": True,
                            "n_records": 5,
                            "n_groups": 5,
                            "sidecar": {
                                "path": relative_sidecar,
                                "sha256": helpers.sha256_file(sidecar),
                            },
                        },
                        "artifacts": [
                            {"path": relative_prediction, "sha256": prediction.sha256}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            helpers.validate_freeze_manifest(freeze_path, prediction, "final_ema")
            sidecar.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "sidecar_sha256"):
                helpers.validate_freeze_manifest(freeze_path, prediction, "final_ema")

    def test_percentile_bootstrap_does_not_emit_a_null_test_p_value(self) -> None:
        helpers = load_paired_helpers()
        y = np.asarray([[0], [1], [0], [1], [0], [1]], dtype=np.float32)
        full = np.asarray([[0.1], [0.9], [0.2], [0.8], [0.3], [0.7]], dtype=np.float32)
        comparator = np.full_like(full, 0.5)
        spec = helpers.MetricSpec(
            "brier_macro",
            "Brier macro",
            "calibration",
            False,
            lambda truth, prob: float(np.mean((truth - prob) ** 2)),
        )
        result, _ = helpers.paired_bootstrap_difference(
            y_true=y,
            full_prob=full,
            comparator_prob=comparator,
            spec=spec,
            n_boot=200,
            seed=42,
        )
        self.assertTrue(math.isnan(result["p_value_two_sided"]))
        self.assertEqual(
            helpers.interpretation_from_ci(result["ci_low"], result["ci_high"]),
            "full_nominal_95ci_better",
        )

    def test_reviewer_runners_do_not_advertise_bootstrap_tail_significance(self) -> None:
        names = [
            "11_paired_full_vs_minirocket.py",
            "15_paired_full_vs_resnet.py",
            "17_paired_full_vs_raw_mamba.py",
            "25_paired_full_vs_transformer.py",
            "27_paired_full_vs_hybrid_morphology.py",
            "32_paired_external_comparators.py",
            "40_paired_morphology_learnability.py",
            "43_structured_ablation_5fold.py",
        ]
        for name in names:
            text = (ROOT / "scripts" / "revision" / name).read_text(encoding="utf-8")
            self.assertNotIn("two-sided sign bootstrap", text, name)
            self.assertNotIn("full_significantly_better", text, name)
            self.assertNotIn("comparator_significantly_better", text, name)

    def test_generic_paired_outputs_require_group_contract_and_exact_bootstrap(self) -> None:
        group = {
            "status": "verified",
            "bootstrap_unit": "chapman_record_subject",
            "one_record_per_group": True,
            "n_records": 5,
            "n_groups": 5,
            "group_semantics": "reviewed",
            "group_semantics_reference": "source",
            "sidecar": {"path": "groups.npz", "sha256": "a" * 64},
        }
        valid_ci = {
            "n_boot_valid": 10,
            "bootstrap_mean": 0.0,
            "ci_low": -0.1,
            "ci_high": 0.1,
            "raw_diff_ci_low": -0.1,
            "raw_diff_ci_high": 0.1,
        }
        for module_name, filename in [
            ("paired_transformer_hardening", "25_paired_full_vs_transformer.py"),
            ("paired_morphology_hardening", "40_paired_morphology_learnability.py"),
        ]:
            module = load_revision_script(module_name, filename)
            contract = module.authenticated_group_contract({"group_contract": group})
            self.assertEqual(contract["group_sidecar_sha256"], "a" * 64)
            module.validate_paired_interval(valid_ci, [{} for _ in range(10)], 10)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                module.validate_paired_interval(valid_ci, [{} for _ in range(9)], 10)
            with self.assertRaisesRegex(RuntimeError, "legacy significance"):
                module.validate_paired_interval(
                    {**valid_ci, "interpretation": "full_significantly_better"},
                    [{} for _ in range(10)],
                    10,
                )

            source = (ROOT / "scripts" / "revision" / filename).read_text(encoding="utf-8")
            self.assertIn('"group_contract": group_contract', source)
            self.assertNotIn('"p_value_two_sided": ci["p_value_two_sided"]', source)

    def test_robustness_percentile_bootstrap_does_not_emit_tail_p_values(self) -> None:
        text = (ROOT / "scripts" / "revision" / "12_robustness_stress.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("def p_two_sided", text)
        self.assertIn('INFERENCE_SCOPE = "pointwise_percentile_ci_effect_size_only"', text)
        self.assertIn('MULTIPLICITY_ADJUSTMENT = "not_applicable_no_null_test"', text)
        self.assertIn('"cache_version": 2', text)


if __name__ == "__main__":
    unittest.main()
