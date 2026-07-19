from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ComparatorTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = load_module(
            "comparator_contract_common_test",
            ROOT / "scripts" / "revision" / "common.py",
        )
        cls.audit = load_module(
            "comparator_contract_audit_test",
            ROOT / "scripts" / "revision" / "47_forensic_notebook_audit.py",
        )

    def test_optimizer_steps_are_derived_per_fold(self):
        contract = self.common.build_comparator_training_contract(
            display_name="Test comparator",
            n_records=10,
            folds=[2, 1],
            preprocessing={"input": "test"},
            training_unit="record",
            training_units_per_fold={1: 9, 2: 8},
            batch_size=4,
            epochs=3,
            loss="bce",
            regularization={"weight_decay": 0.0},
            amp=False,
            seed=42,
            fold_seed_formula="seed + fold",
            checkpoint_rule="fixed_final_epoch",
            tuning_provenance="fixed CLI",
            protocol="test",
        )
        self.assertEqual(contract["folds"], [1, 2])
        self.assertEqual(contract["optimizer_steps"]["total"], 15)
        self.assertEqual(
            [row["optimizer_steps"] for row in contract["optimizer_steps"]["per_fold"]],
            [9, 6],
        )
        self.assertEqual(contract["comparison_scope"], "same_folds_not_budget_matched")

    def test_missing_legacy_training_units_are_explicit(self):
        contract = self.common.build_comparator_training_contract(
            display_name="Legacy primary",
            n_records=10,
            folds=[1, 2],
            preprocessing="declared",
            training_unit="not_recorded",
            training_units_per_fold=None,
            batch_size="not_recorded_in_legacy_checkpoint",
            epochs=20,
            loss="declared",
            regularization="not_recorded_in_legacy_checkpoint",
            amp=True,
            seed="not_recorded_in_legacy_checkpoint",
            fold_seed_formula="not_recorded_in_legacy_checkpoint",
            checkpoint_rule="fixed_final_epoch",
            tuning_provenance="legacy",
            protocol="test",
            comparison_scope="primary_model_reference",
        )
        self.assertEqual(
            contract["optimizer_steps"]["total"],
            "not_recorded_in_legacy_checkpoint",
        )
        self.assertEqual(contract["comparison_scope"], "primary_model_reference")

    def test_full_model_does_not_relabel_inference_batch_as_training_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("metrics", "manifests", "predictions"):
                (root / directory).mkdir(parents=True)
            (root / "metrics" / "oof_final_ema_prediction_summary.json").write_text(
                json.dumps({"batch_size": 512, "protocol": "inference_only"}),
                encoding="utf-8",
            )
            np.savez_compressed(
                root / "predictions" / "oof_final_ema_predictions.npz",
                y_true=np.zeros((5, 1), dtype=np.int8),
                fold_id=np.arange(1, 6, dtype=np.int16),
            )
            rows, _ = self.audit.comparator_contract_rows(root, "a" * 64, "b" * 64)
            full = next(row for row in rows if row["comparator"] == "Full ECG-RAMBA")
            self.assertEqual(full["batch_size"], "not_declared")
            self.assertNotEqual(full["batch_size"], 512)

    def test_baseline_runners_emit_standard_contract(self):
        for relative in (
            "scripts/revision/01_generate_predictions.py",
            "scripts/revision/10_minirocket_only_baseline.py",
            "scripts/revision/14_resnet1d_cnn_baseline.py",
            "scripts/revision/16_raw_mamba_baseline.py",
            "scripts/revision/26_hybrid_morphology_baseline.py",
        ):
            with self.subTest(runner=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn('"comparator_contract"', source)
                self.assertIn("build_comparator_training_contract", source)


if __name__ == "__main__":
    unittest.main()
