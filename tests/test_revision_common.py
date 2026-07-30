import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from configs.config import setup_paths
from scripts.revision.common import (
    aggregate_record_probabilities,
    brier_macro_only,
    calibration_summary,
    ece_macro_only,
    f1_macro_only,
    multilabel_metrics,
    power_mean,
    save_csv,
)


class PowerMeanTests(unittest.TestCase):
    def setUp(self):
        self.probs = np.asarray(
            [
                [0.10, 0.20, 0.90],
                [0.90, 0.80, 0.30],
            ],
            dtype=np.float64,
        )

    def test_q3_matches_generalized_mean_definition(self):
        expected = np.mean(self.probs**3, axis=0) ** (1.0 / 3.0)
        np.testing.assert_allclose(
            power_mean(self.probs, q=3.0, axis=0),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_q1_is_arithmetic_mean(self):
        np.testing.assert_allclose(
            power_mean(self.probs, q=1.0, axis=0),
            np.mean(self.probs, axis=0),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_q0_is_geometric_mean(self):
        expected = np.exp(np.mean(np.log(self.probs), axis=0))
        np.testing.assert_allclose(
            power_mean(self.probs, q=0.0, axis=0),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_q_changes_result(self):
        q2 = power_mean(self.probs, q=2.0, axis=0)
        q8 = power_mean(self.probs, q=8.0, axis=0)
        self.assertFalse(np.allclose(q2, q8))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            power_mean(np.empty((0, 3)), q=3.0, axis=0)

    def test_reaggregation_groups_slices_by_record(self):
        slice_prob = np.asarray(
            [
                [0.10, 0.20],
                [0.90, 0.80],
                [0.25, 0.75],
            ],
            dtype=np.float32,
        )
        record_id = np.asarray([0, 0, 1], dtype=np.int64)
        y_prob, valid, counts = aggregate_record_probabilities(
            slice_prob,
            record_id,
            2,
            q=3.0,
        )
        np.testing.assert_allclose(
            y_prob[0],
            np.mean(slice_prob[:2] ** 3, axis=0) ** (1.0 / 3.0),
            rtol=1e-6,
        )
        np.testing.assert_allclose(y_prob[1], slice_prob[2], rtol=1e-6)
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_array_equal(counts, [2, 1])


class DatasetPathTests(unittest.TestCase):
    def test_named_chapman_archive_precedes_legacy_archive(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            named = root / "WFDB-ChapmanShaoxing.zip"
            legacy = root / "archive.zip"
            named.touch()
            legacy.touch()
            with patch.dict(
                "os.environ",
                {
                    "ECG_RAMBA_DRIVE_ROOT": str(root),
                    "ECG_RAMBA_CHAPMAN_ZIP": "",
                },
                clear=False,
            ):
                paths = setup_paths(27, 3072, "test")
            self.assertEqual(Path(paths["zip_path"]), named)

    def test_explicit_chapman_archive_overrides_named_candidates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            named = root / "WFDB-ChapmanShaoxing.zip"
            explicit = root / "custom-chapman.zip"
            named.touch()
            explicit.touch()
            with patch.dict(
                "os.environ",
                {
                    "ECG_RAMBA_DRIVE_ROOT": str(root),
                    "ECG_RAMBA_CHAPMAN_ZIP": str(explicit),
                },
                clear=False,
            ):
                paths = setup_paths(27, 3072, "test")
            self.assertEqual(Path(paths["zip_path"]), explicit)

    def test_explicit_model_directory_precedes_repo_candidates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "drive-model"
            model_dir.mkdir()
            (model_dir / "fold1_best.pt").touch()
            with patch.dict(
                "os.environ",
                {
                    "ECG_RAMBA_DRIVE_ROOT": str(root),
                    "ECG_RAMBA_MODEL_DIR": str(model_dir),
                },
                clear=False,
            ):
                paths = setup_paths(27, 3072, "test")
            self.assertEqual(Path(paths["model_dir"]), model_dir)


class CsvOutputTests(unittest.TestCase):
    def test_heterogeneous_rows_use_stable_union_of_columns(self):
        rows = [
            {"dataset": "ptbxl", "model": "full", "primary_value": 0.6},
            {
                "dataset": "ptbxl",
                "model": "full",
                "comparator": "resnet",
                "improvement_full_over_comparator": -0.1,
            },
        ]
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "primary.csv"
            save_csv(output, rows)
            with output.open(newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))

        self.assertEqual(
            list(saved[0]),
            [
                "dataset",
                "model",
                "primary_value",
                "comparator",
                "improvement_full_over_comparator",
            ],
        )
        self.assertEqual(saved[0]["comparator"], "")
        self.assertEqual(saved[1]["primary_value"], "")
        self.assertEqual(saved[1]["comparator"], "resnet")


class ScalarMetricParityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260730)
        self.y_true = rng.binomial(1, 0.2, size=(2048, 7)).astype(np.int8)
        self.y_prob = rng.random((2048, 7), dtype=np.float64)

    def assert_scalar_equal(self, observed, expected):
        self.assertAlmostEqual(observed, expected, places=14)

    def test_specialized_metrics_match_full_summaries(self):
        metrics = multilabel_metrics(self.y_true, self.y_prob, threshold=0.5)
        calibration = calibration_summary(self.y_true, self.y_prob, n_bins=15)
        self.assert_scalar_equal(f1_macro_only(self.y_true, self.y_prob, threshold=0.5), metrics["f1_macro"])
        self.assert_scalar_equal(brier_macro_only(self.y_true, self.y_prob), calibration["brier_macro"])
        self.assert_scalar_equal(ece_macro_only(self.y_true, self.y_prob, n_bins=15), calibration["ece_macro"])

    def test_specialized_metrics_match_on_bootstrap_resamples(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            idx = rng.integers(0, len(self.y_true), size=len(self.y_true))
            y_true = self.y_true[idx]
            y_prob = self.y_prob[idx]
            metrics = multilabel_metrics(y_true, y_prob, threshold=0.5)
            calibration = calibration_summary(y_true, y_prob, n_bins=15)
            self.assert_scalar_equal(f1_macro_only(y_true, y_prob, threshold=0.5), metrics["f1_macro"])
            self.assert_scalar_equal(brier_macro_only(y_true, y_prob), calibration["brier_macro"])
            self.assert_scalar_equal(ece_macro_only(y_true, y_prob, n_bins=15), calibration["ece_macro"])

    def test_single_label_semantics_match(self):
        y_true = self.y_true[:, :1]
        y_prob = self.y_prob[:, :1]
        metrics = multilabel_metrics(y_true, y_prob, threshold=0.5)
        calibration = calibration_summary(y_true, y_prob, n_bins=15)
        self.assert_scalar_equal(f1_macro_only(y_true, y_prob, threshold=0.5), metrics["f1_macro"])
        self.assert_scalar_equal(brier_macro_only(y_true, y_prob), calibration["brier_macro"])
        self.assert_scalar_equal(ece_macro_only(y_true, y_prob, n_bins=15), calibration["ece_macro"])


if __name__ == "__main__":
    unittest.main()
