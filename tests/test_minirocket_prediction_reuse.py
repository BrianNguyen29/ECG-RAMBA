import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "revision"
    / "10_minirocket_only_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("minirocket_only_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MiniRocketPredictionReuseTest(unittest.TestCase):
    def test_stale_oof_sha_rejects_cache_without_aborting_runner(self):
        y = np.asarray([[1.0], [0.0]], dtype=np.float32)
        record_id = np.asarray([10, 11], dtype=np.int64)
        args = argparse.Namespace(limit_records=2)

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "minirocket_only_oof_predictions.npz"
            np.savez_compressed(
                cache,
                y_true=y,
                y_prob=np.asarray([[0.8], [0.2]], dtype=np.float32),
                record_id=record_id,
                fold_id=np.asarray([1, 2], dtype=np.int16),
                class_names=np.asarray(["class_a"]),
                oof_predictions_sha256=np.asarray("old-oof-sha"),
            )

            expected_metadata = {
                "oof_predictions_sha256": np.asarray("current-oof-sha")
            }
            output = io.StringIO()
            with mock.patch.object(
                MODULE, "_prediction_metadata", return_value=expected_metadata
            ), contextlib.redirect_stdout(output):
                result = MODULE.load_existing_prediction_npz(
                    cache,
                    y=y,
                    record_id=record_id,
                    class_names=["class_a"],
                    args=args,
                    load_info={},
                    model_name="test",
                    classifier_params={},
                )

        self.assertIsNone(result)
        self.assertIn("prediction NPZ rejected", output.getvalue())
        self.assertIn("the fold-safe heads will be refit", output.getvalue())
        self.assertIn("old-oof-sha != current-oof-sha", output.getvalue())

    def test_matching_contract_reuses_probabilities(self):
        y = np.asarray([[1.0], [0.0]], dtype=np.float32)
        y_prob = np.asarray([[0.8], [0.2]], dtype=np.float32)
        record_id = np.asarray([10, 11], dtype=np.int64)
        args = argparse.Namespace(limit_records=2)

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "minirocket_only_oof_predictions.npz"
            np.savez_compressed(
                cache,
                y_true=y,
                y_prob=y_prob,
                record_id=record_id,
                fold_id=np.asarray([1, 2], dtype=np.int16),
                class_names=np.asarray(["class_a"]),
                oof_predictions_sha256=np.asarray("current-oof-sha"),
            )

            with mock.patch.object(
                MODULE,
                "_prediction_metadata",
                return_value={
                    "oof_predictions_sha256": np.asarray("current-oof-sha")
                },
            ):
                result = MODULE.load_existing_prediction_npz(
                    cache,
                    y=y,
                    record_id=record_id,
                    class_names=["class_a"],
                    args=args,
                    load_info={},
                    model_name="test",
                    classifier_params={},
                )

        self.assertIsNotNone(result)
        reused_prob, fold_id = result
        np.testing.assert_array_equal(reused_prob, y_prob)
        np.testing.assert_array_equal(fold_id, np.asarray([1, 2], dtype=np.int16))


if __name__ == "__main__":
    unittest.main()
