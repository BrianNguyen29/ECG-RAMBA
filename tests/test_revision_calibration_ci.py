import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.revision.common import sha256_file


calibration_ci = importlib.import_module("scripts.revision.04_calibration_ci")


class CalibrationCITests(unittest.TestCase):
    def test_prediction_payload_validation_rejects_invalid_probabilities(self):
        payload = {
            "y_true": np.asarray([[0, 1], [1, 0]], dtype=np.float32),
            "y_prob": np.asarray([[0.2, 1.2], [0.7, 0.1]], dtype=np.float32),
            "class_names": np.asarray(["A", "B"]),
        }
        with self.assertRaises(ValueError):
            calibration_ci.validate_prediction_payload(payload, Path("bad.npz"))

    def test_freeze_manifest_validation_checks_checksum_shape_and_classes(self):
        with tempfile.TemporaryDirectory(dir=calibration_ci.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            pred = root / "predictions.npz"
            y_true = np.asarray([[0, 1], [1, 0], [1, 1]], dtype=np.float32)
            y_prob = np.asarray([[0.1, 0.8], [0.9, 0.2], [0.6, 0.7]], dtype=np.float32)
            np.savez_compressed(
                pred,
                y_true=y_true,
                y_prob=y_prob,
                class_names=np.asarray(["A", "B"]),
            )
            relative = pred.resolve().relative_to(calibration_ci.PROJECT_ROOT.resolve()).as_posix()
            freeze = root / "freeze.json"
            freeze.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "manuscript_ready": True,
                        "validated_records": 3,
                        "n_classes": 2,
                        "class_names": ["A", "B"],
                        "artifacts": [
                            {
                                "path": relative,
                                "size_bytes": pred.stat().st_size,
                                "sha256": sha256_file(pred),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = calibration_ci.load_prediction_payload(pred)
            y_true_loaded, _, class_names = calibration_ci.validate_prediction_payload(payload, pred)
            self.assertEqual(
                calibration_ci.validate_freeze_manifest(
                    freeze_path=freeze,
                    pred_path=pred,
                    y_true=y_true_loaded,
                    class_names=class_names,
                ),
                sha256_file(freeze),
            )

    def test_calibration_helpers_report_micro_and_per_class_outputs(self):
        y_true = np.asarray([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.float32)
        y_prob = np.asarray([[0.1, 0.8], [0.9, 0.2], [0.6, 0.7], [0.3, 0.1]], dtype=np.float32)
        micro = calibration_ci.calibration_micro_summary(y_true, y_prob, n_bins=5)
        rows = calibration_ci.per_class_calibration_rows(
            y_true,
            y_prob,
            ["A", "B"],
            n_bins=5,
        )
        self.assertEqual(micro["n_label_record_pairs"], 8)
        self.assertIn("ece_micro", micro)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["evaluated"] for row in rows))


if __name__ == "__main__":
    unittest.main()
