import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.revision.common import (
    AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    CHAPMAN_GROUP_REFERENCE,
    CHAPMAN_GROUP_SEMANTICS,
    sha256_file,
)


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
            sidecar = root / "test_group_sidecar.npz"
            sidecar.write_bytes(b"authenticated-sidecar")
            sidecar_relative = sidecar.resolve().relative_to(calibration_ci.PROJECT_ROOT.resolve()).as_posix()
            freeze = root / "freeze.json"
            freeze.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "manuscript_ready": True,
                        "validated_records": 3,
                        "n_classes": 2,
                        "class_names": ["A", "B"],
                        "group_contract": {
                            "status": "verified",
                            "group_semantics": CHAPMAN_GROUP_SEMANTICS,
                            "group_semantics_reference": CHAPMAN_GROUP_REFERENCE,
                            "bootstrap_unit": AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
                            "one_record_per_group": True,
                            "n_records": 3,
                            "n_groups": 3,
                            "sidecar": {
                                "path": sidecar_relative,
                                "sha256": sha256_file(sidecar),
                            },
                        },
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
            contract = calibration_ci.validate_freeze_manifest(
                freeze_path=freeze,
                pred_path=pred,
                y_true=y_true_loaded,
                class_names=class_names,
            )
            self.assertEqual(contract["freeze_manifest_sha256"], sha256_file(freeze))
            self.assertEqual(
                contract["bootstrap"]["unit"],
                AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
            )
            sidecar.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "sidecar_sha256"):
                calibration_ci.validate_freeze_manifest(
                    freeze_path=freeze,
                    pred_path=pred,
                    y_true=y_true_loaded,
                    class_names=class_names,
                )

    def test_freeze_manifest_rejects_unbound_group_contract(self):
        with tempfile.TemporaryDirectory(dir=calibration_ci.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            pred = root / "predictions.npz"
            y_true = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
            np.savez_compressed(
                pred,
                y_true=y_true,
                y_prob=np.asarray([[0.2, 0.8], [0.7, 0.1]], dtype=np.float32),
                class_names=np.asarray(["A", "B"]),
            )
            relative = pred.resolve().relative_to(calibration_ci.PROJECT_ROOT.resolve()).as_posix()
            freeze = root / "freeze.json"
            freeze.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "manuscript_ready": True,
                        "validated_records": 2,
                        "n_classes": 2,
                        "class_names": ["A", "B"],
                        "artifacts": [
                            {"path": relative, "sha256": sha256_file(pred)}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "authenticated patient-record"):
                calibration_ci.validate_freeze_manifest(
                    freeze_path=freeze,
                    pred_path=pred,
                    y_true=y_true,
                    class_names=["A", "B"],
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

    def test_main_writes_metrics_from_loaded_dict_payload(self):
        experimental_root = calibration_ci.REVISION_DIR / "experimental"
        experimental_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=experimental_root) as tmp:
            root = Path(tmp)
            pred = root / "predictions.npz"
            out = root / "calibration.json"
            y_true = np.asarray(
                [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1]],
                dtype=np.float32,
            )
            y_prob = np.asarray(
                [[0.1, 0.8], [0.9, 0.2], [0.6, 0.7], [0.3, 0.1], [0.7, 0.4], [0.2, 0.9]],
                dtype=np.float32,
            )
            np.savez_compressed(
                pred,
                y_true=y_true,
                y_prob=y_prob,
                class_names=np.asarray(["A", "B"]),
                dataset=np.asarray("unit_test"),
                protocol=np.asarray("unit_protocol"),
            )
            with patch(
                "sys.argv",
                [
                    "04_calibration_ci.py",
                    "--predictions",
                    str(pred),
                    "--out",
                    str(out),
                    "--n-boot",
                    "3",
                    "--n-bins",
                    "3",
                    "--allow-unauthenticated-exploratory",
                ],
            ):
                calibration_ci.main()
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"], "unit_test")
            self.assertEqual(payload["protocol"], "unit_protocol")
            self.assertFalse(payload["manuscript_ready"])
            self.assertEqual(payload["evidence_status"], "experimental_unauthenticated")
            self.assertIn("calibration_micro", payload)
            self.assertIn("ece_macro", payload["bootstrap_ci"])

    def test_main_rejects_unauthenticated_canonical_output_by_default(self):
        with tempfile.TemporaryDirectory(dir=calibration_ci.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            pred = root / "predictions.npz"
            out = root / "calibration.json"
            np.savez_compressed(
                pred,
                y_true=np.asarray([[0], [1]], dtype=np.float32),
                y_prob=np.asarray([[0.2], [0.8]], dtype=np.float32),
                class_names=np.asarray(["A"]),
            )
            with patch(
                "sys.argv",
                ["04_calibration_ci.py", "--predictions", str(pred), "--out", str(out), "--n-boot", "2"],
            ):
                with self.assertRaisesRegex(RuntimeError, "Canonical calibration requires"):
                    calibration_ci.main()

    def test_bootstrap_ci_requires_exact_finite_replicates(self):
        valid = {"ece_macro": {"mean": 0.1, "lo": 0.05, "hi": 0.2, "n_boot_valid": 10}}
        calibration_ci.validate_bootstrap_ci_payload(valid, 10)
        invalid_count = {"ece_macro": {**valid["ece_macro"], "n_boot_valid": 9}}
        with self.assertRaisesRegex(RuntimeError, "exactly 10"):
            calibration_ci.validate_bootstrap_ci_payload(invalid_count, 10)
        invalid_finite = {"ece_macro": {**valid["ece_macro"], "hi": float("nan")}}
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            calibration_ci.validate_bootstrap_ci_payload(invalid_finite, 10)


if __name__ == "__main__":
    unittest.main()
