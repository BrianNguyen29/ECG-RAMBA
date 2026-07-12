import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROBUSTNESS = importlib.import_module("scripts.revision.12_robustness_stress")
COMPARATOR_STRESS = importlib.import_module(
    "scripts.revision.23_generate_comparator_stress_predictions"
)


class RobustnessCacheContractTests(unittest.TestCase):
    def test_minirocket_cache_requires_matching_head_and_clean_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.npz"
            y = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            fold_id = np.asarray([1, 2], dtype=np.int16)
            prob = np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
            metadata = {
                "minirocket_heads_manifest": {
                    "params_hash": "params-a",
                    "clean_prediction_sha256": "clean-a",
                }
            }
            np.savez_compressed(
                path,
                y_true=y,
                y_prob=prob,
                fold_id=fold_id,
                protocol=np.asarray(ROBUSTNESS.PROTOCOL),
                stress_name=np.asarray("snr20db"),
                model_label=np.asarray("MiniRocket-only"),
                metadata_json=np.asarray(json.dumps(metadata)),
            )

            accepted = ROBUSTNESS.load_existing_prediction(
                path,
                y=y,
                fold_id=fold_id,
                expected_stress="snr20db",
                expected_model_label="MiniRocket-only",
                expected_contract_hash=None,
                expected_minirocket_params_hash="params-a",
                expected_minirocket_clean_prediction_sha256="clean-a",
            )
            np.testing.assert_allclose(accepted, prob)

            rejected = ROBUSTNESS.load_existing_prediction(
                path,
                y=y,
                fold_id=fold_id,
                expected_stress="snr20db",
                expected_model_label="MiniRocket-only",
                expected_contract_hash=None,
                expected_minirocket_params_hash="params-a",
                expected_minirocket_clean_prediction_sha256="clean-b",
            )
            self.assertIsNone(rejected)

    def test_corrupt_npz_is_rejected_instead_of_crashing_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.npz"
            path.write_bytes(b"not-a-zip")
            y = np.zeros((2, 2), dtype=np.float32)
            fold_id = np.asarray([1, 2], dtype=np.int16)
            result = ROBUSTNESS.load_existing_prediction(
                path,
                y=y,
                fold_id=fold_id,
                expected_stress="snr20db",
                expected_model_label="Full ECG-RAMBA",
                expected_contract_hash="contract",
                expected_checkpoint_sha_by_fold={1: "a", 2: "b"},
            )
            self.assertIsNone(result)

    def test_learned_comparator_cache_requires_record_class_and_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resnet.npz"
            y = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            probability = np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
            fold_id = np.asarray([1, 2], dtype=np.int16)
            record_id = np.asarray([0, 1], dtype=np.int64)
            class_names = ["A", "B"]
            hashes = ["fold-1", "fold-2"]
            freeze = {
                "oof_predictions_sha256": "oof",
                "freeze_manifest_sha256": "freeze",
            }
            np.savez_compressed(
                path,
                y_true=y,
                y_prob=probability,
                fold_id=fold_id,
                record_id=record_id,
                class_names=np.asarray(class_names),
                protocol=np.asarray(COMPARATOR_STRESS.PROTOCOL),
                comparator=np.asarray("resnet"),
                stress_test=np.asarray("snr20db"),
                aggregation_implementation=np.asarray(
                    COMPARATOR_STRESS.POWER_MEAN_IMPLEMENTATION
                ),
                power_mean_q=np.asarray(float(COMPARATOR_STRESS.CONFIG["power_mean_q"])),
                oof_predictions_sha256=np.asarray("oof"),
                freeze_manifest_sha256=np.asarray("freeze"),
                checkpoint_sha256=np.asarray(hashes),
            )
            self.assertTrue(
                COMPARATOR_STRESS.validate_existing(
                    path,
                    y,
                    fold_id,
                    record_id,
                    class_names,
                    comparator="resnet",
                    stress="snr20db",
                    freeze_contract=freeze,
                    checkpoint_hashes=hashes,
                )
            )
            self.assertFalse(
                COMPARATOR_STRESS.validate_existing(
                    path,
                    y,
                    fold_id,
                    record_id[::-1],
                    class_names,
                    comparator="resnet",
                    stress="snr20db",
                    freeze_contract=freeze,
                    checkpoint_hashes=hashes,
                )
            )

    def test_stress_inference_trusts_only_checkpoint_set_bound_by_baseline_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_dir = root / "manifests"
            checkpoint_dir = root / "checkpoints"
            manifest_dir.mkdir()
            checkpoint_dir.mkdir()
            rows = []
            paths = []
            for fold in range(1, 6):
                path = checkpoint_dir / f"fold{fold}.pt"
                path.write_bytes(f"checkpoint-{fold}".encode("ascii"))
                paths.append(path)
                rows.append(
                    {
                        "fold": fold,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": COMPARATOR_STRESS.sha256_file(path),
                    }
                )
            (manifest_dir / "resnet1d_cnn_baseline_manifest.json").write_text(
                json.dumps(
                    {
                        "protocol": (
                            "resnet1d_cnn_raw_same_folds_power_mean_v2_q3_threshold_0.5"
                        ),
                        "checkpoint_contract": {
                            "status": "complete",
                            "checkpoints": rows,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(COMPARATOR_STRESS, "MANIFEST_DIR", manifest_dir):
                hashes = COMPARATOR_STRESS.validate_checkpoint_set("resnet", paths)
                self.assertEqual(hashes, [row["sha256"] for row in rows])
                paths[2].write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "size mismatch|SHA mismatch"):
                    COMPARATOR_STRESS.validate_checkpoint_set("resnet", paths)


if __name__ == "__main__":
    unittest.main()
