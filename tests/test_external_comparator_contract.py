import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


paired = importlib.import_module("scripts.revision.32_paired_external_comparators")
fewshot = importlib.import_module("scripts.revision.35_true_fewshot_head_adaptation")


class ExternalComparatorContractTests(unittest.TestCase):
    def _write_prediction(self, path: Path, probabilities: np.ndarray) -> None:
        y_true = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        np.savez_compressed(
            path,
            y_true=y_true,
            y_prob=np.asarray(probabilities, dtype=np.float32),
            record_id=np.asarray(["r1", "r2", "r3"]),
            group_id=np.asarray(["g1", "g2", "g3"]),
            split_id=np.asarray(["test", "test", "test"]),
            class_names=np.asarray(["a", "b"]),
            dataset=np.asarray("ptbxl"),
            protocol=np.asarray("test"),
            task_scope=np.asarray("record_level_mapped_external_task"),
            group_unit=np.asarray("patient"),
            adaptation_labels_used=np.asarray(0, dtype=np.int16),
        )

    def test_paired_external_loader_rejects_out_of_range_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            self._write_prediction(
                path,
                np.asarray([[1.1, 0.2], [0.3, 0.8], [0.7, 0.9]], dtype=np.float32),
            )
            with self.assertRaises(ValueError):
                paired.load_predictions(path, "ptbxl")

    def test_embedding_requires_current_manifest_and_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_path = root / "prediction.npz"
            self._write_prediction(
                prediction_path,
                np.asarray([[0.8, 0.2], [0.3, 0.8], [0.7, 0.9]], dtype=np.float32),
            )
            prediction = fewshot.load_prediction(prediction_path, "ptbxl")
            embedding_path = root / "embedding.npz"
            embeddings = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
            np.savez_compressed(
                embedding_path,
                fold_embeddings=embeddings,
                y_true=prediction["y_true"],
                record_id=prediction["record_id"],
                group_id=prediction["group_id"],
                split_id=prediction["split_id"],
                class_names=prediction["class_names"],
                model=np.asarray("full"),
                source_prediction_sha256=np.asarray(prediction["sha256"]),
                input_fingerprint=np.asarray("f" * 64),
                protocol_version=np.asarray(2, dtype=np.int16),
                representation=np.asarray("mean_of_preclassifier_slice_embeddings_per_fold"),
            )
            canonical = {"oof_sha256": "a" * 64, "freeze_sha256": "b" * 64}
            manifest_path = root / "embedding_manifest.json"
            extractor = (
                fewshot.PROJECT_ROOT
                / "scripts"
                / "revision"
                / "34_extract_external_representations.py"
            )
            manifest = {
                "status": "complete",
                "protocol": "frozen_encoder_external_record_representation_v2_source_bound",
                "runner_sha256": fewshot.sha256_file(extractor),
                "canonical_contract": canonical,
                "input_fingerprint": "f" * 64,
                "representation": "mean_of_preclassifier_slice_embeddings_per_fold",
                "source_prediction": {"sha256": prediction["sha256"]},
                "output": {"sha256": fewshot.sha256_file(embedding_path)},
                "checkpoints": [
                    {"fold": fold, "sha256": str(fold) * 64}
                    for fold in range(1, 6)
                ],
                "checkpoint_source_contract": {"status": "authenticated"},
                "source_provenance": {"archive": {"sha256": "c" * 64}},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = fewshot.load_embeddings(
                embedding_path,
                manifest_path,
                prediction,
                "full",
                canonical,
            )
            np.testing.assert_array_equal(loaded["embedding"], embeddings)

            manifest["canonical_contract"] = {"oof_sha256": "stale"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                fewshot.load_embeddings(
                    embedding_path,
                    manifest_path,
                    prediction,
                    "full",
                    canonical,
                )

            manifest["canonical_contract"] = canonical
            manifest["source_provenance"] = {}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                fewshot.load_embeddings(
                    embedding_path,
                    manifest_path,
                    prediction,
                    "full",
                    canonical,
                )


if __name__ == "__main__":
    unittest.main()
