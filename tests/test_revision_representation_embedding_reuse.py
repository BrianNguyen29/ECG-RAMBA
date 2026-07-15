import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np


extract = importlib.import_module("scripts.revision.22_extract_representations")


class RepresentationEmbeddingReuseTests(unittest.TestCase):
    def _fixture(self, root: Path):
        n_records = 10
        y_true = np.zeros((n_records, len(extract.CLASSES)), dtype=np.float32)
        y_true[::2, 0] = 1.0
        fold_id = np.asarray([1, 2, 3, 4, 5] * 2, dtype=np.int16)
        record_id = np.arange(n_records, dtype=np.int64)
        checkpoint_contracts = {
            fold: {"fold": fold, "sha256": f"checkpoint-{fold}"}
            for fold in range(1, 6)
        }
        old_oof = {
            "path": root / "oof.npz",
            "sha256": "old-oof-sha",
            "freeze_manifest": root / "freeze.json",
            "freeze_manifest_sha256": "old-freeze-sha",
            "y_true": y_true,
            "record_id": record_id,
            "fold_id": fold_id,
            "class_names": np.asarray(extract.CLASSES).astype(str),
        }
        current_oof = dict(old_oof)
        current_oof.update(
            {
                "sha256": "current-oof-sha",
                "freeze_manifest_sha256": "current-freeze-sha",
            }
        )
        payload = {
            "checkpoint_kind": "final_ema",
            "dataset_record_order_fingerprint": "record-order",
            "fold_summaries": [
                {"fold": fold, "checkpoint_sha256": f"checkpoint-{fold}"}
                for fold in range(1, 6)
            ],
        }
        embeddings = {
            key: np.full((n_records, 3), index, dtype=np.float32)
            for index, key in enumerate(extract.EMBEDDING_KEYS, start=1)
        }
        return old_oof, current_oof, checkpoint_contracts, payload, embeddings

    def test_semantically_equivalent_oof_repack_refreshes_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "embedding.npz"
            old_oof, current_oof, checkpoint_contracts, payload, embeddings = self._fixture(root)
            extract.write_final_embedding_npz(
                path=path,
                oof=old_oof,
                embeddings=embeddings,
                fold_id=old_oof["fold_id"],
                slice_count=np.ones(len(old_oof["record_id"]), dtype=np.int16),
                payload=payload,
            )

            audit = extract.inspect_final_embedding_reuse(
                path, current_oof, "final_ema", checkpoint_contracts
            )
            self.assertTrue(audit["reusable"])
            self.assertTrue(audit["semantic_contract_match"])
            self.assertTrue(audit["checkpoint_contract_match"])
            self.assertFalse(audit["exact_source_contract"])

            attestation = extract.refresh_final_embedding_contract(
                path=path,
                oof=current_oof,
                checkpoint_kind="final_ema",
                reuse_audit=audit,
            )
            self.assertEqual(attestation["status"], "verified_semantic_repack")

            refreshed = extract.inspect_final_embedding_reuse(
                path, current_oof, "final_ema", checkpoint_contracts
            )
            self.assertTrue(refreshed["reusable"])
            self.assertTrue(refreshed["exact_source_contract"])
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(str(data["oof_predictions_sha256"].item()), "current-oof-sha")
                self.assertEqual(str(data["freeze_manifest_sha256"].item()), "current-freeze-sha")

    def test_checkpoint_contract_mismatch_blocks_semantic_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "embedding.npz"
            old_oof, current_oof, checkpoint_contracts, payload, embeddings = self._fixture(root)
            extract.write_final_embedding_npz(
                path=path,
                oof=old_oof,
                embeddings=embeddings,
                fold_id=old_oof["fold_id"],
                slice_count=np.ones(len(old_oof["record_id"]), dtype=np.int16),
                payload=payload,
            )
            checkpoint_contracts[3]["sha256"] = "different-checkpoint"

            audit = extract.inspect_final_embedding_reuse(
                path, current_oof, "final_ema", checkpoint_contracts
            )
            self.assertFalse(audit["reusable"])
            self.assertIn("checkpoint_sha_contract_mismatch_or_incomplete", audit["issues"])


if __name__ == "__main__":
    unittest.main()
