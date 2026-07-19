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

    def test_semantically_equivalent_but_sha_changed_oof_requires_regeneration(self):
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
            self.assertFalse(audit["reusable"])
            self.assertTrue(audit["semantic_contract_match"])
            self.assertTrue(audit["checkpoint_contract_match"])
            self.assertFalse(audit["exact_source_contract"])
            self.assertIn("canonical_source_sha_mismatch", audit["issues"])

            with self.assertRaisesRegex(RuntimeError, "cannot be refreshed"):
                extract.refresh_final_embedding_contract(
                    path=path,
                    oof=current_oof,
                    checkpoint_kind="final_ema",
                    reuse_audit=audit,
                )

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

    def test_fold_assignment_mismatch_blocks_reuse_with_diagnostics(self):
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
            current_oof["fold_id"] = np.roll(current_oof["fold_id"], 1)

            audit = extract.inspect_final_embedding_reuse(
                path, current_oof, "final_ema", checkpoint_contracts
            )
            self.assertFalse(audit["reusable"])
            self.assertFalse(audit["semantic_field_match"]["fold_id"])
            self.assertTrue(audit["semantic_field_match"]["y_true"])
            self.assertIn("oof_fold_assignment_mismatch", audit["issues"])
            self.assertEqual(audit["fold_assignment_mismatch_count"], len(old_oof["fold_id"]))

    def test_folds_are_derived_from_frozen_oof_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_oof, _, _, _, _ = self._fixture(Path(tmp))
            folds = extract.folds_from_frozen_oof(old_oof)
            self.assertEqual([int(row["fold_num"]) for row in folds], [1, 2, 3, 4, 5])
            for row in folds:
                fold_num = int(row["fold_num"])
                expected_val = np.flatnonzero(old_oof["fold_id"] == fold_num)
                self.assertTrue(np.array_equal(row["va_idx"], expected_val))
                self.assertTrue(np.all(old_oof["fold_id"][row["tr_idx"]] != fold_num))
                self.assertEqual(len(row["tr_idx"]) + len(row["va_idx"]), len(old_oof["fold_id"]))

    def test_checkpoint_folds_must_match_frozen_oof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_oof, _, checkpoint_contracts, _, _ = self._fixture(root)
            folds = extract.folds_from_frozen_oof(old_oof)
            extract.joblib.dump(
                [{"tr_idx": row["tr_idx"], "va_idx": row["va_idx"]} for row in folds],
                root / "folds.pkl",
            )
            for fold, contract in checkpoint_contracts.items():
                contract["path"] = str(root / f"fold{fold}_final_ema.pt")

            split_contract = extract.validate_checkpoint_fold_contract(
                old_oof, checkpoint_contracts
            )
            self.assertEqual(
                split_contract["source"],
                "frozen_oof_fold_id_verified_against_checkpoint_folds",
            )
            mismatched = dict(old_oof)
            mismatched["fold_id"] = np.roll(old_oof["fold_id"], 1)
            with self.assertRaisesRegex(
                RuntimeError, "membership differs|mismatched_records"
            ):
                extract.validate_checkpoint_fold_contract(mismatched, checkpoint_contracts)


if __name__ == "__main__":
    unittest.main()
