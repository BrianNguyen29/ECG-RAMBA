import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np

from configs.config import CLASSES
from scripts.revision.common import aggregate_record_probabilities, sha256_file


freeze_oof = importlib.import_module("scripts.revision.06_freeze_oof")


class FreezeOOFContractTests(unittest.TestCase):
    def _fixture(
        self,
        *,
        y_true_override=None,
        y_prob_override=None,
        record_id_override=None,
        record_fold_override=None,
        slice_record_override=None,
        slice_fold_override=None,
        checkpoint_split_override=None,
        sidecar_record_sha_override=None,
        sidecar_groups_override=None,
        sidecar_semantics_override=None,
        sidecar_reference_override=None,
        sidecar_counts_override=None,
        sidecar_one_record_override=None,
        sidecar_archive_sha_override=None,
    ):
        temporary = tempfile.TemporaryDirectory(dir=freeze_oof.PROJECT_ROOT)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        pred_dir = root / "predictions"
        metric_dir = root / "metrics"
        table_dir = root / "tables"
        manifest_dir = root / "manifests"
        log_dir = root / "logs"
        for path in (pred_dir, metric_dir, table_dir, manifest_dir, log_dir):
            path.mkdir()

        n_records = 4
        n_classes = len(CLASSES)
        rng = np.random.default_rng(20260718)
        canonical_record_fold = np.asarray([1, 2, 1, 2], dtype=np.int16)
        record_fold = np.asarray(
            canonical_record_fold if record_fold_override is None else record_fold_override,
            dtype=np.int16,
        )
        slice_prob = rng.uniform(0.05, 0.95, size=(8, n_classes)).astype(np.float32)
        slice_record_id = np.asarray(
            np.repeat(np.arange(n_records), 2)
            if slice_record_override is None
            else slice_record_override,
        )
        slice_fold_id = np.asarray(
            np.repeat(record_fold, 2) if slice_fold_override is None else slice_fold_override,
            dtype=np.int16,
        )
        y_prob, valid, counts = aggregate_record_probabilities(
            slice_prob,
            np.repeat(np.arange(n_records), 2),
            n_records,
            q=3.0,
        )
        if y_prob_override is not None:
            y_prob = np.asarray(y_prob_override, dtype=np.float32)
        y_true = (rng.uniform(size=(n_records, n_classes)) > 0.8).astype(np.float32)
        if y_true_override is not None:
            y_true = np.asarray(y_true_override)
        record_id = np.asarray(
            np.arange(n_records) if record_id_override is None else record_id_override,
        )
        dataset_fingerprint = "chapman-record-order-v1"

        folds = [
            {
                "tr_idx": np.asarray([1, 3], dtype=np.int64),
                "va_idx": np.asarray([0, 2], dtype=np.int64),
            },
            {
                "tr_idx": np.asarray([0, 2], dtype=np.int64),
                "va_idx": np.asarray([1, 3], dtype=np.int64),
            },
        ]
        folds_file = root / "folds.pkl"
        joblib.dump(folds, folds_file)

        checkpoint_rows = []
        for fold_num, split in enumerate(folds, start=1):
            split_metadata = {
                "train_count": len(split["tr_idx"]),
                "val_count": len(split["va_idx"]),
                "train_index_hash": freeze_oof.index_fingerprint(split["tr_idx"]),
                "val_index_hash": freeze_oof.index_fingerprint(split["va_idx"]),
            }
            if checkpoint_split_override and fold_num in checkpoint_split_override:
                split_metadata = checkpoint_split_override[fold_num]
            checkpoint_rows.append(
                {
                    "fold": fold_num,
                    "path": f"fold{fold_num}_final_ema.pt",
                    "sha256": f"sha{fold_num}",
                    "size_bytes": 1,
                    "dataset_record_order_fingerprint": dataset_fingerprint,
                    "split": split_metadata,
                }
            )

        record_file = pred_dir / "oof_final_ema_predictions.npz"
        slice_file = pred_dir / "oof_final_ema_slice_predictions.npz"
        np.savez_compressed(
            record_file,
            y_true=y_true,
            y_prob=y_prob,
            record_id=record_id,
            class_names=np.asarray(CLASSES),
            fold_id=record_fold,
            valid_record_mask=valid,
            slice_count=counts,
            aggregation_q=np.asarray(3.0, dtype=np.float32),
            aggregation_implementation=np.asarray(freeze_oof.POWER_MEAN_IMPLEMENTATION),
            cache_schema_version=np.asarray(freeze_oof.CACHE_SCHEMA_VERSION, dtype=np.int16),
            checkpoint_kind=np.asarray("final_ema"),
            source_config_hash=np.asarray("source-config"),
            evaluation_config_hash=np.asarray(freeze_oof.EVALUATION_CONFIG_HASH),
            dataset_record_order_fingerprint=np.asarray(dataset_fingerprint),
        )
        np.savez_compressed(
            slice_file,
            slice_prob=slice_prob,
            record_id=slice_record_id,
            fold_id=slice_fold_id,
            class_names=np.asarray(CLASSES),
        )

        summary_file = metric_dir / "oof_final_ema_prediction_summary.json"
        class_table = table_dir / "oof_final_ema_class_summary.csv"
        run_manifest = manifest_dir / "oof_final_ema_prediction_run_manifest.json"
        summary_file.write_text(
            json.dumps(
                {
                    "dataset": "chapman_oof",
                    "n_records": n_records,
                    "n_valid_predictions": n_records,
                }
            ),
            encoding="utf-8",
        )
        class_table.write_text(
            "class_name\n" + "\n".join(CLASSES) + "\n",
            encoding="utf-8",
        )
        run_manifest.write_text(
            json.dumps(
                {
                    "dataset": "chapman_oof",
                    "dataset_record_order_fingerprint": dataset_fingerprint,
                    "inputs": {"checkpoints": checkpoint_rows},
                    "outputs": {
                        "prediction_file": {
                            "path": str(record_file),
                            "sha256": sha256_file(record_file),
                            "size_bytes": record_file.stat().st_size,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (log_dir / "oof_generate_predictions.log").write_text(
            "complete\n",
            encoding="utf-8",
        )

        group_sidecar = root / "chapman_groups.npz"
        source_archive = root / "WFDB-ChapmanShaoxing.zip"
        source_archive.write_bytes(b"reviewed-chapman-source-archive")
        np.savez_compressed(
            group_sidecar,
            record_id=np.arange(n_records, dtype=np.int64),
            group_id=np.asarray(
                ["subject-0", "subject-1", "subject-2", "subject-3"]
                if sidecar_groups_override is None
                else sidecar_groups_override
            ),
            group_unit=np.asarray("Chapman-Shaoxing/Ningbo source patient-record"),
            group_semantics=np.asarray(
                freeze_oof.SUBJECT_GROUP_SEMANTICS
                if sidecar_semantics_override is None
                else sidecar_semantics_override
            ),
            group_semantics_reference=np.asarray(
                freeze_oof.SUBJECT_GROUP_REFERENCE
                if sidecar_reference_override is None
                else sidecar_reference_override
            ),
            source_patient_record_counts_json=np.asarray(
                json.dumps(
                    freeze_oof.SUBJECT_GROUP_REFERENCE_COUNTS
                    if sidecar_counts_override is None
                    else sidecar_counts_override,
                    sort_keys=True,
                )
            ),
            one_record_per_group=np.asarray(
                True if sidecar_one_record_override is None else sidecar_one_record_override,
                dtype=np.bool_,
            ),
            dataset_record_order_fingerprint=np.asarray(dataset_fingerprint),
            record_file_sha256=np.asarray(
                sha256_file(record_file)
                if sidecar_record_sha_override is None
                else sidecar_record_sha_override
            ),
            source_archive_sha256=np.asarray(
                sha256_file(source_archive)
                if sidecar_archive_sha_override is None
                else sidecar_archive_sha_override
            ),
        )

        args = SimpleNamespace(
            record_file=record_file,
            slice_file=slice_file,
            summary_file=summary_file,
            class_table=class_table,
            run_manifest=run_manifest,
            freeze_manifest=manifest_dir / "oof_final_ema_freeze_manifest.json",
            expected_records=n_records,
            expected_folds=2,
            q=3.0,
            expected_checkpoint_kind="final_ema",
            check_only=True,
            allow_missing_log=False,
            manuscript_ready_strict=True,
            folds_file=folds_file,
            group_sidecar=group_sidecar,
            group_sidecar_sha256=sha256_file(group_sidecar),
            source_archive=source_archive,
        )
        return args, log_dir, checkpoint_rows, y_true, y_prob

    def _validate(self, args, log_dir, checkpoint_rows):
        with patch.object(freeze_oof, "LOG_DIR", log_dir), patch.object(
            freeze_oof,
            "current_checkpoint_rows",
            return_value=checkpoint_rows,
        ), patch.dict(freeze_oof.PATHS, {"zip_path": str(args.source_archive)}):
            return freeze_oof.validate_oof(args)

    def test_strict_contract_accepts_authenticated_membership_and_groups(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture()

        payload = self._validate(args, log_dir, checkpoint_rows)

        self.assertTrue(payload["manuscript_ready"])
        self.assertTrue(payload["strict_manuscript_contract"])
        self.assertEqual(payload["membership_contract"]["status"], "verified")
        self.assertTrue(payload["group_contract"]["one_record_per_group"])
        self.assertEqual(
            payload["group_contract"]["group_semantics"],
            freeze_oof.SUBJECT_GROUP_SEMANTICS,
        )
        self.assertEqual(
            payload["group_contract"]["source_patient_record_counts"],
            freeze_oof.SUBJECT_GROUP_REFERENCE_COUNTS,
        )

    def test_metadata_refresh_authenticates_unchanged_oof_without_generation_log(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture()
        initial = self._validate(args, log_dir, checkpoint_rows)
        args.freeze_manifest.write_text(json.dumps(initial), encoding="utf-8")
        args.metadata_refresh_from_existing_oof = True
        (log_dir / "oof_generate_predictions.log").unlink()

        refreshed = self._validate(args, log_dir, checkpoint_rows)

        provenance = refreshed["generation_provenance"]
        self.assertEqual(provenance["status"], "verified_metadata_only_refresh")
        self.assertFalse(provenance["prediction_values_changed"])
        self.assertEqual(provenance["record_file_sha256"], sha256_file(args.record_file))

    def test_metadata_refresh_rejects_prior_freeze_with_different_oof_sha(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture()
        initial = self._validate(args, log_dir, checkpoint_rows)
        for row in initial["artifacts"]:
            if Path(str(row.get("path", ""))).name == args.record_file.name:
                row["sha256"] = "0" * 64
        args.freeze_manifest.write_text(json.dumps(initial), encoding="utf-8")
        args.metadata_refresh_from_existing_oof = True
        (log_dir / "oof_generate_predictions.log").unlink()

        with self.assertRaisesRegex(RuntimeError, "Prior freeze manifest does not authenticate"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_metadata_refresh_rejects_run_manifest_with_different_oof_sha(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture()
        initial = self._validate(args, log_dir, checkpoint_rows)
        run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        run_manifest["outputs"]["prediction_file"]["sha256"] = "0" * 64
        args.run_manifest.write_text(json.dumps(run_manifest), encoding="utf-8")
        for row in initial["artifacts"]:
            if Path(str(row.get("path", ""))).name == args.run_manifest.name:
                row["sha256"] = sha256_file(args.run_manifest)
                row["size_bytes"] = args.run_manifest.stat().st_size
        args.freeze_manifest.write_text(json.dumps(initial), encoding="utf-8")
        args.metadata_refresh_from_existing_oof = True
        (log_dir / "oof_generate_predictions.log").unlink()

        with self.assertRaisesRegex(RuntimeError, "run manifest does not authenticate"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_metadata_refresh_rejects_run_manifest_not_bound_by_prior_freeze(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture()
        initial = self._validate(args, log_dir, checkpoint_rows)
        args.freeze_manifest.write_text(json.dumps(initial), encoding="utf-8")
        run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        run_manifest["created_utc"] = "mutated-after-freeze"
        args.run_manifest.write_text(json.dumps(run_manifest), encoding="utf-8")
        args.metadata_refresh_from_existing_oof = True
        (log_dir / "oof_generate_predictions.log").unlink()

        with self.assertRaisesRegex(RuntimeError, "does not authenticate the current OOF prediction run manifest"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_non_strict_freeze_is_not_manuscript_ready(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture()
        args.manuscript_ready_strict = False
        args.group_sidecar = None
        args.group_sidecar_sha256 = None

        payload = self._validate(args, log_dir, checkpoint_rows)

        self.assertFalse(payload["manuscript_ready"])
        self.assertEqual(payload["membership_contract"]["status"], "not_requested")
        self.assertEqual(payload["group_contract"]["status"], "not_requested")
        self.assertEqual(
            payload["claim_boundary"],
            "exploratory_frozen_oof_not_for_manuscript_claims",
        )

    def test_checkpoint_membership_is_read_from_checkpoint_payload(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is required to read manuscript checkpoints")
        temporary = tempfile.TemporaryDirectory(dir=freeze_oof.PROJECT_ROOT)
        self.addCleanup(temporary.cleanup)
        checkpoint = Path(temporary.name) / "fold1_final_ema.pt"
        torch.save(
            {
                "fold": 1,
                "split": {
                    "train_count": 2,
                    "val_count": 2,
                    "train_index_hash": "train-hash",
                    "val_index_hash": "val-hash",
                },
                "dataset_record_order_fingerprint": "records-v1",
            },
            checkpoint,
        )

        fold, split, fingerprint, source = freeze_oof.checkpoint_membership_metadata(
            {"fold": 1, "path": str(checkpoint), "sha256": sha256_file(checkpoint)}
        )

        self.assertEqual(fold, 1)
        self.assertEqual(split["train_index_hash"], "train-hash")
        self.assertEqual(fingerprint, "records-v1")
        self.assertEqual(source, "checkpoint_payload")

    def test_artifact_info_supports_canonical_files_outside_repo(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "folds.pkl"
            artifact.write_bytes(b"fold-contract")

            info = freeze_oof.artifact_info(artifact)

            self.assertEqual(info["path"], artifact.resolve().as_posix())
            self.assertEqual(info["sha256"], sha256_file(artifact))

    def test_rejects_checkpoint_fold_membership_swap(self):
        fold_one = np.asarray([1, 3], dtype=np.int64)
        fold_two = np.asarray([0, 2], dtype=np.int64)
        swapped = {
            1: {
                "train_count": 2,
                "val_count": 2,
                "train_index_hash": freeze_oof.index_fingerprint(fold_two),
                "val_index_hash": freeze_oof.index_fingerprint(fold_one),
            }
        }
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            checkpoint_split_override=swapped
        )

        with self.assertRaisesRegex(RuntimeError, "split metadata differs from folds.pkl"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_source_archive_sha_mismatch(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            sidecar_archive_sha_override="0" * 64
        )

        with self.assertRaisesRegex(RuntimeError, "source_archive_sha256 differs"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_record_fold_assignment_swap(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            record_fold_override=[2, 1, 2, 1]
        )

        with self.assertRaisesRegex(RuntimeError, "record fold assignment differs"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_nonbinary_labels(self):
        base = np.zeros((4, len(CLASSES)), dtype=np.float32)
        base[0, 0] = 0.5
        args, log_dir, checkpoint_rows, _, _ = self._fixture(y_true_override=base)

        with self.assertRaisesRegex(ValueError, "binary values"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_nonfinite_record_probabilities(self):
        args, log_dir, checkpoint_rows, _, y_prob = self._fixture()
        invalid = y_prob.copy()
        invalid[0, 0] = np.nan
        with np.load(args.record_file, allow_pickle=False) as existing:
            payload = {key: np.asarray(existing[key]) for key in existing.files}
        payload["y_prob"] = invalid
        np.savez_compressed(args.record_file, **payload)

        with self.assertRaisesRegex(ValueError, "record probabilities"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_nonfinite_slice_probabilities(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture()
        with np.load(args.slice_file, allow_pickle=False) as existing:
            payload = {key: np.asarray(existing[key]) for key in existing.files}
        payload["slice_prob"] = payload["slice_prob"].copy()
        payload["slice_prob"][0, 0] = np.inf
        np.savez_compressed(args.slice_file, **payload)

        with self.assertRaisesRegex(ValueError, "Slice probabilities"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_duplicate_record_ids(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            record_id_override=[0, 1, 1, 3]
        )

        with self.assertRaisesRegex(ValueError, "record_id values must be unique"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_slice_without_record_parent(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            slice_record_override=[0, 0, 1, 1, 2, 2, 3, 99]
        )

        with self.assertRaisesRegex(ValueError, "without a record-level parent"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_slice_fold_mismatch(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            slice_fold_override=[2, 1, 2, 2, 1, 1, 2, 2]
        )

        with self.assertRaisesRegex(ValueError, "parent record fold_id"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_stale_group_sidecar_record_binding(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            sidecar_record_sha_override="0" * 64
        )

        with self.assertRaisesRegex(RuntimeError, "record_file_sha256"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_duplicate_groups(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            sidecar_groups_override=["subject-0", "subject-0", "subject-2", "subject-3"]
        )

        with self.assertRaisesRegex(RuntimeError, "one-record-per-group"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_unreviewed_group_semantics(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            sidecar_semantics_override="record_ids_are_probably_subjects"
        )

        with self.assertRaisesRegex(RuntimeError, "patient-record semantics"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_unreviewed_group_reference(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            sidecar_reference_override="https://example.invalid/unreviewed"
        )

        with self.assertRaisesRegex(RuntimeError, "reviewed PhysioNet source"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_rejects_source_patient_record_count_drift(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture(
            sidecar_counts_override={
                "chapman_shaoxing": {"patients": 10247, "recordings": 10248},
                "ningbo": {"patients": 34905, "recordings": 34905},
            }
        )

        with self.assertRaisesRegex(RuntimeError, "reviewed corpus contract"):
            self._validate(args, log_dir, checkpoint_rows)

    def test_strict_mode_blocks_missing_checkpoint_membership_metadata(self):
        args, log_dir, checkpoint_rows, _, _ = self._fixture()
        for row in checkpoint_rows:
            row.pop("split")
        manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        manifest["inputs"]["checkpoints"] = checkpoint_rows
        args.run_manifest.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "Manuscript-ready OOF freeze blocker"):
            self._validate(args, log_dir, checkpoint_rows)


if __name__ == "__main__":
    unittest.main()
