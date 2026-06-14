import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from configs.config import CLASSES
from scripts.revision import artifact_mirror
from scripts.revision.common import aggregate_record_probabilities, sha256_file


freeze_oof = importlib.import_module("scripts.revision.06_freeze_oof")
pooling = importlib.import_module("scripts.revision.07_pooling_sensitivity")
a0_gate = importlib.import_module("scripts.revision.00_a0_resolution_gate")
generate_predictions = importlib.import_module("scripts.revision.01_generate_predictions")


class RevisionArtifactContractTests(unittest.TestCase):
    def test_freeze_artifact_info_accepts_relative_paths(self):
        with tempfile.TemporaryDirectory(dir=freeze_oof.PROJECT_ROOT) as tmp:
            path = Path(tmp) / "relative_artifact.txt"
            path.write_text("ok", encoding="utf-8")
            relative = path.relative_to(freeze_oof.PROJECT_ROOT)

            info = freeze_oof.artifact_info(relative)

            self.assertEqual(info["path"], relative.as_posix())
            self.assertEqual(info["size_bytes"], path.stat().st_size)
            self.assertEqual(info["sha256"], sha256_file(path))

    def test_oof_artifact_stem_separates_best_and_final_outputs(self):
        self.assertEqual(generate_predictions.oof_artifact_stem("best"), "oof_full")
        self.assertEqual(generate_predictions.oof_artifact_stem("final"), "oof_final")
        self.assertEqual(generate_predictions.oof_artifact_stem("best_ema"), "oof_best_ema")
        self.assertEqual(generate_predictions.oof_artifact_stem("final_ema"), "oof_final_ema")
        self.assertEqual(generate_predictions.oof_artifact_stem("final_raw"), "oof_final_raw")

    def test_checkpoint_path_requires_exact_kind_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "fold1_best.pt").write_bytes(b"legacy")
            with patch.dict(generate_predictions.PATHS, {"model_dir": str(model_dir)}):
                with self.assertRaises(FileNotFoundError):
                    generate_predictions.checkpoint_path(1, "best_ema")
                self.assertEqual(
                    generate_predictions.checkpoint_path(1, "best_ema", allow_fallback=True),
                    model_dir / "fold1_best.pt",
                )

    def test_explicit_checkpoint_kind_requires_explicit_weights_metadata(self):
        generate_predictions.validate_checkpoint_weights_kind("best_ema", "ema", Path("fold1_best_ema.pt"))
        generate_predictions.validate_checkpoint_weights_kind("final_raw", "raw", Path("fold1_final_raw.pt"))
        generate_predictions.validate_checkpoint_weights_kind("best", None, Path("fold1_best.pt"))

        with self.assertRaises(ValueError):
            generate_predictions.validate_checkpoint_weights_kind("best_ema", None, Path("fold1_best_ema.pt"))
        with self.assertRaises(ValueError):
            generate_predictions.validate_checkpoint_weights_kind("best_ema", "raw", Path("fold1_best_ema.pt"))

    def test_fold_cache_without_complete_slice_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold.npz"
            n_classes = len(CLASSES)
            np.savez_compressed(
                path,
                record_id=np.asarray([0, 1], dtype=np.int64),
                y_prob=np.zeros((2, n_classes), dtype=np.float32),
                valid_record_mask=np.asarray([True, True]),
                slice_count=np.asarray([1, 1], dtype=np.int16),
                fold_summary_json=np.asarray(json.dumps({"fold": 1})),
                cache_schema_version=np.asarray(2, dtype=np.int16),
                checkpoint_sha256=np.asarray("checkpoint"),
                aggregation_implementation=np.asarray("power_mean_v2"),
                slice_prob=np.zeros((1, n_classes), dtype=np.float32),
                slice_record_id=np.asarray([0], dtype=np.int64),
                slice_index=np.asarray([0], dtype=np.int16),
                slice_fold_id=np.asarray([1], dtype=np.int16),
            )
            result = generate_predictions.load_fold_prediction_cache(
                path=path,
                fold_num=1,
                va_idx=np.asarray([0, 1], dtype=np.int64),
                n_classes=n_classes,
                oof_probs=np.zeros((2, n_classes), dtype=np.float32),
                fold_id=np.full(2, -1, dtype=np.int16),
                record_slice_count=np.zeros(2, dtype=np.int16),
                save_slice_probs=True,
                slice_probs_all=[],
                slice_record_index_all=[],
                slice_index_all=[],
                slice_fold_id_all=[],
                checkpoint_sha256="checkpoint",
            )
            self.assertIsNone(result)

    def test_legacy_mirror_manifest_is_normalized(self):
        payload = {
            "mirror_root": "/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision",
            "artifacts": [
                {
                    "mirror": (
                        "/content/drive/MyDrive/ECG-Ramba/revision_artifacts/"
                        "reports/revision/predictions/oof_full_predictions.npz"
                    ),
                    "size_bytes": 123,
                    "sha256": "abc",
                }
            ],
        }
        rows = artifact_mirror.normalize_manifest_rows(payload, Path("/different/local/root"))
        self.assertEqual(rows[0]["relative_path"], "predictions/oof_full_predictions.npz")

    def test_mirror_restore_verifies_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            source = revision / "predictions" / "oof_full_predictions.npz"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"verified-oof")
            with patch.object(artifact_mirror, "REVISION_DIR", revision), patch.object(
                artifact_mirror,
                "ensure_revision_dirs",
                return_value=None,
            ):
                artifact_mirror.publish(mirror)
                manifest_path = mirror / "manifests" / "mirror_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertNotIn(
                    "manifests/mirror_manifest.json",
                    {row["relative_path"] for row in manifest["artifacts"]},
                )
                source.unlink()
                artifact_mirror.restore(mirror, replace_mismatched=True)
                self.assertEqual(source.read_bytes(), b"verified-oof")
                (mirror / "predictions" / source.name).write_bytes(b"corrupt")
                with self.assertRaises(RuntimeError):
                    artifact_mirror.restore(mirror, replace_mismatched=True)

    def test_restore_ignores_legacy_self_referential_mirror_manifest_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            artifact = mirror / "metrics" / "summary.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"ok": true}', encoding="utf-8")
            manifest_path = mirror / "manifests" / "mirror_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mirror_root": str(mirror),
                        "artifacts": [
                            {
                                "relative_path": "manifests/mirror_manifest.json",
                                "size_bytes": 1,
                                "sha256": "legacy-stale-self-hash",
                            },
                            {
                                "relative_path": "metrics/summary.json",
                                "size_bytes": artifact.stat().st_size,
                                "sha256": sha256_file(artifact),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(artifact_mirror, "REVISION_DIR", revision), patch.object(
                artifact_mirror,
                "ensure_revision_dirs",
                return_value=None,
            ):
                artifact_mirror.restore(mirror, replace_mismatched=True)
            self.assertEqual(
                (revision / "metrics" / "summary.json").read_text(encoding="utf-8"),
                '{"ok": true}',
            )

    def test_a0_update_preserves_other_task_board_rows(self):
        original = (
            "id,status,notes\n"
            'A0,completed,"old note"\n'
            'A1,pending,"preserve, quoted note"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task_board.csv"
            path.write_text(original, encoding="utf-8")

            a0_gate.update_a0_task_board(
                path,
                "audit_complete_with_deferred_blockers",
                "new note",
            )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                (
                    "id,status,notes\n"
                    "A0,audit_complete_with_deferred_blockers,new note\n"
                    'A1,pending,"preserve, quoted note"\n'
                ),
            )

    def test_max_pooling_groups_by_record(self):
        probs = np.asarray([[0.1, 0.8], [0.9, 0.2], [0.4, 0.5]], dtype=np.float32)
        record_ids = np.asarray([0, 0, 1], dtype=np.int64)
        result, valid, counts = pooling.aggregate_max(probs, record_ids, 2)
        np.testing.assert_allclose(result, [[0.9, 0.8], [0.4, 0.5]])
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_array_equal(counts, [2, 1])

    def test_freeze_validates_reaggregation_and_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory(dir=freeze_oof.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            pred_dir = root / "predictions"
            metric_dir = root / "metrics"
            table_dir = root / "tables"
            manifest_dir = root / "manifests"
            log_dir = root / "logs"
            for path in [pred_dir, metric_dir, table_dir, manifest_dir, log_dir]:
                path.mkdir()

            n_records = 4
            n_classes = len(CLASSES)
            rng = np.random.default_rng(42)
            slice_prob = rng.uniform(0.05, 0.95, size=(8, n_classes)).astype(np.float32)
            slice_record_id = np.repeat(np.arange(n_records), 2)
            slice_fold_id = np.repeat([1, 2, 1, 2], 2).astype(np.int16)
            y_prob, valid, counts = aggregate_record_probabilities(
                slice_prob,
                slice_record_id,
                n_records,
                q=3.0,
            )
            y_true = (rng.uniform(size=(n_records, n_classes)) > 0.8).astype(np.float32)
            dataset_fingerprint = "records123"
            checkpoint_rows = [
                {
                    "fold": 1,
                    "path": "fold1_final_ema.pt",
                    "sha256": "sha1",
                    "size_bytes": 1,
                    "dataset_record_order_fingerprint": dataset_fingerprint,
                },
                {
                    "fold": 2,
                    "path": "fold2_final_ema.pt",
                    "sha256": "sha2",
                    "size_bytes": 1,
                    "dataset_record_order_fingerprint": dataset_fingerprint,
                },
            ]

            record_file = pred_dir / "oof_full_predictions.npz"
            slice_file = pred_dir / "oof_full_slice_predictions.npz"
            np.savez_compressed(
                record_file,
                y_true=y_true,
                y_prob=y_prob,
                record_id=np.arange(n_records),
                class_names=np.asarray(CLASSES),
                fold_id=np.asarray([1, 2, 1, 2], dtype=np.int16),
                valid_record_mask=valid,
                slice_count=counts,
                aggregation_q=np.asarray(3.0, dtype=np.float32),
                aggregation_implementation=np.asarray("power_mean_v2"),
                cache_schema_version=np.asarray(2, dtype=np.int16),
                checkpoint_kind=np.asarray("final_ema"),
                source_config_hash=np.asarray("source"),
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
            summary_file = metric_dir / "oof_full_prediction_summary.json"
            class_table = table_dir / "oof_full_class_summary.csv"
            run_manifest = manifest_dir / "oof_full_prediction_run_manifest.json"
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
                        "dataset_record_order_fingerprint": dataset_fingerprint,
                        "inputs": {"checkpoints": checkpoint_rows},
                    }
                ),
                encoding="utf-8",
            )
            (log_dir / "oof_reaggregate.log").write_text("reaggregated", encoding="utf-8")

            args = SimpleNamespace(
                record_file=record_file,
                slice_file=slice_file,
                summary_file=summary_file,
                class_table=class_table,
                run_manifest=run_manifest,
                freeze_manifest=manifest_dir / "oof_freeze_manifest.json",
                expected_records=n_records,
                expected_folds=2,
                q=3.0,
                expected_checkpoint_kind="final_ema",
                check_only=True,
                allow_missing_log=False,
            )
            with patch.object(freeze_oof, "LOG_DIR", log_dir), patch.object(
                freeze_oof,
                "current_checkpoint_rows",
                return_value=checkpoint_rows,
            ):
                payload = freeze_oof.validate_oof(args)
            self.assertEqual(payload["status"], "frozen")
            self.assertTrue(payload["checkpoint_fingerprints_match"])
            self.assertEqual(payload["validated_records"], n_records)


if __name__ == "__main__":
    unittest.main()
