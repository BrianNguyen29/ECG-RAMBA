import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


gate = importlib.import_module("scripts.revision.18_external_protocol_gate")


class ExternalProtocolGateTests(unittest.TestCase):
    def test_missing_external_artifacts_write_blocked_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                external_root=root / "external",
                expected_checkpoint_kind="final_ema",
                threshold=0.5,
                n_bins=5,
                n_boot=0,
                seed=42,
                reuse_existing=False,
                oof_run_manifest=root / "missing_oof_manifest.json",
            )
            with (
                patch.object(gate, "METRIC_DIR", root / "metrics"),
                patch.object(gate, "TABLE_DIR", root / "tables"),
                patch.object(gate, "MANIFEST_DIR", root / "manifests"),
            ):
                payload, metrics, labels = gate.validate_dataset("ptbxl", args, {})
                self.assertFalse(payload["protocol_gate_passed"])
                self.assertEqual(payload["status"], "blocked_missing_external_artifacts")
                self.assertEqual(metrics, [])
                self.assertEqual(labels, [])
                self.assertTrue((root / "metrics" / "external_ptbxl_protocol_gate.json").exists())

    def test_synthetic_ptb_external_artifacts_pass_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external_root = root / "external"
            output = external_root / "ptbxl"
            output.mkdir(parents=True)
            metrics_dir = root / "metrics"
            tables_dir = root / "tables"
            manifests_dir = root / "manifests"
            oof_manifest = root / "oof_manifest.json"
            archive = root / "PTB-XL.zip"
            archive.write_bytes(b"synthetic archive")
            pca_rows = []
            oof_checkpoints = {}
            checkpoint_rows = []
            for fold in range(1, 6):
                pca = root / f"fold{fold}_pca.joblib"
                ckpt = root / f"fold{fold}_final_ema.pt"
                pca.write_bytes(f"pca {fold}".encode())
                ckpt.write_bytes(f"ckpt {fold}".encode())
                pca_rows.append(
                    {
                        "fold": fold,
                        "path": str(pca),
                        "sha256": gate.sha256_file(pca),
                        "scope": "chapman_training_fold_only",
                    }
                )
                ckpt_row = {
                    "fold": fold,
                    "path": str(ckpt),
                    "sha256": gate.sha256_file(ckpt),
                    "weights_kind": "ema",
                }
                checkpoint_rows.append(ckpt_row)
                oof_checkpoints[fold] = ckpt_row
            oof_manifest.write_text(json.dumps({"checkpoints": checkpoint_rows}), encoding="utf-8")

            y_true = np.asarray(
                [
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                ],
                dtype=np.float32,
            )
            y_prob = np.asarray(
                [
                    [0.9, 0.1, 0.2, 0.8],
                    [0.1, 0.8, 0.7, 0.2],
                    [0.7, 0.6, 0.3, 0.3],
                    [0.2, 0.4, 0.8, 0.7],
                ],
                dtype=np.float32,
            )
            prediction = output / "ptbxl_full_predictions.npz"
            np.savez_compressed(
                prediction,
                y_true=y_true,
                y_prob=y_prob,
                record_id=np.asarray(["1", "2", "3", "4"]),
                group_id=np.asarray(["patient_1", "patient_2", "patient_3", "patient_4"]),
                split_id=np.asarray(["ptbxl_fold10"] * 4),
                group_unit=np.asarray("patient_id"),
                class_names=np.asarray(["NORM", "MI", "STTC", "CD"]),
                dataset=np.asarray("ptbxl"),
                evidence_status=np.asarray("experimental"),
                manuscript_ready=np.asarray(False),
                aggregation_method=np.asarray("power_mean"),
                aggregation_q=np.asarray(float(gate.CONFIG["power_mean_q"]), dtype=np.float32),
                cache_schema_version=np.asarray(gate.CACHE_SCHEMA_VERSION, dtype=np.int16),
                checkpoint_fingerprints_json=np.asarray(json.dumps(checkpoint_rows)),
                pca_fingerprints_json=np.asarray(json.dumps(pca_rows)),
            )
            np.savez_compressed(
                output / "ptbxl_full_slice_predictions.npz",
                slice_prob=y_prob,
                record_index=np.asarray([0, 1, 2, 3], dtype=np.int64),
                record_id=np.asarray(["1", "2", "3", "4"]),
                group_id=np.asarray(["patient_1", "patient_2", "patient_3", "patient_4"]),
                split_id=np.asarray(["ptbxl_fold10"] * 4),
                slice_index=np.asarray([0, 0, 0, 0], dtype=np.int64),
                class_names=np.asarray(["NORM", "MI", "STTC", "CD"]),
                dataset=np.asarray("ptbxl"),
                cache_schema_version=np.asarray(gate.CACHE_SCHEMA_VERSION, dtype=np.int16),
                evidence_status=np.asarray("experimental"),
                manuscript_ready=np.asarray(False),
            )
            (output / "ptbxl_full_class_summary.csv").write_text(
                "dataset,class_name\nptbxl,NORM\n",
                encoding="utf-8",
            )
            summary = {
                "dataset": "ptbxl",
                "evidence_status": "experimental",
                "manuscript_ready": False,
                "n_records": 4,
                "n_classes": 4,
                "checkpoint_kind": "final_ema",
                "load_summary": {
                    "label_protocol": gate.EXPECTED_EXTERNAL_PROTOCOLS["ptbxl"],
                    "unsupported_superclasses": {"HYP": 2},
                    "records_without_supported_superclass": 0,
                },
            }
            (output / "ptbxl_full_prediction_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            manifest = {
                "evidence_status": "experimental",
                "manuscript_ready": False,
                "archive": {
                    "path": str(archive),
                    "size_bytes": archive.stat().st_size,
                    "fingerprint": gate.file_fingerprint(archive),
                },
                "pca": {
                    "scope": "fold_specific_chapman_training_only",
                    "folds": pca_rows,
                },
                "checkpoints": checkpoint_rows,
            }
            (output / "ptbxl_full_prediction_run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                external_root=external_root,
                expected_checkpoint_kind="final_ema",
                threshold=0.5,
                n_bins=5,
                n_boot=0,
                seed=42,
                reuse_existing=False,
                oof_run_manifest=oof_manifest,
            )
            with (
                patch.object(gate, "METRIC_DIR", metrics_dir),
                patch.object(gate, "TABLE_DIR", tables_dir),
                patch.object(gate, "MANIFEST_DIR", manifests_dir),
                patch.dict(gate.PATHS, {"ptb_zip": str(archive)}),
            ):
                payload, metrics, labels = gate.validate_dataset(
                    "ptbxl",
                    args,
                    oof_checkpoints,
                )
                args.reuse_existing = True
                cached_payload, cached_metrics, cached_labels = gate.validate_dataset(
                    "ptbxl",
                    args,
                    oof_checkpoints,
                )
            self.assertTrue(payload["protocol_gate_passed"])
            self.assertEqual(payload["status"], "protocol_gate_passed")
            self.assertEqual(payload["gate_schema_version"], gate.GATE_SCHEMA_VERSION)
            self.assertTrue(payload["gate_cache_key"])
            self.assertEqual(len(metrics), 1)
            self.assertEqual(len(labels), 4)
            self.assertTrue(cached_payload["reused_existing"])
            self.assertEqual(cached_payload["gate_cache_key"], payload["gate_cache_key"])
            self.assertEqual(cached_metrics, [])
            self.assertEqual(cached_labels, [])

    def test_synthetic_ptb_gate_blocks_without_oof_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external_root = root / "external"
            output = external_root / "ptbxl"
            output.mkdir(parents=True)
            metrics_dir = root / "metrics"
            tables_dir = root / "tables"
            manifests_dir = root / "manifests"
            archive = root / "PTB-XL.zip"
            archive.write_bytes(b"synthetic archive")
            pca_rows = []
            checkpoint_rows = []
            for fold in range(1, 6):
                pca = root / f"fold{fold}_pca.joblib"
                ckpt = root / f"fold{fold}_final_ema.pt"
                pca.write_bytes(f"pca {fold}".encode())
                ckpt.write_bytes(f"ckpt {fold}".encode())
                pca_rows.append(
                    {
                        "fold": fold,
                        "path": str(pca),
                        "sha256": gate.sha256_file(pca),
                        "scope": "chapman_training_fold_only",
                    }
                )
                checkpoint_rows.append(
                    {
                        "fold": fold,
                        "path": str(ckpt),
                        "sha256": gate.sha256_file(ckpt),
                        "weights_kind": "ema",
                    }
                )
            y_true = np.asarray(
                [[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]],
                dtype=np.float32,
            )
            y_prob = np.asarray(
                [[0.9, 0.1, 0.2, 0.8], [0.1, 0.8, 0.7, 0.2], [0.7, 0.6, 0.3, 0.3], [0.2, 0.4, 0.8, 0.7]],
                dtype=np.float32,
            )
            np.savez_compressed(
                output / "ptbxl_full_predictions.npz",
                y_true=y_true,
                y_prob=y_prob,
                record_id=np.asarray(["1", "2", "3", "4"]),
                class_names=np.asarray(["NORM", "MI", "STTC", "CD"]),
                dataset=np.asarray("ptbxl"),
                evidence_status=np.asarray("experimental"),
                manuscript_ready=np.asarray(False),
                aggregation_method=np.asarray("power_mean"),
                aggregation_q=np.asarray(float(gate.CONFIG["power_mean_q"]), dtype=np.float32),
                cache_schema_version=np.asarray(gate.CACHE_SCHEMA_VERSION, dtype=np.int16),
                checkpoint_fingerprints_json=np.asarray(json.dumps(checkpoint_rows)),
                pca_fingerprints_json=np.asarray(json.dumps(pca_rows)),
            )
            np.savez_compressed(
                output / "ptbxl_full_slice_predictions.npz",
                slice_prob=y_prob,
                record_index=np.asarray([0, 1, 2, 3], dtype=np.int64),
                record_id=np.asarray(["1", "2", "3", "4"]),
                slice_index=np.asarray([0, 0, 0, 0], dtype=np.int64),
                class_names=np.asarray(["NORM", "MI", "STTC", "CD"]),
                dataset=np.asarray("ptbxl"),
                cache_schema_version=np.asarray(gate.CACHE_SCHEMA_VERSION, dtype=np.int16),
                evidence_status=np.asarray("experimental"),
                manuscript_ready=np.asarray(False),
            )
            (output / "ptbxl_full_class_summary.csv").write_text("dataset,class_name\nptbxl,NORM\n", encoding="utf-8")
            (output / "ptbxl_full_prediction_summary.json").write_text(
                json.dumps(
                    {
                        "dataset": "ptbxl",
                        "evidence_status": "experimental",
                        "manuscript_ready": False,
                        "n_records": 4,
                        "n_classes": 4,
                        "checkpoint_kind": "final_ema",
                        "load_summary": {
                            "label_protocol": gate.EXPECTED_EXTERNAL_PROTOCOLS["ptbxl"],
                            "unsupported_superclasses": {"HYP": 2},
                            "records_without_supported_superclass": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output / "ptbxl_full_prediction_run_manifest.json").write_text(
                json.dumps(
                    {
                        "evidence_status": "experimental",
                        "manuscript_ready": False,
                        "archive": {
                            "path": str(archive),
                            "size_bytes": archive.stat().st_size,
                            "fingerprint": gate.file_fingerprint(archive),
                        },
                        "pca": {"scope": "fold_specific_chapman_training_only", "folds": pca_rows},
                        "checkpoints": checkpoint_rows,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                external_root=external_root,
                expected_checkpoint_kind="final_ema",
                threshold=0.5,
                n_bins=5,
                n_boot=0,
                seed=42,
                reuse_existing=False,
                oof_run_manifest=root / "missing_oof_manifest.json",
            )
            with (
                patch.object(gate, "METRIC_DIR", metrics_dir),
                patch.object(gate, "TABLE_DIR", tables_dir),
                patch.object(gate, "MANIFEST_DIR", manifests_dir),
                patch.dict(gate.PATHS, {"ptb_zip": str(archive)}),
            ):
                payload, metrics, labels = gate.validate_dataset("ptbxl", args, {})
            self.assertFalse(payload["protocol_gate_passed"])
            self.assertEqual(payload["status"], "blocked_protocol_gate_failed")
            self.assertIn("frozen OOF checkpoint contract", "; ".join(payload["issues"]))
            self.assertEqual(metrics[0]["protocol_gate_passed"], False)


if __name__ == "__main__":
    unittest.main()
