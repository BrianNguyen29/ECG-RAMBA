import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "revision"
    / "38_pipeline_storage_audit.py"
)
SPEC = importlib.util.spec_from_file_location("pipeline_storage_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PipelineStorageAuditTest(unittest.TestCase):
    def test_colab_drive_manifest_path_maps_to_local_synced_drive_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "drive"
            canonical = drive_root / "revision_artifacts" / "reports" / "revision"
            canonical.mkdir(parents=True)
            resolved = MODULE.resolve_declared_path(
                "/content/drive/MyDrive/ECG-Ramba/model_runs/run/fold1.pt",
                canonical,
            )
            self.assertEqual(resolved, drive_root / "model_runs" / "run" / "fold1.pt")

    def test_counts_required_slots_not_matching_file_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "predictions" / "folds"
            fold_dir.mkdir(parents=True)
            for variant in range(5):
                (fold_dir / f"oof_fold1_final_ema_cfg_hash{variant}_v2.npz").write_bytes(
                    b"cache"
                )

            prediction_dir = root / "predictions"
            for stress in MODULE.STRESSES:
                if stress == "snr5db":
                    continue
                (prediction_dir / f"robustness_minirocket_{stress}_predictions.npz").write_bytes(
                    b"prediction"
                )
            (prediction_dir / "robustness_minirocket_clean_ref_predictions.npz").write_bytes(
                b"reference"
            )

            rows = {
                row.stage: row for row in MODULE.audit_stages(root, manifest_rows={})
            }
            self.assertEqual(rows["oof_fold_cache"].found_count, 1)
            self.assertEqual(rows["oof_fold_cache"].status, "partial")
            self.assertEqual(rows["minirocket_stress_predictions"].found_count, 5)
            self.assertEqual(rows["minirocket_stress_predictions"].status, "partial")
            self.assertEqual(
                rows["minirocket_stress_predictions"].missing_items,
                "snr5db",
            )

    def test_complete_requires_manifest_coverage_for_every_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "experimental" / "transformer_ecg_checkpoints"
            checkpoint_dir.mkdir(parents=True)
            manifest_rows = {}
            for fold in range(1, 6):
                path = checkpoint_dir / f"fold{fold}_transformer_ecg_final.pt"
                path.write_bytes(b"checkpoint")
                if fold != 5:
                    manifest_rows[path.relative_to(root).as_posix()] = {}

            row = next(
                row
                for row in MODULE.audit_stages(root, manifest_rows)
                if row.stage == "transformer_checkpoints"
            )
            self.assertEqual(row.found_count, 5)
            self.assertEqual(row.manifest_covered_count, 4)
            self.assertEqual(row.unmanifested_items, "fold5")
            self.assertEqual(row.status, "complete_needs_publish")

    def test_full_model_checkpoint_contract_uses_oof_manifest_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_dir = root / "manifests"
            model_dir = root / "model_runs" / "frozen"
            manifest_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            checkpoints = []
            for fold in range(1, 6):
                path = model_dir / f"fold{fold}_final_ema.pt"
                path.write_bytes(f"fold-{fold}".encode("ascii"))
                checkpoints.append(
                    {
                        "fold": fold,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": MODULE.sha256_file(path),
                    }
                )
            (manifest_dir / "oof_final_ema_prediction_run_manifest.json").write_text(
                json.dumps({"inputs": {"checkpoints": checkpoints}}),
                encoding="utf-8",
            )

            row, issues = MODULE.audit_full_model_checkpoints(root, full_sha=True)
            self.assertEqual(issues, [])
            self.assertEqual(row.status, "complete_manifested")
            self.assertEqual(row.found_count, 5)
            self.assertEqual(row.manifest_covered_count, 5)

    def test_learned_prediction_contract_matches_checkpoint_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "manifest": "manifests/test_baseline_manifest.json",
                "predictions": ("predictions/test_oof.npz",),
                "checkpoint": "experimental/test/fold{fold}.pt",
            }
            checkpoint_rows = []
            mirror_rows = {}
            hashes = []
            for fold in range(1, 6):
                path = root / config["checkpoint"].format(fold=fold)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"checkpoint-{fold}".encode("ascii"))
                digest = MODULE.sha256_file(path)
                hashes.append(digest)
                checkpoint_rows.append({"fold": fold, "sha256": digest})
                mirror_rows[path.relative_to(root).as_posix()] = {"sha256": digest}

            baseline_manifest = root / config["manifest"]
            baseline_manifest.parent.mkdir(parents=True, exist_ok=True)
            baseline_manifest.write_text(
                json.dumps(
                    {
                        "checkpoint_contract": {
                            "status": "complete",
                            "checkpoints": checkpoint_rows,
                        }
                    }
                ),
                encoding="utf-8",
            )
            mirror_rows[baseline_manifest.relative_to(root).as_posix()] = {
                "sha256": MODULE.sha256_file(baseline_manifest)
            }

            prediction = root / config["predictions"][0]
            prediction.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction,
                checkpoint_folds=np.asarray([1, 2, 3, 4, 5], dtype=np.int16),
                checkpoint_sha256=np.asarray(hashes),
            )
            mirror_rows[prediction.relative_to(root).as_posix()] = {
                "sha256": MODULE.sha256_file(prediction)
            }

            with patch.object(MODULE, "LEARNED_BASELINE_CONTRACTS", {"test": config}):
                row, issues = MODULE.audit_learned_prediction_contracts(
                    root,
                    mirror_rows,
                    full_sha=True,
                )
                self.assertEqual(issues, [])
                self.assertEqual(row.status, "complete_manifested")

                np.savez_compressed(
                    prediction,
                    checkpoint_folds=np.asarray([1, 2, 3, 4, 5], dtype=np.int16),
                    checkpoint_sha256=np.asarray(["wrong"] * 5),
                )
                row, issues = MODULE.audit_learned_prediction_contracts(
                    root,
                    mirror_rows,
                    full_sha=False,
                )
                self.assertEqual(row.status, "partial")
                self.assertTrue(any("checkpoint SHA contract mismatch" in issue for issue in issues))

    def test_fold_pca_manifest_binds_exact_five_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifests" / "fold_pca_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            rows = []
            for fold in range(1, 6):
                path = root / "pca" / f"fold{fold}.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"pca-{fold}".encode("ascii"))
                rows.append(
                    {
                        "fold": fold,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": MODULE.sha256_file(path),
                    }
                )
            manifest_path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "checkpoint_kind": "final_ema",
                        "fold_pca": rows,
                    }
                ),
                encoding="utf-8",
            )

            row, issues = MODULE.audit_fold_pca_models(root, full_sha=True)
            self.assertEqual(row.status, "complete_manifested")
            self.assertEqual(row.found_count, 5)
            self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
