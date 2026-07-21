import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


lock_runner = load_script("51_ptbxl_adaptation_analysis_lock.py", "_test_ptbxl_lock")
audit = load_script("52_ptbxl_fold_protocol_audit.py", "_test_ptbxl_audit")
refresh = load_script("50_refresh_in_domain_paired_contracts.py", "_test_paired_refresh")


class PTBXLFoldProtocolAuditTests(unittest.TestCase):
    @staticmethod
    def lock_args() -> argparse.Namespace:
        return argparse.Namespace(
            models="full,resnet,raw_mamba,transformer",
            fractions="0,0.01,0.05,0.10",
            primary_fraction=0.10,
            seeds="42,43,44,45,46",
            threshold=0.5,
            n_bins=15,
            n_boot=1000,
            head_c=1.0,
            max_iter=5000,
        )

    def test_analysis_lock_is_exact_and_explicitly_not_preregistration(self):
        expected = lock_runner.expected_lock(self.lock_args())
        self.assertEqual(expected["protocol"]["adaptation_split"], "official_ptbxl_fold9")
        self.assertEqual(expected["protocol"]["test_split"], "official_ptbxl_fold10")
        self.assertEqual(expected["protocol"]["patient_overlap_required"], 0)
        self.assertIn("not a preregistration", expected["temporal_qualification"])
        self.assertEqual(lock_runner.validate_existing(expected, expected), [])
        changed = json.loads(json.dumps(expected))
        changed["protocol"]["primary_fraction"] = 0.05
        self.assertIn("protocol", lock_runner.validate_existing(changed, expected))

        implementation_drift = json.loads(json.dumps(expected))
        implementation_drift["runner_sources"][0]["sha256"] = "0" * 64
        self.assertEqual(lock_runner.validate_existing(implementation_drift, expected), [])
        drift = lock_runner.runner_source_drift(implementation_drift, expected)
        self.assertEqual(len(drift), 1)
        self.assertEqual(
            drift[0]["classification"],
            "implementation_changed_after_protocol_lock",
        )

        malformed_sources = json.loads(json.dumps(expected))
        malformed_sources["runner_sources"][0]["sha256"] = "not-a-sha"
        self.assertIn(
            "runner_sources",
            lock_runner.validate_existing(malformed_sources, expected),
        )

    def test_source_attestation_preserves_immutable_lock_bytes(self):
        expected = lock_runner.expected_lock(self.lock_args())
        locked = json.loads(json.dumps(expected))
        locked["created_utc"] = "2026-07-20T00:00:00+00:00"
        locked["runner_sources"][0]["sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / "lock.json"
            attestation_path = root / "attestation.json"
            lock_runner.save_json_atomic(lock_path, locked)
            before = lock_runner.sha256_file(lock_path)

            attestation = lock_runner.write_source_attestation(
                attestation_path,
                lock_path=lock_path,
                lock=locked,
                current=expected,
            )

            self.assertEqual(lock_runner.sha256_file(lock_path), before)
            self.assertEqual(attestation["analysis_lock"]["sha256"], before)
            self.assertTrue(attestation["protocol_unchanged"])
            self.assertTrue(attestation["locked_runner_sources_preserved"])
            self.assertEqual(len(attestation["runner_source_drift"]), 1)
            self.assertEqual(
                json.loads(attestation_path.read_text(encoding="utf-8"))["analysis_lock"]["sha256"],
                before,
            )

    def test_fold_prediction_loader_and_patient_overlap_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold10.npz"
            np.savez_compressed(
                path,
                y_true=np.asarray([[1, 0], [0, 0], [0, 1]], dtype=np.float32),
                y_prob=np.asarray([[0.8, 0.1], [0.2, 0.3], [0.1, 0.9]], dtype=np.float32),
                record_id=np.asarray(["r1", "r2", "r3"]),
                group_id=np.asarray(["p1", "p2", "p3"]),
                split_id=np.asarray(["ptbxl_fold10"] * 3),
                class_names=np.asarray(["NORM", "CD"]),
                dataset=np.asarray("ptbxl"),
                group_unit=np.asarray("patient_id"),
            )
            payload = audit.load_prediction(path, "ptbxl_fold10")
            self.assertEqual(int(np.sum(~np.any(payload["y_true"] > 0.5, axis=1))), 1)
            self.assertEqual(audit.patient_overlap(np.asarray(["p9", "p2"]), payload["group_id"]), ["p2"])
            candidate = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in payload.items()}
            audit.validate_same_reference(payload, candidate, "same")
            candidate["record_id"][0] = "changed"
            with self.assertRaisesRegex(RuntimeError, "record_id differs"):
                audit.validate_same_reference(payload, candidate, "changed")

    def test_official_archive_binds_patient_ids_and_fold_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "ptbxl.zip"
            metadata = (
                "ecg_id,patient_id,strat_fold\n"
                "1,101,9\n"
                "2,202,10\n"
                "3,303,10\n"
            )
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv", metadata)

            official, provenance = audit.load_official_ptbxl_metadata(archive_path)
            self.assertEqual(official["1"], ("101", 9))
            self.assertEqual(provenance["metadata_rows"], 3)
            self.assertEqual(len(provenance["archive_sha256"]), 64)

            fold10 = {
                "record_id": np.asarray(["2.0", "3"]),
                "group_id": np.asarray(["202", "303.0"]),
            }
            audit.validate_against_official_metadata(
                fold10, official, expected_fold=10, label="fold10"
            )

            wrong_patient = {**fold10, "group_id": np.asarray(["999", "303"])}
            with self.assertRaisesRegex(RuntimeError, "patient IDs differ"):
                audit.validate_against_official_metadata(
                    wrong_patient, official, expected_fold=10, label="fold10"
                )

            with self.assertRaisesRegex(RuntimeError, "official strat_fold differs"):
                audit.validate_against_official_metadata(
                    fold10, official, expected_fold=9, label="fold9"
                )

    def test_protocol_lock_validation_rejects_unlocked_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            payload = {**lock_runner.expected_lock(self.lock_args()), "created_utc": "test"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            args = argparse.Namespace(threshold=0.5, n_bins=15, n_boot=1000)
            models = ["full", "resnet", "raw_mamba", "transformer"]
            validated = audit.validate_lock(path, models, args)
            self.assertEqual(validated["protocol_sha256"], payload["protocol_sha256"])
            payload["protocol"]["test_split"] = "ptbxl_fold9"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "analysis lock"):
                audit.validate_lock(path, models, args)

    def test_paired_refresh_requires_exact_current_oof_and_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "paired": root / "paired.json",
                "table": root / "table.csv",
                "samples": root / "samples.csv",
                "paired_manifest": root / "manifest.json",
            }
            for key in ("table", "samples"):
                config[key].write_text("x\n1\n", encoding="utf-8")
            metrics = {
                name: {"n_boot_valid": 1000}
                for name in ("pr_auc_macro", "roc_auc_macro", "f1_macro", "brier_macro", "ece_macro")
            }
            config["paired"].write_text(
                json.dumps(
                    {
                        "inputs": {
                            "full_predictions": {"sha256": "oof"},
                            "freeze_manifest": {"sha256": "freeze"},
                        },
                        "metrics": metrics,
                    }
                ),
                encoding="utf-8",
            )
            config["paired_manifest"].write_text(
                json.dumps(
                    {
                        "comparison": "full_vs_test",
                        "input_sha256": {"full_predictions": "oof", "freeze_manifest": "freeze"},
                        "paired_bootstrap": {"n_boot": 1000},
                        "artifact_sha256": {
                            "json": refresh.sha256_file(config["paired"]),
                            "table": refresh.sha256_file(config["table"]),
                            "bootstrap_samples": refresh.sha256_file(config["samples"]),
                        },
                    }
                ),
                encoding="utf-8",
            )
            ready, issues = refresh.validate_paired(
                "resnet", config, {"oof_sha256": "oof", "freeze_sha256": "freeze"}, 1000
            )
            self.assertTrue(ready, issues)
            ready, issues = refresh.validate_paired(
                "resnet", config, {"oof_sha256": "oof", "freeze_sha256": "new"}, 1000
            )
            self.assertFalse(ready)
            self.assertIn("canonical_freeze_sha256", issues)


if __name__ == "__main__":
    unittest.main()
