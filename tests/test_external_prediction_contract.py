import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from configs.config import CLASS_TO_IDX, CLASSES


external = importlib.import_module("scripts.revision.03_generate_external_predictions")


class ExternalPredictionContractTests(unittest.TestCase):
    def test_external_checkpoint_files_must_match_oof_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            rows = []
            for fold in range(1, 6):
                path = root / f"fold{fold}_final_ema.pt"
                path.write_bytes(f"trusted-fold-{fold}".encode())
                paths.append(path)
                rows.append(
                    {
                        "fold": fold,
                        "sha256": external.sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            expected_oof_sha = "a" * 64
            manifest = manifest_dir / "oof_final_ema_prediction_run_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "inputs": {"checkpoints": rows},
                        "outputs": {"prediction_file": {"sha256": expected_oof_sha}},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(external, "MANIFEST_DIR", manifest_dir):
                contract = external.validate_checkpoint_files_against_oof_run_manifest(
                    paths, "final_ema", expected_oof_sha
                )
                self.assertEqual(contract["prediction_sha256"], expected_oof_sha)
                paths[2].write_bytes(b"tampered")
                with self.assertRaises(RuntimeError):
                    external.validate_checkpoint_files_against_oof_run_manifest(
                        paths, "final_ema", expected_oof_sha
                    )

    def test_checkpoint_provenance_requires_matching_explicit_ema_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for fold in (1, 2):
                path = Path(tmp) / f"fold{fold}_final_ema.pt"
                torch.save(
                    {
                        "model": {},
                        "weights_kind": "ema",
                        "epoch": 20,
                        "selection_rule": "fixed_final_epoch",
                        "config_hash": "source123",
                        "dataset_record_order_fingerprint": "records123",
                    },
                    path,
                )
                paths.append(path)
            rows, source_hash = external.checkpoint_provenance(paths, "final_ema")
            self.assertEqual(source_hash, "source123")
            self.assertEqual([row["weights_kind"] for row in rows], ["ema", "ema"])

    def test_ptb_mapping_uses_declared_chapman_classes(self):
        probs = np.zeros((2, len(CLASSES)), dtype=np.float32)
        probs[:, CLASS_TO_IDX["SNR"]] = [0.8, 0.1]
        probs[:, CLASS_TO_IDX["QAb"]] = [0.2, 0.7]
        mapped, names = external.map_model_probabilities("ptbxl", probs)
        self.assertEqual(list(names), ["NORM", "MI", "STTC", "CD"])
        np.testing.assert_allclose(mapped[:, 0], [0.8, 0.1])
        np.testing.assert_allclose(mapped[:, 1], [0.2, 0.7])

    def test_cpsc_af_output_combines_af_and_afl(self):
        probs = np.zeros((2, len(CLASSES)), dtype=np.float32)
        probs[:, CLASS_TO_IDX["AF"]] = [0.6, 0.2]
        probs[:, CLASS_TO_IDX["AFL"]] = [0.3, 0.9]
        mapped, names = external.map_model_probabilities("cpsc2021", probs)
        self.assertEqual(list(names), ["AF_or_AFL"])
        np.testing.assert_allclose(mapped[:, 0], [0.6, 0.9])

    def test_checkpoint_compatible_hrv_keeps_amplitude_slots_zero(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(12, 5000)).astype(np.float32)
        features = external.checkpoint_compatible_hrv36(signal)
        self.assertEqual(features.shape, (36,))
        np.testing.assert_array_equal(features[25:30], np.zeros(5, dtype=np.float32))

    def test_cpsc_annotation_failure_is_not_converted_to_negative(self):
        with patch.object(external.wfdb, "rdann", side_effect=FileNotFoundError("missing atr")):
            with self.assertRaises(FileNotFoundError):
                external.cpsc_af_intervals(Path("missing"), signal_length=1000)

    def test_cpsc_intervals_follow_annotation_boundaries(self):
        annotation = SimpleNamespace(
            sample=np.asarray([0, 100, 300, 500]),
            aux_note=["(N", "(AFIB", "(N", "(AFL"],
        )
        with patch.object(external.wfdb, "rdann", return_value=annotation):
            intervals = external.cpsc_af_intervals(Path("record"), signal_length=700)
        self.assertEqual(intervals, [(100, 300), (500, 700)])
        self.assertEqual(external.interval_overlap(intervals, 250, 550), 100)

    def test_cpsc_rhythm_intervals_keep_normal_boundaries(self):
        annotation = SimpleNamespace(
            sample=np.asarray([0, 100, 300, 500]),
            aux_note=["(N", "(AFIB", "(N", "(AFL"],
        )
        with patch.object(external.wfdb, "rdann", return_value=annotation):
            intervals, counts = external.cpsc_rhythm_intervals(Path("record"), signal_length=700)
        self.assertEqual(
            intervals,
            [(0, 100, "normal"), (100, 300, "AF_or_AFL"), (300, 500, "normal"), (500, 700, "AF_or_AFL")],
        )
        self.assertEqual(counts["normal_intervals"], 2)
        self.assertEqual(external.interval_overlap(intervals, 300, 500, "normal"), 200)

    def test_cpsc_disk_backed_loader_avoids_full_signal_ram_accumulation(self):
        record = SimpleNamespace(
            p_signal=np.zeros((5000, 12), dtype=np.float32),
            d_signal=None,
            sig_name=list(external.STANDARD_LEADS),
            fs=500.0,
        )
        metadata = [{"record_id": "cpsc-1", "record_path": Path("cpsc-1")}]
        intervals = [(0, 5000, "normal")]
        interval_counts = {
            "recognized_intervals": 1,
            "af_intervals": 0,
            "normal_intervals": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            memmap_path = Path(tmp) / "cpsc_windows.npy"
            with (
                patch.object(
                    external,
                    "cpsc_metadata",
                    return_value=(metadata, {"label_protocol": "test"}),
                ),
                patch.object(external, "cpsc_window_capacity", return_value=2),
                patch.object(external.wfdb, "rdrecord", return_value=record),
                patch.object(
                    external,
                    "cpsc_rhythm_intervals",
                    return_value=(intervals, interval_counts),
                ),
                patch.object(external, "bandpass_filter", side_effect=lambda value, fs: value),
                patch.object(external, "normalize_signal", side_effect=lambda value: value),
            ):
                signals, labels, record_ids, group_ids, split_ids, summary = (
                    external.load_cpsc_windows(
                        Path(tmp),
                        limit=0,
                        signal_memmap_path=memmap_path,
                    )
                )

            self.assertIsInstance(signals, np.memmap)
            self.assertEqual(signals.dtype, np.float32)
            self.assertEqual(signals.shape, (1, 12, 5000))
            np.testing.assert_array_equal(labels, [[0.0]])
            np.testing.assert_array_equal(record_ids, ["cpsc-1:0:5000"])
            np.testing.assert_array_equal(group_ids, ["cpsc-1"])
            np.testing.assert_array_equal(split_ids, ["cpsc2021_external_pool"])
            self.assertEqual(
                summary["signal_storage"],
                "disk_backed_float32_npy_source_bound_resumable",
            )
            self.assertTrue(memmap_path.exists())
            self.assertTrue((Path(tmp) / "cpsc_windows.npy.contract.npz").exists())
            self.assertFalse((Path(tmp) / ".cpsc_windows.npy.partial.npy").exists())
            self.assertFalse((Path(tmp) / ".cpsc_windows.npy.progress.json").exists())
            mmap_owner = getattr(signals, "_mmap", None)
            if mmap_owner is not None:
                mmap_owner.close()
            del signals

            with (
                patch.object(
                    external,
                    "cpsc_metadata",
                    return_value=(metadata, {"label_protocol": "test"}),
                ),
                patch.object(
                    external.wfdb,
                    "rdrecord",
                    side_effect=AssertionError("completed CPSC cache should avoid signal loading"),
                ),
            ):
                reused = external.load_cpsc_windows(
                    Path(tmp),
                    limit=0,
                    signal_memmap_path=memmap_path,
                )
            self.assertEqual(reused[0].shape, (1, 12, 5000))
            reused_owner = getattr(reused[0], "_mmap", None)
            if reused_owner is not None:
                reused_owner.close()

    def test_georgia_skips_records_without_mapped_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mapped.hea").write_text("#Dx: 164889003\n", encoding="utf-8")
            (root / "unmapped.hea").write_text("#Dx: 999999999\n", encoding="utf-8")
            rows, summary = external.georgia_metadata(root, limit=0)
        self.assertEqual([row["record_id"] for row in rows], ["mapped"])
        self.assertEqual(summary["skipped_records_without_mapped_label"], 1)
        self.assertEqual(summary["unmapped_snomed_codes"], {"999999999": 1})

    def test_georgia_review_file_disables_unreviewed_builtin_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = root / "review.csv"
            review.write_text(
                "source_code,source_label,action,mapped_target,rationale,source_reference\n"
                "428750005,Nonspecific ST-T abnormality,defer_requires_domain_review,,No exact target,test\n",
                encoding="utf-8",
            )
            (root / "unreviewed_builtin.hea").write_text("#Dx: 164889003\n", encoding="utf-8")
            (root / "deferred.hea").write_text("#Dx: 428750005\n", encoding="utf-8")
            rows, summary = external.georgia_metadata(root, limit=0, mapping_review_path=review)
        self.assertEqual(rows, [])
        self.assertEqual(summary["mapping_review_mapped_codes"], 0)
        self.assertEqual(summary["skipped_records_without_mapped_label"], 2)
        self.assertEqual(summary["unmapped_snomed_codes"], {"428750005": 1, "164889003": 1})

    def test_georgia_explicit_missing_review_file_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mapped.hea").write_text("#Dx: 164889003\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                external.georgia_metadata(root, limit=0, mapping_review_path=root / "missing_review.csv")

    def test_ptb_uses_positive_likelihood_not_only_100(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "ecg_id": [1],
                    "patient_id": [101],
                    "strat_fold": [10],
                    "scp_codes": ["{'NORM': 50.0, 'HYP': 25.0}"],
                    "filename_hr": ["records/00001_hr"],
                }
            ).set_index("ecg_id").to_csv(root / "ptbxl_database.csv")
            pd.DataFrame(
                {
                    "diagnostic": [1, 1],
                    "diagnostic_class": ["NORM", "HYP"],
                },
                index=["NORM", "HYP"],
            ).to_csv(root / "scp_statements.csv")
            rows, summary = external.ptb_metadata(root, limit=0)
        np.testing.assert_array_equal(rows[0]["y_true"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(summary["unsupported_superclasses"], {"HYP": 1})


if __name__ == "__main__":
    unittest.main()
