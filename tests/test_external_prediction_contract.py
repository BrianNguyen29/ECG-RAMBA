import importlib
import json
import os
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

    def test_rocket_feature_precision_and_readonly_memmap_boundary(self):
        values = np.asarray([[0.10001, -1.2345, 123.4567]], dtype=np.float32)
        expected = values.astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(
            external.training_pca_compatible_rocket_values(values), expected
        )

        signals = np.zeros((1, 12, 5000), dtype=np.float32)
        signals.setflags(write=False)
        batch = external.writable_signal_batch(signals, 0, 1)
        self.assertTrue(batch.flags.writeable)
        self.assertTrue(batch.flags.c_contiguous)
        self.assertEqual(batch.dtype, np.float32)

    def test_rocket_feature_device_and_batch_configuration(self):
        with patch.object(external.torch.cuda, "is_available", return_value=False):
            self.assertEqual(external.resolve_rocket_feature_device("auto"), "cpu")
            self.assertEqual(external.resolve_rocket_feature_device("cpu"), "cpu")
            with self.assertRaises(RuntimeError):
                external.resolve_rocket_feature_device("cuda")
        self.assertEqual(
            external.resolve_rocket_feature_batch_size("cpu", 0),
            external.DEFAULT_ROCKET_CPU_BATCH_SIZE,
        )
        self.assertEqual(
            external.resolve_rocket_feature_batch_size("cuda", 0),
            external.DEFAULT_ROCKET_CUDA_BATCH_SIZE,
        )
        self.assertEqual(external.resolve_rocket_feature_batch_size("cuda", 256), 256)
        with self.assertRaises(ValueError):
            external.resolve_rocket_feature_batch_size("cpu", -1)

    def test_external_feature_cache_override_requires_canonical_absolute_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "source.zip"
            archive.write_bytes(b"source")
            pca = root / "fold1_pca.joblib"
            pca.write_bytes(b"pca")
            canonical_cache = root / "canonical" / "predictions" / "external_feature_cache"
            signals = np.zeros((2, 12, 5000), dtype=np.float32)
            record_ids = np.asarray(["record-a", "record-b"])

            with patch.dict(
                os.environ,
                {external.EXTERNAL_FEATURE_CACHE_DIR_ENV: str(canonical_cache)},
                clear=False,
            ):
                path, _ = external.feature_cache_path(
                    "cpsc2021", archive, signals, [pca], record_ids
                )
            self.assertEqual(path.parent, canonical_cache)
            self.assertTrue(canonical_cache.is_dir())

            with patch.dict(
                os.environ,
                {external.EXTERNAL_FEATURE_CACHE_DIR_ENV: "relative-cache"},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    external.feature_cache_path("cpsc2021", archive, signals, [pca], record_ids)

    def test_rocket_feature_partial_cache_resumes_only_matching_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "features.npz"
            contract = {
                "archive_sha256": "archive-a",
                "record_id_fingerprint": "records-a",
                "rocket_feature_value_contract": external.EXTERNAL_FEATURE_VALUE_CONTRACT,
            }
            raw, raw_path, progress_path, completed, resume_contract = (
                external.open_rocket_resume_cache(
                    cache_path,
                    contract,
                    n_records=3,
                    n_features=2,
                    batch_size=2,
                )
            )
            raw[0] = np.asarray([1.0, 2.0], dtype=np.float16)
            raw.flush()
            completed.add(0)
            external.save_rocket_resume_progress(progress_path, resume_contract, completed)
            owner = getattr(raw, "_mmap", None)
            if owner is not None:
                owner.close()

            reused, raw_path_2, progress_path_2, completed_2, resume_contract_2 = (
                external.open_rocket_resume_cache(
                    cache_path,
                    contract,
                    n_records=3,
                    n_features=2,
                    batch_size=2,
                )
            )
            self.assertEqual((raw_path_2, progress_path_2), (raw_path, progress_path))
            self.assertEqual(completed_2, {0})
            self.assertEqual(resume_contract_2, resume_contract)
            np.testing.assert_array_equal(reused[0], np.asarray([1.0, 2.0], dtype=np.float16))
            reused_owner = getattr(reused, "_mmap", None)
            if reused_owner is not None:
                reused_owner.close()

            # Starts are only meaningful under the same batch partition. A
            # changed batch size must quarantine rather than skip an unwritten
            # portion of a previously larger batch.
            changed, changed_raw, changed_progress, changed_completed, changed_contract = (
                external.open_rocket_resume_cache(
                    cache_path,
                    contract,
                    n_records=3,
                    n_features=2,
                    batch_size=1,
                )
            )
            self.assertEqual(changed_completed, set())
            self.assertEqual(changed_contract["batch_size"], 1)
            self.assertTrue(list(cache_path.parent.glob(f"{raw_path.name}.stale.*")))
            external.cleanup_rocket_resume(changed, changed_raw, changed_progress)
            self.assertFalse(raw_path.exists())
            self.assertFalse(progress_path.exists())

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

    def test_cpsc_capacity_prescan_counts_only_primary_eligible_windows(self):
        metadata = [{"record_id": "cpsc-1", "record_path": Path("cpsc-1")}]
        header = SimpleNamespace(fs=500.0, sig_len=15000)
        intervals = [
            (0, 5000, "normal"),
            (5000, 10000, "AF_or_AFL"),
            (10000, 12500, "normal"),
            (12500, 15000, "AF_or_AFL"),
        ]
        with (
            patch.object(external.wfdb, "rdheader", return_value=header),
            patch.object(
                external,
                "cpsc_rhythm_intervals",
                return_value=(intervals, {}),
            ),
        ):
            capacity = external.cpsc_window_capacity(metadata)
        self.assertEqual(capacity, 2)

    def test_cpsc_cache_contract_declares_exact_annotation_capacity(self):
        contract = external.cpsc_window_cache_contract(
            [{"record_id": "cpsc-1", "record_path": Path("cpsc-1")}],
            limit=0,
            source_archive_sha256="a" * 64,
        )
        self.assertEqual(contract["schema_version"], 3)
        self.assertEqual(
            contract["capacity_policy"],
            "annotation_prescan_exact_primary_windows_v1",
        )

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
