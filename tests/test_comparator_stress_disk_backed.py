import importlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


STRESS = importlib.import_module(
    "scripts.revision.23_generate_comparator_stress_predictions"
)


class ComparatorStressDiskBackedTests(unittest.TestCase):
    def test_npz_member_extraction_preserves_npy_payload(self):
        values = np.arange(72, dtype=np.float32).reshape(2, 3, 12)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "cache.npz"
            destination = root / "cache_x.npy"
            np.savez_compressed(source, X=values)

            shape, dtype, member_size = STRESS.npz_member_contract(source, "X.npy")
            source_sha = STRESS.sha256_file(source)
            contract = STRESS.extract_npz_member_atomic(
                source,
                "X.npy",
                destination,
                source_sha256=source_sha,
            )
            restored = np.load(destination, mmap_mode="r", allow_pickle=False)

            self.assertEqual(shape, values.shape)
            self.assertEqual(dtype, values.dtype)
            self.assertEqual(destination.stat().st_size, member_size)
            np.testing.assert_array_equal(restored, values)
            self.assertEqual(contract["source_npz_sha256"], source_sha)
            self.assertEqual(contract["member_npy_sha256"], STRESS.sha256_file(destination))
            restored._mmap.close()
            del restored

            with destination.open("r+b") as handle:
                handle.seek(-1, 2)
                original = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([original[0] ^ 0xFF]))
            STRESS.extract_npz_member_atomic(
                source,
                "X.npy",
                destination,
                source_sha256=source_sha,
            )
            repaired = np.load(destination, mmap_mode="r", allow_pickle=False)
            np.testing.assert_array_equal(repaired, values)
            repaired._mmap.close()
            del repaired
            stale = destination.with_name(f".{destination.name}.partial.12345")
            stale.write_bytes(b"interrupted")
            STRESS.extract_npz_member_atomic(
                source,
                "X.npy",
                destination,
                source_sha256=source_sha,
            )
            self.assertFalse(stale.exists())

    def test_disk_backed_perturbations_match_in_memory_reference(self):
        rng = np.random.default_rng(123)
        signals = rng.normal(size=(131, 12, 40)).astype(np.float32)
        specs = [
            {"name": "snr20db", "kind": "additive_noise", "snr_db": 20.0, "seed": 42},
            {"name": "random_3_lead_dropout", "kind": "random_lead_dropout", "n_drop": 3, "seed": 42},
            {"name": "precordial_dropout", "kind": "fixed_lead_dropout", "lead_indices": [6, 7, 8, 9, 10, 11]},
            {"name": "resample_250hz", "kind": "resample_down_up", "source_hz": 500, "target_hz": 250},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(STRESS.robust_helpers, "EXPERIMENTAL_DIR", root / "experimental"):
                for spec in specs:
                    with self.subTest(stress=spec["name"]):
                        expected, _ = STRESS.robust_helpers.perturb_signals(signals, spec)
                        observed, metadata, _path = STRESS.perturb_signals_disk_backed(
                            signals,
                            spec,
                            out_dir=root / "mmap",
                            raw_cache_sha256="a" * 64,
                            source_bundle_sha256="b" * 64,
                        )
                        np.testing.assert_array_equal(observed, expected)
                        self.assertTrue(metadata["disk_backed"])
                        self.assertEqual(
                            metadata["perturbation_mmap_capability"],
                            STRESS.DISK_BACKED_PERTURBATION_CAPABILITY,
                        )
                        observed._mmap.close()
                        del observed

                        resumed, resumed_metadata, _ = STRESS.perturb_signals_disk_backed(
                            signals,
                            spec,
                            out_dir=root / "mmap",
                            raw_cache_sha256="a" * 64,
                            source_bundle_sha256="b" * 64,
                        )
                        np.testing.assert_array_equal(resumed, expected)
                        self.assertEqual(resumed_metadata, metadata)
                        resumed._mmap.close()
                        del resumed

    def test_npz_member_extraction_rejects_source_change_during_stream(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "cache.npz"
            destination = root / "cache_x.npy"
            np.savez_compressed(source, X=values)
            source_sha = STRESS.sha256_file(source)
            real_sha256_file = STRESS.sha256_file

            def changed_source_sha(path):
                path = Path(path)
                if path == source:
                    return "f" * 64
                return real_sha256_file(path)

            with patch.object(STRESS, "sha256_file", side_effect=changed_source_sha):
                with self.assertRaisesRegex(RuntimeError, "source changed during member extraction"):
                    STRESS.extract_npz_member_atomic(
                        source,
                        "X.npy",
                        destination,
                        source_sha256=source_sha,
                    )
            self.assertFalse(destination.exists())

    def test_stale_partial_is_removed_before_generation(self):
        signals = np.zeros((2, 12, 20), dtype=np.float32)
        spec = {"name": "precordial_dropout", "kind": "fixed_lead_dropout", "lead_indices": [6, 7]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "mmap"
            out_dir.mkdir()
            spec_hash = STRESS.robust_helpers.stable_hash(spec)
            output = out_dir / (
                f"{spec['name']}_{spec_hash}_{'a' * 16}_{'b' * 16}_N{len(signals)}.npy"
            )
            stale = output.with_name(f".{output.name}.partial.12345.npy")
            stale.write_bytes(b"interrupted")
            observed, _metadata, _ = STRESS.perturb_signals_disk_backed(
                signals,
                spec,
                out_dir=out_dir,
                raw_cache_sha256="a" * 64,
                source_bundle_sha256="b" * 64,
            )
            self.assertFalse(stale.exists())
            observed._mmap.close()
            del observed
            stale.write_bytes(b"interrupted-after-commit")
            resumed, _resumed_metadata, _ = STRESS.perturb_signals_disk_backed(
                signals,
                spec,
                out_dir=out_dir,
                raw_cache_sha256="a" * 64,
                source_bundle_sha256="b" * 64,
            )
            self.assertFalse(stale.exists())
            resumed._mmap.close()
            del resumed

    def test_startup_sweep_covers_prediction_cache_hit_fast_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / ".snr5db.npy.partial.12345.npy"
            stale.write_bytes(b"interrupted")
            removed = STRESS.cleanup_stale_scratch_partials(root)
            self.assertIn(str(stale), removed)
            self.assertFalse(stale.exists())
        main_source = inspect.getsource(STRESS.main)
        cleanup_index = main_source.index(
            "cleanup_stale_scratch_partials(args.perturbation_mmap_dir)"
        )
        self.assertLess(
            cleanup_index,
            main_source.index("for spec in stresses:", cleanup_index),
        )

    def test_same_size_corrupt_perturbation_is_regenerated(self):
        signals = np.arange(480, dtype=np.float32).reshape(2, 12, 20)
        spec = {"name": "precordial_dropout", "kind": "fixed_lead_dropout", "lead_indices": [6, 7]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observed, metadata, output = STRESS.perturb_signals_disk_backed(
                signals,
                spec,
                out_dir=root,
                raw_cache_sha256="a" * 64,
                source_bundle_sha256="b" * 64,
            )
            expected = np.asarray(observed).copy()
            observed._mmap.close()
            del observed
            with output.open("r+b") as handle:
                handle.seek(-1, 2)
                original = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([original[0] ^ 0xFF]))
            repaired, repaired_metadata, _ = STRESS.perturb_signals_disk_backed(
                signals,
                spec,
                out_dir=root,
                raw_cache_sha256="a" * 64,
                source_bundle_sha256="b" * 64,
            )
            np.testing.assert_array_equal(repaired, expected)
            self.assertEqual(repaired_metadata, metadata)
            repaired._mmap.close()
            del repaired

    def test_perturbation_writer_refuses_concurrent_single_writer_lock(self):
        signals = np.zeros((2, 12, 20), dtype=np.float32)
        spec = {"name": "precordial_dropout", "kind": "fixed_lead_dropout", "lead_indices": [6, 7]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with STRESS.PublishLock(root, run_id="active-test-writer"):
                with self.assertRaisesRegex(RuntimeError, "active or unverifiable publish lock"):
                    STRESS.perturb_signals_disk_backed(
                        signals,
                        spec,
                        out_dir=root,
                        raw_cache_sha256="a" * 64,
                        source_bundle_sha256="b" * 64,
                    )

    def test_selected_comparators_must_share_raw_cache_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for comparator, (name, protocol) in STRESS.BASELINE_CHECKPOINT_CONTRACTS.items():
                (root / name).write_text(
                    '{"protocol": "' + protocol + '", "load_info": {"raw_cache_sha256": "same"}}',
                    encoding="utf-8",
                )
            with patch.object(STRESS, "MANIFEST_DIR", root):
                self.assertEqual(
                    STRESS.expected_raw_cache_sha256(["resnet", "raw_mamba", "transformer"]),
                    "same",
                )
                raw_name = STRESS.BASELINE_CHECKPOINT_CONTRACTS["raw_mamba"][0]
                (root / raw_name).write_text(
                    '{"load_info": {"raw_cache_sha256": "different"}}',
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "raw-cache SHA mismatch"):
                    STRESS.expected_raw_cache_sha256(["resnet", "raw_mamba", "transformer"])


if __name__ == "__main__":
    unittest.main()
