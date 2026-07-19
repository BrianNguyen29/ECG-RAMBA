from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src import data_loader, features
from src.provenance import ndarray_sha256, save_npz_atomic


class _FakeRocket:
    calls = 0

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def cpu(self):
        return self

    def eval(self):
        return self

    def __call__(self, tensor):
        type(self).calls += 1
        return torch.zeros((len(tensor), 20000), dtype=torch.float32)


class CacheProvenanceContractTests(unittest.TestCase):
    @staticmethod
    def _prediction_generator():
        path = Path(__file__).resolve().parents[1] / "scripts" / "revision" / "01_generate_predictions.py"
        spec = importlib.util.spec_from_file_location("_cache_contract_generator", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_ndarray_hash_changes_when_signal_changes_but_ids_do_not(self):
        first = np.zeros((2, 12, 16), dtype=np.float32)
        second = first.copy()
        second[1, 3, 7] = 1.0
        self.assertNotEqual(ndarray_sha256(first), ndarray_sha256(second))

    def test_fixed_transform_cache_does_not_reuse_same_ids_different_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cache = features.PATHS["cache_dir"]
            features.PATHS["cache_dir"] = tmp
            _FakeRocket.calls = 0
            ids = np.asarray(["A", "B"])
            first = np.zeros((2, 12, 16), dtype=np.float32)
            second = first.copy()
            second[0, 0, 0] = 2.0
            try:
                with mock.patch.object(features, "MiniRocketNative", _FakeRocket), mock.patch(
                    "builtins.print"
                ):
                    features.generate_raw_rocket_cache(first, ids)
                    features.generate_raw_rocket_cache(second, ids)
                    features.generate_raw_rocket_cache(first, ids)
                self.assertEqual(_FakeRocket.calls, 2)
                caches = list(Path(tmp).glob("rocket_raw_*.npz"))
                self.assertEqual(len(caches), 2)
            finally:
                features.PATHS["cache_dir"] = old_cache

    def test_clean_cache_contract_binds_archive_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "chapman.zip"
            archive.write_bytes(b"first archive")
            first = data_loader.clean_cache_source_contract(str(archive))
            archive.write_bytes(b"second archive")
            second = data_loader.clean_cache_source_contract(str(archive))
            self.assertNotEqual(first["archive_sha256"], second["archive_sha256"])
            self.assertEqual(first["preprocessing_source_sha256"], second["preprocessing_source_sha256"])

    def test_atomic_npz_never_exposes_partial_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "cache.npz"
            save_npz_atomic(destination, values=np.arange(5))
            self.assertTrue(destination.is_file())
            self.assertEqual(list(Path(tmp).glob("*.partial.*")), [])
            with np.load(destination, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["values"], np.arange(5))

    def test_fold_cache_key_changes_with_input_content_contract(self):
        generator = self._prediction_generator()
        tr_idx = np.asarray([0, 1], dtype=np.int64)
        va_idx = np.asarray([2], dtype=np.int64)
        first_base = {"contract_sha256": "a" * 64}
        second_base = {"contract_sha256": "b" * 64}
        first = generator.scoped_cache_contract(
            first_base,
            artifact_kind="oof_fold_prediction",
            fold_num=1,
            tr_idx=tr_idx,
            va_idx=va_idx,
            source_config_hash="config",
            checkpoint_sha256="checkpoint",
        )
        second = generator.scoped_cache_contract(
            second_base,
            artifact_kind="oof_fold_prediction",
            fold_num=1,
            tr_idx=tr_idx,
            va_idx=va_idx,
            source_config_hash="config",
            checkpoint_sha256="checkpoint",
        )
        self.assertNotEqual(first["contract_sha256"], second["contract_sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            first_path = generator.fold_prediction_cache_path(
                1, "final_ema", "c" * 64, Path(tmp), first
            )
            second_path = generator.fold_prediction_cache_path(
                1, "final_ema", "c" * 64, Path(tmp), second
            )
        self.assertNotEqual(first_path.name, second_path.name)


if __name__ == "__main__":
    unittest.main()
