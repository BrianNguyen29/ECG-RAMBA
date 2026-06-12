import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import src.features as features


class _FakeMiniRocket:
    def __init__(self, *args, **kwargs):
        pass

    def cpu(self):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        base = torch.linspace(
            -1.0,
            1.0,
            20000,
            dtype=torch.float32,
        )
        return base.repeat(len(batch), 1)


class FeatureCacheContractTests(unittest.TestCase):
    def test_rocket_cold_and_warm_cache_values_match(self):
        signals = np.zeros((2, 12, 16), dtype=np.float32)
        record_ids = np.asarray(["record-a", "record-b"])
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(features.PATHS, {"cache_dir": tmp}),
                mock.patch.object(features, "MiniRocketNative", _FakeMiniRocket),
                mock.patch.object(features, "tqdm", side_effect=lambda values, **kwargs: values),
                mock.patch("builtins.print"),
            ):
                cold = features.generate_raw_rocket_cache(signals, record_ids)
                warm = features.generate_raw_rocket_cache(signals, record_ids)

            np.testing.assert_array_equal(cold, warm)
            self.assertEqual(cold.dtype, np.float32)
            cache_path = next(Path(tmp).glob("rocket_raw_*.npz"))
            with np.load(cache_path, allow_pickle=False) as payload:
                self.assertEqual(str(payload["storage_dtype"].item()), "float16")
                self.assertEqual(
                    str(payload["quantization_contract"].item()),
                    "float16_storage_roundtrip_v1",
                )

    def test_hrv_cold_and_warm_cache_values_match(self):
        signals = np.zeros((2, 12, 16), dtype=np.float32)
        raw_amplitude = np.zeros((2, 5), dtype=np.float32)
        record_ids = np.asarray(["record-a", "record-b"])
        hrv = np.linspace(0.0, 1.0, 25, dtype=np.float32)
        amplitude = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        global_stats = np.linspace(0.0, 1.0, 6, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(features.PATHS, {"cache_dir": tmp}),
                mock.patch.object(features, "extract_hrv_features", return_value=hrv),
                mock.patch.object(features, "extract_amplitude_features", return_value=amplitude),
                mock.patch.object(features, "extract_global_record_stats", return_value=global_stats),
                mock.patch.object(features, "tqdm", side_effect=lambda values, **kwargs: values),
                mock.patch("builtins.print"),
            ):
                cold = features.generate_hrv_cache(
                    signals,
                    raw_amplitude,
                    record_ids,
                )
                warm = features.generate_hrv_cache(
                    signals,
                    raw_amplitude,
                    record_ids,
                )

            np.testing.assert_array_equal(cold, warm)
            self.assertEqual(cold.dtype, np.float32)
            cache_path = next(Path(tmp).glob("hrv36_*.npz"))
            with np.load(cache_path, allow_pickle=False) as payload:
                self.assertEqual(str(payload["storage_dtype"].item()), "float16")
                self.assertEqual(
                    str(payload["quantization_contract"].item()),
                    "float16_storage_roundtrip_v1",
                )


if __name__ == "__main__":
    unittest.main()
