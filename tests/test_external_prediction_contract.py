import importlib
import unittest

import numpy as np

from configs.config import CLASS_TO_IDX, CLASSES


external = importlib.import_module("scripts.revision.03_generate_external_predictions")


class ExternalPredictionContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
