import unittest

import numpy as np

from configs.config import NUM_CLASSES, SEQ_LEN
from src.data_loader import validate_clean_cache_arrays
from src.provenance import record_order_fingerprint


class CleanCacheContractTests(unittest.TestCase):
    def test_valid_cache_returns_record_order_fingerprint(self):
        signals = np.zeros((2, 12, SEQ_LEN), dtype=np.float32)
        labels = np.zeros((2, NUM_CLASSES), dtype=np.float32)
        labels[0, 0] = 1
        labels[1, 1] = 1
        amplitude = np.zeros((2, 5), dtype=np.float32)
        subjects = np.asarray(["record-a", "record-b"])

        fingerprint = validate_clean_cache_arrays(
            signals,
            labels,
            amplitude,
            subjects,
        )
        self.assertEqual(fingerprint, record_order_fingerprint(subjects))

    def test_duplicate_record_ids_are_rejected(self):
        signals = np.zeros((2, 12, SEQ_LEN), dtype=np.float32)
        labels = np.zeros((2, NUM_CLASSES), dtype=np.float32)
        amplitude = np.zeros((2, 5), dtype=np.float32)
        subjects = np.asarray(["record-a", "record-a"])

        with self.assertRaisesRegex(ValueError, "not unique"):
            validate_clean_cache_arrays(
                signals,
                labels,
                amplitude,
                subjects,
            )


if __name__ == "__main__":
    unittest.main()
