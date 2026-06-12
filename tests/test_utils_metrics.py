import unittest

import numpy as np

from src.utils import compute_metrics


class UtilityMetricTests(unittest.TestCase):
    def test_fixed_threshold_includes_probability_equal_to_threshold(self):
        y_true = np.asarray([[1.0], [0.0]], dtype=np.float32)
        y_prob = np.asarray([[0.5], [0.1]], dtype=np.float32)
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        self.assertEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(metrics["precision_macro"], 1.0)
        self.assertEqual(metrics["recall_macro"], 1.0)


if __name__ == "__main__":
    unittest.main()
