import unittest

import numpy as np

from scripts.revision.common import aggregate_record_probabilities, power_mean


class PowerMeanTests(unittest.TestCase):
    def setUp(self):
        self.probs = np.asarray(
            [
                [0.10, 0.20, 0.90],
                [0.90, 0.80, 0.30],
            ],
            dtype=np.float64,
        )

    def test_q3_matches_generalized_mean_definition(self):
        expected = np.mean(self.probs**3, axis=0) ** (1.0 / 3.0)
        np.testing.assert_allclose(
            power_mean(self.probs, q=3.0, axis=0),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_q1_is_arithmetic_mean(self):
        np.testing.assert_allclose(
            power_mean(self.probs, q=1.0, axis=0),
            np.mean(self.probs, axis=0),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_q0_is_geometric_mean(self):
        expected = np.exp(np.mean(np.log(self.probs), axis=0))
        np.testing.assert_allclose(
            power_mean(self.probs, q=0.0, axis=0),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_q_changes_result(self):
        q2 = power_mean(self.probs, q=2.0, axis=0)
        q8 = power_mean(self.probs, q=8.0, axis=0)
        self.assertFalse(np.allclose(q2, q8))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            power_mean(np.empty((0, 3)), q=3.0, axis=0)

    def test_reaggregation_groups_slices_by_record(self):
        slice_prob = np.asarray(
            [
                [0.10, 0.20],
                [0.90, 0.80],
                [0.25, 0.75],
            ],
            dtype=np.float32,
        )
        record_id = np.asarray([0, 0, 1], dtype=np.int64)
        y_prob, valid, counts = aggregate_record_probabilities(
            slice_prob,
            record_id,
            2,
            q=3.0,
        )
        np.testing.assert_allclose(
            y_prob[0],
            np.mean(slice_prob[:2] ** 3, axis=0) ** (1.0 / 3.0),
            rtol=1e-6,
        )
        np.testing.assert_allclose(y_prob[1], slice_prob[2], rtol=1e-6)
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_array_equal(counts, [2, 1])


if __name__ == "__main__":
    unittest.main()
