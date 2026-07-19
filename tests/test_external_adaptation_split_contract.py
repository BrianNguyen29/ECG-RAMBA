from __future__ import annotations

import unittest

import numpy as np

from scripts.revision.common import hash_group_train_test_split


class ExternalAdaptationSplitContractTests(unittest.TestCase):
    def test_hash_split_is_invariant_to_held_out_label_permutation(self) -> None:
        groups = np.asarray([f"patient-{index:04d}" for index in range(200)])
        labels = np.random.default_rng(42).integers(0, 2, size=(200, 4))
        permuted_labels = labels[np.random.default_rng(43).permutation(len(labels))]

        train_a, test_a, audit_a = hash_group_train_test_split(groups, 0.5, 42)
        # Deliberately touch two different label arrays to document that neither
        # participates in the split API or its assignment contract.
        self.assertFalse(np.array_equal(labels, permuted_labels))
        train_b, test_b, audit_b = hash_group_train_test_split(groups, 0.5, 42)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        self.assertEqual(audit_a["assignment_sha256"], audit_b["assignment_sha256"])
        self.assertTrue(audit_a["label_independent"])
        self.assertEqual(set(train_a) & set(test_a), set())
        self.assertEqual(set(train_a) | set(test_a), set(groups))

    def test_seed_changes_assignment_but_remains_deterministic(self) -> None:
        groups = np.asarray([f"record-{index}" for index in range(50)])
        train_a, test_a, audit_a = hash_group_train_test_split(groups, 0.4, 7)
        train_b, test_b, audit_b = hash_group_train_test_split(groups, 0.4, 8)
        self.assertNotEqual(audit_a["assignment_sha256"], audit_b["assignment_sha256"])
        self.assertFalse(np.array_equal(test_a, test_b))
        self.assertEqual(len(test_a), len(test_b))
        self.assertEqual(len(train_a), len(train_b))


if __name__ == "__main__":
    unittest.main()
