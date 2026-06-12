import unittest

import numpy as np

from src.training_data import (
    LazyECGSliceDataset,
    audit_fold_splits,
    build_slice_index,
)


class TrainingDataTests(unittest.TestCase):
    def test_lazy_dataset_preserves_slice_order_and_values(self):
        signals = np.arange(3 * 2 * 10, dtype=np.float32).reshape(3, 2, 10)
        hydra = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
        hrv = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
        labels = np.eye(3, dtype=np.float32)

        record_ids, starts, positions, skipped = build_slice_index(
            np.asarray([2, 0]),
            signals,
            slice_length=4,
            slice_stride=3,
            max_slices_per_record=2,
        )
        self.assertEqual(skipped, 0)
        np.testing.assert_array_equal(record_ids, [2, 2, 0, 0])
        np.testing.assert_array_equal(starts, [0, 3, 0, 3])
        np.testing.assert_allclose(positions, [0.2, 0.5, 0.2, 0.5])

        dataset = LazyECGSliceDataset(
            signals,
            hydra,
            hrv,
            labels,
            record_ids,
            starts,
            positions,
            slice_length=4,
        )
        signal, hydra_row, hrv_row, label, record_id, position = dataset[1]
        np.testing.assert_array_equal(signal.numpy(), signals[2, :, 3:7])
        np.testing.assert_array_equal(hydra_row.numpy(), hydra[2])
        np.testing.assert_array_equal(hrv_row.numpy(), hrv[2])
        np.testing.assert_array_equal(label.numpy(), labels[2])
        self.assertEqual(record_id, 2)
        self.assertAlmostEqual(float(position), 0.5)

    def test_short_records_are_reported_as_skipped(self):
        signals = np.zeros((2, 1, 3), dtype=np.float32)
        record_ids, starts, positions, skipped = build_slice_index(
            np.asarray([0, 1]),
            signals,
            slice_length=4,
            slice_stride=2,
            max_slices_per_record=3,
        )
        self.assertEqual(skipped, 2)
        self.assertEqual(len(record_ids), 0)
        self.assertEqual(len(starts), 0)
        self.assertEqual(len(positions), 0)

    def test_fold_audit_requires_subject_isolation_and_exact_coverage(self):
        folds = [
            {"tr_idx": np.asarray([2, 3]), "va_idx": np.asarray([0, 1])},
            {"tr_idx": np.asarray([0, 1]), "va_idx": np.asarray([2, 3])},
        ]
        audit = audit_fold_splits(
            folds,
            np.asarray(["a", "b", "c", "d"]),
            n_records=4,
        )
        self.assertTrue(audit["all_records_covered_once"])
        self.assertTrue(audit["subject_isolation"])

        leaking_folds = [
            {"tr_idx": np.asarray([1, 2, 3]), "va_idx": np.asarray([0])},
            {"tr_idx": np.asarray([0]), "va_idx": np.asarray([1, 2, 3])},
        ]
        with self.assertRaisesRegex(ValueError, "subject overlaps"):
            audit_fold_splits(
                leaking_folds,
                np.asarray(["same", "same", "c", "d"]),
                n_records=4,
            )

    def test_fold_audit_rejects_duplicate_validation_coverage(self):
        folds = [
            {"tr_idx": np.asarray([2, 3]), "va_idx": np.asarray([0, 1])},
            {"tr_idx": np.asarray([0, 3]), "va_idx": np.asarray([1, 2])},
        ]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            audit_fold_splits(
                folds,
                np.asarray(["a", "b", "c", "d"]),
                n_records=4,
            )


if __name__ == "__main__":
    unittest.main()
