"""Memory-efficient fold datasets for ECG-RAMBA training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.provenance import record_order_fingerprint


def audit_fold_splits(
    folds: list[dict[str, np.ndarray]],
    subjects: np.ndarray,
    *,
    n_records: int,
) -> dict:
    """Validate record coverage and subject isolation for cross-validation."""
    subjects = np.asarray(subjects)
    if len(subjects) != n_records:
        raise ValueError(
            f"subjects length {len(subjects)} does not match n_records={n_records}"
        )

    validation_coverage = np.zeros(n_records, dtype=np.int16)
    fold_rows = []
    for fold_num, split in enumerate(folds, start=1):
        train_indices = np.asarray(split["tr_idx"], dtype=np.int64)
        val_indices = np.asarray(split["va_idx"], dtype=np.int64)
        if np.any(train_indices < 0) or np.any(train_indices >= n_records):
            raise ValueError(f"Fold {fold_num} has out-of-range training indices")
        if np.any(val_indices < 0) or np.any(val_indices >= n_records):
            raise ValueError(f"Fold {fold_num} has out-of-range validation indices")

        record_overlap = np.intersect1d(train_indices, val_indices)
        if len(record_overlap):
            raise ValueError(
                f"Fold {fold_num} has {len(record_overlap)} train/validation record overlaps"
            )
        subject_overlap = np.intersect1d(
            np.unique(subjects[train_indices]),
            np.unique(subjects[val_indices]),
        )
        if len(subject_overlap):
            raise ValueError(
                f"Fold {fold_num} has {len(subject_overlap)} train/validation subject overlaps"
            )

        validation_coverage[val_indices] += 1
        fold_rows.append(
            {
                "fold": fold_num,
                "train_records": int(len(train_indices)),
                "validation_records": int(len(val_indices)),
                "train_subjects": int(len(np.unique(subjects[train_indices]))),
                "validation_subjects": int(len(np.unique(subjects[val_indices]))),
                "record_overlap": 0,
                "subject_overlap": 0,
            }
        )

    missing = np.flatnonzero(validation_coverage == 0)
    duplicated = np.flatnonzero(validation_coverage > 1)
    if len(missing) or len(duplicated):
        raise ValueError(
            "Validation folds must cover every record exactly once: "
            f"missing={len(missing)}, duplicated={len(duplicated)}"
        )

    return {
        "n_records": int(n_records),
        "n_folds": int(len(folds)),
        "record_order_fingerprint": record_order_fingerprint(subjects),
        "validation_coverage_min": int(validation_coverage.min()),
        "validation_coverage_max": int(validation_coverage.max()),
        "all_records_covered_once": True,
        "subject_isolation": True,
        "folds": fold_rows,
    }


def build_slice_index(
    record_indices: np.ndarray,
    signals: np.ndarray,
    *,
    slice_length: int,
    slice_stride: int,
    max_slices_per_record: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build record/start metadata without copying ECG or feature arrays."""
    record_ids: list[int] = []
    starts: list[int] = []
    positions: list[float] = []
    skipped = 0

    for record_id in np.asarray(record_indices, dtype=np.int64):
        signal_length = int(signals[int(record_id)].shape[-1])
        if signal_length < slice_length:
            skipped += 1
            continue
        count = 0
        for start in range(0, signal_length - slice_length + 1, slice_stride):
            record_ids.append(int(record_id))
            starts.append(start)
            positions.append((start + slice_length / 2) / signal_length)
            count += 1
            if count >= max_slices_per_record:
                break

    return (
        np.asarray(record_ids, dtype=np.int64),
        np.asarray(starts, dtype=np.int32),
        np.asarray(positions, dtype=np.float32),
        skipped,
    )


class LazyECGSliceDataset(Dataset):
    """Materialize one ECG slice at a time from record-level arrays."""

    def __init__(
        self,
        signals: np.ndarray,
        hydra_by_record: np.ndarray,
        hrv_by_record: np.ndarray,
        labels: np.ndarray,
        record_ids: np.ndarray,
        starts: np.ndarray,
        positions: np.ndarray,
        *,
        slice_length: int,
    ):
        if not (len(record_ids) == len(starts) == len(positions)):
            raise ValueError("record_ids, starts, and positions must have equal length")
        self.signals = signals
        self.hydra_by_record = hydra_by_record
        self.hrv_by_record = hrv_by_record
        self.labels = labels
        self.record_ids = np.asarray(record_ids, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int32)
        self.positions = np.asarray(positions, dtype=np.float32)
        self.slice_length = int(slice_length)

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int):
        record_id = int(self.record_ids[index])
        start = int(self.starts[index])
        stop = start + self.slice_length

        # Keep the NumPy view here. The default collate step creates the
        # contiguous batch once, avoiding a second full slice copy per item.
        signal = self.signals[record_id, :, start:stop]
        return (
            torch.from_numpy(signal).float(),
            torch.from_numpy(self.hydra_by_record[record_id]).float(),
            torch.from_numpy(self.hrv_by_record[record_id]).float(),
            torch.from_numpy(self.labels[record_id]).float(),
            record_id,
            self.positions[index],
        )
