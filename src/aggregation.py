"""Shared record-level aggregation helpers for ECG-RAMBA.

Training, OOF export, calibration, and pooling sensitivity must use the same
implementation to keep manuscript metrics reproducible.
"""

from __future__ import annotations

import numpy as np


POWER_MEAN_IMPLEMENTATION = "power_mean_v2"


def power_mean(probs: np.ndarray, q: float = 3.0, axis: int = 0, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable generalized mean in probability space.

    q=1 is the arithmetic mean, q=0 is the geometric mean, and larger
    positive q values place more weight on high-confidence slices.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        raise ValueError("power_mean requires at least one probability value")
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, got {q}")
    probs = np.clip(probs, eps, 1.0 - eps)
    log_probs = np.log(probs)
    if abs(q) < 1e-12:
        return np.exp(np.mean(log_probs, axis=axis)).astype(np.float32)

    scaled = q * log_probs
    max_scaled = np.max(scaled, axis=axis, keepdims=True)
    log_mean_power = (
        np.squeeze(max_scaled, axis=axis)
        + np.log(np.mean(np.exp(scaled - max_scaled), axis=axis))
    )
    return np.exp(log_mean_power / q).astype(np.float32)


def aggregate_record_probabilities(
    slice_prob: np.ndarray,
    slice_record_id: np.ndarray,
    n_records: int,
    q: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate slice probabilities and return probabilities, validity, counts."""
    slice_prob = np.asarray(slice_prob)
    slice_record_id = np.asarray(slice_record_id, dtype=np.int64)
    if slice_prob.ndim != 2 or len(slice_prob) != len(slice_record_id):
        raise ValueError("slice_prob and slice_record_id have incompatible shapes")
    n_classes = slice_prob.shape[1]
    y_prob = np.zeros((n_records, n_classes), dtype=np.float32)
    valid_mask = np.zeros(n_records, dtype=bool)
    slice_count = np.zeros(n_records, dtype=np.int16)
    order = np.argsort(slice_record_id, kind="stable")
    sorted_ids = slice_record_id[order]
    sorted_probs = slice_prob[order].astype(np.float32, copy=False)
    unique_ids, starts, counts = np.unique(
        sorted_ids,
        return_index=True,
        return_counts=True,
    )
    for record_id, start, count in zip(unique_ids, starts, counts):
        rid = int(record_id)
        if rid < 0 or rid >= n_records:
            raise ValueError(f"Slice record id {rid} is outside [0, {n_records})")
        y_prob[rid] = power_mean(sorted_probs[start : start + count], q=q, axis=0)
        valid_mask[rid] = True
        slice_count[rid] = int(count)
    return y_prob, valid_mask, slice_count
