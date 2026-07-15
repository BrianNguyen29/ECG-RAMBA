"""Shared helpers for Colab-based reviewer revision experiments.

These helpers are intentionally dependency-light and are safe to import from
small runner scripts. Heavy model/data loading stays in the experiment scripts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from src.aggregation import (  # noqa: E402
    POWER_MEAN_IMPLEMENTATION,
    aggregate_record_probabilities,
    power_mean,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_DIR = PROJECT_ROOT / "reports" / "revision"
FIGURE_DIR = REVISION_DIR / "figures"
LOG_DIR = REVISION_DIR / "logs"
MANIFEST_DIR = REVISION_DIR / "manifests"
METRIC_DIR = REVISION_DIR / "metrics"
PREDICTION_DIR = REVISION_DIR / "predictions"
TABLE_DIR = REVISION_DIR / "tables"
EXPERIMENTAL_DIR = REVISION_DIR / "experimental"

CACHE_SCHEMA_VERSION = 2


CURRENT_HRV36_SCHEMA = [
    ("rr_mean_ms", 0),
    ("rr_std_ms", 1),
    ("rr_median_ms", 2),
    ("rr_min_ms", 3),
    ("rr_max_ms", 4),
    ("reserved_zero_05", 5),
    ("reserved_zero_06", 6),
    ("reserved_zero_07", 7),
    ("reserved_zero_08", 8),
    ("reserved_zero_09", 9),
    ("reserved_zero_10", 10),
    ("reserved_zero_11", 11),
    ("reserved_zero_12", 12),
    ("reserved_zero_13", 13),
    ("reserved_zero_14", 14),
    ("reserved_zero_15", 15),
    ("reserved_zero_16", 16),
    ("reserved_zero_17", 17),
    ("reserved_zero_18", 18),
    ("reserved_zero_19", 19),
    ("reserved_zero_20", 20),
    ("reserved_zero_21", 21),
    ("reserved_zero_22", 22),
    ("reserved_zero_23", 23),
    ("reserved_zero_24", 24),
    ("amp_mean", 25),
    ("amp_min", 26),
    ("amp_max", 27),
    ("amp_limb_mean", 28),
    ("amp_precordial_mean", 29),
    ("global_z_mean", 30),
    ("global_z_std", 31),
    ("global_z_energy", 32),
    ("global_z_kurtosis", 33),
    ("global_z_skew", 34),
    ("global_z_p95", 35),
]


PTB_SUPERCLASS_MAPPING = {
    "NORM": {
        "codes": ["SNR", "SB", "STach", "SA"],
        "snomed": ["426783006", "426177001", "427084000", "427393009"],
        "rationale": "Normal sinus variants",
    },
    "MI": {
        "codes": ["QAb"],
        "snomed": ["164917005"],
        "rationale": "Q-wave abnormality proxy for infarction-related pattern",
    },
    "STTC": {
        "codes": ["TInv", "TAb", "LQT"],
        "snomed": ["59931005", "164934002", "111975006"],
        "rationale": "Repolarization abnormalities",
    },
    "CD": {
        "codes": ["LBBB", "RBBB", "CRBBB", "IRBBB", "IAVB", "LAnFB", "NSIVCB"],
        "snomed": [
            "164909002",
            "59118001",
            "713427006",
            "713426002",
            "270492004",
            "445118002",
            "698252002",
        ],
        "rationale": "Bundle branch blocks and conduction delays",
    },
}


def ensure_revision_dirs() -> None:
    for path in [
        REVISION_DIR,
        FIGURE_DIR,
        LOG_DIR,
        MANIFEST_DIR,
        METRIC_DIR,
        PREDICTION_DIR,
        TABLE_DIR,
        EXPERIMENTAL_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: os.PathLike[str] | str, payload: dict) -> None:
    """Persist JSON atomically so an interrupted job cannot leave a reusable partial file."""

    save_json_atomic(path, payload)


def save_json_atomic(path: os.PathLike[str] | str, payload: dict) -> None:
    """Write JSON through a same-directory temporary file before replacing it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_npz_compressed_atomic(path: os.PathLike[str] | str, **arrays: np.ndarray) -> None:
    """Atomically persist a compressed NPZ cache so interrupted writes are not reusable."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.partial.npz")
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_csv(path: os.PathLike[str] | str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        if not rows:
            with tmp_path.open("w", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        else:
            with tmp_path.open("w", newline="", encoding="utf-8") as f:
                fieldnames = list(dict.fromkeys(key for row in rows for key in row))
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def npz_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error for one binary label."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return ece


def mce_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Maximum calibration error for one binary label."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        gaps.append(abs(acc - conf))
    return float(np.max(gaps)) if gaps else math.nan


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> dict:
    """Macro calibration summary for multi-label predictions."""
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_prob.shape}")

    eces, mces, briers = [], [], []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        eces.append(ece_binary(y_true[:, c], y_prob[:, c], n_bins=n_bins))
        mces.append(mce_binary(y_true[:, c], y_prob[:, c], n_bins=n_bins))
        briers.append(brier_score_loss(y_true[:, c], y_prob[:, c]))

    return {
        "ece_macro": float(np.mean(eces)) if eces else math.nan,
        "ece_max": float(np.max(eces)) if eces else math.nan,
        "mce_macro": float(np.mean(mces)) if mces else math.nan,
        "mce_max": float(np.max(mces)) if mces else math.nan,
        "brier_macro": float(np.mean(briers)) if briers else math.nan,
        "n_classes_evaluated": int(len(eces)),
    }


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Fixed-threshold multi-label metrics plus macro ROC-AUC/PR-AUC."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(np.float32)

    roc_scores, pr_scores = [], []
    sensitivity_scores, specificity_scores, ppv_scores, npv_scores = [], [], [], []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        roc_scores.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
        pr_scores.append(average_precision_score(y_true[:, c], y_prob[:, c]))
        yt = y_true[:, c].astype(bool)
        yp = y_pred[:, c].astype(bool)
        tp = float(np.sum(yt & yp))
        tn = float(np.sum(~yt & ~yp))
        fp = float(np.sum(~yt & yp))
        fn = float(np.sum(yt & ~yp))
        if tp + fn > 0:
            sensitivity_scores.append(tp / (tp + fn))
        if tn + fp > 0:
            specificity_scores.append(tn / (tn + fp))
        if tp + fp > 0:
            ppv_scores.append(tp / (tp + fp))
        if tn + fn > 0:
            npv_scores.append(tn / (tn + fn))

    if y_true.shape[1] == 1:
        # sklearn treats an (N, 1) target as binary rather than multilabel-indicator.
        # Flatten explicitly so a one-label mapped task retains positive-label
        # multilabel semantics instead of averaging the negative and positive classes.
        yt_metric = y_true[:, 0]
        yp_metric = y_pred[:, 0]
        f1_macro = f1_micro = f1_score(yt_metric, yp_metric, average="binary", zero_division=0)
        precision_macro = precision_score(yt_metric, yp_metric, average="binary", zero_division=0)
        recall_macro = recall_score(yt_metric, yp_metric, average="binary", zero_division=0)
    else:
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "sensitivity_macro": float(np.mean(sensitivity_scores)) if sensitivity_scores else math.nan,
        "specificity_macro": float(np.mean(specificity_scores)) if specificity_scores else math.nan,
        "ppv_macro": float(np.mean(ppv_scores)) if ppv_scores else math.nan,
        "npv_macro": float(np.mean(npv_scores)) if npv_scores else math.nan,
        "roc_auc_macro": float(np.mean(roc_scores)) if roc_scores else math.nan,
        "pr_auc_macro": float(np.mean(pr_scores)) if pr_scores else math.nan,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Record-level bootstrap CI for a scalar metric function."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            value = metric_fn(y_true[idx], y_prob[idx])
            if np.isfinite(value):
                values.append(float(value))
        except ValueError:
            continue
    if not values:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n_boot_valid": 0}
    lo, hi = np.quantile(values, [alpha / 2, 1.0 - alpha / 2])
    return {
        "mean": float(np.mean(values)),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot_valid": int(len(values)),
    }


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap complete independent groups rather than individual rows."""

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups).astype(str)
    if y_true.shape != y_prob.shape or len(groups) != len(y_true):
        raise ValueError(
            f"Cluster bootstrap shape mismatch: y_true={y_true.shape}, "
            f"y_prob={y_prob.shape}, groups={groups.shape}"
        )
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2:
        raise ValueError("Cluster bootstrap requires at least two independent groups.")
    members = [np.where(inverse == idx)[0] for idx in range(len(unique_groups))]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(n_boot)):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        row_idx = np.concatenate([members[int(group_idx)] for group_idx in sampled])
        try:
            value = float(metric_fn(y_true[row_idx], y_prob[row_idx]))
        except (ValueError, RuntimeError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n_boot_valid": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "lo": float(np.quantile(array, alpha / 2)),
        "hi": float(np.quantile(array, 1 - alpha / 2)),
        "n_boot_valid": int(len(array)),
        "n_groups": int(len(unique_groups)),
        "sample_unit": "group",
    }


def paired_cluster_bootstrap_delta(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paired group bootstrap for ``metric(a) - metric(b)``."""

    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    groups = np.asarray(groups).astype(str)
    if y_true.shape != y_prob_a.shape or y_true.shape != y_prob_b.shape:
        raise ValueError("Paired cluster bootstrap prediction shapes differ.")
    if len(groups) != len(y_true):
        raise ValueError("Paired cluster bootstrap group length differs from predictions.")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2:
        raise ValueError("Paired cluster bootstrap requires at least two independent groups.")
    members = [np.where(inverse == idx)[0] for idx in range(len(unique_groups))]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(n_boot)):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        row_idx = np.concatenate([members[int(group_idx)] for group_idx in sampled])
        try:
            value = float(metric_fn(y_true[row_idx], y_prob_a[row_idx])) - float(
                metric_fn(y_true[row_idx], y_prob_b[row_idx])
            )
        except (ValueError, RuntimeError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n_boot_valid": 0}
    array = np.asarray(values, dtype=np.float64)
    point = float(metric_fn(y_true, y_prob_a)) - float(metric_fn(y_true, y_prob_b))
    return {
        "point_delta_a_minus_b": point,
        "mean": float(np.mean(array)),
        "lo": float(np.quantile(array, alpha / 2)),
        "hi": float(np.quantile(array, 1 - alpha / 2)),
        "n_boot_valid": int(len(array)),
        "n_groups": int(len(unique_groups)),
        "sample_unit": "group",
    }


def balanced_group_train_test_split(
    y_true: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
    seed: int,
    n_candidates: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Choose a leakage-safe group split with stable label coverage.

    Candidate splits are random group permutations. Selection uses labels only
    to avoid losing globally evaluable classes and then minimize train/test
    prevalence drift. No test outcomes are used for model or threshold tuning.
    """

    y_true = np.asarray(y_true)
    groups = np.asarray(groups).astype(str)
    if y_true.ndim != 2 or len(y_true) != len(groups):
        raise ValueError(f"Group split shape mismatch: y_true={y_true.shape}, groups={groups.shape}")
    if not 0 < float(test_fraction) < 1:
        raise ValueError("test_fraction must be in (0,1)")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Group split requires at least two independent groups")
    n_test = min(max(int(round(len(unique_groups) * float(test_fraction))), 1), len(unique_groups) - 1)
    globally_evaluable = np.logical_and(y_true.sum(axis=0) > 0, y_true.sum(axis=0) < len(y_true))
    rng = np.random.default_rng(int(seed))
    best: tuple[tuple[int, float, float], np.ndarray, np.ndarray, dict] | None = None
    for candidate in range(max(int(n_candidates), 1)):
        order = rng.permutation(unique_groups)
        test_groups = order[:n_test]
        train_groups = order[n_test:]
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        train_y = y_true[train_mask]
        test_y = y_true[test_mask]
        train_evaluable = np.logical_and(train_y.sum(axis=0) > 0, train_y.sum(axis=0) < len(train_y))
        test_evaluable = np.logical_and(test_y.sum(axis=0) > 0, test_y.sum(axis=0) < len(test_y))
        coverage_failures = int(np.sum(globally_evaluable & ~(train_evaluable & test_evaluable)))
        prevalence_drift = float(np.mean(np.abs(train_y.mean(axis=0) - test_y.mean(axis=0))))
        row_fraction_error = abs(float(np.mean(test_mask)) - float(test_fraction))
        score = (coverage_failures, prevalence_drift, row_fraction_error)
        audit = {
            "candidate_index": candidate,
            "n_candidates": max(int(n_candidates), 1),
            "coverage_failures": coverage_failures,
            "prevalence_drift_mean_abs": prevalence_drift,
            "row_test_fraction": float(np.mean(test_mask)),
            "target_test_fraction": float(test_fraction),
            "train_groups": int(len(train_groups)),
            "test_groups": int(len(test_groups)),
            "split_policy": "best_of_random_group_candidates_label_coverage_then_prevalence",
        }
        if best is None or score < best[0]:
            best = (score, train_groups, test_groups, audit)
    assert best is not None
    return best[1], best[2], best[3]


def macro_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        scores.append(average_precision_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(scores)) if scores else math.nan


def macro_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(scores)) if scores else math.nan
