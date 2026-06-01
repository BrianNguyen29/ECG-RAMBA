"""Shared helpers for Colab-based reviewer revision experiments.

These helpers are intentionally dependency-light and are safe to import from
small runner scripts. Heavy model/data loading stays in the experiment scripts.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_DIR = PROJECT_ROOT / "reports" / "revision"
FIGURE_DIR = REVISION_DIR / "figures"
LOG_DIR = REVISION_DIR / "logs"
MANIFEST_DIR = REVISION_DIR / "manifests"
METRIC_DIR = REVISION_DIR / "metrics"
PREDICTION_DIR = REVISION_DIR / "predictions"
TABLE_DIR = REVISION_DIR / "tables"


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
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: os.PathLike[str] | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def save_csv(path: os.PathLike[str] | str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def power_mean(probs: np.ndarray, q: float = 3.0, axis: int = 0, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable Power Mean aggregation in probability space."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0 - eps)
    return np.exp(np.mean(q * np.log(probs), axis=axis) / q).astype(np.float32)


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

    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
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
