"""Run actual HRV-only and HRV-domain analyses for the revision package.

This script intentionally avoids model checkpoint inference. It uses the
existing Chapman HRV36 feature contract, the same OOF fold split, and light
classical models to answer two reviewer-facing questions:

1. Does HRV36 carry label signal under the frozen Chapman OOF split?
2. Does HRV36 encode dataset/source identity when compared with available
   external HRV36 feature caches?

Outputs are written under reports/revision and are safe for notebook 05 to
inventory and mirror. Robustness stress tests remain a separate, heavier
inference task and are not executed here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG, CONFIG_HASH, PATHS  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    bootstrap_ci,
    calibration_summary,
    ensure_revision_dirs,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_json,
    sha256_file,
)


PROTOCOL = "hrv36_logistic_regression_same_folds_threshold_0.5"
DOMAIN_PROTOCOL = "hrv36_domain_logistic_regression_stratified_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=PREDICTION_DIR / "oof_full_predictions.npz",
        help="Frozen OOF NPZ used for y_true, class order, and fold_id.",
    )
    parser.add_argument(
        "--chapman-hrv-cache",
        type=Path,
        default=None,
        help="Optional explicit HRV36 NPZ path. Defaults to cache_dir/hrv36_N{N}_C12_L5000.npz.",
    )
    parser.add_argument(
        "--allow-raw-chapman-fallback",
        action="store_true",
        help="Allow loading raw Chapman signals to generate HRV36 if cache is missing. Off by default to avoid Colab OOM.",
    )
    parser.add_argument("--domain-max-per-domain", type=int, default=2000)
    parser.add_argument("--domain-n-splits", type=int, default=5)
    parser.add_argument(
        "--skip-domain-classifier",
        action="store_true",
        help="Only run Chapman HRV-only OOF baseline.",
    )
    parser.add_argument(
        "--ptbxl-hrv",
        type=Path,
        default=Path(PATHS["model_dir"]) / "ptbxl_hrv36.npz",
    )
    parser.add_argument(
        "--cpsc2021-hrv",
        type=Path,
        default=Path(PATHS["model_dir"]) / "cpsc2021_hrv36.npz",
    )
    return parser.parse_args()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _path_info(path: os.PathLike[str] | str, *, with_sha256: bool = False) -> dict:
    p = Path(path)
    info = {
        "path": str(p),
        "exists": p.exists(),
        "is_file": p.is_file() if p.exists() else False,
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
    }
    if with_sha256 and p.exists() and p.is_file():
        info["sha256"] = sha256_file(p)
    return info


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def sanitize_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def filter_folds_to_limit(folds: list[dict[str, np.ndarray]], n_records: int) -> list[dict[str, np.ndarray]]:
    filtered = []
    for fold in folds:
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        tr_idx = tr_idx[tr_idx < n_records]
        va_idx = va_idx[va_idx < n_records]
        if len(va_idx):
            filtered.append({"tr_idx": tr_idx, "va_idx": va_idx})
    if not filtered:
        raise ValueError("No validation records remain after applying --limit-records.")
    return filtered


def fit_predict_hrv_oof(
    X_hrv: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    *,
    seed: int = 42,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fit fold-safe one-vs-rest logistic models and return OOF probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_hrv = sanitize_features(X_hrv)
    y = np.asarray(y, dtype=np.float32)
    if X_hrv.ndim != 2:
        raise ValueError(f"X_hrv must be 2D, got {X_hrv.shape}")
    if len(X_hrv) != len(y):
        raise ValueError(f"X_hrv/y length mismatch: {len(X_hrv)} vs {len(y)}")

    n_records, n_classes = y.shape
    y_prob = np.full((n_records, n_classes), np.nan, dtype=np.float32)
    fold_id = np.full(n_records, -1, dtype=np.int16)
    fold_rows: list[dict] = []

    for fold_num, fold in enumerate(folds, start=1):
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            raise ValueError(f"Fold {fold_num} has empty train/validation indices.")
        if tr_idx.max(initial=-1) >= n_records or va_idx.max(initial=-1) >= n_records:
            raise IndexError(f"Fold {fold_num} contains index outside [0, {n_records}).")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_hrv[tr_idx])
        X_val = scaler.transform(X_hrv[va_idx])
        constant_class_count = 0

        for class_idx in range(n_classes):
            y_train = y[tr_idx, class_idx].astype(np.int8)
            unique = np.unique(y_train)
            if len(unique) < 2:
                constant_class_count += 1
                y_prob[va_idx, class_idx] = float(unique[0])
                continue
            model = LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=max_iter,
                random_state=seed + fold_num + class_idx,
            )
            model.fit(X_train, y_train)
            y_prob[va_idx, class_idx] = model.predict_proba(X_val)[:, 1].astype(np.float32)

        fold_id[va_idx] = fold_num
        fold_rows.append(
            {
                "fold": fold_num,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "constant_class_count": int(constant_class_count),
                "validation_positive_labels": int(np.sum(y[va_idx])),
            }
        )

    missing = np.where(fold_id < 0)[0]
    if len(missing):
        raise RuntimeError(f"OOF prediction coverage is incomplete; missing records: {len(missing)}")
    if not np.all(np.isfinite(y_prob)):
        raise RuntimeError("OOF probabilities contain non-finite values.")
    return np.clip(y_prob, 0.0, 1.0).astype(np.float32), fold_id, fold_rows


def per_class_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    threshold: float,
) -> list[dict]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = (y_prob >= threshold).astype(np.float32)
    rows = []
    for idx, name in enumerate(class_names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        pred = y_pred[:, idx]
        has_both = len(np.unique(yt)) >= 2
        rows.append(
            {
                "class_index": idx,
                "class_name": name,
                "n_records": int(len(yt)),
                "n_positive": int(np.sum(yt)),
                "prevalence": float(np.mean(yt)),
                "predicted_positive": int(np.sum(pred)),
                "predicted_positive_rate": float(np.mean(pred)),
                "roc_auc": float(roc_auc_score(yt, yp)) if has_both else math.nan,
                "pr_auc": float(average_precision_score(yt, yp)) if has_both else math.nan,
                "f1": float(f1_score(yt, pred, zero_division=0)),
                "precision": float(precision_score(yt, pred, zero_division=0)),
                "recall": float(recall_score(yt, pred, zero_division=0)),
            }
        )
    return rows


def _save_csv(path: Path, rows: Iterable[dict]) -> None:
    import csv

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def folds_from_fold_id(fold_id: np.ndarray) -> list[dict[str, np.ndarray]]:
    fold_id = np.asarray(fold_id, dtype=np.int16)
    folds = []
    for fold in sorted(int(x) for x in np.unique(fold_id) if int(x) > 0):
        va_idx = np.where(fold_id == fold)[0].astype(np.int64)
        tr_idx = np.where((fold_id > 0) & (fold_id != fold))[0].astype(np.int64)
        if len(va_idx) and len(tr_idx):
            folds.append({"tr_idx": tr_idx, "va_idx": va_idx})
    if not folds:
        raise ValueError("Could not derive folds from OOF fold_id.")
    return folds


def default_chapman_hrv_cache_path(n_records: int) -> Path:
    return Path(PATHS["cache_dir"]) / f"hrv36_N{n_records}_C12_L5000.npz"


def load_oof_labels_and_folds(oof_predictions: Path, limit_records: int) -> tuple[np.ndarray, list[dict[str, np.ndarray]], dict]:
    if not oof_predictions.exists():
        raise FileNotFoundError(
            f"Missing frozen OOF predictions: {oof_predictions}. "
            "Run/restore notebook 02 artifacts before notebook 05."
        )
    with np.load(oof_predictions, allow_pickle=False) as data:
        required = {"y_true", "fold_id", "class_names", "record_id"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{oof_predictions} is missing required keys: {sorted(missing)}")
        y = np.asarray(data["y_true"], dtype=np.float32)
        fold_id = np.asarray(data["fold_id"], dtype=np.int16)
        record_id = np.asarray(data["record_id"], dtype=np.int64)
        class_names = np.asarray(data["class_names"]).astype(str).tolist()

    if y.ndim != 2 or y.shape[1] != len(CLASSES):
        raise ValueError(f"Unexpected y_true shape in {oof_predictions}: {y.shape}")
    if len(fold_id) != len(y) or len(record_id) != len(y):
        raise ValueError("OOF y_true/fold_id/record_id length mismatch.")
    if not np.array_equal(record_id, np.arange(len(y), dtype=np.int64)):
        raise ValueError("OOF record_id must be exactly 0..N-1.")
    if class_names != CLASSES:
        raise ValueError("OOF class_names differ from configs.config.CLASSES.")

    if limit_records > 0:
        y = y[:limit_records]
        fold_id = fold_id[:limit_records]

    folds = folds_from_fold_id(fold_id)
    info = {
        "oof_predictions": str(oof_predictions),
        "oof_predictions_sha256": sha256_file(oof_predictions),
        "oof_records_total": int(len(record_id)),
        "oof_records_used": int(len(y)),
        "fold_count": int(len(folds)),
        "fold_counts": {str(fold): int(np.sum(fold_id == fold)) for fold in sorted(np.unique(fold_id)) if int(fold) > 0},
    }
    return y, folds, info


def load_cached_chapman_hrv(
    *,
    n_records: int,
    explicit_cache: Path | None,
    limit_records: int,
    allow_raw_fallback: bool,
) -> tuple[np.ndarray, dict]:
    candidates = []
    if explicit_cache is not None:
        candidates.append(explicit_cache)
    candidates.append(default_chapman_hrv_cache_path(n_records))
    candidates.append(PROJECT_ROOT / f"hrv36_N{n_records}_C12_L5000.npz")

    checked = []
    for path in candidates:
        path = Path(path)
        checked.append(str(path))
        if not path.exists():
            continue
        X_hrv = load_hrv_npz(path, expected_dim=int(CONFIG["hrv_dim"]))
        if len(X_hrv) < n_records:
            raise ValueError(f"HRV cache has fewer records than OOF labels: {X_hrv.shape} vs {n_records}")
        if len(X_hrv) != n_records:
            raise ValueError(f"HRV cache record count must match OOF labels: {X_hrv.shape} vs {n_records}")
        if limit_records > 0:
            X_hrv = X_hrv[:limit_records]
        return X_hrv, {
            "chapman_hrv_cache": str(path),
            "chapman_hrv_cache_sha256": sha256_file(path),
            "hrv_shape": list(X_hrv.shape),
            "raw_chapman_loaded": False,
            "checked_cache_paths": checked,
        }

    if not allow_raw_fallback:
        raise FileNotFoundError(
            "Missing Chapman HRV36 cache and raw fallback is disabled to avoid Colab OOM. "
            "Expected one of: " + "; ".join(checked)
        )

    # Explicit opt-in fallback for local debugging only. Notebook 05 does not use this path.
    import importlib

    pred_mod = importlib.import_module("scripts.revision.01_generate_predictions")
    from src.features import generate_hrv_cache

    X, _y, X_raw_amp, _subjects = pred_mod.prepare_clean_chapman(limit_records=limit_records)
    X_hrv = generate_hrv_cache(X, X_raw_amp)
    return sanitize_features(X_hrv), {
        "chapman_hrv_cache": None,
        "chapman_hrv_cache_sha256": None,
        "hrv_shape": list(X_hrv.shape),
        "raw_chapman_loaded": True,
        "checked_cache_paths": checked,
    }


def load_chapman_hrv_and_folds(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]], dict]:
    y, folds, oof_info = load_oof_labels_and_folds(args.oof_predictions, args.limit_records)
    X_hrv, hrv_info = load_cached_chapman_hrv(
        n_records=oof_info["oof_records_total"],
        explicit_cache=args.chapman_hrv_cache,
        limit_records=args.limit_records,
        allow_raw_fallback=args.allow_raw_chapman_fallback,
    )
    if len(X_hrv) != len(y):
        raise ValueError(f"HRV/y length mismatch after loading: {len(X_hrv)} vs {len(y)}")
    load_info = {
        "chapman_records": int(len(y)),
        "chapman_classes": int(y.shape[1]),
        "hrv_shape": list(X_hrv.shape),
        "fold_count": int(len(folds)),
        "limit_records": int(args.limit_records),
        **oof_info,
        **hrv_info,
    }
    return sanitize_features(X_hrv), y.astype(np.float32), folds, load_info


def compute_hrv_baseline(
    X_hrv: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    *,
    threshold: float,
    n_bins: int,
    n_boot: int,
    seed: int,
) -> dict:
    y_prob, fold_id, fold_rows = fit_predict_hrv_oof(X_hrv, y, folds, seed=seed)
    metrics = multilabel_metrics(y, y_prob, threshold=threshold)
    calibration = calibration_summary(y, y_prob, n_bins=n_bins)
    ci = {
        "macro_pr_auc": bootstrap_ci(y, y_prob, macro_pr_auc, n_boot=n_boot, seed=seed),
        "macro_roc_auc": bootstrap_ci(y, y_prob, macro_roc_auc, n_boot=n_boot, seed=seed),
        "f1_macro": bootstrap_ci(
            y,
            y_prob,
            lambda yt, yp: multilabel_metrics(yt, yp, threshold=threshold)["f1_macro"],
            n_boot=n_boot,
            seed=seed,
        ),
    }
    return {
        "y_prob": y_prob,
        "fold_id": fold_id,
        "fold_rows": fold_rows,
        "metrics": metrics,
        "calibration": calibration,
        "bootstrap_ci": ci,
        "per_class_rows": per_class_rows(y, y_prob, CLASSES, threshold),
    }


def load_hrv_npz(path: Path, *, expected_dim: int = 36) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    if "X" not in data.files:
        raise KeyError(f"{path} must contain key 'X'; found {data.files}")
    X = sanitize_features(data["X"])
    if X.ndim != 2 or X.shape[1] != expected_dim:
        raise ValueError(f"{path} expected shape (N, {expected_dim}), got {X.shape}")
    return X


def resolve_hrv_feature_file(path: Path, filename: str) -> Path:
    candidates = [
        Path(path),
        Path(PATHS["model_dir"]) / filename,
        PROJECT_ROOT / "model" / filename,
        PROJECT_ROOT / "models" / filename,
        Path(PATHS["cache_dir"]) / "model" / filename,
        Path(PATHS["cache_dir"]) / "models" / filename,
    ]
    checked = []
    for candidate in candidates:
        candidate = Path(candidate)
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {filename}. Checked: " + "; ".join(checked))


def balanced_domain_arrays(
    features_by_domain: dict[str, np.ndarray],
    *,
    max_per_domain: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    if len(features_by_domain) < 2:
        raise ValueError("At least two domains are required for domain classification.")
    rng = np.random.default_rng(seed)
    Xs, ys, rows = [], [], []
    domains = sorted(features_by_domain)
    for domain_idx, domain in enumerate(domains):
        X = sanitize_features(features_by_domain[domain])
        if len(X) == 0:
            raise ValueError(f"Domain {domain} has no records.")
        take = min(int(max_per_domain), len(X)) if max_per_domain > 0 else len(X)
        selected = np.sort(rng.choice(len(X), size=take, replace=False))
        Xs.append(X[selected])
        ys.append(np.full(take, domain_idx, dtype=np.int64))
        rows.append(
            {
                "domain": domain,
                "available_records": int(len(X)),
                "sampled_records": int(take),
                "feature_dim": int(X.shape[1]),
            }
        )
    return np.vstack(Xs).astype(np.float32), np.concatenate(ys), domains, rows


def run_domain_classifier_cv(
    features_by_domain: dict[str, np.ndarray],
    *,
    max_per_domain: int = 2000,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, label_binarize

    X, y, domains, sample_rows = balanced_domain_arrays(
        features_by_domain,
        max_per_domain=max_per_domain,
        seed=seed,
    )
    class_counts = np.bincount(y, minlength=len(domains))
    effective_splits = min(int(n_splits), int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError(f"Need at least two samples per domain for CV, got counts {class_counts.tolist()}")

    y_prob = np.full((len(y), len(domains)), np.nan, dtype=np.float32)
    fold_id = np.full(len(y), -1, dtype=np.int16)
    fold_rows: list[dict] = []
    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    for fold_num, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[va_idx])
        model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed + fold_num,
        )
        model.fit(X_train, y[tr_idx])
        probs = model.predict_proba(X_val)
        class_to_col = {int(cls): idx for idx, cls in enumerate(model.classes_)}
        for cls in range(len(domains)):
            y_prob[va_idx, cls] = probs[:, class_to_col[cls]]
        fold_id[va_idx] = fold_num
        fold_rows.append(
            {
                "fold": fold_num,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
            }
        )
    if np.any(fold_id < 0) or not np.all(np.isfinite(y_prob)):
        raise RuntimeError("Domain classifier OOF coverage is incomplete.")

    y_pred = np.argmax(y_prob, axis=1)
    y_binary = label_binarize(y, classes=np.arange(len(domains)))
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "domain_roc_auc_ovr_macro": float(
            roc_auc_score(y_binary, y_prob, average="macro", multi_class="ovr")
        ),
        "domain_pr_auc_macro": float(average_precision_score(y_binary, y_prob, average="macro")),
    }
    matrix = confusion_matrix(y, y_pred, labels=np.arange(len(domains)))
    confusion_rows = []
    for true_idx, domain in enumerate(domains):
        row_total = int(matrix[true_idx].sum())
        for pred_idx, pred_domain in enumerate(domains):
            confusion_rows.append(
                {
                    "true_domain": domain,
                    "predicted_domain": pred_domain,
                    "count": int(matrix[true_idx, pred_idx]),
                    "row_fraction": float(matrix[true_idx, pred_idx] / row_total) if row_total else math.nan,
                }
            )

    return {
        "X_shape": list(X.shape),
        "domains": domains,
        "sample_rows": sample_rows,
        "fold_rows": fold_rows,
        "fold_id": fold_id,
        "y_true_domain": y,
        "y_prob_domain": y_prob,
        "metrics": metrics,
        "confusion_rows": confusion_rows,
    }


def interpretation_for_domain_auc(domain_auc: float | None) -> str:
    if domain_auc is None or not np.isfinite(domain_auc):
        return "domain_classifier_unavailable"
    if domain_auc >= 0.85:
        return "high_domain_sensitivity_present_as_limitation"
    if domain_auc >= 0.70:
        return "moderate_domain_sensitivity_report_with_caution"
    return "low_to_moderate_domain_sensitivity"


def write_outputs(
    args: argparse.Namespace,
    load_info: dict,
    baseline: dict,
    domain_result: dict | None,
    domain_blocker: str | None = None,
) -> dict:
    ensure_revision_dirs()
    for directory in [PREDICTION_DIR, METRIC_DIR, TABLE_DIR, MANIFEST_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    created_utc = _now_utc()
    y = baseline["y_true"]
    y_prob = baseline["y_prob"]
    fold_id = baseline["fold_id"]

    prediction_path = PREDICTION_DIR / "hrv_only_oof_predictions.npz"
    np.savez_compressed(
        prediction_path,
        y_true=y.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        record_id=np.arange(len(y), dtype=np.int64),
        fold_id=fold_id.astype(np.int16),
        class_names=np.asarray(CLASSES),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(PROTOCOL),
        threshold=np.asarray(float(args.threshold)),
        config_hash=np.asarray(CONFIG_HASH),
        git_commit=np.asarray(_git_output(["rev-parse", "HEAD"]) or ""),
        manuscript_ready=np.asarray(True),
    )

    per_class_path = TABLE_DIR / "table_hrv_only_class_metrics.csv"
    _save_csv(per_class_path, baseline["per_class_rows"])
    fold_path = TABLE_DIR / "table_hrv_only_fold_summary.csv"
    _save_csv(fold_path, baseline["fold_rows"])

    baseline_summary = {
        "created_utc": created_utc,
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "dataset": "chapman_oof",
        "protocol": PROTOCOL,
        "feature_contract": "hrv36",
        "model": "fold_safe_one_vs_rest_logistic_regression",
        "n_records": int(len(y)),
        "n_classes": int(y.shape[1]),
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "n_boot": int(args.n_boot),
        "metrics": baseline["metrics"],
        "calibration": baseline["calibration"],
        "bootstrap_ci": baseline["bootstrap_ci"],
        "load_info": load_info,
        "artifacts": {
            "predictions_npz": str(prediction_path),
            "per_class_table": str(per_class_path),
            "fold_summary_table": str(fold_path),
        },
    }
    baseline_summary_path = METRIC_DIR / "hrv_only_baseline_summary.json"
    save_json(baseline_summary_path, _json_safe(baseline_summary))

    domain_summary_path = None
    domain_confusion_path = None
    domain_prediction_path = None
    domain_auc = None
    domain_summary_path = METRIC_DIR / "hrv_domain_classifier_summary.json"
    if domain_result is not None:
        domain_prediction_path = PREDICTION_DIR / "hrv_domain_oof_predictions.npz"
        np.savez_compressed(
            domain_prediction_path,
            y_true_domain=domain_result["y_true_domain"].astype(np.int64),
            y_prob_domain=domain_result["y_prob_domain"].astype(np.float32),
            fold_id=domain_result["fold_id"].astype(np.int16),
            domains=np.asarray(domain_result["domains"]),
            protocol=np.asarray(DOMAIN_PROTOCOL),
            git_commit=np.asarray(_git_output(["rev-parse", "HEAD"]) or ""),
        )
        domain_confusion_path = TABLE_DIR / "table_hrv_domain_classifier_confusion.csv"
        _save_csv(domain_confusion_path, domain_result["confusion_rows"])
        domain_fold_path = TABLE_DIR / "table_hrv_domain_classifier_fold_summary.csv"
        _save_csv(domain_fold_path, domain_result["fold_rows"])
        domain_auc = domain_result["metrics"]["domain_roc_auc_ovr_macro"]
        domain_summary = {
            "created_utc": created_utc,
            "git_commit": _git_output(["rev-parse", "HEAD"]),
            "protocol": DOMAIN_PROTOCOL,
            "feature_contract": "hrv36",
            "model": "standardized_multiclass_logistic_regression",
            "domains": domain_result["domains"],
            "sample_rows": domain_result["sample_rows"],
            "fold_rows": domain_result["fold_rows"],
            "metrics": domain_result["metrics"],
            "interpretation": interpretation_for_domain_auc(domain_auc),
            "artifacts": {
                "predictions_npz": str(domain_prediction_path),
                "confusion_table": str(domain_confusion_path),
                "fold_summary_table": str(domain_fold_path),
            },
        }
        save_json(domain_summary_path, _json_safe(domain_summary))
    else:
        domain_summary = {
            "created_utc": created_utc,
            "git_commit": _git_output(["rev-parse", "HEAD"]),
            "protocol": DOMAIN_PROTOCOL,
            "feature_contract": "hrv36",
            "model": "standardized_multiclass_logistic_regression",
            "status": "blocked_external_hrv_missing",
            "blocker": domain_blocker or "External HRV36 domain feature files were unavailable.",
            "domains": [],
            "sample_rows": [],
            "fold_rows": [],
            "metrics": {},
            "interpretation": "domain_classifier_blocked_external_hrv_missing",
            "artifacts": {
                "predictions_npz": None,
                "confusion_table": None,
                "fold_summary_table": None,
            },
        }
        save_json(domain_summary_path, _json_safe(domain_summary))

    hrv_rows = [
        {
            "analysis_name": "HRV-only baseline",
            "required_output": "HRV-only metrics under same split/threshold protocol",
            "status": "complete",
            "metric_name": "roc_auc_macro",
            "metric_value": baseline["metrics"]["roc_auc_macro"],
            "secondary_metric_name": "pr_auc_macro",
            "secondary_metric_value": baseline["metrics"]["pr_auc_macro"],
            "evidence_path": str(baseline_summary_path),
            "blocker": "",
            "safe_wording": "Report as HRV-only feature baseline under the same frozen Chapman OOF split, not as full-model performance.",
        },
        {
            "analysis_name": "HRV domain classifier",
            "required_output": "Domain AUC and interpretation of HRV dataset-source sensitivity",
            "status": "complete" if domain_result is not None else "blocked_external_hrv_missing",
            "metric_name": "domain_roc_auc_ovr_macro",
            "metric_value": domain_auc if domain_auc is not None else math.nan,
            "secondary_metric_name": "balanced_accuracy",
            "secondary_metric_value": domain_result["metrics"]["balanced_accuracy"] if domain_result is not None else math.nan,
            "evidence_path": str(domain_summary_path),
            "blocker": "" if domain_result is not None else (domain_blocker or "External HRV36 domain feature files were unavailable."),
            "safe_wording": (
                "If domain AUC is high, present HRV as domain-sensitive and keep robustness/domain-generalization wording limited."
                if domain_result is not None
                else "Do not claim HRV domain-sensitivity evidence until external HRV36 features are available."
            ),
        },
        {
            "analysis_name": "Duration/noise HRV sensitivity",
            "required_output": "Duration/noise sensitivity summary for HRV features",
            "status": "blocked_runner_tbd",
            "metric_name": "sensitivity_delta",
            "metric_value": math.nan,
            "secondary_metric_name": "",
            "secondary_metric_value": math.nan,
            "evidence_path": "",
            "blocker": "No implemented duration/noise HRV sensitivity runner in scripts/revision.",
            "safe_wording": "Do not present HRV descriptors as validated against duration/noise shifts.",
        },
    ]
    hrv_summary_csv = METRIC_DIR / "hrv_domain_summary.csv"
    hrv_summary_table = TABLE_DIR / "table_hrv_domain_status.csv"
    _save_csv(hrv_summary_csv, hrv_rows)
    _save_csv(hrv_summary_table, hrv_rows)

    outputs = [
        prediction_path,
        per_class_path,
        fold_path,
        baseline_summary_path,
        domain_summary_path,
        hrv_summary_csv,
        hrv_summary_table,
    ]
    if domain_result is not None:
        outputs.extend([domain_prediction_path, domain_confusion_path, domain_summary_path])

    manifest = {
        "created_utc": created_utc,
        "git": {
            "commit": _git_output(["rev-parse", "HEAD"]),
            "branch": _git_output(["branch", "--show-current"]),
            "status_short": _git_output(["status", "--short", "--branch"]),
        },
        "config_hash": CONFIG_HASH,
        "args": vars(args),
        "input_paths": {
            "model_dir": _path_info(PATHS["model_dir"]),
            "cache_dir": _path_info(PATHS["cache_dir"]),
            "ptbxl_hrv": _path_info(args.ptbxl_hrv, with_sha256=True),
            "cpsc2021_hrv": _path_info(args.cpsc2021_hrv, with_sha256=True),
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in outputs
            if path is not None and path.exists()
        },
        "claim_boundary": {
            "hrv_only_baseline": "supported_as_feature_baseline_only",
            "hrv_domain_classifier": interpretation_for_domain_auc(domain_auc),
            "robustness": "not_run_by_this_script",
        },
    }
    manifest_path = MANIFEST_DIR / "hrv_domain_analysis_manifest.json"
    save_json(manifest_path, _json_safe(manifest))

    final_summary = {
        "status": True,
        "hrv_only_roc_auc_macro": baseline["metrics"]["roc_auc_macro"],
        "hrv_only_pr_auc_macro": baseline["metrics"]["pr_auc_macro"],
        "hrv_only_f1_macro": baseline["metrics"]["f1_macro"],
        "domain_roc_auc_ovr_macro": domain_auc,
        "domain_status": "complete" if domain_result is not None else "blocked_external_hrv_missing",
        "domain_blocker": domain_blocker,
        "domain_interpretation": interpretation_for_domain_auc(domain_auc),
        "outputs": {
            "hrv_only_summary_json": str(baseline_summary_path),
            "hrv_domain_classifier_summary_json": str(domain_summary_path),
            "hrv_domain_summary_csv": str(hrv_summary_csv),
            "hrv_domain_summary_table": str(hrv_summary_table),
            "manifest_json": str(manifest_path),
        },
        "recommended_next_step": "implement_robustness_stress_runner_or_keep_robustness_blocked",
    }
    return final_summary


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    X_hrv, y, folds, load_info = load_chapman_hrv_and_folds(args)
    baseline = compute_hrv_baseline(
        X_hrv,
        y,
        folds,
        threshold=args.threshold,
        n_bins=args.n_bins,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    baseline["y_true"] = y

    domain_result = None
    domain_blocker = None
    if not args.skip_domain_classifier:
        try:
            features_by_domain = {
                "chapman": X_hrv,
                "ptbxl": load_hrv_npz(resolve_hrv_feature_file(args.ptbxl_hrv, "ptbxl_hrv36.npz")),
                "cpsc2021": load_hrv_npz(resolve_hrv_feature_file(args.cpsc2021_hrv, "cpsc2021_hrv36.npz")),
            }
            domain_result = run_domain_classifier_cv(
                features_by_domain,
                max_per_domain=args.domain_max_per_domain,
                n_splits=args.domain_n_splits,
                seed=args.seed,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            domain_blocker = str(exc)
            print(f"WARNING: HRV domain classifier blocked: {domain_blocker}", flush=True)
    else:
        domain_blocker = "Domain classifier skipped by --skip-domain-classifier."

    final_summary = write_outputs(args, load_info, baseline, domain_result, domain_blocker=domain_blocker)
    print(json.dumps(_json_safe(final_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
