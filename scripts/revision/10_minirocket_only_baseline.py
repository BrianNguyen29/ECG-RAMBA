"""Run a fold-safe MiniRocket-only baseline under the frozen OOF protocol.

This runner is intentionally separate from the full ECG-RAMBA model. It uses
the cached deterministic RAW MiniRocket feature matrix, the exact frozen OOF
fold split/labels, and a lightweight linear logistic head. The goal is
reviewer-facing baseline evidence, not checkpoint inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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


PROTOCOL = "minirocket_raw_standardized_torch_linear_same_folds_threshold_0.5"
DEFAULT_OOF_PREDICTIONS = PREDICTION_DIR / "oof_final_ema_predictions.npz"
DEFAULT_FREEZE_MANIFEST = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--backend", choices=["torch_linear", "sgd"], default="torch_linear")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--stats-batch-size", type=int, default=1024)
    parser.add_argument("--torch-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu for torch_linear backend.")
    parser.add_argument("--standardize", choices=["train_fold", "none"], default="train_fold")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=DEFAULT_OOF_PREDICTIONS,
        help="Frozen final_ema OOF NPZ used for labels and fold_id.",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=DEFAULT_FREEZE_MANIFEST,
        help="Frozen final_ema OOF manifest used to verify provenance.",
    )
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument(
        "--minirocket-cache",
        type=Path,
        default=None,
        help=(
            "Optional explicit RAW MiniRocket NPZ. Default requires a "
            "record-fingerprinted cache matching the frozen OOF record order."
        ),
    )
    parser.add_argument(
        "--allow-legacy-shape-cache",
        action="store_true",
        help=(
            "Allow a legacy rocket_raw_N*_C12_L5000_K10000_S42.npz cache that "
            "lacks record_order_fingerprint metadata. Off by default for "
            "manuscript integrity."
        ),
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


def _resolve_project_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    return resolved.resolve()


def _project_relative(path: Path) -> str:
    return _resolve_project_path(path).relative_to(PROJECT_ROOT.resolve()).as_posix()


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


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_oof_freeze_contract(
    *,
    freeze_manifest: Path,
    oof_predictions: Path,
    expected_checkpoint_kind: str,
) -> dict:
    freeze_path = _resolve_project_path(freeze_manifest)
    pred_path = _resolve_project_path(oof_predictions)
    if not freeze_path.exists():
        raise FileNotFoundError(f"Missing OOF freeze manifest: {freeze_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {pred_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("manuscript_ready") is not True:
        raise ValueError("OOF freeze manifest must be status=frozen and manuscript_ready=true.")
    if freeze.get("checkpoint_kind") != expected_checkpoint_kind:
        raise ValueError(
            "Unexpected OOF checkpoint kind: "
            f"{freeze.get('checkpoint_kind')} != {expected_checkpoint_kind}"
        )
    relative = _project_relative(pred_path)
    artifacts = {row["path"]: row for row in freeze.get("artifacts", [])}
    if relative not in artifacts:
        raise ValueError(f"Freeze manifest does not include OOF predictions: {relative}")
    pred_sha = sha256_file(pred_path)
    expected_sha = artifacts[relative].get("sha256")
    if pred_sha != expected_sha:
        raise RuntimeError(
            f"OOF prediction SHA256 changed after freeze: {relative} "
            f"{pred_sha} != {expected_sha}"
        )
    if int(freeze.get("validated_records", -1)) != 44186:
        raise ValueError(f"Unexpected freeze validated_records: {freeze.get('validated_records')}")
    if int(freeze.get("n_classes", -1)) != len(CLASSES):
        raise ValueError(f"Unexpected freeze n_classes: {freeze.get('n_classes')}")
    return {
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "oof_predictions_relative": relative,
        "oof_predictions_sha256": pred_sha,
        "checkpoint_kind": freeze.get("checkpoint_kind"),
        "validated_records": int(freeze.get("validated_records")),
        "n_classes": int(freeze.get("n_classes")),
        "dataset_record_order_fingerprint": freeze.get("dataset_record_order_fingerprint"),
    }


def npz_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def folds_from_fold_id(fold_id: np.ndarray) -> list[dict[str, np.ndarray]]:
    fold_id = np.asarray(fold_id, dtype=np.int16)
    folds = []
    for fold in sorted(int(x) for x in np.unique(fold_id) if int(x) > 0):
        va_idx = np.where(fold_id == fold)[0].astype(np.int64)
        tr_idx = np.where((fold_id > 0) & (fold_id != fold))[0].astype(np.int64)
        if len(va_idx) and len(tr_idx):
            folds.append({"fold": fold, "tr_idx": tr_idx, "va_idx": va_idx})
    if not folds:
        raise ValueError("Could not derive folds from OOF fold_id.")
    return folds


def load_oof_labels_and_folds(
    oof_predictions: Path,
    *,
    limit_records: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, np.ndarray]], dict]:
    pred_path = _resolve_project_path(oof_predictions)
    with np.load(pred_path, allow_pickle=False) as data:
        required = {"y_true", "fold_id", "class_names", "record_id"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{pred_path} is missing required keys: {sorted(missing)}")
        y = np.asarray(data["y_true"], dtype=np.float32)
        fold_id = np.asarray(data["fold_id"], dtype=np.int16)
        record_id = np.asarray(data["record_id"], dtype=np.int64)
        class_names = np.asarray(data["class_names"]).astype(str).tolist()
        dataset_record_fingerprint = str(npz_scalar(data, "dataset_record_order_fingerprint", "") or "")

    if y.ndim != 2 or y.shape[1] != len(CLASSES):
        raise ValueError(f"Unexpected y_true shape in {pred_path}: {y.shape}")
    if class_names != CLASSES:
        raise ValueError("OOF class_names differ from configs.config.CLASSES.")
    if len(fold_id) != len(y) or len(record_id) != len(y):
        raise ValueError("OOF y_true/fold_id/record_id length mismatch.")
    if not np.array_equal(record_id, np.arange(len(y), dtype=np.int64)):
        raise ValueError("OOF record_id must be exactly 0..N-1.")

    total_records = len(y)
    if limit_records > 0:
        y = y[:limit_records]
        fold_id = fold_id[:limit_records]
        record_id = record_id[:limit_records]

    folds = folds_from_fold_id(fold_id)
    info = {
        "oof_predictions": str(pred_path),
        "oof_predictions_sha256": sha256_file(pred_path),
        "oof_records_total": int(total_records),
        "oof_records_used": int(len(y)),
        "fold_count": int(len(folds)),
        "fold_counts": {
            str(fold): int(np.sum(fold_id == fold))
            for fold in sorted(np.unique(fold_id))
            if int(fold) > 0
        },
        "dataset_record_order_fingerprint": dataset_record_fingerprint,
    }
    return y, fold_id, record_id, class_names, folds, info


def candidate_minirocket_cache_paths(
    *,
    n_records: int,
    record_fingerprint: str,
    explicit_cache: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if explicit_cache is not None:
        candidates.append(Path(explicit_cache))

    roots = [Path(PATHS["cache_dir"]), PROJECT_ROOT]
    drive_root = os.environ.get("ECG_RAMBA_DRIVE_ROOT")
    if drive_root:
        roots.append(Path(drive_root))

    seen_roots: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        if not root.exists():
            continue
        if record_fingerprint:
            candidates.append(
                root
                / f"rocket_raw_N{n_records}_C12_L5000_K10000_S42_R{record_fingerprint}.npz"
            )
        candidates.extend(
            sorted(
                root.glob(f"rocket_raw_N{n_records}_C12_L5000_K10000_S42_R*.npz"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
        candidates.append(root / f"rocket_raw_N{n_records}_C12_L5000_K10000_S42.npz")

    deduped: list[Path] = []
    seen_paths: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        deduped.append(path)
    return deduped


def load_minirocket_cache(
    *,
    n_records: int,
    record_fingerprint: str,
    explicit_cache: Path | None,
    allow_legacy_shape_cache: bool,
    limit_records: int,
) -> tuple[np.ndarray, dict]:
    candidates = candidate_minirocket_cache_paths(
        n_records=n_records,
        record_fingerprint=record_fingerprint,
        explicit_cache=explicit_cache,
    )
    checked = []
    for path in candidates:
        path = Path(path)
        checked.append(str(path))
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            if "X" not in payload.files:
                raise KeyError(f"{path} must contain key 'X'; found {payload.files}")
            X = np.asarray(payload["X"])
            cached_fingerprint = str(npz_scalar(payload, "record_order_fingerprint", "") or "")
            storage_dtype = str(npz_scalar(payload, "storage_dtype", str(payload["X"].dtype)))
            consumer_dtype = str(npz_scalar(payload, "consumer_dtype", "float32"))
            quantization_contract = str(npz_scalar(payload, "quantization_contract", "unknown"))

        if X.ndim != 2 or X.shape != (n_records, 20000):
            raise ValueError(f"RAW MiniRocket cache expected {(n_records, 20000)}, got {X.shape}: {path}")
        if not np.isfinite(X).all():
            raise ValueError(f"RAW MiniRocket cache contains non-finite values: {path}")

        cache_kind = "record_fingerprinted" if cached_fingerprint else "legacy_shape_only"
        if record_fingerprint and cached_fingerprint != record_fingerprint:
            if cached_fingerprint:
                raise ValueError(
                    "RAW MiniRocket cache fingerprint mismatch: "
                    f"{cached_fingerprint} != {record_fingerprint} ({path})"
                )
            if not allow_legacy_shape_cache:
                continue
        if cache_kind == "legacy_shape_only" and not allow_legacy_shape_cache:
            continue

        if limit_records > 0:
            X = X[:limit_records]
        return X, {
            "minirocket_cache": str(path),
            "minirocket_cache_sha256": sha256_file(path),
            "minirocket_cache_kind": cache_kind,
            "minirocket_cache_record_order_fingerprint": cached_fingerprint,
            "minirocket_storage_dtype": storage_dtype,
            "minirocket_consumer_dtype": consumer_dtype,
            "minirocket_quantization_contract": quantization_contract,
            "minirocket_shape": list(X.shape),
            "minirocket_loaded_dtype": str(X.dtype),
            "checked_cache_paths": checked,
        }

    raise FileNotFoundError(
        "Missing manuscript-safe RAW MiniRocket cache. Expected a record-fingerprinted "
        f"cache for fingerprint {record_fingerprint or '<missing>'}. Checked: "
        + "; ".join(checked)
    )


def fit_predict_minirocket_sgd_oof(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    *,
    seed: int,
    max_iter: int,
    tol: float,
    alpha: float,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=np.float32)
    if len(X) != len(y):
        raise ValueError(f"Feature/label length mismatch: {len(X)} vs {len(y)}")

    n_records, n_classes = y.shape
    y_prob = np.full((n_records, n_classes), np.nan, dtype=np.float32)
    fold_id_out = np.full(n_records, -1, dtype=np.int16)
    fold_rows: list[dict] = []

    for fold in folds:
        fold_num = int(fold["fold"])
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        print(
            f"Fold {fold_num}/5 | train={len(tr_idx)} | val={len(va_idx)} | "
            f"features={X.shape[1]}",
            flush=True,
        )
        scaler = StandardScaler(with_mean=False, copy=True)
        X_train = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[va_idx])
        constant_class_count = 0

        for class_idx, class_name in enumerate(CLASSES):
            y_train = y[tr_idx, class_idx].astype(np.int8)
            unique = np.unique(y_train)
            if len(unique) < 2:
                constant_class_count += 1
                y_prob[va_idx, class_idx] = float(unique[0])
                continue
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                class_weight="balanced",
                random_state=seed + fold_num * 100 + class_idx,
                n_jobs=n_jobs,
            )
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)[:, 1]
            else:
                logits = model.decision_function(X_val)
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            y_prob[va_idx, class_idx] = probs.astype(np.float32)
            if (class_idx + 1) % 9 == 0:
                print(f"  fold {fold_num}: fitted {class_idx + 1}/{n_classes} classes", flush=True)

        fold_id_out[va_idx] = fold_num
        fold_rows.append(
            {
                "fold": fold_num,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "constant_class_count": int(constant_class_count),
                "validation_positive_labels": int(np.sum(y[va_idx])),
                "classifier": "SGDClassifier(log_loss)",
                "max_iter": int(max_iter),
                "tol": float(tol),
                "alpha": float(alpha),
            }
        )

    missing = np.where(fold_id_out < 0)[0]
    if len(missing):
        raise RuntimeError(f"OOF prediction coverage is incomplete; missing records: {len(missing)}")
    if not np.all(np.isfinite(y_prob)):
        raise RuntimeError("OOF probabilities contain non-finite values.")
    return np.clip(y_prob, 0.0, 1.0).astype(np.float32), fold_id_out, fold_rows


def iter_index_batches(indices: np.ndarray, batch_size: int, *, shuffle: bool, rng: np.random.Generator):
    indices = np.asarray(indices, dtype=np.int64)
    if shuffle:
        indices = indices.copy()
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def compute_train_standardization(
    X: np.ndarray,
    train_idx: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature mean/std on the training fold without materializing it."""
    if batch_size <= 0:
        raise ValueError("--stats-batch-size must be positive.")
    n_features = int(X.shape[1])
    total = np.zeros(n_features, dtype=np.float64)
    total_sq = np.zeros(n_features, dtype=np.float64)
    count = 0
    rng = np.random.default_rng(0)
    for batch_idx in iter_index_batches(train_idx, batch_size, shuffle=False, rng=rng):
        xb = np.asarray(X[batch_idx], dtype=np.float32)
        total += np.sum(xb, axis=0, dtype=np.float64)
        total_sq += np.sum(xb * xb, axis=0, dtype=np.float64)
        count += int(len(batch_idx))
    if count == 0:
        raise ValueError("Cannot standardize with an empty training fold.")
    mean = total / float(count)
    var = np.maximum(total_sq / float(count) - mean * mean, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def prepare_feature_batch(
    X: np.ndarray,
    batch_idx: np.ndarray,
    *,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    xb = np.asarray(X[batch_idx], dtype=np.float32)
    if mean is not None and std is not None:
        xb = xb.copy()
        xb -= mean
        xb /= std
    return np.ascontiguousarray(xb, dtype=np.float32)


def fit_predict_minirocket_torch_oof(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    *,
    seed: int,
    batch_size: int,
    stats_batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device_name: str,
    standardize: str,
    allow_tf32: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fit a fold-safe linear logistic head on MiniRocket features in batches."""
    import torch
    import torch.nn as nn

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if epochs <= 0:
        raise ValueError("--torch-epochs must be positive.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"Torch linear backend device={device} batch_size={batch_size} "
        f"epochs={epochs} standardize={standardize} allow_tf32={allow_tf32}",
        flush=True,
    )

    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float32)
    if len(X) != len(y):
        raise ValueError(f"Feature/label length mismatch: {len(X)} vs {len(y)}")

    n_records, n_classes = y.shape
    n_features = int(X.shape[1])
    y_prob = np.full((n_records, n_classes), np.nan, dtype=np.float32)
    fold_id_out = np.full(n_records, -1, dtype=np.int16)
    fold_rows: list[dict] = []

    for fold in folds:
        fold_num = int(fold["fold"])
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        rng = np.random.default_rng(seed + fold_num)
        torch.manual_seed(seed + fold_num)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + fold_num)

        print(
            f"Fold {fold_num}/5 | train={len(tr_idx)} | val={len(va_idx)} | "
            f"features={n_features}",
            flush=True,
        )
        if standardize == "train_fold":
            print(f"  fold {fold_num}: computing train-fold feature standardization", flush=True)
            mean, std = compute_train_standardization(
                X,
                tr_idx,
                batch_size=stats_batch_size,
            )
        else:
            mean, std = None, None
        pos = np.sum(y[tr_idx], axis=0).astype(np.float32)
        neg = float(len(tr_idx)) - pos
        pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0).astype(np.float32)

        model = nn.Linear(n_features, n_classes).to(device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_seen = 0
            for batch_idx in iter_index_batches(tr_idx, batch_size, shuffle=True, rng=rng):
                xb = prepare_feature_batch(X, batch_idx, mean=mean, std=std)
                xb_t = torch.as_tensor(xb, dtype=torch.float32, device=device)
                yb_t = torch.as_tensor(y[batch_idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb_t), yb_t)
                loss.backward()
                optimizer.step()
                batch_n = int(len(batch_idx))
                total_loss += float(loss.detach().cpu()) * batch_n
                total_seen += batch_n
            print(
                f"  fold {fold_num}: epoch {epoch:02d}/{epochs} "
                f"loss={total_loss / max(total_seen, 1):.5f}",
                flush=True,
            )

        model.eval()
        with torch.no_grad():
            for batch_idx in iter_index_batches(va_idx, batch_size, shuffle=False, rng=rng):
                xb = prepare_feature_batch(X, batch_idx, mean=mean, std=std)
                xb_t = torch.as_tensor(xb, dtype=torch.float32, device=device)
                probs = torch.sigmoid(model(xb_t)).detach().cpu().numpy()
                y_prob[batch_idx] = probs.astype(np.float32)

        fold_id_out[va_idx] = fold_num
        fold_rows.append(
            {
                "fold": fold_num,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "validation_positive_labels": int(np.sum(y[va_idx])),
                "classifier": "torch.nn.Linear+BCEWithLogitsLoss",
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "stats_batch_size": int(stats_batch_size),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "standardize": standardize,
                "allow_tf32": bool(allow_tf32),
                "device": str(device),
            }
        )
        del model, optimizer, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    missing = np.where(fold_id_out < 0)[0]
    if len(missing):
        raise RuntimeError(f"OOF prediction coverage is incomplete; missing records: {len(missing)}")
    if not np.all(np.isfinite(y_prob)):
        raise RuntimeError("OOF probabilities contain non-finite values.")
    return np.clip(y_prob, 0.0, 1.0).astype(np.float32), fold_id_out, fold_rows


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


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    freeze_contract = validate_oof_freeze_contract(
        freeze_manifest=args.freeze_manifest,
        oof_predictions=args.oof_predictions,
        expected_checkpoint_kind=args.expected_checkpoint_kind,
    )
    y, _fold_id, record_id, class_names, folds, oof_info = load_oof_labels_and_folds(
        args.oof_predictions,
        limit_records=args.limit_records,
    )
    if oof_info["oof_records_total"] != int(freeze_contract["validated_records"]):
        raise ValueError(
            "OOF prediction record count does not match freeze manifest: "
            f"{oof_info['oof_records_total']} != {freeze_contract['validated_records']}"
        )
    if int(args.limit_records) == 0 and oof_info["fold_count"] != 5:
        raise ValueError(f"Canonical MiniRocket-only baseline requires five folds, got {oof_info['fold_count']}")
    record_fingerprint = (
        oof_info.get("dataset_record_order_fingerprint")
        or freeze_contract.get("dataset_record_order_fingerprint")
        or ""
    )
    if not record_fingerprint:
        raise ValueError(
            "Frozen OOF artifacts must carry dataset_record_order_fingerprint "
            "before a MiniRocket cache can be manuscript-safe."
        )
    X, cache_info = load_minirocket_cache(
        n_records=oof_info["oof_records_total"],
        record_fingerprint=record_fingerprint,
        explicit_cache=args.minirocket_cache,
        allow_legacy_shape_cache=args.allow_legacy_shape_cache,
        limit_records=args.limit_records,
    )
    if len(X) != len(y):
        raise ValueError(f"MiniRocket/y length mismatch after loading: {len(X)} vs {len(y)}")

    if args.backend == "torch_linear":
        y_prob, fold_id, fold_rows = fit_predict_minirocket_torch_oof(
            X,
            y,
            folds,
            seed=args.seed,
            batch_size=args.batch_size,
            stats_batch_size=args.stats_batch_size,
            epochs=args.torch_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device_name=args.device,
            standardize=args.standardize,
            allow_tf32=args.allow_tf32,
        )
        model_name = "fold_safe_torch_linear_logistic_head"
        classifier_params = {
            "backend": args.backend,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "batch_size": int(args.batch_size),
            "stats_batch_size": int(args.stats_batch_size),
            "epochs": int(args.torch_epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pos_weight": "fold_train_negative_positive_ratio",
            "device": args.device,
            "standardize": args.standardize,
            "allow_tf32": bool(args.allow_tf32),
        }
    else:
        y_prob, fold_id, fold_rows = fit_predict_minirocket_sgd_oof(
            X,
            y,
            folds,
            seed=args.seed,
            max_iter=args.max_iter,
            tol=args.tol,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
        )
        model_name = "fold_safe_one_vs_rest_sgd_logistic_regression"
        classifier_params = {
            "backend": args.backend,
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": float(args.alpha),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "class_weight": "balanced",
        }
    metrics = multilabel_metrics(y, y_prob, threshold=args.threshold)
    calibration = calibration_summary(y, y_prob, n_bins=args.n_bins)
    ci = {
        "macro_pr_auc": bootstrap_ci(y, y_prob, macro_pr_auc, n_boot=args.n_boot, seed=args.seed),
        "macro_roc_auc": bootstrap_ci(y, y_prob, macro_roc_auc, n_boot=args.n_boot, seed=args.seed),
        "f1_macro": bootstrap_ci(
            y,
            y_prob,
            lambda yt, yp: multilabel_metrics(yt, yp, threshold=args.threshold)["f1_macro"],
            n_boot=args.n_boot,
            seed=args.seed,
        ),
        "brier_macro": bootstrap_ci(
            y,
            y_prob,
            lambda yt, yp: calibration_summary(yt, yp, n_bins=args.n_bins)["brier_macro"],
            n_boot=args.n_boot,
            seed=args.seed,
        ),
        "ece_macro": bootstrap_ci(
            y,
            y_prob,
            lambda yt, yp: calibration_summary(yt, yp, n_bins=args.n_bins)["ece_macro"],
            n_boot=args.n_boot,
            seed=args.seed,
        ),
    }

    prediction_path = PREDICTION_DIR / "minirocket_only_oof_predictions.npz"
    np.savez_compressed(
        prediction_path,
        y_true=y.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        record_id=record_id.astype(np.int64),
        fold_id=fold_id.astype(np.int16),
        class_names=np.asarray(class_names),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(PROTOCOL),
        feature_contract=np.asarray("minirocket_raw"),
        feature_preprocessing=np.asarray(
            "fold_train_standardization" if args.standardize == "train_fold" else "none"
        ),
        threshold=np.asarray(float(args.threshold)),
        config_hash=np.asarray(CONFIG_HASH),
        git_commit=np.asarray(_git_output(["rev-parse", "HEAD"]) or ""),
        manuscript_ready=np.asarray(args.limit_records == 0),
    )

    per_class_path = TABLE_DIR / "table_minirocket_only_class_metrics.csv"
    fold_path = TABLE_DIR / "table_minirocket_only_fold_summary.csv"
    _save_csv(per_class_path, per_class_rows(y, y_prob, class_names, args.threshold))
    _save_csv(fold_path, fold_rows)

    load_info = {
        **oof_info,
        **cache_info,
        "freeze_contract": freeze_contract,
        "limit_records": int(args.limit_records),
    }
    summary = {
        "created_utc": _now_utc(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "dataset": "chapman_oof",
        "protocol": PROTOCOL,
        "feature_contract": "minirocket_raw",
        "feature_preprocessing": "fold_train_standardization" if args.standardize == "train_fold" else "none",
        "model": model_name,
        "classifier_params": classifier_params,
        "n_records": int(len(y)),
        "n_classes": int(y.shape[1]),
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "n_boot": int(args.n_boot),
        "metrics": metrics,
        "calibration": calibration,
        "bootstrap_ci": ci,
        "load_info": load_info,
        "artifacts": {
            "predictions_npz": str(prediction_path),
            "per_class_table": str(per_class_path),
            "fold_summary_table": str(fold_path),
        },
        "manuscript_ready": args.limit_records == 0,
    }
    summary_path = METRIC_DIR / "minirocket_only_baseline_summary.json"
    save_json(summary_path, _json_safe(summary))

    manifest = {
        "created_utc": _now_utc(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "protocol": PROTOCOL,
        "feature_contract": "minirocket_raw",
        "feature_preprocessing": "fold_train_standardization" if args.standardize == "train_fold" else "none",
        "freeze_contract": freeze_contract,
        "load_info": load_info,
        "artifacts": {
            "summary": str(summary_path),
            "predictions": str(prediction_path),
            "per_class_table": str(per_class_path),
            "fold_summary_table": str(fold_path),
        },
        "artifact_sha256": {
            "summary": sha256_file(summary_path),
            "predictions": sha256_file(prediction_path),
            "per_class_table": sha256_file(per_class_path),
            "fold_summary_table": sha256_file(fold_path),
        },
    }
    manifest_path = MANIFEST_DIR / "minirocket_only_baseline_manifest.json"
    save_json(manifest_path, _json_safe(manifest))

    print(
        json.dumps(
            {
                "status": True,
                "protocol": PROTOCOL,
                "metrics": metrics,
                "calibration": calibration,
                "outputs": summary["artifacts"],
            },
            indent=2,
        )
    )
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
