"""Generate reviewer-ready prediction NPZ files.

Current scope:
    - oof: Chapman out-of-fold predictions using fold checkpoints.

Run from repo root on Colab:
    python scripts/revision/01_generate_predictions.py --dataset oof

Outputs:
    reports/revision/predictions/oof_full_predictions.npz
    reports/revision/predictions/oof_full_slice_predictions.npz
    reports/revision/metrics/oof_full_prediction_summary.json

Prediction files follow the artifact contract in docs/revision_plan/.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG, CONFIG_HASH, PATHS, DEVICE  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    LOG_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    ensure_revision_dirs,
    multilabel_metrics,
    power_mean,
    save_csv,
    save_json,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: str | os.PathLike[str], *, with_sha256: bool = False) -> dict:
    p = Path(path)
    payload = {
        "path": str(p),
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
        "modified_utc": datetime.fromtimestamp(
            p.stat().st_mtime, tz=timezone.utc
        ).isoformat() if p.exists() and p.is_file() else None,
    }
    if with_sha256 and p.exists() and p.is_file():
        payload["sha256"] = sha256_file(p)
    return payload


def git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def runtime_metadata(args: argparse.Namespace, created_utc: str) -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "created_utc": created_utc,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "args": vars(args),
        "project_root": str(PROJECT_ROOT),
        "cwd": str(Path.cwd()),
        "git": {
            "commit": git_output(["rev-parse", "HEAD"]),
            "branch": git_output(["branch", "--show-current"]),
            "status_short": git_output(["status", "--short", "--branch"]),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": DEVICE,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        },
    }


def config_snapshot() -> dict:
    keys = [
        "d_model",
        "n_layers",
        "hydra_dim",
        "hrv_dim",
        "slice_length",
        "slice_stride",
        "max_slices_per_record",
        "default_threshold",
        "aggregation_method",
        "n_folds",
        "cv_strategy",
        "group_key",
        "hydra_pca_mode",
        "use_hrv",
        "use_rocket",
        "use_cross_attention_fusion",
    ]
    return {
        "config_hash": CONFIG_HASH,
        "core": {key: CONFIG.get(key) for key in keys},
        "classes": CLASSES,
        "paths": {key: path_info(value) for key, value in PATHS.items()},
    }


def slice_ordinals(record_ids: np.ndarray) -> np.ndarray:
    counts: dict[int, int] = {}
    out = np.zeros(len(record_ids), dtype=np.int16)
    for i, rid in enumerate(record_ids):
        rid_int = int(rid)
        out[i] = counts.get(rid_int, 0)
        counts[rid_int] = int(out[i]) + 1
    return out


def summarize_slice_counts(slice_counts: np.ndarray) -> dict:
    counts = np.asarray(slice_counts)
    nonzero = counts[counts > 0]
    hist = {str(int(k)): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    return {
        "min": int(counts.min()) if len(counts) else 0,
        "max": int(counts.max()) if len(counts) else 0,
        "mean": float(counts.mean()) if len(counts) else 0.0,
        "median": float(np.median(counts)) if len(counts) else 0.0,
        "nonzero_mean": float(nonzero.mean()) if len(nonzero) else 0.0,
        "zero_slice_records": int(np.sum(counts == 0)),
        "histogram": hist,
    }


def per_class_summary_rows(
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
                "prob_mean": float(np.mean(yp)),
                "prob_min": float(np.min(yp)),
                "prob_max": float(np.max(yp)),
                "roc_auc": float(roc_auc_score(yt, yp)) if has_both else np.nan,
                "pr_auc": float(average_precision_score(yt, yp)) if has_both else np.nan,
                "f1": float(f1_score(yt, pred, zero_division=0)),
                "precision": float(precision_score(yt, pred, zero_division=0)),
                "recall": float(recall_score(yt, pred, zero_division=0)),
            }
        )
    return rows


def index_fingerprint(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]


def hydra_fold_cache_path(fold_num: int, tr_idx: np.ndarray, va_idx: np.ndarray) -> Path:
    cache_dir = Path(PATHS["cache_dir"]) / "revision_feature_cache"
    train_hash = index_fingerprint(tr_idx)
    val_hash = index_fingerprint(va_idx)
    name = (
        f"hydra_oof_{CONFIG_HASH}_fold{fold_num}_"
        f"train{len(tr_idx)}_{train_hash}_val{len(va_idx)}_{val_hash}_"
        f"D{CONFIG['hydra_dim']}.npz"
    )
    return cache_dir / name


def load_or_compute_fold_hydra(
    *,
    fold_num: int,
    X_rocket_raw: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> tuple[np.ndarray, float, Path, bool]:
    from src.features import apply_pca, fit_pca_on_train

    cache_path = hydra_fold_cache_path(fold_num, tr_idx, va_idx)
    expected_shape = (len(va_idx), int(CONFIG["hydra_dim"]))
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            hydra_va = cached["hydra_va"]
            pca_var = float(cached["pca_explained_variance"])
            if hydra_va.shape == expected_shape:
                print(f"✅ Loaded fold-aware Hydra/PCA cache for fold {fold_num}: {cache_path}", flush=True)
                return hydra_va.astype(np.float32), pca_var, cache_path, True
            print(
                f"⚠️  Hydra cache shape mismatch for fold {fold_num}: "
                f"{hydra_va.shape} vs {expected_shape}. Recomputing.",
                flush=True,
            )
        except Exception as exc:
            print(f"⚠️  Could not load Hydra cache for fold {fold_num}: {exc}. Recomputing.", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Fitting fold-aware PCA for fold {fold_num}: "
        f"train={len(tr_idx)} x {X_rocket_raw.shape[1]} -> D={CONFIG['hydra_dim']}",
        flush=True,
    )
    start = time.time()
    pca = fit_pca_on_train(X_rocket_raw[tr_idx], CONFIG["hydra_dim"])
    print(f"Transforming validation Hydra features for fold {fold_num}: val={len(va_idx)}", flush=True)
    hydra_va = apply_pca(pca, X_rocket_raw[va_idx])
    pca_var = float(pca.explained_variance_ratio_.sum())
    elapsed = time.time() - start
    print(f"PCA fold {fold_num} finished in {elapsed / 60:.1f} min | variance={pca_var:.6f}", flush=True)
    print(f"Saving fold-aware Hydra/PCA cache: {cache_path}", flush=True)
    np.savez_compressed(
        cache_path,
        hydra_va=hydra_va.astype(np.float16),
        pca_explained_variance=np.asarray(pca_var, dtype=np.float32),
        fold=np.asarray(fold_num, dtype=np.int16),
        config_hash=np.asarray(CONFIG_HASH),
        train_index_hash=np.asarray(index_fingerprint(tr_idx)),
        val_index_hash=np.asarray(index_fingerprint(va_idx)),
    )
    return hydra_va.astype(np.float32), pca_var, cache_path, False


class ECGSliceDatasetInfer(Dataset):
    def __init__(self, xs: np.ndarray, xh: np.ndarray, xhr: np.ndarray, rids: np.ndarray):
        self.xs = xs
        self.xh = xh
        self.xhr = xhr
        self.rids = rids

    def __len__(self) -> int:
        return len(self.rids)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.xs[idx], dtype=torch.float32),
            torch.tensor(self.xh[idx], dtype=torch.float32),
            torch.tensor(self.xhr[idx], dtype=torch.float32),
            torch.tensor(self.rids[idx], dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["oof"], default="oof")
    parser.add_argument("--checkpoint-kind", choices=["best", "final"], default="best")
    parser.add_argument("--batch-size", type=int, default=max(1, int(CONFIG["batch_size"])))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(max(0, int(CONFIG.get("num_workers", 0))), 4),
        help="DataLoader workers for inference. Colab Drive runs are usually more stable with 0-4.",
    )
    parser.add_argument("--limit-records", type=int, default=0, help="Debug only. 0 means all records.")
    parser.add_argument("--save-slice-probs", action="store_true", default=True)
    parser.add_argument("--no-save-slice-probs", dest="save_slice_probs", action="store_false")
    return parser.parse_args()


def slice_record(x: np.ndarray) -> list[np.ndarray]:
    if x.shape[-1] < CONFIG["slice_length"]:
        return []
    slices = []
    for start in range(
        0,
        x.shape[-1] - CONFIG["slice_length"] + 1,
        CONFIG["slice_stride"],
    ):
        slices.append(x[..., start : start + CONFIG["slice_length"]])
        if len(slices) >= CONFIG["max_slices_per_record"]:
            break
    return slices


def load_folds(y: np.ndarray, subjects: np.ndarray) -> list[dict[str, np.ndarray]]:
    folds_path = Path(PATHS["model_dir"]) / "folds.pkl"
    if folds_path.exists():
        folds = joblib.load(folds_path)
        normalized = []
        for fold in folds:
            normalized.append(
                {
                    "tr_idx": np.asarray(fold["tr_idx"], dtype=np.int64),
                    "va_idx": np.asarray(fold["va_idx"], dtype=np.int64),
                }
            )
        print(f"Loaded folds from: {folds_path}")
        return normalized

    print("folds.pkl not found. Recomputing StratifiedGroupKFold from config.")
    from sklearn.model_selection import StratifiedGroupKFold

    y_strat = y.sum(axis=1).clip(max=3).astype(int)
    sgkf = StratifiedGroupKFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seeds"][0],
    )
    return [
        {"tr_idx": tr_idx.astype(np.int64), "va_idx": va_idx.astype(np.int64)}
        for tr_idx, va_idx in sgkf.split(np.zeros(len(y)), y_strat, groups=subjects)
    ]


def checkpoint_path(fold: int, checkpoint_kind: str) -> Path:
    preferred = Path(PATHS["model_dir"]) / f"fold{fold}_{checkpoint_kind}.pt"
    fallback_kind = "final" if checkpoint_kind == "best" else "best"
    fallback = Path(PATHS["model_dir"]) / f"fold{fold}_{fallback_kind}.pt"
    if preferred.exists():
        return preferred
    if fallback.exists():
        print(f"Checkpoint fallback for fold {fold}: {preferred.name} missing, using {fallback.name}")
        return fallback
    raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {preferred} or {fallback}")


def load_model_for_fold(
    fold: int,
    checkpoint_kind: str,
    checkpoint_file: Path | None = None,
) -> torch.nn.Module:
    from src.model import ECGRambaV7Advanced

    path = checkpoint_file or checkpoint_path(fold, checkpoint_kind)
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model = ECGRambaV7Advanced(cfg=CONFIG).to(DEVICE)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"Strict checkpoint load failed; retrying strict=False. Reason: {exc}")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def prepare_clean_chapman(limit_records: int = 0):
    from src.data_loader import load_chapman_multilabel

    print("Loading Chapman data")
    X, y, X_raw_amp, subjects = load_chapman_multilabel()
    subjects = np.asarray(subjects)

    min_samples = 5
    class_counts = y.sum(axis=0)
    keep_mask = class_counts >= min_samples
    if not keep_mask.all():
        print(f"Dropping {np.sum(~keep_mask)} classes with <{min_samples} samples")
        y = y[:, keep_mask]
        valid = y.sum(axis=1) > 0
        X, y, X_raw_amp, subjects = X[valid], y[valid], X_raw_amp[valid], subjects[valid]

    if y.shape[1] != len(CLASSES):
        raise ValueError(
            f"Class dimension mismatch after cleaning: y has {y.shape[1]}, config has {len(CLASSES)}"
        )

    if limit_records > 0:
        print(f"Debug limit enabled: first {limit_records} records only")
        X = X[:limit_records]
        y = y[:limit_records]
        X_raw_amp = X_raw_amp[:limit_records]
        subjects = subjects[:limit_records]

    print(f"Chapman records: {len(y)} | classes: {y.shape[1]}")
    return X, y.astype(np.float32), X_raw_amp, subjects


def build_fold_slices(
    va_idx: np.ndarray,
    X: np.ndarray,
    X_hrv: np.ndarray,
    hydra_va_by_record: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    xs, xh, xhr, rids = [], [], [], []
    slice_counts: dict[int, int] = {}

    for rid in va_idx:
        slices = slice_record(X[rid])
        slice_counts[int(rid)] = len(slices)
        for signal_slice in slices:
            xs.append(signal_slice)
            xh.append(hydra_va_by_record[int(rid)])
            xhr.append(X_hrv[rid])
            rids.append(int(rid))

    if not xs:
        return (
            np.empty((0, 12, CONFIG["slice_length"]), dtype=np.float32),
            np.empty((0, CONFIG["hydra_dim"]), dtype=np.float32),
            np.empty((0, CONFIG["hrv_dim"]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            slice_counts,
        )

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(xh, dtype=np.float32),
        np.asarray(xhr, dtype=np.float32),
        np.asarray(rids, dtype=np.int64),
        slice_counts,
    )


def infer_slices(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_probs, all_rids = [], []
    with torch.no_grad():
        for x, xh, xhr, rid in tqdm(loader, desc="Inference", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            xh = xh.to(DEVICE, non_blocking=True)
            xhr = xhr.to(DEVICE, non_blocking=True)

            if DEVICE == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(x, xh, xhr)
            else:
                logits = model(x, xh, xhr)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_probs.append(probs)
            all_rids.append(rid.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_rids, axis=0)


def make_inference_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": DEVICE == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def infer_with_retries(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, int, int, list[dict]]:
    """Run inference with conservative fallbacks for Colab GPU/DataLoader failures."""
    current_batch = max(1, int(batch_size))
    current_workers = max(0, int(num_workers))
    attempts: list[dict] = []

    while True:
        print(f"Inference attempt: batch_size={current_batch} | num_workers={current_workers}")
        loader = make_inference_loader(
            dataset,
            batch_size=current_batch,
            num_workers=current_workers,
        )
        try:
            probs, rids = infer_slices(model, loader)
            return probs, rids, current_batch, current_workers, attempts
        except RuntimeError as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            is_cuda_oom = DEVICE == "cuda" and (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "cuda out of memory" in msg_lower
                or "out of memory" in msg_lower
            )
            is_worker_error = "dataloader worker" in msg_lower or "worker exited unexpectedly" in msg_lower
            attempts.append(
                {
                    "batch_size": int(current_batch),
                    "num_workers": int(current_workers),
                    "error_type": type(exc).__name__,
                    "message": msg[:1000],
                }
            )
            del loader
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if is_cuda_oom and current_batch > 1:
                next_batch = max(1, current_batch // 2)
                print(
                    f"CUDA OOM at batch_size={current_batch}. "
                    f"Retrying with batch_size={next_batch}."
                )
                current_batch = next_batch
                continue
            if is_worker_error and current_workers > 0:
                print(
                    f"DataLoader worker failure with num_workers={current_workers}. "
                    "Retrying with num_workers=0."
                )
                current_workers = 0
                continue
            raise


def generate_oof(args: argparse.Namespace) -> None:
    from src.features import (
        apply_pca,
        fit_pca_on_train,
        generate_hrv_cache,
        generate_raw_rocket_cache,
    )
    from src.utils import set_seed

    ensure_revision_dirs()
    set_seed(CONFIG["seeds"][0])
    created_utc = datetime.now(timezone.utc).isoformat()
    run_meta = runtime_metadata(args, created_utc)
    config_meta = config_snapshot()

    X, y, X_raw_amp, subjects = prepare_clean_chapman(limit_records=args.limit_records)
    n_records, n_classes = y.shape

    X_rocket_raw = generate_raw_rocket_cache(X)
    X_hrv = generate_hrv_cache(X, X_raw_amp) if CONFIG["use_hrv"] else np.zeros(
        (n_records, CONFIG["hrv_dim"]), dtype=np.float32
    )
    folds = load_folds(y, subjects)

    if args.limit_records > 0:
        allowed = set(range(n_records))
        folds = [
            {
                "tr_idx": np.asarray([idx for idx in fold["tr_idx"] if int(idx) in allowed], dtype=np.int64),
                "va_idx": np.asarray([idx for idx in fold["va_idx"] if int(idx) in allowed], dtype=np.int64),
            }
            for fold in folds
        ]

    oof_probs = np.zeros((n_records, n_classes), dtype=np.float32)
    fold_id = np.zeros(n_records, dtype=np.int16) - 1
    record_slice_count = np.zeros(n_records, dtype=np.int16)
    pca_variance = []
    fold_summaries = []
    checkpoint_infos = []
    slice_probs_all = []
    slice_record_index_all = []
    slice_index_all = []
    slice_fold_id_all = []

    for fold_num, fold in enumerate(folds, start=1):
        tr_idx = fold["tr_idx"]
        va_idx = fold["va_idx"]
        if len(va_idx) == 0:
            print(f"Fold {fold_num}: no validation records after filtering; skipping")
            continue

        print("\n" + "=" * 80)
        print(f"Fold {fold_num}/{len(folds)} | train={len(tr_idx)} | val={len(va_idx)}")
        print("=" * 80)

        hydra_va, pca_var, pca_cache_path, pca_cache_hit = load_or_compute_fold_hydra(
            fold_num=fold_num,
            X_rocket_raw=X_rocket_raw,
            tr_idx=tr_idx,
            va_idx=va_idx,
        )
        pca_variance.append(
            {
                "fold": fold_num,
                "explained_variance": pca_var,
                "cache_path": str(pca_cache_path),
                "cache_hit": bool(pca_cache_hit),
            }
        )
        print(f"PCA variance retained: {pca_var:.6f}")

        hydra_va_by_record = {
            int(record_idx): hydra_va[offset]
            for offset, record_idx in enumerate(va_idx)
        }
        xs, xh, xhr, rids, slice_counts = build_fold_slices(
            va_idx=va_idx,
            X=X,
            X_hrv=X_hrv,
            hydra_va_by_record=hydra_va_by_record,
        )
        for rid, count in slice_counts.items():
            record_slice_count[rid] = count

        fold_summary = {
            "fold": fold_num,
            "n_train": int(len(tr_idx)),
            "n_validation": int(len(va_idx)),
            "n_slices": int(len(rids)),
            "n_records_with_slices": int(sum(count > 0 for count in slice_counts.values())),
            "n_zero_slice_records": int(sum(count == 0 for count in slice_counts.values())),
            "pca_explained_variance": pca_var,
            "pca_cache_path": str(pca_cache_path),
            "pca_cache_hit": bool(pca_cache_hit),
        }

        if len(rids) == 0:
            print(f"Fold {fold_num}: no slices generated; skipping")
            fold_summary["n_predicted_records"] = 0
            fold_summaries.append(fold_summary)
            continue

        ckpt_path = checkpoint_path(fold_num, args.checkpoint_kind)
        ckpt_info = {"fold": fold_num, **path_info(ckpt_path, with_sha256=True)}
        checkpoint_infos.append(ckpt_info)
        fold_summary["checkpoint"] = ckpt_info

        model = load_model_for_fold(fold_num, args.checkpoint_kind, checkpoint_file=ckpt_path)
        infer_dataset = ECGSliceDatasetInfer(xs, xh, xhr, rids)
        slice_probs, slice_rids, actual_batch_size, actual_num_workers, inference_attempts = infer_with_retries(
            model,
            infer_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        for rid in va_idx:
            mask = slice_rids == int(rid)
            if not np.any(mask):
                continue
            preds = slice_probs[mask]
            preds = preds[np.isfinite(preds).all(axis=1)]
            if len(preds) == 0:
                continue
            oof_probs[int(rid)] = power_mean(preds, q=3.0, axis=0)
            fold_id[int(rid)] = fold_num

        fold_summary["n_predicted_records"] = int(np.sum(fold_id[va_idx] >= 0))
        fold_summary["slice_prob_shape"] = list(slice_probs.shape)
        fold_summary["slice_prob_nan_count"] = int(np.isnan(slice_probs).sum())
        fold_summary["slice_prob_min"] = float(np.nanmin(slice_probs))
        fold_summary["slice_prob_max"] = float(np.nanmax(slice_probs))
        fold_summary["requested_batch_size"] = int(args.batch_size)
        fold_summary["actual_batch_size"] = int(actual_batch_size)
        fold_summary["requested_num_workers"] = int(args.num_workers)
        fold_summary["actual_num_workers"] = int(actual_num_workers)
        fold_summary["inference_retry_attempts"] = inference_attempts
        fold_summaries.append(fold_summary)

        if args.save_slice_probs:
            slice_probs_all.append(slice_probs.astype(np.float16))
            slice_record_index_all.append(slice_rids.astype(np.int64))
            slice_index_all.append(slice_ordinals(slice_rids))
            slice_fold_id_all.append(np.full(len(slice_rids), fold_num, dtype=np.int16))

        del model
        torch.cuda.empty_cache()
        gc.collect()

    valid_records = fold_id >= 0
    if not np.all(valid_records):
        missing = int(np.sum(~valid_records))
        print(f"Warning: {missing} records did not receive predictions")

    git_commit = run_meta["git"]["commit"] or ""
    protocol = "fold_best_power_mean_q3_threshold_0.5"
    threshold = 0.5
    aggregation_q = 3.0

    out_path = PREDICTION_DIR / "oof_full_predictions.npz"
    np.savez_compressed(
        out_path,
        y_true=y,
        y_prob=oof_probs,
        record_id=np.arange(n_records, dtype=np.int64),
        class_names=np.asarray(CLASSES),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(protocol),
        fold_id=fold_id,
        slice_count=record_slice_count,
        valid_record_mask=valid_records.astype(np.bool_),
        config_hash=np.asarray(CONFIG_HASH),
        git_commit=np.asarray(git_commit),
        created_utc=np.asarray(created_utc),
        checkpoint_kind=np.asarray(args.checkpoint_kind),
        batch_size=np.asarray(args.batch_size, dtype=np.int32),
        aggregation_method=np.asarray("power_mean"),
        aggregation_q=np.asarray(aggregation_q, dtype=np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
    )
    print(f"Wrote: {out_path}")

    slice_out_path = None
    if args.save_slice_probs and slice_probs_all:
        slice_out_path = PREDICTION_DIR / "oof_full_slice_predictions.npz"
        np.savez_compressed(
            slice_out_path,
            slice_prob=np.concatenate(slice_probs_all, axis=0),
            record_id=np.concatenate(slice_record_index_all, axis=0),
            slice_index=np.concatenate(slice_index_all, axis=0),
            fold_id=np.concatenate(slice_fold_id_all, axis=0),
            class_names=np.asarray(CLASSES),
            dataset=np.asarray("chapman_oof"),
            protocol=np.asarray("slice_level_fold_best"),
            config_hash=np.asarray(CONFIG_HASH),
            git_commit=np.asarray(git_commit),
            created_utc=np.asarray(created_utc),
        )
        print(f"Wrote: {slice_out_path}")

    metrics = multilabel_metrics(y[valid_records], oof_probs[valid_records], threshold=threshold)
    class_summary_path = TABLE_DIR / "oof_full_class_summary.csv"
    class_rows = per_class_summary_rows(
        y[valid_records],
        oof_probs[valid_records],
        CLASSES,
        threshold=threshold,
    )
    save_csv(class_summary_path, class_rows)
    print(f"Wrote: {class_summary_path}")

    manifest_path = MANIFEST_DIR / "oof_full_prediction_run_manifest.json"
    summary_path = METRIC_DIR / "oof_full_prediction_summary.json"

    output_paths = {
        "prediction_file": out_path,
        "slice_prediction_file": slice_out_path,
        "prediction_summary_json": summary_path,
        "class_summary_csv": class_summary_path,
    }

    summary = {
        "dataset": "chapman_oof",
        "created_utc": created_utc,
        "git_commit": git_commit,
        "config_hash": CONFIG_HASH,
        "prediction_file": str(out_path),
        "slice_prediction_file": str(slice_out_path) if slice_out_path else None,
        "class_summary_csv": str(class_summary_path),
        "run_manifest_json": str(manifest_path),
        "protocol": protocol,
        "n_records": int(n_records),
        "n_valid_predictions": int(np.sum(valid_records)),
        "n_missing_predictions": int(np.sum(~valid_records)),
        "n_classes": int(n_classes),
        "class_names": CLASSES,
        "checkpoint_kind_requested": args.checkpoint_kind,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "limit_records": int(args.limit_records),
        "save_slice_probs": bool(args.save_slice_probs),
        "aggregation": {"method": "power_mean", "q": aggregation_q},
        "threshold": threshold,
        "slice_count_summary": summarize_slice_counts(record_slice_count),
        "pca_variance": pca_variance,
        "fold_summaries": fold_summaries,
        "checkpoints": checkpoint_infos,
        "metrics": metrics,
    }
    save_json(summary_path, summary)

    output_info = {
        name: path_info(path, with_sha256=True)
        for name, path in output_paths.items()
        if path is not None and Path(path).exists()
    }
    run_manifest = {
        "dataset": "chapman_oof",
        "created_utc": created_utc,
        "protocol": protocol,
        "runtime": run_meta,
        "config": config_meta,
        "inputs": {
            "dataset_paths": {
                key: path_info(PATHS[key])
                for key in ["zip_path", "data_cache", "model_dir"]
                if key in PATHS
            },
            "checkpoints": checkpoint_infos,
        },
        "outputs": output_info,
        "fold_summaries": fold_summaries,
        "slice_count_summary": summarize_slice_counts(record_slice_count),
        "prediction_quality": {
            "y_prob_shape": list(oof_probs.shape),
            "y_prob_min": float(np.nanmin(oof_probs[valid_records])),
            "y_prob_max": float(np.nanmax(oof_probs[valid_records])),
            "y_prob_nan_count": int(np.isnan(oof_probs).sum()),
            "valid_record_count": int(np.sum(valid_records)),
            "missing_record_count": int(np.sum(~valid_records)),
        },
    }
    save_json(manifest_path, run_manifest)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {manifest_path}")


def main() -> None:
    args = parse_args()
    if args.dataset == "oof":
        generate_oof(args)
    else:
        raise ValueError(args.dataset)


def write_exception_report(exc: BaseException) -> None:
    try:
        ensure_revision_dirs()
        created_utc = datetime.now(timezone.utc).isoformat()
        stamp = created_utc.replace(":", "").replace("+", "_")
        trace = traceback.format_exc()
        text_path = LOG_DIR / "01_generate_predictions_last_error.txt"
        json_path = LOG_DIR / "01_generate_predictions_last_error.json"
        stamped_path = LOG_DIR / f"01_generate_predictions_error_{stamp}.txt"
        text_path.write_text(trace, encoding="utf-8")
        stamped_path.write_text(trace, encoding="utf-8")
        save_json(
            json_path,
            {
                "created_utc": created_utc,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "command": " ".join([Path(sys.executable).name, *sys.argv]),
                "traceback_txt": str(text_path),
                "stamped_traceback_txt": str(stamped_path),
            },
        )
        print("")
        print("=" * 80)
        print("Prediction script failed. Error report written:")
        print(f"  {text_path}")
        print(f"  {json_path}")
        print("=" * 80)
    except Exception as report_exc:
        print(f"Failed to write exception report: {report_exc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_exception_report(exc)
        raise
