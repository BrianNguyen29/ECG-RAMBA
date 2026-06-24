"""Run a fold-safe Raw Mamba comparator under the frozen OOF protocol.

This runner trains the ECG-RAMBA raw-signal Mamba backbone from scratch after
structurally removing MiniRocket/PCA and HRV inputs. It is a fair comparator,
not an inference ablation of the already-trained full model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG, CONFIG_HASH  # noqa: E402
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
    save_csv,
    save_json,
    sha256_file,
)
from src.aggregation import POWER_MEAN_IMPLEMENTATION, aggregate_record_probabilities  # noqa: E402
from src.training_data import build_slice_index  # noqa: E402
from src.utils import AsymmetricLossMultiLabel, EMA  # noqa: E402


PROTOCOL = "raw_mamba_retrained_weighted_bce_same_folds_power_mean_v2_q3_threshold_0.5"
FEATURE_CONTRACT = "raw_ecg_12lead_mamba_only"
DEFAULT_OOF_PREDICTIONS = PREDICTION_DIR / "oof_final_ema_predictions.npz"
DEFAULT_FREEZE_MANIFEST = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
PREDICTION_PATH = PREDICTION_DIR / "raw_mamba_oof_predictions.npz"
SLICE_PREDICTION_PATH = PREDICTION_DIR / "raw_mamba_slice_predictions.npz"
SUMMARY_PATH = METRIC_DIR / "raw_mamba_baseline_summary.json"
MANIFEST_PATH = MANIFEST_DIR / "raw_mamba_baseline_manifest.json"
PER_CLASS_TABLE = TABLE_DIR / "table_raw_mamba_class_metrics.csv"
FOLD_TABLE = TABLE_DIR / "table_raw_mamba_fold_summary.csv"
FOLD_PREDICTION_DIR = PREDICTION_DIR / "folds"
FOLD_CACHE_STATUS_JSON = METRIC_DIR / "raw_mamba_fold_cache_status.json"
FOLD_CACHE_STATUS_TABLE = TABLE_DIR / "table_raw_mamba_fold_cache_status.csv"


def load_revision_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


resnet_helpers = load_revision_module("14_resnet1d_cnn_baseline.py", "_raw_mamba_resnet_helpers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF_PREDICTIONS)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--raw-cache", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=int(CONFIG["epochs"]))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=float(CONFIG["lr_max"]))
    parser.add_argument("--lr-min", type=float, default=float(CONFIG["lr_min"]))
    parser.add_argument("--weight-decay", type=float, default=float(CONFIG["weight_decay"]))
    parser.add_argument("--grad-clip", type=float, default=float(CONFIG["grad_clip"]))
    parser.add_argument(
        "--bce-pos-weight",
        choices=["fold", "none"],
        default="fold",
        help="Use fold-train negative/positive ratios as BCE pos_weight during warm-up.",
    )
    parser.add_argument("--asym-start-epoch", type=int, default=int(CONFIG["asym_start_epoch"]))
    parser.add_argument("--ema-decay", type=float, default=float(CONFIG["ema_decay"]))
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use torch AMP on CUDA.")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--only-folds",
        default="",
        help=(
            "Comma-separated fold numbers to train/cache only, e.g. '1' or '2,3'. "
            "Subset mode writes fold caches and exits before OOF aggregation/CI."
        ),
    )
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "revision" / "experimental" / "raw_mamba_checkpoints",
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


def npz_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def parse_only_folds(value: str) -> set[int]:
    text = str(value or "").strip()
    if not text:
        return set()
    folds: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"--only-folds accepts comma-separated fold numbers, got {value!r}")
        fold = int(token)
        if fold < 1:
            raise ValueError(f"Invalid fold in --only-folds: {fold}")
        folds.add(fold)
    if not folds:
        raise ValueError(f"--only-folds did not contain any fold numbers: {value!r}")
    return folds


def set_seed(seed: int) -> None:
    resnet_helpers.set_seed(seed)


def select_device(name: str) -> torch.device:
    return resnet_helpers.select_device(name)


def autocast_context(device: torch.device, args: argparse.Namespace):
    if device.type == "cuda" and args.amp:
        dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def make_scaler(device: torch.device, args: argparse.Namespace):
    enabled = device.type == "cuda" and args.amp and args.amp_dtype == "float16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_model() -> nn.Module:
    from src.model import ECGRambaV7Advanced

    ablation = {"no_rocket": True, "no_hrv": True, "no_fusion": True}
    return ECGRambaV7Advanced(cfg=CONFIG, ablation=ablation)


def zero_aux(batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    xh = torch.zeros(batch_size, int(CONFIG["hydra_dim"]), device=device, dtype=dtype)
    xhr = torch.zeros(batch_size, int(CONFIG["hrv_dim"]), device=device, dtype=dtype)
    return xh, xhr


def forward_raw_mamba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    xh, xhr = zero_aux(x.shape[0], x.device, x.dtype)
    return model(x, xh, xhr, use_rocket=False, use_hrv=False, use_fusion=False)


def predict_slice_probabilities(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, record_ids, starts = [], [], []
    with torch.no_grad():
        for x, _y, rid, start in loader:
            x = x.to(device, non_blocking=True)
            with autocast_context(device, args):
                logits = forward_raw_mamba(model, x)
            batch_prob = torch.sigmoid(logits).float().detach().cpu().numpy().astype(np.float32)
            probs.append(batch_prob)
            record_ids.append(np.asarray(rid, dtype=np.int64))
            starts.append(np.asarray(start, dtype=np.int32))
    return (
        np.concatenate(probs, axis=0),
        np.concatenate(record_ids, axis=0),
        np.concatenate(starts, axis=0),
    )


def record_probabilities_for_fold(
    y: np.ndarray,
    slice_prob: np.ndarray,
    slice_record_id: np.ndarray,
    val_indices: np.ndarray,
) -> np.ndarray:
    y_prob_all, valid_mask, _slice_count = aggregate_record_probabilities(
        slice_prob,
        slice_record_id,
        n_records=len(y),
        q=float(CONFIG["power_mean_q"]),
    )
    missing = sorted(set(int(x) for x in val_indices) - set(np.where(valid_mask)[0].astype(int)))
    if missing:
        raise RuntimeError(f"Validation records without slice predictions: {missing[:10]}")
    return y_prob_all


def prediction_diagnostics(y_prob: np.ndarray, threshold: float) -> dict:
    return {
        "prob_min": float(np.min(y_prob)),
        "prob_max": float(np.max(y_prob)),
        "prob_mean": float(np.mean(y_prob)),
        "prob_std": float(np.std(y_prob)),
        "pred_positive_rate": float(np.mean(y_prob >= threshold)),
    }


def record_metrics_for_fold(
    y: np.ndarray,
    slice_prob: np.ndarray,
    slice_record_id: np.ndarray,
    val_indices: np.ndarray,
    threshold: float,
) -> dict:
    y_prob_all = record_probabilities_for_fold(y, slice_prob, slice_record_id, val_indices)
    return multilabel_metrics(y[val_indices], y_prob_all[val_indices], threshold=threshold)


def fold_prediction_path(fold: int) -> Path:
    return FOLD_PREDICTION_DIR / f"raw_mamba_fold{fold}_predictions.npz"


def fold_prediction_matches(
    path: Path,
    *,
    y: np.ndarray,
    val_indices: np.ndarray,
    load_info: dict,
    args: argparse.Namespace,
    model_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"fold", "val_indices", "y_prob", "slice_count", "slice_prob", "slice_record_id", "slice_start"}
            missing = required - set(data.files)
            if missing:
                print(f"Fold cache rejected {path}: missing {sorted(missing)}", flush=True)
                return None
            if str(npz_scalar(data, "protocol", "")) != PROTOCOL:
                return None
            if str(npz_scalar(data, "oof_predictions_sha256", "")) != load_info["oof_predictions_sha256"]:
                return None
            if str(npz_scalar(data, "raw_cache_sha256", "")) != load_info["raw_cache_sha256"]:
                return None
            if int(npz_scalar(data, "epochs", -1)) != int(args.epochs):
                return None
            expected_model_params = json.dumps(_json_safe(model_params), sort_keys=True)
            if str(npz_scalar(data, "model_params_json", "")) != expected_model_params:
                return None
            cached_val = np.asarray(data["val_indices"], dtype=np.int64)
            if not np.array_equal(cached_val, val_indices):
                return None
            y_prob = np.asarray(data["y_prob"], dtype=np.float32)
            slice_count = np.asarray(data["slice_count"], dtype=np.int16)
            slice_prob = np.asarray(data["slice_prob"], dtype=np.float32)
            slice_record_id = np.asarray(data["slice_record_id"], dtype=np.int64)
            slice_start = np.asarray(data["slice_start"], dtype=np.int32)
        if y_prob.shape != (len(y), y.shape[1]) or len(slice_count) != len(y):
            return None
        if slice_prob.ndim != 2 or slice_prob.shape[1] != y.shape[1]:
            return None
        if len(slice_record_id) != len(slice_prob) or len(slice_start) != len(slice_prob):
            return None
        if np.any(slice_record_id < 0) or np.any(slice_record_id >= len(y)):
            return None
        if not np.all(np.isfinite(y_prob)) or not np.all(np.isfinite(slice_prob)):
            return None
        if float(np.min(y_prob)) < -1e-6 or float(np.max(y_prob)) > 1.0 + 1e-6:
            return None
        if float(np.min(slice_prob)) < -1e-6 or float(np.max(slice_prob)) > 1.0 + 1e-6:
            return None
        reconstructed_counts = np.bincount(slice_record_id, minlength=len(y)).astype(np.int16)
        if not np.array_equal(reconstructed_counts, slice_count):
            print("Fold cache rejected: slice_count does not match slice artifact", flush=True)
            return None
        return y_prob, slice_count, slice_prob, slice_record_id, slice_start
    except Exception as exc:
        print(f"Fold cache rejected {path}: {exc!r}", flush=True)
        return None


def save_fold_predictions(
    path: Path,
    *,
    fold: int,
    val_indices: np.ndarray,
    y_prob: np.ndarray,
    slice_count: np.ndarray,
    slice_prob: np.ndarray,
    slice_record_id: np.ndarray,
    slice_start: np.ndarray,
    load_info: dict,
    args: argparse.Namespace,
    model_params: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fold=np.asarray(int(fold)),
        protocol=np.asarray(PROTOCOL),
        feature_contract=np.asarray(FEATURE_CONTRACT),
        val_indices=val_indices.astype(np.int64),
        y_prob=y_prob.astype(np.float32),
        slice_count=slice_count.astype(np.int16),
        slice_prob=slice_prob.astype(np.float32),
        slice_record_id=slice_record_id.astype(np.int64),
        slice_start=slice_start.astype(np.int32),
        oof_predictions_sha256=np.asarray(load_info["oof_predictions_sha256"]),
        raw_cache_sha256=np.asarray(load_info["raw_cache_sha256"]),
        model_params_json=np.asarray(json.dumps(_json_safe(model_params), sort_keys=True)),
        epochs=np.asarray(int(args.epochs)),
        seed=np.asarray(int(args.seed)),
        fold_seed=np.asarray(int(args.seed) + int(fold)),
        weights_kind=np.asarray("ema" if args.ema_decay > 0 else "raw"),
    )
    print(f"Wrote fold prediction cache: {path}", flush=True)


def fold_cache_status_rows(
    *,
    folds: list[dict[str, np.ndarray]],
    y: np.ndarray,
    load_info: dict,
    args: argparse.Namespace,
    model_params: dict,
    selected_folds: set[int],
) -> list[dict]:
    rows = []
    for split in folds:
        fold = int(split["fold"])
        va_idx = np.asarray(split["va_idx"], dtype=np.int64)
        path = fold_prediction_path(fold)
        reusable = fold_prediction_matches(
            path,
            y=y,
            val_indices=va_idx,
            load_info=load_info,
            args=args,
            model_params=model_params,
        )
        rows.append(
            {
                "fold": fold,
                "selected_in_this_run": bool(fold in selected_folds),
                "cache_ready": bool(reusable is not None),
                "path": str(path),
                "validation_records": int(len(va_idx)),
                "size_bytes": int(path.stat().st_size) if path.exists() else None,
            }
        )
    return rows


def write_fold_cache_status(
    *,
    folds: list[dict[str, np.ndarray]],
    y: np.ndarray,
    load_info: dict,
    args: argparse.Namespace,
    model_params: dict,
    selected_folds: set[int],
    fold_rows: list[dict],
) -> dict:
    rows = fold_cache_status_rows(
        folds=folds,
        y=y,
        load_info=load_info,
        args=args,
        model_params=model_params,
        selected_folds=selected_folds,
    )
    save_csv(FOLD_CACHE_STATUS_TABLE, rows)
    ready_folds = [int(row["fold"]) for row in rows if row["cache_ready"]]
    missing_folds = [int(row["fold"]) for row in rows if not row["cache_ready"]]
    payload = {
        "created_utc": _now_utc(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "selected_folds": sorted(int(x) for x in selected_folds),
        "ready_folds": ready_folds,
        "missing_folds": missing_folds,
        "all_folds_ready": len(missing_folds) == 0,
        "fold_cache_table": str(FOLD_CACHE_STATUS_TABLE),
        "fold_rows": fold_rows,
        "note": (
            "Subset mode intentionally writes fold caches only. Run again without "
            "--only-folds and with --reuse-predictions after all five folds are ready "
            "to create canonical OOF predictions, summary, manifest, and paired evidence."
        ),
    }
    save_json(FOLD_CACHE_STATUS_JSON, _json_safe(payload))
    return payload


def train_one_fold(
    *,
    fold: int,
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    load_info: dict,
    model_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    fold_seed = int(args.seed) + int(fold)
    set_seed(fold_seed)
    fold_path = fold_prediction_path(fold)
    if not args.force_rerun:
        reusable = fold_prediction_matches(
            fold_path,
            y=y,
            val_indices=va_idx,
            load_info=load_info,
            args=args,
            model_params=model_params,
        )
        if reusable is not None:
            y_prob, slice_count, slice_prob, slice_record_id, slice_start = reusable
            fold_metrics = multilabel_metrics(y[va_idx], y_prob[va_idx], threshold=args.threshold)
            print(f"Fold {fold}: reused fold prediction cache {fold_path}", flush=True)
            return y_prob, slice_count, slice_prob, slice_record_id, slice_start, {
                "fold": int(fold),
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "train_slices": None,
                "validation_slices": int(len(slice_record_id)),
                "reused_fold_predictions": True,
                "final_epoch": int(args.epochs),
                "final_weights_kind": "ema" if args.ema_decay > 0 else "raw",
                **{f"final_{k}": v for k, v in fold_metrics.items()},
            }

    print("=" * 80, flush=True)
    print(f"Raw Mamba fold {fold}/5 | train={len(tr_idx)} | val={len(va_idx)}", flush=True)
    slice_length = int(CONFIG["slice_length"])
    slice_stride = int(CONFIG["slice_stride"])
    max_slices = int(CONFIG["max_slices_per_record"])

    train_record_ids, train_starts, _train_pos, train_skipped = build_slice_index(
        tr_idx,
        X,
        slice_length=slice_length,
        slice_stride=slice_stride,
        max_slices_per_record=max_slices,
    )
    val_record_ids, val_starts, _val_pos, val_skipped = build_slice_index(
        va_idx,
        X,
        slice_length=slice_length,
        slice_stride=slice_stride,
        max_slices_per_record=max_slices,
    )
    if train_skipped or val_skipped:
        raise RuntimeError(f"Unexpected skipped slices: train={train_skipped}, val={val_skipped}")
    print(
        f"Fold {fold}: train_slices={len(train_record_ids)} val_slices={len(val_record_ids)} "
        f"slice_length={slice_length} stride={slice_stride}",
        flush=True,
    )

    train_ds = resnet_helpers.RawECGSliceDataset(X, y, train_record_ids, train_starts, slice_length=slice_length)
    val_ds = resnet_helpers.RawECGSliceDataset(X, y, val_record_ids, val_starts, slice_length=slice_length)
    train_loader = resnet_helpers.build_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed + fold,
        device=device,
    )
    val_loader = resnet_helpers.build_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1000 + fold,
        device=device,
    )

    model = build_model().to(device)
    bce_pos_weight = None
    if args.bce_pos_weight == "fold":
        bce_pos_weight = resnet_helpers.pos_weight_from_labels(y[tr_idx]).to(device)
        print(
            f"Fold {fold}: BCE pos_weight enabled "
            f"min={float(bce_pos_weight.min().detach().cpu()):.3f} "
            f"mean={float(bce_pos_weight.mean().detach().cpu()):.3f} "
            f"max={float(bce_pos_weight.max().detach().cpu()):.3f}",
            flush=True,
        )
    else:
        print(f"Fold {fold}: BCE pos_weight disabled", flush=True)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
    asym_criterion = AsymmetricLossMultiLabel(
        gamma_neg=float(CONFIG["asym_gamma_neg"]),
        gamma_pos=float(CONFIG["asym_gamma_pos"]),
        clip=float(CONFIG["asym_clip"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(args.lr_min),
    )
    ema = EMA(model, decay=float(args.ema_decay)) if args.ema_decay > 0 else None
    scaler = make_scaler(device, args)

    epoch_rows = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        use_bce = epoch <= int(args.asym_start_epoch)
        for x, target, _rid, _start in train_loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args):
                logits = forward_raw_mamba(model, x)
                loss = bce_criterion(logits, target) if use_bce else asym_criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Fold {fold} epoch {epoch} produced non-finite loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            losses.append(float(loss.detach().cpu().item()))
        scheduler.step()

        row = {
            "fold": int(fold),
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss_name": "BCE" if use_bce else "ASYM",
            "train_loss": float(np.mean(losses)) if losses else math.nan,
        }
        eval_with_ema = ema is not None and epoch > int(args.asym_start_epoch)
        if args.eval_every > 0 and (epoch == args.epochs or epoch % args.eval_every == 0):
            if eval_with_ema:
                ema.apply_shadow(model)
            try:
                slice_prob, slice_record_id, _slice_start = predict_slice_probabilities(
                    model,
                    val_loader,
                    device=device,
                    args=args,
                )
                val_prob_all = record_probabilities_for_fold(y, slice_prob, slice_record_id, va_idx)
                val_prob = val_prob_all[va_idx]
                val_metrics = multilabel_metrics(y[va_idx], val_prob, threshold=args.threshold)
                val_diag = prediction_diagnostics(val_prob, args.threshold)
            finally:
                if eval_with_ema:
                    ema.restore(model)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_diag.items()})
            row["val_weights_kind"] = "ema" if eval_with_ema else "raw"
            print(
                f"Fold {fold} Ep {epoch:03d}/{args.epochs} "
                f"{row['loss_name']} loss={row['train_loss']:.5f} "
                f"Eval={row['val_weights_kind']} F1={val_metrics['f1_macro']:.4f} "
                f"PR={val_metrics['pr_auc_macro']:.4f} ROC={val_metrics['roc_auc_macro']:.4f} "
                f"Pmean={val_diag['prob_mean']:.4f} Pstd={val_diag['prob_std']:.4f} "
                f"P>=thr={val_diag['pred_positive_rate']:.4f}",
                flush=True,
            )
        else:
            print(
                f"Fold {fold} Ep {epoch:03d}/{args.epochs} {row['loss_name']} loss={row['train_loss']:.5f}",
                flush=True,
            )
        epoch_rows.append(row)

    final_weights_kind = "ema" if ema is not None else "raw"
    if ema is not None:
        ema.apply_shadow(model)
    try:
        slice_prob, slice_record_id, slice_start = predict_slice_probabilities(
            model,
            val_loader,
            device=device,
            args=args,
        )
    finally:
        if ema is not None:
            ema.restore(model)

    y_prob_all, valid_mask, slice_count = aggregate_record_probabilities(
        slice_prob,
        slice_record_id,
        n_records=len(y),
        q=float(CONFIG["power_mean_q"]),
    )
    missing = sorted(set(int(x) for x in va_idx) - set(np.where(valid_mask)[0].astype(int)))
    if missing:
        raise RuntimeError(f"Fold {fold} missing validation predictions: {missing[:10]}")
    final_metrics = multilabel_metrics(y[va_idx], y_prob_all[va_idx], threshold=args.threshold)

    if args.save_checkpoints:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.checkpoint_dir / f"fold{fold}_raw_mamba_final_{final_weights_kind}.pt"
        if ema is not None:
            ema.apply_shadow(model)
        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold": int(fold),
                    "protocol": PROTOCOL,
                    "feature_contract": FEATURE_CONTRACT,
                    "weights_kind": final_weights_kind,
                    "args": vars(args),
                    "load_info": _json_safe(load_info),
                    "final_metrics": _json_safe(final_metrics),
                },
                checkpoint_path,
            )
        finally:
            if ema is not None:
                ema.restore(model)
        print(f"Wrote fold checkpoint: {checkpoint_path}", flush=True)

    save_fold_predictions(
        fold_path,
        fold=fold,
        val_indices=va_idx,
        y_prob=y_prob_all,
        slice_count=slice_count,
        slice_prob=slice_prob,
        slice_record_id=slice_record_id,
        slice_start=slice_start,
        load_info=load_info,
        args=args,
        model_params=model_params,
    )

    fold_summary = {
        "fold": int(fold),
        "train_records": int(len(tr_idx)),
        "validation_records": int(len(va_idx)),
        "train_slices": int(len(train_record_ids)),
        "validation_slices": int(len(val_record_ids)),
        "reused_fold_predictions": False,
        "final_epoch": int(args.epochs),
        "fold_seed": int(fold_seed),
        "final_weights_kind": final_weights_kind,
        "epoch_rows_json": json.dumps(_json_safe(epoch_rows), sort_keys=True),
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }
    return y_prob_all, slice_count, slice_prob, slice_record_id, slice_start, fold_summary


def _prediction_metadata(*, args: argparse.Namespace, load_info: dict, model_params: dict) -> dict:
    return {
        "dataset": np.asarray("chapman_oof"),
        "protocol": np.asarray(PROTOCOL),
        "feature_contract": np.asarray(FEATURE_CONTRACT),
        "aggregation_implementation": np.asarray(POWER_MEAN_IMPLEMENTATION),
        "aggregation_method": np.asarray("power_mean"),
        "power_mean_q": np.asarray(float(CONFIG["power_mean_q"])),
        "threshold": np.asarray(float(args.threshold)),
        "config_hash": np.asarray(CONFIG_HASH),
        "git_commit": np.asarray(_git_output(["rev-parse", "HEAD"]) or ""),
        "manuscript_ready": np.asarray(args.limit_records == 0),
        "model": np.asarray("raw_mamba_retrained_baseline"),
        "model_params_json": np.asarray(json.dumps(_json_safe(model_params), sort_keys=True)),
        "oof_predictions_sha256": np.asarray(load_info.get("oof_predictions_sha256", "")),
        "freeze_manifest_sha256": np.asarray(
            (load_info.get("freeze_contract") or {}).get("freeze_manifest_sha256", "")
        ),
        "raw_cache_sha256": np.asarray(load_info.get("raw_cache_sha256", "")),
        "dataset_record_order_fingerprint": np.asarray(load_info.get("dataset_record_order_fingerprint", "")),
        "weights_kind": np.asarray("ema" if args.ema_decay > 0 else "raw"),
    }


def write_prediction_npz(
    path: Path,
    *,
    y: np.ndarray,
    y_prob: np.ndarray,
    record_id: np.ndarray,
    fold_id: np.ndarray,
    slice_count: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
    load_info: dict,
    model_params: dict,
) -> None:
    print(f"Writing Raw Mamba predictions: {path}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=y.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        record_id=record_id.astype(np.int64),
        fold_id=fold_id.astype(np.int16),
        slice_count=slice_count.astype(np.int16),
        class_names=np.asarray(class_names),
        **_prediction_metadata(args=args, load_info=load_info, model_params=model_params),
    )


def write_slice_prediction_npz(
    path: Path,
    *,
    slice_prob: np.ndarray,
    slice_record_id: np.ndarray,
    slice_fold_id: np.ndarray,
    slice_start: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
    load_info: dict,
    model_params: dict,
) -> None:
    print(f"Writing Raw Mamba slice predictions: {path}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        slice_prob=slice_prob.astype(np.float32),
        slice_record_id=slice_record_id.astype(np.int64),
        slice_fold_id=slice_fold_id.astype(np.int16),
        slice_start=slice_start.astype(np.int32),
        class_names=np.asarray(class_names),
        **_prediction_metadata(args=args, load_info=load_info, model_params=model_params),
    )


def load_existing_prediction_npz(
    path: Path,
    *,
    y: np.ndarray,
    expected_fold_id: np.ndarray,
    record_id: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
    load_info: dict,
    model_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    if not SLICE_PREDICTION_PATH.exists():
        print(f"Existing Raw Mamba NPZ rejected: missing paired slice artifact {SLICE_PREDICTION_PATH}", flush=True)
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"y_true", "y_prob", "record_id", "fold_id", "slice_count", "class_names"}
            missing = required - set(data.files)
            if missing:
                print(f"Existing Raw Mamba NPZ rejected: missing {sorted(missing)}", flush=True)
                return None
            if str(npz_scalar(data, "protocol", "")) != PROTOCOL:
                return None
            if str(npz_scalar(data, "model_params_json", "")) != json.dumps(_json_safe(model_params), sort_keys=True):
                return None
            if str(npz_scalar(data, "oof_predictions_sha256", "")) != load_info["oof_predictions_sha256"]:
                return None
            if str(npz_scalar(data, "raw_cache_sha256", "")) != load_info["raw_cache_sha256"]:
                return None
            y_existing = np.asarray(data["y_true"], dtype=np.float32)
            record_existing = np.asarray(data["record_id"], dtype=np.int64)
            fold_existing = np.asarray(data["fold_id"], dtype=np.int16)
            y_prob_existing = np.asarray(data["y_prob"], dtype=np.float32)
            slice_count_existing = np.asarray(data["slice_count"], dtype=np.int16)
            classes_existing = np.asarray(data["class_names"]).astype(str).tolist()
            if not np.array_equal(y_existing, y):
                return None
            if not np.array_equal(record_existing, record_id):
                return None
            if not np.array_equal(fold_existing, expected_fold_id):
                return None
            if classes_existing != class_names:
                return None
            if y_prob_existing.shape != y.shape:
                return None
            if len(slice_count_existing) != len(y) or np.any(slice_count_existing <= 0):
                return None
            if not np.all(np.isfinite(y_prob_existing)):
                return None
            if float(np.min(y_prob_existing)) < -1e-6 or float(np.max(y_prob_existing)) > 1.0 + 1e-6:
                return None
    except Exception as exc:
        print(f"Existing Raw Mamba NPZ rejected: {exc!r}", flush=True)
        return None

    try:
        with np.load(SLICE_PREDICTION_PATH, allow_pickle=False) as slice_data:
            required_slice = {"slice_prob", "slice_record_id", "slice_fold_id", "slice_start", "class_names"}
            missing_slice = required_slice - set(slice_data.files)
            if missing_slice:
                print(f"Existing Raw Mamba NPZ rejected: slice artifact missing {sorted(missing_slice)}", flush=True)
                return None
            if str(npz_scalar(slice_data, "protocol", "")) != PROTOCOL:
                return None
            if str(npz_scalar(slice_data, "feature_contract", "")) != FEATURE_CONTRACT:
                return None
            if str(npz_scalar(slice_data, "model_params_json", "")) != json.dumps(_json_safe(model_params), sort_keys=True):
                return None
            if str(npz_scalar(slice_data, "oof_predictions_sha256", "")) != load_info["oof_predictions_sha256"]:
                return None
            if str(npz_scalar(slice_data, "raw_cache_sha256", "")) != load_info["raw_cache_sha256"]:
                return None
            slice_prob = np.asarray(slice_data["slice_prob"], dtype=np.float32)
            slice_record_id = np.asarray(slice_data["slice_record_id"], dtype=np.int64)
            slice_fold_id = np.asarray(slice_data["slice_fold_id"], dtype=np.int16)
            slice_start = np.asarray(slice_data["slice_start"], dtype=np.int32)
            slice_classes = np.asarray(slice_data["class_names"]).astype(str).tolist()
        if slice_classes != class_names:
            return None
        if slice_prob.ndim != 2 or slice_prob.shape[1] != y.shape[1]:
            return None
        if len(slice_record_id) != len(slice_prob) or len(slice_fold_id) != len(slice_prob) or len(slice_start) != len(slice_prob):
            return None
        if np.any(slice_record_id < 0) or np.any(slice_record_id >= len(y)):
            return None
        if not np.array_equal(slice_fold_id, expected_fold_id[slice_record_id]):
            return None
        if not np.all(np.isfinite(slice_prob)):
            return None
        if float(np.min(slice_prob)) < -1e-6 or float(np.max(slice_prob)) > 1.0 + 1e-6:
            return None
        reconstructed_counts = np.bincount(slice_record_id, minlength=len(y)).astype(np.int16)
        if not np.array_equal(reconstructed_counts, slice_count_existing):
            print("Existing Raw Mamba NPZ rejected: slice_count does not match slice artifact", flush=True)
            return None
    except Exception as exc:
        print(f"Existing Raw Mamba NPZ rejected: could not validate slice artifact: {exc!r}", flush=True)
        return None

    return y_prob_existing, fold_existing, slice_count_existing


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    set_seed(args.seed)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = select_device(args.device)
    print("=" * 80, flush=True)
    print("RAW MAMBA FAIR COMPARATOR UNDER FROZEN OOF", flush=True)
    print("=" * 80, flush=True)
    print(f"protocol={PROTOCOL}", flush=True)
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"epochs={args.epochs} bce_epochs={args.asym_start_epoch} "
        f"asym_start={args.asym_start_epoch + 1} ema_decay={args.ema_decay}",
        flush=True,
    )

    freeze_contract = resnet_helpers.validate_oof_freeze_contract(
        freeze_manifest=args.freeze_manifest,
        oof_predictions=args.oof_predictions,
        expected_checkpoint_kind=args.expected_checkpoint_kind,
    )
    y, fold_id, record_id, class_names, folds, oof_info = resnet_helpers.load_oof_labels_and_folds(
        args.oof_predictions,
        limit_records=args.limit_records,
    )
    if oof_info["oof_records_total"] != int(freeze_contract["validated_records"]):
        raise ValueError(
            "OOF prediction record count does not match freeze manifest: "
            f"{oof_info['oof_records_total']} != {freeze_contract['validated_records']}"
        )
    if int(args.limit_records) == 0 and oof_info["fold_count"] != 5:
        raise ValueError(f"Canonical Raw Mamba baseline requires five folds, got {oof_info['fold_count']}")
    record_fingerprint = (
        oof_info.get("dataset_record_order_fingerprint")
        or freeze_contract.get("dataset_record_order_fingerprint")
        or ""
    )
    if not record_fingerprint:
        raise ValueError("Frozen OOF artifacts must carry dataset_record_order_fingerprint.")

    X, cache_info = resnet_helpers.load_raw_cache(
        expected_y=y,
        expected_record_fingerprint=record_fingerprint,
        explicit_cache=args.raw_cache,
        limit_records=args.limit_records,
    )
    if len(X) != len(y):
        raise ValueError(f"Raw ECG/y length mismatch after loading: {len(X)} vs {len(y)}")

    load_info = {
        **oof_info,
        **cache_info,
        "freeze_contract": freeze_contract,
        "limit_records": int(args.limit_records),
    }
    model_params = {
        "architecture": "ecg_ramba_raw_mamba_structural_ablation",
        "ablation": {"no_rocket": True, "no_hrv": True, "no_fusion": True},
        "input_shape": [12, int(CONFIG["slice_length"])],
        "slice_length": int(CONFIG["slice_length"]),
        "slice_stride": int(CONFIG["slice_stride"]),
        "max_slices_per_record": int(CONFIG["max_slices_per_record"]),
        "aggregation_method": "power_mean",
        "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
        "power_mean_q": float(CONFIG["power_mean_q"]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "fold_seed_formula": "seed + fold",
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "allow_tf32": bool(args.allow_tf32),
        "lr": float(args.lr),
        "lr_min": float(args.lr_min),
        "weight_decay": float(args.weight_decay),
        "loss": "bce_then_asymmetric",
        "bce_pos_weight": str(args.bce_pos_weight),
        "bce_pos_weight_formula": (
            "fold_train_negative_positive_ratio_clipped_1_100"
            if args.bce_pos_weight == "fold"
            else "none"
        ),
        "asym_start_epoch": int(args.asym_start_epoch),
        "asym_gamma_neg": float(CONFIG["asym_gamma_neg"]),
        "asym_gamma_pos": float(CONFIG["asym_gamma_pos"]),
        "asym_clip": float(CONFIG["asym_clip"]),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "selection_rule": "fixed_final_epoch",
        "weights_kind": "ema" if args.ema_decay > 0 else "raw",
        "ema_decay": float(args.ema_decay),
        "uses_hrv": False,
        "uses_minirocket": False,
        "uses_pca": False,
        "uses_ecg_ramba_checkpoints": False,
        "training_from_scratch": True,
    }

    selected_folds = parse_only_folds(args.only_folds)
    available_folds = {int(split["fold"]) for split in folds}
    invalid_folds = sorted(selected_folds - available_folds)
    if invalid_folds:
        raise ValueError(f"--only-folds contains fold(s) not present in frozen OOF: {invalid_folds}")
    if selected_folds:
        print(
            "Subset fold-cache mode enabled. "
            f"Selected folds: {sorted(selected_folds)}. "
            "Canonical OOF aggregation/CI will be skipped in this run.",
            flush=True,
        )
        if args.n_boot:
            print("--n-boot is ignored in subset fold-cache mode.", flush=True)

    if args.reuse_predictions and not args.force_rerun and not selected_folds:
        reusable = load_existing_prediction_npz(
            PREDICTION_PATH,
            y=y,
            expected_fold_id=fold_id,
            record_id=record_id,
            class_names=class_names,
            args=args,
            load_info=load_info,
            model_params=model_params,
        )
    else:
        reusable = None

    fold_rows = []
    all_slice_prob, all_slice_record_id, all_slice_start, all_slice_fold_id = [], [], [], []
    if reusable is not None:
        y_prob, fold_id_out, slice_count = reusable
        print(f"Reusing existing Raw Mamba predictions: {PREDICTION_PATH}", flush=True)
        for fold in sorted(int(x) for x in np.unique(fold_id_out) if int(x) > 0):
            va_idx = np.where(fold_id_out == fold)[0]
            fold_metrics = multilabel_metrics(y[va_idx], y_prob[va_idx], threshold=args.threshold)
            fold_seed = int(args.seed) + int(fold)
            fold_rows.append(
                {
                    "fold": int(fold),
                    "train_records": int(np.sum(fold_id_out != fold)),
                    "validation_records": int(len(va_idx)),
                    "train_slices": None,
                    "validation_slices": int(np.sum(slice_count[va_idx])),
                    "reused_fold_predictions": True,
                    "final_epoch": int(args.epochs),
                    "fold_seed": int(fold_seed),
                    "final_weights_kind": "ema" if args.ema_decay > 0 else "raw",
                    **{f"final_{k}": v for k, v in fold_metrics.items()},
                }
            )
    else:
        y_prob = np.zeros_like(y, dtype=np.float32)
        slice_count = np.zeros(len(y), dtype=np.int16)
        fold_id_out = np.zeros(len(y), dtype=np.int16)
        splits_to_run = [
            split for split in folds if not selected_folds or int(split["fold"]) in selected_folds
        ]
        for split in splits_to_run:
            fold = int(split["fold"])
            tr_idx = np.asarray(split["tr_idx"], dtype=np.int64)
            va_idx = np.asarray(split["va_idx"], dtype=np.int64)
            fold_prob, fold_slice_count, slice_prob, slice_record_id, slice_start, fold_summary = train_one_fold(
                fold=fold,
                X=X,
                y=y,
                tr_idx=tr_idx,
                va_idx=va_idx,
                device=device,
                args=args,
                load_info=load_info,
                model_params=model_params,
            )
            y_prob[va_idx] = fold_prob[va_idx]
            slice_count[va_idx] = fold_slice_count[va_idx]
            fold_id_out[va_idx] = fold
            all_slice_prob.append(slice_prob)
            all_slice_record_id.append(slice_record_id)
            all_slice_start.append(slice_start)
            all_slice_fold_id.append(np.full(len(slice_record_id), fold, dtype=np.int16))
            fold_rows.append(fold_summary)

        if selected_folds:
            status_payload = write_fold_cache_status(
                folds=folds,
                y=y,
                load_info=load_info,
                args=args,
                model_params=model_params,
                selected_folds=selected_folds,
                fold_rows=fold_rows,
            )
            print(
                json.dumps(
                    {
                        "status": True,
                        "mode": "fold_cache_only",
                        "selected_folds": status_payload["selected_folds"],
                        "ready_folds": status_payload["ready_folds"],
                        "missing_folds": status_payload["missing_folds"],
                        "all_folds_ready": status_payload["all_folds_ready"],
                        "next_step": (
                            "Publish the revision artifacts mirror now. After all five folds are ready, "
                            "rerun without --only-folds and with --reuse-predictions to aggregate."
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )
            print(f"Wrote: {FOLD_CACHE_STATUS_JSON}", flush=True)
            print(f"Wrote: {FOLD_CACHE_STATUS_TABLE}", flush=True)
            return

        if np.any(fold_id_out <= 0):
            raise RuntimeError("OOF fold_id coverage incomplete after Raw Mamba training.")
        if np.any(slice_count <= 0):
            raise RuntimeError("Some records have no Raw Mamba slice predictions.")
        write_prediction_npz(
            PREDICTION_PATH,
            y=y,
            y_prob=y_prob,
            record_id=record_id,
            fold_id=fold_id_out,
            slice_count=slice_count,
            class_names=class_names,
            args=args,
            load_info=load_info,
            model_params=model_params,
        )
        if all_slice_prob:
            write_slice_prediction_npz(
                SLICE_PREDICTION_PATH,
                slice_prob=np.concatenate(all_slice_prob, axis=0),
                slice_record_id=np.concatenate(all_slice_record_id, axis=0),
                slice_fold_id=np.concatenate(all_slice_fold_id, axis=0),
                slice_start=np.concatenate(all_slice_start, axis=0),
                class_names=class_names,
                args=args,
                load_info=load_info,
                model_params=model_params,
            )

    print("Computing point metrics...", flush=True)
    metrics = multilabel_metrics(y, y_prob, threshold=args.threshold)
    print("Computing calibration metrics...", flush=True)
    calibration = calibration_summary(y, y_prob, n_bins=args.n_bins)
    print(f"Computing bootstrap CI with n_boot={args.n_boot}; this stage is CPU-bound.", flush=True)

    def _bootstrap_metric(name: str, fn):
        print(f"  bootstrap {name} start", flush=True)
        result = bootstrap_ci(y, y_prob, fn, n_boot=args.n_boot, seed=args.seed)
        print(f"  bootstrap {name} done: {result}", flush=True)
        return result

    ci = {
        "macro_pr_auc": _bootstrap_metric("macro_pr_auc", macro_pr_auc),
        "macro_roc_auc": _bootstrap_metric("macro_roc_auc", macro_roc_auc),
        "f1_macro": _bootstrap_metric(
            "f1_macro",
            lambda yt, yp: multilabel_metrics(yt, yp, threshold=args.threshold)["f1_macro"],
        ),
        "brier_macro": _bootstrap_metric(
            "brier_macro",
            lambda yt, yp: calibration_summary(yt, yp, n_bins=args.n_bins)["brier_macro"],
        ),
        "ece_macro": _bootstrap_metric(
            "ece_macro",
            lambda yt, yp: calibration_summary(yt, yp, n_bins=args.n_bins)["ece_macro"],
        ),
    }

    save_csv(PER_CLASS_TABLE, resnet_helpers.per_class_rows(y, y_prob, class_names, args.threshold))
    save_csv(FOLD_TABLE, fold_rows)

    summary = {
        "created_utc": _now_utc(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "dataset": "chapman_oof",
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "model": "raw_mamba_retrained_baseline",
        "model_params": model_params,
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
            "predictions_npz": str(PREDICTION_PATH),
            "slice_predictions_npz": str(SLICE_PREDICTION_PATH) if SLICE_PREDICTION_PATH.exists() else None,
            "per_class_table": str(PER_CLASS_TABLE),
            "fold_summary_table": str(FOLD_TABLE),
        },
        "manuscript_ready": args.limit_records == 0,
    }
    save_json(SUMMARY_PATH, _json_safe(summary))

    artifact_sha256 = {
        "summary": sha256_file(SUMMARY_PATH),
        "predictions": sha256_file(PREDICTION_PATH),
        "per_class_table": sha256_file(PER_CLASS_TABLE),
        "fold_summary_table": sha256_file(FOLD_TABLE),
    }
    if SLICE_PREDICTION_PATH.exists():
        artifact_sha256["slice_predictions"] = sha256_file(SLICE_PREDICTION_PATH)
    manifest = {
        "created_utc": _now_utc(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "freeze_contract": freeze_contract,
        "load_info": load_info,
        "model_params": model_params,
        "artifacts": summary["artifacts"],
        "artifact_sha256": artifact_sha256,
    }
    save_json(MANIFEST_PATH, _json_safe(manifest))

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
        ),
        flush=True,
    )
    print(f"Wrote: {SUMMARY_PATH}", flush=True)
    print(f"Wrote: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
