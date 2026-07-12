"""Frozen random-convolution morphology MLP-head control under frozen OOF.

This optional reviewer-facing runner tests whether the fixed-seed ROCKET-family
random-convolution morphology branch benefits from a nonlinear head. It reuses
the same fold-safe MAX+PPV feature cache and frozen OOF contract as the legacy
linear random-transform baseline, but replaces the linear logistic head with a
small MLP. It does not alter the ECG-RAMBA checkpoint and must be interpreted
only as morphology-branch head-capacity sensitivity evidence; it does not make
the convolution kernels learnable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG_HASH  # noqa: E402
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
    save_npz_compressed_atomic,
    sha256_file,
)


PROTOCOL = "fixed_seed_rocket_family_max_ppv_mlp_head_same_folds_threshold_0.5"
FEATURE_CONTRACT = "fixed_seed_rocket_family_random_convolution_max_ppv_frozen_mlp_head"
EVALUATED_TRANSFORM_NAME = "fixed_seed_rocket_family_random_convolution_max_ppv"
FOLD_CACHE_DIR = PREDICTION_DIR / "folds"
HELPER_PATH = PROJECT_ROOT / "scripts" / "revision" / "10_minirocket_only_baseline.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("_hybrid_minirocket_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MiniRocket helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_hybrid_minirocket_helpers"] = module
    spec.loader.exec_module(module)
    return module


helpers = load_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--stats-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--standardize", choices=["train_fold", "none"], default="train_fold")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--only-folds",
        default="",
        help="Comma-separated folds to train/cache only; rerun without this flag to aggregate all folds.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist exact fold checkpoints (default: enabled; required for reusable reviewer evidence).",
    )
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "revision" / "experimental" / "hybrid_morphology_checkpoints",
    )
    parser.add_argument(
        "--fold-cache-dir",
        type=Path,
        default=FOLD_CACHE_DIR,
        help=(
            "Directory for resumable per-fold prediction caches. Point this to the "
            "canonical Drive mirror so completed folds survive Colab disconnects."
        ),
    )
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--oof-predictions", type=Path, default=helpers.DEFAULT_OOF_PREDICTIONS)
    parser.add_argument("--freeze-manifest", type=Path, default=helpers.DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--minirocket-cache", type=Path, default=None)
    parser.add_argument("--allow-legacy-shape-cache", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def resolve_device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_only_folds(value: str) -> set[int]:
    folds = {int(item.strip()) for item in str(value).split(",") if item.strip()}
    invalid = sorted(fold for fold in folds if fold < 1 or fold > 5)
    if invalid:
        raise ValueError(f"Invalid --only-folds values: {invalid}")
    return folds


def fold_cache_path(fold: int) -> Path:
    return FOLD_CACHE_DIR / f"hybrid_morphology_fold{fold}_predictions.npz"


def fold_checkpoint_path(args: argparse.Namespace, fold: int) -> Path:
    return Path(args.checkpoint_dir) / f"fold{fold}_hybrid_morphology_final.pt"


def save_torch_atomic(path: Path, payload: dict) -> None:
    """Prevent a disconnect from leaving a checkpoint that looks reusable."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_checkpoint_contract(args: argparse.Namespace) -> dict:
    rows = []
    missing = []
    for fold in range(1, 6):
        path = fold_checkpoint_path(args, fold)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(fold)
            continue
        rows.append(
            {
                "fold": fold,
                "path": helpers._project_relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "status": "complete" if not missing and len(rows) == 5 else "partial",
        "expected_folds": [1, 2, 3, 4, 5],
        "missing_folds": missing,
        "checkpoints": rows,
    }


def require_complete_checkpoint_contract(checkpoint_contract: dict) -> list[dict]:
    rows = sorted(
        [row for row in checkpoint_contract.get("checkpoints", []) if isinstance(row, dict)],
        key=lambda row: int(row.get("fold", -1)),
    )
    folds = [int(row.get("fold", -1)) for row in rows]
    hashes = [str(row.get("sha256", "")) for row in rows]
    if (
        checkpoint_contract.get("status") != "complete"
        or folds != [1, 2, 3, 4, 5]
        or any(len(value) != 64 for value in hashes)
    ):
        raise RuntimeError(
            "Hybrid morphology canonical predictions require five exact saved checkpoints; "
            f"missing_folds={checkpoint_contract.get('missing_folds', [])}. "
            "Use --save-checkpoints for training or restore the checkpoint directory before reuse."
        )
    return rows


def checkpoint_contract_metadata(checkpoint_contract: dict) -> dict:
    rows = require_complete_checkpoint_contract(checkpoint_contract)
    return {
        "checkpoint_folds": np.asarray([int(row["fold"]) for row in rows], dtype=np.int16),
        "checkpoint_sha256": np.asarray([str(row["sha256"]) for row in rows]),
    }


def prediction_checkpoint_contract_matches(data, checkpoint_contract: dict) -> bool:
    expected = checkpoint_contract_metadata(checkpoint_contract)
    required = set(expected)
    if not required.issubset(data.files):
        print(
            "Existing Hybrid morphology artifact rejected: missing checkpoint provenance "
            f"{sorted(required - set(data.files))}",
            flush=True,
        )
        return False
    return all(np.array_equal(np.asarray(data[key]), value) for key, value in expected.items())


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def metadata(
    args: argparse.Namespace,
    load_info: dict,
    model_name: str,
    classifier_params: dict,
    checkpoint_contract: dict,
) -> dict:
    return {
        "dataset": np.asarray("chapman_oof"),
        "protocol": np.asarray(PROTOCOL),
        "feature_contract": np.asarray(FEATURE_CONTRACT),
        "evaluated_transform_name": np.asarray(EVALUATED_TRANSFORM_NAME),
        "canonical_minirocket": np.asarray(False),
        "feature_preprocessing": np.asarray(
            "fold_train_standardization" if args.standardize == "train_fold" else "none"
        ),
        "threshold": np.asarray(float(args.threshold)),
        "config_hash": np.asarray(CONFIG_HASH),
        "git_commit": np.asarray(git_output(["rev-parse", "HEAD"]) or ""),
        "manuscript_ready": np.asarray(args.limit_records == 0),
        "model": np.asarray(model_name),
        "classifier_params_json": np.asarray(json.dumps(json_safe(classifier_params), sort_keys=True)),
        "oof_predictions_sha256": np.asarray(load_info.get("oof_predictions_sha256", "")),
        "freeze_manifest_sha256": np.asarray((load_info.get("freeze_contract") or {}).get("freeze_manifest_sha256", "")),
        "minirocket_cache_sha256": np.asarray(load_info.get("minirocket_cache_sha256", "")),
        "dataset_record_order_fingerprint": np.asarray(load_info.get("dataset_record_order_fingerprint", "")),
        **checkpoint_contract_metadata(checkpoint_contract),
    }


def load_existing(
    path: Path,
    y: np.ndarray,
    record_id: np.ndarray,
    class_names: list[str],
    args,
    load_info,
    model_name,
    classifier_params,
    checkpoint_contract: dict,
):
    if not path.exists():
        return None
    print(f"Checking reusable Hybrid morphology prediction NPZ: {path}", flush=True)
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "fold_id", "class_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Reusable hybrid prediction NPZ missing keys: {sorted(missing)}")
        expected_meta = metadata(
            args,
            load_info,
            model_name,
            classifier_params,
            checkpoint_contract,
        )
        advisory = {"git_commit"}
        for key, expected in expected_meta.items():
            if key not in data.files:
                raise KeyError(f"Reusable hybrid prediction NPZ lacks metadata key: {key}")
            if key in advisory:
                continue
            actual_value = data[key].item() if np.ndim(data[key]) == 0 else data[key]
            expected_value = expected.item() if np.ndim(expected) == 0 else expected
            if str(actual_value) != str(expected_value):
                raise ValueError(f"Reusable hybrid metadata mismatch for {key}: {actual_value} != {expected_value}")
        y_existing = np.asarray(data["y_true"], dtype=np.float32)
        y_prob = np.asarray(data["y_prob"], dtype=np.float32)
        rid_existing = np.asarray(data["record_id"], dtype=np.int64)
        fold_id = np.asarray(data["fold_id"], dtype=np.int16)
        classes_existing = np.asarray(data["class_names"]).astype(str).tolist()
        if not prediction_checkpoint_contract_matches(data, checkpoint_contract):
            return None
    if not np.array_equal(y_existing, y.astype(np.float32)):
        raise ValueError("Reusable hybrid prediction y_true differs from frozen OOF labels.")
    if not np.array_equal(rid_existing, record_id.astype(np.int64)):
        raise ValueError("Reusable hybrid prediction record_id differs from frozen OOF.")
    if classes_existing != class_names:
        raise ValueError("Reusable hybrid prediction class_names differ from frozen OOF.")
    if len(np.unique(fold_id[fold_id > 0])) != 5 and int(args.limit_records) == 0:
        raise ValueError("Reusable hybrid prediction does not cover all five folds.")
    print("Reusable Hybrid morphology prediction NPZ passed contract checks.", flush=True)
    return np.clip(y_prob, 0.0, 1.0).astype(np.float32), fold_id


def load_reusable_fold_cache(
    *,
    cache_path: Path,
    checkpoint_path: Path,
    val_indices: np.ndarray,
    n_classes: int,
    params_json: str,
    load_info: dict,
) -> np.ndarray | None:
    if not cache_path.exists() or not checkpoint_path.exists():
        return None
    try:
        expected_checkpoint_sha256 = sha256_file(checkpoint_path)
        with np.load(cache_path, allow_pickle=False) as cached:
            required = {
                "protocol",
                "classifier_params_json",
                "oof_predictions_sha256",
                "minirocket_cache_sha256",
                "checkpoint_sha256",
                "val_indices",
                "y_prob",
            }
            missing = required - set(cached.files)
            if missing:
                print(
                    f"Fold cache rejected {cache_path}: missing {sorted(missing)}",
                    flush=True,
                )
                return None
            valid = (
                str(cached["protocol"].item()) == PROTOCOL
                and str(cached["classifier_params_json"].item()) == params_json
                and str(cached["oof_predictions_sha256"].item())
                == load_info.get("oof_predictions_sha256")
                and str(cached["minirocket_cache_sha256"].item())
                == load_info.get("minirocket_cache_sha256")
                and str(cached["checkpoint_sha256"].item()) == expected_checkpoint_sha256
                and np.array_equal(np.asarray(cached["val_indices"], dtype=np.int64), val_indices)
            )
            cached_prob = np.asarray(cached["y_prob"], dtype=np.float32)
        if (
            not valid
            or cached_prob.shape != (len(val_indices), n_classes)
            or not np.isfinite(cached_prob).all()
            or float(np.min(cached_prob)) < -1e-6
            or float(np.max(cached_prob)) > 1.0 + 1e-6
        ):
            print(f"Fold cache rejected {cache_path}: contract/value mismatch", flush=True)
            return None
        return np.clip(cached_prob, 0.0, 1.0).astype(np.float32)
    except Exception as exc:
        print(f"Fold cache rejected {cache_path}: {exc!r}", flush=True)
        return None


def fit_predict_mlp_oof(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    *,
    args: argparse.Namespace,
    load_info: dict,
    classifier_params: dict,
):
    import torch
    import torch.nn as nn

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.benchmark = False
    print(
        f"Frozen-transform morphology MLP device={device} batch_size={args.batch_size} "
        f"epochs={args.epochs} hidden_dim={args.hidden_dim} dropout={args.dropout}",
        flush=True,
    )

    n_records, n_classes = y.shape
    n_features = int(X.shape[1])
    y_prob = np.full((n_records, n_classes), np.nan, dtype=np.float32)
    fold_id_out = np.full(n_records, -1, dtype=np.int16)
    fold_rows: list[dict] = []

    selected_folds = parse_only_folds(args.only_folds)
    params_json = json.dumps(json_safe(classifier_params), sort_keys=True)
    for fold in folds:
        fold_num = int(fold["fold"])
        if selected_folds and fold_num not in selected_folds:
            continue
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        cache_path = fold_cache_path(fold_num)
        checkpoint_path = fold_checkpoint_path(args, fold_num)
        cached_prob = None
        if not args.force_rerun:
            cached_prob = load_reusable_fold_cache(
                cache_path=cache_path,
                checkpoint_path=checkpoint_path,
                val_indices=va_idx,
                n_classes=n_classes,
                params_json=params_json,
                load_info=load_info,
            )
        if cached_prob is not None:
            checkpoint_sha256 = sha256_file(checkpoint_path)
            y_prob[va_idx] = cached_prob
            fold_id_out[va_idx] = fold_num
            fold_rows.append(
                {
                    "fold": fold_num,
                    "train_records": int(len(tr_idx)),
                    "validation_records": int(len(va_idx)),
                    "validation_positive_labels": int(np.sum(y[va_idx])),
                    "classifier": "FrozenTransformMLP",
                    "evaluated_transform_name": EVALUATED_TRANSFORM_NAME,
                    "hidden_dim": int(args.hidden_dim),
                    "dropout": float(args.dropout),
                    "epochs": int(args.epochs),
                    "reused_fold_predictions": True,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
            print(f"Fold {fold_num}: reused fold cache {cache_path}", flush=True)
            continue
        rng = np.random.default_rng(int(args.seed) + fold_num)
        torch.manual_seed(int(args.seed) + fold_num)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed) + fold_num)
        print(f"Fold {fold_num}/5 | train={len(tr_idx)} | val={len(va_idx)} | features={n_features}", flush=True)

        checkpoint_payload = None
        if args.reuse_checkpoints and not args.force_rerun and checkpoint_path.exists():
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                not isinstance(checkpoint_payload, dict)
                or checkpoint_payload.get("protocol") != PROTOCOL
                or int(checkpoint_payload.get("fold", -1)) != fold_num
                or checkpoint_payload.get("classifier_params") != json_safe(classifier_params)
                or checkpoint_payload.get("oof_predictions_sha256")
                != load_info.get("oof_predictions_sha256")
                or checkpoint_payload.get("minirocket_cache_sha256")
                != load_info.get("minirocket_cache_sha256")
            ):
                raise ValueError(f"Hybrid checkpoint contract mismatch: {checkpoint_path}")

        if checkpoint_payload is not None:
            mean = np.asarray(checkpoint_payload["feature_mean"], dtype=np.float32)
            std = np.asarray(checkpoint_payload["feature_std"], dtype=np.float32)
        elif args.standardize == "train_fold":
            print(f"  fold {fold_num}: computing train-fold feature standardization", flush=True)
            mean, std = helpers.compute_train_standardization(
                X,
                tr_idx,
                batch_size=int(args.stats_batch_size),
            )
        else:
            mean = np.zeros(n_features, dtype=np.float32)
            std = np.ones(n_features, dtype=np.float32)

        model = nn.Sequential(
            nn.Linear(n_features, int(args.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(int(args.hidden_dim), n_classes),
        ).to(device)
        reused_checkpoint = checkpoint_payload is not None
        if reused_checkpoint:
            model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
            print(f"  fold {fold_num}: reused checkpoint {checkpoint_path}", flush=True)
        else:
            pos = np.maximum(y[tr_idx].sum(axis=0), 1.0)
            neg = np.maximum(len(tr_idx) - y[tr_idx].sum(axis=0), 1.0)
            pos_weight = np.clip(neg / pos, 1.0, 100.0).astype(np.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device))
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

            def amp_context():
                return torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                total_loss = 0.0
                total_seen = 0
                for batch_idx in helpers.iter_index_batches(tr_idx, int(args.batch_size), shuffle=True, rng=rng):
                    xb = (np.asarray(X[batch_idx], dtype=np.float32) - mean) / std
                    xb_t = torch.as_tensor(xb, dtype=torch.float32, device=device)
                    yb_t = torch.as_tensor(y[batch_idx], dtype=torch.float32, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with amp_context():
                        loss = criterion(model(xb_t), yb_t)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += float(loss.detach().cpu()) * len(batch_idx)
                    total_seen += len(batch_idx)
                print(f"  fold {fold_num}: epoch {epoch:02d}/{args.epochs} loss={total_loss / max(total_seen, 1):.5f}", flush=True)

            if args.save_checkpoints:
                save_torch_atomic(
                    checkpoint_path,
                    {
                        "model_state_dict": model.state_dict(),
                        "fold": fold_num,
                        "protocol": PROTOCOL,
                        "classifier_params": json_safe(classifier_params),
                        "feature_mean": mean,
                        "feature_std": std,
                        "oof_predictions_sha256": load_info.get("oof_predictions_sha256"),
                        "minirocket_cache_sha256": load_info.get("minirocket_cache_sha256"),
                    },
                )
                print(f"  fold {fold_num}: wrote checkpoint {checkpoint_path}", flush=True)

        model.eval()
        out = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        offset = 0
        with torch.no_grad():
            for start in range(0, len(va_idx), int(args.batch_size)):
                batch_idx = va_idx[start:start + int(args.batch_size)]
                xb = (np.asarray(X[batch_idx], dtype=np.float32) - mean) / std
                xb_t = torch.as_tensor(xb, dtype=torch.float32, device=device)
                logits = model(xb_t)
                prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                out[offset:offset + len(batch_idx)] = prob
                offset += len(batch_idx)
        y_prob[va_idx] = out
        fold_id_out[va_idx] = fold_num
        if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
            raise RuntimeError(
                f"Fold {fold_num} has predictions but no durable checkpoint. "
                "Rerun with --save-checkpoints before publishing reviewer evidence."
            )
        checkpoint_sha256 = sha256_file(checkpoint_path)
        save_npz_compressed_atomic(
            cache_path,
            fold=np.asarray(fold_num, dtype=np.int16),
            protocol=np.asarray(PROTOCOL),
            classifier_params_json=np.asarray(params_json),
            oof_predictions_sha256=np.asarray(load_info.get("oof_predictions_sha256", "")),
            minirocket_cache_sha256=np.asarray(load_info.get("minirocket_cache_sha256", "")),
            checkpoint_sha256=np.asarray(checkpoint_sha256),
            val_indices=va_idx,
            y_prob=out,
        )
        print(f"  fold {fold_num}: wrote fold cache {cache_path}", flush=True)
        fold_rows.append(
            {
                "fold": fold_num,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "validation_positive_labels": int(np.sum(y[va_idx])),
                "classifier": "FrozenTransformMLP",
                "evaluated_transform_name": EVALUATED_TRANSFORM_NAME,
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "epochs": int(args.epochs),
                "reused_checkpoint": bool(reused_checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
            }
        )

    if not selected_folds and np.any(fold_id_out < 0):
        raise RuntimeError("Hybrid morphology OOF prediction coverage is incomplete.")
    if not selected_folds and not np.all(np.isfinite(y_prob)):
        raise RuntimeError("Hybrid morphology predictions contain non-finite values.")
    return np.clip(y_prob, 0.0, 1.0).astype(np.float32), fold_id_out, fold_rows


def write_prediction_npz(
    path: Path,
    *,
    y,
    y_prob,
    record_id,
    fold_id,
    class_names,
    args,
    load_info,
    model_name,
    classifier_params,
    checkpoint_contract,
) -> None:
    print(f"Writing Hybrid morphology predictions: {path}", flush=True)
    save_npz_compressed_atomic(
        path,
        y_true=y.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        record_id=record_id.astype(np.int64),
        fold_id=fold_id.astype(np.int16),
        class_names=np.asarray(class_names),
        **metadata(args, load_info, model_name, classifier_params, checkpoint_contract),
    )


def bootstrap_metric(name: str, y: np.ndarray, y_prob: np.ndarray, args, fn):
    print(f"  bootstrap {name} start", flush=True)
    result = bootstrap_ci(y, y_prob, fn, n_boot=int(args.n_boot), seed=int(args.seed))
    print(f"  bootstrap {name} done: {result}", flush=True)
    return result


def main() -> None:
    global FOLD_CACHE_DIR
    args = parse_args()
    args.device = resolve_device_name(args.device)
    args.checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir.is_absolute()
        else PROJECT_ROOT / args.checkpoint_dir
    ).resolve()
    if args.save_checkpoints or args.reuse_checkpoints:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    FOLD_CACHE_DIR = (
        args.fold_cache_dir
        if args.fold_cache_dir.is_absolute()
        else PROJECT_ROOT / args.fold_cache_dir
    ).resolve()
    FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_revision_dirs()
    print("=" * 80, flush=True)
    print("HYBRID FIXED-SEED ROCKET-FAMILY MORPHOLOGY MLP BASELINE", flush=True)
    print("=" * 80, flush=True)
    freeze_contract = helpers.validate_oof_freeze_contract(
        freeze_manifest=args.freeze_manifest,
        oof_predictions=args.oof_predictions,
        expected_checkpoint_kind=args.expected_checkpoint_kind,
    )
    y, fold_id, record_id, class_names, folds, oof_info = helpers.load_oof_labels_and_folds(
        args.oof_predictions,
        limit_records=int(args.limit_records),
    )
    record_fingerprint = oof_info.get("dataset_record_order_fingerprint") or freeze_contract.get(
        "dataset_record_order_fingerprint"
    )
    if not record_fingerprint:
        raise ValueError("Frozen OOF artifacts must carry dataset_record_order_fingerprint.")
    X, cache_info = helpers.load_minirocket_cache(
        n_records=oof_info["oof_records_total"],
        record_fingerprint=record_fingerprint,
        explicit_cache=args.minirocket_cache,
        allow_legacy_shape_cache=args.allow_legacy_shape_cache,
        limit_records=int(args.limit_records),
    )
    if len(X) != len(y):
        raise ValueError(f"MiniRocket/y length mismatch after loading: {len(X)} vs {len(y)}")
    load_info = {**oof_info, **cache_info, "freeze_contract": freeze_contract, "limit_records": int(args.limit_records)}

    prediction_path = PREDICTION_DIR / "hybrid_morphology_oof_predictions.npz"
    per_class_path = TABLE_DIR / "table_hybrid_morphology_class_metrics.csv"
    fold_path = TABLE_DIR / "table_hybrid_morphology_fold_summary.csv"
    summary_path = METRIC_DIR / "hybrid_morphology_baseline_summary.json"
    manifest_path = MANIFEST_DIR / "hybrid_morphology_baseline_manifest.json"
    model_name = "fold_safe_frozen_random_convolution_mlp_head"
    classifier_params = {
        "loss": "BCEWithLogitsLoss",
        "optimizer": "AdamW",
        "batch_size": int(args.batch_size),
        "stats_batch_size": int(args.stats_batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "pos_weight": "fold_train_negative_positive_ratio",
        "standardize": args.standardize,
        "evaluated_transform_name": EVALUATED_TRANSFORM_NAME,
        "canonical_minirocket": False,
        "seed": int(args.seed),
        "allow_tf32": bool(args.allow_tf32),
    }
    runtime_config = {
        "device": args.device,
        "allow_tf32": bool(args.allow_tf32),
    }

    reusable = None
    selected_folds = parse_only_folds(args.only_folds)
    checkpoint_contract = build_checkpoint_contract(args)
    if (
        not selected_folds
        and int(args.limit_records) == 0
        and checkpoint_contract.get("status") != "complete"
        and not args.save_checkpoints
    ):
        raise RuntimeError(
            "Hybrid morphology canonical run has no complete checkpoint contract. "
            "Restore the five checkpoints or rerun with --save-checkpoints."
        )
    if (
        args.reuse_predictions
        and not selected_folds
        and not args.force_rerun
        and checkpoint_contract.get("status") == "complete"
    ):
        try:
            reusable = load_existing(
                prediction_path,
                y,
                record_id,
                class_names,
                args,
                load_info,
                model_name,
                classifier_params,
                checkpoint_contract,
            )
        except Exception as exc:
            print(f"Reusable Hybrid morphology NPZ rejected: {exc!r}", flush=True)
            reusable = None
    if reusable is None:
        y_prob, fold_id_out, fold_rows = fit_predict_mlp_oof(
            X,
            y,
            folds,
            args=args,
            load_info=load_info,
            classifier_params=classifier_params,
        )
        if selected_folds:
            params_json = json.dumps(json_safe(classifier_params), sort_keys=True)
            ready = []
            checkpoints = []
            for split in folds:
                fold_num = int(split["fold"])
                checkpoint_path = fold_checkpoint_path(args, fold_num)
                if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
                    checkpoints.append(fold_num)
                if load_reusable_fold_cache(
                    cache_path=fold_cache_path(fold_num),
                    checkpoint_path=checkpoint_path,
                    val_indices=np.asarray(split["va_idx"], dtype=np.int64),
                    n_classes=int(y.shape[1]),
                    params_json=params_json,
                    load_info=load_info,
                ) is not None:
                    ready.append(fold_num)
            print(
                json.dumps(
                    {
                        "status": True,
                        "mode": "fold_cache_only",
                        "selected_folds": sorted(selected_folds),
                        "ready_fold_caches": ready,
                        "ready_checkpoints": checkpoints,
                        "next_step": "Publish the mirror, then rerun without --only-folds to aggregate.",
                    },
                    indent=2,
                )
            )
            return
        checkpoint_contract = build_checkpoint_contract(args)
        require_complete_checkpoint_contract(checkpoint_contract)
        write_prediction_npz(
            prediction_path,
            y=y,
            y_prob=y_prob,
            record_id=record_id,
            fold_id=fold_id_out,
            class_names=class_names,
            args=args,
            load_info=load_info,
            model_name=model_name,
            classifier_params=classifier_params,
            checkpoint_contract=checkpoint_contract,
        )
    else:
        y_prob, fold_id_out = reusable
        fold_rows = [
            {
                "fold": int(fold),
                "train_records": int(sum(np.asarray(fold_id_out) != fold)),
                "validation_records": int(np.sum(np.asarray(fold_id_out) == fold)),
                "validation_positive_labels": int(np.sum(y[np.asarray(fold_id_out) == fold])),
                "classifier": model_name,
                "reused_predictions": True,
            }
            for fold in sorted(int(x) for x in np.unique(fold_id_out) if int(x) > 0)
        ]

    print("Computing point metrics...", flush=True)
    metrics = multilabel_metrics(y, y_prob, threshold=float(args.threshold))
    print("Computing calibration metrics...", flush=True)
    calibration = calibration_summary(y, y_prob, n_bins=int(args.n_bins))
    print(f"Computing bootstrap CI with n_boot={args.n_boot}; this stage is CPU-bound.", flush=True)
    ci = {
        "macro_pr_auc": bootstrap_metric("macro_pr_auc", y, y_prob, args, macro_pr_auc),
        "macro_roc_auc": bootstrap_metric("macro_roc_auc", y, y_prob, args, macro_roc_auc),
        "f1_macro": bootstrap_metric(
            "f1_macro",
            y,
            y_prob,
            args,
            lambda yt, yp: multilabel_metrics(yt, yp, threshold=float(args.threshold))["f1_macro"],
        ),
        "brier_macro": bootstrap_metric(
            "brier_macro",
            y,
            y_prob,
            args,
            lambda yt, yp: calibration_summary(yt, yp, n_bins=int(args.n_bins))["brier_macro"],
        ),
        "ece_macro": bootstrap_metric(
            "ece_macro",
            y,
            y_prob,
            args,
            lambda yt, yp: calibration_summary(yt, yp, n_bins=int(args.n_bins))["ece_macro"],
        ),
    }
    helpers._save_csv(per_class_path, helpers.per_class_rows(y, y_prob, class_names, float(args.threshold)))
    helpers._save_csv(fold_path, fold_rows)
    require_complete_checkpoint_contract(checkpoint_contract)

    summary = {
        "created_utc": now_utc(),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "dataset": "chapman_oof",
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "evaluated_transform_name": EVALUATED_TRANSFORM_NAME,
        "canonical_minirocket": False,
        "feature_preprocessing": "fold_train_standardization" if args.standardize == "train_fold" else "none",
        "model": model_name,
        "classifier_params": classifier_params,
        "runtime_config": runtime_config,
        "n_records": int(len(y)),
        "n_classes": int(y.shape[1]),
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "n_boot": int(args.n_boot),
        "metrics": metrics,
        "calibration": calibration,
        "bootstrap_ci": ci,
        "load_info": load_info,
        "checkpoint_contract": checkpoint_contract,
        "artifacts": {
            "predictions_npz": str(prediction_path),
            "per_class_table": str(per_class_path),
            "fold_summary_table": str(fold_path),
        },
        "manuscript_ready": args.limit_records == 0,
    }
    save_json(summary_path, json_safe(summary))
    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "evaluated_transform_name": EVALUATED_TRANSFORM_NAME,
        "canonical_minirocket": False,
        "freeze_contract": freeze_contract,
        "load_info": load_info,
        "classifier_params": classifier_params,
        "runtime_config": runtime_config,
        "checkpoint_contract": checkpoint_contract,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
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
    save_json(manifest_path, json_safe(manifest))
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
    print(f"Wrote: {summary_path}", flush=True)
    print(f"Wrote: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
