"""Controlled frozen-vs-partially-learnable morphology-bank experiment.

This reviewer-facing runner isolates one narrow design change: whether a
seeded random-convolution morphology bank is entirely frozen or whether a
pre-specified fraction of its kernel weights can learn.  Both variants use the
same initial kernels, thresholds, differentiable pooling, MLP head, folds,
optimizer, and final-epoch selection rule.

The bank is deliberately reduced (default: 256 kernels) so the controlled
experiment is tractable on one A100.  It is not the 10,000-kernel frozen branch
inside the evaluated ECG-RAMBA checkpoint and must not be used to claim a
causal mechanism for the full model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    calibration_summary,
    ensure_revision_dirs,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_csv,
    save_json,
    save_npz_compressed_atomic,
    sha256_file,
)
from src.aggregation import POWER_MEAN_IMPLEMENTATION, aggregate_record_probabilities  # noqa: E402
from src.training_data import build_slice_index  # noqa: E402


PROTOCOL = "morphology_learnability_control_v1_same_folds_power_mean_q3_threshold_0.5"
FEATURE_CONTRACT = "reduced_seeded_random_convolution_max_soft_ppv_control"
VARIANTS = {
    "frozen": 0.0,
    "partial": 0.25,
}
DEFAULT_FOLD_CACHE_DIR = PREDICTION_DIR / "folds"
DEFAULT_CHECKPOINT_DIR = (
    PROJECT_ROOT / "reports" / "revision" / "experimental" / "morphology_learnability_checkpoints"
)


def load_raw_helpers():
    path = PROJECT_ROOT / "scripts" / "revision" / "14_resnet1d_cnn_baseline.py"
    spec = importlib.util.spec_from_file_location("_morphology_learnability_raw_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load raw-ECG baseline helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


raw_helpers = load_raw_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-predictions", type=Path, default=PREDICTION_DIR / "oof_final_ema_predictions.npz")
    parser.add_argument("--freeze-manifest", type=Path, default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json")
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--raw-cache", type=Path, default=None)
    parser.add_argument("--variants", default="frozen,partial")
    parser.add_argument("--only-folds", default="")
    parser.add_argument("--num-kernels", type=int, default=256)
    parser.add_argument("--trainable-fraction", type=float, default=0.25)
    parser.add_argument("--dilations", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--kernel-length", type=int, default=9)
    parser.add_argument("--ppv-temperature", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--fold-cache-dir", type=Path, default=DEFAULT_FOLD_CACHE_DIR)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_csv_ints(value: str, *, minimum: int = 1) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or any(item < minimum for item in values):
        raise ValueError(f"Expected comma-separated integers >= {minimum}: {value!r}")
    return values


def parse_variants(value: str) -> list[str]:
    variants = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown or not variants:
        raise ValueError(f"Unknown or empty variants: {unknown or variants}")
    return list(dict.fromkeys(variants))


def parse_folds(value: str) -> set[int]:
    folds = {int(item.strip()) for item in str(value).split(",") if item.strip()}
    invalid = sorted(fold for fold in folds if fold < 1 or fold > 5)
    if invalid:
        raise ValueError(f"Invalid folds: {invalid}")
    return folds


def stable_json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def tensor_state_hash(model: nn.Module, *, include_masks: bool = False) -> str:
    """Hash deterministic initialized tensors without depending on serialization."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not include_masks and (
            "trainable_mask_" in name or "initial_weight_" in name
        ):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class ControlledRandomConvolutionBank(nn.Module):
    """Seeded bank with channel-wise gradient masks for partial learnability."""

    def __init__(
        self,
        *,
        c_in: int,
        num_kernels: int,
        kernel_length: int,
        dilations: list[int],
        trainable_fraction: float,
        ppv_temperature: float,
        seed: int,
    ) -> None:
        super().__init__()
        if not 0.0 <= trainable_fraction <= 1.0:
            raise ValueError("trainable_fraction must be in [0, 1].")
        if num_kernels < len(dilations) or num_kernels % len(dilations):
            raise ValueError("num_kernels must be divisible by the number of dilations.")
        if ppv_temperature <= 0:
            raise ValueError("ppv_temperature must be positive.")

        self.num_kernels = int(num_kernels)
        self.trainable_fraction = float(trainable_fraction)
        self.ppv_temperature = float(ppv_temperature)
        kernels_per_dilation = num_kernels // len(dilations)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.convs = nn.ModuleList()
        self._trainable_mask_names: list[str] = []
        self._initial_weight_names: list[str] = []
        self._threshold_names: list[str] = []

        for index, dilation in enumerate(dilations):
            conv = nn.Conv1d(
                c_in,
                kernels_per_dilation,
                kernel_size=kernel_length,
                dilation=int(dilation),
                padding="same",
                bias=False,
            )
            weights = torch.randint(-1, 2, conv.weight.shape, generator=generator).float()
            conv.weight.data.copy_(weights)
            trainable_count = int(round(kernels_per_dilation * trainable_fraction))
            permutation = torch.randperm(kernels_per_dilation, generator=generator)
            mask = torch.zeros(kernels_per_dilation, 1, 1, dtype=torch.float32)
            if trainable_count:
                mask[permutation[:trainable_count]] = 1.0
            mask_name = f"trainable_mask_{index}"
            weight_name = f"initial_weight_{index}"
            threshold_name = f"threshold_{index}"
            self.register_buffer(mask_name, mask)
            self.register_buffer(weight_name, weights.clone())
            self.register_buffer(
                threshold_name,
                torch.randn(kernels_per_dilation, generator=generator).float() * 0.1,
            )
            self._trainable_mask_names.append(mask_name)
            self._initial_weight_names.append(weight_name)
            self._threshold_names.append(threshold_name)
            if trainable_count:
                conv.weight.requires_grad_(True)
                conv.weight.register_hook(lambda grad, m=mask: grad * m.to(grad.device, grad.dtype))
            else:
                conv.weight.requires_grad_(False)
            self.convs.append(conv)

    @property
    def output_dim(self) -> int:
        return self.num_kernels * 2

    @property
    def trainable_kernel_count(self) -> int:
        return int(
            sum(float(getattr(self, name).sum().item()) for name in self._trainable_mask_names)
        )

    @torch.no_grad()
    def restore_frozen_kernel_channels(self) -> None:
        for conv, mask_name, initial_name in zip(
            self.convs, self._trainable_mask_names, self._initial_weight_names
        ):
            mask = getattr(self, mask_name).to(conv.weight.device, conv.weight.dtype)
            initial = getattr(self, initial_name).to(conv.weight.device, conv.weight.dtype)
            conv.weight.data.mul_(mask).add_(initial * (1.0 - mask))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for conv, threshold_name in zip(self.convs, self._threshold_names):
            response = conv(x)
            maximum = response.amax(dim=-1)
            threshold = getattr(self, threshold_name).view(1, -1, 1)
            scaled = ((response - threshold) / self.ppv_temperature).clamp(-20.0, 20.0)
            soft_ppv = torch.sigmoid(scaled).mean(dim=-1)
            features.extend([maximum, soft_ppv])
        return torch.cat(features, dim=1)


class MorphologyLearnabilityControl(nn.Module):
    def __init__(self, *, model_params: dict[str, Any], trainable_fraction: float) -> None:
        super().__init__()
        self.bank = ControlledRandomConvolutionBank(
            c_in=12,
            num_kernels=int(model_params["num_kernels"]),
            kernel_length=int(model_params["kernel_length"]),
            dilations=[int(value) for value in model_params["dilations"]],
            trainable_fraction=float(trainable_fraction),
            ppv_temperature=float(model_params["ppv_temperature"]),
            seed=int(model_params["bank_seed"]),
        )
        feature_dim = self.bank.output_dim
        hidden_dim = int(model_params["hidden_dim"])
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(model_params["dropout"])),
            nn.Linear(hidden_dim, int(model_params["n_classes"])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.bank(x))


def model_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "architecture": "controlled_seeded_random_convolution_morphology_bank",
        "input_shape": [12, int(raw_helpers.CONFIG["slice_length"])],
        "slice_length": int(raw_helpers.CONFIG["slice_length"]),
        "slice_stride": int(raw_helpers.CONFIG["slice_stride"]),
        "max_slices_per_record": int(raw_helpers.CONFIG["max_slices_per_record"]),
        "aggregation": "power_mean_q3",
        "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
        "num_kernels": int(args.num_kernels),
        "kernel_length": int(args.kernel_length),
        "dilations": parse_csv_ints(args.dilations),
        "pooling_statistics": ["MAX", "soft_PPV"],
        "ppv_temperature": float(args.ppv_temperature),
        "partial_trainable_fraction": float(args.trainable_fraction),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "n_classes": len(raw_helpers.CLASSES),
        "bank_seed": int(args.seed),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "loss": "BCEWithLogitsLoss_fold_pos_weight",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "selection_rule": "fixed_final_epoch",
        "scope": "reduced_bank_mechanism_sensitivity_not_checkpoint_reproduction",
    }


def variant_fraction(args: argparse.Namespace, variant: str) -> float:
    return 0.0 if variant == "frozen" else float(args.trainable_fraction)


def checkpoint_path(args: argparse.Namespace, variant: str, fold: int) -> Path:
    return resolve(args.checkpoint_dir) / variant / f"fold{fold}_morphology_learnability_{variant}_final.pt"


def fold_cache_path(args: argparse.Namespace, variant: str, fold: int) -> Path:
    return resolve(args.fold_cache_dir) / f"morphology_learnability_{variant}_fold{fold}_predictions.npz"


def output_prediction_path(variant: str) -> Path:
    return PREDICTION_DIR / f"morphology_learnability_{variant}_oof_predictions.npz"


def input_contract(load_info: dict[str, Any], params_hash: str, variant: str, fraction: float) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "variant": variant,
        "trainable_fraction": float(fraction),
        "model_params_sha256": params_hash,
        "oof_predictions_sha256": load_info["oof_predictions_sha256"],
        "raw_cache_sha256": load_info["raw_cache_sha256"],
        "freeze_manifest_sha256": load_info["freeze_contract"]["freeze_manifest_sha256"],
        "dataset_record_order_fingerprint": load_info["dataset_record_order_fingerprint"],
    }


def load_fold_cache(
    *,
    path: Path,
    checkpoint: Path,
    fold: int,
    va_idx: np.ndarray,
    contract: dict[str, Any],
) -> np.ndarray | None:
    if not path.exists() or not checkpoint.exists() or path.stat().st_size == 0 or checkpoint.stat().st_size == 0:
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"val_indices", "y_prob", "checkpoint_sha256", *contract.keys()}
            if required - set(payload.files):
                return None
            if int(payload["fold"].item()) != int(fold):
                return None
            if not np.array_equal(np.asarray(payload["val_indices"], dtype=np.int64), va_idx):
                return None
            for key, expected in contract.items():
                observed = payload[key].item() if np.ndim(payload[key]) == 0 else payload[key]
                if str(observed) != str(expected):
                    return None
            if str(payload["checkpoint_sha256"].item()) != sha256_file(checkpoint):
                return None
            y_prob = np.asarray(payload["y_prob"], dtype=np.float32)
        if y_prob.shape != (len(va_idx), len(raw_helpers.CLASSES)) or not np.all(np.isfinite(y_prob)):
            return None
        return np.clip(y_prob, 0.0, 1.0)
    except Exception as exc:
        print(f"Fold cache rejected {path}: {exc!r}", flush=True)
        return None


def save_fold_cache(
    *,
    path: Path,
    checkpoint: Path,
    fold: int,
    va_idx: np.ndarray,
    y_prob: np.ndarray,
    contract: dict[str, Any],
) -> None:
    save_npz_compressed_atomic(
        path,
        fold=np.asarray(int(fold)),
        val_indices=np.asarray(va_idx, dtype=np.int64),
        y_prob=np.asarray(y_prob, dtype=np.float32),
        checkpoint_sha256=np.asarray(sha256_file(checkpoint)),
        **{key: np.asarray(value) for key, value in contract.items()},
    )
    print(f"Wrote fold cache: {path}", flush=True)


def checkpoint_is_compatible(path: Path, *, fold: int, contract: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        return None
    if int(payload.get("fold", -1)) != int(fold) or payload.get("input_contract") != contract:
        return None
    return payload


def build_variant_loaders(
    *,
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    args: argparse.Namespace,
    fold: int,
    device: torch.device,
):
    slice_length = int(raw_helpers.CONFIG["slice_length"])
    stride = int(raw_helpers.CONFIG["slice_stride"])
    max_slices = int(raw_helpers.CONFIG["max_slices_per_record"])
    tr_records, tr_starts, _tr_pos, tr_skipped = build_slice_index(
        tr_idx, X, slice_length=slice_length, slice_stride=stride, max_slices_per_record=max_slices
    )
    va_records, va_starts, _va_pos, va_skipped = build_slice_index(
        va_idx, X, slice_length=slice_length, slice_stride=stride, max_slices_per_record=max_slices
    )
    if tr_skipped or va_skipped:
        raise RuntimeError(f"Unexpected skipped records: train={tr_skipped} validation={va_skipped}")
    train_ds = raw_helpers.RawECGSliceDataset(X, y, tr_records, tr_starts, slice_length=slice_length)
    val_ds = raw_helpers.RawECGSliceDataset(X, y, va_records, va_starts, slice_length=slice_length)
    train_loader = raw_helpers.build_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed + fold,
        device=device,
    )
    val_loader = raw_helpers.build_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1000 + fold,
        device=device,
    )
    return train_loader, val_loader, len(tr_records), len(va_records)


def train_or_predict_fold(
    *,
    variant: str,
    fold: int,
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    load_info: dict[str, Any],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    params_hash = stable_json_hash(params)
    fraction = variant_fraction(args, variant)
    contract = input_contract(load_info, params_hash, variant, fraction)
    checkpoint = checkpoint_path(args, variant, fold)
    cache = fold_cache_path(args, variant, fold)
    if not args.force_rerun:
        reusable = load_fold_cache(
            path=cache, checkpoint=checkpoint, fold=fold, va_idx=va_idx, contract=contract
        )
        if reusable is not None:
            print(f"{variant} fold {fold}: reused authenticated fold cache", flush=True)
            checkpoint_payload = checkpoint_is_compatible(
                checkpoint, fold=fold, contract=contract
            )
            if checkpoint_payload is None:
                raise RuntimeError(
                    f"{variant} fold {fold}: fold cache passed but checkpoint contract failed"
                )
            metrics = multilabel_metrics(y[va_idx], reusable, threshold=args.threshold)
            return reusable, {
                "variant": variant,
                "fold": fold,
                "train_records": len(tr_idx),
                "validation_records": len(va_idx),
                "reused_fold_cache": True,
                "checkpoint_sha256": sha256_file(checkpoint),
                "initialization_sha256": checkpoint_payload.get("initialization_sha256"),
                "trainable_kernel_count": checkpoint_payload.get("trainable_kernel_count"),
                **{f"final_{key}": value for key, value in metrics.items()},
            }

    train_loader, val_loader, train_slices, val_slices = build_variant_loaders(
        X=X, y=y, tr_idx=tr_idx, va_idx=va_idx, args=args, fold=fold, device=device
    )
    raw_helpers.set_seed(args.seed + fold * 100)
    model = MorphologyLearnabilityControl(model_params=params, trainable_fraction=fraction).to(device)
    initialization_sha256 = tensor_state_hash(model)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    reused_checkpoint = False
    saved = None
    if args.reuse_checkpoints and not args.force_rerun:
        saved = checkpoint_is_compatible(checkpoint, fold=fold, contract=contract)
    if saved is not None:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.to(device)
        reused_checkpoint = True
        print(f"{variant} fold {fold}: checkpoint accepted; inference only", flush=True)
    else:
        pos_weight = raw_helpers.pos_weight_from_labels(y[tr_idx]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=max(args.lr * 0.02, 1e-6)
        )
        scaler = raw_helpers.make_scaler(device, args.amp)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, target, _record, _start in train_loader:
                x = x.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with raw_helpers.autocast_context(device, args.amp):
                    loss = criterion(model(x), target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                model.bank.restore_frozen_kernel_channels()
                losses.append(float(loss.detach().cpu().item()))
            scheduler.step()
            message = (
                f"{variant} fold {fold} epoch {epoch:02d}/{args.epochs} "
                f"loss={float(np.mean(losses)):.6f}"
            )
            if args.eval_every > 0 and (epoch == args.epochs or epoch % args.eval_every == 0):
                slice_prob, slice_record, _slice_start = raw_helpers.predict_slice_probabilities(
                    model, val_loader, device=device, use_amp=args.amp
                )
                fold_metrics = raw_helpers.record_metrics_for_fold(
                    y, slice_prob, slice_record, va_idx, args.threshold
                )
                message += (
                    f" F1={fold_metrics['f1_macro']:.4f}"
                    f" PR={fold_metrics['pr_auc_macro']:.4f}"
                    f" ROC={fold_metrics['roc_auc_macro']:.4f}"
                )
            print(message, flush=True)
        if args.save_checkpoints:
            raw_helpers.save_torch_atomic(
                checkpoint,
                {
                    "model_state_dict": model.state_dict(),
                    "fold": fold,
                    "variant": variant,
                    "protocol": PROTOCOL,
                    "input_contract": contract,
                    "model_params": params,
                    "initialization_sha256": initialization_sha256,
                    "trainable_kernel_count": model.bank.trainable_kernel_count,
                    "args": json_safe(vars(args)),
                    "created_utc": now_utc(),
                },
            )
            print(f"Wrote checkpoint: {checkpoint}", flush=True)
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"Missing durable checkpoint for {variant} fold {fold}: {checkpoint}")

    slice_prob, slice_record, _slice_start = raw_helpers.predict_slice_probabilities(
        model, val_loader, device=device, use_amp=args.amp
    )
    aggregated, valid, _counts = aggregate_record_probabilities(
        slice_prob, slice_record, n_records=len(y), q=3.0
    )
    missing = sorted(set(va_idx.astype(int)) - set(np.where(valid)[0].astype(int)))
    if missing:
        raise RuntimeError(f"{variant} fold {fold}: validation records without predictions: {missing[:10]}")
    y_prob = np.asarray(aggregated[va_idx], dtype=np.float32)
    save_fold_cache(
        path=cache,
        checkpoint=checkpoint,
        fold=fold,
        va_idx=va_idx,
        y_prob=y_prob,
        contract=contract,
    )
    metrics = multilabel_metrics(y[va_idx], y_prob, threshold=args.threshold)
    return y_prob, {
        "variant": variant,
        "fold": fold,
        "train_records": len(tr_idx),
        "validation_records": len(va_idx),
        "train_slices": train_slices,
        "validation_slices": val_slices,
        "reused_fold_cache": False,
        "reused_checkpoint": reused_checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "initialization_sha256": initialization_sha256,
        "trainable_kernel_count": model.bank.trainable_kernel_count,
        **{f"final_{key}": value for key, value in metrics.items()},
    }


def checkpoint_contract(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    rows = []
    missing = []
    for fold in range(1, 6):
        path = checkpoint_path(args, variant, fold)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(fold)
        else:
            rows.append({"fold": fold, "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {
        "status": "complete" if not missing else "partial",
        "variant": variant,
        "missing_folds": missing,
        "checkpoints": rows,
    }


def write_prediction_artifact(
    *,
    variant: str,
    y: np.ndarray,
    y_prob: np.ndarray,
    fold_id: np.ndarray,
    record_id: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
    load_info: dict[str, Any],
    params: dict[str, Any],
    checkpoints: dict[str, Any],
) -> Path:
    path = output_prediction_path(variant)
    rows = sorted(checkpoints["checkpoints"], key=lambda row: row["fold"])
    save_npz_compressed_atomic(
        path,
        y_true=y.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        fold_id=fold_id.astype(np.int16),
        record_id=record_id.astype(np.int64),
        class_names=np.asarray(class_names),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(PROTOCOL),
        feature_contract=np.asarray(FEATURE_CONTRACT),
        variant=np.asarray(variant),
        trainable_fraction=np.asarray(variant_fraction(args, variant)),
        model_params_json=np.asarray(json.dumps(json_safe(params), sort_keys=True)),
        oof_predictions_sha256=np.asarray(load_info["oof_predictions_sha256"]),
        freeze_manifest_sha256=np.asarray(load_info["freeze_contract"]["freeze_manifest_sha256"]),
        raw_cache_sha256=np.asarray(load_info["raw_cache_sha256"]),
        dataset_record_order_fingerprint=np.asarray(load_info["dataset_record_order_fingerprint"]),
        checkpoint_folds=np.asarray([row["fold"] for row in rows], dtype=np.int16),
        checkpoint_sha256=np.asarray([row["sha256"] for row in rows]),
    )
    return path


def class_rows(variant: str, y: np.ndarray, y_prob: np.ndarray, class_names: list[str], threshold: float) -> list[dict]:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    rows = []
    for index, name in enumerate(class_names):
        target = y[:, index]
        probability = y_prob[:, index]
        both = len(np.unique(target)) == 2
        rows.append(
            {
                "variant": variant,
                "class_index": index,
                "class_name": name,
                "prevalence": float(np.mean(target)),
                "roc_auc": float(roc_auc_score(target, probability)) if both else math.nan,
                "pr_auc": float(average_precision_score(target, probability)) if both else math.nan,
                "f1": float(f1_score(target, probability >= threshold, zero_division=0)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    variants = parse_variants(args.variants)
    selected_folds = parse_folds(args.only_folds)
    if args.num_kernels % len(parse_csv_ints(args.dilations)):
        raise ValueError("--num-kernels must be divisible by the number of dilations.")
    if not 0.0 < args.trainable_fraction < 1.0:
        raise ValueError("--trainable-fraction must be strictly between 0 and 1.")
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = raw_helpers.select_device(args.device)
    print("=" * 80, flush=True)
    print("CONTROLLED MORPHOLOGY KERNEL LEARNABILITY", flush=True)
    print("=" * 80, flush=True)
    print(f"protocol={PROTOCOL}", flush=True)
    print(f"variants={variants} selected_folds={sorted(selected_folds) or 'all'}", flush=True)
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    if device.type != "cuda" and not args.limit_records:
        raise RuntimeError("Canonical morphology learnability training requires a CUDA runtime.")

    freeze_contract = raw_helpers.validate_oof_freeze_contract(
        freeze_manifest=args.freeze_manifest,
        oof_predictions=args.oof_predictions,
        expected_checkpoint_kind=args.expected_checkpoint_kind,
    )
    y, fold_id, record_id, class_names, folds, oof_info = raw_helpers.load_oof_labels_and_folds(
        args.oof_predictions, limit_records=args.limit_records
    )
    record_fingerprint = (
        oof_info.get("dataset_record_order_fingerprint")
        or freeze_contract.get("dataset_record_order_fingerprint")
        or ""
    )
    X, raw_info = raw_helpers.load_raw_cache(
        expected_y=y,
        expected_record_fingerprint=record_fingerprint,
        explicit_cache=args.raw_cache,
        limit_records=args.limit_records,
    )
    load_info = {**oof_info, **raw_info, "freeze_contract": freeze_contract}
    params = model_params(args)
    params_hash = stable_json_hash(params)
    available_folds = {int(split["fold"]) for split in folds}
    if selected_folds - available_folds:
        raise ValueError(f"Requested folds absent from OOF: {sorted(selected_folds - available_folds)}")

    fold_rows: list[dict[str, Any]] = []
    variant_probabilities: dict[str, np.ndarray] = {}
    run_folds = [split for split in folds if not selected_folds or int(split["fold"]) in selected_folds]
    for variant in variants:
        y_prob_all = np.full(y.shape, np.nan, dtype=np.float32)
        for split in run_folds:
            fold = int(split["fold"])
            y_prob, row = train_or_predict_fold(
                variant=variant,
                fold=fold,
                X=X,
                y=y,
                tr_idx=np.asarray(split["tr_idx"], dtype=np.int64),
                va_idx=np.asarray(split["va_idx"], dtype=np.int64),
                args=args,
                device=device,
                load_info=load_info,
                params=params,
            )
            y_prob_all[np.asarray(split["va_idx"], dtype=np.int64)] = y_prob
            fold_rows.append(row)
        variant_probabilities[variant] = y_prob_all

    status_path = METRIC_DIR / "morphology_learnability_fold_cache_status.json"
    status_table = TABLE_DIR / "table_morphology_learnability_fold_cache_status.csv"
    status_rows = []
    for variant in variants:
        contract = checkpoint_contract(args, variant)
        for fold in sorted(available_folds):
            cache = fold_cache_path(args, variant, fold)
            checkpoint = checkpoint_path(args, variant, fold)
            status_rows.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "cache_exists": cache.exists() and cache.stat().st_size > 0,
                    "checkpoint_exists": checkpoint.exists() and checkpoint.stat().st_size > 0,
                    "cache_path": str(cache),
                    "checkpoint_path": str(checkpoint),
                }
            )
    save_csv(status_table, status_rows)
    save_json(
        status_path,
        {
            "status": "complete" if all(row["cache_exists"] and row["checkpoint_exists"] for row in status_rows) else "partial",
            "created_utc": now_utc(),
            "selected_folds": sorted(selected_folds),
            "rows": status_rows,
        },
    )
    if selected_folds:
        print(json.dumps({"status": "fold_cache_only", "status_file": str(status_path)}, indent=2), flush=True)
        return

    if set(variants) != set(VARIANTS):
        raise RuntimeError("Canonical aggregation requires both frozen and partial variants.")

    initialization_by_fold: dict[int, set[str]] = {}
    for row in fold_rows:
        fingerprint = str(row.get("initialization_sha256") or "")
        if len(fingerprint) != 64:
            raise RuntimeError(
                f"Missing initialization fingerprint for {row.get('variant')} fold {row.get('fold')}"
            )
        initialization_by_fold.setdefault(int(row["fold"]), set()).add(fingerprint)
    mismatched_initializations = {
        fold: sorted(values)
        for fold, values in initialization_by_fold.items()
        if len(values) != 1
    }
    if mismatched_initializations:
        raise RuntimeError(
            "Frozen and partially learnable variants were not identically initialized: "
            + repr(mismatched_initializations)
        )
    for variant in variants:
        if np.any(~np.isfinite(variant_probabilities[variant])):
            # The current invocation may have reused no selected folds. Load all authenticated caches.
            restored = np.full(y.shape, np.nan, dtype=np.float32)
            for split in folds:
                fold = int(split["fold"])
                va_idx = np.asarray(split["va_idx"], dtype=np.int64)
                contract = input_contract(load_info, params_hash, variant, variant_fraction(args, variant))
                cached = load_fold_cache(
                    path=fold_cache_path(args, variant, fold),
                    checkpoint=checkpoint_path(args, variant, fold),
                    fold=fold,
                    va_idx=va_idx,
                    contract=contract,
                )
                if cached is None:
                    raise RuntimeError(f"Missing authenticated {variant} fold {fold} cache/checkpoint.")
                restored[va_idx] = cached
            variant_probabilities[variant] = restored

    summary_variants = {}
    class_table_rows = []
    artifact_rows = []
    for variant in variants:
        checkpoints = checkpoint_contract(args, variant)
        if checkpoints["status"] != "complete":
            raise RuntimeError(f"Incomplete checkpoint contract for {variant}: {checkpoints['missing_folds']}")
        probability = variant_probabilities[variant]
        prediction = write_prediction_artifact(
            variant=variant,
            y=y,
            y_prob=probability,
            fold_id=fold_id,
            record_id=record_id,
            class_names=class_names,
            args=args,
            load_info=load_info,
            params=params,
            checkpoints=checkpoints,
        )
        metrics = multilabel_metrics(y, probability, threshold=args.threshold)
        calibration = calibration_summary(y, probability, n_bins=args.n_bins)
        summary_variants[variant] = {
            "trainable_fraction": variant_fraction(args, variant),
            "metrics": metrics,
            "calibration": calibration,
            "prediction_path": str(prediction),
            "prediction_sha256": sha256_file(prediction),
            "checkpoint_contract": checkpoints,
        }
        class_table_rows.extend(class_rows(variant, y, probability, class_names, args.threshold))
        artifact_rows.append({"kind": f"{variant}_predictions", "path": str(prediction), "sha256": sha256_file(prediction)})

    model_table = TABLE_DIR / "table_morphology_learnability_model_metrics.csv"
    fold_table = TABLE_DIR / "table_morphology_learnability_fold_summary.csv"
    class_table = TABLE_DIR / "table_morphology_learnability_class_metrics.csv"
    summary_path = METRIC_DIR / "morphology_learnability_summary.json"
    manifest_path = MANIFEST_DIR / "morphology_learnability_manifest.json"
    model_rows = []
    for variant, payload in summary_variants.items():
        model_rows.append(
            {
                "variant": variant,
                "trainable_fraction": payload["trainable_fraction"],
                **payload["metrics"],
                **payload["calibration"],
            }
        )
    save_csv(model_table, model_rows)
    save_csv(fold_table, fold_rows)
    save_csv(class_table, class_table_rows)
    summary = {
        "status": True,
        "created_utc": now_utc(),
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "model_params": params,
        "model_params_sha256": params_hash,
        "matched_initialization_sha256_by_fold": {
            str(fold): next(iter(values))
            for fold, values in sorted(initialization_by_fold.items())
        },
        "canonical_contract": input_contract(load_info, params_hash, "variant_specific", -1.0),
        "variants": summary_variants,
        "claim_guidance": {
            "allowed": "Use as a reduced-bank controlled sensitivity comparison between frozen and partially learnable seeded kernels.",
            "not_allowed": "Do not call this the evaluated 10,000-kernel ECG-RAMBA branch or infer a causal mechanism for the full model.",
        },
        "outputs": {
            "model_table": str(model_table),
            "fold_table": str(fold_table),
            "class_table": str(class_table),
        },
    }
    save_json(summary_path, json_safe(summary))
    artifacts = [summary_path, model_table, fold_table, class_table, status_path, status_table]
    artifacts.extend(output_prediction_path(variant) for variant in variants)
    manifest = {
        "status": "complete",
        "created_utc": now_utc(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": PROTOCOL,
        "inputs": {
            "oof_predictions": {"path": load_info["oof_predictions"], "sha256": load_info["oof_predictions_sha256"]},
            "freeze_manifest": {"path": str(resolve(args.freeze_manifest)), "sha256": load_info["freeze_contract"]["freeze_manifest_sha256"]},
            "raw_cache": {"path": load_info["raw_cache"], "sha256": load_info["raw_cache_sha256"]},
        },
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in artifacts
        ],
        "checkpoint_contracts": {
            variant: summary_variants[variant]["checkpoint_contract"] for variant in variants
        },
    }
    save_json(manifest_path, json_safe(manifest))
    print(json.dumps({"status": True, "variants": summary_variants, "manifest": str(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
