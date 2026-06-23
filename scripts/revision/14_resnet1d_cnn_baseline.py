"""Run a fold-safe ResNet1D/CNN baseline under the frozen OOF protocol.

This runner is reviewer-facing baseline evidence. It trains a small raw-ECG
1D ResNet/CNN from scratch on the exact frozen Chapman OOF folds and labels.
It deliberately does not use MiniRocket, HRV, PCA, or ECG-RAMBA checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
from src.aggregation import POWER_MEAN_IMPLEMENTATION, aggregate_record_probabilities  # noqa: E402
from src.provenance import record_order_fingerprint  # noqa: E402
from src.training_data import build_slice_index  # noqa: E402


PROTOCOL = "resnet1d_cnn_raw_same_folds_power_mean_v2_q3_threshold_0.5"
FEATURE_CONTRACT = "raw_ecg_12lead"
DEFAULT_OOF_PREDICTIONS = PREDICTION_DIR / "oof_final_ema_predictions.npz"
DEFAULT_FREEZE_MANIFEST = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
PREDICTION_PATH = PREDICTION_DIR / "resnet1d_cnn_oof_predictions.npz"
SLICE_PREDICTION_PATH = PREDICTION_DIR / "resnet1d_cnn_slice_predictions.npz"
SUMMARY_PATH = METRIC_DIR / "resnet1d_cnn_baseline_summary.json"
MANIFEST_PATH = MANIFEST_DIR / "resnet1d_cnn_baseline_manifest.json"
PER_CLASS_TABLE = TABLE_DIR / "table_resnet1d_cnn_class_metrics.csv"
FOLD_TABLE = TABLE_DIR / "table_resnet1d_cnn_fold_summary.csv"
FOLD_PREDICTION_DIR = PREDICTION_DIR / "folds"


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use float16 AMP on CUDA.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "reports" / "revision" / "experimental" / "resnet1d_cnn_checkpoints")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    return device


def validate_oof_freeze_contract(
    *,
    freeze_manifest: Path,
    oof_predictions: Path,
    expected_checkpoint_kind: str,
) -> dict:
    freeze_path = _resolve_project_path(freeze_manifest)
    pred_path = _resolve_project_path(oof_predictions)
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


def candidate_raw_cache_paths(explicit_cache: Path | None) -> list[Path]:
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
        candidates.append(root / "ecg_data_27c_subject.npz")
        candidates.extend(
            sorted(
                [
                    path for path in root.glob("ecg_data_27c_subject*.npz")
                    if ".corrupt_" not in path.name
                ],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

    deduped: list[Path] = []
    seen_paths: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        deduped.append(path)
    return deduped


def load_raw_cache(
    *,
    expected_y: np.ndarray,
    expected_record_fingerprint: str,
    explicit_cache: Path | None,
    limit_records: int,
) -> tuple[np.ndarray, dict]:
    checked = []
    for path in candidate_raw_cache_paths(explicit_cache):
        checked.append(str(path))
        if not path.exists():
            continue
        print(f"Loading raw ECG cache candidate: {path}", flush=True)
        with np.load(path, allow_pickle=False) as data:
            required = {"X", "y", "subjects"}
            missing = required - set(data.files)
            if missing:
                print(f"  rejected: missing keys {sorted(missing)}", flush=True)
                continue
            X = np.asarray(data["X"], dtype=np.float32)
            y_cache = np.asarray(data["y"], dtype=np.float32)
            subjects = np.asarray(data["subjects"]).astype(str)
            stored_record_fingerprint = str(npz_scalar(data, "record_order_fingerprint", "") or "")

        if X.ndim != 3 or X.shape[1] != 12 or X.shape[2] != 5000:
            print(f"  rejected: unexpected X shape {X.shape}", flush=True)
            continue
        if len(X) != len(y_cache) or len(subjects) != len(X):
            print("  rejected: X/y/subjects length mismatch", flush=True)
            continue
        computed_fingerprint = record_order_fingerprint(subjects)
        if stored_record_fingerprint and stored_record_fingerprint != computed_fingerprint:
            print("  rejected: stored and computed record fingerprints differ", flush=True)
            continue
        if expected_record_fingerprint and computed_fingerprint != expected_record_fingerprint:
            print(
                "  rejected: raw cache record fingerprint does not match frozen OOF "
                f"{computed_fingerprint} != {expected_record_fingerprint}",
                flush=True,
            )
            continue
        if len(y_cache) < len(expected_y):
            print(f"  rejected: cache has too few records {len(y_cache)} < {len(expected_y)}", flush=True)
            continue
        if not np.array_equal(y_cache[: len(expected_y)], expected_y):
            print("  rejected: raw-cache labels do not exactly match frozen OOF labels", flush=True)
            continue

        if limit_records > 0:
            X = X[:limit_records]
            y_cache = y_cache[:limit_records]
            subjects = subjects[:limit_records]

        cache_info = {
            "raw_cache": str(path.resolve()),
            "raw_cache_sha256": sha256_file(path),
            "raw_cache_kind": "record_fingerprinted",
            "raw_cache_record_order_fingerprint": computed_fingerprint,
            "raw_cache_stored_record_order_fingerprint": stored_record_fingerprint,
            "raw_cache_shape": list(X.shape),
        }
        print(f"Using raw ECG cache: {path}", flush=True)
        print(f"Raw ECG shape: {X.shape}", flush=True)
        return X, cache_info

    raise FileNotFoundError(
        "No manuscript-safe raw ECG cache found. Checked:\n- " + "\n- ".join(checked)
    )


class RawECGSliceDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        record_ids: np.ndarray,
        starts: np.ndarray,
        *,
        slice_length: int,
    ) -> None:
        if len(record_ids) != len(starts):
            raise ValueError("record_ids and starts must have equal length")
        self.signals = signals
        self.labels = labels
        self.record_ids = np.asarray(record_ids, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int32)
        self.slice_length = int(slice_length)

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int):
        record_id = int(self.record_ids[index])
        start = int(self.starts[index])
        stop = start + self.slice_length
        signal = self.signals[record_id, :, start:stop]
        return (
            torch.from_numpy(signal).float(),
            torch.from_numpy(self.labels[record_id]).float(),
            record_id,
            start,
        )


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


class ResNet1DCNN(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        n_classes: int,
        base_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 6
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(c1),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            BasicBlock1D(c1, c1, stride=1, dropout=dropout),
            BasicBlock1D(c1, c1, stride=1, dropout=dropout),
        )
        self.layer2 = nn.Sequential(
            BasicBlock1D(c1, c2, stride=2, dropout=dropout),
            BasicBlock1D(c2, c2, stride=1, dropout=dropout),
        )
        self.layer3 = nn.Sequential(
            BasicBlock1D(c2, c3, stride=2, dropout=dropout),
            BasicBlock1D(c3, c3, stride=1, dropout=dropout),
        )
        self.layer4 = nn.Sequential(
            BasicBlock1D(c3, c4, stride=2, dropout=dropout),
            BasicBlock1D(c4, c4, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)


def build_model(args: argparse.Namespace) -> nn.Module:
    return ResNet1DCNN(
        in_channels=12,
        n_classes=len(CLASSES),
        base_channels=args.base_channels,
        dropout=args.dropout,
    )


def pos_weight_from_labels(y_train: np.ndarray) -> torch.Tensor:
    positives = np.sum(y_train, axis=0).astype(np.float64)
    negatives = len(y_train) - positives
    weight = negatives / np.maximum(positives, 1.0)
    weight = np.clip(weight, 1.0, 100.0)
    return torch.tensor(weight, dtype=torch.float32)


def autocast_context(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(device: torch.device, use_amp: bool):
    enabled = device.type == "cuda" and use_amp
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def predict_slice_probabilities(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, record_ids, starts = [], [], []
    with torch.no_grad():
        for x, _y, rid, start in loader:
            x = x.to(device, non_blocking=True)
            with autocast_context(device, use_amp):
                logits = model(x)
            batch_prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            probs.append(batch_prob)
            record_ids.append(np.asarray(rid, dtype=np.int64))
            starts.append(np.asarray(start, dtype=np.int32))
    return (
        np.concatenate(probs, axis=0),
        np.concatenate(record_ids, axis=0),
        np.concatenate(starts, axis=0),
    )


def record_metrics_for_fold(
    y: np.ndarray,
    slice_prob: np.ndarray,
    slice_record_id: np.ndarray,
    val_indices: np.ndarray,
    threshold: float,
) -> dict:
    y_prob_all, valid_mask, _slice_count = aggregate_record_probabilities(
        slice_prob,
        slice_record_id,
        n_records=len(y),
        q=float(CONFIG["power_mean_q"]),
    )
    missing = sorted(set(int(x) for x in val_indices) - set(np.where(valid_mask)[0].astype(int)))
    if missing:
        raise RuntimeError(f"Validation records without slice predictions: {missing[:10]}")
    return multilabel_metrics(y[val_indices], y_prob_all[val_indices], threshold=threshold)


def fold_prediction_path(fold: int) -> Path:
    return FOLD_PREDICTION_DIR / f"resnet1d_cnn_fold{fold}_predictions.npz"


def fold_prediction_matches(
    path: Path,
    *,
    y: np.ndarray,
    val_indices: np.ndarray,
    load_info: dict,
    args: argparse.Namespace,
    model_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
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
        if y_prob.shape != (len(y), y.shape[1]):
            return None
        if len(slice_count) != len(y):
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
    )
    print(f"Wrote fold prediction cache: {path}", flush=True)


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
                **{f"final_{k}": v for k, v in fold_metrics.items()},
            }

    print("=" * 80, flush=True)
    print(f"ResNet1D/CNN fold {fold}/5 | train={len(tr_idx)} | val={len(va_idx)}", flush=True)
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

    train_ds = RawECGSliceDataset(X, y, train_record_ids, train_starts, slice_length=slice_length)
    val_ds = RawECGSliceDataset(X, y, val_record_ids, val_starts, slice_length=slice_length)
    train_loader = build_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed + fold,
        device=device,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1000 + fold,
        device=device,
    )

    model = build_model(args).to(device)
    pos_weight = pos_weight_from_labels(y[tr_idx]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=max(args.lr * 0.02, 1e-6),
    )
    scaler = make_scaler(device, args.amp)

    epoch_rows = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for x, target, _rid, _start in train_loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                logits = model(x)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
        scheduler.step()

        row = {
            "fold": int(fold),
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)) if losses else math.nan,
        }
        if args.eval_every > 0 and (epoch == args.epochs or epoch % args.eval_every == 0):
            slice_prob, slice_record_id, _slice_start = predict_slice_probabilities(
                model,
                val_loader,
                device=device,
                use_amp=args.amp,
            )
            val_metrics = record_metrics_for_fold(
                y,
                slice_prob,
                slice_record_id,
                va_idx,
                threshold=args.threshold,
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            print(
                f"Fold {fold} Ep {epoch:03d}/{args.epochs} "
                f"loss={row['train_loss']:.5f} "
                f"F1={val_metrics['f1_macro']:.4f} "
                f"PR={val_metrics['pr_auc_macro']:.4f} "
                f"ROC={val_metrics['roc_auc_macro']:.4f}",
                flush=True,
            )
        else:
            print(
                f"Fold {fold} Ep {epoch:03d}/{args.epochs} loss={row['train_loss']:.5f}",
                flush=True,
            )
        epoch_rows.append(row)

    slice_prob, slice_record_id, slice_start = predict_slice_probabilities(
        model,
        val_loader,
        device=device,
        use_amp=args.amp,
    )
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
        checkpoint_path = args.checkpoint_dir / f"fold{fold}_resnet1d_cnn_final.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fold": int(fold),
                "protocol": PROTOCOL,
                "args": vars(args),
                "load_info": _json_safe(load_info),
                "final_metrics": _json_safe(final_metrics),
            },
            checkpoint_path,
        )
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
        "epoch_rows_json": json.dumps(_json_safe(epoch_rows), sort_keys=True),
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }
    return y_prob_all, slice_count, slice_prob, slice_record_id, slice_start, fold_summary


def _prediction_metadata(
    *,
    args: argparse.Namespace,
    load_info: dict,
    model_params: dict,
) -> dict:
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
        "model": np.asarray("resnet1d_cnn_raw_baseline"),
        "model_params_json": np.asarray(json.dumps(_json_safe(model_params), sort_keys=True)),
        "oof_predictions_sha256": np.asarray(load_info.get("oof_predictions_sha256", "")),
        "freeze_manifest_sha256": np.asarray(
            (load_info.get("freeze_contract") or {}).get("freeze_manifest_sha256", "")
        ),
        "raw_cache_sha256": np.asarray(load_info.get("raw_cache_sha256", "")),
        "dataset_record_order_fingerprint": np.asarray(
            load_info.get("dataset_record_order_fingerprint", "")
        ),
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
    print(f"Writing ResNet1D/CNN predictions: {path}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=y.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        record_id=record_id.astype(np.int64),
        fold_id=fold_id.astype(np.int16),
        slice_count=slice_count.astype(np.int16),
        class_names=np.asarray(class_names),
        **_prediction_metadata(
            args=args,
            load_info=load_info,
            model_params=model_params,
        ),
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
    print(f"Writing ResNet1D/CNN slice predictions: {path}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        slice_prob=slice_prob.astype(np.float32),
        slice_record_id=slice_record_id.astype(np.int64),
        slice_fold_id=slice_fold_id.astype(np.int16),
        slice_start=slice_start.astype(np.int32),
        class_names=np.asarray(class_names),
        **_prediction_metadata(
            args=args,
            load_info=load_info,
            model_params=model_params,
        ),
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
        print(
            f"Existing ResNet1D/CNN NPZ rejected: missing paired slice artifact {SLICE_PREDICTION_PATH}",
            flush=True,
        )
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"y_true", "y_prob", "record_id", "fold_id", "slice_count", "class_names"}
            missing = required - set(data.files)
            if missing:
                print(f"Existing ResNet1D/CNN NPZ rejected: missing {sorted(missing)}", flush=True)
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
        print(f"Existing ResNet1D/CNN NPZ rejected: {exc!r}", flush=True)
        return None

    try:
        with np.load(SLICE_PREDICTION_PATH, allow_pickle=False) as slice_data:
            required_slice = {"slice_prob", "slice_record_id", "slice_fold_id", "slice_start", "class_names"}
            missing_slice = required_slice - set(slice_data.files)
            if missing_slice:
                print(f"Existing ResNet1D/CNN NPZ rejected: slice artifact missing {sorted(missing_slice)}", flush=True)
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
            print("Existing ResNet1D/CNN NPZ rejected: slice_count does not match slice artifact", flush=True)
            return None
    except Exception as exc:
        print(f"Existing ResNet1D/CNN NPZ rejected: could not validate slice artifact: {exc!r}", flush=True)
        return None

    return (
        y_prob_existing,
        fold_existing,
        slice_count_existing,
    )


def per_class_rows(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str], threshold: float) -> list[dict]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = (y_prob >= threshold).astype(np.float32)
    rows = []
    for idx, name in enumerate(class_names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        pred = y_pred[:, idx]
        has_both = len(np.unique(yt)) > 1
        rows.append(
            {
                "class_index": int(idx),
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
    set_seed(args.seed)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = select_device(args.device)
    print("=" * 80, flush=True)
    print("RESNET1D/CNN FAIR BASELINE UNDER FROZEN OOF", flush=True)
    print("=" * 80, flush=True)
    print(f"protocol={PROTOCOL}", flush=True)
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    freeze_contract = validate_oof_freeze_contract(
        freeze_manifest=args.freeze_manifest,
        oof_predictions=args.oof_predictions,
        expected_checkpoint_kind=args.expected_checkpoint_kind,
    )
    y, fold_id, record_id, class_names, folds, oof_info = load_oof_labels_and_folds(
        args.oof_predictions,
        limit_records=args.limit_records,
    )
    if oof_info["oof_records_total"] != int(freeze_contract["validated_records"]):
        raise ValueError(
            "OOF prediction record count does not match freeze manifest: "
            f"{oof_info['oof_records_total']} != {freeze_contract['validated_records']}"
        )
    if int(args.limit_records) == 0 and oof_info["fold_count"] != 5:
        raise ValueError(f"Canonical ResNet1D/CNN baseline requires five folds, got {oof_info['fold_count']}")
    record_fingerprint = (
        oof_info.get("dataset_record_order_fingerprint")
        or freeze_contract.get("dataset_record_order_fingerprint")
        or ""
    )
    if not record_fingerprint:
        raise ValueError("Frozen OOF artifacts must carry dataset_record_order_fingerprint.")

    X, cache_info = load_raw_cache(
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
        "architecture": "resnet1d_cnn_basicblock",
        "input_shape": [12, int(CONFIG["slice_length"])],
        "slice_length": int(CONFIG["slice_length"]),
        "slice_stride": int(CONFIG["slice_stride"]),
        "max_slices_per_record": int(CONFIG["max_slices_per_record"]),
        "aggregation_method": "power_mean",
        "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
        "power_mean_q": float(CONFIG["power_mean_q"]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "base_channels": int(args.base_channels),
        "loss": "BCEWithLogitsLoss",
        "pos_weight": "fold_train_negative_positive_ratio_clipped_1_100",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "selection_rule": "fixed_final_epoch",
        "uses_hrv": False,
        "uses_minirocket": False,
        "uses_pca": False,
        "uses_ecg_ramba_checkpoints": False,
    }

    if args.reuse_predictions and not args.force_rerun:
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
        print(f"Reusing existing ResNet1D/CNN predictions: {PREDICTION_PATH}", flush=True)
        for fold in sorted(int(x) for x in np.unique(fold_id_out) if int(x) > 0):
            va_idx = np.where(fold_id_out == fold)[0]
            fold_metrics = multilabel_metrics(y[va_idx], y_prob[va_idx], threshold=args.threshold)
            fold_rows.append(
                {
                    "fold": int(fold),
                    "train_records": int(np.sum(fold_id_out != fold)),
                    "validation_records": int(len(va_idx)),
                    "train_slices": None,
                    "validation_slices": int(np.sum(slice_count[va_idx])),
                    "reused_fold_predictions": True,
                    "final_epoch": int(args.epochs),
                    **{f"final_{k}": v for k, v in fold_metrics.items()},
                }
            )
    else:
        y_prob = np.zeros_like(y, dtype=np.float32)
        slice_count = np.zeros(len(y), dtype=np.int16)
        fold_id_out = np.zeros(len(y), dtype=np.int16)
        for split in folds:
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

        if np.any(fold_id_out <= 0):
            raise RuntimeError("OOF fold_id coverage incomplete after ResNet1D/CNN training.")
        if np.any(slice_count <= 0):
            raise RuntimeError("Some records have no ResNet1D/CNN slice predictions.")
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

    _save_csv(PER_CLASS_TABLE, per_class_rows(y, y_prob, class_names, args.threshold))
    _save_csv(FOLD_TABLE, fold_rows)

    summary = {
        "created_utc": _now_utc(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "dataset": "chapman_oof",
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "model": "resnet1d_cnn_raw_baseline",
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
