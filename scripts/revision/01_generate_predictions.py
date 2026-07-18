"""Generate reviewer-ready prediction NPZ files.

Current scope:
    - oof: Chapman out-of-fold predictions using fold checkpoints.

Run from repo root on Colab:
    python scripts/revision/01_generate_predictions.py --dataset oof

Outputs:
    reports/revision/predictions/<artifact_stem>_predictions.npz
    reports/revision/predictions/<artifact_stem>_slice_predictions.npz
    reports/revision/metrics/<artifact_stem>_prediction_summary.json

    `--checkpoint-kind best` preserves the legacy `oof_full` stem. Canonical
    manuscript OOF uses fixed-epoch `--checkpoint-kind final_ema`, which writes
    `oof_final_ema_*`. Validation-selected `best_ema` remains diagnostic.

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

from configs.config import (  # noqa: E402
    CLASSES,
    CONFIG,
    CONFIG_HASH,
    DEVICE,
    EVALUATION_CONFIG_HASH,
    PATHS,
)
from scripts.revision.common import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    LOG_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    POWER_MEAN_IMPLEMENTATION,
    TABLE_DIR,
    ensure_revision_dirs,
    multilabel_metrics,
    power_mean,
    save_csv,
    save_json,
    sha256_file,
)
from src.provenance import record_order_fingerprint  # noqa: E402
from src.features import (  # noqa: E402
    HRV36_CHECKPOINT_SEMANTICS,
    checkpoint_compatible_hrv36_contract,
)

CHECKPOINT_KINDS = ["best", "final", "best_ema", "final_ema", "best_raw", "final_raw"]
AMP_DTYPE = (
    torch.bfloat16
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
)
AMP_DTYPE_NAME = (
    str(AMP_DTYPE).replace("torch.", "")
    if DEVICE == "cuda"
    else "float32"
)


def oof_protocol_names(
    checkpoint_kind: str,
    ablation_variant: str,
    aggregation_q: float,
) -> tuple[str, str]:
    """Keep the canonical Full protocol stable while naming retrained removals."""

    variant_token = "" if ablation_variant == "full" else f"_{ablation_variant}"
    record_protocol = (
        f"fold_{checkpoint_kind}{variant_token}_{POWER_MEAN_IMPLEMENTATION}_"
        f"q{float(aggregation_q):g}_threshold_0.5"
    )
    slice_protocol = f"slice_level_fold_{checkpoint_kind}{variant_token}"
    return record_protocol, slice_protocol


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
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
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
        "source_runtime_config_hash": CONFIG_HASH,
        "evaluation_config_hash": EVALUATION_CONFIG_HASH,
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


def hydra_fold_cache_path(
    fold_num: int,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    source_config_hash: str,
) -> Path:
    cache_dir = Path(PATHS["cache_dir"]) / "revision_feature_cache"
    train_hash = index_fingerprint(tr_idx)
    val_hash = index_fingerprint(va_idx)
    name = (
        f"hydra_oof_v{CACHE_SCHEMA_VERSION}_{source_config_hash}_fold{fold_num}_"
        f"train{len(tr_idx)}_{train_hash}_val{len(va_idx)}_{val_hash}_"
        f"D{CONFIG['hydra_dim']}.npz"
    )
    return cache_dir / name


def fold_pca_model_path(
    fold_num: int,
    tr_idx: np.ndarray,
    source_config_hash: str,
) -> Path:
    cache_dir = Path(
        os.environ.get("ECG_RAMBA_PCA_CACHE_DIR")
        or (Path(PATHS["cache_dir"]) / "revision_pca_models")
    )
    train_hash = index_fingerprint(tr_idx)
    return cache_dir / (
        f"fold{fold_num}_pca_v{CACHE_SCHEMA_VERSION}_{source_config_hash}_"
        f"train{len(tr_idx)}_{train_hash}_D{CONFIG['hydra_dim']}.joblib"
    )


def load_or_compute_fold_hydra(
    *,
    fold_num: int,
    X_rocket_raw: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    source_config_hash: str,
) -> tuple[np.ndarray, float, Path, bool]:
    from src.features import apply_pca, fit_pca_on_train

    cache_path = hydra_fold_cache_path(
        fold_num,
        tr_idx,
        va_idx,
        source_config_hash,
    )
    expected_shape = (len(va_idx), int(CONFIG["hydra_dim"]))
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            hydra_va = cached["hydra_va"]
            pca_var = float(cached["pca_explained_variance"])
            schema_version = int(cached["cache_schema_version"]) if "cache_schema_version" in cached.files else 0
            if (
                hydra_va.shape == expected_shape
                and hydra_va.dtype == np.float32
                and schema_version == CACHE_SCHEMA_VERSION
            ):
                print(f"✅ Loaded fold-aware Hydra/PCA cache for fold {fold_num}: {cache_path}", flush=True)
                return hydra_va, pca_var, cache_path, True
            print(
                f"⚠️  Hydra cache contract mismatch for fold {fold_num}: "
                f"shape={hydra_va.shape}, dtype={hydra_va.dtype}, schema={schema_version}. "
                "Recomputing as float32.",
                flush=True,
            )
        except Exception as exc:
            print(f"⚠️  Could not load Hydra cache for fold {fold_num}: {exc}. Recomputing.", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pca_model_path = fold_pca_model_path(
        fold_num,
        tr_idx,
        source_config_hash,
    )
    if pca_model_path.exists():
        print(f"Loading training-fold PCA object: {pca_model_path}", flush=True)
        pca = joblib.load(pca_model_path)
    else:
        print(
            f"Fitting fold-aware PCA for fold {fold_num}: "
            f"train={len(tr_idx)} x {X_rocket_raw.shape[1]} -> D={CONFIG['hydra_dim']}",
            flush=True,
        )
        pca = fit_pca_on_train(X_rocket_raw[tr_idx], CONFIG["hydra_dim"])
        pca_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, pca_model_path)
        print(f"Saved fold-aware PCA object: {pca_model_path}", flush=True)
    print(f"Transforming validation Hydra features for fold {fold_num}: val={len(va_idx)}", flush=True)
    hydra_va = apply_pca(pca, X_rocket_raw[va_idx])
    pca_var = float(pca.explained_variance_ratio_.sum())
    elapsed = time.time() - start
    print(f"PCA fold {fold_num} finished in {elapsed / 60:.1f} min | variance={pca_var:.6f}", flush=True)
    print(f"Saving fold-aware Hydra/PCA cache: {cache_path}", flush=True)
    np.savez_compressed(
        cache_path,
        hydra_va=hydra_va.astype(np.float32),
        pca_explained_variance=np.asarray(pca_var, dtype=np.float32),
        cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        fold=np.asarray(fold_num, dtype=np.int16),
        source_config_hash=np.asarray(source_config_hash),
        evaluation_config_hash=np.asarray(EVALUATION_CONFIG_HASH),
        train_index_hash=np.asarray(index_fingerprint(tr_idx)),
        val_index_hash=np.asarray(index_fingerprint(va_idx)),
    )
    return hydra_va.astype(np.float32), pca_var, cache_path, False


def fold_prediction_cache_path(
    fold_num: int,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    fold_cache_dir: Path,
) -> Path:
    return (
        fold_cache_dir
        / (
            f"oof_fold{fold_num}_{checkpoint_kind}_{EVALUATION_CONFIG_HASH}_"
            f"{checkpoint_sha256[:12]}_v{CACHE_SCHEMA_VERSION}.npz"
        )
    )


def oof_artifact_stem(checkpoint_kind: str) -> str:
    if checkpoint_kind == "best":
        return "oof_full"
    return f"oof_{checkpoint_kind}"


def load_fold_prediction_cache(
    *,
    path: Path,
    fold_num: int,
    va_idx: np.ndarray,
    n_classes: int,
    oof_probs: np.ndarray,
    fold_id: np.ndarray,
    record_slice_count: np.ndarray,
    save_slice_probs: bool,
    slice_probs_all: list[np.ndarray],
    slice_record_index_all: list[np.ndarray],
    slice_index_all: list[np.ndarray],
    slice_fold_id_all: list[np.ndarray],
    checkpoint_sha256: str,
) -> dict | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        record_id = data["record_id"].astype(np.int64)
        cached_y_prob = data["y_prob"]
        y_prob = cached_y_prob.astype(np.float32)
        valid_mask = data["valid_record_mask"].astype(bool)
        slice_count = data["slice_count"].astype(np.int16)
        summary = json.loads(str(data["fold_summary_json"].item()))
        schema_version = int(data["cache_schema_version"]) if "cache_schema_version" in data else 0
        cached_checkpoint_sha = (
            str(data["checkpoint_sha256"].item())
            if "checkpoint_sha256" in data
            else ""
        )
        aggregation_implementation = (
            str(data["aggregation_implementation"].item())
            if "aggregation_implementation" in data
            else ""
        )
        if (
            record_id.shape != va_idx.shape
            or not np.array_equal(record_id, va_idx.astype(np.int64))
            or y_prob.shape != (len(va_idx), n_classes)
            or valid_mask.shape != va_idx.shape
            or cached_y_prob.dtype != np.float32
            or schema_version != CACHE_SCHEMA_VERSION
            or cached_checkpoint_sha != checkpoint_sha256
            or aggregation_implementation != POWER_MEAN_IMPLEMENTATION
        ):
            print(
                    f"WARNING: Fold cache contract/fingerprint mismatch; recomputing fold {fold_num}: {path}",
                flush=True,
            )
            return None

        oof_probs[record_id] = y_prob
        fold_id[record_id[valid_mask]] = fold_num
        record_slice_count[record_id] = slice_count

        if save_slice_probs:
            required_slice_keys = {
                "slice_prob",
                "slice_record_id",
                "slice_index",
                "slice_fold_id",
            }
            if not required_slice_keys.issubset(data) or len(data["slice_prob"]) == 0:
                print(
                    f"WARNING: Fold cache lacks required slice probabilities; recomputing fold {fold_num}: {path}",
                    flush=True,
                )
                return None
            cached_slice_prob = data["slice_prob"]
            cached_slice_record_id = data["slice_record_id"]
            cached_slice_index = data["slice_index"]
            cached_slice_fold_id = data["slice_fold_id"]
            expected_slice_records = set(int(x) for x in record_id[valid_mask])
            actual_slice_records = set(int(x) for x in np.unique(cached_slice_record_id))
            cached_slice_counts = np.bincount(
                cached_slice_record_id,
                minlength=int(np.max(record_id)) + 1,
            )[record_id]
            if (
                cached_slice_prob.dtype != np.float32
                or cached_slice_prob.ndim != 2
                or cached_slice_prob.shape[1] != n_classes
                or len(cached_slice_prob) != len(cached_slice_record_id)
                or len(cached_slice_prob) != len(cached_slice_index)
                or len(cached_slice_prob) != len(cached_slice_fold_id)
                or not np.isfinite(cached_slice_prob).all()
                or not np.all(cached_slice_fold_id == fold_num)
                or actual_slice_records != expected_slice_records
                or not np.array_equal(cached_slice_counts.astype(np.int16), slice_count)
            ):
                print(
                    f"WARNING: Fold slice cache contract mismatch; recomputing fold {fold_num}: {path}",
                    flush=True,
                )
                return None
            slice_probs_all.append(cached_slice_prob)
            slice_record_index_all.append(cached_slice_record_id.astype(np.int64))
            slice_index_all.append(cached_slice_index.astype(np.int16))
            slice_fold_id_all.append(cached_slice_fold_id.astype(np.int16))

        print(f"Loaded cached predictions for fold {fold_num}: {path}", flush=True)
        return summary
    except Exception as exc:
        print(
            f"WARNING: Could not load fold prediction cache for fold {fold_num}: {exc}. Recomputing.",
            flush=True,
        )
        return None


def save_fold_prediction_cache(
    *,
    path: Path,
    fold_num: int,
    va_idx: np.ndarray,
    oof_probs: np.ndarray,
    fold_id: np.ndarray,
    record_slice_count: np.ndarray,
    fold_summary: dict,
    slice_probs: np.ndarray | None,
    slice_rids: np.ndarray | None,
    checkpoint_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_mask = fold_id[va_idx] >= 0
    payload = {
        "record_id": va_idx.astype(np.int64),
        "y_prob": oof_probs[va_idx].astype(np.float32),
        "valid_record_mask": valid_mask.astype(np.bool_),
        "slice_count": record_slice_count[va_idx].astype(np.int16),
        "fold": np.asarray(fold_num, dtype=np.int16),
        "config_hash": np.asarray(EVALUATION_CONFIG_HASH),
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "checkpoint_sha256": np.asarray(checkpoint_sha256),
        "aggregation_implementation": np.asarray(POWER_MEAN_IMPLEMENTATION),
        "fold_summary_json": np.asarray(json.dumps(fold_summary, sort_keys=True)),
    }
    if slice_probs is not None and slice_rids is not None:
        payload.update(
            {
                "slice_prob": slice_probs.astype(np.float32),
                "slice_record_id": slice_rids.astype(np.int64),
                "slice_index": slice_ordinals(slice_rids),
                "slice_fold_id": np.full(len(slice_rids), fold_num, dtype=np.int16),
            }
        )
    else:
        payload.update(
            {
                "slice_prob": np.empty((0, len(CLASSES)), dtype=np.float32),
                "slice_record_id": np.empty((0,), dtype=np.int64),
                "slice_index": np.empty((0,), dtype=np.int16),
                "slice_fold_id": np.empty((0,), dtype=np.int16),
            }
        )
    tmp_path = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, path)
    print(f"💾 Saved fold prediction cache for fold {fold_num}: {path}", flush=True)


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
    parser.add_argument("--checkpoint-kind", choices=CHECKPOINT_KINDS, default="best")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Checkpoint directory override for a matched retrained architecture variant.",
    )
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=None,
        help="Explicit canonical folds.pkl. Defaults to <model-dir>/folds.pkl.",
    )
    parser.add_argument(
        "--artifact-stem",
        default=None,
        help="Output stem override; required for non-Full architecture variants.",
    )
    parser.add_argument(
        "--ablation-variant",
        default="full",
        choices=["full", "no_morphology", "no_rhythm", "no_fusion", "no_context_fusion"],
    )
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
    parser.add_argument("--resume-fold-cache", action="store_true", default=True)
    parser.add_argument("--no-resume-fold-cache", dest="resume_fold_cache", action="store_false")
    parser.add_argument("--force-rerun-folds", action="store_true", default=False)
    parser.add_argument(
        "--fold-cache-dir",
        type=Path,
        default=PREDICTION_DIR / "folds",
        help=(
            "Directory for resumable per-fold OOF caches. Point this to the canonical "
            "Drive mirror in Colab so completed folds survive runtime disconnects."
        ),
    )
    parser.add_argument(
        "--allow-checkpoint-fallback",
        action="store_true",
        help="Legacy/debug only. Manuscript exports must use the exact requested checkpoint files.",
    )
    parser.add_argument("--min-system-ram-gb", type=float, default=24.0)
    parser.add_argument("--allow-low-ram", action="store_true", default=False)
    return parser.parse_args()


def system_ram_gb() -> float:
    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        return float(page_size * physical_pages / (1024 ** 3))
    return 0.0


def validate_runtime_memory(args: argparse.Namespace) -> None:
    ram_gb = system_ram_gb()
    print(f"System RAM detected: {ram_gb:.1f} GiB", flush=True)
    if ram_gb and ram_gb < args.min_system_ram_gb and not args.allow_low_ram:
        raise RuntimeError(
            f"Insufficient system RAM for full OOF export: {ram_gb:.1f} GiB detected, "
            f"{args.min_system_ram_gb:.1f} GiB required. Standard Colab T4 runtimes can be "
            "killed with exit code 137 because raw ECG, MiniRocket, PCA, and fold slices coexist "
            "in host RAM. Use a High-RAM/A100 runtime. Batch size mainly controls GPU memory. "
            "Pass --allow-low-ram only for controlled debugging, not manuscript prediction export."
        )


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


def load_folds(
    y: np.ndarray,
    subjects: np.ndarray,
    folds_path_override: Path | None = None,
) -> list[dict[str, np.ndarray]]:
    folds_path = (
        folds_path_override
        if folds_path_override is not None
        else Path(PATHS["model_dir"]) / "folds.pkl"
    )
    folds_path = folds_path if folds_path.is_absolute() else PROJECT_ROOT / folds_path
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


def checkpoint_path(fold: int, checkpoint_kind: str, *, allow_fallback: bool = False) -> Path:
    preferred = Path(PATHS["model_dir"]) / f"fold{fold}_{checkpoint_kind}.pt"
    if preferred.exists():
        return preferred
    if allow_fallback:
        fallback_candidates = []
        if checkpoint_kind == "best":
            fallback_candidates = ["best_ema", "best_raw", "final"]
        elif checkpoint_kind == "final":
            fallback_candidates = ["final_raw", "final_ema", "best"]
        elif checkpoint_kind.endswith("_ema"):
            fallback_candidates = [checkpoint_kind.replace("_ema", "")]
        elif checkpoint_kind.endswith("_raw"):
            fallback_candidates = [checkpoint_kind.replace("_raw", "")]
        for fallback_kind in fallback_candidates:
            fallback = Path(PATHS["model_dir"]) / f"fold{fold}_{fallback_kind}.pt"
            if fallback.exists():
                print(
                    f"Checkpoint fallback for fold {fold}: {preferred.name} missing, "
                    f"using {fallback.name}. This is not manuscript-safe."
                )
                return fallback
    raise FileNotFoundError(f"Missing exact checkpoint for fold {fold}: {preferred}")


def expected_weights_kind(checkpoint_kind: str) -> str | None:
    if checkpoint_kind.endswith("_ema"):
        return "ema"
    if checkpoint_kind.endswith("_raw"):
        return "raw"
    return None


def validate_checkpoint_weights_kind(checkpoint_kind: str, actual: str | None, path: Path) -> None:
    expected = expected_weights_kind(checkpoint_kind)
    if expected is not None and actual != expected:
        raise ValueError(
            f"Checkpoint kind {checkpoint_kind} requires explicit weights_kind={expected}, "
            f"but {path} reports weights_kind={actual}. "
            "Retrain with the explicit checkpoint contract before manuscript OOF export."
        )


def load_checkpoint_payload(path: Path, checkpoint_kind: str) -> tuple[dict, dict]:
    # These are user-owned training checkpoints selected from the frozen model
    # run. PyTorch 2.6+ defaults to weights_only=True, which rejects the rich
    # metadata dictionaries (for example Path values) stored by train.py.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"Checkpoint lacks the explicit metadata contract: {path}"
        )
    actual = checkpoint.get("weights_kind")
    validate_checkpoint_weights_kind(checkpoint_kind, actual, path)
    source_config_hash = checkpoint.get("config_hash")
    if not source_config_hash:
        raise ValueError(f"Checkpoint lacks config_hash provenance: {path}")
    expected_feature_contract = {"hrv36": checkpoint_compatible_hrv36_contract()}
    declared_feature_contract = checkpoint.get("feature_contract")
    if declared_feature_contract is None:
        feature_contract = expected_feature_contract
        feature_contract_provenance = "inferred_audited_legacy_chapman_checkpoint"
    else:
        feature_contract = dict(declared_feature_contract)
        feature_contract_provenance = "declared_in_checkpoint"
        if feature_contract != expected_feature_contract:
            raise ValueError(
                f"Checkpoint feature contract mismatch for {path}: "
                f"found={feature_contract!r} expected={expected_feature_contract!r}"
            )
    architecture_contract = checkpoint.get("architecture_contract")
    pca_contract = checkpoint.get("pca_contract")
    if architecture_contract == "ecg_ramba_structured_ablation_v1":
        if not isinstance(pca_contract, dict):
            raise ValueError(f"Structured-ablation checkpoint lacks PCA contract: {path}")
        pca_path_raw = str(pca_contract.get("path") or "")
        pca_path = Path(pca_path_raw) if pca_path_raw else None
        if (
            pca_path is None
            or not pca_path.is_file()
            or sha256_file(pca_path) != pca_contract.get("sha256")
        ):
            raise ValueError(f"Structured-ablation checkpoint PCA artifact is missing/stale: {path}")
    metadata = {
        "source_config_hash": str(source_config_hash),
        "weights_kind": actual,
        "epoch": int(checkpoint.get("epoch", -1)),
        "selection_rule": checkpoint.get("selection_rule"),
        "metrics_weights_kind": checkpoint.get("metrics_weights_kind"),
        "checkpoint_contract": checkpoint.get("checkpoint_contract"),
        "training_protocol": checkpoint.get("training_protocol"),
        "dataset_record_order_fingerprint": checkpoint.get(
            "dataset_record_order_fingerprint"
        ),
        "ablation_variant": checkpoint.get("ablation_variant", "full"),
        "ablation_spec": checkpoint.get("ablation_spec", {}),
        "architecture_contract": architecture_contract,
        "feature_contract": feature_contract,
        "feature_contract_provenance": feature_contract_provenance,
        "pca_contract": pca_contract,
    }
    return checkpoint, metadata


def load_model_for_fold(
    fold: int,
    checkpoint_kind: str,
    checkpoint_file: Path | None = None,
    checkpoint_payload: dict | None = None,
    ablation_variant: str | None = None,
) -> torch.nn.Module:
    from src.model import ECGRambaV7Advanced, resolve_structured_ablation

    path = checkpoint_file or checkpoint_path(fold, checkpoint_kind)
    print(f"Loading checkpoint: {path}")
    checkpoint = checkpoint_payload
    if checkpoint is None:
        checkpoint, _ = load_checkpoint_payload(path, checkpoint_kind)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    actual = checkpoint.get("weights_kind") if isinstance(checkpoint, dict) else None
    validate_checkpoint_weights_kind(checkpoint_kind, actual, path)

    checkpoint_variant = checkpoint.get("ablation_variant", "full")
    variant, expected_spec = resolve_structured_ablation(ablation_variant or checkpoint_variant)
    checkpoint_spec = dict(checkpoint.get("ablation_spec", {}))
    if checkpoint_variant != variant or checkpoint_spec != expected_spec:
        raise RuntimeError(
            f"Checkpoint architecture contract mismatch for {path}: "
            f"checkpoint=({checkpoint_variant}, {checkpoint_spec}) requested=({variant}, {expected_spec})"
        )
    model = ECGRambaV7Advanced(cfg=CONFIG, ablation=expected_spec).to(DEVICE)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        if expected_weights_kind(checkpoint_kind) is not None:
            raise RuntimeError(
                f"Strict checkpoint load failed for explicit checkpoint kind "
                f"{checkpoint_kind}: {path}. Architecture/config drift must be "
                "resolved before manuscript evaluation."
            ) from exc
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
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
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

    args.fold_cache_dir = (
        args.fold_cache_dir
        if args.fold_cache_dir.is_absolute()
        else PROJECT_ROOT / args.fold_cache_dir
    ).resolve()
    args.fold_cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_revision_dirs()
    set_seed(CONFIG["seeds"][0])
    created_utc = datetime.now(timezone.utc).isoformat()
    run_meta = runtime_metadata(args, created_utc)
    config_meta = config_snapshot()

    X, y, X_raw_amp, subjects = prepare_clean_chapman(limit_records=args.limit_records)
    n_records, n_classes = y.shape
    dataset_record_fingerprint = record_order_fingerprint(subjects)

    X_rocket_raw = generate_raw_rocket_cache(X, subjects)
    X_hrv = generate_hrv_cache(X, X_raw_amp, subjects) if CONFIG["use_hrv"] else np.zeros(
        (n_records, CONFIG["hrv_dim"]), dtype=np.float32
    )
    folds = load_folds(y, subjects, args.folds_path)

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
    source_config_hashes: set[str] = set()
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

        ckpt_path = checkpoint_path(
            fold_num,
            args.checkpoint_kind,
            allow_fallback=args.allow_checkpoint_fallback,
        )
        ckpt_info = {"fold": fold_num, **path_info(ckpt_path, with_sha256=True)}
        checkpoint_payload, checkpoint_meta = load_checkpoint_payload(
            ckpt_path,
            args.checkpoint_kind,
        )
        if checkpoint_meta["ablation_variant"] != args.ablation_variant:
            raise RuntimeError(
                f"Fold {fold_num} checkpoint variant {checkpoint_meta['ablation_variant']} "
                f"does not match requested {args.ablation_variant}"
            )
        if (
            args.limit_records == 0
            and checkpoint_meta["dataset_record_order_fingerprint"]
            != dataset_record_fingerprint
        ):
            raise RuntimeError(
                f"Fold {fold_num} checkpoint record-order fingerprint does not "
                "match the loaded Chapman dataset"
            )
        ckpt_info.update(checkpoint_meta)
        source_config_hash = checkpoint_meta["source_config_hash"]
        source_config_hashes.add(source_config_hash)
        checkpoint_sha256 = ckpt_info["sha256"]
        checkpoint_infos.append(ckpt_info)
        fold_cache_path = fold_prediction_cache_path(
            fold_num,
            args.checkpoint_kind,
            checkpoint_sha256,
            args.fold_cache_dir,
        )
        if args.resume_fold_cache and not args.force_rerun_folds:
            cached_summary = load_fold_prediction_cache(
                path=fold_cache_path,
                fold_num=fold_num,
                va_idx=va_idx,
                n_classes=n_classes,
                oof_probs=oof_probs,
                fold_id=fold_id,
                record_slice_count=record_slice_count,
                save_slice_probs=args.save_slice_probs,
                slice_probs_all=slice_probs_all,
                slice_record_index_all=slice_record_index_all,
                slice_index_all=slice_index_all,
                slice_fold_id_all=slice_fold_id_all,
                checkpoint_sha256=checkpoint_sha256,
            )
            if cached_summary is not None:
                del checkpoint_payload
                fold_summaries.append(cached_summary)
                if "pca_explained_variance" in cached_summary:
                    pca_variance.append(
                        {
                            "fold": fold_num,
                            "explained_variance": cached_summary.get("pca_explained_variance"),
                            "cache_path": cached_summary.get("pca_cache_path"),
                            "cache_hit": True,
                            "prediction_cache_hit": True,
                        }
                    )
                continue

        hydra_va, pca_var, pca_cache_path, pca_cache_hit = load_or_compute_fold_hydra(
            fold_num=fold_num,
            X_rocket_raw=X_rocket_raw,
            tr_idx=tr_idx,
            va_idx=va_idx,
            source_config_hash=source_config_hash,
        )
        pca_object_path = fold_pca_model_path(
            fold_num,
            tr_idx,
            source_config_hash,
        )
        pca_variance.append(
            {
                "fold": fold_num,
                "explained_variance": pca_var,
                "cache_path": str(pca_cache_path),
                "cache_hit": bool(pca_cache_hit),
                "pca_object_path": str(pca_object_path) if pca_object_path.exists() else None,
                "pca_object_sha256": sha256_file(pca_object_path) if pca_object_path.exists() else None,
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
            "pca_object_path": str(pca_object_path) if pca_object_path.exists() else None,
            "pca_object_sha256": sha256_file(pca_object_path) if pca_object_path.exists() else None,
            "checkpoint": ckpt_info,
        }

        if len(rids) == 0:
            print(f"Fold {fold_num}: no slices generated; skipping")
            del checkpoint_payload
            fold_summary["n_predicted_records"] = 0
            fold_summaries.append(fold_summary)
            save_fold_prediction_cache(
                path=fold_cache_path,
                fold_num=fold_num,
                va_idx=va_idx,
                oof_probs=oof_probs,
                fold_id=fold_id,
                record_slice_count=record_slice_count,
                fold_summary=fold_summary,
                slice_probs=None,
                slice_rids=None,
                checkpoint_sha256=checkpoint_sha256,
            )
            continue

        model = load_model_for_fold(
            fold_num,
            args.checkpoint_kind,
            checkpoint_file=ckpt_path,
            checkpoint_payload=checkpoint_payload,
            ablation_variant=args.ablation_variant,
        )
        del checkpoint_payload
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
            oof_probs[int(rid)] = power_mean(
                preds,
                q=float(CONFIG["power_mean_q"]),
                axis=0,
            )
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
            slice_probs_all.append(slice_probs.astype(np.float32))
            slice_record_index_all.append(slice_rids.astype(np.int64))
            slice_index_all.append(slice_ordinals(slice_rids))
            slice_fold_id_all.append(np.full(len(slice_rids), fold_num, dtype=np.int16))

        save_fold_prediction_cache(
            path=fold_cache_path,
            fold_num=fold_num,
            va_idx=va_idx,
            oof_probs=oof_probs,
            fold_id=fold_id,
            record_slice_count=record_slice_count,
            fold_summary=fold_summary,
            slice_probs=slice_probs if args.save_slice_probs else None,
            slice_rids=slice_rids if args.save_slice_probs else None,
            checkpoint_sha256=checkpoint_sha256,
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    valid_records = fold_id >= 0
    if not np.all(valid_records):
        missing = int(np.sum(~valid_records))
        print(f"Warning: {missing} records did not receive predictions")

    git_commit = run_meta["git"]["commit"] or ""
    threshold = 0.5
    aggregation_q = float(CONFIG["power_mean_q"])
    protocol, slice_protocol = oof_protocol_names(
        args.checkpoint_kind,
        args.ablation_variant,
        aggregation_q,
    )
    if len(source_config_hashes) != 1:
        raise RuntimeError(
            "All fold checkpoints must share one source_config_hash; found "
            f"{sorted(source_config_hashes)}"
        )
    source_config_hash = next(iter(source_config_hashes))
    expected_feature_contract = {"hrv36": checkpoint_compatible_hrv36_contract()}
    observed_feature_contracts = {
        json.dumps(info.get("feature_contract"), sort_keys=True)
        for info in checkpoint_infos
    }
    if observed_feature_contracts != {json.dumps(expected_feature_contract, sort_keys=True)}:
        raise RuntimeError(
            "Fold checkpoints do not share the checkpoint-compatible HRV36 feature contract"
        )
    feature_contract_json = json.dumps(expected_feature_contract, sort_keys=True)
    checkpoint_fingerprints_json = json.dumps(
        sorted(checkpoint_infos, key=lambda row: int(row["fold"])),
        sort_keys=True,
    )

    artifact_stem = args.artifact_stem or oof_artifact_stem(args.checkpoint_kind)
    out_path = PREDICTION_DIR / f"{artifact_stem}_predictions.npz"
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
        config_hash=np.asarray(EVALUATION_CONFIG_HASH),
        source_config_hash=np.asarray(source_config_hash),
        evaluation_config_hash=np.asarray(EVALUATION_CONFIG_HASH),
        dataset_record_order_fingerprint=np.asarray(dataset_record_fingerprint),
        git_commit=np.asarray(git_commit),
        created_utc=np.asarray(created_utc),
        checkpoint_kind=np.asarray(args.checkpoint_kind),
        batch_size=np.asarray(args.batch_size, dtype=np.int32),
        aggregation_method=np.asarray("power_mean"),
        aggregation_q=np.asarray(aggregation_q, dtype=np.float32),
        aggregation_implementation=np.asarray(POWER_MEAN_IMPLEMENTATION),
        cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        checkpoint_fingerprints_json=np.asarray(checkpoint_fingerprints_json),
        threshold=np.asarray(threshold, dtype=np.float32),
        ablation_variant=np.asarray(args.ablation_variant),
        architecture_contract=np.asarray("ecg_ramba_structured_ablation_v1"),
        hrv_feature_semantics=np.asarray(HRV36_CHECKPOINT_SEMANTICS),
        feature_contract_json=np.asarray(feature_contract_json),
    )
    print(f"Wrote: {out_path}")

    slice_out_path = None
    if args.save_slice_probs and slice_probs_all:
        slice_out_path = PREDICTION_DIR / f"{artifact_stem}_slice_predictions.npz"
        np.savez_compressed(
            slice_out_path,
            slice_prob=np.concatenate(slice_probs_all, axis=0),
            record_id=np.concatenate(slice_record_index_all, axis=0),
            slice_index=np.concatenate(slice_index_all, axis=0),
            fold_id=np.concatenate(slice_fold_id_all, axis=0),
            class_names=np.asarray(CLASSES),
            dataset=np.asarray("chapman_oof"),
            protocol=np.asarray(slice_protocol),
            config_hash=np.asarray(EVALUATION_CONFIG_HASH),
            source_config_hash=np.asarray(source_config_hash),
            evaluation_config_hash=np.asarray(EVALUATION_CONFIG_HASH),
            dataset_record_order_fingerprint=np.asarray(dataset_record_fingerprint),
            git_commit=np.asarray(git_commit),
            created_utc=np.asarray(created_utc),
            probability_dtype=np.asarray("float32"),
            cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
            checkpoint_fingerprints_json=np.asarray(checkpoint_fingerprints_json),
        )
        print(f"Wrote: {slice_out_path}")

    metrics = multilabel_metrics(y[valid_records], oof_probs[valid_records], threshold=threshold)
    class_summary_path = TABLE_DIR / f"{artifact_stem}_class_summary.csv"
    class_rows = per_class_summary_rows(
        y[valid_records],
        oof_probs[valid_records],
        CLASSES,
        threshold=threshold,
    )
    save_csv(class_summary_path, class_rows)
    print(f"Wrote: {class_summary_path}")

    manifest_path = MANIFEST_DIR / f"{artifact_stem}_prediction_run_manifest.json"
    summary_path = METRIC_DIR / f"{artifact_stem}_prediction_summary.json"

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
        "source_config_hash": source_config_hash,
        "evaluation_config_hash": EVALUATION_CONFIG_HASH,
        "dataset_record_order_fingerprint": dataset_record_fingerprint,
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
        "ablation_variant": args.ablation_variant,
        "architecture_contract": "ecg_ramba_structured_ablation_v1",
        "feature_contract": expected_feature_contract,
        "batch_size": int(args.batch_size),
        "inference_amp_dtype": AMP_DTYPE_NAME,
        "num_workers": int(args.num_workers),
        "limit_records": int(args.limit_records),
        "save_slice_probs": bool(args.save_slice_probs),
        "aggregation": {"method": "power_mean", "q": aggregation_q},
        "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
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
        "ablation_variant": args.ablation_variant,
        "architecture_contract": "ecg_ramba_structured_ablation_v1",
        "feature_contract": expected_feature_contract,
        "dataset_record_order_fingerprint": dataset_record_fingerprint,
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
    if args.model_dir is not None:
        model_dir = args.model_dir if args.model_dir.is_absolute() else PROJECT_ROOT / args.model_dir
        PATHS["model_dir"] = str(model_dir.resolve())
    if args.ablation_variant != "full" and not args.artifact_stem:
        raise ValueError("--artifact-stem is required for non-Full ablation exports")
    validate_runtime_memory(args)
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
