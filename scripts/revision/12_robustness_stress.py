"""Run perturbation robustness tests versus a fixed-seed ROCKET-family head.

This runner is intentionally heavier than the HRV/domain notebook cells. It
re-runs inference on perturbed Chapman OOF validation records instead of
deriving robustness claims from clean predictions.

Protocol guarantees:
- The Full model uses frozen fold checkpoints, existing training-fold PCA
  objects, Q=3 power-mean slice aggregation, and the same OOF fold contract.
- Linear heads are trained on clean train-fold fixed-seed random-convolution
  MAX+PPV features and evaluated on matched perturbed validation features. The
  heads are reusable and tied to the legacy-named feature-cache SHA and
  hyperparameters; the transform is not canonical MiniRocket.
- Paired bootstrap resamples records with one shared index vector across clean,
  perturbed, Full, and MiniRocket predictions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG, EVALUATION_CONFIG_HASH, PATHS, SEQ_LEN  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    calibration_summary,
    ensure_revision_dirs,
    git_commit,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_json,
    save_npz_compressed_atomic,
    sha256_file,
)
from src.aggregation import POWER_MEAN_IMPLEMENTATION, power_mean  # noqa: E402
from src.features import (  # noqa: E402
    MiniRocketNative,
    extract_amplitude_features,
    extract_global_record_stats,
    extract_hrv_features,
)
from src.provenance import (  # noqa: E402
    exclusive_cache_writer,
    ndarray_sha256,
    record_order_fingerprint,
)


PROTOCOL = "robustness_full_vs_fixed_seed_rocket_perturbation_v2_source_bound"
CI_SCOPE = "nominal_95_percentile_paired_record_bootstrap_unadjusted"
INFERENCE_SCOPE = "pointwise_percentile_ci_effect_size_only"
NULL_TEST = "not_run"
MULTIPLICITY_ADJUSTMENT = "not_applicable_no_null_test"
TRAINING_VARIABILITY_SCOPE = "fixed_trained_folds_and_checkpoints_not_retrained_within_bootstrap"
PERTURBATION_REALIZATION_SCOPE = "single_fixed_seed_conditional_stress_audit"
MINIROCKET_HEAD_PROTOCOL = "minirocket_clean_heads_for_robustness_v1"
EXPECTED_MINIROCKET_PROTOCOL = "minirocket_raw_standardized_torch_linear_same_folds_threshold_0.5"
DEFAULT_STRESS_TESTS = [
    "snr20db",
    "snr10db",
    "snr5db",
    "random_3_lead_dropout",
    "precordial_dropout",
    "resample_250hz",
]
SOURCE_BUNDLE_PATHS = (
    "scripts/revision/12_robustness_stress.py",
    "scripts/revision/common.py",
    "scripts/revision/01_generate_predictions.py",
    "scripts/revision/10_minirocket_only_baseline.py",
    "src/aggregation.py",
    "src/features.py",
    "src/provenance.py",
    "configs/config.py",
)


def source_bundle_contract() -> dict:
    files = {relative: sha256_file(PROJECT_ROOT / relative) for relative in SOURCE_BUNDLE_PATHS}
    return {
        "schema_version": 1,
        "files": files,
        "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    higher_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]


@dataclass
class MiniRocketHeads:
    head_dir: Path
    manifest: dict
    clean_prob: np.ndarray
    fold_id: np.ndarray
    fold_rows: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--stress-tests", default=",".join(DEFAULT_STRESS_TESTS))
    parser.add_argument("--checkpoint-kind", default="final_ema")
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--minirocket-feature-batch-size", type=int, default=64)
    parser.add_argument("--minirocket-batch-size", type=int, default=4096)
    parser.add_argument("--minirocket-stats-batch-size", type=int, default=1024)
    parser.add_argument("--minirocket-epochs", type=int, default=20)
    parser.add_argument("--minirocket-lr", type=float, default=1e-3)
    parser.add_argument("--minirocket-weight-decay", type=float, default=1e-4)
    parser.add_argument("--minirocket-device", default="auto")
    parser.add_argument("--minirocket-feature-device", choices=["cpu", "cuda"], default="cpu")
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--features-only",
        action="store_true",
        help=(
            "Generate and validate source-bound perturbed ROCKET-family and HRV feature caches, "
            "then exit before loading any ECG-RAMBA checkpoint. Intended for a CPU High-RAM stage."
        ),
    )
    execution_group.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Require existing source-bound perturbation feature caches and run prediction inference only. "
            "Missing or stale caches are blockers instead of being recomputed on the GPU runtime."
        ),
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=PREDICTION_DIR / "robustness_feature_cache",
        help=(
            "Durable directory for source-bound perturbation feature caches. In Colab this should point "
            "inside the canonical revision-artifact Drive root."
        ),
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=None,
        help="Authenticated Chapman clean-cache NPZ. Defaults to configs.config.PATHS['data_cache'].",
    )
    parser.add_argument(
        "--disk-backed-signals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Extract only X.npy from the authenticated clean-cache NPZ into local scratch and mmap it. "
            "This is required for full-size Colab runs because the cleaned signals exceed CPU-runtime RAM."
        ),
    )
    parser.add_argument(
        "--raw-cache-mmap-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "ECG_RAMBA_ROBUSTNESS_RAW_MMAP_DIR",
                "/content/ecg_ramba_runtime/robustness_raw_cache",
            )
        ),
        help="Local scratch directory for the authenticated clean-signal memmap.",
    )
    parser.add_argument(
        "--perturbation-mmap-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "ECG_RAMBA_ROBUSTNESS_PERTURB_MMAP_DIR",
                "/content/ecg_ramba_runtime/robustness_perturbations",
            )
        ),
        help="Local scratch directory for one exact perturbed-signal memmap at a time.",
    )
    parser.add_argument(
        "--feature-work-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "ECG_RAMBA_ROBUSTNESS_FEATURE_WORK_DIR",
                "/content/ecg_ramba_runtime/robustness_feature_work",
            )
        ),
        help="Local scratch directory for bounded-memory ROCKET-family feature assembly.",
    )
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-metric-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse per-stress/per-metric robustness bootstrap caches when their "
            "input fingerprints match. This makes long Colab aggregation runs resumable."
        ),
    )
    parser.add_argument(
        "--bootstrap-progress-every",
        type=int,
        default=100,
        help="Print paired-bootstrap progress every N resamples; use 0 to disable.",
    )
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "robustness_metric_cache",
        help=(
            "Directory for resumable per-stress/per-metric bootstrap caches. "
            "Use a Drive-backed path on Colab if runtime disconnections are expected."
        ),
    )
    parser.add_argument(
        "--require-existing-stress-predictions",
        action="store_true",
        help=(
            "Fail early unless every requested Full/MiniRocket stress prediction and "
            "the MiniRocket clean robustness reference already exist. This enables "
            "a low-memory final aggregation pass without loading raw Chapman signals."
        ),
    )
    parser.add_argument("--reuse-minirocket-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-perturbed-caches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-legacy-shape-cache", action="store_true")
    parser.add_argument("--full-clean-predictions", type=Path, default=PREDICTION_DIR / "oof_final_ema_predictions.npz")
    parser.add_argument(
        "--minirocket-clean-predictions",
        type=Path,
        default=PREDICTION_DIR / "minirocket_only_oof_predictions.npz",
    )
    parser.add_argument("--freeze-manifest", type=Path, default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json")
    parser.add_argument(
        "--oof-run-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_prediction_run_manifest.json",
        help=(
            "OOF prediction run manifest that records the exact fold checkpoints "
            "used for the frozen Full ECG-RAMBA predictions."
        ),
    )
    parser.add_argument("--minirocket-manifest", type=Path, default=MANIFEST_DIR / "minirocket_only_baseline_manifest.json")
    parser.add_argument("--minirocket-summary", type=Path, default=METRIC_DIR / "minirocket_only_baseline_summary.json")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    return resolve_path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def log_path(path: Path) -> str:
    try:
        return project_relative(path)
    except ValueError:
        return str(resolve_path(path))


class IndexedSignalRows:
    """Expose cleaned Chapman rows without materializing the 10+ GiB signal array."""

    def __init__(self, base: np.ndarray, row_indices: np.ndarray):
        self.base = base
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.shape = (len(self.row_indices), *base.shape[1:])
        self.dtype = base.dtype

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, key):
        return self.base[self.row_indices[key]]


def npz_member_contract(path: Path, member: str) -> tuple[tuple[int, ...], np.dtype, int]:
    """Read an NPY member header from an NPZ without materializing its array."""

    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise KeyError(f"Missing {member} in Chapman clean cache: {path}")
        info = archive.getinfo(member)
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY member version {version}: {path}!{member}")
    return tuple(int(value) for value in shape), np.dtype(dtype), int(info.file_size)


def npz_scalar(payload: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in payload.files:
        return default
    value = payload[key]
    return value.item() if np.ndim(value) == 0 else value


def inspect_disk_backed_chapman_cache(
    *,
    expected_y: np.ndarray,
    expected_record_fingerprint: str,
    expected_archive_sha256: str,
    explicit_cache: Path | None,
    limit_records: int,
    expected_cache_contract: dict | None = None,
) -> dict:
    """Authenticate clean-cache metadata and derive the exact cleaned row mapping."""

    candidates = []
    if explicit_cache is not None:
        candidates.append(resolve_path(explicit_cache))
    candidates.append(Path(PATHS["data_cache"]))
    checked: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if str(path) in seen:
            continue
        seen.add(str(path))
        checked.append(str(path))
        if not path.is_file() or path.stat().st_size == 0:
            continue
        print(f"Authenticating disk-backed Chapman clean cache: {path}", flush=True)
        try:
            x_shape, x_dtype, x_member_size = npz_member_contract(path, "X.npy")
            with np.load(path, allow_pickle=True) as payload:
                required = {"y", "X_raw_amp", "subjects"}
                missing = required - set(payload.files)
                if missing:
                    raise KeyError(f"missing keys {sorted(missing)}")
                y_cache = np.asarray(payload["y"], dtype=np.float32)
                raw_amp = np.asarray(payload["X_raw_amp"], dtype=np.float32)
                subjects = np.asarray(payload["subjects"]).astype(str)
                stored_archive_sha = str(npz_scalar(payload, "archive_sha256", "") or "")
                preprocessing_config_sha = str(
                    npz_scalar(payload, "preprocessing_config_sha256", "") or ""
                )
                preprocessing_source_sha = str(
                    npz_scalar(payload, "preprocessing_source_sha256", "") or ""
                )
                cache_schema_version = int(npz_scalar(payload, "cache_schema_version", 0) or 0)
                stored_record_fp = str(npz_scalar(payload, "record_order_fingerprint", "") or "")
        except Exception as exc:
            print(f"  rejected: {type(exc).__name__}: {exc}", flush=True)
            continue
        if x_shape != (len(y_cache), 12, int(SEQ_LEN)) or x_dtype != np.dtype(np.float32):
            print(f"  rejected: unexpected X contract shape={x_shape} dtype={x_dtype}", flush=True)
            continue
        if raw_amp.shape != (len(y_cache), 5) or subjects.shape != (len(y_cache),):
            print("  rejected: y/X_raw_amp/subjects shape mismatch", flush=True)
            continue
        if not np.isfinite(y_cache).all() or not np.isfinite(raw_amp).all():
            print("  rejected: non-finite labels or amplitude features", flush=True)
            continue
        if not np.logical_or(y_cache == 0, y_cache == 1).all():
            print("  rejected: labels are not binary", flush=True)
            continue
        raw_record_fp = record_order_fingerprint(subjects)
        if stored_record_fp and stored_record_fp != raw_record_fp:
            print("  rejected: stored raw-cache record fingerprint mismatch", flush=True)
            continue
        provenance_mode = "archive_metadata_bound"
        if expected_archive_sha256 and stored_archive_sha != expected_archive_sha256:
            if stored_archive_sha:
                print(
                    f"  rejected: archive SHA {stored_archive_sha} != frozen {expected_archive_sha256}",
                    flush=True,
                )
                continue
            expected_cache_contract = expected_cache_contract or {}
            expected_path = expected_cache_contract.get("path")
            expected_size = expected_cache_contract.get("size_bytes")
            expected_fp = expected_cache_contract.get("record_order_fingerprint")
            if (
                not expected_path
                or Path(str(expected_path)).expanduser().resolve() != path
                or int(expected_size or -1) != int(path.stat().st_size)
                or not expected_fp
                or str(expected_fp) != str(expected_record_fingerprint)
            ):
                print(
                    "  rejected: legacy cache lacks archive SHA and does not match the "
                    "SHA-frozen OOF run-manifest path/size/order contract",
                    flush=True,
                )
                continue
            provenance_mode = "frozen_oof_legacy_cache_content_bound"

        keep_classes = y_cache.sum(axis=0) >= 5
        y_clean = y_cache[:, keep_classes]
        row_indices = np.arange(len(y_cache), dtype=np.int64)
        if not keep_classes.all():
            valid_rows = y_clean.sum(axis=1) > 0
            row_indices = row_indices[valid_rows]
            y_clean = y_clean[valid_rows]
            raw_amp = raw_amp[valid_rows]
            subjects = subjects[valid_rows]
        if y_clean.shape[1] != len(CLASSES):
            print(f"  rejected: cleaned class count {y_clean.shape[1]} != {len(CLASSES)}", flush=True)
            continue
        if limit_records > 0:
            row_indices = row_indices[:limit_records]
            y_clean = y_clean[:limit_records]
            raw_amp = raw_amp[:limit_records]
            subjects = subjects[:limit_records]
        if y_clean.shape != expected_y.shape or not np.array_equal(y_clean, expected_y):
            print(
                f"  rejected: cleaned labels differ from frozen OOF: {y_clean.shape} != {expected_y.shape}",
                flush=True,
            )
            continue
        cleaned_record_fp = record_order_fingerprint(subjects)
        if limit_records == 0 and expected_record_fingerprint and cleaned_record_fp != expected_record_fingerprint:
            print(
                f"  rejected: cleaned record fingerprint {cleaned_record_fp} != frozen "
                f"{expected_record_fingerprint}",
                flush=True,
            )
            continue
        source_sha = sha256_file(path)
        return {
            "path": path,
            "sha256": source_sha,
            "size_bytes": path.stat().st_size,
            "x_shape": x_shape,
            "x_dtype": str(x_dtype),
            "x_member_size_bytes": x_member_size,
            "row_indices": row_indices,
            "y": y_clean.astype(np.float32),
            "X_raw_amp": raw_amp.astype(np.float32),
            "subjects": subjects,
            "record_order_fingerprint": cleaned_record_fp,
            "raw_record_order_fingerprint": raw_record_fp,
            "archive_sha256": stored_archive_sha,
            "provenance_mode": provenance_mode,
            "legacy_cache_attestation": (
                dict(expected_cache_contract or {})
                if provenance_mode == "frozen_oof_legacy_cache_content_bound"
                else None
            ),
            "cache_schema_version": cache_schema_version,
            "preprocessing_source_sha256": preprocessing_source_sha,
            "preprocessing_config_sha256": preprocessing_config_sha,
        }
    raise FileNotFoundError(
        "No authenticated Chapman clean cache satisfies the frozen OOF contract. Checked:\n- "
        + "\n- ".join(checked)
    )


def open_disk_backed_chapman_signals(cache: dict, mmap_dir: Path) -> IndexedSignalRows:
    """Extract X.npy atomically to local scratch and return only the cleaned indexed rows."""

    source = Path(cache["path"])
    mmap_dir = resolve_path(mmap_dir)
    mmap_dir.mkdir(parents=True, exist_ok=True)
    destination = mmap_dir / (
        f"chapman_clean_x_{cache['sha256'][:16]}_N{cache['x_shape'][0]}.npy"
    )
    contract_path = destination.with_suffix(destination.suffix + ".contract.json")
    expected_contract = {
        "schema_version": 1,
        "source_npz_sha256": cache["sha256"],
        "source_npz_size_bytes": cache["size_bytes"],
        "member": "X.npy",
        "member_size_bytes": cache["x_member_size_bytes"],
        "shape": list(cache["x_shape"]),
        "dtype": cache["x_dtype"],
    }
    with exclusive_cache_writer(destination):
        reusable = False
        if destination.is_file() and contract_path.is_file():
            try:
                stored = json.loads(contract_path.read_text(encoding="utf-8"))
                reusable = (
                    all(stored.get(key) == value for key, value in expected_contract.items())
                    and destination.stat().st_size == cache["x_member_size_bytes"]
                    and sha256_file(destination) == stored.get("member_npy_sha256")
                )
            except Exception as exc:
                print(f"Rejecting local Chapman memmap contract: {type(exc).__name__}: {exc}", flush=True)
        if not reusable:
            destination.unlink(missing_ok=True)
            contract_path.unlink(missing_ok=True)
            required = int(cache["x_member_size_bytes"]) + 512 * 1024 * 1024
            free = shutil.disk_usage(mmap_dir).free
            if free < required:
                raise RuntimeError(
                    f"Insufficient local scratch for Chapman X.npy: free={free}, required>{required}"
                )
            partial = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
            partial.unlink(missing_ok=True)
            print(f"Extracting authenticated X.npy to local scratch: {destination}", flush=True)
            with zipfile.ZipFile(source) as archive, archive.open("X.npy") as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if partial.stat().st_size != cache["x_member_size_bytes"]:
                partial.unlink(missing_ok=True)
                raise RuntimeError("Extracted Chapman X.npy size mismatch")
            member_sha = sha256_file(partial)
            os.replace(partial, destination)
            save_json(contract_path, {**expected_contract, "member_npy_sha256": member_sha})
    base = np.load(destination, mmap_mode="r", allow_pickle=False)
    if base.shape != tuple(cache["x_shape"]) or base.dtype != np.float32:
        raise RuntimeError(f"Local Chapman memmap contract mismatch: {base.shape}, {base.dtype}")
    print(
        f"Using disk-backed Chapman signals: raw={base.shape[0]} cleaned={len(cache['row_indices'])} "
        f"path={destination}",
        flush=True,
    )
    return IndexedSignalRows(base, cache["row_indices"])


def perturb_signals_disk_backed(
    X: IndexedSignalRows | np.ndarray,
    spec: dict,
    *,
    out_dir: Path,
    raw_cache_sha256: str,
    source_bundle_sha256: str,
) -> tuple[np.memmap, dict, Path]:
    """Apply the canonical perturbation in bounded chunks into one local memmap."""

    out_dir = resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / (
        f"{spec['name']}_{stable_hash(spec)}_{raw_cache_sha256[:16]}_"
        f"{source_bundle_sha256[:16]}_N{len(X)}.npy"
    )
    contract_path = output.with_suffix(output.suffix + ".contract.json")
    expected = {
        "schema_version": 1,
        "raw_cache_sha256": raw_cache_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "stress_spec": spec,
        "shape": list(X.shape),
        "dtype": "float32",
    }
    with exclusive_cache_writer(output):
        reusable = False
        metadata: dict = {}
        if output.is_file() and contract_path.is_file():
            try:
                stored = json.loads(contract_path.read_text(encoding="utf-8"))
                reusable = (
                    all(stored.get(key) == value for key, value in expected.items())
                    and sha256_file(output) == stored.get("output_npy_sha256")
                )
                metadata = stored.get("metadata") or {}
            except Exception as exc:
                print(f"Rejecting local perturbation memmap: {type(exc).__name__}: {exc}", flush=True)
        if not reusable:
            output.unlink(missing_ok=True)
            contract_path.unlink(missing_ok=True)
            required = int(np.prod(X.shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
            free = shutil.disk_usage(out_dir).free
            if free < required + 512 * 1024 * 1024:
                raise RuntimeError(
                    f"Insufficient local scratch for perturbed ECG: free={free}, required>{required}"
                )
            partial = output.with_name(f".{output.name}.partial.{os.getpid()}.npy")
            partial.unlink(missing_ok=True)
            out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=X.shape)
            metadata = {"spec": dict(spec), "disk_backed": True}
            kind = spec["kind"]
            if kind == "additive_noise":
                rng = np.random.default_rng(int(spec["seed"]))
                target = 10.0 ** (float(spec["snr_db"]) / 10.0)
                for start in tqdm(range(0, len(X), 128), desc=f"perturb {spec['name']}"):
                    stop = min(len(X), start + 128)
                    xb = np.asarray(X[start:stop], dtype=np.float32)
                    power = np.mean(xb * xb, axis=(1, 2), keepdims=True)
                    noise_std = np.sqrt(np.maximum(power / target, 1e-12)).astype(np.float32)
                    out[start:stop] = xb + rng.standard_normal(size=xb.shape).astype(np.float32) * noise_std
                metadata["snr_db"] = float(spec["snr_db"])
            elif kind == "random_lead_dropout":
                rng = np.random.default_rng(int(spec["seed"]))
                n_drop = int(spec["n_drop"])
                dropped = np.empty((len(X), n_drop), dtype=np.int16)
                for index in tqdm(range(len(X)), desc=f"perturb {spec['name']}"):
                    out[index] = X[index]
                    leads = np.sort(rng.choice(X.shape[1], size=n_drop, replace=False)).astype(np.int16)
                    out[index, leads, :] = 0.0
                    dropped[index] = leads
                map_dir = EXPERIMENTAL_DIR / "robustness_perturbations"
                map_dir.mkdir(parents=True, exist_ok=True)
                map_path = map_dir / f"{spec['name']}_{stable_hash(spec)}_dropped_leads.npz"
                save_npz_compressed_atomic(
                    map_path,
                    dropped_leads=dropped,
                    stress_json=np.asarray(json.dumps(spec, sort_keys=True)),
                    record_count=np.asarray(len(X), dtype=np.int64),
                )
                metadata.update(
                    {
                        "dropped_leads_shape": list(dropped.shape),
                        "dropped_leads_sha256": hashlib.sha256(
                            np.ascontiguousarray(dropped).view(np.uint8)
                        ).hexdigest(),
                        "lead_indices_file": str(map_path),
                        "lead_indices_file_sha256": sha256_file(map_path),
                    }
                )
            elif kind == "fixed_lead_dropout":
                leads = np.asarray(spec["lead_indices"], dtype=np.int64)
                for start in tqdm(range(0, len(X), 128), desc=f"perturb {spec['name']}"):
                    stop = min(len(X), start + 128)
                    out[start:stop] = X[start:stop]
                    out[start:stop, leads, :] = 0.0
                metadata["lead_indices"] = leads.astype(int).tolist()
            elif kind == "resample_down_up":
                from scipy.signal import resample_poly

                for start in tqdm(range(0, len(X), 64), desc=f"perturb {spec['name']}"):
                    stop = min(len(X), start + 64)
                    down = resample_poly(X[start:stop], up=1, down=2, axis=-1).astype(np.float32)
                    restored = resample_poly(down, up=2, down=1, axis=-1).astype(np.float32)
                    if restored.shape[-1] < X.shape[-1]:
                        restored = np.pad(
                            restored,
                            ((0, 0), (0, 0), (0, X.shape[-1] - restored.shape[-1])),
                        )
                    out[start:stop] = restored[..., : X.shape[-1]]
                metadata.update(
                    {
                        "source_hz": int(spec["source_hz"]),
                        "target_hz": int(spec["target_hz"]),
                        "length_restored_to": int(X.shape[-1]),
                    }
                )
            else:
                raise ValueError(f"Unhandled perturbation kind: {kind}")
            out.flush()
            del out
            with partial.open("r+b") as handle:
                os.fsync(handle.fileno())
            output_sha = sha256_file(partial)
            os.replace(partial, output)
            save_json(
                contract_path,
                {**expected, "output_npy_sha256": output_sha, "metadata": metadata},
            )
    stressed = np.load(output, mmap_mode="r", allow_pickle=False)
    if stressed.shape != X.shape or stressed.dtype != np.float32:
        raise RuntimeError(f"Perturbed memmap contract mismatch: {stressed.shape}, {stressed.dtype}")
    return stressed, metadata, output


def close_memmap(array) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def remove_local_perturbation(array, path: Path) -> None:
    close_memmap(array)
    Path(path).unlink(missing_ok=True)
    Path(path).with_suffix(Path(path).suffix + ".contract.json").unlink(missing_ok=True)


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


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_cache_path(cache_dir: Path, stress_name: str, metric_name: str) -> Path:
    cache_dir = cache_dir if cache_dir.is_absolute() else resolve_path(cache_dir)
    return cache_dir / f"{stress_name}_{metric_name}.json"


def load_metric_cache(path: Path, expected_metadata: dict) -> tuple[dict, list[dict]] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("metadata") != expected_metadata:
        return None
    row = payload.get("row")
    samples = payload.get("samples")
    if not isinstance(row, dict) or not isinstance(samples, list):
        return None
    expected_n_boot = int(expected_metadata.get("n_boot", 0))
    if expected_n_boot < 1 or int(row.get("n_boot_valid", -1)) != expected_n_boot:
        return None
    if len(samples) != expected_n_boot:
        return None
    expected_indices = list(range(expected_n_boot))
    try:
        observed_indices = [int(sample["bootstrap_index"]) for sample in samples]
    except (KeyError, TypeError, ValueError):
        return None
    if observed_indices != expected_indices:
        return None
    sample_numeric_fields = (
        "clean_full",
        "stress_full",
        "clean_minirocket",
        "stress_minirocket",
        "degradation_full",
        "degradation_minirocket",
        "robustness_advantage_full_less_degradation",
        "stressed_advantage_full_over_minirocket",
    )
    for sample in samples:
        if sample.get("stress_test") != expected_metadata.get("stress_test"):
            return None
        if sample.get("metric") != expected_metadata.get("metric"):
            return None
        try:
            if not all(math.isfinite(float(sample[field])) for field in sample_numeric_fields):
                return None
        except (KeyError, TypeError, ValueError):
            return None
    required_row_fields = (
        "clean_full",
        "stress_full",
        "degradation_full",
        "clean_minirocket",
        "stress_minirocket",
        "degradation_minirocket",
        "stressed_advantage_full_over_minirocket",
        "stressed_advantage_ci_low",
        "stressed_advantage_ci_high",
        "degradation_advantage_full_less_degradation",
        "degradation_advantage_ci_low",
        "degradation_advantage_ci_high",
    )
    try:
        if not all(math.isfinite(float(row[field])) for field in required_row_fields):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    if row.get("inference_scope") != INFERENCE_SCOPE:
        return None
    if row.get("null_test") != NULL_TEST:
        return None
    if row.get("multiplicity_adjustment") != MULTIPLICITY_ADJUSTMENT:
        return None
    for key, value in row.items():
        lowered = str(key).lower()
        if "p_value" in lowered and value not in (None, "", "not_reported"):
            try:
                if math.isfinite(float(value)):
                    return None
            except (TypeError, ValueError):
                return None
        if "significant" in str(value).lower():
            return None
    return row, samples


def write_metric_cache(path: Path, metadata: dict, row: dict, samples: list[dict]) -> None:
    save_json(
        path,
        json_safe(
            {
                "metadata": metadata,
                "row": row,
                "samples": samples,
            }
        ),
    )


def load_revision_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def metric_specs(threshold: float, n_bins: int) -> list[MetricSpec]:
    return [
        MetricSpec("pr_auc_macro", "ranking", True, macro_pr_auc),
        MetricSpec("roc_auc_macro", "ranking", True, macro_roc_auc),
        MetricSpec(
            "f1_macro",
            "fixed_threshold",
            True,
            lambda y, p: multilabel_metrics(y, p, threshold=threshold)["f1_macro"],
        ),
        MetricSpec(
            "brier_macro",
            "calibration",
            False,
            lambda y, p: calibration_summary(y, p, n_bins=n_bins)["brier_macro"],
        ),
        MetricSpec(
            "ece_macro",
            "calibration",
            False,
            lambda y, p: calibration_summary(y, p, n_bins=n_bins)["ece_macro"],
        ),
    ]


def metric_value(spec: MetricSpec, y: np.ndarray, prob: np.ndarray) -> float:
    value = float(spec.fn(y, prob))
    return value if np.isfinite(value) else math.nan


def degradation(clean_value: float, stress_value: float, spec: MetricSpec) -> float:
    if not np.isfinite(clean_value) or not np.isfinite(stress_value):
        return math.nan
    return clean_value - stress_value if spec.higher_is_better else stress_value - clean_value


def load_prediction_npz(path: Path, label: str) -> dict:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} predictions: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "fold_id", "class_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing required keys: {sorted(missing)}")
        payload = {
            "path": path,
            "sha256": sha256_file(path),
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "y_prob": np.asarray(data["y_prob"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"], dtype=np.int64),
            "class_names": np.asarray(data["class_names"]).astype(str).tolist(),
            "fold_id": np.asarray(data["fold_id"], dtype=np.int16) if "fold_id" in data.files else None,
            "metadata": {
                key: (data[key].item() if np.ndim(data[key]) == 0 else data[key].tolist())
                for key in data.files
                if key
                in {
                    "dataset",
                    "protocol",
                    "checkpoint_kind",
                    "dataset_record_order_fingerprint",
                    "aggregation_method",
                    "aggregation_q",
                    "aggregation_implementation",
                    "feature_contract",
                    "feature_preprocessing",
                    "manuscript_ready",
                }
            },
        }
    y_true = payload["y_true"]
    y_prob = payload["y_prob"]
    if y_true.ndim != 2 or y_prob.shape != y_true.shape:
        raise ValueError(f"{label} shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")
    if len(payload["record_id"]) != len(y_true):
        raise ValueError(f"{label} record_id length mismatch")
    if payload["fold_id"] is not None and len(payload["fold_id"]) != len(y_true):
        raise ValueError(f"{label} fold_id length mismatch")
    if payload["class_names"] != CLASSES:
        raise ValueError(f"{label} class_names do not match config CLASSES")
    if not np.array_equal(payload["record_id"], np.arange(len(y_true), dtype=np.int64)):
        raise ValueError(f"{label} record_id must be exactly 0..N-1")
    if not np.isfinite(y_prob).all():
        raise ValueError(f"{label} probabilities contain non-finite values")
    payload["y_prob"] = np.clip(y_prob, 0.0, 1.0).astype(np.float32)
    return payload


def validate_clean_prediction_contract(
    full: dict,
    mini: dict | None,
    args: argparse.Namespace,
    *,
    require_minirocket: bool = True,
) -> dict:
    if require_minirocket and mini is None:
        raise ValueError("MiniRocket clean predictions are required outside features-only mode.")
    if mini is not None and not np.array_equal(full["y_true"], mini["y_true"]):
        raise ValueError("Full and MiniRocket clean y_true arrays differ.")
    if mini is not None and not np.array_equal(full["record_id"], mini["record_id"]):
        raise ValueError("Full and MiniRocket clean record_id arrays differ.")
    if mini is not None and full["class_names"] != mini["class_names"]:
        raise ValueError("Full and MiniRocket class_names differ.")
    if full["fold_id"] is None:
        raise ValueError("Full clean predictions must include fold_id.")
    if mini is not None and mini["fold_id"] is not None and not np.array_equal(full["fold_id"], mini["fold_id"]):
        raise ValueError("Full and MiniRocket clean fold_id arrays differ.")

    freeze_path = resolve_path(args.freeze_manifest)
    if not freeze_path.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("manuscript_ready") is not True:
        raise ValueError("Freeze manifest must be status=frozen and manuscript_ready=true.")
    if freeze.get("checkpoint_kind") != args.expected_checkpoint_kind:
        raise ValueError(
            f"Unexpected freeze checkpoint kind {freeze.get('checkpoint_kind')} "
            f"!= {args.expected_checkpoint_kind}"
        )
    artifacts = {row.get("path"): row for row in freeze.get("artifacts", [])}
    rel = project_relative(full["path"])
    if rel not in artifacts:
        raise ValueError(f"Freeze manifest does not include Full predictions: {rel}")
    if full["sha256"] != artifacts[rel].get("sha256"):
        raise RuntimeError("Full clean prediction SHA does not match freeze manifest.")

    group_contract = freeze.get("group_contract") or {}
    sidecar_artifact = group_contract.get("sidecar") or {}
    sidecar_path_text = sidecar_artifact.get("path")
    if (
        group_contract.get("status") != "verified"
        or group_contract.get("one_record_per_group") is not True
        or not sidecar_path_text
        or not sidecar_artifact.get("sha256")
    ):
        raise RuntimeError("Freeze manifest lacks a verified one-record-per-group sidecar contract.")
    sidecar_path = resolve_path(Path(str(sidecar_path_text)))
    if not sidecar_path.exists() or sidecar_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing authenticated OOF group sidecar: {sidecar_path}")
    sidecar_sha256 = sha256_file(sidecar_path)
    if sidecar_sha256 != sidecar_artifact.get("sha256"):
        raise RuntimeError("OOF group sidecar SHA does not match the frozen group contract.")
    if int(group_contract.get("n_records", -1)) != int(full["y_true"].shape[0]):
        raise RuntimeError("OOF group contract record count does not match clean predictions.")
    if int(group_contract.get("n_groups", -1)) != int(full["y_true"].shape[0]):
        raise RuntimeError("Robustness record bootstrap requires the reviewed one-record-per-group contract.")

    contract = {
        "freeze_manifest": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
            "checkpoint_kind": freeze.get("checkpoint_kind"),
            "validated_records": freeze.get("validated_records"),
            "dataset_record_order_fingerprint": freeze.get("dataset_record_order_fingerprint"),
            "source_archive_sha256": (group_contract.get("source_archive") or {}).get("sha256"),
        },
        "group_contract": {
            "status": group_contract.get("status"),
            "one_record_per_group": group_contract.get("one_record_per_group"),
            "n_records": group_contract.get("n_records"),
            "n_groups": group_contract.get("n_groups"),
            "bootstrap_unit": group_contract.get("bootstrap_unit"),
            "group_semantics": group_contract.get("group_semantics"),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sidecar_sha256,
        },
    }
    if not require_minirocket:
        return contract

    mini_manifest = resolve_path(args.minirocket_manifest)
    mini_summary = resolve_path(args.minirocket_summary)
    if not mini_manifest.exists() or not mini_summary.exists():
        raise FileNotFoundError("MiniRocket baseline manifest/summary are required before robustness tests.")
    mini_manifest_payload = json.loads(mini_manifest.read_text(encoding="utf-8"))
    mini_summary_payload = json.loads(mini_summary.read_text(encoding="utf-8"))
    for source_name, payload in [("summary", mini_summary_payload), ("manifest", mini_manifest_payload)]:
        if payload.get("protocol") != EXPECTED_MINIROCKET_PROTOCOL:
            raise ValueError(
                f"MiniRocket {source_name} protocol mismatch: "
                f"{payload.get('protocol')} != {EXPECTED_MINIROCKET_PROTOCOL}"
            )
        if payload.get("feature_contract") != "minirocket_raw":
            raise ValueError(f"MiniRocket {source_name} feature_contract must be minirocket_raw.")
    if mini_summary_payload.get("manuscript_ready") is not True:
        raise ValueError("MiniRocket summary must report manuscript_ready=true.")
    if int(mini_summary_payload.get("n_records", -1)) != int(full["y_true"].shape[0]):
        raise ValueError("MiniRocket summary n_records does not match Full clean predictions.")
    manifest_prediction_sha = (mini_manifest_payload.get("artifact_sha256") or {}).get("predictions")
    if manifest_prediction_sha and manifest_prediction_sha != mini["sha256"]:
        raise RuntimeError(
            f"MiniRocket prediction SHA mismatch: manifest {manifest_prediction_sha} != file {mini['sha256']}"
        )
    contract.update({
        "minirocket_manifest": {
            "path": str(mini_manifest),
            "sha256": sha256_file(mini_manifest),
            "protocol": mini_manifest_payload.get("protocol"),
            "prediction_sha256": manifest_prediction_sha,
        },
        "minirocket_summary": {
            "path": str(mini_summary),
            "sha256": sha256_file(mini_summary),
            "protocol": mini_summary_payload.get("protocol"),
            "manuscript_ready": mini_summary_payload.get("manuscript_ready"),
            "n_records": mini_summary_payload.get("n_records"),
        },
    })
    return contract


def frozen_oof_raw_cache_contract(args: argparse.Namespace, contract: dict) -> dict:
    """Authenticate the legacy clean-cache identity recorded by the frozen OOF run."""

    manifest_path = resolve_path(args.oof_run_manifest)
    if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing frozen OOF run manifest: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    freeze_path = Path(contract["freeze_manifest"]["path"])
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_sha = (freeze.get("generation_provenance") or {}).get("run_manifest_sha256")
    if not frozen_sha or manifest_sha != frozen_sha:
        raise RuntimeError(
            "OOF run manifest is not authenticated by the freeze generation provenance: "
            f"observed={manifest_sha} frozen={frozen_sha}"
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = ((payload.get("config") or {}).get("paths") or {}).get("data_cache") or {}
    path = cache.get("path")
    size_bytes = cache.get("size_bytes")
    record_fp = payload.get("dataset_record_order_fingerprint")
    source_config_hash = (payload.get("config") or {}).get("source_runtime_config_hash")
    if not path or int(size_bytes or -1) <= 0 or not record_fp or not source_config_hash:
        raise RuntimeError("Frozen OOF run manifest lacks the legacy clean-cache identity contract.")
    return {
        "path": str(Path(str(path)).expanduser().resolve()),
        "size_bytes": int(size_bytes),
        "record_order_fingerprint": str(record_fp),
        "source_config_hash": str(source_config_hash),
        "oof_run_manifest_path": str(manifest_path),
        "oof_run_manifest_sha256": manifest_sha,
        "attestation_scope": "path_size_record_order_plus_exact_label_equality",
    }


def fold_list_from_fold_id(fold_id: np.ndarray) -> list[dict[str, np.ndarray]]:
    folds = []
    for fold in sorted(int(x) for x in np.unique(fold_id) if int(x) > 0):
        va_idx = np.where(fold_id == fold)[0].astype(np.int64)
        tr_idx = np.where((fold_id > 0) & (fold_id != fold))[0].astype(np.int64)
        if len(tr_idx) and len(va_idx):
            folds.append({"fold": fold, "tr_idx": tr_idx, "va_idx": va_idx})
    if not folds:
        raise ValueError("Could not derive folds from fold_id.")
    return folds


def stable_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def stress_feature_hash(stress: dict, contract: dict, record_fp: str) -> str:
    """Bind perturbation caches to data content, ordering, code, and protocol."""

    freeze = contract["freeze_manifest"]
    return stable_hash(
        {
            "stress": stress,
            "protocol": PROTOCOL,
            "source_bundle_sha256": source_bundle_contract()["sha256"],
            "freeze_manifest_sha256": freeze["sha256"],
            "source_archive_sha256": freeze.get("source_archive_sha256"),
            "record_order_fingerprint": record_fp,
        }
    )


def feature_cache_hash(stress_hash: str, raw_cache: dict | None) -> str:
    """Bind cached features to the exact raw-cache bytes and preprocessing provenance."""

    if raw_cache is None:
        return stable_hash({"stress_hash": stress_hash, "raw_cache": "in_memory_unbound"})
    return stable_hash(
        {
            "stress_hash": stress_hash,
            "raw_cache_sha256": raw_cache.get("sha256"),
            "raw_cache_schema_version": raw_cache.get("cache_schema_version"),
            "preprocessing_source_sha256": raw_cache.get("preprocessing_source_sha256"),
            "preprocessing_config_sha256": raw_cache.get("preprocessing_config_sha256"),
        }
    )


def expected_weights_kind(checkpoint_kind: str) -> str | None:
    if checkpoint_kind.endswith("_ema"):
        return "ema"
    if checkpoint_kind.endswith("_raw"):
        return "raw"
    return None


def load_oof_checkpoint_contract(args: argparse.Namespace) -> dict:
    """Load exact checkpoint paths/SHA from the frozen OOF run manifest."""
    manifest_path = resolve_path(args.oof_run_manifest)
    if not manifest_path.exists():
        print(
            f"WARNING: OOF run manifest not found: {manifest_path}. "
            "Falling back to config model_dir checkpoint lookup.",
            flush=True,
        )
        return {
            "path": str(manifest_path),
            "sha256": None,
            "status": "missing_fallback_to_model_dir",
            "checkpoints": {},
        }

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_kind = payload.get("checkpoint_kind_requested")
    if requested_kind and requested_kind != args.expected_checkpoint_kind:
        raise ValueError(
            f"OOF run manifest checkpoint kind mismatch: {requested_kind} != {args.expected_checkpoint_kind}"
        )
    records = payload.get("checkpoints") or (payload.get("inputs") or {}).get("checkpoints") or []
    if not records:
        raise ValueError(f"OOF run manifest contains no checkpoint records: {manifest_path}")

    checkpoints: dict[int, dict] = {}
    expected_kind = expected_weights_kind(args.expected_checkpoint_kind)
    for row in records:
        fold = int(row["fold"])
        weights_kind = row.get("weights_kind")
        if expected_kind and weights_kind and weights_kind != expected_kind:
            raise ValueError(
                f"OOF run manifest fold {fold} weights_kind mismatch: {weights_kind} != {expected_kind}"
            )
        ckpt_path = resolve_path(Path(row["path"]))
        checkpoints[fold] = {
            "path": str(ckpt_path),
            "sha256": row.get("sha256"),
            "weights_kind": weights_kind,
            "source_config_hash": row.get("source_config_hash"),
            "dataset_record_order_fingerprint": row.get("dataset_record_order_fingerprint"),
        }

    pca_objects: dict[int, dict] = {}
    for row in payload.get("fold_summaries", []):
        fold = int(row.get("fold", 0))
        pca_path_raw = row.get("pca_object_path")
        pca_sha = row.get("pca_object_sha256")
        if fold > 0 and pca_path_raw and pca_sha:
            pca_objects[fold] = {
                "path": str(resolve_path(Path(pca_path_raw))),
                "sha256": str(pca_sha),
            }
    if set(pca_objects) != set(checkpoints):
        raise ValueError(
            "OOF run manifest lacks a complete fold-PCA object contract: "
            f"pca_folds={sorted(pca_objects)} checkpoint_folds={sorted(checkpoints)}"
        )

    print(f"Loaded OOF checkpoint contract: {manifest_path} | folds={sorted(checkpoints)}", flush=True)
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "status": "loaded",
        "protocol": payload.get("protocol"),
        "dataset_record_order_fingerprint": payload.get("dataset_record_order_fingerprint"),
        "checkpoints": checkpoints,
        "pca_objects": pca_objects,
    }


def stress_specs(names: list[str], seed: int) -> list[dict]:
    specs = []
    for name in names:
        key = name.strip()
        if not key:
            continue
        if key == "snr20db":
            specs.append({"name": key, "kind": "additive_noise", "snr_db": 20.0, "seed": seed + 2001})
        elif key == "snr10db":
            specs.append({"name": key, "kind": "additive_noise", "snr_db": 10.0, "seed": seed + 2011})
        elif key == "snr5db":
            specs.append({"name": key, "kind": "additive_noise", "snr_db": 5.0, "seed": seed + 2021})
        elif key == "random_3_lead_dropout":
            specs.append({"name": key, "kind": "random_lead_dropout", "n_drop": 3, "seed": seed + 3001})
        elif key == "precordial_dropout":
            specs.append({"name": key, "kind": "fixed_lead_dropout", "lead_indices": list(range(6, 12)), "seed": seed})
        elif key == "resample_250hz":
            specs.append({"name": key, "kind": "resample_down_up", "source_hz": 500, "target_hz": 250, "seed": seed})
        else:
            raise ValueError(f"Unknown stress test: {key}")
    if not specs:
        raise ValueError("No stress tests requested.")
    return specs


def perturb_signals(X: np.ndarray, spec: dict) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=np.float32)
    meta = {"spec": dict(spec)}
    kind = spec["kind"]
    if kind == "additive_noise":
        rng = np.random.default_rng(int(spec["seed"]))
        out = np.empty_like(X, dtype=np.float32)
        target = 10.0 ** (float(spec["snr_db"]) / 10.0)
        for start in tqdm(range(0, len(X), 128), desc=f"perturb {spec['name']}"):
            stop = min(len(X), start + 128)
            xb = X[start:stop]
            power = np.mean(xb * xb, axis=(1, 2), keepdims=True)
            noise_std = np.sqrt(np.maximum(power / target, 1e-12)).astype(np.float32)
            noise = rng.standard_normal(size=xb.shape).astype(np.float32) * noise_std
            out[start:stop] = xb + noise
        meta["snr_db"] = float(spec["snr_db"])
        return out, meta
    if kind == "random_lead_dropout":
        rng = np.random.default_rng(int(spec["seed"]))
        out = X.copy()
        n_drop = int(spec["n_drop"])
        dropped = np.empty((len(X), n_drop), dtype=np.int16)
        for i in range(len(X)):
            leads = np.sort(rng.choice(X.shape[1], size=n_drop, replace=False)).astype(np.int16)
            out[i, leads, :] = 0.0
            dropped[i] = leads
        map_dir = EXPERIMENTAL_DIR / "robustness_perturbations"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / f"{spec['name']}_{stable_hash(spec)}_dropped_leads.npz"
        np.savez_compressed(
            map_path,
            dropped_leads=dropped,
            stress_json=np.asarray(json.dumps(spec, sort_keys=True)),
            record_count=np.asarray(len(X), dtype=np.int64),
        )
        meta["dropped_leads_shape"] = list(dropped.shape)
        meta["dropped_leads_sha256"] = hashlib.sha256(np.ascontiguousarray(dropped).view(np.uint8)).hexdigest()
        meta["lead_indices_file"] = str(map_path)
        meta["lead_indices_file_sha256"] = sha256_file(map_path)
        return out, meta
    if kind == "fixed_lead_dropout":
        leads = np.asarray(spec["lead_indices"], dtype=np.int64)
        out = X.copy()
        out[:, leads, :] = 0.0
        meta["lead_indices"] = leads.astype(int).tolist()
        return out, meta
    if kind == "resample_down_up":
        from scipy.signal import resample_poly

        out = np.empty_like(X, dtype=np.float32)
        for start in tqdm(range(0, len(X), 64), desc=f"perturb {spec['name']}"):
            stop = min(len(X), start + 64)
            down = resample_poly(X[start:stop], up=1, down=2, axis=-1).astype(np.float32)
            restored = resample_poly(down, up=2, down=1, axis=-1).astype(np.float32)
            if restored.shape[-1] < X.shape[-1]:
                pad = X.shape[-1] - restored.shape[-1]
                restored = np.pad(restored, ((0, 0), (0, 0), (0, pad)), mode="constant")
            out[start:stop] = restored[..., : X.shape[-1]]
        meta["source_hz"] = int(spec["source_hz"])
        meta["target_hz"] = int(spec["target_hz"])
        meta["length_restored_to"] = int(X.shape[-1])
        return out, meta
    raise ValueError(f"Unhandled perturbation kind: {kind}")


def robust_feature_cache_path(
    cache_dir: Path,
    prefix: str,
    stress_name: str,
    stress_hash: str,
    n_records: int,
    record_fp: str,
) -> Path:
    cache_dir = resolve_path(cache_dir)
    return cache_dir / f"{prefix}_{stress_name}_{stress_hash}_N{n_records}_R{record_fp}.npz"


def feature_cache_contract_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".contract.json")


def inspect_feature_cache(
    cache_path: Path,
    *,
    feature_kind: str,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
    stress_name: str,
    stress_hash: str,
    record_fp: str,
) -> dict | None:
    """Validate an atomic source-bound feature cache without loading its large X array."""

    cache_path = resolve_path(cache_path)
    sidecar_path = feature_cache_contract_path(cache_path)
    if not cache_path.is_file() or not sidecar_path.is_file():
        return None
    expected = {
        "schema_version": 1,
        "feature_kind": feature_kind,
        "shape": list(expected_shape),
        "storage_dtype": str(np.dtype(expected_dtype)),
        "stress_name": stress_name,
        "stress_hash": stress_hash,
        "record_order_fingerprint": record_fp,
    }
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not all(sidecar.get(key) == value for key, value in expected.items()):
            return None
        if sha256_file(cache_path) != sidecar.get("cache_file_sha256"):
            return None
        shape, dtype, _member_size = npz_member_contract(cache_path, "X.npy")
        if shape != expected_shape or dtype != np.dtype(expected_dtype):
            return None
        with np.load(cache_path, allow_pickle=False) as payload:
            if str(npz_scalar(payload, "stress_name", "")) != stress_name:
                return None
            if str(npz_scalar(payload, "stress_hash", "")) != stress_hash:
                return None
            if str(npz_scalar(payload, "record_order_fingerprint", "")) != record_fp:
                return None
        return {
            "path": str(cache_path),
            "sha256": sidecar["cache_file_sha256"],
            "cache_hit": True,
            "storage_dtype": str(dtype),
            "feature_array_sha256": sidecar.get("feature_array_sha256"),
            "contract_path": str(sidecar_path),
            "contract_sha256": sha256_file(sidecar_path),
        }
    except Exception as exc:
        print(f"Rejecting feature cache {cache_path.name}: {type(exc).__name__}: {exc}", flush=True)
        return None


def write_feature_cache_contract(
    cache_path: Path,
    *,
    feature_kind: str,
    features: np.ndarray,
    stress_name: str,
    stress_hash: str,
    record_fp: str,
) -> dict:
    sidecar_path = feature_cache_contract_path(cache_path)
    contract = {
        "schema_version": 1,
        "feature_kind": feature_kind,
        "shape": list(features.shape),
        "storage_dtype": str(features.dtype),
        "stress_name": stress_name,
        "stress_hash": stress_hash,
        "record_order_fingerprint": record_fp,
        "feature_array_sha256": ndarray_sha256(features),
        "cache_file_sha256": sha256_file(cache_path),
    }
    save_json(sidecar_path, contract)
    return {
        "path": str(cache_path),
        "sha256": contract["cache_file_sha256"],
        "cache_hit": False,
        "storage_dtype": str(features.dtype),
        "feature_array_sha256": contract["feature_array_sha256"],
        "contract_path": str(sidecar_path),
        "contract_sha256": sha256_file(sidecar_path),
    }


def generate_minirocket_features(
    X: np.ndarray,
    *,
    stress_name: str,
    stress_hash: str,
    record_fp: str,
    batch_size: int,
    device_name: str,
    save_cache: bool,
    cache_dir: Path,
    require_existing: bool = False,
    materialize: bool = True,
    work_dir: Path | None = None,
) -> tuple[np.ndarray | None, dict]:
    cache_path = robust_feature_cache_path(
        cache_dir,
        "robust_minirocket_raw",
        stress_name,
        stress_hash,
        len(X),
        record_fp,
    )
    cached_info = inspect_feature_cache(
        cache_path,
        feature_kind="fixed_seed_rocket_family_max_ppv",
        expected_shape=(len(X), 20000),
        expected_dtype=np.dtype(np.float16),
        stress_name=stress_name,
        stress_hash=stress_hash,
        record_fp=record_fp,
    )
    if cached_info is not None:
        print(f"Loaded perturbed ROCKET-family cache contract: {cache_path}", flush=True)
        if not materialize:
            return None, cached_info
        with np.load(cache_path, allow_pickle=False) as payload:
            feats = np.asarray(payload["X"], dtype=np.float32)
        if not np.isfinite(feats).all():
            raise RuntimeError(f"Authenticated ROCKET-family cache contains non-finite values: {cache_path}")
        return feats, cached_info

    if require_existing:
        raise FileNotFoundError(
            f"Required source-bound perturbed ROCKET-family cache is missing or invalid: {cache_path}"
        )

    device = torch.device(device_name)
    model = MiniRocketNative(c_in=X.shape[1], seq_len=X.shape[-1], num_kernels=10000, seed=42).to(device).eval()
    work_dir = resolve_path(work_dir or Path("/tmp/ecg_ramba_robustness_feature_work"))
    work_dir.mkdir(parents=True, exist_ok=True)
    work_path = work_dir / f"rocket_{stress_name}_{stress_hash}_N{len(X)}.f16.npy"
    work_path.unlink(missing_ok=True)
    feature_store = np.lib.format.open_memmap(
        work_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(X), 20000),
    )
    try:
        with torch.no_grad():
            for start in tqdm(
                range(0, len(X), batch_size),
                desc=f"ROCKET-family {stress_name}",
                unit="batch",
            ):
                stop = min(len(X), start + batch_size)
                signal_batch = np.array(X[start:stop], dtype=np.float32, copy=True, order="C")
                xb = torch.as_tensor(signal_batch, dtype=torch.float32, device=device)
                feature_store[start:stop] = model(xb).detach().cpu().numpy().astype(np.float16)
        feature_store.flush()
        for start in range(0, len(feature_store), 256):
            if not np.isfinite(feature_store[start : start + 256]).all():
                raise RuntimeError(f"ROCKET-family extraction produced non-finite values near row {start}")
        if not save_cache:
            if materialize:
                return np.asarray(feature_store, dtype=np.float32), {
                    "path": str(cache_path),
                    "sha256": None,
                    "cache_hit": False,
                    "feature_device": device_name,
                }
            raise ValueError("materialize=False requires save_cache=True")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz_compressed_atomic(
            cache_path,
            X=feature_store,
            storage_dtype=np.asarray("float16"),
            consumer_dtype=np.asarray("float32"),
            stress_name=np.asarray(stress_name),
            stress_hash=np.asarray(stress_hash),
            record_order_fingerprint=np.asarray(record_fp),
        )
        info = write_feature_cache_contract(
            cache_path,
            feature_kind="fixed_seed_rocket_family_max_ppv",
            features=feature_store,
            stress_name=stress_name,
            stress_hash=stress_hash,
            record_fp=record_fp,
        )
        info["feature_device"] = device_name
        print(f"Saved perturbed ROCKET-family cache: {cache_path}", flush=True)
        return (np.asarray(feature_store, dtype=np.float32) if materialize else None), info
    finally:
        close_memmap(feature_store)
        work_path.unlink(missing_ok=True)


def generate_hrv36_features(
    X: np.ndarray,
    X_raw_amp: np.ndarray,
    *,
    stress_name: str,
    stress_hash: str,
    record_fp: str,
    save_cache: bool,
    cache_dir: Path,
    require_existing: bool = False,
    materialize: bool = True,
) -> tuple[np.ndarray | None, dict]:
    cache_path = robust_feature_cache_path(
        cache_dir,
        "robust_hrv36",
        stress_name,
        stress_hash,
        len(X),
        record_fp,
    )
    cached_info = inspect_feature_cache(
        cache_path,
        feature_kind="hrv36_perturbed_signal_amp_contract",
        expected_shape=(len(X), int(CONFIG["hrv_dim"])),
        expected_dtype=np.dtype(np.float16),
        stress_name=stress_name,
        stress_hash=stress_hash,
        record_fp=record_fp,
    )
    if cached_info is not None:
        print(f"Loaded perturbed HRV36 cache contract: {cache_path}", flush=True)
        if not materialize:
            return None, cached_info
        with np.load(cache_path, allow_pickle=False) as payload:
            feats = np.asarray(payload["X"], dtype=np.float32)
        if not np.isfinite(feats).all():
            raise RuntimeError(f"Authenticated HRV36 cache contains non-finite values: {cache_path}")
        return feats, cached_info

    if require_existing:
        raise FileNotFoundError(
            f"Required source-bound perturbed HRV cache is missing or invalid: {cache_path}"
        )

    feats = np.zeros((len(X), int(CONFIG["hrv_dim"])), dtype=np.float32)
    for i, sig in enumerate(tqdm(X, desc=f"HRV36 {stress_name}")):
        hrv = extract_hrv_features(sig)
        amp = extract_amplitude_features(X_raw_amp[i])
        gstat = extract_global_record_stats(sig)
        feats[i] = np.concatenate([hrv, amp, gstat]).astype(np.float32)
    if not np.isfinite(feats).all():
        raise RuntimeError("Perturbed HRV36 extraction produced non-finite values")
    info = {"path": str(cache_path), "sha256": None, "cache_hit": False}
    if save_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz_compressed_atomic(
            cache_path,
            X=feats.astype(np.float16),
            storage_dtype=np.asarray("float16"),
            consumer_dtype=np.asarray("float32"),
            stress_name=np.asarray(stress_name),
            stress_hash=np.asarray(stress_hash),
            record_order_fingerprint=np.asarray(record_fp),
            hrv_contract=np.asarray("hrv_and_global_recomputed_on_perturbed_signal_amp_uses_existing_pipeline_contract"),
        )
        stored_features = feats.astype(np.float16)
        info = write_feature_cache_contract(
            cache_path,
            feature_kind="hrv36_perturbed_signal_amp_contract",
            features=stored_features,
            stress_name=stress_name,
            stress_hash=stress_hash,
            record_fp=record_fp,
        )
        print(f"Saved perturbed HRV36 cache: {cache_path}", flush=True)
    return (feats if materialize else None), info


def fit_or_load_minirocket_heads(
    *,
    X_clean: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    clean_cache_info: dict,
    args: argparse.Namespace,
) -> MiniRocketHeads:
    head_dir = EXPERIMENTAL_DIR / "robustness_minirocket_heads"
    manifest_path = MANIFEST_DIR / "robustness_minirocket_heads_manifest.json"
    clean_pred_path = PREDICTION_DIR / "robustness_minirocket_clean_ref_predictions.npz"
    params = {
        "protocol": MINIROCKET_HEAD_PROTOCOL,
        "clean_minirocket_cache_sha256": clean_cache_info.get("minirocket_cache_sha256"),
        "epochs": int(args.minirocket_epochs),
        "batch_size": int(args.minirocket_batch_size),
        "stats_batch_size": int(args.minirocket_stats_batch_size),
        "lr": float(args.minirocket_lr),
        "weight_decay": float(args.minirocket_weight_decay),
        "device": args.minirocket_device,
        "allow_tf32": bool(args.allow_tf32),
        "seed": int(args.seed),
        "n_records": int(len(y)),
        "n_classes": int(y.shape[1]),
        "config_hash": EVALUATION_CONFIG_HASH,
    }
    params_hash = stable_hash(params)
    if args.reuse_minirocket_heads and manifest_path.exists() and clean_pred_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("params_hash") == params_hash:
            head_paths = [head_dir / f"fold{fold['fold']}_head.pt" for fold in folds]
            if all(path.exists() for path in head_paths):
                with np.load(clean_pred_path, allow_pickle=False) as data:
                    clean_prob = np.asarray(data["y_prob"], dtype=np.float32)
                    fold_id = np.asarray(data["fold_id"], dtype=np.int16)
                if clean_prob.shape == y.shape and np.all(np.isfinite(clean_prob)):
                    print(f"Reusing MiniRocket robustness heads: {head_dir}", flush=True)
                    return MiniRocketHeads(head_dir, manifest, clean_prob, fold_id, manifest.get("fold_rows", []))

    import torch.nn as nn

    device_name = args.minirocket_device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    head_dir.mkdir(parents=True, exist_ok=True)
    clean_prob = np.full_like(y, np.nan, dtype=np.float32)
    fold_id = np.full(len(y), -1, dtype=np.int16)
    fold_rows: list[dict] = []
    mini = load_revision_module("10_minirocket_only_baseline.py", "_ecg_ramba_minirocket_baseline_for_robustness")

    for fold in folds:
        fold_num = int(fold["fold"])
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        rng = np.random.default_rng(args.seed + fold_num)
        torch.manual_seed(args.seed + fold_num)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + fold_num)
        print(f"MiniRocket clean head fold {fold_num}/5 | train={len(tr_idx)} | val={len(va_idx)}", flush=True)
        mean, std = mini.compute_train_standardization(
            X_clean,
            tr_idx,
            batch_size=args.minirocket_stats_batch_size,
        )
        pos = np.sum(y[tr_idx], axis=0).astype(np.float32)
        neg = float(len(tr_idx)) - pos
        pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0).astype(np.float32)
        model = nn.Linear(X_clean.shape[1], y.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.minirocket_lr, weight_decay=args.minirocket_weight_decay)
        model.train()
        for epoch in range(1, args.minirocket_epochs + 1):
            total_loss = 0.0
            total_seen = 0
            for batch_idx in mini.iter_index_batches(
                tr_idx,
                args.minirocket_batch_size,
                shuffle=True,
                rng=rng,
            ):
                xb = mini.prepare_feature_batch(X_clean, batch_idx, mean=mean, std=std)
                xb_t = torch.as_tensor(xb, dtype=torch.float32, device=device)
                yb_t = torch.as_tensor(y[batch_idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb_t), yb_t)
                loss.backward()
                optimizer.step()
                batch_n = int(len(batch_idx))
                total_loss += float(loss.detach().cpu()) * batch_n
                total_seen += batch_n
            print(f"  MiniRocket fold {fold_num}: epoch {epoch:02d}/{args.minirocket_epochs} loss={total_loss / max(total_seen, 1):.5f}", flush=True)
        model.eval()
        with torch.no_grad():
            for batch_idx in mini.iter_index_batches(va_idx, args.minirocket_batch_size, shuffle=False, rng=rng):
                xb = mini.prepare_feature_batch(X_clean, batch_idx, mean=mean, std=std)
                probs = torch.sigmoid(model(torch.as_tensor(xb, dtype=torch.float32, device=device))).detach().cpu().numpy()
                clean_prob[batch_idx] = probs.astype(np.float32)
        fold_id[va_idx] = fold_num
        head_path = head_dir / f"fold{fold_num}_head.pt"
        torch.save(
            {
                "fold": fold_num,
                "weight": model.weight.detach().cpu(),
                "bias": model.bias.detach().cpu(),
                "mean": torch.as_tensor(mean.astype(np.float32)),
                "std": torch.as_tensor(std.astype(np.float32)),
                "params_hash": params_hash,
                "protocol": MINIROCKET_HEAD_PROTOCOL,
            },
            head_path,
        )
        fold_rows.append(
            {
                "fold": fold_num,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "head_path": str(head_path),
                "head_sha256": sha256_file(head_path),
                "epochs": int(args.minirocket_epochs),
            }
        )
        del model, optimizer, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.any(fold_id < 0) or not np.isfinite(clean_prob).all():
        raise RuntimeError("MiniRocket robustness clean-head prediction coverage is incomplete.")
    save_npz_compressed_atomic(
        clean_pred_path,
        y_true=y.astype(np.float32),
        y_prob=np.clip(clean_prob, 0.0, 1.0).astype(np.float32),
        record_id=np.arange(len(y), dtype=np.int64),
        fold_id=fold_id.astype(np.int16),
        class_names=np.asarray(CLASSES),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(MINIROCKET_HEAD_PROTOCOL),
        feature_contract=np.asarray("minirocket_raw"),
        feature_preprocessing=np.asarray("fold_train_standardization"),
        threshold=np.asarray(float(args.threshold)),
        manuscript_ready=np.asarray(True),
    )
    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "protocol": MINIROCKET_HEAD_PROTOCOL,
        "params": params,
        "params_hash": params_hash,
        "head_dir": str(head_dir),
        "clean_prediction_file": str(clean_pred_path),
        "clean_prediction_sha256": sha256_file(clean_pred_path),
        "fold_rows": fold_rows,
    }
    save_json(manifest_path, json_safe(manifest))
    print(f"Wrote MiniRocket robustness heads manifest: {manifest_path}", flush=True)
    return MiniRocketHeads(head_dir, manifest, clean_prob, fold_id, fold_rows)


def predict_minirocket_heads(
    X_features: np.ndarray,
    y_shape: tuple[int, int],
    folds: list[dict[str, np.ndarray]],
    heads: MiniRocketHeads,
    *,
    batch_size: int,
    device_name: str,
) -> np.ndarray:
    import torch.nn.functional as F

    mini = load_revision_module("10_minirocket_only_baseline.py", "_ecg_ramba_minirocket_baseline_predict_for_robustness")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    out = np.full(y_shape, np.nan, dtype=np.float32)
    for fold in folds:
        fold_num = int(fold["fold"])
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        head = torch.load(
            heads.head_dir / f"fold{fold_num}_head.pt",
            map_location="cpu",
            weights_only=False,
        )
        weight = head["weight"].to(device=device, dtype=torch.float32)
        bias = head["bias"].to(device=device, dtype=torch.float32)
        mean = head["mean"].detach().cpu().numpy().astype(np.float32)
        std = head["std"].detach().cpu().numpy().astype(np.float32)
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for batch_idx in mini.iter_index_batches(va_idx, batch_size, shuffle=False, rng=rng):
                xb = mini.prepare_feature_batch(X_features, batch_idx, mean=mean, std=std)
                logits = F.linear(torch.as_tensor(xb, dtype=torch.float32, device=device), weight, bias)
                out[batch_idx] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        del weight, bias
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(out).all():
        raise RuntimeError("MiniRocket stress prediction contains missing/non-finite values.")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def predict_full_model(
    *,
    X: np.ndarray,
    X_rocket: np.ndarray,
    X_hrv: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
    stress_name: str,
    dataset_record_fingerprint: str,
    checkpoint_contract: dict[int, dict],
    pca_contract: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    gen = load_revision_module("01_generate_predictions.py", "_ecg_ramba_generate_predictions_for_robustness")
    n_records, n_classes = y.shape
    oof_prob = np.zeros((n_records, n_classes), dtype=np.float32)
    fold_id = np.full(n_records, -1, dtype=np.int16)
    slice_count = np.zeros(n_records, dtype=np.int16)
    fold_rows: list[dict] = []
    for fold in folds:
        fold_num = int(fold["fold"])
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        print(f"Full ECG-RAMBA stress={stress_name} fold {fold_num}/5 | val={len(va_idx)}", flush=True)
        ckpt_record = checkpoint_contract.get(fold_num)
        if ckpt_record:
            ckpt_path = resolve_path(Path(ckpt_record["path"]))
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint recorded by frozen OOF manifest for fold {fold_num}: {ckpt_path}. "
                    "Restore/copy the model_runs checkpoints that produced oof_final_ema_predictions.npz; "
                    "do not substitute legacy model/fold*_final.pt checkpoints."
                )
            expected_sha = ckpt_record.get("sha256")
            if expected_sha:
                actual_sha = sha256_file(ckpt_path)
                if actual_sha != expected_sha:
                    raise RuntimeError(
                        f"Fold {fold_num} checkpoint SHA mismatch: {actual_sha} != {expected_sha} ({ckpt_path})"
                    )
        else:
            ckpt_path = gen.checkpoint_path(fold_num, args.checkpoint_kind, allow_fallback=False)
        checkpoint_payload, checkpoint_meta = gen.load_checkpoint_payload(ckpt_path, args.checkpoint_kind)
        checkpoint_fp = checkpoint_meta.get("dataset_record_order_fingerprint")
        if args.limit_records == 0 and checkpoint_fp and checkpoint_fp != dataset_record_fingerprint:
            raise RuntimeError(
                f"Fold {fold_num} checkpoint record fingerprint {checkpoint_fp} "
                f"!= loaded dataset fingerprint {dataset_record_fingerprint}"
            )
        source_config_hash = checkpoint_meta["source_config_hash"]
        pca_record = pca_contract.get(fold_num)
        if not pca_record:
            raise RuntimeError(f"Missing frozen OOF PCA contract for fold {fold_num}")
        pca_path = resolve_path(Path(pca_record["path"]))
        if not pca_path.exists():
            raise FileNotFoundError(
                f"Missing training-fold PCA object for fold {fold_num}: {pca_path}. "
                "Run the final_ema OOF export/training pipeline first; robustness must not fit a new PCA."
            )
        actual_pca_sha = sha256_file(pca_path)
        if actual_pca_sha != pca_record.get("sha256"):
            raise RuntimeError(
                f"Fold {fold_num} PCA SHA mismatch: {actual_pca_sha} != "
                f"{pca_record.get('sha256')} ({pca_path})"
            )
        pca = joblib.load(pca_path)
        hydra_va = pca.transform(X_rocket[va_idx]).astype(np.float32)
        hydra_va_by_record = {int(record_idx): hydra_va[offset] for offset, record_idx in enumerate(va_idx)}
        xs, xh, xhr, rids, counts = gen.build_fold_slices(
            va_idx=va_idx,
            X=X,
            X_hrv=X_hrv,
            hydra_va_by_record=hydra_va_by_record,
        )
        for rid, count in counts.items():
            slice_count[int(rid)] = int(count)
        fold_summary = {
            "fold": fold_num,
            "validation_records": int(len(va_idx)),
            "n_slices": int(len(rids)),
            "pca_object_path": str(pca_path),
            "pca_object_sha256": sha256_file(pca_path),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path),
        }
        if len(rids) == 0:
            fold_rows.append(fold_summary)
            del checkpoint_payload
            continue
        model = gen.load_model_for_fold(
            fold_num,
            args.checkpoint_kind,
            checkpoint_file=ckpt_path,
            checkpoint_payload=checkpoint_payload,
        )
        del checkpoint_payload
        dataset = gen.ECGSliceDatasetInfer(xs, xh, xhr, rids)
        slice_prob, slice_rids, actual_batch, actual_workers, attempts = gen.infer_with_retries(
            model,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for rid in va_idx:
            mask = slice_rids == int(rid)
            if not np.any(mask):
                continue
            preds = slice_prob[mask]
            preds = preds[np.isfinite(preds).all(axis=1)]
            if len(preds):
                oof_prob[int(rid)] = power_mean(preds, q=float(CONFIG["power_mean_q"]), axis=0)
                fold_id[int(rid)] = fold_num
        fold_summary.update(
            {
                "n_predicted_records": int(np.sum(fold_id[va_idx] >= 0)),
                "actual_batch_size": int(actual_batch),
                "actual_num_workers": int(actual_workers),
                "inference_retry_attempts": attempts,
                "slice_prob_min": float(np.min(slice_prob)),
                "slice_prob_max": float(np.max(slice_prob)),
            }
        )
        fold_rows.append(fold_summary)
        del model, dataset, xs, xh, xhr, rids, slice_prob, slice_rids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    missing = int(np.sum(fold_id < 0))
    if missing:
        raise RuntimeError(f"Full model stress inference missing {missing} records.")
    return np.clip(oof_prob, 0.0, 1.0).astype(np.float32), slice_count, fold_rows


def write_prediction(path: Path, *, y: np.ndarray, prob: np.ndarray, fold_id: np.ndarray, stress: dict, model_label: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_compressed_atomic(
        path,
        y_true=y.astype(np.float32),
        y_prob=prob.astype(np.float32),
        record_id=np.arange(len(y), dtype=np.int64),
        fold_id=fold_id.astype(np.int16),
        class_names=np.asarray(CLASSES),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(PROTOCOL),
        model_label=np.asarray(model_label),
        stress_name=np.asarray(stress["name"]),
        stress_kind=np.asarray(stress["kind"]),
        stress_json=np.asarray(json.dumps(stress, sort_keys=True)),
        threshold=np.asarray(float(metadata["threshold"])),
        n_bins=np.asarray(int(metadata["n_bins"])),
        git_commit=np.asarray(git_commit()),
        created_utc=np.asarray(now_utc()),
        manuscript_ready=np.asarray(True),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Wrote predictions: {path}", flush=True)


def load_existing_prediction(
    path: Path,
    *,
    y: np.ndarray,
    fold_id: np.ndarray,
    expected_stress: str,
    expected_stress_spec: dict,
    expected_model_label: str,
    expected_contract_hash: str | None,
    expected_checkpoint_sha_by_fold: dict[int, str] | None = None,
    expected_minirocket_params_hash: str | None = None,
    expected_minirocket_clean_prediction_sha256: str | None = None,
) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"y_true", "y_prob", "fold_id"}
            if not required.issubset(data.files):
                return None
            cached_y = np.asarray(data["y_true"], dtype=np.float32)
            cached_prob = np.asarray(data["y_prob"], dtype=np.float32)
            cached_fold = np.asarray(data["fold_id"], dtype=np.int16)
            cached_protocol = str(np.asarray(data["protocol"]).item()) if "protocol" in data.files else ""
            cached_stress = str(np.asarray(data["stress_name"]).item()) if "stress_name" in data.files else ""
            cached_stress_spec = (
                json.loads(str(np.asarray(data["stress_json"]).item()))
                if "stress_json" in data.files
                else None
            )
            cached_model = str(np.asarray(data["model_label"]).item()) if "model_label" in data.files else ""
            metadata = (
                json.loads(str(np.asarray(data["metadata_json"]).item()))
                if "metadata_json" in data.files
                else {}
            )
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        print(f"Rejecting unreadable stress prediction {path}: {exc}", flush=True)
        return None

    cached_contract_hash = metadata.get("prediction_contract_hash")
    contract_matches = bool(
        expected_contract_hash
        and cached_contract_hash
        and cached_contract_hash == expected_contract_hash
    )
    # Reuse is exact-contract only. Checkpoint/head equality alone cannot attest
    # that the same perturbation and feature-extraction source produced a cache.
    if (
        cached_y.shape == y.shape
        and cached_prob.shape == y.shape
        and np.array_equal(cached_y, y)
        and np.array_equal(cached_fold, fold_id)
        and cached_protocol == PROTOCOL
        and cached_stress == expected_stress
        and isinstance(cached_stress_spec, dict)
        and json.dumps(cached_stress_spec, sort_keys=True)
        == json.dumps(expected_stress_spec, sort_keys=True)
        and cached_model == expected_model_label
        and contract_matches
    ):
        if np.isfinite(cached_prob).all():
            print(f"Reusing existing stress prediction: {path}", flush=True)
            return np.clip(cached_prob, 0.0, 1.0).astype(np.float32)
    return None


def prediction_contract_hash(
    *,
    model_label: str,
    checkpoint_contract_payload: dict | None = None,
    minirocket_heads_manifest: dict | None = None,
) -> str | None:
    source_bundle = source_bundle_contract()
    if model_label == "Full ECG-RAMBA":
        payload = checkpoint_contract_payload or {}
        checkpoints = payload.get("checkpoints") or {}
        checkpoint_sha = {
            str(fold): row.get("sha256")
            for fold, row in sorted(checkpoints.items())
        }
        if not checkpoint_sha or any(not value for value in checkpoint_sha.values()):
            return None
        contract = {
            "protocol": PROTOCOL,
            "model_label": model_label,
            "oof_run_manifest_sha256": payload.get("sha256"),
            "checkpoint_sha256_by_fold": checkpoint_sha,
            "source_bundle_sha256": source_bundle["sha256"],
        }
    else:
        manifest = minirocket_heads_manifest or {}
        if not manifest.get("params_hash"):
            return None
        contract = {
            "protocol": PROTOCOL,
            "model_label": model_label,
            "minirocket_head_protocol": manifest.get("protocol"),
            "minirocket_head_params_hash": manifest.get("params_hash"),
            "minirocket_clean_prediction_sha256": manifest.get(
                "clean_prediction_sha256"
            ),
            "source_bundle_sha256": source_bundle["sha256"],
        }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_existing_minirocket_clean_reference(*, y: np.ndarray, fold_id: np.ndarray) -> MiniRocketHeads | None:
    """Load the MiniRocket clean reference without materializing MiniRocket features.

    This is used for final robustness aggregation after all per-stress prediction
    artifacts have already been generated and mirrored. It avoids reloading the
    raw Chapman array and the 20k-feature MiniRocket cache just to recompute
    clean-reference metrics.
    """

    clean_pred_path = PREDICTION_DIR / "robustness_minirocket_clean_ref_predictions.npz"
    manifest_path = MANIFEST_DIR / "robustness_minirocket_heads_manifest.json"
    head_dir = EXPERIMENTAL_DIR / "robustness_minirocket_heads"
    if not clean_pred_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != MINIROCKET_HEAD_PROTOCOL:
        raise ValueError(
            f"MiniRocket robustness clean reference protocol mismatch: "
            f"{manifest.get('protocol')} != {MINIROCKET_HEAD_PROTOCOL}"
        )
    expected_sha = manifest.get("clean_prediction_sha256")
    if expected_sha and sha256_file(clean_pred_path) != expected_sha:
        raise RuntimeError("MiniRocket robustness clean reference SHA does not match its manifest.")
    with np.load(clean_pred_path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "fold_id", "class_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{clean_pred_path} missing required keys: {sorted(missing)}")
        cached_y = np.asarray(data["y_true"], dtype=np.float32)
        cached_prob = np.asarray(data["y_prob"], dtype=np.float32)
        cached_fold = np.asarray(data["fold_id"], dtype=np.int16)
        class_names = np.asarray(data["class_names"]).astype(str).tolist()
    if class_names != CLASSES:
        raise ValueError("MiniRocket robustness clean reference class_names do not match config CLASSES.")
    if cached_y.shape != y.shape or cached_prob.shape != y.shape:
        raise ValueError(
            f"MiniRocket robustness clean reference shape mismatch: "
            f"y_true={cached_y.shape}, y_prob={cached_prob.shape}, expected={y.shape}"
        )
    if not np.array_equal(cached_y, y):
        raise ValueError("MiniRocket robustness clean reference y_true does not match frozen OOF labels.")
    if not np.array_equal(cached_fold, fold_id):
        raise ValueError("MiniRocket robustness clean reference fold_id does not match frozen OOF folds.")
    if not np.isfinite(cached_prob).all():
        raise ValueError("MiniRocket robustness clean reference contains non-finite probabilities.")
    print(f"Reusing MiniRocket robustness clean reference: {clean_pred_path}", flush=True)
    return MiniRocketHeads(
        head_dir=head_dir,
        manifest=manifest,
        clean_prob=np.clip(cached_prob, 0.0, 1.0).astype(np.float32),
        fold_id=cached_fold,
        fold_rows=manifest.get("fold_rows", []),
    )


def paired_bootstrap_robustness(
    *,
    y: np.ndarray,
    clean_full: np.ndarray,
    stress_full: np.ndarray,
    clean_mini: np.ndarray,
    stress_mini: np.ndarray,
    spec: MetricSpec,
    n_boot: int,
    seed: int,
    stress_name: str,
    progress_every: int = 0,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = []
    for boot_idx in range(n_boot):
        if progress_every > 0 and boot_idx > 0 and boot_idx % progress_every == 0:
            print(
                f"  paired bootstrap {stress_name} {spec.name}: {boot_idx}/{n_boot}",
                flush=True,
            )
        idx = rng.integers(0, n, size=n)
        try:
            cf = metric_value(spec, y[idx], clean_full[idx])
            sf = metric_value(spec, y[idx], stress_full[idx])
            cm = metric_value(spec, y[idx], clean_mini[idx])
            sm = metric_value(spec, y[idx], stress_mini[idx])
        except ValueError:
            continue
        vals = [cf, sf, cm, sm]
        if not all(np.isfinite(v) for v in vals):
            continue
        full_deg = degradation(cf, sf, spec)
        mini_deg = degradation(cm, sm, spec)
        stressed_raw_diff = sf - sm
        stressed_adv = stressed_raw_diff if spec.higher_is_better else -stressed_raw_diff
        deg_adv = mini_deg - full_deg
        samples.append(
            {
                "stress_test": stress_name,
                "metric": spec.name,
                "bootstrap_index": int(boot_idx),
                "clean_full": cf,
                "stress_full": sf,
                "clean_minirocket": cm,
                "stress_minirocket": sm,
                "degradation_full": full_deg,
                "degradation_minirocket": mini_deg,
                "robustness_advantage_full_less_degradation": deg_adv,
                "stressed_advantage_full_over_minirocket": stressed_adv,
            }
        )
    if not samples:
        return {
            "degradation_advantage_mean": math.nan,
            "degradation_advantage_ci_low": math.nan,
            "degradation_advantage_ci_high": math.nan,
            "stressed_advantage_mean": math.nan,
            "stressed_advantage_ci_low": math.nan,
            "stressed_advantage_ci_high": math.nan,
            "p_value_degradation_two_sided": None,
            "p_value_stressed_two_sided": None,
            "n_boot_valid": 0,
        }, samples
    deg_adv = np.asarray([row["robustness_advantage_full_less_degradation"] for row in samples], dtype=np.float64)
    stressed_adv = np.asarray([row["stressed_advantage_full_over_minirocket"] for row in samples], dtype=np.float64)

    deg_lo, deg_hi = np.quantile(deg_adv, [0.025, 0.975])
    stress_lo, stress_hi = np.quantile(stressed_adv, [0.025, 0.975])
    return {
        "degradation_advantage_mean": float(np.mean(deg_adv)),
        "degradation_advantage_ci_low": float(deg_lo),
        "degradation_advantage_ci_high": float(deg_hi),
        "stressed_advantage_mean": float(np.mean(stressed_adv)),
        "stressed_advantage_ci_low": float(stress_lo),
        "stressed_advantage_ci_high": float(stress_hi),
        # Percentile bootstrap draws estimate the sampling distribution of an
        # effect; tail mass around zero is not a null-centred randomisation test.
        "p_value_degradation_two_sided": None,
        "p_value_stressed_two_sided": None,
        "n_boot_valid": int(len(samples)),
    }, samples


def interpretation(
    ci_low: float,
    ci_high: float,
    *,
    positive_label: str,
    negative_label: str,
    inconclusive_label: str,
) -> str:
    if ci_low > 0.0:
        return positive_label
    if ci_high < 0.0:
        return negative_label
    return inconclusive_label


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    print("=" * 80, flush=True)
    print("ROBUSTNESS STRESS TEST: FULL ECG-RAMBA VS FIXED-SEED ROCKET-FAMILY HEAD", flush=True)
    print("=" * 80, flush=True)
    print(f"stress_tests={args.stress_tests}", flush=True)
    print(f"n_boot={args.n_boot} threshold={args.threshold} n_bins={args.n_bins}", flush=True)
    execution_mode = "features_only" if args.features_only else "inference_only" if args.inference_only else "combined"
    print(f"execution_mode={execution_mode}", flush=True)
    print(f"feature_cache_dir={resolve_path(args.feature_cache_dir)}", flush=True)
    print(f"python={sys.version}", flush=True)
    print(f"platform={platform.platform()}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    if args.inference_only and not torch.cuda.is_available():
        raise RuntimeError("--inference-only requires a CUDA runtime for Full ECG-RAMBA checkpoint inference.")

    full_clean = load_prediction_npz(args.full_clean_predictions, "Full clean")
    mini_clean_canonical = (
        None
        if args.features_only
        else load_prediction_npz(args.minirocket_clean_predictions, "MiniRocket canonical clean")
    )
    contract = validate_clean_prediction_contract(
        full_clean,
        mini_clean_canonical,
        args,
        require_minirocket=not args.features_only,
    )
    raw_cache_contract = (
        frozen_oof_raw_cache_contract(args, contract) if args.disk_backed_signals else None
    )
    y = full_clean["y_true"]
    fold_id = np.asarray(full_clean["fold_id"], dtype=np.int16)
    folds = fold_list_from_fold_id(fold_id)
    if args.limit_records > 0:
        keep = np.arange(min(args.limit_records, len(y)), dtype=np.int64)
        y = y[keep]
        fold_id = fold_id[keep]
        full_clean["y_prob"] = full_clean["y_prob"][keep]
        if mini_clean_canonical is not None:
            mini_clean_canonical["y_prob"] = mini_clean_canonical["y_prob"][keep]
        folds = fold_list_from_fold_id(fold_id)
        print(f"Debug limit active: {len(y)} records", flush=True)

    specs = stress_specs(args.stress_tests.split(","), args.seed)
    expected_fp = contract["freeze_manifest"].get("dataset_record_order_fingerprint")
    if args.features_only:
        if not args.save_perturbed_caches:
            raise ValueError("--features-only requires --save-perturbed-caches.")
        raw_cache_info = None
        if args.disk_backed_signals:
            raw_cache_info = inspect_disk_backed_chapman_cache(
                expected_y=y,
                expected_record_fingerprint=str(expected_fp or ""),
                expected_archive_sha256=str(
                    contract["freeze_manifest"].get("source_archive_sha256") or ""
                ),
                expected_cache_contract=raw_cache_contract,
                explicit_cache=args.raw_cache,
                limit_records=args.limit_records,
            )
            y_loaded = raw_cache_info["y"]
            X_raw_amp = raw_cache_info["X_raw_amp"]
            subjects = raw_cache_info["subjects"]
            X = None
        else:
            gen = load_revision_module(
                "01_generate_predictions.py",
                "_ecg_ramba_generate_predictions_data_for_robustness_features",
            )
            X, y_loaded, X_raw_amp, subjects = gen.prepare_clean_chapman(
                limit_records=args.limit_records
            )
        if y_loaded.shape != y.shape or not np.array_equal(y_loaded, y):
            raise ValueError("Loaded Chapman labels do not match frozen OOF y_true.")
        record_fp = record_order_fingerprint(subjects)
        if args.limit_records == 0 and expected_fp and record_fp != expected_fp:
            raise RuntimeError(f"Loaded Chapman record fingerprint {record_fp} != frozen {expected_fp}")
        if args.limit_records > 0:
            print(
                "Debug limit active: skipping full-record fingerprint equality check "
                f"(subset fingerprint={record_fp}, frozen full fingerprint={expected_fp}).",
                flush=True,
            )

        feature_rows = []
        source_bundle = source_bundle_contract()
        for stress in specs:
            stress_name = stress["name"]
            stress_hash = stress_feature_hash(stress, contract, record_fp)
            cache_hash = feature_cache_hash(stress_hash, raw_cache_info)
            print("\n" + "=" * 80, flush=True)
            print(
                f"Feature cache stage: {stress_name} | stress_hash={stress_hash} "
                f"cache_hash={cache_hash}",
                flush=True,
            )
            print("=" * 80, flush=True)
            rocket_path = robust_feature_cache_path(
                args.feature_cache_dir,
                "robust_minirocket_raw",
                stress_name,
                cache_hash,
                len(y),
                record_fp,
            )
            hrv_path = robust_feature_cache_path(
                args.feature_cache_dir,
                "robust_hrv36",
                stress_name,
                cache_hash,
                len(y),
                record_fp,
            )
            rocket_info = inspect_feature_cache(
                rocket_path,
                feature_kind="fixed_seed_rocket_family_max_ppv",
                expected_shape=(len(y), 20000),
                expected_dtype=np.dtype(np.float16),
                stress_name=stress_name,
                stress_hash=cache_hash,
                record_fp=record_fp,
            )
            hrv_info = inspect_feature_cache(
                hrv_path,
                feature_kind="hrv36_perturbed_signal_amp_contract",
                expected_shape=(len(y), int(CONFIG["hrv_dim"])),
                expected_dtype=np.dtype(np.float16),
                stress_name=stress_name,
                stress_hash=cache_hash,
                record_fp=record_fp,
            )
            if rocket_info is not None and hrv_info is not None:
                print(f"Reusing complete source-bound feature pair: {stress_name}", flush=True)
                feature_rows.append(
                    {
                        "stress_test": stress_name,
                        "stress_hash": stress_hash,
                        "feature_cache_hash": cache_hash,
                        "stress_spec": stress,
                        "perturbation": {"spec": stress, "reused_feature_pair": True},
                        "rocket_family": rocket_info,
                        "hrv36": hrv_info,
                    }
                )
                continue
            if X is None:
                assert raw_cache_info is not None
                X = open_disk_backed_chapman_signals(raw_cache_info, args.raw_cache_mmap_dir)
            perturbation_path = None
            if args.disk_backed_signals:
                assert raw_cache_info is not None
                X_stress, perturb_meta, perturbation_path = perturb_signals_disk_backed(
                    X,
                    stress,
                    out_dir=args.perturbation_mmap_dir,
                    raw_cache_sha256=raw_cache_info["sha256"],
                    source_bundle_sha256=source_bundle["sha256"],
                )
            else:
                X_stress, perturb_meta = perturb_signals(X, stress)
            X_rocket, rocket_info = generate_minirocket_features(
                X_stress,
                stress_name=stress_name,
                stress_hash=cache_hash,
                record_fp=record_fp,
                batch_size=args.minirocket_feature_batch_size,
                device_name=args.minirocket_feature_device,
                save_cache=True,
                cache_dir=args.feature_cache_dir,
                materialize=False,
                work_dir=args.feature_work_dir,
            )
            X_hrv, hrv_info = generate_hrv36_features(
                X_stress,
                X_raw_amp,
                stress_name=stress_name,
                stress_hash=cache_hash,
                record_fp=record_fp,
                save_cache=True,
                cache_dir=args.feature_cache_dir,
                materialize=False,
            )
            feature_rows.append(
                {
                    "stress_test": stress_name,
                    "stress_hash": stress_hash,
                    "feature_cache_hash": cache_hash,
                    "stress_spec": stress,
                    "perturbation": perturb_meta,
                    "rocket_family": rocket_info,
                    "hrv36": hrv_info,
                }
            )
            if perturbation_path is not None:
                remove_local_perturbation(X_stress, perturbation_path)
            del X_stress, X_rocket, X_hrv
            gc.collect()

        raw_manifest = None
        if raw_cache_info is not None:
            raw_manifest = {
                key: json_safe(value)
                for key, value in raw_cache_info.items()
                if key not in {"row_indices", "y", "X_raw_amp", "subjects"}
            }
        feature_manifest = {
            "schema_version": 2,
            "status": "complete",
            "mode": "features_only",
            "created_utc": now_utc(),
            "protocol": PROTOCOL,
            "source_bundle": source_bundle_contract(),
            "freeze_manifest": contract["freeze_manifest"],
            "record_order_fingerprint": record_fp,
            "n_records": int(len(y)),
            "cache_dir": str(resolve_path(args.feature_cache_dir)),
            "disk_backed_signals": bool(args.disk_backed_signals),
            "raw_cache": raw_manifest,
            "stress_rows": feature_rows,
        }
        feature_manifest_path = MANIFEST_DIR / "robustness_feature_cache_manifest.json"
        save_json(feature_manifest_path, feature_manifest)
        print(f"Wrote feature-cache manifest: {feature_manifest_path}", flush=True)
        print(f"Feature-only stage complete: {len(feature_rows)}/{len(specs)} stresses", flush=True)
        return

    assert mini_clean_canonical is not None
    checkpoint_contract_payload = load_oof_checkpoint_contract(args)
    checkpoint_contract = checkpoint_contract_payload.get("checkpoints", {})
    pca_contract = checkpoint_contract_payload.get("pca_objects", {})
    reusable_heads = (
        load_existing_minirocket_clean_reference(y=y, fold_id=fold_id)
        if args.reuse_existing
        else None
    )
    expected_checkpoint_sha_by_fold = {
        int(fold): str(row.get("sha256"))
        for fold, row in checkpoint_contract.items()
        if row.get("sha256")
    }
    full_prediction_contract_hash = prediction_contract_hash(
        model_label="Full ECG-RAMBA",
        checkpoint_contract_payload=checkpoint_contract_payload,
    )
    mini_prediction_contract_hash = prediction_contract_hash(
        model_label="MiniRocket-only",
        minirocket_heads_manifest=(reusable_heads.manifest if reusable_heads else None),
    )
    existing_stress_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing_existing: list[str] = []
    if args.reuse_existing:
        for stress in specs:
            stress_name = stress["name"]
            full_pred_path = PREDICTION_DIR / f"robustness_full_{stress_name}_predictions.npz"
            mini_pred_path = PREDICTION_DIR / f"robustness_minirocket_{stress_name}_predictions.npz"
            full_prob = load_existing_prediction(
                full_pred_path,
                y=y,
                fold_id=fold_id,
                expected_stress=stress_name,
                expected_stress_spec=stress,
                expected_model_label="Full ECG-RAMBA",
                expected_contract_hash=full_prediction_contract_hash,
                expected_checkpoint_sha_by_fold=expected_checkpoint_sha_by_fold,
            )
            mini_prob = load_existing_prediction(
                mini_pred_path,
                y=y,
                fold_id=fold_id,
                expected_stress=stress_name,
                expected_stress_spec=stress,
                expected_model_label="MiniRocket-only",
                expected_contract_hash=mini_prediction_contract_hash,
                expected_minirocket_params_hash=(
                    reusable_heads.manifest.get("params_hash")
                    if reusable_heads
                    else None
                ),
                expected_minirocket_clean_prediction_sha256=(
                    reusable_heads.manifest.get("clean_prediction_sha256")
                    if reusable_heads
                    else None
                ),
            )
            if full_prob is None:
                missing_existing.append(project_relative(full_pred_path))
            if mini_prob is None:
                missing_existing.append(project_relative(mini_pred_path))
            if full_prob is not None and mini_prob is not None:
                existing_stress_probs[stress_name] = (full_prob, mini_prob)

    aggregation_only = bool(args.reuse_existing and not missing_existing)
    raw_cache_info = None
    heads = reusable_heads if aggregation_only else None
    if args.require_existing_stress_predictions and (missing_existing or heads is None):
        missing = list(missing_existing)
        if heads is None:
            missing.extend(
                [
                    project_relative(PREDICTION_DIR / "robustness_minirocket_clean_ref_predictions.npz"),
                    project_relative(MANIFEST_DIR / "robustness_minirocket_heads_manifest.json"),
                ]
            )
        raise FileNotFoundError(
            "Low-memory aggregation was requested, but required existing artifacts are missing or invalid: "
            + "; ".join(missing)
        )

    if aggregation_only and heads is not None:
        print(
            "All requested stress predictions and MiniRocket clean reference are reusable; "
            "running low-memory aggregation without loading raw Chapman signals.",
            flush=True,
        )
        record_fp = str(expected_fp or "")
        clean_rocket_info = {
            "aggregation_only_reused_clean_reference": True,
            "clean_prediction_file": str(PREDICTION_DIR / "robustness_minirocket_clean_ref_predictions.npz"),
            "clean_prediction_sha256": sha256_file(PREDICTION_DIR / "robustness_minirocket_clean_ref_predictions.npz"),
        }
        mini_clean_ref = heads.clean_prob
        X = None
        X_raw_amp = None
    else:
        mini = load_revision_module("10_minirocket_only_baseline.py", "_ecg_ramba_minirocket_data_for_robustness")
        if args.disk_backed_signals:
            raw_cache_info = inspect_disk_backed_chapman_cache(
                expected_y=y,
                expected_record_fingerprint=str(expected_fp or ""),
                expected_archive_sha256=str(
                    contract["freeze_manifest"].get("source_archive_sha256") or ""
                ),
                expected_cache_contract=raw_cache_contract,
                explicit_cache=args.raw_cache,
                limit_records=args.limit_records,
            )
            X = open_disk_backed_chapman_signals(raw_cache_info, args.raw_cache_mmap_dir)
            y_loaded = raw_cache_info["y"]
            X_raw_amp = raw_cache_info["X_raw_amp"]
            subjects = raw_cache_info["subjects"]
        else:
            gen = load_revision_module(
                "01_generate_predictions.py",
                "_ecg_ramba_generate_predictions_data_for_robustness",
            )
            X, y_loaded, X_raw_amp, subjects = gen.prepare_clean_chapman(
                limit_records=args.limit_records
            )
        if y_loaded.shape != y.shape or not np.array_equal(y_loaded, y):
            raise ValueError("Loaded Chapman labels do not match frozen OOF y_true.")
        record_fp = record_order_fingerprint(subjects)
        if args.limit_records == 0 and expected_fp and record_fp != expected_fp:
            raise RuntimeError(f"Loaded Chapman record fingerprint {record_fp} != frozen {expected_fp}")
        if args.limit_records > 0:
            print(
                "Debug limit active: skipping full-record fingerprint equality check "
                f"(subset fingerprint={record_fp}, frozen full fingerprint={expected_fp}).",
                flush=True,
            )

        # Debug subsets still consume the manuscript full-record cache and slice it
        # inside the MiniRocket loader. This avoids looking for an artificial
        # N=<limit> cache keyed by the subset fingerprint.
        clean_cache_n_records = int(contract["freeze_manifest"].get("validated_records") or len(full_clean["record_id"]))
        clean_cache_fingerprint = expected_fp if args.limit_records > 0 and expected_fp else record_fp
        clean_X_rocket, clean_rocket_info = mini.load_minirocket_cache(
            n_records=clean_cache_n_records if args.limit_records > 0 else len(X),
            record_fingerprint=clean_cache_fingerprint,
            explicit_cache=None,
            allow_legacy_shape_cache=args.allow_legacy_shape_cache,
            limit_records=args.limit_records,
        )
        heads = fit_or_load_minirocket_heads(
            X_clean=clean_X_rocket,
            y=y,
            folds=folds,
            clean_cache_info=clean_rocket_info,
            args=args,
        )
        mini_clean_ref = heads.clean_prob

    mini_prediction_contract_hash = prediction_contract_hash(
        model_label="MiniRocket-only",
        minirocket_heads_manifest=heads.manifest,
    )
    rows: list[dict] = []
    sample_rows: list[dict] = []
    artifact_rows: list[dict] = []
    stress_meta_rows: list[dict] = []
    clean_metrics = {
        "full": multilabel_metrics(y, full_clean["y_prob"], threshold=args.threshold),
        "minirocket_canonical": multilabel_metrics(y, mini_clean_canonical["y_prob"], threshold=args.threshold),
        "minirocket_robustness_head_ref": multilabel_metrics(y, mini_clean_ref, threshold=args.threshold),
    }
    clean_calibration = {
        "full": calibration_summary(y, full_clean["y_prob"], n_bins=args.n_bins),
        "minirocket_canonical": calibration_summary(y, mini_clean_canonical["y_prob"], n_bins=args.n_bins),
        "minirocket_robustness_head_ref": calibration_summary(y, mini_clean_ref, n_bins=args.n_bins),
    }
    mini_clean_ref_delta = {
        "max_abs_probability_delta_vs_canonical": float(
            np.max(np.abs(mini_clean_ref - mini_clean_canonical["y_prob"]))
        ),
        "mean_abs_probability_delta_vs_canonical": float(
            np.mean(np.abs(mini_clean_ref - mini_clean_canonical["y_prob"]))
        ),
    }
    print(
        "MiniRocket clean reference delta vs canonical: "
        f"max_abs={mini_clean_ref_delta['max_abs_probability_delta_vs_canonical']:.6g} "
        f"mean_abs={mini_clean_ref_delta['mean_abs_probability_delta_vs_canonical']:.6g}",
        flush=True,
    )

    for stress in specs:
        stress_hash = stress_feature_hash(stress, contract, record_fp)
        cache_hash = feature_cache_hash(stress_hash, raw_cache_info)
        stress_name = stress["name"]
        print("\n" + "=" * 80, flush=True)
        print(
            f"Stress test: {stress_name} | stress_hash={stress_hash} cache_hash={cache_hash}",
            flush=True,
        )
        print("=" * 80, flush=True)
        full_pred_path = PREDICTION_DIR / f"robustness_full_{stress_name}_predictions.npz"
        mini_pred_path = PREDICTION_DIR / f"robustness_minirocket_{stress_name}_predictions.npz"
        if stress_name in existing_stress_probs:
            full_prob, mini_prob = existing_stress_probs[stress_name]
        else:
            full_prob = load_existing_prediction(
                full_pred_path,
                y=y,
                fold_id=fold_id,
                expected_stress=stress_name,
                expected_stress_spec=stress,
                expected_model_label="Full ECG-RAMBA",
                expected_contract_hash=full_prediction_contract_hash,
                expected_checkpoint_sha_by_fold=expected_checkpoint_sha_by_fold,
            ) if args.reuse_existing else None
            mini_prob = load_existing_prediction(
                mini_pred_path,
                y=y,
                fold_id=fold_id,
                expected_stress=stress_name,
                expected_stress_spec=stress,
                expected_model_label="MiniRocket-only",
                expected_contract_hash=mini_prediction_contract_hash,
                expected_minirocket_params_hash=(
                    heads.manifest.get("params_hash")
                    if heads
                    else None
                ),
                expected_minirocket_clean_prediction_sha256=(
                    heads.manifest.get("clean_prediction_sha256")
                    if heads
                    else None
                ),
            ) if args.reuse_existing else None
        feature_infos = {}
        perturb_meta = {}
        perturbation_path = None
        if full_prob is None or mini_prob is None:
            if X is None or X_raw_amp is None:
                raise RuntimeError(
                    f"Missing reusable predictions for {stress_name} during low-memory aggregation. "
                    "Run this stress individually first or disable --require-existing-stress-predictions."
                )
            if args.disk_backed_signals:
                if raw_cache_info is None:
                    raise RuntimeError("Disk-backed signal contract is unavailable for stress inference.")
                X_stress, perturb_meta, perturbation_path = perturb_signals_disk_backed(
                    X,
                    stress,
                    out_dir=args.perturbation_mmap_dir,
                    raw_cache_sha256=raw_cache_info["sha256"],
                    source_bundle_sha256=source_bundle_contract()["sha256"],
                )
            else:
                X_stress, perturb_meta = perturb_signals(X, stress)
            X_rocket, rocket_info = generate_minirocket_features(
                X_stress,
                stress_name=stress_name,
                stress_hash=cache_hash,
                record_fp=record_fp,
                batch_size=args.minirocket_feature_batch_size,
                device_name=args.minirocket_feature_device,
                save_cache=args.save_perturbed_caches,
                cache_dir=args.feature_cache_dir,
                require_existing=args.inference_only,
                work_dir=args.feature_work_dir,
            )
            feature_infos["minirocket"] = rocket_info
            if full_prob is None:
                X_hrv, hrv_info = generate_hrv36_features(
                    X_stress,
                    X_raw_amp,
                    stress_name=stress_name,
                    stress_hash=cache_hash,
                    record_fp=record_fp,
                    save_cache=args.save_perturbed_caches,
                    cache_dir=args.feature_cache_dir,
                    require_existing=args.inference_only,
                )
                feature_infos["hrv36"] = hrv_info
                full_prob, slice_count, full_fold_rows = predict_full_model(
                    X=X_stress,
                    X_rocket=X_rocket,
                    X_hrv=X_hrv,
                    y=y,
                    folds=folds,
                    args=args,
                    stress_name=stress_name,
                    dataset_record_fingerprint=record_fp,
                    checkpoint_contract=checkpoint_contract,
                    pca_contract=pca_contract,
                )
                write_prediction(
                    full_pred_path,
                    y=y,
                    prob=full_prob,
                    fold_id=fold_id,
                    stress=stress,
                    model_label="Full ECG-RAMBA",
                    metadata={
                        "threshold": args.threshold,
                        "n_bins": args.n_bins,
                        "prediction_contract_hash": full_prediction_contract_hash,
                        "source_bundle": source_bundle_contract(),
                        "feature_infos": feature_infos,
                        "fold_rows": full_fold_rows,
                        "slice_count_min": int(slice_count.min()),
                        "slice_count_max": int(slice_count.max()),
                    },
                )
            if perturbation_path is not None:
                remove_local_perturbation(X_stress, perturbation_path)
            del X_stress
            gc.collect()
            if mini_prob is None:
                mini_prob = predict_minirocket_heads(
                    X_rocket,
                    y.shape,
                    folds,
                    heads,
                    batch_size=args.minirocket_batch_size,
                    device_name=args.minirocket_device,
                )
                write_prediction(
                    mini_pred_path,
                    y=y,
                    prob=mini_prob,
                    fold_id=fold_id,
                    stress=stress,
                    model_label="MiniRocket-only",
                    metadata={
                        "threshold": args.threshold,
                        "n_bins": args.n_bins,
                        "prediction_contract_hash": mini_prediction_contract_hash,
                        "source_bundle": source_bundle_contract(),
                        "feature_infos": feature_infos,
                        "minirocket_heads_manifest": heads.manifest,
                    },
                )
            del X_rocket
            if "X_hrv" in locals():
                del X_hrv
            gc.collect()

        full_pred_sha = sha256_file(full_pred_path)
        mini_pred_sha = sha256_file(mini_pred_path)
        mini_clean_ref_path = PREDICTION_DIR / "robustness_minirocket_clean_ref_predictions.npz"
        mini_clean_ref_sha = sha256_file(mini_clean_ref_path) if mini_clean_ref_path.exists() else None

        artifact_rows.extend(
            [
                {
                    "stress_test": stress_name,
                    "artifact": "full_predictions",
                    "path": project_relative(full_pred_path),
                    "sha256": full_pred_sha,
                },
                {
                    "stress_test": stress_name,
                    "artifact": "minirocket_predictions",
                    "path": project_relative(mini_pred_path),
                    "sha256": mini_pred_sha,
                },
            ]
        )
        stress_meta_rows.append(
            {
                "stress_test": stress_name,
                "stress_hash": stress_hash,
                "feature_cache_hash": cache_hash,
                "stress_json": json.dumps(stress, sort_keys=True),
                "perturbation_metadata": json.dumps(perturb_meta, sort_keys=True),
            }
        )
        for metric_index, spec in enumerate(metric_specs(args.threshold, args.n_bins)):
            metric_seed = args.seed + 10007 * (metric_index + 1) + int(stress_hash[:6], 16)
            cache_metadata = {
                "protocol": PROTOCOL,
                "cache_version": 2,
                "stress_test": stress_name,
                "stress_hash": stress_hash,
                "stress_json": json.dumps(stress, sort_keys=True),
                "metric": spec.name,
                "metric_family": spec.family,
                "higher_is_better": bool(spec.higher_is_better),
                "threshold": float(args.threshold),
                "n_bins": int(args.n_bins),
                "n_boot": int(args.n_boot),
                "seed": int(metric_seed),
                "n_records": int(len(y)),
                "n_classes": int(y.shape[1]),
                "full_clean_predictions_sha256": full_clean["sha256"],
                "minirocket_clean_predictions_sha256": mini_clean_canonical["sha256"],
                "minirocket_clean_reference_sha256": mini_clean_ref_sha,
                "full_stress_predictions_sha256": full_pred_sha,
                "minirocket_stress_predictions_sha256": mini_pred_sha,
                "expected_checkpoint_kind": str(args.expected_checkpoint_kind),
                "source_bundle_sha256": source_bundle_contract()["sha256"],
                "freeze_manifest_sha256": contract["freeze_manifest"]["sha256"],
                "group_sidecar_sha256": contract["group_contract"]["sidecar_sha256"],
                "bootstrap_unit": contract["group_contract"].get("bootstrap_unit"),
            }
            cache_path = metric_cache_path(args.metric_cache_dir, stress_name, spec.name)
            cached_metric = load_metric_cache(cache_path, cache_metadata) if args.reuse_metric_cache else None
            if cached_metric is not None:
                cached_row, cached_samples = cached_metric
                rows.append(cached_row)
                sample_rows.extend(cached_samples)
                print(
                    f"Reusing robustness metric cache: {stress_name} {spec.name} "
                    f"({len(cached_samples)} bootstrap rows)",
                    flush=True,
                )
                continue

            clean_full_value = metric_value(spec, y, full_clean["y_prob"])
            stress_full_value = metric_value(spec, y, full_prob)
            clean_mini_value = metric_value(spec, y, mini_clean_ref)
            stress_mini_value = metric_value(spec, y, mini_prob)
            deg_full = degradation(clean_full_value, stress_full_value, spec)
            deg_mini = degradation(clean_mini_value, stress_mini_value, spec)
            stressed_raw_diff = stress_full_value - stress_mini_value
            stressed_adv = stressed_raw_diff if spec.higher_is_better else -stressed_raw_diff
            deg_adv = deg_mini - deg_full
            print(
                f"{stress_name} {spec.name}: full_stress={stress_full_value:.6f} "
                f"mini_stress={stress_mini_value:.6f} degradation_adv_full={deg_adv:.6f}",
                flush=True,
            )
            print(f"  paired bootstrap {stress_name} {spec.name} start", flush=True)
            ci, samples = paired_bootstrap_robustness(
                y=y,
                clean_full=full_clean["y_prob"],
                stress_full=full_prob,
                clean_mini=mini_clean_ref,
                stress_mini=mini_prob,
                spec=spec,
                n_boot=args.n_boot,
                seed=metric_seed,
                stress_name=stress_name,
                progress_every=max(0, int(args.bootstrap_progress_every)),
            )
            print(
                f"  paired bootstrap {stress_name} {spec.name} done: "
                f"n_boot_valid={ci['n_boot_valid']}",
                flush=True,
            )
            if int(ci["n_boot_valid"]) != int(args.n_boot):
                raise RuntimeError(
                    f"Robustness bootstrap for {stress_name}/{spec.name} produced "
                    f"{ci['n_boot_valid']} valid rows, expected exactly {args.n_boot}. "
                    "Do not publish a short manuscript-ready cache."
                )
            row = {
                "stress_test": stress_name,
                "metric": spec.name,
                "metric_family": spec.family,
                "higher_is_better": bool(spec.higher_is_better),
                "clean_full": clean_full_value,
                "stress_full": stress_full_value,
                "degradation_full": deg_full,
                "clean_minirocket": clean_mini_value,
                "stress_minirocket": stress_mini_value,
                "degradation_minirocket": deg_mini,
                "stressed_advantage_full_over_minirocket": stressed_adv,
                "stressed_advantage_ci_low": ci["stressed_advantage_ci_low"],
                "stressed_advantage_ci_high": ci["stressed_advantage_ci_high"],
                "degradation_advantage_full_less_degradation": deg_adv,
                "degradation_advantage_ci_low": ci["degradation_advantage_ci_low"],
                "degradation_advantage_ci_high": ci["degradation_advantage_ci_high"],
                "p_value_degradation_two_sided": ci["p_value_degradation_two_sided"],
                "p_value_stressed_two_sided": ci["p_value_stressed_two_sided"],
                "n_boot_valid": ci["n_boot_valid"],
                "ci_scope": CI_SCOPE,
                "inference_scope": INFERENCE_SCOPE,
                "null_test": NULL_TEST,
                "multiplicity_adjustment": MULTIPLICITY_ADJUSTMENT,
                "training_variability_scope": TRAINING_VARIABILITY_SCOPE,
                "perturbation_realization_scope": PERTURBATION_REALIZATION_SCOPE,
                "degradation_interpretation": interpretation(
                    ci["degradation_advantage_ci_low"],
                    ci["degradation_advantage_ci_high"],
                    positive_label="full_nominal_95ci_less_degraded",
                    negative_label="minirocket_nominal_95ci_less_degraded",
                    inconclusive_label="nominal_95ci_inconclusive_degradation_difference",
                ),
                "stressed_performance_interpretation": interpretation(
                    ci["stressed_advantage_ci_low"],
                    ci["stressed_advantage_ci_high"],
                    positive_label="full_nominal_95ci_better_under_stress",
                    negative_label="minirocket_nominal_95ci_better_under_stress",
                    inconclusive_label="nominal_95ci_inconclusive_stressed_difference",
                ),
            }
            rows.append(row)
            sample_rows.extend(samples)
            write_metric_cache(cache_path, cache_metadata, row, samples)
            print(f"  cached metric result: {log_path(cache_path)}", flush=True)
        del full_prob, mini_prob
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    summary_csv = METRIC_DIR / "robustness_summary.csv"
    table_csv = TABLE_DIR / "table_robustness.csv"
    comparison_json = METRIC_DIR / "robustness_full_vs_minirocket_comparison.json"
    samples_csv = METRIC_DIR / "robustness_full_vs_minirocket_bootstrap_samples.csv"
    manifest_path = MANIFEST_DIR / "robustness_stress_manifest.json"
    stress_meta_csv = TABLE_DIR / "table_robustness_stress_metadata.csv"
    artifacts_csv = TABLE_DIR / "table_robustness_artifacts.csv"

    save_csv(summary_csv, rows)
    save_csv(table_csv, rows)
    save_csv(samples_csv, sample_rows)
    save_csv(stress_meta_csv, stress_meta_rows)
    save_csv(artifacts_csv, artifact_rows)

    payload = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "protocol": PROTOCOL,
        "runner_sha256": sha256_file(Path(__file__)),
        "source_bundle": source_bundle_contract(),
        "statistical_inference": {
            "ci_scope": CI_SCOPE,
            "inference_scope": INFERENCE_SCOPE,
            "null_test": NULL_TEST,
            "multiplicity_adjustment": MULTIPLICITY_ADJUSTMENT,
        },
        "claim_guidance": {
            "robustness": (
                "Report metric- and perturbation-specific paired degradation estimates with nominal "
                "95% percentile CIs. Do not call them statistically significant because no null-centred "
                "test or multiplicity correction is performed."
            ),
            "ranking_warning": (
                "Stressed AUROC/AUPRC may still favor MiniRocket-only even when fixed-threshold/calibration "
                "degradation favors Full ECG-RAMBA."
            ),
        },
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "n_boot": int(args.n_boot),
        "shape": {"n_records": int(len(y)), "n_classes": int(y.shape[1])},
        "inputs": {
            "full_clean_predictions": {"path": str(full_clean["path"]), "sha256": full_clean["sha256"]},
            "minirocket_clean_predictions": {
                "path": str(mini_clean_canonical["path"]),
                "sha256": mini_clean_canonical["sha256"],
            },
            "contract": contract,
            "oof_checkpoint_contract": checkpoint_contract_payload,
            "clean_minirocket_cache": clean_rocket_info,
            "minirocket_heads": heads.manifest,
        },
        "clean_metrics": clean_metrics,
        "clean_calibration": clean_calibration,
        "minirocket_clean_reference_delta": mini_clean_ref_delta,
        "stress_tests": specs,
        "metrics": rows,
        "outputs": {
            "summary_csv": str(summary_csv),
            "table_csv": str(table_csv),
            "bootstrap_samples_csv": str(samples_csv),
            "stress_metadata_csv": str(stress_meta_csv),
            "artifacts_csv": str(artifacts_csv),
            "comparison_json": str(comparison_json),
            "manifest": str(manifest_path),
            "metric_cache_dir": str(resolve_path(args.metric_cache_dir)),
        },
    }
    save_json(comparison_json, json_safe(payload))
    manifest = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "protocol": PROTOCOL,
        "runner_sha256": sha256_file(Path(__file__)),
        "source_bundle": source_bundle_contract(),
        "args": vars(args),
        "artifact_sha256": {
            "summary_csv": sha256_file(summary_csv),
            "table_csv": sha256_file(table_csv),
            "bootstrap_samples_csv": sha256_file(samples_csv),
            "stress_metadata_csv": sha256_file(stress_meta_csv),
            "artifacts_csv": sha256_file(artifacts_csv),
            "comparison_json": sha256_file(comparison_json),
        },
        "artifact_rows": artifact_rows,
    }
    save_json(manifest_path, json_safe(manifest))
    print("Wrote:", comparison_json, flush=True)
    print("Wrote:", table_csv, flush=True)
    print("Wrote:", manifest_path, flush=True)
    print(json.dumps({"status": True, "stress_tests": [s["name"] for s in specs]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
