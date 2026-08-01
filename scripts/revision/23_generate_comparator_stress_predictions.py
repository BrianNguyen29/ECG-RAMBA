"""Generate stressed predictions for learned raw-ECG comparators.

This runner is intentionally inference-only. It requires comparator checkpoints
created by the fair-baseline runners with ``--save-checkpoints`` and refuses to
retrain or reconstruct weights from clean OOF predictions. The output NPZ files
match the contract consumed by ``21_robustness_multicomparator.py``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    PREDICTION_DIR,
    ensure_revision_dirs,
    save_json,
    save_npz_compressed_atomic,
    sha256_file,
)
from scripts.revision.artifact_mirror import PublishLock  # noqa: E402
from src.aggregation import POWER_MEAN_IMPLEMENTATION, aggregate_record_probabilities  # noqa: E402
from src.training_data import build_slice_index  # noqa: E402


PROTOCOL = "comparator_stress_predictions_v2_source_bound_same_folds_power_mean_v2_q3"
SOURCE_BUNDLE_SCHEMA_VERSION = 1
DISK_BACKED_RAW_CACHE_CAPABILITY = "source_bound_npz_member_memmap_v1"
DISK_BACKED_PERTURBATION_CAPABILITY = "chunk_exact_local_memmap_v1"
DEFAULT_OOF = PREDICTION_DIR / "oof_final_ema_predictions.npz"
DEFAULT_FREEZE = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
DEFAULT_RESNET_CKPT_DIR = PROJECT_ROOT / "reports" / "revision" / "experimental" / "resnet1d_cnn_checkpoints"
DEFAULT_RAW_MAMBA_CKPT_DIR = PROJECT_ROOT / "reports" / "revision" / "experimental" / "raw_mamba_checkpoints"
DEFAULT_TRANSFORMER_CKPT_DIR = PROJECT_ROOT / "reports" / "revision" / "experimental" / "transformer_ecg_checkpoints"
BASELINE_CHECKPOINT_CONTRACTS = {
    "resnet": (
        "resnet1d_cnn_baseline_manifest.json",
        "resnet1d_cnn_raw_same_folds_power_mean_v2_q3_threshold_0.5",
    ),
    "raw_mamba": (
        "raw_mamba_baseline_manifest.json",
        "raw_mamba_retrained_weighted_bce_same_folds_power_mean_v2_q3_threshold_0.5",
    ),
    "transformer": (
        "transformer_ecg_baseline_manifest.json",
        "transformer_ecg_raw_same_folds_power_mean_v2_q3_threshold_0.5",
    ),
}

SOURCE_BUNDLE_PATHS = (
    "scripts/revision/23_generate_comparator_stress_predictions.py",
    "scripts/revision/12_robustness_stress.py",
    "scripts/revision/14_resnet1d_cnn_baseline.py",
    "scripts/revision/16_raw_mamba_baseline.py",
    "scripts/revision/24_transformer_ecg_baseline.py",
    "scripts/revision/artifact_mirror.py",
    "src/aggregation.py",
    "src/training_data.py",
    "configs/config.py",
)


def source_bundle_contract() -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in SOURCE_BUNDLE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing comparator-stress source dependency: {path}")
        files[relative] = sha256_file(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "files": files,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def load_revision_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


resnet_helpers = load_revision_module("14_resnet1d_cnn_baseline.py", "_stress_resnet_helpers")
raw_mamba_helpers = load_revision_module("16_raw_mamba_baseline.py", "_stress_raw_mamba_helpers")
transformer_helpers = load_revision_module("24_transformer_ecg_baseline.py", "_stress_transformer_helpers")
robust_helpers = load_revision_module("12_robustness_stress.py", "_stress_perturb_helpers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparators", default="resnet,raw_mamba,transformer")
    parser.add_argument("--stress-tests", default="snr20db,snr10db,snr5db,random_3_lead_dropout,precordial_dropout,resample_250hz")
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--raw-cache", type=Path, default=None)
    parser.add_argument(
        "--disk-backed-raw-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract the authenticated X.npy member to local scratch and mmap it instead of loading all ECGs into RAM.",
    )
    parser.add_argument(
        "--raw-cache-mmap-dir",
        type=Path,
        default=Path(os.environ.get("ECG_RAMBA_RAW_CACHE_MMAP_DIR", "/content/ecg_ramba_runtime/raw_cache")),
    )
    parser.add_argument(
        "--perturbation-mmap-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "ECG_RAMBA_PERTURBATION_MMAP_DIR",
                "/content/ecg_ramba_runtime/robustness_perturbations",
            )
        ),
    )
    parser.add_argument(
        "--keep-perturbation-mmaps",
        action="store_true",
        help="Keep completed local stressed-signal memmaps after their predictions are written.",
    )
    parser.add_argument("--resnet-checkpoint-dir", type=Path, default=DEFAULT_RESNET_CKPT_DIR)
    parser.add_argument("--raw-mamba-checkpoint-dir", type=Path, default=DEFAULT_RAW_MAMBA_CKPT_DIR)
    parser.add_argument("--transformer-checkpoint-dir", type=Path, default=DEFAULT_TRANSFORMER_CKPT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--finalize-manifest-only",
        action="store_true",
        help="Validate existing stress artifacts and rebuild the combined manifest without loading raw ECG.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--out-manifest", type=Path, default=MANIFEST_DIR / "comparator_stress_prediction_manifest.json")
    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    path = resolve(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return device


def autocast_resnet(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def output_path(comparator: str, stress: str) -> Path:
    stem = {"resnet": "resnet1d_cnn", "raw_mamba": "raw_mamba", "transformer": "transformer_ecg"}[comparator]
    return PREDICTION_DIR / f"robustness_{stem}_{stress}_predictions.npz"


def checkpoint_path(args: argparse.Namespace, comparator: str, fold: int) -> Path:
    if comparator == "resnet":
        return resolve(args.resnet_checkpoint_dir) / f"fold{fold}_resnet1d_cnn_final.pt"
    if comparator == "raw_mamba":
        return resolve(args.raw_mamba_checkpoint_dir) / f"fold{fold}_raw_mamba_final_ema.pt"
    if comparator == "transformer":
        return resolve(args.transformer_checkpoint_dir) / f"fold{fold}_transformer_ecg_final.pt"
    raise ValueError(f"Unknown comparator: {comparator}")


def validate_checkpoint_set(comparator: str, paths: list[Path]) -> list[str]:
    manifest_name, expected_protocol = BASELINE_CHECKPOINT_CONTRACTS[comparator]
    manifest_path = MANIFEST_DIR / manifest_name
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {comparator} baseline manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != expected_protocol:
        raise RuntimeError(f"{comparator} baseline manifest protocol mismatch: {manifest_path}")
    contract = payload.get("checkpoint_contract") or {}
    rows = contract.get("checkpoints") or []
    by_fold = {
        int(row["fold"]): row
        for row in rows
        if isinstance(row, dict) and row.get("fold") is not None
    }
    if contract.get("status") != "complete" or sorted(by_fold) != [1, 2, 3, 4, 5]:
        raise RuntimeError(
            f"{comparator} baseline checkpoint contract is incomplete: "
            f"status={contract.get('status')!r}, declared_folds={sorted(by_fold)}, "
            f"missing_folds={sorted({1, 2, 3, 4, 5} - set(by_fold))}. "
            "Rerun the corresponding Notebook 04 baseline cell in aggregate/reuse mode "
            "with the canonical checkpoint directory, then publish the refreshed manifest."
        )
    if len(paths) != 5:
        raise RuntimeError(f"{comparator} stress inference requires exact checkpoint folds 1..5")

    hashes: list[str] = []
    for fold, path in enumerate(paths, start=1):
        path = resolve(path).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing {comparator} fold {fold} checkpoint: {path}")
        row = by_fold[fold]
        declared_path = Path(str(row.get("path") or "")).expanduser()
        if not declared_path.is_absolute():
            declared_path = PROJECT_ROOT / declared_path
        if declared_path.resolve() != path:
            raise RuntimeError(
                f"{comparator} fold {fold} checkpoint path differs from baseline manifest: "
                f"{path} != {declared_path.resolve()}"
            )
        expected_size = int(row.get("size_bytes", -1))
        if expected_size >= 0 and path.stat().st_size != expected_size:
            raise RuntimeError(f"{comparator} fold {fold} checkpoint size mismatch")
        expected_sha = str(row.get("sha256") or "")
        actual_sha = sha256_file(path)
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError(f"{comparator} fold {fold} checkpoint SHA mismatch")
        hashes.append(actual_sha)
    return hashes


def expected_raw_cache_sha256(comparators: list[str]) -> str:
    """Require every selected learned comparator to name the same training raw cache."""
    declared: dict[str, str] = {}
    for comparator in comparators:
        manifest_name, _expected_protocol = BASELINE_CHECKPOINT_CONTRACTS[comparator]
        manifest_path = MANIFEST_DIR / manifest_name
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = str((payload.get("load_info") or {}).get("raw_cache_sha256") or "")
        if not value:
            raise RuntimeError(f"{comparator} baseline manifest does not declare raw_cache_sha256")
        declared[comparator] = value
    unique = sorted(set(declared.values()))
    if len(unique) != 1:
        raise RuntimeError(f"Comparator baseline raw-cache SHA mismatch: {declared}")
    return unique[0]


def npz_member_contract(path: Path, member: str) -> tuple[tuple[int, ...], np.dtype, int]:
    """Read an NPY member header without materializing its array."""
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise KeyError(f"Missing {member} in raw cache: {path}")
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


def remove_stale_partials(destination: Path) -> list[str]:
    """Remove process-local scratch partials left by interrupted runtimes."""
    removed: list[str] = []
    for partial in destination.parent.glob(f".{destination.name}.partial.*"):
        if partial.is_file():
            partial.unlink()
            removed.append(str(partial))
    if removed:
        print(f"Removed {len(removed)} stale local partial(s) for {destination.name}", flush=True)
    return removed


def cleanup_stale_scratch_partials(root: Path) -> list[str]:
    """Sweep interrupted local scratch writes while excluding active writers."""
    root = resolve(root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"startup-cleanup-{os.getpid()}"
    removed: list[str] = []
    with PublishLock(root, run_id=run_id):
        for partial in root.glob(".*.partial.*"):
            if partial.is_file():
                partial.unlink()
                removed.append(str(partial))
    if removed:
        print(f"Removed {len(removed)} interrupted scratch partial(s) under {root}", flush=True)
    return removed


def extract_npz_member_atomic(
    source: Path,
    member: str,
    destination: Path,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    """Serialize validation, extraction, and commit for one local raw-cache member."""
    destination = Path(destination)
    run_id = f"raw-{hashlib.sha256(str(destination).encode('utf-8')).hexdigest()[:16]}-{os.getpid()}"
    with PublishLock(destination.parent, run_id=run_id):
        return _extract_npz_member_atomic_locked(
            source,
            member,
            destination,
            source_sha256=source_sha256,
        )


def _extract_npz_member_atomic_locked(
    source: Path,
    member: str,
    destination: Path,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    """Stream an NPY member to local scratch and bind reuse to its content hash."""
    shape, dtype, expected_size = npz_member_contract(source, member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contract_path = destination.with_suffix(destination.suffix + ".contract.json")
    expected_contract = {
        "capability": DISK_BACKED_RAW_CACHE_CAPABILITY,
        "source_npz_sha256": source_sha256,
        "source_npz_size_bytes": source.stat().st_size,
        "member": member,
        "member_size_bytes": expected_size,
        "shape": list(shape),
        "dtype": str(dtype),
    }
    if destination.is_file() and contract_path.is_file():
        try:
            stored = json.loads(contract_path.read_text(encoding="utf-8"))
            base_matches = all(stored.get(key) == value for key, value in expected_contract.items())
            content_sha = str(stored.get("member_npy_sha256") or "")
            if (
                base_matches
                and destination.stat().st_size == expected_size
                and content_sha
                and sha256_file(destination) == content_sha
            ):
                remove_stale_partials(destination)
                return stored
        except Exception as exc:
            print(f"Rejecting local raw memmap contract: {type(exc).__name__}: {exc}", flush=True)
    destination.unlink(missing_ok=True)
    contract_path.unlink(missing_ok=True)
    remove_stale_partials(destination)
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < expected_size + 512 * 1024 * 1024:
        raise RuntimeError(
            f"Insufficient local scratch for raw ECG memmap: free={free_bytes}, required>{expected_size}"
        )
    partial = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    if partial.exists():
        partial.unlink()
    print(f"Extracting authenticated {member} to local memmap scratch: {destination}", flush=True)
    with zipfile.ZipFile(source) as archive, archive.open(member) as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Raw ECG member extraction size mismatch: {partial.stat().st_size} != {expected_size}"
        )
    source_sha_after = sha256_file(source)
    if source_sha_after != source_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "Raw ECG source changed during member extraction: "
            f"before={source_sha256} after={source_sha_after}"
        )
    contract = {
        **expected_contract,
        "member_npy_sha256": sha256_file(partial),
    }
    os.replace(partial, destination)
    save_json(contract_path, contract)
    return contract


def load_disk_backed_raw_cache(
    *,
    expected_y: np.ndarray,
    expected_record_fingerprint: str,
    explicit_cache: Path | None,
    limit_records: int,
    comparators: list[str],
    mmap_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Authenticate the baseline NPZ, then mmap its X member from local scratch."""
    expected_sha = expected_raw_cache_sha256(comparators)
    checked: list[str] = []
    for candidate in resnet_helpers.candidate_raw_cache_paths(explicit_cache):
        path = resolve(Path(candidate)).resolve()
        checked.append(str(path))
        if not path.is_file() or path.stat().st_size == 0:
            continue
        print(f"Authenticating disk-backed raw ECG cache candidate: {path}", flush=True)
        try:
            x_shape, x_dtype, _member_size = npz_member_contract(path, "X.npy")
            with np.load(path, allow_pickle=False) as data:
                required = {"X", "y", "subjects"}
                missing = required - set(data.files)
                if missing:
                    raise ValueError(f"missing keys {sorted(missing)}")
                y_cache = np.asarray(data["y"], dtype=np.float32)
                subjects = np.asarray(data["subjects"]).astype(str)
                stored_fingerprint = str(resnet_helpers.npz_scalar(data, "record_order_fingerprint", "") or "")
        except Exception as exc:
            print(f"  rejected: {type(exc).__name__}: {exc}", flush=True)
            continue
        if x_shape != (len(y_cache), 12, 5000) or x_dtype != np.dtype(np.float32):
            print(f"  rejected: unexpected X contract shape={x_shape} dtype={x_dtype}", flush=True)
            continue
        computed_fingerprint = resnet_helpers.record_order_fingerprint(subjects)
        if stored_fingerprint and stored_fingerprint != computed_fingerprint:
            print("  rejected: stored and computed record fingerprints differ", flush=True)
            continue
        if expected_record_fingerprint and computed_fingerprint != expected_record_fingerprint:
            print("  rejected: record fingerprint differs from frozen OOF", flush=True)
            continue
        if len(y_cache) < len(expected_y) or not np.array_equal(y_cache[: len(expected_y)], expected_y):
            print("  rejected: labels differ from frozen OOF", flush=True)
            continue
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            print(f"  rejected: raw-cache SHA {actual_sha} != baseline contract {expected_sha}", flush=True)
            continue

        mmap_path = resolve(mmap_dir) / (
            f"chapman_clean_x_{actual_sha[:16]}_{computed_fingerprint}_N{len(y_cache)}.npy"
        )
        mmap_contract = extract_npz_member_atomic(
            path,
            "X.npy",
            mmap_path,
            source_sha256=actual_sha,
        )
        X = np.load(mmap_path, mmap_mode="r", allow_pickle=False)
        if X.shape != x_shape or X.dtype != np.float32:
            raise RuntimeError(f"Extracted raw ECG memmap contract mismatch: {X.shape}, {X.dtype}")
        if limit_records > 0:
            X = X[:limit_records]
            y_cache = y_cache[:limit_records]
            subjects = subjects[:limit_records]
        info = {
            "raw_cache": str(path),
            "raw_cache_sha256": actual_sha,
            "raw_cache_kind": "record_fingerprinted_disk_backed_npz_member",
            "raw_cache_record_order_fingerprint": computed_fingerprint,
            "raw_cache_stored_record_order_fingerprint": stored_fingerprint,
            "raw_cache_shape": list(X.shape),
            "raw_cache_mmap_path": str(mmap_path),
            "raw_cache_mmap_capability": DISK_BACKED_RAW_CACHE_CAPABILITY,
            "raw_cache_mmap_sha256": mmap_contract["member_npy_sha256"],
        }
        print(f"Using disk-backed raw ECG cache: {path}", flush=True)
        print(f"Raw ECG memmap: {mmap_path} shape={X.shape}", flush=True)
        return X, info
    raise FileNotFoundError(
        "No manuscript-safe disk-backed raw ECG cache found. Checked:\n- " + "\n- ".join(checked)
    )


def perturb_signals_disk_backed(
    X: np.ndarray,
    spec: dict[str, Any],
    *,
    out_dir: Path,
    raw_cache_sha256: str,
    source_bundle_sha256: str,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    """Serialize validation, generation, and commit for one perturbation memmap."""
    out_dir = resolve(out_dir)
    lock = perturbation_writer_lock(
        X,
        spec,
        out_dir=out_dir,
        raw_cache_sha256=raw_cache_sha256,
        source_bundle_sha256=source_bundle_sha256,
    )
    with lock:
        return _perturb_signals_disk_backed_locked(
            X,
            spec,
            out_dir=out_dir,
            raw_cache_sha256=raw_cache_sha256,
            source_bundle_sha256=source_bundle_sha256,
        )


def perturbation_writer_lock(
    X: np.ndarray,
    spec: dict[str, Any],
    *,
    out_dir: Path,
    raw_cache_sha256: str,
    source_bundle_sha256: str,
) -> PublishLock:
    """Return the single-writer lease used through perturbation consumption."""
    out_dir = resolve(out_dir)
    lock_key = json.dumps(
        {
            "stress": spec,
            "raw_cache_sha256": raw_cache_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "shape": list(X.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    run_id = f"perturb-{hashlib.sha256(lock_key.encode('utf-8')).hexdigest()[:16]}-{os.getpid()}"
    return PublishLock(out_dir, run_id=run_id)


def _perturb_signals_disk_backed_locked(
    X: np.ndarray,
    spec: dict[str, Any],
    *,
    out_dir: Path,
    raw_cache_sha256: str,
    source_bundle_sha256: str,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    """Apply the canonical perturbation in identical chunks into a local NPY memmap."""
    out_dir = resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_hash = robust_helpers.stable_hash(spec)
    output = out_dir / (
        f"{spec['name']}_{spec_hash}_{raw_cache_sha256[:16]}_"
        f"{source_bundle_sha256[:16]}_N{len(X)}.npy"
    )
    contract_path = output.with_suffix(output.suffix + ".contract.json")
    contract_base = {
        "capability": DISK_BACKED_PERTURBATION_CAPABILITY,
        "raw_cache_sha256": raw_cache_sha256,
        "implementation_source_bundle_sha256": source_bundle_sha256,
        "stress_spec": spec,
        "shape": list(X.shape),
        "dtype": "float32",
    }
    if output.is_file() and contract_path.is_file():
        try:
            stored = json.loads(contract_path.read_text(encoding="utf-8"))
            base_matches = all(stored.get(key) == value for key, value in contract_base.items())
            output_sha = str(stored.get("output_npy_sha256") or "")
            metadata = stored.get("metadata") or {}
            sidecar_path = Path(str(metadata.get("lead_indices_file") or ""))
            sidecar_valid = True
            if metadata.get("lead_indices_file"):
                sidecar_valid = (
                    sidecar_path.is_file()
                    and sha256_file(sidecar_path)
                    == str(metadata.get("lead_indices_file_sha256") or "")
                )
            if (
                base_matches
                and output_sha
                and sha256_file(output) == output_sha
                and sidecar_valid
            ):
                cached = np.load(output, mmap_mode="r", allow_pickle=False)
                if cached.shape == X.shape and cached.dtype == np.float32:
                    remove_stale_partials(output)
                    print(f"Reusing completed local perturbation memmap: {output}", flush=True)
                    return cached, metadata, output
        except Exception as exc:
            print(f"Rejecting local perturbation cache: {type(exc).__name__}: {exc}", flush=True)
    output.unlink(missing_ok=True)
    contract_path.unlink(missing_ok=True)
    remove_stale_partials(output)

    required_bytes = int(np.prod(X.shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    free_bytes = shutil.disk_usage(out_dir).free
    if free_bytes < required_bytes + 512 * 1024 * 1024:
        raise RuntimeError(
            f"Insufficient local scratch for perturbed ECG memmap: free={free_bytes}, required>{required_bytes}"
        )
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}.npy")
    if partial.exists():
        partial.unlink()
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=X.shape)
    meta: dict[str, Any] = {
        "spec": dict(spec),
        "disk_backed": True,
        "perturbation_mmap_path": str(output),
        "perturbation_mmap_capability": DISK_BACKED_PERTURBATION_CAPABILITY,
        "implementation_source_bundle_sha256": source_bundle_sha256,
    }
    kind = spec["kind"]
    if kind == "additive_noise":
        rng = np.random.default_rng(int(spec["seed"]))
        target = 10.0 ** (float(spec["snr_db"]) / 10.0)
        for start in robust_helpers.tqdm(range(0, len(X), 128), desc=f"perturb {spec['name']}"):
            stop = min(len(X), start + 128)
            xb = np.asarray(X[start:stop], dtype=np.float32)
            power = np.mean(xb * xb, axis=(1, 2), keepdims=True)
            noise_std = np.sqrt(np.maximum(power / target, 1e-12)).astype(np.float32)
            noise = rng.standard_normal(size=xb.shape).astype(np.float32) * noise_std
            out[start:stop] = xb + noise
        meta["snr_db"] = float(spec["snr_db"])
    elif kind == "random_lead_dropout":
        rng = np.random.default_rng(int(spec["seed"]))
        n_drop = int(spec["n_drop"])
        dropped = np.empty((len(X), n_drop), dtype=np.int16)
        for index in robust_helpers.tqdm(range(len(X)), desc=f"perturb {spec['name']}"):
            out[index] = X[index]
            leads = np.sort(rng.choice(X.shape[1], size=n_drop, replace=False)).astype(np.int16)
            out[index, leads, :] = 0.0
            dropped[index] = leads
        map_dir = robust_helpers.EXPERIMENTAL_DIR / "robustness_perturbations"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / (
            f"{spec['name']}_{spec_hash}_{raw_cache_sha256[:16]}_"
            f"{source_bundle_sha256[:16]}_N{len(X)}_dropped_leads.npz"
        )
        save_npz_compressed_atomic(
            map_path,
            dropped_leads=dropped,
            stress_json=np.asarray(json.dumps(spec, sort_keys=True)),
            record_count=np.asarray(len(X), dtype=np.int64),
            raw_cache_sha256=np.asarray(raw_cache_sha256),
            implementation_source_bundle_sha256=np.asarray(source_bundle_sha256),
        )
        meta["dropped_leads_shape"] = list(dropped.shape)
        meta["dropped_leads_sha256"] = hashlib.sha256(
            np.ascontiguousarray(dropped).view(np.uint8)
        ).hexdigest()
        meta["lead_indices_file"] = str(map_path)
        meta["lead_indices_file_sha256"] = sha256_file(map_path)
    elif kind == "fixed_lead_dropout":
        leads = np.asarray(spec["lead_indices"], dtype=np.int64)
        for start in robust_helpers.tqdm(range(0, len(X), 128), desc=f"perturb {spec['name']}"):
            stop = min(len(X), start + 128)
            out[start:stop] = X[start:stop]
            out[start:stop, leads, :] = 0.0
        meta["lead_indices"] = leads.astype(int).tolist()
    elif kind == "resample_down_up":
        from scipy.signal import resample_poly

        for start in robust_helpers.tqdm(range(0, len(X), 64), desc=f"perturb {spec['name']}"):
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
    else:
        raise ValueError(f"Unhandled perturbation kind: {kind}")
    out.flush()
    del out
    with partial.open("r+b") as handle:
        os.fsync(handle.fileno())
    output_sha256 = sha256_file(partial)
    os.replace(partial, output)
    contract = {
        **contract_base,
        "output_npy_sha256": output_sha256,
        "metadata": meta,
    }
    save_json(contract_path, contract)
    stressed = np.load(output, mmap_mode="r", allow_pickle=False)
    return stressed, meta, output


def scalar(data: Any, key: str, default: Any = "") -> Any:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.ndim == 0 else value


def validate_existing(
    path: Path,
    y: np.ndarray,
    fold_id: np.ndarray,
    record_id: np.ndarray,
    class_names: list[str],
    *,
    comparator: str,
    stress: str,
    stress_spec: dict[str, Any],
    freeze_contract: dict[str, Any],
    checkpoint_hashes: list[str],
    source_bundle_sha256: str,
    raw_cache_sha256: str | None = None,
) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            stress_metadata = json.loads(str(scalar(data, "stress_metadata_json", "")) or "{}")
            return (
                "y_true" in data.files
                and "y_prob" in data.files
                and np.asarray(data["y_true"]).shape == y.shape
                and np.asarray(data["y_prob"]).shape == y.shape
                and np.array_equal(np.asarray(data["y_true"], dtype=np.float32), y)
                and np.array_equal(np.asarray(data["fold_id"], dtype=np.int16), fold_id)
                and np.array_equal(np.asarray(data["record_id"], dtype=np.int64), record_id)
                and np.asarray(data["class_names"]).astype(str).tolist() == class_names
                and np.isfinite(np.asarray(data["y_prob"], dtype=np.float32)).all()
                and float(np.min(np.asarray(data["y_prob"], dtype=np.float32))) >= -1e-6
                and float(np.max(np.asarray(data["y_prob"], dtype=np.float32))) <= 1.0 + 1e-6
                and str(scalar(data, "protocol")) == PROTOCOL
                and str(scalar(data, "comparator")) == comparator
                and str(scalar(data, "stress_test")) == stress
                and isinstance(stress_metadata.get("spec"), dict)
                and json.dumps(stress_metadata["spec"], sort_keys=True)
                == json.dumps(stress_spec, sort_keys=True)
                and str(scalar(data, "aggregation_implementation")) == POWER_MEAN_IMPLEMENTATION
                and float(scalar(data, "power_mean_q", -1.0)) == float(CONFIG["power_mean_q"])
                and str(scalar(data, "oof_predictions_sha256"))
                == str(freeze_contract.get("oof_predictions_sha256", ""))
                and str(scalar(data, "freeze_manifest_sha256"))
                == str(freeze_contract.get("freeze_manifest_sha256", ""))
                and np.array_equal(
                    np.asarray(data["checkpoint_sha256"]).astype(str),
                    np.asarray(checkpoint_hashes).astype(str),
                )
                and str(scalar(data, "source_bundle_sha256")) == source_bundle_sha256
                and str(scalar(data, "producer_runner_sha256"))
                == source_bundle_contract()["files"]["scripts/revision/23_generate_comparator_stress_predictions.py"]
                and (
                    raw_cache_sha256 is None
                    or str(scalar(data, "raw_cache_sha256")) == raw_cache_sha256
                )
            )
    except Exception:
        return False


def load_trusted_checkpoint(path: Path) -> Any:
    """Load checkpoints generated by the local revision baseline runners."""

    # PyTorch 2.6 defaults torch.load to weights_only=True. Our checkpoint
    # payloads intentionally include trusted metadata such as args/load_info
    # plus the tensor state_dict, so full-pickle loading is required here.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_resnet_model(args: argparse.Namespace, ckpt: Path, device: torch.device):
    model_args = argparse.Namespace(base_channels=64, dropout=0.20)
    payload = load_trusted_checkpoint(ckpt)
    if isinstance(payload, dict) and "args" in payload:
        saved_args = payload.get("args") or {}
        model_args.base_channels = int(saved_args.get("base_channels", model_args.base_channels))
        model_args.dropout = float(saved_args.get("dropout", model_args.dropout))
    model = resnet_helpers.build_model(model_args)
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) else payload)
    return model.to(device).eval()


def load_raw_mamba_model(ckpt: Path, device: torch.device):
    payload = load_trusted_checkpoint(ckpt)
    model = raw_mamba_helpers.build_model()
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) else payload)
    return model.to(device).eval()


def load_transformer_model(ckpt: Path, device: torch.device):
    payload = load_trusted_checkpoint(ckpt)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Transformer checkpoint lacks model_state_dict: {ckpt}")
    saved_args = payload.get("args") or {}
    model_params = payload.get("model_params") or {}
    base_channels = int(saved_args.get("base_channels", 64))
    embed_dim = int(
        model_params.get("embed_dim")
        or saved_args.get("transformer_embed_dim")
        or base_channels
    )
    n_heads = int(
        model_params.get("n_heads")
        or saved_args.get("transformer_heads")
        or (4 if embed_dim % 4 == 0 else 2)
    )
    depth = int(model_params.get("depth") or saved_args.get("transformer_depth") or 3)
    patch_size = int(model_params.get("patch_size") or saved_args.get("transformer_patch_size") or 50)
    patch_stride = int(model_params.get("patch_stride") or saved_args.get("transformer_patch_stride") or 25)
    ff_multiplier = int(
        model_params.get("feed_forward_multiplier")
        or saved_args.get("transformer_ff_multiplier")
        or 4
    )
    dropout = float(saved_args.get("dropout", 0.20))
    if embed_dim % n_heads != 0:
        raise ValueError(
            f"Transformer checkpoint architecture is invalid: embed_dim={embed_dim}, heads={n_heads}"
        )
    model = transformer_helpers.ECGPatchTransformer(
        n_classes=len(CLASSES),
        embed_dim=embed_dim,
        n_heads=n_heads,
        depth=depth,
        patch_size=patch_size,
        patch_stride=patch_stride,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
        max_length=int(CONFIG["slice_length"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def predict_fold(
    *,
    comparator: str,
    model: torch.nn.Module,
    signals: np.ndarray,
    y: np.ndarray,
    val_indices: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record_ids, starts, _positions, skipped = build_slice_index(
        val_indices,
        signals,
        slice_length=int(CONFIG["slice_length"]),
        slice_stride=int(CONFIG["slice_stride"]),
        max_slices_per_record=int(CONFIG["max_slices_per_record"]),
    )
    if skipped:
        raise RuntimeError(f"Fold {fold} skipped validation records under stress: {skipped}")
    dataset = resnet_helpers.RawECGSliceDataset(
        signals,
        y,
        record_ids,
        starts,
        slice_length=int(CONFIG["slice_length"]),
    )
    loader = resnet_helpers.build_loader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        seed=int(args.seed) + int(fold),
        device=device,
    )
    if comparator in {"resnet", "transformer"}:
        slice_prob, slice_record_id, slice_start = resnet_helpers.predict_slice_probabilities(
            model,
            loader,
            device=device,
            use_amp=bool(args.amp),
        )
    else:
        raw_args = argparse.Namespace(amp=bool(args.amp), amp_dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
        slice_prob, slice_record_id, slice_start = raw_mamba_helpers.predict_slice_probabilities(
            model,
            loader,
            device=device,
            args=raw_args,
        )
    y_prob_all, valid_mask, slice_count = aggregate_record_probabilities(
        slice_prob,
        slice_record_id,
        n_records=len(y),
        q=float(CONFIG["power_mean_q"]),
    )
    missing = sorted(set(int(x) for x in val_indices) - set(np.where(valid_mask)[0].astype(int)))
    if missing:
        raise RuntimeError(f"Fold {fold} missing stressed predictions for records: {missing[:10]}")
    return y_prob_all, slice_count, slice_prob, slice_record_id, slice_start


def write_stress_npz(
    path: Path,
    *,
    comparator: str,
    stress: str,
    y: np.ndarray,
    y_prob: np.ndarray,
    record_id: np.ndarray,
    fold_id: np.ndarray,
    slice_count: np.ndarray,
    class_names: list[str],
    stress_meta: dict[str, Any],
    checkpoint_paths: list[Path],
    checkpoint_hashes: list[str],
    raw_cache_info: dict[str, Any],
    freeze_contract: dict[str, Any],
    source_bundle: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_compressed_atomic(
        path,
        y_true=y.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        record_id=record_id.astype(np.int64),
        fold_id=fold_id.astype(np.int16),
        slice_count=slice_count.astype(np.int16),
        class_names=np.asarray(class_names),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(PROTOCOL),
        comparator=np.asarray(comparator),
        stress_test=np.asarray(stress),
        stress_metadata_json=np.asarray(json.dumps(stress_meta, sort_keys=True)),
        aggregation_method=np.asarray("power_mean"),
        aggregation_implementation=np.asarray(POWER_MEAN_IMPLEMENTATION),
        power_mean_q=np.asarray(float(CONFIG["power_mean_q"])),
        checkpoint_paths=np.asarray([project_relative(path) for path in checkpoint_paths]),
        checkpoint_sha256=np.asarray(checkpoint_hashes),
        raw_cache_sha256=np.asarray(raw_cache_info.get("raw_cache_sha256", "")),
        source_bundle_schema_version=np.asarray(int(source_bundle["schema_version"])),
        source_bundle_sha256=np.asarray(str(source_bundle["sha256"])),
        source_bundle_json=np.asarray(json.dumps(source_bundle, sort_keys=True)),
        producer_runner_sha256=np.asarray(
            str(source_bundle["files"]["scripts/revision/23_generate_comparator_stress_predictions.py"])
        ),
        freeze_manifest_sha256=np.asarray(freeze_contract.get("freeze_manifest_sha256", "")),
        oof_predictions_sha256=np.asarray(freeze_contract.get("oof_predictions_sha256", "")),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
    )


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cpu") if args.finalize_manifest_only else select_device(args.device)
    comparators = parse_list(args.comparators)
    stresses = robust_helpers.stress_specs(parse_list(args.stress_tests), int(args.seed))
    source_bundle = source_bundle_contract()
    unknown = sorted(set(comparators) - {"resnet", "raw_mamba", "transformer"})
    if unknown:
        raise ValueError(f"Unsupported comparators for this runner: {unknown}")

    print("=" * 80, flush=True)
    print("COMPARATOR STRESS PREDICTION GENERATOR", flush=True)
    print("=" * 80, flush=True)
    print(f"comparators={comparators} stresses={[s['name'] for s in stresses]} device={device}", flush=True)

    freeze_contract = resnet_helpers.validate_oof_freeze_contract(
        freeze_manifest=args.freeze_manifest,
        oof_predictions=args.oof_predictions,
        expected_checkpoint_kind=args.expected_checkpoint_kind,
    )
    y, fold_id, record_id, class_names, folds, oof_info = resnet_helpers.load_oof_labels_and_folds(
        args.oof_predictions,
        limit_records=int(args.limit_records),
    )
    folds = sorted(folds, key=lambda split: int(split["fold"]))
    checkpoint_contracts: dict[str, dict[str, Any]] = {}
    for comparator in comparators:
        paths = [checkpoint_path(args, comparator, int(split["fold"])) for split in folds]
        absent = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
        hashes = validate_checkpoint_set(comparator, paths) if not absent else []
        checkpoint_contracts[comparator] = {
            "paths": paths,
            "absent": absent,
            "sha256": hashes,
        }
    expected_raw_sha256 = expected_raw_cache_sha256(comparators)

    if args.finalize_manifest_only:
        artifacts: list[dict[str, Any]] = []
        missing: list[str] = []
        for spec in stresses:
            stress = spec["name"]
            for comparator in comparators:
                contract = checkpoint_contracts[comparator]
                if contract["absent"]:
                    missing.append(
                        f"{comparator}/{stress} missing checkpoints: " + "; ".join(contract["absent"])
                    )
                    continue
                out = output_path(comparator, stress)
                if not validate_existing(
                    out,
                    y,
                    fold_id,
                    record_id,
                    class_names,
                    comparator=comparator,
                    stress=stress,
                    stress_spec=spec,
                    freeze_contract=freeze_contract,
                    checkpoint_hashes=contract["sha256"],
                    source_bundle_sha256=str(source_bundle["sha256"]),
                    raw_cache_sha256=expected_raw_sha256,
                ):
                    missing.append(f"{comparator}/{stress} missing or stale prediction: {out}")
                    continue
                artifacts.append(
                    {
                        "comparator": comparator,
                        "stress": stress,
                        "path": project_relative(out),
                        "sha256": sha256_file(out),
                        "reused": True,
                    }
                )
        payload = {
            "status": "complete" if not missing else "blocked_missing_or_stale_artifacts",
            "protocol": PROTOCOL,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "comparators": comparators,
            "stress_tests": [spec["name"] for spec in stresses],
            "artifacts": artifacts,
            "missing": missing,
            "device": "not_loaded_manifest_only",
            "finalize_manifest_only": True,
            "canonical_contract": freeze_contract,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_bundle": source_bundle,
            "requires_saved_comparator_checkpoints": True,
        }
        save_json(resolve(args.out_manifest), payload)
        print(json.dumps({"status": payload["status"], "artifacts": len(artifacts), "missing": len(missing)}, indent=2), flush=True)
        print(f"Wrote manifest: {resolve(args.out_manifest)}", flush=True)
        if missing and args.strict:
            raise RuntimeError("Missing/stale comparator stress artifacts prevent manifest finalization.")
        return

    raw_x: np.ndarray | None = None
    raw_cache_info: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    if args.disk_backed_raw_cache:
        cleanup_stale_scratch_partials(args.raw_cache_mmap_dir)
        cleanup_stale_scratch_partials(args.perturbation_mmap_dir)
    for spec in stresses:
        stress = spec["name"]
        reuse_by_comparator: dict[str, bool] = {}
        for comparator in comparators:
            contract = checkpoint_contracts[comparator]
            out = output_path(comparator, stress)
            reuse_by_comparator[comparator] = bool(
                args.reuse_existing
                and not contract["absent"]
                and validate_existing(
                    out,
                    y,
                    fold_id,
                    record_id,
                    class_names,
                    comparator=comparator,
                    stress=stress,
                    stress_spec=spec,
                    freeze_contract=freeze_contract,
                    checkpoint_hashes=contract["sha256"],
                    source_bundle_sha256=str(source_bundle["sha256"]),
                    raw_cache_sha256=expected_raw_sha256,
                )
            )
        if all(reuse_by_comparator.values()):
            for comparator in comparators:
                out = output_path(comparator, stress)
                print(f"Reusing existing {comparator}/{stress}: {out}", flush=True)
                artifacts.append(
                    {
                        "comparator": comparator,
                        "stress": stress,
                        "path": project_relative(out),
                        "sha256": sha256_file(out),
                        "reused": True,
                    }
                )
            continue

        if raw_x is None:
            if args.disk_backed_raw_cache:
                raw_x, raw_cache_info = load_disk_backed_raw_cache(
                    expected_y=y,
                    expected_record_fingerprint=oof_info.get("dataset_record_order_fingerprint", ""),
                    explicit_cache=args.raw_cache,
                    limit_records=int(args.limit_records),
                    comparators=comparators,
                    mmap_dir=args.raw_cache_mmap_dir,
                )
            else:
                raw_x, raw_cache_info = resnet_helpers.load_raw_cache(
                    expected_y=y,
                    expected_record_fingerprint=oof_info.get("dataset_record_order_fingerprint", ""),
                    explicit_cache=args.raw_cache,
                    limit_records=int(args.limit_records),
                )
            if str(raw_cache_info.get("raw_cache_sha256") or "") != expected_raw_sha256:
                raise RuntimeError("Loaded raw cache differs from the shared comparator baseline contract")

        print(f"\nStress {stress}: perturbing raw ECG", flush=True)
        perturbation_path: Path | None = None
        perturbation_lock: PublishLock | None = None
        if args.disk_backed_raw_cache:
            perturbation_lock = perturbation_writer_lock(
                raw_x,
                spec,
                out_dir=args.perturbation_mmap_dir,
                raw_cache_sha256=expected_raw_sha256,
                source_bundle_sha256=str(source_bundle["sha256"]),
            )
            perturbation_lock.acquire()
            stressed_x, stress_meta, perturbation_path = _perturb_signals_disk_backed_locked(
                raw_x,
                spec,
                out_dir=args.perturbation_mmap_dir,
                raw_cache_sha256=expected_raw_sha256,
                source_bundle_sha256=str(source_bundle["sha256"]),
            )
        else:
            stressed_x, stress_meta = robust_helpers.perturb_signals(raw_x, spec)
        for comparator in comparators:
            out = output_path(comparator, stress)
            contract = checkpoint_contracts[comparator]
            if reuse_by_comparator[comparator]:
                print(f"Reusing existing {comparator}/{stress}: {out}", flush=True)
                artifacts.append({"comparator": comparator, "stress": stress, "path": project_relative(out), "sha256": sha256_file(out), "reused": True})
                continue
            ckpts = contract["paths"]
            absent = contract["absent"]
            if absent:
                msg = f"{comparator}/{stress} missing checkpoints: " + "; ".join(absent)
                print("BLOCKED:", msg, flush=True)
                missing.append(msg)
                continue
            y_prob = np.zeros_like(y, dtype=np.float32)
            slice_count = np.zeros(len(y), dtype=np.int16)
            for split in folds:
                fold = int(split["fold"])
                va_idx = np.asarray(split["va_idx"], dtype=np.int64)
                ckpt = checkpoint_path(args, comparator, fold)
                print(f"{comparator} stress={stress} fold {fold}/5 | val={len(va_idx)} checkpoint={ckpt}", flush=True)
                if comparator == "resnet":
                    model = load_resnet_model(args, ckpt, device)
                elif comparator == "raw_mamba":
                    model = load_raw_mamba_model(ckpt, device)
                elif comparator == "transformer":
                    model = load_transformer_model(ckpt, device)
                else:
                    raise ValueError(f"Unknown comparator: {comparator}")
                fold_prob, fold_slice_count, _slice_prob, _slice_record_id, _slice_start = predict_fold(
                    comparator=comparator,
                    model=model,
                    signals=stressed_x,
                    y=y,
                    val_indices=va_idx,
                    fold=fold,
                    args=args,
                    device=device,
                )
                y_prob[va_idx] = fold_prob[va_idx]
                slice_count[va_idx] = fold_slice_count[va_idx]
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if np.any(slice_count <= 0):
                raise RuntimeError(f"{comparator}/{stress} incomplete slice_count coverage")
            write_stress_npz(
                out,
                comparator=comparator,
                stress=stress,
                y=y,
                y_prob=y_prob,
                record_id=record_id,
                fold_id=fold_id,
                slice_count=slice_count,
                class_names=class_names,
                stress_meta=stress_meta,
                checkpoint_paths=ckpts,
                checkpoint_hashes=contract["sha256"],
                raw_cache_info=raw_cache_info,
                freeze_contract=freeze_contract,
                source_bundle=source_bundle,
            )
            print(f"Wrote {comparator}/{stress}: {out}", flush=True)
            artifacts.append({"comparator": comparator, "stress": stress, "path": project_relative(out), "sha256": sha256_file(out), "reused": False})
        del stressed_x
        gc.collect()
        if perturbation_path is not None and not args.keep_perturbation_mmaps:
            contract_path = perturbation_path.with_suffix(perturbation_path.suffix + ".contract.json")
            perturbation_path.unlink(missing_ok=True)
            contract_path.unlink(missing_ok=True)
        if perturbation_lock is not None:
            perturbation_lock.release()

    if raw_x is not None:
        del raw_x
        gc.collect()

    payload = {
        "status": "complete" if not missing else "blocked_missing_checkpoints",
        "protocol": PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparators": comparators,
        "stress_tests": [spec["name"] for spec in stresses],
        "artifacts": artifacts,
        "missing": missing,
        "device": str(device),
        "canonical_contract": freeze_contract,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_bundle": source_bundle,
        "requires_saved_comparator_checkpoints": True,
    }
    save_json(resolve(args.out_manifest), payload)
    print(json.dumps({"status": payload["status"], "artifacts": len(artifacts), "missing": len(missing)}, indent=2), flush=True)
    print(f"Wrote manifest: {resolve(args.out_manifest)}", flush=True)
    if missing and args.strict:
        raise RuntimeError("Missing comparator checkpoints prevent stressed prediction generation.")


if __name__ == "__main__":
    main()
