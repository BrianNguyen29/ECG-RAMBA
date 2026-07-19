"""Extract frozen ECG-RAMBA branch embeddings for representation probes.

This runner uses the fold assignment frozen in the manuscript OOF artifact and
reuses the PCA/checkpoint contract from ``01_generate_predictions.py``. For
each checkpoint it extracts both that fold's training and validation records,
so downstream probes never cross independently trained latent coordinate
systems. It also preserves the record-level OOF embedding artifact used for
descriptive projections by ``20_representation_probe.py``.

Outputs are cached per fold so Colab interruptions can be resumed without
silently mixing checkpoint/config variants.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

gen = importlib.import_module("scripts.revision.01_generate_predictions")

from configs.config import (  # noqa: E402
    CLASSES,
    CONFIG,
    DEVICE,
    EVALUATION_CONFIG_HASH,
    PATHS,
)
from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    PREDICTION_DIR,
    ensure_revision_dirs,
    save_json,
    sha256_file,
)
from src.provenance import (  # noqa: E402
    canonical_json_sha256,
    exclusive_cache_writer,
    record_order_fingerprint,
)


PROTOCOL = "ecg_ramba_final_ema_branch_embedding_extraction_v1"
LOCAL_COORDINATE_PROTOCOL = "checkpoint_local_train_validation_embeddings_v1"
LOCAL_COORDINATE_SCHEMA_VERSION = 1
REPRESENTATION_FOLD_CACHE_SCHEMA_VERSION = 2
REPRESENTATION_FOLD_CACHE_CONTRACT_SCHEMA_VERSION = 1
EMBEDDING_KEYS = [
    "morphology_embedding",
    "rhythm_embedding",
    "context_embedding",
    "fused_embedding",
]


def array_sha256(array: np.ndarray, dtype: np.dtype | type | None = None) -> str:
    values = np.asarray(array, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def folds_from_frozen_oof(oof: dict[str, Any]) -> list[dict[str, np.ndarray | int | str]]:
    """Build the exact evaluation folds recorded in the frozen OOF artifact."""
    fold_id = np.asarray(oof["fold_id"], dtype=np.int16)
    n_records = len(np.asarray(oof["record_id"]))
    if fold_id.shape != (n_records,):
        raise ValueError(f"Invalid frozen OOF fold_id shape: {fold_id.shape}")

    expected_folds = list(range(1, int(CONFIG["n_folds"]) + 1))
    observed_folds = sorted(int(value) for value in np.unique(fold_id))
    if observed_folds != expected_folds:
        raise ValueError(
            "Frozen OOF fold_id does not contain the expected folds: "
            f"observed={observed_folds} expected={expected_folds}"
        )

    folds: list[dict[str, np.ndarray | int | str]] = []
    for fold_num in expected_folds:
        va_idx = np.flatnonzero(fold_id == fold_num).astype(np.int64)
        tr_idx = np.flatnonzero(fold_id != fold_num).astype(np.int64)
        if len(va_idx) == 0 or len(tr_idx) + len(va_idx) != n_records:
            raise ValueError(f"Frozen OOF fold {fold_num} has an invalid train/validation partition.")
        folds.append(
            {
                "fold_num": fold_num,
                "tr_idx": tr_idx,
                "va_idx": va_idx,
                "train_index_sha256": array_sha256(tr_idx, np.int64),
                "validation_index_sha256": array_sha256(va_idx, np.int64),
            }
        )
    return folds


def validate_checkpoint_fold_contract(
    oof: dict[str, Any], checkpoint_contracts: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Verify that frozen OOF membership matches the checkpoints' persisted folds.pkl."""
    checkpoint_paths = [
        Path(str(row.get("path") or ""))
        for _, row in sorted(checkpoint_contracts.items())
        if str(row.get("path") or "")
    ]
    checkpoint_dirs = {path.parent for path in checkpoint_paths}
    if len(checkpoint_paths) != int(CONFIG["n_folds"]) or len(checkpoint_dirs) != 1:
        raise RuntimeError(
            "Checkpoint fold provenance is incomplete or spans multiple model directories."
        )

    folds_path = next(iter(checkpoint_dirs)) / "folds.pkl"
    if not folds_path.exists() or folds_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Persisted training folds are required beside the checkpoints: {folds_path}"
        )
    persisted_folds = joblib.load(folds_path)
    if len(persisted_folds) != int(CONFIG["n_folds"]):
        raise RuntimeError(
            f"Persisted fold count mismatch: {len(persisted_folds)} != {CONFIG['n_folds']}"
        )

    n_records = len(np.asarray(oof["record_id"]))
    persisted_fold_id = np.full(n_records, -1, dtype=np.int16)
    partition_contracts: list[dict[str, Any]] = []
    for fold_num, fold in enumerate(persisted_folds, start=1):
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        if (
            np.any(tr_idx < 0)
            or np.any(tr_idx >= n_records)
            or np.any(va_idx < 0)
            or np.any(va_idx >= n_records)
        ):
            raise RuntimeError(f"Persisted fold {fold_num} contains out-of-range indices.")
        if len(np.unique(tr_idx)) != len(tr_idx) or len(np.unique(va_idx)) != len(va_idx):
            raise RuntimeError(f"Persisted fold {fold_num} contains duplicate indices.")
        if np.intersect1d(tr_idx, va_idx).size:
            raise RuntimeError(f"Persisted fold {fold_num} train/validation indices overlap.")
        expected_va = np.flatnonzero(np.asarray(oof["fold_id"]) == fold_num).astype(np.int64)
        expected_tr = np.flatnonzero(np.asarray(oof["fold_id"]) != fold_num).astype(np.int64)
        if not np.array_equal(np.sort(va_idx), expected_va) or not np.array_equal(
            np.sort(tr_idx), expected_tr
        ):
            raise RuntimeError(
                f"Persisted fold {fold_num} train/validation membership differs from frozen OOF."
            )
        if np.any(persisted_fold_id[va_idx] != -1):
            raise RuntimeError("Persisted folds assign at least one record to multiple validation folds.")
        persisted_fold_id[va_idx] = fold_num
        partition_contracts.append(
            {
                "fold": fold_num,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "train_index_sha256": array_sha256(tr_idx, np.int64),
                "validation_index_sha256": array_sha256(va_idx, np.int64),
            }
        )
    if np.any(persisted_fold_id < 0):
        raise RuntimeError(
            f"Persisted folds do not cover {int(np.sum(persisted_fold_id < 0))} OOF records."
        )

    current_fold_id = np.asarray(oof["fold_id"], dtype=np.int16)
    if not np.array_equal(persisted_fold_id, current_fold_id):
        mismatch_count = int(np.sum(persisted_fold_id != current_fold_id))
        raise RuntimeError(
            "Frozen OOF fold assignment differs from the folds persisted beside the checkpoints: "
            f"mismatched_records={mismatch_count}."
        )
    return {
        "source": "frozen_oof_fold_id_verified_against_checkpoint_folds",
        "folds_path": str(folds_path),
        "folds_file_sha256": sha256_file(folds_path),
        "fold_assignment_sha256": array_sha256(current_fold_id, np.int16),
        "fold_partitions": partition_contracts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kind", default="final_ema", choices=gen.CHECKPOINT_KINDS)
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=PREDICTION_DIR / "oof_final_ema_predictions.npz",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json",
    )
    parser.add_argument(
        "--oof-run-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_prediction_run_manifest.json",
        help="Prediction run manifest containing exact checkpoint paths and SHA256 values.",
    )
    parser.add_argument(
        "--out-embedding",
        type=Path,
        default=PREDICTION_DIR / "representation_embeddings_final_ema.npz",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "representation_embedding_manifest.json",
    )
    parser.add_argument(
        "--fold-cache-dir",
        type=Path,
        default=PREDICTION_DIR / "folds",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--record-batch-size",
        type=int,
        default=256,
        help="Records materialized at once while extracting checkpoint-local embeddings.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument(
        "--only-folds",
        default="",
        help="Optional comma-separated fold numbers to compute now. Other folds are loaded from cache.",
    )
    parser.add_argument("--resume-fold-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun-folds", action="store_true", default=False)
    parser.add_argument("--min-system-ram-gb", type=float, default=24.0)
    parser.add_argument("--allow-low-ram", action="store_true", default=False)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    path = resolve(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def validate_mamba_runtime_for_extraction() -> None:
    missing = [
        module
        for module in ("mamba_ssm", "causal_conv1d")
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    raise ImportError(
        "Representation extraction requires the ECG-RAMBA Mamba runtime before "
        "loading data or fitting fold PCA. Missing modules: "
        + ", ".join(missing)
        + ". In Colab, run Notebook 00 bootstrap or the Notebook 02 model "
        "dependency/Mamba install cell in the same GPU runtime, restart only if "
        "that installer asks you to, then rerun Notebook 06 from Setup. Existing "
        "fold Hydra/PCA caches are safe to reuse."
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def validate_npz_payload(path: Path, payload: dict[str, Any]) -> None:
    """Fully decompress and compare every array in a temporary NPZ."""
    with np.load(path, allow_pickle=False) as check:
        if set(check.files) != set(payload):
            raise RuntimeError(f"Incomplete temporary NPZ payload: {path}")
        for key in sorted(payload):
            observed = np.asarray(check[key])
            expected = np.asarray(payload[key])
            if observed.shape != expected.shape or observed.dtype != expected.dtype:
                raise RuntimeError(
                    f"Temporary NPZ field mismatch for {key}: "
                    f"{observed.shape}/{observed.dtype} != {expected.shape}/{expected.dtype}"
                )
            if array_sha256(observed) != array_sha256(expected):
                raise RuntimeError(f"Temporary NPZ field checksum mismatch for {key}: {path}")


def atomic_savez_compressed(path: Path, payload: dict[str, Any]) -> None:
    """Validate a compressed NPZ fully, then expose it under an exclusive lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}.npz"
    )
    with exclusive_cache_writer(path):
        try:
            with tmp_path.open("wb") as handle:
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            validate_npz_payload(tmp_path, payload)
            os.replace(tmp_path, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            tmp_path.unlink(missing_ok=True)


def parse_only_folds(value: str) -> set[int] | None:
    if not value.strip():
        return None
    out = {int(item.strip()) for item in value.split(",") if item.strip()}
    invalid = sorted(fold for fold in out if fold < 1 or fold > int(CONFIG["n_folds"]))
    if invalid:
        raise ValueError(f"Invalid fold numbers in --only-folds: {invalid}")
    return out


def load_oof_contract(path: Path, freeze_manifest: Path, limit_records: int) -> dict[str, Any]:
    path = resolve(path)
    freeze_manifest = resolve(freeze_manifest)
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {path}")
    if limit_records == 0 and not freeze_manifest.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {freeze_manifest}")

    with np.load(path, allow_pickle=False) as data:
        required = ["y_true", "record_id", "fold_id", "class_names"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"OOF predictions missing required keys: {missing}")
        y_true = np.asarray(data["y_true"], dtype=np.float32)
        record_id = np.asarray(data["record_id"]).astype(np.int64)
        fold_id = np.asarray(data["fold_id"]).astype(np.int16)
        class_names = np.asarray(data["class_names"]).astype(str)

    if limit_records > 0:
        y_true = y_true[:limit_records]
        record_id = record_id[:limit_records]
        fold_id = fold_id[:limit_records]

    if not np.array_equal(class_names, np.asarray(CLASSES).astype(str)):
        raise ValueError("OOF class_names do not match current config CLASSES.")
    if y_true.shape != (len(record_id), len(CLASSES)):
        raise ValueError(f"Invalid OOF y_true shape: {y_true.shape}")
    if not np.array_equal(record_id, np.arange(len(record_id), dtype=np.int64)):
        raise ValueError("OOF record_id must be the canonical 0..N-1 order for embedding extraction.")

    freeze_payload: dict[str, Any] | None = None
    if freeze_manifest.exists():
        freeze_payload = json.loads(freeze_manifest.read_text(encoding="utf-8"))
        expected_sha = freeze_payload.get("record_file_sha256") or freeze_payload.get("predictions_sha256")
        if expected_sha is None:
            for artifact in freeze_payload.get("artifacts", []):
                artifact_path = str(artifact.get("path", ""))
                if artifact_path.endswith(path.name):
                    expected_sha = artifact.get("sha256")
                    break
        actual_sha = sha256_file(path)
        if limit_records == 0 and not expected_sha:
            raise RuntimeError("Freeze manifest does not declare the OOF prediction SHA256")
        if expected_sha and str(expected_sha) != actual_sha:
            raise RuntimeError(
                "Freeze manifest does not match OOF predictions: "
                f"manifest={expected_sha} actual={actual_sha}"
            )

    return {
        "path": path,
        "sha256": sha256_file(path),
        "freeze_manifest": freeze_manifest,
        "freeze_manifest_sha256": sha256_file(freeze_manifest) if freeze_manifest.exists() else None,
        "freeze_payload": freeze_payload,
        "y_true": y_true,
        "record_id": record_id,
        "fold_id": fold_id,
        "class_names": class_names,
    }


def load_checkpoint_contracts(run_manifest_path: Path, checkpoint_kind: str) -> dict[int, dict[str, Any]]:
    path = resolve(run_manifest_path)
    if not path.exists():
        print(
            f"WARNING: OOF run manifest not found: {path}. "
            "Falling back to canonical checkpoint path lookup.",
            flush=True,
        )
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_rows = payload.get("inputs", {}).get("checkpoints", [])
    contracts: dict[int, dict[str, Any]] = {}
    for row in checkpoint_rows:
        fold = int(row.get("fold", -1))
        row_path = Path(str(row.get("path", "")))
        if fold <= 0 or not row_path.name.endswith(f"_{checkpoint_kind}.pt"):
            continue
        contracts[fold] = row
    if contracts:
        print(
            f"Loaded checkpoint contract from OOF run manifest: {path} | folds={sorted(contracts)}",
            flush=True,
        )
    else:
        print(
            f"WARNING: OOF run manifest has no {checkpoint_kind} checkpoint rows: {path}",
            flush=True,
        )
    return contracts


def resolve_checkpoint_for_fold(
    *,
    fold_num: int,
    checkpoint_kind: str,
    checkpoint_contracts: dict[int, dict[str, Any]],
) -> tuple[Path, str | None]:
    contract = checkpoint_contracts.get(fold_num)
    candidates: list[Path] = []
    expected_sha = None
    if contract:
        expected_sha = str(contract.get("sha256") or "")
        manifest_path = Path(str(contract.get("path", "")))
        if str(manifest_path):
            candidates.append(manifest_path)

            # Support local clones whose Drive root differs from the original
            # Colab absolute path recorded in the manifest.
            marker = "/ECG-Ramba/"
            manifest_posix = manifest_path.as_posix()
            if marker in manifest_posix:
                relative_to_drive = manifest_posix.split(marker, 1)[1]
                drive_root = Path(os.environ.get("ECG_RAMBA_DRIVE_ROOT", ""))
                if str(drive_root):
                    candidates.append(drive_root / relative_to_drive)
                candidates.append(PROJECT_ROOT.parent / relative_to_drive)

    try:
        candidates.append(gen.checkpoint_path(fold_num, checkpoint_kind, allow_fallback=False))
    except FileNotFoundError:
        pass

    # Common retraining output location used by Notebook 02a.
    drive_root = Path(os.environ.get("ECG_RAMBA_DRIVE_ROOT", ""))
    for root in [
        drive_root / "model_runs" / "ema_protocol_e20_v2" if str(drive_root) else None,
        PROJECT_ROOT.parent / "model_runs" / "ema_protocol_e20_v2",
        PROJECT_ROOT / "model_runs" / "ema_protocol_e20_v2",
        Path(PATHS["model_dir"]),
    ]:
        if root is not None:
            candidates.append(root / f"fold{fold_num}_{checkpoint_kind}.pt")

    seen: set[str] = set()
    checked: list[str] = []
    for candidate in candidates:
        if not candidate or not str(candidate):
            continue
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if not candidate.exists():
            continue
        actual_sha = sha256_file(candidate)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(
                f"Checkpoint SHA mismatch for fold {fold_num}: {candidate} "
                f"expected={expected_sha} actual={actual_sha}"
            )
        return candidate, actual_sha

    raise FileNotFoundError(
        f"Missing exact checkpoint for fold {fold_num} ({checkpoint_kind}). "
        "Checked paths: " + "; ".join(checked) + ". "
        "Restore/copy the model_runs/ema_protocol_e20_v2 checkpoints to Drive, "
        "or rerun Notebook 02a retraining before representation extraction."
    )


def fold_embedding_cache_path(
    fold_num: int,
    checkpoint_kind: str,
    fold_cache_dir: Path,
    cache_contract: dict[str, Any],
) -> Path:
    contract_sha = str(cache_contract["contract_sha256"])
    return resolve(fold_cache_dir) / (
        f"representation_local_{checkpoint_kind}_fold{fold_num}_{EVALUATION_CONFIG_HASH}_"
        f"C{contract_sha[:24]}_v{REPRESENTATION_FOLD_CACHE_SCHEMA_VERSION}_"
        f"lc{LOCAL_COORDINATE_SCHEMA_VERSION}.npz"
    )


def representation_source_contract() -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "revision" / "01_generate_predictions.py",
        PROJECT_ROOT / "configs" / "config.py",
        PROJECT_ROOT / "src" / "data_loader.py",
        PROJECT_ROOT / "src" / "features.py",
        PROJECT_ROOT / "src" / "layers.py",
        PROJECT_ROOT / "src" / "model.py",
    ]
    rows = [
        {
            "path": project_relative(path),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {"files": rows, "bundle_sha256": digest.hexdigest()}


def ordered_record_sha256(record_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    values = np.asarray(record_ids).astype(str)
    digest.update(len(values).to_bytes(8, "little", signed=False))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def inspect_pca_artifact(
    *,
    path: Path,
    expected_contract: dict[str, Any],
    expected_raw_dim: int,
) -> tuple[Any, dict[str, Any]] | None:
    contract_path = gen._pca_contract_path(path)
    if not path.is_file() or not contract_path.is_file():
        return None
    try:
        persisted_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if persisted_contract != expected_contract:
            return None
        sha_before = sha256_file(path)
        pca = joblib.load(path)
        if sha256_file(path) != sha_before:
            return None
        components = np.asarray(getattr(pca, "components_", None))
        explained = np.asarray(getattr(pca, "explained_variance_ratio_", None))
        expected_components = int(CONFIG["hydra_dim"])
        if (
            int(getattr(pca, "n_components_", -1)) != expected_components
            or components.shape != (expected_components, expected_raw_dim)
            or explained.shape != (expected_components,)
            or not np.isfinite(components).all()
            or not np.isfinite(explained).all()
        ):
            return None
        identity = {
            "pca_model_path": str(path),
            "pca_model_sha256": sha_before,
            "pca_model_size_bytes": int(path.stat().st_size),
            "pca_contract_path": str(contract_path),
            "pca_contract_file_sha256": sha256_file(contract_path),
            "pca_contract_sha256": str(expected_contract["contract_sha256"]),
            "pca_contract": expected_contract,
            "pca_n_components": expected_components,
            "pca_raw_dim": int(expected_raw_dim),
        }
        return pca, identity
    except Exception:
        return None


def ensure_pca_artifact(
    *,
    path: Path,
    expected_contract: dict[str, Any],
    X_rocket_raw: np.ndarray,
    tr_idx: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    verified = inspect_pca_artifact(
        path=path,
        expected_contract=expected_contract,
        expected_raw_dim=int(X_rocket_raw.shape[1]),
    )
    if verified is not None:
        return verified
    from src.features import fit_pca_on_train

    print(f"PCA artifact is missing/stale; fitting exact fold contract: {path}", flush=True)
    pca = fit_pca_on_train(X_rocket_raw[tr_idx], int(CONFIG["hydra_dim"]))
    gen._atomic_joblib_dump(pca, path, expected_contract)
    verified = inspect_pca_artifact(
        path=path,
        expected_contract=expected_contract,
        expected_raw_dim=int(X_rocket_raw.shape[1]),
    )
    if verified is None:
        raise RuntimeError(f"PCA artifact failed post-write contract validation: {path}")
    return verified


def build_fold_embedding_cache_contract(
    *,
    fold_num: int,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    source_config_hash: str,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    oof_sha256: str,
    freeze_sha256: str,
    split_contract: dict[str, Any],
    cache_provenance: dict[str, Any],
    dataset_record_fingerprint: str,
    dataset_record_order_sha256: str,
    hrv_input_sha256: str,
    pca_identity: dict[str, Any],
) -> dict[str, Any]:
    source_contract = representation_source_contract()
    contract: dict[str, Any] = {
        "schema_version": REPRESENTATION_FOLD_CACHE_CONTRACT_SCHEMA_VERSION,
        "artifact_kind": "checkpoint_local_train_validation_embeddings",
        "cache_schema_version": REPRESENTATION_FOLD_CACHE_SCHEMA_VERSION,
        "local_coordinate_schema_version": LOCAL_COORDINATE_SCHEMA_VERSION,
        "coordinate_protocol": LOCAL_COORDINATE_PROTOCOL,
        "fold": int(fold_num),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "source_config_hash": source_config_hash,
        "evaluation_config_hash": EVALUATION_CONFIG_HASH,
        "oof_predictions_sha256": oof_sha256,
        "freeze_manifest_sha256": freeze_sha256,
        "train_index_sha256": array_sha256(tr_idx, np.int64),
        "validation_index_sha256": array_sha256(va_idx, np.int64),
        "fold_assignment_sha256": str(split_contract["fold_assignment_sha256"]),
        "folds_file_sha256": str(split_contract["folds_file_sha256"]),
        "dataset_record_order_fingerprint": dataset_record_fingerprint,
        "dataset_record_order_sha256": dataset_record_order_sha256,
        "hrv_input_sha256": hrv_input_sha256,
        "raw_input_contract_sha256": str(cache_provenance["contract_sha256"]),
        "raw_input_contract": cache_provenance,
        "representation_source_bundle_sha256": str(source_contract["bundle_sha256"]),
        "representation_source_contract": source_contract,
        **pca_identity,
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def validate_stored_fold_cache_contract(
    *,
    contract: dict[str, Any],
    fold_num: int,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    oof: dict[str, Any],
    split_contract: dict[str, Any],
    current_archive_sha256: str | None = None,
) -> list[str]:
    issues: list[str] = []
    required = {
        "schema_version",
        "artifact_kind",
        "cache_schema_version",
        "local_coordinate_schema_version",
        "coordinate_protocol",
        "fold",
        "checkpoint_kind",
        "checkpoint_sha256",
        "source_config_hash",
        "evaluation_config_hash",
        "oof_predictions_sha256",
        "freeze_manifest_sha256",
        "train_index_sha256",
        "validation_index_sha256",
        "fold_assignment_sha256",
        "folds_file_sha256",
        "dataset_record_order_fingerprint",
        "dataset_record_order_sha256",
        "hrv_input_sha256",
        "raw_input_contract_sha256",
        "raw_input_contract",
        "representation_source_bundle_sha256",
        "representation_source_contract",
        "pca_model_path",
        "pca_model_sha256",
        "pca_model_size_bytes",
        "pca_contract_path",
        "pca_contract_file_sha256",
        "pca_contract_sha256",
        "pca_contract",
        "pca_n_components",
        "pca_raw_dim",
        "contract_sha256",
    }
    missing = sorted(required - set(contract))
    if missing:
        return [f"missing_contract_fields={','.join(missing)}"]
    observed_sha = str(contract.get("contract_sha256") or "")
    actual_sha = canonical_json_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )
    if observed_sha != actual_sha:
        issues.append("cache_contract_sha256_mismatch")
    current_source = representation_source_contract()
    raw_contract = contract.get("raw_input_contract") or {}
    if not isinstance(raw_contract, dict):
        return ["raw_input_contract_not_mapping"]
    if canonical_json_sha256(
        {key: value for key, value in raw_contract.items() if key != "contract_sha256"}
    ) != str(raw_contract.get("contract_sha256") or ""):
        issues.append("raw_input_contract_sha256_invalid")
    if contract.get("raw_input_contract_sha256") != raw_contract.get("contract_sha256"):
        issues.append("raw_input_contract_binding_mismatch")
    if (
        contract.get("dataset_record_order_fingerprint")
        != raw_contract.get("dataset_record_order_fingerprint")
        or raw_contract.get("evaluation_config_hash") != EVALUATION_CONFIG_HASH
    ):
        issues.append("raw_input_record_or_config_contract_mismatch")
    archive_path = Path(PATHS["zip_path"]).resolve()
    if not archive_path.is_file():
        issues.append("source_archive_missing")
    elif raw_contract.get("source_archive_sha256") != (
        current_archive_sha256 or sha256_file(archive_path)
    ):
        issues.append("source_archive_sha256_mismatch")
    expected_values = {
        "schema_version": REPRESENTATION_FOLD_CACHE_CONTRACT_SCHEMA_VERSION,
        "artifact_kind": "checkpoint_local_train_validation_embeddings",
        "cache_schema_version": REPRESENTATION_FOLD_CACHE_SCHEMA_VERSION,
        "local_coordinate_schema_version": LOCAL_COORDINATE_SCHEMA_VERSION,
        "coordinate_protocol": LOCAL_COORDINATE_PROTOCOL,
        "fold": fold_num,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_config_hash": EVALUATION_CONFIG_HASH,
        "oof_predictions_sha256": oof["sha256"],
        "freeze_manifest_sha256": str(oof["freeze_manifest_sha256"] or ""),
        "train_index_sha256": array_sha256(tr_idx, np.int64),
        "validation_index_sha256": array_sha256(va_idx, np.int64),
        "fold_assignment_sha256": str(split_contract.get("fold_assignment_sha256") or ""),
        "folds_file_sha256": str(split_contract.get("folds_file_sha256") or ""),
        "representation_source_bundle_sha256": current_source["bundle_sha256"],
    }
    for key, expected in expected_values.items():
        if contract.get(key) != expected:
            issues.append(f"{key}_mismatch")
    if contract.get("representation_source_contract") != current_source:
        issues.append("representation_source_contract_mismatch")

    pca_path = Path(str(contract.get("pca_model_path") or ""))
    expected_pca_contract = contract.get("pca_contract") or {}
    if (
        expected_pca_contract.get("artifact_kind") != "fold_train_pca"
        or int(expected_pca_contract.get("fold", -1)) != fold_num
        or expected_pca_contract.get("source_config_hash")
        != contract.get("source_config_hash")
        or expected_pca_contract.get("base_contract_sha256")
        != raw_contract.get("contract_sha256")
        or expected_pca_contract.get("base_contract") != raw_contract
        or expected_pca_contract.get("train_index_hash")
        != gen.index_fingerprint(tr_idx)
    ):
        issues.append("pca_raw_input_or_split_contract_mismatch")
    verified_pca = inspect_pca_artifact(
        path=pca_path,
        expected_contract=expected_pca_contract,
        expected_raw_dim=int(contract.get("pca_raw_dim", -1)),
    )
    if verified_pca is None:
        issues.append("pca_artifact_or_contract_invalid")
    else:
        _, identity = verified_pca
        for key in (
            "pca_model_path",
            "pca_model_sha256",
            "pca_model_size_bytes",
            "pca_contract_path",
            "pca_contract_file_sha256",
            "pca_contract_sha256",
            "pca_n_components",
            "pca_raw_dim",
        ):
            if contract.get(key) != identity.get(key):
                issues.append(f"{key}_mismatch")
    return sorted(set(issues))


def load_fold_embedding_cache(
    *,
    path: Path,
    fold_num: int,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    expected_contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
            expected_keys = {
                "train_record_id",
                "validation_record_id",
                "train_slice_count",
                "validation_slice_count",
                "fold",
                "cache_schema_version",
                "local_coordinate_schema_version",
                "coordinate_protocol",
                "source_config_hash",
                "evaluation_config_hash",
                "checkpoint_sha256",
                "oof_predictions_sha256",
                "freeze_manifest_sha256",
                "source_bundle_sha256",
                "train_index_sha256",
                "validation_index_sha256",
                "cache_contract_sha256",
                "cache_contract_json",
                "fold_summary_json",
                *{f"train_{key}" for key in EMBEDDING_KEYS},
                *{f"validation_{key}" for key in EMBEDDING_KEYS},
            }
            if set(data.files) != expected_keys:
                raise ValueError(
                    "Representation fold cache field set mismatch: "
                    f"missing={sorted(expected_keys - set(data.files))} "
                    f"unexpected={sorted(set(data.files) - expected_keys)}"
                )
            # Materialize every field while the archive is open. This detects
            # truncated/corrupt compressed members before reuse.
            arrays = {key: np.asarray(data[key]) for key in data.files}

            train_record_id = arrays["train_record_id"]
            validation_record_id = arrays["validation_record_id"]
            train_slice_count = arrays["train_slice_count"]
            validation_slice_count = arrays["validation_slice_count"]
            if train_record_id.dtype != np.int64 or validation_record_id.dtype != np.int64:
                raise ValueError("Representation cache record ids must be int64")
            if train_slice_count.dtype != np.int16 or validation_slice_count.dtype != np.int16:
                raise ValueError("Representation cache slice counts must be int16")

            cached_contract = json.loads(str(arrays["cache_contract_json"].item()))
            cached_contract_sha = str(arrays["cache_contract_sha256"].item())
            if (
                cached_contract != expected_contract
                or cached_contract_sha != expected_contract.get("contract_sha256")
                or canonical_json_sha256(
                    {key: value for key, value in cached_contract.items() if key != "contract_sha256"}
                )
                != cached_contract_sha
            ):
                raise ValueError("Representation fold cache exact contract mismatch")

            summary = json.loads(str(arrays["fold_summary_json"].item()))
            split_payloads = {
                "train": {
                    "record_id": train_record_id,
                    "slice_count": train_slice_count,
                    "embeddings": {
                        key: arrays[f"train_{key}"]
                        for key in EMBEDDING_KEYS
                    },
                },
                "validation": {
                    "record_id": validation_record_id,
                    "slice_count": validation_slice_count,
                    "embeddings": {
                        key: arrays[f"validation_{key}"]
                        for key in EMBEDDING_KEYS
                    },
                },
            }
        if (
            int(arrays["fold"].item()) != fold_num
            or int(arrays["cache_schema_version"].item())
            != REPRESENTATION_FOLD_CACHE_SCHEMA_VERSION
            or int(arrays["local_coordinate_schema_version"].item())
            != LOCAL_COORDINATE_SCHEMA_VERSION
            or str(arrays["checkpoint_sha256"].item())
            != expected_contract["checkpoint_sha256"]
            or str(arrays["evaluation_config_hash"].item())
            != expected_contract["evaluation_config_hash"]
            or str(arrays["coordinate_protocol"].item()) != LOCAL_COORDINATE_PROTOCOL
            or str(arrays["oof_predictions_sha256"].item())
            != expected_contract["oof_predictions_sha256"]
            or str(arrays["freeze_manifest_sha256"].item())
            != expected_contract["freeze_manifest_sha256"]
            or str(arrays["source_bundle_sha256"].item())
            != expected_contract["representation_source_bundle_sha256"]
            or str(arrays["source_config_hash"].item())
            != expected_contract["source_config_hash"]
            or str(arrays["train_index_sha256"].item())
            != expected_contract["train_index_sha256"]
            or str(arrays["validation_index_sha256"].item())
            != expected_contract["validation_index_sha256"]
            or not np.array_equal(train_record_id, tr_idx.astype(np.int64))
            or not np.array_equal(validation_record_id, va_idx.astype(np.int64))
            or train_slice_count.shape != tr_idx.shape
            or validation_slice_count.shape != va_idx.shape
            or np.intersect1d(train_record_id, validation_record_id).size != 0
        ):
            print(f"WARNING: Representation fold cache contract mismatch: {path}", flush=True)
            return None
        embedding_dims: set[int] = set()
        for split_name, expected_idx in (("train", tr_idx), ("validation", va_idx)):
            split = split_payloads[split_name]
            if np.any(split["slice_count"] <= 0):
                print(f"WARNING: Invalid cached {split_name} slice counts in {path}", flush=True)
                return None
            for key, arr in split["embeddings"].items():
                if arr.ndim != 2 or arr.shape[0] != len(expected_idx) or arr.dtype != np.float32:
                    print(
                        f"WARNING: Invalid cached {split_name}/{key} shape/dtype in "
                        f"{path}: {arr.shape} {arr.dtype}",
                        flush=True,
                    )
                    return None
                if not np.isfinite(arr).all():
                    print(f"WARNING: Non-finite cached {split_name}/{key} in {path}", flush=True)
                    return None
                embedding_dims.add(int(arr.shape[1]))
        if len(embedding_dims) != 1 or next(iter(embedding_dims), 0) <= 0:
            print(f"WARNING: Inconsistent cached embedding dimensions in {path}", flush=True)
            return None
        print(f"Loaded representation cache for fold {fold_num}: {path}", flush=True)
        return split_payloads, summary
    except Exception as exc:
        print(f"WARNING: Could not load representation cache {path}: {exc}", flush=True)
        return None


def save_fold_embedding_cache(
    *,
    path: Path,
    fold_num: int,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    train_embeddings: dict[str, np.ndarray],
    validation_embeddings: dict[str, np.ndarray],
    train_slice_count: np.ndarray,
    validation_slice_count: np.ndarray,
    cache_contract: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cache_contract.get("contract_sha256") != canonical_json_sha256(
        {key: value for key, value in cache_contract.items() if key != "contract_sha256"}
    ):
        raise ValueError("Refusing to write representation cache with invalid contract SHA")
    payload: dict[str, Any] = {
        "train_record_id": tr_idx.astype(np.int64),
        "validation_record_id": va_idx.astype(np.int64),
        "train_slice_count": train_slice_count.astype(np.int16),
        "validation_slice_count": validation_slice_count.astype(np.int16),
        "fold": np.asarray(fold_num, dtype=np.int16),
        "cache_schema_version": np.asarray(
            REPRESENTATION_FOLD_CACHE_SCHEMA_VERSION, dtype=np.int16
        ),
        "local_coordinate_schema_version": np.asarray(
            LOCAL_COORDINATE_SCHEMA_VERSION, dtype=np.int16
        ),
        "coordinate_protocol": np.asarray(LOCAL_COORDINATE_PROTOCOL),
        "source_config_hash": np.asarray(cache_contract["source_config_hash"]),
        "evaluation_config_hash": np.asarray(cache_contract["evaluation_config_hash"]),
        "checkpoint_sha256": np.asarray(cache_contract["checkpoint_sha256"]),
        "oof_predictions_sha256": np.asarray(cache_contract["oof_predictions_sha256"]),
        "freeze_manifest_sha256": np.asarray(cache_contract["freeze_manifest_sha256"]),
        "source_bundle_sha256": np.asarray(
            cache_contract["representation_source_bundle_sha256"]
        ),
        "train_index_sha256": np.asarray(array_sha256(tr_idx, np.int64)),
        "validation_index_sha256": np.asarray(array_sha256(va_idx, np.int64)),
        "cache_contract_sha256": np.asarray(cache_contract["contract_sha256"]),
        "cache_contract_json": np.asarray(json.dumps(cache_contract, sort_keys=True)),
        "fold_summary_json": np.asarray(json.dumps(summary, sort_keys=True)),
    }
    for key in EMBEDDING_KEYS:
        payload[f"train_{key}"] = train_embeddings[key].astype(np.float32)
        payload[f"validation_{key}"] = validation_embeddings[key].astype(np.float32)
    atomic_savez_compressed(path, payload)
    print(f"Wrote representation cache for fold {fold_num}: {path}", flush=True)


def forward_with_embeddings(
    model: torch.nn.Module,
    x: torch.Tensor,
    xh: torch.Tensor,
    xhr: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Expose embeddings through the model's parity-preserving forward hook."""
    hook = getattr(model, "forward_with_embeddings", None)
    if not callable(hook):
        raise RuntimeError(
            "Loaded model lacks forward_with_embeddings(); use the current src/model.py."
        )
    logits, embeddings = hook(x, xh, xhr)
    missing = sorted(set(EMBEDDING_KEYS) - set(embeddings))
    if missing:
        raise RuntimeError(f"Model embedding hook is incomplete: missing={missing}")
    return logits, {key: embeddings[key].float() for key in EMBEDDING_KEYS}


def extract_fold_embeddings(
    *,
    fold_num: int,
    model: torch.nn.Module,
    xs: np.ndarray,
    xh: np.ndarray,
    xhr: np.ndarray,
    rids: np.ndarray,
    va_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    split_name: str = "validation",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if len(rids) == 0:
        raise RuntimeError(f"Fold {fold_num} has no slices.")

    dataset = gen.ECGSliceDatasetInfer(xs, xh, xhr, rids)
    loader = gen.make_inference_loader(
        dataset,
        batch_size=max(1, int(batch_size)),
        num_workers=max(0, int(num_workers)),
    )
    rid_to_pos = {int(rid): pos for pos, rid in enumerate(va_idx.astype(np.int64))}
    slice_count = np.zeros(len(va_idx), dtype=np.int32)
    sums: dict[str, np.ndarray] | None = None

    with torch.no_grad():
        for x_batch, xh_batch, xhr_batch, rid_batch in tqdm(
            loader,
            desc=f"Embeddings fold {fold_num} {split_name}",
            leave=False,
        ):
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            xh_batch = xh_batch.to(DEVICE, non_blocking=True)
            xhr_batch = xhr_batch.to(DEVICE, non_blocking=True)

            if DEVICE == "cuda":
                with torch.amp.autocast("cuda", dtype=gen.AMP_DTYPE):
                    _, embeddings_t = forward_with_embeddings(model, x_batch, xh_batch, xhr_batch)
            else:
                _, embeddings_t = forward_with_embeddings(model, x_batch, xh_batch, xhr_batch)

            rid_np = rid_batch.cpu().numpy().astype(np.int64)
            positions = np.asarray([rid_to_pos[int(rid)] for rid in rid_np], dtype=np.int64)
            if sums is None:
                sums = {
                    key: np.zeros((len(va_idx), value.shape[1]), dtype=np.float32)
                    for key, value in embeddings_t.items()
                }
            np.add.at(slice_count, positions, 1)
            for key, value in embeddings_t.items():
                arr = value.detach().cpu().numpy().astype(np.float32)
                np.add.at(sums[key], positions, arr)

    if sums is None:
        raise RuntimeError(f"Fold {fold_num} produced no embeddings.")
    if np.any(slice_count <= 0):
        missing = va_idx[np.where(slice_count <= 0)[0]][:20].tolist()
        raise RuntimeError(f"Fold {fold_num} has records without extracted slices: {missing}")
    averaged = {
        key: (arr / slice_count[:, None].astype(np.float32)).astype(np.float32)
        for key, arr in sums.items()
    }
    return averaged, slice_count.astype(np.int16)


def extract_partition_embeddings(
    *,
    fold_num: int,
    split_name: str,
    model: torch.nn.Module,
    X: np.ndarray,
    X_hrv: np.ndarray,
    X_rocket_raw: np.ndarray,
    record_indices: np.ndarray,
    pca: Any,
    precomputed_hydra: np.ndarray | None,
    record_batch_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract one checkpoint's embeddings without materializing all slices."""
    from src.features import apply_pca

    record_indices = np.asarray(record_indices, dtype=np.int64)
    if not len(record_indices):
        raise RuntimeError(f"Fold {fold_num} {split_name} partition is empty.")
    if precomputed_hydra is not None:
        precomputed_hydra = np.asarray(precomputed_hydra, dtype=np.float32)
        if precomputed_hydra.shape != (len(record_indices), int(CONFIG["hydra_dim"])):
            raise ValueError(
                f"Fold {fold_num} {split_name} Hydra shape mismatch: "
                f"{precomputed_hydra.shape}"
            )

    output: dict[str, np.ndarray] | None = None
    slice_count = np.zeros(len(record_indices), dtype=np.int16)
    chunk_size = max(1, int(record_batch_size))
    for start in range(0, len(record_indices), chunk_size):
        stop = min(start + chunk_size, len(record_indices))
        chunk_idx = record_indices[start:stop]
        if precomputed_hydra is None:
            hydra_chunk = apply_pca(pca, X_rocket_raw[chunk_idx])
        else:
            hydra_chunk = precomputed_hydra[start:stop]
        hydra_by_record = {
            int(record_id): hydra_chunk[pos]
            for pos, record_id in enumerate(chunk_idx)
        }
        xs, xh, xhr, rids, build_slice_counts = gen.build_fold_slices(
            chunk_idx,
            X,
            X_hrv,
            hydra_by_record,
        )
        chunk_embeddings, chunk_slice_count = extract_fold_embeddings(
            fold_num=fold_num,
            model=model,
            xs=xs,
            xh=xh,
            xhr=xhr,
            rids=rids,
            va_idx=chunk_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            split_name=split_name,
        )
        expected_counts = np.asarray(
            [build_slice_counts[int(record_id)] for record_id in chunk_idx],
            dtype=np.int16,
        )
        if not np.array_equal(chunk_slice_count, expected_counts):
            raise RuntimeError(
                f"Fold {fold_num} {split_name} slice counts changed during extraction."
            )
        if output is None:
            output = {
                key: np.empty((len(record_indices), values.shape[1]), dtype=np.float32)
                for key, values in chunk_embeddings.items()
            }
        for key in EMBEDDING_KEYS:
            output[key][start:stop] = chunk_embeddings[key]
        slice_count[start:stop] = chunk_slice_count
        print(
            f"Fold {fold_num} {split_name}: records {stop}/{len(record_indices)}",
            flush=True,
        )
        del xs, xh, xhr, rids, hydra_chunk, chunk_embeddings
        gc.collect()

    if output is None or any(not np.isfinite(values).all() for values in output.values()):
        raise RuntimeError(f"Fold {fold_num} {split_name} produced invalid embeddings.")
    return output, slice_count


def write_final_embedding_npz(
    *,
    path: Path,
    oof: dict[str, Any],
    embeddings: dict[str, np.ndarray],
    fold_id: np.ndarray,
    slice_count: np.ndarray,
    payload: dict[str, Any],
) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "y_true": oof["y_true"].astype(np.float32),
        "record_id": oof["record_id"].astype(np.int64),
        "fold_id": fold_id.astype(np.int16),
        "slice_count": slice_count.astype(np.int16),
        "class_names": oof["class_names"].astype(str),
        "protocol": np.asarray(PROTOCOL),
        "coordinate_protocol": np.asarray(LOCAL_COORDINATE_PROTOCOL),
        "checkpoint_kind": np.asarray(payload["checkpoint_kind"]),
        "oof_predictions_sha256": np.asarray(oof["sha256"]),
        "freeze_manifest_sha256": np.asarray(oof["freeze_manifest_sha256"] or ""),
        "source_bundle_sha256": np.asarray(
            payload.get("source_contract", representation_source_contract())["bundle_sha256"]
        ),
        "local_fold_cache_index_json": np.asarray(
            json.dumps(payload.get("local_fold_cache_index", []), sort_keys=True)
        ),
        "dataset_record_order_fingerprint": np.asarray(payload["dataset_record_order_fingerprint"]),
        "embedding_manifest_json": np.asarray(json.dumps(jsonable(payload), sort_keys=True)),
    }
    arrays.update({key: value.astype(np.float32) for key, value in embeddings.items()})
    atomic_savez_compressed(path, arrays)
    print(f"Wrote representation embeddings: {path}", flush=True)


def inspect_final_embedding_reuse(
    path: Path,
    oof: dict[str, Any],
    checkpoint_kind: str,
    checkpoint_contracts: dict[int, dict[str, Any]],
    split_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = resolve(path)
    audit: dict[str, Any] = {
        "reusable": False,
        "issues": [],
        "semantic_fields": ["y_true", "record_id", "fold_id", "class_names"],
    }
    if not path.exists() or path.stat().st_size == 0:
        audit["issues"].append("embedding_missing_or_empty")
        return audit
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "y_true",
                "record_id",
                "fold_id",
                "class_names",
                "protocol",
                "checkpoint_kind",
                "oof_predictions_sha256",
                "freeze_manifest_sha256",
                "source_bundle_sha256",
                "coordinate_protocol",
                "local_fold_cache_index_json",
                "slice_count",
                "embedding_manifest_json",
                *EMBEDDING_KEYS,
            }
            missing = sorted(required - set(data.files))
            if missing:
                audit["issues"].append(f"missing_fields={','.join(missing)}")
                return audit

            source_oof_sha = str(data["oof_predictions_sha256"].item())
            source_freeze_sha = str(data["freeze_manifest_sha256"].item())
            audit.update(
                {
                    "source_oof_sha256": source_oof_sha,
                    "source_freeze_sha256": source_freeze_sha,
                    "current_oof_sha256": oof["sha256"],
                    "current_freeze_sha256": str(oof["freeze_manifest_sha256"] or ""),
                    "exact_oof_sha_match": source_oof_sha == oof["sha256"],
                    "exact_freeze_sha_match": source_freeze_sha
                    == str(oof["freeze_manifest_sha256"] or ""),
                }
            )
            if str(data["protocol"].item()) != PROTOCOL:
                audit["issues"].append("protocol_mismatch")
            if str(data["coordinate_protocol"].item()) != LOCAL_COORDINATE_PROTOCOL:
                audit["issues"].append("coordinate_protocol_mismatch")
            if str(data["checkpoint_kind"].item()) != checkpoint_kind:
                audit["issues"].append("checkpoint_kind_mismatch")
            current_source_bundle_sha = representation_source_contract()["bundle_sha256"]
            observed_source_bundle_sha = str(data["source_bundle_sha256"].item())
            audit["source_bundle_sha256"] = observed_source_bundle_sha
            audit["current_source_bundle_sha256"] = current_source_bundle_sha
            if observed_source_bundle_sha != current_source_bundle_sha:
                audit["issues"].append("representation_source_bundle_mismatch")

            embedded_fold_id = np.asarray(data["fold_id"], dtype=np.int16)
            current_fold_id = np.asarray(oof["fold_id"], dtype=np.int16)
            semantic_field_match = {
                "y_true": bool(
                    np.array_equal(np.asarray(data["y_true"], dtype=np.float32), oof["y_true"])
                ),
                "record_id": bool(
                    np.array_equal(np.asarray(data["record_id"]).astype(np.int64), oof["record_id"])
                ),
                "fold_id": bool(np.array_equal(embedded_fold_id, current_fold_id)),
                "class_names": bool(
                    np.array_equal(np.asarray(data["class_names"]).astype(str), oof["class_names"])
                ),
            }
            audit["semantic_field_match"] = semantic_field_match
            semantic_match = all(semantic_field_match.values())
            audit["semantic_contract_match"] = semantic_match
            if not semantic_match:
                audit["issues"].append("oof_semantic_contract_mismatch")
            if not semantic_field_match["fold_id"]:
                audit["issues"].append("oof_fold_assignment_mismatch")
                audit["fold_assignment_mismatch_count"] = int(
                    np.sum(embedded_fold_id != current_fold_id)
                ) if embedded_fold_id.shape == current_fold_id.shape else None
                audit["source_fold_assignment_sha256"] = array_sha256(
                    embedded_fold_id, np.int16
                )
                audit["current_fold_assignment_sha256"] = array_sha256(
                    current_fold_id, np.int16
                )

            embedded_manifest = json.loads(str(data["embedding_manifest_json"].item()))
            local_cache_index = json.loads(str(data["local_fold_cache_index_json"].item()))
            audit["existing_semantic_reuse_attestation"] = embedded_manifest.get(
                "semantic_reuse_attestation"
            )
            observed_checkpoint_shas = {
                int(row.get("fold", -1)): str(row.get("checkpoint_sha256") or "")
                for row in embedded_manifest.get("fold_summaries", [])
                if int(row.get("fold", -1)) > 0
            }
            expected_checkpoint_shas = {
                int(fold): str(row.get("sha256") or "")
                for fold, row in checkpoint_contracts.items()
                if str(row.get("sha256") or "")
            }
            expected_folds = set(range(1, int(CONFIG["n_folds"]) + 1))
            audit["checkpoint_contract_match"] = bool(
                set(expected_checkpoint_shas) == expected_folds
                and observed_checkpoint_shas == expected_checkpoint_shas
            )
            audit["checkpoint_sha256_by_fold"] = observed_checkpoint_shas
            if not audit["checkpoint_contract_match"]:
                audit["issues"].append("checkpoint_sha_contract_mismatch_or_incomplete")

            archive_path = Path(PATHS["zip_path"]).resolve()
            current_archive_sha256 = (
                sha256_file(archive_path) if archive_path.is_file() else None
            )
            audit["current_source_archive"] = str(archive_path)
            audit["current_source_archive_sha256"] = current_archive_sha256
            normalized_folds = folds_from_frozen_oof(oof)
            effective_split_contract = split_contract or {
                "fold_assignment_sha256": array_sha256(oof["fold_id"], np.int16),
                "folds_file_sha256": "",
            }
            local_cache_by_fold = {
                int(row.get("fold", -1)): row
                for row in local_cache_index
                if int(row.get("fold", -1)) > 0
            }
            expected_folds = set(range(1, int(CONFIG["n_folds"]) + 1))
            local_cache_contracts: list[dict[str, Any]] = []
            if set(local_cache_by_fold) != expected_folds:
                audit["issues"].append("checkpoint_local_cache_index_incomplete")
            else:
                for fold in normalized_folds:
                    fold_num = int(fold["fold_num"])
                    row = local_cache_by_fold[fold_num]
                    cache_path = resolve(Path(str(row.get("path") or "")))
                    expected_cache_sha = str(row.get("sha256") or "")
                    expected_checkpoint_sha = str(
                        checkpoint_contracts.get(fold_num, {}).get("sha256") or ""
                    )
                    stored_cache_contract = row.get("cache_contract")
                    stored_cache_contract_sha = str(
                        row.get("cache_contract_sha256") or ""
                    )
                    cache_status = {
                        "fold": fold_num,
                        "path": str(cache_path),
                        "sha256": expected_cache_sha,
                        "valid": False,
                    }
                    if (
                        not cache_path.exists()
                        or not expected_cache_sha
                        or sha256_file(cache_path) != expected_cache_sha
                        or str(row.get("checkpoint_sha256") or "")
                        != expected_checkpoint_sha
                        or str(row.get("coordinate_system_id") or "")
                        != f"fold{fold_num}:{expected_checkpoint_sha}"
                        or str(row.get("train_index_sha256") or "")
                        != str(fold["train_index_sha256"])
                        or str(row.get("validation_index_sha256") or "")
                        != str(fold["validation_index_sha256"])
                        or not isinstance(stored_cache_contract, dict)
                        or stored_cache_contract_sha
                        != str((stored_cache_contract or {}).get("contract_sha256") or "")
                    ):
                        audit["issues"].append(
                            f"checkpoint_local_cache_contract_mismatch_fold{fold_num}"
                        )
                    else:
                        provenance_issues = validate_stored_fold_cache_contract(
                            contract=stored_cache_contract,
                            fold_num=fold_num,
                            tr_idx=np.asarray(fold["tr_idx"], dtype=np.int64),
                            va_idx=np.asarray(fold["va_idx"], dtype=np.int64),
                            checkpoint_kind=checkpoint_kind,
                            checkpoint_sha256=expected_checkpoint_sha,
                            oof=oof,
                            split_contract=effective_split_contract,
                            current_archive_sha256=current_archive_sha256,
                        )
                        if provenance_issues:
                            cache_status["provenance_issues"] = provenance_issues
                            audit["issues"].append(
                                f"checkpoint_local_cache_provenance_mismatch_fold{fold_num}"
                            )
                            local_cache_contracts.append(cache_status)
                            continue
                        loaded_cache = load_fold_embedding_cache(
                            path=cache_path,
                            fold_num=fold_num,
                            tr_idx=np.asarray(fold["tr_idx"], dtype=np.int64),
                            va_idx=np.asarray(fold["va_idx"], dtype=np.int64),
                            expected_contract=stored_cache_contract,
                        )
                        cache_status["valid"] = loaded_cache is not None
                        if loaded_cache is None:
                            audit["issues"].append(
                                f"checkpoint_local_cache_payload_invalid_fold{fold_num}"
                            )
                        else:
                            split_payloads, _ = loaded_cache
                            for key in EMBEDDING_KEYS:
                                if not np.array_equal(
                                    np.asarray(data[key], dtype=np.float32)[
                                        np.asarray(fold["va_idx"], dtype=np.int64)
                                    ],
                                    split_payloads["validation"]["embeddings"][key],
                                ):
                                    cache_status["valid"] = False
                                    audit["issues"].append(
                                        f"global_local_embedding_mismatch_fold{fold_num}_{key}"
                                    )
                    local_cache_contracts.append(cache_status)
            audit["checkpoint_local_cache_contracts"] = local_cache_contracts
            audit["local_fold_cache_index"] = local_cache_index

            embeddings_valid = all(
                np.asarray(data[key]).ndim == 2
                and np.asarray(data[key]).shape[0] == len(oof["record_id"])
                and np.isfinite(np.asarray(data[key], dtype=np.float32)).all()
                for key in EMBEDDING_KEYS
            )
            audit["embedding_arrays_valid"] = embeddings_valid
            if not embeddings_valid:
                audit["issues"].append("embedding_arrays_invalid")
    except Exception as exc:
        audit["issues"].append(f"{type(exc).__name__}: {exc}")
        return audit

    audit["exact_source_contract"] = bool(
        audit.get("exact_oof_sha_match") and audit.get("exact_freeze_sha_match")
    )
    if not audit["exact_source_contract"]:
        audit["issues"].append("canonical_source_sha_mismatch")
    audit["reusable"] = not audit["issues"]
    return audit


def refresh_final_embedding_contract(
    *,
    path: Path,
    oof: dict[str, Any],
    checkpoint_kind: str,
    reuse_audit: dict[str, Any],
    split_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not reuse_audit.get("exact_source_contract"):
        raise RuntimeError(
            "Representation provenance cannot be refreshed across OOF/freeze SHA changes; "
            "regenerate checkpoint-local embeddings instead."
        )
    path = resolve(path)
    with np.load(path, allow_pickle=False) as data:
        embeddings = {key: np.asarray(data[key], dtype=np.float32) for key in EMBEDDING_KEYS}
        fold_id = np.asarray(data["fold_id"], dtype=np.int16)
        slice_count = np.asarray(data["slice_count"], dtype=np.int16)
        payload = json.loads(str(data["embedding_manifest_json"].item()))

    attestation = {
        "status": "verified_semantic_repack",
        "source_oof_sha256": reuse_audit.get("source_oof_sha256"),
        "source_freeze_sha256": reuse_audit.get("source_freeze_sha256"),
        "current_oof_sha256": oof["sha256"],
        "current_freeze_sha256": str(oof["freeze_manifest_sha256"] or ""),
        "semantic_contract_match": reuse_audit.get("semantic_contract_match") is True,
        "checkpoint_contract_match": reuse_audit.get("checkpoint_contract_match") is True,
        "semantic_fields": reuse_audit.get("semantic_fields", []),
        "semantic_field_match": reuse_audit.get("semantic_field_match", {}),
    }
    payload.update(
        {
            "created_utc": now_utc(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint_kind": checkpoint_kind,
            "oof_predictions": project_relative(oof["path"]),
            "oof_predictions_sha256": oof["sha256"],
            "freeze_manifest": project_relative(oof["freeze_manifest"]),
            "freeze_manifest_sha256": oof["freeze_manifest_sha256"],
            "canonical_contract": {
                "oof_sha256": oof["sha256"],
                "freeze_sha256": oof["freeze_manifest_sha256"],
            },
            "split_contract": split_contract
            or {
                "source": "frozen_oof_fold_id",
                "fold_assignment_sha256": array_sha256(oof["fold_id"], np.int16),
            },
            "semantic_reuse_attestation": attestation,
        }
    )
    write_final_embedding_npz(
        path=path,
        oof=oof,
        embeddings=embeddings,
        fold_id=fold_id,
        slice_count=slice_count,
        payload=payload,
    )
    return attestation


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    created_utc = now_utc()
    only_folds = parse_only_folds(args.only_folds)

    oof = load_oof_contract(args.oof_predictions, args.freeze_manifest, args.limit_records)
    frozen_kind = (oof.get("freeze_payload") or {}).get("checkpoint_kind")
    if args.limit_records == 0 and frozen_kind and str(frozen_kind) != args.checkpoint_kind:
        raise RuntimeError(
            f"Freeze manifest checkpoint_kind={frozen_kind} does not match requested "
            f"--checkpoint-kind={args.checkpoint_kind}."
        )

    print("=" * 80, flush=True)
    print("ECG-RAMBA REPRESENTATION EMBEDDING EXTRACTION", flush=True)
    print("=" * 80, flush=True)
    print(f"checkpoint_kind={args.checkpoint_kind}", flush=True)
    print(f"oof_predictions={resolve(args.oof_predictions)} sha256={oof['sha256']}", flush=True)
    print(f"oof_run_manifest={resolve(args.oof_run_manifest)}", flush=True)
    print(f"only_folds={sorted(only_folds) if only_folds else 'all'}", flush=True)
    print(
        f"batch_size={args.batch_size} record_batch_size={args.record_batch_size} "
        f"num_workers={args.num_workers}",
        flush=True,
    )
    checkpoint_contracts = load_checkpoint_contracts(args.oof_run_manifest, args.checkpoint_kind)
    checkpoint_split_contract = validate_checkpoint_fold_contract(oof, checkpoint_contracts)
    print(f"checkpoint_split_contract={checkpoint_split_contract}", flush=True)
    final_reuse_audit = inspect_final_embedding_reuse(
        args.out_embedding,
        oof,
        args.checkpoint_kind,
        checkpoint_contracts,
        checkpoint_split_contract,
    )
    if not only_folds and final_reuse_audit.get("reusable"):
        semantic_reuse_attestation = None
        if not final_reuse_audit.get("exact_source_contract"):
            semantic_reuse_attestation = refresh_final_embedding_contract(
                path=args.out_embedding,
                oof=oof,
                checkpoint_kind=args.checkpoint_kind,
                reuse_audit=final_reuse_audit,
                split_contract=checkpoint_split_contract,
            )
            print(
                "Refreshed representation embedding OOF/freeze metadata after verified "
                "semantic and checkpoint-contract reuse.",
                flush=True,
            )
        manifest_path = resolve(args.out_manifest)
        existing = {}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(
            {
                "status": "complete",
                "protocol": PROTOCOL,
                "created_utc": created_utc,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "checkpoint_kind": args.checkpoint_kind,
                "oof_predictions": project_relative(resolve(args.oof_predictions)),
                "oof_predictions_sha256": oof["sha256"],
                "freeze_manifest": project_relative(resolve(args.freeze_manifest)),
                "freeze_manifest_sha256": oof["freeze_manifest_sha256"],
                "canonical_contract": {
                    "oof_sha256": oof["sha256"],
                    "freeze_sha256": oof["freeze_manifest_sha256"],
                },
                "split_contract": checkpoint_split_contract,
                "source_contract": representation_source_contract(),
                "coordinate_protocol": LOCAL_COORDINATE_PROTOCOL,
                "local_coordinate_schema_version": LOCAL_COORDINATE_SCHEMA_VERSION,
                "local_fold_cache_index": final_reuse_audit.get(
                    "local_fold_cache_index", existing.get("local_fold_cache_index", [])
                ),
                "n_records": int(len(oof["record_id"])),
                "n_classes": int(oof["y_true"].shape[1]),
                "covered_records": int(len(oof["record_id"])),
                "missing_records": 0,
                "missing_fold_caches": [],
                "outputs": {
                    "embedding_npz": project_relative(resolve(args.out_embedding)),
                    "embedding_npz_sha256": sha256_file(resolve(args.out_embedding)),
                    "manifest": project_relative(manifest_path),
                },
                "safe_wording": (
                    "Use downstream representation-probe results as suggestive branch-specific "
                    "information only; do not claim proven morphology-rhythm disentanglement."
                ),
                "reused_verified_final_embedding": True,
                "semantic_reuse_attestation": semantic_reuse_attestation
                or final_reuse_audit.get("existing_semantic_reuse_attestation")
                or existing.get("semantic_reuse_attestation"),
            }
        )
        save_json(manifest_path, jsonable(existing))
        print(f"Reusing verified final representation embedding: {resolve(args.out_embedding)}", flush=True)
        print(f"Wrote manifest: {manifest_path}", flush=True)
        return
    if final_reuse_audit.get("issues"):
        print(
            "Final representation embedding is not reusable: "
            + "; ".join(str(issue) for issue in final_reuse_audit["issues"]),
            flush=True,
        )
    gen.validate_runtime_memory(args)
    validate_mamba_runtime_for_extraction()

    from src.features import generate_hrv_cache, generate_raw_rocket_cache

    X, y, X_raw_amp, subjects = gen.prepare_clean_chapman(limit_records=args.limit_records)
    n_records, n_classes = y.shape
    if n_records != len(oof["record_id"]) or n_classes != len(CLASSES):
        raise ValueError(f"Dataset/OOF shape mismatch: data={y.shape}, oof={oof['y_true'].shape}")
    if not np.array_equal(y.astype(np.float32), oof["y_true"].astype(np.float32)):
        raise ValueError("Chapman labels do not match frozen OOF y_true.")

    dataset_record_fingerprint = record_order_fingerprint(subjects)
    dataset_record_order_sha256 = ordered_record_sha256(subjects)
    print(
        f"dataset_record_order_fingerprint={dataset_record_fingerprint} "
        f"full_sha256={dataset_record_order_sha256}",
        flush=True,
    )
    X_rocket_raw, rocket_contract = generate_raw_rocket_cache(
        X,
        subjects,
        return_contract=True,
    )
    cache_provenance = gen.build_oof_cache_provenance(
        rocket_contract=rocket_contract,
        dataset_record_fingerprint=dataset_record_fingerprint,
    )
    X_hrv = generate_hrv_cache(X, X_raw_amp, subjects) if CONFIG["use_hrv"] else np.zeros(
        (n_records, CONFIG["hrv_dim"]), dtype=np.float32
    )
    hrv_input_sha256 = array_sha256(X_hrv, np.float32)

    normalized_folds = folds_from_frozen_oof(oof)
    print(
        "Representation split source=frozen_oof_fold_id "
        f"sha256={array_sha256(oof['fold_id'], np.int16)}",
        flush=True,
    )

    global_embeddings: dict[str, np.ndarray] | None = None
    global_fold_id = np.full(n_records, -1, dtype=np.int16)
    global_slice_count = np.zeros(n_records, dtype=np.int16)
    fold_summaries: list[dict[str, Any]] = []
    local_fold_cache_index: list[dict[str, Any]] = []
    missing_fold_caches: list[int] = []

    for fold in normalized_folds:
        fold_idx = int(fold["fold_num"])
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        checkpoint_file, checkpoint_sha = resolve_checkpoint_for_fold(
            fold_num=fold_idx,
            checkpoint_kind=args.checkpoint_kind,
            checkpoint_contracts=checkpoint_contracts,
        )
        if checkpoint_sha is None:
            checkpoint_sha = sha256_file(checkpoint_file)
        checkpoint_payload, checkpoint_meta = gen.load_checkpoint_payload(checkpoint_file, args.checkpoint_kind)
        source_config_hash = checkpoint_meta["source_config_hash"]
        if args.limit_records == 0 and checkpoint_meta["dataset_record_order_fingerprint"] != dataset_record_fingerprint:
            raise RuntimeError(
                f"Fold {fold_idx} checkpoint fingerprint mismatch: "
                f"checkpoint={checkpoint_meta['dataset_record_order_fingerprint']} "
                f"data={dataset_record_fingerprint}"
            )

        pca_contract = gen.scoped_cache_contract(
            cache_provenance,
            artifact_kind="fold_train_pca",
            fold_num=fold_idx,
            tr_idx=tr_idx,
            va_idx=None,
            source_config_hash=source_config_hash,
        )
        pca_model_path = gen.fold_pca_model_path(
            fold_idx,
            tr_idx,
            source_config_hash,
            pca_contract,
        )
        verified_pca = inspect_pca_artifact(
            path=pca_model_path,
            expected_contract=pca_contract,
            expected_raw_dim=int(X_rocket_raw.shape[1]),
        )
        fold_selected = only_folds is None or fold_idx in only_folds
        if verified_pca is None and not fold_selected:
            print(
                f"Fold {fold_idx}: PCA artifact is missing/stale and fold is not selected; "
                "leaving representation cache incomplete.",
                flush=True,
            )
            missing_fold_caches.append(fold_idx)
            continue
        if verified_pca is None:
            pca, pca_identity = ensure_pca_artifact(
                path=pca_model_path,
                expected_contract=pca_contract,
                X_rocket_raw=X_rocket_raw,
                tr_idx=tr_idx,
            )
        else:
            pca, pca_identity = verified_pca

        cache_contract = build_fold_embedding_cache_contract(
            fold_num=fold_idx,
            checkpoint_kind=args.checkpoint_kind,
            checkpoint_sha256=checkpoint_sha,
            source_config_hash=source_config_hash,
            tr_idx=tr_idx,
            va_idx=va_idx,
            oof_sha256=oof["sha256"],
            freeze_sha256=str(oof["freeze_manifest_sha256"] or ""),
            split_contract=checkpoint_split_contract,
            cache_provenance=cache_provenance,
            dataset_record_fingerprint=dataset_record_fingerprint,
            dataset_record_order_sha256=dataset_record_order_sha256,
            hrv_input_sha256=hrv_input_sha256,
            pca_identity=pca_identity,
        )
        cache_path = fold_embedding_cache_path(
            fold_idx,
            args.checkpoint_kind,
            args.fold_cache_dir,
            cache_contract,
        )
        cached = None
        if args.resume_fold_cache and not args.force_rerun_folds:
            cached = load_fold_embedding_cache(
                path=cache_path,
                fold_num=fold_idx,
                tr_idx=tr_idx,
                va_idx=va_idx,
                expected_contract=cache_contract,
            )

        should_compute = cached is None and fold_selected
        if cached is None and not should_compute:
            print(
                f"Fold {fold_idx}: cache missing and not selected by --only-folds; leaving incomplete.",
                flush=True,
            )
            missing_fold_caches.append(fold_idx)
            continue

        if cached is not None:
            split_payloads, summary = cached
            train_embeddings = split_payloads["train"]["embeddings"]
            validation_embeddings = split_payloads["validation"]["embeddings"]
            train_slice_count = split_payloads["train"]["slice_count"]
            validation_slice_count = split_payloads["validation"]["slice_count"]
        else:
            print("=" * 80, flush=True)
            print(
                f"Fold {fold_idx}/{len(normalized_folds)} | "
                f"train={len(tr_idx)} val={len(va_idx)}",
                flush=True,
            )
            hydra_va, pca_var, hydra_cache_path, hydra_cache_hit = gen.load_or_compute_fold_hydra(
                fold_num=fold_idx,
                X_rocket_raw=X_rocket_raw,
                tr_idx=tr_idx,
                va_idx=va_idx,
                source_config_hash=source_config_hash,
                cache_provenance=cache_provenance,
            )
            model = gen.load_model_for_fold(
                fold_idx,
                args.checkpoint_kind,
                checkpoint_file=checkpoint_file,
                checkpoint_payload=checkpoint_payload,
            )
            train_embeddings, train_slice_count = extract_partition_embeddings(
                fold_num=fold_idx,
                split_name="train",
                model=model,
                X=X,
                X_hrv=X_hrv,
                X_rocket_raw=X_rocket_raw,
                record_indices=tr_idx,
                pca=pca,
                precomputed_hydra=None,
                record_batch_size=args.record_batch_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            validation_embeddings, validation_slice_count = extract_partition_embeddings(
                fold_num=fold_idx,
                split_name="validation",
                model=model,
                X=X,
                X_hrv=X_hrv,
                X_rocket_raw=X_rocket_raw,
                record_indices=va_idx,
                pca=pca,
                precomputed_hydra=hydra_va,
                record_batch_size=args.record_batch_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            summary = {
                "fold": fold_idx,
                "coordinate_system_id": f"fold{fold_idx}:{checkpoint_sha}",
                "coordinate_protocol": LOCAL_COORDINATE_PROTOCOL,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "train_index_sha256": str(fold["train_index_sha256"]),
                "validation_index_sha256": str(fold["validation_index_sha256"]),
                "train_slices": int(train_slice_count.sum()),
                "validation_slices": int(validation_slice_count.sum()),
                "train_slice_count_min": int(train_slice_count.min()),
                "train_slice_count_max": int(train_slice_count.max()),
                "validation_slice_count_min": int(validation_slice_count.min()),
                "validation_slice_count_max": int(validation_slice_count.max()),
                "checkpoint_file": str(checkpoint_file),
                "checkpoint_sha256": checkpoint_sha,
                "source_config_hash": source_config_hash,
                "weights_kind": checkpoint_meta.get("weights_kind"),
                "checkpoint_epoch": checkpoint_meta.get("epoch"),
                "checkpoint_selection_rule": checkpoint_meta.get("selection_rule"),
                "hydra_cache_path": str(hydra_cache_path),
                "hydra_cache_hit": bool(hydra_cache_hit),
                "pca_model_path": str(pca_model_path),
                "pca_model_sha256": pca_identity["pca_model_sha256"],
                "pca_contract_sha256": pca_identity["pca_contract_sha256"],
                "pca_contract_file_sha256": pca_identity["pca_contract_file_sha256"],
                "hydra_pca_explained_variance": float(pca_var),
                "source_bundle_sha256": representation_source_contract()["bundle_sha256"],
                "raw_input_contract_sha256": cache_provenance["contract_sha256"],
                "dataset_record_order_sha256": dataset_record_order_sha256,
                "hrv_input_sha256": hrv_input_sha256,
                "representation_cache_contract_sha256": cache_contract["contract_sha256"],
            }
            save_fold_embedding_cache(
                path=cache_path,
                fold_num=fold_idx,
                tr_idx=tr_idx,
                va_idx=va_idx,
                train_embeddings=train_embeddings,
                validation_embeddings=validation_embeddings,
                train_slice_count=train_slice_count,
                validation_slice_count=validation_slice_count,
                cache_contract=cache_contract,
                summary=summary,
            )
            del model, hydra_va
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if global_embeddings is None:
            global_embeddings = {
                key: np.zeros((n_records, arr.shape[1]), dtype=np.float32)
                for key, arr in validation_embeddings.items()
            }
        for key in EMBEDDING_KEYS:
            global_embeddings[key][va_idx] = validation_embeddings[key]
        global_fold_id[va_idx] = fold_idx
        global_slice_count[va_idx] = validation_slice_count.astype(np.int16)
        fold_summaries.append(summary)
        local_fold_cache_index.append(
            {
                "fold": fold_idx,
                "path": project_relative(cache_path),
                "sha256": sha256_file(cache_path),
                "checkpoint_sha256": checkpoint_sha,
                "coordinate_system_id": summary.get(
                    "coordinate_system_id", f"fold{fold_idx}:{checkpoint_sha}"
                ),
                "train_index_sha256": str(fold["train_index_sha256"]),
                "validation_index_sha256": str(fold["validation_index_sha256"]),
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "cache_contract_sha256": cache_contract["contract_sha256"],
                "cache_contract": cache_contract,
            }
        )

    covered = global_fold_id >= 0
    missing_records = int(np.sum(~covered))
    payload = {
        "status": "complete" if missing_records == 0 else "partial_missing_fold_caches",
        "protocol": PROTOCOL,
        "created_utc": created_utc,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint_kind": args.checkpoint_kind,
        "oof_predictions": project_relative(resolve(args.oof_predictions)),
        "oof_predictions_sha256": oof["sha256"],
        "freeze_manifest": project_relative(resolve(args.freeze_manifest)),
        "freeze_manifest_sha256": oof["freeze_manifest_sha256"],
        "canonical_contract": {
            "oof_sha256": oof["sha256"],
            "freeze_sha256": oof["freeze_manifest_sha256"],
        },
        "dataset_record_order_fingerprint": dataset_record_fingerprint,
        "dataset_record_order_sha256": dataset_record_order_sha256,
        "hrv_input_sha256": hrv_input_sha256,
        "raw_input_contract": cache_provenance,
        "split_contract": checkpoint_split_contract,
        "evaluation_config_hash": EVALUATION_CONFIG_HASH,
        "source_contract": representation_source_contract(),
        "coordinate_protocol": LOCAL_COORDINATE_PROTOCOL,
        "local_coordinate_schema_version": LOCAL_COORDINATE_SCHEMA_VERSION,
        "local_fold_cache_index": local_fold_cache_index,
        "embedding_views": EMBEDDING_KEYS,
        "slice_embedding_aggregation": "arithmetic_mean_over_record_slices_within_checkpoint",
        "n_records": int(n_records),
        "n_classes": int(n_classes),
        "covered_records": int(np.sum(covered)),
        "missing_records": missing_records,
        "missing_fold_caches": missing_fold_caches,
        "fold_summaries": fold_summaries,
        "outputs": {
            "embedding_npz": project_relative(resolve(args.out_embedding)) if missing_records == 0 else None,
            "manifest": project_relative(resolve(args.out_manifest)),
        },
        "safe_wording": (
            "Use downstream representation-probe results as suggestive branch-specific "
            "information only; do not claim proven morphology-rhythm disentanglement."
        ),
        "runtime": jsonable(gen.runtime_metadata(args, created_utc)),
    }

    if missing_records == 0:
        if global_embeddings is None:
            raise RuntimeError("No embeddings were assembled.")
        for key, arr in global_embeddings.items():
            if arr.shape[0] != n_records or not np.isfinite(arr).all():
                raise RuntimeError(f"Invalid final embedding array {key}: {arr.shape}")
        write_final_embedding_npz(
            path=args.out_embedding,
            oof=oof,
            embeddings=global_embeddings,
            fold_id=global_fold_id,
            slice_count=global_slice_count,
            payload=payload,
        )
        payload["outputs"]["embedding_npz_sha256"] = sha256_file(
            resolve(args.out_embedding)
        )
    else:
        print(
            f"Partial extraction only: missing_records={missing_records}. "
            "Run remaining folds or restore fold caches before running representation probe.",
            flush=True,
        )

    save_json(resolve(args.out_manifest), jsonable(payload))
    print(json.dumps(jsonable({"status": payload["status"], "missing_records": missing_records}), indent=2), flush=True)
    print(f"Wrote manifest: {resolve(args.out_manifest)}", flush=True)


if __name__ == "__main__":
    main()
