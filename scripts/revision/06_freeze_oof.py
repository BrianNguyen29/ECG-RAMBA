"""Validate and freeze the canonical Chapman OOF prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
    CLASSES,
    EVALUATION_CONFIG_HASH,
    PATHS,
)
from scripts.revision.common import (  # noqa: E402
    AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    CACHE_SCHEMA_VERSION,
    CHAPMAN_GROUP_REFERENCE,
    CHAPMAN_GROUP_REFERENCE_COUNTS,
    CHAPMAN_GROUP_SEMANTICS,
    LOG_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    POWER_MEAN_IMPLEMENTATION,
    PREDICTION_DIR,
    TABLE_DIR,
    aggregate_record_probabilities,
    ensure_revision_dirs,
    git_commit,
    npz_scalar,
    save_json,
    sha256_file,
)


DEFAULT_RECORD = PREDICTION_DIR / "oof_full_predictions.npz"
DEFAULT_SLICE = PREDICTION_DIR / "oof_full_slice_predictions.npz"
DEFAULT_SUMMARY = METRIC_DIR / "oof_full_prediction_summary.json"
DEFAULT_CLASS_TABLE = TABLE_DIR / "oof_full_class_summary.csv"
DEFAULT_RUN_MANIFEST = MANIFEST_DIR / "oof_full_prediction_run_manifest.json"
DEFAULT_FREEZE_MANIFEST = MANIFEST_DIR / "oof_freeze_manifest.json"
SUBJECT_GROUP_SEMANTICS = CHAPMAN_GROUP_SEMANTICS
SUBJECT_GROUP_REFERENCE = CHAPMAN_GROUP_REFERENCE
SUBJECT_GROUP_REFERENCE_COUNTS = CHAPMAN_GROUP_REFERENCE_COUNTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-file", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--slice-file", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--class-table", type=Path, default=DEFAULT_CLASS_TABLE)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--expected-records", type=int, default=44186)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--q", type=float, default=3.0)
    parser.add_argument(
        "--expected-checkpoint-kind",
        default="final_ema",
        help=(
            "Checkpoint kind allowed for manuscript-ready OOF. The default "
            "uses the pre-specified final EMA epoch and rejects validation-selected best checkpoints."
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--check-existing-freeze",
        action="store_true",
        help=(
            "Validate the current OOF/checkpoint contract against the existing freeze manifest "
            "without requiring a runtime-local generation log or rewriting the freeze manifest."
        ),
    )
    parser.add_argument("--allow-missing-log", action="store_true")
    parser.add_argument(
        "--metadata-refresh-from-existing-oof",
        action="store_true",
        help=(
            "Permit a metadata-only strict refreeze when the original generation log is unavailable, "
            "but only after the prior frozen manifest and prediction run manifest independently bind "
            "the exact current OOF prediction SHA256. No predictions are regenerated."
        ),
    )
    parser.add_argument(
        "--manuscript-ready-strict",
        action="store_true",
        help=(
            "Require authenticated folds.pkl/checkpoint split membership and a SHA-bound "
            "one-record-per-group sidecar before declaring the OOF freeze manuscript-ready."
        ),
    )
    parser.add_argument(
        "--folds-file",
        type=Path,
        default=None,
        help="Canonical folds.pkl. Defaults to <model-dir>/folds.pkl in strict mode.",
    )
    parser.add_argument(
        "--group-sidecar",
        type=Path,
        default=None,
        help=(
            "NPZ/JSON sidecar with record_id, group_id (or subject_id), "
            "dataset_record_order_fingerprint, record_file_sha256, and the reviewed "
            "PhysioNet one-patient-per-record semantics contract."
        ),
    )
    parser.add_argument(
        "--group-sidecar-sha256",
        default=None,
        help="Optional expected SHA256 for the group sidecar.",
    )
    return parser.parse_args()


def checkpoint_fold(row: dict) -> int | None:
    if row.get("fold") is not None:
        return int(row["fold"])
    match = re.search(r"fold(\d+)", Path(str(row.get("path", ""))).name)
    return int(match.group(1)) if match else None


def normalize_checkpoint_rows(rows: list[dict]) -> dict[int, dict]:
    normalized = {}
    for row in rows:
        fold = checkpoint_fold(row)
        sha = row.get("sha256")
        if fold is None or not sha:
            continue
        normalized[fold] = {
            "fold": fold,
            "path": str(row.get("path", "")),
            "sha256": str(sha),
            "size_bytes": int(row.get("size_bytes", 0)),
        }
    return normalized


def checkpoint_rows_from_manifest(payload: dict) -> list[dict]:
    candidates = [
        payload.get("source_checkpoints"),
        payload.get("checkpoints"),
        payload.get("inputs", {}).get("checkpoints"),
    ]
    for rows in candidates:
        if isinstance(rows, list) and rows:
            return rows
    return []


def current_checkpoint_rows(kind: str, expected_folds: int) -> list[dict]:
    model_dir = Path(PATHS["model_dir"])
    rows = []
    for fold in range(1, expected_folds + 1):
        path = model_dir / f"fold{fold}_{kind}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing exact checkpoint for fold {fold}: {path}")
        rows.append(
            {
                "fold": fold,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def artifact_info(path: Path) -> dict:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        display_path = resolved.as_posix()
    return {
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


STRICT_BLOCKER = "Manuscript-ready OOF freeze blocker"


def strict_blocker(message: str) -> RuntimeError:
    return RuntimeError(f"{STRICT_BLOCKER}: {message}")


def index_fingerprint(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]


def assignment_fingerprint(fold_id: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(fold_id, dtype=np.int16))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def integral_vector(value: np.ndarray, *, name: str, expected_length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},), found {raw.shape}")
    if raw.dtype.kind not in "iufb":
        raise ValueError(f"{name} must be numeric integer identifiers")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} contains non-finite identifiers")
    if not np.equal(raw, np.floor(raw)).all():
        raise ValueError(f"{name} contains non-integer identifiers")
    return raw.astype(np.int64, copy=False)


def validate_record_arrays(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    record_id_raw: np.ndarray,
    fold_id_raw: np.ndarray,
    expected_records: int,
    expected_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    if y_true.shape != (expected_records, len(CLASSES)):
        raise ValueError(f"Unexpected y_true shape: {y_true.shape}")
    if y_prob.shape != y_true.shape:
        raise ValueError(f"Prediction shape mismatch: {y_prob.shape} vs {y_true.shape}")
    try:
        labels_finite = bool(np.isfinite(y_true).all())
    except TypeError as exc:
        raise ValueError("OOF y_true must be numeric, finite, and binary") from exc
    if not labels_finite:
        raise ValueError("OOF y_true contains non-finite values")
    if not np.logical_or(y_true == 0, y_true == 1).all():
        raise ValueError("OOF y_true must contain only binary values {0, 1}")
    try:
        probabilities_finite = bool(np.isfinite(y_prob).all())
    except TypeError as exc:
        raise ValueError("OOF record probabilities must be numeric and finite") from exc
    if not probabilities_finite or np.min(y_prob) < 0 or np.max(y_prob) > 1:
        raise ValueError("OOF record probabilities are not finite values in [0, 1]")

    record_id = integral_vector(
        record_id_raw,
        name="record_id",
        expected_length=expected_records,
    )
    if len(np.unique(record_id)) != expected_records:
        raise ValueError("record_id values must be unique")
    if not np.array_equal(record_id, np.arange(expected_records, dtype=np.int64)):
        raise ValueError("record_id must be exactly 0..N-1")

    fold_id = integral_vector(
        fold_id_raw,
        name="fold_id",
        expected_length=expected_records,
    )
    expected_fold_ids = set(range(1, expected_folds + 1))
    actual_fold_ids = set(int(x) for x in np.unique(fold_id))
    if actual_fold_ids != expected_fold_ids:
        raise ValueError(f"OOF folds mismatch: {sorted(actual_fold_ids)}")
    return record_id, fold_id.astype(np.int16, copy=False)


def validate_slice_arrays(
    *,
    slice_prob: np.ndarray,
    slice_record_id_raw: np.ndarray,
    slice_fold_id_raw: np.ndarray,
    record_id: np.ndarray,
    record_fold_id: np.ndarray,
    expected_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    if slice_prob.dtype != np.float32:
        raise ValueError(f"Slice probabilities must be float32, found {slice_prob.dtype}")
    if slice_prob.ndim != 2 or slice_prob.shape[1] != len(CLASSES):
        raise ValueError(f"Unexpected slice probability shape: {slice_prob.shape}")
    if len(slice_prob) == 0:
        raise ValueError("Slice artifact contains no prediction rows")
    if not np.isfinite(slice_prob).all() or np.min(slice_prob) < 0 or np.max(slice_prob) > 1:
        raise ValueError("Slice probabilities are not finite values in [0, 1]")

    n_slices = len(slice_prob)
    slice_record_id = integral_vector(
        slice_record_id_raw,
        name="slice record_id",
        expected_length=n_slices,
    )
    slice_fold_id = integral_vector(
        slice_fold_id_raw,
        name="slice fold_id",
        expected_length=n_slices,
    )
    valid_parent_ids = set(int(x) for x in record_id)
    observed_parent_ids = set(int(x) for x in np.unique(slice_record_id))
    invalid_parent_ids = sorted(observed_parent_ids - valid_parent_ids)
    if invalid_parent_ids:
        raise ValueError(
            "Slice artifact contains record_id values without a record-level parent: "
            f"{invalid_parent_ids[:10]}"
        )
    if observed_parent_ids != valid_parent_ids:
        missing = sorted(valid_parent_ids - observed_parent_ids)
        raise ValueError(f"Slice artifact does not cover all OOF record parents: {missing[:10]}")

    expected_fold_ids = set(range(1, expected_folds + 1))
    if set(int(x) for x in np.unique(slice_fold_id)) != expected_fold_ids:
        raise ValueError("Slice artifact does not contain all expected folds")
    expected_slice_fold = record_fold_id[slice_record_id]
    mismatch = np.flatnonzero(slice_fold_id != expected_slice_fold)
    if len(mismatch):
        first = int(mismatch[0])
        raise ValueError(
            "Slice fold_id does not match its parent record fold_id: "
            f"slice={first}, record_id={int(slice_record_id[first])}, "
            f"slice_fold={int(slice_fold_id[first])}, "
            f"record_fold={int(expected_slice_fold[first])}"
        )
    return slice_record_id, slice_fold_id.astype(np.int16, copy=False)


def resolve_contract_path(path: Path | None, default: Path) -> Path:
    resolved = path if path is not None else default
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def normalize_fold_splits(
    folds: object,
    *,
    expected_folds: int,
    expected_records: int,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    if not isinstance(folds, (list, tuple)) or len(folds) != expected_folds:
        raise strict_blocker(
            f"folds.pkl must contain exactly {expected_folds} folds"
        )
    normalized = []
    validation_coverage = np.zeros(expected_records, dtype=np.int16)
    for fold_num, split in enumerate(folds, start=1):
        if not isinstance(split, dict) or "tr_idx" not in split or "va_idx" not in split:
            raise strict_blocker(f"fold {fold_num} lacks tr_idx/va_idx")
        train_idx = integral_vector(
            np.asarray(split["tr_idx"]),
            name=f"fold {fold_num} tr_idx",
            expected_length=len(split["tr_idx"]),
        )
        val_idx = integral_vector(
            np.asarray(split["va_idx"]),
            name=f"fold {fold_num} va_idx",
            expected_length=len(split["va_idx"]),
        )
        if len(np.unique(train_idx)) != len(train_idx):
            raise strict_blocker(f"fold {fold_num} training indices contain duplicates")
        if len(np.unique(val_idx)) != len(val_idx):
            raise strict_blocker(f"fold {fold_num} validation indices contain duplicates")
        if (
            np.any(train_idx < 0)
            or np.any(train_idx >= expected_records)
            or np.any(val_idx < 0)
            or np.any(val_idx >= expected_records)
        ):
            raise strict_blocker(f"fold {fold_num} contains out-of-range record indices")
        if len(np.intersect1d(train_idx, val_idx)):
            raise strict_blocker(f"fold {fold_num} has train/validation record overlap")
        if len(np.union1d(train_idx, val_idx)) != expected_records:
            raise strict_blocker(
                f"fold {fold_num} train/validation indices do not partition all records"
            )
        validation_coverage[val_idx] += 1
        normalized.append({"tr_idx": train_idx, "va_idx": val_idx})
    if not np.all(validation_coverage == 1):
        raise strict_blocker(
            "fold validation memberships must cover every record exactly once"
        )
    return normalized, validation_coverage


def split_metadata_from_row(row: dict) -> tuple[dict | None, str | None]:
    candidates = [
        (row.get("split"), "run_manifest_checkpoint_row"),
        ((row.get("metadata") or {}).get("split"), "run_manifest_checkpoint_metadata"),
        (
            (row.get("checkpoint_metadata") or {}).get("split"),
            "run_manifest_checkpoint_metadata",
        ),
    ]
    for candidate, source in candidates:
        if isinstance(candidate, dict):
            return candidate, source
    return None, None


def checkpoint_membership_metadata(row: dict) -> tuple[int | None, dict, str | None, str]:
    inline_split, inline_source = split_metadata_from_row(row)
    if inline_split is not None:
        return (
            checkpoint_fold(row),
            inline_split,
            row.get("dataset_record_order_fingerprint"),
            str(inline_source),
        )

    raw_path = str(row.get("path") or "")
    path = Path(raw_path) if raw_path else None
    if path is not None and not path.is_absolute():
        model_candidate = Path(PATHS["model_dir"]) / path.name
        project_candidate = PROJECT_ROOT / path
        path = model_candidate if model_candidate.exists() else project_candidate
    if path is None or not path.is_file():
        raise strict_blocker(
            f"checkpoint split metadata is unavailable for fold {checkpoint_fold(row)}"
        )
    try:
        import torch

        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except (TypeError, RuntimeError):
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise strict_blocker(
            f"cannot read checkpoint split metadata from {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("split"), dict):
        raise strict_blocker(f"checkpoint lacks split metadata: {path}")
    return (
        int(payload["fold"]) if payload.get("fold") is not None else None,
        dict(payload["split"]),
        payload.get("dataset_record_order_fingerprint"),
        "checkpoint_payload",
    )


def validate_membership_contract(
    *,
    args: argparse.Namespace,
    record_fold_id: np.ndarray,
    dataset_record_order_fingerprint: str,
    source_checkpoint_rows: list[dict],
    current_checkpoint_rows_list: list[dict],
) -> tuple[dict, Path]:
    try:
        import joblib
    except ImportError as exc:
        raise strict_blocker("joblib is required to authenticate folds.pkl") from exc

    folds_path = resolve_contract_path(
        getattr(args, "folds_file", None),
        Path(PATHS["model_dir"]) / "folds.pkl",
    )
    if not folds_path.is_file() or folds_path.stat().st_size == 0:
        raise strict_blocker(f"canonical folds.pkl is missing: {folds_path}")
    try:
        folds, _ = normalize_fold_splits(
            joblib.load(folds_path),
            expected_folds=args.expected_folds,
            expected_records=args.expected_records,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise strict_blocker(
            f"cannot load canonical folds.pkl {folds_path}: {type(exc).__name__}: {exc}"
        ) from exc

    expected_assignment = np.zeros(args.expected_records, dtype=np.int16)
    for fold_num, split in enumerate(folds, start=1):
        expected_assignment[split["va_idx"]] = fold_num
    mismatch = np.flatnonzero(expected_assignment != record_fold_id)
    if len(mismatch):
        raise strict_blocker(
            "OOF record fold assignment differs from folds.pkl: "
            f"mismatched_records={len(mismatch)}, first_record={int(mismatch[0])}"
        )

    source_by_fold = {
        checkpoint_fold(row): row
        for row in source_checkpoint_rows
        if checkpoint_fold(row) is not None
    }
    current_by_fold = {
        checkpoint_fold(row): row
        for row in current_checkpoint_rows_list
        if checkpoint_fold(row) is not None
    }
    fold_contracts = []
    for fold_num, split in enumerate(folds, start=1):
        expected_split = {
            "train_count": int(len(split["tr_idx"])),
            "val_count": int(len(split["va_idx"])),
            "train_index_hash": index_fingerprint(split["tr_idx"]),
            "val_index_hash": index_fingerprint(split["va_idx"]),
        }
        source_row = source_by_fold.get(fold_num, {})
        current_row = current_by_fold.get(fold_num, {})
        # Prefer the currently SHA-authenticated checkpoint. Tests and migrated
        # manifests may carry the same checkpoint metadata inline, but real
        # manuscript runs read the split contract from the checkpoint payload.
        metadata_row = current_row or source_row
        metadata_fold, observed_split, observed_fingerprint, metadata_source = (
            checkpoint_membership_metadata(metadata_row)
        )
        if metadata_fold != fold_num:
            raise strict_blocker(
                f"checkpoint fold metadata mismatch: expected fold {fold_num}, found {metadata_fold}"
            )
        mismatched_fields = [
            key
            for key, expected in expected_split.items()
            if observed_split.get(key) != expected
        ]
        if mismatched_fields:
            raise strict_blocker(
                f"checkpoint fold {fold_num} split metadata differs from folds.pkl: "
                + ", ".join(mismatched_fields)
            )
        if str(observed_fingerprint or "") != dataset_record_order_fingerprint:
            raise strict_blocker(
                f"checkpoint fold {fold_num} has a missing/stale dataset record-order fingerprint"
            )
        fold_contracts.append(
            {
                "fold": fold_num,
                **expected_split,
                "checkpoint_sha256": str(current_row.get("sha256") or source_row.get("sha256")),
                "metadata_source": metadata_source,
            }
        )
    return (
        {
            "status": "verified",
            "folds_file": artifact_info(folds_path),
            "record_fold_assignment_sha256": assignment_fingerprint(record_fold_id),
            "validation_coverage": "exactly_once",
            "checkpoint_split_metadata_verified": True,
            "folds": fold_contracts,
        },
        folds_path,
    )


def load_group_sidecar(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            group_field = "group_id" if "group_id" in data.files else "subject_id"
            required = {"record_id", group_field}
            if group_field not in data.files or not required.issubset(data.files):
                raise ValueError("group sidecar NPZ must contain record_id and group_id/subject_id")
            return {
                "record_id": np.asarray(data["record_id"]),
                "group_id": np.asarray(data[group_field]),
                "group_field": group_field,
                "group_unit": str(npz_scalar(data, "group_unit", "")),
                "group_semantics": str(npz_scalar(data, "group_semantics", "")),
                "group_semantics_reference": str(
                    npz_scalar(data, "group_semantics_reference", "")
                ),
                "source_patient_record_counts_json": str(
                    npz_scalar(data, "source_patient_record_counts_json", "")
                ),
                "one_record_per_group": bool(
                    npz_scalar(data, "one_record_per_group", False)
                ),
                "dataset_record_order_fingerprint": str(
                    npz_scalar(data, "dataset_record_order_fingerprint", "")
                ),
                "record_file_sha256": str(npz_scalar(data, "record_file_sha256", "")),
                "source_archive_sha256": str(
                    npz_scalar(data, "source_archive_sha256", "")
                ),
            }
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        group_field = "group_id" if "group_id" in payload else "subject_id"
        if "record_id" not in payload or group_field not in payload:
            raise ValueError("group sidecar JSON must contain record_id and group_id/subject_id")
        return {
            "record_id": np.asarray(payload["record_id"]),
            "group_id": np.asarray(payload[group_field]),
            "group_field": group_field,
            "group_unit": str(payload.get("group_unit", "")),
            "group_semantics": str(payload.get("group_semantics", "")),
            "group_semantics_reference": str(
                payload.get("group_semantics_reference", "")
            ),
            "source_patient_record_counts_json": (
                json.dumps(payload.get("source_patient_record_counts"), sort_keys=True)
                if isinstance(payload.get("source_patient_record_counts"), dict)
                else str(payload.get("source_patient_record_counts_json", ""))
            ),
            "one_record_per_group": bool(payload.get("one_record_per_group", False)),
            "dataset_record_order_fingerprint": str(
                payload.get("dataset_record_order_fingerprint", "")
            ),
            "record_file_sha256": str(payload.get("record_file_sha256", "")),
            "source_archive_sha256": str(payload.get("source_archive_sha256", "")),
        }
    raise ValueError("group sidecar must be an NPZ or JSON file")


def validate_group_contract(
    *,
    sidecar_path: Path,
    expected_sidecar_sha256: str | None,
    record_file: Path,
    record_id: np.ndarray,
    dataset_record_order_fingerprint: str,
) -> dict:
    sidecar_path = resolve_contract_path(sidecar_path, sidecar_path)
    if not sidecar_path.is_file() or sidecar_path.stat().st_size == 0:
        raise strict_blocker(f"group sidecar is missing: {sidecar_path}")
    actual_sidecar_sha256 = sha256_file(sidecar_path)
    if expected_sidecar_sha256 and actual_sidecar_sha256 != expected_sidecar_sha256:
        raise strict_blocker(
            "group sidecar SHA256 differs from --group-sidecar-sha256; sidecar is stale"
        )
    try:
        sidecar = load_group_sidecar(sidecar_path)
    except Exception as exc:
        raise strict_blocker(
            f"cannot read group sidecar {sidecar_path}: {type(exc).__name__}: {exc}"
        ) from exc

    sidecar_record_id = integral_vector(
        sidecar["record_id"],
        name="group sidecar record_id",
        expected_length=len(record_id),
    )
    if not np.array_equal(sidecar_record_id, record_id):
        raise strict_blocker("group sidecar record_id order differs from the OOF artifact")
    groups = np.asarray(sidecar["group_id"])
    if groups.shape != (len(record_id),):
        raise strict_blocker(
            f"group sidecar group_id shape mismatch: {groups.shape}"
        )
    if groups.dtype.kind in "fiu":
        if not np.isfinite(groups).all():
            raise strict_blocker("group sidecar contains non-finite group IDs")
        normalized_groups = groups.astype(str)
    else:
        normalized_groups = groups.astype(str)
        if any(not value.strip() or value.strip().lower() in {"none", "nan"} for value in normalized_groups):
            raise strict_blocker("group sidecar contains empty group IDs")
    n_groups = int(len(np.unique(normalized_groups)))
    if n_groups != len(record_id):
        raise strict_blocker(
            "one-record-per-group independence is not verified: "
            f"records={len(record_id)}, unique_groups={n_groups}"
        )
    if not sidecar["one_record_per_group"]:
        raise strict_blocker("group sidecar does not attest one_record_per_group=true")
    if sidecar["group_semantics"] != SUBJECT_GROUP_SEMANTICS:
        raise strict_blocker(
            "group sidecar has missing or unreviewed patient-record semantics: "
            f"observed={sidecar['group_semantics']!r}"
        )
    if sidecar["group_semantics_reference"] != SUBJECT_GROUP_REFERENCE:
        raise strict_blocker(
            "group sidecar patient-record semantics are not bound to the reviewed PhysioNet source"
        )
    try:
        source_counts = json.loads(sidecar["source_patient_record_counts_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise strict_blocker(
            "group sidecar source patient/record counts are missing or invalid JSON"
        ) from exc
    if source_counts != SUBJECT_GROUP_REFERENCE_COUNTS:
        raise strict_blocker(
            "group sidecar source patient/record counts differ from the reviewed corpus contract: "
            f"observed={source_counts!r}"
        )
    if sidecar["dataset_record_order_fingerprint"] != dataset_record_order_fingerprint:
        raise strict_blocker("group sidecar has a stale dataset record-order fingerprint")
    actual_record_sha256 = sha256_file(record_file)
    if sidecar["record_file_sha256"] != actual_record_sha256:
        raise strict_blocker("group sidecar record_file_sha256 does not authenticate the OOF artifact")
    source_archive = Path(PATHS["zip_path"])
    if not source_archive.is_file() or source_archive.stat().st_size == 0:
        raise strict_blocker(f"Chapman source archive is missing for group-contract authentication: {source_archive}")
    actual_archive_sha256 = sha256_file(source_archive)
    if not sidecar["source_archive_sha256"]:
        raise strict_blocker("group sidecar source_archive_sha256 is missing")
    if sidecar["source_archive_sha256"] != actual_archive_sha256:
        raise strict_blocker("group sidecar source_archive_sha256 differs from the active Chapman archive")
    return {
        "status": "verified",
        "sidecar": artifact_info(sidecar_path),
        "group_field": sidecar["group_field"],
        "record_file_sha256": actual_record_sha256,
        "dataset_record_order_fingerprint": dataset_record_order_fingerprint,
        "n_records": int(len(record_id)),
        "n_groups": n_groups,
        "one_record_per_group": True,
        "group_unit": sidecar["group_unit"],
        "group_semantics": SUBJECT_GROUP_SEMANTICS,
        "group_semantics_reference": SUBJECT_GROUP_REFERENCE,
        "source_patient_record_counts": source_counts,
        "source_archive": artifact_info(source_archive),
        "bootstrap_unit": AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    }


def validate_metadata_refresh_provenance(args: argparse.Namespace) -> dict:
    """Authenticate an unchanged OOF artifact for a metadata-only refreeze."""

    if not args.freeze_manifest.is_file() or args.freeze_manifest.stat().st_size == 0:
        raise FileNotFoundError(
            "Metadata-only refresh requires a non-empty prior freeze manifest"
        )
    prior_freeze_sha = sha256_file(args.freeze_manifest)
    prior = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if prior.get("status") != "frozen" or prior.get("manuscript_ready") is not True:
        raise RuntimeError(
            "Metadata-only refresh requires a prior frozen, manuscript-ready OOF manifest"
        )
    record_sha = sha256_file(args.record_file)
    prior_record_rows = [
        row
        for row in prior.get("artifacts", [])
        if Path(str(row.get("path", ""))).name == args.record_file.name
    ]
    if len(prior_record_rows) != 1 or prior_record_rows[0].get("sha256") != record_sha:
        raise RuntimeError(
            "Prior freeze manifest does not authenticate the exact current OOF prediction SHA256"
        )
    current_run_manifest_sha = sha256_file(args.run_manifest)
    prior_run_manifest_rows = [
        row
        for row in prior.get("artifacts", [])
        if Path(str(row.get("path", ""))).name == args.run_manifest.name
    ]
    if (
        len(prior_run_manifest_rows) != 1
        or prior_run_manifest_rows[0].get("sha256") != current_run_manifest_sha
    ):
        raise RuntimeError(
            "Prior freeze manifest does not authenticate the current OOF prediction run manifest"
        )
    run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    run_prediction = (run_manifest.get("outputs") or {}).get("prediction_file") or {}
    if run_prediction.get("sha256") != record_sha:
        raise RuntimeError(
            "OOF prediction run manifest does not authenticate the exact current OOF prediction SHA256"
        )
    if int(run_prediction.get("size_bytes", -1)) != args.record_file.stat().st_size:
        raise RuntimeError("OOF prediction run manifest size differs from the current OOF artifact")
    return {
        "status": "verified_metadata_only_refresh",
        "prediction_values_changed": False,
        "record_file_sha256": record_sha,
        "prior_freeze_manifest_sha256": prior_freeze_sha,
        "run_manifest_sha256": current_run_manifest_sha,
        "reason": "group/split provenance metadata refresh with unchanged SHA-bound OOF predictions",
    }


def validate_existing_freeze(args: argparse.Namespace, current: dict) -> dict:
    if not args.freeze_manifest.exists() or args.freeze_manifest.stat().st_size == 0:
        raise FileNotFoundError(f"Existing freeze manifest is missing: {args.freeze_manifest}")
    frozen = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    stable_fields = (
        "schema_version",
        "status",
        "manuscript_ready",
        "claim_boundary",
        "dataset",
        "expected_records",
        "validated_records",
        "n_classes",
        "class_names",
        "expected_folds",
        "fold_counts",
        "slice_count",
        "slice_count_min",
        "slice_count_max",
        "aggregation",
        "source_config_hash",
        "dataset_record_order_fingerprint",
        "evaluation_config_hash",
        "current_evaluation_config_hash",
        "checkpoint_kind",
        "checkpoint_fingerprints_match",
    )
    mismatched_fields = [field for field in stable_fields if frozen.get(field) != current.get(field)]
    if mismatched_fields:
        raise RuntimeError(
            "Existing freeze manifest differs from the current OOF contract: "
            + ", ".join(mismatched_fields)
        )

    if current.get("strict_manuscript_contract"):
        strict_fields = (
            "strict_manuscript_contract",
            "membership_contract",
            "group_contract",
        )
        strict_mismatches = [
            field for field in strict_fields if frozen.get(field) != current.get(field)
        ]
        if strict_mismatches:
            raise RuntimeError(
                "Existing freeze manifest differs from the current strict OOF contract: "
                + ", ".join(strict_mismatches)
            )

    for key in ("source_checkpoints", "current_checkpoints"):
        frozen_rows = normalize_checkpoint_rows(frozen.get(key) or [])
        current_rows = normalize_checkpoint_rows(current.get(key) or [])
        if {
            fold: row["sha256"] for fold, row in frozen_rows.items()
        } != {
            fold: row["sha256"] for fold, row in current_rows.items()
        }:
            raise RuntimeError(f"Existing freeze manifest has stale {key} SHA256 evidence")

    frozen_artifacts = {
        str(row.get("path", "")).replace("\\", "/"): row
        for row in frozen.get("artifacts", [])
        if isinstance(row, dict) and row.get("path")
    }
    for path in (
        args.record_file,
        args.slice_file,
        args.summary_file,
        args.class_table,
        args.run_manifest,
    ):
        actual = artifact_info(path)
        row = frozen_artifacts.get(actual["path"])
        if (
            row is None
            or int(row.get("size_bytes", -1)) != actual["size_bytes"]
            or row.get("sha256") != actual["sha256"]
        ):
            raise RuntimeError(
                f"Existing freeze manifest does not authenticate current artifact: {actual['path']}"
            )
    return frozen


def validate_oof(args: argparse.Namespace) -> dict:
    strict_manuscript_contract = bool(
        getattr(args, "manuscript_ready_strict", False)
    )
    group_sidecar = getattr(args, "group_sidecar", None)
    group_sidecar_sha256 = getattr(args, "group_sidecar_sha256", None)
    if strict_manuscript_contract and group_sidecar is None:
        raise strict_blocker(
            "--group-sidecar is required to verify one-record-per-group independence"
        )
    if group_sidecar_sha256 and group_sidecar is None:
        raise ValueError("--group-sidecar-sha256 requires --group-sidecar")

    required = [
        args.record_file,
        args.slice_file,
        args.summary_file,
        args.class_table,
        args.run_manifest,
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing/empty OOF artifacts: {missing}")

    summary = json.loads(args.summary_file.read_text(encoding="utf-8"))
    run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    if summary.get("dataset") != "chapman_oof":
        raise ValueError(f"Unexpected summary dataset: {summary.get('dataset')}")
    if int(summary.get("n_records", -1)) != args.expected_records:
        raise ValueError(f"Unexpected summary record count: {summary.get('n_records')}")
    if int(summary.get("n_valid_predictions", -1)) != args.expected_records:
        raise ValueError("Summary does not report complete OOF prediction coverage")
    with args.class_table.open(newline="", encoding="utf-8") as f:
        class_rows = list(csv.DictReader(f))
    class_column = [str(row.get("class_name", "")) for row in class_rows]
    if class_column != CLASSES:
        raise ValueError("OOF class summary rows differ from configs.config.CLASSES")
    if run_manifest.get("dataset") not in {None, "chapman_oof"}:
        raise ValueError(f"Unexpected run manifest dataset: {run_manifest.get('dataset')}")
    with np.load(args.record_file, allow_pickle=False) as record_data, np.load(
        args.slice_file,
        allow_pickle=False,
    ) as slice_data:
        required_record = {
            "y_true",
            "y_prob",
            "record_id",
            "class_names",
            "fold_id",
            "aggregation_q",
            "aggregation_implementation",
            "cache_schema_version",
            "dataset_record_order_fingerprint",
        }
        required_slice = {"slice_prob", "record_id", "fold_id", "class_names"}
        if not required_record.issubset(record_data.files):
            raise ValueError(f"Record artifact lacks: {sorted(required_record - set(record_data.files))}")
        if not required_slice.issubset(slice_data.files):
            raise ValueError(f"Slice artifact lacks: {sorted(required_slice - set(slice_data.files))}")

        y_true = np.asarray(record_data["y_true"])
        y_prob = np.asarray(record_data["y_prob"])
        record_id_raw = np.asarray(record_data["record_id"])
        fold_id_raw = np.asarray(record_data["fold_id"])
        class_names = np.asarray(record_data["class_names"]).astype(str)
        valid_mask = (
            np.asarray(record_data["valid_record_mask"], dtype=bool)
            if "valid_record_mask" in record_data.files
            else np.ones(args.expected_records, dtype=bool)
        )
        if valid_mask.shape != (args.expected_records,):
            raise ValueError(
                "valid_record_mask must have shape "
                f"({args.expected_records},), found {valid_mask.shape}"
            )
        record_id, fold_id = validate_record_arrays(
            y_true=y_true,
            y_prob=y_prob,
            record_id_raw=record_id_raw,
            fold_id_raw=fold_id_raw,
            expected_records=args.expected_records,
            expected_folds=args.expected_folds,
        )
        if not np.array_equal(class_names, np.asarray(CLASSES)):
            raise ValueError("OOF class order differs from configs.config.CLASSES")
        if not np.all(valid_mask):
            raise ValueError(f"{int(np.sum(~valid_mask))} OOF records are missing predictions")

        implementation = str(npz_scalar(record_data, "aggregation_implementation", ""))
        schema_version = int(npz_scalar(record_data, "cache_schema_version", 0))
        aggregation_q = float(npz_scalar(record_data, "aggregation_q", np.nan))
        if implementation != POWER_MEAN_IMPLEMENTATION:
            raise ValueError(f"Unexpected aggregation implementation: {implementation}")
        if schema_version < CACHE_SCHEMA_VERSION:
            raise ValueError(f"Legacy cache schema: {schema_version}")
        if not np.isclose(aggregation_q, args.q):
            raise ValueError(f"Unexpected aggregation q={aggregation_q}")

        slice_prob = np.asarray(slice_data["slice_prob"])
        slice_record_id_raw = np.asarray(slice_data["record_id"])
        slice_fold_id_raw = np.asarray(slice_data["fold_id"])
        slice_classes = np.asarray(slice_data["class_names"]).astype(str)
        slice_record_id, slice_fold_id = validate_slice_arrays(
            slice_prob=slice_prob,
            slice_record_id_raw=slice_record_id_raw,
            slice_fold_id_raw=slice_fold_id_raw,
            record_id=record_id,
            record_fold_id=fold_id,
            expected_folds=args.expected_folds,
        )
        if not np.array_equal(slice_classes, np.asarray(CLASSES)):
            raise ValueError("Slice class order differs from configs.config.CLASSES")

        rebuilt, rebuilt_valid, slice_count = aggregate_record_probabilities(
            slice_prob,
            slice_record_id,
            args.expected_records,
            q=args.q,
        )
        if not np.all(rebuilt_valid):
            raise ValueError("Re-aggregation did not cover all records")
        max_abs_delta = float(np.max(np.abs(rebuilt - y_prob)))
        if max_abs_delta > 2e-6:
            raise ValueError(f"Record probabilities do not match Q={args.q:g} slices: max delta={max_abs_delta}")

        source_config_hash = str(
            npz_scalar(
                record_data,
                "source_config_hash",
                npz_scalar(record_data, "config_hash", ""),
            )
        )
        evaluation_config_hash = str(
            npz_scalar(
                record_data,
                "evaluation_config_hash",
                npz_scalar(record_data, "config_hash", ""),
            )
        )
        checkpoint_kind = str(npz_scalar(record_data, "checkpoint_kind", "best"))
        dataset_record_order_fingerprint = str(
            npz_scalar(record_data, "dataset_record_order_fingerprint", "")
        )
        if not dataset_record_order_fingerprint:
            raise ValueError("OOF artifact lacks the Chapman record-order fingerprint")
        if evaluation_config_hash != EVALUATION_CONFIG_HASH:
            raise ValueError(
                "OOF evaluation_config_hash differs from the current evaluation config. "
                "Re-run 02_reaggregate_oof.py before freezing."
            )
        if checkpoint_kind != args.expected_checkpoint_kind:
            raise ValueError(
                f"Checkpoint kind {checkpoint_kind} is not manuscript-ready under this "
                f"protocol; expected {args.expected_checkpoint_kind}. Validation-selected "
                "best checkpoints are diagnostic unless evaluated in a nested CV design."
            )

    raw_source_checkpoint_rows = checkpoint_rows_from_manifest(run_manifest)
    manifest_dataset_fingerprint = run_manifest.get(
        "dataset_record_order_fingerprint"
    )
    if manifest_dataset_fingerprint != dataset_record_order_fingerprint:
        raise ValueError(
            "Run manifest dataset record-order fingerprint differs from the OOF artifact"
        )
    checkpoint_dataset_fingerprints = {
        str(row.get("dataset_record_order_fingerprint"))
        for row in raw_source_checkpoint_rows
        if row.get("dataset_record_order_fingerprint")
    }
    if checkpoint_dataset_fingerprints != {dataset_record_order_fingerprint}:
        raise ValueError(
            "Source checkpoint dataset record-order fingerprints are incomplete "
            "or differ from the OOF artifact"
        )
    source_checkpoints = normalize_checkpoint_rows(raw_source_checkpoint_rows)
    current_checkpoints_list = current_checkpoint_rows(checkpoint_kind, args.expected_folds)
    current_checkpoints = normalize_checkpoint_rows(current_checkpoints_list)
    if set(source_checkpoints) != set(range(1, args.expected_folds + 1)):
        raise ValueError("Run manifest lacks complete source checkpoint SHA256 evidence")
    mismatches = []
    for fold in range(1, args.expected_folds + 1):
        if source_checkpoints[fold]["sha256"] != current_checkpoints[fold]["sha256"]:
            mismatches.append(fold)
    if mismatches:
        raise ValueError(f"Checkpoint fingerprint mismatch for folds: {mismatches}")

    membership_contract = {
        "status": "not_requested",
        "checkpoint_split_metadata_verified": False,
    }
    folds_path = None
    if strict_manuscript_contract:
        membership_contract, folds_path = validate_membership_contract(
            args=args,
            record_fold_id=fold_id,
            dataset_record_order_fingerprint=dataset_record_order_fingerprint,
            source_checkpoint_rows=raw_source_checkpoint_rows,
            current_checkpoint_rows_list=current_checkpoints_list,
        )

    group_contract = {
        "status": "not_requested",
        "one_record_per_group": False,
    }
    resolved_group_sidecar = None
    if group_sidecar is not None:
        resolved_group_sidecar = resolve_contract_path(group_sidecar, group_sidecar)
        group_contract = validate_group_contract(
            sidecar_path=resolved_group_sidecar,
            expected_sidecar_sha256=group_sidecar_sha256,
            record_file=args.record_file,
            record_id=record_id,
            dataset_record_order_fingerprint=dataset_record_order_fingerprint,
        )

    log_candidates = []
    for pattern in ["oof*_generate_predictions.log", "oof*_reaggregate.log", "oof_reaggregate.log"]:
        log_candidates.extend(LOG_DIR.glob(pattern))
    logs = sorted(
        {
            path
            for path in log_candidates
            if path.is_file() and path.stat().st_size > 0
        }
    )
    metadata_refresh_provenance = None
    if not logs and getattr(args, "metadata_refresh_from_existing_oof", False):
        metadata_refresh_provenance = validate_metadata_refresh_provenance(args)
    elif not logs and not args.allow_missing_log:
        raise FileNotFoundError("No non-empty OOF generation/re-aggregation log found")

    support_files = required + logs
    if folds_path is not None:
        support_files.append(folds_path)
    if resolved_group_sidecar is not None:
        support_files.append(resolved_group_sidecar)
    fold_cache_dir = PREDICTION_DIR / "folds"
    support_files.extend(sorted(fold_cache_dir.glob("oof_fold*.npz")))
    artifacts = [artifact_info(path) for path in support_files]
    manuscript_ready = bool(
        strict_manuscript_contract
        and membership_contract.get("status") == "verified"
        and group_contract.get("status") == "verified"
        and group_contract.get("one_record_per_group") is True
    )
    return {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "manuscript_ready": manuscript_ready,
        "claim_boundary": (
            "manuscript_ready_authenticated_oof"
            if manuscript_ready
            else "exploratory_frozen_oof_not_for_manuscript_claims"
        ),
        "dataset": "chapman_oof",
        "expected_records": args.expected_records,
        "validated_records": args.expected_records,
        "n_classes": len(CLASSES),
        "class_names": CLASSES,
        "expected_folds": args.expected_folds,
        "fold_counts": {
            str(fold): int(np.sum(fold_id == fold))
            for fold in range(1, args.expected_folds + 1)
        },
        "slice_count": int(len(slice_prob)),
        "slice_count_min": int(np.min(slice_count)),
        "slice_count_max": int(np.max(slice_count)),
        "aggregation": {
            "method": "power_mean",
            "q": args.q,
            "implementation": POWER_MEAN_IMPLEMENTATION,
            "max_abs_reaggregation_delta": max_abs_delta,
        },
        "source_config_hash": source_config_hash,
        "dataset_record_order_fingerprint": dataset_record_order_fingerprint,
        "evaluation_config_hash": evaluation_config_hash,
        "current_evaluation_config_hash": EVALUATION_CONFIG_HASH,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_fingerprints_match": True,
        "strict_manuscript_contract": strict_manuscript_contract,
        "membership_contract": membership_contract,
        "group_contract": group_contract,
        "generation_provenance": (
            metadata_refresh_provenance
            or {
                "status": "generation_or_reaggregation_log_present",
                "prediction_values_changed": None,
                "logs": [artifact_info(path) for path in logs],
            }
        ),
        "source_checkpoints": [source_checkpoints[i] for i in sorted(source_checkpoints)],
        "current_checkpoints": current_checkpoints_list,
        "git_commit": git_commit(),
        "run_manifest": str(args.run_manifest),
        "artifacts": artifacts,
    }


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    if args.check_existing_freeze:
        if args.check_only:
            raise ValueError("Use only one of --check-only and --check-existing-freeze")
        args.allow_missing_log = True
        current = validate_oof(args)
        payload = validate_existing_freeze(args, current)
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Validated without rewrite: {args.freeze_manifest}")
        return
    payload = validate_oof(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.check_only:
        save_json(args.freeze_manifest, payload)
        print(f"Wrote: {args.freeze_manifest}")


if __name__ == "__main__":
    main()
