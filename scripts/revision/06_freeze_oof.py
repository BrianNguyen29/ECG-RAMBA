"""Validate and freeze the canonical Chapman OOF prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG_HASH, PATHS  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
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
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-missing-log", action="store_true")
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
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_oof(args: argparse.Namespace) -> dict:
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
        }
        required_slice = {"slice_prob", "record_id", "fold_id", "class_names"}
        if not required_record.issubset(record_data.files):
            raise ValueError(f"Record artifact lacks: {sorted(required_record - set(record_data.files))}")
        if not required_slice.issubset(slice_data.files):
            raise ValueError(f"Slice artifact lacks: {sorted(required_slice - set(slice_data.files))}")

        y_true = np.asarray(record_data["y_true"])
        y_prob = np.asarray(record_data["y_prob"])
        record_id = np.asarray(record_data["record_id"], dtype=np.int64)
        fold_id = np.asarray(record_data["fold_id"], dtype=np.int16)
        class_names = np.asarray(record_data["class_names"]).astype(str)
        valid_mask = (
            np.asarray(record_data["valid_record_mask"], dtype=bool)
            if "valid_record_mask" in record_data.files
            else np.ones(args.expected_records, dtype=bool)
        )
        if y_true.shape != (args.expected_records, len(CLASSES)):
            raise ValueError(f"Unexpected y_true shape: {y_true.shape}")
        if y_prob.shape != y_true.shape:
            raise ValueError(f"Prediction shape mismatch: {y_prob.shape} vs {y_true.shape}")
        if not np.array_equal(record_id, np.arange(args.expected_records, dtype=np.int64)):
            raise ValueError("record_id must be exactly 0..N-1")
        if not np.array_equal(class_names, np.asarray(CLASSES)):
            raise ValueError("OOF class order differs from configs.config.CLASSES")
        if not np.all(valid_mask):
            raise ValueError(f"{int(np.sum(~valid_mask))} OOF records are missing predictions")
        expected_fold_ids = set(range(1, args.expected_folds + 1))
        actual_fold_ids = set(int(x) for x in np.unique(fold_id))
        if actual_fold_ids != expected_fold_ids:
            raise ValueError(f"OOF folds mismatch: {sorted(actual_fold_ids)}")
        if not np.isfinite(y_prob).all() or np.min(y_prob) < 0 or np.max(y_prob) > 1:
            raise ValueError("OOF record probabilities are not finite values in [0, 1]")

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
        slice_record_id = np.asarray(slice_data["record_id"], dtype=np.int64)
        slice_fold_id = np.asarray(slice_data["fold_id"], dtype=np.int16)
        slice_classes = np.asarray(slice_data["class_names"]).astype(str)
        if slice_prob.dtype != np.float32:
            raise ValueError(f"Slice probabilities must be float32, found {slice_prob.dtype}")
        if slice_prob.ndim != 2 or slice_prob.shape[1] != len(CLASSES):
            raise ValueError(f"Unexpected slice probability shape: {slice_prob.shape}")
        if len(slice_prob) != len(slice_record_id) or len(slice_prob) != len(slice_fold_id):
            raise ValueError("Slice arrays have incompatible lengths")
        if not np.array_equal(slice_classes, np.asarray(CLASSES)):
            raise ValueError("Slice class order differs from configs.config.CLASSES")
        if set(int(x) for x in np.unique(slice_fold_id)) != expected_fold_ids:
            raise ValueError("Slice artifact does not contain all expected folds")
        if set(int(x) for x in np.unique(slice_record_id)) != set(range(args.expected_records)):
            raise ValueError("Slice artifact does not cover exactly all OOF records")

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
        if evaluation_config_hash != CONFIG_HASH:
            raise ValueError(
                "OOF evaluation_config_hash differs from the current evaluation config. "
                "Re-run 02_reaggregate_oof.py before freezing."
            )

    source_checkpoints = normalize_checkpoint_rows(checkpoint_rows_from_manifest(run_manifest))
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
    if not logs and not args.allow_missing_log:
        raise FileNotFoundError("No non-empty OOF generation/re-aggregation log found")

    support_files = required + logs
    fold_cache_dir = PREDICTION_DIR / "folds"
    support_files.extend(sorted(fold_cache_dir.glob("oof_fold*.npz")))
    artifacts = [artifact_info(path) for path in support_files]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "manuscript_ready": True,
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
        "evaluation_config_hash": evaluation_config_hash,
        "current_evaluation_config_hash": CONFIG_HASH,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_fingerprints_match": True,
        "source_checkpoints": [source_checkpoints[i] for i in sorted(source_checkpoints)],
        "current_checkpoints": current_checkpoints_list,
        "git_commit": git_commit(),
        "run_manifest": str(args.run_manifest),
        "artifacts": artifacts,
    }


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    payload = validate_oof(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.check_only:
        save_json(args.freeze_manifest, payload)
        print(f"Wrote: {args.freeze_manifest}")


if __name__ == "__main__":
    main()
