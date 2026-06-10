"""Rebuild record-level OOF probabilities from saved slice probabilities.

This script repairs legacy OOF artifacts produced by the incorrect geometric
mean implementation without rerunning model inference. It only writes outputs
when all expected folds and all previously valid records are represented.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONFIG_HASH  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    MANIFEST_DIR,
    METRIC_DIR,
    POWER_MEAN_IMPLEMENTATION,
    PREDICTION_DIR,
    TABLE_DIR,
    aggregate_record_probabilities,
    ensure_revision_dirs,
    multilabel_metrics,
    save_csv,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slice-file",
        type=Path,
        default=PREDICTION_DIR / "oof_full_slice_predictions.npz",
    )
    parser.add_argument(
        "--record-file",
        type=Path,
        default=PREDICTION_DIR / "oof_full_predictions.npz",
        help="Legacy record artifact used only for labels and record metadata.",
    )
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--q", type=float, default=3.0)
    parser.add_argument(
        "--if-possible",
        action="store_true",
        help="Return success without writing when complete slice artifacts are unavailable.",
    )
    return parser.parse_args()


def per_class_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: np.ndarray,
    threshold: float,
) -> list[dict]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rows = []
    for idx, name in enumerate(class_names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        pred = yp >= threshold
        has_both = len(np.unique(yt)) >= 2
        rows.append(
            {
                "class_index": idx,
                "class_name": str(name),
                "n_records": int(len(yt)),
                "n_positive": int(np.sum(yt)),
                "prevalence": float(np.mean(yt)),
                "roc_auc": float(roc_auc_score(yt, yp)) if has_both else np.nan,
                "pr_auc": float(average_precision_score(yt, yp)) if has_both else np.nan,
                "f1": float(f1_score(yt, pred, zero_division=0)),
                "precision": float(precision_score(yt, pred, zero_division=0)),
                "recall": float(recall_score(yt, pred, zero_division=0)),
            }
        )
    return rows


def archive_legacy(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    archive_dir = PREDICTION_DIR / "invalidated_legacy_aggregation"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{path.name}.{stamp}.legacy"
    shutil.copy2(path, target)
    return target


def validate_sources(
    record_data: np.lib.npyio.NpzFile,
    slice_data: np.lib.npyio.NpzFile,
    expected_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_record = {"y_true", "record_id", "class_names"}
    required_slice = {"slice_prob", "record_id", "fold_id"}
    if not required_record.issubset(record_data.files):
        raise ValueError(f"Record artifact lacks keys: {sorted(required_record - set(record_data.files))}")
    if not required_slice.issubset(slice_data.files):
        raise ValueError(f"Slice artifact lacks keys: {sorted(required_slice - set(slice_data.files))}")

    slice_prob = np.asarray(slice_data["slice_prob"])
    slice_record_id = np.asarray(slice_data["record_id"], dtype=np.int64)
    slice_fold_id = np.asarray(slice_data["fold_id"], dtype=np.int16)
    if slice_prob.ndim != 2 or len(slice_prob) != len(slice_record_id):
        raise ValueError("Slice probability and record-id arrays have incompatible shapes")
    if not np.isfinite(slice_prob).all():
        raise ValueError("Slice probabilities contain NaN or infinity")

    expected = set(range(1, expected_folds + 1))
    actual = set(int(x) for x in np.unique(slice_fold_id))
    if actual != expected:
        raise ValueError(f"Incomplete fold coverage: expected {sorted(expected)}, found {sorted(actual)}")

    record_ids = np.asarray(record_data["record_id"], dtype=np.int64)
    valid_mask = (
        np.asarray(record_data["valid_record_mask"], dtype=bool)
        if "valid_record_mask" in record_data.files
        else np.ones(len(record_ids), dtype=bool)
    )
    expected_records = set(int(x) for x in record_ids[valid_mask])
    actual_records = set(int(x) for x in np.unique(slice_record_id))
    if actual_records != expected_records:
        missing = sorted(expected_records - actual_records)[:20]
        extra = sorted(actual_records - expected_records)[:20]
        raise ValueError(
            f"Slice record coverage mismatch: missing={missing}, extra={extra}, "
            f"expected={len(expected_records)}, actual={len(actual_records)}"
        )
    return slice_prob, slice_record_id, slice_fold_id


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    if not args.record_file.exists() or not args.slice_file.exists():
        message = (
            f"Re-aggregation unavailable: record_file={args.record_file.exists()}, "
            f"slice_file={args.slice_file.exists()}"
        )
        if args.if_possible:
            print(message)
            return
        raise FileNotFoundError(message)

    try:
        with np.load(args.record_file, allow_pickle=False) as record_data, np.load(
            args.slice_file,
            allow_pickle=False,
        ) as slice_data:
            slice_prob, slice_record_id, slice_fold_id = validate_sources(
                record_data,
                slice_data,
                args.expected_folds,
            )
            y_true = np.asarray(record_data["y_true"], dtype=np.float32)
            record_id = np.asarray(record_data["record_id"], dtype=np.int64)
            class_names = np.asarray(record_data["class_names"])
            source_config_hash = (
                str(record_data["config_hash"].item())
                if "config_hash" in record_data.files
                else ""
            )
            metadata = {
                key: record_data[key].copy()
                for key in [
                    "dataset",
                    "git_commit",
                    "checkpoint_kind",
                    "batch_size",
                    "threshold",
                ]
                if key in record_data.files
            }
            source_slice_dtype = str(slice_prob.dtype)
            slice_metadata = {
                key: slice_data[key].copy()
                for key in [
                    "record_id",
                    "slice_index",
                    "fold_id",
                    "class_names",
                    "dataset",
                    "protocol",
                    "config_hash",
                    "git_commit",
                    "created_utc",
                ]
                if key in slice_data.files
            }

        y_prob, valid_mask, slice_count = aggregate_record_probabilities(
            slice_prob,
            slice_record_id,
            len(record_id),
            args.q,
        )
        if not np.all(valid_mask):
            raise ValueError(f"{int(np.sum(~valid_mask))} records are missing after aggregation")
        record_fold_id = np.full(len(record_id), -1, dtype=np.int16)
        for rid in np.unique(slice_record_id):
            folds = np.unique(slice_fold_id[slice_record_id == rid])
            if len(folds) != 1:
                raise ValueError(f"Record {int(rid)} is associated with multiple folds: {folds.tolist()}")
            record_fold_id[int(rid)] = int(folds[0])

        threshold = float(metadata.get("threshold", np.asarray(0.5)))
        metadata.pop("threshold", None)
        created_utc = datetime.now(timezone.utc).isoformat()
        stamp = created_utc.replace(":", "").replace("+", "_")
        archived_record = archive_legacy(args.record_file, stamp)
        archived_slice = archive_legacy(args.slice_file, stamp)
        checkpoint_kind = str(metadata.get("checkpoint_kind", np.asarray("best")).item())
        protocol = (
            f"fold_{checkpoint_kind}_{POWER_MEAN_IMPLEMENTATION}_"
            f"q{args.q:g}_threshold_{threshold:g}"
        )

        np.savez_compressed(
            args.slice_file,
            slice_prob=slice_prob.astype(np.float32),
            cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
            probability_dtype=np.asarray("float32"),
            standardized_utc=np.asarray(created_utc),
            **slice_metadata,
        )
        np.savez_compressed(
            args.record_file,
            y_true=y_true,
            y_prob=y_prob.astype(np.float32),
            record_id=record_id,
            class_names=class_names,
            fold_id=record_fold_id,
            slice_count=slice_count,
            valid_record_mask=valid_mask,
            protocol=np.asarray(protocol),
            config_hash=np.asarray(CONFIG_HASH),
            created_utc=np.asarray(created_utc),
            aggregation_method=np.asarray("power_mean"),
            aggregation_q=np.asarray(args.q, dtype=np.float32),
            aggregation_implementation=np.asarray(POWER_MEAN_IMPLEMENTATION),
            cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
            reaggregated_from=np.asarray(str(args.slice_file)),
            source_slice_dtype=np.asarray(source_slice_dtype),
            source_config_hash=np.asarray(source_config_hash),
            **metadata,
        )

        metrics = multilabel_metrics(y_true, y_prob, threshold=threshold)
        class_path = TABLE_DIR / "oof_full_class_summary.csv"
        summary_path = METRIC_DIR / "oof_full_prediction_summary.json"
        manifest_path = MANIFEST_DIR / "oof_full_prediction_run_manifest.json"
        save_csv(class_path, per_class_rows(y_true, y_prob, class_names, threshold))
        save_json(
            summary_path,
            {
                "dataset": "chapman_oof",
                "created_utc": created_utc,
                "config_hash": CONFIG_HASH,
                "prediction_file": str(args.record_file),
                "slice_prediction_file": str(args.slice_file),
                "class_summary_csv": str(class_path),
                "protocol": protocol,
                "n_records": int(len(y_true)),
                "n_valid_predictions": int(np.sum(valid_mask)),
                "n_missing_predictions": int(np.sum(~valid_mask)),
                "n_classes": int(y_true.shape[1]),
                "class_names": [str(x) for x in class_names],
                "aggregation": {"method": "power_mean", "q": args.q},
                "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "source_slice_dtype": source_slice_dtype,
                "source_config_hash": source_config_hash,
                "metrics": metrics,
            },
        )
        save_json(
            manifest_path,
            {
                "created_utc": created_utc,
                "action": "reaggregate_saved_oof_slices",
                "source_slice_file": str(args.slice_file),
                "source_slice_dtype": source_slice_dtype,
                "source_config_hash": source_config_hash,
                "source_fold_ids": sorted(int(x) for x in np.unique(slice_fold_id)),
                "expected_folds": args.expected_folds,
                "archived_invalid_record_artifact": str(archived_record) if archived_record else None,
                "archived_legacy_slice_artifact": str(archived_slice) if archived_slice else None,
                "output_record_file": str(args.record_file),
                "output_slice_file": str(args.slice_file),
                "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
                "aggregation_q": args.q,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "metrics": metrics,
            },
        )
        print(f"Re-aggregated {len(y_true)} records from {len(slice_prob)} saved slices.")
        print(f"Source slice dtype: {source_slice_dtype}")
        print(f"Wrote: {args.record_file}")
        print(f"Wrote: {summary_path}")
        print(f"Wrote: {manifest_path}")
    except (FileNotFoundError, ValueError) as exc:
        if args.if_possible:
            print(f"Re-aggregation skipped: {exc}")
            return
        raise


if __name__ == "__main__":
    main()
    aggregate_record_probabilities,
