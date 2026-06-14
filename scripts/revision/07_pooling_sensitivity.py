"""Compute pooling sensitivity from frozen OOF slice probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    aggregate_record_probabilities,
    multilabel_metrics,
    save_csv,
    save_json,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json",
    )
    parser.add_argument(
        "--record-file",
        type=Path,
        default=PREDICTION_DIR / "oof_final_ema_predictions.npz",
    )
    parser.add_argument(
        "--slice-file",
        type=Path,
        default=PREDICTION_DIR / "oof_final_ema_slice_predictions.npz",
    )
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def project_relative_path(path: Path) -> tuple[Path, str]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    return resolved, resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()


def verify_frozen_artifact(manifest: dict, path: Path) -> None:
    resolved, relative = project_relative_path(path)
    rows = {row["path"]: row for row in manifest.get("artifacts", [])}
    if relative not in rows:
        raise ValueError(f"Freeze manifest does not include {relative}")
    if sha256_file(resolved) != rows[relative]["sha256"]:
        raise RuntimeError(f"Frozen artifact checksum changed: {relative}")


def aggregate_max(
    slice_prob: np.ndarray,
    slice_record_id: np.ndarray,
    n_records: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_classes = slice_prob.shape[1]
    if np.any(slice_record_id < 0) or np.any(slice_record_id >= n_records):
        raise ValueError("slice_record_id contains values outside the record range")
    result = np.full((n_records, n_classes), -np.inf, dtype=np.float32)
    np.maximum.at(result, slice_record_id, slice_prob)
    counts = np.bincount(slice_record_id, minlength=n_records).astype(np.int16)
    valid = counts > 0
    result[~valid] = 0.0
    return result, valid, counts


def main() -> None:
    args = parse_args()
    if not args.freeze_manifest.exists():
        raise FileNotFoundError(
            f"OOF must be frozen before pooling sensitivity: {args.freeze_manifest}"
        )
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("manuscript_ready") is not True:
        raise ValueError("OOF freeze manifest is not manuscript-ready")
    if freeze.get("checkpoint_kind") != args.expected_checkpoint_kind:
        raise ValueError(
            "Unexpected OOF checkpoint kind: "
            f"{freeze.get('checkpoint_kind')} != {args.expected_checkpoint_kind}"
        )
    verify_frozen_artifact(freeze, args.record_file)
    verify_frozen_artifact(freeze, args.slice_file)

    with np.load(args.record_file, allow_pickle=False) as record_data, np.load(
        args.slice_file,
        allow_pickle=False,
    ) as slice_data:
        y_true = np.asarray(record_data["y_true"], dtype=np.float32)
        slice_prob = np.asarray(slice_data["slice_prob"], dtype=np.float32)
        slice_record_id = np.asarray(slice_data["record_id"], dtype=np.int64)

    rows = []
    detail = {}
    methods: list[tuple[str, float | None]] = [
        ("mean", 1.0),
        ("power_mean_q2", 2.0),
        ("power_mean_q3", 3.0),
        ("power_mean_q4", 4.0),
        ("power_mean_q8", 8.0),
        ("max", None),
    ]
    for name, q in methods:
        if q is None:
            y_prob, valid, counts = aggregate_max(slice_prob, slice_record_id, len(y_true))
        else:
            y_prob, valid, counts = aggregate_record_probabilities(
                slice_prob,
                slice_record_id,
                len(y_true),
                q=q,
            )
        if not np.all(valid):
            raise ValueError(f"{name}: {int(np.sum(~valid))} records have no slices")
        metrics = multilabel_metrics(y_true, y_prob, threshold=args.threshold)
        row = {
            "dataset": "chapman_oof",
            "pooling": name,
            "q": q if q is not None else "max",
            "threshold": args.threshold,
            "n_records": len(y_true),
            "slice_count_min": int(np.min(counts)),
            "slice_count_max": int(np.max(counts)),
            **metrics,
        }
        rows.append(row)
        detail[name] = row

    csv_path = METRIC_DIR / "pooling_sensitivity.csv"
    json_path = METRIC_DIR / "pooling_sensitivity.json"
    save_csv(csv_path, rows)
    save_json(
        json_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "chapman_oof",
            "source_freeze_manifest": str(args.freeze_manifest),
            "source_freeze_manifest_sha256": sha256_file(args.freeze_manifest),
            "record_file": str(args.record_file),
            "slice_file": str(args.slice_file),
            "threshold": args.threshold,
            "results": detail,
        },
    )
    print(json.dumps(detail, indent=2))
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
