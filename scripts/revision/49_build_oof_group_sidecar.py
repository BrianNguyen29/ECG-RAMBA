"""Build the SHA-bound Chapman OOF one-record-per-patient group sidecar.

The PhysioNet ECG Arrhythmia Database describes one recording per patient for
both Chapman-Shaoxing and Ningbo. The canonical OOF file stores source record
indices rather than separate patient identifiers, so under that reviewed source
contract each record index is also a unique resampling group. This script binds
that identity mapping to the exact OOF file, record order, and source archive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PATHS
from scripts.revision.common import (
    CHAPMAN_GROUP_REFERENCE,
    CHAPMAN_GROUP_REFERENCE_COUNTS,
    CHAPMAN_GROUP_SEMANTICS,
    git_commit,
    npz_scalar,
    sha256_file,
)


GROUP_SIDECAR_CAPABILITY = "chapman_oof_group_sidecar_v1"
GROUP_SIDECAR_SCHEMA_VERSION = 1
DEFAULT_OOF = Path("reports/revision/predictions/oof_final_ema_predictions.npz")
DEFAULT_OUTPUT = Path("reports/revision/manifests/oof_final_ema_group_sidecar.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--source-archive", type=Path, default=Path(PATHS["zip_path"]))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-records", type=int, default=44186)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def integral_record_ids(values: np.ndarray, expected_records: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (expected_records,):
        raise ValueError(f"record_id shape mismatch: {raw.shape} != {(expected_records,)}")
    if raw.dtype.kind in "iu":
        result = raw.astype(np.int64, copy=False)
    elif raw.dtype.kind == "f" and np.isfinite(raw).all() and np.all(raw == np.floor(raw)):
        result = raw.astype(np.int64)
    else:
        raise ValueError("record_id must be a finite integral vector")
    if len(np.unique(result)) != expected_records:
        raise ValueError("record_id must contain exactly one unique source record per OOF row")
    return result


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_sidecar(
    oof_path: Path,
    archive_path: Path,
    output_path: Path,
    *,
    expected_records: int,
    reuse_existing: bool = False,
) -> dict:
    for label, path in {"OOF predictions": oof_path, "Chapman archive": archive_path}.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")

    with np.load(oof_path, allow_pickle=False) as data:
        if "record_id" not in data.files:
            raise ValueError("OOF prediction artifact lacks record_id")
        record_id = integral_record_ids(data["record_id"], expected_records)
        order_fingerprint = str(npz_scalar(data, "dataset_record_order_fingerprint", ""))
        if not order_fingerprint:
            raise ValueError("OOF prediction artifact lacks dataset_record_order_fingerprint")
        dataset = str(npz_scalar(data, "dataset", ""))
        if dataset != "chapman_oof":
            raise ValueError(f"Expected dataset='chapman_oof', observed {dataset!r}")

    oof_sha = sha256_file(oof_path)
    archive_sha = sha256_file(archive_path)
    producer_sha = sha256_file(Path(__file__).resolve())
    if reuse_existing and output_path.is_file() and output_path.stat().st_size > 0:
        try:
            with np.load(output_path, allow_pickle=False) as existing:
                existing_record_id = np.asarray(existing["record_id"], dtype=np.int64)
                existing_group_id = np.asarray(existing["group_id"], dtype=np.int64)
                reusable = (
                    np.array_equal(existing_record_id, record_id)
                    and np.array_equal(existing_group_id, record_id)
                    and str(npz_scalar(existing, "record_file_sha256", "")) == oof_sha
                    and str(npz_scalar(existing, "source_archive_sha256", "")) == archive_sha
                    and str(npz_scalar(existing, "dataset_record_order_fingerprint", ""))
                    == order_fingerprint
                    and str(npz_scalar(existing, "producer_sha256", "")) == producer_sha
                    and int(npz_scalar(existing, "schema_version", -1))
                    == GROUP_SIDECAR_SCHEMA_VERSION
                )
        except (OSError, KeyError, ValueError):
            reusable = False
        if reusable:
            result = {
                "status": "reused",
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "n_records": int(len(record_id)),
                "n_groups": int(len(np.unique(record_id))),
                "record_file_sha256": oof_sha,
                "source_archive_sha256": archive_sha,
                "dataset_record_order_fingerprint": order_fingerprint,
                "group_semantics_reference": CHAPMAN_GROUP_REFERENCE,
            }
            print(json.dumps(result, indent=2))
            return result
    # Under the reviewed one-record-per-patient source contract, using the
    # source record identifier as group identifier is exact and label-free.
    group_id = record_id.copy()
    write_npz_atomic(
        output_path,
        record_id=record_id,
        group_id=group_id,
        group_unit=np.asarray("authenticated_source_patient_record"),
        group_semantics=np.asarray(CHAPMAN_GROUP_SEMANTICS),
        group_semantics_reference=np.asarray(CHAPMAN_GROUP_REFERENCE),
        source_patient_record_counts_json=np.asarray(
            json.dumps(CHAPMAN_GROUP_REFERENCE_COUNTS, sort_keys=True)
        ),
        one_record_per_group=np.asarray(True),
        dataset_record_order_fingerprint=np.asarray(order_fingerprint),
        record_file_sha256=np.asarray(oof_sha),
        source_archive_sha256=np.asarray(archive_sha),
        source_archive_size_bytes=np.asarray(archive_path.stat().st_size, dtype=np.int64),
        schema_version=np.asarray(GROUP_SIDECAR_SCHEMA_VERSION, dtype=np.int16),
        capability=np.asarray(GROUP_SIDECAR_CAPABILITY),
        producer_git_commit=np.asarray(git_commit()),
        producer_sha256=np.asarray(producer_sha),
    )
    result = {
        "status": "complete",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "n_records": int(len(record_id)),
        "n_groups": int(len(np.unique(group_id))),
        "record_file_sha256": oof_sha,
        "source_archive_sha256": archive_sha,
        "dataset_record_order_fingerprint": order_fingerprint,
        "group_semantics_reference": CHAPMAN_GROUP_REFERENCE,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    build_sidecar(
        args.oof_predictions,
        args.source_archive,
        args.out,
        expected_records=args.expected_records,
        reuse_existing=args.reuse_existing,
    )


if __name__ == "__main__":
    main()
