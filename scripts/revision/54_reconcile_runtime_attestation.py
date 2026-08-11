"""Reconcile one non-evidence canonical runtime attestation after a verified repair."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.artifact_mirror import normalize_manifest_rows, publish
from scripts.revision.common import REVISION_DIR, save_json, sha256_file


RUNTIME_ATTESTATION_RECONCILIATION_CAPABILITY = (
    "known_non_evidence_runtime_attestation_manifest_reconciliation_v1"
)
RUNTIME_ATTESTATION_RECONCILIATION_SCHEMA_VERSION = 1
KNOWN_RUNTIME_ATTESTATION = Path(
    "manifests/robustness_low_memory_execution_attestation_v50.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    return parser.parse_args()


def validate_runtime_only_attestation(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "status": "pre_execution_verified",
        "scope": "source_bound_robustness_inference_all_stresses",
        "mathematical_change": False,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Known runtime attestation has an unexpected scientific/protocol shape: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if not str(payload.get("canonical_runner_path", "")).endswith(
        "scripts/revision/12_robustness_stress.py"
    ):
        raise RuntimeError("Known runtime attestation has an unexpected runner path")
    return payload


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + ".partial." + uuid.uuid4().hex
    )
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            raise RuntimeError("Temporary runtime-attestation copy failed SHA256 verification")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    canonical_root = args.canonical_root.resolve()
    source = canonical_root / KNOWN_RUNTIME_ATTESTATION
    manifest_path = canonical_root / "manifests" / "mirror_manifest.json"
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Runtime-attestation reconciliation requires both canonical files: "
            f"{source}; {manifest_path}"
        )

    validate_runtime_only_attestation(source)
    actual_sha = sha256_file(source)
    actual_size = source.stat().st_size
    rows = {
        row["relative_path"]: row
        for row in normalize_manifest_rows(
            json.loads(manifest_path.read_text(encoding="utf-8")), canonical_root
        )
    }
    previous = rows.get(KNOWN_RUNTIME_ATTESTATION.as_posix())
    already_verified = bool(
        previous
        and int(previous["size_bytes"]) == actual_size
        and previous["sha256"] == actual_sha
    )
    if already_verified:
        print("Known runtime-only attestation already matches the canonical manifest.")
        return

    local_attestation = REVISION_DIR / KNOWN_RUNTIME_ATTESTATION
    atomic_copy(source, local_attestation)
    report_relative = Path("manifests/runtime_attestation_mirror_reconciliation.json")
    report_path = REVISION_DIR / report_relative
    report = {
        "capability": RUNTIME_ATTESTATION_RECONCILIATION_CAPABILITY,
        "schema_version": RUNTIME_ATTESTATION_RECONCILIATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "reconciled_known_runtime_only_attestation",
        "relative_path": KNOWN_RUNTIME_ATTESTATION.as_posix(),
        "previous_manifest_row": previous,
        "verified_canonical": {"size_bytes": actual_size, "sha256": actual_sha},
        "scientific_evidence_affected": False,
        "reason": (
            "A pre-execution low-memory runtime attestation changed after direct canonical "
            "write while retaining the same byte length. It contains no prediction, metric, "
            "or scientific-result payload and is rehashed only after schema validation."
        ),
    }
    save_json(report_path, report)
    publish(
        canonical_root,
        verify_existing="size",
        source_conflict_policy="source",
        refresh_existing_prefixes=[KNOWN_RUNTIME_ATTESTATION.as_posix()],
        include_paths=[
            KNOWN_RUNTIME_ATTESTATION.as_posix(),
            report_relative.as_posix(),
        ],
    )

    refreshed_rows = {
        row["relative_path"]: row
        for row in normalize_manifest_rows(
            json.loads(manifest_path.read_text(encoding="utf-8")), canonical_root
        )
    }
    refreshed = refreshed_rows.get(KNOWN_RUNTIME_ATTESTATION.as_posix())
    if not refreshed or refreshed["sha256"] != actual_sha:
        raise RuntimeError("Canonical manifest was not updated with the verified runtime attestation")
    print("Reconciled known runtime-only attestation:", KNOWN_RUNTIME_ATTESTATION)
    print("Wrote:", report_path)


if __name__ == "__main__":
    main()
