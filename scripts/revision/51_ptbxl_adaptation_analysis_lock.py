"""Create or verify the immutable PTB-XL fold-9/10 adaptation analysis lock.

This is a reproducibility lock for the current revision reruns, not a claim of
historical preregistration: fold-10 results had already been inspected before
this lock was introduced. Once written, a configuration mismatch is a hard
error and requires a new explicitly versioned lock file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    git_commit,
    save_json_atomic,
    sha256_file,
)


PTBXL_ADAPTATION_LOCK_CAPABILITY = "ptbxl_fold9_fold10_analysis_lock_v1"
PTBXL_ADAPTATION_LOCK_SCHEMA_VERSION = 1
PTBXL_ADAPTATION_LOCK_SOURCE_ATTESTATION_CAPABILITY = (
    "ptbxl_analysis_lock_implementation_attestation_v1"
)
PTBXL_ADAPTATION_LOCK_SOURCE_ATTESTATION_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="full,resnet,raw_mamba,transformer")
    parser.add_argument("--fractions", default="0,0.01,0.05,0.10")
    parser.add_argument("--primary-fraction", type=float, default=0.10)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--head-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument(
        "--out-lock",
        type=Path,
        default=MANIFEST_DIR / "ptbxl_adaptation_analysis_lock.json",
    )
    parser.add_argument(
        "--out-source-attestation",
        type=Path,
        default=MANIFEST_DIR / "ptbxl_adaptation_analysis_lock_source_attestation.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    fractions = sorted({float(item.strip()) for item in args.fractions.split(",") if item.strip()})
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if models != ["full", "resnet", "raw_mamba", "transformer"]:
        raise ValueError("The reviewer analysis lock requires full,resnet,raw_mamba,transformer in that order")
    if fractions != [0.0, 0.01, 0.05, 0.1]:
        raise ValueError("The reviewer analysis lock requires fractions 0,0.01,0.05,0.10")
    if seeds != [42, 43, 44, 45, 46]:
        raise ValueError("The reviewer analysis lock requires seeds 42,43,44,45,46")
    return {
        "dataset": "ptbxl",
        "adaptation_split": "official_ptbxl_fold9",
        "test_split": "official_ptbxl_fold10",
        "group_unit": "patient_id",
        "patient_overlap_required": 0,
        "models": models,
        "fractions": fractions,
        "primary_fraction": float(args.primary_fraction),
        "fraction_unit": "independent_patient_groups_from_fold9",
        "fraction_sampling": "nested_seeded_label_independent_group_prefix",
        "seeds": seeds,
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "n_boot": int(args.n_boot),
        "bootstrap_unit": "patient_group",
        "score_calibration": {
            "kind": "per_class_unweighted_monotonic_platt_scaling",
            "regularization_C": 1.0,
            "fit_split": "fold9_only",
        },
        "frozen_encoder_head": {
            "kind": "fold_specific_balanced_logistic_head",
            "regularization_C": float(args.head_c),
            "max_iter": int(args.max_iter),
            "fit_split": "fold9_only",
            "encoder_weights_updated": False,
        },
        "primary_test_access_policy": "evaluate_fold10_only_after_configuration_lock_validation",
        "unsupported_only_sensitivity": {
            "primary": "all_mapped_task_records",
            "sensitivity": "exclude_records_with_no_supported_positive_superclass",
            "selection_uses_predictions": False,
        },
    }


def expected_lock(args: argparse.Namespace) -> dict[str, Any]:
    protocol = protocol_payload(args)
    source_paths = [
        PROJECT_ROOT / "scripts" / "revision" / "31_generate_external_comparator_predictions.py",
        PROJECT_ROOT / "scripts" / "revision" / "32_paired_external_comparators.py",
        PROJECT_ROOT / "scripts" / "revision" / "33_group_safe_score_calibration.py",
        PROJECT_ROOT / "scripts" / "revision" / "35_true_fewshot_head_adaptation.py",
        PROJECT_ROOT / "scripts" / "revision" / "52_ptbxl_fold_protocol_audit.py",
    ]
    return {
        "status": "locked",
        "capability": PTBXL_ADAPTATION_LOCK_CAPABILITY,
        "schema_version": PTBXL_ADAPTATION_LOCK_SCHEMA_VERSION,
        "lock_scope": "reproducibility_lock_for_current_and_future_revision_reruns",
        "temporal_qualification": (
            "post_initial_result_review; this artifact is not a preregistration and must not be described as one"
        ),
        "protocol": protocol,
        "protocol_sha256": canonical_json_sha256(protocol),
        "runner_sources": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for path in source_paths
        ],
    }


def validate_existing(existing: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    issues = []
    for key in (
        "status",
        "capability",
        "schema_version",
        "lock_scope",
        "temporal_qualification",
        "protocol",
        "protocol_sha256",
    ):
        if existing.get(key) != expected.get(key):
            issues.append(key)
    existing_sources = existing.get("runner_sources")
    expected_sources = expected.get("runner_sources") or []
    if not isinstance(existing_sources, list):
        issues.append("runner_sources")
        return issues
    existing_by_path = {
        str(row.get("path")): row
        for row in existing_sources
        if isinstance(row, dict) and row.get("path")
    }
    expected_paths = {str(row["path"]) for row in expected_sources}
    if set(existing_by_path) != expected_paths:
        issues.append("runner_sources")
        return issues

    def valid_sha256(value: Any) -> bool:
        text = str(value or "").lower()
        return len(text) == 64 and all(character in "0123456789abcdef" for character in text)

    if any(not valid_sha256(existing_by_path[path].get("sha256")) for path in expected_paths):
        issues.append("runner_sources")
    return issues


def runner_source_drift(
    existing: dict[str, Any], expected: dict[str, Any]
) -> list[dict[str, str]]:
    """Describe implementation drift without mutating the immutable protocol lock."""

    existing_by_path = {
        str(row["path"]): str(row["sha256"])
        for row in existing.get("runner_sources") or []
    }
    expected_by_path = {
        str(row["path"]): str(row["sha256"])
        for row in expected.get("runner_sources") or []
    }
    return [
        {
            "path": path,
            "locked_sha256": existing_by_path[path],
            "current_sha256": expected_by_path[path],
            "classification": "implementation_changed_after_protocol_lock",
        }
        for path in sorted(expected_by_path)
        if existing_by_path.get(path) != expected_by_path[path]
    ]


def write_source_attestation(
    path: Path,
    *,
    lock_path: Path,
    lock: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    drift = runner_source_drift(lock, current)
    payload = {
        "status": "complete",
        "capability": PTBXL_ADAPTATION_LOCK_SOURCE_ATTESTATION_CAPABILITY,
        "schema_version": PTBXL_ADAPTATION_LOCK_SOURCE_ATTESTATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "analysis_lock": {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path),
            "protocol_sha256": lock["protocol_sha256"],
        },
        "protocol_unchanged": lock.get("protocol") == current.get("protocol"),
        "locked_runner_sources_preserved": True,
        "current_runner_sources": current.get("runner_sources") or [],
        "runner_source_drift": drift,
        "interpretation": (
            "The immutable post-initial-review lock fixes analysis choices, not future source bytes. "
            "Runner changes are disclosed here; each generated result manifest must independently bind "
            "the current runner SHA and the unchanged lock SHA."
        ),
    }
    save_json_atomic(path, payload)
    return payload


def main() -> None:
    args = parse_args()
    out = resolve(args.out_lock)
    attestation_out = resolve(args.out_source_attestation)
    expected = expected_lock(args)
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        issues = validate_existing(existing, expected)
        if issues:
            raise RuntimeError(
                f"PTB-XL analysis lock mismatch in {out}: {issues}. "
                "Do not overwrite it silently; use a new versioned lock after documented protocol review."
            )
        attestation = write_source_attestation(
            attestation_out,
            lock_path=out,
            lock=existing,
            current=expected,
        )
        print(f"Reusing exact PTB-XL adaptation protocol lock: {out}", flush=True)
        print(
            "Implementation drift attested without modifying the lock: "
            f"{[row['path'] for row in attestation['runner_source_drift']]}",
            flush=True,
        )
        print(f"Wrote source attestation: {attestation_out}", flush=True)
        return
    payload = {
        **expected,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json_atomic(out, payload)
    write_source_attestation(
        attestation_out,
        lock_path=out,
        lock=payload,
        current=expected,
    )
    print(f"Wrote immutable PTB-XL adaptation analysis lock: {out}", flush=True)
    print(f"protocol_sha256={payload['protocol_sha256']}", flush=True)
    print(f"Wrote source attestation: {attestation_out}", flush=True)


if __name__ == "__main__":
    main()
