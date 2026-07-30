"""Validate immutable baseline artifacts across reviewed runner-only drift.

Baseline predictions remain valid inputs to paired analyses when their original
producer is recoverable from git history and the reviewed source drift did not
change prediction semantics. This module records that narrow compatibility
decision without rewriting historical producer metadata.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPABILITY = "reviewed_baseline_runner_compatibility_attestation_v1"
SCHEMA_VERSION = 1


REVIEWED_RUNNER_COMPATIBILITY: dict[str, dict[str, Any]] = {
    "minirocket": {
        "producer_commit": "78266776c8c89fc233100d151eae70094813d46d",
        "historical_runner_sha256": "e83748251a9d9f71570f2d50d01e9fb1cb05df3c2acf47306c8e7c2823d9b7f5",
        "compatible_current_runner_sha256": "c075c5fc9627e51a577f751f0194892cdee549b575b53c1a8b878a0869547e83",
        "allow_missing_manifest_runner_sha256": True,
        "review_scope": (
            "nomenclature, comparator-contract reporting, and provenance fields only; "
            "the fixed-seed random-convolution MAX+PPV predictions are unchanged"
        ),
    },
    "resnet": {
        "producer_commit": "3b4c384a9db1b70ffbe77f076192a0b181878928",
        "historical_runner_sha256": "2ea4afdd71c42e5c8ba738e2858520385b3b826650bb2aa948b2a314bfb66559",
        "compatible_current_runner_sha256": "47a721463f592b506f3538809edb8af6684cce8d4f235bd5d1dee5d821679d0e",
        "allow_missing_manifest_runner_sha256": False,
        "review_scope": (
            "comparator-contract reporting and legacy Transformer checkpoint metadata "
            "validation only; ResNet prediction semantics are unchanged"
        ),
    },
    "raw_mamba": {
        "producer_commit": "3b4c384a9db1b70ffbe77f076192a0b181878928",
        "historical_runner_sha256": "625903d67a5df59b2ed3263bc337811baedd33c58e69b4dec2c4d4c6b36d1510",
        "compatible_current_runner_sha256": "aefdf37458668ce51197328c92bf323265ffb379f1d619f9692aa0413f7daad9",
        "allow_missing_manifest_runner_sha256": False,
        "review_scope": (
            "same-fold wording, comparator-contract reporting, and provenance fields "
            "only; Raw Mamba prediction semantics are unchanged"
        ),
    },
    "hybrid": {
        "producer_commit": "0bc9361a82dc27f065da4d9cb76ed2fe53b5f84c",
        "historical_runner_sha256": "052006bd54f79d30d685fba561856e115b4d05f51c2bdf64c0ae129357c2d641",
        "compatible_current_runner_sha256": "2d2e02b3afd463804ff621e6a275dc3f3f76ab73d528b9b584aac911938b157a",
        "allow_missing_manifest_runner_sha256": False,
        "review_scope": (
            "comparator-contract reporting and provenance fields only; frozen-transform "
            "morphology MLP predictions are unchanged"
        ),
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def git_bytes(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Could not recover reviewed producer source {relative_path}@{commit}: {detail}"
        ) from exc


def current_authority_source_sha256(relative_path: str) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    tracked_blob = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{relative_path}"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    working_blob = subprocess.check_output(
        ["git", "hash-object", "--path", relative_path, relative_path],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if working_blob != tracked_blob:
        raise RuntimeError(
            f"Baseline producer source differs from the checked-out authority commit: {relative_path}"
        )
    return sha256_bytes(git_bytes(commit, relative_path))


def validate_baseline_producer_provenance(
    *,
    baseline_key: str,
    producer_path: Path,
    summary_path: Path,
    manifest_path: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return an authenticated direct or reviewed-compatibility provenance row."""

    producer_path = Path(producer_path).resolve()
    summary_path = Path(summary_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    producer_relative = project_relative(producer_path)
    current_runner_sha = current_authority_source_sha256(producer_relative)

    summary_sha = sha256_file(summary_path)
    manifest_summary_sha = (manifest.get("artifact_sha256") or {}).get("summary")
    if manifest_summary_sha != summary_sha:
        raise RuntimeError(
            f"{baseline_key} summary is not hash-bound by its producer manifest: "
            f"{summary_sha} != {manifest_summary_sha}"
        )

    summary_commit = str(summary.get("git_commit") or "")
    manifest_commit = str(manifest.get("git_commit") or "")
    if not summary_commit or summary_commit != manifest_commit:
        raise RuntimeError(
            f"{baseline_key} summary/manifest producer commits are missing or inconsistent."
        )

    summary_runner_sha = summary.get("runner_sha256")
    manifest_runner_sha = manifest.get("runner_sha256")
    if summary_runner_sha is not None and summary_runner_sha != manifest_runner_sha:
        raise RuntimeError(
            f"{baseline_key} summary runner SHA differs from its producer manifest."
        )

    if manifest_runner_sha == current_runner_sha:
        return {
            "status": "accepted_current_runner",
            "capability": CAPABILITY,
            "schema_version": SCHEMA_VERSION,
            "producer_commit": manifest_commit,
            "producer_path": producer_relative,
            "observed_runner_sha256": manifest_runner_sha,
            "current_authority_runner_sha256": current_runner_sha,
            "summary_sha256": summary_sha,
            "summary_runner_sha256_inherited_from_manifest": summary_runner_sha is None,
        }

    attestation = REVIEWED_RUNNER_COMPATIBILITY.get(baseline_key)
    if not attestation:
        raise RuntimeError(
            f"{baseline_key} producer runner SHA is stale and has no reviewed compatibility attestation."
        )
    if manifest_commit != attestation["producer_commit"]:
        raise RuntimeError(
            f"{baseline_key} producer commit is outside the reviewed compatibility attestation."
        )
    if current_runner_sha != attestation["compatible_current_runner_sha256"]:
        raise RuntimeError(
            f"{baseline_key} current producer source changed after compatibility review."
        )

    historical_runner_sha = sha256_bytes(
        git_bytes(attestation["producer_commit"], producer_relative)
    )
    if historical_runner_sha != attestation["historical_runner_sha256"]:
        raise RuntimeError(
            f"{baseline_key} historical producer source does not match the reviewed attestation."
        )
    if manifest_runner_sha is None:
        if not attestation["allow_missing_manifest_runner_sha256"]:
            raise RuntimeError(
                f"{baseline_key} manifest omits the historical runner SHA outside an approved legacy case."
            )
    elif manifest_runner_sha != historical_runner_sha:
        raise RuntimeError(
            f"{baseline_key} manifest runner SHA does not match its historical git source."
        )

    return {
        "status": "accepted_reviewed_runner_compatibility",
        "capability": CAPABILITY,
        "schema_version": SCHEMA_VERSION,
        "producer_commit": manifest_commit,
        "producer_path": producer_relative,
        "observed_runner_sha256": manifest_runner_sha,
        "historical_runner_sha256": historical_runner_sha,
        "current_authority_runner_sha256": current_runner_sha,
        "summary_sha256": summary_sha,
        "summary_runner_sha256_inherited_from_manifest": summary_runner_sha is None,
        "review_scope": attestation["review_scope"],
        "claim_boundary": (
            "The attestation authenticates reuse of immutable predictions for paired "
            "analysis; it does not claim that the current runner generated them."
        ),
    }
