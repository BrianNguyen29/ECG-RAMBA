"""Validate immutable baseline artifacts across reviewed source-bundle drift.

Baseline predictions remain valid inputs to paired analyses when their original
producer is recoverable from git history, the exact artifact package is pinned,
and the reviewed executable source-bundle drift did not change prediction
semantics. This module records that narrow compatibility decision without
rewriting historical producer metadata.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPABILITY = "reviewed_baseline_source_bundle_compatibility_attestation_v2"
SCHEMA_VERSION = 2


BASE_SOURCE_BUNDLE = (
    "configs/config.py",
    "scripts/revision/common.py",
    "src/aggregation.py",
    "src/provenance.py",
)


REVIEWED_RUNNER_COMPATIBILITY: dict[str, dict[str, Any]] = {
    "minirocket": {
        "producer_commit": "78266776c8c89fc233100d151eae70094813d46d",
        "source_bundle_paths": (
            "scripts/revision/10_minirocket_only_baseline.py",
            *BASE_SOURCE_BUNDLE,
        ),
        "historical_runner_sha256": "e83748251a9d9f71570f2d50d01e9fb1cb05df3c2acf47306c8e7c2823d9b7f5",
        "compatible_current_runner_sha256": "c075c5fc9627e51a577f751f0194892cdee549b575b53c1a8b878a0869547e83",
        "historical_source_bundle_sha256": "f2d6c1c65eaaac5bb652bf0cd584397e219edc09f95012bc20f112caec182a86",
        "compatible_current_source_bundle_sha256": "20d80b3d7fc87ff12187e85968453cec5d1b02b0061ef73343bbdf92fdec6bb1",
        "expected_manifest_sha256": "8ed1b6e73ea40fd96bc3e9b0b8bd52f868a9dddb059618c7b7ff4c5ea5f77bcf",
        "expected_summary_sha256": "d6a1d6e68af35155bde0481ee1737b4291af1bb9e89045ce7cced671ca3183db",
        "expected_prediction_sha256": "8eb9d43c2fde96f06b3d10c8a64090147b23d626621f92d7bab2c61896c2d1ec",
        "allow_missing_manifest_runner_sha256": True,
        "review_scope": (
            "exact package and executable source bundle reviewed across nomenclature, "
            "comparator-contract reporting, scalar paired-metric helper additions, and "
            "provenance-only drift; the fixed-seed "
            "random-convolution MAX+PPV predictions are unchanged"
        ),
    },
    "resnet": {
        "producer_commit": "3b4c384a9db1b70ffbe77f076192a0b181878928",
        "source_bundle_paths": (
            "scripts/revision/14_resnet1d_cnn_baseline.py",
            *BASE_SOURCE_BUNDLE,
            "src/training_data.py",
        ),
        "historical_runner_sha256": "2ea4afdd71c42e5c8ba738e2858520385b3b826650bb2aa948b2a314bfb66559",
        "compatible_current_runner_sha256": "47a721463f592b506f3538809edb8af6684cce8d4f235bd5d1dee5d821679d0e",
        "historical_source_bundle_sha256": "085a526ebf387d003c1f4846e04bec792b56a73cbfcb2c2180629aab1a01669f",
        "compatible_current_source_bundle_sha256": "456e8ca7a4f845b92550366eed47d727e51b402a0ef774d4f1227964ab7db899",
        "expected_manifest_sha256": "486dc2e9ebc634f52b62312e23727b5bb0107522a992fc03cef4800f78af880b",
        "expected_summary_sha256": "3db0d738fc103cbceacc81f921c4e70eba3058b7617fa5549bfd7dfb7c6854f2",
        "expected_prediction_sha256": "b1af5f641858cb5778deff180a35641468f97723612ce9b3f83bf6c89e401dcd",
        "allow_missing_manifest_runner_sha256": False,
        "review_scope": (
            "exact package and executable source bundle reviewed across comparator-contract "
            "reporting, scalar paired-metric helper additions, and legacy Transformer "
            "checkpoint metadata validation-only drift; "
            "ResNet prediction semantics are unchanged"
        ),
    },
    "raw_mamba": {
        "producer_commit": "3b4c384a9db1b70ffbe77f076192a0b181878928",
        "source_bundle_paths": (
            "scripts/revision/16_raw_mamba_baseline.py",
            "scripts/revision/14_resnet1d_cnn_baseline.py",
            *BASE_SOURCE_BUNDLE,
            "src/training_data.py",
            "src/utils.py",
        ),
        "historical_runner_sha256": "625903d67a5df59b2ed3263bc337811baedd33c58e69b4dec2c4d4c6b36d1510",
        "compatible_current_runner_sha256": "aefdf37458668ce51197328c92bf323265ffb379f1d619f9692aa0413f7daad9",
        "historical_source_bundle_sha256": "c8ba5c09954c6695e8b93b0f9701fbdea992ee499b964b3251c5d12e72ae8c8e",
        "compatible_current_source_bundle_sha256": "fca50938744ee5f04025b417c1739dc48cffd76d0ec4607364047d9392639e3c",
        "expected_manifest_sha256": "334fd492aa8286c704bdeaad223a28993813a5ec0c12b766dae60a9f73ba430f",
        "expected_summary_sha256": "8210b4f3c3d46cd6acf0517582544dc713e3050356e46522ba54e1efdaf7acd4",
        "expected_prediction_sha256": "f8e7102b15675c0bb1cf89012279f45b5faee039cdb032c19e7a8aeee9b3dfd6",
        "allow_missing_manifest_runner_sha256": False,
        "review_scope": (
            "exact package and executable source bundle reviewed across same-fold wording, "
            "comparator-contract reporting, scalar paired-metric helper additions, and "
            "provenance-only drift; Raw Mamba prediction "
            "semantics are unchanged"
        ),
    },
    "transformer": {
        "producer_commit": "e388f10515ab68834f3bff80b41efe1635967b6c",
        "source_bundle_paths": (
            "scripts/revision/24_transformer_ecg_baseline.py",
            "scripts/revision/14_resnet1d_cnn_baseline.py",
            *BASE_SOURCE_BUNDLE,
            "src/training_data.py",
        ),
        "historical_runner_sha256": "65d60f3bb62526456344b6f8cf69450695fc89a8663d13a3780d6a5e8da9a30c",
        "compatible_current_runner_sha256": "65d60f3bb62526456344b6f8cf69450695fc89a8663d13a3780d6a5e8da9a30c",
        "historical_source_bundle_sha256": "658387b569f1992bc3c09decfd1e51a928c24e22fb6772342850021d2a14ef77",
        "compatible_current_source_bundle_sha256": "4fc41e4e10ea893265e4b4ad84fabf1ac95bc2e7fd90e6cb7bdd7b198eac95a0",
        "expected_manifest_sha256": "ab6efd83215a344bfa13176685e899eeb8969579736fff706a5007bf815772c9",
        "expected_summary_sha256": "06788f74235652dbe23e2fcaf09d758686229913f109690204d9416025e73b51",
        "expected_prediction_sha256": "7ab38e492e7e980208094c445684eb8d8d8f2e62ad9fbb6be5b6f23b30da13e9",
        "allow_missing_manifest_runner_sha256": False,
        "review_scope": (
            "exact package and executable source bundle reviewed across shared ResNet "
            "pipeline metadata/provenance drift and scalar paired-metric helper additions; "
            "Transformer prediction semantics are unchanged"
        ),
    },
    "hybrid": {
        "producer_commit": "0bc9361a82dc27f065da4d9cb76ed2fe53b5f84c",
        "source_bundle_paths": (
            "scripts/revision/26_hybrid_morphology_baseline.py",
            "scripts/revision/10_minirocket_only_baseline.py",
            *BASE_SOURCE_BUNDLE,
        ),
        "historical_runner_sha256": "052006bd54f79d30d685fba561856e115b4d05f51c2bdf64c0ae129357c2d641",
        "compatible_current_runner_sha256": "2d2e02b3afd463804ff621e6a275dc3f3f76ab73d528b9b584aac911938b157a",
        "historical_source_bundle_sha256": "a90ee6fc0a274e4e7c3af4a790a2556c4a86c01f13570d920018193e758d423a",
        "compatible_current_source_bundle_sha256": "d44497af0878ee3a20da7dfda2486553b33ddfccb11d47562d718b143f78a16a",
        "expected_manifest_sha256": "dab6cd02f2c7095235c055dd720a030bd0419bd4eb1463d6989d7f9bf5630c4c",
        "expected_summary_sha256": "9ca31ec00b52726242be995cb60810bf4cb28c985162e5f129aa943105f24e6e",
        "expected_prediction_sha256": "c4710db3b978473ddc4e851c1d37af54060fb43dc4db64adc026a5f6e636daac",
        "allow_missing_manifest_runner_sha256": False,
        "review_scope": (
            "exact package and executable source bundle reviewed across comparator-contract "
            "reporting, scalar paired-metric helper additions, and provenance-only drift; "
            "frozen-transform morphology MLP predictions "
            "are unchanged"
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
    commit = current_authority_commit()
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


def current_authority_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def source_bundle_sha256(commit: str, relative_paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(set(relative_paths)):
        path_bytes = relative_path.encode("utf-8")
        source_bytes = git_bytes(commit, relative_path)
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(source_bytes).to_bytes(8, "big"))
        digest.update(source_bytes)
    return digest.hexdigest()


def current_authority_source_bundle_sha256(relative_paths: tuple[str, ...]) -> str:
    commit = current_authority_commit()
    for relative_path in sorted(set(relative_paths)):
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
                "Baseline executable source bundle differs from the checked-out "
                f"authority commit: {relative_path}"
            )
    return source_bundle_sha256(commit, relative_paths)


def require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise RuntimeError(f"{label} must be a nonempty 64-character SHA-256 digest.")
    return text


def validate_baseline_producer_provenance(
    *,
    baseline_key: str,
    producer_path: Path,
    summary_path: Path,
    manifest_path: Path,
    prediction_path: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return an authenticated direct or reviewed-compatibility provenance row."""

    producer_path = Path(producer_path).resolve()
    summary_path = Path(summary_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    prediction_path = Path(prediction_path).resolve()
    producer_relative = project_relative(producer_path)
    current_runner_sha = current_authority_source_sha256(producer_relative)
    current_commit = current_authority_commit()
    attestation = REVIEWED_RUNNER_COMPATIBILITY.get(baseline_key)
    if not attestation:
        raise RuntimeError(
            f"{baseline_key} has no executable source-bundle definition."
        )
    source_bundle_paths = tuple(attestation["source_bundle_paths"])
    current_source_bundle_sha = current_authority_source_bundle_sha256(
        source_bundle_paths
    )

    summary_sha = sha256_file(summary_path)
    manifest_sha = sha256_file(manifest_path)
    prediction_sha = sha256_file(prediction_path)
    artifact_sha256 = manifest.get("artifact_sha256") or {}
    manifest_summary_sha = require_sha256(
        artifact_sha256.get("summary"),
        label=f"{baseline_key} manifest summary SHA",
    )
    manifest_prediction_sha = require_sha256(
        artifact_sha256.get("predictions"),
        label=f"{baseline_key} manifest prediction SHA",
    )
    if manifest_summary_sha != summary_sha:
        raise RuntimeError(
            f"{baseline_key} summary is not hash-bound by its producer manifest: "
            f"{summary_sha} != {manifest_summary_sha}"
        )
    if manifest_prediction_sha != prediction_sha:
        raise RuntimeError(
            f"{baseline_key} prediction is not hash-bound by its producer manifest: "
            f"{prediction_sha} != {manifest_prediction_sha}"
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

    if manifest_commit == current_commit and manifest_runner_sha == current_runner_sha:
        return {
            "status": "accepted_current_authority_producer",
            "capability": CAPABILITY,
            "schema_version": SCHEMA_VERSION,
            "producer_commit": manifest_commit,
            "producer_path": producer_relative,
            "observed_runner_sha256": manifest_runner_sha,
            "current_authority_runner_sha256": current_runner_sha,
            "current_authority_commit": current_commit,
            "source_bundle_paths": list(source_bundle_paths),
            "current_authority_source_bundle_sha256": current_source_bundle_sha,
            "manifest_sha256": manifest_sha,
            "summary_sha256": summary_sha,
            "prediction_sha256": prediction_sha,
            "summary_runner_sha256_inherited_from_manifest": summary_runner_sha is None,
        }

    if manifest_commit != attestation["producer_commit"]:
        raise RuntimeError(
            f"{baseline_key} producer commit is outside the reviewed compatibility attestation."
        )
    exact_package = {
        "manifest": (manifest_sha, attestation["expected_manifest_sha256"]),
        "summary": (summary_sha, attestation["expected_summary_sha256"]),
        "prediction": (prediction_sha, attestation["expected_prediction_sha256"]),
    }
    package_mismatches = [
        role
        for role, (observed, expected) in exact_package.items()
        if observed != expected
    ]
    if package_mismatches:
        raise RuntimeError(
            f"{baseline_key} artifact package differs from the reviewed exact-digest "
            f"attestation: {package_mismatches}"
        )
    if current_runner_sha != attestation["compatible_current_runner_sha256"]:
        raise RuntimeError(
            f"{baseline_key} current producer source changed after compatibility review."
        )
    if (
        current_source_bundle_sha
        != attestation["compatible_current_source_bundle_sha256"]
    ):
        raise RuntimeError(
            f"{baseline_key} current executable source bundle changed after compatibility review."
        )

    historical_runner_sha = sha256_bytes(
        git_bytes(attestation["producer_commit"], producer_relative)
    )
    if historical_runner_sha != attestation["historical_runner_sha256"]:
        raise RuntimeError(
            f"{baseline_key} historical producer source does not match the reviewed attestation."
        )
    historical_source_bundle_sha = source_bundle_sha256(
        attestation["producer_commit"],
        source_bundle_paths,
    )
    if historical_source_bundle_sha != attestation["historical_source_bundle_sha256"]:
        raise RuntimeError(
            f"{baseline_key} historical executable source bundle does not match "
            "the reviewed attestation."
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
        "source_bundle_paths": list(source_bundle_paths),
        "historical_source_bundle_sha256": historical_source_bundle_sha,
        "current_authority_source_bundle_sha256": current_source_bundle_sha,
        "manifest_sha256": manifest_sha,
        "summary_sha256": summary_sha,
        "prediction_sha256": prediction_sha,
        "summary_runner_sha256_inherited_from_manifest": summary_runner_sha is None,
        "review_scope": attestation["review_scope"],
        "claim_boundary": (
            "The attestation authenticates reuse of immutable predictions for paired "
            "analysis; it does not claim that the current runner generated them."
        ),
    }
