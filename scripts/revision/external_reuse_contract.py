"""Lightweight source-bound reuse checks for external prediction artifacts.

This module intentionally has no torch, WFDB, or Mamba dependency so Notebook 02
can decide whether GPU inference is needed from a fresh CPU runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.revision.common import CACHE_SCHEMA_VERSION, save_json, sha256_file
from src.aggregation import aggregate_record_probabilities


EXTERNAL_REUSE_CAPABILITY = "source_bound_external_reuse_v2_dataset_scoped_attestation"
EXTERNAL_REUSE_SCHEMA_VERSION = 2
# The v2 exporter change is confined to the CPSC2021 disk-backed window loader.
# Bind the exception to both immutable source hashes so any later exporter edit
# fails closed. CPSC2021 is deliberately absent and must be regenerated.
RUNNER_COMPATIBILITY_ATTESTATIONS = {
    "ptbxl": {
        "68bd59aad0323a5077a1665edc1a1afd2a73542f4060e4d534998287dff04df8": {
            "compatible_current_runner_sha256": "8ebd81758b8f566f148702d3a7e17b99da93342aa793e9157c8b8ab2960fa216",
            "producer_release": "refs/tags/ecg-ramba-revision-20260721-v1",
            "reviewed_change_scope": "cpsc2021_window_storage_only",
        }
    },
    "georgia": {
        "68bd59aad0323a5077a1665edc1a1afd2a73542f4060e4d534998287dff04df8": {
            "compatible_current_runner_sha256": "8ebd81758b8f566f148702d3a7e17b99da93342aa793e9157c8b8ab2960fa216",
            "producer_release": "refs/tags/ecg-ramba-revision-20260721-v1",
            "reviewed_change_scope": "cpsc2021_window_storage_only",
        }
    },
}
EXPECTED_LABEL_PROTOCOLS = {
    "ptbxl": "official_ptbxl_diagnostic_superclass_any_positive_likelihood",
    "georgia": "chapman_27_class_snomed_intersection",
    "cpsc2021": "annotation_aligned_full_10s_windows_strict_majority_af_or_normal_v2",
}
EXPECTED_GROUP_UNITS = {
    "ptbxl": "patient_id",
    "georgia": "record_id_assumed_independent",
    "cpsc2021": "source_ecg_record",
}


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.reshape(-1)[0].item() if value.size else default


def _cached_archive_sha256(path: Path, cache_dir: Path) -> str:
    """Return a content hash while avoiding repeated multi-GB Drive reads.

    The stat-keyed cache is only a Notebook preflight optimization. The protocol
    gate independently recomputes the full archive SHA256 before accepting the
    dataset as manuscript-ready.
    """

    stat = path.stat()
    stat_contract = {
        "resolved_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    key = hashlib.sha256(
        json.dumps(stat_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{path.stem}_{key[:20]}.json"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        cached_sha = str(payload.get("sha256", ""))
        if payload.get("stat_contract") == stat_contract and len(cached_sha) == 64:
            return cached_sha

    digest = sha256_file(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        cache_path,
        {
            "cache_kind": "external_archive_stat_keyed_sha256_v1",
            "stat_contract": stat_contract,
            "sha256": digest,
            "note": "Preflight cache only; the strict protocol gate re-hashes archive content.",
        },
    )
    return digest


def _artifact_paths(dataset: str, revision_root: Path) -> dict[str, Path]:
    base = revision_root / "experimental" / "external" / dataset
    return {
        "prediction": base / f"{dataset}_full_predictions.npz",
        "slice_prediction": base / f"{dataset}_full_slice_predictions.npz",
        "summary": base / f"{dataset}_full_prediction_summary.json",
        "manifest": base / f"{dataset}_full_prediction_run_manifest.json",
        "class_summary": base / f"{dataset}_full_class_summary.csv",
    }


def validate_external_prediction_reuse(
    dataset: str,
    *,
    revision_root: Path,
    archive_path: Path | None,
    exporter_path: Path,
    oof_path: Path,
    freeze_path: Path,
    archive_hash_cache_dir: Path,
    threshold: float = 0.5,
    q: float = 3.0,
) -> dict[str, Any]:
    """Validate whether a dataset export can be reused without GPU inference."""

    if dataset not in EXPECTED_LABEL_PROTOCOLS:
        raise ValueError(f"Unsupported external dataset: {dataset}")
    paths = _artifact_paths(dataset, Path(revision_root))
    reasons: list[str] = []
    diagnostics: dict[str, Any] = {}

    for name, path in paths.items():
        if not path.exists() or path.stat().st_size <= 0:
            reasons.append(f"missing_or_empty={name}")
    for name, path in {
        "exporter": exporter_path,
        "canonical_oof": oof_path,
        "canonical_freeze": freeze_path,
    }.items():
        if not path.exists() or path.stat().st_size <= 0:
            reasons.append(f"missing_contract_input={name}")
    if archive_path is None or not archive_path.exists() or archive_path.stat().st_size <= 0:
        reasons.append("missing_external_archive")
    if reasons:
        return {"ready": False, "reasons": reasons, "diagnostics": diagnostics, "paths": paths}

    try:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"unreadable_json={type(exc).__name__}:{exc}")
        return {"ready": False, "reasons": reasons, "diagnostics": diagnostics, "paths": paths}

    current_contract = {
        "oof_sha256": sha256_file(oof_path),
        "freeze_sha256": sha256_file(freeze_path),
    }
    observed_contract = manifest.get("canonical_contract") or {}
    diagnostics["current_canonical_contract"] = current_contract
    diagnostics["observed_canonical_contract"] = observed_contract
    if observed_contract != current_contract:
        reasons.append("canonical_oof_or_freeze_sha_mismatch")

    current_runner_sha = sha256_file(exporter_path)
    diagnostics["current_runner_sha256"] = current_runner_sha
    diagnostics["observed_runner_sha256"] = manifest.get("runner_sha256")
    observed_runner_sha = str(manifest.get("runner_sha256", ""))
    if observed_runner_sha != current_runner_sha:
        attestation = RUNNER_COMPATIBILITY_ATTESTATIONS.get(dataset, {}).get(
            observed_runner_sha
        )
        if (
            not attestation
            or attestation.get("compatible_current_runner_sha256") != current_runner_sha
        ):
            reasons.append("external_exporter_sha_mismatch")
        else:
            diagnostics["runner_compatibility_attestation"] = {
                **attestation,
                "observed_runner_sha256": observed_runner_sha,
                "current_runner_sha256": current_runner_sha,
                "status": "accepted_dataset_unaffected_by_reviewed_change",
            }

    manifest_archive = manifest.get("archive") or {}
    current_archive_size = int(archive_path.stat().st_size)
    if int(manifest_archive.get("size_bytes", -1)) != current_archive_size:
        reasons.append("external_archive_size_mismatch")
    else:
        current_archive_sha = _cached_archive_sha256(archive_path, archive_hash_cache_dir)
        diagnostics["current_archive_sha256"] = current_archive_sha
        diagnostics["observed_archive_sha256"] = manifest_archive.get("sha256")
        if manifest_archive.get("sha256") != current_archive_sha:
            reasons.append("external_archive_sha256_mismatch")

    if summary.get("dataset") != dataset or manifest.get("dataset") != dataset:
        reasons.append("dataset_metadata_mismatch")
    load_summary = summary.get("load_summary") or {}
    if load_summary.get("label_protocol") != EXPECTED_LABEL_PROTOCOLS[dataset]:
        reasons.append("label_protocol_mismatch")
    if not math.isclose(float(summary.get("threshold", math.nan)), threshold, abs_tol=1e-7):
        reasons.append("summary_threshold_mismatch")
    aggregation = summary.get("aggregation") or {}
    if aggregation.get("method") != "power_mean" or not math.isclose(
        float(aggregation.get("q", math.nan)), q, rel_tol=1e-6
    ):
        reasons.append("summary_aggregation_mismatch")
    if summary.get("evidence_status") != "experimental" or summary.get("manuscript_ready") is not False:
        reasons.append("summary_evidence_boundary_mismatch")

    manifest_outputs = manifest.get("outputs") or {}
    for name in ("prediction", "slice_prediction", "summary", "class_summary"):
        path = paths[name]
        row = manifest_outputs.get(path.name) or {}
        if int(row.get("size_bytes", -1)) != path.stat().st_size or row.get("sha256") != sha256_file(path):
            reasons.append(f"manifest_output_hash_mismatch={name}")

    try:
        with np.load(paths["prediction"], allow_pickle=False) as record_data:
            required_record_keys = {
                "y_true", "y_prob", "record_id", "group_id", "group_unit", "split_id",
                "class_names", "dataset", "cache_schema_version", "evidence_status",
                "manuscript_ready", "aggregation_method", "aggregation_q", "threshold",
            }
            missing_record_keys = sorted(required_record_keys - set(record_data.files))
            if missing_record_keys:
                reasons.append("record_missing_keys=" + ",".join(missing_record_keys))
                raise ValueError("record NPZ contract incomplete")
            y_true = np.asarray(record_data["y_true"], dtype=np.float32)
            y_prob = np.asarray(record_data["y_prob"], dtype=np.float32)
            record_id = np.asarray(record_data["record_id"]).astype(str)
            group_id = np.asarray(record_data["group_id"]).astype(str)
            split_id = np.asarray(record_data["split_id"]).astype(str)
            group_unit = str(_scalar(record_data, "group_unit", ""))
            if y_true.ndim != 2 or y_prob.shape != y_true.shape or len(record_id) != len(y_true):
                reasons.append("record_shape_or_alignment_invalid")
            if not np.isfinite(y_prob).all() or np.min(y_prob) < 0.0 or np.max(y_prob) > 1.0:
                reasons.append("record_probabilities_invalid")
            if not np.isfinite(y_true).all() or not np.all((y_true == 0.0) | (y_true == 1.0)):
                reasons.append("record_labels_not_finite_binary")
            if len(np.unique(record_id)) != len(record_id):
                reasons.append("record_ids_not_unique")
            if len(group_id) != len(y_true) or len(split_id) != len(y_true):
                reasons.append("record_group_or_split_alignment_invalid")
            if group_unit != EXPECTED_GROUP_UNITS[dataset]:
                reasons.append("group_unit_mismatch")
            if dataset == "ptbxl" and set(split_id.tolist()) != {"ptbxl_fold10"}:
                reasons.append("ptbxl_not_official_fold10")
            if dataset == "cpsc2021" and len(np.unique(group_id)) >= len(group_id):
                reasons.append("cpsc_group_clustering_missing")
            if int(_scalar(record_data, "cache_schema_version", -1)) != CACHE_SCHEMA_VERSION:
                reasons.append("record_cache_schema_stale")
            if str(_scalar(record_data, "dataset", "")) != dataset:
                reasons.append("record_dataset_mismatch")
            if str(_scalar(record_data, "aggregation_method", "")) != "power_mean":
                reasons.append("record_aggregation_method_mismatch")
            if not math.isclose(float(_scalar(record_data, "aggregation_q", math.nan)), q, rel_tol=1e-6):
                reasons.append("record_aggregation_q_mismatch")
            if not math.isclose(float(_scalar(record_data, "threshold", math.nan)), threshold, abs_tol=1e-7):
                reasons.append("record_threshold_mismatch")

        with np.load(paths["slice_prediction"], allow_pickle=False) as slice_data:
            required_slice_keys = {"slice_prob", "record_index", "record_id", "group_id", "split_id"}
            missing_slice_keys = sorted(required_slice_keys - set(slice_data.files))
            if missing_slice_keys:
                reasons.append("slice_missing_keys=" + ",".join(missing_slice_keys))
                raise ValueError("slice NPZ contract incomplete")
            slice_prob = np.asarray(slice_data["slice_prob"], dtype=np.float32)
            record_index = np.asarray(slice_data["record_index"], dtype=np.int64)
            if slice_prob.ndim != 2 or len(slice_prob) == 0 or len(record_index) != len(slice_prob):
                reasons.append("slice_shape_or_alignment_invalid")
            elif np.min(record_index) < 0 or np.max(record_index) >= len(record_id):
                reasons.append("slice_record_index_out_of_bounds")
            else:
                if not np.array_equal(np.asarray(slice_data["record_id"]).astype(str), record_id[record_index]):
                    reasons.append("slice_record_id_linkage_invalid")
                if not np.array_equal(np.asarray(slice_data["group_id"]).astype(str), group_id[record_index]):
                    reasons.append("slice_group_id_linkage_invalid")
                if not np.array_equal(np.asarray(slice_data["split_id"]).astype(str), split_id[record_index]):
                    reasons.append("slice_split_id_linkage_invalid")
                reconstructed, valid, counts = aggregate_record_probabilities(
                    slice_prob, record_index, len(record_id), q=q
                )
                max_abs = float(np.max(np.abs(reconstructed[valid] - y_prob[valid]))) if np.any(valid) else math.inf
                diagnostics["q3_reconstruction_max_abs"] = max_abs
                if not np.all(counts > 0) or max_abs > 1e-6:
                    reasons.append(f"q3_reconstruction_mismatch={max_abs}")
    except (OSError, ValueError, TypeError, FloatingPointError) as exc:
        if not any(item.startswith(("record_", "slice_")) for item in reasons):
            reasons.append(f"unreadable_prediction_contract={type(exc).__name__}:{exc}")

    if dataset == "cpsc2021":
        cpsc_required = {
            "primary_requires_full_window": True,
            "primary_excludes_transition_windows": True,
            "primary_requires_strict_majority": True,
        }
        for key, expected in cpsc_required.items():
            if load_summary.get(key) is not expected:
                reasons.append(f"cpsc_{key}_mismatch")
        if int(load_summary.get("primary_window_length_samples", -1)) != 5000:
            reasons.append("cpsc_primary_window_length_samples_mismatch")
        if not math.isclose(float(load_summary.get("primary_window_length_seconds", math.nan)), 10.0):
            reasons.append("cpsc_primary_window_length_seconds_mismatch")
        for key in ("partial_windows_excluded", "tie_windows_excluded", "transition_windows_excluded"):
            if key not in load_summary:
                reasons.append(f"cpsc_missing_{key}")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "diagnostics": diagnostics,
        "paths": paths,
    }
