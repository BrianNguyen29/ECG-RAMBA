"""Validate and select resumable learned-comparator robustness profiles.

Notebook 05 can emit several profile-scoped ledgers. Screening profiles are
useful reviewer audits but are not interchangeable with the canonical
six-stress, five-metric, 1,000-bootstrap package. This module keeps that
distinction explicit while enforcing a common artifact/provenance contract.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.revision.common import (
    AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    CHAPMAN_GROUP_REFERENCE,
    CHAPMAN_GROUP_SEMANTICS,
    ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION,
    sha256_file,
)


PROTOCOL = "robustness_multicomparator_aggregation_v2_source_bound"
CI_SCOPE = "nominal_95_percentile_paired_record_bootstrap_unadjusted"
BOOTSTRAP_UNIT = AUTHENTICATED_RECORD_BOOTSTRAP_UNIT
TRAINING_VARIABILITY_SCOPE = "fixed_trained_folds_and_checkpoints_not_retrained_within_bootstrap"
METRIC_CACHE_SCHEMA_VERSION = ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION
MACRO_CLASS_SUPPORT_POLICY = (
    "rank_calibration_omit_single_resampled_class_f1_keeps_all_labels_zero_division_zero"
)
ALLOWED_INTERPRETATIONS = {
    "full_nominal_95ci_more_favorable_change",
    "comparator_nominal_95ci_more_favorable_change",
    "nominal_95ci_inconclusive_change_difference",
}
CANONICAL_STRESSES = {
    "snr20db",
    "snr10db",
    "snr5db",
    "random_3_lead_dropout",
    "precordial_dropout",
    "resample_250hz",
}
CANONICAL_METRICS = {
    "pr_auc_macro",
    "roc_auc_macro",
    "f1_macro",
    "brier_macro",
    "ece_macro",
}
CANONICAL_COMPARATORS = {"full", "minirocket", "resnet", "raw_mamba", "transformer"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def profile_paths(revision_root: Path, profile: str) -> dict[str, Path]:
    stem = "robustness_multicomparator"
    if profile != "canonical":
        stem = f"{stem}_{profile}"
    return {
        "summary": revision_root / "metrics" / f"{stem}_summary.csv",
        "pairwise": revision_root / "metrics" / f"{stem}_pairwise.json",
        "table": revision_root / "tables" / f"table_{stem}.csv",
        "manifest": revision_root / "manifests" / f"{stem}_manifest.json",
    }


def discover_profiles(revision_root: Path) -> list[str]:
    manifest_dir = revision_root / "manifests"
    profiles: set[str] = set()
    canonical = manifest_dir / "robustness_multicomparator_manifest.json"
    if canonical.exists():
        profiles.add("canonical")
    prefix = "robustness_multicomparator_"
    suffix = "_manifest.json"
    for path in manifest_dir.glob(f"{prefix}*{suffix}"):
        profile = path.name[len(prefix) : -len(suffix)]
        if profile:
            profiles.add(profile)
    return sorted(profiles)


def _classification(
    *,
    profile: str,
    n_boot: int,
    stresses: set[str],
    metrics: set[str],
    comparators: set[str],
) -> dict[str, Any]:
    comparator_complete = CANONICAL_COMPARATORS.issubset(comparators)
    canonical_gate_ready = (
        profile == "canonical"
        and n_boot >= 1000
        and CANONICAL_STRESSES.issubset(stresses)
        and CANONICAL_METRICS.issubset(metrics)
        and comparator_complete
    )
    metric_specific_ci_ready = (
        n_boot >= 1000 and bool(stresses) and bool(metrics) and comparator_complete
    )
    if canonical_gate_ready:
        return {
            "evidence_tier": "canonical_full_ledger",
            "audit_ready": True,
            "metric_specific_ci_ready": True,
            "canonical_gate_ready": True,
            "safe_wording": (
                "Use only named stress-, metric-, and comparator-specific signed change differences. "
                "CIs are nominal and unadjusted across the comparison family; completion does not "
                "establish broad robustness superiority."
            ),
        }
    if metric_specific_ci_ready:
        return {
            "evidence_tier": "final_metric_specific_subset",
            "audit_ready": True,
            "metric_specific_ci_ready": True,
            "canonical_gate_ready": False,
            "safe_wording": (
                "Use only the named stress-, metric-, and comparator-specific signed change differences "
                "with nominal unadjusted CIs. The subset does not represent the predefined full ledger."
            ),
        }
    return {
        "evidence_tier": "screening_subset",
        "audit_ready": True,
        "metric_specific_ci_ready": False,
        "canonical_gate_ready": False,
        "safe_wording": (
            "Treat this profile as a screening audit only because it uses fewer than 1,000 bootstrap replicates "
            "and/or incomplete stress/metric coverage. Do not use it for a final robustness claim."
        ),
    }


def validate_profile(
    revision_root: Path,
    profile: str,
    *,
    canonical_contract: dict[str, str],
    runner_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    paths = profile_paths(revision_root, profile)
    issues: list[str] = []
    for label, path in paths.items():
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"missing_or_empty:{label}={path.as_posix()}")
    if issues:
        return {
            "profile": profile,
            "valid": False,
            "issues": issues,
            "paths": {name: path.as_posix() for name, path in paths.items()},
            "evidence_tier": "invalid_or_incomplete",
            "audit_ready": False,
            "metric_specific_ci_ready": False,
            "canonical_gate_ready": False,
        }

    try:
        manifest = _read_json(paths["manifest"])
        pairwise = _read_json(paths["pairwise"])
        rows = _read_csv(paths["summary"])
        table_rows = _read_csv(paths["table"])
    except Exception as exc:
        return {
            "profile": profile,
            "valid": False,
            "issues": [f"parse_failed:{type(exc).__name__}:{exc}"],
            "paths": {name: path.as_posix() for name, path in paths.items()},
            "evidence_tier": "invalid_or_incomplete",
            "audit_ready": False,
            "metric_specific_ci_ready": False,
            "canonical_gate_ready": False,
        }

    expected_runner_sha = sha256_file(runner_path) if runner_path.exists() else ""
    for label, payload in (("manifest", manifest), ("pairwise", pairwise)):
        if payload.get("status") != "complete":
            issues.append(f"{label}.status={payload.get('status')!r}")
        if payload.get("protocol") != PROTOCOL:
            issues.append(f"{label}.protocol={payload.get('protocol')!r}")
        if payload.get("output_profile") != profile:
            issues.append(f"{label}.output_profile={payload.get('output_profile')!r}")
        if payload.get("canonical_contract") != canonical_contract:
            issues.append(f"{label}.canonical_contract_mismatch")
        if payload.get("runner_sha256") != expected_runner_sha:
            issues.append(f"{label}.runner_sha256_mismatch")
        if payload.get("ci_scope") != CI_SCOPE:
            issues.append(f"{label}.ci_scope={payload.get('ci_scope')!r}")
        if payload.get("bootstrap_unit") != BOOTSTRAP_UNIT:
            issues.append(f"{label}.bootstrap_unit={payload.get('bootstrap_unit')!r}")
        if payload.get("training_variability_scope") != TRAINING_VARIABILITY_SCOPE:
            issues.append(
                f"{label}.training_variability_scope={payload.get('training_variability_scope')!r}"
            )
        if int(payload.get("metric_cache_schema_version", 0)) != METRIC_CACHE_SCHEMA_VERSION:
            issues.append(f"{label}.metric_cache_schema_version_invalid")
        if payload.get("macro_class_support_policy") != MACRO_CLASS_SUPPORT_POLICY:
            issues.append(f"{label}.macro_class_support_policy_invalid")
        independence = payload.get("bootstrap_independence_contract") or {}
        if (
            independence.get("unit") != BOOTSTRAP_UNIT
            or independence.get("independence_contract") != CHAPMAN_GROUP_SEMANTICS
            or independence.get("group_semantics_reference") != CHAPMAN_GROUP_REFERENCE
            or not independence.get("group_sidecar_sha256")
            or independence.get("training_variability_scope") != TRAINING_VARIABILITY_SCOPE
        ):
            issues.append(f"{label}.bootstrap_independence_contract_invalid")
        source_value = independence.get("source")
        if source_value:
            source_path = _project_path(project_root, source_value)
            if not source_path.exists() or source_path.stat().st_size == 0:
                issues.append(f"{label}.bootstrap_contract_source_missing")
            elif independence.get("source_sha256") != sha256_file(source_path):
                issues.append(f"{label}.bootstrap_contract_source_sha256_mismatch")
        else:
            issues.append(f"{label}.bootstrap_contract_source_missing")
        if set((payload.get("stress_contracts") or {}).keys()) != set(
            payload.get("stress_tests") or []
        ):
            issues.append(f"{label}.stress_contracts_incomplete")

    stresses = {str(item) for item in manifest.get("stress_tests") or []}
    metrics = {str(item) for item in manifest.get("metrics") or []}
    comparators = {str(item) for item in manifest.get("comparators") or []}
    n_boot = _as_int(manifest.get("n_boot"))
    if {str(item) for item in pairwise.get("stress_tests") or []} != stresses:
        issues.append("pairwise_stress_coverage_mismatch")
    if {str(item) for item in pairwise.get("metrics") or []} != metrics:
        issues.append("pairwise_metric_coverage_mismatch")
    if {str(item) for item in pairwise.get("comparators") or []} != comparators:
        issues.append("pairwise_comparator_coverage_mismatch")
    if _as_int(pairwise.get("n_boot")) != n_boot:
        issues.append("pairwise_n_boot_mismatch")
    expected_rows = len(stresses) * len(metrics) * max(0, len(comparators) - 1)
    if not stresses or not metrics or not comparators:
        issues.append("manifest_coverage_empty")
    if "full" not in comparators:
        issues.append("manifest_comparators_missing_full")
    if _as_int(manifest.get("completed_rows")) != expected_rows:
        issues.append("manifest_completed_rows_mismatch")
    if _as_int(manifest.get("blocked_rows"), 0) != 0:
        issues.append("manifest_has_blocked_rows")
    if _as_int(pairwise.get("completed_rows")) != expected_rows:
        issues.append("pairwise_completed_rows_mismatch")
    if _as_int(pairwise.get("blocked_rows"), 0) != 0:
        issues.append("pairwise_has_blocked_rows")
    if len(rows) != expected_rows or len(table_rows) != expected_rows:
        issues.append(
            f"row_count_mismatch:expected={expected_rows},summary={len(rows)},table={len(table_rows)}"
        )
    if table_rows != rows:
        issues.append("table_summary_content_mismatch")

    expected_keys = {
        (stress, comparator, metric)
        for stress in stresses
        for comparator in comparators - {"full"}
        for metric in metrics
    }
    observed_keys = {
        (str(row.get("stress")), str(row.get("comparator")), str(row.get("metric")))
        for row in rows
    }
    if observed_keys != expected_keys:
        issues.append("summary_row_coverage_mismatch")
    if any(str(row.get("status")) != "complete" for row in rows):
        issues.append("summary_has_noncomplete_rows")
    if any(str(row.get("output_profile")) != profile for row in rows):
        issues.append("summary_output_profile_mismatch")
    if any(_as_int(row.get("n_boot")) != n_boot for row in rows):
        issues.append("summary_n_boot_mismatch")
    if any(str(row.get("ci_scope")) != CI_SCOPE for row in rows):
        issues.append("summary_ci_scope_mismatch")
    if any(str(row.get("bootstrap_unit")) != BOOTSTRAP_UNIT for row in rows):
        issues.append("summary_bootstrap_unit_mismatch")
    if any(
        str(row.get("training_variability_scope")) != TRAINING_VARIABILITY_SCOPE
        for row in rows
    ):
        issues.append("summary_training_variability_scope_mismatch")
    if any(str(row.get("macro_class_support_policy")) != MACRO_CLASS_SUPPORT_POLICY for row in rows):
        issues.append("summary_macro_class_support_policy_mismatch")
    if any(str(row.get("interpretation")) not in ALLOWED_INTERPRETATIONS for row in rows):
        issues.append("summary_interpretation_not_claim_safe")
    invalid_artifacts = [
        row for row in manifest.get("artifact_status") or [] if row.get("status") != "ready"
    ]
    expected_artifact_rows = len(comparators) * (1 + len(stresses))
    if len(manifest.get("artifact_status") or []) != expected_artifact_rows:
        issues.append("manifest_artifact_provenance_grid_incomplete")
    if invalid_artifacts:
        issues.append(f"manifest_has_invalid_artifacts:{len(invalid_artifacts)}")
    minimum_valid_bootstraps = max(1, math.ceil(0.95 * n_boot))
    ci_fields = (
        "degradation_adv_ci_low",
        "degradation_adv_ci_high",
        "stressed_adv_ci_low",
        "stressed_adv_ci_high",
    )
    for index, row in enumerate(rows):
        n_boot_valid = _as_int(row.get("n_boot_valid"), 0)
        if n_boot_valid < minimum_valid_bootstraps or n_boot_valid > n_boot:
            issues.append(
                f"summary_invalid_bootstrap_count:row={index},valid={n_boot_valid},requested={n_boot}"
            )
        try:
            ci_values = {name: float(row.get(name, "nan")) for name in ci_fields}
        except (TypeError, ValueError):
            issues.append(f"summary_non_numeric_ci:row={index}")
            continue
        if not all(math.isfinite(value) for value in ci_values.values()):
            issues.append(f"summary_non_finite_ci:row={index}")
        if ci_values["degradation_adv_ci_low"] > ci_values["degradation_adv_ci_high"]:
            issues.append(f"summary_reversed_degradation_ci:row={index}")
        if ci_values["stressed_adv_ci_low"] > ci_values["stressed_adv_ci_high"]:
            issues.append(f"summary_reversed_stressed_ci:row={index}")

    pairwise_items = pairwise.get("items") or {}
    if set(pairwise_items) != {
        f"{stress}/{comparator}/{metric}"
        for stress, comparator, metric in expected_keys
    }:
        issues.append("pairwise_item_coverage_mismatch")
    else:
        for row in rows:
            key = f"{row.get('stress')}/{row.get('comparator')}/{row.get('metric')}"
            item = pairwise_items[key]
            for field in (
                "status",
                "output_profile",
                "interpretation",
                "n_boot",
                "n_boot_valid",
            ):
                if str(item.get(field)) != str(row.get(field)):
                    issues.append(f"pairwise_summary_mismatch:{key}:{field}")
                    break

    artifact_sha = manifest.get("artifact_sha256") or {}
    for label in ("summary", "table", "pairwise"):
        expected_sha = artifact_sha.get(label)
        if expected_sha != sha256_file(paths[label]):
            issues.append(f"manifest_artifact_sha256_mismatch:{label}")

    sidecar_hashes = artifact_sha.get("comparator_sidecars") or {}
    sidecar_paths = (manifest.get("outputs") or {}).get("comparator_sidecars") or {}
    for comparator in sorted(comparators - {"full", "minirocket"}):
        sidecar_value = sidecar_paths.get(comparator)
        if not sidecar_value:
            issues.append(f"missing_sidecar_path:{comparator}")
            continue
        sidecar_path = _project_path(project_root, sidecar_value)
        if not sidecar_path.exists() or sidecar_path.stat().st_size == 0:
            issues.append(f"missing_sidecar:{comparator}")
            continue
        if sidecar_hashes.get(comparator) != sha256_file(sidecar_path):
            issues.append(f"sidecar_sha256_mismatch:{comparator}")
            continue
        sidecar = _read_json(sidecar_path)
        if sidecar.get("status") != "complete":
            issues.append(f"sidecar_status:{comparator}={sidecar.get('status')!r}")
        if sidecar.get("canonical_contract") != canonical_contract:
            issues.append(f"sidecar_contract_mismatch:{comparator}")
        if sidecar.get("output_profile") != profile:
            issues.append(f"sidecar_profile_mismatch:{comparator}")
        if sidecar.get("runner_sha256") != expected_runner_sha:
            issues.append(f"sidecar_runner_sha256_mismatch:{comparator}")
        if sidecar.get("ci_scope") != CI_SCOPE:
            issues.append(f"sidecar_ci_scope_mismatch:{comparator}")
        if sidecar.get("bootstrap_unit") != BOOTSTRAP_UNIT:
            issues.append(f"sidecar_bootstrap_unit_mismatch:{comparator}")
        if sidecar.get("training_variability_scope") != TRAINING_VARIABILITY_SCOPE:
            issues.append(f"sidecar_training_variability_scope_mismatch:{comparator}")
        if int(sidecar.get("metric_cache_schema_version", 0)) != METRIC_CACHE_SCHEMA_VERSION:
            issues.append(f"sidecar_metric_cache_schema_version_mismatch:{comparator}")
        if sidecar.get("macro_class_support_policy") != MACRO_CLASS_SUPPORT_POLICY:
            issues.append(f"sidecar_macro_class_support_policy_mismatch:{comparator}")
        if sidecar.get("bootstrap_independence_contract") != manifest.get(
            "bootstrap_independence_contract"
        ):
            issues.append(f"sidecar_bootstrap_independence_contract_mismatch:{comparator}")
        if sidecar.get("source_pairwise_sha256") != sha256_file(paths["pairwise"]):
            issues.append(f"sidecar_pairwise_sha256_mismatch:{comparator}")
        sidecar_rows = sidecar.get("rows") or []
        expected_sidecar_keys = {
            (stress, comparator, metric)
            for stress in stresses
            for metric in metrics
        }
        observed_sidecar_keys = {
            (str(row.get("stress")), str(row.get("comparator")), str(row.get("metric")))
            for row in sidecar_rows
        }
        if observed_sidecar_keys != expected_sidecar_keys:
            issues.append(f"sidecar_row_coverage_mismatch:{comparator}")

    issues = list(dict.fromkeys(issues))
    classification = _classification(
        profile=profile,
        n_boot=n_boot,
        stresses=stresses,
        metrics=metrics,
        comparators=comparators,
    )
    valid = not issues
    interpretation_counts = Counter(str(row.get("interpretation") or "missing") for row in rows)
    result = {
        "profile": profile,
        "valid": valid,
        "issues": issues,
        "protocol": manifest.get("protocol"),
        "n_boot": n_boot,
        "stresses": sorted(stresses),
        "metrics": sorted(metrics),
        "comparators": sorted(comparators),
        "completed_rows": len(rows),
        "blocked_rows": sum(str(row.get("status")) != "complete" for row in rows),
        "interpretation_counts": dict(sorted(interpretation_counts.items())),
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "canonical_contract": manifest.get("canonical_contract"),
        **classification,
    }
    if not valid:
        result.update(
            {
                "evidence_tier": "invalid_or_incomplete",
                "audit_ready": False,
                "metric_specific_ci_ready": False,
                "canonical_gate_ready": False,
                "safe_wording": "Do not use this robustness package; its artifact or provenance contract is invalid.",
            }
        )
    return result


def _selection_key(result: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
    return (
        int(bool(result.get("valid"))),
        int(bool(result.get("canonical_gate_ready"))),
        int(bool(result.get("metric_specific_ci_ready"))),
        len(result.get("stresses") or []),
        len(result.get("metrics") or []),
        str(result.get("profile") or ""),
    )


def select_best_profile(
    revision_root: Path,
    *,
    canonical_contract: dict[str, str],
    runner_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    candidates = [
        validate_profile(
            revision_root,
            profile,
            canonical_contract=canonical_contract,
            runner_path=runner_path,
            project_root=project_root,
        )
        for profile in discover_profiles(revision_root)
    ]
    valid = [item for item in candidates if item.get("valid")]
    if not valid:
        return {
            "status": "not_available_or_invalid",
            "complete": False,
            "selected_profile": None,
            "selected": {},
            "candidates": candidates,
            "key_numbers": "learned_comparator_robustness=not_available_or_invalid",
            "safe_wording": "Do not claim learned-comparator robustness; no authenticated profile is available.",
            "blocker": "Run Notebook 05 or repair the profile artifact/provenance contract.",
        }

    selected = max(valid, key=_selection_key)
    interpretations = selected.get("interpretation_counts") or {}
    interpretation_text = ",".join(f"{key}:{value}" for key, value in interpretations.items())
    return {
        "status": selected["evidence_tier"],
        "complete": True,
        "selected_profile": selected["profile"],
        "selected": selected,
        "candidates": candidates,
        "key_numbers": (
            f"learned_comparator_robustness_profile={selected['profile']}; "
            f"tier={selected['evidence_tier']}; n_boot={selected['n_boot']}; "
            f"stresses={','.join(selected['stresses'])}; metrics={','.join(selected['metrics'])}; "
            f"rows={selected['completed_rows']}; interpretations={interpretation_text or 'none'}"
        ),
        "safe_wording": selected["safe_wording"],
        "blocker": (
            "The canonical six-stress, five-metric, n_boot=1000 robustness gate is incomplete."
            if not selected.get("canonical_gate_ready")
            else ""
        ),
    }
