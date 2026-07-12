"""Claim readiness gates for reviewer-sensitive ECG-RAMBA claims.

This lightweight audit writes explicit blocker/status artifacts for claims that
cannot be inferred from the main frozen OOF evaluation. It is intentionally
conservative: a row is marked complete only when the required downstream
artifacts already exist. Otherwise it records the missing evidence and the
manuscript-safe wording.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    TABLE_DIR,
    ensure_revision_dirs,
    git_commit,
    save_csv,
    save_json_atomic,
    sha256_file,
)


ROBUSTNESS_PROTOCOL = "robustness_multicomparator_aggregation_v1"
ROBUSTNESS_STRESSES = [
    "snr20db",
    "snr10db",
    "snr5db",
    "random_3_lead_dropout",
    "precordial_dropout",
    "resample_250hz",
]
ROBUSTNESS_COMPARATORS = ["full", "minirocket", "resnet", "raw_mamba", "transformer"]
ROBUSTNESS_PAIRED_COMPARATORS = ["minirocket", "resnet", "raw_mamba", "transformer"]
ROBUSTNESS_LEARNED_COMPARATORS = ["resnet", "raw_mamba", "transformer"]
ROBUSTNESS_METRICS = ["pr_auc_macro", "roc_auc_macro", "f1_macro", "brier_macro", "ece_macro"]
ROBUSTNESS_N_BOOT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "claim_readiness_gates.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_claim_readiness_gates.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "claim_readiness_gates_manifest.json",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    return resolve(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def present(path: Path) -> bool:
    candidate = resolve(path)
    return candidate.exists() and candidate.stat().st_size > 0


def status_from_required(required: list[Path], complete_status: str, blocked_status: str) -> tuple[str, list[str]]:
    missing = [p.as_posix() for p in required if not present(p)]
    return (complete_status, missing) if not missing else (blocked_status, missing)


def read_json_if_present(path: Path) -> dict:
    candidate = resolve(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def nested(payload: dict, *keys: str):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def canonical_contract() -> dict[str, str]:
    oof = PROJECT_ROOT / "reports" / "revision" / "predictions" / "oof_final_ema_predictions.npz"
    freeze = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
    if not present(oof) or not present(freeze):
        raise FileNotFoundError("Canonical final_ema OOF predictions and freeze manifest are required.")
    freeze_payload = read_json_if_present(freeze)
    if (
        freeze_payload.get("status") != "frozen"
        or freeze_payload.get("manuscript_ready") is not True
        or freeze_payload.get("checkpoint_kind") != "final_ema"
    ):
        raise RuntimeError("Canonical freeze manifest is not frozen/manuscript_ready final_ema evidence.")
    oof_sha = sha256_file(oof)
    frozen_oof_rows = [
        row
        for row in freeze_payload.get("artifacts") or []
        if str(row.get("path", "")).replace("\\", "/").endswith(f"/{oof.name}")
    ]
    if len(frozen_oof_rows) != 1 or frozen_oof_rows[0].get("sha256") != oof_sha:
        raise RuntimeError("Canonical freeze manifest does not bind the current final_ema OOF prediction SHA256.")
    return {"oof_sha256": oof_sha, "freeze_sha256": sha256_file(freeze)}


def contract_matches(actual: object, expected: dict[str, str]) -> bool:
    """Accept an extended contract only when every canonical key matches exactly."""

    return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())


def baseline_contract_issues(
    summary_path: Path,
    paired_path: Path,
    *,
    protocol: str,
    contract: dict[str, str],
) -> list[str]:
    """Validate provenance, not just existence, for an OOF comparator row."""
    issues: list[str] = []
    summary = read_json_if_present(summary_path)
    paired = read_json_if_present(paired_path)
    if summary.get("_read_error"):
        return [f"summary_unreadable={summary['_read_error']}"]
    if paired.get("_read_error"):
        return [f"paired_unreadable={paired['_read_error']}"]
    if summary and summary.get("manuscript_ready") is not True:
        issues.append("summary.manuscript_ready!=true")
    if summary and summary.get("protocol") != protocol:
        issues.append(f"summary.protocol={summary.get('protocol')!r}")
    if contract and summary:
        if nested(summary, "load_info", "oof_predictions_sha256") != contract["oof_sha256"]:
            issues.append("summary.load_info.oof_predictions_sha256_mismatch")
        if nested(summary, "load_info", "freeze_contract", "freeze_manifest_sha256") != contract["freeze_sha256"]:
            issues.append("summary.load_info.freeze_contract.freeze_manifest_sha256_mismatch")
    if paired and paired.get("status") not in (True, "complete"):
        issues.append(f"paired.status={paired.get('status')!r}")
    if contract and paired:
        if nested(paired, "inputs", "full_predictions", "sha256") != contract["oof_sha256"]:
            issues.append("paired.inputs.full_predictions.sha256_mismatch")
        if nested(paired, "inputs", "freeze_manifest", "sha256") != contract["freeze_sha256"]:
            issues.append("paired.inputs.freeze_manifest.sha256_mismatch")
    return issues


def manifest_contract_issues(
    path: Path,
    *,
    expected_status: str | bool | None = None,
    expected_protocol: str | None = None,
    canonical: dict[str, str] | None = None,
) -> list[str]:
    payload = read_json_if_present(path)
    if not payload:
        return []
    if payload.get("_read_error"):
        return [f"manifest_unreadable={payload['_read_error']}"]
    issues: list[str] = []
    if expected_status is not None and payload.get("status") != expected_status:
        issues.append(f"manifest.status={payload.get('status')!r}")
    if expected_protocol is not None and payload.get("protocol") != expected_protocol:
        issues.append(f"manifest.protocol={payload.get('protocol')!r}")
    if canonical and not contract_matches(payload.get("canonical_contract"), canonical):
        issues.append("manifest.canonical_contract_mismatch")
    return issues


def manifest_runner_issues(path: Path, runner_name: str) -> list[str]:
    payload = read_json_if_present(path)
    if not payload or payload.get("_read_error"):
        return []
    runner = PROJECT_ROOT / "scripts" / "revision" / runner_name
    if not runner.exists():
        return [f"runner_missing={runner_name}"]
    expected = sha256_file(runner)
    if payload.get("runner_sha256") != expected:
        return [f"manifest.runner_sha256_mismatch={runner_name}"]
    return []


def read_csv_if_present(path: Path) -> list[dict[str, str]]:
    candidate = resolve(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return []
    try:
        with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def int_or_zero(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def robustness_contract_issues(
    *,
    manifest_path: Path,
    pairwise_path: Path,
    summary_path: Path,
    table_path: Path,
    sidecar_paths: dict[str, Path],
    canonical: dict[str, str],
) -> list[str]:
    """Require the canonical 6-stress/5-metric/1000-bootstrap ledger."""

    issues: list[str] = []
    manifest = read_json_if_present(manifest_path)
    pairwise = read_json_if_present(pairwise_path)
    runner = PROJECT_ROOT / "scripts" / "revision" / "21_robustness_multicomparator.py"
    runner_sha = sha256_file(runner) if runner.exists() else None
    expected_rows = len(ROBUSTNESS_STRESSES) * len(ROBUSTNESS_PAIRED_COMPARATORS) * len(ROBUSTNESS_METRICS)

    for label, payload in [("manifest", manifest), ("pairwise", pairwise)]:
        if not payload:
            continue
        if payload.get("_read_error"):
            issues.append(f"{label}_unreadable")
            continue
        if payload.get("status") != "complete":
            issues.append(f"{label}.status={payload.get('status')!r}")
        if payload.get("protocol") != ROBUSTNESS_PROTOCOL:
            issues.append(f"{label}.protocol={payload.get('protocol')!r}")
        if payload.get("output_profile") != "canonical":
            issues.append(f"{label}.output_profile={payload.get('output_profile')!r}")
        if not contract_matches(payload.get("canonical_contract"), canonical):
            issues.append(f"{label}.canonical_contract_mismatch")
        if runner_sha is None or payload.get("runner_sha256") != runner_sha:
            issues.append(f"{label}.runner_sha256_mismatch")
        if list(payload.get("comparators") or []) != ROBUSTNESS_COMPARATORS:
            issues.append(f"{label}.comparators_incomplete")
        if list(payload.get("stress_tests") or []) != ROBUSTNESS_STRESSES:
            issues.append(f"{label}.stress_tests_incomplete")
        if list(payload.get("metrics") or []) != ROBUSTNESS_METRICS:
            issues.append(f"{label}.metrics_incomplete")
        if int_or_zero(payload.get("n_boot")) != ROBUSTNESS_N_BOOT:
            issues.append(f"{label}.n_boot={payload.get('n_boot')!r}")
        if int_or_zero(payload.get("blocked_rows")) != 0:
            issues.append(f"{label}.blocked_rows={payload.get('blocked_rows')!r}")
        if int_or_zero(payload.get("completed_rows")) != expected_rows:
            issues.append(f"{label}.completed_rows={payload.get('completed_rows')!r}")

    if pairwise and len(pairwise.get("items") or {}) != expected_rows:
        issues.append(f"pairwise.items_count={len(pairwise.get('items') or {})}")

    expected_pairwise_rel = rel(pairwise_path)
    pairwise_sha = sha256_file(resolve(pairwise_path)) if present(pairwise_path) else ""
    for comparator, path in sidecar_paths.items():
        payload = read_json_if_present(path)
        label = path.name
        if not payload:
            continue
        if payload.get("status") != "complete":
            issues.append(f"{label}:status={payload.get('status')!r}")
        if payload.get("protocol") != ROBUSTNESS_PROTOCOL:
            issues.append(f"{label}:protocol_mismatch")
        if payload.get("comparator") != comparator:
            issues.append(f"{label}:comparator_mismatch")
        if payload.get("output_profile") != "canonical":
            issues.append(f"{label}:not_canonical")
        if not contract_matches(payload.get("canonical_contract"), canonical):
            issues.append(f"{label}:canonical_contract_mismatch")
        if runner_sha is None or payload.get("runner_sha256") != runner_sha:
            issues.append(f"{label}:runner_sha256_mismatch")
        if payload.get("source_pairwise") != expected_pairwise_rel:
            issues.append(f"{label}:source_pairwise_mismatch")
        if payload.get("source_pairwise_sha256") != pairwise_sha:
            issues.append(f"{label}:source_pairwise_sha256_mismatch")
        if list(payload.get("stress_tests") or []) != ROBUSTNESS_STRESSES:
            issues.append(f"{label}:stress_tests_incomplete")
        if list(payload.get("metrics") or []) != ROBUSTNESS_METRICS:
            issues.append(f"{label}:metrics_incomplete")
        if int_or_zero(payload.get("n_boot")) != ROBUSTNESS_N_BOOT:
            issues.append(f"{label}:n_boot={payload.get('n_boot')!r}")
        expected_sidecar_rows = len(ROBUSTNESS_STRESSES) * len(ROBUSTNESS_METRICS)
        if int_or_zero(payload.get("blocked_rows")) != 0:
            issues.append(f"{label}:blocked_rows={payload.get('blocked_rows')!r}")
        if int_or_zero(payload.get("completed_rows")) != expected_sidecar_rows:
            issues.append(f"{label}:completed_rows={payload.get('completed_rows')!r}")
        if len(payload.get("rows") or []) != expected_sidecar_rows:
            issues.append(f"{label}:rows_count={len(payload.get('rows') or [])}")

    for label, path in [("summary", summary_path), ("table", table_path)]:
        rows = read_csv_if_present(path)
        if not rows:
            continue
        if len(rows) != expected_rows:
            issues.append(f"{label}.rows={len(rows)}")
        if any(row.get("status") != "complete" for row in rows):
            issues.append(f"{label}.contains_noncomplete_rows")
        if {row.get("stress") for row in rows} != set(ROBUSTNESS_STRESSES):
            issues.append(f"{label}.stress_set_mismatch")
        if {row.get("comparator") for row in rows} != set(ROBUSTNESS_PAIRED_COMPARATORS):
            issues.append(f"{label}.comparator_set_mismatch")
        if {row.get("metric") for row in rows} != set(ROBUSTNESS_METRICS):
            issues.append(f"{label}.metric_set_mismatch")
        if any(row.get("output_profile") != "canonical" for row in rows):
            issues.append(f"{label}.contains_noncanonical_rows")
        if any(int_or_zero(row.get("n_boot")) != ROBUSTNESS_N_BOOT for row in rows):
            issues.append(f"{label}.n_boot_mismatch")

    artifact_sha = manifest.get("artifact_sha256") if isinstance(manifest, dict) else {}
    expected_hashes = {
        "summary": sha256_file(resolve(summary_path)) if present(summary_path) else "",
        "table": sha256_file(resolve(table_path)) if present(table_path) else "",
        "pairwise": pairwise_sha,
    }
    for key, value in expected_hashes.items():
        if value and (artifact_sha or {}).get(key) != value:
            issues.append(f"manifest.artifact_sha256.{key}_mismatch")
    sidecar_hashes = (artifact_sha or {}).get("comparator_sidecars") or {}
    for comparator, path in sidecar_paths.items():
        if present(path) and sidecar_hashes.get(comparator) != sha256_file(resolve(path)):
            issues.append(f"manifest.artifact_sha256.sidecar_{comparator}_mismatch")
    return issues


def external_comparator_manifest_issues(
    manifests: list[Path],
    *,
    canonical: dict[str, str],
) -> list[str]:
    """Check external comparator provenance against the currently frozen Chapman OOF."""

    issues: list[str] = []
    runner = PROJECT_ROOT / "scripts" / "revision" / "31_generate_external_comparator_predictions.py"
    runner_sha = sha256_file(runner) if runner.exists() else None
    for path in manifests:
        payload = read_json_if_present(path)
        label = path.name
        if not payload:
            continue
        if payload.get("_read_error"):
            issues.append(f"{label}:unreadable")
            continue
        if payload.get("status") != "complete_experimental_requires_external_comparator_gate":
            issues.append(f"{label}:status={payload.get('status')!r}")
        if canonical and not contract_matches(payload.get("canonical_contract"), canonical):
            issues.append(f"{label}:canonical_oof_freeze_mismatch")
        source = payload.get("source_contract") if isinstance(payload.get("source_contract"), dict) else {}
        if not source.get("archive_sha256") or not source.get("runner_sha256"):
            issues.append(f"{label}:missing_source_contract")
        elif runner_sha and source.get("runner_sha256") != runner_sha:
            issues.append(f"{label}:runner_sha_mismatch")
    return issues


def complete_if_valid(
    *,
    required: list[Path],
    complete_status: str,
    blocked_status: str,
    contract_issues: list[str],
) -> tuple[str, list[str]]:
    status, issues = status_from_required(required, complete_status, blocked_status)
    issues.extend(f"contract:{item}" for item in contract_issues)
    if contract_issues:
        status = blocked_status
    return status, issues


def row(
    *,
    claim_id: str,
    claim_area: str,
    status: str,
    manuscript_ready: bool,
    evidence_status: str,
    required_artifacts: list[Path],
    missing_artifacts: list[str] | None,
    safe_wording: str,
    blocker: str,
    next_action: str,
) -> dict:
    existing = [p.as_posix() for p in required_artifacts if present(p)]
    return {
        "claim_id": claim_id,
        "claim_area": claim_area,
        "status": status,
        "manuscript_ready": bool(manuscript_ready),
        "evidence_status": evidence_status,
        "required_artifacts": ";".join(p.as_posix() for p in required_artifacts),
        "existing_artifacts": ";".join(existing),
        "missing_artifacts": ";".join(missing_artifacts or []),
        "safe_wording": safe_wording,
        "blocker": blocker,
        "next_action": next_action,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    save_csv(resolve(path), rows)


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    canonical = canonical_contract()
    print("=" * 80, flush=True)

    method_identity_required = [
        TABLE_DIR / "table_morphology_transform_contract.csv",
        MANIFEST_DIR / "morphology_transform_identity_gate.json",
        MANIFEST_DIR / "reviewer_completion_input_contract.json",
    ]
    method_identity_payload = read_json_if_present(method_identity_required[1])
    method_identity_issues = manifest_contract_issues(
        method_identity_required[1], expected_status="complete", canonical=canonical
    )
    if method_identity_payload and method_identity_payload.get("manuscript_ready") is not True:
        method_identity_issues.append("manifest.manuscript_ready!=true")
    method_identity_status, method_identity_missing = complete_if_valid(
        required=method_identity_required,
        complete_status="complete_custom_transform_identified",
        blocked_status="blocked_morphology_transform_identity_not_audited",
        contract_issues=method_identity_issues,
    )

    presentation_required = [
        Path("reports/revision/figures/figure_calibration_audit.png"),
        TABLE_DIR / "table_calibration_ci_compact.csv",
        TABLE_DIR / "table_paired_baseline_ci_compact.csv",
        TABLE_DIR / "table_pooling_sensitivity_compact.csv",
        TABLE_DIR / "table_fold_pca_provenance.csv",
        TABLE_DIR / "table_training_configuration.csv",
    ]
    presentation_status, presentation_missing = complete_if_valid(
        required=presentation_required,
        complete_status="complete_reviewer_presentation_assets_available",
        blocked_status="blocked_reviewer_presentation_assets_missing",
        contract_issues=manifest_contract_issues(
            MANIFEST_DIR / "reviewer_completion_input_contract.json",
            expected_status=True,
            canonical=canonical,
        )
        + manifest_runner_issues(
            MANIFEST_DIR / "reviewer_completion_input_contract.json",
            "29_reviewer_presentation_assets.py",
        ),
    )
    print("CLAIM READINESS GATES", flush=True)
    print("=" * 80, flush=True)

    transformer_required = [
        METRIC_DIR / "transformer_ecg_baseline_summary.json",
        Path("reports/revision/predictions/transformer_ecg_oof_predictions.npz"),
        MANIFEST_DIR / "transformer_ecg_baseline_manifest.json",
        METRIC_DIR / "paired_full_vs_transformer_comparison.json",
        TABLE_DIR / "table_paired_full_vs_transformer.csv",
        MANIFEST_DIR / "paired_full_vs_transformer_manifest.json",
    ]
    transformer_status, transformer_missing = complete_if_valid(
        required=transformer_required,
        complete_status="complete_optional_comparator_available",
        blocked_status="not_run_or_stale_optional_comparator",
        contract_issues=baseline_contract_issues(
            transformer_required[0],
            transformer_required[3],
            protocol="transformer_ecg_raw_same_folds_power_mean_v2_q3_threshold_0.5",
            contract=canonical,
        ),
    )

    hybrid_required = [
        METRIC_DIR / "hybrid_morphology_baseline_summary.json",
        Path("reports/revision/predictions/hybrid_morphology_oof_predictions.npz"),
        MANIFEST_DIR / "hybrid_morphology_baseline_manifest.json",
        METRIC_DIR / "paired_full_vs_hybrid_morphology_comparison.json",
        TABLE_DIR / "table_paired_full_vs_hybrid_morphology.csv",
        MANIFEST_DIR / "paired_full_vs_hybrid_morphology_manifest.json",
    ]
    hybrid_status, hybrid_missing = complete_if_valid(
        required=hybrid_required,
        complete_status="complete_optional_morphology_sensitivity_available",
        blocked_status="not_run_or_stale_optional_morphology_sensitivity",
        contract_issues=baseline_contract_issues(
            hybrid_required[0],
            hybrid_required[3],
            protocol="fixed_seed_rocket_family_max_ppv_mlp_head_same_folds_threshold_0.5",
            contract=canonical,
        ),
    )

    robustness_required = [
        METRIC_DIR / "robustness_full_vs_resnet_comparison.json",
        METRIC_DIR / "robustness_full_vs_raw_mamba_comparison.json",
        METRIC_DIR / "robustness_full_vs_transformer_comparison.json",
        METRIC_DIR / "robustness_multicomparator_pairwise.json",
        METRIC_DIR / "robustness_multicomparator_summary.csv",
        TABLE_DIR / "table_robustness_multicomparator.csv",
        MANIFEST_DIR / "robustness_multicomparator_manifest.json",
    ]
    robustness_manifest = MANIFEST_DIR / "robustness_multicomparator_manifest.json"
    robustness_sidecars = {
        "resnet": METRIC_DIR / "robustness_full_vs_resnet_comparison.json",
        "raw_mamba": METRIC_DIR / "robustness_full_vs_raw_mamba_comparison.json",
        "transformer": METRIC_DIR / "robustness_full_vs_transformer_comparison.json",
    }
    robustness_status, robustness_missing = complete_if_valid(
        required=robustness_required,
        complete_status="complete_multicomparator_robustness_available",
        blocked_status="blocked_missing_or_stale_learned_comparator_stress_evidence",
        contract_issues=robustness_contract_issues(
            manifest_path=robustness_manifest,
            pairwise_path=METRIC_DIR / "robustness_multicomparator_pairwise.json",
            summary_path=METRIC_DIR / "robustness_multicomparator_summary.csv",
            table_path=TABLE_DIR / "table_robustness_multicomparator.csv",
            sidecar_paths=robustness_sidecars,
            canonical=canonical,
        ),
    )

    representation_required = [
        METRIC_DIR / "representation_evidence_status.json",
        METRIC_DIR / "representation_probe_summary.json",
        TABLE_DIR / "table_representation_probe.csv",
        TABLE_DIR / "table_representation_probe_by_fold.csv",
        TABLE_DIR / "table_representation_cka.csv",
        Path("reports/revision/figures/figure_representation_audit.png"),
        MANIFEST_DIR / "representation_probe_manifest.json",
    ]
    representation_status, representation_missing = complete_if_valid(
        required=representation_required,
        complete_status="audit_available_not_mechanistic_proof",
        blocked_status="blocked_missing_or_stale_representation_audit",
        contract_issues=manifest_contract_issues(
            MANIFEST_DIR / "representation_probe_manifest.json",
            expected_status="complete",
            expected_protocol="representation_probe_fold_safe_v3_projection_and_fold_audit",
            canonical=canonical,
        )
        + manifest_runner_issues(
            MANIFEST_DIR / "representation_probe_manifest.json",
            "20_representation_probe.py",
        ),
    )

    group_safe_calibration_required = [
        METRIC_DIR / "group_safe_score_calibration_ptbxl_summary.csv",
        TABLE_DIR / "table_group_safe_score_calibration_ptbxl.csv",
        METRIC_DIR / "group_safe_score_calibration_ptbxl_bootstrap.json",
        MANIFEST_DIR / "group_safe_score_calibration_ptbxl_manifest.json",
    ]
    group_safe_calibration_status, group_safe_calibration_missing = complete_if_valid(
        required=group_safe_calibration_required,
        complete_status="complete_group_safe_score_calibration",
        blocked_status="not_run_or_stale_group_safe_score_calibration",
        contract_issues=manifest_contract_issues(
            group_safe_calibration_required[-1],
            expected_status="complete_group_safe_score_calibration",
            expected_protocol="group_safe_score_calibration_v2_gated_external",
            canonical=canonical,
        )
        + manifest_runner_issues(
            group_safe_calibration_required[-1], "33_group_safe_score_calibration.py"
        ),
    )

    true_fewshot_required = [
        METRIC_DIR / "true_fewshot_head_ptbxl_summary.csv",
        TABLE_DIR / "table_true_fewshot_head_ptbxl.csv",
        TABLE_DIR / "table_true_fewshot_head_ptbxl_paired.csv",
        METRIC_DIR / "true_fewshot_head_ptbxl_bootstrap.json",
        TABLE_DIR / "table_true_fewshot_head_ptbxl_coefficients.csv",
        MANIFEST_DIR / "true_fewshot_head_ptbxl_splits.npz",
        MANIFEST_DIR / "true_fewshot_head_ptbxl_manifest.json",
    ]
    true_fewshot_status, true_fewshot_missing = complete_if_valid(
        required=true_fewshot_required,
        complete_status="complete_true_fewshot_frozen_encoder_head_adaptation",
        blocked_status="not_run_or_stale_true_fewshot_head_adaptation",
        contract_issues=manifest_contract_issues(
            true_fewshot_required[-1],
            expected_status="complete_true_classifier_head_adaptation",
            expected_protocol="frozen_encoder_true_linear_head_adaptation_v2_group_safe_gated",
            canonical=canonical,
        )
        + manifest_runner_issues(
            true_fewshot_required[-1], "35_true_fewshot_head_adaptation.py"
        ),
    )

    external_comparator_required = [
        METRIC_DIR / "external_comparator_paired_summary.json",
        TABLE_DIR / "table_external_comparator_paired.csv",
        MANIFEST_DIR / "external_comparator_paired_manifest.json",
        MANIFEST_DIR / "external_ptbxl_resnet1d_cnn_manifest.json",
        MANIFEST_DIR / "external_ptbxl_raw_mamba_manifest.json",
        MANIFEST_DIR / "external_georgia_resnet1d_cnn_manifest.json",
        MANIFEST_DIR / "external_georgia_raw_mamba_manifest.json",
    ]
    external_comparator_contract_issues = (
        manifest_contract_issues(
            MANIFEST_DIR / "external_comparator_paired_manifest.json",
            expected_status="complete",
            canonical=canonical,
        )
        + manifest_runner_issues(
            MANIFEST_DIR / "external_comparator_paired_manifest.json",
            "32_paired_external_comparators.py",
        )
        + external_comparator_manifest_issues(external_comparator_required[3:], canonical=canonical)
    )
    external_comparator_status, external_comparator_missing = complete_if_valid(
        required=external_comparator_required,
        complete_status="complete_external_learned_comparator_audit",
        blocked_status="not_run_or_stale_external_learned_comparator_audit",
        contract_issues=external_comparator_contract_issues,
    )

    marked_manuscript_required = [MANIFEST_DIR / "marked_manuscript_manifest.json"]
    marked_payload = read_json_if_present(marked_manuscript_required[0])
    marked_pdf = Path(str(nested(marked_payload, "outputs", "marked_pdf", "path") or ""))
    if marked_pdf and marked_pdf.as_posix() not in (".", ""):
        marked_manuscript_required.append(marked_pdf)
    marked_contract_issues = manifest_contract_issues(
        marked_manuscript_required[0], expected_status="complete_marked_manuscript"
    )
    if marked_payload and marked_payload.get("editorial_ready") is not True:
        marked_contract_issues.append("manifest.editorial_ready!=true")
    marked_status, marked_missing = complete_if_valid(
        required=marked_manuscript_required,
        complete_status="complete_marked_manuscript_pdf",
        blocked_status="blocked_marked_manuscript_pdf_not_built",
        contract_issues=marked_contract_issues,
    )

    rows = [
        row(
            claim_id="morphology_transform_identity",
            claim_area="Morphology transform identity and PCA input dimension",
            status=method_identity_status,
            manuscript_ready=method_identity_status.startswith("complete"),
            evidence_status="method_contract",
            required_artifacts=method_identity_required,
            missing_artifacts=method_identity_missing,
            safe_wording=(
                "Describe the evaluated branch as a fixed-seed ROCKET-family random-convolution "
                "MAX+PPV transform (10,000 requested kernels, 20,000 outputs), not canonical MiniRocket."
            ),
            blocker="The evaluated transform has not passed the method-identity gate." if method_identity_missing else "",
            next_action="Run scripts/revision/29_reviewer_presentation_assets.py after restoring canonical artifacts.",
        ),
        row(
            claim_id="reviewer_presentation_assets",
            claim_area="Reliability, CI, Q=3, PCA, and training appendix assets",
            status=presentation_status,
            manuscript_ready=presentation_status.startswith("complete"),
            evidence_status="presentation_only",
            required_artifacts=presentation_required,
            missing_artifacts=presentation_missing,
            safe_wording="Use generated tables/figure directly; do not type numeric values independently.",
            blocker="Reviewer-facing presentation assets are incomplete." if presentation_missing else "",
            next_action="Run scripts/revision/29_reviewer_presentation_assets.py --strict after paired artifacts are current.",
        ),
        row(
            claim_id="transformer_ecg_baseline",
            claim_area="Compact Transformer ECG fair comparator",
            status=transformer_status,
            manuscript_ready=transformer_status.startswith("complete"),
            evidence_status="optional_comparator_specific",
            required_artifacts=transformer_required,
            missing_artifacts=transformer_missing,
            safe_wording=(
                "Use only comparator-specific wording if the compact Transformer ECG baseline and paired "
                "bootstrap artifacts are complete. It is trained from scratch and is not a foundation model."
            ),
            blocker="Missing Transformer ECG baseline and/or paired Full-vs-Transformer artifacts."
            if transformer_missing
            else "",
            next_action=(
                "Run scripts/revision/24_transformer_ecg_baseline.py, then "
                "scripts/revision/25_paired_full_vs_transformer.py under the frozen OOF contract."
            ),
        ),
        row(
            claim_id="hybrid_morphology_mlp",
            claim_area="Frozen ROCKET-family random-convolution MLP-head sensitivity",
            status=hybrid_status,
            manuscript_ready=hybrid_status.startswith("complete"),
            evidence_status="optional_mechanism_sensitivity",
            required_artifacts=hybrid_required,
            missing_artifacts=hybrid_missing,
            safe_wording=(
                "Use only as a fixed-seed ROCKET-family transform-head sensitivity control. The MLP does "
                "not make convolution kernels learnable and cannot isolate deterministic kernels from "
                "regularization, optimization, or head-capacity effects."
            ),
            blocker="Missing frozen-transform MLP-head baseline and/or paired comparison artifacts."
            if hybrid_missing
            else "",
            next_action=(
                "Run scripts/revision/26_hybrid_morphology_baseline.py, then "
                "scripts/revision/27_paired_full_vs_hybrid_morphology.py."
            ),
        ),
        row(
            claim_id="fewshot_score_calibration_v1",
            claim_area="Existing row-split external score calibration",
            status="blocked_not_group_safe",
            manuscript_ready=False,
            evidence_status="exploratory_provenance_only",
            required_artifacts=[],
            missing_artifacts=["group_id", "zero_group_overlap_audit", "group_cluster_bootstrap"],
            safe_wording=(
                "Do not call the v1 score-calibration analysis leakage-audited or true few-shot adaptation. "
                "It permutes prediction rows and leaves ECG-RAMBA weights unchanged."
            ),
            blocker="Repeated patient/source-record observations are not kept in one split.",
            next_action=(
                "Regenerate external predictions with group_id and run the group-safe v2 calibration/adaptation runner."
            ),
        ),
        row(
            claim_id="group_safe_score_calibration_v2",
            claim_area="PTB-XL group-safe score calibration on frozen predictions",
            status=group_safe_calibration_status,
            manuscript_ready=group_safe_calibration_status.startswith("complete"),
            evidence_status="calibration_only_not_weight_adaptation",
            required_artifacts=group_safe_calibration_required,
            missing_artifacts=group_safe_calibration_missing,
            safe_wording=(
                "If complete, describe this only as group-safe, dataset-specific score calibration of "
                "frozen predictions. It changes decision scores/threshold behavior, not encoder or classifier weights."
            ),
            blocker=(
                "PTB-XL group-safe score-calibration artifacts are absent, stale, or fail their protocol contract."
                if group_safe_calibration_missing
                else ""
            ),
            next_action=(
                "Run scripts/revision/33_group_safe_score_calibration.py after the PTB-XL protocol gate, "
                "then report F1 and ranking metrics separately."
            ),
        ),
        row(
            claim_id="true_fewshot_frozen_encoder_head_adaptation",
            claim_area="PTB-XL true few-shot frozen-encoder classifier-head adaptation",
            status=true_fewshot_status,
            manuscript_ready=true_fewshot_status.startswith("complete"),
            evidence_status="parameter_adaptation_not_end_to_end_finetuning",
            required_artifacts=true_fewshot_required,
            missing_artifacts=true_fewshot_missing,
            safe_wording=(
                "If complete, report PTB-XL results as group-safe adaptation of new linear classifier heads on "
                "frozen Chapman-trained encoders. This is parameter adaptation, but not end-to-end fine-tuning "
                "and not evidence of general few-shot superiority."
            ),
            blocker=(
                "No complete group-safe frozen-encoder head-adaptation package is available for PTB-XL."
                if true_fewshot_missing
                else ""
            ),
            next_action=(
                "Generate fold-specific external representations and paired external comparator evidence, then run "
                "scripts/revision/35_true_fewshot_head_adaptation.py for PTB-XL official folds 9/10."
            ),
        ),
        row(
            claim_id="external_learned_comparator_audit",
            claim_area="PTB-XL/Georgia learned-comparator zero-target-label audit",
            status=external_comparator_status,
            manuscript_ready=external_comparator_status.startswith("complete"),
            evidence_status="dataset_specific_mapped_task_only",
            required_artifacts=external_comparator_required,
            missing_artifacts=external_comparator_missing,
            safe_wording=(
                "If complete, report PTB-XL and Georgia separately as zero-target-label mapped-task comparisons "
                "against the named Chapman-trained comparators. Do not average them with CPSC2021 or infer broad "
                "external-transfer superiority."
            ),
            blocker=(
                "Required PTB-XL/Georgia comparator predictions or group-paired external comparisons are missing or stale."
                if external_comparator_missing
                else ""
            ),
            next_action=(
                "Run scripts/revision/31_generate_external_comparator_predictions.py for ResNet1D/CNN and Raw Mamba, "
                "then scripts/revision/32_paired_external_comparators.py. Run Transformer only after its in-domain gate completes."
            ),
        ),
        row(
            claim_id="robustness_learned_comparators",
            claim_area="Robustness vs learned comparators",
            status=robustness_status,
            manuscript_ready=robustness_status.startswith("complete"),
            evidence_status="metric_specific_if_complete",
            required_artifacts=robustness_required,
            missing_artifacts=robustness_missing,
            safe_wording=(
                "Use learned-comparator robustness only when the complete paired stress ledger is current. "
                "All robustness wording remains metric-, stress-, and comparator-specific."
            ),
            blocker="Missing or stale ResNet1D/CNN, Raw Mamba, and/or multi-comparator stress evidence."
            if robustness_missing
            else "",
            next_action=(
                "Save/reuse learned-comparator checkpoints, run comparator stress predictions, then run "
                "scripts/revision/21_robustness_multicomparator.py."
            ),
        ),
        row(
            claim_id="full_hrv_feature_set",
            claim_area="Full HRV feature set",
            status="blocked_retrain_required",
            manuscript_ready=False,
            evidence_status="not_available_for_current_checkpoint",
            required_artifacts=[],
            missing_artifacts=["true_RMSSD_SDNN_LFHF_checkpoint_training_contract"],
            safe_wording=(
                "Do not claim RMSSD, SDNN, LF/HF, or a complete HRV feature set for the current final-EMA "
                "checkpoint. The current HRV slots are checkpoint-compatible and partly reserved."
            ),
            blocker="Full HRV semantics require a new feature schema and full retraining; they cannot be retrofitted.",
            next_action="Define a true HRV schema, retrain all folds, regenerate OOF/calibration/baselines, and re-freeze evidence.",
        ),
        row(
            claim_id="mechanistic_disentanglement",
            claim_area="Mechanistic morphology-rhythm disentanglement",
            status=representation_status,
            manuscript_ready=representation_status.startswith("audit_available"),
            evidence_status="audit_only",
            required_artifacts=representation_required,
            missing_artifacts=representation_missing,
            safe_wording=(
                "Representation/CKA results may be described only as a conservative branch-embedding audit. "
                "They do not prove morphology-rhythm disentanglement."
            ),
            blocker="" if not representation_missing else "Missing representation probe/CKA artifacts.",
            next_action=(
                "If stronger mechanism evidence is required, add preregistered probes with leakage-safe splits and "
                "keep wording as suggestive unless tests support stronger claims."
            ),
        ),
        row(
            claim_id="marked_highlighted_manuscript",
            claim_area="Editorial marked/highlighted manuscript PDF",
            status=marked_status,
            manuscript_ready=marked_status.startswith("complete"),
            evidence_status="editorial_deliverable",
            required_artifacts=marked_manuscript_required,
            missing_artifacts=marked_missing,
            safe_wording=(
                "Do not state that a marked manuscript has been supplied until latexdiff and LaTeX compilation "
                "produce the verified marked PDF and its manifest."
            ),
            blocker=(
                "The requested marked/highlighted manuscript PDF has not been built and verified."
                if marked_missing
                else ""
            ),
            next_action=(
                "Run scripts/revision/36_build_marked_manuscript.py in an environment with latexdiff and latexmk, "
                "then retain its manifest with the submission package."
            ),
        ),
        row(
            claim_id="clinical_deployment_readiness",
            claim_area="Clinical deployment/safety readiness",
            status="blocked_no_prospective_or_clinical_utility_validation",
            manuscript_ready=False,
            evidence_status="not_available",
            required_artifacts=[],
            missing_artifacts=["prospective_validation", "clinical_threshold_target", "decision_curve_or_utility_analysis"],
            safe_wording=(
                "Avoid clinical-use, safety-readiness, or prospective-utility wording. "
                "Current evidence is retrospective/model-evaluation evidence only."
            ),
            blocker="No prospective validation, clinical utility analysis, or prespecified deployment threshold target.",
            next_action="Plan a prospective or clinically curated external validation with utility/threshold analysis.",
        ),
        row(
            claim_id="broad_in_domain_global_superiority",
            claim_area="Broad in-domain/fair-baseline advantage",
            status="contradicted_by_current_fair_baselines",
            manuscript_ready=False,
            evidence_status="contradicted",
            required_artifacts=[
                METRIC_DIR / "paired_full_vs_resnet_comparison.json",
                METRIC_DIR / "paired_full_vs_raw_mamba_comparison.json",
            ],
            missing_artifacts=[],
            safe_wording=(
                "Avoid performance-leading, best-in-domain, or broad fair-baseline advantage wording. "
                "Use metric-specific and comparator-specific wording only."
            ),
            blocker="ResNet1D/CNN and Raw Mamba outperform ECG-RAMBA on multiple principal frozen OOF metrics.",
            next_action="Keep the manuscript framed as structured analysis and evidence-bounded tradeoffs, not broad performance advantage.",
        ),
    ]

    out_table = resolve(args.out_table)
    out_json = resolve(args.out_json)
    out_manifest = resolve(args.out_manifest)
    write_csv(out_table, rows)
    payload = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "rows": rows,
        "claim_guidance": {
            "use": "Use this table to keep manuscript and rebuttal wording bounded by completed evidence.",
            "do_not_use": "Do not convert blocked rows into positive claims.",
        },
    }
    save_json_atomic(out_json, payload)
    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "artifacts": {
            "json": rel(out_json),
            "table": rel(out_table),
        },
        "artifact_sha256": {
            "json": sha256_file(out_json),
            "table": sha256_file(out_table),
        },
    }
    save_json_atomic(out_manifest, manifest)
    print(json.dumps({"status": True, "rows": len(rows), "table": rel(out_table)}, indent=2), flush=True)
    print(f"Wrote: {out_json}", flush=True)
    print(f"Wrote: {out_table}", flush=True)
    print(f"Wrote: {out_manifest}", flush=True)


if __name__ == "__main__":
    main()
