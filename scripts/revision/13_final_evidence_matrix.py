"""Build the final reviewer evidence matrix from frozen revision artifacts.

This script does not recompute model predictions. It reads the artifacts
produced by notebooks 01-06, validates the key contracts, and writes compact
tables that can be used to draft the rebuttal/manuscript without overstating
the evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    REVISION_DIR,
    TABLE_DIR,
    ensure_revision_dirs,
    git_commit,
    save_csv,
    save_json,
    sha256_file,
)


REQUIRED_ROBUSTNESS_STRESSES = {
    "snr20db",
    "snr10db",
    "snr5db",
    "random_3_lead_dropout",
    "precordial_dropout",
    "resample_250hz",
}
REQUIRED_ROBUSTNESS_METRICS = {
    "pr_auc_macro",
    "roc_auc_macro",
    "f1_macro",
    "brier_macro",
    "ece_macro",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "final_evidence_matrix.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_final_evidence_matrix.csv",
    )
    parser.add_argument(
        "--out-safe-wording",
        type=Path,
        default=TABLE_DIR / "table_final_safe_wording.csv",
    )
    parser.add_argument(
        "--out-blockers",
        type=Path,
        default=TABLE_DIR / "table_final_blocker_status.csv",
    )
    parser.add_argument(
        "--out-robustness",
        type=Path,
        default=TABLE_DIR / "table_final_robustness_claims.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "final_evidence_matrix_manifest.json",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "" if not math.isfinite(value) else f"{value:.{digits}f}"


def csv_index(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {str(row.get(key, "")): row for row in rows}


def assert_robustness_contract(rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    stresses = {str(row.get("stress_test", "")) for row in rows}
    metrics = {str(row.get("metric", "")) for row in rows}
    if stresses != REQUIRED_ROBUSTNESS_STRESSES:
        issues.append(f"robustness stress set mismatch: {sorted(stresses)}")
    if metrics != REQUIRED_ROBUSTNESS_METRICS:
        issues.append(f"robustness metric set mismatch: {sorted(metrics)}")
    expected_rows = len(REQUIRED_ROBUSTNESS_STRESSES) * len(REQUIRED_ROBUSTNESS_METRICS)
    if len(rows) != expected_rows:
        issues.append(f"robustness row count {len(rows)} != {expected_rows}")
    return issues


def robustness_claim_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    claim_rows = []
    for row in rows:
        degradation = str(row.get("degradation_interpretation", ""))
        stressed = str(row.get("stressed_performance_interpretation", ""))
        metric = str(row.get("metric", ""))
        if degradation == "full_significantly_less_degraded":
            degradation_wording = "Full ECG-RAMBA degrades less for this stress/metric."
        elif degradation == "minirocket_significantly_less_degraded":
            degradation_wording = "MiniRocket-only degrades less for this stress/metric."
        else:
            degradation_wording = "No paired degradation advantage should be claimed."

        if stressed == "full_significantly_better_under_stress":
            stressed_wording = "Full ECG-RAMBA is better under the stressed operating point."
        elif stressed == "minirocket_significantly_better_under_stress":
            stressed_wording = "MiniRocket-only is better under the stressed operating point."
        else:
            stressed_wording = "No stressed-performance superiority should be claimed."

        claim_rows.append(
            {
                "stress_test": row.get("stress_test", ""),
                "metric": metric,
                "metric_family": row.get("metric_family", ""),
                "clean_full": row.get("clean_full", ""),
                "stress_full": row.get("stress_full", ""),
                "degradation_full": row.get("degradation_full", ""),
                "clean_minirocket": row.get("clean_minirocket", ""),
                "stress_minirocket": row.get("stress_minirocket", ""),
                "degradation_minirocket": row.get("degradation_minirocket", ""),
                "degradation_advantage_full_less_degradation": row.get(
                    "degradation_advantage_full_less_degradation", ""
                ),
                "degradation_ci_low": row.get("degradation_advantage_ci_low", ""),
                "degradation_ci_high": row.get("degradation_advantage_ci_high", ""),
                "stressed_advantage_full_over_minirocket": row.get(
                    "stressed_advantage_full_over_minirocket", ""
                ),
                "stressed_ci_low": row.get("stressed_advantage_ci_low", ""),
                "stressed_ci_high": row.get("stressed_advantage_ci_high", ""),
                "degradation_interpretation": degradation,
                "stressed_performance_interpretation": stressed,
                "safe_wording_degradation": degradation_wording,
                "safe_wording_stressed_performance": stressed_wording,
            }
        )
    return sorted(claim_rows, key=lambda r: (str(r["stress_test"]), str(r["metric"])))


def summarize_robustness(rows: list[dict[str, str]]) -> dict[str, Any]:
    full_less_degraded = [
        row
        for row in rows
        if row.get("degradation_interpretation") == "full_significantly_less_degraded"
    ]
    mini_less_degraded = [
        row
        for row in rows
        if row.get("degradation_interpretation") == "minirocket_significantly_less_degraded"
    ]
    full_better_stress = [
        row
        for row in rows
        if row.get("stressed_performance_interpretation")
        == "full_significantly_better_under_stress"
    ]
    mini_better_stress = [
        row
        for row in rows
        if row.get("stressed_performance_interpretation")
        == "minirocket_significantly_better_under_stress"
    ]
    return {
        "n_rows": len(rows),
        "full_less_degraded_count": len(full_less_degraded),
        "minirocket_less_degraded_count": len(mini_less_degraded),
        "full_better_under_stress_count": len(full_better_stress),
        "minirocket_better_under_stress_count": len(mini_better_stress),
        "full_less_degraded_metrics": [
            f"{row.get('stress_test')}:{row.get('metric')}" for row in full_less_degraded
        ],
        "minirocket_less_degraded_metrics": [
            f"{row.get('stress_test')}:{row.get('metric')}" for row in mini_less_degraded
        ],
    }


def artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "sha256": "", "size_bytes": 0}
    return {
        "path": rel(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()

    paths = {
        "calibration": METRIC_DIR / "calibration_ci_oof_final_ema_predictions.json",
        "pooling": METRIC_DIR / "pooling_sensitivity.csv",
        "baseline": METRIC_DIR / "baseline_summary.csv",
        "component": METRIC_DIR / "component_check_summary.json",
        "hrv_domain": METRIC_DIR / "hrv_domain_summary.csv",
        "robustness": METRIC_DIR / "robustness_summary.csv",
        "paired_minirocket": METRIC_DIR / "paired_full_vs_minirocket_comparison.json",
        "paired_resnet": METRIC_DIR / "paired_full_vs_resnet_comparison.json",
        "a0_status": REVISION_DIR / "a0_resolution_status.json",
        "claim_map": PROJECT_ROOT / "docs" / "revision_plan" / "claim_evidence_map.csv",
        "task_board": PROJECT_ROOT / "docs" / "revision_plan" / "task_board.csv",
    }
    optional_paths = {
        "paired_raw_mamba": METRIC_DIR / "paired_full_vs_raw_mamba_comparison.json",
        "external_protocol_gate_summary": METRIC_DIR / "external_protocol_gate_summary.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if args.strict and missing:
        raise FileNotFoundError(
            "Missing required final evidence inputs: "
            + "; ".join(f"{name}={paths[name]}" for name in missing)
        )

    calibration = read_json(paths["calibration"], required=args.strict)
    pooling_rows = read_csv_rows(paths["pooling"], required=args.strict)
    baseline_rows = read_csv_rows(paths["baseline"], required=args.strict)
    hrv_rows = read_csv_rows(paths["hrv_domain"], required=args.strict)
    robustness_rows = read_csv_rows(paths["robustness"], required=args.strict)
    paired_minirocket = read_json(paths["paired_minirocket"], required=False)
    paired_resnet = read_json(paths["paired_resnet"], required=False)
    paired_raw_mamba = read_json(optional_paths["paired_raw_mamba"], required=False)
    external_gate_rows = read_csv_rows(
        optional_paths["external_protocol_gate_summary"],
        required=False,
    )
    a0 = read_json(paths["a0_status"], required=args.strict)
    claim_map = read_csv_rows(paths["claim_map"], required=args.strict)
    task_board = read_csv_rows(paths["task_board"], required=False)

    contract_issues: list[str] = []
    if robustness_rows:
        contract_issues.extend(assert_robustness_contract(robustness_rows))
    if args.strict and contract_issues:
        raise RuntimeError("; ".join(contract_issues))

    baseline_by_name = csv_index(baseline_rows, "baseline_name")
    hrv_by_name = csv_index(hrv_rows, "analysis_name")
    pooling_by_name = csv_index(pooling_rows, "pooling")
    claim_by_id = csv_index(claim_map, "claim_id")
    task_by_id = csv_index(task_board, "id")

    full = baseline_by_name.get("Full ECG-RAMBA frozen OOF", {})
    mini = baseline_by_name.get("MiniRocket-only", {})
    hrv_only = baseline_by_name.get("HRV-only", {})
    resnet = baseline_by_name.get("ResNet1D/CNN", {})
    raw_mamba = baseline_by_name.get("Raw Mamba", {})
    q3 = pooling_by_name.get("power_mean_q3", {})
    robustness_summary = summarize_robustness(robustness_rows)
    external_gate_passed = [
        row.get("dataset", "")
        for row in external_gate_rows
        if str(row.get("protocol_gate_passed", "")).lower() in {"true", "1", "yes"}
    ]
    external_gate_blocked = [
        row.get("dataset", "")
        for row in external_gate_rows
        if str(row.get("protocol_gate_passed", "")).lower() not in {"true", "1", "yes"}
    ]
    external_gate_status = (
        "not_run"
        if not external_gate_rows
        else (
            "all_requested_passed"
            if external_gate_passed and not external_gate_blocked
            else "partial_or_blocked"
        )
    )
    if external_gate_status == "all_requested_passed":
        c06_evidence_status = "oof_supported_external_protocol_gated_for_passed_datasets"
    elif external_gate_status == "partial_or_blocked":
        c06_evidence_status = "oof_supported_external_partial_or_blocked"
    else:
        c06_evidence_status = "oof_supported_external_not_run_or_deferred"

    complete_fair_statuses = {
        "complete_frozen_oof",
        "complete_feature_baseline_from_notebook05",
        "complete_feature_baseline_from_script10",
        "complete_architecture_baseline_from_script14",
        "complete_architecture_baseline_from_script16",
    }
    required_fair_comparators = ["Raw Mamba", "ResNet1D/CNN"]
    missing_fair_comparators = [
        name
        for name in required_fair_comparators
        if baseline_by_name.get(name, {}).get("status") not in complete_fair_statuses
    ]
    if missing_fair_comparators:
        c01_evidence_status = "blocked_fair_baselines_missing"
        c01_blocker = (
            f"{', '.join(missing_fair_comparators)} fair comparator row(s) remain incomplete "
            "under the frozen OOF protocol."
        )
    else:
        c01_evidence_status = "complete_baseline_matrix_requires_metric_specific_interpretation"
        c01_blocker = "No missing required fair comparator rows; interpret only metric-specific paired deltas."

    paired_minirocket_metrics = paired_minirocket.get("metrics", {}) if isinstance(paired_minirocket, dict) else {}
    paired_resnet_metrics = paired_resnet.get("metrics", {}) if isinstance(paired_resnet, dict) else {}
    paired_raw_mamba_metrics = paired_raw_mamba.get("metrics", {}) if isinstance(paired_raw_mamba, dict) else {}
    paired_f1 = paired_minirocket_metrics.get("f1_macro", {})
    paired_pr = paired_minirocket_metrics.get("pr_auc_macro", {})
    paired_brier = paired_minirocket_metrics.get("brier_macro", {})
    paired_ece = paired_minirocket_metrics.get("ece_macro", {})
    paired_resnet_f1 = paired_resnet_metrics.get("f1_macro", {})
    paired_resnet_pr = paired_resnet_metrics.get("pr_auc_macro", {})
    paired_resnet_brier = paired_resnet_metrics.get("brier_macro", {})
    paired_resnet_ece = paired_resnet_metrics.get("ece_macro", {})
    paired_raw_f1 = paired_raw_mamba_metrics.get("f1_macro", {})
    paired_raw_pr = paired_raw_mamba_metrics.get("pr_auc_macro", {})
    paired_raw_brier = paired_raw_mamba_metrics.get("brier_macro", {})
    paired_raw_ece = paired_raw_mamba_metrics.get("ece_macro", {})

    calibration_metrics = calibration.get("metrics", {}) if isinstance(calibration, dict) else {}
    calibration_summary = calibration.get("calibration", {}) if isinstance(calibration, dict) else {}
    calibration_micro = calibration.get("calibration_micro", {}) if isinstance(calibration, dict) else {}

    blockers = a0.get("blockers", []) if isinstance(a0, dict) else []
    blocker_rows = [
        {
            "blocker_id": row.get("blocker_id", ""),
            "blocker": row.get("blocker", ""),
            "resolution_status": row.get("resolution_status", ""),
            "decision": row.get("decision", ""),
            "restriction": row.get("restriction", ""),
            "valid": row.get("valid", ""),
            "evidence_paths": row.get("evidence_paths", ""),
        }
        for row in blockers
    ]
    unresolved_blockers = [
        row
        for row in blocker_rows
        if row.get("resolution_status") not in {"resolved", "deferred", "manuscript-corrected"}
        or row.get("valid") is False
    ]

    matrix_rows = [
        {
            "claim_id": "C01",
            "claim_topic": "Fair baseline superiority / external transfer",
            "evidence_status": c01_evidence_status,
            "key_numbers": (
                f"Full PR-AUC={fmt(full.get('pr_auc_macro'))}, F1={fmt(full.get('f1_macro'))}; "
                f"MiniRocket PR-AUC={fmt(mini.get('pr_auc_macro'))}, F1={fmt(mini.get('f1_macro'))}; "
                f"ResNet1D/CNN PR-AUC={fmt(resnet.get('pr_auc_macro'))}, F1={fmt(resnet.get('f1_macro'))}; "
                f"Raw Mamba PR-AUC={fmt(raw_mamba.get('pr_auc_macro'))}, F1={fmt(raw_mamba.get('f1_macro'))}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/baseline_summary.csv;"
                "reports/revision/metrics/paired_full_vs_minirocket_comparison.json;"
                "reports/revision/metrics/paired_full_vs_resnet_comparison.json;"
                "reports/revision/metrics/paired_full_vs_raw_mamba_comparison.json"
            ),
            "safe_wording": (
                "Do not claim superiority over all fair baselines. Report comparator-specific, "
                "metric-specific paired deltas. In-domain fair comparators show ResNet1D/CNN "
                "and Raw Mamba are stronger on discrimination/F1 metrics; narrow ECG-RAMBA "
                "claims to supported calibration tradeoffs, architecture analysis, and "
                "documented limitations."
            ),
            "blocker": c01_blocker,
            "source_claim_status": c01_evidence_status,
        },
        {
            "claim_id": "C02",
            "claim_topic": "Fixed-threshold ranking-decision gap",
            "evidence_status": "supported_with_limitations",
            "key_numbers": (
                f"OOF F1={fmt(calibration_metrics.get('f1_macro'))}, "
                f"PR-AUC={fmt(calibration_metrics.get('pr_auc_macro'))}, "
                f"ECE={fmt(calibration_summary.get('ece_macro'))}, "
                f"Brier={fmt(calibration_summary.get('brier_macro'))}; "
                f"paired F1={paired_f1.get('interpretation', '')}, "
                f"Brier={paired_brier.get('interpretation', '')}, "
                f"ECE={paired_ece.get('interpretation', '')}, "
                f"PR-AUC={paired_pr.get('interpretation', '')}; "
                f"ResNet paired F1={paired_resnet_f1.get('interpretation', '')}, "
                f"Brier={paired_resnet_brier.get('interpretation', '')}, "
                f"ECE={paired_resnet_ece.get('interpretation', '')}, "
                f"PR-AUC={paired_resnet_pr.get('interpretation', '')}; "
                f"Raw Mamba paired F1={paired_raw_f1.get('interpretation', '')}, "
                f"Brier={paired_raw_brier.get('interpretation', '')}, "
                f"ECE={paired_raw_ece.get('interpretation', '')}, "
                f"PR-AUC={paired_raw_pr.get('interpretation', '')}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json;"
                "reports/revision/tables/table_paired_full_vs_minirocket.csv;"
                "reports/revision/tables/table_paired_full_vs_resnet.csv;"
                "reports/revision/tables/table_paired_full_vs_raw_mamba.csv"
            ),
            "safe_wording": (
                "Frozen OOF supports only metric-specific operating-point statements. ECG-RAMBA "
                "has calibration/error advantages over MiniRocket-only and Raw Mamba, but "
                "ResNet1D/CNN is stronger on PR-AUC, ROC-AUC, F1, Brier, and ECE; do not "
                "claim a general fixed-threshold or calibration advantage."
            ),
            "blocker": "",
            "source_claim_status": claim_by_id.get("C02", {}).get("status", ""),
        },
        {
            "claim_id": "C03",
            "claim_topic": "HRV feature evidence and domain sensitivity",
            "evidence_status": "partially_supported_with_domain_limitation",
            "key_numbers": (
                f"HRV-only ROC-AUC={fmt(hrv_only.get('roc_auc_macro'))}, "
                f"PR-AUC={fmt(hrv_only.get('pr_auc_macro'))}, F1={fmt(hrv_only.get('f1_macro'))}; "
                f"domain status={hrv_by_name.get('HRV domain classifier', {}).get('status', '')}, "
                f"domain AUC={fmt(hrv_by_name.get('HRV domain classifier', {}).get('metric_value'))}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/hrv_domain_summary.csv;"
                "reports/revision/metrics/hrv_only_baseline_summary.json;"
                "reports/revision/metrics/hrv_domain_classifier_summary.json"
            ),
            "safe_wording": (
                "Report HRV-only as a feature baseline. The near-perfect HRV domain classifier "
                "indicates strong domain sensitivity, so avoid domain-invariance wording. Do "
                "not describe reserved HRV36 slots as implemented RMSSD/SDNN/LF-HF features."
            ),
            "blocker": "Current HRV36 schema still contains reserved zero slots and no full RMSSD/SDNN/LF-HF claim.",
            "source_claim_status": claim_by_id.get("C03", {}).get("status", ""),
        },
        {
            "claim_id": "C04",
            "claim_topic": "Morphology-rhythm separation",
            "evidence_status": "blocked_representation_probe_missing",
            "key_numbers": "No completed UMAP/probing/CKA representation artifact; representation separation remains unproven.",
            "evidence_paths": (
                "reports/revision/metrics/robustness_summary.csv;"
                "reports/revision/metrics/representation_evidence_status.json"
            ),
            "safe_wording": (
                "Do not claim proven morphology-rhythm disentanglement. State that the architecture "
                "is designed to combine complementary streams and that representation separation remains future work."
            ),
            "blocker": "No completed UMAP/probing/CKA representation artifact.",
            "source_claim_status": claim_by_id.get("C04", {}).get("status", ""),
        },
        {
            "claim_id": "C05",
            "claim_topic": "Q=3 pooling operating point",
            "evidence_status": "supported_as_tradeoff_not_optimality_claim",
            "key_numbers": (
                f"Q=3 PR-AUC={fmt(q3.get('pr_auc_macro'))}, "
                f"ROC-AUC={fmt(q3.get('roc_auc_macro'))}, F1={fmt(q3.get('f1_macro'))}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/pooling_sensitivity.csv;"
                "reports/revision/metrics/pooling_decision_summary.json"
            ),
            "safe_wording": (
                "Present Q=3 as the pre-specified/frozen operating point and a sensitivity-tested "
                "tradeoff, not as globally optimal."
            ),
            "blocker": "",
            "source_claim_status": claim_by_id.get("C05", {}).get("status", ""),
        },
        {
            "claim_id": "C06",
            "claim_topic": "Protocol-faithful OOF evaluation",
            "evidence_status": c06_evidence_status,
            "key_numbers": (
                f"A0 audit_complete={a0.get('audit_complete')}; "
                f"blockers={a0.get('blocker_count')}; "
                f"calibration n={calibration.get('shape', {}).get('y_true', [''])[0] if isinstance(calibration.get('shape'), dict) else ''}; "
                f"micro ECE={fmt(calibration_micro.get('ece_micro'))}; "
                f"external_gate_status={external_gate_status}; "
                f"external_gate_passed={','.join(external_gate_passed) if external_gate_passed else 'none'}"
            ),
            "evidence_paths": (
                "reports/revision/manifests/oof_final_ema_freeze_manifest.json;"
                "reports/revision/a0_resolution_status.json;"
                "reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json;"
                "reports/revision/metrics/external_protocol_gate_summary.csv"
            ),
            "safe_wording": (
                "Claim protocol-faithful frozen Chapman OOF evaluation. External datasets may be "
                "described only as protocol-gated mapped-task evaluations for datasets that pass "
                "scripts/revision/18_external_protocol_gate.py; otherwise keep PTB/Georgia/CPSC "
                "outputs experimental. Do not claim external zero-shot superiority."
            ),
            "blocker": (
                "Deferred blockers remain documented; protocol_ready is distinct from audit_complete. "
                f"External gate status: {external_gate_status}."
            ),
            "source_claim_status": claim_by_id.get("C06", {}).get("status", ""),
        },
    ]

    safe_rows = [
        {
            "claim_id": row["claim_id"],
            "claim_topic": row["claim_topic"],
            "evidence_status": row["evidence_status"],
            "safe_wording": row["safe_wording"],
            "blocker": row["blocker"],
        }
        for row in matrix_rows
    ]
    robustness_claims = robustness_claim_rows(robustness_rows)

    final_ready = (
        not missing
        and not contract_issues
        and not unresolved_blockers
        and len(matrix_rows) == 6
        and len(robustness_claims)
        == len(REQUIRED_ROBUSTNESS_STRESSES) * len(REQUIRED_ROBUSTNESS_METRICS)
    )
    payload = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "final_ready_for_rebuttal": final_ready,
        "all_claims_supported": False,
        "missing_inputs": missing,
        "contract_issues": contract_issues,
        "unresolved_blockers": unresolved_blockers,
        "claim_guidance": {
            "global_superiority": "Do not claim global superiority over all fair baselines.",
            "resnet_in_domain": (
                "The completed paired ResNet1D/CNN comparison favors ResNet on frozen Chapman OOF "
                "PR-AUC, ROC-AUC, F1, Brier, and ECE; do not claim ECG-RAMBA in-domain superiority "
                "over fair CNN/ResNet baselines."
            ),
            "operating_point": (
                "ECG-RAMBA operating-point advantages are comparator-specific. The MiniRocket-only "
                "F1/Brier/ECE result does not generalize to ResNet1D/CNN."
            ),
            "robustness": (
                "Use only metric-specific robustness claims supported by paired degradation CIs."
            ),
            "external": "Keep external dataset outputs experimental unless protocol-specific checks are complete.",
            "external_protocol_gate": (
                "Use only protocol-gated mapped-task wording for external datasets that pass "
                "the external protocol gate; this still does not support zero-shot superiority."
            ),
            "hrv": "Do not describe reserved HRV slots as implemented RMSSD/SDNN/LF-HF features.",
            "raw_mamba": (
                "Use Raw Mamba only as a comparator-specific fair-baseline result. "
                "It does not restore global superiority if ResNet1D/CNN remains stronger."
            ),
        },
        "inputs": {name: artifact(path) for name, path in paths.items()},
        "optional_inputs": {name: artifact(path) for name, path in optional_paths.items()},
        "robustness_summary": robustness_summary,
        "task_status": {
            key: {
                "status": row.get("status", ""),
                "notebook": row.get("notebook", ""),
                "report_artifacts": row.get("report_artifacts", ""),
            }
            for key, row in task_by_id.items()
        },
        "evidence_matrix": matrix_rows,
        "safe_wording": safe_rows,
        "blockers": blocker_rows,
        "robustness_claims": robustness_claims,
        "outputs": {
            "json": rel(args.out_json),
            "table": rel(args.out_table),
            "safe_wording": rel(args.out_safe_wording),
            "blockers": rel(args.out_blockers),
            "robustness_claims": rel(args.out_robustness),
            "manifest": rel(args.out_manifest),
        },
    }

    save_csv(args.out_table, matrix_rows)
    save_csv(args.out_safe_wording, safe_rows)
    save_csv(args.out_blockers, blocker_rows)
    save_csv(args.out_robustness, robustness_claims)
    save_json(args.out_json, payload)

    output_paths = [
        args.out_json,
        args.out_table,
        args.out_safe_wording,
        args.out_blockers,
        args.out_robustness,
    ]
    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "protocol": "final_reviewer_evidence_matrix_v1",
        "strict": bool(args.strict),
        "artifact_sha256": {
            rel(path): sha256_file(path) for path in output_paths if path.exists()
        },
        "input_sha256": {
            name: sha256_file(path) for name, path in paths.items() if path.exists()
        },
        "final_ready_for_rebuttal": final_ready,
        "all_claims_supported": False,
    }
    save_json(args.out_manifest, manifest)

    print(json.dumps({
        "status": True,
        "final_ready_for_rebuttal": final_ready,
        "evidence_rows": len(matrix_rows),
        "robustness_rows": len(robustness_claims),
        "outputs": payload["outputs"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
