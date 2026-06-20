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
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


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
        "a0_status": REVISION_DIR / "a0_resolution_status.json",
        "claim_map": PROJECT_ROOT / "docs" / "revision_plan" / "claim_evidence_map.csv",
        "task_board": PROJECT_ROOT / "docs" / "revision_plan" / "task_board.csv",
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
    paired = read_json(paths["paired_minirocket"], required=False)
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
    q3 = pooling_by_name.get("power_mean_q3", {})
    robustness_summary = summarize_robustness(robustness_rows)

    paired_metrics = paired.get("metrics", {}) if isinstance(paired, dict) else {}
    paired_f1 = paired_metrics.get("f1_macro", {})
    paired_pr = paired_metrics.get("pr_auc_macro", {})
    paired_brier = paired_metrics.get("brier_macro", {})
    paired_ece = paired_metrics.get("ece_macro", {})

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
            "evidence_status": "blocked_fair_baselines_missing",
            "key_numbers": (
                f"Full PR-AUC={fmt(full.get('pr_auc_macro'))}, F1={fmt(full.get('f1_macro'))}; "
                f"MiniRocket PR-AUC={fmt(mini.get('pr_auc_macro'))}, F1={fmt(mini.get('f1_macro'))}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/baseline_summary.csv;"
                "reports/revision/metrics/paired_full_vs_minirocket_comparison.json"
            ),
            "safe_wording": (
                "Do not claim superiority over all fair baselines. Report that MiniRocket-only "
                "is stronger on rank-based discrimination while Full ECG-RAMBA is stronger for "
                "fixed-threshold/calibrated operating metrics where paired CIs support it."
            ),
            "blocker": "Raw Mamba and ResNet1D/CNN fair runners remain TBD.",
            "source_claim_status": claim_by_id.get("C01", {}).get("status", ""),
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
                f"PR-AUC={paired_pr.get('interpretation', '')}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json;"
                "reports/revision/tables/table_paired_full_vs_minirocket.csv"
            ),
            "safe_wording": (
                "Frozen OOF supports a calibrated/fixed-threshold operating-point advantage, "
                "not a rank-based discrimination advantage."
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
                "Report HRV-only as a feature baseline. If domain AUC is high, present HRV "
                "as domain-sensitive and avoid domain-invariance wording."
            ),
            "blocker": "Current HRV36 schema still contains reserved zero slots and no full RMSSD/SDNN/LF-HF claim.",
            "source_claim_status": claim_by_id.get("C03", {}).get("status", ""),
        },
        {
            "claim_id": "C04",
            "claim_topic": "Morphology-rhythm separation",
            "evidence_status": "blocked_representation_probe_missing",
            "key_numbers": (
                f"Robustness rows={robustness_summary['n_rows']}; "
                f"Full less-degraded metrics={robustness_summary['full_less_degraded_count']}; "
                f"MiniRocket less-degraded metrics={robustness_summary['minirocket_less_degraded_count']}"
            ),
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
            "evidence_status": "oof_supported_external_experimental",
            "key_numbers": (
                f"A0 audit_complete={a0.get('audit_complete')}; "
                f"blockers={a0.get('blocker_count')}; "
                f"calibration n={calibration.get('shape', {}).get('y_true', [''])[0] if isinstance(calibration.get('shape'), dict) else ''}; "
                f"micro ECE={fmt(calibration_micro.get('ece_micro'))}"
            ),
            "evidence_paths": (
                "reports/revision/manifests/oof_final_ema_freeze_manifest.json;"
                "reports/revision/a0_resolution_status.json;"
                "reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json"
            ),
            "safe_wording": (
                "Claim protocol-faithful frozen Chapman OOF evaluation. Keep PTB/Georgia/CPSC outputs "
                "experimental unless their dataset-specific protocols are separately completed."
            ),
            "blocker": "Deferred blockers remain documented; protocol_ready is distinct from audit_complete.",
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

    final_ready = not contract_issues and not unresolved_blockers and bool(matrix_rows)
    payload = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "final_ready_for_rebuttal": final_ready,
        "all_claims_supported": False,
        "contract_issues": contract_issues,
        "unresolved_blockers": unresolved_blockers,
        "claim_guidance": {
            "global_superiority": "Do not claim global superiority over all fair baselines.",
            "robustness": (
                "Use only metric-specific robustness claims supported by paired degradation CIs."
            ),
            "external": "Keep external dataset outputs experimental unless protocol-specific checks are complete.",
            "hrv": "Do not describe reserved HRV slots as implemented RMSSD/SDNN/LF-HF features.",
        },
        "inputs": {name: artifact(path) for name, path in paths.items()},
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
