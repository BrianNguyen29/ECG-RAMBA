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
EXPECTED_EXTERNAL_DATASETS = ("ptbxl", "georgia", "cpsc2021")


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


def summarize_representation(
    status_payload: dict[str, Any],
    probe_rows: list[dict[str, str]],
    cka_rows: list[dict[str, str]],
) -> dict[str, Any]:
    status = str(status_payload.get("status", "") or "")
    complete = status in {"complete_probe_available", "complete"} and bool(probe_rows) and bool(cka_rows)
    complete_probe_rows = [row for row in probe_rows if row.get("status") == "complete"]
    best_probe = max(complete_probe_rows, key=lambda row: fnum(row.get("macro_roc_auc")), default={})
    morphology_probe = next(
        (
            row
            for row in complete_probe_rows
            if row.get("view") == "morphology" and row.get("label_group") == "morphology_labels"
        ),
        {},
    )
    rhythm_probe = next(
        (
            row
            for row in complete_probe_rows
            if row.get("view") == "rhythm" and row.get("label_group") == "rhythm_labels"
        ),
        {},
    )
    cka_by_pair = {
        f"{row.get('left_view')}/{row.get('right_view')}": row
        for row in cka_rows
        if row.get("status", "complete") == "complete"
    }
    best_cka = max(cka_by_pair.values(), key=lambda row: fnum(row.get("linear_cka")), default={})
    morph_rhythm_cka = cka_by_pair.get("morphology/rhythm", {})

    if complete:
        key_numbers = (
            f"Probe/CKA complete; best probe ROC-AUC={fmt(best_probe.get('macro_roc_auc'))} "
            f"({best_probe.get('view', '')}/{best_probe.get('label_group', '')}); "
            f"morphology->morphology ROC-AUC={fmt(morphology_probe.get('macro_roc_auc'))}, "
            f"PR-AUC={fmt(morphology_probe.get('macro_pr_auc'))}; "
            f"rhythm->rhythm ROC-AUC={fmt(rhythm_probe.get('macro_roc_auc'))}, "
            f"PR-AUC={fmt(rhythm_probe.get('macro_pr_auc'))}; "
            f"CKA morphology/rhythm={fmt(morph_rhythm_cka.get('linear_cka'))}, "
            f"max CKA={fmt(best_cka.get('linear_cka'))} "
            f"({best_cka.get('left_view', '')}/{best_cka.get('right_view', '')})"
        )
        safe_wording = (
            "Representation probes and CKA provide an audit of branch embeddings. "
            "CKA shows the branch embeddings are not identical, but fold-safe linear "
            "probes are near chance and do not support label-aligned morphology-rhythm "
            "disentanglement. Treat branch specialization as suggestive architecture "
            "analysis and a limitation, not an established mechanism."
        )
        blocker = (
            "Probe/CKA artifacts are complete, but mechanistic morphology-rhythm "
            "disentanglement remains unproven because fold-safe linear probes are weak."
        )
        evidence_status = "complete_probe_available_with_disentanglement_limitation"
    else:
        key_numbers = (
            "No completed UMAP/probing/CKA representation artifact; "
            "representation separation remains unproven."
        )
        safe_wording = (
            "Do not claim proven morphology-rhythm disentanglement. State that the architecture "
            "is designed to combine complementary streams and that representation separation remains future work."
        )
        blocker = "No completed UMAP/probing/CKA representation artifact."
        evidence_status = "blocked_representation_probe_missing"

    return {
        "complete": complete,
        "status": status,
        "evidence_status": evidence_status,
        "key_numbers": key_numbers,
        "safe_wording": safe_wording,
        "blocker": blocker,
        "best_probe": best_probe,
        "morphology_probe": morphology_probe,
        "rhythm_probe": rhythm_probe,
        "morphology_rhythm_cka": morph_rhythm_cka,
        "best_cka": best_cka,
    }


def summarize_fewshot(rows: list[dict[str, str]], manifest: dict[str, Any]) -> dict[str, Any]:
    complete = bool(rows) and manifest.get("status") == "complete"
    if not complete:
        return {
            "complete": False,
            "status": manifest.get("status", "missing"),
            "key_numbers": "fewshot_status=not_run_or_deferred",
            "safe_wording": (
                "Do not claim few-shot adaptation. The few-shot package is absent or incomplete."
            ),
            "blocker": "PTB-XL few-shot score-calibration artifact is absent or incomplete.",
            "best_fraction": {},
            "best_f1_fraction": {},
            "best_pr_auc_fraction": {},
            "zero_fraction": {},
        }

    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(fnum(row.get("fraction")), []).append(row)

    def summarize_fraction(fraction: float) -> dict[str, Any]:
        items = grouped.get(fraction, [])
        out: dict[str, Any] = {"fraction": fraction, "n_seeds": len(items)}
        if not items:
            return out
        metrics = [
            "train_records",
            "test_records",
            "f1_macro",
            "pr_auc_macro",
            "roc_auc_macro",
            "brier_macro",
            "ece_macro",
            "adapted_classes",
        ]
        for metric in metrics:
            values = [fnum(row.get(metric)) for row in items]
            finite = [value for value in values if math.isfinite(value)]
            out[f"{metric}_mean"] = sum(finite) / len(finite) if finite else math.nan
        out["mode"] = ",".join(sorted({str(row.get("mode", "")) for row in items}))
        return out

    finite_fractions = sorted(fraction for fraction in grouped if math.isfinite(fraction))
    zero_fraction = summarize_fraction(0.0 if 0.0 in grouped else finite_fractions[0])
    fraction_summaries = [summarize_fraction(fraction) for fraction in finite_fractions]
    best_f1_fraction = max(
        fraction_summaries,
        key=lambda item: fnum(item.get("f1_macro_mean")),
        default={},
    )
    best_pr_auc_fraction = max(
        fraction_summaries,
        key=lambda item: fnum(item.get("pr_auc_macro_mean")),
        default={},
    )
    f1_gain = fnum(best_f1_fraction.get("f1_macro_mean")) - fnum(
        zero_fraction.get("f1_macro_mean")
    )
    pr_gain = fnum(best_pr_auc_fraction.get("pr_auc_macro_mean")) - fnum(
        zero_fraction.get("pr_auc_macro_mean")
    )
    if not math.isfinite(f1_gain):
        f1_gain = math.nan
    if not math.isfinite(pr_gain):
        pr_gain = math.nan
    key_numbers = (
        "fewshot_status=complete; "
        f"protocol={manifest.get('protocol', '')}; "
        f"adaptation_kind={manifest.get('adaptation_kind', '')}; "
        f"zero-shot PR-AUC={fmt(zero_fraction.get('pr_auc_macro_mean'))}, "
        f"F1={fmt(zero_fraction.get('f1_macro_mean'))}; "
        f"F1-best fraction={fmt(best_f1_fraction.get('fraction'), digits=2)}, "
        f"train_records_mean={fmt(best_f1_fraction.get('train_records_mean'), digits=1)}, "
        f"F1={fmt(best_f1_fraction.get('f1_macro_mean'))}, "
        f"F1_gain_vs_zero={fmt(f1_gain)}; "
        f"rank-best fraction={fmt(best_pr_auc_fraction.get('fraction'), digits=2)}, "
        f"PR-AUC={fmt(best_pr_auc_fraction.get('pr_auc_macro_mean'))}, "
        f"PR-AUC_gain_vs_zero={fmt(pr_gain)}"
    )
    return {
        "complete": True,
        "status": "complete",
        "key_numbers": key_numbers,
        "safe_wording": (
            "Report PTB-XL few-shot only as leakage-audited score calibration on frozen "
            "protocol-gated external predictions. Emphasize fixed-threshold F1 changes "
            "separately from ranking metrics because score calibration may not improve "
            "PR-AUC/ROC-AUC. It leaves ECG-RAMBA weights unchanged and does not establish "
            "general zero-shot or few-shot superiority."
        ),
        "blocker": "",
        "best_fraction": best_f1_fraction,
        "best_f1_fraction": best_f1_fraction,
        "best_pr_auc_fraction": best_pr_auc_fraction,
        "f1_gain_vs_zero": f1_gain,
        "pr_auc_gain_vs_zero": pr_gain,
        "zero_fraction": zero_fraction,
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
        "paired_transformer": METRIC_DIR / "paired_full_vs_transformer_comparison.json",
        "external_protocol_gate_summary": METRIC_DIR / "external_protocol_gate_summary.csv",
        "representation_evidence_status": METRIC_DIR / "representation_evidence_status.json",
        "representation_probe_summary": METRIC_DIR / "representation_probe_summary.json",
        "representation_probe_table": TABLE_DIR / "table_representation_probe.csv",
        "representation_cka_table": TABLE_DIR / "table_representation_cka.csv",
        "fewshot_ptbxl_summary": METRIC_DIR / "fewshot_ptbxl_summary.csv",
        "fewshot_ptbxl_table": TABLE_DIR / "table_fewshot_ptbxl.csv",
        "fewshot_ptbxl_bootstrap": METRIC_DIR / "fewshot_ptbxl_bootstrap.json",
        "fewshot_ptbxl_manifest": MANIFEST_DIR / "fewshot_ptbxl_run_manifest.json",
        "robustness_multicomparator_summary": METRIC_DIR / "robustness_multicomparator_summary.csv",
        "robustness_multicomparator_pairwise": METRIC_DIR / "robustness_multicomparator_pairwise.json",
        "robustness_multicomparator_table": TABLE_DIR / "table_robustness_multicomparator.csv",
        "robustness_multicomparator_manifest": MANIFEST_DIR / "robustness_multicomparator_manifest.json",
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
    paired_transformer = read_json(optional_paths["paired_transformer"], required=False)
    external_gate_rows = read_csv_rows(
        optional_paths["external_protocol_gate_summary"],
        required=False,
    )
    representation_status = read_json(
        optional_paths["representation_evidence_status"],
        required=False,
    )
    representation_probe_summary = read_json(
        optional_paths["representation_probe_summary"],
        required=False,
    )
    representation_probe_rows = read_csv_rows(
        optional_paths["representation_probe_table"],
        required=False,
    )
    representation_cka_rows = read_csv_rows(
        optional_paths["representation_cka_table"],
        required=False,
    )
    fewshot_rows = read_csv_rows(optional_paths["fewshot_ptbxl_summary"], required=False)
    fewshot_manifest = read_json(optional_paths["fewshot_ptbxl_manifest"], required=False)
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
    transformer = baseline_by_name.get("Transformer ECG", {})
    q3 = pooling_by_name.get("power_mean_q3", {})
    robustness_summary = summarize_robustness(robustness_rows)
    representation_summary = summarize_representation(
        representation_status or representation_probe_summary,
        representation_probe_rows,
        representation_cka_rows,
    )
    fewshot_summary = summarize_fewshot(fewshot_rows, fewshot_manifest)
    external_gate_by_dataset = {
        str(row.get("dataset", "")).strip().lower(): row
        for row in external_gate_rows
        if str(row.get("dataset", "")).strip()
    }
    external_gate_passed = sorted(
        dataset
        for dataset, row in external_gate_by_dataset.items()
        if str(row.get("protocol_gate_passed", "")).lower() in {"true", "1", "yes"}
    )
    external_gate_blocked = sorted(
        dataset
        for dataset, row in external_gate_by_dataset.items()
        if str(row.get("protocol_gate_passed", "")).lower() not in {"true", "1", "yes"}
    )
    external_gate_deferred = [
        dataset for dataset in EXPECTED_EXTERNAL_DATASETS if dataset not in external_gate_by_dataset
    ]
    if not external_gate_rows:
        external_gate_status = "not_run_all_external_deferred"
    elif external_gate_passed and not external_gate_blocked and not external_gate_deferred:
        external_gate_status = "all_expected_passed"
    elif external_gate_passed:
        external_gate_status = "partial_pass_with_deferred_or_blocked_external"
    else:
        external_gate_status = "blocked_or_deferred"

    if external_gate_status == "all_expected_passed":
        c06_evidence_status = "oof_supported_external_protocol_gated_all_expected"
    elif external_gate_status == "partial_pass_with_deferred_or_blocked_external":
        c06_evidence_status = "oof_supported_external_protocol_gated_partial_with_deferred"
    elif external_gate_status == "blocked_or_deferred":
        c06_evidence_status = "oof_supported_external_blocked_or_deferred"
    else:
        c06_evidence_status = "oof_supported_external_not_run_or_deferred"

    external_gate_limited = sorted(set(external_gate_blocked) | set(external_gate_deferred))
    external_passed_text = ",".join(external_gate_passed) if external_gate_passed else "none"
    external_limited_text = ",".join(external_gate_limited) if external_gate_limited else "none"
    external_safe_sentence = (
        f"Protocol-gated mapped-task external evaluation is available only for: {external_passed_text}. "
        f"Keep blocked/deferred external datasets limited: {external_limited_text}. "
        "No unqualified external-transfer or cross-dataset superiority claim is supported. "
    )

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
    paired_transformer_metrics = paired_transformer.get("metrics", {}) if isinstance(paired_transformer, dict) else {}
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
    paired_transformer_f1 = paired_transformer_metrics.get("f1_macro", {})
    paired_transformer_pr = paired_transformer_metrics.get("pr_auc_macro", {})

    transformer_key_numbers = (
        f"; Transformer ECG PR-AUC={fmt(transformer.get('pr_auc_macro'))}, "
        f"F1={fmt(transformer.get('f1_macro'))}"
        if transformer
        else ""
    )
    transformer_paired_key_numbers = (
        f"; Transformer paired F1={paired_transformer_f1.get('interpretation', '')}, "
        f"PR-AUC={paired_transformer_pr.get('interpretation', '')}"
        if paired_transformer_metrics
        else "; Transformer paired comparison=not_run_optional"
    )
    transformer_evidence_paths = (
        ";reports/revision/metrics/paired_full_vs_transformer_comparison.json;"
        "reports/revision/tables/table_paired_full_vs_transformer.csv"
        if paired_transformer_metrics
        else ""
    )

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

    fewshot_blocker_text = (
        f" Few-shot blocker: {fewshot_summary['blocker']}"
        if fewshot_summary.get("blocker")
        else ""
    )

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
                f"{transformer_key_numbers}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/baseline_summary.csv;"
                "reports/revision/metrics/paired_full_vs_minirocket_comparison.json;"
                "reports/revision/metrics/paired_full_vs_resnet_comparison.json;"
                "reports/revision/metrics/paired_full_vs_raw_mamba_comparison.json"
                f"{transformer_evidence_paths}"
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
                f"{transformer_paired_key_numbers}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json;"
                "reports/revision/tables/table_paired_full_vs_minirocket.csv;"
                "reports/revision/tables/table_paired_full_vs_resnet.csv;"
                "reports/revision/tables/table_paired_full_vs_raw_mamba.csv"
                f"{transformer_evidence_paths}"
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
            "evidence_status": representation_summary["evidence_status"],
            "key_numbers": representation_summary["key_numbers"],
            "evidence_paths": (
                "reports/revision/metrics/representation_evidence_status.json;"
                "reports/revision/metrics/representation_probe_summary.json;"
                "reports/revision/tables/table_representation_probe.csv;"
                "reports/revision/tables/table_representation_cka.csv"
            ),
            "safe_wording": representation_summary["safe_wording"],
            "blocker": representation_summary["blocker"],
            "source_claim_status": representation_summary["evidence_status"],
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
                f"external_gate_passed={','.join(external_gate_passed) if external_gate_passed else 'none'}; "
                f"external_gate_blocked={','.join(external_gate_blocked) if external_gate_blocked else 'none'}; "
                f"external_gate_deferred={','.join(external_gate_deferred) if external_gate_deferred else 'none'}; "
                f"{fewshot_summary['key_numbers']}"
            ),
            "evidence_paths": (
                "reports/revision/manifests/oof_final_ema_freeze_manifest.json;"
                "reports/revision/a0_resolution_status.json;"
                "reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json;"
                "reports/revision/metrics/external_protocol_gate_summary.csv;"
                "reports/revision/metrics/fewshot_ptbxl_summary.csv;"
                "reports/revision/manifests/fewshot_ptbxl_run_manifest.json"
            ),
            "safe_wording": (
                "Claim protocol-faithful frozen Chapman OOF evaluation. "
                f"{external_safe_sentence}"
                f"{fewshot_summary['safe_wording']}"
            ),
            "blocker": (
                "Deferred blockers remain documented; protocol_ready is distinct from audit_complete. "
                f"External gate status: {external_gate_status}; "
                f"deferred external datasets: {','.join(external_gate_deferred) if external_gate_deferred else 'none'}."
                f"{fewshot_blocker_text}"
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
            "global_superiority": "Do not claim broad superiority over all fair baselines.",
            "resnet_in_domain": (
                "The completed paired ResNet1D/CNN comparison favors ResNet on frozen Chapman OOF "
                "PR-AUC, ROC-AUC, F1, Brier, and ECE; do not claim an ECG-RAMBA in-domain "
                "performance advantage over fair CNN/ResNet baselines."
            ),
            "operating_point": (
                "ECG-RAMBA operating-point advantages are comparator-specific. The MiniRocket-only "
                "F1/Brier/ECE result does not generalize to ResNet1D/CNN."
            ),
            "robustness": (
                "Use only metric-specific robustness claims supported by paired degradation CIs. "
                "If robustness_multicomparator artifacts are present, treat missing ResNet/Raw-Mamba "
                "stress predictions as explicit blocked evidence rather than support for broad robustness."
            ),
            "fewshot": (
                "Few-shot evidence is optional and gated. Report it only when the dataset-specific "
                "external protocol gate passed and scripts/revision/19_fewshot_adaptation.py produced "
                "a completed leakage-audited sensitivity package; do not describe it as model-weight updating."
            ),
            "external": (
                f"Use only protocol-gated mapped-task wording for passed external datasets: {external_passed_text}. "
                f"Keep blocked/deferred external datasets limited: {external_limited_text}."
            ),
            "external_protocol_gate": (
                "Use only protocol-gated mapped-task wording for external datasets that pass "
                "the external protocol gate; this still does not support unqualified "
                "external-transfer or cross-dataset performance claims."
            ),
            "hrv": "Do not describe reserved HRV slots as implemented RMSSD/SDNN/LF-HF features.",
            "representation": (
                "Use representation probe/CKA only as a conservative audit. CKA may show branch "
                "embeddings are not identical, but weak fold-safe linear probes do not support "
                "established morphology-rhythm separation."
            ),
            "raw_mamba": (
                "Use Raw Mamba only as a comparator-specific fair-baseline result. "
                "It does not restore a broad fair-baseline advantage if ResNet1D/CNN remains stronger."
            ),
            "transformer": (
                "Use Transformer ECG only as optional comparator-specific evidence if "
                "scripts/revision/24_transformer_ecg_baseline.py and paired bootstrap output are complete. "
                "It must not be used to imply broad model-family superiority."
            ),
        },
        "inputs": {name: artifact(path) for name, path in paths.items()},
        "optional_inputs": {name: artifact(path) for name, path in optional_paths.items()},
        "external_gate_summary": {
            "expected_datasets": list(EXPECTED_EXTERNAL_DATASETS),
            "status": external_gate_status,
            "passed": external_gate_passed,
            "blocked": external_gate_blocked,
            "deferred": external_gate_deferred,
        },
        "fewshot_summary": fewshot_summary,
        "robustness_summary": robustness_summary,
        "representation_summary": representation_summary,
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
