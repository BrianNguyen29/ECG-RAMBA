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
    save_json_atomic,
    sha256_file,
)
from scripts.revision.robustness_profile_audit import select_best_profile  # noqa: E402


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

# Stable capability contract consumed by Notebook 07. Keep this declarative so
# the notebook can validate generator support without depending on internal
# helper names, which may change during refactors.
FINAL_EVIDENCE_SCHEMA_VERSION = 7
FINAL_EVIDENCE_CAPABILITIES = (
    "claim_readiness_gates",
    "external_learned_comparator_audit",
    "group_safe_score_calibration_v2",
    "hybrid_morphology_paired",
    "learned_comparator_robustness_audit",
    "representation_probe_v3",
    "reviewer_presentation_assets",
    "reviewer_gap_closure_v1",
    "morphology_kernel_learnability_control",
    "external_zero_target_group_paired_ci",
    "pooling_q3_cross_dataset_sensitivity",
    "transformer_paired",
    "true_fewshot_frozen_encoder_head_v2",
)


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


def paired_oof_contract_issues(
    payload: dict[str, Any],
    *,
    label: str,
    oof_sha256: str,
    freeze_sha256: str,
) -> list[str]:
    if not payload:
        return [f"{label}: paired payload missing"]
    if payload.get("status") not in (True, "complete"):
        return [f"{label}: paired status={payload.get('status')!r}"]
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    issues: list[str] = []
    if ((inputs.get("full_predictions") or {}).get("sha256")) != oof_sha256:
        issues.append(f"{label}: full OOF SHA mismatch")
    if ((inputs.get("freeze_manifest") or {}).get("sha256")) != freeze_sha256:
        issues.append(f"{label}: freeze manifest SHA mismatch")
    return issues


def external_comparator_audit_contract_issues(
    manifests: list[Path],
    *,
    oof_sha256: str,
    freeze_sha256: str,
) -> list[str]:
    """Validate optional external learned-comparator outputs against the current frozen OOF."""

    issues: list[str] = []
    expected_contract = {"oof_sha256": oof_sha256, "freeze_sha256": freeze_sha256}
    runner = PROJECT_ROOT / "scripts" / "revision" / "31_generate_external_comparator_predictions.py"
    runner_sha = sha256_file(runner) if runner.exists() else None
    for path in manifests:
        if not path.exists():
            continue
        payload = read_json(path, required=False)
        label = path.name
        if payload.get("status") != "complete_experimental_requires_external_comparator_gate":
            issues.append(f"{label}: status={payload.get('status')!r}")
        if payload.get("canonical_contract") != expected_contract:
            issues.append(f"{label}: canonical OOF/freeze mismatch")
        sources = payload.get("source_contract") if isinstance(payload.get("source_contract"), dict) else {}
        if not sources.get("archive_sha256") or not sources.get("runner_sha256"):
            issues.append(f"{label}: source contract missing")
        elif runner_sha and sources.get("runner_sha256") != runner_sha:
            issues.append(f"{label}: runner SHA mismatch")
    return issues


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
        if degradation in {"full_nominal_95ci_less_degraded", "full_significantly_less_degraded"}:
            normalized_degradation = "full_nominal_95ci_less_degraded"
            degradation_wording = (
                "For this named stress/metric, the nominal unadjusted paired 95% record-bootstrap CI "
                "favors Full ECG-RAMBA for the clean-to-stress change."
            )
        elif degradation in {
            "minirocket_nominal_95ci_less_degraded",
            "minirocket_significantly_less_degraded",
        }:
            normalized_degradation = "minirocket_nominal_95ci_less_degraded"
            degradation_wording = (
                "For this named stress/metric, the nominal unadjusted paired 95% record-bootstrap CI "
                "favors the fixed-transform-only comparator for the clean-to-stress change."
            )
        else:
            normalized_degradation = "nominal_95ci_inconclusive_degradation_difference"
            degradation_wording = (
                "The nominal unadjusted paired 95% record-bootstrap CI is inconclusive for the "
                "clean-to-stress change difference."
            )

        if stressed in {
            "full_nominal_95ci_better_under_stress",
            "full_significantly_better_under_stress",
        }:
            normalized_stressed = "full_nominal_95ci_better_under_stress"
            stressed_wording = (
                "For this named stress/metric, the nominal unadjusted paired 95% record-bootstrap CI "
                "favors Full ECG-RAMBA at the stressed operating point."
            )
        elif stressed in {
            "minirocket_nominal_95ci_better_under_stress",
            "minirocket_significantly_better_under_stress",
        }:
            normalized_stressed = "minirocket_nominal_95ci_better_under_stress"
            stressed_wording = (
                "For this named stress/metric, the nominal unadjusted paired 95% record-bootstrap CI "
                "favors the fixed-transform-only comparator at the stressed operating point."
            )
        else:
            normalized_stressed = "nominal_95ci_inconclusive_stressed_difference"
            stressed_wording = (
                "The nominal unadjusted paired 95% record-bootstrap CI is inconclusive at the "
                "stressed operating point."
            )

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
                "degradation_interpretation": normalized_degradation,
                "stressed_performance_interpretation": normalized_stressed,
                "ci_scope": "nominal_95_percentile_paired_record_bootstrap_unadjusted",
                "training_variability_scope": (
                    "fixed_trained_folds_and_checkpoints_not_retrained_within_bootstrap"
                ),
                "safe_wording_degradation": degradation_wording,
                "safe_wording_stressed_performance": stressed_wording,
            }
        )
    return sorted(claim_rows, key=lambda r: (str(r["stress_test"]), str(r["metric"])))


def summarize_robustness(rows: list[dict[str, str]]) -> dict[str, Any]:
    full_less_degraded = [
        row
        for row in rows
        if row.get("degradation_interpretation")
        in {"full_nominal_95ci_less_degraded", "full_significantly_less_degraded"}
    ]
    mini_less_degraded = [
        row
        for row in rows
        if row.get("degradation_interpretation")
        in {"minirocket_nominal_95ci_less_degraded", "minirocket_significantly_less_degraded"}
    ]
    full_better_stress = [
        row
        for row in rows
        if row.get("stressed_performance_interpretation")
        in {"full_nominal_95ci_better_under_stress", "full_significantly_better_under_stress"}
    ]
    mini_better_stress = [
        row
        for row in rows
        if row.get("stressed_performance_interpretation")
        in {
            "minirocket_nominal_95ci_better_under_stress",
            "minirocket_significantly_better_under_stress",
        }
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
    manifest_payload: dict[str, Any],
    probe_rows: list[dict[str, str]],
    cka_rows: list[dict[str, str]],
    canonical_contract: dict[str, str],
) -> dict[str, Any]:
    status = str(status_payload.get("status", "") or "")
    runner = PROJECT_ROOT / "scripts" / "revision" / "20_representation_probe.py"
    runner_sha = sha256_file(runner) if runner.exists() else None
    complete = (
        status
        in {
            "complete_probe_available",
            "complete_probe_available_with_disentanglement_limitation",
            "complete",
        }
        and manifest_payload.get("status") == "complete"
        and manifest_payload.get("protocol") == "representation_probe_fold_safe_v3_projection_and_fold_audit"
        and manifest_payload.get("canonical_contract") == canonical_contract
        and runner_sha is not None
        and manifest_payload.get("runner_sha256") == runner_sha
        and bool(probe_rows)
        and bool(cka_rows)
    )
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


def summarize_external_adaptation(
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
    dataset: str,
    *,
    expected_status: str,
    expected_protocol: str,
    adaptation_label: str,
    safe_wording: str,
    canonical_contract: dict[str, str],
    runner_name: str,
    required_manifest_fields: dict[str, Any] | None = None,
    required_row_fields: dict[str, str] | None = None,
    model_filter: str | None = None,
    primary_fraction: float = 0.10,
    expected_fractions: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10),
    required_output_paths: tuple[Path, ...] = (),
    primary_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Summarize only an explicitly versioned, protocol-valid adaptation package."""
    if model_filter is not None:
        rows = [row for row in rows if str(row.get("model", "")) == model_filter]
    runner = PROJECT_ROOT / "scripts" / "revision" / runner_name
    runner_sha = sha256_file(runner) if runner.exists() else None
    observed_fractions = {fnum(row.get("fraction")) for row in rows}
    expected_seed_set = {int(value) for value in manifest.get("seeds", [])}
    complete_seed_grid = bool(expected_seed_set) and all(
        {
            int(seed_value)
            for row in rows
            if math.isclose(fnum(row.get("fraction")), fraction, abs_tol=1e-12)
            for seed_value in [fnum(row.get("seed"))]
            if math.isfinite(seed_value)
        }
        == expected_seed_set
        for fraction in expected_fractions
    )
    manifest_outputs = {
        Path(str(row.get("path", ""))).name: row
        for row in manifest.get("outputs", [])
        if isinstance(row, dict) and row.get("path")
    }
    output_contract_complete = all(
        path.exists()
        and path.stat().st_size > 0
        and (manifest_outputs.get(path.name) or {}).get("size_bytes") == path.stat().st_size
        and (manifest_outputs.get(path.name) or {}).get("sha256") == sha256_file(path)
        for path in required_output_paths
    )
    expected_metrics = {"pr_auc_macro", "roc_auc_macro", "f1_macro", "brier_macro", "ece_macro"}
    primary_adapted_rows = [
        row
        for row in (primary_rows or [])
        if row.get("comparison_type") == "adapted_vs_zero_target_label"
        and (model_filter is None or row.get("model") == model_filter)
        and math.isclose(fnum(row.get("primary_fraction")), primary_fraction, abs_tol=1e-12)
    ]
    expected_comparators = {
        str(model) for model in manifest.get("models", []) if str(model) != "full"
    }
    primary_paired_rows = [
        row
        for row in (primary_rows or [])
        if row.get("comparison_type") == "full_vs_comparator_at_primary_fraction"
        and math.isclose(fnum(row.get("primary_fraction")), primary_fraction, abs_tol=1e-12)
    ]
    primary_contract_complete = primary_rows is None or (
        {str(row.get("metric")) for row in primary_adapted_rows} == expected_metrics
        and len(primary_adapted_rows) == len(expected_metrics)
        and {
            (str(row.get("comparator")), str(row.get("metric")))
            for row in primary_paired_rows
        }
        == {
            (comparator, metric)
            for comparator in expected_comparators
            for metric in expected_metrics
        }
        and all(
            math.isfinite(fnum(row.get("n_boot_valid")))
            and int(fnum(row.get("n_boot_valid")))
            == int(manifest.get("primary_endpoint_inference", {}).get("n_boot", -1))
            and math.isfinite(fnum(row.get("n_seeds")))
            and int(fnum(row.get("n_seeds"))) == len(expected_seed_set)
            and all(
                math.isfinite(fnum(row.get(field)))
                for field in ("improvement_ci_low", "improvement_ci_high")
            )
            for row in primary_adapted_rows + primary_paired_rows
        )
    )
    complete = (
        bool(rows)
        and manifest.get("status") == expected_status
        and manifest.get("protocol") == expected_protocol
        and manifest.get("zero_group_overlap_all_splits") is True
        and manifest.get("canonical_contract") == canonical_contract
        and runner_sha is not None
        and manifest.get("runner_sha256") == runner_sha
        and all(
            manifest.get(key) == value
            for key, value in (required_manifest_fields or {}).items()
        )
        and all(
            str(row.get(key, "")) == value
            for row in rows
            for key, value in (required_row_fields or {}).items()
        )
        and observed_fractions == set(expected_fractions)
        and complete_seed_grid
        and len(rows) == len(expected_fractions) * len(expected_seed_set)
        and output_contract_complete
        and primary_contract_complete
    )
    if not complete:
        return {
            "dataset": dataset,
            "complete": False,
            "status": manifest.get("status", "missing_or_protocol_mismatch"),
            "key_numbers": f"{adaptation_label}_status=not_run_or_protocol_invalid",
            "safe_wording": f"Do not claim {adaptation_label}. The required group-safe package is absent or invalid.",
            "blocker": f"{dataset} {adaptation_label} artifact is absent, incomplete, or uses a different protocol.",
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
            "train_records_or_windows",
            "test_records_or_windows",
            "f1_macro",
            "pr_auc_macro",
            "roc_auc_macro",
            "brier_macro",
            "ece_macro",
            "adapted_classes",
        ]
        for metric in metrics:
            aliases = {
                "train_records_or_windows": ("train_records_or_windows", "train_records"),
                "test_records_or_windows": ("test_records_or_windows", "test_records"),
                "adapted_classes": ("adapted_classes", "fold_heads"),
            }.get(metric, (metric,))
            values = [fnum(next((row.get(key) for key in aliases if row.get(key) not in (None, "")), None)) for row in items]
            finite = [value for value in values if math.isfinite(value)]
            out[f"{metric}_mean"] = sum(finite) / len(finite) if finite else math.nan
        out["mode"] = ",".join(sorted({str(row.get("mode", "")) for row in items}))
        return out

    finite_fractions = sorted(fraction for fraction in grouped if math.isfinite(fraction))
    zero_fraction = summarize_fraction(0.0 if 0.0 in grouped else finite_fractions[0])
    fraction_summaries = [summarize_fraction(fraction) for fraction in finite_fractions]
    primary = summarize_fraction(primary_fraction)
    primary_ci_by_metric = {
        str(row.get("metric")): row for row in primary_adapted_rows
    }
    f1_gain = fnum(primary.get("f1_macro_mean")) - fnum(
        zero_fraction.get("f1_macro_mean")
    )
    pr_gain = fnum(primary.get("pr_auc_macro_mean")) - fnum(
        zero_fraction.get("pr_auc_macro_mean")
    )
    if not math.isfinite(f1_gain):
        f1_gain = math.nan
    if not math.isfinite(pr_gain):
        pr_gain = math.nan
    f1_ci_row = primary_ci_by_metric.get("f1_macro")
    pr_ci_row = primary_ci_by_metric.get("pr_auc_macro")
    f1_ci_text = (
        f", F1 95% group CI=[{fmt(f1_ci_row.get('primary_value_ci_low'))}, "
        f"{fmt(f1_ci_row.get('primary_value_ci_high'))}]"
        if f1_ci_row
        else ""
    )
    pr_ci_text = (
        f", PR-AUC 95% group CI=[{fmt(pr_ci_row.get('primary_value_ci_low'))}, "
        f"{fmt(pr_ci_row.get('primary_value_ci_high'))}]"
        if pr_ci_row
        else ""
    )
    key_numbers = (
        f"{adaptation_label}_status=complete; "
        f"dataset={dataset}; "
        f"protocol={expected_protocol}; "
        f"zero-shot PR-AUC={fmt(zero_fraction.get('pr_auc_macro_mean'))}, "
        f"F1={fmt(zero_fraction.get('f1_macro_mean'))}; "
        f"pre-specified primary fraction={fmt(primary.get('fraction'), digits=2)}, "
        f"train_units_mean={fmt(primary.get('train_records_or_windows_mean'), digits=1)}, "
        f"F1={fmt(primary.get('f1_macro_mean'))}{f1_ci_text}, "
        f"F1_gain_vs_zero={fmt(f1_gain)}; "
        f"PR-AUC={fmt(primary.get('pr_auc_macro_mean'))}{pr_ci_text}, "
        f"PR-AUC_gain_vs_zero={fmt(pr_gain)}"
    )
    return {
        "dataset": dataset,
        "complete": True,
        "status": "complete",
        "key_numbers": key_numbers,
        "safe_wording": safe_wording,
        "blocker": "",
        "primary_fraction": primary,
        "primary_fraction_policy": "pre_specified_0.10_no_test_set_budget_selection",
        "primary_endpoint_rows": primary_rows or [],
        "best_fraction": primary,
        "best_f1_fraction": primary,
        "best_pr_auc_fraction": primary,
        "f1_gain_vs_zero": f1_gain,
        "pr_auc_gain_vs_zero": pr_gain,
        "zero_fraction": zero_fraction,
    }


def combine_adaptation_summaries(
    summaries_by_dataset: dict[str, dict[str, Any]],
    *,
    label: str,
    safe_wording: str,
) -> dict[str, Any]:
    completed = {
        dataset: summary
        for dataset, summary in summaries_by_dataset.items()
        if summary.get("complete") is True
    }
    deferred = [
        dataset
        for dataset, summary in summaries_by_dataset.items()
        if summary.get("complete") is not True
    ]
    if not completed:
        return {
            "complete": False,
            "status": "not_run_or_deferred",
            "datasets_complete": [],
            "datasets_deferred": deferred,
            "by_dataset": summaries_by_dataset,
            "key_numbers": f"{label}_status=not_run_or_deferred",
            "safe_wording": f"Do not claim {label}. No protocol-valid dataset-specific package is complete.",
            "blocker": f"No complete dataset-specific {label} artifact.",
        }

    completed_names = sorted(completed)
    key_numbers = "; ".join(
        f"{dataset}({completed[dataset]['key_numbers']})" for dataset in completed_names
    )
    blocker = (
        f"Dataset-specific {label} not available for: "
        + ",".join(deferred)
        if deferred
        else ""
    )
    return {
        "complete": True,
        "status": "complete",
        "datasets_complete": completed_names,
        "datasets_deferred": deferred,
        "by_dataset": summaries_by_dataset,
        "key_numbers": key_numbers,
        "safe_wording": safe_wording,
        "blocker": blocker,
    }


def summarize_external_comparator_audit(
    rows: list[dict[str, Any]], manifest: dict[str, Any], contract_issues: list[str]
) -> dict[str, Any]:
    complete = (
        bool(rows)
        and manifest.get("status") == "complete"
        and not manifest.get("failures")
        and not contract_issues
    )
    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
    comparators = sorted({str(row.get("comparator", "")) for row in rows if row.get("comparator")})
    return {
        "complete": complete,
        "datasets": datasets,
        "comparators": comparators,
        "key_numbers": (
            f"external_learned_comparator_audit={'complete' if complete else 'not_run_or_incomplete'}; "
            f"datasets={','.join(datasets) if datasets else 'none'}; "
            f"comparators={','.join(comparators) if comparators else 'none'}; rows={len(rows)}"
        ),
        "safe_wording": (
            "Report paired external results only by dataset, comparator, and metric. PTB-XL and Georgia are "
            "separate mapped record-level tasks; CPSC2021 remains a separate annotation-aligned window task."
        ),
        "blocker": (
            "External learned-comparator paired audit is incomplete."
            if not complete and not contract_issues
            else "; ".join(contract_issues)
        ),
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


def reviewer_gap_closure_contract_issues(
    status: dict[str, Any],
    manifest: dict[str, Any],
    *,
    required_outputs: list[Path],
) -> list[str]:
    """Authenticate the four reviewer-item closure rows and compact outputs."""

    issues: list[str] = []
    expected_items = {"R1-C2", "R1-C5", "R1-C6", "R2-C3"}
    rows = status.get("rows") if isinstance(status.get("rows"), list) else []
    by_item = {
        str(row.get("reviewer_item")): row
        for row in rows
        if isinstance(row, dict) and row.get("reviewer_item")
    }
    if status.get("status") is not True or set(by_item) != expected_items:
        issues.append("reviewer gap closure status/grid is incomplete")
    for reviewer_item in sorted(expected_items):
        row = by_item.get(reviewer_item) or {}
        if row.get("status") != "complete" or row.get("manuscript_ready") is not True:
            issues.append(f"reviewer gap {reviewer_item} is not manuscript-ready")
        if row.get("issues"):
            issues.append(f"reviewer gap {reviewer_item} reports issues")

    runner = PROJECT_ROOT / "scripts" / "revision" / "41_reviewer_gap_closure.py"
    if manifest.get("status") != "complete":
        issues.append(f"reviewer gap closure manifest status={manifest.get('status')!r}")
    if not runner.exists() or manifest.get("runner_sha256") != sha256_file(runner):
        issues.append("reviewer gap closure runner SHA mismatch")
    artifact_rows = {
        Path(str(row.get("path", ""))).name: row
        for row in manifest.get("artifacts") or []
        if isinstance(row, dict) and row.get("path")
    }
    for path in required_outputs:
        if not path.exists() or path.stat().st_size == 0:
            continue
        row = artifact_rows.get(path.name)
        if row is None:
            issues.append(f"reviewer gap closure manifest missing {path.name}")
            continue
        if int(row.get("size_bytes", -1)) != path.stat().st_size:
            issues.append(f"reviewer gap closure size mismatch for {path.name}")
        if row.get("sha256") != sha256_file(path):
            issues.append(f"reviewer gap closure SHA mismatch for {path.name}")
    return issues


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()

    paths = {
        "oof_predictions": REVISION_DIR / "predictions" / "oof_final_ema_predictions.npz",
        "freeze_manifest": MANIFEST_DIR / "oof_final_ema_freeze_manifest.json",
        "calibration": METRIC_DIR / "calibration_ci_oof_final_ema_predictions.json",
        "pooling": METRIC_DIR / "pooling_sensitivity.csv",
        "baseline": METRIC_DIR / "baseline_summary.csv",
        "component": METRIC_DIR / "component_check_summary.json",
        "hrv_domain": METRIC_DIR / "hrv_domain_summary.csv",
        "robustness": METRIC_DIR / "robustness_summary.csv",
        "paired_minirocket": METRIC_DIR / "paired_full_vs_minirocket_comparison.json",
        "paired_resnet": METRIC_DIR / "paired_full_vs_resnet_comparison.json",
        "reviewer_gap_closure_status": METRIC_DIR / "reviewer_gap_closure_status.json",
        "reviewer_gap_closure_table": TABLE_DIR / "table_reviewer_gap_closure_status.csv",
        "reviewer_gap_closure_manifest": MANIFEST_DIR / "reviewer_gap_closure_manifest.json",
        "external_zero_target_ci_compact": TABLE_DIR / "table_external_zero_target_ci_compact.csv",
        "external_zero_target_ci_compact_tex": TABLE_DIR / "table_external_zero_target_ci_compact.tex",
        "pooling_cross_dataset_compact": TABLE_DIR / "table_pooling_cross_dataset_compact.csv",
        "pooling_cross_dataset_compact_tex": TABLE_DIR / "table_pooling_cross_dataset_compact.tex",
        "morphology_learnability_compact": TABLE_DIR / "table_morphology_learnability_compact.csv",
        "morphology_learnability_compact_tex": TABLE_DIR / "table_morphology_learnability_compact.tex",
        "robustness_six_stress_compact": TABLE_DIR / "table_robustness_six_stress_compact.csv",
        "robustness_six_stress_compact_tex": TABLE_DIR / "table_robustness_six_stress_compact.tex",
        "a0_status": REVISION_DIR / "a0_resolution_status.json",
        "claim_map": PROJECT_ROOT / "docs" / "revision_plan" / "claim_evidence_map.csv",
        "task_board": PROJECT_ROOT / "docs" / "revision_plan" / "task_board.csv",
    }
    optional_paths = {
        "paired_raw_mamba": METRIC_DIR / "paired_full_vs_raw_mamba_comparison.json",
        "paired_transformer": METRIC_DIR / "paired_full_vs_transformer_comparison.json",
        "paired_hybrid_morphology": METRIC_DIR / "paired_full_vs_hybrid_morphology_comparison.json",
        "hybrid_morphology_summary": METRIC_DIR / "hybrid_morphology_baseline_summary.json",
        "claim_readiness_gates": METRIC_DIR / "claim_readiness_gates.json",
        "claim_readiness_table": TABLE_DIR / "table_claim_readiness_gates.csv",
        "external_protocol_gate_summary": METRIC_DIR / "external_protocol_gate_summary.csv",
        "representation_evidence_status": METRIC_DIR / "representation_evidence_status.json",
        "representation_probe_summary": METRIC_DIR / "representation_probe_summary.json",
        "representation_probe_table": TABLE_DIR / "table_representation_probe.csv",
        "representation_probe_by_fold_table": TABLE_DIR / "table_representation_probe_by_fold.csv",
        "representation_cka_table": TABLE_DIR / "table_representation_cka.csv",
        "representation_figure": REVISION_DIR / "figures" / "figure_representation_audit.png",
        "representation_probe_manifest": MANIFEST_DIR / "representation_probe_manifest.json",
        "fewshot_ptbxl_summary": METRIC_DIR / "fewshot_ptbxl_summary.csv",
        "fewshot_ptbxl_table": TABLE_DIR / "table_fewshot_ptbxl.csv",
        "fewshot_ptbxl_bootstrap": METRIC_DIR / "fewshot_ptbxl_bootstrap.json",
        "fewshot_ptbxl_manifest": MANIFEST_DIR / "fewshot_ptbxl_run_manifest.json",
        "fewshot_cpsc2021_summary": METRIC_DIR / "fewshot_cpsc2021_summary.csv",
        "fewshot_cpsc2021_table": TABLE_DIR / "table_fewshot_cpsc2021.csv",
        "fewshot_cpsc2021_bootstrap": METRIC_DIR / "fewshot_cpsc2021_bootstrap.json",
        "fewshot_cpsc2021_manifest": MANIFEST_DIR / "fewshot_cpsc2021_run_manifest.json",
        "fewshot_georgia_summary": METRIC_DIR / "fewshot_georgia_summary.csv",
        "fewshot_georgia_table": TABLE_DIR / "table_fewshot_georgia.csv",
        "fewshot_georgia_bootstrap": METRIC_DIR / "fewshot_georgia_bootstrap.json",
        "fewshot_georgia_manifest": MANIFEST_DIR / "fewshot_georgia_run_manifest.json",
        "group_safe_score_calibration_ptbxl_summary": METRIC_DIR / "group_safe_score_calibration_ptbxl_summary.csv",
        "group_safe_score_calibration_ptbxl_table": TABLE_DIR / "table_group_safe_score_calibration_ptbxl.csv",
        "group_safe_score_calibration_ptbxl_bootstrap": METRIC_DIR / "group_safe_score_calibration_ptbxl_bootstrap.json",
        "group_safe_score_calibration_ptbxl_splits": MANIFEST_DIR / "group_safe_score_calibration_ptbxl_splits.npz",
        "group_safe_score_calibration_ptbxl_coefficients": TABLE_DIR / "table_group_safe_score_calibration_ptbxl_coefficients.csv",
        "group_safe_score_calibration_ptbxl_manifest": MANIFEST_DIR / "group_safe_score_calibration_ptbxl_manifest.json",
        "true_fewshot_head_ptbxl_summary": METRIC_DIR / "true_fewshot_head_ptbxl_summary.csv",
        "true_fewshot_head_ptbxl_table": TABLE_DIR / "table_true_fewshot_head_ptbxl.csv",
        "true_fewshot_head_ptbxl_paired": TABLE_DIR / "table_true_fewshot_head_ptbxl_paired.csv",
        "true_fewshot_head_ptbxl_primary": TABLE_DIR / "table_true_fewshot_head_ptbxl_primary.csv",
        "true_fewshot_head_ptbxl_bootstrap": METRIC_DIR / "true_fewshot_head_ptbxl_bootstrap.json",
        "true_fewshot_head_ptbxl_coefficients": TABLE_DIR / "table_true_fewshot_head_ptbxl_coefficients.csv",
        "true_fewshot_head_ptbxl_splits": MANIFEST_DIR / "true_fewshot_head_ptbxl_splits.npz",
        "true_fewshot_head_ptbxl_manifest": MANIFEST_DIR / "true_fewshot_head_ptbxl_manifest.json",
        "external_comparator_summary": METRIC_DIR / "external_comparator_paired_summary.json",
        "external_comparator_table": TABLE_DIR / "table_external_comparator_paired.csv",
        "external_comparator_manifest": MANIFEST_DIR / "external_comparator_paired_manifest.json",
        "external_ptbxl_resnet_manifest": MANIFEST_DIR / "external_ptbxl_resnet1d_cnn_manifest.json",
        "external_ptbxl_raw_mamba_manifest": MANIFEST_DIR / "external_ptbxl_raw_mamba_manifest.json",
        "external_georgia_resnet_manifest": MANIFEST_DIR / "external_georgia_resnet1d_cnn_manifest.json",
        "external_georgia_raw_mamba_manifest": MANIFEST_DIR / "external_georgia_raw_mamba_manifest.json",
        "reviewer_presentation_manifest": MANIFEST_DIR / "reviewer_completion_input_contract.json",
        "marked_manuscript_manifest": MANIFEST_DIR / "marked_manuscript_manifest.json",
        "robustness_multicomparator_summary": METRIC_DIR / "robustness_multicomparator_summary.csv",
        "robustness_multicomparator_pairwise": METRIC_DIR / "robustness_multicomparator_pairwise.json",
        "robustness_multicomparator_table": TABLE_DIR / "table_robustness_multicomparator.csv",
        "robustness_multicomparator_manifest": MANIFEST_DIR / "robustness_multicomparator_manifest.json",
        "robustness_full_vs_resnet": METRIC_DIR / "robustness_full_vs_resnet_comparison.json",
        "robustness_full_vs_raw_mamba": METRIC_DIR / "robustness_full_vs_raw_mamba_comparison.json",
        "robustness_full_vs_transformer": METRIC_DIR / "robustness_full_vs_transformer_comparison.json",
        "morphology_learnability_summary": METRIC_DIR / "morphology_learnability_summary.json",
        "paired_morphology_learnability": METRIC_DIR / "paired_morphology_learnability_comparison.json",
        "paired_morphology_learnability_table": TABLE_DIR / "table_paired_morphology_learnability.csv",
        "paired_morphology_learnability_manifest": MANIFEST_DIR / "paired_morphology_learnability_manifest.json",
        "pooling_sensitivity_external": METRIC_DIR / "pooling_sensitivity_external.csv",
        "pooling_q3_paired_bootstrap": METRIC_DIR / "pooling_q3_paired_bootstrap.json",
        "pooling_sensitivity_external_manifest": MANIFEST_DIR / "pooling_sensitivity_external_manifest.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if args.strict and missing:
        raise FileNotFoundError(
            "Missing required final evidence inputs: "
            + "; ".join(f"{name}={paths[name]}" for name in missing)
        )

    current_oof_sha256 = (
        sha256_file(paths["oof_predictions"]) if paths["oof_predictions"].exists() else ""
    )
    current_freeze_sha256 = (
        sha256_file(paths["freeze_manifest"]) if paths["freeze_manifest"].exists() else ""
    )
    canonical_contract = {
        "oof_sha256": current_oof_sha256,
        "freeze_sha256": current_freeze_sha256,
    }
    learned_robustness_audit = select_best_profile(
        REVISION_DIR,
        canonical_contract=canonical_contract,
        runner_path=PROJECT_ROOT / "scripts" / "revision" / "21_robustness_multicomparator.py",
        project_root=PROJECT_ROOT,
    )
    selected_robustness_paths = (
        (learned_robustness_audit.get("selected") or {}).get("paths") or {}
    )
    for label, value in selected_robustness_paths.items():
        optional_paths[f"learned_robustness_selected_{label}"] = Path(value)
    selected_robustness_evidence_paths = [
        rel(Path(value)) for value in selected_robustness_paths.values()
    ]

    calibration = read_json(paths["calibration"], required=args.strict)
    pooling_rows = read_csv_rows(paths["pooling"], required=args.strict)
    baseline_rows = read_csv_rows(paths["baseline"], required=args.strict)
    hrv_rows = read_csv_rows(paths["hrv_domain"], required=args.strict)
    robustness_rows = read_csv_rows(paths["robustness"], required=args.strict)
    paired_minirocket = read_json(paths["paired_minirocket"], required=False)
    paired_resnet = read_json(paths["paired_resnet"], required=False)
    paired_raw_mamba = read_json(optional_paths["paired_raw_mamba"], required=False)
    paired_transformer = read_json(optional_paths["paired_transformer"], required=False)
    paired_hybrid_morphology = read_json(optional_paths["paired_hybrid_morphology"], required=False)
    hybrid_morphology_summary = read_json(optional_paths["hybrid_morphology_summary"], required=False)
    claim_readiness_gates = read_json(optional_paths["claim_readiness_gates"], required=False)
    reviewer_gap_closure = read_json(paths["reviewer_gap_closure_status"], required=args.strict)
    reviewer_gap_closure_manifest = read_json(
        paths["reviewer_gap_closure_manifest"], required=args.strict
    )
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
    representation_probe_manifest = read_json(
        optional_paths["representation_probe_manifest"],
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
    # Legacy v1 score-calibration outputs are deliberately retained only as
    # provenance. They were row-split analyses and must never become a final
    # few-shot claim. The v2 artifacts below have explicit group-overlap audits.
    legacy_fewshot_dataset_summaries = {}
    for dataset in EXPECTED_EXTERNAL_DATASETS:
        manifest = read_json(optional_paths[f"fewshot_{dataset}_manifest"], required=False)
        legacy_fewshot_dataset_summaries[dataset] = {
            "dataset": dataset,
            "status": manifest.get("status", "missing"),
            "protocol": manifest.get("protocol", ""),
            "claim_ready": False,
            "reason": "legacy_row_split_score_calibration_not_group_safe",
        }
    score_calibration_summaries = {
        "ptbxl": summarize_external_adaptation(
            read_csv_rows(optional_paths["group_safe_score_calibration_ptbxl_summary"], required=False),
            read_json(optional_paths["group_safe_score_calibration_ptbxl_manifest"], required=False),
            "ptbxl",
            expected_status="complete_group_safe_score_calibration",
            expected_protocol="group_safe_score_calibration_v2_gated_external",
            adaptation_label="group_safe_score_calibration",
            canonical_contract=canonical_contract,
            runner_name="33_group_safe_score_calibration.py",
            primary_fraction=0.10,
            required_output_paths=(
                optional_paths["group_safe_score_calibration_ptbxl_summary"],
                optional_paths["group_safe_score_calibration_ptbxl_table"],
                optional_paths["group_safe_score_calibration_ptbxl_bootstrap"],
                optional_paths["group_safe_score_calibration_ptbxl_splits"],
                optional_paths["group_safe_score_calibration_ptbxl_coefficients"],
            ),
            required_manifest_fields={
                "fraction_unit": "independent_target_groups_from_adaptation_pool",
                "fraction_sampling": "nested_random_group_prefix_per_seed",
                "primary_fraction": 0.10,
                "primary_fraction_policy": "pre_specified_before_test_metric_evaluation",
            },
            required_row_fields={
                "fraction_unit": "independent_target_groups_from_adaptation_pool",
                "fraction_sampling": "nested_random_group_prefix_per_seed",
            },
            safe_wording=(
                "Report PTB-XL only as group-safe, dataset-specific score calibration of frozen predictions, "
                "using the pre-specified 10% target-group budget as primary and 1%/5% as sensitivity points. "
                "It can change fixed-threshold F1/calibration but cannot establish model-weight adaptation, "
                "few-shot fine-tuning, or broad transfer superiority."
            ),
        )
    }
    score_calibration_summary = combine_adaptation_summaries(
        score_calibration_summaries,
        label="group_safe_score_calibration",
        safe_wording=(
            "Use group-safe score-calibration results only for the named dataset and distinguish threshold/calibration "
            "changes from ranking metrics. Frozen model and encoder weights remain unchanged."
        ),
    )
    true_fewshot_summaries = {
        "ptbxl": summarize_external_adaptation(
            read_csv_rows(optional_paths["true_fewshot_head_ptbxl_summary"], required=False),
            read_json(optional_paths["true_fewshot_head_ptbxl_manifest"], required=False),
            "ptbxl",
            expected_status="complete_true_classifier_head_adaptation",
            expected_protocol="frozen_encoder_true_linear_head_adaptation_v2_group_safe_gated",
            adaptation_label="true_fewshot_frozen_encoder_head",
            canonical_contract=canonical_contract,
            runner_name="35_true_fewshot_head_adaptation.py",
            model_filter="full",
            primary_fraction=0.10,
            primary_rows=read_csv_rows(
                optional_paths["true_fewshot_head_ptbxl_primary"], required=False
            ),
            required_output_paths=(
                optional_paths["true_fewshot_head_ptbxl_summary"],
                optional_paths["true_fewshot_head_ptbxl_table"],
                optional_paths["true_fewshot_head_ptbxl_paired"],
                optional_paths["true_fewshot_head_ptbxl_primary"],
                optional_paths["true_fewshot_head_ptbxl_bootstrap"],
                optional_paths["true_fewshot_head_ptbxl_coefficients"],
                optional_paths["true_fewshot_head_ptbxl_splits"],
            ),
            required_manifest_fields={
                "fraction_unit": "independent_target_groups_from_adaptation_pool",
                "fraction_sampling": "nested_random_group_prefix_per_seed",
                "primary_fraction": 0.10,
                "primary_fraction_policy": "pre_specified_before_test_metric_evaluation",
            },
            required_row_fields={
                "fraction_unit": "independent_target_groups_from_adaptation_pool",
                "representation_pooling": "mean_of_preclassifier_slice_embeddings_per_fold",
            },
            safe_wording=(
                "Report PTB-XL results as group-safe fitting of new linear classifier heads on frozen encoder, "
                "mean-pooled record representations. Fractions are nested fractions of independent target groups. "
                "This is true parameter adaptation, but not end-to-end encoder fine-tuning or general superiority."
            ),
        )
    }
    true_fewshot_summary = combine_adaptation_summaries(
        true_fewshot_summaries,
        label="true_fewshot_frozen_encoder_head",
        safe_wording=(
            "Use true few-shot results only for the named dataset, frozen encoder, split protocol, and comparator set. "
            "Do not describe the experiment as end-to-end fine-tuning or broad transfer superiority."
        ),
    )
    fewshot_summary = {
        "complete": bool(score_calibration_summary.get("complete") or true_fewshot_summary.get("complete")),
        "status": "complete" if score_calibration_summary.get("complete") or true_fewshot_summary.get("complete") else "not_run_or_deferred",
        "key_numbers": "; ".join(
            item["key_numbers"]
            for item in (score_calibration_summary, true_fewshot_summary)
            if item.get("complete")
        ) or "adaptation_status=not_run_or_deferred",
        "safe_wording": " ".join(
            item["safe_wording"]
            for item in (score_calibration_summary, true_fewshot_summary)
        ),
        "blocker": "; ".join(
            item["blocker"]
            for item in (score_calibration_summary, true_fewshot_summary)
            if item.get("blocker")
        ),
    }
    a0 = read_json(paths["a0_status"], required=args.strict)
    claim_map = read_csv_rows(paths["claim_map"], required=args.strict)
    task_board = read_csv_rows(paths["task_board"], required=False)

    contract_issues: list[str] = []
    contract_issues.extend(
        reviewer_gap_closure_contract_issues(
            reviewer_gap_closure,
            reviewer_gap_closure_manifest,
            required_outputs=[
                paths["reviewer_gap_closure_status"],
                paths["reviewer_gap_closure_table"],
                paths["external_zero_target_ci_compact"],
                paths["external_zero_target_ci_compact_tex"],
                paths["pooling_cross_dataset_compact"],
                paths["pooling_cross_dataset_compact_tex"],
                paths["morphology_learnability_compact"],
                paths["morphology_learnability_compact_tex"],
                paths["robustness_six_stress_compact"],
                paths["robustness_six_stress_compact_tex"],
            ],
        )
    )
    external_comparator_payload = read_json(
        optional_paths["external_comparator_summary"], required=False
    )
    external_comparator_contract_issues = external_comparator_audit_contract_issues(
        [
            optional_paths["external_ptbxl_resnet_manifest"],
            optional_paths["external_ptbxl_raw_mamba_manifest"],
            optional_paths["external_georgia_resnet_manifest"],
            optional_paths["external_georgia_raw_mamba_manifest"],
        ],
        oof_sha256=current_oof_sha256,
        freeze_sha256=current_freeze_sha256,
    )
    external_paired_manifest = read_json(
        optional_paths["external_comparator_manifest"], required=False
    )
    external_paired_runner = (
        PROJECT_ROOT / "scripts" / "revision" / "32_paired_external_comparators.py"
    )
    if external_paired_manifest:
        if external_paired_manifest.get("status") != "complete":
            external_comparator_contract_issues.append(
                f"external paired manifest status={external_paired_manifest.get('status')!r}"
            )
        if external_paired_manifest.get("canonical_contract") != canonical_contract:
            external_comparator_contract_issues.append(
                "external paired manifest canonical OOF/freeze mismatch"
            )
        if (
            not external_paired_runner.exists()
            or external_paired_manifest.get("runner_sha256")
            != sha256_file(external_paired_runner)
        ):
            external_comparator_contract_issues.append(
                "external paired manifest runner SHA mismatch"
            )
    external_comparator_audit = summarize_external_comparator_audit(
        list(external_comparator_payload.get("rows") or []),
        external_paired_manifest,
        external_comparator_contract_issues,
    )
    if calibration.get("predictions_sha256") != current_oof_sha256:
        contract_issues.append("Calibration CI: OOF SHA mismatch")
    if calibration.get("freeze_manifest_sha256") != current_freeze_sha256:
        contract_issues.append("Calibration CI: freeze manifest SHA mismatch")
    for label, paired_payload, required in (
        ("Fixed-transform-only (legacy MiniRocket artifact)", paired_minirocket, True),
        ("ResNet1D/CNN", paired_resnet, True),
        ("Raw Mamba", paired_raw_mamba, False),
        ("Transformer ECG", paired_transformer, False),
        ("Frozen-transform MLP head", paired_hybrid_morphology, False),
    ):
        if paired_payload or required:
            contract_issues.extend(
                paired_oof_contract_issues(
                    paired_payload,
                    label=label,
                    oof_sha256=current_oof_sha256,
                    freeze_sha256=current_freeze_sha256,
                )
            )
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
    hybrid_morphology = (
        {
            **(hybrid_morphology_summary.get("metrics", {}) if isinstance(hybrid_morphology_summary, dict) else {}),
            "status": "complete_optional_morphology_sensitivity",
        }
        if isinstance(hybrid_morphology_summary, dict) and hybrid_morphology_summary.get("metrics")
        else {}
    )
    q3 = pooling_by_name.get("power_mean_q3", {})
    robustness_summary = summarize_robustness(robustness_rows)
    representation_summary = summarize_representation(
        representation_status or representation_probe_summary,
        representation_probe_manifest,
        representation_probe_rows,
        representation_cka_rows,
        canonical_contract,
    )
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
        "No unqualified external-transfer or cross-dataset performance-advantage claim is supported. "
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
    paired_hybrid_metrics = (
        paired_hybrid_morphology.get("metrics", {}) if isinstance(paired_hybrid_morphology, dict) else {}
    )
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
    paired_hybrid_f1 = paired_hybrid_metrics.get("f1_macro", {})
    paired_hybrid_pr = paired_hybrid_metrics.get("pr_auc_macro", {})

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
    hybrid_key_numbers = (
        f"; Hybrid fixed-transform MLP PR-AUC={fmt(hybrid_morphology.get('pr_auc_macro'))}, "
        f"F1={fmt(hybrid_morphology.get('f1_macro'))}"
        if hybrid_morphology
        else ""
    )
    hybrid_paired_key_numbers = (
        f"; Hybrid fixed-transform MLP paired F1={paired_hybrid_f1.get('interpretation', '')}, "
        f"PR-AUC={paired_hybrid_pr.get('interpretation', '')}"
        if paired_hybrid_metrics
        else "; Hybrid fixed-transform MLP paired comparison=not_run_optional"
    )
    hybrid_evidence_paths = (
        ";reports/revision/metrics/paired_full_vs_hybrid_morphology_comparison.json;"
        "reports/revision/tables/table_paired_full_vs_hybrid_morphology.csv"
        if paired_hybrid_metrics
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
        f" Few-shot deferred note: {fewshot_summary['blocker']}"
        if fewshot_summary.get("blocker")
        else ""
    )
    fewshot_evidence_paths = []
    if score_calibration_summary.get("complete"):
        fewshot_evidence_paths.extend(
            [
                "reports/revision/metrics/group_safe_score_calibration_ptbxl_summary.csv",
                "reports/revision/tables/table_group_safe_score_calibration_ptbxl.csv",
                "reports/revision/metrics/group_safe_score_calibration_ptbxl_bootstrap.json",
                "reports/revision/manifests/group_safe_score_calibration_ptbxl_splits.npz",
                "reports/revision/manifests/group_safe_score_calibration_ptbxl_manifest.json",
            ]
        )
    if true_fewshot_summary.get("complete"):
        fewshot_evidence_paths.extend(
            [
                "reports/revision/metrics/true_fewshot_head_ptbxl_summary.csv",
                "reports/revision/tables/table_true_fewshot_head_ptbxl_paired.csv",
                "reports/revision/tables/table_true_fewshot_head_ptbxl_primary.csv",
                "reports/revision/metrics/true_fewshot_head_ptbxl_bootstrap.json",
                "reports/revision/manifests/true_fewshot_head_ptbxl_splits.npz",
                "reports/revision/manifests/true_fewshot_head_ptbxl_manifest.json",
            ]
        )
    fewshot_evidence_path_text = (
        ";" + ";".join(fewshot_evidence_paths) if fewshot_evidence_paths else ""
    )
    reviewer_gap_by_item = {
        str(item.get("reviewer_item")): item
        for item in reviewer_gap_closure.get("rows") or []
        if isinstance(item, dict) and item.get("reviewer_item")
    }
    reviewer_gap_status_text = ", ".join(
        f"{item}={reviewer_gap_by_item.get(item, {}).get('status', 'missing')}"
        for item in ("R1-C2", "R1-C5", "R1-C6", "R2-C3")
    )

    matrix_rows = [
        {
            "claim_id": "C01",
            "claim_topic": "Fair baseline superiority / external transfer",
            "evidence_status": c01_evidence_status,
            "key_numbers": (
                f"Full PR-AUC={fmt(full.get('pr_auc_macro'))}, F1={fmt(full.get('f1_macro'))}; "
                f"Fixed-transform-only PR-AUC={fmt(mini.get('pr_auc_macro'))}, F1={fmt(mini.get('f1_macro'))}; "
                f"ResNet1D/CNN PR-AUC={fmt(resnet.get('pr_auc_macro'))}, F1={fmt(resnet.get('f1_macro'))}; "
                f"Raw Mamba PR-AUC={fmt(raw_mamba.get('pr_auc_macro'))}, F1={fmt(raw_mamba.get('f1_macro'))}"
                f"{transformer_key_numbers}"
                f"{hybrid_key_numbers}; {external_comparator_audit['key_numbers']}; "
                f"{learned_robustness_audit['key_numbers']}; reviewer closure: {reviewer_gap_status_text}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/baseline_summary.csv;"
                "reports/revision/metrics/paired_full_vs_minirocket_comparison.json;"
                "reports/revision/metrics/paired_full_vs_resnet_comparison.json;"
                "reports/revision/metrics/paired_full_vs_raw_mamba_comparison.json"
                f"{transformer_evidence_paths}"
                f"{hybrid_evidence_paths};reports/revision/metrics/external_comparator_paired_summary.json;"
                "reports/revision/tables/table_external_comparator_paired.csv;"
                "reports/revision/tables/table_external_zero_target_ci_compact.csv;"
                "reports/revision/tables/table_morphology_learnability_compact.csv;"
                "reports/revision/tables/table_robustness_six_stress_compact.csv;"
                + ";".join(selected_robustness_evidence_paths)
            ),
            "safe_wording": (
                "Do not claim superiority over all fair baselines. Report comparator-specific, "
                "metric-specific paired deltas. In-domain fair comparators show ResNet1D/CNN "
                "and Raw Mamba are stronger on discrimination/F1 metrics; narrow ECG-RAMBA "
                "claims to supported calibration tradeoffs, architecture analysis, and "
                "documented limitations. "
                f"{external_comparator_audit['safe_wording']} "
                f"{learned_robustness_audit['safe_wording']} "
                "The reduced morphology learnability control is a bounded mechanism sensitivity, not a causal explanation of the full model."
            ),
            "blocker": "; ".join(
                item
                for item in [
                    c01_blocker,
                    external_comparator_audit["blocker"],
                    learned_robustness_audit["blocker"],
                ]
                if item
            ),
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
                f"{hybrid_paired_key_numbers}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/calibration_ci_oof_final_ema_predictions.json;"
                "reports/revision/tables/table_paired_full_vs_minirocket.csv;"
                "reports/revision/tables/table_paired_full_vs_resnet.csv;"
                "reports/revision/tables/table_paired_full_vs_raw_mamba.csv"
                f"{transformer_evidence_paths}"
                f"{hybrid_evidence_paths}"
            ),
            "safe_wording": (
                "Frozen OOF supports only metric-specific operating-point statements. ECG-RAMBA "
                "has calibration/error advantages over the fixed-transform-only comparator and Raw Mamba, but "
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
                "reports/revision/tables/table_representation_probe_by_fold.csv;"
                "reports/revision/tables/table_representation_cka.csv;"
                "reports/revision/figures/figure_representation_audit.png;"
                "reports/revision/manifests/representation_probe_manifest.json"
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
                f"ROC-AUC={fmt(q3.get('roc_auc_macro'))}, F1={fmt(q3.get('f1_macro'))}; "
                f"cross-dataset gate={reviewer_gap_by_item.get('R1-C6', {}).get('status', 'missing')}"
            ),
            "evidence_paths": (
                "reports/revision/metrics/pooling_sensitivity.csv;"
                "reports/revision/metrics/pooling_decision_summary.json;"
                "reports/revision/metrics/pooling_sensitivity_external.csv;"
                "reports/revision/metrics/pooling_q3_paired_bootstrap.json;"
                "reports/revision/tables/table_pooling_cross_dataset_compact.csv"
            ),
            "safe_wording": (
                "Present Q=3 as the pre-specified/frozen operating point and show the group-bootstrap "
                "sensitivity separately for Chapman, PTB-XL, Georgia, and the CPSC2021 mapped-window task. "
                "Treat it as a tested tradeoff, not a universally optimal pooling rule."
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
                "reports/revision/metrics/external_protocol_gate_summary.csv"
                f"{fewshot_evidence_path_text}"
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
            "global_superiority": "Avoid broad fair-baseline advantage wording.",
            "resnet_in_domain": (
                "The completed paired ResNet1D/CNN comparison favors ResNet on frozen Chapman OOF "
                "PR-AUC, ROC-AUC, F1, Brier, and ECE; do not claim an ECG-RAMBA in-domain "
                "performance advantage over fair CNN/ResNet baselines."
            ),
            "operating_point": (
                "ECG-RAMBA operating-point advantages are comparator-specific. The fixed-transform-only "
                "F1/Brier/ECE result does not generalize to ResNet1D/CNN."
            ),
            "robustness": (
                "Use only metric-specific robustness claims supported by paired degradation CIs. "
                f"{learned_robustness_audit['safe_wording']} "
                "Treat the multi-comparator 95% intervals as pointwise; they are not multiplicity-adjusted across the full stress/comparator/metric grid."
            ),
            "fewshot": (
                "Legacy row-split score calibration is not claim-ready. Group-safe score calibration changes no "
                "model weights. True few-shot evidence requires the frozen-encoder linear-head protocol and must "
                "not be described as end-to-end fine-tuning or broad transfer superiority."
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
                "Use only the v3 fold-aware projection/probe/CKA audit. CKA may show branch embeddings are not "
                "identical, but weak fold-safe linear probes do not support established morphology-rhythm separation."
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
            "hybrid_morphology": (
                "Use the frozen ROCKET-family transform MLP only as optional morphology-head sensitivity evidence if "
                "scripts/revision/26_hybrid_morphology_baseline.py and paired bootstrap output are complete. "
                "It must not be used as causal proof of deterministic morphology, regularization, or disentanglement."
            ),
            "external_comparators": (
                "PTB-XL and Georgia learned-comparator audits must remain separate mapped tasks. Run ResNet1D/CNN and "
                "Raw Mamba first; Transformer external inference is secondary to a complete in-domain Transformer gate. "
                "CPSC2021 is a separate 10-second AF/AFL mapped-window task."
            ),
            "marked_manuscript": (
                "The editorial marked/highlighted manuscript is complete only after latexdiff and LaTeX create a verified PDF."
            ),
            "claim_readiness_gates": (
                "Use scripts/revision/28_claim_readiness_gates.py as a blocker ledger for optional or "
                "not-supported claims; blocked rows must not be converted into positive manuscript claims."
            ),
            "reviewer_gap_closure": (
                "Use reviewer_gap_closure_status.json and its authenticated compact tables for R1-C2, R1-C5, "
                "R1-C6, and R2-C3. A reviewer item is usable only when its manuscript_ready field is true."
            ),
            "morphology_learnability": (
                "The controlled reduced-bank frozen-versus-partially-learnable comparison isolates kernel "
                "learnability only within that matched sensitivity experiment; it is not causal proof for ECG-RAMBA."
            ),
            "external_zero_target_ci": (
                "Report paired bootstrap CIs separately by dataset, comparator, and metric, with Holm-adjusted conclusions for each dataset family. "
                "PTB-XL uses patient IDs, CPSC2021 uses source-record groups, and Georgia uses record-level resampling under an explicit independence assumption because patient IDs are unavailable. "
                "Do not pool PTB-XL, Georgia, and CPSC2021 or claim general zero-shot superiority."
            ),
            "pooling_cross_dataset": (
                "Q=3 is the frozen operating point with cross-dataset sensitivity evidence, not a universally optimal rule."
            ),
        },
        "inputs": {name: artifact(path) for name, path in paths.items()},
        "optional_inputs": {name: artifact(path) for name, path in optional_paths.items()},
        "claim_readiness_gates": claim_readiness_gates if isinstance(claim_readiness_gates, dict) else {},
        "reviewer_gap_closure": reviewer_gap_closure if isinstance(reviewer_gap_closure, dict) else {},
        "external_gate_summary": {
            "expected_datasets": list(EXPECTED_EXTERNAL_DATASETS),
            "status": external_gate_status,
            "passed": external_gate_passed,
            "blocked": external_gate_blocked,
            "deferred": external_gate_deferred,
        },
        "fewshot_summary": fewshot_summary,
        "legacy_row_split_score_calibration": legacy_fewshot_dataset_summaries,
        "group_safe_score_calibration_summary": score_calibration_summary,
        "true_fewshot_head_adaptation_summary": true_fewshot_summary,
        "external_learned_comparator_audit": external_comparator_audit,
        "robustness_summary": robustness_summary,
        "learned_comparator_robustness_audit": learned_robustness_audit,
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
    save_json_atomic(args.out_json, payload)

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
        "optional_input_sha256": {
            name: sha256_file(path)
            for name, path in optional_paths.items()
            if path.exists() and path.stat().st_size > 0
        },
        "selected_learned_robustness_profile": learned_robustness_audit.get(
            "selected_profile"
        ),
        "final_ready_for_rebuttal": final_ready,
        "all_claims_supported": False,
    }
    save_json_atomic(args.out_manifest, manifest)

    print(json.dumps({
        "status": True,
        "final_ready_for_rebuttal": final_ready,
        "evidence_rows": len(matrix_rows),
        "robustness_rows": len(robustness_claims),
        "outputs": payload["outputs"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
