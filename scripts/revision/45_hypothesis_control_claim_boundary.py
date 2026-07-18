"""Build the hypothesis-control-finding-claim-boundary evidence ledger."""

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
    git_commit,
    save_csv,
    save_json,
    sha256_file,
)
from src.features import checkpoint_compatible_hrv36_contract  # noqa: E402


SCHEMA_VERSION = 2
TITLE = "ECG-RAMBA: A Protocol-Faithful Evaluation of Structured Morphology-Rhythm ECG Modeling"
CENTRAL_QUESTION = (
    "We test whether explicit morphology and rhythm interfaces provide benefits that survive "
    "matched comparisons in discrimination, calibration, perturbation robustness, and "
    "target-label adaptation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_hypothesis_control_finding_claim_boundary.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "hypothesis_control_claim_boundary.json",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "hypothesis_control_claim_boundary_manifest.json",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=TABLE_DIR / "table_hypothesis_control_finding_claim_boundary.tex",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    path = resolve(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> list[dict]:
    path = resolve(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def report_relative(raw_path: str | Path) -> str:
    normalized = str(raw_path).replace("\\", "/")
    marker = "reports/revision/"
    if marker in normalized:
        return marker + normalized.split(marker, 1)[1]
    return normalized


def manifest_output_hashes(manifest: dict) -> dict[str, str]:
    outputs = manifest.get("outputs") or {}
    rows = outputs if isinstance(outputs, list) else [
        {"path": path, "sha256": sha}
        for path, sha in outputs.items()
        if isinstance(sha, str)
    ]
    return {
        report_relative(row.get("path", "")): str(row.get("sha256", ""))
        for row in rows
        if isinstance(row, dict) and row.get("path") and row.get("sha256")
    }


def manifest_authenticates(
    manifest_path: Path,
    required_paths: list[Path],
    *,
    protocol: str,
    accepted_statuses: set[str],
) -> bool:
    manifest = read_json(manifest_path)
    if manifest.get("protocol") != protocol or manifest.get("status") not in accepted_statuses:
        return False
    output_hashes = manifest_output_hashes(manifest)
    for path in required_paths:
        resolved = resolve(path)
        if not resolved.is_file() or resolved.stat().st_size == 0:
            return False
        if output_hashes.get(report_relative(path)) != sha256_file(resolved):
            return False
    return True


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_tex_table(path: Path, rows: list[dict]) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Hypothesis--control--finding--claim-boundary ledger. Findings are activated only by authenticated final-evidence artifacts.}",
        r"\label{tab:hypothesis_claim_boundary}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabularx}{\textwidth}{p{0.13\textwidth}p{0.20\textwidth}X p{0.27\textwidth}}",
        r"\toprule",
        r"Hypothesis & Matched control & Finding & Claim boundary \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} \\\\".format(
                latex_escape(row["hypothesis"]),
                latex_escape(row["matched_control"]),
                latex_escape(row["finding"]),
                latex_escape(row["claim_boundary"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_wording(topic: str) -> str:
    rows = read_csv(TABLE_DIR / "table_final_safe_wording.csv")
    for row in rows:
        if topic.lower() in str(row.get("claim_topic", "")).lower():
            return str(row.get("safe_wording", "")).strip()
    return ""


def ablation_finding(control: str) -> tuple[str, str]:
    summary_path = METRIC_DIR / "structured_ablation_5fold_summary.json"
    paired_path = TABLE_DIR / "table_paired_structured_ablation_5fold.csv"
    summary = read_json(summary_path)
    rows = read_csv(paired_path)
    selected = [row for row in rows if row.get("comparison") == f"full_vs_{control}"]
    expected_metrics = {"pr_auc_macro", "roc_auc_macro", "f1_macro", "brier_macro", "ece_macro"}
    selected_metrics = {str(row.get("metric", "")) for row in selected}
    authenticated = manifest_authenticates(
        MANIFEST_DIR / "structured_ablation_5fold_manifest.json",
        [summary_path, paired_path],
        protocol="matched_retrained_structured_ablation_5fold_v3",
        accepted_statuses={"complete"},
    )
    if (
        summary.get("status") != "complete"
        or summary.get("paired_analysis_uses_fresh_matched_full") is not True
        or (summary.get("training_contract") or {}).get("hrv36_feature_contract")
        != checkpoint_compatible_hrv36_contract()
        or not (summary.get("training_contract") or {}).get("pca_contract")
        or len(summary.get("pca_contract_by_fold") or {}) != 5
        or len(selected) != len(expected_metrics)
        or selected_metrics != expected_metrics
        or int((summary.get("training_contract") or {}).get("n_boot", 0)) != 1000
        or any(int(float(row.get("n_boot_valid", 0))) != 1000 for row in selected)
        or any(not row.get("ci_low") or not row.get("ci_high") for row in selected)
        or (summary.get("paired_bootstrap_contract") or {}).get("complete") is not True
        or not authenticated
    ):
        return (
            "pending_matched_retraining",
            "Five-fold matched removal evidence is not complete; no component-benefit conclusion is entered.",
        )
    counts = {
        label: sum(row.get("interpretation") == label for row in selected)
        for label in [
            "full_significantly_better",
            "control_significantly_better",
            "inconclusive",
        ]
    }
    return (
        "complete_paired_ablation",
        (
            f"Across {len(selected)} pre-specified metrics, paired record bootstrap favors Full on "
            f"{counts['full_significantly_better']}, favors the removal control on "
            f"{counts['control_significantly_better']}, and is inconclusive on "
            f"{counts['inconclusive']}."
        ),
    )


def calibration_finding() -> tuple[str, str]:
    summary_path = METRIC_DIR / "matched_oof_calibration_summary.json"
    bootstrap_path = METRIC_DIR / "matched_oof_calibration_bootstrap.json"
    paired_path = TABLE_DIR / "table_paired_matched_oof_calibration.csv"
    summary = read_json(summary_path)
    bootstrap = read_json(bootstrap_path)
    authenticated = manifest_authenticates(
        MANIFEST_DIR / "matched_oof_calibration_manifest.json",
        [summary_path, bootstrap_path, paired_path],
        protocol="matched_cross_fitted_per_class_platt_v2",
        accepted_statuses={"complete"},
    )
    if (
        summary.get("status") != "complete"
        or "full" not in summary.get("models", {})
        or summary.get("bootstrap_unit") != "Chapman record; one record per subject"
        or "pools record-class label instances"
        not in str(summary.get("reliability_figure_scope", ""))
        or (summary.get("completeness_contract") or {}).get("coefficient_grid_complete") is not True
        or (summary.get("completeness_contract") or {}).get("raw_vs_calibrated_bootstrap_complete") is not True
        or (summary.get("completeness_contract") or {}).get("matched_model_bootstrap_complete") is not True
        or int((summary.get("completeness_contract") or {}).get("required_valid_bootstrap_replicates", 0)) != 1000
        or not authenticated
    ):
        return (
            "pending_matched_calibration",
            "Cross-fitted OOF-score calibration results are not complete; raw calibration claims remain unchanged.",
        )
    full = summary["models"]["full"]
    raw = full.get("raw", {})
    calibrated = full.get("cross_fitted_platt", {})
    interpretations = bootstrap.get("models", {}).get("full", {})
    improved = sorted(
        metric
        for metric, payload in interpretations.items()
        if payload.get("interpretation") == "calibrated_significantly_better"
    )
    return (
        "complete_oof_score_calibration_sensitivity",
        (
            f"For ECG-RAMBA, cross-fitted OOF-score Platt changes Brier {raw.get('brier_macro', float('nan')):.4f}"
            f" to {calibrated.get('brier_macro', float('nan')):.4f}, ECE "
            f"{raw.get('ece_macro', float('nan')):.4f} to "
            f"{calibrated.get('ece_macro', float('nan')):.4f}; CI-supported improvements: "
            f"{', '.join(improved) if improved else 'none'}."
        ),
    )


def adaptation_finding() -> tuple[str, str]:
    primary_path = TABLE_DIR / "table_true_fewshot_head_ptbxl_primary.csv"
    learning_curve_path = TABLE_DIR / "table_true_fewshot_head_ptbxl_learning_curve.csv"
    rows = read_csv(primary_path)
    learning_curve = read_csv(learning_curve_path)
    learning_curve_figure = resolve(
        Path("reports/revision/figures/figure_true_fewshot_head_ptbxl_learning_curve.png")
    )
    selected = [
        row
        for row in rows
        if row.get("comparison_type") == "adapted_vs_zero_target_label"
        and row.get("model") == "full"
        and row.get("metric") == "f1_macro"
    ]
    authenticated = manifest_authenticates(
        MANIFEST_DIR / "true_fewshot_head_ptbxl_manifest.json",
        [primary_path, learning_curve_path, learning_curve_figure],
        protocol="frozen_encoder_true_linear_head_adaptation_v2_group_safe_gated",
        accepted_statuses={"complete_true_classifier_head_adaptation"},
    )
    if not selected or not authenticated:
        return "pending_learning_curve", "The true frozen-encoder adaptation endpoint is unavailable."
    row = selected[0]
    expected_curve_keys = {
        (model, fraction, metric)
        for model in ("full", "resnet", "raw_mamba", "transformer")
        for fraction in (0.0, 0.01, 0.05, 0.10)
        for metric in ("f1_macro", "pr_auc_macro")
    }
    observed_curve_keys = {
        (
            str(item.get("model", "")),
            float(item.get("fraction", "nan")),
            str(item.get("metric", "")),
        )
        for item in learning_curve
        if item.get("comparison_type") == "adapted_vs_zero_target_label"
    }
    curve_complete = (
        expected_curve_keys.issubset(observed_curve_keys)
        and learning_curve_figure.exists()
        and learning_curve_figure.stat().st_size > 0
    )
    return (
        "complete_learning_curve" if curve_complete else "complete_endpoint_pending_learning_curve_asset",
        (
            f"On PTB-XL fold 9-to-10 frozen-encoder adaptation, ECG-RAMBA macro-F1 changes from "
            f"{float(row['zero_target_label_value']):.4f} to "
            f"{float(row['primary_value_mean_across_seeds']):.4f} at 10% target groups "
            f"(within-model gain {float(row['improvement_primary_over_zero']):+.4f}, "
            f"95% CI [{float(row['improvement_ci_low']):+.4f}, "
            f"{float(row['improvement_ci_high']):+.4f}])."
        ),
    )


def robustness_finding() -> tuple[str, str]:
    summary_path = METRIC_DIR / "robustness_multicomparator_summary.csv"
    table_path = TABLE_DIR / "table_robustness_multicomparator.csv"
    pairwise_path = METRIC_DIR / "robustness_multicomparator_pairwise.json"
    manifest = read_json(MANIFEST_DIR / "robustness_multicomparator_manifest.json")
    rows = read_csv(table_path)
    expected_stresses = {
        "snr20db",
        "snr10db",
        "snr5db",
        "random_3_lead_dropout",
        "precordial_dropout",
        "resample_250hz",
    }
    expected_comparators = {"minirocket", "resnet", "raw_mamba", "transformer"}
    expected_metrics = {"pr_auc_macro", "roc_auc_macro", "f1_macro", "brier_macro", "ece_macro"}
    expected_keys = {
        (stress, comparator, metric)
        for stress in expected_stresses
        for comparator in expected_comparators
        for metric in expected_metrics
    }
    observed_keys = {
        (str(row.get("stress", "")), str(row.get("comparator", "")), str(row.get("metric", "")))
        for row in rows
    }
    artifact_sha = manifest.get("artifact_sha256") or {}
    authenticated = (
        manifest.get("status") == "complete"
        and manifest.get("protocol") == "robustness_multicomparator_aggregation_v1"
        and set(manifest.get("stress_tests") or []) == expected_stresses
        and expected_comparators.issubset(set(manifest.get("comparators") or []))
        and all(path.is_file() and path.stat().st_size > 0 for path in [summary_path, table_path, pairwise_path])
        and artifact_sha.get("summary") == sha256_file(summary_path)
        and artifact_sha.get("table") == sha256_file(table_path)
        and artifact_sha.get("pairwise") == sha256_file(pairwise_path)
    )
    if (
        len(rows) != len(expected_keys)
        or observed_keys != expected_keys
        or any(row.get("status") != "complete" for row in rows)
        or not authenticated
    ):
        return (
            "pending_complete_learned_comparator_ledger",
            "The canonical six-stress, four-comparator, five-metric ledger is not authenticated as complete.",
        )
    interpretations = [str(row.get("interpretation", "")) for row in rows]
    full = sum(value.startswith("full_significantly") for value in interpretations)
    comparator = sum(value.startswith("comparator_significantly") for value in interpretations)
    inconclusive = len(rows) - full - comparator
    return (
        "complete_metric_specific_robustness_ledger",
        (
            f"Across 120 nominal pointwise paired degradation intervals, {full} favor Full, "
            f"{comparator} favor a comparator, and {inconclusive} overlap zero."
        ),
    )


def external_finding() -> tuple[str, str]:
    rows = read_csv(METRIC_DIR / "external_protocol_gate_summary.csv")
    passed_rows = [
        row
        for row in rows
        if str(row.get("protocol_gate_passed", "")).lower() == "true"
        and str(row.get("manuscript_ready", "")).lower() == "true"
    ]
    passed = []
    for row in passed_rows:
        dataset = str(row.get("dataset", ""))
        manifest = read_json(MANIFEST_DIR / f"external_{dataset}_protocol_gate_manifest.json")
        gate_path = METRIC_DIR / f"external_{dataset}_protocol_gate.json"
        metrics_path = TABLE_DIR / f"table_external_{dataset}_metrics.csv"
        artifacts = manifest.get("artifacts") or {}
        gate_artifact = artifacts.get("gate_json") or {}
        metrics_artifact = artifacts.get("metrics_table") or {}
        authenticated = (
            manifest.get("dataset") == dataset
            and manifest.get("status") == "protocol_gate_passed"
            and manifest.get("protocol_gate_passed") is True
            and not manifest.get("issues")
            and gate_path.is_file()
            and metrics_path.is_file()
            and gate_artifact.get("sha256") == sha256_file(gate_path)
            and metrics_artifact.get("sha256") == sha256_file(metrics_path)
        )
        if authenticated:
            passed.append(dataset)
    passed = sorted(passed)
    if not passed:
        return "pending_external_protocol_gates", "No mapped-task external protocol gate is complete."
    return (
        "complete_protocol_gated_mapped_tasks",
        (
            "Dataset-specific protocol gates pass for "
            + ", ".join(passed)
            + "; tasks and comparator effects remain separate rather than pooled."
        ),
    )


def main() -> None:
    args = parse_args()
    morphology_status, morphology_finding = ablation_finding("no_morphology")
    rhythm_status, rhythm_finding = ablation_finding("no_rhythm")
    fusion_status, fusion_finding = ablation_finding("no_context_fusion")
    calibration_status, calibration_text = calibration_finding()
    adaptation_status, adaptation_text = adaptation_finding()
    robustness_status, robustness_text = robustness_finding()
    external_status, external_text = external_finding()

    rows = [
        {
            "hypothesis": "Morphology interface",
            "matched_control": (
                "Five-fold retrained removal of the fixed-transform morphology stream and its dependent "
                "cross-attention interaction; raw ECG retained"
            ),
            "evidence_status": morphology_status,
            "finding": morphology_finding,
            "claim_boundary": (
                "A Full advantage supports the within-architecture contribution of the morphology/fusion "
                "interface as a unit; it does not isolate the fixed transform, prove exclusive morphology "
                "encoding, or establish global model superiority."
            ),
        },
        {
            "hypothesis": "Rhythm interface",
            "matched_control": (
                "Five-fold retrained removal of the checkpoint-compatible five-RR-plus-six-global-statistics "
                "conditioning interface; raw ECG retained"
            ),
            "evidence_status": rhythm_status,
            "finding": rhythm_finding,
            "claim_boundary": (
                "The control concerns the implemented checkpoint-compatible rhythm statistics; it does "
                "not imply that RMSSD/SDNN/LF-HF were present or that rhythm is isolated to one branch."
            ),
        },
        {
            "hypothesis": "Context/fusion stack",
            "matched_control": "Five-fold retrained joint No-context/fusion stack",
            "evidence_status": fusion_status,
            "finding": fusion_finding,
            "claim_boundary": (
                "This joint removal tests the stack as a unit and cannot identify cross-attention, "
                "Perceiver, or BiMamba as the sole causal mechanism."
            ),
        },
        {
            "hypothesis": "Score calibration",
            "matched_control": "Raw vs per-class cross-fitted Platt on frozen OOF scores",
            "evidence_status": calibration_status,
            "finding": calibration_text,
            "claim_boundary": (
                "Secondary post-hoc OOF-score audit, not a fully nested calibration-pipeline estimate; "
                "improvement does not compensate for lower discrimination or establish clinical threshold safety."
            ),
        },
        {
            "hypothesis": "Perturbation robustness",
            "matched_control": "Same record perturbations and paired degradation CIs across learned comparators",
            "evidence_status": robustness_status,
            "finding": robustness_text,
            "claim_boundary": "No general robustness superiority; report only named stress, metric, and comparator.",
        },
        {
            "hypothesis": "External transfer",
            "matched_control": (
                "Dataset-specific mapped-task protocol gates; learned-comparator effects remain separate"
            ),
            "evidence_status": external_status,
            "finding": external_text,
            "claim_boundary": (
                "No unqualified zero-shot or cross-dataset superiority; CPSC2021 is an annotation-aligned "
                "10-second mapped-window task, not the official challenge endpoint."
            ),
        },
        {
            "hypothesis": "Target-label adaptation",
            "matched_control": "Shared PTB-XL fold 9 adaptation pool, fold 10 test, 0/1/5/10%, five seeds",
            "evidence_status": adaptation_status,
            "finding": adaptation_text,
            "claim_boundary": (
                "Frozen-encoder linear-head adaptation learning curve; absolute comparator performance "
                "is reported separately and no general few-shot superiority is claimed."
            ),
        },
    ]
    required_complete = {
        "morphology": morphology_status == "complete_paired_ablation",
        "rhythm": rhythm_status == "complete_paired_ablation",
        "context_fusion": fusion_status == "complete_paired_ablation",
        "calibration": calibration_status == "complete_oof_score_calibration_sensitivity",
        "robustness": robustness_status == "complete_metric_specific_robustness_ledger",
        "external": external_status == "complete_protocol_gated_mapped_tasks",
        "adaptation": adaptation_status == "complete_learning_curve",
    }
    payload = {
        "status": "complete" if all(required_complete.values()) else "incomplete_required_experiments",
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "recommended_title": TITLE,
        "central_question": CENTRAL_QUESTION,
        "required_complete": required_complete,
        "rows": rows,
    }
    save_csv(resolve(args.out_table), rows)
    save_json(resolve(args.out_json), payload)
    write_tex_table(args.out_tex, rows)
    manifest = {
        "status": payload["status"],
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "outputs": {
            str(path): sha256_file(resolve(path))
            for path in [args.out_table, args.out_json, args.out_tex]
        },
    }
    save_json(resolve(args.out_manifest), manifest)
    print(json.dumps({"status": payload["status"], "required_complete": required_complete}, indent=2))
    if args.strict and payload["status"] != "complete":
        missing = [name for name, complete in required_complete.items() if not complete]
        raise RuntimeError(f"Hypothesis-testing evidence is incomplete: {missing}")


if __name__ == "__main__":
    main()
