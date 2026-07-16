"""Validate and present the four reviewer-gap closure packages.

The runner is presentation-only: it never recomputes predictions or bootstrap
replicates. It validates the producing manifests/tables and writes compact CSV
and LaTeX tables plus a machine-readable closure gate for R1-C2, R1-C5,
R1-C6, and R2-C3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    save_json,
    sha256_file,
)


DATASETS = ("ptbxl", "georgia", "cpsc2021")
EXTERNAL_COMPARATORS = ("resnet", "raw_mamba", "transformer")
ROBUSTNESS_COMPARATORS = ("minirocket", "resnet", "raw_mamba", "transformer")
STRESSES = (
    "snr20db",
    "snr10db",
    "snr5db",
    "random_3_lead_dropout",
    "precordial_dropout",
    "resample_250hz",
)
METRICS = ("pr_auc_macro", "roc_auc_macro", "f1_macro", "brier_macro", "ece_macro")
POOLING_METHODS = (
    "mean",
    "power_mean_q2",
    "power_mean_q3",
    "power_mean_q4",
    "power_mean_q8",
    "max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--external-table",
        type=Path,
        default=TABLE_DIR / "table_external_comparator_paired.csv",
    )
    parser.add_argument(
        "--external-summary",
        type=Path,
        default=METRIC_DIR / "external_comparator_paired_summary.json",
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=MANIFEST_DIR / "external_comparator_paired_manifest.json",
    )
    parser.add_argument(
        "--pooling-table",
        type=Path,
        default=TABLE_DIR / "table_pooling_sensitivity_across_datasets.csv",
    )
    parser.add_argument(
        "--pooling-bootstrap",
        type=Path,
        default=METRIC_DIR / "pooling_q3_paired_bootstrap.json",
    )
    parser.add_argument(
        "--pooling-manifest",
        type=Path,
        default=MANIFEST_DIR / "pooling_sensitivity_external_manifest.json",
    )
    parser.add_argument(
        "--morphology-table",
        type=Path,
        default=TABLE_DIR / "table_paired_morphology_learnability.csv",
    )
    parser.add_argument(
        "--morphology-json",
        type=Path,
        default=METRIC_DIR / "paired_morphology_learnability_comparison.json",
    )
    parser.add_argument(
        "--morphology-manifest",
        type=Path,
        default=MANIFEST_DIR / "paired_morphology_learnability_manifest.json",
    )
    parser.add_argument(
        "--robustness-table",
        type=Path,
        default=TABLE_DIR / "table_robustness_multicomparator.csv",
    )
    parser.add_argument(
        "--robustness-pairwise",
        type=Path,
        default=METRIC_DIR / "robustness_multicomparator_pairwise.json",
    )
    parser.add_argument(
        "--robustness-manifest",
        type=Path,
        default=MANIFEST_DIR / "robustness_multicomparator_manifest.json",
    )
    parser.add_argument(
        "--out-status",
        type=Path,
        default=METRIC_DIR / "reviewer_gap_closure_status.json",
    )
    parser.add_argument(
        "--out-status-table",
        type=Path,
        default=TABLE_DIR / "table_reviewer_gap_closure_status.csv",
    )
    parser.add_argument(
        "--out-external",
        type=Path,
        default=TABLE_DIR / "table_external_zero_target_ci_compact.csv",
    )
    parser.add_argument(
        "--out-pooling",
        type=Path,
        default=TABLE_DIR / "table_pooling_cross_dataset_compact.csv",
    )
    parser.add_argument(
        "--out-morphology",
        type=Path,
        default=TABLE_DIR / "table_morphology_learnability_compact.csv",
    )
    parser.add_argument(
        "--out-robustness",
        type=Path,
        default=TABLE_DIR / "table_robustness_six_stress_compact.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "reviewer_gap_closure_manifest.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    path = resolve(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def float_value(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}: {row}")
    return value


def int_value(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def output_hash_matches(manifest: dict[str, Any], path: Path) -> bool:
    path = resolve(path)
    for item in manifest.get("outputs") or []:
        if Path(str(item.get("path", ""))).name == path.name:
            return item.get("sha256") == sha256_file(path)
    return False


def named_artifact_hash_matches(manifest: dict[str, Any], key: str, path: Path) -> bool:
    expected = (manifest.get("artifact_sha256") or {}).get(key)
    path = resolve(path)
    return bool(expected) and expected == sha256_file(path)


def runner_hash_matches(manifest: dict[str, Any], runner_name: str) -> bool:
    runner = PROJECT_ROOT / "scripts" / "revision" / runner_name
    expected = manifest.get("runner_sha256")
    return runner.exists() and bool(expected) and expected == sha256_file(runner)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def latex_escape(value: Any) -> str:
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
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(columns) - 1)
    content = [f"\\begin{{tabular}}{{{alignment}}}", r"\toprule"]
    content.append(" & ".join(latex_escape(column) for column in columns) + r" \\")
    content.append(r"\midrule")
    for row in rows:
        content.append(" & ".join(latex_escape(row.get(column, "")) for column in columns) + r" \\")
    content.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def validate_external(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table_path = resolve(args.external_table)
    rows = read_csv(table_path)
    summary = read_json(args.external_summary)
    manifest = read_json(args.external_manifest)
    expected = {
        (dataset, comparator, metric)
        for dataset in DATASETS
        for comparator in EXTERNAL_COMPARATORS
        for metric in METRICS
    }
    observed = {(row["dataset"], row["comparator"], row["metric"]) for row in rows}
    issues = []
    if observed != expected or len(rows) != len(expected):
        issues.append(f"expected {len(expected)} unique rows, observed {len(rows)}")
    if manifest.get("status") != "complete" or manifest.get("failures"):
        issues.append("external paired manifest is incomplete")
    if not output_hash_matches(manifest, table_path):
        issues.append("external paired table SHA is not authenticated by its manifest")
    if not runner_hash_matches(manifest, "32_paired_external_comparators.py"):
        issues.append("external paired runner SHA does not match the current implementation")
    for row in rows:
        if int_value(row, "n_boot_valid") < 1000:
            issues.append("external paired CI has fewer than 1000 valid bootstrap replicates")
            break
        if int_value(row, "n_groups") < 2 or not row.get("group_unit"):
            issues.append("external paired CI lacks patient/source group bootstrap provenance")
            break
        for key in (
            "full_value",
            "comparator_value",
            "improvement_full_over_comparator",
            "improvement_ci_low",
            "improvement_ci_high",
            "holm_p_value_two_sided",
        ):
            float_value(row, key)
    if summary.get("status") not in {True, "complete"}:
        issues.append("external paired summary status is not true")

    compact = []
    for row in rows:
        if row["metric"] not in {"pr_auc_macro", "f1_macro"}:
            continue
        compact.append(
            {
                "Dataset": row["dataset"],
                "Comparator": row["comparator_label"],
                "Metric": row["metric"].replace("_macro", ""),
                "Full": fmt(row["full_value"]),
                "Comparator value": fmt(row["comparator_value"]),
                "Delta Full-comparator": fmt(row["improvement_full_over_comparator"]),
                "95% CI": f"[{fmt(row['improvement_ci_low'])}, {fmt(row['improvement_ci_high'])}]",
                "Holm p": fmt(row["holm_p_value_two_sided"]),
                "Groups": int_value(row, "n_groups"),
                "Group unit": row["group_unit"],
                "Interpretation": row["interpretation"],
            }
        )
    return {
        "reviewer_item": "R1-C5",
        "status": "complete" if not issues else "incomplete",
        "manuscript_ready": not issues,
        "issues": issues,
        "evidence": [str(resolve(args.external_table)), str(resolve(args.external_manifest))],
        "safe_wording": (
            "Report zero-target-label paired effects separately by dataset, comparator, and metric with group-bootstrap CIs; do not pool tasks."
        ),
    }, compact


def validate_pooling(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv(args.pooling_table)
    bootstrap = read_json(args.pooling_bootstrap)
    manifest = read_json(args.pooling_manifest)
    expected = {(dataset, method) for dataset in DATASETS for method in POOLING_METHODS}
    observed = {(row["dataset"], row["pooling"]) for row in rows}
    issues = []
    if observed != expected or len(rows) != len(expected):
        issues.append(f"expected {len(expected)} dataset/method rows, observed {len(rows)}")
    if manifest.get("status") is not True:
        issues.append("external pooling manifest is incomplete")
    if manifest.get("protocol") != "external_pooling_sensitivity_v2_group_bootstrap":
        issues.append("unexpected external pooling protocol")
    if not manifest.get("strict_group_bootstrap"):
        issues.append("strict group bootstrap was not enabled")
    if sorted(manifest.get("datasets") or []) != sorted(DATASETS):
        issues.append("external pooling manifest does not cover all three named datasets")
    if int(manifest.get("n_boot", 0)) < 1000 or int(bootstrap.get("n_boot", 0)) < 1000:
        issues.append("external pooling sensitivity has fewer than 1000 bootstrap replicates")
    if not output_hash_matches(manifest, resolve(args.pooling_table)):
        issues.append("external pooling table SHA is not authenticated by its manifest")
    if not output_hash_matches(manifest, resolve(args.pooling_bootstrap)):
        issues.append("external pooling bootstrap SHA is not authenticated by its manifest")
    if not runner_hash_matches(manifest, "30_pooling_sensitivity_external.py"):
        issues.append("external pooling runner SHA does not match the current implementation")
    bootstrap_items = bootstrap.get("items") or {}
    expected_bootstrap_keys = {
        f"{dataset}__q3_vs_{method}__{metric}"
        for dataset in DATASETS
        for method in POOLING_METHODS
        if method != "power_mean_q3"
        for metric in METRICS
    }
    if set(bootstrap_items) != expected_bootstrap_keys:
        issues.append("external pooling paired-bootstrap grid is incomplete")
    for key, item in bootstrap_items.items():
        if int(item.get("n_boot_valid", 0)) < 1000:
            issues.append(f"external pooling bootstrap is short for {key}")
            break
        if not bool_value(item.get("group_safe")) or int(item.get("n_groups", 0)) < 2:
            issues.append(f"external pooling bootstrap lacks group provenance for {key}")
            break
        for field in ("point_delta_a_minus_b", "lo", "hi"):
            float_value(item, field)
    for row in rows:
        if not bool_value(row.get("group_safe")) or int_value(row, "n_groups") < 2:
            issues.append(f"{row.get('dataset')} lacks a group-safe pooling contract")
            break

    by_key = {(row["dataset"], row["pooling"]): row for row in rows}
    compact = []
    for dataset in DATASETS:
        q3 = by_key.get((dataset, "power_mean_q3"), {})
        mean = by_key.get((dataset, "mean"), {})
        for metric in ("pr_auc_macro", "roc_auc_macro", "f1_macro"):
            item = bootstrap_items.get(f"{dataset}__q3_vs_mean__{metric}") or {}
            values = [float_value(by_key[(dataset, method)], metric) for method in POOLING_METHODS if (dataset, method) in by_key]
            compact.append(
                {
                    "Dataset": dataset,
                    "Metric": metric.replace("_macro", ""),
                    "Q=3": fmt(q3[metric]) if q3 else "",
                    "Mean": fmt(mean[metric]) if mean else "",
                    "Tested min-max": f"[{fmt(min(values))}, {fmt(max(values))}]" if values else "",
                    "Delta Q3-mean": fmt(item.get("point_delta_a_minus_b", math.nan)),
                    "95% group CI": f"[{fmt(item.get('lo', math.nan))}, {fmt(item.get('hi', math.nan))}]",
                    "Groups": int_value(q3, "n_groups") if q3 else 0,
                    "Group unit": q3.get("group_unit", ""),
                }
            )
    return {
        "reviewer_item": "R1-C6",
        "status": "complete" if not issues else "incomplete",
        "manuscript_ready": not issues,
        "issues": issues,
        "evidence": [str(resolve(args.pooling_table)), str(resolve(args.pooling_bootstrap))],
        "safe_wording": "Q=3 is a frozen operating point evaluated across named datasets; it is not universally optimal.",
    }, compact


def validate_morphology(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues = []
    compact: list[dict[str, Any]] = []
    try:
        rows = read_csv(args.morphology_table)
        payload = read_json(args.morphology_json)
        manifest = read_json(args.morphology_manifest)
        expected = {
            (comparison, metric)
            for comparison in ("partial_vs_frozen", "full_vs_partial")
            for metric in METRICS
        }
        observed = {(row["comparison"], row["metric"]) for row in rows}
        if observed != expected or len(rows) != len(expected):
            issues.append("controlled morphology paired table does not contain two complete five-metric comparisons")
        if payload.get("status") is not True or manifest.get("status") != "complete":
            issues.append("controlled morphology paired package is incomplete")
        if not output_hash_matches(manifest, resolve(args.morphology_table)):
            issues.append("controlled morphology table SHA is not authenticated by its manifest")
        if not output_hash_matches(manifest, resolve(args.morphology_json)):
            issues.append("controlled morphology JSON SHA is not authenticated by its manifest")
        if not runner_hash_matches(manifest, "40_paired_morphology_learnability.py"):
            issues.append("controlled morphology runner SHA does not match the current implementation")
        for row in rows:
            if int_value(row, "n_boot_valid") < 1000:
                issues.append("controlled morphology comparison has fewer than 1000 bootstrap replicates")
                break
            compact.append(
                {
                    "Comparison": row["comparison"],
                    "Metric": row["metric"].replace("_macro", ""),
                    "First": fmt(row["first_value"]),
                    "Second": fmt(row["second_value"]),
                    "Oriented delta": fmt(row["improvement_first_over_second"]),
                    "95% CI": f"[{fmt(row['improvement_ci_low'])}, {fmt(row['improvement_ci_high'])}]",
                    "Holm p": fmt(row["holm_p_value_two_sided"]),
                    "Interpretation": row["interpretation"],
                }
            )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        issues.append(str(exc))
    return {
        "reviewer_item": "R1-C2",
        "status": "complete" if not issues else "incomplete",
        "manuscript_ready": not issues,
        "issues": issues,
        "evidence": [str(resolve(args.morphology_table)), str(resolve(args.morphology_manifest))],
        "safe_wording": "Use only endpoint-specific results from the reduced identically initialized frozen-versus-partially-learnable kernel control.",
    }, compact


def validate_robustness(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues = []
    compact: list[dict[str, Any]] = []
    try:
        rows = read_csv(args.robustness_table)
        pairwise = read_json(args.robustness_pairwise)
        manifest = read_json(args.robustness_manifest)
        expected = {
            (stress, comparator, metric)
            for stress in STRESSES
            for comparator in ROBUSTNESS_COMPARATORS
            for metric in METRICS
        }
        observed = {(row["stress"], row["comparator"], row["metric"]) for row in rows}
        if observed != expected or len(rows) != len(expected):
            issues.append(f"expected {len(expected)} canonical robustness rows, observed {len(rows)}")
        if pairwise.get("status") not in {True, "complete"} or manifest.get("status") != "complete":
            issues.append("canonical robustness pairwise package is incomplete")
        if int(pairwise.get("n_boot", 0)) < 1000 or int(manifest.get("n_boot", 0)) < 1000:
            issues.append("canonical robustness package has fewer than 1000 bootstrap replicates")
        if pairwise.get("output_profile") != "canonical" or manifest.get("output_profile") != "canonical":
            issues.append("robustness package is a screening/custom profile, not canonical")
        if not named_artifact_hash_matches(manifest, "table", resolve(args.robustness_table)):
            issues.append("canonical robustness table SHA is not authenticated by its manifest")
        if not named_artifact_hash_matches(manifest, "pairwise", resolve(args.robustness_pairwise)):
            issues.append("canonical robustness pairwise SHA is not authenticated by its manifest")
        if not runner_hash_matches(manifest, "21_robustness_multicomparator.py"):
            issues.append("canonical robustness runner SHA does not match the current implementation")
        for row in rows:
            if int_value(row, "n_boot_valid") < 1000 or row.get("status") != "complete":
                issues.append("canonical robustness table contains incomplete/bootstrap-short rows")
                break

        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["stress"], row["comparator"])].append(row)
        for stress in STRESSES:
            for comparator in ROBUSTNESS_COMPARATORS:
                group = grouped.get((stress, comparator), [])
                counts = Counter(row.get("interpretation", "") for row in group)
                f1 = next((row for row in group if row["metric"] == "f1_macro"), None)
                pr = next((row for row in group if row["metric"] == "pr_auc_macro"), None)
                if not f1 or not pr:
                    continue
                compact.append(
                    {
                        "Stress": stress,
                        "Comparator": f1["comparator_label"],
                        "F1 degradation advantage": fmt(f1["degradation_advantage_full"]),
                        "F1 95% CI": f"[{fmt(f1['degradation_adv_ci_low'])}, {fmt(f1['degradation_adv_ci_high'])}]",
                        "PR-AUC degradation advantage": fmt(pr["degradation_advantage_full"]),
                        "PR-AUC 95% CI": f"[{fmt(pr['degradation_adv_ci_low'])}, {fmt(pr['degradation_adv_ci_high'])}]",
                        "Full less degraded": counts["full_significantly_less_degraded"],
                        "Comparator less degraded": counts["comparator_significantly_less_degraded"],
                        "Inconclusive": counts["no_significant_degradation_difference"],
                    }
                )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        issues.append(str(exc))
    return {
        "reviewer_item": "R2-C3",
        "status": "complete" if not issues else "incomplete",
        "manuscript_ready": not issues,
        "issues": issues,
        "evidence": [str(resolve(args.robustness_table)), str(resolve(args.robustness_manifest))],
        "safe_wording": "Report robustness only by named stress, metric, and comparator; do not claim general robustness superiority.",
    }, compact


def write_compact(path: Path, rows: list[dict[str, Any]]) -> list[Path]:
    path = resolve(path)
    save_csv(path, rows)
    tex_path = path.with_suffix(".tex")
    if rows:
        write_latex(tex_path, rows, list(rows[0]))
    else:
        tex_path.write_text("% No complete reviewer-ready rows.\n", encoding="utf-8")
    return [path, tex_path]


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    print("=" * 80)
    print("REVIEWER GAP CLOSURE GATE")
    print("=" * 80)
    validators = [
        ("R1-C2", validate_morphology, args.out_morphology),
        ("R1-C5", validate_external, args.out_external),
        ("R1-C6", validate_pooling, args.out_pooling),
        ("R2-C3", validate_robustness, args.out_robustness),
    ]
    status_rows = []
    output_paths: list[Path] = []
    for reviewer_item, validator, output in validators:
        try:
            status, compact = validator(args)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            status = {
                "reviewer_item": reviewer_item,
                "status": "incomplete",
                "manuscript_ready": False,
                "issues": [str(exc)],
                "evidence": [],
                "safe_wording": "Do not claim this reviewer item complete until its authenticated closure package passes.",
            }
            compact = []
        status_rows.append(status)
        output_paths.extend(write_compact(output, compact))
        print(json.dumps(status, indent=2), flush=True)

    status_table = resolve(args.out_status_table)
    save_csv(
        status_table,
        [
            {
                **{key: value for key, value in row.items() if key not in {"issues", "evidence"}},
                "issues": "; ".join(row["issues"]),
                "evidence": "; ".join(row["evidence"]),
            }
            for row in status_rows
        ],
    )
    overall = all(row["manuscript_ready"] for row in status_rows)
    status_path = resolve(args.out_status)
    save_json(
        status_path,
        {
            "status": overall,
            "created_utc": now_utc(),
            "rows": status_rows,
            "claim_guidance": {
                "allowed": "Only use reviewer-item wording whose manuscript_ready field is true.",
                "not_allowed": "Do not promote incomplete or screening evidence to a final claim.",
            },
        },
    )
    output_paths.extend([status_path, status_table])
    manifest_path = resolve(args.out_manifest)
    save_json(
        manifest_path,
        {
            "status": "complete" if overall else "incomplete",
            "created_utc": now_utc(),
            "git_commit": git_commit(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "reviewer_items": [row["reviewer_item"] for row in status_rows],
            "artifacts": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in output_paths
            ],
        },
    )
    print(json.dumps({"status": overall, "status_file": str(status_path)}, indent=2))
    if args.strict and not overall:
        incomplete = [row["reviewer_item"] for row in status_rows if not row["manuscript_ready"]]
        raise RuntimeError("Reviewer gap closure incomplete: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
