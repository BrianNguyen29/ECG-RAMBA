"""Paired bootstrap comparison between Full ECG-RAMBA and Transformer ECG.

This optional reviewer comparator uses the same record-level paired bootstrap
contract as the ResNet1D/CNN and Raw Mamba comparisons. It is only valid after
``24_transformer_ecg_baseline.py`` has produced a manuscript-ready frozen OOF
prediction artifact under the same folds, labels, threshold, and Q=3
aggregation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    ensure_revision_dirs,
    git_commit,
    save_json,
    sha256_file,
)


FULL_LABEL = "Full ECG-RAMBA frozen OOF"
COMPARATOR_LABEL = "Transformer ECG"
EXPECTED_TRANSFORMER_PROTOCOL = "transformer_ecg_raw_same_folds_power_mean_v2_q3_threshold_0.5"
EXPECTED_TRANSFORMER_FEATURE_CONTRACT = "raw_ecg_12lead"


def load_revision_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


paired_helpers = load_revision_module("15_paired_full_vs_resnet.py", "_transformer_paired_helpers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-predictions", type=Path, default=PREDICTION_DIR / "oof_final_ema_predictions.npz")
    parser.add_argument(
        "--comparator-predictions",
        type=Path,
        default=PREDICTION_DIR / "transformer_ecg_oof_predictions.npz",
    )
    parser.add_argument("--freeze-manifest", type=Path, default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json")
    parser.add_argument(
        "--comparator-summary",
        type=Path,
        default=METRIC_DIR / "transformer_ecg_baseline_summary.json",
    )
    parser.add_argument(
        "--comparator-manifest",
        type=Path,
        default=MANIFEST_DIR / "transformer_ecg_baseline_manifest.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "paired_full_vs_transformer_comparison.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_paired_full_vs_transformer.csv",
    )
    parser.add_argument(
        "--out-bootstrap-samples",
        type=Path,
        default=METRIC_DIR / "paired_full_vs_transformer_bootstrap_samples.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "paired_full_vs_transformer_manifest.json",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-records", type=int, default=44186)
    parser.add_argument("--expected-classes", type=int, default=27)
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--require-manuscript-ready", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    return resolve_path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def authenticated_group_contract(freeze_info: dict) -> dict:
    group = freeze_info.get("group_contract") or {}
    sidecar = group.get("sidecar") or {}
    required = {
        "status": group.get("status"),
        "bootstrap_unit": group.get("bootstrap_unit"),
        "one_record_per_group": group.get("one_record_per_group"),
        "n_records": group.get("n_records"),
        "n_groups": group.get("n_groups"),
        "group_semantics": group.get("group_semantics"),
        "group_semantics_reference": group.get("group_semantics_reference"),
        "group_sidecar": sidecar.get("path"),
        "group_sidecar_sha256": sidecar.get("sha256"),
    }
    if (
        required["status"] != "verified"
        or required["one_record_per_group"] is not True
        or int(required["n_records"] or -1) != int(required["n_groups"] or -2)
        or not required["group_sidecar_sha256"]
    ):
        raise RuntimeError("Paired Transformer output lacks a verified one-record-per-group contract")
    required["group_contract_sha256"] = canonical_json_sha256(group)
    return required


def validate_paired_interval(ci: dict, samples: list[dict], n_boot: int) -> None:
    if int(ci.get("n_boot_valid", -1)) != int(n_boot) or len(samples) != int(n_boot):
        raise RuntimeError(
            f"Paired Transformer bootstrap is incomplete: "
            f"n_boot_valid={ci.get('n_boot_valid')} samples={len(samples)} expected={n_boot}"
        )
    for key in ("bootstrap_mean", "ci_low", "ci_high", "raw_diff_ci_low", "raw_diff_ci_high"):
        try:
            value = float(ci[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Paired Transformer bootstrap lacks numeric {key}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"Paired Transformer bootstrap has non-finite {key}")
    if "significant" in json.dumps(ci, sort_keys=True).lower():
        raise RuntimeError("Paired Transformer bootstrap contains prohibited legacy significance wording")


def validate_transformer_artifacts(
    *,
    summary_path: Path,
    manifest_path: Path,
    comparator,
    require_manuscript_ready: bool,
) -> dict:
    summary_path = resolve_path(summary_path)
    manifest_path = resolve_path(manifest_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Transformer ECG summary JSON: {summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Transformer ECG manifest JSON: {manifest_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source_name, payload in [("summary", summary), ("manifest", manifest)]:
        if payload.get("protocol") != EXPECTED_TRANSFORMER_PROTOCOL:
            raise ValueError(
                f"Transformer ECG {source_name} protocol mismatch: "
                f"{payload.get('protocol')} != {EXPECTED_TRANSFORMER_PROTOCOL}"
            )
        if payload.get("feature_contract") != EXPECTED_TRANSFORMER_FEATURE_CONTRACT:
            raise ValueError(
                f"Transformer ECG {source_name} feature_contract must be "
                f"{EXPECTED_TRANSFORMER_FEATURE_CONTRACT}."
            )
    if require_manuscript_ready and summary.get("manuscript_ready") is not True:
        raise ValueError("Transformer ECG summary is not manuscript_ready=true.")
    artifact_sha = manifest.get("artifact_sha256", {})
    expected_pred_sha = artifact_sha.get("predictions")
    if expected_pred_sha and expected_pred_sha != comparator.sha256:
        raise RuntimeError(f"Transformer ECG prediction SHA mismatch: {comparator.sha256} != {expected_pred_sha}")
    if comparator.metadata.get("protocol") not in (None, EXPECTED_TRANSFORMER_PROTOCOL):
        raise ValueError(f"Transformer ECG prediction protocol mismatch: {comparator.metadata.get('protocol')}")
    if comparator.metadata.get("feature_contract") not in (None, EXPECTED_TRANSFORMER_FEATURE_CONTRACT):
        raise ValueError(
            f"Transformer ECG prediction feature contract mismatch: {comparator.metadata.get('feature_contract')}"
        )
    return {
        "summary": {
            "path": str(summary_path),
            "relative_path": project_relative(summary_path),
            "sha256": sha256_file(summary_path),
            "manuscript_ready": summary.get("manuscript_ready"),
        },
        "manifest": {
            "path": str(manifest_path),
            "relative_path": project_relative(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "protocol": summary.get("protocol"),
        "feature_contract": summary.get("feature_contract"),
    }


def safe_wording(metric_family: str, interpretation: str) -> str:
    if interpretation in {"comparator_nominal_95ci_better", "full_nominal_95ci_better"}:
        favored = "Transformer ECG" if interpretation.startswith("comparator") else "Full ECG-RAMBA"
        return (
            f"The nominal pointwise 95% interval favored {favored} for this {metric_family} "
            "endpoint under frozen Chapman OOF; no multiplicity-adjusted superiority test was performed."
        )
    return "Do not claim a paired Full-vs-Transformer difference for this metric without qualification."


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    print("=" * 80, flush=True)
    print("PAIRED FULL ECG-RAMBA VS TRANSFORMER ECG COMPARISON", flush=True)
    print("=" * 80, flush=True)
    print(f"Full predictions      : {resolve_path(args.full_predictions)}", flush=True)
    print(f"Comparator predictions: {resolve_path(args.comparator_predictions)}", flush=True)
    print(f"Freeze manifest       : {resolve_path(args.freeze_manifest)}", flush=True)
    print(f"Transformer manifest  : {resolve_path(args.comparator_manifest)}", flush=True)
    print(f"n_boot={args.n_boot} seed={args.seed} threshold={args.threshold} n_bins={args.n_bins}", flush=True)

    full = paired_helpers.load_prediction_set(args.full_predictions, FULL_LABEL)
    comparator = paired_helpers.load_prediction_set(args.comparator_predictions, COMPARATOR_LABEL)
    print(f"Loaded Full: shape={full.y_true.shape} sha256={full.sha256}", flush=True)
    print(f"Loaded Transformer ECG: shape={comparator.y_true.shape} sha256={comparator.sha256}", flush=True)

    freeze_info = paired_helpers.validate_freeze_manifest(args.freeze_manifest, full, args.expected_checkpoint_kind)
    group_contract = authenticated_group_contract(freeze_info)
    runner_sha256 = sha256_file(Path(__file__).resolve())
    paired_helper_sha256 = sha256_file(Path(paired_helpers.__file__).resolve())
    transformer_info = validate_transformer_artifacts(
        summary_path=args.comparator_summary,
        manifest_path=args.comparator_manifest,
        comparator=comparator,
        require_manuscript_ready=args.require_manuscript_ready,
    )
    paired_helpers.validate_pair(full, comparator, args.expected_records, args.expected_classes)
    print("Input contract validated: same y_true, record_id, class_names, fold_id, and frozen Full SHA.", flush=True)

    rows: list[dict] = []
    sample_rows: list[dict] = []
    for metric_index, spec in enumerate(paired_helpers.metric_specs(args.threshold, args.n_bins)):
        full_value = float(spec.fn(full.y_true, full.y_prob))
        comparator_value = float(spec.fn(comparator.y_true, comparator.y_prob))
        raw_diff = full_value - comparator_value
        observed_improvement = raw_diff if spec.higher_is_better else -raw_diff
        print(
            f"{spec.name}: full={full_value:.6f} comparator={comparator_value:.6f} "
            f"improvement_full={observed_improvement:.6f}",
            flush=True,
        )
        print(f"  paired bootstrap {spec.name} start", flush=True)
        ci, samples = paired_helpers.paired_bootstrap_difference(
            y_true=full.y_true,
            full_prob=full.y_prob,
            comparator_prob=comparator.y_prob,
            spec=spec,
            n_boot=args.n_boot,
            seed=args.seed + metric_index * 1009,
        )
        validate_paired_interval(ci, samples, args.n_boot)
        print(f"  paired bootstrap {spec.name} done: {ci}", flush=True)
        row = {
            "comparison": "full_ecg_ramba_vs_transformer_ecg",
            "metric": spec.name,
            "display_name": spec.display_name,
            "metric_family": spec.family,
            "higher_is_better": bool(spec.higher_is_better),
            "full_label": FULL_LABEL,
            "comparator_label": COMPARATOR_LABEL,
            "full_value": full_value,
            "comparator_value": comparator_value,
            "raw_difference_full_minus_comparator": raw_diff,
            "improvement_full_over_comparator": observed_improvement,
            "improvement_bootstrap_mean": ci["bootstrap_mean"],
            "improvement_ci_low": ci["ci_low"],
            "improvement_ci_high": ci["ci_high"],
            "raw_diff_ci_low": ci["raw_diff_ci_low"],
            "raw_diff_ci_high": ci["raw_diff_ci_high"],
            "n_boot_valid": ci["n_boot_valid"],
            "interpretation": paired_helpers.interpretation_from_ci(ci["ci_low"], ci["ci_high"]),
        }
        rows.append(row)
        sample_rows.extend(samples)

    paired_helpers.mark_pointwise_inference(rows)
    for row in rows:
        row["safe_wording"] = safe_wording(row["metric_family"], row["interpretation"])

    out_table = resolve_path(args.out_table)
    out_bootstrap = resolve_path(args.out_bootstrap_samples)
    out_json = resolve_path(args.out_json)
    out_manifest = resolve_path(args.out_manifest)
    paired_helpers.save_csv(out_table, rows)
    paired_helpers.save_csv(out_bootstrap, sample_rows)

    payload = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "runner_sha256": runner_sha256,
        "paired_helper_sha256": paired_helper_sha256,
        "comparison": "full_ecg_ramba_vs_transformer_ecg",
        "full_label": FULL_LABEL,
        "comparator_label": COMPARATOR_LABEL,
        "paired_bootstrap": {
            "sample_unit": group_contract["bootstrap_unit"],
            "description": (
                "One bootstrap index vector is applied to both model prediction matrices; "
                "the authenticated freeze contract verifies one Chapman record per subject/group."
            ),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "alpha": 0.05,
            "p_value": "not reported; percentile bootstrap is used only for pointwise effect-size confidence intervals",
            "group_contract": group_contract,
        },
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "shape": {
            "n_records": int(full.y_true.shape[0]),
            "n_classes": int(full.y_true.shape[1]),
        },
        "inputs": {
            "full_predictions": {
                "path": str(full.path),
                "relative_path": project_relative(full.path),
                "sha256": full.sha256,
                "metadata": full.metadata,
            },
            "comparator_predictions": {
                "path": str(comparator.path),
                "relative_path": project_relative(comparator.path),
                "sha256": comparator.sha256,
                "metadata": comparator.metadata,
            },
            "freeze_manifest": freeze_info,
            "transformer": transformer_info,
        },
        "metrics": {row["metric"]: row for row in rows},
        "outputs": {
            "table": str(out_table),
            "bootstrap_samples": str(out_bootstrap),
            "json": str(out_json),
            "manifest": str(out_manifest),
        },
        "claim_guidance": {
            "transformer_comparator": "Use only as comparator-specific evidence after the Transformer ECG baseline is run.",
            "global_superiority": "Do not claim global superiority from this optional comparator.",
        },
    }
    save_json(out_json, paired_helpers.json_safe(payload))

    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "runner_sha256": runner_sha256,
        "paired_helper_sha256": paired_helper_sha256,
        "comparison": "full_ecg_ramba_vs_transformer_ecg",
        "input_sha256": {
            "full_predictions": full.sha256,
            "comparator_predictions": comparator.sha256,
            "freeze_manifest": freeze_info["sha256"],
            "transformer_summary": transformer_info["summary"]["sha256"],
            "transformer_manifest": transformer_info["manifest"]["sha256"],
            "runner": runner_sha256,
            "paired_helper": paired_helper_sha256,
            "group_contract": group_contract["group_contract_sha256"],
            "group_sidecar": group_contract["group_sidecar_sha256"],
        },
        "artifacts": {
            "json": str(out_json),
            "table": str(out_table),
            "bootstrap_samples": str(out_bootstrap),
        },
        "artifact_sha256": {
            "json": sha256_file(out_json),
            "table": sha256_file(out_table),
            "bootstrap_samples": sha256_file(out_bootstrap),
        },
        "paired_bootstrap": payload["paired_bootstrap"],
    }
    save_json(out_manifest, paired_helpers.json_safe(manifest))

    print("Wrote:", out_json, flush=True)
    print("Wrote:", out_table, flush=True)
    print("Wrote:", out_bootstrap, flush=True)
    print("Wrote:", out_manifest, flush=True)
    print(json.dumps({"status": True, "metrics": {row["metric"]: row["interpretation"] for row in rows}}, indent=2))


if __name__ == "__main__":
    main()
