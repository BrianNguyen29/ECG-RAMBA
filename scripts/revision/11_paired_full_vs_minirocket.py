"""Paired bootstrap comparison with a fixed-seed ROCKET-family linear head.

The comparison uses record-level paired resampling: each bootstrap replicate
draws one index vector and applies it to both prediction matrices. This is the
right uncertainty estimate for model differences on the same frozen OOF cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    CHAPMAN_GROUP_REFERENCE,
    CHAPMAN_GROUP_SEMANTICS,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    calibration_summary,
    ensure_revision_dirs,
    git_commit,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_json,
    sha256_file,
)


FULL_LABEL = "Full ECG-RAMBA frozen OOF"
COMPARATOR_LABEL = "Fixed-seed ROCKET-family MAX+PPV linear head"
PAIRED_INFERENCE_SCHEMA_VERSION = 2
EXPECTED_MINIROCKET_PROTOCOL = "minirocket_raw_standardized_torch_linear_same_folds_threshold_0.5"


@dataclass(frozen=True)
class PredictionSet:
    label: str
    path: Path
    sha256: str
    y_true: np.ndarray
    y_prob: np.ndarray
    record_id: np.ndarray
    fold_id: np.ndarray | None
    class_names: list[str]
    metadata: dict


@dataclass(frozen=True)
class MetricSpec:
    name: str
    display_name: str
    family: str
    higher_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-predictions", type=Path, default=PREDICTION_DIR / "oof_final_ema_predictions.npz")
    parser.add_argument(
        "--comparator-predictions",
        type=Path,
        default=PREDICTION_DIR / "minirocket_only_oof_predictions.npz",
    )
    parser.add_argument("--freeze-manifest", type=Path, default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json")
    parser.add_argument(
        "--comparator-summary",
        type=Path,
        default=METRIC_DIR / "minirocket_only_baseline_summary.json",
    )
    parser.add_argument(
        "--comparator-manifest",
        type=Path,
        default=MANIFEST_DIR / "minirocket_only_baseline_manifest.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "paired_full_vs_minirocket_comparison.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_paired_full_vs_minirocket.csv",
    )
    parser.add_argument(
        "--out-bootstrap-samples",
        type=Path,
        default=METRIC_DIR / "paired_full_vs_minirocket_bootstrap_samples.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "paired_full_vs_minirocket_manifest.json",
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


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def scalar_from_npz(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def load_prediction_set(path: Path, label: str) -> PredictionSet:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction NPZ for {label}: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "fold_id", "class_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing required keys: {sorted(missing)}")
        y_true = np.asarray(data["y_true"], dtype=np.float32)
        y_prob = np.asarray(data["y_prob"], dtype=np.float32)
        record_id = np.asarray(data["record_id"], dtype=np.int64)
        fold_id = np.asarray(data["fold_id"], dtype=np.int16)
        class_names = np.asarray(data["class_names"]).astype(str).tolist()
        metadata = {
            key: json_safe(scalar_from_npz(data, key))
            for key in [
                "dataset",
                "protocol",
                "checkpoint_kind",
                "feature_contract",
                "feature_preprocessing",
                "threshold",
                "config_hash",
                "git_commit",
                "manuscript_ready",
                "dataset_record_order_fingerprint",
                "oof_predictions_sha256",
                "freeze_manifest_sha256",
                "minirocket_cache_sha256",
            ]
            if key in data.files
        }
    if y_true.ndim != 2 or y_prob.shape != y_true.shape:
        raise ValueError(f"{label} prediction shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")
    if len(record_id) != len(y_true):
        raise ValueError(f"{label} record_id length mismatch: {len(record_id)} vs {len(y_true)}")
    if len(fold_id) != len(y_true):
        raise ValueError(f"{label} fold_id length mismatch: {len(fold_id)} vs {len(y_true)}")
    if not np.all(np.isfinite(y_prob)):
        raise ValueError(f"{label} probabilities contain non-finite values.")
    if float(np.min(y_prob)) < -1e-6 or float(np.max(y_prob)) > 1.0 + 1e-6:
        raise ValueError(f"{label} probabilities are outside [0, 1].")
    return PredictionSet(
        label=label,
        path=path,
        sha256=sha256_file(path),
        y_true=y_true,
        y_prob=np.clip(y_prob, 0.0, 1.0).astype(np.float32),
        record_id=record_id,
        fold_id=fold_id,
        class_names=class_names,
        metadata=metadata,
    )


def validate_freeze_manifest(path: Path, full: PredictionSet, expected_checkpoint_kind: str) -> dict:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Full OOF freeze manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError(f"OOF freeze manifest status must be frozen, got {payload.get('status')}")
    if payload.get("checkpoint_kind") != expected_checkpoint_kind:
        raise ValueError(
            f"Unexpected checkpoint kind: {payload.get('checkpoint_kind')} != {expected_checkpoint_kind}"
        )
    artifacts = {row.get("path"): row for row in payload.get("artifacts", [])}
    rel = project_relative(full.path)
    if rel not in artifacts:
        raise ValueError(f"Freeze manifest does not list Full predictions: {rel}")
    expected_sha = artifacts[rel].get("sha256")
    if full.sha256 != expected_sha:
        raise RuntimeError(f"Full OOF SHA mismatch: {full.sha256} != {expected_sha}")
    group = payload.get("group_contract") or {}
    group_errors = []
    if group.get("status") != "verified":
        group_errors.append("status")
    if group.get("group_semantics") != CHAPMAN_GROUP_SEMANTICS:
        group_errors.append("group_semantics")
    if group.get("group_semantics_reference") != CHAPMAN_GROUP_REFERENCE:
        group_errors.append("group_semantics_reference")
    if group.get("bootstrap_unit") != AUTHENTICATED_RECORD_BOOTSTRAP_UNIT:
        group_errors.append("bootstrap_unit")
    if group.get("one_record_per_group") is not True:
        group_errors.append("one_record_per_group")
    if int(group.get("n_records", -1)) != len(full.y_true) or int(group.get("n_groups", -1)) != len(full.y_true):
        group_errors.append("group_counts")
    sidecar = group.get("sidecar") or {}
    sidecar_path = Path(str(sidecar.get("path", "")))
    if sidecar_path and not sidecar_path.is_absolute():
        sidecar_path = PROJECT_ROOT / sidecar_path
    if not sidecar_path.is_file():
        group_errors.append("sidecar_missing")
    elif sha256_file(sidecar_path) != sidecar.get("sha256"):
        group_errors.append("sidecar_sha256")
    if group_errors:
        raise RuntimeError(
            "Paired OOF comparison lacks an authenticated patient/group bootstrap contract: "
            + ", ".join(group_errors)
        )
    return {
        "path": str(path),
        "relative_path": project_relative(path),
        "sha256": sha256_file(path),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "validated_records": payload.get("validated_records"),
        "n_classes": payload.get("n_classes"),
        "dataset_record_order_fingerprint": payload.get("dataset_record_order_fingerprint"),
        "group_contract": group,
    }


def validate_minirocket_artifacts(
    *,
    summary_path: Path,
    manifest_path: Path,
    comparator: PredictionSet,
    require_manuscript_ready: bool,
) -> dict:
    summary_path = resolve_path(summary_path)
    manifest_path = resolve_path(manifest_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing MiniRocket summary JSON: {summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing MiniRocket manifest JSON: {manifest_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    producer = PROJECT_ROOT / "scripts" / "revision" / "10_minirocket_only_baseline.py"
    producer_sha256 = sha256_file(producer)
    for source_name, payload in (("summary", summary), ("manifest", manifest)):
        if payload.get("runner_sha256") != producer_sha256:
            raise RuntimeError(f"MiniRocket {source_name} producer runner SHA is stale.")
    for source_name, payload in [("summary", summary), ("manifest", manifest)]:
        if payload.get("protocol") != EXPECTED_MINIROCKET_PROTOCOL:
            raise ValueError(
                f"MiniRocket {source_name} protocol mismatch: "
                f"{payload.get('protocol')} != {EXPECTED_MINIROCKET_PROTOCOL}"
            )
        if payload.get("feature_contract") != "minirocket_raw":
            raise ValueError(f"MiniRocket {source_name} feature_contract must be minirocket_raw.")
        if payload.get("feature_preprocessing") != "fold_train_standardization":
            raise ValueError(f"MiniRocket {source_name} must use fold_train_standardization.")
    if require_manuscript_ready and summary.get("manuscript_ready") is not True:
        raise ValueError("MiniRocket summary is not manuscript_ready=true.")
    artifact_sha = manifest.get("artifact_sha256", {})
    expected_pred_sha = artifact_sha.get("predictions")
    if expected_pred_sha and expected_pred_sha != comparator.sha256:
        raise RuntimeError(
            f"MiniRocket prediction SHA mismatch: {comparator.sha256} != {expected_pred_sha}"
        )
    if comparator.metadata.get("protocol") not in (None, EXPECTED_MINIROCKET_PROTOCOL):
        raise ValueError(f"MiniRocket prediction metadata protocol mismatch: {comparator.metadata.get('protocol')}")
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
        "feature_preprocessing": summary.get("feature_preprocessing"),
    }


def validate_pair(full: PredictionSet, comparator: PredictionSet, expected_records: int, expected_classes: int) -> None:
    if full.y_true.shape != comparator.y_true.shape:
        raise ValueError(f"Prediction shape mismatch: {full.y_true.shape} vs {comparator.y_true.shape}")
    if full.y_true.shape != (expected_records, expected_classes):
        raise ValueError(
            f"Unexpected prediction shape: {full.y_true.shape} != {(expected_records, expected_classes)}"
        )
    if not np.array_equal(full.y_true, comparator.y_true):
        raise ValueError("Full and MiniRocket y_true arrays differ; paired comparison is invalid.")
    if not np.array_equal(full.record_id, comparator.record_id):
        raise ValueError("Full and MiniRocket record_id arrays differ; record-level pairing is invalid.")
    if full.class_names != comparator.class_names:
        raise ValueError("Full and MiniRocket class_names differ.")
    if full.fold_id is None or comparator.fold_id is None:
        raise ValueError("Full and comparator predictions must both declare fold_id for a paired OOF comparison.")
    if not np.array_equal(full.fold_id, comparator.fold_id):
        raise ValueError("Full and MiniRocket fold_id arrays differ.")
    folds = sorted(int(x) for x in np.unique(full.fold_id) if int(x) > 0)
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected five OOF folds [1..5], got {folds}")


def metric_specs(threshold: float, n_bins: int) -> list[MetricSpec]:
    return [
        MetricSpec("pr_auc_macro", "PR-AUC macro", "ranking", True, macro_pr_auc),
        MetricSpec("roc_auc_macro", "ROC-AUC macro", "ranking", True, macro_roc_auc),
        MetricSpec(
            "f1_macro",
            "F1 macro",
            "fixed_threshold",
            True,
            lambda y, p: multilabel_metrics(y, p, threshold=threshold)["f1_macro"],
        ),
        MetricSpec(
            "brier_macro",
            "Brier macro",
            "calibration",
            False,
            lambda y, p: calibration_summary(y, p, n_bins=n_bins)["brier_macro"],
        ),
        MetricSpec(
            "ece_macro",
            "ECE macro",
            "calibration",
            False,
            lambda y, p: calibration_summary(y, p, n_bins=n_bins)["ece_macro"],
        ),
    ]


def paired_bootstrap_difference(
    *,
    y_true: np.ndarray,
    full_prob: np.ndarray,
    comparator_prob: np.ndarray,
    spec: MetricSpec,
    n_boot: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = []
    for boot_idx in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            full_value = float(spec.fn(y_true[idx], full_prob[idx]))
            comparator_value = float(spec.fn(y_true[idx], comparator_prob[idx]))
        except ValueError:
            continue
        if not np.isfinite(full_value) or not np.isfinite(comparator_value):
            continue
        raw_diff = full_value - comparator_value
        improvement = raw_diff if spec.higher_is_better else -raw_diff
        samples.append(
            {
                "metric": spec.name,
                "bootstrap_index": int(boot_idx),
                "full_value": full_value,
                "comparator_value": comparator_value,
                "raw_difference_full_minus_comparator": raw_diff,
                "improvement_full_over_comparator": improvement,
            }
        )
    if not samples:
        return {
            "bootstrap_mean": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "raw_diff_ci_low": math.nan,
            "raw_diff_ci_high": math.nan,
            "p_value_two_sided": math.nan,
            "n_boot_valid": 0,
        }, samples
    improvements = np.asarray([row["improvement_full_over_comparator"] for row in samples], dtype=np.float64)
    raw_diffs = np.asarray([row["raw_difference_full_minus_comparator"] for row in samples], dtype=np.float64)
    ci_low, ci_high = np.quantile(improvements, [0.025, 0.975])
    raw_low, raw_high = np.quantile(raw_diffs, [0.025, 0.975])
    return {
        "bootstrap_mean": float(np.mean(improvements)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "raw_diff_ci_low": float(raw_low),
        "raw_diff_ci_high": float(raw_high),
        # A percentile bootstrap is an effect-size interval, not a
        # null-centered randomization test. Do not manufacture a p-value from
        # the fraction of bootstrap effects crossing zero.
        "p_value_two_sided": math.nan,
        "n_boot_valid": int(len(improvements)),
    }, samples


def mark_pointwise_inference(rows: list[dict]) -> None:
    for row in rows:
        row["holm_p_value_two_sided"] = math.nan
        row["multiplicity_adjustment"] = "not_applicable_no_null_test"


def interpretation_from_ci(ci_low: float, ci_high: float) -> str:
    if ci_low > 0.0:
        return "full_nominal_95ci_better"
    if ci_high < 0.0:
        return "comparator_nominal_95ci_better"
    return "inconclusive"


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    print("=" * 80, flush=True)
    print("PAIRED FULL ECG-RAMBA VS FIXED-SEED ROCKET-FAMILY LINEAR HEAD", flush=True)
    print("=" * 80, flush=True)
    print(f"Full predictions      : {resolve_path(args.full_predictions)}", flush=True)
    print(f"Comparator predictions: {resolve_path(args.comparator_predictions)}", flush=True)
    print(f"Freeze manifest       : {resolve_path(args.freeze_manifest)}", flush=True)
    print(f"Comparator manifest   : {resolve_path(args.comparator_manifest)}", flush=True)
    print(f"n_boot={args.n_boot} seed={args.seed} threshold={args.threshold} n_bins={args.n_bins}", flush=True)

    full = load_prediction_set(args.full_predictions, FULL_LABEL)
    comparator = load_prediction_set(args.comparator_predictions, COMPARATOR_LABEL)
    print(f"Loaded Full: shape={full.y_true.shape} sha256={full.sha256}", flush=True)
    print(f"Loaded comparator: shape={comparator.y_true.shape} sha256={comparator.sha256}", flush=True)

    freeze_info = validate_freeze_manifest(args.freeze_manifest, full, args.expected_checkpoint_kind)
    mini_info = validate_minirocket_artifacts(
        summary_path=args.comparator_summary,
        manifest_path=args.comparator_manifest,
        comparator=comparator,
        require_manuscript_ready=args.require_manuscript_ready,
    )
    validate_pair(full, comparator, args.expected_records, args.expected_classes)
    print("Input contract validated: same y_true, record_id, class_names, fold_id, and frozen Full SHA.", flush=True)

    rows: list[dict] = []
    sample_rows: list[dict] = []
    for metric_index, spec in enumerate(metric_specs(args.threshold, args.n_bins)):
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
        ci, samples = paired_bootstrap_difference(
            y_true=full.y_true,
            full_prob=full.y_prob,
            comparator_prob=comparator.y_prob,
            spec=spec,
            n_boot=args.n_boot,
            seed=args.seed + metric_index * 1009,
        )
        print(f"  paired bootstrap {spec.name} done: {ci}", flush=True)
        row = {
            "comparison": "full_ecg_ramba_vs_minirocket_only",
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
            "p_value_two_sided": ci["p_value_two_sided"],
            "n_boot_valid": ci["n_boot_valid"],
            "interpretation": interpretation_from_ci(ci["ci_low"], ci["ci_high"]),
        }
        rows.append(row)
        sample_rows.extend(samples)

    mark_pointwise_inference(rows)
    for row in rows:
        row["inference_scope"] = "pointwise_percentile_ci_effect_size_only"
        if row["metric_family"] == "ranking" and row["interpretation"] == "comparator_nominal_95ci_better":
            row["safe_wording"] = (
                "The nominal pointwise 95% interval favored the fixed-seed ROCKET-family "
                "MAX+PPV linear head for rank-based discrimination on frozen Chapman OOF; "
                "no multiplicity-adjusted superiority test was performed."
            )
        elif row["metric_family"] in {"fixed_threshold", "calibration"} and row["interpretation"] == "full_nominal_95ci_better":
            row["safe_wording"] = (
                "The nominal pointwise 95% interval favored Full ECG-RAMBA for this "
                "fixed-threshold/calibration endpoint; no multiplicity-adjusted superiority "
                "test was performed."
            )
        else:
            row["safe_wording"] = "Do not claim a paired difference for this metric without qualification."

    args.out_table = resolve_path(args.out_table)
    args.out_bootstrap_samples = resolve_path(args.out_bootstrap_samples)
    args.out_json = resolve_path(args.out_json)
    args.out_manifest = resolve_path(args.out_manifest)
    save_csv(args.out_table, rows)
    save_csv(args.out_bootstrap_samples, sample_rows)

    payload = {
        "status": True,
        "paired_inference_schema_version": PAIRED_INFERENCE_SCHEMA_VERSION,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "comparison": "full_ecg_ramba_vs_minirocket_only",
        "full_label": FULL_LABEL,
        "comparator_label": COMPARATOR_LABEL,
        "paired_bootstrap": {
            "sample_unit": "record",
            "description": "One bootstrap index vector is applied to both model prediction matrices.",
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "alpha": 0.05,
            "p_value": "not_reported; percentile bootstrap is used only for pointwise effect-size confidence intervals",
            "null_test": "not_run",
            "multiplicity_adjustment": "not_applicable_no_null_test",
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
            "minirocket": mini_info,
        },
        "metrics": {row["metric"]: row for row in rows},
        "outputs": {
            "table": str(args.out_table),
            "bootstrap_samples": str(args.out_bootstrap_samples),
            "json": str(args.out_json),
            "manifest": str(args.out_manifest),
        },
        "claim_guidance": {
            "ranking": "Do not claim Full ECG-RAMBA is superior to the fixed-seed ROCKET-family linear head on AUROC/AUPRC when paired intervals favor the comparator.",
            "operating_point": "It is acceptable to state that Full ECG-RAMBA has a better fixed-threshold/calibrated operating point when F1/Brier/ECE paired CIs favor Full.",
        },
    }
    save_json(args.out_json, json_safe(payload))

    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "paired_inference_schema_version": PAIRED_INFERENCE_SCHEMA_VERSION,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "comparison": "full_ecg_ramba_vs_minirocket_only",
        "input_sha256": {
            "full_predictions": full.sha256,
            "comparator_predictions": comparator.sha256,
            "freeze_manifest": freeze_info["sha256"],
            "minirocket_summary": mini_info["summary"]["sha256"],
            "minirocket_manifest": mini_info["manifest"]["sha256"],
        },
        "artifacts": {
            "json": str(args.out_json),
            "table": str(args.out_table),
            "bootstrap_samples": str(args.out_bootstrap_samples),
        },
        "artifact_sha256": {
            "json": sha256_file(args.out_json),
            "table": sha256_file(args.out_table),
            "bootstrap_samples": sha256_file(args.out_bootstrap_samples),
        },
        "paired_bootstrap": payload["paired_bootstrap"],
    }
    save_json(args.out_manifest, json_safe(manifest))

    print("Wrote:", args.out_json, flush=True)
    print("Wrote:", args.out_table, flush=True)
    print("Wrote:", args.out_bootstrap_samples, flush=True)
    print("Wrote:", args.out_manifest, flush=True)
    print(json.dumps({"status": True, "metrics": {row["metric"]: row["interpretation"] for row in rows}}, indent=2))


if __name__ == "__main__":
    main()
