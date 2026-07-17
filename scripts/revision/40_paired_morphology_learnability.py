"""Paired bootstrap for the controlled morphology learnability experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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


EXPECTED_PROTOCOL = "morphology_learnability_control_v1_same_folds_power_mean_q3_threshold_0.5"
EXPECTED_FEATURE_CONTRACT = "reduced_seeded_random_convolution_max_soft_ppv_control"


def load_helpers():
    path = PROJECT_ROOT / "scripts" / "revision" / "15_paired_full_vs_resnet.py"
    spec = importlib.util.spec_from_file_location("_morphology_learnability_paired_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load paired bootstrap helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = load_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-predictions", type=Path, default=PREDICTION_DIR / "oof_final_ema_predictions.npz")
    parser.add_argument(
        "--frozen-predictions",
        type=Path,
        default=PREDICTION_DIR / "morphology_learnability_frozen_oof_predictions.npz",
    )
    parser.add_argument(
        "--partial-predictions",
        type=Path,
        default=PREDICTION_DIR / "morphology_learnability_partial_oof_predictions.npz",
    )
    parser.add_argument("--freeze-manifest", type=Path, default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json")
    parser.add_argument("--experiment-summary", type=Path, default=METRIC_DIR / "morphology_learnability_summary.json")
    parser.add_argument("--experiment-manifest", type=Path, default=MANIFEST_DIR / "morphology_learnability_manifest.json")
    parser.add_argument("--out-json", type=Path, default=METRIC_DIR / "paired_morphology_learnability_comparison.json")
    parser.add_argument("--out-table", type=Path, default=TABLE_DIR / "table_paired_morphology_learnability.csv")
    parser.add_argument(
        "--out-bootstrap-samples",
        type=Path,
        default=METRIC_DIR / "paired_morphology_learnability_bootstrap_samples.csv",
    )
    parser.add_argument("--out-manifest", type=Path, default=MANIFEST_DIR / "paired_morphology_learnability_manifest.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-records", type=int, default=44186)
    parser.add_argument("--expected-classes", type=int, default=27)
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def npz_scalar(payload: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in payload.files:
        return default
    value = payload[key]
    return value.item() if np.ndim(value) == 0 else value


def validate_control_prediction(path: Path, *, variant: str, canonical) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "y_true",
            "y_prob",
            "fold_id",
            "record_id",
            "class_names",
            "protocol",
            "feature_contract",
            "variant",
            "oof_predictions_sha256",
            "freeze_manifest_sha256",
            "checkpoint_folds",
            "checkpoint_sha256",
        }
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"{path} missing {sorted(missing)}")
        if npz_scalar(payload, "protocol") != EXPECTED_PROTOCOL:
            raise ValueError(f"{path}: protocol mismatch")
        if npz_scalar(payload, "feature_contract") != EXPECTED_FEATURE_CONTRACT:
            raise ValueError(f"{path}: feature contract mismatch")
        if npz_scalar(payload, "variant") != variant:
            raise ValueError(f"{path}: expected variant={variant}")
        if npz_scalar(payload, "oof_predictions_sha256") != canonical.sha256:
            raise ValueError(f"{path}: canonical OOF SHA mismatch")
        if not np.array_equal(np.asarray(payload["y_true"], dtype=np.float32), canonical.y_true):
            raise ValueError(f"{path}: y_true mismatch")
        if not np.array_equal(np.asarray(payload["fold_id"], dtype=np.int16), canonical.fold_id):
            raise ValueError(f"{path}: fold_id mismatch")
        if not np.array_equal(np.asarray(payload["record_id"]), canonical.record_id):
            raise ValueError(f"{path}: record_id mismatch")
        if np.asarray(payload["class_names"]).astype(str).tolist() != canonical.class_names:
            raise ValueError(f"{path}: class_names mismatch")
        folds = np.asarray(payload["checkpoint_folds"], dtype=np.int16).tolist()
        hashes = np.asarray(payload["checkpoint_sha256"]).astype(str).tolist()
        if folds != [1, 2, 3, 4, 5] or len(hashes) != 5 or any(len(value) != 64 for value in hashes):
            raise ValueError(f"{path}: incomplete checkpoint contract")
        probability = np.asarray(payload["y_prob"], dtype=np.float32)
        fraction = float(npz_scalar(payload, "trainable_fraction", -1.0))
        freeze_sha = str(npz_scalar(payload, "freeze_manifest_sha256", ""))
    if probability.shape != canonical.y_prob.shape or not np.all(np.isfinite(probability)):
        raise ValueError(f"{path}: invalid probability matrix")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "y_prob": np.clip(probability, 0.0, 1.0),
        "trainable_fraction": fraction,
        "freeze_manifest_sha256": freeze_sha,
        "checkpoint_sha256": hashes,
    }


def validate_experiment_package(summary_path: Path, manifest_path: Path, controls: dict[str, dict]) -> dict:
    summary_path = resolve(summary_path)
    manifest_path = resolve(manifest_path)
    if not summary_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Morphology learnability summary/manifest is missing.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary.get("status") is not True or summary.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("Morphology learnability summary is not complete/current.")
    if manifest.get("status") != "complete" or manifest.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("Morphology learnability manifest is not complete/current.")
    for variant, control in controls.items():
        expected = ((summary.get("variants") or {}).get(variant) or {}).get("prediction_sha256")
        if expected != control["sha256"]:
            raise ValueError(f"Summary prediction SHA mismatch for {variant}")
    initialization = summary.get("matched_initialization_sha256_by_fold") or {}
    if sorted(int(key) for key in initialization) != [1, 2, 3, 4, 5]:
        raise ValueError("Matched frozen/partial initialization attestation is incomplete.")
    if controls["frozen"]["trainable_fraction"] != 0.0:
        raise ValueError("Frozen control unexpectedly has trainable kernels.")
    if not 0.0 < controls["partial"]["trainable_fraction"] < 1.0:
        raise ValueError("Partial control lacks a strict partial trainable-kernel fraction.")
    return {
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "model_params": summary.get("model_params"),
        "model_params_sha256": summary.get("model_params_sha256"),
        "matched_initialization_sha256_by_fold": initialization,
    }


def safe_wording(comparison: str, metric_family: str, interpretation: str) -> str:
    if comparison == "partial_vs_frozen":
        if interpretation == "full_significantly_better":
            return (
                "In the reduced-bank controlled sensitivity experiment, the partially learnable bank is stronger "
                f"for this {metric_family} endpoint than the identically initialized frozen bank."
            )
        if interpretation == "comparator_significantly_better":
            return (
                "In the reduced-bank controlled sensitivity experiment, the frozen bank is stronger "
                f"for this {metric_family} endpoint than the partially learnable bank."
            )
        return (
            "The reduced-bank controlled sensitivity experiment does not resolve a paired difference "
            f"for this {metric_family} endpoint."
        )
    if interpretation == "full_significantly_better":
        return "Full ECG-RAMBA is stronger for this paired endpoint than the reduced partially learnable control."
    if interpretation == "comparator_significantly_better":
        return "The reduced partially learnable morphology control is stronger for this paired endpoint than Full ECG-RAMBA."
    return "The Full ECG-RAMBA versus reduced partially learnable control difference is inconclusive for this endpoint."


def compare(
    *,
    comparison: str,
    first_label: str,
    second_label: str,
    y_true: np.ndarray,
    first_prob: np.ndarray,
    second_prob: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    rows = []
    samples = []
    for metric_index, spec in enumerate(helpers.metric_specs(args.threshold, args.n_bins)):
        first_value = float(spec.fn(y_true, first_prob))
        second_value = float(spec.fn(y_true, second_prob))
        raw_difference = first_value - second_value
        improvement = raw_difference if spec.higher_is_better else -raw_difference
        print(f"{comparison}: paired bootstrap {spec.name} start", flush=True)
        ci, metric_samples = helpers.paired_bootstrap_difference(
            y_true=y_true,
            full_prob=first_prob,
            comparator_prob=second_prob,
            spec=spec,
            n_boot=args.n_boot,
            seed=args.seed + metric_index * 1009,
        )
        print(f"{comparison}: paired bootstrap {spec.name} done", flush=True)
        rows.append(
            {
                "comparison": comparison,
                "first_label": first_label,
                "second_label": second_label,
                "metric": spec.name,
                "metric_family": spec.family,
                "higher_is_better": bool(spec.higher_is_better),
                "first_value": first_value,
                "second_value": second_value,
                "raw_difference_first_minus_second": raw_difference,
                "improvement_first_over_second": improvement,
                "improvement_bootstrap_mean": ci["bootstrap_mean"],
                "improvement_ci_low": ci["ci_low"],
                "improvement_ci_high": ci["ci_high"],
                "p_value_two_sided": ci["p_value_two_sided"],
                "n_boot_valid": ci["n_boot_valid"],
                "interpretation": helpers.interpretation_from_ci(ci["ci_low"], ci["ci_high"]),
            }
        )
        for item in metric_samples:
            samples.append({"comparison": comparison, **item})
    helpers.add_holm_adjustment(rows)
    for row in rows:
        row["safe_wording"] = safe_wording(comparison, row["metric_family"], row["interpretation"])
    return rows, samples


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    print("=" * 80, flush=True)
    print("PAIRED MORPHOLOGY LEARNABILITY CONTROL", flush=True)
    print("=" * 80, flush=True)
    canonical = helpers.load_prediction_set(args.full_predictions, "Full ECG-RAMBA")
    freeze_info = helpers.validate_freeze_manifest(
        args.freeze_manifest, canonical, args.expected_checkpoint_kind
    )
    if canonical.y_true.shape != (args.expected_records, args.expected_classes):
        raise ValueError(f"Unexpected canonical shape: {canonical.y_true.shape}")
    controls = {
        variant: validate_control_prediction(path, variant=variant, canonical=canonical)
        for variant, path in {
            "frozen": args.frozen_predictions,
            "partial": args.partial_predictions,
        }.items()
    }
    for variant, control in controls.items():
        if control["freeze_manifest_sha256"] != freeze_info["sha256"]:
            raise ValueError(f"{variant}: freeze manifest SHA mismatch")
    package = validate_experiment_package(args.experiment_summary, args.experiment_manifest, controls)

    rows_primary, samples_primary = compare(
        comparison="partial_vs_frozen",
        first_label="Partially learnable random-convolution bank",
        second_label="Frozen random-convolution bank",
        y_true=canonical.y_true,
        first_prob=controls["partial"]["y_prob"],
        second_prob=controls["frozen"]["y_prob"],
        args=args,
    )
    rows_secondary, samples_secondary = compare(
        comparison="full_vs_partial",
        first_label="Full ECG-RAMBA",
        second_label="Partially learnable random-convolution bank",
        y_true=canonical.y_true,
        first_prob=canonical.y_prob,
        second_prob=controls["partial"]["y_prob"],
        args=args,
    )
    rows = rows_primary + rows_secondary
    samples = samples_primary + samples_secondary
    out_json = resolve(args.out_json)
    out_table = resolve(args.out_table)
    out_samples = resolve(args.out_bootstrap_samples)
    out_manifest = resolve(args.out_manifest)
    helpers.save_csv(out_table, rows)
    helpers.save_csv(out_samples, samples)
    payload = {
        "status": True,
        "created_utc": now_utc(),
        "comparison_scope": "controlled_reduced_bank_mechanism_sensitivity",
        "protocol": EXPECTED_PROTOCOL,
        "paired_bootstrap": {
            "sample_unit": "Chapman record/subject",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "alpha": 0.05,
            "holm_adjustment": "within each five-metric comparison",
        },
        "inputs": {
            "full_predictions": {"path": str(canonical.path), "sha256": canonical.sha256},
            "freeze_manifest": freeze_info,
            "frozen_predictions": {"path": str(controls["frozen"]["path"]), "sha256": controls["frozen"]["sha256"]},
            "partial_predictions": {"path": str(controls["partial"]["path"]), "sha256": controls["partial"]["sha256"]},
            "experiment_package": package,
        },
        "comparisons": {
            "partial_vs_frozen": {row["metric"]: row for row in rows_primary},
            "full_vs_partial": {row["metric"]: row for row in rows_secondary},
        },
        "claim_guidance": {
            "allowed": "Use only endpoint-specific paired conclusions for the reduced-bank frozen-versus-partially-learnable control.",
            "not_allowed": "Do not infer that kernel learnability explains ECG-RAMBA performance or claim causal determinism/regularization separation.",
        },
    }
    save_json(out_json, helpers.json_safe(payload))
    manifest = {
        "status": "complete",
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": EXPECTED_PROTOCOL,
        "comparison_scope": "controlled_reduced_bank_mechanism_sensitivity",
        "canonical_contract": {
            "oof_sha256": canonical.sha256,
            "freeze_sha256": freeze_info["sha256"],
        },
        "n_boot": args.n_boot,
        "bootstrap_unit": "Chapman record/subject",
        "input_sha256": {
            "full_predictions": canonical.sha256,
            "freeze_manifest": freeze_info["sha256"],
            "frozen_predictions": controls["frozen"]["sha256"],
            "partial_predictions": controls["partial"]["sha256"],
            "experiment_summary": package["summary"]["sha256"],
            "experiment_manifest": package["manifest"]["sha256"],
        },
        "artifacts": {
            "json": str(out_json),
            "table": str(out_table),
            "bootstrap_samples": str(out_samples),
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (out_json, out_table, out_samples)
        ],
        "artifact_sha256": {
            "json": sha256_file(out_json),
            "table": sha256_file(out_table),
            "bootstrap_samples": sha256_file(out_samples),
        },
    }
    save_json(out_manifest, helpers.json_safe(manifest))
    print(json.dumps({"status": True, "outputs": [str(out_json), str(out_table), str(out_manifest)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
