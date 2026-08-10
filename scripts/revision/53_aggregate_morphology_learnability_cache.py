"""Aggregate authenticated morphology fold caches without running model training.

The original A100 runner writes one prediction cache and one checkpoint per
fold and variant. This CPU-only finalizer validates those producer artifacts
against the frozen OOF contract, then creates the OOF summary consumed by the
paired analysis. It never materializes the multi-GB signal array.
"""

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


def load_control():
    path = PROJECT_ROOT / "scripts" / "revision" / "39_morphology_learnability_control.py"
    spec = importlib.util.spec_from_file_location("_morphology_control_producer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load morphology control producer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


control = load_control()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-predictions", type=Path, default=control.PREDICTION_DIR / "oof_final_ema_predictions.npz")
    parser.add_argument("--freeze-manifest", type=Path, default=control.MANIFEST_DIR / "oof_final_ema_freeze_manifest.json")
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--raw-cache", type=Path, default=None)
    parser.add_argument("--num-kernels", type=int, default=256)
    parser.add_argument("--trainable-fraction", type=float, default=0.25)
    parser.add_argument("--dilations", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--kernel-length", type=int, default=9)
    parser.add_argument("--ppv-temperature", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=control.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--fold-cache-dir", type=Path, default=control.DEFAULT_FOLD_CACHE_DIR)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def raw_metadata(*, expected_y: np.ndarray, fingerprint: str, explicit: Path | None) -> dict[str, Any]:
    """Validate raw identity without decompressing the ECG signal tensor."""

    checked: list[str] = []
    for path in control.raw_helpers.candidate_raw_cache_paths(explicit):
        checked.append(str(path))
        if not path.is_file() or not path.stat().st_size:
            continue
        with np.load(path, allow_pickle=False) as payload:
            if {"X", "y", "subjects"} - set(payload.files):
                continue
            labels = np.asarray(payload["y"], dtype=np.float32)
            subjects = np.asarray(payload["subjects"]).astype(str)
            stored = str(control.raw_helpers.npz_scalar(payload, "record_order_fingerprint", "") or "")
        if len(labels) < len(expected_y) or len(subjects) < len(expected_y):
            continue
        observed = control.raw_helpers.record_order_fingerprint(subjects[: len(expected_y)])
        if (stored and stored != observed) or (fingerprint and observed != fingerprint):
            continue
        if not np.array_equal(labels[: len(expected_y)], expected_y):
            continue
        return {
            "raw_cache": str(path.resolve()),
            "raw_cache_sha256": control.sha256_file(path),
            "raw_cache_kind": "record_fingerprinted_metadata_only",
            "raw_cache_record_order_fingerprint": observed,
            "raw_cache_stored_record_order_fingerprint": stored,
        }
    raise FileNotFoundError("No source-bound raw cache matches the frozen OOF. Checked:\n- " + "\n- ".join(checked))


def fold_result(variant, fold, y, va_idx, probability, checkpoint, payload, threshold):
    metrics = control.multilabel_metrics(y[va_idx], probability, threshold=threshold)
    return {
        "variant": variant,
        "fold": fold,
        "train_records": len(y) - len(va_idx),
        "validation_records": len(va_idx),
        "reused_fold_cache": True,
        "reused_checkpoint": True,
        "checkpoint_sha256": control.sha256_file(checkpoint),
        "initialization_sha256": payload.get("initialization_sha256"),
        "trainable_kernel_count": payload.get("trainable_kernel_count"),
        **{f"final_{key}": value for key, value in metrics.items()},
    }


def main() -> None:
    args = parse_args()
    control.ensure_revision_dirs()
    if args.num_kernels % len(control.parse_csv_ints(args.dilations)):
        raise ValueError("--num-kernels must be divisible by the number of dilations.")
    print("=" * 80, flush=True)
    print("MORPHOLOGY LEARNABILITY AUTHENTICATED CACHE AGGREGATION", flush=True)
    print("=" * 80, flush=True)
    freeze = control.raw_helpers.validate_oof_freeze_contract(
        freeze_manifest=args.freeze_manifest,
        oof_predictions=args.oof_predictions,
        expected_checkpoint_kind=args.expected_checkpoint_kind,
    )
    y, fold_id, record_id, class_names, folds, oof_info = control.raw_helpers.load_oof_labels_and_folds(
        args.oof_predictions, limit_records=0
    )
    fingerprint = oof_info.get("dataset_record_order_fingerprint") or freeze.get("dataset_record_order_fingerprint") or ""
    raw_info = raw_metadata(expected_y=y, fingerprint=fingerprint, explicit=args.raw_cache)
    load_info = {**oof_info, **raw_info, "freeze_contract": freeze}
    params = control.model_params(args)
    params_hash = control.stable_json_hash(params)
    if {int(split["fold"]) for split in folds} != {1, 2, 3, 4, 5}:
        raise RuntimeError("Frozen OOF must contain exactly folds 1..5.")

    probabilities: dict[str, np.ndarray] = {}
    fold_rows: list[dict[str, Any]] = []
    initialization: dict[int, set[str]] = {}
    for variant in control.VARIANTS:
        contract = control.input_contract(load_info, params_hash, variant, control.variant_fraction(args, variant))
        probability_all = np.full(y.shape, np.nan, dtype=np.float32)
        for split in folds:
            fold = int(split["fold"])
            va_idx = np.asarray(split["va_idx"], dtype=np.int64)
            checkpoint = control.checkpoint_path(args, variant, fold)
            cached = control.load_fold_cache(
                path=control.fold_cache_path(args, variant, fold), checkpoint=checkpoint,
                fold=fold, va_idx=va_idx, contract=contract,
            )
            payload = control.checkpoint_is_compatible(checkpoint, fold=fold, contract=contract)
            if cached is None or payload is None:
                raise RuntimeError(f"{variant} fold {fold} is not an authenticated reusable cache/checkpoint.")
            init_sha = str(payload.get("initialization_sha256") or "")
            if len(init_sha) != 64:
                raise RuntimeError(f"{variant} fold {fold} has no initialization SHA.")
            initialization.setdefault(fold, set()).add(init_sha)
            probability_all[va_idx] = cached
            fold_rows.append(fold_result(variant, fold, y, va_idx, cached, checkpoint, payload, args.threshold))
            print(f"{variant} fold {fold}: authenticated cache reused", flush=True)
        if not np.all(np.isfinite(probability_all)):
            raise RuntimeError(f"{variant} cache aggregation has missing predictions.")
        probabilities[variant] = probability_all
    if sorted(initialization) != [1, 2, 3, 4, 5] or any(len(values) != 1 for values in initialization.values()):
        raise RuntimeError("Frozen/partial initializations are not matched by fold.")

    variants: dict[str, Any] = {}
    class_rows: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    for variant, probability in probabilities.items():
        checkpoints = control.checkpoint_contract(args, variant)
        if checkpoints["status"] != "complete":
            raise RuntimeError(f"Incomplete checkpoint contract for {variant}.")
        prediction = control.write_prediction_artifact(
            variant=variant, y=y, y_prob=probability, fold_id=fold_id, record_id=record_id,
            class_names=class_names, args=args, load_info=load_info, params=params, checkpoints=checkpoints,
        )
        variants[variant] = {
            "trainable_fraction": control.variant_fraction(args, variant),
            "metrics": control.multilabel_metrics(y, probability, threshold=args.threshold),
            "calibration": control.calibration_summary(y, probability, n_bins=args.n_bins),
            "prediction_path": str(prediction),
            "prediction_sha256": control.sha256_file(prediction),
            "checkpoint_contract": checkpoints,
        }
        class_rows.extend(control.class_rows(variant, y, probability, class_names, args.threshold))
        artifacts.append(prediction)

    model_table = control.TABLE_DIR / "table_morphology_learnability_model_metrics.csv"
    fold_table = control.TABLE_DIR / "table_morphology_learnability_fold_summary.csv"
    class_table = control.TABLE_DIR / "table_morphology_learnability_class_metrics.csv"
    status_path = control.METRIC_DIR / "morphology_learnability_fold_cache_status.json"
    status_table = control.TABLE_DIR / "table_morphology_learnability_fold_cache_status.csv"
    summary_path = control.METRIC_DIR / "morphology_learnability_summary.json"
    manifest_path = control.MANIFEST_DIR / "morphology_learnability_manifest.json"
    control.save_csv(model_table, [{"variant": name, "trainable_fraction": data["trainable_fraction"], **data["metrics"], **data["calibration"]} for name, data in variants.items()])
    control.save_csv(fold_table, fold_rows)
    control.save_csv(class_table, class_rows)
    status_rows = [{"variant": variant, "fold": fold, "cache_exists": True, "checkpoint_exists": True, "cache_path": str(control.fold_cache_path(args, variant, fold)), "checkpoint_path": str(control.checkpoint_path(args, variant, fold))} for variant in control.VARIANTS for fold in range(1, 6)]
    control.save_csv(status_table, status_rows)
    producer_sha = control.sha256_file(Path(control.__file__).resolve())
    control.save_json(status_path, {"status": "complete", "created_utc": now_utc(), "aggregation_mode": "authenticated_cache_only", "producer_runner_sha256": producer_sha, "rows": status_rows})
    summary = {
        "status": True, "created_utc": now_utc(), "protocol": control.PROTOCOL,
        "feature_contract": control.FEATURE_CONTRACT, "model_params": params,
        "model_params_sha256": params_hash,
        "matched_initialization_sha256_by_fold": {str(fold): next(iter(values)) for fold, values in sorted(initialization.items())},
        "canonical_contract": control.input_contract(load_info, params_hash, "variant_specific", -1.0),
        "cache_aggregation": {"mode": "authenticated_cache_only", "producer_runner_sha256": producer_sha, "raw_tensor_materialized": False},
        "variants": variants,
        "claim_guidance": {"allowed": "Use as a reduced-bank controlled sensitivity comparison between frozen and partially learnable seeded kernels.", "not_allowed": "Do not call this the evaluated 10,000-kernel ECG-RAMBA branch or infer a causal mechanism for the full model."},
        "outputs": {"model_table": str(model_table), "fold_table": str(fold_table), "class_table": str(class_table)},
    }
    control.save_json(summary_path, control.json_safe(summary))
    artifacts.extend([summary_path, model_table, fold_table, class_table, status_path, status_table])
    manifest = {
        "status": "complete", "created_utc": now_utc(), "git_commit": control.git_output("rev-parse", "HEAD"),
        "git_status_short": control.git_output("status", "--short"),
        "runner_sha256": control.sha256_file(Path(__file__).resolve()), "producer_runner_sha256": producer_sha,
        "protocol": control.PROTOCOL,
        "inputs": {"oof_predictions": {"path": load_info["oof_predictions"], "sha256": load_info["oof_predictions_sha256"]}, "freeze_manifest": {"path": str(control.resolve(args.freeze_manifest)), "sha256": freeze["freeze_manifest_sha256"]}, "raw_cache": {"path": raw_info["raw_cache"], "sha256": raw_info["raw_cache_sha256"]}},
        "artifacts": [{"path": str(path), "sha256": control.sha256_file(path), "size_bytes": path.stat().st_size} for path in artifacts],
        "checkpoint_contracts": {variant: variants[variant]["checkpoint_contract"] for variant in control.VARIANTS},
    }
    control.save_json(manifest_path, control.json_safe(manifest))
    print(json.dumps({"status": True, "mode": "authenticated_cache_only", "manifest": str(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
