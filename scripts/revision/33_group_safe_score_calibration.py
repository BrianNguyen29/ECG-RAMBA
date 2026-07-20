"""Group-safe target-domain score calibration on frozen predictions.

This is a corrected successor to ``19_fewshot_adaptation.py``. It keeps every
patient/source ECG record in exactly one split and uses cluster bootstrap
intervals. ECG-RAMBA model weights remain unchanged, so outputs must be called
score calibration rather than few-shot model fine-tuning.

For PTB-XL, official fold 9 is the adaptation pool and fold 10 is the fixed
test set. Georgia/CPSC2021 use SHA256-seeded, label-independent group splits
with nested adaptation-group prefixes because no equivalent official target
split exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    calibration_summary,
    cluster_bootstrap_ci,
    hash_group_train_test_split,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    paired_cluster_bootstrap_delta,
    save_csv,
    save_json,
    save_json_atomic,
    save_npz_compressed_atomic,
    sha256_file,
)


PROTOCOL = "group_safe_score_calibration_v2_gated_external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ptbxl", "georgia", "cpsc2021"])
    parser.add_argument("--external-root", type=Path, default=EXPERIMENTAL_DIR / "external")
    parser.add_argument("--test-predictions", type=Path, default=None)
    parser.add_argument("--adaptation-predictions", type=Path, default=None)
    parser.add_argument("--gate-json", type=Path, default=None)
    parser.add_argument("--fractions", default="0,0.01,0.05,0.10")
    parser.add_argument("--primary-fraction", type=float, default=0.10)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--test-fraction", type=float, default=0.50)
    parser.add_argument("--split-candidates", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "group_safe_score_calibration_metric_cache",
    )
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--analysis-lock",
        type=Path,
        default=MANIFEST_DIR / "ptbxl_adaptation_analysis_lock.json",
    )
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--out-table", type=Path, default=None)
    parser.add_argument("--out-bootstrap", type=Path, default=None)
    parser.add_argument("--out-splits", type=Path, default=None)
    parser.add_argument("--out-coefficients", type=Path, default=None)
    parser.add_argument("--out-manifest", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def budget_role(fraction: float, primary_fraction: float) -> str:
    if fraction == 0:
        return "zero_target_label_reference"
    if np.isclose(fraction, primary_fraction, atol=1e-12):
        return "analysis_locked_primary"
    return "sensitivity"


def canonical_contract() -> dict[str, str]:
    oof = PREDICTION_DIR / "oof_final_ema_predictions.npz"
    freeze = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
    if not oof.exists() or not freeze.exists():
        raise FileNotFoundError("Canonical frozen OOF artifacts are required before score calibration.")
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or payload.get("manuscript_ready") is not True:
        raise RuntimeError("Canonical freeze manifest is not frozen/manuscript_ready.")
    oof_sha = sha256_file(oof)
    expected = next(
        (
            row.get("sha256")
            for row in payload.get("artifacts", [])
            if str(row.get("path", "")).replace("\\", "/").endswith(oof.name)
        ),
        None,
    )
    if expected != oof_sha:
        raise RuntimeError(f"Freeze OOF SHA mismatch: {expected} != {oof_sha}")
    return {"oof_sha256": oof_sha, "freeze_sha256": sha256_file(freeze)}


def parse_float_list(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def validate_ptbxl_analysis_lock(
    path: Path,
    *,
    fractions: list[float],
    primary_fraction: float,
    seeds: list[int],
    threshold: float,
    n_bins: int,
    n_boot: int,
) -> dict[str, str]:
    path = resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"PTB-XL adaptation analysis lock is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    expected = {
        "status": payload.get("status") == "locked",
        "adaptation_split": protocol.get("adaptation_split") == "official_ptbxl_fold9",
        "test_split": protocol.get("test_split") == "official_ptbxl_fold10",
        "group_unit": protocol.get("group_unit") == "patient_id",
        "fractions": protocol.get("fractions") == fractions,
        "primary_fraction": np.isclose(float(protocol.get("primary_fraction", np.nan)), primary_fraction),
        "seeds": protocol.get("seeds") == seeds,
        "threshold": np.isclose(float(protocol.get("threshold", np.nan)), threshold),
        "n_bins": int(protocol.get("n_bins", -1)) == n_bins,
        "n_boot": int(protocol.get("n_boot", -1)) == n_boot,
        "calibrator": (protocol.get("score_calibration") or {}).get("fit_split") == "fold9_only",
    }
    failed = [key for key, valid in expected.items() if not valid]
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("protocol_sha256") != protocol_sha:
        failed.append("protocol_sha256")
    if failed:
        raise RuntimeError(f"PTB-XL analysis lock mismatch: {failed}")
    return {"path": str(path), "sha256": sha256_file(path), "protocol_sha256": protocol_sha}


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = resolve(args.external_root) / args.dataset
    test = args.test_predictions or root / f"{args.dataset}_full_predictions.npz"
    adaptation = args.adaptation_predictions
    if adaptation is None and args.dataset == "ptbxl":
        adaptation = root / "ptbxl_full_fold9_predictions.npz"
    return {
        "test": resolve(test),
        "adaptation": resolve(adaptation) if adaptation is not None else Path(),
        "gate": resolve(args.gate_json or METRIC_DIR / f"external_{args.dataset}_protocol_gate.json"),
        "summary": resolve(args.out_summary or METRIC_DIR / f"group_safe_score_calibration_{args.dataset}_summary.csv"),
        "table": resolve(args.out_table or TABLE_DIR / f"table_group_safe_score_calibration_{args.dataset}.csv"),
        "bootstrap": resolve(args.out_bootstrap or METRIC_DIR / f"group_safe_score_calibration_{args.dataset}_bootstrap.json"),
        "splits": resolve(args.out_splits or MANIFEST_DIR / f"group_safe_score_calibration_{args.dataset}_splits.npz"),
        "coefficients": resolve(args.out_coefficients or TABLE_DIR / f"table_group_safe_score_calibration_{args.dataset}_coefficients.csv"),
        "manifest": resolve(args.out_manifest or MANIFEST_DIR / f"group_safe_score_calibration_{args.dataset}_manifest.json"),
    }


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def load_prediction(path: Path, dataset: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "group_id", "split_id", "class_names", "dataset"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing group-safe keys: {sorted(missing)}")
        payload = {
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "y_prob": np.asarray(data["y_prob"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]).astype(str),
            "group_id": np.asarray(data["group_id"]).astype(str),
            "split_id": np.asarray(data["split_id"]).astype(str),
            "class_names": np.asarray(data["class_names"]).astype(str),
            "dataset": str(scalar(data, "dataset", "")),
            "group_unit": str(scalar(data, "group_unit", "group")),
            "protocol": str(scalar(data, "protocol", "")),
        }
    if payload["dataset"] != dataset:
        raise ValueError(f"Dataset mismatch: {payload['dataset']} != {dataset}")
    if payload["y_true"].ndim != 2 or payload["y_true"].shape != payload["y_prob"].shape:
        raise ValueError(f"{path}: y_true/y_prob shape mismatch")
    n_records, n_classes = payload["y_true"].shape
    for key in ("record_id", "group_id", "split_id"):
        if len(payload[key]) != n_records:
            raise ValueError(f"{path}: {key} length mismatch")
        if np.any(np.char.str_len(payload[key].astype(str)) == 0):
            raise ValueError(f"{path}: {key} contains empty identifiers")
    if len(np.unique(payload["record_id"])) != n_records:
        raise ValueError(f"{path}: record_id values are not unique")
    if len(payload["class_names"]) != n_classes:
        raise ValueError(f"{path}: class_names length mismatch")
    if not np.isfinite(payload["y_true"]).all() or not np.isfinite(payload["y_prob"]).all():
        raise ValueError(f"{path}: labels or probabilities contain NaN/Inf")
    if not np.isin(payload["y_true"], [0.0, 1.0]).all():
        raise ValueError(f"{path}: labels must be binary")
    if float(payload["y_prob"].min()) < 0.0 or float(payload["y_prob"].max()) > 1.0:
        raise ValueError(f"{path}: probabilities must be in [0,1]")
    if len(np.unique(payload["group_id"])) < 2:
        raise ValueError(f"{path}: fewer than two independent groups")
    payload["path"] = path
    payload["sha256"] = sha256_file(path)
    return payload


def load_gate(path: Path, test: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("gate_schema_version", 0) < 4:
        raise RuntimeError("External gate predates group-safe schema v4")
    if gate.get("protocol_gate_passed") is not True or gate.get("manuscript_ready") is not True:
        raise RuntimeError("External protocol gate has not passed")
    expected_sha = ((gate.get("artifacts") or {}).get("prediction") or {}).get("sha256")
    if expected_sha != test["sha256"]:
        raise RuntimeError("External gate is stale for the test prediction artifact")
    return gate


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def calibrate_scores(
    y_train: np.ndarray,
    p_train: np.ndarray,
    p_test: np.ndarray,
    class_names: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression

    out = np.asarray(p_test, dtype=np.float64).copy()
    x_train = logit(p_train)
    x_test = logit(p_test)
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        labels = y_train[:, class_index].astype(int)
        if len(np.unique(labels)) < 2:
            rows.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "status": "skipped_single_label",
                    "coefficient": np.nan,
                    "intercept": np.nan,
                    "n_iter": 0,
                }
            )
            continue
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
        model.fit(x_train[:, [class_index]], labels)
        if int(model.n_iter_[0]) >= int(model.max_iter):
            raise RuntimeError(f"Platt calibrator did not converge for class {class_name}")
        coefficient = float(model.coef_[0, 0])
        if coefficient <= 0:
            rows.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "status": "skipped_nonpositive_slope",
                    "coefficient": coefficient,
                    "intercept": float(model.intercept_[0]),
                    "n_iter": int(model.n_iter_[0]),
                }
            )
            continue
        out[:, class_index] = model.predict_proba(x_test[:, [class_index]])[:, 1]
        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "status": "adapted",
                "coefficient": coefficient,
                "intercept": float(model.intercept_[0]),
                "n_iter": int(model.n_iter_[0]),
            }
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32), rows


def metric_functions(threshold: float, n_bins: int) -> dict[str, tuple[Callable, bool]]:
    return {
        "pr_auc_macro": (macro_pr_auc, True),
        "roc_auc_macro": (macro_roc_auc, True),
        "f1_macro": (lambda y, p: multilabel_metrics(y, p, threshold=threshold)["f1_macro"], True),
        "brier_macro": (lambda y, p: calibration_summary(y, p, n_bins=n_bins)["brier_macro"], False),
        "ece_macro": (lambda y, p: calibration_summary(y, p, n_bins=n_bins)["ece_macro"], False),
    }


def point_bundle(y: np.ndarray, p: np.ndarray, threshold: float, n_bins: int) -> dict[str, float]:
    return {**multilabel_metrics(y, p, threshold=threshold), **calibration_summary(y, p, n_bins=n_bins)}


def group_rows(groups: np.ndarray, selected_groups: np.ndarray) -> np.ndarray:
    return np.where(np.isin(groups, selected_groups))[0].astype(np.int64)


def nested_group_subset(pool_groups: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0:
        return np.asarray([], dtype=pool_groups.dtype)
    n = min(max(int(round(len(pool_groups) * fraction)), 1), len(pool_groups))
    return pool_groups[:n]


def metric_cache_key(
    args: argparse.Namespace,
    *,
    test_sha: str,
    adaptation_sha: str,
    split_key: str,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    metric_name: str,
    analysis_lock_sha256: str | None,
) -> str:
    payload = {
        "protocol": PROTOCOL,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "calibrator": "unweighted_monotonic_platt_logit_lbfgs_c1",
        "dataset": args.dataset,
        "test_sha256": test_sha,
        "adaptation_sha256": adaptation_sha,
        "split_key": split_key,
        "train_groups_sha256": hashlib.sha256(np.asarray(train_groups).astype(str).tobytes()).hexdigest(),
        "test_groups_sha256": hashlib.sha256(np.asarray(test_groups).astype(str).tobytes()).hexdigest(),
        "metric": metric_name,
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "analysis_lock_sha256": analysis_lock_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def exact_zero_delta(n_boot: int, n_groups: int) -> dict[str, Any]:
    return {
        "point_delta_a_minus_b": 0.0,
        "mean": 0.0,
        "lo": 0.0,
        "hi": 0.0,
        "n_boot_valid": int(n_boot),
        "n_groups": int(n_groups),
        "sample_unit": "group",
    }


def main() -> None:
    args = parse_args()
    canonical = canonical_contract()
    paths = default_paths(args)
    for key, path in paths.items():
        if key != "adaptation" or str(path):
            path.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 80, flush=True)
    print("GROUP-SAFE EXTERNAL SCORE CALIBRATION", flush=True)
    print("=" * 80, flush=True)
    print(f"dataset={args.dataset} protocol={PROTOCOL}", flush=True)

    fractions = parse_float_list(args.fractions)
    seeds = parse_int_list(args.seeds)
    if not fractions or any(value < 0 or value > 1 for value in fractions):
        raise ValueError(f"Invalid fractions: {fractions}")
    if not any(np.isclose(value, args.primary_fraction, atol=1e-12) for value in fractions):
        raise ValueError("--primary-fraction must be included in --fractions")
    if not seeds:
        raise ValueError("At least one seed is required")
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be in (0,1)")
    analysis_lock = (
        validate_ptbxl_analysis_lock(
            args.analysis_lock,
            fractions=fractions,
            primary_fraction=args.primary_fraction,
            seeds=seeds,
            threshold=args.threshold,
            n_bins=args.n_bins,
            n_boot=args.n_boot,
        )
        if args.dataset == "ptbxl"
        else None
    )

    test = load_prediction(paths["test"], args.dataset)
    gate = load_gate(paths["gate"], test)
    if args.dataset == "ptbxl":
        if not paths["adaptation"].exists():
            raise FileNotFoundError(
                f"PTB-XL group-safe calibration requires official fold-9 predictions: {paths['adaptation']}"
            )
        adaptation = load_prediction(paths["adaptation"], args.dataset)
        if set(test["split_id"]) != {"ptbxl_fold10"} or set(adaptation["split_id"]) != {"ptbxl_fold9"}:
            raise RuntimeError("PTB-XL calibration requires fold 9 adaptation and fold 10 test artifacts")
        if not np.array_equal(test["class_names"], adaptation["class_names"]):
            raise ValueError("PTB-XL adaptation/test class order differs")
        overlap = set(test["group_id"]) & set(adaptation["group_id"])
        if overlap:
            raise RuntimeError(f"PTB-XL patient leakage between folds 9 and 10: {sorted(overlap)[:10]}")
    else:
        adaptation = test

    rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    bootstrap_items: dict[str, Any] = {}
    split_arrays: dict[str, np.ndarray] = {}
    metric_cache_dir = resolve(args.metric_cache_dir) / args.dataset
    metric_cache_dir.mkdir(parents=True, exist_ok=True)
    split_audits: dict[str, Any] = {}
    for seed in seeds:
        if args.dataset == "ptbxl":
            pool_groups = np.random.default_rng(seed).permutation(np.unique(adaptation["group_id"]))
            test_groups = np.unique(test["group_id"])
            test_idx = np.arange(len(test["y_true"]), dtype=np.int64)
            train_source = adaptation
            split_audits[f"seed{seed}"] = {
                "split_policy": "official_ptbxl_fold9_adaptation_fold10_test",
                "train_groups": int(len(pool_groups)),
                "test_groups": int(len(test_groups)),
                "group_overlap": 0,
            }
        else:
            pool_groups, test_groups, split_audit = hash_group_train_test_split(
                test["group_id"],
                args.test_fraction,
                seed,
            )
            split_audits[f"seed{seed}"] = split_audit
            test_idx = group_rows(test["group_id"], test_groups)
            train_source = test
        split_arrays[f"seed{seed}_test_group_id"] = np.asarray(test_groups)
        split_arrays[f"seed{seed}_test_index"] = test_idx
        zero_prob = test["y_prob"][test_idx]
        zero_y = test["y_true"][test_idx]
        zero_groups = test["group_id"][test_idx]
        for fraction in fractions:
            selected_train_groups = nested_group_subset(pool_groups, fraction)
            train_idx = group_rows(train_source["group_id"], selected_train_groups)
            split_key = f"seed{seed}_frac{fraction:g}".replace(".", "p")
            split_arrays[f"{split_key}_train_group_id"] = np.asarray(selected_train_groups)
            split_arrays[f"{split_key}_train_index"] = train_idx
            overlap = set(selected_train_groups) & set(test_groups)
            if overlap:
                raise RuntimeError(f"Group leakage in {split_key}: {sorted(overlap)[:10]}")
            if len(train_idx) == 0:
                adapted_prob = zero_prob
                coefficients: list[dict[str, Any]] = []
                mode = "zero_target_label_identity"
            else:
                adapted_prob, coefficients = calibrate_scores(
                    train_source["y_true"][train_idx],
                    train_source["y_prob"][train_idx],
                    zero_prob,
                    test["class_names"],
                    seed,
                )
                mode = "group_safe_score_calibration"
            points = point_bundle(zero_y, adapted_prob, args.threshold, args.n_bins)
            row = {
                "dataset": args.dataset,
                "protocol": PROTOCOL,
                "mode": mode,
                "seed": seed,
                "fraction": fraction,
                "budget_role": budget_role(fraction, args.primary_fraction),
                "fraction_unit": "independent_target_groups_from_adaptation_pool",
                "fraction_sampling": "nested_seeded_label_independent_group_prefix",
                "train_groups": int(len(selected_train_groups)),
                "train_records_or_windows": int(len(train_idx)),
                "test_groups": int(len(np.unique(zero_groups))),
                "test_records_or_windows": int(len(test_idx)),
                "group_unit": test["group_unit"],
                "group_overlap": 0,
                "split_policy": (
                    "official_ptbxl_fold9_adaptation_fold10_test"
                    if args.dataset == "ptbxl"
                    else "sha256_label_independent_fixed_test_nested_adaptation_prefix"
                ),
                **points,
            }
            rows.append(row)
            for coefficient in coefficients:
                coefficient_rows.append(
                    {"dataset": args.dataset, "seed": seed, "fraction": fraction, **coefficient}
                )
            item: dict[str, Any] = {
                "seed": seed,
                "fraction": fraction,
                "mode": mode,
                "train_groups": int(len(selected_train_groups)),
                "test_groups": int(len(np.unique(zero_groups))),
                "group_overlap": 0,
                "metrics": {},
            }
            for metric_name, (metric_fn, higher_is_better) in metric_functions(
                args.threshold, args.n_bins
            ).items():
                cache_key = metric_cache_key(
                    args,
                    test_sha=test["sha256"],
                    adaptation_sha=adaptation["sha256"],
                    split_key=split_key,
                    train_groups=selected_train_groups,
                    test_groups=test_groups,
                    metric_name=metric_name,
                    analysis_lock_sha256=(analysis_lock or {}).get("sha256"),
                )
                cache_path = metric_cache_dir / f"{split_key}_{metric_name}_{cache_key[:16]}.json"
                if args.reuse_existing and not args.force_rerun and cache_path.exists():
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cached.get("cache_key") != cache_key:
                        raise RuntimeError(f"Metric cache key mismatch: {cache_path}")
                    ci = cached["cluster_ci"]
                    paired = cached["paired_vs_zero"]
                    print(f"Reusing {split_key}/{metric_name}: {cache_path}", flush=True)
                else:
                    ci = cluster_bootstrap_ci(
                        zero_y,
                        adapted_prob,
                        zero_groups,
                        metric_fn,
                        n_boot=args.n_boot,
                        seed=seed,
                    )
                    paired = (
                        exact_zero_delta(args.n_boot, len(np.unique(zero_groups)))
                        if np.array_equal(adapted_prob, zero_prob)
                        else paired_cluster_bootstrap_delta(
                            zero_y,
                            adapted_prob,
                            zero_prob,
                            zero_groups,
                            metric_fn,
                            n_boot=args.n_boot,
                            seed=seed,
                        )
                    )
                    save_json_atomic(
                        cache_path,
                        {
                            "cache_key": cache_key,
                            "split_key": split_key,
                            "metric": metric_name,
                            "cluster_ci": ci,
                            "paired_vs_zero": paired,
                        },
                    )
                if not higher_is_better:
                    paired = {
                        **paired,
                        "point_improvement_adapted_over_zero": -paired["point_delta_a_minus_b"],
                        "improvement_ci_low": -paired["hi"],
                        "improvement_ci_high": -paired["lo"],
                    }
                else:
                    paired = {
                        **paired,
                        "point_improvement_adapted_over_zero": paired["point_delta_a_minus_b"],
                        "improvement_ci_low": paired["lo"],
                        "improvement_ci_high": paired["hi"],
                    }
                item["metrics"][metric_name] = {"cluster_ci": ci, "paired_vs_zero": paired}
            bootstrap_items[split_key] = item
            print(
                f"{split_key}: mode={mode} train_groups={len(selected_train_groups)} "
                f"test_groups={len(np.unique(zero_groups))} F1={points['f1_macro']:.4f} "
                f"PR={points['pr_auc_macro']:.4f} ROC={points['roc_auc_macro']:.4f}",
                flush=True,
            )

    save_npz_compressed_atomic(
        paths["splits"],
        protocol=np.asarray(PROTOCOL),
        dataset=np.asarray(args.dataset),
        test_prediction_sha256=np.asarray(test["sha256"]),
        adaptation_prediction_sha256=np.asarray(adaptation["sha256"]),
        class_names=test["class_names"],
        **split_arrays,
    )
    save_csv(paths["summary"], rows)
    save_csv(paths["table"], rows)
    save_csv(paths["coefficients"], coefficient_rows)
    save_json(
        paths["bootstrap"],
        {
            "status": "complete",
            "protocol": PROTOCOL,
            "dataset": args.dataset,
            "n_boot": args.n_boot,
            "bootstrap_unit": "patient/source-record group",
            "items": bootstrap_items,
        },
    )
    outputs = [paths["summary"], paths["table"], paths["bootstrap"], paths["splits"], paths["coefficients"]]
    save_json(
        paths["manifest"],
        {
            "status": "complete_group_safe_score_calibration",
            "created_utc": now_utc(),
            "protocol": PROTOCOL,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "canonical_contract": canonical,
            "analysis_lock": analysis_lock,
            "dataset": args.dataset,
            "adaptation_kind": "score_calibration_only_model_weights_unchanged",
            "calibrator": {
                "kind": "per_class_unweighted_monotonic_platt_scaling",
                "input": "logit_of_frozen_probability",
                "solver": "lbfgs",
                "regularization_C": 1.0,
                "nonpositive_slope_policy": "identity_fallback",
                "ranking_invariance_expected": True,
            },
            "group_unit": test["group_unit"],
            "zero_group_overlap_all_splits": True,
            "gate": {"path": str(paths["gate"]), "sha256": sha256_file(paths["gate"]), "schema": gate.get("gate_schema_version")},
            "test_predictions": {"path": str(test["path"]), "sha256": test["sha256"]},
            "adaptation_predictions": {"path": str(adaptation["path"]), "sha256": adaptation["sha256"]},
            "fractions": fractions,
            "primary_fraction": args.primary_fraction,
            "primary_fraction_policy": "fixed_by_post_initial_review_analysis_lock_before_current_rerun",
            "fraction_unit": "independent_target_groups_from_adaptation_pool",
            "fraction_sampling": "nested_seeded_label_independent_group_prefix",
            "seeds": seeds,
            "split_audits": split_audits,
            "n_boot": args.n_boot,
            "safe_wording": (
                "Group-safe target-domain score calibration on frozen predictions with a locked 10% "
                "primary target-group budget; model weights were not fine-tuned. The lock is a "
                "post-initial-review reproducibility lock, not a preregistration."
            ),
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in outputs
            ],
        },
    )
    print(json.dumps({"status": True, "rows": len(rows), "manifest": str(paths["manifest"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
