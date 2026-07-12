"""Aggregate perturbation robustness across multiple comparators.

This script does not generate new stress predictions. It validates and compares
existing clean/stressed prediction artifacts for Full ECG-RAMBA, MiniRocket-only,
ResNet1D/CNN, Raw Mamba, and Transformer ECG. Missing comparator-stress artifacts are recorded as
blocked rows rather than silently omitted.

Use this runner only for metric-specific robustness statements. It is designed
to prevent broad robustness claims when learned-comparator stress artifacts have
not been generated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
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
    save_csv,
    save_json,
    sha256_file,
)


PROTOCOL = "robustness_multicomparator_aggregation_v1"
DEFAULT_STRESSES = (
    "snr20db",
    "snr10db",
    "snr5db",
    "random_3_lead_dropout",
    "precordial_dropout",
    "resample_250hz",
)
COMPARATORS = {
    "full": {
        "label": "Full ECG-RAMBA",
        "clean": "oof_final_ema_predictions.npz",
        "stress": "robustness_full_{stress}_predictions.npz",
    },
    "minirocket": {
        "label": "MiniRocket-only",
        # Stress predictions use the dedicated robustness heads. Degradation
        # must therefore use the clean reference from those exact heads, not
        # the separately trained canonical MiniRocket-only baseline.
        "clean": "robustness_minirocket_clean_ref_predictions.npz",
        "stress": "robustness_minirocket_{stress}_predictions.npz",
    },
    "resnet": {
        "label": "ResNet1D/CNN",
        "clean": "resnet1d_cnn_oof_predictions.npz",
        "stress": "robustness_resnet1d_cnn_{stress}_predictions.npz",
        "baseline_manifest": "resnet1d_cnn_baseline_manifest.json",
        "baseline_protocol": "resnet1d_cnn_raw_same_folds_power_mean_v2_q3_threshold_0.5",
    },
    "raw_mamba": {
        "label": "Raw Mamba",
        "clean": "raw_mamba_oof_predictions.npz",
        "stress": "robustness_raw_mamba_{stress}_predictions.npz",
        "baseline_manifest": "raw_mamba_baseline_manifest.json",
        "baseline_protocol": "raw_mamba_retrained_weighted_bce_same_folds_power_mean_v2_q3_threshold_0.5",
    },
    "transformer": {
        "label": "Transformer ECG",
        "clean": "transformer_ecg_oof_predictions.npz",
        "stress": "robustness_transformer_ecg_{stress}_predictions.npz",
        "baseline_manifest": "transformer_ecg_baseline_manifest.json",
        "baseline_protocol": "transformer_ecg_raw_same_folds_power_mean_v2_q3_threshold_0.5",
    },
}
OOF_RUN_MANIFEST = MANIFEST_DIR / "oof_final_ema_prediction_run_manifest.json"
OOF_FREEZE_MANIFEST = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
MINIROCKET_HEADS_MANIFEST = MANIFEST_DIR / "robustness_minirocket_heads_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparators", default="full,minirocket,resnet,raw_mamba,transformer")
    parser.add_argument("--stress-tests", default=",".join(DEFAULT_STRESSES))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--metrics",
        default="pr_auc_macro,roc_auc_macro,f1_macro,brier_macro,ece_macro",
        help=(
            "Comma-separated metric subset. Use pr_auc_macro,roc_auc_macro,f1_macro "
            "for a faster reviewer screening pass; include brier_macro,ece_macro "
            "for calibration/error robustness."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-jobs",
        type=int,
        default=1,
        help=(
            "Thread workers for bootstrap replicate evaluation. RNG indices are still "
            "generated serially, so changing this value does not change sampled records."
        ),
    )
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "robustness_multicomparator_metric_cache",
        help="Directory for resumable per-stress/per-comparator/per-metric bootstrap caches.",
    )
    parser.add_argument("--reuse-metric-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=METRIC_DIR / "robustness_multicomparator_summary.csv",
    )
    parser.add_argument(
        "--out-pairwise",
        type=Path,
        default=METRIC_DIR / "robustness_multicomparator_pairwise.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_robustness_multicomparator.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "robustness_multicomparator_manifest.json",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    path = resolve(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def cache_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def metric_cache_path(cache_dir: Path, stress: str, comparator: str, metric: str) -> Path:
    return resolve(cache_dir) / f"{cache_slug(stress)}__{cache_slug(comparator)}__{cache_slug(metric)}.json"


def output_profile_name(out_pairwise: Path) -> str:
    stem = Path(out_pairwise).stem
    canonical_stem = "robustness_multicomparator_pairwise"
    if stem == canonical_stem:
        return "canonical"
    profile = stem.replace("robustness_multicomparator", "", 1)
    profile = profile.replace("_pairwise", "").strip("_")
    return cache_slug(profile) if profile else "custom"


def comparator_sidecar_path(out_pairwise: Path, comparator: str) -> Path:
    profile = output_profile_name(out_pairwise)
    profile_suffix = "" if profile == "canonical" else f"_{profile}"
    return resolve(out_pairwise).parent / (
        f"robustness_full_vs_{cache_slug(comparator)}{profile_suffix}_comparison.json"
    )


def cache_metadata(
    *,
    args: argparse.Namespace,
    stress: str,
    comparator: str,
    spec: dict[str, Any],
    full_clean: dict[str, Any],
    full_stress: dict[str, Any],
    comp_clean: dict[str, Any],
    comp_stress: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "stress": stress,
        "comparator": comparator,
        "metric": spec["name"],
        "direction": spec["direction"],
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "n_boot": int(args.n_boot),
        "seed": int(seed),
        "full_clean_sha256": full_clean["sha256"],
        "full_stress_sha256": full_stress["sha256"],
        "comp_clean_sha256": comp_clean["sha256"],
        "comp_stress_sha256": comp_stress["sha256"],
    }


def read_metric_cache(path: Path, metadata: dict[str, Any]) -> dict[str, Any] | None:
    path = resolve(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: could not read metric cache {path}: {exc}", flush=True)
        return None
    if payload.get("metadata") != metadata:
        return None
    row = payload.get("row")
    return row if isinstance(row, dict) else None


def write_metric_cache(path: Path, metadata: dict[str, Any], row: dict[str, Any]) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, {"metadata": metadata, "row": row, "created_utc": now_utc()})


def load_npz(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = ["y_true", "y_prob", "record_id", "class_names", "fold_id"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} missing keys={missing}")
        payload = {key: data[key] for key in data.files}
    payload["y_true"] = np.asarray(payload["y_true"], dtype=np.float32)
    payload["y_prob"] = np.asarray(payload["y_prob"], dtype=np.float32)
    payload["record_id"] = np.asarray(payload["record_id"]).astype(str)
    payload["class_names"] = np.asarray(payload["class_names"]).astype(str)
    payload["fold_id"] = np.asarray(payload["fold_id"]).astype(int)
    if payload["y_true"].shape != payload["y_prob"].shape:
        raise ValueError(f"{path} shape mismatch: {payload['y_true'].shape} vs {payload['y_prob'].shape}")
    if payload["y_true"].ndim != 2:
        raise ValueError(f"{path} predictions must be a two-dimensional record-by-class matrix")
    n_records, n_classes = payload["y_true"].shape
    if len(payload["record_id"]) != n_records or len(payload["fold_id"]) != n_records:
        raise ValueError(f"{path} record/fold arrays do not match the prediction row count")
    if len(payload["class_names"]) != n_classes:
        raise ValueError(f"{path} class_names do not match the prediction column count")
    if np.any(~np.isfinite(payload["y_true"])) or not np.all(
        np.logical_or(payload["y_true"] == 0.0, payload["y_true"] == 1.0)
    ):
        raise ValueError(f"{path} y_true must contain finite binary labels")
    if np.any(~np.isfinite(payload["y_prob"])):
        raise ValueError(f"{path} contains non-finite probabilities")
    if np.any((payload["y_prob"] < 0.0) | (payload["y_prob"] > 1.0)):
        raise ValueError(f"{path} contains probabilities outside [0, 1]")
    payload["path"] = path
    payload["sha256"] = sha256_file(path)
    return payload


def validate_same_contract(reference: dict[str, Any], other: dict[str, Any], label: str) -> None:
    for key in ["y_true", "record_id", "class_names", "fold_id"]:
        if key not in reference or key not in other:
            continue
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"{label} differs from Full contract on {key}")


def scalar(payload: dict[str, Any], key: str, default: Any = "") -> Any:
    if key not in payload:
        return default
    value = np.asarray(payload[key])
    return value.item() if value.ndim == 0 else value


def _checkpoint_sha_rows(rows: list[dict[str, Any]], *, label: str) -> list[str]:
    try:
        ordered = sorted(rows, key=lambda row: int(row["fold"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} checkpoint rows are malformed") from exc
    folds = [int(row.get("fold", -1)) for row in ordered]
    hashes = [str(row.get("sha256") or row.get("checkpoint_sha256") or "") for row in ordered]
    if folds != [1, 2, 3, 4, 5] or any(not value for value in hashes):
        raise RuntimeError(f"{label} checkpoint contract must cover exact folds 1..5 with SHA256")
    return hashes


def load_clean_checkpoint_contract(comparator: str, clean_data: dict[str, Any]) -> list[str]:
    clean_sha256 = str(clean_data.get("sha256") or "")
    if comparator == "full":
        if not OOF_RUN_MANIFEST.exists() or OOF_RUN_MANIFEST.stat().st_size == 0:
            raise FileNotFoundError(f"Missing Full OOF run manifest: {OOF_RUN_MANIFEST}")
        payload = json.loads(OOF_RUN_MANIFEST.read_text(encoding="utf-8"))
        if payload.get("protocol") != "fold_final_ema_power_mean_v2_q3_threshold_0.5":
            raise RuntimeError("Full clean prediction run manifest has an unexpected protocol")
        expected_clean_sha = (
            (payload.get("outputs") or {}).get("prediction_file") or {}
        ).get("sha256")
        if expected_clean_sha != clean_sha256:
            raise RuntimeError(
                "Full clean prediction SHA does not match its OOF run manifest: "
                f"{clean_sha256} != {expected_clean_sha}"
            )
        return _checkpoint_sha_rows(
            list((payload.get("inputs") or {}).get("checkpoints") or []),
            label="Full ECG-RAMBA",
        )

    if comparator == "minirocket":
        if not MINIROCKET_HEADS_MANIFEST.exists() or MINIROCKET_HEADS_MANIFEST.stat().st_size == 0:
            raise FileNotFoundError(f"Missing MiniRocket robustness-head manifest: {MINIROCKET_HEADS_MANIFEST}")
        payload = json.loads(MINIROCKET_HEADS_MANIFEST.read_text(encoding="utf-8"))
        if payload.get("protocol") != "minirocket_clean_heads_for_robustness_v1":
            raise RuntimeError("MiniRocket robustness-head manifest has an unexpected protocol")
        if payload.get("clean_prediction_sha256") != clean_sha256:
            raise RuntimeError("MiniRocket clean reference SHA does not match its robustness-head manifest")
        fold_rows = sorted(payload.get("fold_rows") or [], key=lambda row: int(row.get("fold", -1)))
        if (
            [int(row.get("fold", -1)) for row in fold_rows] != [1, 2, 3, 4, 5]
            or any(not row.get("head_sha256") for row in fold_rows)
            or not payload.get("params_hash")
        ):
            raise RuntimeError("MiniRocket robustness-head contract is incomplete")
        return []

    manifest_name = COMPARATORS[comparator].get("baseline_manifest")
    if not manifest_name:
        return []
    path = MANIFEST_DIR / manifest_name
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {comparator} baseline manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_protocol = COMPARATORS[comparator].get("baseline_protocol")
    if expected_protocol and payload.get("protocol") != expected_protocol:
        raise RuntimeError(f"{comparator} baseline manifest has an unexpected protocol: {path}")
    expected_clean_sha = (payload.get("artifact_sha256") or {}).get("predictions")
    if expected_clean_sha != clean_sha256:
        raise RuntimeError(
            f"{comparator} clean prediction SHA does not match baseline manifest: "
            f"{clean_sha256} != {expected_clean_sha}"
        )
    contract = payload.get("checkpoint_contract") or {}
    rows = sorted(contract.get("checkpoints") or [], key=lambda row: int(row["fold"]))
    if (
        contract.get("status") != "complete"
        or [int(row.get("fold", -1)) for row in rows] != [1, 2, 3, 4, 5]
        or any(not row.get("sha256") for row in rows)
    ):
        raise RuntimeError(f"{comparator} baseline checkpoint contract is incomplete: {path}")
    expected_hashes = np.asarray([str(row["sha256"]) for row in rows])
    embedded_folds = np.asarray(clean_data.get("checkpoint_folds", []), dtype=np.int16)
    embedded_hashes = np.asarray(clean_data.get("checkpoint_sha256", [])).astype(str)
    if not np.array_equal(embedded_folds, np.asarray([1, 2, 3, 4, 5], dtype=np.int16)):
        raise RuntimeError(f"{comparator} clean predictions lack the exact five-fold checkpoint contract")
    if not np.array_equal(embedded_hashes, expected_hashes):
        raise RuntimeError(
            f"{comparator} clean prediction checkpoint SHA contract differs from its baseline manifest"
        )
    return expected_hashes.tolist()


def load_canonical_contract(clean_sha256: str) -> dict[str, str]:
    if not OOF_FREEZE_MANIFEST.exists() or OOF_FREEZE_MANIFEST.stat().st_size == 0:
        raise FileNotFoundError(f"Missing frozen OOF manifest: {OOF_FREEZE_MANIFEST}")
    payload = json.loads(OOF_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or payload.get("checkpoint_kind") != "final_ema":
        raise RuntimeError("Frozen OOF manifest is not the canonical final_ema contract")
    prediction_rows = [
        row
        for row in payload.get("artifacts") or []
        if str(row.get("path", "")).endswith("/oof_final_ema_predictions.npz")
    ]
    if len(prediction_rows) != 1 or prediction_rows[0].get("sha256") != clean_sha256:
        raise RuntimeError("Full clean prediction SHA does not match the frozen OOF manifest")
    return {
        "oof_sha256": clean_sha256,
        "freeze_sha256": sha256_file(OOF_FREEZE_MANIFEST),
    }


def validate_stress_provenance(
    comparator: str,
    stress: str,
    clean: dict[str, Any],
    stress_data: dict[str, Any],
) -> None:
    if comparator == "full":
        expected = np.asarray(clean.get("checkpoint_sha256", [])).astype(str)
        metadata = json.loads(str(scalar(stress_data, "metadata_json", "{}")))
        fold_rows = list(metadata.get("fold_rows") or [])
        try:
            actual = np.asarray(
                [
                    str(row.get("checkpoint_sha256") or "")
                    for row in sorted(fold_rows, key=lambda row: int(row.get("fold", -1)))
                ]
            )
            actual_folds = [
                int(row.get("fold", -1))
                for row in sorted(fold_rows, key=lambda row: int(row.get("fold", -1)))
            ]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"full/{stress} checkpoint metadata is malformed") from exc
        if expected.shape != (5,) or actual_folds != [1, 2, 3, 4, 5] or not np.array_equal(actual, expected):
            raise RuntimeError(f"full/{stress} stress checkpoints do not match the frozen Full OOF contract")

    if comparator in {"resnet", "raw_mamba", "transformer"}:
        expected = np.asarray(clean.get("checkpoint_sha256", [])).astype(str)
        actual = np.asarray(stress_data.get("checkpoint_sha256", [])).astype(str)
        if expected.shape != (5,) or not np.array_equal(actual, expected):
            raise RuntimeError(
                f"{comparator}/{stress} stress checkpoints do not match the clean baseline contract"
            )
        if str(scalar(stress_data, "protocol")) != "comparator_stress_predictions_v1_same_folds_power_mean_v2_q3":
            raise RuntimeError(f"{comparator}/{stress} has an unexpected stress protocol")
        if str(scalar(stress_data, "comparator")) != comparator:
            raise RuntimeError(f"{comparator}/{stress} comparator tag mismatch")
        if str(scalar(stress_data, "stress_test")) != stress:
            raise RuntimeError(f"{comparator}/{stress} stress tag mismatch")
        return

    if str(scalar(stress_data, "protocol")) != "robustness_full_vs_minirocket_perturbation_v1":
        raise RuntimeError(f"{comparator}/{stress} has an unexpected stress protocol")
    if str(scalar(stress_data, "stress_name")) != stress:
        raise RuntimeError(f"{comparator}/{stress} stress tag mismatch")
    expected_model = "Full ECG-RAMBA" if comparator == "full" else "MiniRocket-only"
    if str(scalar(stress_data, "model_label")) != expected_model:
        raise RuntimeError(f"{comparator}/{stress} model tag mismatch")
    if comparator == "minirocket":
        manifest_path = MINIROCKET_HEADS_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("clean_prediction_sha256") != clean.get("sha256"):
            raise RuntimeError("MiniRocket robustness clean reference does not match its head manifest")
        metadata = json.loads(str(scalar(stress_data, "metadata_json", "{}")))
        cached_manifest = metadata.get("minirocket_heads_manifest") or {}
        if cached_manifest.get("params_hash") != manifest.get("params_hash"):
            raise RuntimeError(
                f"minirocket/{stress} was generated by a different robustness head contract"
            )
        if cached_manifest.get("clean_prediction_sha256") != clean.get("sha256"):
            raise RuntimeError(f"minirocket/{stress} references a different clean robustness prediction")


def metric_specs(threshold: float, n_bins: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "pr_auc_macro",
            "direction": "higher",
            "fn": macro_pr_auc,
        },
        {
            "name": "roc_auc_macro",
            "direction": "higher",
            "fn": macro_roc_auc,
        },
        {
            "name": "f1_macro",
            "direction": "higher",
            "fn": lambda y, p: multilabel_metrics(y, p, threshold=threshold)["f1_macro"],
        },
        {
            "name": "brier_macro",
            "direction": "lower",
            "fn": lambda y, p: calibration_summary(y, p, n_bins=n_bins)["brier_macro"],
        },
        {
            "name": "ece_macro",
            "direction": "lower",
            "fn": lambda y, p: calibration_summary(y, p, n_bins=n_bins)["ece_macro"],
        },
    ]


def filter_metric_specs(specs: list[dict[str, Any]], requested: list[str]) -> list[dict[str, Any]]:
    available = {spec["name"]: spec for spec in specs}
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}; choices={sorted(available)}")
    return [available[name] for name in requested]


def benefit(value: float, direction: str) -> float:
    return value if direction == "higher" else -value


def metric_value(spec: dict[str, Any], data: dict[str, Any], idx: np.ndarray | None = None) -> float:
    y = data["y_true"] if idx is None else data["y_true"][idx]
    p = data["y_prob"] if idx is None else data["y_prob"][idx]
    try:
        value = float(spec["fn"](y, p))
    except ValueError:
        return float("nan")
    return value


def paired_bootstrap(
    spec: dict[str, Any],
    full_clean: dict[str, Any],
    full_stress: dict[str, Any],
    comp_clean: dict[str, Any],
    comp_stress: dict[str, Any],
    n_boot: int,
    seed: int,
    n_jobs: int = 1,
    shared_full_cache: dict[tuple[Any, ...], list[tuple[float, float]]] | None = None,
    shared_full_cache_key: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(full_clean["y_true"])
    values = []
    stressed_values = []

    cached_full_values = (
        shared_full_cache.get(shared_full_cache_key)
        if shared_full_cache is not None and shared_full_cache_key is not None
        else None
    )
    if cached_full_values is not None and len(cached_full_values) != n_boot:
        raise RuntimeError("Shared Full bootstrap cache length does not match n_boot")

    def evaluate(item: tuple[int, np.ndarray]) -> tuple[tuple[float, float], tuple[float, float] | None]:
        ordinal, idx = item
        if cached_full_values is None:
            fc = metric_value(spec, full_clean, idx)
            fs = metric_value(spec, full_stress, idx)
        else:
            fc, fs = cached_full_values[ordinal]
        cc = metric_value(spec, comp_clean, idx)
        cs = metric_value(spec, comp_stress, idx)
        full_pair = (float(fc), float(fs))
        if not all(np.isfinite([fc, fs, cc, cs])):
            return full_pair, None
        full_deg = benefit(fs, spec["direction"]) - benefit(fc, spec["direction"])
        comp_deg = benefit(cs, spec["direction"]) - benefit(cc, spec["direction"])
        return full_pair, (
            float(full_deg - comp_deg),
            float(benefit(fs, spec["direction"]) - benefit(cs, spec["direction"])),
        )

    n_jobs = max(1, min(int(n_jobs), int(n_boot)))
    batch_size = max(1, n_jobs * 2)
    executor_context = (
        concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs)
        if n_jobs > 1
        else None
    )
    computed_full_values: list[tuple[float, float]] = []
    ordinal = 0
    try:
        for start in range(0, n_boot, batch_size):
            count = min(batch_size, n_boot - start)
            # Generate indices in the caller thread and in the same order as
            # the historical sequential implementation. Executor.map returns
            # results in input order, preserving deterministic quantiles.
            indices = [rng.integers(0, n, size=n) for _ in range(count)]
            indexed = list(enumerate(indices, start=ordinal))
            ordinal += count
            results = (
                executor_context.map(evaluate, indexed)
                if executor_context is not None
                else map(evaluate, indexed)
            )
            for full_pair, result in results:
                if cached_full_values is None:
                    computed_full_values.append(full_pair)
                if result is None:
                    continue
                degradation_value, stressed_value = result
                values.append(degradation_value)
                stressed_values.append(stressed_value)
    finally:
        if executor_context is not None:
            executor_context.shutdown(wait=True)
    if (
        cached_full_values is None
        and shared_full_cache is not None
        and shared_full_cache_key is not None
    ):
        if len(computed_full_values) != n_boot:
            raise RuntimeError("Could not build the complete shared Full bootstrap cache")
        shared_full_cache[shared_full_cache_key] = computed_full_values
    if not values:
        return {"n_boot_valid": 0, "degradation_adv_ci_low": math.nan, "degradation_adv_ci_high": math.nan}
    lo, hi = np.quantile(values, [0.025, 0.975])
    slo, shi = np.quantile(stressed_values, [0.025, 0.975])
    return {
        "n_boot_valid": int(len(values)),
        "degradation_adv_mean": float(np.mean(values)),
        "degradation_adv_ci_low": float(lo),
        "degradation_adv_ci_high": float(hi),
        "stressed_adv_mean": float(np.mean(stressed_values)),
        "stressed_adv_ci_low": float(slo),
        "stressed_adv_ci_high": float(shi),
    }


def interpretation(ci_low: float, ci_high: float) -> str:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "insufficient_bootstrap"
    if ci_low > 0:
        return "full_significantly_less_degraded"
    if ci_high < 0:
        return "comparator_significantly_less_degraded"
    return "no_significant_degradation_difference"


def row_interpretation(row: dict[str, Any]) -> str:
    cached = str(row.get("interpretation") or "").strip()
    if cached:
        return cached
    return interpretation(
        float(row.get("degradation_adv_ci_low", math.nan)),
        float(row.get("degradation_adv_ci_high", math.nan)),
    )


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    for path in [args.out_summary, args.out_pairwise, args.out_table, args.out_manifest]:
        path.parent.mkdir(parents=True, exist_ok=True)
    resolve(args.metric_cache_dir).mkdir(parents=True, exist_ok=True)

    comparators = parse_list(args.comparators)
    stresses = parse_list(args.stress_tests)
    unknown = [item for item in comparators if item not in COMPARATORS]
    if unknown:
        raise ValueError(f"Unknown comparators: {unknown}; choices={sorted(COMPARATORS)}")
    if "full" not in comparators:
        comparators = ["full", *comparators]

    print("=" * 80, flush=True)
    print("ROBUSTNESS MULTI-COMPARATOR AGGREGATION", flush=True)
    print("=" * 80, flush=True)
    print(f"comparators={comparators}", flush=True)
    print(f"stress_tests={stresses}", flush=True)
    requested_metrics = parse_list(args.metrics)
    if not requested_metrics:
        raise ValueError("--metrics must contain at least one metric name.")
    print(f"metrics={requested_metrics}", flush=True)
    if args.bootstrap_jobs < 1:
        raise ValueError("--bootstrap-jobs must be at least 1")
    print(f"bootstrap_jobs={args.bootstrap_jobs}", flush=True)
    print(f"metric_cache_dir={resolve(args.metric_cache_dir)} reuse={args.reuse_metric_cache}", flush=True)

    clean: dict[str, dict[str, Any]] = {}
    artifact_status: list[dict[str, Any]] = []
    for comp in comparators:
        path = PREDICTION_DIR / COMPARATORS[comp]["clean"]
        try:
            clean[comp] = load_npz(path)
            checkpoint_sha = load_clean_checkpoint_contract(
                comp,
                clean[comp],
            )
            if checkpoint_sha:
                clean[comp]["checkpoint_sha256"] = np.asarray(checkpoint_sha)
            artifact_status.append(
                {
                    "comparator": comp,
                    "kind": "clean",
                    "path": project_relative(path),
                    "exists": True,
                    "sha256": clean[comp]["sha256"],
                    "status": "ready",
                }
            )
        except Exception as exc:
            artifact_status.append(
                {
                    "comparator": comp,
                    "kind": "clean",
                    "path": project_relative(path),
                    "exists": False,
                    "sha256": "",
                    "status": f"missing_or_invalid:{exc}",
                }
            )

    if "full" not in clean:
        payload = {
            "status": "blocked_missing_full_clean_predictions",
            "protocol": PROTOCOL,
            "created_utc": now_utc(),
            "artifact_status": artifact_status,
            "safe_wording": "Cannot evaluate robustness without frozen Full ECG-RAMBA clean predictions.",
            "git_commit": git_commit(),
        }
        save_json(args.out_manifest, payload)
        save_json(args.out_pairwise, payload)
        save_csv(args.out_summary, artifact_status)
        save_csv(args.out_table, artifact_status)
        if args.strict:
            raise FileNotFoundError("Missing Full clean predictions.")
        print(json.dumps(payload, indent=2), flush=True)
        return

    full_clean = clean["full"]
    canonical_contract = load_canonical_contract(full_clean["sha256"])
    for comp, data in list(clean.items()):
        if comp == "full":
            continue
        try:
            validate_same_contract(full_clean, data, comp)
        except Exception as exc:
            artifact_status.append(
                {
                    "comparator": comp,
                    "kind": "contract",
                    "path": "",
                    "exists": True,
                    "sha256": "",
                    "status": f"contract_failed:{exc}",
                }
            )
            del clean[comp]

    specs = filter_metric_specs(metric_specs(args.threshold, args.n_bins), requested_metrics)
    rows: list[dict[str, Any]] = []
    pairwise: dict[str, Any] = {
        "status": "complete_with_possible_missing_comparators",
        "protocol": PROTOCOL,
        "created_utc": now_utc(),
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "bootstrap_jobs": args.bootstrap_jobs,
        "metrics": requested_metrics,
        "metric_cache_dir": project_relative(args.metric_cache_dir),
        "output_profile": output_profile_name(args.out_pairwise),
        "comparators": comparators,
        "stress_tests": stresses,
        "canonical_contract": canonical_contract,
        "runner_sha256": sha256_file(Path(__file__)),
        "items": {},
    }

    for stress in stresses:
        stress_data: dict[str, dict[str, Any]] = {}
        shared_full_bootstrap_cache: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
        for comp in comparators:
            if comp not in clean:
                continue
            stress_path = PREDICTION_DIR / COMPARATORS[comp]["stress"].format(stress=stress)
            try:
                stress_data[comp] = load_npz(stress_path)
                validate_same_contract(full_clean, stress_data[comp], f"{comp}/{stress}")
                validate_stress_provenance(comp, stress, clean[comp], stress_data[comp])
                artifact_status.append(
                    {
                        "comparator": comp,
                        "kind": f"stress:{stress}",
                        "path": project_relative(stress_path),
                        "exists": True,
                        "sha256": stress_data[comp]["sha256"],
                        "status": "ready",
                    }
                )
            except Exception as exc:
                artifact_status.append(
                    {
                        "comparator": comp,
                        "kind": f"stress:{stress}",
                        "path": project_relative(stress_path),
                        "exists": False,
                        "sha256": "",
                        "status": f"missing_or_invalid:{exc}",
                    }
                )

        for comp in [c for c in comparators if c != "full"]:
            for spec_idx, spec in enumerate(specs):
                base_row: dict[str, Any] = {
                    "stress": stress,
                    "comparator": comp,
                    "comparator_label": COMPARATORS.get(comp, {}).get("label", comp),
                    "metric": spec["name"],
                    "direction": spec["direction"],
                    "output_profile": output_profile_name(args.out_pairwise),
                    "threshold": args.threshold,
                    "n_bins": args.n_bins,
                    "n_boot": args.n_boot,
                }
                if comp not in clean:
                    rows.append({**base_row, "status": "blocked_missing_clean_comparator"})
                    continue
                if "full" not in stress_data or comp not in stress_data:
                    rows.append({**base_row, "status": "blocked_missing_stress_predictions"})
                    continue

                fc = metric_value(spec, full_clean)
                fs = metric_value(spec, stress_data["full"])
                cc = metric_value(spec, clean[comp])
                cs = metric_value(spec, stress_data[comp])
                full_deg = benefit(fs, spec["direction"]) - benefit(fc, spec["direction"])
                comp_deg = benefit(cs, spec["direction"]) - benefit(cc, spec["direction"])
                deg_adv = full_deg - comp_deg
                stressed_adv = benefit(fs, spec["direction"]) - benefit(cs, spec["direction"])
                seed = args.seed + spec_idx
                metadata = cache_metadata(
                    args=args,
                    stress=stress,
                    comparator=comp,
                    spec=spec,
                    full_clean=full_clean,
                    full_stress=stress_data["full"],
                    comp_clean=clean[comp],
                    comp_stress=stress_data[comp],
                    seed=seed,
                )
                cache_path = metric_cache_path(args.metric_cache_dir, stress, comp, spec["name"])
                row = read_metric_cache(cache_path, metadata) if args.reuse_metric_cache else None
                if row is not None:
                    print(f"{stress} {comp} {spec['name']}: cache hit {project_relative(cache_path)}", flush=True)
                else:
                    print(f"{stress} {comp} {spec['name']}: bootstrap start", flush=True)
                    boot = paired_bootstrap(
                        spec,
                        full_clean,
                        stress_data["full"],
                        clean[comp],
                        stress_data[comp],
                        args.n_boot,
                        seed,
                        args.bootstrap_jobs,
                        shared_full_bootstrap_cache,
                        (
                            stress,
                            spec["name"],
                            seed,
                            args.n_boot,
                            stress_data["full"]["sha256"],
                        ),
                    )
                    interp = interpretation(
                        boot.get("degradation_adv_ci_low", math.nan),
                        boot.get("degradation_adv_ci_high", math.nan),
                    )
                    row = {
                        **base_row,
                        "status": "complete",
                        "clean_full": fc,
                        "stress_full": fs,
                        "degradation_full_benefit": full_deg,
                        "clean_comparator": cc,
                        "stress_comparator": cs,
                        "degradation_comparator_benefit": comp_deg,
                        "degradation_advantage_full": deg_adv,
                        "stressed_advantage_full": stressed_adv,
                        "degradation_adv_ci_low": boot.get("degradation_adv_ci_low"),
                        "degradation_adv_ci_high": boot.get("degradation_adv_ci_high"),
                        "stressed_adv_ci_low": boot.get("stressed_adv_ci_low"),
                        "stressed_adv_ci_high": boot.get("stressed_adv_ci_high"),
                        "n_boot_valid": boot.get("n_boot_valid"),
                        "interpretation": interp,
                    }
                    write_metric_cache(cache_path, metadata, row)
                    print(f"{stress} {comp} {spec['name']}: bootstrap done", flush=True)
                interp = row_interpretation(row)
                rows.append(row)
                pairwise["items"][f"{stress}/{comp}/{spec['name']}"] = row
                print(
                    f"{stress} {comp} {spec['name']}: stress_full={fs:.6f} "
                    f"stress_comp={cs:.6f} deg_adv={deg_adv:.6f} {interp}",
                    flush=True,
                )

    completed = [row for row in rows if row.get("status") == "complete"]
    blocked = [row for row in rows if row.get("status") != "complete"]
    pairwise["completed_rows"] = len(completed)
    pairwise["blocked_rows"] = len(blocked)
    pairwise["status"] = "complete_with_blockers" if blocked else "complete"
    pairwise["artifact_status"] = artifact_status
    pairwise["safe_wording"] = (
        "Use only metric-specific and comparator-specific robustness statements. "
        "Missing stress artifacts keep broad robustness superiority blocked."
    )
    manifest = {
        "status": "complete_with_blockers" if blocked else "complete",
        "protocol": PROTOCOL,
        "created_utc": now_utc(),
        "comparators": comparators,
        "stress_tests": stresses,
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "bootstrap_jobs": args.bootstrap_jobs,
        "metrics": requested_metrics,
        "output_profile": output_profile_name(args.out_pairwise),
        "completed_rows": len(completed),
        "blocked_rows": len(blocked),
        "artifact_status": artifact_status,
        "outputs": {
            "summary": project_relative(args.out_summary),
            "table": project_relative(args.out_table),
            "pairwise": project_relative(args.out_pairwise),
            "manifest": project_relative(args.out_manifest),
        },
        "canonical_contract": canonical_contract,
        "runner_sha256": sha256_file(Path(__file__)),
        "git_commit": git_commit(),
    }
    save_csv(args.out_summary, rows)
    save_csv(args.out_table, rows)
    save_json(args.out_pairwise, pairwise)
    pairwise_sha256 = sha256_file(resolve(args.out_pairwise))
    sidecar_outputs = {}
    for comp in comparators:
        # The canonical Full-vs-MiniRocket comparison is owned by script 12 and
        # has a different schema. Never overwrite it from this ledger.
        if comp in {"full", "minirocket"}:
            continue
        comp_rows = [row for row in rows if row.get("comparator") == comp]
        comp_blocked = [row for row in comp_rows if row.get("status") != "complete"]
        expected_rows = len(stresses) * len(requested_metrics)
        sidecar_path = comparator_sidecar_path(args.out_pairwise, comp)
        sidecar = {
            "status": (
                "complete"
                if len(comp_rows) == expected_rows and not comp_blocked
                else "complete_with_blockers"
            ),
            "protocol": PROTOCOL,
            "created_utc": now_utc(),
            "comparator": comp,
            "comparator_label": COMPARATORS[comp]["label"],
            "stress_tests": stresses,
            "metrics": requested_metrics,
            "threshold": args.threshold,
            "n_bins": args.n_bins,
            "n_boot": args.n_boot,
            "bootstrap_jobs": args.bootstrap_jobs,
            "output_profile": output_profile_name(args.out_pairwise),
            "expected_rows": expected_rows,
            "completed_rows": len(comp_rows) - len(comp_blocked),
            "blocked_rows": len(comp_blocked),
            "rows": comp_rows,
            "source_pairwise": project_relative(args.out_pairwise),
            "source_pairwise_sha256": pairwise_sha256,
            "canonical_contract": canonical_contract,
            "runner_sha256": sha256_file(Path(__file__)),
            "safe_wording": (
                "Use only stress-, metric-, and comparator-specific paired degradation CIs."
            ),
        }
        save_json(sidecar_path, sidecar)
        sidecar_outputs[comp] = project_relative(sidecar_path)
    manifest["outputs"]["comparator_sidecars"] = sidecar_outputs
    manifest["artifact_sha256"] = {
        "summary": sha256_file(resolve(args.out_summary)),
        "table": sha256_file(resolve(args.out_table)),
        "pairwise": pairwise_sha256,
        "comparator_sidecars": {
            comp: sha256_file(comparator_sidecar_path(args.out_pairwise, comp))
            for comp in sidecar_outputs
        },
    }
    save_json(args.out_manifest, manifest)
    print(json.dumps({"status": True, "completed_rows": len(completed), "blocked_rows": len(blocked)}, indent=2))
    if args.strict and blocked:
        raise RuntimeError(f"Blocked robustness rows remain: {len(blocked)}")


if __name__ == "__main__":
    main()
