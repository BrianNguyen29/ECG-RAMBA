"""Aggregate perturbation robustness across multiple comparators.

This script does not generate new stress predictions. It validates and compares
existing clean/stressed prediction artifacts for Full ECG-RAMBA, the fixed-seed
ROCKET-family MAX+PPV linear head,
ResNet1D/CNN, Raw Mamba, and Transformer ECG. Missing comparator-stress artifacts are recorded as
blocked rows rather than silently omitted.

Use this runner only for metric-specific robustness statements. It is designed
to prevent broad robustness claims when learned-comparator stress artifacts have
not been generated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import sparse

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
    ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION,
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


PROTOCOL = "robustness_multicomparator_aggregation_v2_source_bound"
BOOTSTRAP_ENGINE = "paired_record_resample_presorted_rank_sparse_ece_weighted_counts_v2"
METRIC_CACHE_SCHEMA_VERSION = ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION
CI_SCOPE = "nominal_95_percentile_paired_record_bootstrap_unadjusted"
BOOTSTRAP_UNIT = AUTHENTICATED_RECORD_BOOTSTRAP_UNIT
TRAINING_VARIABILITY_SCOPE = "fixed_trained_folds_and_checkpoints_not_retrained_within_bootstrap"
COMPARATOR_STRESS_PROTOCOL = "comparator_stress_predictions_v2_source_bound_same_folds_power_mean_v2_q3"
COMPARATOR_STRESS_SOURCE_PATHS = (
    "scripts/revision/23_generate_comparator_stress_predictions.py",
    "scripts/revision/12_robustness_stress.py",
    "scripts/revision/common.py",
    "scripts/revision/14_resnet1d_cnn_baseline.py",
    "scripts/revision/16_raw_mamba_baseline.py",
    "scripts/revision/24_transformer_ecg_baseline.py",
    "src/aggregation.py",
    "src/training_data.py",
    "configs/config.py",
)
ROCKET_STRESS_PROTOCOL = "robustness_full_vs_fixed_seed_rocket_perturbation_v2_source_bound"
ROCKET_STRESS_SOURCE_PATHS = (
    "scripts/revision/12_robustness_stress.py",
    "scripts/revision/common.py",
    "scripts/revision/01_generate_predictions.py",
    "scripts/revision/10_minirocket_only_baseline.py",
    "src/aggregation.py",
    "src/features.py",
    "src/provenance.py",
    "configs/config.py",
)


def current_comparator_stress_source_bundle() -> dict[str, Any]:
    files = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in COMPARATOR_STRESS_SOURCE_PATHS
    }
    return {
        "schema_version": 1,
        "files": files,
        "sha256": _canonical_json_sha256(files),
    }


def current_rocket_stress_source_bundle() -> dict[str, Any]:
    files = {relative: sha256_file(PROJECT_ROOT / relative) for relative in ROCKET_STRESS_SOURCE_PATHS}
    return {"schema_version": 1, "files": files, "sha256": _canonical_json_sha256(files)}
MACRO_CLASS_SUPPORT_POLICY = (
    "rank_calibration_omit_single_resampled_class_f1_keeps_all_labels_zero_division_zero"
)
STRESS_INPUT_SPACE = "bandpass_filtered_per_lead_z_normalized_model_input"
CHAPMAN_LEAD_ORDER = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
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
        "label": "Fixed-seed ROCKET-family MAX+PPV linear head",
        # Stress predictions use the dedicated robustness heads. Degradation
        # must therefore use the clean reference from those exact heads, not
        # the separately trained fixed-seed ROCKET-family MAX+PPV baseline.
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
CALIBRATION_CI = METRIC_DIR / "calibration_ci_oof_final_ema_predictions.json"


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
            "Thread workers for bootstrap replicate evaluation. Seeded record draws are "
            "generated serially and cached, so changing this value does not change sampled records."
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


def expected_stress_spec(name: str, seed: int) -> dict[str, Any]:
    """Return the exact perturbation contract used by scripts 12 and 23."""

    if name == "snr20db":
        return {"name": name, "kind": "additive_noise", "snr_db": 20.0, "seed": seed + 2001}
    if name == "snr10db":
        return {"name": name, "kind": "additive_noise", "snr_db": 10.0, "seed": seed + 2011}
    if name == "snr5db":
        return {"name": name, "kind": "additive_noise", "snr_db": 5.0, "seed": seed + 2021}
    if name == "random_3_lead_dropout":
        return {"name": name, "kind": "random_lead_dropout", "n_drop": 3, "seed": seed + 3001}
    if name == "precordial_dropout":
        return {
            "name": name,
            "kind": "fixed_lead_dropout",
            "lead_indices": list(range(6, 12)),
            "seed": seed,
        }
    if name == "resample_250hz":
        return {
            "name": name,
            "kind": "resample_down_up",
            "source_hz": 500,
            "target_hz": 250,
            "seed": seed,
        }
    raise ValueError(f"Unknown stress test: {name}")


def stress_contract_description(spec: dict[str, Any]) -> dict[str, Any]:
    description: dict[str, Any] = {
        "spec": spec,
        "input_space": STRESS_INPUT_SPACE,
        "same_realization_across_models": True,
        "realization_scope": "single_fixed_seed_conditional_stress_audit",
    }
    if spec["kind"] in {"random_lead_dropout", "fixed_lead_dropout"}:
        description["lead_order"] = list(CHAPMAN_LEAD_ORDER)
    if spec["kind"] == "additive_noise":
        description["implementation"] = (
            "iid_gaussian_noise_scaled_per_record_from_global_mean_square_across_leads_and_time"
        )
        description["snr_definition"] = "signal_power_over_noise_power_in_model_input_space"
        description["amplitude_clipping"] = False
    elif spec["kind"] == "random_lead_dropout":
        description["implementation"] = (
            "per_record_uniform_without_replacement_three_of_twelve_leads_zero_filled"
        )
    elif spec["kind"] == "fixed_lead_dropout":
        description["implementation"] = "fixed_v1_to_v6_zero_filled"
    if spec["kind"] == "resample_down_up":
        description["implementation"] = (
            "scipy_resample_poly_500_to_250_then_250_to_500_default_antialias_fir_trim_or_zero_pad"
        )
        description["interpretation"] = (
            "anti_aliased_500_to_250_to_500_hz_bandwidth_perturbation_not_native_250hz_deployment"
        )
    return description


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
    canonical_contract: dict[str, Any],
    bootstrap_contract: dict[str, Any],
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
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "statistical_helper_sha256": sha256_file(
            PROJECT_ROOT / "scripts" / "revision" / "common.py"
        ),
        "metric_cache_schema_version": METRIC_CACHE_SCHEMA_VERSION,
        "bootstrap_engine_contract": BOOTSTRAP_ENGINE,
        "macro_class_support_policy": MACRO_CLASS_SUPPORT_POLICY,
        "full_clean_sha256": full_clean["sha256"],
        "full_stress_sha256": full_stress["sha256"],
        "comp_clean_sha256": comp_clean["sha256"],
        "comp_stress_sha256": comp_stress["sha256"],
        "oof_sha256": canonical_contract["oof_sha256"],
        "freeze_sha256": canonical_contract["freeze_sha256"],
        "group_contract_sha256": canonical_contract["group_contract_sha256"],
        "group_sidecar_sha256": canonical_contract["group_sidecar_sha256"],
        "bootstrap_contract_source_sha256": bootstrap_contract["source_sha256"],
    }


def validate_metric_cache_row(row: dict[str, Any], metadata: dict[str, Any]) -> None:
    if row.get("status") != "complete":
        raise ValueError("metric cache row status is not complete")
    if int(row.get("n_boot_valid", -1)) != int(metadata["n_boot"]):
        raise ValueError("metric cache does not contain the exact requested bootstrap count")
    if row.get("bootstrap_engine") != BOOTSTRAP_ENGINE:
        raise ValueError("metric cache bootstrap engine differs from the current exact engine")
    for field in (
        "degradation_adv_ci_low",
        "degradation_adv_ci_high",
        "stressed_adv_ci_low",
        "stressed_adv_ci_high",
    ):
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"metric cache lacks a numeric {field}") from exc
        if not math.isfinite(value):
            raise ValueError(f"metric cache contains a non-finite {field}")
    for key, value in row.items():
        if "significant" in str(key).lower() or "significant" in str(value).lower():
            raise ValueError("metric cache contains prohibited legacy significance wording")


def read_metric_cache(path: Path, metadata: dict[str, Any]) -> dict[str, Any] | None:
    path = resolve(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: could not read metric cache {path}: {exc}", flush=True)
        return None
    row = payload.get("row")
    if not isinstance(row, dict):
        return None
    observed_metadata = payload.get("metadata")
    if observed_metadata != metadata:
        return None
    try:
        validate_metric_cache_row(row, metadata)
    except (TypeError, ValueError) as exc:
        print(f"WARNING: rejecting invalid metric cache {path}: {exc}", flush=True)
        return None
    return dict(row)


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


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_group_sidecar(path_value: Any) -> Path:
    path = Path(str(path_value or ""))
    if not str(path):
        raise RuntimeError("Frozen OOF group sidecar path is missing")
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_canonical_contract(clean_sha256: str) -> dict[str, Any]:
    if not OOF_FREEZE_MANIFEST.exists() or OOF_FREEZE_MANIFEST.stat().st_size == 0:
        raise FileNotFoundError(f"Missing frozen OOF manifest: {OOF_FREEZE_MANIFEST}")
    payload = json.loads(OOF_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "frozen"
        or payload.get("manuscript_ready") is not True
        or payload.get("checkpoint_kind") != "final_ema"
    ):
        raise RuntimeError("Frozen OOF manifest is not the canonical final_ema contract")
    prediction_rows = [
        row
        for row in payload.get("artifacts") or []
        if str(row.get("path", "")).endswith("/oof_final_ema_predictions.npz")
    ]
    if len(prediction_rows) != 1 or prediction_rows[0].get("sha256") != clean_sha256:
        raise RuntimeError("Full clean prediction SHA does not match the frozen OOF manifest")
    group = payload.get("group_contract") or {}
    sidecar = group.get("sidecar") or {}
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
    if int(group.get("n_records", -1)) != int(payload.get("validated_records", -2)):
        group_errors.append("n_records")
    if int(group.get("n_groups", -1)) != int(payload.get("validated_records", -2)):
        group_errors.append("n_groups")
    try:
        sidecar_path = _resolve_group_sidecar(sidecar.get("path"))
    except RuntimeError:
        sidecar_path = Path()
        group_errors.append("sidecar_path")
    if not sidecar_path.is_file():
        group_errors.append("sidecar_missing")
    elif sha256_file(sidecar_path) != sidecar.get("sha256"):
        group_errors.append("sidecar_sha256")
    if group_errors:
        raise RuntimeError(
            "Frozen OOF manifest lacks an authenticated live patient/group contract: "
            + ", ".join(group_errors)
        )
    return {
        "oof_sha256": clean_sha256,
        "freeze_sha256": sha256_file(OOF_FREEZE_MANIFEST),
        "group_contract_sha256": _canonical_json_sha256(group),
        "group_sidecar": project_relative(sidecar_path),
        "group_sidecar_sha256": str(sidecar["sha256"]),
        "n_groups": int(group["n_groups"]),
    }


def load_bootstrap_independence_contract(canonical_contract: dict[str, str]) -> dict[str, Any]:
    """Authenticate the subject-level interpretation of a record bootstrap."""

    path = resolve(CALIBRATION_CI)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            "Missing calibration/bootstrap contract required for subject-level robustness CIs: "
            f"{path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    bootstrap = payload.get("bootstrap") or {}
    if bootstrap.get("unit") != AUTHENTICATED_RECORD_BOOTSTRAP_UNIT:
        raise RuntimeError("Calibration contract does not declare authenticated patient-record units")
    if bootstrap.get("independence_contract") != CHAPMAN_GROUP_SEMANTICS:
        raise RuntimeError("Calibration contract does not declare the reviewed patient-record semantics")
    if bootstrap.get("group_semantics_reference") != CHAPMAN_GROUP_REFERENCE:
        raise RuntimeError("Calibration contract is not bound to the reviewed PhysioNet source")
    if not bootstrap.get("group_sidecar") or not bootstrap.get("group_sidecar_sha256"):
        raise RuntimeError("Calibration contract does not authenticate its patient-record sidecar")
    if payload.get("predictions_sha256") != canonical_contract["oof_sha256"]:
        raise RuntimeError("Calibration bootstrap contract references a different canonical OOF artifact")
    if payload.get("freeze_manifest_sha256") != canonical_contract["freeze_sha256"]:
        raise RuntimeError("Calibration bootstrap contract references a different OOF freeze manifest")
    sidecar_path = _resolve_group_sidecar(bootstrap["group_sidecar"])
    if not sidecar_path.is_file():
        raise RuntimeError("Calibration bootstrap group sidecar is missing")
    if sha256_file(sidecar_path) != bootstrap["group_sidecar_sha256"]:
        raise RuntimeError("Calibration bootstrap group sidecar SHA256 is stale")
    if bootstrap["group_sidecar_sha256"] != canonical_contract["group_sidecar_sha256"]:
        raise RuntimeError("Calibration bootstrap group sidecar differs from the frozen OOF contract")
    if int(bootstrap.get("records", -1)) != int(canonical_contract["n_groups"]):
        raise RuntimeError("Calibration bootstrap record count differs from the frozen OOF group contract")
    if int(bootstrap.get("unique_groups", -1)) != int(canonical_contract["n_groups"]):
        raise RuntimeError("Calibration bootstrap group count differs from the frozen OOF group contract")
    return {
        "unit": BOOTSTRAP_UNIT,
        "independence_contract": CHAPMAN_GROUP_SEMANTICS,
        "group_semantics_reference": CHAPMAN_GROUP_REFERENCE,
        "group_sidecar": project_relative(sidecar_path),
        "group_sidecar_sha256": bootstrap["group_sidecar_sha256"],
        "source": project_relative(path),
        "source_sha256": sha256_file(path),
        "training_variability_scope": TRAINING_VARIABILITY_SCOPE,
    }


def load_validated_clean_artifact(comparator: str, path: Path) -> dict[str, Any]:
    """Validate a clean artifact completely before exposing it to aggregation."""

    candidate = load_npz(path)
    checkpoint_sha = load_clean_checkpoint_contract(comparator, candidate)
    if checkpoint_sha:
        candidate["checkpoint_sha256"] = np.asarray(checkpoint_sha)
    return candidate


def artifact_stress_spec(comparator: str, stress_data: dict[str, Any]) -> dict[str, Any]:
    if comparator in {"full", "minirocket"}:
        raw = str(scalar(stress_data, "stress_json", ""))
        if not raw:
            raise RuntimeError(f"{comparator} stress artifact lacks stress_json")
        payload = json.loads(raw)
    else:
        raw = str(scalar(stress_data, "stress_metadata_json", ""))
        if not raw:
            raise RuntimeError(f"{comparator} stress artifact lacks stress_metadata_json")
        payload = (json.loads(raw) or {}).get("spec")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{comparator} stress specification is malformed")
    return payload


def validate_stress_provenance(
    comparator: str,
    stress: str,
    clean: dict[str, Any],
    stress_data: dict[str, Any],
    expected_spec: dict[str, Any],
) -> None:
    observed_spec = artifact_stress_spec(comparator, stress_data)
    if json.dumps(observed_spec, sort_keys=True) != json.dumps(expected_spec, sort_keys=True):
        raise RuntimeError(
            f"{comparator}/{stress} perturbation specification mismatch: "
            f"{observed_spec!r} != {expected_spec!r}"
        )

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
        if str(scalar(stress_data, "protocol")) != COMPARATOR_STRESS_PROTOCOL:
            raise RuntimeError(f"{comparator}/{stress} has an unexpected stress protocol")
        source_bundle = current_comparator_stress_source_bundle()
        if str(scalar(stress_data, "source_bundle_sha256")) != source_bundle["sha256"]:
            raise RuntimeError(
                f"{comparator}/{stress} was produced by a stale comparator-stress source bundle"
            )
        if str(scalar(stress_data, "producer_runner_sha256")) != source_bundle["files"][
            "scripts/revision/23_generate_comparator_stress_predictions.py"
        ]:
            raise RuntimeError(f"{comparator}/{stress} producer runner SHA is stale")
        if str(scalar(stress_data, "comparator")) != comparator:
            raise RuntimeError(f"{comparator}/{stress} comparator tag mismatch")
        if str(scalar(stress_data, "stress_test")) != stress:
            raise RuntimeError(f"{comparator}/{stress} stress tag mismatch")
        for field in (
            "raw_cache_sha256",
            "oof_predictions_sha256",
            "freeze_manifest_sha256",
            "aggregation_implementation",
            "power_mean_q",
        ):
            expected_value = str(scalar(clean, field, ""))
            actual_value = str(scalar(stress_data, field, ""))
            if not expected_value or actual_value != expected_value:
                raise RuntimeError(
                    f"{comparator}/{stress} {field} differs from its clean baseline contract"
                )
        slice_count = np.asarray(stress_data.get("slice_count", []), dtype=np.int64)
        if slice_count.shape != (len(stress_data["y_true"]),) or np.any(slice_count <= 0):
            raise RuntimeError(f"{comparator}/{stress} has incomplete record slice coverage")
        return

    if str(scalar(stress_data, "protocol")) != ROCKET_STRESS_PROTOCOL:
        raise RuntimeError(f"{comparator}/{stress} has an unexpected stress protocol")
    metadata = json.loads(str(scalar(stress_data, "metadata_json", "{}")))
    source_bundle = metadata.get("source_bundle") or {}
    expected_bundle = current_rocket_stress_source_bundle()
    if source_bundle.get("sha256") != expected_bundle["sha256"]:
        raise RuntimeError(f"{comparator}/{stress} was produced by a stale robustness source bundle")
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


def load_validated_stress_artifact(
    comparator: str,
    stress: str,
    path: Path,
    clean: dict[str, Any],
    full_clean: dict[str, Any],
    expected_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate a stress artifact completely before exposing it to aggregation."""

    candidate = load_npz(path)
    validate_same_contract(full_clean, candidate, f"{comparator}/{stress}")
    validate_stress_provenance(comparator, stress, clean, candidate, expected_spec)
    return candidate


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
            "threshold": float(threshold),
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
            "n_bins": int(n_bins),
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


_RANK_CONTEXT_CACHE: dict[str, list[dict[str, np.ndarray]]] = {}
_ECE_CONTEXT_CACHE: dict[str, dict[str, Any]] = {}
_RESAMPLE_COUNT_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, ...]] = {}


def rank_context(data: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    """Cache score order/tie boundaries once for exact weighted PR/ROC bootstrap."""

    source_sha = str(data.get("sha256") or "")
    cached = _RANK_CONTEXT_CACHE.get(source_sha) if source_sha else data.get("_rank_context")
    if cached is not None:
        return cached

    y_true = np.asarray(data["y_true"])
    y_prob = np.asarray(data["y_prob"])
    contexts: list[dict[str, np.ndarray]] = []
    for class_idx in range(y_true.shape[1]):
        # Match sklearn's stable descending score order and collapse equal-score ties.
        order = np.argsort(y_prob[:, class_idx], kind="mergesort")[::-1]
        sorted_prob = y_prob[order, class_idx]
        boundaries = np.r_[np.where(np.diff(sorted_prob))[0], len(order) - 1]
        contexts.append(
            {
                "order": order.astype(np.int32, copy=False),
                "boundaries": boundaries.astype(np.int32, copy=False),
                "y_sorted": y_true[order, class_idx].astype(np.uint8, copy=False),
            }
        )
    if source_sha:
        _RANK_CONTEXT_CACHE[source_sha] = contexts
    else:
        data["_rank_context"] = contexts
    return contexts


def weighted_rank_metric(
    contexts: list[dict[str, np.ndarray]],
    counts: np.ndarray,
    *,
    metric: str,
) -> float:
    """Compute macro AP/AUC for an integer-weighted record resample without re-sorting."""

    scores: list[float] = []
    for context in contexts:
        weights = counts[context["order"]]
        positives = weights * context["y_sorted"]
        negatives = weights - positives
        tp = np.cumsum(positives, dtype=np.float64)[context["boundaries"]]
        fp = np.cumsum(negatives, dtype=np.float64)[context["boundaries"]]
        total_positive = float(tp[-1])
        total_negative = float(fp[-1])
        if total_positive <= 0.0 or total_negative <= 0.0:
            continue
        if metric == "roc_auc_macro":
            tpr = tp / total_positive
            fpr = fp / total_negative
            tpr = np.r_[0.0, tpr]
            fpr = np.r_[0.0, fpr]
            scores.append(float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) * 0.5)))
        elif metric == "pr_auc_macro":
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            recall = tp / total_positive
            scores.append(float(np.sum(np.diff(np.r_[0.0, recall]) * precision)))
        else:
            raise ValueError(f"Unsupported weighted rank metric: {metric}")
    return float(np.mean(scores)) if scores else math.nan


def ece_context(data: dict[str, Any], n_bins: int) -> dict[str, Any]:
    """Cache a sparse record-to-class-bin residual matrix for exact ECE."""

    source_sha = str(data.get("sha256") or "")
    cache_key = f"{source_sha}:bins={n_bins}" if source_sha else ""
    memory_cache = data.setdefault("_ece_contexts", {}) if not source_sha else None
    cached = _ECE_CONTEXT_CACHE.get(cache_key) if source_sha else memory_cache.get(n_bins)
    if cached is not None:
        return cached

    y_true = np.asarray(data["y_true"], dtype=np.float64)
    y_prob = np.asarray(data["y_prob"], dtype=np.float64)
    n_records, n_classes = y_true.shape
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.searchsorted(bins, y_prob, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    row_index = (np.arange(n_classes)[:, None] * n_bins + bin_index.T).ravel()
    column_index = np.broadcast_to(np.arange(n_records), (n_classes, n_records)).ravel()
    residual = (y_true - y_prob).T.ravel()
    matrix = sparse.csr_matrix(
        (residual, (row_index, column_index)),
        shape=(n_classes * n_bins, n_records),
    )
    cached = {"matrix": matrix, "n_classes": n_classes, "n_bins": n_bins}
    if source_sha:
        _ECE_CONTEXT_CACHE[cache_key] = cached
    else:
        memory_cache[n_bins] = cached
    return cached


def bootstrap_record_counts(n_records: int, n_boot: int, seed: int) -> tuple[np.ndarray, ...]:
    """Reuse the exact seeded record-resample counts across paired comparators."""

    cache_key = (int(n_records), int(n_boot), int(seed))
    cached = _RESAMPLE_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed)
    count_dtype = np.uint16 if n_records <= np.iinfo(np.uint16).max else np.uint32
    counts = tuple(
        np.bincount(
            rng.integers(0, n_records, size=n_records),
            minlength=n_records,
        ).astype(count_dtype, copy=False)
        for _ in range(n_boot)
    )
    _RESAMPLE_COUNT_CACHE[cache_key] = counts
    return counts


def weighted_resample_metric(
    spec: dict[str, Any],
    data: dict[str, Any],
    counts: np.ndarray,
    *,
    rank_cache: list[dict[str, np.ndarray]] | None = None,
    ece_cache: dict[str, Any] | None = None,
) -> float:
    """Exact metric for the same integer record resample represented by counts."""

    metric = str(spec["name"])
    if metric in {"pr_auc_macro", "roc_auc_macro"}:
        return weighted_rank_metric(
            rank_cache if rank_cache is not None else rank_context(data),
            counts,
            metric=metric,
        )

    y_true_raw = np.asarray(data["y_true"])
    y_prob_raw = np.asarray(data["y_prob"])
    y_true = y_true_raw.astype(np.float64, copy=False)
    y_prob = y_prob_raw.astype(np.float64, copy=False)
    weights = np.asarray(counts, dtype=np.float64)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return math.nan

    if metric == "f1_macro":
        predicted = y_prob >= float(spec["threshold"])
        positive = y_true == 1.0
        expanded = weights[:, None]
        tp = np.sum(expanded * (positive & predicted), axis=0, dtype=np.float64)
        fp = np.sum(expanded * (~positive & predicted), axis=0, dtype=np.float64)
        fn = np.sum(expanded * (positive & ~predicted), axis=0, dtype=np.float64)
        denominator = 2.0 * tp + fp + fn
        per_class = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
        return float(np.mean(per_class))

    positive_weight = weights @ y_true
    valid_classes = (positive_weight > 0.0) & (positive_weight < total_weight)
    if not np.any(valid_classes):
        return math.nan

    if metric == "brier_macro":
        # Preserve the original prediction dtype for parity with sklearn's
        # brier_score_loss on the explicitly repeated bootstrap sample.
        squared_error = np.square(y_true_raw - y_prob_raw)
        per_class = (weights @ squared_error) / total_weight
        return float(np.mean(per_class[valid_classes]))

    if metric == "ece_macro":
        n_bins = int(spec["n_bins"])
        context = ece_cache if ece_cache is not None else ece_context(data, n_bins)
        weighted_residual = np.asarray(context["matrix"] @ weights).reshape(
            int(context["n_classes"]),
            n_bins,
        )
        per_class = np.sum(np.abs(weighted_residual), axis=1) / total_weight
        return float(np.mean(per_class[valid_classes]))

    raise ValueError(f"Unsupported weighted resample metric: {metric}")


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

    optimized_metric = str(spec.get("name")) in {
        "pr_auc_macro",
        "roc_auc_macro",
        "f1_macro",
        "brier_macro",
        "ece_macro",
    }
    cached_record_counts = bootstrap_record_counts(n, n_boot, seed) if optimized_metric else None
    rng = np.random.default_rng(seed) if not optimized_metric else None
    rank_caches = (
        {
            "full_clean": rank_context(full_clean),
            "full_stress": rank_context(full_stress),
            "comp_clean": rank_context(comp_clean),
            "comp_stress": rank_context(comp_stress),
        }
        if str(spec.get("name")) in {"pr_auc_macro", "roc_auc_macro"}
        else {}
    )
    ece_caches = (
        {
            "full_clean": ece_context(full_clean, int(spec["n_bins"])),
            "full_stress": ece_context(full_stress, int(spec["n_bins"])),
            "comp_clean": ece_context(comp_clean, int(spec["n_bins"])),
            "comp_stress": ece_context(comp_stress, int(spec["n_bins"])),
        }
        if str(spec.get("name")) == "ece_macro"
        else {}
    )

    def evaluate(
        item: tuple[int, np.ndarray | None, np.ndarray],
    ) -> tuple[tuple[float, float], tuple[float, float] | None]:
        ordinal, idx, counts = item
        if cached_full_values is None:
            if optimized_metric:
                fc = weighted_resample_metric(
                    spec,
                    full_clean,
                    counts,
                    rank_cache=rank_caches.get("full_clean"),
                    ece_cache=ece_caches.get("full_clean"),
                )
                fs = weighted_resample_metric(
                    spec,
                    full_stress,
                    counts,
                    rank_cache=rank_caches.get("full_stress"),
                    ece_cache=ece_caches.get("full_stress"),
                )
            else:
                assert idx is not None
                fc = metric_value(spec, full_clean, idx)
                fs = metric_value(spec, full_stress, idx)
        else:
            fc, fs = cached_full_values[ordinal]
        if optimized_metric:
            cc = weighted_resample_metric(
                spec,
                comp_clean,
                counts,
                rank_cache=rank_caches.get("comp_clean"),
                ece_cache=ece_caches.get("comp_clean"),
            )
            cs = weighted_resample_metric(
                spec,
                comp_stress,
                counts,
                rank_cache=rank_caches.get("comp_stress"),
                ece_cache=ece_caches.get("comp_stress"),
            )
        else:
            assert idx is not None
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
            if optimized_metric:
                assert cached_record_counts is not None
                indexed = [
                    (sample_ordinal, None, cached_record_counts[sample_ordinal])
                    for sample_ordinal in range(ordinal, ordinal + count)
                ]
            else:
                assert rng is not None
                indices = [rng.integers(0, n, size=n) for _ in range(count)]
                indexed = [
                    (sample_ordinal, idx, np.bincount(idx, minlength=n).astype(np.float64, copy=False))
                    for sample_ordinal, idx in enumerate(indices, start=ordinal)
                ]
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
        "bootstrap_engine": BOOTSTRAP_ENGINE if optimized_metric else "record_index_resample_v1",
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
        return "full_nominal_95ci_more_favorable_change"
    if ci_high < 0:
        return "comparator_nominal_95ci_more_favorable_change"
    return "nominal_95ci_inconclusive_change_difference"


def row_interpretation(row: dict[str, Any]) -> str:
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
    print(f"bootstrap_engine={BOOTSTRAP_ENGINE}", flush=True)
    print(f"metric_cache_dir={resolve(args.metric_cache_dir)} reuse={args.reuse_metric_cache}", flush=True)

    clean: dict[str, dict[str, Any]] = {}
    artifact_status: list[dict[str, Any]] = []
    for comp in comparators:
        path = PREDICTION_DIR / COMPARATORS[comp]["clean"]
        try:
            candidate_clean = load_validated_clean_artifact(comp, path)
            clean[comp] = candidate_clean
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
    bootstrap_independence_contract = load_bootstrap_independence_contract(canonical_contract)
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
    expected_stress_specs = {
        stress: expected_stress_spec(stress, int(args.seed)) for stress in stresses
    }
    rows: list[dict[str, Any]] = []
    pairwise: dict[str, Any] = {
        "status": "complete_with_possible_missing_comparators",
        "protocol": PROTOCOL,
        "created_utc": now_utc(),
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "bootstrap_jobs": args.bootstrap_jobs,
        "bootstrap_engine": BOOTSTRAP_ENGINE,
        "metric_cache_schema_version": METRIC_CACHE_SCHEMA_VERSION,
        "macro_class_support_policy": MACRO_CLASS_SUPPORT_POLICY,
        "bootstrap_unit": BOOTSTRAP_UNIT,
        "bootstrap_independence_contract": bootstrap_independence_contract,
        "training_variability_scope": TRAINING_VARIABILITY_SCOPE,
        "ci_scope": CI_SCOPE,
        "endpoint_definition": (
            "difference_in_signed_clean_to_stress_benefit_change_full_minus_comparator"
        ),
        "metrics": requested_metrics,
        "metric_cache_dir": project_relative(args.metric_cache_dir),
        "output_profile": output_profile_name(args.out_pairwise),
        "comparators": comparators,
        "stress_tests": stresses,
        "stress_contracts": {
            name: stress_contract_description(spec)
            for name, spec in expected_stress_specs.items()
        },
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
                candidate_stress = load_validated_stress_artifact(
                    comp,
                    stress,
                    stress_path,
                    clean[comp],
                    full_clean,
                    expected_stress_specs[stress],
                )
                stress_data[comp] = candidate_stress
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
                invalid_exists = stress_path.exists() and stress_path.stat().st_size > 0
                artifact_status.append(
                    {
                        "comparator": comp,
                        "kind": f"stress:{stress}",
                        "path": project_relative(stress_path),
                        "exists": invalid_exists,
                        "sha256": sha256_file(stress_path) if invalid_exists else "",
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
                    "bootstrap_unit": BOOTSTRAP_UNIT,
                    "training_variability_scope": TRAINING_VARIABILITY_SCOPE,
                    "ci_scope": CI_SCOPE,
                    "macro_class_support_policy": MACRO_CLASS_SUPPORT_POLICY,
                    "perturbation_realization_scope": "single_fixed_seed_conditional_stress_audit",
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
                    canonical_contract=canonical_contract,
                    bootstrap_contract=bootstrap_independence_contract,
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
                        "bootstrap_engine": boot.get("bootstrap_engine"),
                        "interpretation": interp,
                    }
                    validate_metric_cache_row(row, metadata)
                    write_metric_cache(cache_path, metadata, row)
                    print(f"{stress} {comp} {spec['name']}: bootstrap done", flush=True)
                validate_metric_cache_row(row, metadata)
                interp = row_interpretation(row)
                row.update(
                    {
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
                        "interpretation": interp,
                    }
                )
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
        "Use only named stress-, metric-, and comparator-specific signed change differences. "
        "The paired 95% percentile CIs are nominal and unadjusted across the full comparison family; "
        "stochastic stresses use one fixed seeded realization, and trained folds/checkpoints are held fixed. "
        "Do not claim broad robustness superiority or training-run uncertainty."
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
        "bootstrap_engine": BOOTSTRAP_ENGINE,
        "metric_cache_schema_version": METRIC_CACHE_SCHEMA_VERSION,
        "macro_class_support_policy": MACRO_CLASS_SUPPORT_POLICY,
        "bootstrap_unit": BOOTSTRAP_UNIT,
        "bootstrap_independence_contract": bootstrap_independence_contract,
        "training_variability_scope": TRAINING_VARIABILITY_SCOPE,
        "ci_scope": CI_SCOPE,
        "endpoint_definition": (
            "difference_in_signed_clean_to_stress_benefit_change_full_minus_comparator"
        ),
        "metrics": requested_metrics,
        "stress_contracts": {
            name: stress_contract_description(spec)
            for name, spec in expected_stress_specs.items()
        },
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
            "bootstrap_engine": BOOTSTRAP_ENGINE,
            "metric_cache_schema_version": METRIC_CACHE_SCHEMA_VERSION,
            "macro_class_support_policy": MACRO_CLASS_SUPPORT_POLICY,
            "bootstrap_unit": BOOTSTRAP_UNIT,
            "bootstrap_independence_contract": bootstrap_independence_contract,
            "training_variability_scope": TRAINING_VARIABILITY_SCOPE,
            "ci_scope": CI_SCOPE,
            "endpoint_definition": (
                "difference_in_signed_clean_to_stress_benefit_change_full_minus_comparator"
            ),
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
                "Use only this named stress/metric/comparator signed change difference with its "
                "nominal unadjusted paired 95% record-bootstrap CI, conditional on the fixed trained folds."
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
