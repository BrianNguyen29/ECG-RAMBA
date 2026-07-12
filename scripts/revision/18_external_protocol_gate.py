"""Validate external PTB-XL, Georgia, and CPSC2021 artifacts before claims.

This gate intentionally does not run model inference. It reads the experimental
external outputs produced by ``03_generate_external_predictions.py`` and checks
whether a dataset-specific artifact set is complete enough to be cited as
protocol-gated external evidence. Passing this gate does not support unqualified
external-transfer advantage, benchmark-leading external performance, or clinical
deployment claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG, PATHS  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PTB_SUPERCLASS_MAPPING,
    TABLE_DIR,
    calibration_summary,
    cluster_bootstrap_ci,
    ensure_revision_dirs,
    git_commit,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_csv,
    save_json,
    sha256_file,
)


DATASETS = ("ptbxl", "georgia", "cpsc2021")
GATE_SCHEMA_VERSION = 4
METRIC_IMPLEMENTATION_PATH = PROJECT_ROOT / "scripts" / "revision" / "common.py"
EXPECTED_EXTERNAL_PROTOCOLS = {
    "ptbxl": "official_ptbxl_diagnostic_superclass_any_positive_likelihood",
    "georgia": "chapman_27_class_snomed_intersection",
    "cpsc2021": "annotation_aligned_nonoverlapping_10s_windows_majority_af_or_normal",
}
EXPECTED_CLASS_NAMES = {
    "ptbxl": tuple(PTB_SUPERCLASS_MAPPING.keys()),
    "georgia": tuple(CLASSES),
    "cpsc2021": ("AF_or_AFL",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        action="append",
        default=None,
        help="Dataset(s) to validate. Defaults to all.",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=EXPERIMENTAL_DIR / "external",
        help="Root containing external/<dataset> experimental artifacts.",
    )
    parser.add_argument(
        "--oof-run-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_prediction_run_manifest.json",
    )
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse gate outputs when the source artifact checksums and gate parameters match.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero if any requested dataset does not pass the gate.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=METRIC_DIR / "external_protocol_gate_summary.csv",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_payload_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def archive_path(dataset: str) -> Path:
    key = {
        "ptbxl": "ptb_zip",
        "georgia": "georgia_zip",
        "cpsc2021": "cpsc_zip",
    }[dataset]
    return Path(PATHS[key])


def as_scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def json_scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    value = as_scalar(data, key, default=None)
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_weight_kind(checkpoint_kind: str) -> str:
    if checkpoint_kind.endswith("_ema"):
        return "ema"
    if checkpoint_kind.endswith("_raw"):
        return "raw"
    return ""


def artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": project_relative(path),
            "exists": False,
            "size_bytes": 0,
            "sha256": "",
        }
    return {
        "path": project_relative(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def gate_cache_key(
    dataset: str,
    args: argparse.Namespace,
    required_paths: dict[str, Path],
) -> str:
    payload = {
        "dataset": dataset,
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "expected_checkpoint_kind": args.expected_checkpoint_kind,
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "metric_implementation_sha256": sha256_file(METRIC_IMPLEMENTATION_PATH),
        "external_root": project_relative(args.external_root),
        "oof_run_manifest": artifact(args.oof_run_manifest),
        "source_artifacts": {name: artifact(path) for name, path in sorted(required_paths.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_oof_checkpoint_contract(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    records = payload.get("checkpoints") or (payload.get("inputs") or {}).get("checkpoints") or []
    rows: dict[int, dict[str, Any]] = {}
    if isinstance(records, dict):
        records = records.values()
    for row in records:
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError):
            continue
        rows[fold] = row
    return rows


def metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
    n_bins: int,
) -> dict[str, dict[str, float | int]]:
    if n_boot <= 0:
        return {}

    def f1_macro(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(multilabel_metrics(yt, yp, threshold=threshold)["f1_macro"])

    def brier_macro(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(calibration_summary(yt, yp, n_bins=n_bins)["brier_macro"])

    def ece_macro(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(calibration_summary(yt, yp, n_bins=n_bins)["ece_macro"])

    return {
        "macro_pr_auc": cluster_bootstrap_ci(y_true, y_prob, groups, macro_pr_auc, n_boot=n_boot, seed=seed),
        "macro_roc_auc": cluster_bootstrap_ci(y_true, y_prob, groups, macro_roc_auc, n_boot=n_boot, seed=seed + 1),
        "f1_macro": cluster_bootstrap_ci(y_true, y_prob, groups, f1_macro, n_boot=n_boot, seed=seed + 2),
        "brier_macro": cluster_bootstrap_ci(y_true, y_prob, groups, brier_macro, n_boot=n_boot, seed=seed + 3),
        "ece_macro": cluster_bootstrap_ci(y_true, y_prob, groups, ece_macro, n_boot=n_boot, seed=seed + 4),
    }


def class_rows(dataset: str, y_true: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for idx, name in enumerate(class_names):
        row: dict[str, Any] = {
            "dataset": dataset,
            "class_index": idx,
            "class_name": name,
            "n_records": int(y_true.shape[0]),
            "n_positive": int(np.sum(y_true[:, idx])),
            "prevalence": float(np.mean(y_true[:, idx])),
        }
        if dataset == "ptbxl":
            spec = PTB_SUPERCLASS_MAPPING[name]
            row["mapped_chapman_proxy_codes"] = ";".join(spec["codes"])
            row["mapping_scope"] = "ptbxl_diagnostic_superclass_to_chapman_proxy"
        elif dataset == "cpsc2021":
            row["mapped_chapman_proxy_codes"] = "AF;AFL"
            row["mapping_scope"] = "annotation_window_af_or_afl_to_chapman_af_afl_proxy"
        else:
            row["mapped_chapman_proxy_codes"] = name
            row["mapping_scope"] = "georgia_snomed_to_chapman_27_intersection"
        rows.append(row)
    return rows


def validate_dataset(
    dataset: str,
    args: argparse.Namespace,
    oof_checkpoints: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = args.external_root / dataset
    prediction_path = output_root / f"{dataset}_full_predictions.npz"
    slice_path = output_root / f"{dataset}_full_slice_predictions.npz"
    summary_path = output_root / f"{dataset}_full_prediction_summary.json"
    class_path = output_root / f"{dataset}_full_class_summary.csv"
    manifest_path = output_root / f"{dataset}_full_prediction_run_manifest.json"

    required_paths = {
        "prediction": prediction_path,
        "slice_prediction": slice_path,
        "summary": summary_path,
        "class_summary": class_path,
        "manifest": manifest_path,
    }
    missing = [name for name, path in required_paths.items() if not path.exists() or path.stat().st_size == 0]
    issues: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    created_utc = now_utc()
    gate_path = METRIC_DIR / f"external_{dataset}_protocol_gate.json"
    label_table = TABLE_DIR / f"table_external_{dataset}_label_mapping.csv"
    metrics_table = TABLE_DIR / f"table_external_{dataset}_metrics.csv"
    gate_manifest = MANIFEST_DIR / f"external_{dataset}_protocol_gate_manifest.json"

    if missing:
        issues.append("missing_or_empty_artifacts=" + ",".join(missing))
        payload = {
            "dataset": dataset,
            "created_utc": created_utc,
            "git_commit": git_commit(),
            "gate_schema_version": GATE_SCHEMA_VERSION,
            "gate_cache_key": "",
            "reused_existing": False,
            "status": "blocked_missing_external_artifacts",
            "protocol_gate_passed": False,
            "manuscript_ready": False,
            "issues": issues,
            "warnings": warnings,
            "artifacts": {
                **{name: artifact(path) for name, path in required_paths.items()},
                "gate_json": artifact(gate_path),
                "label_table": artifact(label_table),
                "metrics_table": artifact(metrics_table),
                "gate_manifest": artifact(gate_manifest),
            },
            "safe_claim": "External outputs remain experimental and are not manuscript-ready.",
            "safe_wording": "External outputs remain experimental and are not manuscript-ready.",
        }
        save_csv(label_table, [])
        save_csv(metrics_table, [])
        save_json(gate_path, payload)
        payload["artifacts"]["gate_json"] = artifact(gate_path)
        payload["artifacts"]["label_table"] = artifact(label_table)
        payload["artifacts"]["metrics_table"] = artifact(metrics_table)
        save_json(
            gate_manifest,
            {
                "dataset": dataset,
                "created_utc": created_utc,
                "status": payload["status"],
                "protocol_gate_passed": False,
                "artifacts": payload["artifacts"],
                "issues": issues,
                "warnings": warnings,
            },
        )
        payload["artifacts"]["gate_manifest"] = artifact(gate_manifest)
        save_json(gate_path, payload)
        return payload, [], []

    cache_key = gate_cache_key(dataset, args, required_paths)
    if args.reuse_existing and gate_path.exists():
        try:
            cached = read_json(gate_path)
        except (OSError, json.JSONDecodeError):
            cached = {}
        cache_artifacts = {
            "gate_json": gate_path,
            "label_table": label_table,
            "metrics_table": metrics_table,
            "gate_manifest": gate_manifest,
        }
        cached_artifacts = cached.get("artifacts", {})
        supplemental_keys = []
        if dataset == "georgia":
            supplemental_keys.append("georgia_mapping_inventory")
        elif dataset == "cpsc2021":
            supplemental_keys.append("cpsc_annotation_audit")
        for key in supplemental_keys:
            cached_path = resolve_payload_path((cached_artifacts.get(key) or {}).get("path"))
            if cached_path is not None:
                cache_artifacts[key] = cached_path
        cache_outputs_exist = all(path.exists() and path.stat().st_size > 0 for path in cache_artifacts.values())
        cache_outputs_exist = cache_outputs_exist and all(key in cache_artifacts for key in supplemental_keys)
        cache_outputs_match = True
        for name, path in cache_artifacts.items():
            if name == "gate_json":
                continue
            expected_sha = (cached_artifacts.get(name) or {}).get("sha256")
            if not expected_sha or not path.exists() or sha256_file(path) != expected_sha:
                cache_outputs_match = False
                break
        if (
            cached.get("gate_schema_version") == GATE_SCHEMA_VERSION
            and cached.get("gate_cache_key") == cache_key
            and cache_outputs_exist
            and cache_outputs_match
        ):
            cached["reused_existing"] = True
            cached["reused_utc"] = created_utc
            print(f"Reusing cached external gate for {dataset}: {gate_path}")
            return cached, [], []

    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    with np.load(prediction_path, allow_pickle=False) as data:
        prediction_keys = set(data.files)
        y_true = np.asarray(data["y_true"], dtype=np.float32)
        y_prob = np.asarray(data["y_prob"], dtype=np.float32)
        record_id = np.asarray(data["record_id"])
        group_id = (
            np.asarray(data["group_id"]).astype(str)
            if "group_id" in prediction_keys
            else np.asarray([], dtype=str)
        )
        split_id = (
            np.asarray(data["split_id"]).astype(str)
            if "split_id" in prediction_keys
            else np.asarray([], dtype=str)
        )
        group_unit = str(as_scalar(data, "group_unit", ""))
        class_names = [str(x) for x in np.asarray(data["class_names"]).tolist()]
        npz_dataset = str(as_scalar(data, "dataset", ""))
        evidence_status = str(as_scalar(data, "evidence_status", ""))
        manuscript_ready = bool(as_scalar(data, "manuscript_ready", False))
        aggregation_method = str(as_scalar(data, "aggregation_method", ""))
        aggregation_q = float(as_scalar(data, "aggregation_q", math.nan))
        cache_schema = int(as_scalar(data, "cache_schema_version", 0))
        checkpoint_fingerprints = json_scalar(data, "checkpoint_fingerprints_json", [])
        pca_fingerprints = json_scalar(data, "pca_fingerprints_json", [])

    with np.load(slice_path, allow_pickle=False) as slice_data:
        slice_keys = set(slice_data.files)
        required_slice_keys = {
            "slice_prob",
            "record_index",
            "record_id",
            "group_id",
            "split_id",
            "slice_index",
            "class_names",
            "dataset",
            "cache_schema_version",
            "evidence_status",
            "manuscript_ready",
        }
        slice_required_keys_present = required_slice_keys.issubset(slice_keys)
        if slice_required_keys_present:
            slice_prob = np.asarray(slice_data["slice_prob"], dtype=np.float32)
            slice_record_index = np.asarray(slice_data["record_index"])
            slice_record_id = np.asarray(slice_data["record_id"])
            slice_group_id = np.asarray(slice_data["group_id"]).astype(str)
            slice_split_id = np.asarray(slice_data["split_id"]).astype(str)
            slice_class_names = [str(x) for x in np.asarray(slice_data["class_names"]).tolist()]
            slice_dataset = str(as_scalar(slice_data, "dataset", ""))
            slice_cache_schema = int(as_scalar(slice_data, "cache_schema_version", 0))
            slice_evidence_status = str(as_scalar(slice_data, "evidence_status", ""))
            slice_manuscript_ready = bool(as_scalar(slice_data, "manuscript_ready", False))
        else:
            slice_prob = np.empty((0, 0), dtype=np.float32)
            slice_record_index = np.asarray([], dtype=np.int64)
            slice_record_id = np.asarray([])
            slice_group_id = np.asarray([], dtype=str)
            slice_split_id = np.asarray([], dtype=str)
            slice_class_names = []
            slice_dataset = ""
            slice_cache_schema = 0
            slice_evidence_status = ""
            slice_manuscript_ready = True

    checks["dataset_matches"] = npz_dataset == dataset and summary.get("dataset") == dataset
    checks["shape_matches"] = y_true.shape == y_prob.shape and y_true.ndim == 2
    checks["record_id_count_matches"] = len(record_id) == y_true.shape[0]
    checks["group_id_present"] = "group_id" in prediction_keys and len(group_id) == y_true.shape[0]
    checks["split_id_present"] = "split_id" in prediction_keys and len(split_id) == y_true.shape[0]
    checks["group_ids_nonempty"] = bool(
        checks["group_id_present"] and np.all(np.char.str_len(group_id.astype(str)) > 0)
    )
    checks["split_ids_nonempty"] = bool(
        checks["split_id_present"] and np.all(np.char.str_len(split_id.astype(str)) > 0)
    )
    checks["group_unit_declared"] = bool(group_unit.strip())
    checks["at_least_two_independent_groups"] = bool(
        checks["group_id_present"] and len(np.unique(group_id)) >= 2
    )
    checks["finite_probabilities"] = bool(np.isfinite(y_prob).all())
    checks["probabilities_in_unit_interval"] = bool(np.min(y_prob) >= 0.0 and np.max(y_prob) <= 1.0)
    checks["finite_labels"] = bool(np.isfinite(y_true).all())
    checks["binary_labels"] = bool(np.all((y_true == 0.0) | (y_true == 1.0)))
    checks["class_names_match_expected"] = tuple(class_names) == EXPECTED_CLASS_NAMES[dataset]
    checks["source_marked_experimental"] = evidence_status == "experimental" and manuscript_ready is False
    checks["summary_marked_experimental"] = (
        summary.get("evidence_status") == "experimental" and summary.get("manuscript_ready") is False
    )
    checks["manifest_marked_experimental"] = (
        manifest.get("evidence_status") == "experimental" and manifest.get("manuscript_ready") is False
    )
    checks["aggregation_power_mean_q"] = aggregation_method == "power_mean" and math.isclose(
        aggregation_q,
        float(CONFIG["power_mean_q"]),
        rel_tol=1e-6,
    )
    checks["cache_schema_current"] = cache_schema == CACHE_SCHEMA_VERSION
    checks["oof_run_manifest_exists"] = bool(args.oof_run_manifest.exists() and args.oof_run_manifest.stat().st_size > 0)
    checks["oof_checkpoint_contract_count_5"] = len(oof_checkpoints) == int(CONFIG["n_folds"])
    checks["slice_required_keys_present"] = slice_required_keys_present
    checks["slice_count_positive"] = bool(slice_prob.shape[0] > 0)
    checks["slice_shape_matches_classes"] = (
        slice_prob.ndim == 2 and y_prob.ndim == 2 and slice_prob.shape[1] == y_prob.shape[1]
    )
    checks["slice_finite_probabilities"] = bool(slice_prob.size > 0 and np.isfinite(slice_prob).all())
    checks["slice_probabilities_in_unit_interval"] = bool(
        slice_prob.size > 0 and np.min(slice_prob) >= 0.0 and np.max(slice_prob) <= 1.0
    )
    checks["slice_record_index_count_matches"] = bool(len(slice_record_index) == slice_prob.shape[0])
    if checks["slice_record_index_count_matches"] and len(slice_record_index) > 0:
        try:
            slice_record_index_int = slice_record_index.astype(np.int64, copy=False)
            slice_record_index_valid = bool(
                np.min(slice_record_index_int) >= 0 and np.max(slice_record_index_int) < len(record_id)
            )
        except (TypeError, ValueError):
            slice_record_index_int = np.asarray([], dtype=np.int64)
            slice_record_index_valid = False
    else:
        slice_record_index_int = np.asarray([], dtype=np.int64)
        slice_record_index_valid = False
    checks["slice_record_index_valid"] = slice_record_index_valid
    checks["slice_record_id_count_matches"] = bool(len(slice_record_id) == slice_prob.shape[0])
    checks["slice_record_id_matches_index"] = bool(
        slice_record_index_valid
        and len(slice_record_id) == len(slice_record_index_int)
        and np.array_equal(
            np.asarray(slice_record_id, dtype=str),
            np.asarray(record_id[slice_record_index_int], dtype=str),
        )
    )
    checks["slice_group_id_matches_index"] = bool(
        slice_record_index_valid
        and checks["group_id_present"]
        and len(slice_group_id) == len(slice_record_index_int)
        and np.array_equal(slice_group_id, group_id[slice_record_index_int])
    )
    checks["slice_split_id_matches_index"] = bool(
        slice_record_index_valid
        and checks["split_id_present"]
        and len(slice_split_id) == len(slice_record_index_int)
        and np.array_equal(slice_split_id, split_id[slice_record_index_int])
    )
    checks["slice_class_names_match"] = tuple(slice_class_names) == tuple(class_names)
    checks["slice_dataset_matches"] = slice_dataset == dataset
    checks["slice_cache_schema_current"] = slice_cache_schema == CACHE_SCHEMA_VERSION
    checks["slice_source_marked_experimental"] = (
        slice_evidence_status == "experimental" and slice_manuscript_ready is False
    )

    if not checks["dataset_matches"]:
        issues.append("dataset metadata mismatch")
    if not checks["shape_matches"]:
        issues.append(f"label/prediction shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if not checks["record_id_count_matches"]:
        issues.append("record_id count does not match y_true rows")
    if not checks["group_id_present"] or not checks["group_ids_nonempty"]:
        issues.append("group_id is missing, empty, or does not match prediction rows")
    if not checks["split_id_present"] or not checks["split_ids_nonempty"]:
        issues.append("split_id is missing, empty, or does not match prediction rows")
    if not checks["group_unit_declared"]:
        issues.append("group_unit metadata is missing")
    if not checks["at_least_two_independent_groups"]:
        issues.append("external evaluation requires at least two independent groups")
    if not checks["finite_probabilities"] or not checks["probabilities_in_unit_interval"]:
        issues.append("invalid prediction probabilities")
    if not checks["finite_labels"] or not checks["binary_labels"]:
        issues.append("labels must be finite binary indicators")
    if not checks["class_names_match_expected"]:
        issues.append(f"class_names mismatch: {class_names} != {EXPECTED_CLASS_NAMES[dataset]}")
    if not checks["source_marked_experimental"]:
        issues.append("source external artifacts must remain evidence_status=experimental/manuscript_ready=false")
    if not checks["summary_marked_experimental"]:
        issues.append("summary artifact must remain evidence_status=experimental/manuscript_ready=false")
    if not checks["manifest_marked_experimental"]:
        issues.append("manifest artifact must remain evidence_status=experimental/manuscript_ready=false")
    if not checks["aggregation_power_mean_q"]:
        issues.append("external predictions must use shared Power Mean Q protocol")
    if not checks["cache_schema_current"]:
        issues.append("external prediction cache schema is stale")
    if not checks["oof_run_manifest_exists"]:
        issues.append(f"frozen OOF run manifest missing or empty: {args.oof_run_manifest}")
    if not checks["oof_checkpoint_contract_count_5"]:
        issues.append("frozen OOF checkpoint contract must include all five folds")
    if not checks["slice_required_keys_present"]:
        missing_slice = sorted(required_slice_keys - slice_keys)
        issues.append("slice prediction missing required keys=" + ",".join(missing_slice))
    if not checks["slice_count_positive"]:
        issues.append("slice prediction artifact has no slices")
    if not checks["slice_shape_matches_classes"]:
        issues.append("slice prediction class dimension does not match record predictions")
    if not checks["slice_finite_probabilities"] or not checks["slice_probabilities_in_unit_interval"]:
        issues.append("invalid slice probabilities")
    if not checks["slice_record_index_count_matches"] or not checks["slice_record_index_valid"]:
        issues.append("slice record_index is missing or out of bounds")
    if not checks["slice_record_id_count_matches"] or not checks["slice_record_id_matches_index"]:
        issues.append("slice record_id does not match record_index")
    if not checks["slice_group_id_matches_index"]:
        issues.append("slice group_id does not match record_index")
    if not checks["slice_split_id_matches_index"]:
        issues.append("slice split_id does not match record_index")
    if not checks["slice_class_names_match"]:
        issues.append("slice class_names do not match record predictions")
    if not checks["slice_dataset_matches"]:
        issues.append("slice dataset metadata mismatch")
    if not checks["slice_cache_schema_current"]:
        issues.append("slice prediction cache schema is stale")
    if not checks["slice_source_marked_experimental"]:
        issues.append("slice artifact must remain evidence_status=experimental/manuscript_ready=false")

    load_summary = summary.get("load_summary", {})
    expected_label_protocol = EXPECTED_EXTERNAL_PROTOCOLS[dataset]
    checks["label_protocol_expected"] = load_summary.get("label_protocol") == expected_label_protocol
    if not checks["label_protocol_expected"]:
        issues.append(
            f"label_protocol mismatch: {load_summary.get('label_protocol')} != {expected_label_protocol}"
        )
    checks["summary_record_count_matches"] = int(summary.get("n_records", -1)) == int(y_true.shape[0])
    checks["summary_class_count_matches"] = int(summary.get("n_classes", -1)) == int(y_true.shape[1])
    if not checks["summary_record_count_matches"]:
        issues.append("summary n_records does not match predictions")
    if not checks["summary_class_count_matches"]:
        issues.append("summary n_classes does not match predictions")

    if dataset == "ptbxl":
        unsupported = load_summary.get("unsupported_superclasses", {})
        records_without = int(load_summary.get("records_without_supported_superclass", -1))
        checks["ptb_reports_unsupported_hyp"] = isinstance(unsupported, dict)
        checks["ptb_reports_records_without_supported_superclass"] = records_without >= 0
        checks["ptb_official_test_fold_only"] = bool(
            checks["split_id_present"] and set(split_id.tolist()) == {"ptbxl_fold10"}
        )
        checks["ptb_group_unit_patient"] = group_unit == "patient_id"
        if not checks["ptb_reports_unsupported_hyp"]:
            issues.append("PTB gate requires reported unsupported superclass counts")
        if not checks["ptb_reports_records_without_supported_superclass"]:
            issues.append("PTB gate requires records_without_supported_superclass")
        if not checks["ptb_official_test_fold_only"]:
            issues.append("Primary PTB-XL gate requires official strat_fold=10 test records only")
        if not checks["ptb_group_unit_patient"]:
            issues.append("PTB-XL gate requires patient_id grouping")
        warnings.append("PTB predictions are Chapman proxy superclasses; HYP remains unsupported.")
    elif dataset == "georgia":
        skipped = load_summary.get("skipped_records_without_mapped_label")
        checks["georgia_reports_unmapped_skips"] = isinstance(skipped, int) and skipped >= 0
        inventory_path = resolve_payload_path(load_summary.get("mapping_inventory_csv"))
        checks["georgia_mapping_review_exists"] = bool(load_summary.get("mapping_review_file_exists"))
        checks["georgia_review_has_mapped_codes"] = int(load_summary.get("mapping_review_mapped_codes", 0)) > 0
        checks["georgia_reports_mapping_inventory"] = bool(
            inventory_path is not None and inventory_path.exists() and inventory_path.stat().st_size > 0
        )
        checks["georgia_group_unit_declared"] = group_unit == "record_id_assumed_independent"
        if not checks["georgia_reports_unmapped_skips"]:
            issues.append("Georgia gate requires skipped_records_without_mapped_label")
        if not checks["georgia_mapping_review_exists"]:
            issues.append("Georgia gate requires a reviewed mapping CSV, even if all reviewed rows are deferred")
        if not checks["georgia_review_has_mapped_codes"]:
            issues.append("Georgia gate requires at least one reviewed map/include row for the frozen taxonomy")
        if not checks["georgia_reports_mapping_inventory"]:
            issues.append("Georgia gate requires a non-empty reviewed mapping/code inventory table")
        if not checks["georgia_group_unit_declared"]:
            issues.append("Georgia gate requires explicit record-level independence assumption")
    elif dataset == "cpsc2021":
        checks["cpsc_reports_annotation_skips"] = "skipped_annotation_records" in load_summary
        checks["cpsc_has_positive_and_negative_windows"] = bool(
            y_true.shape[1] == 1 and 0 < np.sum(y_true[:, 0]) < y_true.shape[0]
        )
        checks["cpsc_windows_loaded"] = int(load_summary.get("loaded_windows", 0)) == int(y_true.shape[0])
        checks["cpsc_reports_negative_windows"] = int(load_summary.get("negative_windows", 0)) > 0
        checks["cpsc_reports_ambiguous_windows"] = "ambiguous_windows" in load_summary
        audit_path = resolve_payload_path(load_summary.get("annotation_audit_csv"))
        checks["cpsc_annotation_audit_exists"] = bool(
            load_summary.get("annotation_audit_csv")
            and audit_path is not None
            and audit_path.exists()
            and audit_path.stat().st_size > 0
        )
        checks["cpsc_group_unit_source_record"] = group_unit == "source_ecg_record"
        checks["cpsc_multiple_windows_per_group"] = bool(
            checks["group_id_present"] and len(np.unique(group_id)) < len(group_id)
        )
        if not checks["cpsc_reports_annotation_skips"]:
            issues.append("CPSC gate requires skipped_annotation_records")
        if not checks["cpsc_has_positive_and_negative_windows"]:
            issues.append("CPSC gate requires both positive and negative annotation windows")
        if not checks["cpsc_windows_loaded"]:
            issues.append("CPSC loaded_windows does not match prediction rows")
        if not checks["cpsc_reports_negative_windows"]:
            issues.append("CPSC gate requires explicitly counted normal/negative windows")
        if not checks["cpsc_reports_ambiguous_windows"]:
            issues.append("CPSC gate requires ambiguous_windows audit count")
        if not checks["cpsc_annotation_audit_exists"]:
            issues.append("CPSC gate requires a non-empty per-record annotation audit table")
        if not checks["cpsc_group_unit_source_record"]:
            issues.append("CPSC gate requires source_ecg_record grouping")
        if not checks["cpsc_multiple_windows_per_group"]:
            issues.append("CPSC gate expected repeated windows clustered by source record")
        warnings.append("CPSC is evaluated on annotation-aligned windows, not official episode score.")

    archive = archive_path(dataset)
    if archive.exists():
        manifest_archive = manifest.get("archive", {})
        checks["archive_fingerprint_matches"] = manifest_archive.get("fingerprint") == file_fingerprint(archive)
        checks["archive_size_matches"] = int(manifest_archive.get("size_bytes", -1)) == archive.stat().st_size
        if not checks["archive_fingerprint_matches"] or not checks["archive_size_matches"]:
            issues.append("archive fingerprint/size mismatch")
    else:
        checks["archive_exists"] = False
        issues.append(f"archive missing: {archive}")

    pca_rows = (manifest.get("pca") or {}).get("folds", [])
    checks["pca_fold_count_5"] = len(pca_rows) == int(CONFIG["n_folds"])
    checks["pca_fingerprints_present"] = len(pca_fingerprints or []) == int(CONFIG["n_folds"])
    if not checks["pca_fold_count_5"]:
        issues.append("fold PCA manifest must include all five folds")
    for row in pca_rows:
        path = Path(row.get("path", ""))
        if not path.exists():
            issues.append(f"fold PCA missing: {path}")
            continue
        if sha256_file(path) != row.get("sha256"):
            issues.append(f"fold PCA checksum mismatch: {path}")

    checkpoints = manifest.get("checkpoints", [])
    checks["checkpoint_count_5"] = len(checkpoints) == int(CONFIG["n_folds"])
    checks["checkpoint_kind_expected"] = summary.get("checkpoint_kind") == args.expected_checkpoint_kind
    checks["checkpoint_fingerprints_present"] = len(checkpoint_fingerprints or []) == int(CONFIG["n_folds"])
    if not checks["checkpoint_count_5"]:
        issues.append("external manifest must include all five checkpoint fingerprints")
    if not checks["checkpoint_kind_expected"]:
        issues.append(f"checkpoint_kind mismatch: {summary.get('checkpoint_kind')} != {args.expected_checkpoint_kind}")
    expected_kind = expected_weight_kind(args.expected_checkpoint_kind)
    for row in checkpoints:
        fold = int(row.get("fold", -1))
        if expected_kind and row.get("weights_kind") != expected_kind:
            issues.append(f"fold {fold} checkpoint weights_kind mismatch")
        oof_row = oof_checkpoints.get(fold)
        if oof_row:
            if row.get("sha256") != oof_row.get("sha256"):
                issues.append(f"fold {fold} checkpoint sha does not match frozen OOF run manifest")
        else:
            issues.append(f"fold {fold} missing from frozen OOF run manifest")
        path = Path(row.get("path", ""))
        if path.exists() and sha256_file(path) != row.get("sha256"):
            issues.append(f"fold {fold} checkpoint path checksum mismatch: {path}")

    if y_true.shape == y_prob.shape:
        class_has_two_labels = [len(np.unique(y_true[:, idx])) >= 2 for idx in range(y_true.shape[1])]
        checks["at_least_one_evaluable_class"] = any(class_has_two_labels)
        if not checks["at_least_one_evaluable_class"]:
            issues.append("no class has both positive and negative labels")
    else:
        class_has_two_labels = []
        checks["at_least_one_evaluable_class"] = False

    metrics = multilabel_metrics(y_true, y_prob, threshold=args.threshold) if not issues else {}
    calib = calibration_summary(y_true, y_prob, n_bins=args.n_bins) if not issues else {}
    bootstrap = (
        metric_ci(y_true, y_prob, group_id, args.threshold, args.n_boot, args.seed, args.n_bins)
        if not issues
        else {}
    )

    metric_rows = [
        {
            "dataset": dataset,
            "protocol_gate_passed": not issues,
            "manuscript_ready": not issues,
            "n_records": int(y_true.shape[0]),
            "n_groups": int(len(np.unique(group_id))) if len(group_id) else 0,
            "group_unit": group_unit,
            "n_classes": int(y_true.shape[1]),
            "threshold": args.threshold,
            "n_bins": args.n_bins,
            "n_boot": args.n_boot,
            **metrics,
            **{f"calibration_{key}": value for key, value in calib.items()},
        }
    ]
    label_rows = class_rows(dataset, y_true, class_names) if y_true.shape == y_prob.shape else []

    status = "protocol_gate_passed" if not issues else "blocked_protocol_gate_failed"

    payload = {
        "dataset": dataset,
        "created_utc": created_utc,
        "git_commit": git_commit(),
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "gate_cache_key": cache_key,
        "reused_existing": False,
        "status": status,
        "protocol_gate_passed": not issues,
        "manuscript_ready": not issues,
        "source_external_artifacts_remain_experimental": True,
        "safe_claim": (
            "Protocol-gated external evaluation under mapped label/task definitions only."
            if not issues
            else "External outputs remain experimental and are not manuscript-ready."
        ),
        "unsafe_claims": [
            "unqualified external-transfer advantage",
            "benchmark-leading external performance",
            "unqualified cross-dataset robustness advantage",
            "clinical deployment readiness",
        ],
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
        "n_records": int(y_true.shape[0]),
        "n_groups": int(len(np.unique(group_id))) if len(group_id) else 0,
        "group_unit": group_unit,
        "n_classes": int(y_true.shape[1]),
        "class_names": class_names,
        "metrics": metrics,
        "calibration": calib,
        "bootstrap_ci": bootstrap,
        "load_summary": load_summary,
        "artifacts": {
            **{name: artifact(path) for name, path in required_paths.items()},
            "oof_run_manifest": artifact(args.oof_run_manifest),
            "gate_json": artifact(gate_path),
            "label_table": artifact(label_table),
            "metrics_table": artifact(metrics_table),
            "gate_manifest": artifact(gate_manifest),
            "georgia_mapping_inventory": artifact(resolve_payload_path(load_summary.get("mapping_inventory_csv")))
            if dataset == "georgia" and resolve_payload_path(load_summary.get("mapping_inventory_csv")) is not None
            else {"path": "", "exists": False, "size_bytes": 0, "sha256": ""},
            "cpsc_annotation_audit": artifact(resolve_payload_path(load_summary.get("annotation_audit_csv")))
            if dataset == "cpsc2021" and resolve_payload_path(load_summary.get("annotation_audit_csv")) is not None
            else {"path": "", "exists": False, "size_bytes": 0, "sha256": ""},
        },
        "contract": {
            "expected_checkpoint_kind": args.expected_checkpoint_kind,
            "expected_label_protocol": expected_label_protocol,
            "expected_class_names": list(EXPECTED_CLASS_NAMES[dataset]),
            "aggregation": {"method": "power_mean", "q": float(CONFIG["power_mean_q"])},
            "bootstrap_unit": "patient/source-record group",
            "metric_implementation": artifact(METRIC_IMPLEMENTATION_PATH),
            "single_label_metric_semantics": "positive_label_multilabel_reduction",
            "group_unit": group_unit,
            "threshold": args.threshold,
            "n_bins": args.n_bins,
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
    }

    save_csv(label_table, label_rows)
    save_csv(metrics_table, metric_rows)
    save_json(gate_path, payload)
    payload["artifacts"]["gate_json"] = artifact(gate_path)
    payload["artifacts"]["label_table"] = artifact(label_table)
    payload["artifacts"]["metrics_table"] = artifact(metrics_table)
    save_json(
        gate_manifest,
        {
            "dataset": dataset,
            "created_utc": created_utc,
            "git_commit": payload["git_commit"],
            "gate_schema_version": GATE_SCHEMA_VERSION,
            "gate_cache_key": cache_key,
            "status": status,
            "protocol_gate_passed": not issues,
            "artifacts": payload["artifacts"],
            "issues": issues,
            "warnings": warnings,
            "contract": payload["contract"],
            "source_manifest": artifact(manifest_path),
            "source_prediction": artifact(prediction_path),
        },
    )
    payload["artifacts"]["gate_manifest"] = artifact(gate_manifest)
    save_json(gate_path, payload)
    return payload, metric_rows, label_rows


def summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    calib = payload.get("calibration", {})
    issues = payload.get("issues", [])
    return {
        "dataset": payload.get("dataset", ""),
        "status": payload.get("status", ""),
        "protocol_gate_passed": payload.get("protocol_gate_passed", False),
        "manuscript_ready": payload.get("manuscript_ready", False),
        "n_records": payload.get("n_records", 0),
        "n_groups": payload.get("n_groups", 0),
        "group_unit": payload.get("group_unit", ""),
        "n_classes": payload.get("n_classes", 0),
        "roc_auc_macro": metrics.get("roc_auc_macro", math.nan),
        "pr_auc_macro": metrics.get("pr_auc_macro", math.nan),
        "f1_macro": metrics.get("f1_macro", math.nan),
        "brier_macro": calib.get("brier_macro", math.nan),
        "ece_macro": calib.get("ece_macro", math.nan),
        "issue_count": len(issues),
        "issues": "; ".join(str(issue) for issue in issues[:8]),
        "safe_claim": payload.get("safe_claim", ""),
        "gate_schema_version": payload.get("gate_schema_version", ""),
        "gate_cache_key": payload.get("gate_cache_key", ""),
        "reused_existing": payload.get("reused_existing", False),
        "gate_json": payload.get("artifacts", {}).get("gate_json", {}).get("path", ""),
        "gate_manifest": payload.get("artifacts", {}).get("gate_manifest", {}).get("path", ""),
        "prediction_sha256": payload.get("artifacts", {}).get("prediction", {}).get("sha256", ""),
        "slice_prediction_sha256": payload.get("artifacts", {}).get("slice_prediction", {}).get("sha256", ""),
    }


def normalize_datasets(values: list[str] | None) -> list[str]:
    if not values or "all" in values:
        return list(DATASETS)
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    datasets = normalize_datasets(args.dataset)
    oof_checkpoints = load_oof_checkpoint_contract(args.oof_run_manifest)
    payloads = []
    print("=" * 80)
    print("EXTERNAL PROTOCOL GATE")
    print("=" * 80)
    print(f"datasets={','.join(datasets)}")
    print(f"external_root={args.external_root}")
    print(f"oof_run_manifest={args.oof_run_manifest} exists={args.oof_run_manifest.exists()}")
    print(
        f"threshold={args.threshold} n_bins={args.n_bins} n_boot={args.n_boot} "
        f"seed={args.seed} reuse_existing={args.reuse_existing}"
    )
    for dataset in datasets:
        payload, _, _ = validate_dataset(dataset, args, oof_checkpoints)
        payloads.append(payload)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "status": payload["status"],
                    "protocol_gate_passed": payload["protocol_gate_passed"],
                    "issues": payload["issues"],
                },
                indent=2,
            )
        )
    rows = [summary_row(payload) for payload in payloads]
    save_csv(args.out_summary, rows)
    print(f"Wrote: {args.out_summary}")
    if args.strict and any(not row["protocol_gate_passed"] for row in rows):
        failed = [row["dataset"] for row in rows if not row["protocol_gate_passed"]]
        raise SystemExit(f"External protocol gate failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
