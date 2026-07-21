"""Audit PTB-XL fold 9/10 isolation and unsupported-only sensitivity.

The primary mapped task retains records with no positive label among the four
supported superclasses. This runner additionally reports a pre-declared
sensitivity analysis excluding those records. Selection depends only on the
mapped reference labels, never on model predictions. All intervals resample
intact PTB-XL patient groups and paired model deltas use shared resamples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    TABLE_DIR,
    calibration_summary,
    cluster_bootstrap_ci,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    paired_cluster_bootstrap_delta,
    save_csv,
    save_json,
    save_json_atomic,
    sha256_file,
)


PTBXL_FOLD_PROTOCOL_AUDIT_CAPABILITY = "ptbxl_fold9_fold10_patient_audit_v1"
PTBXL_FOLD_PROTOCOL_AUDIT_SCHEMA_VERSION = 2

MODEL_STEMS = {
    "full": "full",
    "resnet": "resnet1d_cnn",
    "raw_mamba": "raw_mamba",
    "transformer": "transformer_ecg",
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    higher_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="full,resnet,raw_mamba,transformer")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-root", type=Path, default=EXPERIMENTAL_DIR / "external")
    parser.add_argument(
        "--ptbxl-archive",
        type=Path,
        required=True,
        help="Authoritative PTB-XL ZIP containing ptbxl_database.csv.",
    )
    parser.add_argument(
        "--analysis-lock",
        type=Path,
        default=MANIFEST_DIR / "ptbxl_adaptation_analysis_lock.json",
    )
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "ptbxl_fold_protocol_metric_cache",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "ptbxl_fold_protocol_audit.json",
    )
    parser.add_argument(
        "--out-audit-table",
        type=Path,
        default=TABLE_DIR / "table_ptbxl_fold_protocol_audit.csv",
    )
    parser.add_argument(
        "--out-sensitivity-table",
        type=Path,
        default=TABLE_DIR / "table_ptbxl_unsupported_only_sensitivity.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "ptbxl_fold_protocol_audit_manifest.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def prediction_path(root: Path, model: str, fold9: bool) -> Path:
    stem = MODEL_STEMS[model]
    suffix = "_fold9" if fold9 else ""
    return resolve(root) / "ptbxl" / f"ptbxl_{stem}{suffix}_predictions.npz"


def _normalize_integer_id(value: Any, *, field: str) -> str:
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Invalid integral {field}: {value!r}")
    return str(int(numeric))


def load_official_ptbxl_metadata(archive_path: Path) -> tuple[dict[str, tuple[str, int]], dict[str, Any]]:
    archive_path = resolve(archive_path)
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise FileNotFoundError(f"Authoritative PTB-XL archive is missing: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"PTB-XL archive is not a valid ZIP: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.replace("\\", "/").lower().endswith("/ptbxl_database.csv")
            or name.replace("\\", "/").lower() == "ptbxl_database.csv"
        )
        if len(members) != 1:
            raise RuntimeError(
                f"Expected exactly one ptbxl_database.csv in {archive_path}, found {members}"
            )
        member = members[0]
        raw = archive.read(member)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    required = {"ecg_id", "patient_id", "strat_fold"}
    if not reader.fieldnames or required - set(reader.fieldnames):
        raise ValueError(f"Official PTB-XL metadata lacks columns: {sorted(required)}")
    rows: dict[str, tuple[str, int]] = {}
    for row in reader:
        ecg_id = _normalize_integer_id(row["ecg_id"], field="ecg_id")
        patient_id = _normalize_integer_id(row["patient_id"], field="patient_id")
        fold = int(_normalize_integer_id(row["strat_fold"], field="strat_fold"))
        if ecg_id in rows:
            raise RuntimeError(f"Official PTB-XL metadata contains duplicate ecg_id={ecg_id}")
        rows[ecg_id] = (patient_id, fold)
    if not rows:
        raise RuntimeError("Official PTB-XL metadata is empty")
    return rows, {
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "metadata_member": member,
        "metadata_member_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata_rows": len(rows),
        "source_columns": ["ecg_id", "patient_id", "strat_fold"],
    }


def validate_against_official_metadata(
    prediction: dict[str, Any],
    official: dict[str, tuple[str, int]],
    *,
    expected_fold: int,
    label: str,
) -> None:
    try:
        record_ids = np.asarray(
            [_normalize_integer_id(value, field="record_id") for value in prediction["record_id"]]
        )
        observed_groups = np.asarray(
            [_normalize_integer_id(value, field="group_id") for value in prediction["group_id"]]
        )
    except ValueError as exc:
        raise RuntimeError(f"{label}: prediction IDs are not valid PTB-XL integer IDs") from exc
    missing = [record_id for record_id in record_ids if record_id not in official]
    if missing:
        raise RuntimeError(f"{label}: records absent from official PTB-XL metadata: {missing[:10]}")
    expected_groups = np.asarray([official[record_id][0] for record_id in record_ids])
    observed_folds = np.asarray([official[record_id][1] for record_id in record_ids])
    if not np.array_equal(observed_groups, expected_groups):
        mismatch = int(np.sum(observed_groups != expected_groups))
        raise RuntimeError(f"{label}: {mismatch} patient IDs differ from ptbxl_database.csv")
    if not np.all(observed_folds == expected_fold):
        raise RuntimeError(
            f"{label}: official strat_fold differs; observed={sorted(set(observed_folds.tolist()))}"
        )


def load_prediction(path: Path, expected_split: str) -> dict[str, Any]:
    path = resolve(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "group_id", "split_id", "class_names", "dataset"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path}: missing keys {sorted(missing)}")
        out = {
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "y_prob": np.asarray(data["y_prob"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]).astype(str),
            "group_id": np.asarray(data["group_id"]).astype(str),
            "split_id": np.asarray(data["split_id"]).astype(str),
            "class_names": np.asarray(data["class_names"]).astype(str),
            "dataset": str(scalar(data, "dataset", "")),
            "group_unit": str(scalar(data, "group_unit", "")),
            "path": path,
            "sha256": sha256_file(path),
        }
    y_true, y_prob = out["y_true"], out["y_prob"]
    if out["dataset"] != "ptbxl":
        raise ValueError(f"{path}: dataset must be ptbxl")
    if y_true.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError(f"{path}: y_true/y_prob shape mismatch")
    if not np.isin(y_true, [0.0, 1.0]).all() or not np.isfinite(y_prob).all():
        raise ValueError(f"{path}: labels must be binary and probabilities finite")
    if float(y_prob.min()) < 0.0 or float(y_prob.max()) > 1.0:
        raise ValueError(f"{path}: probabilities outside [0,1]")
    n = len(y_true)
    for key in ("record_id", "group_id", "split_id"):
        if len(out[key]) != n or np.any(np.char.str_len(out[key]) == 0):
            raise ValueError(f"{path}: invalid {key}")
    if len(np.unique(out["record_id"])) != n:
        raise ValueError(f"{path}: duplicate record_id")
    if set(out["split_id"]) != {expected_split}:
        raise ValueError(f"{path}: expected split {expected_split}, observed {sorted(set(out['split_id']))}")
    if "patient" not in out["group_unit"].lower():
        raise ValueError(f"{path}: PTB-XL group_unit must identify patients, got {out['group_unit']!r}")
    return out


def validate_lock(path: Path, models: list[str], args: argparse.Namespace) -> dict[str, Any]:
    path = resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"PTB-XL adaptation analysis lock is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    expected = {
        "status": payload.get("status") == "locked",
        "splits": protocol.get("adaptation_split") == "official_ptbxl_fold9" and protocol.get("test_split") == "official_ptbxl_fold10",
        "models": protocol.get("models") == models,
        "threshold": math.isclose(float(protocol.get("threshold", math.nan)), args.threshold, abs_tol=1e-12),
        "n_bins": int(protocol.get("n_bins", -1)) == args.n_bins,
        "n_boot": int(protocol.get("n_boot", -1)) == args.n_boot,
        "group_unit": protocol.get("group_unit") == "patient_id",
    }
    failed = [key for key, ok in expected.items() if not ok]
    if failed:
        raise RuntimeError(f"PTB-XL analysis lock does not match this audit: {failed}")
    if payload.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise RuntimeError("PTB-XL analysis lock protocol SHA mismatch")
    return {"path": str(path), "sha256": sha256_file(path), "protocol_sha256": payload["protocol_sha256"]}


def validate_same_reference(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    for key in ("y_true", "record_id", "group_id", "split_id", "class_names"):
        if not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"{label}: {key} differs from Full")


def patient_overlap(adaptation_groups: np.ndarray, test_groups: np.ndarray) -> list[str]:
    """Return sorted shared patient IDs for an explicit leakage gate."""

    return sorted(
        set(np.asarray(adaptation_groups).astype(str))
        & set(np.asarray(test_groups).astype(str))
    )


def metrics(threshold: float, n_bins: int) -> list[MetricSpec]:
    return [
        MetricSpec("pr_auc_macro", "ranking", True, macro_pr_auc),
        MetricSpec("roc_auc_macro", "ranking", True, macro_roc_auc),
        MetricSpec("f1_macro", "fixed_threshold", True, lambda y, p: multilabel_metrics(y, p, threshold=threshold)["f1_macro"]),
        MetricSpec("brier_macro", "calibration", False, lambda y, p: calibration_summary(y, p, n_bins=n_bins)["brier_macro"]),
        MetricSpec("ece_macro", "calibration", False, lambda y, p: calibration_summary(y, p, n_bins=n_bins)["ece_macro"]),
    ]


def cached_metric(path: Path, contract: dict[str, Any], compute: Callable[[], dict[str, Any]], reuse: bool) -> dict[str, Any]:
    key = canonical_json_sha256(contract)
    if reuse and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cache_key") == key and payload.get("contract") == contract:
            return payload["result"]
    result = compute()
    save_json_atomic(path, {"cache_key": key, "contract": contract, "result": result})
    return result


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if models != ["full", "resnet", "raw_mamba", "transformer"]:
        raise ValueError("Audit requires full,resnet,raw_mamba,transformer in that order")
    lock = validate_lock(args.analysis_lock, models, args)
    official_metadata, official_contract = load_official_ptbxl_metadata(args.ptbxl_archive)
    root = resolve(args.external_root)
    test = {model: load_prediction(prediction_path(root, model, False), "ptbxl_fold10") for model in models}
    fold9 = {model: load_prediction(prediction_path(root, model, True), "ptbxl_fold9") for model in models}
    for model in models[1:]:
        validate_same_reference(test["full"], test[model], f"fold10/{model}")
        validate_same_reference(fold9["full"], fold9[model], f"fold9/{model}")
    validate_against_official_metadata(
        fold9["full"], official_metadata, expected_fold=9, label="fold9/full"
    )
    validate_against_official_metadata(
        test["full"], official_metadata, expected_fold=10, label="fold10/full"
    )
    if not np.array_equal(test["full"]["class_names"], fold9["full"]["class_names"]):
        raise RuntimeError("PTB-XL fold9/fold10 class order differs")
    overlap = patient_overlap(fold9["full"]["group_id"], test["full"]["group_id"])
    if overlap:
        raise RuntimeError(f"PTB-XL patient leakage between fold9 and fold10: {overlap[:10]}")

    audit_rows = []
    for split_name, payload in (("fold9_adaptation", fold9["full"]), ("fold10_test", test["full"])):
        supported_mask = np.any(payload["y_true"] > 0.5, axis=1)
        audit_rows.append(
            {
                "split": split_name,
                "n_records": int(len(supported_mask)),
                "n_patients": int(len(np.unique(payload["group_id"]))),
                "unsupported_only_records": int(np.sum(~supported_mask)),
                "supported_positive_records": int(np.sum(supported_mask)),
                "split_id": str(np.unique(payload["split_id"])[0]),
                "group_unit": payload["group_unit"],
                "patient_overlap_with_other_split": 0,
            }
        )

    cache_dir = resolve(args.metric_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    runner_sha = sha256_file(Path(__file__).resolve())
    sensitivity_rows: list[dict[str, Any]] = []
    reference = test["full"]
    masks = {
        "primary_all_mapped_records": np.ones(len(reference["y_true"]), dtype=bool),
        "sensitivity_exclude_unsupported_only": np.any(reference["y_true"] > 0.5, axis=1),
    }
    for analysis_index, (analysis_set, mask) in enumerate(masks.items()):
        if int(mask.sum()) == 0:
            raise RuntimeError(f"{analysis_set}: empty evaluable subset")
        y = reference["y_true"][mask]
        groups = reference["group_id"][mask]
        mask_sha = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        for metric_index, spec in enumerate(metrics(args.threshold, args.n_bins)):
            seed = args.seed + analysis_index * 100_003 + metric_index * 1009
            for model in models:
                prob = test[model]["y_prob"][mask]
                contract = {
                    "schema": PTBXL_FOLD_PROTOCOL_AUDIT_SCHEMA_VERSION,
                    "runner_sha256": runner_sha,
                    "analysis_set": analysis_set,
                    "model": model,
                    "metric": spec.name,
                    "prediction_sha256": test[model]["sha256"],
                    "mask_sha256": mask_sha,
                    "n_boot": args.n_boot,
                    "seed": seed,
                    "analysis_lock_sha256": lock["sha256"],
                    "official_metadata_member_sha256": official_contract["metadata_member_sha256"],
                }
                cache = cache_dir / f"{analysis_set}__{model}__{spec.name}__{canonical_json_sha256(contract)[:16]}.json"
                ci = cached_metric(
                    cache,
                    contract,
                    lambda y=y, prob=prob, groups=groups, spec=spec, seed=seed: cluster_bootstrap_ci(
                        y, prob, groups, spec.fn, n_boot=args.n_boot, seed=seed
                    ),
                    args.reuse_existing,
                )
                point = float(spec.fn(y, prob))
                sensitivity_rows.append(
                    {
                        "row_type": "model_metric",
                        "analysis_set": analysis_set,
                        "selection_basis": "mapped_reference_labels_only",
                        "model": model,
                        "comparator": "",
                        "metric": spec.name,
                        "metric_family": spec.family,
                        "higher_is_better": spec.higher_is_better,
                        "point_value": point,
                        "ci_low": ci["lo"],
                        "ci_high": ci["hi"],
                        "improvement_full_over_comparator": "",
                        "improvement_ci_low": "",
                        "improvement_ci_high": "",
                        "n_records": int(mask.sum()),
                        "n_groups": int(len(np.unique(groups))),
                        "n_boot_valid": int(ci["n_boot_valid"]),
                        "inference_scope": "patient_group_percentile_ci_effect_size_only",
                    }
                )
            for comparator in models[1:]:
                full_prob = test["full"]["y_prob"][mask]
                comp_prob = test[comparator]["y_prob"][mask]
                contract = {
                    "schema": PTBXL_FOLD_PROTOCOL_AUDIT_SCHEMA_VERSION,
                    "runner_sha256": runner_sha,
                    "analysis_set": analysis_set,
                    "comparison": f"full_vs_{comparator}",
                    "metric": spec.name,
                    "full_sha256": test["full"]["sha256"],
                    "comparator_sha256": test[comparator]["sha256"],
                    "mask_sha256": mask_sha,
                    "n_boot": args.n_boot,
                    "seed": seed,
                    "analysis_lock_sha256": lock["sha256"],
                    "official_metadata_member_sha256": official_contract["metadata_member_sha256"],
                }
                cache = cache_dir / f"{analysis_set}__full_vs_{comparator}__{spec.name}__{canonical_json_sha256(contract)[:16]}.json"
                paired = cached_metric(
                    cache,
                    contract,
                    lambda y=y, full_prob=full_prob, comp_prob=comp_prob, groups=groups, spec=spec, seed=seed: paired_cluster_bootstrap_delta(
                        y, full_prob, comp_prob, groups, spec.fn, n_boot=args.n_boot, seed=seed
                    ),
                    args.reuse_existing,
                )
                if spec.higher_is_better:
                    point, low, high = paired["point_delta_a_minus_b"], paired["lo"], paired["hi"]
                else:
                    point, low, high = -paired["point_delta_a_minus_b"], -paired["hi"], -paired["lo"]
                sensitivity_rows.append(
                    {
                        "row_type": "paired_full_vs_comparator",
                        "analysis_set": analysis_set,
                        "selection_basis": "mapped_reference_labels_only",
                        "model": "full",
                        "comparator": comparator,
                        "metric": spec.name,
                        "metric_family": spec.family,
                        "higher_is_better": spec.higher_is_better,
                        "point_value": "",
                        "ci_low": "",
                        "ci_high": "",
                        "improvement_full_over_comparator": point,
                        "improvement_ci_low": low,
                        "improvement_ci_high": high,
                        "n_records": int(mask.sum()),
                        "n_groups": int(len(np.unique(groups))),
                        "n_boot_valid": int(paired["n_boot_valid"]),
                        "inference_scope": "paired_patient_group_percentile_ci_effect_size_only",
                    }
                )

    if args.strict:
        incomplete = [row for row in sensitivity_rows if int(row["n_boot_valid"]) != args.n_boot]
        if incomplete:
            raise RuntimeError(f"Incomplete patient-group bootstrap rows: {len(incomplete)}")
    out_audit = resolve(args.out_audit_table)
    out_sensitivity = resolve(args.out_sensitivity_table)
    out_json = resolve(args.out_json)
    out_manifest = resolve(args.out_manifest)
    save_csv(out_audit, audit_rows)
    save_csv(out_sensitivity, sensitivity_rows)
    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capability": PTBXL_FOLD_PROTOCOL_AUDIT_CAPABILITY,
        "schema_version": PTBXL_FOLD_PROTOCOL_AUDIT_SCHEMA_VERSION,
        "analysis_lock": lock,
        "official_metadata_contract": official_contract,
        "split_contract": {
            "adaptation_split": "ptbxl_fold9",
            "test_split": "ptbxl_fold10",
            "group_unit": "patient_id",
            "patient_overlap": 0,
            "fold9_records": int(len(fold9["full"]["y_true"])),
            "fold9_patients": int(len(np.unique(fold9["full"]["group_id"]))),
            "fold10_records": int(len(test["full"]["y_true"])),
            "fold10_patients": int(len(np.unique(test["full"]["group_id"]))),
        },
        "unsupported_only": {
            "definition": "no positive label among the four supported mapped superclasses",
            "selection_uses_predictions": False,
            "fold9_count": audit_rows[0]["unsupported_only_records"],
            "fold10_count": audit_rows[1]["unsupported_only_records"],
            "primary_analysis": "all mapped-task records",
            "sensitivity_analysis": "exclude unsupported-only records",
        },
        "inference": {
            "bootstrap_unit": "patient_id",
            "n_boot": args.n_boot,
            "confidence_interval": "patient-group percentile bootstrap",
            "paired_interval": "shared patient-group resamples",
            "p_values": "not_reported",
        },
        "inputs": {
            split: {
                model: {"path": str(data["path"]), "sha256": data["sha256"]}
                for model, data in collection.items()
            }
            for split, collection in (("fold9", fold9), ("fold10", test))
        },
        "outputs": {
            "audit_table": str(out_audit),
            "sensitivity_table": str(out_sensitivity),
        },
        "claim_boundary": (
            "Report the unsupported-only exclusion as a label-defined sensitivity analysis; "
            "do not replace the all-record mapped-task primary result or infer general external superiority."
        ),
    }
    save_json(out_json, payload)
    artifacts = [out_json, out_audit, out_sensitivity]
    save_json(
        out_manifest,
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": runner_sha,
            "analysis_lock": lock,
            "official_metadata_contract": official_contract,
            "inputs": payload["inputs"],
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in artifacts
            ],
        },
    )
    print(json.dumps({"status": True, "audit_rows": len(audit_rows), "sensitivity_rows": len(sensitivity_rows), "unsupported_fold10": audit_rows[1]["unsupported_only_records"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
