"""Paired group-bootstrap external comparison for Chapman-trained models.

All models must predict the same mapped external records/windows without using
target labels. Patient/source-record groups are resampled as intact clusters.
CPSC2021 remains a separate AF/AFL mapped-window task and is never combined
with PTB-XL or Georgia.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    CHAPMAN_GROUP_REFERENCE,
    CHAPMAN_GROUP_SEMANTICS,
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    calibration_summary,
    git_commit,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_csv,
    save_json,
    save_json_atomic,
    sha256_file,
)


PROTOCOL_VERSION = 3
COMPARATOR_STEMS = {
    "resnet": "resnet1d_cnn",
    "raw_mamba": "raw_mamba",
    "transformer": "transformer_ecg",
}
COMPARATOR_LABELS = {
    "resnet": "ResNet1D/CNN",
    "raw_mamba": "Raw Mamba",
    "transformer": "Transformer ECG",
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    higher_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["ptbxl", "georgia", "cpsc2021", "all"],
    )
    parser.add_argument("--comparators", default="resnet,raw_mamba,transformer")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=EXPERIMENTAL_DIR / "external",
    )
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "external_comparator_paired_metric_cache",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_external_comparator_paired.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "external_comparator_paired_summary.json",
    )
    parser.add_argument(
        "--out-samples",
        type=Path,
        default=METRIC_DIR / "external_comparator_paired_bootstrap_samples.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "external_comparator_paired_manifest.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_contract_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_contract() -> dict[str, Any]:
    oof = PREDICTION_DIR / "oof_final_ema_predictions.npz"
    freeze = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
    if not oof.exists() or not freeze.exists():
        raise FileNotFoundError("Canonical frozen OOF/freeze artifacts are required for external pairing.")
    freeze_payload = json.loads(freeze.read_text(encoding="utf-8"))
    if freeze_payload.get("status") != "frozen" or freeze_payload.get("manuscript_ready") is not True:
        raise RuntimeError("Canonical freeze manifest is not frozen/manuscript_ready.")
    oof_sha = sha256_file(oof)
    expected_oof_sha = next(
        (
            row.get("sha256")
            for row in freeze_payload.get("artifacts", [])
            if str(row.get("path", "")).replace("\\", "/").endswith(oof.name)
        ),
        None,
    )
    if expected_oof_sha != oof_sha:
        raise RuntimeError(f"Freeze OOF SHA mismatch: {expected_oof_sha} != {oof_sha}")
    group = freeze_payload.get("group_contract") or {}
    sidecar = group.get("sidecar") or {}
    errors = []
    if group.get("status") != "verified":
        errors.append("status")
    if group.get("group_semantics") != CHAPMAN_GROUP_SEMANTICS:
        errors.append("group_semantics")
    if group.get("group_semantics_reference") != CHAPMAN_GROUP_REFERENCE:
        errors.append("group_semantics_reference")
    if group.get("bootstrap_unit") != AUTHENTICATED_RECORD_BOOTSTRAP_UNIT:
        errors.append("bootstrap_unit")
    if group.get("one_record_per_group") is not True:
        errors.append("one_record_per_group")
    if int(group.get("n_records", -1)) != int(freeze_payload.get("validated_records", -2)):
        errors.append("n_records")
    if int(group.get("n_groups", -1)) != int(freeze_payload.get("validated_records", -2)):
        errors.append("n_groups")
    sidecar_path = resolve_contract_path(sidecar.get("path"))
    if not str(sidecar.get("path") or "") or not sidecar_path.is_file():
        errors.append("group_sidecar_missing")
    elif not sidecar.get("sha256") or sha256_file(sidecar_path) != sidecar.get("sha256"):
        errors.append("group_sidecar_sha256")
    if errors:
        raise RuntimeError(
            "Canonical freeze lacks an authenticated live patient/group contract: "
            + ", ".join(errors)
        )
    return {
        "oof_sha256": oof_sha,
        "freeze_sha256": sha256_file(freeze),
        "group_contract_sha256": canonical_json_sha256(group),
        "group_sidecar_sha256": sidecar["sha256"],
        "bootstrap_unit": AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def selected_datasets(values: list[str] | None) -> list[str]:
    items = values or ["ptbxl", "georgia", "cpsc2021"]
    if "all" in items:
        return ["ptbxl", "georgia", "cpsc2021"]
    return list(dict.fromkeys(items))


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def prediction_path(root: Path, dataset: str, model: str) -> Path:
    if model == "full":
        return root / dataset / f"{dataset}_full_predictions.npz"
    return root / dataset / f"{dataset}_{COMPARATOR_STEMS[model]}_predictions.npz"


def comparator_manifest_path(dataset: str, comparator: str) -> Path:
    return MANIFEST_DIR / f"external_{dataset}_{COMPARATOR_STEMS[comparator]}_manifest.json"


def load_predictions(path: Path, expected_dataset: str) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "group_id", "class_names", "dataset"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing group-safe keys: {sorted(missing)}")
        payload = {
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "y_prob": np.asarray(data["y_prob"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]).astype(str),
            "group_id": np.asarray(data["group_id"]).astype(str),
            "split_id": np.asarray(data["split_id"]).astype(str)
            if "split_id" in data.files
            else np.asarray([f"{expected_dataset}_unspecified"] * len(data["record_id"])),
            "class_names": np.asarray(data["class_names"]).astype(str),
            "dataset": str(scalar(data, "dataset", "")),
            "protocol": str(scalar(data, "protocol", "")),
            "task_scope": str(scalar(data, "task_scope", "")),
            "group_unit": str(scalar(data, "group_unit", "group")),
            "adaptation_labels_used": int(scalar(data, "adaptation_labels_used", 0)),
        }
    if payload["dataset"] != expected_dataset:
        raise ValueError(f"Dataset mismatch: {payload['dataset']} != {expected_dataset}")
    if payload["y_true"].ndim != 2 or payload["y_true"].shape != payload["y_prob"].shape:
        raise ValueError(f"{path}: label/prediction shape mismatch")
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
    group_rows = [
        f"{record_id}\x1e{group_id}\x1e{split_id}"
        for record_id, group_id, split_id in zip(
            payload["record_id"], payload["group_id"], payload["split_id"]
        )
    ]
    payload["group_assignment_sha256"] = hashlib.sha256(
        "\x1f".join(group_rows).encode("utf-8")
    ).hexdigest()
    return payload


def validate_pair(full: dict[str, Any], comparator: dict[str, Any], dataset: str) -> None:
    for key in ("y_true", "record_id", "group_id", "split_id", "class_names"):
        if not np.array_equal(full[key], comparator[key]):
            raise ValueError(f"{dataset}: Full/comparator {key} differs")
    if comparator["adaptation_labels_used"] != 0:
        raise ValueError(f"{dataset}: comparator predictions used target labels")
    if dataset == "cpsc2021":
        if "af_afl" not in comparator["task_scope"].lower():
            raise ValueError("CPSC comparator artifact does not declare the AF/AFL mapped-window task")
    elif "record" not in comparator["task_scope"].lower():
        raise ValueError(f"{dataset}: comparator artifact is not a record-level mapped task")


def validate_full_gate(dataset: str, full: dict[str, Any]) -> dict[str, Any]:
    path = METRIC_DIR / f"external_{dataset}_protocol_gate.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing passed Full external gate: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("protocol_gate_passed") is not True or gate.get("manuscript_ready") is not True:
        raise RuntimeError(f"Full external gate has not passed for {dataset}")
    expected_sha = ((gate.get("artifacts") or {}).get("prediction") or {}).get("sha256")
    if expected_sha and expected_sha != full["sha256"]:
        raise RuntimeError(f"{dataset}: Full external gate is stale for prediction NPZ")
    if int(gate.get("n_records", -1)) != len(full["record_id"]):
        raise RuntimeError(f"{dataset}: Full external gate record count differs from predictions")
    if int(gate.get("n_groups", -1)) != len(np.unique(full["group_id"])):
        raise RuntimeError(f"{dataset}: Full external gate group count differs from predictions")
    if str(gate.get("group_unit") or "") != full["group_unit"]:
        raise RuntimeError(f"{dataset}: Full external gate group unit differs from predictions")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "gate_cache_key": gate.get("gate_cache_key"),
        "group_assignment_sha256": full["group_assignment_sha256"],
        "n_groups": int(len(np.unique(full["group_id"]))),
        "group_unit": full["group_unit"],
    }


def metric_specs(threshold: float, n_bins: int) -> list[MetricSpec]:
    return [
        MetricSpec("pr_auc_macro", "ranking", True, macro_pr_auc),
        MetricSpec("roc_auc_macro", "ranking", True, macro_roc_auc),
        MetricSpec(
            "f1_macro",
            "fixed_threshold",
            True,
            lambda y, p: multilabel_metrics(y, p, threshold=threshold)["f1_macro"],
        ),
        MetricSpec(
            "brier_macro",
            "calibration",
            False,
            lambda y, p: calibration_summary(y, p, n_bins=n_bins)["brier_macro"],
        ),
        MetricSpec(
            "ece_macro",
            "calibration",
            False,
            lambda y, p: calibration_summary(y, p, n_bins=n_bins)["ece_macro"],
        ),
    ]


def metric_cache_contract(
    dataset: str,
    comparator: str,
    metric: str,
    full_sha: str,
    comparator_sha: str,
    args: argparse.Namespace,
    *,
    group_assignment_sha256: str,
    canonical: dict[str, Any],
    full_gate_sha256: str,
    comparator_manifest_sha256: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset": dataset,
        "comparator": comparator,
        "metric": metric,
        "full_sha": full_sha,
        "comparator_sha": comparator_sha,
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "seed": int(bootstrap_seed),
        "group_assignment_sha256": group_assignment_sha256,
        "canonical_oof_sha256": canonical["oof_sha256"],
        "canonical_freeze_sha256": canonical["freeze_sha256"],
        "canonical_group_contract_sha256": canonical["group_contract_sha256"],
        "canonical_group_sidecar_sha256": canonical["group_sidecar_sha256"],
        "full_gate_sha256": full_gate_sha256,
        "comparator_manifest_sha256": comparator_manifest_sha256,
    }


def metric_cache_key(contract: dict[str, Any]) -> str:
    return canonical_json_sha256(contract)


def validate_bootstrap_payload(
    result: dict[str, Any],
    samples: np.ndarray,
    *,
    n_boot: int,
) -> None:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or len(samples) != int(n_boot) or not np.isfinite(samples).all():
        raise RuntimeError("Paired external metric cache lacks the exact finite bootstrap sample count")
    if int(result.get("n_boot_valid", -1)) != int(n_boot):
        raise RuntimeError("Paired external metric cache n_boot_valid differs from the request")
    for field in ("improvement_ci_low", "improvement_ci_high"):
        try:
            value = float(result[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Paired external metric cache lacks numeric {field}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"Paired external metric cache contains non-finite {field}")
    for key, value in result.items():
        if "significant" in str(key).lower() or "significant" in str(value).lower():
            raise RuntimeError("Paired external metric cache contains legacy significance wording")


def paired_group_bootstrap(
    y_true: np.ndarray,
    full_prob: np.ndarray,
    comparator_prob: np.ndarray,
    groups: np.ndarray,
    spec: MetricSpec,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    groups = np.asarray(groups).astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Paired external bootstrap requires at least two groups")
    members = [np.where(inverse == idx)[0] for idx in range(len(unique))]
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(n_boot)):
        sampled_groups = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[int(group)] for group in sampled_groups])
        try:
            raw = float(spec.fn(y_true[idx], full_prob[idx])) - float(
                spec.fn(y_true[idx], comparator_prob[idx])
            )
        except (ValueError, RuntimeError):
            continue
        improvement = raw if spec.higher_is_better else -raw
        if np.isfinite(improvement):
            samples.append(improvement)
    if not samples:
        raise RuntimeError("No valid paired group-bootstrap samples")
    values = np.asarray(samples, dtype=np.float64)
    full_value = float(spec.fn(y_true, full_prob))
    comparator_value = float(spec.fn(y_true, comparator_prob))
    raw = full_value - comparator_value
    point = raw if spec.higher_is_better else -raw
    return (
        {
            "full_value": full_value,
            "comparator_value": comparator_value,
            "raw_difference_full_minus_comparator": raw,
            "improvement_full_over_comparator": point,
            "improvement_bootstrap_mean": float(np.mean(values)),
            "improvement_ci_low": float(np.quantile(values, 0.025)),
            "improvement_ci_high": float(np.quantile(values, 0.975)),
            "p_value_two_sided": None,
            "n_boot_valid": int(len(values)),
            "n_groups": int(len(unique)),
            "sample_unit": "patient/source-record group",
            "inference_scope": "pointwise_percentile_ci_effect_size_only",
            "null_test": "not_run",
        },
        values,
    )


def mark_pointwise_inference(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["holm_p_value_two_sided"] = None
        row["multiplicity_adjustment"] = "not_applicable_no_null_test"


def interpretation(row: dict[str, Any]) -> str:
    low = float(row["improvement_ci_low"])
    high = float(row["improvement_ci_high"])
    if low > 0:
        return "full_nominal_95ci_better"
    if high < 0:
        return "comparator_nominal_95ci_better"
    return "paired_difference_inconclusive"


def safe_wording(row: dict[str, Any]) -> str:
    dataset = row["dataset"]
    comparator = row["comparator_label"]
    metric = row["metric"]
    scope = "CPSC2021 AF/AFL mapped-window" if dataset == "cpsc2021" else f"{dataset} mapped-task"
    outcome = row["interpretation"]
    direction = "higher" if row["higher_is_better"] else "lower"
    if outcome == "full_nominal_95ci_better":
        return f"The pointwise 95% paired effect-size interval favors ECG-RAMBA ({direction} {metric}) over {comparator} for this zero-target-label {scope} comparison."
    if outcome == "comparator_nominal_95ci_better":
        return f"The pointwise 95% paired effect-size interval favors {comparator} ({direction} {metric}) over ECG-RAMBA for this zero-target-label {scope} comparison."
    return f"The paired {metric} difference between ECG-RAMBA and {comparator} is inconclusive for this zero-target-label {scope} comparison."


def main() -> None:
    args = parse_args()
    canonical = canonical_contract()
    datasets = selected_datasets(args.dataset)
    comparators = parse_list(args.comparators)
    unknown = sorted(set(comparators) - set(COMPARATOR_STEMS))
    if unknown:
        raise ValueError(f"Unknown comparators: {unknown}")
    external_root = resolve(args.external_root)
    cache_dir = resolve(args.metric_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80, flush=True)
    print("PAIRED EXTERNAL LEARNED-COMPARATOR AUDIT", flush=True)
    print("=" * 80, flush=True)
    print(f"datasets={datasets} comparators={comparators} n_boot={args.n_boot}", flush=True)

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dataset in datasets:
        try:
            full = load_predictions(prediction_path(external_root, dataset, "full"), dataset)
            gate_info = validate_full_gate(dataset, full)
            for comparator in comparators:
                comp = load_predictions(prediction_path(external_root, dataset, comparator), dataset)
                validate_pair(full, comp, dataset)
                manifest_path = comparator_manifest_path(dataset, comparator)
                if not manifest_path.exists():
                    raise FileNotFoundError(manifest_path)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_comp_sha = ((manifest.get("artifacts") or {}).get("predictions") or {}).get("sha256")
                if expected_comp_sha != comp["sha256"]:
                    raise RuntimeError(f"{dataset}/{comparator}: comparator manifest is stale")
                if manifest.get("canonical_contract") != canonical:
                    raise RuntimeError(f"{dataset}/{comparator}: comparator manifest uses a stale OOF/freeze contract")
                source = manifest.get("source_contract") if isinstance(manifest.get("source_contract"), dict) else {}
                runner = PROJECT_ROOT / "scripts" / "revision" / "31_generate_external_comparator_predictions.py"
                loader = PROJECT_ROOT / "scripts" / "revision" / "03_generate_external_predictions.py"
                if (
                    not source.get("archive_sha256")
                    or not source.get("runner_sha256")
                    or source.get("runner_sha256") != sha256_file(runner)
                    or source.get("external_loader_sha256") != sha256_file(loader)
                ):
                    raise RuntimeError(f"{dataset}/{comparator}: comparator source contract is missing or stale")
                inputs.extend(
                    [
                        {"dataset": dataset, "model": "full", "path": str(full["path"]), "sha256": full["sha256"]},
                        {"dataset": dataset, "model": comparator, "path": str(comp["path"]), "sha256": comp["sha256"]},
                    ]
                )
                for metric_index, spec in enumerate(metric_specs(args.threshold, args.n_bins)):
                    bootstrap_seed = args.seed + metric_index * 1009
                    manifest_sha256 = sha256_file(manifest_path)
                    cache_contract = metric_cache_contract(
                        dataset,
                        comparator,
                        spec.name,
                        full["sha256"],
                        comp["sha256"],
                        args,
                        group_assignment_sha256=full["group_assignment_sha256"],
                        canonical=canonical,
                        full_gate_sha256=gate_info["sha256"],
                        comparator_manifest_sha256=manifest_sha256,
                        bootstrap_seed=bootstrap_seed,
                    )
                    key = metric_cache_key(cache_contract)
                    path = cache_dir / f"{dataset}_{comparator}_{spec.name}_{key[:16]}.json"
                    if args.reuse_existing and path.exists():
                        cached = json.loads(path.read_text(encoding="utf-8"))
                        if (
                            cached.get("cache_key") != key
                            or cached.get("contract") != cache_contract
                            or cached.get("dataset") != dataset
                            or cached.get("comparator") != comparator
                            or cached.get("metric") != spec.name
                        ):
                            raise RuntimeError(f"Paired external metric cache contract mismatch: {path}")
                        result = cached["result"]
                        samples = np.asarray(cached["samples"], dtype=np.float64)
                        validate_bootstrap_payload(result, samples, n_boot=args.n_boot)
                    else:
                        result, samples = paired_group_bootstrap(
                            full["y_true"],
                            full["y_prob"],
                            comp["y_prob"],
                            full["group_id"],
                            spec,
                            args.n_boot,
                            bootstrap_seed,
                        )
                        validate_bootstrap_payload(result, samples, n_boot=args.n_boot)
                        save_json_atomic(
                            path,
                            {
                                "cache_key": key,
                                "contract": cache_contract,
                                "dataset": dataset,
                                "comparator": comparator,
                                "metric": spec.name,
                                "result": result,
                                "samples": samples.tolist(),
                            },
                        )
                    row = {
                        "dataset": dataset,
                        "task_scope": (
                            "cpsc2021_10s_af_afl_mapped_windows"
                            if dataset == "cpsc2021"
                            else "record_level_mapped_external_task"
                        ),
                        "comparison": f"full_vs_{comparator}",
                        "comparator": comparator,
                        "comparator_label": COMPARATOR_LABELS[comparator],
                        "metric": spec.name,
                        "metric_family": spec.family,
                        "higher_is_better": spec.higher_is_better,
                        "group_unit": full["group_unit"],
                        "group_assignment_sha256": full["group_assignment_sha256"],
                        **result,
                        "full_gate_sha256": gate_info["sha256"],
                        "comparator_manifest_sha256": manifest_sha256,
                    }
                    rows.append(row)
                    sample_rows.extend(
                        {
                            "dataset": dataset,
                            "comparator": comparator,
                            "metric": spec.name,
                            "bootstrap_index": idx,
                            "improvement_full_over_comparator": float(value),
                        }
                        for idx, value in enumerate(samples)
                    )
                    print(
                        f"{dataset}/{comparator}/{spec.name}: improvement={result['improvement_full_over_comparator']:.6f} "
                        f"CI=[{result['improvement_ci_low']:.6f}, {result['improvement_ci_high']:.6f}]",
                        flush=True,
                    )
        except Exception as exc:
            failures.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{dataset} failed: {type(exc).__name__}: {exc}", flush=True)
            if args.strict:
                raise

    mark_pointwise_inference(rows)
    for row in rows:
        row["interpretation"] = interpretation(row)
        row["safe_wording"] = safe_wording(row)
    out_table = resolve(args.out_table)
    out_json = resolve(args.out_json)
    out_samples = resolve(args.out_samples)
    out_manifest = resolve(args.out_manifest)
    save_csv(out_table, rows)
    save_csv(out_samples, sample_rows)
    save_json(
        out_json,
        {
            "status": "complete" if not failures else "incomplete",
            "created_utc": now_utc(),
            "protocol": "paired_group_bootstrap_external_zero_target_label_v2_pointwise_ci",
            "inference": {
                "confidence_interval": "paired group-percentile bootstrap",
                "p_value": "not_reported",
                "reason": "bootstrap-tail proportions are not null-centered tests",
                "multiplicity_adjustment": "not_applicable_no_null_test",
            },
            "n_boot": args.n_boot,
            "seed": args.seed,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "canonical_contract": canonical,
            "rows": rows,
            "failures": failures,
            "safe_wording": (
                "Report each dataset, comparator, and metric separately; do not infer general external superiority."
            ),
        },
    )
    artifacts = [out_table, out_json, out_samples]
    save_json(
        out_manifest,
        {
            "status": "complete" if not failures else "incomplete",
            "created_utc": now_utc(),
            "git_commit": git_commit(),
            "protocol_version": PROTOCOL_VERSION,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "canonical_contract": canonical,
            "inputs": inputs,
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in artifacts
            ],
            "failures": failures,
        },
    )
    print(json.dumps({"status": not failures, "rows": len(rows), "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
