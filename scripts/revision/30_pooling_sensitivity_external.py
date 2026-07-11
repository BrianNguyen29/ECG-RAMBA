"""Post-hoc pooling sensitivity on existing external slice predictions.

No model inference is performed.  The runner validates that Q=3 reconstructs
the stored record predictions before evaluating alternative aggregation rules.
CPSC2021 remains a separate mapped-window task in all outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    TABLE_DIR,
    calibration_summary,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    paired_cluster_bootstrap_delta,
    save_csv,
    save_json,
    save_json_atomic,
    sha256_file,
)
from src.aggregation import POWER_MEAN_IMPLEMENTATION, aggregate_record_probabilities  # noqa: E402


DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT / "reports" / "revision" / "experimental" / "external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=["ptbxl", "georgia", "cpsc2021"])
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--q-values", default="1,2,3,4,8,max")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--strict-group-bootstrap", action="store_true")
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "pooling_external_metric_cache",
    )
    parser.add_argument("--out-summary", type=Path, default=METRIC_DIR / "pooling_sensitivity_external.csv")
    parser.add_argument("--out-table", type=Path, default=TABLE_DIR / "table_pooling_sensitivity_across_datasets.csv")
    parser.add_argument("--out-bootstrap", type=Path, default=METRIC_DIR / "pooling_q3_paired_bootstrap.json")
    parser.add_argument("--out-manifest", type=Path, default=MANIFEST_DIR / "pooling_sensitivity_external_manifest.json")
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


def parse_methods(value: str) -> list[tuple[str, float | str]]:
    methods: list[tuple[str, float | str]] = []
    for item in str(value).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "max":
            methods.append(("max", "max"))
        else:
            q = float(token)
            if q <= 0:
                raise ValueError("Pooling q values must be positive.")
            name = "mean" if q == 1 else f"power_mean_q{q:g}"
            methods.append((name, q))
    if not any(q == 3 for _, q in methods):
        raise ValueError("Q=3 must be included as the frozen reference.")
    return methods


def derive_groups(dataset: str, record_ids: np.ndarray, payload: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str, bool]:
    if "group_id" in payload.files:
        return np.asarray(payload["group_id"]).astype(str), str(npz_scalar(payload, "group_unit", "group")), True
    ids = np.asarray(record_ids).astype(str)
    if dataset == "cpsc2021":
        groups = np.asarray([value.rsplit(":", 2)[0] if value.count(":") >= 2 else value for value in ids])
        return groups, "source_ecg_record_derived_from_window_id", True
    if dataset == "georgia":
        return ids, "record_id_assumed_independent", True
    return ids, "record_proxy_patient_id_unavailable", False


def aggregate_max(slice_prob: np.ndarray, record_index: np.ndarray, n_records: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros((n_records, slice_prob.shape[1]), dtype=np.float32)
    counts = np.bincount(record_index, minlength=n_records).astype(np.int16)
    valid = counts > 0
    for record in np.where(valid)[0]:
        out[record] = np.max(slice_prob[record_index == record], axis=0)
    return out, valid, counts


def metric_functions(threshold: float, n_bins: int) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "pr_auc_macro": macro_pr_auc,
        "roc_auc_macro": macro_roc_auc,
        "f1_macro": lambda y, p: multilabel_metrics(y, p, threshold=threshold)["f1_macro"],
        "brier_macro": lambda y, p: calibration_summary(y, p, n_bins=n_bins)["brier_macro"],
        "ece_macro": lambda y, p: calibration_summary(y, p, n_bins=n_bins)["ece_macro"],
    }


def metric_cache_key(
    *,
    dataset: str,
    alternative: str,
    metric: str,
    record_prediction_sha256: str,
    slice_prediction_sha256: str,
    groups: np.ndarray,
    threshold: float,
    n_bins: int,
    n_boot: int,
    seed: int,
) -> str:
    payload = {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset": dataset,
        "reference": "power_mean_q3",
        "alternative": alternative,
        "metric": metric,
        "record_prediction_sha256": record_prediction_sha256,
        "slice_prediction_sha256": slice_prediction_sha256,
        "group_fingerprint": hashlib.sha256(
            "\n".join(np.asarray(groups).astype(str)).encode("utf-8")
        ).hexdigest(),
        "threshold": float(threshold),
        "n_bins": int(n_bins),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    args = parse_args()
    datasets = args.dataset or ["ptbxl", "georgia", "cpsc2021"]
    methods = parse_methods(args.q_values)
    external_root = resolve(args.external_root)
    cache_dir = resolve(args.metric_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("EXTERNAL POOLING SENSITIVITY")
    print("=" * 80)
    print(f"datasets={datasets} methods={[name for name, _ in methods]} n_boot={args.n_boot}")

    rows: list[dict] = []
    bootstrap_items: dict[str, dict] = {}
    inputs: list[dict] = []
    for dataset in datasets:
        root = external_root / dataset
        pred_path = root / f"{dataset}_full_predictions.npz"
        slice_path = root / f"{dataset}_full_slice_predictions.npz"
        gate_path = METRIC_DIR / f"external_{dataset}_protocol_gate.json"
        if not pred_path.exists() or not slice_path.exists():
            raise FileNotFoundError(f"Missing external prediction/slice artifacts for {dataset}: {root}")
        if not gate_path.exists():
            raise FileNotFoundError(f"Missing protocol gate for {dataset}: {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("status") != "protocol_gate_passed"
            or gate.get("protocol_gate_passed") is not True
            or gate.get("manuscript_ready") is not True
        ):
            raise RuntimeError(f"{dataset}: external protocol gate is not manuscript-ready")
        with np.load(pred_path, allow_pickle=False) as pred:
            y_true = np.asarray(pred["y_true"], dtype=np.float32)
            stored_prob = np.asarray(pred["y_prob"], dtype=np.float32)
            record_ids = np.asarray(pred["record_id"]).astype(str)
            class_names = np.asarray(pred["class_names"]).astype(str)
            groups, group_unit, group_safe = derive_groups(dataset, record_ids, pred)
        with np.load(slice_path, allow_pickle=False) as slices:
            slice_prob = np.asarray(slices["slice_prob"], dtype=np.float32)
            record_index = np.asarray(slices["record_index"], dtype=np.int64)
            slice_classes = np.asarray(slices["class_names"]).astype(str)
        if not np.array_equal(class_names, slice_classes):
            raise ValueError(f"{dataset} class order differs between record and slice predictions.")
        if np.any(record_index < 0) or np.any(record_index >= len(y_true)):
            raise ValueError(f"{dataset} slice record_index is out of range.")
        if args.strict_group_bootstrap and not group_safe:
            raise RuntimeError(f"{dataset} has no validated patient/source group IDs.")
        pred_sha = sha256_file(pred_path)
        slice_sha = sha256_file(slice_path)
        gate_prediction_sha = (((gate.get("artifacts") or {}).get("prediction") or {}).get("sha256"))
        gate_slice_sha = (((gate.get("artifacts") or {}).get("slice_prediction") or {}).get("sha256"))
        if gate_prediction_sha != pred_sha or gate_slice_sha != slice_sha:
            raise RuntimeError(f"{dataset}: external prediction artifacts differ from the passed protocol gate")
        inputs.extend(
            [
                {"dataset": dataset, "kind": "protocol_gate", "path": str(gate_path), "sha256": sha256_file(gate_path)},
                {"dataset": dataset, "kind": "record_predictions", "path": str(pred_path), "sha256": pred_sha},
                {"dataset": dataset, "kind": "slice_predictions", "path": str(slice_path), "sha256": slice_sha},
            ]
        )

        probabilities: dict[str, np.ndarray] = {}
        counts_reference = None
        for method_name, q in methods:
            if q == "max":
                prob, valid, counts = aggregate_max(slice_prob, record_index, len(y_true))
            else:
                prob, valid, counts = aggregate_record_probabilities(
                    slice_prob, record_index, len(y_true), q=float(q)
                )
            if not np.all(valid):
                raise RuntimeError(f"{dataset}/{method_name}: records without slices={int(np.sum(~valid))}")
            if counts_reference is None:
                counts_reference = counts
            elif not np.array_equal(counts_reference, counts):
                raise RuntimeError(f"{dataset}: slice counts changed across pooling methods.")
            probabilities[method_name] = np.asarray(prob, dtype=np.float32)
            point = {**multilabel_metrics(y_true, prob, threshold=args.threshold), **calibration_summary(y_true, prob, n_bins=args.n_bins)}
            rows.append(
                {
                    "dataset": dataset,
                    "task_scope": "cpsc2021_10s_af_afl_mapped_windows" if dataset == "cpsc2021" else "mapped_external_records",
                    "pooling": method_name,
                    "q": q,
                    "threshold": args.threshold,
                    "n_records": len(y_true),
                    "n_groups": len(np.unique(groups)),
                    "group_unit": group_unit,
                    "group_safe": group_safe,
                    "slice_count_min": int(np.min(counts)),
                    "slice_count_max": int(np.max(counts)),
                    "f1_macro": point["f1_macro"],
                    "roc_auc_macro": point["roc_auc_macro"],
                    "pr_auc_macro": point["pr_auc_macro"],
                    "brier_macro": point["brier_macro"],
                    "ece_macro": point["ece_macro"],
                }
            )

        q3 = probabilities["power_mean_q3"]
        max_abs = float(np.max(np.abs(q3 - stored_prob)))
        if max_abs > 2e-6:
            raise RuntimeError(f"{dataset}: reconstructed Q=3 differs from stored predictions (max_abs={max_abs})")

        for method_name, _q in methods:
            if method_name == "power_mean_q3":
                continue
            for metric_name, metric_fn in metric_functions(args.threshold, args.n_bins).items():
                key = f"{dataset}__q3_vs_{method_name}__{metric_name}"
                cache_key = metric_cache_key(
                    dataset=dataset,
                    alternative=method_name,
                    metric=metric_name,
                    record_prediction_sha256=pred_sha,
                    slice_prediction_sha256=slice_sha,
                    groups=groups,
                    threshold=args.threshold,
                    n_bins=args.n_bins,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                cache_path = cache_dir / f"{key}_{cache_key[:16]}.json"
                if args.reuse_existing and cache_path.exists():
                    item = json.loads(cache_path.read_text(encoding="utf-8"))
                    if item.get("cache_key") != cache_key:
                        raise RuntimeError(f"Pooling metric cache key mismatch: {cache_path}")
                else:
                    item = paired_cluster_bootstrap_delta(
                        y_true,
                        q3,
                        probabilities[method_name],
                        groups,
                        metric_fn,
                        n_boot=args.n_boot,
                        seed=args.seed,
                    )
                    item.update(
                        {
                            "dataset": dataset,
                            "reference": "power_mean_q3",
                            "alternative": method_name,
                            "metric": metric_name,
                            "group_unit": group_unit,
                            "group_safe": group_safe,
                            "cache_key": cache_key,
                        }
                    )
                    save_json_atomic(cache_path, item)
                bootstrap_items[key] = item
                print(f"{key}: delta={item.get('point_delta_a_minus_b')} CI=[{item.get('lo')}, {item.get('hi')}]", flush=True)

    out_summary = resolve(args.out_summary)
    out_table = resolve(args.out_table)
    out_bootstrap = resolve(args.out_bootstrap)
    out_manifest = resolve(args.out_manifest)
    save_csv(out_summary, rows)
    save_csv(out_table, rows)
    save_json(
        out_bootstrap,
        {
            "status": True,
            "created_utc": now_utc(),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "items": bootstrap_items,
        },
    )
    outputs = [out_summary, out_table, out_bootstrap]
    save_json(
        out_manifest,
        {
            "status": True,
            "created_utc": now_utc(),
            "protocol": "external_pooling_sensitivity_v2_group_bootstrap",
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "datasets": datasets,
            "methods": [name for name, _ in methods],
            "threshold": float(args.threshold),
            "n_bins": int(args.n_bins),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "strict_group_bootstrap": bool(args.strict_group_bootstrap),
            "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
            "frozen_reference": "power_mean_q3",
            "safe_wording": "Q=3 is a frozen operating point, not a universally optimal pooling exponent.",
            "inputs": inputs,
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in outputs
            ],
        },
    )
    print(json.dumps({"status": True, "rows": len(rows), "bootstrap_items": len(bootstrap_items)}, indent=2))


if __name__ == "__main__":
    main()
