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
        "clean": "minirocket_only_oof_predictions.npz",
        "stress": "robustness_minirocket_{stress}_predictions.npz",
    },
    "resnet": {
        "label": "ResNet1D/CNN",
        "clean": "resnet1d_cnn_oof_predictions.npz",
        "stress": "robustness_resnet1d_cnn_{stress}_predictions.npz",
    },
    "raw_mamba": {
        "label": "Raw Mamba",
        "clean": "raw_mamba_oof_predictions.npz",
        "stress": "robustness_raw_mamba_{stress}_predictions.npz",
    },
    "transformer": {
        "label": "Transformer ECG",
        "clean": "transformer_ecg_oof_predictions.npz",
        "stress": "robustness_transformer_ecg_{stress}_predictions.npz",
    },
}


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
    with np.load(path, allow_pickle=True) as data:
        required = ["y_true", "y_prob", "record_id", "class_names"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} missing keys={missing}")
        payload = {key: data[key] for key in data.files}
    payload["y_true"] = np.asarray(payload["y_true"], dtype=np.float32)
    payload["y_prob"] = np.asarray(payload["y_prob"], dtype=np.float32)
    payload["record_id"] = np.asarray(payload["record_id"]).astype(str)
    payload["class_names"] = np.asarray(payload["class_names"]).astype(str)
    payload["fold_id"] = np.asarray(payload.get("fold_id", np.zeros(len(payload["record_id"])))).astype(int)
    if payload["y_true"].shape != payload["y_prob"].shape:
        raise ValueError(f"{path} shape mismatch: {payload['y_true'].shape} vs {payload['y_prob'].shape}")
    if np.any(~np.isfinite(payload["y_prob"])):
        raise ValueError(f"{path} contains non-finite probabilities")
    payload["path"] = path
    payload["sha256"] = sha256_file(path)
    return payload


def validate_same_contract(reference: dict[str, Any], other: dict[str, Any], label: str) -> None:
    for key in ["y_true", "record_id", "class_names", "fold_id"]:
        if key not in reference or key not in other:
            continue
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"{label} differs from Full contract on {key}")


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
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(full_clean["y_true"])
    values = []
    stressed_values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fc = metric_value(spec, full_clean, idx)
        fs = metric_value(spec, full_stress, idx)
        cc = metric_value(spec, comp_clean, idx)
        cs = metric_value(spec, comp_stress, idx)
        if not all(np.isfinite([fc, fs, cc, cs])):
            continue
        full_deg = benefit(fs, spec["direction"]) - benefit(fc, spec["direction"])
        comp_deg = benefit(cs, spec["direction"]) - benefit(cc, spec["direction"])
        values.append(float(full_deg - comp_deg))
        stressed_values.append(float(benefit(fs, spec["direction"]) - benefit(cs, spec["direction"])))
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
    print(f"metric_cache_dir={resolve(args.metric_cache_dir)} reuse={args.reuse_metric_cache}", flush=True)

    clean: dict[str, dict[str, Any]] = {}
    artifact_status: list[dict[str, Any]] = []
    for comp in comparators:
        path = PREDICTION_DIR / COMPARATORS[comp]["clean"]
        try:
            clean[comp] = load_npz(path)
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
        "metrics": requested_metrics,
        "metric_cache_dir": project_relative(args.metric_cache_dir),
        "items": {},
    }

    for stress in stresses:
        stress_data: dict[str, dict[str, Any]] = {}
        for comp in comparators:
            if comp not in clean:
                continue
            stress_path = PREDICTION_DIR / COMPARATORS[comp]["stress"].format(stress=stress)
            try:
                stress_data[comp] = load_npz(stress_path)
                validate_same_contract(full_clean, stress_data[comp], f"{comp}/{stress}")
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
        "metrics": requested_metrics,
        "completed_rows": len(completed),
        "blocked_rows": len(blocked),
        "artifact_status": artifact_status,
        "outputs": {
            "summary": project_relative(args.out_summary),
            "table": project_relative(args.out_table),
            "pairwise": project_relative(args.out_pairwise),
            "manifest": project_relative(args.out_manifest),
        },
        "git_commit": git_commit(),
    }
    save_csv(args.out_summary, rows)
    save_csv(args.out_table, rows)
    save_json(args.out_pairwise, pairwise)
    save_json(args.out_manifest, manifest)
    print(json.dumps({"status": True, "completed_rows": len(completed), "blocked_rows": len(blocked)}, indent=2))
    if args.strict and blocked:
        raise RuntimeError(f"Blocked robustness rows remain: {len(blocked)}")


if __name__ == "__main__":
    main()
