"""Run perturbation robustness tests for Full ECG-RAMBA vs MiniRocket-only.

This runner is intentionally heavier than the HRV/domain notebook cells. It
re-runs inference on perturbed Chapman OOF validation records instead of
deriving robustness claims from clean predictions.

Protocol guarantees:
- The Full model uses frozen fold checkpoints, existing training-fold PCA
  objects, Q=3 power-mean slice aggregation, and the same OOF fold contract.
- MiniRocket-only heads are trained on clean train-fold MiniRocket features and
  evaluated on perturbed validation-fold MiniRocket features. The heads are
  reusable and tied to the clean MiniRocket cache SHA and hyperparameters.
- Paired bootstrap resamples records with one shared index vector across clean,
  perturbed, Full, and MiniRocket predictions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG, EVALUATION_CONFIG_HASH, PATHS  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    EXPERIMENTAL_DIR,
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
    save_json,
    sha256_file,
)
from src.aggregation import POWER_MEAN_IMPLEMENTATION, power_mean  # noqa: E402
from src.features import (  # noqa: E402
    MiniRocketNative,
    extract_amplitude_features,
    extract_global_record_stats,
    extract_hrv_features,
)
from src.provenance import record_order_fingerprint  # noqa: E402


PROTOCOL = "robustness_full_vs_minirocket_perturbation_v1"
MINIROCKET_HEAD_PROTOCOL = "minirocket_clean_heads_for_robustness_v1"
EXPECTED_MINIROCKET_PROTOCOL = "minirocket_raw_standardized_torch_linear_same_folds_threshold_0.5"
DEFAULT_STRESS_TESTS = [
    "snr20db",
    "snr10db",
    "snr5db",
    "random_3_lead_dropout",
    "precordial_dropout",
    "resample_250hz",
]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    higher_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]


@dataclass
class MiniRocketHeads:
    head_dir: Path
    manifest: dict
    clean_prob: np.ndarray
    fold_id: np.ndarray
    fold_rows: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--stress-tests", default=",".join(DEFAULT_STRESS_TESTS))
    parser.add_argument("--checkpoint-kind", default="final_ema")
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--minirocket-feature-batch-size", type=int, default=64)
    parser.add_argument("--minirocket-batch-size", type=int, default=4096)
    parser.add_argument("--minirocket-stats-batch-size", type=int, default=1024)
    parser.add_argument("--minirocket-epochs", type=int, default=20)
    parser.add_argument("--minirocket-lr", type=float, default=1e-3)
    parser.add_argument("--minirocket-weight-decay", type=float, default=1e-4)
    parser.add_argument("--minirocket-device", default="auto")
    parser.add_argument("--minirocket-feature-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-minirocket-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-perturbed-caches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-legacy-shape-cache", action="store_true")
    parser.add_argument("--full-clean-predictions", type=Path, default=PREDICTION_DIR / "oof_final_ema_predictions.npz")
    parser.add_argument(
        "--minirocket-clean-predictions",
        type=Path,
        default=PREDICTION_DIR / "minirocket_only_oof_predictions.npz",
    )
    parser.add_argument("--freeze-manifest", type=Path, default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json")
    parser.add_argument("--minirocket-manifest", type=Path, default=MANIFEST_DIR / "minirocket_only_baseline_manifest.json")
    parser.add_argument("--minirocket-summary", type=Path, default=METRIC_DIR / "minirocket_only_baseline_summary.json")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    return resolve_path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_revision_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def metric_value(spec: MetricSpec, y: np.ndarray, prob: np.ndarray) -> float:
    value = float(spec.fn(y, prob))
    return value if np.isfinite(value) else math.nan


def degradation(clean_value: float, stress_value: float, spec: MetricSpec) -> float:
    if not np.isfinite(clean_value) or not np.isfinite(stress_value):
        return math.nan
    return clean_value - stress_value if spec.higher_is_better else stress_value - clean_value


def load_prediction_npz(path: Path, label: str) -> dict:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} predictions: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "class_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing required keys: {sorted(missing)}")
        payload = {
            "path": path,
            "sha256": sha256_file(path),
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "y_prob": np.asarray(data["y_prob"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"], dtype=np.int64),
            "class_names": np.asarray(data["class_names"]).astype(str).tolist(),
            "fold_id": np.asarray(data["fold_id"], dtype=np.int16) if "fold_id" in data.files else None,
            "metadata": {
                key: (data[key].item() if np.ndim(data[key]) == 0 else data[key].tolist())
                for key in data.files
                if key
                in {
                    "dataset",
                    "protocol",
                    "checkpoint_kind",
                    "dataset_record_order_fingerprint",
                    "aggregation_method",
                    "aggregation_q",
                    "aggregation_implementation",
                    "feature_contract",
                    "feature_preprocessing",
                    "manuscript_ready",
                }
            },
        }
    y_true = payload["y_true"]
    y_prob = payload["y_prob"]
    if y_true.ndim != 2 or y_prob.shape != y_true.shape:
        raise ValueError(f"{label} shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")
    if len(payload["record_id"]) != len(y_true):
        raise ValueError(f"{label} record_id length mismatch")
    if payload["fold_id"] is not None and len(payload["fold_id"]) != len(y_true):
        raise ValueError(f"{label} fold_id length mismatch")
    if payload["class_names"] != CLASSES:
        raise ValueError(f"{label} class_names do not match config CLASSES")
    if not np.array_equal(payload["record_id"], np.arange(len(y_true), dtype=np.int64)):
        raise ValueError(f"{label} record_id must be exactly 0..N-1")
    if not np.isfinite(y_prob).all():
        raise ValueError(f"{label} probabilities contain non-finite values")
    payload["y_prob"] = np.clip(y_prob, 0.0, 1.0).astype(np.float32)
    return payload


def validate_clean_prediction_contract(full: dict, mini: dict, args: argparse.Namespace) -> dict:
    if not np.array_equal(full["y_true"], mini["y_true"]):
        raise ValueError("Full and MiniRocket clean y_true arrays differ.")
    if not np.array_equal(full["record_id"], mini["record_id"]):
        raise ValueError("Full and MiniRocket clean record_id arrays differ.")
    if full["class_names"] != mini["class_names"]:
        raise ValueError("Full and MiniRocket class_names differ.")
    if full["fold_id"] is None:
        raise ValueError("Full clean predictions must include fold_id.")
    if mini["fold_id"] is not None and not np.array_equal(full["fold_id"], mini["fold_id"]):
        raise ValueError("Full and MiniRocket clean fold_id arrays differ.")

    freeze_path = resolve_path(args.freeze_manifest)
    if not freeze_path.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("manuscript_ready") is not True:
        raise ValueError("Freeze manifest must be status=frozen and manuscript_ready=true.")
    if freeze.get("checkpoint_kind") != args.expected_checkpoint_kind:
        raise ValueError(
            f"Unexpected freeze checkpoint kind {freeze.get('checkpoint_kind')} "
            f"!= {args.expected_checkpoint_kind}"
        )
    artifacts = {row.get("path"): row for row in freeze.get("artifacts", [])}
    rel = project_relative(full["path"])
    if rel not in artifacts:
        raise ValueError(f"Freeze manifest does not include Full predictions: {rel}")
    if full["sha256"] != artifacts[rel].get("sha256"):
        raise RuntimeError("Full clean prediction SHA does not match freeze manifest.")

    mini_manifest = resolve_path(args.minirocket_manifest)
    mini_summary = resolve_path(args.minirocket_summary)
    if not mini_manifest.exists() or not mini_summary.exists():
        raise FileNotFoundError("MiniRocket baseline manifest/summary are required before robustness tests.")
    mini_manifest_payload = json.loads(mini_manifest.read_text(encoding="utf-8"))
    mini_summary_payload = json.loads(mini_summary.read_text(encoding="utf-8"))
    for source_name, payload in [("summary", mini_summary_payload), ("manifest", mini_manifest_payload)]:
        if payload.get("protocol") != EXPECTED_MINIROCKET_PROTOCOL:
            raise ValueError(
                f"MiniRocket {source_name} protocol mismatch: "
                f"{payload.get('protocol')} != {EXPECTED_MINIROCKET_PROTOCOL}"
            )
        if payload.get("feature_contract") != "minirocket_raw":
            raise ValueError(f"MiniRocket {source_name} feature_contract must be minirocket_raw.")
    if mini_summary_payload.get("manuscript_ready") is not True:
        raise ValueError("MiniRocket summary must report manuscript_ready=true.")
    if int(mini_summary_payload.get("n_records", -1)) != int(full["y_true"].shape[0]):
        raise ValueError("MiniRocket summary n_records does not match Full clean predictions.")
    manifest_prediction_sha = (mini_manifest_payload.get("artifact_sha256") or {}).get("predictions")
    if manifest_prediction_sha and manifest_prediction_sha != mini["sha256"]:
        raise RuntimeError(
            f"MiniRocket prediction SHA mismatch: manifest {manifest_prediction_sha} != file {mini['sha256']}"
        )
    return {
        "freeze_manifest": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
            "checkpoint_kind": freeze.get("checkpoint_kind"),
            "validated_records": freeze.get("validated_records"),
            "dataset_record_order_fingerprint": freeze.get("dataset_record_order_fingerprint"),
        },
        "minirocket_manifest": {
            "path": str(mini_manifest),
            "sha256": sha256_file(mini_manifest),
            "protocol": mini_manifest_payload.get("protocol"),
            "prediction_sha256": manifest_prediction_sha,
        },
        "minirocket_summary": {
            "path": str(mini_summary),
            "sha256": sha256_file(mini_summary),
            "protocol": mini_summary_payload.get("protocol"),
            "manuscript_ready": mini_summary_payload.get("manuscript_ready"),
            "n_records": mini_summary_payload.get("n_records"),
        },
    }


def fold_list_from_fold_id(fold_id: np.ndarray) -> list[dict[str, np.ndarray]]:
    folds = []
    for fold in sorted(int(x) for x in np.unique(fold_id) if int(x) > 0):
        va_idx = np.where(fold_id == fold)[0].astype(np.int64)
        tr_idx = np.where((fold_id > 0) & (fold_id != fold))[0].astype(np.int64)
        if len(tr_idx) and len(va_idx):
            folds.append({"fold": fold, "tr_idx": tr_idx, "va_idx": va_idx})
    if not folds:
        raise ValueError("Could not derive folds from fold_id.")
    return folds


def stable_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def stress_specs(names: list[str], seed: int) -> list[dict]:
    specs = []
    for name in names:
        key = name.strip()
        if not key:
            continue
        if key == "snr20db":
            specs.append({"name": key, "kind": "additive_noise", "snr_db": 20.0, "seed": seed + 2001})
        elif key == "snr10db":
            specs.append({"name": key, "kind": "additive_noise", "snr_db": 10.0, "seed": seed + 2011})
        elif key == "snr5db":
            specs.append({"name": key, "kind": "additive_noise", "snr_db": 5.0, "seed": seed + 2021})
        elif key == "random_3_lead_dropout":
            specs.append({"name": key, "kind": "random_lead_dropout", "n_drop": 3, "seed": seed + 3001})
        elif key == "precordial_dropout":
            specs.append({"name": key, "kind": "fixed_lead_dropout", "lead_indices": list(range(6, 12)), "seed": seed})
        elif key == "resample_250hz":
            specs.append({"name": key, "kind": "resample_down_up", "source_hz": 500, "target_hz": 250, "seed": seed})
        else:
            raise ValueError(f"Unknown stress test: {key}")
    if not specs:
        raise ValueError("No stress tests requested.")
    return specs


def perturb_signals(X: np.ndarray, spec: dict) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=np.float32)
    meta = {"spec": dict(spec)}
    kind = spec["kind"]
    if kind == "additive_noise":
        rng = np.random.default_rng(int(spec["seed"]))
        out = np.empty_like(X, dtype=np.float32)
        target = 10.0 ** (float(spec["snr_db"]) / 10.0)
        for start in tqdm(range(0, len(X), 128), desc=f"perturb {spec['name']}"):
            stop = min(len(X), start + 128)
            xb = X[start:stop]
            power = np.mean(xb * xb, axis=(1, 2), keepdims=True)
            noise_std = np.sqrt(np.maximum(power / target, 1e-12)).astype(np.float32)
            noise = rng.standard_normal(size=xb.shape).astype(np.float32) * noise_std
            out[start:stop] = xb + noise
        meta["snr_db"] = float(spec["snr_db"])
        return out, meta
    if kind == "random_lead_dropout":
        rng = np.random.default_rng(int(spec["seed"]))
        out = X.copy()
        n_drop = int(spec["n_drop"])
        dropped = np.empty((len(X), n_drop), dtype=np.int16)
        for i in range(len(X)):
            leads = np.sort(rng.choice(X.shape[1], size=n_drop, replace=False)).astype(np.int16)
            out[i, leads, :] = 0.0
            dropped[i] = leads
        map_dir = EXPERIMENTAL_DIR / "robustness_perturbations"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / f"{spec['name']}_{stable_hash(spec)}_dropped_leads.npz"
        np.savez_compressed(
            map_path,
            dropped_leads=dropped,
            stress_json=np.asarray(json.dumps(spec, sort_keys=True)),
            record_count=np.asarray(len(X), dtype=np.int64),
        )
        meta["dropped_leads_shape"] = list(dropped.shape)
        meta["dropped_leads_sha256"] = hashlib.sha256(np.ascontiguousarray(dropped).view(np.uint8)).hexdigest()
        meta["lead_indices_file"] = str(map_path)
        meta["lead_indices_file_sha256"] = sha256_file(map_path)
        return out, meta
    if kind == "fixed_lead_dropout":
        leads = np.asarray(spec["lead_indices"], dtype=np.int64)
        out = X.copy()
        out[:, leads, :] = 0.0
        meta["lead_indices"] = leads.astype(int).tolist()
        return out, meta
    if kind == "resample_down_up":
        from scipy.signal import resample_poly

        out = np.empty_like(X, dtype=np.float32)
        for start in tqdm(range(0, len(X), 64), desc=f"perturb {spec['name']}"):
            stop = min(len(X), start + 64)
            down = resample_poly(X[start:stop], up=1, down=2, axis=-1).astype(np.float32)
            restored = resample_poly(down, up=2, down=1, axis=-1).astype(np.float32)
            if restored.shape[-1] < X.shape[-1]:
                pad = X.shape[-1] - restored.shape[-1]
                restored = np.pad(restored, ((0, 0), (0, 0), (0, pad)), mode="constant")
            out[start:stop] = restored[..., : X.shape[-1]]
        meta["source_hz"] = int(spec["source_hz"])
        meta["target_hz"] = int(spec["target_hz"])
        meta["length_restored_to"] = int(X.shape[-1])
        return out, meta
    raise ValueError(f"Unhandled perturbation kind: {kind}")


def robust_feature_cache_path(prefix: str, stress_name: str, stress_hash: str, n_records: int, record_fp: str) -> Path:
    cache_dir = Path(PATHS["cache_dir"]) / "revision_feature_cache"
    return cache_dir / f"{prefix}_{stress_name}_{stress_hash}_N{n_records}_R{record_fp}.npz"


def generate_minirocket_features(
    X: np.ndarray,
    *,
    stress_name: str,
    stress_hash: str,
    record_fp: str,
    batch_size: int,
    device_name: str,
    save_cache: bool,
) -> tuple[np.ndarray, dict]:
    cache_path = robust_feature_cache_path("robust_minirocket_raw", stress_name, stress_hash, len(X), record_fp)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            feats = np.asarray(payload["X"])
            cached_hash = str(payload["stress_hash"].item()) if "stress_hash" in payload.files else ""
        if feats.shape == (len(X), 20000) and cached_hash == stress_hash and np.isfinite(feats).all():
            print(f"Loaded perturbed MiniRocket cache: {cache_path}", flush=True)
            return feats.astype(np.float32), {
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
                "cache_hit": True,
                "storage_dtype": str(feats.dtype),
            }
        print(f"Perturbed MiniRocket cache mismatch, regenerating: {cache_path}", flush=True)

    device = torch.device(device_name)
    model = MiniRocketNative(c_in=X.shape[1], seq_len=X.shape[-1], num_kernels=10000, seed=42).to(device).eval()
    feats = []
    with torch.no_grad():
        for start in tqdm(range(0, len(X), batch_size), desc=f"MiniRocket {stress_name}", unit="batch"):
            xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            out = model(xb).detach().cpu().numpy()
            feats.append(out)
    X_rocket = np.vstack(feats).astype(np.float32)
    if X_rocket.shape != (len(X), 20000) or not np.isfinite(X_rocket).all():
        raise RuntimeError(f"Invalid perturbed MiniRocket output: {X_rocket.shape}")
    info = {
        "path": str(cache_path),
        "sha256": None,
        "cache_hit": False,
        "feature_device": device_name,
    }
    if save_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X_rocket.astype(np.float16),
            storage_dtype=np.asarray("float16"),
            consumer_dtype=np.asarray("float32"),
            stress_name=np.asarray(stress_name),
            stress_hash=np.asarray(stress_hash),
            record_order_fingerprint=np.asarray(record_fp),
        )
        info["sha256"] = sha256_file(cache_path)
        print(f"Saved perturbed MiniRocket cache: {cache_path}", flush=True)
    return X_rocket, info


def generate_hrv36_features(
    X: np.ndarray,
    X_raw_amp: np.ndarray,
    *,
    stress_name: str,
    stress_hash: str,
    record_fp: str,
    save_cache: bool,
) -> tuple[np.ndarray, dict]:
    cache_path = robust_feature_cache_path("robust_hrv36", stress_name, stress_hash, len(X), record_fp)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            feats = np.asarray(payload["X"])
            cached_hash = str(payload["stress_hash"].item()) if "stress_hash" in payload.files else ""
        if feats.shape == (len(X), int(CONFIG["hrv_dim"])) and cached_hash == stress_hash and np.isfinite(feats).all():
            print(f"Loaded perturbed HRV36 cache: {cache_path}", flush=True)
            return feats.astype(np.float32), {
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
                "cache_hit": True,
                "storage_dtype": str(feats.dtype),
            }
        print(f"Perturbed HRV36 cache mismatch, regenerating: {cache_path}", flush=True)

    feats = np.zeros((len(X), int(CONFIG["hrv_dim"])), dtype=np.float32)
    for i, sig in enumerate(tqdm(X, desc=f"HRV36 {stress_name}")):
        hrv = extract_hrv_features(sig)
        amp = extract_amplitude_features(X_raw_amp[i])
        gstat = extract_global_record_stats(sig)
        feats[i] = np.concatenate([hrv, amp, gstat]).astype(np.float32)
    if not np.isfinite(feats).all():
        raise RuntimeError("Perturbed HRV36 extraction produced non-finite values")
    info = {"path": str(cache_path), "sha256": None, "cache_hit": False}
    if save_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=feats.astype(np.float16),
            storage_dtype=np.asarray("float16"),
            consumer_dtype=np.asarray("float32"),
            stress_name=np.asarray(stress_name),
            stress_hash=np.asarray(stress_hash),
            record_order_fingerprint=np.asarray(record_fp),
            hrv_contract=np.asarray("hrv_and_global_recomputed_on_perturbed_signal_amp_uses_existing_pipeline_contract"),
        )
        info["sha256"] = sha256_file(cache_path)
        print(f"Saved perturbed HRV36 cache: {cache_path}", flush=True)
    return feats, info


def fit_or_load_minirocket_heads(
    *,
    X_clean: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    clean_cache_info: dict,
    args: argparse.Namespace,
) -> MiniRocketHeads:
    head_dir = EXPERIMENTAL_DIR / "robustness_minirocket_heads"
    manifest_path = MANIFEST_DIR / "robustness_minirocket_heads_manifest.json"
    clean_pred_path = PREDICTION_DIR / "robustness_minirocket_clean_ref_predictions.npz"
    params = {
        "protocol": MINIROCKET_HEAD_PROTOCOL,
        "clean_minirocket_cache_sha256": clean_cache_info.get("minirocket_cache_sha256"),
        "epochs": int(args.minirocket_epochs),
        "batch_size": int(args.minirocket_batch_size),
        "stats_batch_size": int(args.minirocket_stats_batch_size),
        "lr": float(args.minirocket_lr),
        "weight_decay": float(args.minirocket_weight_decay),
        "device": args.minirocket_device,
        "allow_tf32": bool(args.allow_tf32),
        "seed": int(args.seed),
        "n_records": int(len(y)),
        "n_classes": int(y.shape[1]),
        "config_hash": EVALUATION_CONFIG_HASH,
    }
    params_hash = stable_hash(params)
    if args.reuse_minirocket_heads and manifest_path.exists() and clean_pred_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("params_hash") == params_hash:
            head_paths = [head_dir / f"fold{fold['fold']}_head.pt" for fold in folds]
            if all(path.exists() for path in head_paths):
                with np.load(clean_pred_path, allow_pickle=False) as data:
                    clean_prob = np.asarray(data["y_prob"], dtype=np.float32)
                    fold_id = np.asarray(data["fold_id"], dtype=np.int16)
                if clean_prob.shape == y.shape and np.all(np.isfinite(clean_prob)):
                    print(f"Reusing MiniRocket robustness heads: {head_dir}", flush=True)
                    return MiniRocketHeads(head_dir, manifest, clean_prob, fold_id, manifest.get("fold_rows", []))

    import torch.nn as nn

    device_name = args.minirocket_device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    head_dir.mkdir(parents=True, exist_ok=True)
    clean_prob = np.full_like(y, np.nan, dtype=np.float32)
    fold_id = np.full(len(y), -1, dtype=np.int16)
    fold_rows: list[dict] = []
    mini = load_revision_module("10_minirocket_only_baseline.py", "_ecg_ramba_minirocket_baseline_for_robustness")

    for fold in folds:
        fold_num = int(fold["fold"])
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        rng = np.random.default_rng(args.seed + fold_num)
        torch.manual_seed(args.seed + fold_num)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + fold_num)
        print(f"MiniRocket clean head fold {fold_num}/5 | train={len(tr_idx)} | val={len(va_idx)}", flush=True)
        mean, std = mini.compute_train_standardization(
            X_clean,
            tr_idx,
            batch_size=args.minirocket_stats_batch_size,
        )
        pos = np.sum(y[tr_idx], axis=0).astype(np.float32)
        neg = float(len(tr_idx)) - pos
        pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0).astype(np.float32)
        model = nn.Linear(X_clean.shape[1], y.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.minirocket_lr, weight_decay=args.minirocket_weight_decay)
        model.train()
        for epoch in range(1, args.minirocket_epochs + 1):
            total_loss = 0.0
            total_seen = 0
            for batch_idx in mini.iter_index_batches(
                tr_idx,
                args.minirocket_batch_size,
                shuffle=True,
                rng=rng,
            ):
                xb = mini.prepare_feature_batch(X_clean, batch_idx, mean=mean, std=std)
                xb_t = torch.as_tensor(xb, dtype=torch.float32, device=device)
                yb_t = torch.as_tensor(y[batch_idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb_t), yb_t)
                loss.backward()
                optimizer.step()
                batch_n = int(len(batch_idx))
                total_loss += float(loss.detach().cpu()) * batch_n
                total_seen += batch_n
            print(f"  MiniRocket fold {fold_num}: epoch {epoch:02d}/{args.minirocket_epochs} loss={total_loss / max(total_seen, 1):.5f}", flush=True)
        model.eval()
        with torch.no_grad():
            for batch_idx in mini.iter_index_batches(va_idx, args.minirocket_batch_size, shuffle=False, rng=rng):
                xb = mini.prepare_feature_batch(X_clean, batch_idx, mean=mean, std=std)
                probs = torch.sigmoid(model(torch.as_tensor(xb, dtype=torch.float32, device=device))).detach().cpu().numpy()
                clean_prob[batch_idx] = probs.astype(np.float32)
        fold_id[va_idx] = fold_num
        head_path = head_dir / f"fold{fold_num}_head.pt"
        torch.save(
            {
                "fold": fold_num,
                "weight": model.weight.detach().cpu(),
                "bias": model.bias.detach().cpu(),
                "mean": torch.as_tensor(mean.astype(np.float32)),
                "std": torch.as_tensor(std.astype(np.float32)),
                "params_hash": params_hash,
                "protocol": MINIROCKET_HEAD_PROTOCOL,
            },
            head_path,
        )
        fold_rows.append(
            {
                "fold": fold_num,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "head_path": str(head_path),
                "head_sha256": sha256_file(head_path),
                "epochs": int(args.minirocket_epochs),
            }
        )
        del model, optimizer, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.any(fold_id < 0) or not np.isfinite(clean_prob).all():
        raise RuntimeError("MiniRocket robustness clean-head prediction coverage is incomplete.")
    np.savez_compressed(
        clean_pred_path,
        y_true=y.astype(np.float32),
        y_prob=np.clip(clean_prob, 0.0, 1.0).astype(np.float32),
        record_id=np.arange(len(y), dtype=np.int64),
        fold_id=fold_id.astype(np.int16),
        class_names=np.asarray(CLASSES),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(MINIROCKET_HEAD_PROTOCOL),
        feature_contract=np.asarray("minirocket_raw"),
        feature_preprocessing=np.asarray("fold_train_standardization"),
        threshold=np.asarray(float(args.threshold)),
        manuscript_ready=np.asarray(True),
    )
    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "protocol": MINIROCKET_HEAD_PROTOCOL,
        "params": params,
        "params_hash": params_hash,
        "head_dir": str(head_dir),
        "clean_prediction_file": str(clean_pred_path),
        "clean_prediction_sha256": sha256_file(clean_pred_path),
        "fold_rows": fold_rows,
    }
    save_json(manifest_path, json_safe(manifest))
    print(f"Wrote MiniRocket robustness heads manifest: {manifest_path}", flush=True)
    return MiniRocketHeads(head_dir, manifest, clean_prob, fold_id, fold_rows)


def predict_minirocket_heads(
    X_features: np.ndarray,
    y_shape: tuple[int, int],
    folds: list[dict[str, np.ndarray]],
    heads: MiniRocketHeads,
    *,
    batch_size: int,
    device_name: str,
) -> np.ndarray:
    import torch.nn.functional as F

    mini = load_revision_module("10_minirocket_only_baseline.py", "_ecg_ramba_minirocket_baseline_predict_for_robustness")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    out = np.full(y_shape, np.nan, dtype=np.float32)
    for fold in folds:
        fold_num = int(fold["fold"])
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        head = torch.load(heads.head_dir / f"fold{fold_num}_head.pt", map_location="cpu")
        weight = head["weight"].to(device=device, dtype=torch.float32)
        bias = head["bias"].to(device=device, dtype=torch.float32)
        mean = head["mean"].detach().cpu().numpy().astype(np.float32)
        std = head["std"].detach().cpu().numpy().astype(np.float32)
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for batch_idx in mini.iter_index_batches(va_idx, batch_size, shuffle=False, rng=rng):
                xb = mini.prepare_feature_batch(X_features, batch_idx, mean=mean, std=std)
                logits = F.linear(torch.as_tensor(xb, dtype=torch.float32, device=device), weight, bias)
                out[batch_idx] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        del weight, bias
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(out).all():
        raise RuntimeError("MiniRocket stress prediction contains missing/non-finite values.")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def predict_full_model(
    *,
    X: np.ndarray,
    X_rocket: np.ndarray,
    X_hrv: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
    stress_name: str,
    dataset_record_fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    gen = load_revision_module("01_generate_predictions.py", "_ecg_ramba_generate_predictions_for_robustness")
    n_records, n_classes = y.shape
    oof_prob = np.zeros((n_records, n_classes), dtype=np.float32)
    fold_id = np.full(n_records, -1, dtype=np.int16)
    slice_count = np.zeros(n_records, dtype=np.int16)
    fold_rows: list[dict] = []
    for fold in folds:
        fold_num = int(fold["fold"])
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        print(f"Full ECG-RAMBA stress={stress_name} fold {fold_num}/5 | val={len(va_idx)}", flush=True)
        ckpt_path = gen.checkpoint_path(fold_num, args.checkpoint_kind, allow_fallback=False)
        checkpoint_payload, checkpoint_meta = gen.load_checkpoint_payload(ckpt_path, args.checkpoint_kind)
        checkpoint_fp = checkpoint_meta.get("dataset_record_order_fingerprint")
        if args.limit_records == 0 and checkpoint_fp and checkpoint_fp != dataset_record_fingerprint:
            raise RuntimeError(
                f"Fold {fold_num} checkpoint record fingerprint {checkpoint_fp} "
                f"!= loaded dataset fingerprint {dataset_record_fingerprint}"
            )
        source_config_hash = checkpoint_meta["source_config_hash"]
        pca_path = gen.fold_pca_model_path(fold_num, tr_idx, source_config_hash)
        if not pca_path.exists():
            raise FileNotFoundError(
                f"Missing training-fold PCA object for fold {fold_num}: {pca_path}. "
                "Run the final_ema OOF export/training pipeline first; robustness must not fit a new PCA."
            )
        pca = joblib.load(pca_path)
        hydra_va = pca.transform(X_rocket[va_idx]).astype(np.float32)
        hydra_va_by_record = {int(record_idx): hydra_va[offset] for offset, record_idx in enumerate(va_idx)}
        xs, xh, xhr, rids, counts = gen.build_fold_slices(
            va_idx=va_idx,
            X=X,
            X_hrv=X_hrv,
            hydra_va_by_record=hydra_va_by_record,
        )
        for rid, count in counts.items():
            slice_count[int(rid)] = int(count)
        fold_summary = {
            "fold": fold_num,
            "validation_records": int(len(va_idx)),
            "n_slices": int(len(rids)),
            "pca_object_path": str(pca_path),
            "pca_object_sha256": sha256_file(pca_path),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path),
        }
        if len(rids) == 0:
            fold_rows.append(fold_summary)
            del checkpoint_payload
            continue
        model = gen.load_model_for_fold(
            fold_num,
            args.checkpoint_kind,
            checkpoint_file=ckpt_path,
            checkpoint_payload=checkpoint_payload,
        )
        del checkpoint_payload
        dataset = gen.ECGSliceDatasetInfer(xs, xh, xhr, rids)
        slice_prob, slice_rids, actual_batch, actual_workers, attempts = gen.infer_with_retries(
            model,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for rid in va_idx:
            mask = slice_rids == int(rid)
            if not np.any(mask):
                continue
            preds = slice_prob[mask]
            preds = preds[np.isfinite(preds).all(axis=1)]
            if len(preds):
                oof_prob[int(rid)] = power_mean(preds, q=float(CONFIG["power_mean_q"]), axis=0)
                fold_id[int(rid)] = fold_num
        fold_summary.update(
            {
                "n_predicted_records": int(np.sum(fold_id[va_idx] >= 0)),
                "actual_batch_size": int(actual_batch),
                "actual_num_workers": int(actual_workers),
                "inference_retry_attempts": attempts,
                "slice_prob_min": float(np.min(slice_prob)),
                "slice_prob_max": float(np.max(slice_prob)),
            }
        )
        fold_rows.append(fold_summary)
        del model, dataset, xs, xh, xhr, rids, slice_prob, slice_rids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    missing = int(np.sum(fold_id < 0))
    if missing:
        raise RuntimeError(f"Full model stress inference missing {missing} records.")
    return np.clip(oof_prob, 0.0, 1.0).astype(np.float32), slice_count, fold_rows


def write_prediction(path: Path, *, y: np.ndarray, prob: np.ndarray, fold_id: np.ndarray, stress: dict, model_label: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=y.astype(np.float32),
        y_prob=prob.astype(np.float32),
        record_id=np.arange(len(y), dtype=np.int64),
        fold_id=fold_id.astype(np.int16),
        class_names=np.asarray(CLASSES),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(PROTOCOL),
        model_label=np.asarray(model_label),
        stress_name=np.asarray(stress["name"]),
        stress_kind=np.asarray(stress["kind"]),
        stress_json=np.asarray(json.dumps(stress, sort_keys=True)),
        threshold=np.asarray(float(metadata["threshold"])),
        n_bins=np.asarray(int(metadata["n_bins"])),
        git_commit=np.asarray(git_commit()),
        created_utc=np.asarray(now_utc()),
        manuscript_ready=np.asarray(True),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Wrote predictions: {path}", flush=True)


def load_existing_prediction(path: Path, *, y: np.ndarray, fold_id: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "y_true" not in data.files or "y_prob" not in data.files or "fold_id" not in data.files:
            return None
        cached_y = np.asarray(data["y_true"], dtype=np.float32)
        cached_prob = np.asarray(data["y_prob"], dtype=np.float32)
        cached_fold = np.asarray(data["fold_id"], dtype=np.int16)
    if cached_y.shape == y.shape and cached_prob.shape == y.shape and np.array_equal(cached_y, y) and np.array_equal(cached_fold, fold_id):
        if np.isfinite(cached_prob).all():
            print(f"Reusing existing stress prediction: {path}", flush=True)
            return np.clip(cached_prob, 0.0, 1.0).astype(np.float32)
    return None


def paired_bootstrap_robustness(
    *,
    y: np.ndarray,
    clean_full: np.ndarray,
    stress_full: np.ndarray,
    clean_mini: np.ndarray,
    stress_mini: np.ndarray,
    spec: MetricSpec,
    n_boot: int,
    seed: int,
    stress_name: str,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = []
    for boot_idx in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            cf = metric_value(spec, y[idx], clean_full[idx])
            sf = metric_value(spec, y[idx], stress_full[idx])
            cm = metric_value(spec, y[idx], clean_mini[idx])
            sm = metric_value(spec, y[idx], stress_mini[idx])
        except ValueError:
            continue
        vals = [cf, sf, cm, sm]
        if not all(np.isfinite(v) for v in vals):
            continue
        full_deg = degradation(cf, sf, spec)
        mini_deg = degradation(cm, sm, spec)
        stressed_raw_diff = sf - sm
        stressed_adv = stressed_raw_diff if spec.higher_is_better else -stressed_raw_diff
        deg_adv = mini_deg - full_deg
        samples.append(
            {
                "stress_test": stress_name,
                "metric": spec.name,
                "bootstrap_index": int(boot_idx),
                "clean_full": cf,
                "stress_full": sf,
                "clean_minirocket": cm,
                "stress_minirocket": sm,
                "degradation_full": full_deg,
                "degradation_minirocket": mini_deg,
                "robustness_advantage_full_less_degradation": deg_adv,
                "stressed_advantage_full_over_minirocket": stressed_adv,
            }
        )
    if not samples:
        return {
            "degradation_advantage_mean": math.nan,
            "degradation_advantage_ci_low": math.nan,
            "degradation_advantage_ci_high": math.nan,
            "stressed_advantage_mean": math.nan,
            "stressed_advantage_ci_low": math.nan,
            "stressed_advantage_ci_high": math.nan,
            "p_value_degradation_two_sided": math.nan,
            "p_value_stressed_two_sided": math.nan,
            "n_boot_valid": 0,
        }, samples
    deg_adv = np.asarray([row["robustness_advantage_full_less_degradation"] for row in samples], dtype=np.float64)
    stressed_adv = np.asarray([row["stressed_advantage_full_over_minirocket"] for row in samples], dtype=np.float64)

    def p_two_sided(values: np.ndarray) -> float:
        p_lower = (float(np.sum(values <= 0.0)) + 1.0) / (len(values) + 1.0)
        p_upper = (float(np.sum(values >= 0.0)) + 1.0) / (len(values) + 1.0)
        return float(min(1.0, 2.0 * min(p_lower, p_upper)))

    deg_lo, deg_hi = np.quantile(deg_adv, [0.025, 0.975])
    stress_lo, stress_hi = np.quantile(stressed_adv, [0.025, 0.975])
    return {
        "degradation_advantage_mean": float(np.mean(deg_adv)),
        "degradation_advantage_ci_low": float(deg_lo),
        "degradation_advantage_ci_high": float(deg_hi),
        "stressed_advantage_mean": float(np.mean(stressed_adv)),
        "stressed_advantage_ci_low": float(stress_lo),
        "stressed_advantage_ci_high": float(stress_hi),
        "p_value_degradation_two_sided": p_two_sided(deg_adv),
        "p_value_stressed_two_sided": p_two_sided(stressed_adv),
        "n_boot_valid": int(len(samples)),
    }, samples


def interpretation(ci_low: float, ci_high: float, *, positive_label: str, negative_label: str) -> str:
    if ci_low > 0.0:
        return positive_label
    if ci_high < 0.0:
        return negative_label
    return "inconclusive"


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    print("=" * 80, flush=True)
    print("ROBUSTNESS STRESS TEST: FULL ECG-RAMBA VS MINIROCKET-ONLY", flush=True)
    print("=" * 80, flush=True)
    print(f"stress_tests={args.stress_tests}", flush=True)
    print(f"n_boot={args.n_boot} threshold={args.threshold} n_bins={args.n_bins}", flush=True)
    print(f"python={sys.version}", flush=True)
    print(f"platform={platform.platform()}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    full_clean = load_prediction_npz(args.full_clean_predictions, "Full clean")
    mini_clean_canonical = load_prediction_npz(args.minirocket_clean_predictions, "MiniRocket canonical clean")
    contract = validate_clean_prediction_contract(full_clean, mini_clean_canonical, args)
    y = full_clean["y_true"]
    fold_id = np.asarray(full_clean["fold_id"], dtype=np.int16)
    folds = fold_list_from_fold_id(fold_id)
    if args.limit_records > 0:
        keep = np.arange(min(args.limit_records, len(y)), dtype=np.int64)
        y = y[keep]
        fold_id = fold_id[keep]
        full_clean["y_prob"] = full_clean["y_prob"][keep]
        mini_clean_canonical["y_prob"] = mini_clean_canonical["y_prob"][keep]
        folds = fold_list_from_fold_id(fold_id)
        print(f"Debug limit active: {len(y)} records", flush=True)

    gen = load_revision_module("01_generate_predictions.py", "_ecg_ramba_generate_predictions_data_for_robustness")
    mini = load_revision_module("10_minirocket_only_baseline.py", "_ecg_ramba_minirocket_data_for_robustness")
    X, y_loaded, X_raw_amp, subjects = gen.prepare_clean_chapman(limit_records=args.limit_records)
    if y_loaded.shape != y.shape or not np.array_equal(y_loaded, y):
        raise ValueError("Loaded Chapman labels do not match frozen OOF y_true.")
    record_fp = record_order_fingerprint(subjects)
    expected_fp = contract["freeze_manifest"].get("dataset_record_order_fingerprint")
    if args.limit_records == 0 and expected_fp and record_fp != expected_fp:
        raise RuntimeError(f"Loaded Chapman record fingerprint {record_fp} != frozen {expected_fp}")
    if args.limit_records > 0:
        print(
            "Debug limit active: skipping full-record fingerprint equality check "
            f"(subset fingerprint={record_fp}, frozen full fingerprint={expected_fp}).",
            flush=True,
        )

    # Debug subsets still consume the manuscript full-record cache and slice it
    # inside the MiniRocket loader. This avoids looking for an artificial
    # N=<limit> cache keyed by the subset fingerprint.
    clean_cache_n_records = int(contract["freeze_manifest"].get("validated_records") or len(full_clean["record_id"]))
    clean_cache_fingerprint = expected_fp if args.limit_records > 0 and expected_fp else record_fp
    clean_X_rocket, clean_rocket_info = mini.load_minirocket_cache(
        n_records=clean_cache_n_records if args.limit_records > 0 else len(X),
        record_fingerprint=clean_cache_fingerprint,
        explicit_cache=None,
        allow_legacy_shape_cache=args.allow_legacy_shape_cache,
        limit_records=args.limit_records,
    )
    heads = fit_or_load_minirocket_heads(
        X_clean=clean_X_rocket,
        y=y,
        folds=folds,
        clean_cache_info=clean_rocket_info,
        args=args,
    )
    mini_clean_ref = heads.clean_prob

    specs = stress_specs(args.stress_tests.split(","), args.seed)
    rows: list[dict] = []
    sample_rows: list[dict] = []
    artifact_rows: list[dict] = []
    stress_meta_rows: list[dict] = []
    clean_metrics = {
        "full": multilabel_metrics(y, full_clean["y_prob"], threshold=args.threshold),
        "minirocket_canonical": multilabel_metrics(y, mini_clean_canonical["y_prob"], threshold=args.threshold),
        "minirocket_robustness_head_ref": multilabel_metrics(y, mini_clean_ref, threshold=args.threshold),
    }
    clean_calibration = {
        "full": calibration_summary(y, full_clean["y_prob"], n_bins=args.n_bins),
        "minirocket_canonical": calibration_summary(y, mini_clean_canonical["y_prob"], n_bins=args.n_bins),
        "minirocket_robustness_head_ref": calibration_summary(y, mini_clean_ref, n_bins=args.n_bins),
    }
    mini_clean_ref_delta = {
        "max_abs_probability_delta_vs_canonical": float(
            np.max(np.abs(mini_clean_ref - mini_clean_canonical["y_prob"]))
        ),
        "mean_abs_probability_delta_vs_canonical": float(
            np.mean(np.abs(mini_clean_ref - mini_clean_canonical["y_prob"]))
        ),
    }
    print(
        "MiniRocket clean reference delta vs canonical: "
        f"max_abs={mini_clean_ref_delta['max_abs_probability_delta_vs_canonical']:.6g} "
        f"mean_abs={mini_clean_ref_delta['mean_abs_probability_delta_vs_canonical']:.6g}",
        flush=True,
    )

    for stress in specs:
        stress_hash = stable_hash(stress)
        stress_name = stress["name"]
        print("\n" + "=" * 80, flush=True)
        print(f"Stress test: {stress_name} | hash={stress_hash}", flush=True)
        print("=" * 80, flush=True)
        full_pred_path = PREDICTION_DIR / f"robustness_full_{stress_name}_predictions.npz"
        mini_pred_path = PREDICTION_DIR / f"robustness_minirocket_{stress_name}_predictions.npz"
        full_prob = load_existing_prediction(full_pred_path, y=y, fold_id=fold_id) if args.reuse_existing else None
        mini_prob = load_existing_prediction(mini_pred_path, y=y, fold_id=fold_id) if args.reuse_existing else None
        feature_infos = {}
        perturb_meta = {}
        if full_prob is None or mini_prob is None:
            X_stress, perturb_meta = perturb_signals(X, stress)
            X_rocket, rocket_info = generate_minirocket_features(
                X_stress,
                stress_name=stress_name,
                stress_hash=stress_hash,
                record_fp=record_fp,
                batch_size=args.minirocket_feature_batch_size,
                device_name=args.minirocket_feature_device,
                save_cache=args.save_perturbed_caches,
            )
            feature_infos["minirocket"] = rocket_info
            if full_prob is None:
                X_hrv, hrv_info = generate_hrv36_features(
                    X_stress,
                    X_raw_amp,
                    stress_name=stress_name,
                    stress_hash=stress_hash,
                    record_fp=record_fp,
                    save_cache=args.save_perturbed_caches,
                )
                feature_infos["hrv36"] = hrv_info
                full_prob, slice_count, full_fold_rows = predict_full_model(
                    X=X_stress,
                    X_rocket=X_rocket,
                    X_hrv=X_hrv,
                    y=y,
                    folds=folds,
                    args=args,
                    stress_name=stress_name,
                    dataset_record_fingerprint=record_fp,
                )
                write_prediction(
                    full_pred_path,
                    y=y,
                    prob=full_prob,
                    fold_id=fold_id,
                    stress=stress,
                    model_label="Full ECG-RAMBA",
                    metadata={
                        "threshold": args.threshold,
                        "n_bins": args.n_bins,
                        "feature_infos": feature_infos,
                        "fold_rows": full_fold_rows,
                        "slice_count_min": int(slice_count.min()),
                        "slice_count_max": int(slice_count.max()),
                    },
                )
            if mini_prob is None:
                mini_prob = predict_minirocket_heads(
                    X_rocket,
                    y.shape,
                    folds,
                    heads,
                    batch_size=args.minirocket_batch_size,
                    device_name=args.minirocket_device,
                )
                write_prediction(
                    mini_pred_path,
                    y=y,
                    prob=mini_prob,
                    fold_id=fold_id,
                    stress=stress,
                    model_label="MiniRocket-only",
                    metadata={
                        "threshold": args.threshold,
                        "n_bins": args.n_bins,
                        "feature_infos": feature_infos,
                        "minirocket_heads_manifest": heads.manifest,
                    },
                )
            del X_stress
            gc.collect()

        artifact_rows.extend(
            [
                {
                    "stress_test": stress_name,
                    "artifact": "full_predictions",
                    "path": project_relative(full_pred_path),
                    "sha256": sha256_file(full_pred_path),
                },
                {
                    "stress_test": stress_name,
                    "artifact": "minirocket_predictions",
                    "path": project_relative(mini_pred_path),
                    "sha256": sha256_file(mini_pred_path),
                },
            ]
        )
        stress_meta_rows.append(
            {
                "stress_test": stress_name,
                "stress_hash": stress_hash,
                "stress_json": json.dumps(stress, sort_keys=True),
                "perturbation_metadata": json.dumps(perturb_meta, sort_keys=True),
            }
        )
        for metric_index, spec in enumerate(metric_specs(args.threshold, args.n_bins)):
            clean_full_value = metric_value(spec, y, full_clean["y_prob"])
            stress_full_value = metric_value(spec, y, full_prob)
            clean_mini_value = metric_value(spec, y, mini_clean_ref)
            stress_mini_value = metric_value(spec, y, mini_prob)
            deg_full = degradation(clean_full_value, stress_full_value, spec)
            deg_mini = degradation(clean_mini_value, stress_mini_value, spec)
            stressed_raw_diff = stress_full_value - stress_mini_value
            stressed_adv = stressed_raw_diff if spec.higher_is_better else -stressed_raw_diff
            deg_adv = deg_mini - deg_full
            print(
                f"{stress_name} {spec.name}: full_stress={stress_full_value:.6f} "
                f"mini_stress={stress_mini_value:.6f} degradation_adv_full={deg_adv:.6f}",
                flush=True,
            )
            ci, samples = paired_bootstrap_robustness(
                y=y,
                clean_full=full_clean["y_prob"],
                stress_full=full_prob,
                clean_mini=mini_clean_ref,
                stress_mini=mini_prob,
                spec=spec,
                n_boot=args.n_boot,
                seed=args.seed + 10007 * (metric_index + 1) + int(stress_hash[:6], 16),
                stress_name=stress_name,
            )
            row = {
                "stress_test": stress_name,
                "metric": spec.name,
                "metric_family": spec.family,
                "higher_is_better": bool(spec.higher_is_better),
                "clean_full": clean_full_value,
                "stress_full": stress_full_value,
                "degradation_full": deg_full,
                "clean_minirocket": clean_mini_value,
                "stress_minirocket": stress_mini_value,
                "degradation_minirocket": deg_mini,
                "stressed_advantage_full_over_minirocket": stressed_adv,
                "stressed_advantage_ci_low": ci["stressed_advantage_ci_low"],
                "stressed_advantage_ci_high": ci["stressed_advantage_ci_high"],
                "degradation_advantage_full_less_degradation": deg_adv,
                "degradation_advantage_ci_low": ci["degradation_advantage_ci_low"],
                "degradation_advantage_ci_high": ci["degradation_advantage_ci_high"],
                "p_value_degradation_two_sided": ci["p_value_degradation_two_sided"],
                "p_value_stressed_two_sided": ci["p_value_stressed_two_sided"],
                "n_boot_valid": ci["n_boot_valid"],
                "degradation_interpretation": interpretation(
                    ci["degradation_advantage_ci_low"],
                    ci["degradation_advantage_ci_high"],
                    positive_label="full_significantly_less_degraded",
                    negative_label="minirocket_significantly_less_degraded",
                ),
                "stressed_performance_interpretation": interpretation(
                    ci["stressed_advantage_ci_low"],
                    ci["stressed_advantage_ci_high"],
                    positive_label="full_significantly_better_under_stress",
                    negative_label="minirocket_significantly_better_under_stress",
                ),
            }
            rows.append(row)
            sample_rows.extend(samples)
        del full_prob, mini_prob
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    summary_csv = METRIC_DIR / "robustness_summary.csv"
    table_csv = TABLE_DIR / "table_robustness.csv"
    comparison_json = METRIC_DIR / "robustness_full_vs_minirocket_comparison.json"
    samples_csv = METRIC_DIR / "robustness_full_vs_minirocket_bootstrap_samples.csv"
    manifest_path = MANIFEST_DIR / "robustness_stress_manifest.json"
    stress_meta_csv = TABLE_DIR / "table_robustness_stress_metadata.csv"
    artifacts_csv = TABLE_DIR / "table_robustness_artifacts.csv"

    save_csv(summary_csv, rows)
    save_csv(table_csv, rows)
    save_csv(samples_csv, sample_rows)
    save_csv(stress_meta_csv, stress_meta_rows)
    save_csv(artifacts_csv, artifact_rows)

    payload = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "protocol": PROTOCOL,
        "claim_guidance": {
            "robustness": (
                "Claim Full ECG-RAMBA is more robust than MiniRocket-only only for metrics/perturbations "
                "where degradation_advantage_full_less_degradation has a positive paired CI."
            ),
            "ranking_warning": (
                "Stressed AUROC/AUPRC may still favor MiniRocket-only even when fixed-threshold/calibration "
                "degradation favors Full ECG-RAMBA."
            ),
        },
        "threshold": float(args.threshold),
        "n_bins": int(args.n_bins),
        "n_boot": int(args.n_boot),
        "shape": {"n_records": int(len(y)), "n_classes": int(y.shape[1])},
        "inputs": {
            "full_clean_predictions": {"path": str(full_clean["path"]), "sha256": full_clean["sha256"]},
            "minirocket_clean_predictions": {
                "path": str(mini_clean_canonical["path"]),
                "sha256": mini_clean_canonical["sha256"],
            },
            "contract": contract,
            "clean_minirocket_cache": clean_rocket_info,
            "minirocket_heads": heads.manifest,
        },
        "clean_metrics": clean_metrics,
        "clean_calibration": clean_calibration,
        "minirocket_clean_reference_delta": mini_clean_ref_delta,
        "stress_tests": specs,
        "metrics": rows,
        "outputs": {
            "summary_csv": str(summary_csv),
            "table_csv": str(table_csv),
            "bootstrap_samples_csv": str(samples_csv),
            "stress_metadata_csv": str(stress_meta_csv),
            "artifacts_csv": str(artifacts_csv),
            "comparison_json": str(comparison_json),
            "manifest": str(manifest_path),
        },
    }
    save_json(comparison_json, json_safe(payload))
    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "protocol": PROTOCOL,
        "args": vars(args),
        "artifact_sha256": {
            "summary_csv": sha256_file(summary_csv),
            "table_csv": sha256_file(table_csv),
            "bootstrap_samples_csv": sha256_file(samples_csv),
            "stress_metadata_csv": sha256_file(stress_meta_csv),
            "artifacts_csv": sha256_file(artifacts_csv),
            "comparison_json": sha256_file(comparison_json),
        },
        "artifact_rows": artifact_rows,
    }
    save_json(manifest_path, json_safe(manifest))
    print("Wrote:", comparison_json, flush=True)
    print("Wrote:", table_csv, flush=True)
    print("Wrote:", manifest_path, flush=True)
    print(json.dumps({"status": True, "stress_tests": [s["name"] for s in specs]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
