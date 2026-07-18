"""Train, resume, export, and compare matched ECG-RAMBA structural ablations.

A fresh matched Full control and each structural removal use the same folds,
fixed epoch budget, deterministic fold seed, minibatch-order generator, and
Full-reference initialization for every retained tensor. The canonical
manuscript Full run remains a frozen reproducibility anchor, while paired
ablation conclusions use the fresh matched Full run. Per-fold checkpoints and
OOF caches are resumable and authenticated by checkpoint SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    POWER_MEAN_IMPLEMENTATION,
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
from configs.config import CONFIG  # noqa: E402
from src.features import (  # noqa: E402
    HRV36_CHECKPOINT_SEMANTICS,
    checkpoint_compatible_hrv36_contract,
)
SCHEMA_VERSION = 3
PROTOCOL = "matched_retrained_structured_ablation_5fold_v3"
STRUCTURED_ABLATION_SPECS = {
    "full": {},
    "no_morphology": {"no_rocket": True},
    "no_rhythm": {"no_hrv": True},
    "no_fusion": {"no_fusion": True},
    "no_context_fusion": {"no_context_fusion": True},
}
DEFAULT_VARIANTS = ["full", "no_morphology", "no_rhythm", "no_context_fusion"]
VARIANT_LABELS = {
    "canonical_full": "Canonical manuscript Full ECG-RAMBA",
    "full": "Full ECG-RAMBA",
    "no_morphology": "No fixed-transform morphology/fusion interface",
    "no_rhythm": "No checkpoint-compatible RR/global-statistics interface",
    "no_fusion": "No cross-attention fusion",
    "no_context_fusion": "No context/fusion stack",
    "raw_mamba": "Raw Mamba",
}
VARIANT_CONTROLS = {
    "full": "Fresh matched Full control trained with the same fold-specific seed policy.",
    "no_morphology": (
        "Removes the fixed-transform morphology stream before retraining; raw ECG remains available, "
        "but the cross-attention interaction that requires that stream is also structurally unavailable."
    ),
    "no_rhythm": (
        "Removes the checkpoint-compatible five-RR-plus-six-global-statistics conditioning interface "
        "before retraining; raw ECG remains available."
    ),
    "no_fusion": "Replaces cross-attention fusion with sequence concatenation before retraining.",
    "no_context_fusion": "Jointly removes cross-attention, final Perceiver, and BiMamba context blocks before retraining.",
    "raw_mamba": "Independent raw-signal Mamba architecture comparator trained on the same folds.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--only-folds", default="", help="Comma-separated folds for resumable training.")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-oof", action="store_true")
    parser.add_argument("--aggregate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-complete", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument(
        "--canonical-model-dir",
        type=Path,
        default=None,
        help="Full-model directory containing folds.pkl and fold*_final_ema.pt.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "reports/revision/experimental/structured_ablation_checkpoints",
    )
    parser.add_argument(
        "--fold-cache-root",
        type=Path,
        default=PREDICTION_DIR / "structured_ablation_folds",
    )
    parser.add_argument(
        "--pca-cache-root",
        type=Path,
        default=PROJECT_ROOT
        / "reports/revision/experimental/structured_ablation_pca_models",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=METRIC_DIR / "structured_ablation_5fold_summary.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_structured_ablation_5fold.csv",
    )
    parser.add_argument(
        "--out-paired-table",
        type=Path,
        default=TABLE_DIR / "table_paired_structured_ablation_5fold.csv",
    )
    parser.add_argument(
        "--out-tex-table",
        type=Path,
        default=TABLE_DIR / "table_structured_ablation_5fold.tex",
    )
    parser.add_argument(
        "--out-status-table",
        type=Path,
        default=TABLE_DIR / "table_structured_ablation_checkpoint_status.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "structured_ablation_5fold_manifest.json",
    )
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "structured_ablation_metric_cache",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    path = resolve(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_tex_table(rows: list[dict], path: Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        (
            r"\caption{Matched five-fold structured-ablation results. Fresh Full and removal variants "
            r"use identical folds, training budget, fold seed policy, checkpoint-compatible rhythm "
            r"and global-statistic semantics, and fold-specific PCA artifacts. Canonical Full and Raw Mamba are shown as "
            r"separate anchors. Lower is better for Brier and ECE.}"
        ),
        r"\label{tab:matched_structured_ablation}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuration & PR-AUC & ROC-AUC & F1 & Brier & ECE \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
                latex_escape(row["variant_label"]),
                float(row["pr_auc_macro"]),
                float(row["roc_auc_macro"]),
                float(row["f1_macro"]),
                float(row["brier_macro"]),
                float(row["ece_macro"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_canonical_model_dir(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["ECG_RAMBA_MODEL_DIR"]) if os.environ.get("ECG_RAMBA_MODEL_DIR") else None,
        Path("/content/drive/MyDrive/ECG-Ramba/model_runs/ema_protocol_e20_v2"),
        PROJECT_ROOT / "model",
        PROJECT_ROOT / "models",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = resolve(candidate)
        if (candidate / "folds.pkl").exists() and all(
            (candidate / f"fold{fold}_final_ema.pt").exists() for fold in range(1, 6)
        ):
            return candidate
    raise FileNotFoundError("Could not locate canonical Full model directory with folds.pkl and five final_ema checkpoints")


def parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(variants) - set(STRUCTURED_ABLATION_SPECS))
    if invalid or not variants:
        raise ValueError(f"Invalid ablation variants {invalid}; expected named structural variants")
    if "full" not in variants:
        raise ValueError("Matched structural ablation requires a freshly retrained full control")
    return list(dict.fromkeys(variants))


def parse_folds(raw: str) -> list[int]:
    folds = [int(item.strip()) for item in raw.split(",") if item.strip()] if raw.strip() else [1, 2, 3, 4, 5]
    if not folds or not set(folds).issubset({1, 2, 3, 4, 5}):
        raise ValueError(f"Invalid folds: {folds}")
    return sorted(set(folds))


def canonical_checkpoint_contracts(
    canonical_model_dir: Path,
    canonical_folds: list[dict],
) -> dict[int, dict]:
    import torch

    protocol_keys = [
        "epochs",
        "loss_switch_epoch",
        "bce_reduction",
        "asymmetric_reduction",
        "scheduler",
        "lr_max",
        "lr_min",
        "ema_decay",
        "ema_scope",
        "amp_dtype",
        "model_selection",
    ]
    contracts = {}
    for fold in range(1, 6):
        path = canonical_model_dir / f"fold{fold}_final_ema.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        split = payload.get("split") or {}
        expected_train = np.asarray(canonical_folds[fold - 1]["tr_idx"], dtype=np.int64)
        expected_val = np.asarray(canonical_folds[fold - 1]["va_idx"], dtype=np.int64)
        train_hash = hashlib.sha256(np.ascontiguousarray(expected_train).tobytes()).hexdigest()[:16]
        val_hash = hashlib.sha256(np.ascontiguousarray(expected_val).tobytes()).hexdigest()[:16]
        if split.get("train_index_hash") != train_hash or split.get("val_index_hash") != val_hash:
            raise RuntimeError(f"Canonical Full checkpoint fold {fold} does not match folds.pkl")
        training_protocol = payload.get("training_protocol") or {}
        contracts[fold] = {
            "path": path,
            "sha256": sha256_file(path),
            "config_hash": payload.get("config_hash"),
            "dataset_record_order_fingerprint": payload.get("dataset_record_order_fingerprint"),
            "class_names": list(payload.get("class_names") or []),
            "aggregation": dict(payload.get("aggregation") or {}),
            "training_protocol": {key: training_protocol.get(key) for key in protocol_keys},
            # The original checkpoints predate this explicit field. The fixed
            # contract below records their audited legacy feature semantics.
            "feature_contract": {"hrv36": checkpoint_compatible_hrv36_contract()},
            "pca_explained_variance": float(
                payload.get("pca_explained_variance", math.nan)
            ),
            "train_index_hash": train_hash,
            "val_index_hash": val_hash,
        }
        del payload
    return contracts


def checkpoint_status(
    variant: str,
    model_dir: Path,
    canonical_contracts: dict[int, dict],
) -> list[dict]:
    import torch

    rows = []
    for fold in range(1, 6):
        path = model_dir / f"fold{fold}_final_ema.pt"
        canonical = canonical_contracts[fold]
        row = {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "fold": fold,
            "path": str(path),
            "exists": path.exists() and path.stat().st_size > 0,
            "contract_valid": False,
            "sha256": "",
            "issue": "missing_checkpoint",
            "expected_training_seed": int(CONFIG["seeds"][0]) + fold,
            "canonical_full_checkpoint_sha256": canonical["sha256"],
            "canonical_pca_explained_variance": canonical["pca_explained_variance"],
            "pca_path": "",
            "pca_sha256": "",
            "pca_explained_variance": math.nan,
            "initialization_policy": "",
            "initialization_reference_seed": -1,
            "initialization_group_sha256": {},
        }
        if row["exists"]:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                training_protocol = payload.get("training_protocol") or {}
                expected_protocol = canonical["training_protocol"]
                pca_contract = payload.get("pca_contract") or {}
                pca_path_raw = str(pca_contract.get("path") or "")
                pca_path = Path(pca_path_raw) if pca_path_raw else None
                pca_sha256 = str(pca_contract.get("sha256") or "")
                pca_variance = float(
                    pca_contract.get("explained_variance_ratio_sum", math.nan)
                )
                initialization_contract = payload.get("initialization_contract") or {}
                variant_group_hashes = dict(
                    initialization_contract.get("variant_group_sha256") or {}
                )
                reference_group_hashes = dict(
                    initialization_contract.get("reference_group_sha256") or {}
                )
                pca_file_valid = bool(
                    pca_path is not None
                    and pca_path.is_file()
                    and pca_path.stat().st_size > 0
                    and sha256_file(pca_path) == pca_sha256
                )
                checks = {
                    "fold": int(payload.get("fold", -1)) == fold,
                    "epoch": int(payload.get("epoch", -1)) == int(CONFIG["epochs"]),
                    "weights_kind": payload.get("weights_kind") == "ema",
                    "selection_rule": payload.get("selection_rule") == "fixed_final_epoch",
                    "ablation_variant": payload.get("ablation_variant") == variant,
                    "ablation_spec": dict(payload.get("ablation_spec", {})) == STRUCTURED_ABLATION_SPECS[variant],
                    "architecture_contract": payload.get("architecture_contract")
                    == "ecg_ramba_structured_ablation_v1",
                    "training_seed": int(payload.get("training_seed", -1))
                    == int(CONFIG["seeds"][0]) + fold,
                    "config_hash": payload.get("config_hash") == canonical["config_hash"],
                    "dataset_record_order_fingerprint": payload.get(
                        "dataset_record_order_fingerprint"
                    )
                    == canonical["dataset_record_order_fingerprint"],
                    "class_names": list(payload.get("class_names") or []) == canonical["class_names"],
                    "aggregation": dict(payload.get("aggregation") or {}) == canonical["aggregation"],
                    "training_protocol": {
                        key: training_protocol.get(key) for key in expected_protocol
                    }
                    == expected_protocol,
                    "feature_contract": payload.get("feature_contract")
                    == canonical["feature_contract"],
                    "pca_artifact": pca_file_valid,
                    "pca_training_scope": pca_contract.get("fit_scope")
                    == "training_records_of_this_outer_fold_only",
                    "pca_training_indices": pca_contract.get("train_index_hash")
                    == canonical["train_index_hash"],
                    "pca_output_dim": int(pca_contract.get("output_dim", -1))
                    == int(CONFIG["hydra_dim"]),
                    "pca_variance_internal": np.isfinite(pca_variance)
                    and np.isclose(
                        pca_variance,
                        float(payload.get("pca_explained_variance", math.nan)),
                        rtol=0.0,
                        atol=1e-12,
                    ),
                    "pca_variance_vs_canonical": np.isfinite(pca_variance)
                    and np.isclose(
                        pca_variance,
                        canonical["pca_explained_variance"],
                        rtol=0.0,
                        atol=1e-8,
                    ),
                    "training_indices": (payload.get("split") or {}).get("train_index_hash")
                    == canonical["train_index_hash"],
                    "validation_indices": (payload.get("split") or {}).get("val_index_hash")
                    == canonical["val_index_hash"],
                    "initialization_policy": initialization_contract.get("policy")
                    == "fold_seeded_full_reference_overlap_v1",
                    "initialization_reference_variant": initialization_contract.get(
                        "reference_variant"
                    )
                    == "full",
                    "initialization_reference_seed": int(
                        initialization_contract.get("reference_seed", -1)
                    )
                    == int(CONFIG["seeds"][0]) + fold,
                    "initialization_groups_present": bool(variant_group_hashes),
                    "initialization_matches_reference": bool(variant_group_hashes)
                    and all(
                        value == reference_group_hashes.get(group)
                        for group, value in variant_group_hashes.items()
                    ),
                }
                row["contract_valid"] = all(checks.values())
                row["issue"] = "" if row["contract_valid"] else json.dumps(
                    {key: value for key, value in checks.items() if not value}, sort_keys=True
                )
                row["sha256"] = sha256_file(path)
                row["pca_path"] = pca_path_raw
                row["pca_sha256"] = pca_sha256
                row["pca_explained_variance"] = pca_variance
                row["initialization_policy"] = str(
                    initialization_contract.get("policy") or ""
                )
                row["initialization_reference_seed"] = int(
                    initialization_contract.get("reference_seed", -1)
                )
                row["initialization_group_sha256"] = variant_group_hashes
            except Exception as exc:
                row["issue"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def enforce_shared_pca_contract(status_rows: list[dict]) -> None:
    """Invalidate fold checkpoints unless every available variant used one PCA artifact."""

    for fold in range(1, 6):
        fold_rows = [row for row in status_rows if int(row["fold"]) == fold]
        valid_rows = [row for row in fold_rows if row.get("contract_valid")]
        hashes = {str(row.get("pca_sha256", "")) for row in valid_rows}
        if len(hashes) <= 1:
            continue
        for row in valid_rows:
            row["contract_valid"] = False
            row["issue"] = json.dumps(
                {"shared_fold_pca_sha": False, "observed_hashes": sorted(hashes)},
                sort_keys=True,
            )


def enforce_shared_initialization_contract(status_rows: list[dict]) -> None:
    """Require every retained parameter group to match the same Full step-zero state."""

    for fold in range(1, 6):
        fold_rows = [row for row in status_rows if int(row["fold"]) == fold]
        full_rows = [
            row
            for row in fold_rows
            if row.get("variant") == "full" and row.get("contract_valid")
        ]
        if len(full_rows) != 1:
            continue
        full_hashes = dict(full_rows[0].get("initialization_group_sha256") or {})
        mismatches = {}
        for row in fold_rows:
            if not row.get("contract_valid"):
                continue
            variant_hashes = dict(row.get("initialization_group_sha256") or {})
            bad_groups = {
                group: {"full": full_hashes.get(group), "variant": value}
                for group, value in variant_hashes.items()
                if full_hashes.get(group) != value
            }
            if bad_groups:
                mismatches[str(row["variant"])] = bad_groups
        if not mismatches:
            continue
        for row in fold_rows:
            if row.get("contract_valid"):
                row["contract_valid"] = False
                row["issue"] = json.dumps(
                    {"shared_full_initialization": False, "mismatches": mismatches},
                    sort_keys=True,
                )


def run_training(
    variant: str,
    folds: list[int],
    model_dir: Path,
    folds_path: Path,
    pca_cache_root: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "ECG_RAMBA_MODEL_DIR": str(model_dir),
            "ECG_RAMBA_FOLDS_PATH": str(folds_path),
            "ECG_RAMBA_ABLATION_VARIANT": variant,
            "ECG_RAMBA_ONLY_FOLDS": ",".join(map(str, folds)),
            "ECG_RAMBA_RESUME_TRAINING": "1",
            "ECG_RAMBA_HRV_FEATURE_SEMANTICS": HRV36_CHECKPOINT_SEMANTICS,
            "ECG_RAMBA_PCA_CACHE_DIR": str(pca_cache_root),
        }
    )
    print(f"Training {variant} folds={folds} model_dir={model_dir}", flush=True)
    subprocess.run([sys.executable, "-u", "scripts/train.py"], cwd=PROJECT_ROOT, env=env, check=True)


def run_oof_export(args: argparse.Namespace, variant: str, model_dir: Path, folds_path: Path) -> None:
    stem = f"structured_ablation_{variant}"
    cache_dir = resolve(args.fold_cache_root) / variant
    command = [
        sys.executable,
        "-u",
        "scripts/revision/01_generate_predictions.py",
        "--dataset",
        "oof",
        "--checkpoint-kind",
        "final_ema",
        "--model-dir",
        str(model_dir),
        "--folds-path",
        str(folds_path),
        "--artifact-stem",
        stem,
        "--ablation-variant",
        variant,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--fold-cache-dir",
        str(cache_dir),
        "--resume-fold-cache",
    ]
    print("$ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env["ECG_RAMBA_PCA_CACHE_DIR"] = str(resolve(args.pca_cache_root))
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def load_prediction(path: Path) -> dict:
    path = resolve(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "fold_id", "class_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing {sorted(missing)}")
        payload = {
            "path": path,
            "sha256": sha256_file(path),
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "y_prob": np.asarray(data["y_prob"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]),
            "fold_id": np.asarray(data["fold_id"]),
            "class_names": np.asarray(data["class_names"]).astype(str),
            "ablation_variant": str(data["ablation_variant"].item())
            if "ablation_variant" in data
            else None,
            "protocol": str(data["protocol"].item()) if "protocol" in data else None,
            "checkpoint_fingerprints": json.loads(
                str(data["checkpoint_fingerprints_json"].item())
            )
            if "checkpoint_fingerprints_json" in data
            else [],
            "feature_contract": json.loads(str(data["feature_contract_json"].item()))
            if "feature_contract_json" in data
            else None,
        }
    if payload["y_true"].ndim != 2 or payload["y_prob"].shape != payload["y_true"].shape:
        raise ValueError(
            f"Prediction shape mismatch for {path}: {payload['y_true'].shape} vs "
            f"{payload['y_prob'].shape}"
        )
    if len(payload["record_id"]) != len(payload["y_true"]):
        raise ValueError(f"Prediction record length mismatch for {path}")
    return payload


def validate_pair(full: dict, other: dict, label: str) -> None:
    for key in ["y_true", "record_id", "fold_id", "class_names"]:
        if not np.array_equal(full[key], other[key]):
            raise ValueError(f"{label} {key} differs from canonical Full OOF")
    if not np.isfinite(other["y_prob"]).all():
        raise ValueError(f"{label} contains non-finite probabilities")


def validate_structured_prediction_provenance(
    variant: str,
    prediction: dict,
    status_rows: list[dict],
) -> dict:
    variant_rows = [row for row in status_rows if row.get("variant") == variant]
    if len(variant_rows) != 5 or not all(row.get("contract_valid") for row in variant_rows):
        raise RuntimeError(f"{variant} does not have five contract-valid matched checkpoints")
    expected_hashes = {
        int(row["fold"]): str(row["sha256"])
        for row in variant_rows
    }
    observed_hashes = {
        int(row.get("fold", -1)): str(row.get("sha256", ""))
        for row in prediction.get("checkpoint_fingerprints", [])
    }
    if prediction.get("ablation_variant") != variant:
        raise RuntimeError(
            f"{variant} prediction declares ablation_variant={prediction.get('ablation_variant')!r}"
        )
    expected_feature_contract = {"hrv36": checkpoint_compatible_hrv36_contract()}
    variant_token = "" if variant == "full" else f"_{variant}"
    expected_protocol = (
        f"fold_final_ema{variant_token}_{POWER_MEAN_IMPLEMENTATION}_"
        f"q{float(CONFIG['power_mean_q']):g}_threshold_0.5"
    )
    if prediction.get("protocol") != expected_protocol:
        raise RuntimeError(
            f"{variant} prediction protocol {prediction.get('protocol')!r} "
            f"does not match {expected_protocol!r}"
        )
    if prediction.get("feature_contract") != expected_feature_contract:
        raise RuntimeError(
            f"{variant} prediction does not declare the matched HRV36 feature contract"
        )
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"{variant} prediction checkpoint hashes do not match current checkpoints")

    manifest_path = MANIFEST_DIR / f"structured_ablation_{variant}_prediction_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_output = (manifest.get("outputs") or {}).get("prediction_file") or {}
    if (
        manifest.get("ablation_variant") != variant
        or manifest.get("protocol") != expected_protocol
        or manifest.get("architecture_contract") != "ecg_ramba_structured_ablation_v1"
        or manifest.get("feature_contract") != expected_feature_contract
        or prediction_output.get("sha256") != prediction["sha256"]
    ):
        raise RuntimeError(f"{variant} prediction run manifest does not authenticate the OOF artifact")
    manifest_hashes = {
        int(row.get("fold", -1)): str(row.get("sha256", ""))
        for row in ((manifest.get("inputs") or {}).get("checkpoints") or [])
    }
    if manifest_hashes != expected_hashes:
        raise RuntimeError(f"{variant} run manifest checkpoint hashes do not match current checkpoints")
    return {"path": rel(manifest_path), "sha256": sha256_file(manifest_path)}


def validate_raw_mamba_provenance(
    prediction: dict,
    canonical_full: dict,
    freeze_path: Path,
) -> dict:
    manifest_path = MANIFEST_DIR / "raw_mamba_baseline_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze_contract = manifest.get("freeze_contract") or {}
    artifact_sha = manifest.get("artifact_sha256") or {}
    if artifact_sha.get("predictions") != prediction["sha256"]:
        raise RuntimeError("Raw Mamba manifest does not authenticate the prediction artifact")
    if freeze_contract.get("oof_predictions_sha256") != canonical_full["sha256"]:
        raise RuntimeError("Raw Mamba was not evaluated against the current canonical Full OOF")
    if freeze_contract.get("freeze_manifest_sha256") != sha256_file(freeze_path):
        raise RuntimeError("Raw Mamba freeze-manifest contract is stale")
    return {"path": rel(manifest_path), "sha256": sha256_file(manifest_path)}


def metrics(y: np.ndarray, p: np.ndarray, threshold: float, n_bins: int) -> dict:
    discrimination = multilabel_metrics(y, p, threshold)
    calibration = calibration_summary(y, p, n_bins)
    return {
        "pr_auc_macro": discrimination["pr_auc_macro"],
        "roc_auc_macro": discrimination["roc_auc_macro"],
        "f1_macro": discrimination["f1_macro"],
        "brier_macro": calibration["brier_macro"],
        "ece_macro": calibration["ece_macro"],
    }


def metric_specs(threshold: float, n_bins: int) -> dict[str, tuple[Callable, bool]]:
    return {
        "pr_auc_macro": (macro_pr_auc, True),
        "roc_auc_macro": (macro_roc_auc, True),
        "f1_macro": (lambda y, p: multilabel_metrics(y, p, threshold)["f1_macro"], True),
        "brier_macro": (lambda y, p: calibration_summary(y, p, n_bins)["brier_macro"], False),
        "ece_macro": (lambda y, p: calibration_summary(y, p, n_bins)["ece_macro"], False),
    }


def paired_bootstrap(
    y: np.ndarray,
    full_prob: np.ndarray,
    other_prob: np.ndarray,
    fn: Callable,
    higher: bool,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    point_raw = fn(y, full_prob) - fn(y, other_prob)
    point = point_raw if higher else -point_raw
    for _ in range(n_boot):
        index = rng.integers(0, len(y), size=len(y))
        try:
            delta = fn(y[index], full_prob[index]) - fn(y[index], other_prob[index])
        except ValueError:
            continue
        delta = delta if higher else -delta
        if np.isfinite(delta):
            values.append(float(delta))
    if not values:
        return {"point": point, "ci_low": None, "ci_high": None, "n_boot_valid": 0}
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "point": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot_valid": len(values),
        "interpretation": (
            "full_significantly_better"
            if low > 0
            else "control_significantly_better"
            if high < 0
            else "inconclusive"
        ),
    }


def aggregate(args: argparse.Namespace, variants: list[str], status_rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    canonical_full = load_prediction(PREDICTION_DIR / "oof_final_ema_predictions.npz")
    structured_predictions = {}
    prediction_manifests = {}
    provenance_issues = {}
    for variant in variants:
        path = PREDICTION_DIR / f"structured_ablation_{variant}_predictions.npz"
        if path.exists():
            prediction = load_prediction(path)
            validate_pair(canonical_full, prediction, variant)
            try:
                prediction_manifests[variant] = validate_structured_prediction_provenance(
                    variant, prediction, status_rows
                )
            except Exception as exc:
                provenance_issues[variant] = f"{type(exc).__name__}: {exc}"
                continue
            structured_predictions[variant] = prediction
    matched_full = structured_predictions.get("full")
    controls = {
        name: prediction
        for name, prediction in structured_predictions.items()
        if name != "full"
    }
    raw_path = PREDICTION_DIR / "raw_mamba_oof_predictions.npz"
    if raw_path.exists():
        controls["raw_mamba"] = load_prediction(raw_path)
    for name, prediction in controls.items():
        validate_pair(canonical_full, prediction, name)

    freeze_path = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
    if not freeze_path.exists():
        raise FileNotFoundError(freeze_path)
    freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_artifacts = {
        str(row.get("path", "")): row
        for row in freeze_payload.get("artifacts", [])
        if isinstance(row, dict)
    }
    full_rel = rel(canonical_full["path"])
    if freeze_payload.get("status") != "frozen" or full_rel not in freeze_artifacts:
        raise RuntimeError("Canonical Full prediction is not authenticated by the freeze manifest")
    if freeze_artifacts[full_rel].get("sha256") != canonical_full["sha256"]:
        raise RuntimeError("Canonical Full prediction SHA differs from the freeze manifest")
    if "raw_mamba" in controls:
        try:
            prediction_manifests["raw_mamba"] = validate_raw_mamba_provenance(
                controls["raw_mamba"], canonical_full, freeze_path
            )
        except Exception as exc:
            provenance_issues["raw_mamba"] = f"{type(exc).__name__}: {exc}"
            del controls["raw_mamba"]

    table_rows = [
        {
            "variant": "canonical_full",
            "variant_label": VARIANT_LABELS["canonical_full"],
            "control": "frozen manuscript reproducibility anchor; not the paired ablation anchor",
            **metrics(
                canonical_full["y_true"],
                canonical_full["y_prob"],
                args.threshold,
                args.n_bins,
            ),
            "prediction_path": rel(canonical_full["path"]),
        }
    ]
    if matched_full is not None:
        table_rows.append(
            {
                "variant": "full",
                "variant_label": VARIANT_LABELS["full"],
                "control": VARIANT_CONTROLS["full"],
                **metrics(
                    matched_full["y_true"],
                    matched_full["y_prob"],
                    args.threshold,
                    args.n_bins,
                ),
                "prediction_path": rel(matched_full["path"]),
            }
        )
    for name, prediction in controls.items():
        table_rows.append(
            {
                "variant": name,
                "variant_label": VARIANT_LABELS[name],
                "control": VARIANT_CONTROLS[name],
                **metrics(
                    canonical_full["y_true"],
                    prediction["y_prob"],
                    args.threshold,
                    args.n_bins,
                ),
                "prediction_path": rel(prediction["path"]),
            }
        )

    cache_dir = resolve(args.metric_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    if matched_full is not None:
        paired_controls = controls
    else:
        paired_controls = {}
    for control_index, (name, prediction) in enumerate(paired_controls.items()):
        for metric_index, (metric_name, (fn, higher)) in enumerate(metric_specs(args.threshold, args.n_bins).items()):
            contract = {
                "schema_version": SCHEMA_VERSION,
                "full_sha256": matched_full["sha256"],
                "control_sha256": prediction["sha256"],
                "control": name,
                "metric": metric_name,
                "n_boot": args.n_boot,
                "seed": args.seed + control_index * 100 + metric_index,
            }
            cache_path = cache_dir / f"full_vs_{name}__{metric_name}.json"
            result = None
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("contract") == contract:
                    result = cached.get("result")
            if result is None:
                result = paired_bootstrap(
                    matched_full["y_true"],
                    matched_full["y_prob"],
                    prediction["y_prob"],
                    fn,
                    higher,
                    args.n_boot,
                    contract["seed"],
                )
                save_json(cache_path, {"contract": contract, "result": result})
            paired_rows.append(
                {
                    "comparison": f"full_vs_{name}",
                    "control": name,
                    "control_label": VARIANT_LABELS[name],
                    "metric": metric_name,
                    "higher_is_better": higher,
                    "improvement_full_over_control": result["point"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "n_boot_valid": result["n_boot_valid"],
                    "interpretation": result.get("interpretation"),
                }
            )
            print(f"full vs {name} {metric_name}: {result}", flush=True)
    complete_variants = sorted(set(structured_predictions) & set(variants))
    required_controls = set(variants) | {"raw_mamba"}
    complete_controls = set(structured_predictions) | ({"raw_mamba"} if "raw_mamba" in controls else set())
    missing_required_controls = sorted(required_controls - complete_controls)
    expected_paired_rows = len(paired_controls) * len(metric_specs(args.threshold, args.n_bins))
    paired_bootstrap_complete = (
        matched_full is not None
        and len(paired_rows) == expected_paired_rows
        and all(
            int(row.get("n_boot_valid", 0)) == int(args.n_boot)
            and row.get("ci_low") is not None
            and row.get("ci_high") is not None
            for row in paired_rows
        )
    )
    summary = {
        "status": (
            "complete"
            if not missing_required_controls and paired_bootstrap_complete
            else "incomplete"
        ),
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "protocol": PROTOCOL,
        "canonical_full_reused_as_reproducibility_anchor": True,
        "paired_analysis_uses_fresh_matched_full": matched_full is not None,
        "raw_mamba_control_included": "raw_mamba" in controls,
        "requested_variants": variants,
        "complete_variants": complete_variants,
        "missing_variants": sorted(set(variants) - set(structured_predictions)),
        "missing_required_controls": missing_required_controls,
        "training_contract": {
            "folds": [1, 2, 3, 4, 5],
            "epochs": int(CONFIG["epochs"]),
            "checkpoint": "fixed final epoch EMA",
            "threshold": args.threshold,
            "aggregation": "power mean q=3",
            "uncertainty": "paired record bootstrap",
            "n_boot": int(args.n_boot),
            "fold_seed_policy": "base seed plus one-based fold ID; identical across matched variants",
            "initialization_policy": (
                "Each retained tensor is copied from one fold-seeded Full reference before training; "
                "group hashes are checked across variants."
            ),
            "hrv36_feature_contract": checkpoint_compatible_hrv36_contract(),
            "training_variability_scope": (
                "one deterministic training seed per fold; paired bootstrap conditions on trained models "
                "and does not include retraining variability"
            ),
            "pca_contract": (
                "Each fold PCA is fitted on that fold's training records only. The PCA artifact SHA, "
                "training-index hash, retained variance, and output dimension must be identical across "
                "the fresh Full and all structured removals for the same fold."
            ),
        },
        "pca_contract_by_fold": {
            str(fold): {
                str(row["variant"]): {
                    "sha256": row.get("pca_sha256"),
                    "explained_variance_ratio_sum": row.get("pca_explained_variance"),
                    "contract_valid": bool(row.get("contract_valid")),
                }
                for row in status_rows
                if int(row["fold"]) == fold
            }
            for fold in range(1, 6)
        },
        "initialization_contract_by_fold": {
            str(fold): {
                str(row["variant"]): {
                    "policy": row.get("initialization_policy"),
                    "reference_seed": row.get("initialization_reference_seed"),
                    "group_sha256": row.get("initialization_group_sha256"),
                    "contract_valid": bool(row.get("contract_valid")),
                }
                for row in status_rows
                if int(row["fold"]) == fold
            }
            for fold in range(1, 6)
        },
        "claim_boundary": (
            "A significant Full-versus-removal difference supports within-architecture complementarity "
            "for that named control. The no-morphology control also removes its dependent cross-attention "
            "interaction, and the context/fusion control removes several modules jointly; neither isolates "
            "a single mechanism. No result establishes global superiority."
        ),
        "canonical_contract": {
            "full_prediction": {"path": full_rel, "sha256": canonical_full["sha256"]},
            "freeze_manifest": {"path": rel(freeze_path), "sha256": sha256_file(freeze_path)},
            "records": int(len(canonical_full["record_id"])),
            "folds": sorted(int(value) for value in np.unique(canonical_full["fold_id"])),
            "hrv36_feature_contract": checkpoint_compatible_hrv36_contract(),
        },
        "matched_full_contract": {
            "path": rel(matched_full["path"]) if matched_full is not None else None,
            "sha256": matched_full["sha256"] if matched_full is not None else None,
        },
        "control_predictions": {
            name: {"path": rel(prediction["path"]), "sha256": prediction["sha256"]}
            for name, prediction in controls.items()
        },
        "checkpoint_status": status_rows,
        "prediction_run_manifests": prediction_manifests,
        "provenance_issues": provenance_issues,
        "paired_bootstrap_contract": {
            "complete": paired_bootstrap_complete,
            "expected_rows": expected_paired_rows,
            "observed_rows": len(paired_rows),
            "required_valid_replicates_per_row": int(args.n_boot),
        },
    }
    return summary, table_rows, paired_rows


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    variants = parse_variants(args.variants)
    selected_folds = parse_folds(args.only_folds)
    canonical_model_dir = find_canonical_model_dir(args.canonical_model_dir)
    folds_path = canonical_model_dir / "folds.pkl"
    canonical_folds = joblib.load(folds_path)
    if len(canonical_folds) != 5:
        raise ValueError("Canonical fold contract must contain exactly five folds")
    canonical_contracts = canonical_checkpoint_contracts(canonical_model_dir, canonical_folds)
    model_root = resolve(args.model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    pca_cache_root = resolve(args.pca_cache_root)
    pca_cache_root.mkdir(parents=True, exist_ok=True)

    if args.run_training:
        for variant in variants:
            run_training(
                variant,
                selected_folds,
                model_root / variant,
                folds_path,
                pca_cache_root,
            )

    status_rows = []
    for variant in variants:
        status_rows.extend(checkpoint_status(variant, model_root / variant, canonical_contracts))
    enforce_shared_pca_contract(status_rows)
    enforce_shared_initialization_contract(status_rows)
    save_csv(resolve(args.out_status_table), status_rows)
    ready = {
        variant: all(row["contract_valid"] for row in status_rows if row["variant"] == variant)
        for variant in variants
    }
    print(f"Ablation checkpoint readiness: {ready}", flush=True)

    if args.run_oof:
        for variant in variants:
            if ready[variant]:
                run_oof_export(args, variant, model_root / variant, folds_path)
            else:
                print(f"Skipping OOF export for incomplete variant {variant}", flush=True)

    if args.aggregate:
        summary, table_rows, paired_rows = aggregate(args, variants, status_rows)
    else:
        summary = {
            "status": "checkpoint_audit_only",
            "schema_version": SCHEMA_VERSION,
            "created_utc": now_utc(),
            "protocol": PROTOCOL,
            "requested_variants": variants,
            "checkpoint_readiness": ready,
        }
        table_rows, paired_rows = [], []
    save_csv(resolve(args.out_table), table_rows)
    save_csv(resolve(args.out_paired_table), paired_rows)
    write_tex_table(table_rows, args.out_tex_table)
    save_json(resolve(args.out_summary), summary)
    manifest = {
        "status": summary["status"],
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "protocol": PROTOCOL,
        "git_commit": git_commit(),
        "canonical_model_dir": str(canonical_model_dir),
        "canonical_folds": {"path": str(folds_path), "sha256": sha256_file(folds_path)},
        "hrv36_feature_contract": checkpoint_compatible_hrv36_contract(),
        "canonical_full_checkpoints": {
            str(fold): {
                "path": str(contract["path"]),
                "sha256": contract["sha256"],
            }
            for fold, contract in canonical_contracts.items()
        },
        "model_root": str(model_root),
        "pca_cache_root": str(pca_cache_root),
        "outputs": {
            rel(path): sha256_file(resolve(path))
            for path in [
                args.out_summary,
                args.out_table,
                args.out_paired_table,
                args.out_status_table,
                args.out_tex_table,
            ]
        },
    }
    save_json(resolve(args.out_manifest), manifest)
    print(json.dumps({"status": summary["status"], "readiness": ready}, indent=2))
    if args.strict_complete and summary["status"] != "complete":
        raise RuntimeError(
            "Structured ablation evidence is incomplete: "
            f"{summary.get('missing_required_controls', summary.get('missing_variants'))}; "
            f"provenance_issues={summary.get('provenance_issues', {})}"
        )


if __name__ == "__main__":
    main()
