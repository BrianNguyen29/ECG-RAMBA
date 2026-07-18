
"""
ECG RAMBA - Training Pipeline
==================================================================================
Principles:
- Subject-aware CV (no leakage)
- Fold-wise PCA only
- BCE phase → ONE-TIME switch to FIXED Asymmetric Loss
- NO early stopping (full-epoch training)
- EMA for evaluation only (AFTER the BCE phase)
- Fixed threshold evaluation
- Quiet logging (NaN aggregated per fold only)
"""

import os
import sys
import gc
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from threading import Event, Thread
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import CLASSES, CONFIG, CONFIG_HASH, PATHS, DEVICE
from src.data_loader import load_chapman_multilabel
from src.features import (
    HRV36_CHECKPOINT_SEMANTICS,
    checkpoint_compatible_hrv36_contract,
    generate_raw_rocket_cache,
    generate_hrv_cache,
    fit_pca_on_train,
    apply_pca,
)
from src.model import ECGRambaV7Advanced, resolve_structured_ablation
from src.training_data import (
    LazyECGSliceDataset,
    audit_fold_splits,
    build_slice_index,
)
from src.aggregation import (
    POWER_MEAN_IMPLEMENTATION,
    aggregate_record_probabilities,
)
from src.utils import (
    compute_metrics,
    AsymmetricLossMultiLabel,
    EMA,
    set_seed,
)

# Suppress specific warning
warnings.filterwarnings("ignore", message="The least populated class in y")

FEATURE_CACHE_SCHEMA_VERSION = 2
AMP_DTYPE = (
    torch.bfloat16
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
)
AMP_DTYPE_NAME = (
    str(AMP_DTYPE).replace("torch.", "")
    if DEVICE == "cuda"
    else "float32"
)

def index_fingerprint(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_state_dict(model: torch.nn.Module) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


INITIALIZATION_GROUP_PREFIXES = {
    "raw_tokenizer": ("spatial_attn.", "tok."),
    "morphology_interface": ("rocket_perceiver.",),
    "rhythm_interface": ("hrv_proj.",),
    "cross_fusion": ("cross_fusion.",),
    "feature_projection": ("feature_proj.",),
    "final_perceiver": (
        "final_latents",
        "final_cross_attn.",
        "final_self_attn.",
        "final_norm1.",
        "final_norm2.",
        "final_norm3.",
        "final_ffn.",
    ),
    "context_backbone": ("layers.",),
    "normalization_head": ("norm.", "head."),
}


def initialization_group_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    """Hash initial tensors by architectural group without serializing a checkpoint."""

    output = {}
    for group, prefixes in INITIALIZATION_GROUP_PREFIXES.items():
        keys = sorted(
            key
            for key in state
            if any(key == prefix or key.startswith(prefix) for prefix in prefixes)
        )
        if not keys:
            continue
        digest = hashlib.sha256()
        for key in keys:
            tensor = state[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        output[group] = digest.hexdigest()
    return output


def initialize_matched_structured_model(
    *,
    cfg: dict,
    ablation_variant: str,
    ablation_spec: dict,
    fold_seed: int,
) -> tuple[torch.nn.Module, dict]:
    """Initialize every retained module from one fold-specific Full reference.

    Reusing only a scalar seed is insufficient for a structural ablation: removing
    a module changes RNG consumption and therefore changes the initialization of
    modules constructed later. Building a Full reference first and copying the
    exact overlapping state makes retained modules identical at step zero.
    """

    set_seed(fold_seed)
    reference = ECGRambaV7Advanced(cfg=cfg, ablation={})
    reference_state = cpu_state_dict(reference)
    reference_hashes = initialization_group_hashes(reference_state)

    if ablation_variant == "full":
        model = reference
        variant_state = reference_state
        loaded_keys = sorted(reference_state)
    else:
        model = ECGRambaV7Advanced(cfg=cfg, ablation=ablation_spec)
        variant_initial_state = model.state_dict()
        variant_keys = sorted(variant_initial_state)
        missing_from_reference = [key for key in variant_keys if key not in reference_state]
        shape_mismatches = [
            key
            for key in variant_keys
            if key in reference_state
            and tuple(variant_initial_state[key].shape) != tuple(reference_state[key].shape)
        ]
        if missing_from_reference or shape_mismatches:
            raise RuntimeError(
                "Structured ablation is not a strict subset of the Full initialization: "
                f"missing={missing_from_reference[:8]} shape_mismatch={shape_mismatches[:8]}"
            )
        overlap = {key: reference_state[key] for key in variant_keys}
        model.load_state_dict(overlap, strict=True)
        variant_state = cpu_state_dict(model)
        loaded_keys = variant_keys
        del reference, variant_initial_state

    variant_hashes = initialization_group_hashes(variant_state)
    for group, observed in variant_hashes.items():
        if observed != reference_hashes.get(group):
            raise RuntimeError(
                f"Matched initialization failed for group={group}: "
                f"variant={observed} reference={reference_hashes.get(group)}"
            )
    contract = {
        "policy": "fold_seeded_full_reference_overlap_v1",
        "reference_variant": "full",
        "reference_seed": int(fold_seed),
        "loaded_reference_key_count": len(loaded_keys),
        "variant_state_key_count": len(variant_state),
        "variant_group_sha256": variant_hashes,
        "reference_group_sha256": reference_hashes,
    }
    del reference_state, variant_state
    return model, contract


def float_metrics(metrics: dict | None) -> dict:
    if not metrics:
        return {}
    return {key: float(value) for key, value in metrics.items()}


def atomic_write_csv(frame: pd.DataFrame, path: str, *, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    frame.to_csv(tmp_path, index=index)
    os.replace(tmp_path, path)


def atomic_write_json(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def checkpoint_payload(
    *,
    model_state: dict,
    fold: int,
    epoch: int,
    weights_kind: str,
    selected_by_weights_kind: str,
    metrics: dict | None,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    pca_variance: float,
    pca_path: str,
    pca_sha256: str,
    dataset_record_order_fingerprint: str,
    selection_rule: str,
    metrics_weights_kind: str | None = None,
    selection_metrics: dict | None = None,
    ablation_variant: str = "full",
    ablation_spec: dict | None = None,
    initialization_contract: dict | None = None,
) -> dict:
    metrics = float_metrics(metrics)
    selection_metrics = float_metrics(selection_metrics)
    return {
        "model": model_state,
        "epoch": int(epoch),
        "fold": int(fold),
        "weights_kind": weights_kind,
        "selected_by_weights_kind": selected_by_weights_kind,
        "validation_weights_kind": metrics_weights_kind,
        "metrics_weights_kind": metrics_weights_kind,
        "f1_macro": float(metrics.get("f1_macro", np.nan)),
        "metrics": metrics,
        "selection_rule": selection_rule,
        "selection_metrics": selection_metrics,
        "training_seed": int(CONFIG["seeds"][0]) + int(fold),
        "config_hash": CONFIG_HASH,
        "dataset_record_order_fingerprint": dataset_record_order_fingerprint,
        "class_names": list(CLASSES),
        "aggregation": {
            "method": "power_mean",
            "q": float(CONFIG["power_mean_q"]),
            "implementation": POWER_MEAN_IMPLEMENTATION,
        },
        "training_protocol": {
            "epochs": int(CONFIG["epochs"]),
            "loss_switch_epoch": int(CONFIG["asym_start_epoch"]) + 1,
            "bce_reduction": "mean_over_batch_and_classes",
            "asymmetric_reduction": "sum_over_classes_mean_over_batch",
            "scheduler": "cosine_annealing",
            "lr_max": float(CONFIG["lr_max"]),
            "lr_min": float(CONFIG["lr_min"]),
            "ema_decay": float(CONFIG["ema_decay"]),
            "ema_scope": "trainable_parameters_only",
            "fold_seed_policy": "base_seed_plus_one_based_fold_id",
            "amp_dtype": AMP_DTYPE_NAME,
            "model_selection": (
                "fixed_final_epoch_for_manuscript_oof; "
                "best_ema_is_diagnostic_on_cv_validation"
            ),
        },
        "split": {
            "train_count": int(len(train_indices)),
            "val_count": int(len(val_indices)),
            "train_index_hash": index_fingerprint(train_indices),
            "val_index_hash": index_fingerprint(val_indices),
        },
        "pca_explained_variance": float(pca_variance),
        "pca_contract": {
            "path": str(pca_path),
            "sha256": str(pca_sha256),
            "explained_variance_ratio_sum": float(pca_variance),
            "fit_scope": "training_records_of_this_outer_fold_only",
            "train_index_hash": index_fingerprint(train_indices),
            "output_dim": int(CONFIG["hydra_dim"]),
        },
        "checkpoint_contract": "explicit_weights_kind_v2",
        "ablation_variant": str(ablation_variant),
        "ablation_spec": dict(ablation_spec or {}),
        "architecture_contract": "ecg_ramba_structured_ablation_v1",
        "initialization_contract": dict(initialization_contract or {}),
        "feature_contract": {
            "hrv36": checkpoint_compatible_hrv36_contract(),
        },
    }


def save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def save_fold_pca_model(fold: int, pca, train_indices: np.ndarray) -> str:
    path = fold_pca_model_path(fold, train_indices)
    tmp_path = f"{path}.tmp"
    joblib.dump(pca, tmp_path)
    os.replace(tmp_path, path)
    return path


def fold_pca_model_path(fold: int, train_indices: np.ndarray) -> str:
    cache_dir = os.environ.get("ECG_RAMBA_PCA_CACHE_DIR") or os.path.join(
        PATHS["cache_dir"], "revision_pca_models"
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        (
            f"fold{fold}_pca_v{FEATURE_CACHE_SCHEMA_VERSION}_{CONFIG_HASH}_"
            f"train{len(train_indices)}_{index_fingerprint(train_indices)}_"
            f"D{CONFIG['hydra_dim']}.joblib"
        ),
    )


def validate_fold_pca_model(pca, *, path: str, n_features: int) -> None:
    components = getattr(pca, "components_", None)
    expected_shape = (int(CONFIG["hydra_dim"]), int(n_features))
    if components is None or tuple(components.shape) != expected_shape:
        actual_shape = None if components is None else tuple(components.shape)
        raise ValueError(
            f"PCA cache has invalid components shape: {actual_shape}; "
            f"expected {expected_shape} at {path}"
        )
    explained = getattr(pca, "explained_variance_ratio_", None)
    if explained is None or len(explained) != int(CONFIG["hydra_dim"]):
        actual_len = None if explained is None else len(explained)
        raise ValueError(
            f"PCA cache has invalid explained_variance_ratio_ length: "
            f"{actual_len}; expected {CONFIG['hydra_dim']} at {path}"
        )


def env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def read_existing_rows(paths: list[str]) -> list[dict]:
    for path in paths:
        if os.path.exists(path):
            frame = pd.read_csv(path)
            return frame.to_dict(orient="records")
    return []


def make_loader(dataset, *, shuffle: bool, seed: int | None = None) -> DataLoader:
    workers = int(CONFIG["num_workers"])
    kwargs = {
        "batch_size": int(CONFIG["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": DEVICE == "cuda",
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def run_with_heartbeat(label: str, function, *, interval_seconds: int | None = None):
    """Run a blocking CPU task while emitting elapsed-time heartbeats."""
    if interval_seconds is None:
        interval_seconds = env_int("ECG_RAMBA_PCA_HEARTBEAT_SECONDS", 60)
    stop = Event()
    started = time.perf_counter()

    def heartbeat():
        while not stop.wait(interval_seconds):
            elapsed = (time.perf_counter() - started) / 60
            print(f"   ⏳ {label} still running | elapsed={elapsed:.1f} min", flush=True)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return function()
    finally:
        stop.set()
        thread.join(timeout=1)


# ==================================================================================
# MAIN TRAINING FUNCTION
# ==================================================================================

def main():
    base_training_seed = int(CONFIG["seeds"][0])
    set_seed(base_training_seed)
    hrv_feature_semantics = os.environ.get(
        "ECG_RAMBA_HRV_FEATURE_SEMANTICS",
        HRV36_CHECKPOINT_SEMANTICS,
    ).strip()
    if hrv_feature_semantics != HRV36_CHECKPOINT_SEMANTICS:
        raise ValueError(
            "Structured retraining must preserve the current checkpoint HRV36 semantics. "
            f"Requested {hrv_feature_semantics!r}; expected {HRV36_CHECKPOINT_SEMANTICS!r}. "
            "A corrected/full HRV schema requires a separately named model family and complete retraining."
        )
    ablation_variant, ablation_spec = resolve_structured_ablation(
        os.environ.get("ECG_RAMBA_ABLATION_VARIANT", "full")
    )
    only_folds_raw = os.environ.get("ECG_RAMBA_ONLY_FOLDS", "").strip()
    selected_folds = (
        {int(item.strip()) for item in only_folds_raw.split(",") if item.strip()}
        if only_folds_raw
        else set(range(1, int(CONFIG["n_folds"]) + 1))
    )
    expected_folds = set(range(1, int(CONFIG["n_folds"]) + 1))
    if not selected_folds or not selected_folds.issubset(expected_folds):
        raise ValueError(
            f"ECG_RAMBA_ONLY_FOLDS must be a non-empty subset of {sorted(expected_folds)}; "
            f"got {sorted(selected_folds)}"
        )

    # ==================================================================================
    # 🔧 RUN HEADER
    # ==================================================================================
    print("🔧 ECG RAMBA ")
    print(f"   D={CONFIG['d_model']} | Gamma={CONFIG['asym_gamma_neg']} | LR={CONFIG['lr_max']}")
    print(
        f"   BCE epochs={CONFIG['asym_start_epoch']} | "
        f"ASYM starts={CONFIG['asym_start_epoch'] + 1} | "
        f"Epochs={CONFIG['epochs']} | Folds={CONFIG['n_folds']}"
    )
    print(f"   Structural variant={ablation_variant} | spec={ablation_spec}")
    print(f"   HRV36 semantics={hrv_feature_semantics}")
    print(f"   Selected folds={sorted(selected_folds)} | model_dir={PATHS['model_dir']}")

    # ==================================================================================
    # 🛡️ PHASE 1 | LOAD DATA & AUTO-CLEAN
    # ==================================================================================
    print("\n" + "=" * 80)
    print("PHASE 1 | LOAD DATA & AUTO-CLEAN")
    print("=" * 80)

    X, y, X_raw_amp, subjects = load_chapman_multilabel()
    print(f"Original: {len(y)} records | {y.shape[1]} classes")
    if len(y) != 44186:
        raise RuntimeError(
            f"Canonical Chapman protocol requires 44186 records, found {len(y)}"
        )

    MIN_SAMPLES = 5
    class_counts = y.sum(axis=0)
    keep_mask = class_counts >= MIN_SAMPLES

    if not keep_mask.all():
        rare_classes = [
            CLASSES[index]
            for index in np.flatnonzero(~keep_mask)
        ]
        raise RuntimeError(
            "Fixed 27-class protocol cannot silently drop rare classes. "
            f"Classes below {MIN_SAMPLES} samples: {rare_classes}"
        )

    if y.shape[1] != len(CLASSES):
        raise RuntimeError(
            f"Label dimension {y.shape[1]} does not match fixed taxonomy {len(CLASSES)}"
        )
    print(f"✅ Cleaned → {len(y)} records | {y.shape[1]} classes")

    # ==================================================================================
    # 🧬 PHASE 2 | RAW FEATURE GENERATION (ANTI-LEAKAGE)
    # ==================================================================================
    print("\n" + "=" * 80)
    print("PHASE 2 | RAW FEATURE GENERATION (ANTI-LEAKAGE)")
    print("=" * 80)

    X_rocket_raw = generate_raw_rocket_cache(X, subjects)
    X_hrv_base = (
        generate_hrv_cache(
            X,
            X_raw_amp,
            subjects,
            semantics=hrv_feature_semantics,
        )
        if CONFIG["use_hrv"]
        else None
    )

    print(f"✅ RAW MiniRocket shape: {X_rocket_raw.shape}")
    if X_hrv_base is not None:
        print(f"✅ HRV feature shape  : {X_hrv_base.shape}")

    # ==================================================================================
    # 🚀 PHASE 3 | STRATIFIED GROUP K-FOLD TRAINING
    # ==================================================================================
    print("\n" + "=" * 80)
    print("PHASE 3 | TRAINING WITH FOLD-AWARE PCA")
    print("=" * 80)

    y_strat = y.sum(axis=1).clip(max=3).astype(int)
    sgkf = StratifiedGroupKFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seeds"][0],
    )

    os.makedirs(PATHS["model_dir"], exist_ok=True)
    epoch_log_partial_path = os.path.join(PATHS["model_dir"], "training_log_epochs.partial.csv")
    fold_log_partial_path = os.path.join(PATHS["model_dir"], "cv_results_clean_core.partial.csv")
    epoch_log_path = os.path.join(PATHS["model_dir"], "training_log_epochs.csv")
    fold_log_path = os.path.join(PATHS["model_dir"], "cv_results_clean_core.csv")
    resume_training = env_flag("ECG_RAMBA_RESUME_TRAINING", default=True)
    epoch_logs = (
        read_existing_rows([epoch_log_partial_path, epoch_log_path])
        if resume_training
        else []
    )
    fold_results = (
        read_existing_rows([fold_log_partial_path, fold_log_path])
        if resume_training
        else []
    )

    folds_path = os.path.join(PATHS["model_dir"], "folds.pkl")
    canonical_folds_path = os.environ.get("ECG_RAMBA_FOLDS_PATH", "").strip()
    reusable_folds_path = (
        folds_path
        if os.path.exists(folds_path)
        else canonical_folds_path
        if canonical_folds_path and os.path.exists(canonical_folds_path)
        else ""
    )
    if resume_training and reusable_folds_path:
        folds = joblib.load(reusable_folds_path)
        if os.path.abspath(reusable_folds_path) != os.path.abspath(folds_path):
            joblib.dump(folds, folds_path)
        print(f"✅ Reusing fold split provenance: {reusable_folds_path}")
    else:
        folds = [
            {
                "tr_idx": np.asarray(tr_idx, dtype=np.int64),
                "va_idx": np.asarray(va_idx, dtype=np.int64),
            }
            for tr_idx, va_idx in sgkf.split(X, y_strat, groups=subjects)
        ]
        joblib.dump(folds, folds_path)
        print(f"✅ Saved fold split provenance: {folds_path}")
    split_audit = audit_fold_splits(folds, subjects, n_records=len(y))
    split_audit_path = os.path.join(PATHS["model_dir"], "fold_split_audit.json")
    tmp_split_audit_path = f"{split_audit_path}.tmp"
    with open(tmp_split_audit_path, "w", encoding="utf-8") as handle:
        json.dump(split_audit, handle, indent=2, sort_keys=True)
    os.replace(tmp_split_audit_path, split_audit_path)
    print(f"✅ Saved fold split audit: {split_audit_path}")

    prevalence_rows = []
    for fold_num, split in enumerate(folds, start=1):
        for split_name, indices in (("train", split["tr_idx"]), ("validation", split["va_idx"])):
            prevalence = y[indices].mean(axis=0)
            positives = y[indices].sum(axis=0)
            prevalence_rows.extend(
                {
                    "fold": fold_num,
                    "split": split_name,
                    "class_name": class_name,
                    "n_records": int(len(indices)),
                    "positive_records": int(positives[class_index]),
                    "prevalence": float(prevalence[class_index]),
                }
                for class_index, class_name in enumerate(CLASSES)
            )
    prevalence_path = os.path.join(PATHS["model_dir"], "fold_label_prevalence.csv")
    atomic_write_csv(pd.DataFrame(prevalence_rows), prevalence_path)
    print(f"✅ Saved fold label prevalence audit: {prevalence_path}")

    completed_folds = set()
    resume_audit_rows = []
    for fold in range(1, int(CONFIG["n_folds"]) + 1):
        fold_train_indices = np.asarray(folds[fold - 1]["tr_idx"], dtype=np.int64)
        expected_pca_path = fold_pca_model_path(fold, fold_train_indices)
        expected_pca_sha256 = (
            file_sha256(expected_pca_path) if os.path.exists(expected_pca_path) else None
        )
        epoch_count = sum(
            int(row.get("fold", -1)) == fold
            for row in epoch_logs
        )
        has_result = any(
            int(row.get("fold", -1)) == fold
            for row in fold_results
        )
        checkpoint_paths = [
            os.path.join(PATHS["model_dir"], f"fold{fold}_{kind}.pt")
            for kind in ("best_ema", "best_raw", "final_ema", "final_raw")
        ]
        checkpoint_exists = {
            kind: os.path.exists(os.path.join(PATHS["model_dir"], f"fold{fold}_{kind}.pt"))
            for kind in ("best_ema", "best_raw", "final_ema", "final_raw")
        }
        checkpoint_metadata_valid = False
        checkpoint_metadata_error = ""
        final_ema_metadata = {}
        final_ema_path = os.path.join(
            PATHS["model_dir"],
            f"fold{fold}_final_ema.pt",
        )
        if os.path.exists(final_ema_path):
            try:
                payload = torch.load(final_ema_path, map_location="cpu", weights_only=False)
                final_ema_metadata = {
                    "config_hash": payload.get("config_hash"),
                    "dataset_record_order_fingerprint": payload.get(
                        "dataset_record_order_fingerprint"
                    ),
                    "weights_kind": payload.get("weights_kind"),
                    "selection_rule": payload.get("selection_rule"),
                    "epoch": int(payload.get("epoch", -1)),
                    "checkpoint_contract": payload.get("checkpoint_contract"),
                    "training_protocol": payload.get("training_protocol"),
                    "ablation_variant": payload.get("ablation_variant", "full"),
                    "ablation_spec": payload.get("ablation_spec", {}),
                    "training_seed": payload.get("training_seed"),
                    "feature_contract": payload.get("feature_contract"),
                    "pca_contract": payload.get("pca_contract"),
                    "initialization_contract": payload.get("initialization_contract"),
                }
                pca_contract = payload.get("pca_contract") or {}
                initialization_contract = payload.get("initialization_contract") or {}
                checkpoint_metadata_valid = (
                    payload.get("config_hash") == CONFIG_HASH
                    and payload.get("dataset_record_order_fingerprint")
                    == split_audit["record_order_fingerprint"]
                    and payload.get("weights_kind") == "ema"
                    and payload.get("selection_rule") == "fixed_final_epoch"
                    and int(payload.get("epoch", -1)) == int(CONFIG["epochs"])
                    and payload.get("ablation_variant", "full") == ablation_variant
                    and dict(payload.get("ablation_spec", {})) == ablation_spec
                    and int(payload.get("training_seed", -1)) == base_training_seed + fold
                    and (payload.get("feature_contract") or {}).get("hrv36")
                    == checkpoint_compatible_hrv36_contract()
                    and expected_pca_sha256 is not None
                    and pca_contract.get("sha256") == expected_pca_sha256
                    and pca_contract.get("train_index_hash")
                    == index_fingerprint(fold_train_indices)
                    and int(pca_contract.get("output_dim", -1)) == int(CONFIG["hydra_dim"])
                    and pca_contract.get("fit_scope")
                    == "training_records_of_this_outer_fold_only"
                    and np.isfinite(float(pca_contract.get("explained_variance_ratio_sum", np.nan)))
                    and initialization_contract.get("policy")
                    == "fold_seeded_full_reference_overlap_v1"
                    and initialization_contract.get("reference_variant") == "full"
                    and int(initialization_contract.get("reference_seed", -1))
                    == base_training_seed + fold
                    and bool(initialization_contract.get("variant_group_sha256"))
                    and all(
                        value == (initialization_contract.get("reference_group_sha256") or {}).get(group)
                        for group, value in (
                            initialization_contract.get("variant_group_sha256") or {}
                        ).items()
                    )
                )
                del payload
            except Exception as exc:
                checkpoint_metadata_error = repr(exc)
                print(
                    f"⚠️ Fold {fold} resume checkpoint validation failed: {exc}",
                    flush=True,
                )
        complete_for_resume = (
            epoch_count == int(CONFIG["epochs"])
            and has_result
            and all(os.path.exists(path) for path in checkpoint_paths)
            and checkpoint_metadata_valid
        )
        resume_audit_rows.append(
            {
                "fold": int(fold),
                "resume_training_enabled": bool(resume_training),
                "epoch_log_rows": int(epoch_count),
                "expected_epoch_rows": int(CONFIG["epochs"]),
                "has_cv_result_row": bool(has_result),
                "checkpoint_best_ema_exists": bool(checkpoint_exists["best_ema"]),
                "checkpoint_best_raw_exists": bool(checkpoint_exists["best_raw"]),
                "checkpoint_final_ema_exists": bool(checkpoint_exists["final_ema"]),
                "checkpoint_final_raw_exists": bool(checkpoint_exists["final_raw"]),
                "all_required_checkpoints_exist": bool(all(checkpoint_exists.values())),
                "final_ema_metadata_valid": bool(checkpoint_metadata_valid),
                "final_ema_config_hash": final_ema_metadata.get("config_hash"),
                "expected_config_hash": CONFIG_HASH,
                "final_ema_dataset_record_order_fingerprint": final_ema_metadata.get(
                    "dataset_record_order_fingerprint"
                ),
                "expected_dataset_record_order_fingerprint": split_audit[
                    "record_order_fingerprint"
                ],
                "final_ema_weights_kind": final_ema_metadata.get("weights_kind"),
                "final_ema_selection_rule": final_ema_metadata.get("selection_rule"),
                "final_ema_epoch": final_ema_metadata.get("epoch"),
                "expected_final_epoch": int(CONFIG["epochs"]),
                "final_ema_ablation_variant": final_ema_metadata.get("ablation_variant"),
                "expected_ablation_variant": ablation_variant,
                "final_ema_ablation_spec": final_ema_metadata.get("ablation_spec"),
                "expected_ablation_spec": json.dumps(ablation_spec, sort_keys=True),
                "final_ema_training_seed": final_ema_metadata.get("training_seed"),
                "expected_training_seed": base_training_seed + fold,
                "final_ema_feature_contract": json.dumps(
                    final_ema_metadata.get("feature_contract"), sort_keys=True
                ),
                "expected_feature_contract": json.dumps(
                    {"hrv36": checkpoint_compatible_hrv36_contract()}, sort_keys=True
                ),
                "final_ema_pca_contract": json.dumps(
                    final_ema_metadata.get("pca_contract"), sort_keys=True
                ),
                "final_ema_initialization_contract": json.dumps(
                    final_ema_metadata.get("initialization_contract"), sort_keys=True
                ),
                "expected_pca_path": expected_pca_path,
                "expected_pca_sha256": expected_pca_sha256,
                "checkpoint_contract": final_ema_metadata.get("checkpoint_contract"),
                "training_protocol": final_ema_metadata.get("training_protocol"),
                "metadata_error": checkpoint_metadata_error,
                "complete_for_resume_skip": bool(complete_for_resume),
            }
        )
        if complete_for_resume:
            completed_folds.add(fold)
    resume_audit_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": PATHS["model_dir"],
        "resume_training_enabled": bool(resume_training),
        "config_hash": CONFIG_HASH,
        "expected_epochs": int(CONFIG["epochs"]),
        "expected_folds": int(CONFIG["n_folds"]),
        "selected_folds": sorted(selected_folds),
        "ablation_variant": ablation_variant,
        "ablation_spec": ablation_spec,
        "dataset_record_order_fingerprint": split_audit["record_order_fingerprint"],
        "feature_contract": {"hrv36": checkpoint_compatible_hrv36_contract()},
        "completed_folds_reused": sorted(int(fold) for fold in completed_folds),
        "skip_policy": (
            "A fold may be skipped only when epoch logs cover all configured epochs, "
            "the CV result row exists, all four explicit checkpoints exist, and "
            "fold*_final_ema.pt metadata matches config hash, dataset fingerprint, "
            "weights_kind=ema, selection_rule=fixed_final_epoch, final epoch, and the "
            "checkpoint-compatible HRV36 feature contract. The declared fold-PCA SHA, "
            "training-index hash, and output dimension must also match the current "
            "training-fold-only PCA artifact. The initialization contract must attest "
            "that every retained module was copied from the fold-seeded Full reference."
        ),
        "folds": resume_audit_rows,
    }
    resume_audit_json_path = os.path.join(PATHS["model_dir"], "resume_integrity_audit.json")
    resume_audit_csv_path = os.path.join(PATHS["model_dir"], "resume_integrity_audit.csv")
    atomic_write_json(resume_audit_payload, resume_audit_json_path)
    atomic_write_csv(pd.DataFrame(resume_audit_rows), resume_audit_csv_path)
    print(f"🧾 Wrote resume integrity audit: {resume_audit_json_path}")
    print(f"🧾 Wrote resume integrity audit table: {resume_audit_csv_path}")
    if completed_folds:
        print(
            f"♻️ Resume enabled; completed folds will be reused: "
            f"{sorted(completed_folds)}"
        )

    for fold, split in enumerate(folds, start=1):
        if fold not in selected_folds:
            print(f"\n⏭️ FOLD {fold}/{CONFIG['n_folds']} not selected for this invocation")
            continue
        if fold in completed_folds:
            print(f"\n♻️ FOLD {fold}/{CONFIG['n_folds']} already complete; skipping")
            continue
        epoch_logs = [
            row for row in epoch_logs
            if int(row.get("fold", -1)) != fold
        ]
        fold_results = [
            row for row in fold_results
            if int(row.get("fold", -1)) != fold
        ]
        tr_idx = split["tr_idx"]
        va_idx = split["va_idx"]
        fold_training_seed = base_training_seed + fold
        # A fold-specific seed makes interrupted/resumed single-fold runs reproduce
        # the same initialization and sample order across structural variants.
        set_seed(fold_training_seed)
        fold_started = time.perf_counter()
        rocket_train_gib = (
            len(tr_idx) * X_rocket_raw.shape[1] * X_rocket_raw.dtype.itemsize
        ) / (1024 ** 3)
        print(
            f"\n⚡ FOLD {fold}/{CONFIG['n_folds']} | training_seed={fold_training_seed}",
            flush=True,
        )
        print(
            f"   ⏳ PCA stage starting on CPU | train={len(tr_idx)} | "
            f"features={X_rocket_raw.shape[1]} | input_copy≈{rocket_train_gib:.2f} GiB",
            flush=True,
        )

        pca_path = fold_pca_model_path(fold, tr_idx)
        pca_started = time.perf_counter()
        if resume_training and os.path.exists(pca_path):
            pca_size_mib = os.path.getsize(pca_path) / (1024 ** 2)
            print(
                f"   ♻️ Fold PCA cache candidate: {pca_path} "
                f"({pca_size_mib:.1f} MiB)",
                flush=True,
            )
            try:
                pca = run_with_heartbeat(
                    f"Fold {fold} PCA cache load",
                    lambda: joblib.load(pca_path),
                )
                validate_fold_pca_model(
                    pca,
                    path=pca_path,
                    n_features=X_rocket_raw.shape[1],
                )
                print(
                    f"   ✅ Loaded fold PCA object in "
                    f"{(time.perf_counter() - pca_started) / 60:.1f} min",
                    flush=True,
                )
            except Exception as exc:
                corrupt_path = f"{pca_path}.corrupt_{int(time.time())}"
                try:
                    os.replace(pca_path, corrupt_path)
                    print(
                        f"   ⚠️ PCA cache load failed and was quarantined: "
                        f"{corrupt_path}",
                        flush=True,
                    )
                except OSError:
                    print(
                        f"   ⚠️ PCA cache load failed but could not be "
                        f"quarantined: {pca_path}",
                        flush=True,
                    )
                print(f"   ⚠️ PCA cache error: {exc}. Re-fitting fold PCA.", flush=True)
                pca_started = time.perf_counter()
                print("   ⏳ Materializing fold train MiniRocket matrix...", flush=True)
                X_train_pca = np.ascontiguousarray(X_rocket_raw[tr_idx], dtype=np.float32)
                print(
                    f"   ⏳ Fitting randomized PCA: "
                    f"{X_train_pca.shape[0]} x {X_train_pca.shape[1]} "
                    f"-> D={CONFIG['hydra_dim']}",
                    flush=True,
                )
                pca = run_with_heartbeat(
                    f"Fold {fold} PCA fit",
                    lambda: fit_pca_on_train(X_train_pca, CONFIG["hydra_dim"]),
                )
                del X_train_pca
                print(f"   💾 Saving fold PCA object: {pca_path}", flush=True)
                run_with_heartbeat(
                    f"Fold {fold} PCA cache save",
                    lambda: save_fold_pca_model(fold, pca, tr_idx),
                )
                print(
                    f"   ✅ PCA fit complete in "
                    f"{(time.perf_counter() - pca_started) / 60:.1f} min",
                    flush=True,
                )
        else:
            print(f"   🧭 Fold PCA cache path: {pca_path}", flush=True)
            print("   ⏳ Materializing fold train MiniRocket matrix...", flush=True)
            X_train_pca = np.ascontiguousarray(X_rocket_raw[tr_idx], dtype=np.float32)
            print(
                f"   ⏳ Fitting randomized PCA: "
                f"{X_train_pca.shape[0]} x {X_train_pca.shape[1]} "
                f"-> D={CONFIG['hydra_dim']}",
                flush=True,
            )
            pca = run_with_heartbeat(
                f"Fold {fold} PCA fit",
                lambda: fit_pca_on_train(X_train_pca, CONFIG["hydra_dim"]),
            )
            del X_train_pca
            print(f"   💾 Saving fold PCA object: {pca_path}", flush=True)
            pca_path = run_with_heartbeat(
                f"Fold {fold} PCA cache save",
                lambda: save_fold_pca_model(fold, pca, tr_idx),
            )
            print(
                f"   ✅ PCA fit complete in "
                f"{(time.perf_counter() - pca_started) / 60:.1f} min",
                flush=True,
            )
        print("   ⏳ Transforming train/validation MiniRocket features...", flush=True)
        hydra_tr = apply_pca(pca, X_rocket_raw[tr_idx])
        hydra_va = apply_pca(pca, X_rocket_raw[va_idx])
        pca_variance = float(pca.explained_variance_ratio_.sum())
        pca_sha256 = file_sha256(pca_path)
        print(f"   🛡️ PCA variance retained: {pca_variance:.3f}")
        print(f"   💾 Fold PCA object: {pca_path} | sha256={pca_sha256[:12]}")

        hydra_by_record = np.zeros((len(X), CONFIG["hydra_dim"]), dtype=np.float32)
        hydra_by_record[tr_idx] = hydra_tr
        hydra_by_record[va_idx] = hydra_va
        del hydra_tr, hydra_va

        print("   ⏳ Building lazy train/validation slice indices...", flush=True)
        rid_tr, start_tr, pos_tr, skipped_tr = build_slice_index(
            tr_idx,
            X,
            slice_length=CONFIG["slice_length"],
            slice_stride=CONFIG["slice_stride"],
            max_slices_per_record=CONFIG["max_slices_per_record"],
        )
        rid_va, start_va, pos_va, skipped_va = build_slice_index(
            va_idx,
            X,
            slice_length=CONFIG["slice_length"],
            slice_stride=CONFIG["slice_stride"],
            max_slices_per_record=CONFIG["max_slices_per_record"],
        )

        n_val_records_expected = len(np.unique(va_idx))
        n_val_records_with_slice = len(np.unique(rid_va))
        total_val_slices = len(rid_va)
        avg_slices_per_record = total_val_slices / max(n_val_records_with_slice, 1)

        print(
          f"   🧪 Fold {fold} | EARLY CHECK | "
          f"val_records_with_slice={n_val_records_with_slice}/{n_val_records_expected} | "
          f"total_val_slices={total_val_slices} | "
          f"avg_slices/record={avg_slices_per_record:.2f} | "
          f"skipped_no_slice={skipped_va}"
        )
        if skipped_va or n_val_records_with_slice != n_val_records_expected:
            raise RuntimeError(
                f"Fold {fold} validation slicing is incomplete: "
                f"records_with_slice={n_val_records_with_slice}/"
                f"{n_val_records_expected}, skipped={skipped_va}"
            )

        hrv_by_record = (
            X_hrv_base
            if X_hrv_base is not None
            else np.zeros((len(X), 1), dtype=np.float32)
        )
        train_loader = make_loader(
            LazyECGSliceDataset(
                X,
                hydra_by_record,
                hrv_by_record,
                y,
                rid_tr,
                start_tr,
                pos_tr,
                slice_length=CONFIG["slice_length"],
            ),
            shuffle=True,
            seed=fold_training_seed,
        )

        val_loader = make_loader(
            LazyECGSliceDataset(
                X,
                hydra_by_record,
                hrv_by_record,
                y,
                rid_va,
                start_va,
                pos_va,
                slice_length=CONFIG["slice_length"],
            ),
            shuffle=False,
            seed=fold_training_seed + 100_000,
        )

        model, initialization_contract = initialize_matched_structured_model(
            cfg=CONFIG,
            ablation_variant=ablation_variant,
            ablation_spec=ablation_spec,
            fold_seed=fold_training_seed,
        )
        model = model.to(DEVICE)
        # Keep model stochasticity reproducible independently of constructor RNG use.
        set_seed(fold_training_seed)
        print(
            "   Matched initialization: "
            f"policy={initialization_contract['policy']} | "
            f"groups={sorted(initialization_contract['variant_group_sha256'])}",
            flush=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr_max"],
            weight_decay=CONFIG["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["epochs"],
            eta_min=CONFIG["lr_min"],
        )

        bce_criterion = torch.nn.BCEWithLogitsLoss()
        asym_criterion = AsymmetricLossMultiLabel(
            gamma_neg=CONFIG["asym_gamma_neg"],
            gamma_pos=CONFIG["asym_gamma_pos"],
            clip=CONFIG["asym_clip"],
        )

        ema = EMA(model, decay=CONFIG["ema_decay"])

        best_ema_f1, best_ema_epoch = -np.inf, -1
        best_ema_metrics = None
        best_raw_warmup_f1 = -np.inf
        fold_skipped_nan = 0
        last_eval_metrics = None
        last_eval_weights_kind = None
        selected_state = None
        selected_payload = None
        raw_payload = None
        final_raw_payload = None
        final_ema_payload = None

        best_ema_ckpt_path = os.path.join(PATHS["model_dir"], f"fold{fold}_best_ema.pt")
        best_raw_ckpt_path = os.path.join(PATHS["model_dir"], f"fold{fold}_best_raw.pt")
        final_ema_ckpt_path = os.path.join(PATHS["model_dir"], f"fold{fold}_final_ema.pt")
        final_raw_ckpt_path = os.path.join(PATHS["model_dir"], f"fold{fold}_final_raw.pt")

        for epoch in range(CONFIG["epochs"]):
            model.train()
            loss_sum = 0.0
            lr = float(optimizer.param_groups[0]["lr"])

            use_bce = epoch < CONFIG["asym_start_epoch"]
            loss_name = "BCE" if use_bce else "ASYM"

            for x, xh, xhr, tgt, _, _ in train_loader:
                x, xh, xhr, tgt = x.to(DEVICE), xh.to(DEVICE), xhr.to(DEVICE), tgt.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed Precision usually requires CUDA
                if DEVICE == 'cuda':
                    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                        logits = model(x, xh, xhr)
                        loss = bce_criterion(logits, tgt) if use_bce else asym_criterion(logits, tgt)
                else:
                    logits = model(x, xh, xhr)
                    loss = bce_criterion(logits, tgt) if use_bce else asym_criterion(logits, tgt)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at fold={fold}, epoch={epoch + 1}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                optimizer.step()
                ema.update(model)
                loss_sum += loss.item()

            scheduler.step()

            model.eval()
            eval_weights_kind = (
                "ema"
                if CONFIG.get("use_ema", True) and epoch >= CONFIG["asym_start_epoch"]
                else "raw"
            )
            if eval_weights_kind == "ema":
                ema.apply_shadow(model)

            slice_prob_batches = []
            slice_rid_batches = []
            with torch.no_grad():
                for x, xh, xhr, _, rids, pos in val_loader:
                    x, xh, xhr = x.to(DEVICE), xh.to(DEVICE), xhr.to(DEVICE)
                    
                    if DEVICE == 'cuda':
                        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                            probs = torch.sigmoid(model(x, xh, xhr)).float().cpu().numpy()
                    else:
                        probs = torch.sigmoid(model(x, xh, xhr)).cpu().numpy()
                    if torch.is_tensor(rids):
                        rid_np = rids.cpu().numpy().astype(np.int64)
                    else:
                        rid_np = np.asarray(rids, dtype=np.int64)
                    slice_prob_batches.append(probs.astype(np.float32))
                    slice_rid_batches.append(rid_np)

            if not slice_prob_batches:
                if eval_weights_kind == "ema":
                    ema.restore(model)
                raise RuntimeError(
                    f"Fold {fold} epoch {epoch + 1} produced no validation batches"
                )

            slice_probs = np.concatenate(slice_prob_batches, axis=0)
            slice_rids = np.concatenate(slice_rid_batches, axis=0)
            finite_slice_mask = np.isfinite(slice_probs).all(axis=1)
            if not np.all(finite_slice_mask):
                fold_skipped_nan += int(np.sum(~finite_slice_mask))
                if eval_weights_kind == "ema":
                    ema.restore(model)
                raise FloatingPointError(
                    f"Fold {fold} epoch {epoch + 1} produced "
                    f"{fold_skipped_nan} non-finite validation slices"
                )
            y_prob_all, valid_record_mask, _ = aggregate_record_probabilities(
                slice_probs,
                slice_rids,
                n_records=len(y),
                q=float(CONFIG["power_mean_q"]),
            )
            val_record_mask = np.zeros(len(y), dtype=bool)
            val_record_mask[va_idx] = True
            metric_mask = val_record_mask & valid_record_mask
            n_metric_records = int(np.sum(metric_mask))
            if n_metric_records != n_val_records_expected:
                if eval_weights_kind == "ema":
                    ema.restore(model)
                raise RuntimeError(
                    f"Fold {fold} epoch {epoch + 1} scored "
                    f"{n_metric_records}/{n_val_records_expected} validation records"
                )

            metrics = compute_metrics(
                y[metric_mask],
                y_prob_all[metric_mask],
                threshold=CONFIG["default_threshold"],
            )

            f1m = metrics["f1_macro"]
            is_new_best_ema = eval_weights_kind == "ema" and f1m > best_ema_f1
            selected_state = cpu_state_dict(model) if is_new_best_ema else None

            if eval_weights_kind == "ema":
                ema.restore(model)

            if eval_weights_kind == "raw":
                best_raw_warmup_f1 = max(best_raw_warmup_f1, f1m)

            if is_new_best_ema:
                best_ema_f1 = f1m
                best_ema_epoch = epoch + 1
                best_ema_metrics = metrics.copy()
                for previous_row in epoch_logs:
                    if previous_row["fold"] == fold:
                        previous_row["is_best_epoch"] = False
                selected_payload = checkpoint_payload(
                    model_state=selected_state,
                    fold=fold,
                    epoch=best_ema_epoch,
                    weights_kind="ema",
                    selected_by_weights_kind="ema",
                    metrics=best_ema_metrics,
                    train_indices=tr_idx,
                    val_indices=va_idx,
                    pca_variance=pca_variance,
                    pca_path=pca_path,
                    pca_sha256=pca_sha256,
                    dataset_record_order_fingerprint=split_audit["record_order_fingerprint"],
                    metrics_weights_kind="ema",
                    selection_rule="max_validation_f1_macro",
                    ablation_variant=ablation_variant,
                    ablation_spec=ablation_spec,
                    initialization_contract=initialization_contract,
                )
                save_checkpoint(best_ema_ckpt_path, selected_payload)
                raw_payload = checkpoint_payload(
                    model_state=cpu_state_dict(model),
                    fold=fold,
                    epoch=best_ema_epoch,
                    weights_kind="raw",
                    selected_by_weights_kind="ema",
                    metrics=None,
                    train_indices=tr_idx,
                    val_indices=va_idx,
                    pca_variance=pca_variance,
                    pca_path=pca_path,
                    pca_sha256=pca_sha256,
                    dataset_record_order_fingerprint=split_audit["record_order_fingerprint"],
                    metrics_weights_kind=None,
                    selection_rule="paired_with_best_ema_epoch",
                    selection_metrics=best_ema_metrics,
                    ablation_variant=ablation_variant,
                    ablation_spec=ablation_spec,
                    initialization_contract=initialization_contract,
                )

            last_eval_metrics = metrics.copy()
            last_eval_weights_kind = eval_weights_kind
            displayed_best = (
                best_ema_f1
                if eval_weights_kind == "ema"
                else best_raw_warmup_f1
            )

            print(
                f"Ep {epoch+1:03d} | {loss_name} | LR {lr:.2e} | "
                f"Loss {loss_sum/len(train_loader):.4f} | "
                f"Eval {eval_weights_kind.upper()} | "
                f"F1m {metrics['f1_macro']:.4f} | "
                f"P {metrics['precision_macro']:.4f} | "
                f"R {metrics['recall_macro']:.4f} | "
                f"AP {metrics['auprc_macro']:.4f} | "
                f"Best {displayed_best:.4f}"
            )

            epoch_logs.append(
                dict(
                    fold=fold,
                    epoch=epoch + 1,
                    loss=loss_name,
                    lr=lr,
                    loss_value=loss_sum / len(train_loader),
                    is_best_epoch=bool(is_new_best_ema),
                    validation_weights_kind=eval_weights_kind,
                    aggregation_method="power_mean",
                    aggregation_q=float(CONFIG["power_mean_q"]),
                    aggregation_implementation=POWER_MEAN_IMPLEMENTATION,
                    n_validation_records_scored=int(np.sum(metric_mask)),
                    n_validation_slices_scored=int(len(slice_probs)),
                    **metrics,
                )
            )
            atomic_write_csv(pd.DataFrame(epoch_logs), epoch_log_partial_path)

        if best_ema_epoch < 0 or best_ema_metrics is None:
            raise RuntimeError(
                f"Fold {fold} did not produce an EMA validation checkpoint. "
                "Increase epochs or reduce asym_start_epoch before using this run."
            )
        if last_eval_weights_kind != "ema" or last_eval_metrics is None:
            raise RuntimeError(
                f"Fold {fold} final epoch did not produce EMA validation metrics"
            )
        if raw_payload is None:
            raise RuntimeError(f"Fold {fold} did not retain the raw best-EMA companion")
        save_checkpoint(best_raw_ckpt_path, raw_payload)
        selected_state = None
        selected_payload = None
        raw_payload = None

        final_raw_payload = checkpoint_payload(
            model_state=cpu_state_dict(model),
            fold=fold,
            epoch=CONFIG["epochs"],
            weights_kind="raw",
            selected_by_weights_kind="raw",
            metrics=None,
            train_indices=tr_idx,
            val_indices=va_idx,
            pca_variance=pca_variance,
            pca_path=pca_path,
            pca_sha256=pca_sha256,
            dataset_record_order_fingerprint=split_audit["record_order_fingerprint"],
            metrics_weights_kind=None,
            selection_rule="fixed_final_epoch",
            ablation_variant=ablation_variant,
            ablation_spec=ablation_spec,
            initialization_contract=initialization_contract,
        )
        save_checkpoint(final_raw_ckpt_path, final_raw_payload)

        if CONFIG.get("use_ema", True):
            ema.apply_shadow(model)
            final_ema_payload = checkpoint_payload(
                model_state=cpu_state_dict(model),
                fold=fold,
                epoch=CONFIG["epochs"],
                weights_kind="ema",
                selected_by_weights_kind="ema",
                metrics=last_eval_metrics if last_eval_weights_kind == "ema" else None,
                train_indices=tr_idx,
                val_indices=va_idx,
                pca_variance=pca_variance,
                pca_path=pca_path,
                pca_sha256=pca_sha256,
                dataset_record_order_fingerprint=split_audit["record_order_fingerprint"],
                metrics_weights_kind="ema",
                selection_rule="fixed_final_epoch",
                ablation_variant=ablation_variant,
                ablation_spec=ablation_spec,
                initialization_contract=initialization_contract,
            )
            ema.restore(model)
            save_checkpoint(final_ema_ckpt_path, final_ema_payload)

        fold_results.append(
            dict(
                fold=fold,
                protocol_checkpoint_kind="final_ema",
                protocol_epoch=int(CONFIG["epochs"]),
                ablation_variant=ablation_variant,
                architecture_contract="ecg_ramba_structured_ablation_v1",
                best_ema_epoch=best_ema_epoch,
                pca_explained_variance=pca_variance,
                skipped_nonfinite_validation_slices=int(fold_skipped_nan),
                **last_eval_metrics,
                **{
                    f"best_ema_{key}": value
                    for key, value in best_ema_metrics.items()
                },
            )
        )
        atomic_write_csv(
            pd.DataFrame(fold_results).set_index("fold"),
            fold_log_partial_path,
            index=True,
        )

        print(
            f"   🧹 Releasing Fold {fold} datasets/loaders after "
            f"{(time.perf_counter() - fold_started) / 60:.1f} min",
            flush=True,
        )
        del train_loader, val_loader
        del rid_tr, start_tr, pos_tr
        del rid_va, start_va, pos_va
        del hydra_by_record, hrv_by_record, pca
        del selected_state, selected_payload, raw_payload
        del final_raw_payload, final_ema_payload
        del model, optimizer, scheduler, ema
        torch.cuda.empty_cache()
        gc.collect()
        print(f"   ✅ Fold {fold} memory cleanup complete", flush=True)

    # ==================================================================================
    # 🏁 FINAL REPORT
    # ==================================================================================
    print("\n" + "=" * 80)
    print("FINAL CROSS-VALIDATION RESULTS (OOF)")
    print("=" * 80)

    df_folds = pd.DataFrame(fold_results).sort_values("fold").set_index("fold")
    df_epochs = pd.DataFrame(epoch_logs).sort_values(["fold", "epoch"])

    print(df_folds.round(4))

    for m in ["f1_macro", "f1_micro", "precision_macro", "recall_macro", "auprc_macro"]:
        v = df_folds[m].values
        mean, std = v.mean(), v.std(ddof=1)
        ci = stats.t.interval(0.95, len(v) - 1, loc=mean, scale=std / np.sqrt(len(v)))
        print(f"{m:18s} {mean:.4f} ± {std:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    atomic_write_csv(df_epochs, epoch_log_path)
    atomic_write_csv(df_folds, fold_log_path, index=True)
    for partial_path in (epoch_log_partial_path, fold_log_partial_path):
        if os.path.exists(partial_path):
            os.remove(partial_path)

    print("\n✅ PIPELINE FINISHED.")


if __name__ == "__main__":
    main()
