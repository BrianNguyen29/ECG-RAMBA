"""Generate stressed predictions for learned raw-ECG comparators.

This runner is intentionally inference-only. It requires comparator checkpoints
created by the fair-baseline runners with ``--save-checkpoints`` and refuses to
retrain or reconstruct weights from clean OOF predictions. The output NPZ files
match the contract consumed by ``21_robustness_multicomparator.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG  # noqa: E402
from scripts.revision.common import MANIFEST_DIR, PREDICTION_DIR, ensure_revision_dirs, save_json, sha256_file  # noqa: E402
from src.aggregation import POWER_MEAN_IMPLEMENTATION, aggregate_record_probabilities  # noqa: E402
from src.training_data import build_slice_index  # noqa: E402


PROTOCOL = "comparator_stress_predictions_v1_same_folds_power_mean_v2_q3"
DEFAULT_OOF = PREDICTION_DIR / "oof_final_ema_predictions.npz"
DEFAULT_FREEZE = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
DEFAULT_RESNET_CKPT_DIR = PROJECT_ROOT / "reports" / "revision" / "experimental" / "resnet1d_cnn_checkpoints"
DEFAULT_RAW_MAMBA_CKPT_DIR = PROJECT_ROOT / "reports" / "revision" / "experimental" / "raw_mamba_checkpoints"
DEFAULT_TRANSFORMER_CKPT_DIR = PROJECT_ROOT / "reports" / "revision" / "experimental" / "transformer_ecg_checkpoints"


def load_revision_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


resnet_helpers = load_revision_module("14_resnet1d_cnn_baseline.py", "_stress_resnet_helpers")
raw_mamba_helpers = load_revision_module("16_raw_mamba_baseline.py", "_stress_raw_mamba_helpers")
transformer_helpers = load_revision_module("24_transformer_ecg_baseline.py", "_stress_transformer_helpers")
robust_helpers = load_revision_module("12_robustness_stress.py", "_stress_perturb_helpers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparators", default="resnet,raw_mamba,transformer")
    parser.add_argument("--stress-tests", default="snr20db,snr10db,snr5db,random_3_lead_dropout,precordial_dropout,resample_250hz")
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--raw-cache", type=Path, default=None)
    parser.add_argument("--resnet-checkpoint-dir", type=Path, default=DEFAULT_RESNET_CKPT_DIR)
    parser.add_argument("--raw-mamba-checkpoint-dir", type=Path, default=DEFAULT_RAW_MAMBA_CKPT_DIR)
    parser.add_argument("--transformer-checkpoint-dir", type=Path, default=DEFAULT_TRANSFORMER_CKPT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--out-manifest", type=Path, default=MANIFEST_DIR / "comparator_stress_prediction_manifest.json")
    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    path = resolve(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return device


def autocast_resnet(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def output_path(comparator: str, stress: str) -> Path:
    stem = {"resnet": "resnet1d_cnn", "raw_mamba": "raw_mamba", "transformer": "transformer_ecg"}[comparator]
    return PREDICTION_DIR / f"robustness_{stem}_{stress}_predictions.npz"


def checkpoint_path(args: argparse.Namespace, comparator: str, fold: int) -> Path:
    if comparator == "resnet":
        return resolve(args.resnet_checkpoint_dir) / f"fold{fold}_resnet1d_cnn_final.pt"
    if comparator == "raw_mamba":
        return resolve(args.raw_mamba_checkpoint_dir) / f"fold{fold}_raw_mamba_final_ema.pt"
    if comparator == "transformer":
        return resolve(args.transformer_checkpoint_dir) / f"fold{fold}_transformer_ecg_final.pt"
    raise ValueError(f"Unknown comparator: {comparator}")


def validate_existing(path: Path, y: np.ndarray, fold_id: np.ndarray) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                "y_true" in data.files
                and "y_prob" in data.files
                and np.asarray(data["y_true"]).shape == y.shape
                and np.asarray(data["y_prob"]).shape == y.shape
                and np.array_equal(np.asarray(data["y_true"], dtype=np.float32), y)
                and np.array_equal(np.asarray(data["fold_id"], dtype=np.int16), fold_id)
            )
    except Exception:
        return False


def load_trusted_checkpoint(path: Path) -> Any:
    """Load checkpoints generated by the local revision baseline runners."""

    # PyTorch 2.6 defaults torch.load to weights_only=True. Our checkpoint
    # payloads intentionally include trusted metadata such as args/load_info
    # plus the tensor state_dict, so full-pickle loading is required here.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_resnet_model(args: argparse.Namespace, ckpt: Path, device: torch.device):
    model_args = argparse.Namespace(base_channels=64, dropout=0.20)
    payload = load_trusted_checkpoint(ckpt)
    if isinstance(payload, dict) and "args" in payload:
        saved_args = payload.get("args") or {}
        model_args.base_channels = int(saved_args.get("base_channels", model_args.base_channels))
        model_args.dropout = float(saved_args.get("dropout", model_args.dropout))
    model = resnet_helpers.build_model(model_args)
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) else payload)
    return model.to(device).eval()


def load_raw_mamba_model(ckpt: Path, device: torch.device):
    payload = load_trusted_checkpoint(ckpt)
    model = raw_mamba_helpers.build_model()
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) else payload)
    return model.to(device).eval()


def load_transformer_model(ckpt: Path, device: torch.device):
    payload = load_trusted_checkpoint(ckpt)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Transformer checkpoint lacks model_state_dict: {ckpt}")
    saved_args = payload.get("args") or {}
    model_params = payload.get("model_params") or {}
    base_channels = int(saved_args.get("base_channels", 64))
    embed_dim = int(
        model_params.get("embed_dim")
        or saved_args.get("transformer_embed_dim")
        or base_channels
    )
    n_heads = int(
        model_params.get("n_heads")
        or saved_args.get("transformer_heads")
        or (4 if embed_dim % 4 == 0 else 2)
    )
    depth = int(model_params.get("depth") or saved_args.get("transformer_depth") or 3)
    patch_size = int(model_params.get("patch_size") or saved_args.get("transformer_patch_size") or 50)
    patch_stride = int(model_params.get("patch_stride") or saved_args.get("transformer_patch_stride") or 25)
    ff_multiplier = int(
        model_params.get("feed_forward_multiplier")
        or saved_args.get("transformer_ff_multiplier")
        or 4
    )
    dropout = float(saved_args.get("dropout", 0.20))
    if embed_dim % n_heads != 0:
        raise ValueError(
            f"Transformer checkpoint architecture is invalid: embed_dim={embed_dim}, heads={n_heads}"
        )
    model = transformer_helpers.ECGPatchTransformer(
        n_classes=len(CLASSES),
        embed_dim=embed_dim,
        n_heads=n_heads,
        depth=depth,
        patch_size=patch_size,
        patch_stride=patch_stride,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
        max_length=int(CONFIG["slice_length"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def predict_fold(
    *,
    comparator: str,
    model: torch.nn.Module,
    signals: np.ndarray,
    y: np.ndarray,
    val_indices: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record_ids, starts, _positions, skipped = build_slice_index(
        val_indices,
        signals,
        slice_length=int(CONFIG["slice_length"]),
        slice_stride=int(CONFIG["slice_stride"]),
        max_slices_per_record=int(CONFIG["max_slices_per_record"]),
    )
    if skipped:
        raise RuntimeError(f"Fold {fold} skipped validation records under stress: {skipped}")
    dataset = resnet_helpers.RawECGSliceDataset(
        signals,
        y,
        record_ids,
        starts,
        slice_length=int(CONFIG["slice_length"]),
    )
    loader = resnet_helpers.build_loader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        seed=int(args.seed) + int(fold),
        device=device,
    )
    if comparator in {"resnet", "transformer"}:
        slice_prob, slice_record_id, slice_start = resnet_helpers.predict_slice_probabilities(
            model,
            loader,
            device=device,
            use_amp=bool(args.amp),
        )
    else:
        raw_args = argparse.Namespace(amp=bool(args.amp), amp_dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
        slice_prob, slice_record_id, slice_start = raw_mamba_helpers.predict_slice_probabilities(
            model,
            loader,
            device=device,
            args=raw_args,
        )
    y_prob_all, valid_mask, slice_count = aggregate_record_probabilities(
        slice_prob,
        slice_record_id,
        n_records=len(y),
        q=float(CONFIG["power_mean_q"]),
    )
    missing = sorted(set(int(x) for x in val_indices) - set(np.where(valid_mask)[0].astype(int)))
    if missing:
        raise RuntimeError(f"Fold {fold} missing stressed predictions for records: {missing[:10]}")
    return y_prob_all, slice_count, slice_prob, slice_record_id, slice_start


def write_stress_npz(
    path: Path,
    *,
    comparator: str,
    stress: str,
    y: np.ndarray,
    y_prob: np.ndarray,
    record_id: np.ndarray,
    fold_id: np.ndarray,
    slice_count: np.ndarray,
    class_names: list[str],
    stress_meta: dict[str, Any],
    checkpoint_paths: list[Path],
    raw_cache_info: dict[str, Any],
    freeze_contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=y.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        record_id=record_id.astype(np.int64),
        fold_id=fold_id.astype(np.int16),
        slice_count=slice_count.astype(np.int16),
        class_names=np.asarray(class_names),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray(PROTOCOL),
        comparator=np.asarray(comparator),
        stress_test=np.asarray(stress),
        stress_metadata_json=np.asarray(json.dumps(stress_meta, sort_keys=True)),
        aggregation_method=np.asarray("power_mean"),
        aggregation_implementation=np.asarray(POWER_MEAN_IMPLEMENTATION),
        power_mean_q=np.asarray(float(CONFIG["power_mean_q"])),
        checkpoint_paths=np.asarray([project_relative(path) for path in checkpoint_paths]),
        checkpoint_sha256=np.asarray([sha256_file(path) for path in checkpoint_paths]),
        raw_cache_sha256=np.asarray(raw_cache_info.get("raw_cache_sha256", "")),
        freeze_manifest_sha256=np.asarray(freeze_contract.get("freeze_manifest_sha256", "")),
        oof_predictions_sha256=np.asarray(freeze_contract.get("oof_predictions_sha256", "")),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
    )


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = select_device(args.device)
    comparators = parse_list(args.comparators)
    stresses = robust_helpers.stress_specs(parse_list(args.stress_tests), int(args.seed))
    unknown = sorted(set(comparators) - {"resnet", "raw_mamba", "transformer"})
    if unknown:
        raise ValueError(f"Unsupported comparators for this runner: {unknown}")

    print("=" * 80, flush=True)
    print("COMPARATOR STRESS PREDICTION GENERATOR", flush=True)
    print("=" * 80, flush=True)
    print(f"comparators={comparators} stresses={[s['name'] for s in stresses]} device={device}", flush=True)

    freeze_contract = resnet_helpers.validate_oof_freeze_contract(
        freeze_manifest=args.freeze_manifest,
        oof_predictions=args.oof_predictions,
        expected_checkpoint_kind=args.expected_checkpoint_kind,
    )
    y, fold_id, record_id, class_names, folds, oof_info = resnet_helpers.load_oof_labels_and_folds(
        args.oof_predictions,
        limit_records=int(args.limit_records),
    )
    raw_x, raw_cache_info = resnet_helpers.load_raw_cache(
        expected_y=y,
        expected_record_fingerprint=oof_info.get("dataset_record_order_fingerprint", ""),
        explicit_cache=args.raw_cache,
        limit_records=int(args.limit_records),
    )

    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in stresses:
        stress = spec["name"]
        print(f"\nStress {stress}: perturbing raw ECG", flush=True)
        stressed_x, stress_meta = robust_helpers.perturb_signals(raw_x, spec)
        for comparator in comparators:
            out = output_path(comparator, stress)
            if args.reuse_existing and validate_existing(out, y, fold_id):
                print(f"Reusing existing {comparator}/{stress}: {out}", flush=True)
                artifacts.append({"comparator": comparator, "stress": stress, "path": project_relative(out), "sha256": sha256_file(out), "reused": True})
                continue
            ckpts = [checkpoint_path(args, comparator, int(split["fold"])) for split in folds]
            absent = [str(path) for path in ckpts if not path.exists() or path.stat().st_size == 0]
            if absent:
                msg = f"{comparator}/{stress} missing checkpoints: " + "; ".join(absent)
                print("BLOCKED:", msg, flush=True)
                missing.append(msg)
                continue
            y_prob = np.zeros_like(y, dtype=np.float32)
            slice_count = np.zeros(len(y), dtype=np.int16)
            for split in folds:
                fold = int(split["fold"])
                va_idx = np.asarray(split["va_idx"], dtype=np.int64)
                ckpt = checkpoint_path(args, comparator, fold)
                print(f"{comparator} stress={stress} fold {fold}/5 | val={len(va_idx)} checkpoint={ckpt}", flush=True)
                if comparator == "resnet":
                    model = load_resnet_model(args, ckpt, device)
                elif comparator == "raw_mamba":
                    model = load_raw_mamba_model(ckpt, device)
                elif comparator == "transformer":
                    model = load_transformer_model(ckpt, device)
                else:
                    raise ValueError(f"Unknown comparator: {comparator}")
                fold_prob, fold_slice_count, _slice_prob, _slice_record_id, _slice_start = predict_fold(
                    comparator=comparator,
                    model=model,
                    signals=stressed_x,
                    y=y,
                    val_indices=va_idx,
                    fold=fold,
                    args=args,
                    device=device,
                )
                y_prob[va_idx] = fold_prob[va_idx]
                slice_count[va_idx] = fold_slice_count[va_idx]
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if np.any(slice_count <= 0):
                raise RuntimeError(f"{comparator}/{stress} incomplete slice_count coverage")
            write_stress_npz(
                out,
                comparator=comparator,
                stress=stress,
                y=y,
                y_prob=y_prob,
                record_id=record_id,
                fold_id=fold_id,
                slice_count=slice_count,
                class_names=class_names,
                stress_meta=stress_meta,
                checkpoint_paths=ckpts,
                raw_cache_info=raw_cache_info,
                freeze_contract=freeze_contract,
            )
            print(f"Wrote {comparator}/{stress}: {out}", flush=True)
            artifacts.append({"comparator": comparator, "stress": stress, "path": project_relative(out), "sha256": sha256_file(out), "reused": False})

    payload = {
        "status": "complete" if not missing else "blocked_missing_checkpoints",
        "protocol": PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparators": comparators,
        "stress_tests": [spec["name"] for spec in stresses],
        "artifacts": artifacts,
        "missing": missing,
        "device": str(device),
        "requires_saved_comparator_checkpoints": True,
    }
    save_json(resolve(args.out_manifest), payload)
    print(json.dumps({"status": payload["status"], "artifacts": len(artifacts), "missing": len(missing)}, indent=2), flush=True)
    print(f"Wrote manifest: {resolve(args.out_manifest)}", flush=True)
    if missing and args.strict:
        raise RuntimeError("Missing comparator checkpoints prevent stressed prediction generation.")


if __name__ == "__main__":
    main()
