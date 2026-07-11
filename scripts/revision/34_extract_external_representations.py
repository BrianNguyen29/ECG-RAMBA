"""Extract checkpoint-fingerprinted external record representations.

The output is used only by the true few-shot classifier-head adaptation runner.
Each Chapman fold encoder remains frozen. Record representations are means of
the pre-classifier slice representations, saved separately for every fold.
Fold caches make long Colab runs resumable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONFIG, PATHS  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    PREDICTION_DIR,
    git_commit,
    save_json,
    save_npz_compressed_atomic,
    sha256_file,
)
from src.training_data import build_slice_index  # noqa: E402


PROTOCOL_VERSION = 1
MODEL_STEMS = {
    "full": "ecg_ramba_full",
    "resnet": "resnet1d_cnn",
    "raw_mamba": "raw_mamba",
    "transformer": "transformer_ecg",
}


def load_revision_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


external = load_revision_module("03_generate_external_predictions.py", "_external_repr_dataset")
resnet = load_revision_module("14_resnet1d_cnn_baseline.py", "_external_repr_resnet")
raw_mamba = load_revision_module("16_raw_mamba_baseline.py", "_external_repr_raw_mamba")
comparators = load_revision_module(
    "31_generate_external_comparator_predictions.py", "_external_repr_comparators"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ptbxl", "georgia", "cpsc2021"])
    parser.add_argument("--models", default="full,resnet,raw_mamba")
    parser.add_argument("--ptbxl-folds", default="10")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--checkpoint-kind", default="final_ema")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-folds", default="")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--extract-root", type=Path, default=Path("/content/ecg_ramba_runtime/external"))
    parser.add_argument(
        "--georgia-mapping-review",
        type=Path,
        default=PROJECT_ROOT / "docs" / "revision_plan" / "georgia_label_mapping_review_20260703.csv",
    )
    parser.add_argument(
        "--georgia-code-inventory-out",
        type=Path,
        default=PROJECT_ROOT / "reports" / "revision" / "tables" / "table_georgia_snomed_code_inventory.csv",
    )
    parser.add_argument(
        "--cpsc-annotation-audit-out",
        type=Path,
        default=PROJECT_ROOT / "reports" / "revision" / "tables" / "table_cpsc2021_annotation_audit.csv",
    )
    parser.add_argument(
        "--fold-cache-dir",
        type=Path,
        default=PREDICTION_DIR / "external_representation_folds",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=EXPERIMENTAL_DIR / "external",
    )
    parser.add_argument(
        "--resnet-checkpoint-dir",
        type=Path,
        default=EXPERIMENTAL_DIR / "resnet1d_cnn_checkpoints",
    )
    parser.add_argument(
        "--raw-mamba-checkpoint-dir",
        type=Path,
        default=EXPERIMENTAL_DIR / "raw_mamba_checkpoints",
    )
    parser.add_argument(
        "--transformer-checkpoint-dir",
        type=Path,
        default=EXPERIMENTAL_DIR / "transformer_ecg_checkpoints",
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=PREDICTION_DIR / "oof_final_ema_predictions.npz",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_folds(value: str) -> set[int]:
    folds = {int(item) for item in parse_list(value)}
    if any(fold < 1 or fold > 5 for fold in folds):
        raise ValueError(f"Invalid folds: {sorted(folds)}")
    return folds


def tag_for(args: argparse.Namespace) -> str:
    tag = str(args.output_tag).strip().replace(" ", "_")
    if args.dataset == "ptbxl":
        folds = external.parse_ptbxl_folds(args.ptbxl_folds)
        if folds != (10,) and not tag:
            tag = "folds" + "_".join(str(fold) for fold in folds)
    return tag


def source_prediction_path(args: argparse.Namespace, model: str) -> Path:
    tag = tag_for(args)
    suffix = f"_{tag}" if tag else ""
    root = resolve(args.external_root) / args.dataset
    if model == "full":
        return root / f"{args.dataset}_full{suffix}_predictions.npz"
    return root / f"{args.dataset}_{MODEL_STEMS[model]}{suffix}_predictions.npz"


def load_source_contract(args: argparse.Namespace, model: str) -> dict[str, Any]:
    path = source_prediction_path(args, model)
    if not path.exists():
        raise FileNotFoundError(
            f"Run external prediction inference before representation extraction: {path}"
        )
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "record_id", "group_id", "split_id", "class_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing group-safe keys: {sorted(missing)}")
        payload = {
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]).astype(str),
            "group_id": np.asarray(data["group_id"]).astype(str),
            "split_id": np.asarray(data["split_id"]).astype(str),
            "class_names": np.asarray(data["class_names"]).astype(str),
        }
    payload.update({"path": path, "sha256": sha256_file(path)})
    return payload


def input_fingerprint(contract: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("record_id", "group_id", "split_id"):
        digest.update("\n".join(contract[key]).encode())
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(contract["y_true"]).tobytes())
    return digest.hexdigest()


def loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ptbxl_folds=args.ptbxl_folds,
        georgia_mapping_review=resolve(args.georgia_mapping_review),
        georgia_code_inventory_out=resolve(args.georgia_code_inventory_out),
        cpsc_annotation_audit_out=resolve(args.cpsc_annotation_audit_out),
    )


def load_signals(args: argparse.Namespace, source: dict[str, Any]):
    archive = external.archive_path(args.dataset)
    root = external.extract_archive(args.dataset, archive, resolve(args.extract_root))
    signals, y, record_id, group_id, split_id, load_summary = external.load_records(
        args.dataset, root, int(args.limit_records), loader_args(args)
    )
    for name, actual, expected in (
        ("y_true", y, source["y_true"]),
        ("record_id", np.asarray(record_id).astype(str), source["record_id"]),
        ("group_id", np.asarray(group_id).astype(str), source["group_id"]),
        ("split_id", np.asarray(split_id).astype(str), source["split_id"]),
    ):
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"Reloaded external {name} differs from source prediction artifact")
    return signals, load_summary, archive


def checkpoint_paths(args: argparse.Namespace, model: str) -> list[Path]:
    if model == "full":
        return external.checkpoint_paths(args.checkpoint_kind)
    return [comparators.checkpoint_path(args, model, fold) for fold in range(1, 6)]


def fold_cache_path(args: argparse.Namespace, model: str, fold: int) -> Path:
    tag = tag_for(args)
    suffix = f"_{tag}" if tag else ""
    return resolve(args.fold_cache_dir) / (
        f"{args.dataset}{suffix}_{MODEL_STEMS[model]}_fold{fold}_record_embeddings.npz"
    )


def final_paths(args: argparse.Namespace, model: str) -> tuple[Path, Path]:
    tag = tag_for(args)
    suffix = f"_{tag}" if tag else ""
    return (
        PREDICTION_DIR / f"external_{args.dataset}_{MODEL_STEMS[model]}{suffix}_record_embeddings.npz",
        MANIFEST_DIR / f"external_{args.dataset}_{MODEL_STEMS[model]}{suffix}_embedding_manifest.json",
    )


def final_output_matches(
    args: argparse.Namespace,
    model: str,
    source: dict[str, Any],
    checkpoints: list[Path],
    fingerprint: str,
    canonical: dict[str, Any],
) -> bool:
    output, manifest_path = final_paths(args, model)
    if not output.exists() or output.stat().st_size == 0 or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_rows = manifest.get("checkpoints") or []
        if (
            manifest.get("status") != "complete"
            or manifest.get("protocol") != "frozen_encoder_external_record_representation_v1"
            or manifest.get("runner_sha256") != sha256_file(Path(__file__).resolve())
            or manifest.get("canonical_contract") != canonical
            or manifest.get("input_fingerprint") != fingerprint
            or (manifest.get("source_prediction") or {}).get("sha256") != source["sha256"]
            or [row.get("sha256") for row in checkpoint_rows]
            != [sha256_file(path) for path in checkpoints]
            or (manifest.get("output") or {}).get("sha256") != sha256_file(output)
        ):
            return False
        with np.load(output, allow_pickle=False) as data:
            required = {
                "fold_embeddings",
                "y_true",
                "record_id",
                "group_id",
                "split_id",
                "class_names",
                "model",
                "dataset",
                "source_prediction_sha256",
                "input_fingerprint",
                "protocol_version",
            }
            if required - set(data.files):
                return False
            embeddings = np.asarray(data["fold_embeddings"], dtype=np.float32)
            return bool(
                embeddings.ndim == 3
                and embeddings.shape[0] == 5
                and embeddings.shape[1] == len(source["record_id"])
                and np.isfinite(embeddings).all()
                and str(data["model"].item()) == model
                and str(data["dataset"].item()) == args.dataset
                and str(data["source_prediction_sha256"].item()) == source["sha256"]
                and str(data["input_fingerprint"].item()) == fingerprint
                and int(data["protocol_version"].item()) == PROTOCOL_VERSION
                and np.array_equal(np.asarray(data["y_true"]), source["y_true"])
                and np.array_equal(np.asarray(data["record_id"]).astype(str), source["record_id"])
                and np.array_equal(np.asarray(data["group_id"]).astype(str), source["group_id"])
                and np.array_equal(np.asarray(data["split_id"]).astype(str), source["split_id"])
                and np.array_equal(np.asarray(data["class_names"]).astype(str), source["class_names"])
            )
    except Exception:
        return False


def cache_matches(path: Path, model: str, fold: int, checkpoint_sha: str, fingerprint: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"embedding", "model", "fold", "checkpoint_sha256", "input_fingerprint", "protocol_version"}
            if required - set(data.files):
                return False
            embedding = np.asarray(data["embedding"], dtype=np.float32)
            return bool(
                str(data["model"].item()) == model
                and int(data["fold"].item()) == fold
                and str(data["checkpoint_sha256"].item()) == checkpoint_sha
                and str(data["input_fingerprint"].item()) == fingerprint
                and int(data["protocol_version"].item()) == PROTOCOL_VERSION
                and embedding.ndim == 2
                and np.isfinite(embedding).all()
            )
    except Exception:
        return False


def aggregate_record_mean(features: np.ndarray, record_index: np.ndarray, n_records: int) -> np.ndarray:
    sums = np.zeros((n_records, features.shape[1]), dtype=np.float64)
    np.add.at(sums, record_index, features)
    counts = np.bincount(record_index, minlength=n_records)
    if np.any(counts == 0):
        raise RuntimeError(f"Records without representation slices: {np.where(counts == 0)[0][:10]}")
    return (sums / counts[:, None]).astype(np.float32)


def amp_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def infer_raw_features(model_name: str, model, loader, device: torch.device, use_amp: bool):
    features: list[np.ndarray] = []
    record_ids: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _target, rid, start in tqdm(loader, desc=f"{model_name} representations", leave=False):
            x = x.to(device, non_blocking=True)
            with amp_context(device, use_amp):
                if model_name == "raw_mamba":
                    value = raw_mamba.forward_raw_mamba_features(model, x)
                else:
                    value = model.forward_features(x)
            features.append(value.float().cpu().numpy())
            record_ids.append(np.asarray(rid, dtype=np.int64))
            starts.append(np.asarray(start, dtype=np.int32))
    return np.concatenate(features), np.concatenate(record_ids), np.concatenate(starts)


def infer_full_features(
    model,
    xs: np.ndarray,
    xhr: np.ndarray,
    slice_record_index: np.ndarray,
    hydra: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(xs), batch_size), desc="Full representations", leave=False):
            stop = min(start + batch_size, len(xs))
            xb = torch.from_numpy(xs[start:stop]).to(device)
            rid = slice_record_index[start:stop]
            hb = torch.from_numpy(hydra[rid]).to(device)
            rb = torch.from_numpy(xhr[start:stop]).to(device)
            with amp_context(device, use_amp):
                value = model.forward_features(xb, hb, rb)
            output.append(value.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def main() -> None:
    args = parse_args()
    models = parse_list(args.models)
    unknown = sorted(set(models) - set(MODEL_STEMS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    selected_folds = parse_folds(args.only_folds)
    print("=" * 80, flush=True)
    print("EXTERNAL FROZEN-ENCODER REPRESENTATION EXTRACTION", flush=True)
    print("=" * 80, flush=True)
    print(
        f"dataset={args.dataset} models={models} folds={sorted(selected_folds) or 'all'} "
        f"requested_device={args.device}",
        flush=True,
    )

    canonical = comparators.canonical_contract(args)
    base_source = load_source_contract(args, "full")
    fingerprint = input_fingerprint(base_source)
    source_by_model: dict[str, dict[str, Any]] = {}
    checkpoints_by_model: dict[str, list[Path]] = {}
    for model_name in models:
        source = load_source_contract(args, model_name)
        if not all(
            np.array_equal(source[key], base_source[key])
            for key in ("y_true", "record_id", "group_id", "split_id", "class_names")
        ):
            raise RuntimeError(f"{model_name} external source artifact differs from Full")
        if model_name != "full":
            comparators.validate_in_domain_comparator(model_name, canonical)
        checkpoints = checkpoint_paths(args, model_name)
        missing = [path for path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError("; ".join(str(path) for path in missing))
        source_by_model[model_name] = source
        checkpoints_by_model[model_name] = checkpoints

    if args.reuse_existing and not args.force_rerun and not selected_folds and all(
        final_output_matches(
            args,
            model_name,
            source_by_model[model_name],
            checkpoints_by_model[model_name],
            fingerprint,
            canonical,
        )
        for model_name in models
    ):
        status_rows = []
        for model_name in models:
            output, manifest = final_paths(args, model_name)
            status_rows.append(
                {
                    "model": model_name,
                    "status": "complete_reused_verified",
                    "embedding": str(output),
                    "manifest": str(manifest),
                }
            )
            print(f"Reusing verified final representation: {output}", flush=True)
        status_path = METRIC_DIR / f"external_{args.dataset}_representation_extraction_status.json"
        save_json(
            status_path,
            {
                "status": "complete",
                "created_utc": now_utc(),
                "dataset": args.dataset,
                "rows": status_rows,
                "reused_verified_final_outputs": True,
            },
        )
        return

    device = comparators.model_loaders.select_device(args.device)
    if device.type != "cuda" and str(args.device).lower() == "auto":
        raise RuntimeError(
            "Verified final representation artifacts were not reusable and extraction needs model inference. "
            "Attach a CUDA GPU (A100 High-RAM preferred) or pass --device cpu explicitly if the slow CPU path "
            "is intentional. Existing fold caches remain reusable."
        )
    if args.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"resolved_device={device}", flush=True)
    signals, load_summary, archive = load_signals(args, base_source)
    all_indices = np.arange(len(signals), dtype=np.int64)
    slice_record_ids, starts, _positions, skipped = build_slice_index(
        all_indices,
        signals,
        slice_length=int(CONFIG["slice_length"]),
        slice_stride=int(CONFIG["slice_stride"]),
        max_slices_per_record=int(CONFIG["max_slices_per_record"]),
    )
    if skipped:
        raise RuntimeError(f"Records without valid slices: {skipped[:10]}")
    raw_dataset = resnet.RawECGSliceDataset(
        signals,
        base_source["y_true"],
        slice_record_ids,
        starts,
        slice_length=int(CONFIG["slice_length"]),
    )
    raw_loader = resnet.build_loader(
        raw_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        seed=42,
        device=device,
    )

    full_aux = None
    if "full" in models:
        full_checkpoints = checkpoints_by_model["full"]
        checkpoint_rows, source_hash = external.checkpoint_provenance(full_checkpoints, args.checkpoint_kind)
        expected_record_fp = checkpoint_rows[0]["dataset_record_order_fingerprint"]
        pca_paths = external.fold_pca_paths(
            5,
            source_hash,
            args.checkpoint_kind,
            expected_record_fp,
        )
        hydra, hrv, feature_cache, feature_cache_hit = external.generate_features(
            args.dataset,
            archive,
            signals,
            base_source["record_id"],
            pca_paths,
            False,
        )
        xs, xhr, full_slice_record_index, _slice_index = external.build_slices(signals, hrv)
        if not np.array_equal(full_slice_record_index, slice_record_ids):
            raise RuntimeError("Full/raw slice record order differs")
        full_aux = {
            "hydra": hydra,
            "xs": xs,
            "xhr": xhr,
            "record_index": full_slice_record_index,
            "feature_cache": str(feature_cache),
            "feature_cache_sha256": sha256_file(feature_cache),
            "feature_cache_hit": feature_cache_hit,
        }

    status_rows: list[dict[str, Any]] = []
    for model_name in models:
        source = source_by_model[model_name]
        ckpts = checkpoints_by_model[model_name]
        hashes = [sha256_file(path) for path in ckpts]
        folds_now = selected_folds or set(range(1, 6))
        for fold in sorted(folds_now):
            cache = fold_cache_path(args, model_name, fold)
            if args.reuse_existing and not args.force_rerun and cache_matches(
                cache, model_name, fold, hashes[fold - 1], fingerprint
            ):
                print(f"Reusing {model_name}/fold{fold}: {cache}", flush=True)
                continue
            print(f"Extracting {model_name}/fold{fold}: {ckpts[fold - 1]}", flush=True)
            if model_name == "full":
                from src.model import ECGRambaV7Advanced

                payload = torch.load(ckpts[fold - 1], map_location="cpu", weights_only=False)
                state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
                model = ECGRambaV7Advanced(cfg=CONFIG).to(device)
                model.load_state_dict(state, strict=True)
                assert full_aux is not None
                slice_features = infer_full_features(
                    model,
                    full_aux["xs"],
                    full_aux["xhr"],
                    full_aux["record_index"],
                    full_aux["hydra"][fold - 1],
                    int(args.batch_size),
                    device,
                    bool(args.amp),
                )
                actual_record_ids = full_aux["record_index"]
            else:
                model = comparators.load_model(args, model_name, ckpts[fold - 1], device)
                slice_features, actual_record_ids, actual_starts = infer_raw_features(
                    model_name, model, raw_loader, device, bool(args.amp)
                )
                if not np.array_equal(actual_record_ids, slice_record_ids) or not np.array_equal(actual_starts, starts):
                    raise RuntimeError(f"{model_name}/fold{fold}: slice order changed")
            record_embedding = aggregate_record_mean(slice_features, actual_record_ids, len(signals))
            cache.parent.mkdir(parents=True, exist_ok=True)
            save_npz_compressed_atomic(
                cache,
                embedding=record_embedding,
                model=np.asarray(model_name),
                fold=np.asarray(fold, dtype=np.int16),
                checkpoint_path=np.asarray(str(ckpts[fold - 1])),
                checkpoint_sha256=np.asarray(hashes[fold - 1]),
                input_fingerprint=np.asarray(fingerprint),
                source_prediction_sha256=np.asarray(source["sha256"]),
                protocol_version=np.asarray(PROTOCOL_VERSION, dtype=np.int16),
                created_utc=np.asarray(now_utc()),
            )
            print(f"Wrote fold representation cache: {cache}", flush=True)
            del model, slice_features, record_embedding
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        caches = [fold_cache_path(args, model_name, fold) for fold in range(1, 6)]
        ready = [
            fold
            for fold, cache in enumerate(caches, start=1)
            if cache_matches(cache, model_name, fold, hashes[fold - 1], fingerprint)
        ]
        if len(ready) != 5:
            status_rows.append({"model": model_name, "status": "fold_caches_incomplete", "ready_folds": ready})
            print(f"{model_name}: ready folds={ready}; rerun missing folds before aggregation", flush=True)
            continue
        embeddings = []
        for cache in caches:
            with np.load(cache, allow_pickle=False) as data:
                embeddings.append(np.asarray(data["embedding"], dtype=np.float32))
        stacked = np.stack(embeddings, axis=0)
        out, manifest = final_paths(args, model_name)
        save_npz_compressed_atomic(
            out,
            fold_embeddings=stacked,
            y_true=base_source["y_true"],
            record_id=base_source["record_id"],
            group_id=base_source["group_id"],
            split_id=base_source["split_id"],
            class_names=base_source["class_names"],
            model=np.asarray(model_name),
            dataset=np.asarray(args.dataset),
            representation=np.asarray("mean_of_preclassifier_slice_embeddings_per_fold"),
            source_prediction_sha256=np.asarray(source["sha256"]),
            input_fingerprint=np.asarray(fingerprint),
            protocol_version=np.asarray(PROTOCOL_VERSION, dtype=np.int16),
        )
        save_json(
            manifest,
            {
                "status": "complete",
                "created_utc": now_utc(),
                "git_commit": git_commit(),
                "dataset": args.dataset,
                "model": model_name,
                "protocol": "frozen_encoder_external_record_representation_v1",
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "canonical_contract": canonical,
                "representation": "mean_of_preclassifier_slice_embeddings_per_fold",
                "shape": list(stacked.shape),
                "input_fingerprint": fingerprint,
                "source_prediction": {"path": str(source["path"]), "sha256": source["sha256"]},
                "checkpoints": [
                    {"fold": fold, "path": str(path), "sha256": hashes[fold - 1]}
                    for fold, path in enumerate(ckpts, start=1)
                ],
                "fold_caches": [
                    {"fold": fold, "path": str(path), "sha256": sha256_file(path)}
                    for fold, path in enumerate(caches, start=1)
                ],
                "full_feature_cache": (
                    {
                        "path": full_aux["feature_cache"],
                        "sha256": full_aux["feature_cache_sha256"],
                        "cache_hit": full_aux["feature_cache_hit"],
                    }
                    if model_name == "full" and full_aux is not None
                    else None
                ),
                "output": {"path": str(out), "sha256": sha256_file(out), "size_bytes": out.stat().st_size},
            },
        )
        status_rows.append({"model": model_name, "status": "complete", "embedding": str(out), "manifest": str(manifest)})
        print(f"Wrote external record embeddings: {out}", flush=True)

    status_path = METRIC_DIR / f"external_{args.dataset}_representation_extraction_status.json"
    save_json(
        status_path,
        {
            "status": "complete" if all(row["status"] == "complete" for row in status_rows) else "incomplete",
            "created_utc": now_utc(),
            "dataset": args.dataset,
            "rows": status_rows,
            "load_summary": load_summary,
        },
    )
    print(json.dumps({"status": status_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
