"""Extract frozen ECG-RAMBA branch embeddings for representation probes.

This runner uses the fold assignment frozen in the manuscript OOF artifact and
reuses the PCA/checkpoint contract from ``01_generate_predictions.py``. It
writes a record-level embedding artifact for ``20_representation_probe.py``.
It is intentionally separate from the model class so the inference path used
for reviewer metrics remains unchanged.

Outputs are cached per fold so Colab interruptions can be resumed without
silently mixing checkpoint/config variants.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

gen = importlib.import_module("scripts.revision.01_generate_predictions")

from configs.config import (  # noqa: E402
    CLASSES,
    CONFIG,
    DEVICE,
    EVALUATION_CONFIG_HASH,
    PATHS,
)
from scripts.revision.common import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    MANIFEST_DIR,
    PREDICTION_DIR,
    ensure_revision_dirs,
    save_json,
    sha256_file,
)
from src.provenance import record_order_fingerprint  # noqa: E402


PROTOCOL = "ecg_ramba_final_ema_branch_embedding_extraction_v1"
EMBEDDING_KEYS = [
    "morphology_embedding",
    "rhythm_embedding",
    "context_embedding",
    "fused_embedding",
]


def array_sha256(array: np.ndarray, dtype: np.dtype | type | None = None) -> str:
    values = np.asarray(array, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def folds_from_frozen_oof(oof: dict[str, Any]) -> list[dict[str, np.ndarray | int | str]]:
    """Build the exact evaluation folds recorded in the frozen OOF artifact."""
    fold_id = np.asarray(oof["fold_id"], dtype=np.int16)
    n_records = len(np.asarray(oof["record_id"]))
    if fold_id.shape != (n_records,):
        raise ValueError(f"Invalid frozen OOF fold_id shape: {fold_id.shape}")

    expected_folds = list(range(1, int(CONFIG["n_folds"]) + 1))
    observed_folds = sorted(int(value) for value in np.unique(fold_id))
    if observed_folds != expected_folds:
        raise ValueError(
            "Frozen OOF fold_id does not contain the expected folds: "
            f"observed={observed_folds} expected={expected_folds}"
        )

    folds: list[dict[str, np.ndarray | int | str]] = []
    for fold_num in expected_folds:
        va_idx = np.flatnonzero(fold_id == fold_num).astype(np.int64)
        tr_idx = np.flatnonzero(fold_id != fold_num).astype(np.int64)
        if len(va_idx) == 0 or len(tr_idx) + len(va_idx) != n_records:
            raise ValueError(f"Frozen OOF fold {fold_num} has an invalid train/validation partition.")
        folds.append(
            {
                "fold_num": fold_num,
                "tr_idx": tr_idx,
                "va_idx": va_idx,
                "train_index_sha256": array_sha256(tr_idx, np.int64),
                "validation_index_sha256": array_sha256(va_idx, np.int64),
            }
        )
    return folds


def validate_checkpoint_fold_contract(
    oof: dict[str, Any], checkpoint_contracts: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Verify that frozen OOF membership matches the checkpoints' persisted folds.pkl."""
    checkpoint_paths = [
        Path(str(row.get("path") or ""))
        for _, row in sorted(checkpoint_contracts.items())
        if str(row.get("path") or "")
    ]
    checkpoint_dirs = {path.parent for path in checkpoint_paths}
    if len(checkpoint_paths) != int(CONFIG["n_folds"]) or len(checkpoint_dirs) != 1:
        raise RuntimeError(
            "Checkpoint fold provenance is incomplete or spans multiple model directories."
        )

    folds_path = next(iter(checkpoint_dirs)) / "folds.pkl"
    if not folds_path.exists() or folds_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Persisted training folds are required beside the checkpoints: {folds_path}"
        )
    persisted_folds = joblib.load(folds_path)
    if len(persisted_folds) != int(CONFIG["n_folds"]):
        raise RuntimeError(
            f"Persisted fold count mismatch: {len(persisted_folds)} != {CONFIG['n_folds']}"
        )

    n_records = len(np.asarray(oof["record_id"]))
    persisted_fold_id = np.full(n_records, -1, dtype=np.int16)
    for fold_num, fold in enumerate(persisted_folds, start=1):
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        va_idx = va_idx[(va_idx >= 0) & (va_idx < n_records)]
        if len(np.unique(va_idx)) != len(va_idx):
            raise RuntimeError(f"Persisted fold {fold_num} contains duplicate validation indices.")
        if np.any(persisted_fold_id[va_idx] != -1):
            raise RuntimeError("Persisted folds assign at least one record to multiple validation folds.")
        persisted_fold_id[va_idx] = fold_num
    if np.any(persisted_fold_id < 0):
        raise RuntimeError(
            f"Persisted folds do not cover {int(np.sum(persisted_fold_id < 0))} OOF records."
        )

    current_fold_id = np.asarray(oof["fold_id"], dtype=np.int16)
    if not np.array_equal(persisted_fold_id, current_fold_id):
        mismatch_count = int(np.sum(persisted_fold_id != current_fold_id))
        raise RuntimeError(
            "Frozen OOF fold assignment differs from the folds persisted beside the checkpoints: "
            f"mismatched_records={mismatch_count}."
        )
    return {
        "source": "frozen_oof_fold_id_verified_against_checkpoint_folds",
        "folds_path": str(folds_path),
        "folds_file_sha256": sha256_file(folds_path),
        "fold_assignment_sha256": array_sha256(current_fold_id, np.int16),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kind", default="final_ema", choices=gen.CHECKPOINT_KINDS)
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
    parser.add_argument(
        "--oof-run-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_prediction_run_manifest.json",
        help="Prediction run manifest containing exact checkpoint paths and SHA256 values.",
    )
    parser.add_argument(
        "--out-embedding",
        type=Path,
        default=PREDICTION_DIR / "representation_embeddings_final_ema.npz",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "representation_embedding_manifest.json",
    )
    parser.add_argument(
        "--fold-cache-dir",
        type=Path,
        default=PREDICTION_DIR / "folds",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument(
        "--only-folds",
        default="",
        help="Optional comma-separated fold numbers to compute now. Other folds are loaded from cache.",
    )
    parser.add_argument("--resume-fold-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun-folds", action="store_true", default=False)
    parser.add_argument("--min-system-ram-gb", type=float, default=24.0)
    parser.add_argument("--allow-low-ram", action="store_true", default=False)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    path = resolve(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def validate_mamba_runtime_for_extraction() -> None:
    missing = [
        module
        for module in ("mamba_ssm", "causal_conv1d")
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    raise ImportError(
        "Representation extraction requires the ECG-RAMBA Mamba runtime before "
        "loading data or fitting fold PCA. Missing modules: "
        + ", ".join(missing)
        + ". In Colab, run Notebook 00 bootstrap or the Notebook 02 model "
        "dependency/Mamba install cell in the same GPU runtime, restart only if "
        "that installer asks you to, then rerun Notebook 06 from Setup. Existing "
        "fold Hydra/PCA caches are safe to reuse."
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_only_folds(value: str) -> set[int] | None:
    if not value.strip():
        return None
    out = {int(item.strip()) for item in value.split(",") if item.strip()}
    invalid = sorted(fold for fold in out if fold < 1 or fold > int(CONFIG["n_folds"]))
    if invalid:
        raise ValueError(f"Invalid fold numbers in --only-folds: {invalid}")
    return out


def load_oof_contract(path: Path, freeze_manifest: Path, limit_records: int) -> dict[str, Any]:
    path = resolve(path)
    freeze_manifest = resolve(freeze_manifest)
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {path}")
    if limit_records == 0 and not freeze_manifest.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {freeze_manifest}")

    with np.load(path, allow_pickle=False) as data:
        required = ["y_true", "record_id", "fold_id", "class_names"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"OOF predictions missing required keys: {missing}")
        y_true = np.asarray(data["y_true"], dtype=np.float32)
        record_id = np.asarray(data["record_id"]).astype(np.int64)
        fold_id = np.asarray(data["fold_id"]).astype(np.int16)
        class_names = np.asarray(data["class_names"]).astype(str)

    if limit_records > 0:
        y_true = y_true[:limit_records]
        record_id = record_id[:limit_records]
        fold_id = fold_id[:limit_records]

    if not np.array_equal(class_names, np.asarray(CLASSES).astype(str)):
        raise ValueError("OOF class_names do not match current config CLASSES.")
    if y_true.shape != (len(record_id), len(CLASSES)):
        raise ValueError(f"Invalid OOF y_true shape: {y_true.shape}")
    if not np.array_equal(record_id, np.arange(len(record_id), dtype=np.int64)):
        raise ValueError("OOF record_id must be the canonical 0..N-1 order for embedding extraction.")

    freeze_payload: dict[str, Any] | None = None
    if freeze_manifest.exists():
        freeze_payload = json.loads(freeze_manifest.read_text(encoding="utf-8"))
        expected_sha = freeze_payload.get("record_file_sha256") or freeze_payload.get("predictions_sha256")
        if expected_sha is None:
            for artifact in freeze_payload.get("artifacts", []):
                artifact_path = str(artifact.get("path", ""))
                if artifact_path.endswith(path.name):
                    expected_sha = artifact.get("sha256")
                    break
        actual_sha = sha256_file(path)
        if expected_sha and str(expected_sha) != actual_sha:
            raise RuntimeError(
                "Freeze manifest does not match OOF predictions: "
                f"manifest={expected_sha} actual={actual_sha}"
            )

    return {
        "path": path,
        "sha256": sha256_file(path),
        "freeze_manifest": freeze_manifest,
        "freeze_manifest_sha256": sha256_file(freeze_manifest) if freeze_manifest.exists() else None,
        "freeze_payload": freeze_payload,
        "y_true": y_true,
        "record_id": record_id,
        "fold_id": fold_id,
        "class_names": class_names,
    }


def load_checkpoint_contracts(run_manifest_path: Path, checkpoint_kind: str) -> dict[int, dict[str, Any]]:
    path = resolve(run_manifest_path)
    if not path.exists():
        print(
            f"WARNING: OOF run manifest not found: {path}. "
            "Falling back to canonical checkpoint path lookup.",
            flush=True,
        )
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_rows = payload.get("inputs", {}).get("checkpoints", [])
    contracts: dict[int, dict[str, Any]] = {}
    for row in checkpoint_rows:
        fold = int(row.get("fold", -1))
        row_path = Path(str(row.get("path", "")))
        if fold <= 0 or not row_path.name.endswith(f"_{checkpoint_kind}.pt"):
            continue
        contracts[fold] = row
    if contracts:
        print(
            f"Loaded checkpoint contract from OOF run manifest: {path} | folds={sorted(contracts)}",
            flush=True,
        )
    else:
        print(
            f"WARNING: OOF run manifest has no {checkpoint_kind} checkpoint rows: {path}",
            flush=True,
        )
    return contracts


def resolve_checkpoint_for_fold(
    *,
    fold_num: int,
    checkpoint_kind: str,
    checkpoint_contracts: dict[int, dict[str, Any]],
) -> tuple[Path, str | None]:
    contract = checkpoint_contracts.get(fold_num)
    candidates: list[Path] = []
    expected_sha = None
    if contract:
        expected_sha = str(contract.get("sha256") or "")
        manifest_path = Path(str(contract.get("path", "")))
        if str(manifest_path):
            candidates.append(manifest_path)

            # Support local clones whose Drive root differs from the original
            # Colab absolute path recorded in the manifest.
            marker = "/ECG-Ramba/"
            manifest_posix = manifest_path.as_posix()
            if marker in manifest_posix:
                relative_to_drive = manifest_posix.split(marker, 1)[1]
                drive_root = Path(os.environ.get("ECG_RAMBA_DRIVE_ROOT", ""))
                if str(drive_root):
                    candidates.append(drive_root / relative_to_drive)
                candidates.append(PROJECT_ROOT.parent / relative_to_drive)

    try:
        candidates.append(gen.checkpoint_path(fold_num, checkpoint_kind, allow_fallback=False))
    except FileNotFoundError:
        pass

    # Common retraining output location used by Notebook 02a.
    drive_root = Path(os.environ.get("ECG_RAMBA_DRIVE_ROOT", ""))
    for root in [
        drive_root / "model_runs" / "ema_protocol_e20_v2" if str(drive_root) else None,
        PROJECT_ROOT.parent / "model_runs" / "ema_protocol_e20_v2",
        PROJECT_ROOT / "model_runs" / "ema_protocol_e20_v2",
        Path(PATHS["model_dir"]),
    ]:
        if root is not None:
            candidates.append(root / f"fold{fold_num}_{checkpoint_kind}.pt")

    seen: set[str] = set()
    checked: list[str] = []
    for candidate in candidates:
        if not candidate or not str(candidate):
            continue
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if not candidate.exists():
            continue
        actual_sha = sha256_file(candidate)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(
                f"Checkpoint SHA mismatch for fold {fold_num}: {candidate} "
                f"expected={expected_sha} actual={actual_sha}"
            )
        return candidate, actual_sha

    raise FileNotFoundError(
        f"Missing exact checkpoint for fold {fold_num} ({checkpoint_kind}). "
        "Checked paths: " + "; ".join(checked) + ". "
        "Restore/copy the model_runs/ema_protocol_e20_v2 checkpoints to Drive, "
        "or rerun Notebook 02a retraining before representation extraction."
    )


def fold_embedding_cache_path(
    fold_num: int,
    checkpoint_kind: str,
    checkpoint_sha256: str,
    fold_cache_dir: Path,
) -> Path:
    return resolve(fold_cache_dir) / (
        f"representation_{checkpoint_kind}_fold{fold_num}_{EVALUATION_CONFIG_HASH}_"
        f"{checkpoint_sha256[:12]}_v{CACHE_SCHEMA_VERSION}.npz"
    )


def load_fold_embedding_cache(
    *,
    path: Path,
    fold_num: int,
    va_idx: np.ndarray,
    checkpoint_sha256: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            record_id = np.asarray(data["record_id"]).astype(np.int64)
            slice_count = np.asarray(data["slice_count"]).astype(np.int16)
            schema_version = int(data["cache_schema_version"])
            cached_checkpoint_sha = str(data["checkpoint_sha256"].item())
            cached_eval_hash = str(data["evaluation_config_hash"].item())
            summary = json.loads(str(data["fold_summary_json"].item()))
            embeddings = {key: np.asarray(data[key], dtype=np.float32) for key in EMBEDDING_KEYS}
        if (
            schema_version != CACHE_SCHEMA_VERSION
            or cached_checkpoint_sha != checkpoint_sha256
            or cached_eval_hash != EVALUATION_CONFIG_HASH
            or not np.array_equal(record_id, va_idx.astype(np.int64))
            or slice_count.shape != va_idx.shape
        ):
            print(f"WARNING: Representation fold cache contract mismatch: {path}", flush=True)
            return None
        for key, arr in embeddings.items():
            if arr.ndim != 2 or arr.shape[0] != len(va_idx) or arr.dtype != np.float32:
                print(f"WARNING: Invalid cached {key} shape/dtype in {path}: {arr.shape} {arr.dtype}", flush=True)
                return None
            if not np.isfinite(arr).all():
                print(f"WARNING: Non-finite cached {key} in {path}", flush=True)
                return None
        print(f"Loaded representation cache for fold {fold_num}: {path}", flush=True)
        return embeddings, slice_count, summary
    except Exception as exc:
        print(f"WARNING: Could not load representation cache {path}: {exc}", flush=True)
        return None


def save_fold_embedding_cache(
    *,
    path: Path,
    fold_num: int,
    va_idx: np.ndarray,
    embeddings: dict[str, np.ndarray],
    slice_count: np.ndarray,
    checkpoint_sha256: str,
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "record_id": va_idx.astype(np.int64),
        "slice_count": slice_count.astype(np.int16),
        "fold": np.asarray(fold_num, dtype=np.int16),
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "source_config_hash": np.asarray(summary.get("source_config_hash", "")),
        "evaluation_config_hash": np.asarray(EVALUATION_CONFIG_HASH),
        "checkpoint_sha256": np.asarray(checkpoint_sha256),
        "fold_summary_json": np.asarray(json.dumps(summary, sort_keys=True)),
    }
    for key in EMBEDDING_KEYS:
        payload[key] = embeddings[key].astype(np.float32)
    tmp_path = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, path)
    print(f"Wrote representation cache for fold {fold_num}: {path}", flush=True)


def mean_pool(sequence: torch.Tensor | None, *, batch: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sequence is None:
        return torch.zeros(batch, d_model, device=device, dtype=dtype)
    return sequence.mean(dim=1)


def forward_with_embeddings(
    model: torch.nn.Module,
    x: torch.Tensor,
    xh: torch.Tensor,
    xhr: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mirror ECGRambaV7Advanced.forward and expose branch-level embeddings."""
    batch = x.size(0)
    d_model = int(model.cfg["d_model"])

    if model.spatial_attn is not None:
        x = model.spatial_attn(x)

    mamba_feat = model.tok(x)
    if model.ablation.get("no_multiscale", False):
        mamba_feat = mamba_feat.transpose(1, 2)

    if model.rocket_perceiver is not None:
        rocket_feat = model.rocket_perceiver(xh)
    else:
        rocket_feat = None

    if model.hrv_proj is not None:
        hrv_context = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else nullcontext()
        with hrv_context:
            hrv_feat = model.hrv_proj(xhr.float()).unsqueeze(1)
    else:
        hrv_feat = None

    branch_embeddings = {
        "context_embedding": mean_pool(
            mamba_feat,
            batch=batch,
            d_model=d_model,
            device=x.device,
            dtype=mamba_feat.dtype,
        ),
        "morphology_embedding": mean_pool(
            rocket_feat,
            batch=batch,
            d_model=d_model,
            device=x.device,
            dtype=mamba_feat.dtype,
        ),
        "rhythm_embedding": mean_pool(
            hrv_feat,
            batch=batch,
            d_model=d_model,
            device=x.device,
            dtype=mamba_feat.dtype,
        ),
    }

    if model.use_cross_attn and mamba_feat is not None and rocket_feat is not None:
        fused_seq, _ = model.cross_fusion(mamba_feat, rocket_feat)
    else:
        parts = [part for part in [mamba_feat, rocket_feat] if part is not None]
        if len(parts) > 1:
            fused_seq = torch.cat(parts, dim=1)
        elif len(parts) == 1:
            fused_seq = parts[0]
        else:
            fused_seq = torch.zeros(batch, 1, d_model, device=x.device, dtype=mamba_feat.dtype)

    seq_parts = [part for part in [fused_seq, hrv_feat] if part is not None]
    if seq_parts:
        seq = torch.cat(seq_parts, dim=1)
    else:
        seq = torch.zeros(batch, 1, d_model, device=x.device, dtype=mamba_feat.dtype)

    seq = model.feature_proj(seq)

    if model.use_final_perceiver:
        lat = model.final_latents.expand(batch, -1, -1)
        lat = lat + model.final_cross_attn(
            model.final_norm1(lat), seq, seq, need_weights=False
        )[0]
        lat = lat + model.final_self_attn(
            model.final_norm2(lat),
            model.final_norm2(lat),
            model.final_norm2(lat),
            need_weights=False,
        )[0]
        lat = lat + model.final_ffn(model.final_norm3(lat))
        seq = lat

    for layer in model.layers:
        seq = layer(seq)

    fused_embedding = model.norm(seq).mean(dim=1)
    branch_embeddings["fused_embedding"] = fused_embedding
    logits = model.head(fused_embedding)
    return logits, {key: value.float() for key, value in branch_embeddings.items()}


def extract_fold_embeddings(
    *,
    fold_num: int,
    model: torch.nn.Module,
    xs: np.ndarray,
    xh: np.ndarray,
    xhr: np.ndarray,
    rids: np.ndarray,
    va_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if len(rids) == 0:
        raise RuntimeError(f"Fold {fold_num} has no slices.")

    dataset = gen.ECGSliceDatasetInfer(xs, xh, xhr, rids)
    loader = gen.make_inference_loader(
        dataset,
        batch_size=max(1, int(batch_size)),
        num_workers=max(0, int(num_workers)),
    )
    rid_to_pos = {int(rid): pos for pos, rid in enumerate(va_idx.astype(np.int64))}
    slice_count = np.zeros(len(va_idx), dtype=np.int32)
    sums: dict[str, np.ndarray] | None = None

    with torch.no_grad():
        for x_batch, xh_batch, xhr_batch, rid_batch in tqdm(
            loader,
            desc=f"Embeddings fold {fold_num}",
            leave=False,
        ):
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            xh_batch = xh_batch.to(DEVICE, non_blocking=True)
            xhr_batch = xhr_batch.to(DEVICE, non_blocking=True)

            if DEVICE == "cuda":
                with torch.amp.autocast("cuda", dtype=gen.AMP_DTYPE):
                    _, embeddings_t = forward_with_embeddings(model, x_batch, xh_batch, xhr_batch)
            else:
                _, embeddings_t = forward_with_embeddings(model, x_batch, xh_batch, xhr_batch)

            rid_np = rid_batch.cpu().numpy().astype(np.int64)
            positions = np.asarray([rid_to_pos[int(rid)] for rid in rid_np], dtype=np.int64)
            if sums is None:
                sums = {
                    key: np.zeros((len(va_idx), value.shape[1]), dtype=np.float32)
                    for key, value in embeddings_t.items()
                }
            np.add.at(slice_count, positions, 1)
            for key, value in embeddings_t.items():
                arr = value.detach().cpu().numpy().astype(np.float32)
                np.add.at(sums[key], positions, arr)

    if sums is None:
        raise RuntimeError(f"Fold {fold_num} produced no embeddings.")
    if np.any(slice_count <= 0):
        missing = va_idx[np.where(slice_count <= 0)[0]][:20].tolist()
        raise RuntimeError(f"Fold {fold_num} has records without extracted slices: {missing}")
    averaged = {
        key: (arr / slice_count[:, None].astype(np.float32)).astype(np.float32)
        for key, arr in sums.items()
    }
    return averaged, slice_count.astype(np.int16)


def write_final_embedding_npz(
    *,
    path: Path,
    oof: dict[str, Any],
    embeddings: dict[str, np.ndarray],
    fold_id: np.ndarray,
    slice_count: np.ndarray,
    payload: dict[str, Any],
) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "y_true": oof["y_true"].astype(np.float32),
        "record_id": oof["record_id"].astype(np.int64),
        "fold_id": fold_id.astype(np.int16),
        "slice_count": slice_count.astype(np.int16),
        "class_names": oof["class_names"].astype(str),
        "protocol": np.asarray(PROTOCOL),
        "checkpoint_kind": np.asarray(payload["checkpoint_kind"]),
        "oof_predictions_sha256": np.asarray(oof["sha256"]),
        "freeze_manifest_sha256": np.asarray(oof["freeze_manifest_sha256"] or ""),
        "dataset_record_order_fingerprint": np.asarray(payload["dataset_record_order_fingerprint"]),
        "embedding_manifest_json": np.asarray(json.dumps(jsonable(payload), sort_keys=True)),
    }
    arrays.update({key: value.astype(np.float32) for key, value in embeddings.items()})
    tmp_path = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)
    print(f"Wrote representation embeddings: {path}", flush=True)


def inspect_final_embedding_reuse(
    path: Path,
    oof: dict[str, Any],
    checkpoint_kind: str,
    checkpoint_contracts: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    path = resolve(path)
    audit: dict[str, Any] = {
        "reusable": False,
        "issues": [],
        "semantic_fields": ["y_true", "record_id", "fold_id", "class_names"],
    }
    if not path.exists() or path.stat().st_size == 0:
        audit["issues"].append("embedding_missing_or_empty")
        return audit
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "y_true",
                "record_id",
                "fold_id",
                "class_names",
                "protocol",
                "checkpoint_kind",
                "oof_predictions_sha256",
                "freeze_manifest_sha256",
                "slice_count",
                "embedding_manifest_json",
                *EMBEDDING_KEYS,
            }
            missing = sorted(required - set(data.files))
            if missing:
                audit["issues"].append(f"missing_fields={','.join(missing)}")
                return audit

            source_oof_sha = str(data["oof_predictions_sha256"].item())
            source_freeze_sha = str(data["freeze_manifest_sha256"].item())
            audit.update(
                {
                    "source_oof_sha256": source_oof_sha,
                    "source_freeze_sha256": source_freeze_sha,
                    "current_oof_sha256": oof["sha256"],
                    "current_freeze_sha256": str(oof["freeze_manifest_sha256"] or ""),
                    "exact_oof_sha_match": source_oof_sha == oof["sha256"],
                    "exact_freeze_sha_match": source_freeze_sha
                    == str(oof["freeze_manifest_sha256"] or ""),
                }
            )
            if str(data["protocol"].item()) != PROTOCOL:
                audit["issues"].append("protocol_mismatch")
            if str(data["checkpoint_kind"].item()) != checkpoint_kind:
                audit["issues"].append("checkpoint_kind_mismatch")

            embedded_fold_id = np.asarray(data["fold_id"], dtype=np.int16)
            current_fold_id = np.asarray(oof["fold_id"], dtype=np.int16)
            semantic_field_match = {
                "y_true": bool(
                    np.array_equal(np.asarray(data["y_true"], dtype=np.float32), oof["y_true"])
                ),
                "record_id": bool(
                    np.array_equal(np.asarray(data["record_id"]).astype(np.int64), oof["record_id"])
                ),
                "fold_id": bool(np.array_equal(embedded_fold_id, current_fold_id)),
                "class_names": bool(
                    np.array_equal(np.asarray(data["class_names"]).astype(str), oof["class_names"])
                ),
            }
            audit["semantic_field_match"] = semantic_field_match
            semantic_match = all(semantic_field_match.values())
            audit["semantic_contract_match"] = semantic_match
            if not semantic_match:
                audit["issues"].append("oof_semantic_contract_mismatch")
            if not semantic_field_match["fold_id"]:
                audit["issues"].append("oof_fold_assignment_mismatch")
                audit["fold_assignment_mismatch_count"] = int(
                    np.sum(embedded_fold_id != current_fold_id)
                ) if embedded_fold_id.shape == current_fold_id.shape else None
                audit["source_fold_assignment_sha256"] = array_sha256(
                    embedded_fold_id, np.int16
                )
                audit["current_fold_assignment_sha256"] = array_sha256(
                    current_fold_id, np.int16
                )

            embedded_manifest = json.loads(str(data["embedding_manifest_json"].item()))
            audit["existing_semantic_reuse_attestation"] = embedded_manifest.get(
                "semantic_reuse_attestation"
            )
            observed_checkpoint_shas = {
                int(row.get("fold", -1)): str(row.get("checkpoint_sha256") or "")
                for row in embedded_manifest.get("fold_summaries", [])
                if int(row.get("fold", -1)) > 0
            }
            expected_checkpoint_shas = {
                int(fold): str(row.get("sha256") or "")
                for fold, row in checkpoint_contracts.items()
                if str(row.get("sha256") or "")
            }
            expected_folds = set(range(1, int(CONFIG["n_folds"]) + 1))
            audit["checkpoint_contract_match"] = bool(
                set(expected_checkpoint_shas) == expected_folds
                and observed_checkpoint_shas == expected_checkpoint_shas
            )
            audit["checkpoint_sha256_by_fold"] = observed_checkpoint_shas
            if not audit["checkpoint_contract_match"]:
                audit["issues"].append("checkpoint_sha_contract_mismatch_or_incomplete")

            embeddings_valid = all(
                np.asarray(data[key]).ndim == 2
                and np.asarray(data[key]).shape[0] == len(oof["record_id"])
                and np.isfinite(np.asarray(data[key], dtype=np.float32)).all()
                for key in EMBEDDING_KEYS
            )
            audit["embedding_arrays_valid"] = embeddings_valid
            if not embeddings_valid:
                audit["issues"].append("embedding_arrays_invalid")
    except Exception as exc:
        audit["issues"].append(f"{type(exc).__name__}: {exc}")
        return audit

    audit["exact_source_contract"] = bool(
        audit.get("exact_oof_sha_match") and audit.get("exact_freeze_sha_match")
    )
    audit["reusable"] = not audit["issues"]
    return audit


def refresh_final_embedding_contract(
    *,
    path: Path,
    oof: dict[str, Any],
    checkpoint_kind: str,
    reuse_audit: dict[str, Any],
    split_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = resolve(path)
    with np.load(path, allow_pickle=False) as data:
        embeddings = {key: np.asarray(data[key], dtype=np.float32) for key in EMBEDDING_KEYS}
        fold_id = np.asarray(data["fold_id"], dtype=np.int16)
        slice_count = np.asarray(data["slice_count"], dtype=np.int16)
        payload = json.loads(str(data["embedding_manifest_json"].item()))

    attestation = {
        "status": "verified_semantic_repack",
        "source_oof_sha256": reuse_audit.get("source_oof_sha256"),
        "source_freeze_sha256": reuse_audit.get("source_freeze_sha256"),
        "current_oof_sha256": oof["sha256"],
        "current_freeze_sha256": str(oof["freeze_manifest_sha256"] or ""),
        "semantic_contract_match": reuse_audit.get("semantic_contract_match") is True,
        "checkpoint_contract_match": reuse_audit.get("checkpoint_contract_match") is True,
        "semantic_fields": reuse_audit.get("semantic_fields", []),
        "semantic_field_match": reuse_audit.get("semantic_field_match", {}),
    }
    payload.update(
        {
            "created_utc": now_utc(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint_kind": checkpoint_kind,
            "oof_predictions": project_relative(oof["path"]),
            "oof_predictions_sha256": oof["sha256"],
            "freeze_manifest": project_relative(oof["freeze_manifest"]),
            "freeze_manifest_sha256": oof["freeze_manifest_sha256"],
            "canonical_contract": {
                "oof_sha256": oof["sha256"],
                "freeze_sha256": oof["freeze_manifest_sha256"],
            },
            "split_contract": split_contract
            or {
                "source": "frozen_oof_fold_id",
                "fold_assignment_sha256": array_sha256(oof["fold_id"], np.int16),
            },
            "semantic_reuse_attestation": attestation,
        }
    )
    write_final_embedding_npz(
        path=path,
        oof=oof,
        embeddings=embeddings,
        fold_id=fold_id,
        slice_count=slice_count,
        payload=payload,
    )
    return attestation


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    created_utc = now_utc()
    only_folds = parse_only_folds(args.only_folds)

    oof = load_oof_contract(args.oof_predictions, args.freeze_manifest, args.limit_records)
    frozen_kind = (oof.get("freeze_payload") or {}).get("checkpoint_kind")
    if args.limit_records == 0 and frozen_kind and str(frozen_kind) != args.checkpoint_kind:
        raise RuntimeError(
            f"Freeze manifest checkpoint_kind={frozen_kind} does not match requested "
            f"--checkpoint-kind={args.checkpoint_kind}."
        )

    print("=" * 80, flush=True)
    print("ECG-RAMBA REPRESENTATION EMBEDDING EXTRACTION", flush=True)
    print("=" * 80, flush=True)
    print(f"checkpoint_kind={args.checkpoint_kind}", flush=True)
    print(f"oof_predictions={resolve(args.oof_predictions)} sha256={oof['sha256']}", flush=True)
    print(f"oof_run_manifest={resolve(args.oof_run_manifest)}", flush=True)
    print(f"only_folds={sorted(only_folds) if only_folds else 'all'}", flush=True)
    print(f"batch_size={args.batch_size} num_workers={args.num_workers}", flush=True)
    checkpoint_contracts = load_checkpoint_contracts(args.oof_run_manifest, args.checkpoint_kind)
    checkpoint_split_contract = validate_checkpoint_fold_contract(oof, checkpoint_contracts)
    print(f"checkpoint_split_contract={checkpoint_split_contract}", flush=True)
    final_reuse_audit = inspect_final_embedding_reuse(
        args.out_embedding,
        oof,
        args.checkpoint_kind,
        checkpoint_contracts,
    )
    if not only_folds and final_reuse_audit.get("reusable"):
        semantic_reuse_attestation = None
        if not final_reuse_audit.get("exact_source_contract"):
            semantic_reuse_attestation = refresh_final_embedding_contract(
                path=args.out_embedding,
                oof=oof,
                checkpoint_kind=args.checkpoint_kind,
                reuse_audit=final_reuse_audit,
                split_contract=checkpoint_split_contract,
            )
            print(
                "Refreshed representation embedding OOF/freeze metadata after verified "
                "semantic and checkpoint-contract reuse.",
                flush=True,
            )
        manifest_path = resolve(args.out_manifest)
        existing = {}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(
            {
                "status": "complete",
                "protocol": PROTOCOL,
                "created_utc": created_utc,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "checkpoint_kind": args.checkpoint_kind,
                "oof_predictions": project_relative(resolve(args.oof_predictions)),
                "oof_predictions_sha256": oof["sha256"],
                "freeze_manifest": project_relative(resolve(args.freeze_manifest)),
                "freeze_manifest_sha256": oof["freeze_manifest_sha256"],
                "canonical_contract": {
                    "oof_sha256": oof["sha256"],
                    "freeze_sha256": oof["freeze_manifest_sha256"],
                },
                "split_contract": checkpoint_split_contract,
                "n_records": int(len(oof["record_id"])),
                "n_classes": int(oof["y_true"].shape[1]),
                "covered_records": int(len(oof["record_id"])),
                "missing_records": 0,
                "missing_fold_caches": [],
                "outputs": {
                    "embedding_npz": project_relative(resolve(args.out_embedding)),
                    "embedding_npz_sha256": sha256_file(resolve(args.out_embedding)),
                    "manifest": project_relative(manifest_path),
                },
                "safe_wording": (
                    "Use downstream representation-probe results as suggestive branch-specific "
                    "information only; do not claim proven morphology-rhythm disentanglement."
                ),
                "reused_verified_final_embedding": True,
                "semantic_reuse_attestation": semantic_reuse_attestation
                or final_reuse_audit.get("existing_semantic_reuse_attestation")
                or existing.get("semantic_reuse_attestation"),
            }
        )
        save_json(manifest_path, jsonable(existing))
        print(f"Reusing verified final representation embedding: {resolve(args.out_embedding)}", flush=True)
        print(f"Wrote manifest: {manifest_path}", flush=True)
        return
    if final_reuse_audit.get("issues"):
        print(
            "Final representation embedding is not reusable: "
            + "; ".join(str(issue) for issue in final_reuse_audit["issues"]),
            flush=True,
        )
    gen.validate_runtime_memory(args)
    validate_mamba_runtime_for_extraction()

    from src.features import generate_hrv_cache, generate_raw_rocket_cache

    X, y, X_raw_amp, subjects = gen.prepare_clean_chapman(limit_records=args.limit_records)
    n_records, n_classes = y.shape
    if n_records != len(oof["record_id"]) or n_classes != len(CLASSES):
        raise ValueError(f"Dataset/OOF shape mismatch: data={y.shape}, oof={oof['y_true'].shape}")
    if not np.array_equal(y.astype(np.float32), oof["y_true"].astype(np.float32)):
        raise ValueError("Chapman labels do not match frozen OOF y_true.")

    dataset_record_fingerprint = record_order_fingerprint(subjects)
    print(f"dataset_record_order_fingerprint={dataset_record_fingerprint}", flush=True)
    X_rocket_raw = generate_raw_rocket_cache(X, subjects)
    X_hrv = generate_hrv_cache(X, X_raw_amp, subjects) if CONFIG["use_hrv"] else np.zeros(
        (n_records, CONFIG["hrv_dim"]), dtype=np.float32
    )

    normalized_folds = folds_from_frozen_oof(oof)
    print(
        "Representation split source=frozen_oof_fold_id "
        f"sha256={array_sha256(oof['fold_id'], np.int16)}",
        flush=True,
    )

    global_embeddings: dict[str, np.ndarray] | None = None
    global_fold_id = np.full(n_records, -1, dtype=np.int16)
    global_slice_count = np.zeros(n_records, dtype=np.int16)
    fold_summaries: list[dict[str, Any]] = []
    missing_fold_caches: list[int] = []

    for fold in normalized_folds:
        fold_idx = int(fold["fold_num"])
        tr_idx = np.asarray(fold["tr_idx"], dtype=np.int64)
        va_idx = np.asarray(fold["va_idx"], dtype=np.int64)
        checkpoint_file, checkpoint_sha = resolve_checkpoint_for_fold(
            fold_num=fold_idx,
            checkpoint_kind=args.checkpoint_kind,
            checkpoint_contracts=checkpoint_contracts,
        )
        if checkpoint_sha is None:
            checkpoint_sha = sha256_file(checkpoint_file)
        checkpoint_payload, checkpoint_meta = gen.load_checkpoint_payload(checkpoint_file, args.checkpoint_kind)
        source_config_hash = checkpoint_meta["source_config_hash"]
        if args.limit_records == 0 and checkpoint_meta["dataset_record_order_fingerprint"] != dataset_record_fingerprint:
            raise RuntimeError(
                f"Fold {fold_idx} checkpoint fingerprint mismatch: "
                f"checkpoint={checkpoint_meta['dataset_record_order_fingerprint']} "
                f"data={dataset_record_fingerprint}"
            )

        cache_path = fold_embedding_cache_path(
            fold_idx,
            args.checkpoint_kind,
            checkpoint_sha,
            args.fold_cache_dir,
        )
        cached = None
        if args.resume_fold_cache and not args.force_rerun_folds:
            cached = load_fold_embedding_cache(
                path=cache_path,
                fold_num=fold_idx,
                va_idx=va_idx,
                checkpoint_sha256=checkpoint_sha,
            )

        should_compute = cached is None and (only_folds is None or fold_idx in only_folds)
        if cached is None and not should_compute:
            print(
                f"Fold {fold_idx}: cache missing and not selected by --only-folds; leaving incomplete.",
                flush=True,
            )
            missing_fold_caches.append(fold_idx)
            continue

        if cached is not None:
            fold_embeddings, fold_slice_count, summary = cached
        else:
            print("=" * 80, flush=True)
            print(f"Fold {fold_idx}/{len(normalized_folds)} | val={len(va_idx)}", flush=True)
            hydra_va, pca_var, hydra_cache_path, hydra_cache_hit = gen.load_or_compute_fold_hydra(
                fold_num=fold_idx,
                X_rocket_raw=X_rocket_raw,
                tr_idx=tr_idx,
                va_idx=va_idx,
                source_config_hash=source_config_hash,
            )
            hydra_va_by_record = {
                int(record_id): hydra_va[pos]
                for pos, record_id in enumerate(va_idx.astype(np.int64))
            }
            xs, xh, xhr, rids, build_slice_counts = gen.build_fold_slices(
                va_idx,
                X,
                X_hrv,
                hydra_va_by_record,
            )
            model = gen.load_model_for_fold(
                fold_idx,
                args.checkpoint_kind,
                checkpoint_file=checkpoint_file,
                checkpoint_payload=checkpoint_payload,
            )
            fold_embeddings, fold_slice_count = extract_fold_embeddings(
                fold_num=fold_idx,
                model=model,
                xs=xs,
                xh=xh,
                xhr=xhr,
                rids=rids,
                va_idx=va_idx,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            expected_counts = np.asarray([build_slice_counts[int(rid)] for rid in va_idx], dtype=np.int16)
            if not np.array_equal(fold_slice_count, expected_counts):
                raise RuntimeError(f"Fold {fold_idx} slice counts changed during extraction.")
            summary = {
                "fold": fold_idx,
                "train_records": int(len(tr_idx)),
                "validation_records": int(len(va_idx)),
                "train_index_sha256": str(fold["train_index_sha256"]),
                "validation_index_sha256": str(fold["validation_index_sha256"]),
                "validation_slices": int(len(rids)),
                "slice_count_min": int(fold_slice_count.min()),
                "slice_count_max": int(fold_slice_count.max()),
                "slice_count_mean": float(fold_slice_count.mean()),
                "checkpoint_file": str(checkpoint_file),
                "checkpoint_sha256": checkpoint_sha,
                "source_config_hash": source_config_hash,
                "weights_kind": checkpoint_meta.get("weights_kind"),
                "checkpoint_epoch": checkpoint_meta.get("epoch"),
                "checkpoint_selection_rule": checkpoint_meta.get("selection_rule"),
                "hydra_cache_path": str(hydra_cache_path),
                "hydra_cache_hit": bool(hydra_cache_hit),
                "hydra_pca_explained_variance": float(pca_var),
            }
            save_fold_embedding_cache(
                path=cache_path,
                fold_num=fold_idx,
                va_idx=va_idx,
                embeddings=fold_embeddings,
                slice_count=fold_slice_count,
                checkpoint_sha256=checkpoint_sha,
                summary=summary,
            )
            del model, xs, xh, xhr, rids
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if global_embeddings is None:
            global_embeddings = {
                key: np.zeros((n_records, arr.shape[1]), dtype=np.float32)
                for key, arr in fold_embeddings.items()
            }
        for key in EMBEDDING_KEYS:
            global_embeddings[key][va_idx] = fold_embeddings[key]
        global_fold_id[va_idx] = fold_idx
        global_slice_count[va_idx] = fold_slice_count.astype(np.int16)
        fold_summaries.append(summary)

    covered = global_fold_id >= 0
    missing_records = int(np.sum(~covered))
    payload = {
        "status": "complete" if missing_records == 0 else "partial_missing_fold_caches",
        "protocol": PROTOCOL,
        "created_utc": created_utc,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint_kind": args.checkpoint_kind,
        "oof_predictions": project_relative(resolve(args.oof_predictions)),
        "oof_predictions_sha256": oof["sha256"],
        "freeze_manifest": project_relative(resolve(args.freeze_manifest)),
        "freeze_manifest_sha256": oof["freeze_manifest_sha256"],
        "canonical_contract": {
            "oof_sha256": oof["sha256"],
            "freeze_sha256": oof["freeze_manifest_sha256"],
        },
        "dataset_record_order_fingerprint": dataset_record_fingerprint,
        "split_contract": checkpoint_split_contract,
        "evaluation_config_hash": EVALUATION_CONFIG_HASH,
        "embedding_views": EMBEDDING_KEYS,
        "slice_embedding_aggregation": "arithmetic_mean_over_record_slices",
        "n_records": int(n_records),
        "n_classes": int(n_classes),
        "covered_records": int(np.sum(covered)),
        "missing_records": missing_records,
        "missing_fold_caches": missing_fold_caches,
        "fold_summaries": fold_summaries,
        "outputs": {
            "embedding_npz": project_relative(resolve(args.out_embedding)) if missing_records == 0 else None,
            "manifest": project_relative(resolve(args.out_manifest)),
        },
        "safe_wording": (
            "Use downstream representation-probe results as suggestive branch-specific "
            "information only; do not claim proven morphology-rhythm disentanglement."
        ),
        "runtime": jsonable(gen.runtime_metadata(args, created_utc)),
    }

    if missing_records == 0:
        if global_embeddings is None:
            raise RuntimeError("No embeddings were assembled.")
        for key, arr in global_embeddings.items():
            if arr.shape[0] != n_records or not np.isfinite(arr).all():
                raise RuntimeError(f"Invalid final embedding array {key}: {arr.shape}")
        write_final_embedding_npz(
            path=args.out_embedding,
            oof=oof,
            embeddings=global_embeddings,
            fold_id=global_fold_id,
            slice_count=global_slice_count,
            payload=payload,
        )
        payload["outputs"]["embedding_npz_sha256"] = sha256_file(
            resolve(args.out_embedding)
        )
    else:
        print(
            f"Partial extraction only: missing_records={missing_records}. "
            "Run remaining folds or restore fold caches before running representation probe.",
            flush=True,
        )

    save_json(resolve(args.out_manifest), jsonable(payload))
    print(json.dumps(jsonable({"status": payload["status"], "missing_records": missing_records}), indent=2), flush=True)
    print(f"Wrote manifest: {resolve(args.out_manifest)}", flush=True)


if __name__ == "__main__":
    main()
