"""Build the five fold-specific PCA objects required for external inference.

This reuses the saved Chapman MiniRocket cache and does not run model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONFIG, PATHS  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    MANIFEST_DIR,
    ensure_revision_dirs,
    save_json,
    sha256_file,
)
from src.data_loader import load_chapman_multilabel  # noqa: E402
from src.features import fit_pca_on_train, generate_raw_rocket_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, action="append", help="Build only selected fold(s).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--checkpoint-kind",
        choices=["best_ema", "final_ema", "best_raw", "final_raw"],
        default="final_ema",
    )
    return parser.parse_args()


def index_fingerprint(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]


def checkpoint_source_hashes(
    kind: str,
    n_folds: int,
) -> tuple[dict[int, str], str]:
    source_hashes = {}
    dataset_fingerprints = set()
    for fold in range(1, n_folds + 1):
        path = Path(PATHS["model_dir"]) / f"fold{fold}_{kind}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing exact checkpoint for PCA provenance: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not payload.get("config_hash"):
            raise ValueError(f"Checkpoint lacks config_hash provenance: {path}")
        source_hashes[fold] = str(payload["config_hash"])
        dataset_fingerprint = payload.get("dataset_record_order_fingerprint")
        if not dataset_fingerprint:
            raise ValueError(
                f"Checkpoint lacks dataset record-order provenance: {path}"
            )
        dataset_fingerprints.add(str(dataset_fingerprint))
        del payload
    if len(set(source_hashes.values())) != 1:
        raise ValueError(
            f"Checkpoint config hashes differ across folds: {source_hashes}"
        )
    if len(dataset_fingerprints) != 1:
        raise ValueError(
            "Checkpoint dataset record-order fingerprints differ across folds: "
            f"{sorted(dataset_fingerprints)}"
        )
    return source_hashes, next(iter(dataset_fingerprints))


def pca_path(
    fold: int,
    train_indices: np.ndarray,
    source_config_hash: str,
) -> Path:
    return Path(PATHS["cache_dir"]) / "revision_pca_models" / (
        f"fold{fold}_pca_v{CACHE_SCHEMA_VERSION}_{source_config_hash}_"
        f"train{len(train_indices)}_{index_fingerprint(train_indices)}_"
        f"D{CONFIG['hydra_dim']}.joblib"
    )


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    folds_path = Path(PATHS["model_dir"]) / "folds.pkl"
    if not folds_path.exists():
        raise FileNotFoundError(folds_path)
    folds_path_sha256 = sha256_file(folds_path)
    folds = joblib.load(folds_path)
    source_hashes, dataset_record_fingerprint = checkpoint_source_hashes(
        args.checkpoint_kind,
        len(folds),
    )
    common_source_hash = next(iter(source_hashes.values()))
    requested = set(args.fold or range(1, len(folds) + 1))
    if not requested.issubset(set(range(1, len(folds) + 1))):
        raise ValueError(f"Invalid folds requested: {sorted(requested)}")

    raw_features = None
    rows = []
    for fold_num, fold in enumerate(folds, start=1):
        if fold_num not in requested:
            continue
        train_indices = np.asarray(fold["tr_idx"], dtype=np.int64)
        source_config_hash = source_hashes[fold_num]
        destination = pca_path(fold_num, train_indices, source_config_hash)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.force:
            pca = joblib.load(destination)
            print(f"Reusing fold {fold_num} PCA: {destination}")
        else:
            if raw_features is None:
                X, _, _, subjects = load_chapman_multilabel()
                raw_features = generate_raw_rocket_cache(X, subjects)
            print(
                f"Fitting fold {fold_num} PCA: train={len(train_indices)} "
                f"raw_dim={raw_features.shape[1]} -> {CONFIG['hydra_dim']}",
                flush=True,
            )
            pca = fit_pca_on_train(raw_features[train_indices], CONFIG["hydra_dim"])
            joblib.dump(pca, destination)
            print(f"Wrote: {destination}")
        rows.append(
            {
                "fold": fold_num,
                "path": str(destination),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "train_records": int(len(train_indices)),
                "train_index_hash": index_fingerprint(train_indices),
                "n_components": int(pca.n_components_),
                "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                "source_config_hash": source_config_hash,
                "checkpoint_kind": args.checkpoint_kind,
            }
        )

    manifest_path = MANIFEST_DIR / "fold_pca_manifest.json"
    prior = {}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("folds_path_sha256") != folds_path_sha256
            or prior.get("source_config_hash") != common_source_hash
            or prior.get("checkpoint_kind") != args.checkpoint_kind
            or prior.get("dataset_record_order_fingerprint")
            != dataset_record_fingerprint
        ):
            prior = {}
    merged = {
        int(row["fold"]): row
        for row in prior.get("fold_pca", [])
        if int(row["fold"]) not in requested
    }
    merged.update({int(row["fold"]): row for row in rows})
    payload = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_config_hash": common_source_hash,
        "dataset_record_order_fingerprint": dataset_record_fingerprint,
        "checkpoint_kind": args.checkpoint_kind,
        "folds_path": str(folds_path),
        "folds_path_sha256": folds_path_sha256,
        "fold_pca": [merged[key] for key in sorted(merged)],
        "complete": set(merged) == set(range(1, len(folds) + 1)),
    }
    save_json(manifest_path, payload)
    print(json.dumps(payload, indent=2))
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
