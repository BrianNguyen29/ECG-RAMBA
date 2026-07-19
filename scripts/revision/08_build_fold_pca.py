"""Build the five fold-specific PCA objects required for external inference.

This reuses the saved Chapman MiniRocket cache and does not run model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


PCA_INTEGRITY_SCHEMA_VERSION = 1


def load_prediction_generator():
    path = Path(__file__).with_name("01_generate_predictions.py")
    spec = importlib.util.spec_from_file_location("_ecg_ramba_oof_cache_contract", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def pca_integrity_contract_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".artifact.json")


def validate_pca_object(
    pca: object,
    *,
    expected_raw_dim: int,
    expected_components: int,
) -> None:
    components = np.asarray(getattr(pca, "components_", None))
    explained = np.asarray(getattr(pca, "explained_variance_ratio_", None))
    n_components = int(getattr(pca, "n_components_", -1))
    if n_components != expected_components:
        raise ValueError(
            f"PCA n_components mismatch: {n_components} != {expected_components}"
        )
    if components.shape != (expected_components, expected_raw_dim):
        raise ValueError(
            "PCA components shape mismatch: "
            f"{components.shape} != {(expected_components, expected_raw_dim)}"
        )
    if explained.shape != (expected_components,):
        raise ValueError(
            f"PCA explained-variance shape mismatch: {explained.shape}"
        )
    if not np.isfinite(components).all() or not np.isfinite(explained).all():
        raise ValueError("PCA artifact contains non-finite fitted parameters")


def bind_pca_artifact(
    *,
    path: Path,
    cache_contract: dict,
    expected_raw_dim: int,
    expected_components: int,
) -> dict:
    contract_path = path.with_suffix(path.suffix + ".contract.json")
    persisted_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if persisted_contract != cache_contract:
        raise RuntimeError(f"PCA cache contract mismatch after write: {contract_path}")
    pca = joblib.load(path)
    validate_pca_object(
        pca,
        expected_raw_dim=expected_raw_dim,
        expected_components=expected_components,
    )
    payload = {
        "schema_version": PCA_INTEGRITY_SCHEMA_VERSION,
        "artifact_kind": "fold_train_pca_joblib",
        "artifact_name": path.name,
        "artifact_sha256": sha256_file(path),
        "artifact_size_bytes": int(path.stat().st_size),
        "cache_contract_sha256": str(cache_contract["contract_sha256"]),
        "cache_contract_file_sha256": sha256_file(contract_path),
        "expected_raw_dim": int(expected_raw_dim),
        "expected_components": int(expected_components),
    }
    save_json(pca_integrity_contract_path(path), payload)
    return payload


def load_verified_pca_artifact(
    *,
    path: Path,
    cache_contract: dict,
    expected_raw_dim: int,
    expected_components: int,
) -> tuple[object, dict] | None:
    contract_path = path.with_suffix(path.suffix + ".contract.json")
    integrity_path = pca_integrity_contract_path(path)
    if not path.is_file() or not contract_path.is_file() or not integrity_path.is_file():
        return None
    try:
        persisted_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        if persisted_contract != cache_contract:
            return None
        if (
            int(integrity.get("schema_version", 0)) != PCA_INTEGRITY_SCHEMA_VERSION
            or integrity.get("artifact_kind") != "fold_train_pca_joblib"
            or integrity.get("artifact_name") != path.name
            or integrity.get("artifact_sha256") != sha256_file(path)
            or int(integrity.get("artifact_size_bytes", -1)) != path.stat().st_size
            or integrity.get("cache_contract_sha256")
            != cache_contract.get("contract_sha256")
            or integrity.get("cache_contract_file_sha256") != sha256_file(contract_path)
            or int(integrity.get("expected_raw_dim", -1)) != expected_raw_dim
            or int(integrity.get("expected_components", -1)) != expected_components
        ):
            return None
        sha_before = sha256_file(path)
        pca = joblib.load(path)
        if sha256_file(path) != sha_before or sha_before != integrity["artifact_sha256"]:
            return None
        validate_pca_object(
            pca,
            expected_raw_dim=expected_raw_dim,
            expected_components=expected_components,
        )
        return pca, integrity
    except Exception:
        return None


def pca_manifest_row(
    *,
    fold_num: int,
    destination: Path,
    train_indices: np.ndarray,
    source_config_hash: str,
    checkpoint_kind: str,
    cache_contract: dict,
    pca: object,
    integrity: dict,
) -> dict:
    contract_path = destination.with_suffix(destination.suffix + ".contract.json")
    integrity_path = pca_integrity_contract_path(destination)
    return {
        "fold": fold_num,
        "path": str(destination),
        "sha256": str(integrity["artifact_sha256"]),
        "size_bytes": int(integrity["artifact_size_bytes"]),
        "train_records": int(len(train_indices)),
        "train_index_hash": index_fingerprint(train_indices),
        "n_components": int(getattr(pca, "n_components_")),
        "explained_variance": float(
            np.sum(np.asarray(getattr(pca, "explained_variance_ratio_")))
        ),
        "source_config_hash": source_config_hash,
        "checkpoint_kind": checkpoint_kind,
        "cache_contract_sha256": cache_contract["contract_sha256"],
        "cache_contract_path": str(contract_path),
        "cache_contract_file_sha256": sha256_file(contract_path),
        "integrity_contract_path": str(integrity_path),
        "integrity_contract_sha256": sha256_file(integrity_path),
    }


def recover_verified_pca_manifest_row(
    *,
    fold_num: int,
    destination: Path,
    train_indices: np.ndarray,
    source_config_hash: str,
    checkpoint_kind: str,
    cache_contract: dict,
    prior_row: dict | None,
    expected_raw_dim: int,
    expected_components: int,
) -> dict | None:
    if prior_row is not None and Path(str(prior_row.get("path", ""))) != destination:
        return None
    verified = load_verified_pca_artifact(
        path=destination,
        cache_contract=cache_contract,
        expected_raw_dim=expected_raw_dim,
        expected_components=expected_components,
    )
    if verified is None:
        return None
    pca, integrity = verified
    if prior_row is not None:
        try:
            prior_matches = (
                prior_row.get("sha256") == integrity["artifact_sha256"]
                and int(prior_row.get("size_bytes", -1))
                == integrity["artifact_size_bytes"]
                and prior_row.get("cache_contract_sha256")
                == cache_contract["contract_sha256"]
            )
        except (TypeError, ValueError):
            prior_matches = False
        if not prior_matches:
            return None
    return pca_manifest_row(
        fold_num=fold_num,
        destination=destination,
        train_indices=train_indices,
        source_config_hash=source_config_hash,
        checkpoint_kind=checkpoint_kind,
        cache_contract=cache_contract,
        pca=pca,
        integrity=integrity,
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

    generator = load_prediction_generator()
    X, _, _, subjects = load_chapman_multilabel()
    raw_features, rocket_contract = generate_raw_rocket_cache(
        X,
        subjects,
        return_contract=True,
    )
    loaded_record_fingerprint = generator.record_order_fingerprint(subjects)
    if loaded_record_fingerprint != dataset_record_fingerprint:
        raise RuntimeError(
            "Loaded Chapman record order does not match the checkpoint contract"
        )
    cache_provenance = generator.build_oof_cache_provenance(
        rocket_contract=rocket_contract,
        dataset_record_fingerprint=loaded_record_fingerprint,
    )
    fold_specs: dict[int, dict] = {}
    for fold_num, fold in enumerate(folds, start=1):
        train_indices = np.asarray(fold["tr_idx"], dtype=np.int64)
        source_config_hash = source_hashes[fold_num]
        cache_contract = generator.scoped_cache_contract(
            cache_provenance,
            artifact_kind="fold_train_pca",
            fold_num=fold_num,
            tr_idx=train_indices,
            va_idx=None,
            source_config_hash=source_config_hash,
        )
        destination = generator.fold_pca_model_path(
            fold_num,
            train_indices,
            source_config_hash,
            cache_contract,
        )
        fold_specs[fold_num] = {
            "train_indices": train_indices,
            "source_config_hash": source_config_hash,
            "cache_contract": cache_contract,
            "destination": destination,
        }

    rows_by_fold: dict[int, dict] = {}
    for fold_num in sorted(requested):
        spec = fold_specs[fold_num]
        train_indices = spec["train_indices"]
        source_config_hash = spec["source_config_hash"]
        cache_contract = spec["cache_contract"]
        destination = spec["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        verified = None if args.force else load_verified_pca_artifact(
            path=destination,
            cache_contract=cache_contract,
            expected_raw_dim=int(raw_features.shape[1]),
            expected_components=int(CONFIG["hydra_dim"]),
        )
        if verified is not None:
            pca, integrity = verified
            print(f"Reusing fold {fold_num} PCA: {destination}")
        else:
            print(
                f"Fitting fold {fold_num} PCA: train={len(train_indices)} "
                f"raw_dim={raw_features.shape[1]} -> {CONFIG['hydra_dim']}",
                flush=True,
            )
            pca = fit_pca_on_train(raw_features[train_indices], CONFIG["hydra_dim"])
            generator._atomic_joblib_dump(pca, destination, cache_contract)
            bind_pca_artifact(
                path=destination,
                cache_contract=cache_contract,
                expected_raw_dim=int(raw_features.shape[1]),
                expected_components=int(CONFIG["hydra_dim"]),
            )
            verified = load_verified_pca_artifact(
                path=destination,
                cache_contract=cache_contract,
                expected_raw_dim=int(raw_features.shape[1]),
                expected_components=int(CONFIG["hydra_dim"]),
            )
            if verified is None:
                raise RuntimeError(
                    f"Fold {fold_num} PCA failed post-write integrity validation: {destination}"
                )
            pca, integrity = verified
            print(f"Wrote: {destination}")
        rows_by_fold[fold_num] = pca_manifest_row(
            fold_num=fold_num,
            destination=destination,
            train_indices=train_indices,
            source_config_hash=source_config_hash,
            checkpoint_kind=args.checkpoint_kind,
            cache_contract=cache_contract,
            pca=pca,
            integrity=integrity,
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
    prior_by_fold = {
        int(row["fold"]): row
        for row in prior.get("fold_pca", [])
        if isinstance(row, dict) and int(row.get("fold", -1)) > 0
    }
    invalid_folds: list[int] = []
    for fold_num, spec in sorted(fold_specs.items()):
        if fold_num in rows_by_fold:
            continue
        recovered = recover_verified_pca_manifest_row(
            fold_num=fold_num,
            destination=spec["destination"],
            train_indices=spec["train_indices"],
            source_config_hash=spec["source_config_hash"],
            checkpoint_kind=args.checkpoint_kind,
            cache_contract=spec["cache_contract"],
            prior_row=prior_by_fold.get(fold_num),
            expected_raw_dim=int(raw_features.shape[1]),
            expected_components=int(CONFIG["hydra_dim"]),
        )
        if recovered is None:
            invalid_folds.append(fold_num)
            continue
        rows_by_fold[fold_num] = recovered

    expected_folds = set(range(1, len(folds) + 1))
    payload = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_config_hash": common_source_hash,
        "dataset_record_order_fingerprint": dataset_record_fingerprint,
        "checkpoint_kind": args.checkpoint_kind,
        "folds_path": str(folds_path),
        "folds_path_sha256": folds_path_sha256,
        "fold_pca": [rows_by_fold[key] for key in sorted(rows_by_fold)],
        "validated_folds": sorted(rows_by_fold),
        "invalid_or_missing_folds": sorted(set(invalid_folds) | (expected_folds - set(rows_by_fold))),
        "complete": set(rows_by_fold) == expected_folds,
    }
    save_json(manifest_path, payload)
    print(json.dumps(payload, indent=2))
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
