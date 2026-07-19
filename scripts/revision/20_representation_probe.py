"""Representation probe/CKA runner for branch-specific evidence.

This script consumes a frozen embedding NPZ plus its checkpoint-local fold
caches. Each probe scaler/classifier is fit on train embeddings produced by one
fold checkpoint and evaluated on validation embeddings from that same
checkpoint. Global OOF embeddings are retained only for descriptive projection.
When the provenance package is absent or stale, the script writes a blocked
manifest so the evidence matrix can record the gap without overclaiming.

Expected NPZ keys:

- ``y_true``: (N, C) binary labels.
- ``record_id``: (N,) record identifiers.
- ``fold_id``: (N,) frozen fold identifiers.
- ``class_names``: (C,) class names.
- one or more embedding arrays named ``morphology_embedding``,
  ``rhythm_embedding``, ``context_embedding``, or ``fused_embedding``.
- ``local_fold_cache_index_json``: authenticated train/validation cache index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    FIGURE_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    ensure_revision_dirs,
    git_commit,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_csv,
    save_json,
    sha256_file,
)


PROTOCOL = "representation_probe_fold_safe_v3_projection_and_fold_audit"
LOCAL_COORDINATE_PROTOCOL = "checkpoint_local_train_validation_embeddings_v1"
LOCAL_COORDINATE_SCHEMA_VERSION = 1
VIEW_KEYS = {
    "morphology": ("morphology_embedding", "morphology", "rocket_embedding"),
    "rhythm": ("rhythm_embedding", "rhythm", "hrv_embedding"),
    "context": ("context_embedding", "context", "mamba_embedding"),
    "fused": ("fused_embedding", "fused", "final_embedding"),
}
DEFAULT_MORPHOLOGY_LABELS = "LBBB,RBBB,CRBBB,IRBBB,IAVB,LAnFB,NSIVCB,QAb,TAb,TInv,LAD,RAD,LQRSV"
DEFAULT_RHYTHM_LABELS = "AF,AFL,Brady,SA,SB,SNR,STach,PAC,PVC,SVPB,VPB,PR,LPR,LQT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-npz", type=Path, default=None)
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
    parser.add_argument("--morphology-labels", default=DEFAULT_MORPHOLOGY_LABELS)
    parser.add_argument("--rhythm-labels", default=DEFAULT_RHYTHM_LABELS)
    parser.add_argument("--max-cka-records", type=int, default=6000)
    parser.add_argument("--max-plot-records", type=int, default=3000)
    parser.add_argument("--probe-max-iter", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-summary", type=Path, default=METRIC_DIR / "representation_probe_summary.json")
    parser.add_argument("--out-probe-table", type=Path, default=TABLE_DIR / "table_representation_probe.csv")
    parser.add_argument(
        "--out-fold-probe-table",
        type=Path,
        default=TABLE_DIR / "table_representation_probe_by_fold.csv",
    )
    parser.add_argument("--out-cka-table", type=Path, default=TABLE_DIR / "table_representation_cka.csv")
    parser.add_argument(
        "--out-audit-figure",
        type=Path,
        default=FIGURE_DIR / "figure_representation_audit.png",
    )
    parser.add_argument("--out-manifest", type=Path, default=MANIFEST_DIR / "representation_probe_manifest.json")
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


def array_sha256(array: np.ndarray, dtype: np.dtype | type | None = None) -> str:
    values = np.asarray(array, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def current_extraction_source_bundle_sha256() -> str:
    paths = [
        PROJECT_ROOT / "scripts" / "revision" / "22_extract_representations.py",
        PROJECT_ROOT / "scripts" / "revision" / "01_generate_predictions.py",
        PROJECT_ROOT / "configs" / "config.py",
        PROJECT_ROOT / "src" / "data_loader.py",
        PROJECT_ROOT / "src" / "features.py",
        PROJECT_ROOT / "src" / "layers.py",
        PROJECT_ROOT / "src" / "model.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_label_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def blocked_payload(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "status": "blocked_missing_embeddings",
        "protocol": PROTOCOL,
        "created_utc": now_utc(),
        "reason": reason,
        "safe_wording": (
            "Representation separation remains unproven. Do not claim proven "
            "morphology-rhythm disentanglement without embedding/probe artifacts."
        ),
        "required_embedding_keys": [
            "y_true",
            "record_id",
            "fold_id",
            "class_names",
            "morphology_embedding or rhythm_embedding or context_embedding or fused_embedding",
        ],
        "outputs": {
            "summary_json": project_relative(args.out_summary),
            "probe_table": project_relative(args.out_probe_table),
            "fold_probe_table": project_relative(args.out_fold_probe_table),
            "cka_table": project_relative(args.out_cka_table),
            "audit_figure": project_relative(args.out_audit_figure),
            "manifest": project_relative(args.out_manifest),
        },
        "git_commit": git_commit(),
    }


def load_embeddings(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding NPZ: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in ["y_true", "record_id", "fold_id", "class_names"] if key not in data.files]
        if missing:
            raise KeyError(f"Missing required keys: {missing}")
        def source_scalar(key: str, default: str = "") -> str:
            if key not in data.files:
                return default
            value = data[key]
            return str(value.item() if np.ndim(value) == 0 else value)

        payload: dict[str, Any] = {
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]).astype(str),
            "fold_id": np.asarray(data["fold_id"]).astype(int),
            "class_names": np.asarray(data["class_names"]).astype(str),
            "views": {},
            "path": path,
            "sha256": sha256_file(path),
            "source_protocol": source_scalar("protocol"),
            "source_oof_sha256": source_scalar("oof_predictions_sha256"),
            "source_freeze_sha256": source_scalar("freeze_manifest_sha256"),
            "source_bundle_sha256": source_scalar("source_bundle_sha256"),
            "coordinate_protocol": source_scalar("coordinate_protocol"),
            "checkpoint_kind": source_scalar("checkpoint_kind"),
            "local_fold_cache_index": (
                json.loads(source_scalar("local_fold_cache_index_json", "[]"))
                if "local_fold_cache_index_json" in data.files
                else []
            ),
        }
        n = len(payload["record_id"])
        for view, candidates in VIEW_KEYS.items():
            for key in candidates:
                if key in data.files:
                    arr = np.asarray(data[key], dtype=np.float32)
                    if arr.ndim != 2 or arr.shape[0] != n:
                        raise ValueError(f"Invalid {key} shape: {arr.shape}; expected ({n}, D)")
                    if not np.isfinite(arr).all():
                        raise ValueError(f"Embedding view {key} contains NaN/Inf values")
                    payload["views"][view] = arr
                    break
    if not payload["views"]:
        raise KeyError("No recognized embedding view keys found.")
    if payload["y_true"].ndim != 2 or payload["y_true"].shape[0] != len(payload["record_id"]):
        raise ValueError("y_true row count does not match record_id count.")
    if len(np.unique(payload["record_id"])) != len(payload["record_id"]):
        raise ValueError("record_id values are not unique")
    if len(payload["fold_id"]) != len(payload["record_id"]):
        raise ValueError("fold_id row count does not match record_id count")
    if not set(np.unique(payload["fold_id"])).issubset({1, 2, 3, 4, 5}):
        raise ValueError(f"Unexpected fold identifiers: {sorted(np.unique(payload['fold_id']).tolist())}")
    if len(payload["class_names"]) != payload["y_true"].shape[1]:
        raise ValueError("class_names length does not match label width")
    if payload.get("source_protocol") != "ecg_ramba_final_ema_branch_embedding_extraction_v1":
        raise RuntimeError("Embedding artifact uses an unsupported or missing extraction protocol")
    if not payload.get("source_oof_sha256") or not payload.get("source_freeze_sha256"):
        raise RuntimeError("Embedding artifact lacks canonical OOF/freeze provenance")
    if payload.get("coordinate_protocol") != LOCAL_COORDINATE_PROTOCOL:
        raise RuntimeError(
            "Embedding artifact lacks checkpoint-local train/validation coordinates"
        )
    if not payload.get("source_bundle_sha256"):
        raise RuntimeError("Embedding artifact lacks extraction source-bundle provenance")
    if not isinstance(payload.get("local_fold_cache_index"), list):
        raise RuntimeError("Embedding artifact has an invalid local fold cache index")
    if not np.isfinite(payload["y_true"]).all() or not np.isin(payload["y_true"], [0.0, 1.0]).all():
        raise ValueError("y_true must contain finite binary labels")
    return payload


def load_current_canonical_contract(
    oof_path: Path,
    freeze_manifest_path: Path,
) -> dict[str, Any]:
    oof_path = resolve(oof_path)
    freeze_manifest_path = resolve(freeze_manifest_path)
    if not oof_path.exists() or oof_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing canonical OOF predictions: {oof_path}")
    if not freeze_manifest_path.exists() or freeze_manifest_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing canonical freeze manifest: {freeze_manifest_path}")
    with np.load(oof_path, allow_pickle=False) as data:
        required = {"y_true", "record_id", "fold_id", "class_names"}
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"Canonical OOF artifact missing keys: {missing}")
        payload = {
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]).astype(str),
            "fold_id": np.asarray(data["fold_id"], dtype=np.int16),
            "class_names": np.asarray(data["class_names"]).astype(str),
        }
    payload.update(
        {
            "oof_path": oof_path,
            "freeze_manifest_path": freeze_manifest_path,
            "oof_sha256": sha256_file(oof_path),
            "freeze_sha256": sha256_file(freeze_manifest_path),
        }
    )
    freeze_payload = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    expected_oof_sha = freeze_payload.get("record_file_sha256") or freeze_payload.get(
        "predictions_sha256"
    )
    if expected_oof_sha is None:
        for artifact in freeze_payload.get("artifacts", []):
            if str(artifact.get("path", "")).endswith(oof_path.name):
                expected_oof_sha = artifact.get("sha256")
                break
    if not expected_oof_sha:
        raise RuntimeError("Canonical freeze manifest does not declare the OOF artifact SHA256")
    if str(expected_oof_sha) != payload["oof_sha256"]:
        raise RuntimeError(
            "Canonical freeze manifest does not authenticate the current OOF artifact"
        )
    if (
        not np.isfinite(payload["y_true"]).all()
        or not np.isin(payload["y_true"], [0.0, 1.0]).all()
    ):
        raise ValueError("Canonical OOF y_true must contain finite binary labels")
    if len(np.unique(payload["record_id"])) != len(payload["record_id"]):
        raise ValueError("Canonical OOF record_id values are not unique")
    return payload


def validate_embedding_against_canonical(
    emb: dict[str, Any],
    canonical: dict[str, Any],
) -> None:
    errors: list[str] = []
    if emb.get("source_oof_sha256") != canonical["oof_sha256"]:
        errors.append("oof_predictions_sha256")
    if emb.get("source_freeze_sha256") != canonical["freeze_sha256"]:
        errors.append("freeze_manifest_sha256")
    if emb.get("source_bundle_sha256") != current_extraction_source_bundle_sha256():
        errors.append("source_bundle_sha256")
    for key in ("y_true", "record_id", "fold_id", "class_names"):
        if not np.array_equal(np.asarray(emb[key]), np.asarray(canonical[key])):
            errors.append(f"semantic:{key}")
    if errors:
        raise RuntimeError(
            "Embedding artifact does not match the current canonical OOF/freeze contract: "
            + ", ".join(errors)
        )


def _resolve_cache_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_checkpoint_local_folds(
    emb: dict[str, Any],
    canonical: dict[str, Any],
) -> list[dict[str, Any]]:
    if emb.get("source_oof_sha256") != canonical["oof_sha256"] or emb.get(
        "source_freeze_sha256"
    ) != canonical["freeze_sha256"]:
        raise RuntimeError(
            "Embedding source hashes do not match the current canonical OOF/freeze contract"
        )
    index_rows = emb.get("local_fold_cache_index", [])
    index_by_fold = {
        int(row.get("fold", -1)): row
        for row in index_rows
        if int(row.get("fold", -1)) > 0
    }
    expected_folds = sorted(int(value) for value in np.unique(canonical["fold_id"]))
    if sorted(index_by_fold) != expected_folds:
        raise RuntimeError(
            "Checkpoint-local fold cache index is incomplete: "
            f"observed={sorted(index_by_fold)} expected={expected_folds}"
        )

    local_folds: list[dict[str, Any]] = []
    validation_coverage = np.zeros(len(canonical["record_id"]), dtype=np.int16)
    for fold in expected_folds:
        row = index_by_fold[fold]
        path = _resolve_cache_path(str(row.get("path") or ""))
        expected_sha = str(row.get("sha256") or "")
        if not path.exists() or not expected_sha or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Checkpoint-local cache SHA mismatch for fold {fold}: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {
                "fold",
                "coordinate_protocol",
                "local_coordinate_schema_version",
                "checkpoint_sha256",
                "oof_predictions_sha256",
                "freeze_manifest_sha256",
                "source_bundle_sha256",
                "train_record_id",
                "validation_record_id",
                "train_index_sha256",
                "validation_index_sha256",
                *{f"train_{candidates[0]}" for candidates in VIEW_KEYS.values()},
                *{f"validation_{candidates[0]}" for candidates in VIEW_KEYS.values()},
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise RuntimeError(f"Fold {fold} local cache missing keys: {missing}")
            train_ids = np.asarray(data["train_record_id"], dtype=np.int64)
            validation_ids = np.asarray(data["validation_record_id"], dtype=np.int64)
            if int(data["fold"]) != fold:
                raise RuntimeError(f"Fold identity mismatch in local cache {path}")
            scalar_contract = {
                "coordinate_protocol": str(data["coordinate_protocol"].item()),
                "local_coordinate_schema_version": int(data["local_coordinate_schema_version"]),
                "checkpoint_sha256": str(data["checkpoint_sha256"].item()),
                "oof_predictions_sha256": str(data["oof_predictions_sha256"].item()),
                "freeze_manifest_sha256": str(data["freeze_manifest_sha256"].item()),
                "source_bundle_sha256": str(data["source_bundle_sha256"].item()),
                "train_index_sha256": str(data["train_index_sha256"].item()),
                "validation_index_sha256": str(data["validation_index_sha256"].item()),
            }
            views = {
                view: {
                    "train": np.asarray(data[f"train_{key_candidates[0]}"], dtype=np.float32),
                    "validation": np.asarray(
                        data[f"validation_{key_candidates[0]}"], dtype=np.float32
                    ),
                }
                for view, key_candidates in VIEW_KEYS.items()
            }

        expected_train = np.flatnonzero(canonical["fold_id"] != fold).astype(np.int64)
        expected_validation = np.flatnonzero(canonical["fold_id"] == fold).astype(np.int64)
        if not np.array_equal(train_ids, expected_train) or not np.array_equal(
            validation_ids, expected_validation
        ):
            raise RuntimeError(f"Fold {fold} local cache membership differs from canonical OOF")
        if np.intersect1d(train_ids, validation_ids).size:
            raise RuntimeError(f"Fold {fold} local train/validation records overlap")
        if scalar_contract["coordinate_protocol"] != LOCAL_COORDINATE_PROTOCOL:
            raise RuntimeError(f"Fold {fold} local coordinate protocol mismatch")
        if scalar_contract["local_coordinate_schema_version"] != LOCAL_COORDINATE_SCHEMA_VERSION:
            raise RuntimeError(f"Fold {fold} local coordinate schema mismatch")
        if scalar_contract["oof_predictions_sha256"] != canonical["oof_sha256"]:
            raise RuntimeError(f"Fold {fold} cache is stale for canonical OOF")
        if scalar_contract["freeze_manifest_sha256"] != canonical["freeze_sha256"]:
            raise RuntimeError(f"Fold {fold} cache is stale for canonical freeze manifest")
        if scalar_contract["source_bundle_sha256"] != emb["source_bundle_sha256"]:
            raise RuntimeError(f"Fold {fold} extraction source bundle mismatch")
        if scalar_contract["checkpoint_sha256"] != str(row.get("checkpoint_sha256") or ""):
            raise RuntimeError(f"Fold {fold} checkpoint SHA mismatch")
        expected_coordinate_id = f"fold{fold}:{scalar_contract['checkpoint_sha256']}"
        if str(row.get("coordinate_system_id") or "") != expected_coordinate_id:
            raise RuntimeError(f"Fold {fold} coordinate-system identity mismatch")
        if scalar_contract["train_index_sha256"] != array_sha256(train_ids, np.int64):
            raise RuntimeError(f"Fold {fold} train index hash mismatch")
        if scalar_contract["validation_index_sha256"] != array_sha256(
            validation_ids, np.int64
        ):
            raise RuntimeError(f"Fold {fold} validation index hash mismatch")
        if str(row.get("train_index_sha256") or "") != scalar_contract[
            "train_index_sha256"
        ]:
            raise RuntimeError(f"Fold {fold} cache-index train hash mismatch")
        if str(row.get("validation_index_sha256") or "") != scalar_contract[
            "validation_index_sha256"
        ]:
            raise RuntimeError(f"Fold {fold} cache-index validation hash mismatch")
        for view, split_views in views.items():
            if split_views["train"].shape[0] != len(train_ids):
                raise RuntimeError(f"Fold {fold} {view} train row count mismatch")
            if split_views["validation"].shape[0] != len(validation_ids):
                raise RuntimeError(f"Fold {fold} {view} validation row count mismatch")
            if not all(np.isfinite(values).all() for values in split_views.values()):
                raise RuntimeError(f"Fold {fold} {view} contains non-finite embeddings")
        validation_coverage[validation_ids] += 1
        local_folds.append(
            {
                "fold_id": fold,
                "coordinate_system_id": str(
                    row.get("coordinate_system_id")
                    or f"fold{fold}:{scalar_contract['checkpoint_sha256']}"
                ),
                "checkpoint_sha256": scalar_contract["checkpoint_sha256"],
                "train_record_id": train_ids,
                "validation_record_id": validation_ids,
                "views": views,
            }
        )
    if not np.all(validation_coverage == 1):
        raise RuntimeError("Checkpoint-local validation partitions do not cover each record once")
    return local_folds


def validate_global_embedding_projection(
    emb: dict[str, Any],
    local_folds: list[dict[str, Any]],
) -> None:
    for local_fold in local_folds:
        fold = int(local_fold["fold_id"])
        validation_ids = np.asarray(local_fold["validation_record_id"], dtype=np.int64)
        for view, global_values in emb["views"].items():
            if not np.array_equal(
                np.asarray(global_values, dtype=np.float32)[validation_ids],
                local_fold["views"][view]["validation"],
            ):
                raise RuntimeError(
                    f"Global OOF projection differs from checkpoint-local validation "
                    f"embeddings for fold {fold}/{view}"
                )


def label_indices(class_names: np.ndarray, requested: list[str]) -> list[int]:
    lookup = {name: idx for idx, name in enumerate(class_names.tolist())}
    return [lookup[name] for name in requested if name in lookup]


def linear_cka(x: np.ndarray, y: np.ndarray, max_records: int, seed: int) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have the same row count.")
    n = x.shape[0]
    if max_records > 0 and n > max_records:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_records, replace=False))
        x = x[idx]
        y = y[idx]
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    numerator = float(np.sum(xty * xty))
    denominator = float(np.sqrt(np.sum(xtx * xtx) * np.sum(yty * yty)))
    return numerator / denominator if denominator > 0 else float("nan")


def probe_one_view(
    local_folds: list[dict[str, Any]],
    view: str,
    y: np.ndarray,
    threshold: float,
    seed: int,
    max_iter: int,
) -> dict[str, Any]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    valid_cols = [c for c in range(y.shape[1]) if len(np.unique(y[:, c])) >= 2]
    if not valid_cols:
        return {"status": "blocked_no_binary_labels"}
    yv = y[:, valid_cols]
    pred = np.full(yv.shape, np.nan, dtype=np.float32)
    convergence_warning_count = 0
    convergence_limited_estimators = 0
    solver_iter_max = 0
    fold_rows: list[dict[str, Any]] = []
    for local_fold in local_folds:
        fold = int(local_fold["fold_id"])
        train_ids = np.asarray(local_fold["train_record_id"], dtype=np.int64)
        validation_ids = np.asarray(local_fold["validation_record_id"], dtype=np.int64)
        x_train = np.asarray(local_fold["views"][view]["train"], dtype=np.float32)
        x_validation = np.asarray(
            local_fold["views"][view]["validation"], dtype=np.float32
        )
        if x_train.shape[0] != len(train_ids) or x_validation.shape[0] != len(
            validation_ids
        ):
            raise RuntimeError(f"Fold {fold} {view} local embedding rows are misaligned")
        if x_train.shape[1] != x_validation.shape[1]:
            raise RuntimeError(f"Fold {fold} {view} train/validation dimensions differ")

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_validation_scaled = scaler.transform(x_validation)
        y_train = yv[train_ids]
        y_validation = yv[validation_ids]
        fold_prob = np.tile(
            np.mean(y_train, axis=0, keepdims=True),
            (len(validation_ids), 1),
        ).astype(np.float32)
        usable_cols: list[int] = []
        for column in range(yv.shape[1]):
            if len(np.unique(y_train[:, column])) < 2:
                continue
            estimator = LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=max_iter,
                random_state=seed + fold,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(x_train_scaled, y_train[:, column])
            convergence_warning_count += sum(
                1 for warning in caught if issubclass(warning.category, ConvergenceWarning)
            )
            estimator_iter = int(np.max(estimator.n_iter_))
            solver_iter_max = max(solver_iter_max, estimator_iter)
            if estimator_iter >= max_iter:
                convergence_limited_estimators += 1
            fold_prob[:, column] = estimator.predict_proba(x_validation_scaled)[:, 1]
            usable_cols.append(column)

        pred[validation_ids] = fold_prob
        fold_metrics = multilabel_metrics(y_validation, fold_prob, threshold=threshold)
        fold_metrics["macro_pr_auc"] = macro_pr_auc(y_validation, fold_prob)
        fold_metrics["macro_roc_auc"] = macro_roc_auc(y_validation, fold_prob)
        fold_rows.append(
            {
                "fold_id": int(fold),
                "coordinate_system_id": local_fold["coordinate_system_id"],
                "checkpoint_sha256": local_fold["checkpoint_sha256"],
                "n_train_records": int(len(train_ids)),
                "n_records_evaluated": int(len(validation_ids)),
                "n_labels_evaluated": int(len(valid_cols)),
                "n_labels_fit": int(len(usable_cols)),
                **fold_metrics,
            }
        )
    covered = np.all(np.isfinite(pred), axis=1)
    if not np.any(covered):
        return {"status": "blocked_no_fold_predictions"}
    metrics = multilabel_metrics(yv[covered], pred[covered], threshold=threshold)
    metrics["macro_pr_auc"] = macro_pr_auc(yv[covered], pred[covered])
    metrics["macro_roc_auc"] = macro_roc_auc(yv[covered], pred[covered])
    metrics["n_labels_evaluated"] = int(len(valid_cols))
    metrics["n_records_evaluated"] = int(np.sum(covered))
    metrics["probe_max_iter"] = int(max_iter)
    metrics["solver_iter_max"] = int(solver_iter_max)
    metrics["convergence_warning_count"] = int(convergence_warning_count)
    metrics["convergence_limited_estimators"] = int(convergence_limited_estimators)
    metrics["coordinate_design"] = LOCAL_COORDINATE_PROTOCOL
    metrics["_fold_rows"] = fold_rows
    metrics["status"] = "complete"
    return metrics


def project_embedding(x: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return np.asarray([], dtype=np.int64), np.empty((0, 2), dtype=np.float32), "unavailable"

    rng = np.random.default_rng(args.seed)
    n = x.shape[0]
    idx = np.arange(n)
    if args.max_plot_records > 0 and n > args.max_plot_records:
        idx = np.sort(rng.choice(n, size=args.max_plot_records, replace=False))
    x_plot = x[idx]
    method = "pca"
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=args.seed, n_jobs=1)
        coords = reducer.fit_transform(x_plot)
        method = "umap"
    except Exception:
        coords = PCA(n_components=2, random_state=args.seed).fit_transform(x_plot)
    return idx.astype(np.int64), np.asarray(coords, dtype=np.float32), method


def write_view_projection(
    view_name: str,
    idx: np.ndarray,
    coords: np.ndarray,
    method: str,
    fold_id: np.ndarray,
) -> str:
    if not len(idx):
        return ""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fold_plot = fold_id[idx]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"representation_{method}_{view_name}.png"
    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=fold_plot, s=4, cmap="tab10", alpha=0.75)
    plt.title(f"{view_name} embedding ({method.upper()}, colored by fold)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.colorbar(scatter, label="fold")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return project_relative(out)


def representation_label_groups(
    y_true: np.ndarray,
    morphology_idx: list[int],
    rhythm_idx: list[int],
) -> tuple[np.ndarray, list[str]]:
    morphology_positive = (
        np.any(y_true[:, morphology_idx] > 0, axis=1) if morphology_idx else np.zeros(len(y_true), dtype=bool)
    )
    rhythm_positive = (
        np.any(y_true[:, rhythm_idx] > 0, axis=1) if rhythm_idx else np.zeros(len(y_true), dtype=bool)
    )
    category = morphology_positive.astype(np.int8) + 2 * rhythm_positive.astype(np.int8)
    return category, ["neither", "morphology only", "rhythm only", "both"]


def write_representation_audit_figure(
    projections: dict[str, tuple[np.ndarray, np.ndarray, str]],
    y_true: np.ndarray,
    fold_id: np.ndarray,
    morphology_idx: list[int],
    rhythm_idx: list[int],
    out_path: Path,
) -> str:
    available = [(name, value) for name, value in projections.items() if len(value[0])]
    if not available:
        return ""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    category, category_names = representation_label_groups(y_true, morphology_idx, rhythm_idx)
    fold_colors = plt.get_cmap("tab10")
    category_colors = ["#808080", "#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(len(available), 2, figsize=(10, 3.1 * len(available)), squeeze=False)
    for row_index, (view_name, (idx, coords, method)) in enumerate(available):
        ax_fold, ax_label = axes[row_index]
        fold_plot = fold_id[idx]
        category_plot = category[idx]
        for fold in sorted(np.unique(fold_plot)):
            mask = fold_plot == fold
            ax_fold.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=5,
                alpha=0.55,
                color=fold_colors(int(fold) - 1),
                linewidths=0,
                label=f"Fold {int(fold)}",
            )
        for value, label in enumerate(category_names):
            mask = category_plot == value
            if np.any(mask):
                ax_label.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=5,
                    alpha=0.55,
                    color=category_colors[value],
                    linewidths=0,
                    label=label,
                )
        ax_fold.set_title(f"{view_name}: OOF fold ({method.upper()})")
        ax_label.set_title(f"{view_name}: diagnostic label group ({method.upper()})")
        for axis in (ax_fold, ax_label):
            axis.set_xlabel("Projection 1")
            axis.set_ylabel("Projection 2")
            axis.grid(False)
    fold_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=fold_colors(fold - 1), label=f"Fold {fold}")
        for fold in range(1, 6)
    ]
    label_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=label)
        for color, label in zip(category_colors, category_names)
    ]
    axes[0, 0].legend(handles=fold_handles, loc="best", fontsize=8, frameon=False)
    axes[0, 1].legend(handles=label_handles, loc="best", fontsize=8, frameon=False)
    fig.suptitle("Branch representation audit (descriptive projection; not evidence of disentanglement)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return project_relative(out_path)


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    for path in [
        args.out_summary,
        args.out_probe_table,
        args.out_fold_probe_table,
        args.out_cka_table,
        args.out_audit_figure,
        args.out_manifest,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("REPRESENTATION PROBE / CKA GATE", flush=True)
    print("=" * 80, flush=True)

    try:
        if args.embedding_npz is None:
            raise FileNotFoundError("No --embedding-npz was provided.")
        emb = load_embeddings(args.embedding_npz)
        canonical = load_current_canonical_contract(
            args.oof_predictions,
            args.freeze_manifest,
        )
        validate_embedding_against_canonical(emb, canonical)
        local_folds = load_checkpoint_local_folds(emb, canonical)
        validate_global_embedding_projection(emb, local_folds)
    except Exception as exc:
        payload = blocked_payload(args, str(exc))
        save_json(args.out_summary, payload)
        save_json(args.out_manifest, payload)
        save_csv(args.out_probe_table, [payload])
        save_csv(args.out_fold_probe_table, [payload])
        save_csv(args.out_cka_table, [payload])
        print(json.dumps(payload, indent=2), flush=True)
        if args.strict:
            raise
        return

    y_true = emb["y_true"]
    fold_id = emb["fold_id"]
    class_names = emb["class_names"]
    morphology_idx = label_indices(class_names, parse_label_list(args.morphology_labels))
    rhythm_idx = label_indices(class_names, parse_label_list(args.rhythm_labels))
    groups = {
        "morphology_labels": morphology_idx,
        "rhythm_labels": rhythm_idx,
        "all_labels": list(range(y_true.shape[1])),
    }

    probe_rows: list[dict[str, Any]] = []
    fold_probe_rows: list[dict[str, Any]] = []
    figure_paths: dict[str, str] = {}
    projections: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for view, x in emb["views"].items():
        projection = project_embedding(x, args)
        projections[view] = projection
        figure_paths[view] = write_view_projection(view, *projection, fold_id)
        for group_name, idx in groups.items():
            if not idx:
                probe_rows.append(
                    {
                        "view": view,
                        "label_group": group_name,
                        "status": "blocked_no_matching_labels",
                    }
                )
                continue
            result = probe_one_view(
                local_folds,
                view,
                y_true[:, idx],
                args.threshold,
                args.seed,
                args.probe_max_iter,
            )
            view_fold_rows = result.pop("_fold_rows", [])
            fold_probe_rows.extend(
                {"view": view, "label_group": group_name, **fold_row}
                for fold_row in view_fold_rows
            )
            if args.strict and (
                int(result.get("convergence_warning_count", 0)) > 0
                or int(result.get("convergence_limited_estimators", 0)) > 0
            ):
                raise RuntimeError(
                    f"Probe convergence gate failed for {view}/{group_name}: {result}"
                )
            probe_rows.append(
                {
                    "view": view,
                    "label_group": group_name,
                    **result,
                }
            )
            print(f"probe {view}/{group_name}: {result}", flush=True)

    figure_paths["combined_audit"] = write_representation_audit_figure(
        projections,
        y_true,
        fold_id,
        morphology_idx,
        rhythm_idx,
        args.out_audit_figure,
    )

    cka_rows: list[dict[str, Any]] = []
    for local_fold in local_folds:
        fold = int(local_fold["fold_id"])
        for left, right in combinations(sorted(emb["views"]), 2):
            left_values = local_fold["views"][left]["validation"]
            right_values = local_fold["views"][right]["validation"]
            value = linear_cka(
                left_values,
                right_values,
                args.max_cka_records,
                args.seed + int(fold),
            )
            cka_rows.append(
                {
                    "scope": "checkpoint_local_validation",
                    "fold_id": int(fold),
                    "coordinate_system_id": local_fold["coordinate_system_id"],
                    "checkpoint_sha256": local_fold["checkpoint_sha256"],
                    "left_view": left,
                    "right_view": right,
                    "linear_cka": float(value),
                    "n_records": int(len(local_fold["validation_record_id"])),
                    "max_records": int(args.max_cka_records),
                    "status": "complete" if np.isfinite(value) else "blocked_nonfinite",
                }
            )
            print(
                f"cka checkpoint_local_validation/fold{fold} {left}/{right}: "
                f"{value:.6f}",
                flush=True,
            )

    payload = {
        "status": "complete",
        "protocol": PROTOCOL,
        "created_utc": now_utc(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "canonical_contract": {
            "oof_sha256": canonical["oof_sha256"],
            "freeze_sha256": canonical["freeze_sha256"],
        },
        "safe_wording": (
            "Representation probes provide suggestive branch-specific evidence only; "
            "they do not establish strict morphology-rhythm separation."
        ),
        "embedding_npz": {
            "path": project_relative(emb["path"]),
            "sha256": emb["sha256"],
        },
        "n_records": int(len(emb["record_id"])),
        "n_classes": int(y_true.shape[1]),
        "coordinate_design": LOCAL_COORDINATE_PROTOCOL,
        "n_checkpoint_local_folds": int(len(local_folds)),
        "projection_role": (
            "descriptive_only; global OOF projections mix checkpoint coordinate systems"
        ),
        "probe_max_iter": int(args.probe_max_iter),
        "views": {name: list(arr.shape) for name, arr in emb["views"].items()},
        "label_groups": {
            "morphology_labels": class_names[morphology_idx].tolist() if morphology_idx else [],
            "rhythm_labels": class_names[rhythm_idx].tolist() if rhythm_idx else [],
        },
        "figures": figure_paths,
        "outputs": {
            "summary_json": project_relative(args.out_summary),
            "probe_table": project_relative(args.out_probe_table),
            "fold_probe_table": project_relative(args.out_fold_probe_table),
            "cka_table": project_relative(args.out_cka_table),
            "audit_figure": figure_paths.get("combined_audit", ""),
            "manifest": project_relative(args.out_manifest),
        },
        "git_commit": git_commit(),
    }
    save_csv(args.out_probe_table, probe_rows)
    save_csv(args.out_fold_probe_table, fold_probe_rows)
    save_csv(args.out_cka_table, cka_rows)
    payload["artifact_sha256"] = {
        "probe_table": sha256_file(resolve(args.out_probe_table)),
        "fold_probe_table": sha256_file(resolve(args.out_fold_probe_table)),
        "cka_table": sha256_file(resolve(args.out_cka_table)),
        "audit_figure": (
            sha256_file(resolve(args.out_audit_figure))
            if resolve(args.out_audit_figure).exists()
            else ""
        ),
    }
    save_json(
        args.out_summary,
        {**payload, "probe_rows": probe_rows, "fold_probe_rows": fold_probe_rows, "cka_rows": cka_rows},
    )
    save_json(args.out_manifest, payload)
    print(json.dumps({"status": True, "views": list(emb["views"]), "manifest": project_relative(args.out_manifest)}, indent=2))


if __name__ == "__main__":
    main()
