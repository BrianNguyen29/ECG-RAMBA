"""Representation probe/CKA runner for branch-specific evidence.

This script consumes a frozen embedding NPZ artifact. It does not extract
embeddings from the model by itself; that extraction must be produced by a
separate, checkpoint-fingerprinted hook if stronger representation evidence is
needed. When no embedding artifact is provided, the script writes a blocked
manifest so the evidence matrix can record the gap without overclaiming.

Expected NPZ keys:

- ``y_true``: (N, C) binary labels.
- ``record_id``: (N,) record identifiers.
- ``fold_id``: (N,) frozen fold identifiers.
- ``class_names``: (C,) class names.
- one or more embedding arrays named ``morphology_embedding``,
  ``rhythm_embedding``, ``context_embedding``, or ``fused_embedding``.
"""

from __future__ import annotations

import argparse
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


PROTOCOL = "representation_probe_fold_safe_v2_maxiter_trace"
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
    parser.add_argument("--morphology-labels", default=DEFAULT_MORPHOLOGY_LABELS)
    parser.add_argument("--rhythm-labels", default=DEFAULT_RHYTHM_LABELS)
    parser.add_argument("--max-cka-records", type=int, default=6000)
    parser.add_argument("--max-plot-records", type=int, default=3000)
    parser.add_argument("--probe-max-iter", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-summary", type=Path, default=METRIC_DIR / "representation_probe_summary.json")
    parser.add_argument("--out-probe-table", type=Path, default=TABLE_DIR / "table_representation_probe.csv")
    parser.add_argument("--out-cka-table", type=Path, default=TABLE_DIR / "table_representation_cka.csv")
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
            "cka_table": project_relative(args.out_cka_table),
            "manifest": project_relative(args.out_manifest),
        },
        "git_commit": git_commit(),
    }


def load_embeddings(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in ["y_true", "record_id", "fold_id", "class_names"] if key not in data.files]
        if missing:
            raise KeyError(f"Missing required keys: {missing}")
        payload: dict[str, Any] = {
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]).astype(str),
            "fold_id": np.asarray(data["fold_id"]).astype(int),
            "class_names": np.asarray(data["class_names"]).astype(str),
            "views": {},
            "path": path,
            "sha256": sha256_file(path),
        }
        n = len(payload["record_id"])
        for view, candidates in VIEW_KEYS.items():
            for key in candidates:
                if key in data.files:
                    arr = np.asarray(data[key], dtype=np.float32)
                    if arr.ndim != 2 or arr.shape[0] != n:
                        raise ValueError(f"Invalid {key} shape: {arr.shape}; expected ({n}, D)")
                    payload["views"][view] = arr
                    break
    if not payload["views"]:
        raise KeyError("No recognized embedding view keys found.")
    if payload["y_true"].shape[0] != len(payload["record_id"]):
        raise ValueError("y_true row count does not match record_id count.")
    return payload


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
    x: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    threshold: float,
    seed: int,
    max_iter: int,
) -> dict[str, Any]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    valid_cols = [c for c in range(y.shape[1]) if len(np.unique(y[:, c])) >= 2]
    if not valid_cols:
        return {"status": "blocked_no_binary_labels"}
    yv = y[:, valid_cols]
    pred = np.full(yv.shape, np.nan, dtype=np.float32)
    convergence_warning_count = 0
    convergence_limited_estimators = 0
    solver_iter_max = 0
    for fold in sorted(np.unique(fold_id)):
        train = fold_id != fold
        test = fold_id == fold
        if not np.any(test) or not np.any(train):
            continue
        usable_cols = [c for c in range(yv.shape[1]) if len(np.unique(yv[train, c])) >= 2]
        if not usable_cols:
            continue
        model = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=seed,
                )
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x[train], yv[train][:, usable_cols])
        convergence_warning_count += sum(
            1 for warning in caught if issubclass(warning.category, ConvergenceWarning)
        )
        classifier = model.named_steps["onevsrestclassifier"]
        for estimator in getattr(classifier, "estimators_", []):
            n_iter = getattr(estimator, "n_iter_", np.asarray([], dtype=np.int32))
            if len(n_iter):
                estimator_iter = int(np.max(n_iter))
                solver_iter_max = max(solver_iter_max, estimator_iter)
                if estimator_iter >= max_iter:
                    convergence_limited_estimators += 1
        fold_prob = np.zeros((int(np.sum(test)), yv.shape[1]), dtype=np.float32)
        fold_prob[:] = np.mean(yv[train], axis=0, keepdims=True)
        predicted = model.predict_proba(x[test])
        if isinstance(predicted, list):
            predicted = np.stack([p[:, 1] for p in predicted], axis=1)
        fold_prob[:, usable_cols] = np.asarray(predicted, dtype=np.float32)
        pred[test] = fold_prob
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
    metrics["status"] = "complete"
    return metrics


def maybe_write_umap(view_name: str, x: np.ndarray, fold_id: np.ndarray, args: argparse.Namespace) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception:
        return ""

    rng = np.random.default_rng(args.seed)
    n = x.shape[0]
    idx = np.arange(n)
    if args.max_plot_records > 0 and n > args.max_plot_records:
        idx = np.sort(rng.choice(n, size=args.max_plot_records, replace=False))
    x_plot = x[idx]
    fold_plot = fold_id[idx]
    method = "pca"
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        coords = reducer.fit_transform(x_plot)
        method = "umap"
    except Exception:
        coords = PCA(n_components=2, random_state=args.seed).fit_transform(x_plot)
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


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    for path in [args.out_summary, args.out_probe_table, args.out_cka_table, args.out_manifest]:
        path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("REPRESENTATION PROBE / CKA GATE", flush=True)
    print("=" * 80, flush=True)

    try:
        if args.embedding_npz is None:
            raise FileNotFoundError("No --embedding-npz was provided.")
        emb = load_embeddings(args.embedding_npz)
    except Exception as exc:
        payload = blocked_payload(args, str(exc))
        save_json(args.out_summary, payload)
        save_json(args.out_manifest, payload)
        save_csv(args.out_probe_table, [payload])
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
    figure_paths: dict[str, str] = {}
    for view, x in emb["views"].items():
        figure_paths[view] = maybe_write_umap(view, x, fold_id, args)
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
                x,
                y_true[:, idx],
                fold_id,
                args.threshold,
                args.seed,
                args.probe_max_iter,
            )
            probe_rows.append(
                {
                    "view": view,
                    "label_group": group_name,
                    **result,
                }
            )
            print(f"probe {view}/{group_name}: {result}", flush=True)

    cka_rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(emb["views"]), 2):
        value = linear_cka(emb["views"][left], emb["views"][right], args.max_cka_records, args.seed)
        cka_rows.append(
            {
                "left_view": left,
                "right_view": right,
                "linear_cka": float(value),
                "max_records": int(args.max_cka_records),
                "status": "complete" if np.isfinite(value) else "blocked_nonfinite",
            }
        )
        print(f"cka {left}/{right}: {value:.6f}", flush=True)

    payload = {
        "status": "complete",
        "protocol": PROTOCOL,
        "created_utc": now_utc(),
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
            "cka_table": project_relative(args.out_cka_table),
            "manifest": project_relative(args.out_manifest),
        },
        "git_commit": git_commit(),
    }
    save_csv(args.out_probe_table, probe_rows)
    save_csv(args.out_cka_table, cka_rows)
    save_json(args.out_summary, {**payload, "probe_rows": probe_rows, "cka_rows": cka_rows})
    save_json(args.out_manifest, payload)
    print(json.dumps({"status": True, "views": list(emb["views"]), "manifest": project_relative(args.out_manifest)}, indent=2))


if __name__ == "__main__":
    main()
