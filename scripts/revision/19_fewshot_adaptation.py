"""Few-shot adaptation under a passed external protocol gate.

This runner intentionally starts with a conservative, reproducible adaptation
mode: per-class score calibration on frozen external predictions. It does not
fine-tune ECG-RAMBA weights and therefore must not be described as model-level
few-shot transfer. Its purpose is to provide a leakage-audited target-domain
few-shot sensitivity analysis after a dataset-specific external gate has passed.

If the requested dataset has not passed ``18_external_protocol_gate.py``, the
script writes blocked/deferred artifacts and exits successfully unless
``--strict`` is supplied. This prevents accidental overclaiming.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    TABLE_DIR,
    bootstrap_ci,
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


PROTOCOL = "fewshot_score_calibration_v1_gated_external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ptbxl", "georgia", "cpsc2021"])
    parser.add_argument("--external-root", type=Path, default=EXPERIMENTAL_DIR / "external")
    parser.add_argument("--gate-json", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--fractions", default="0,0.01,0.05,0.10")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--test-fraction", type=float, default=0.50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--out-table", type=Path, default=None)
    parser.add_argument("--out-bootstrap", type=Path, default=None)
    parser.add_argument("--out-splits", type=Path, default=None)
    parser.add_argument("--out-manifest", type=Path, default=None)
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


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in items]


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    dataset = args.dataset
    return {
        "gate_json": args.gate_json or METRIC_DIR / f"external_{dataset}_protocol_gate.json",
        "predictions": args.predictions
        or args.external_root
        / dataset
        / f"{dataset}_full_predictions.npz",
        "summary": args.out_summary or METRIC_DIR / f"fewshot_{dataset}_summary.csv",
        "table": args.out_table or TABLE_DIR / f"table_fewshot_{dataset}.csv",
        "bootstrap": args.out_bootstrap or METRIC_DIR / f"fewshot_{dataset}_bootstrap.json",
        "splits": args.out_splits or MANIFEST_DIR / f"fewshot_{dataset}_splits.npz",
        "manifest": args.out_manifest or MANIFEST_DIR / f"fewshot_{dataset}_run_manifest.json",
    }


def blocked_payload(args: argparse.Namespace, paths: dict[str, Path], reason: str) -> dict[str, Any]:
    return {
        "status": "blocked_precondition",
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "created_utc": now_utc(),
        "reason": reason,
        "safe_wording": (
            "Few-shot adaptation remains deferred. Do not state that few-shot "
            "experiments were added until this runner completes on a protocol-gated dataset."
        ),
        "required_preconditions": [
            "external protocol gate must pass",
            "external prediction NPZ must exist",
            "fixed target-domain splits must be recorded",
        ],
        "outputs": {key: project_relative(path) for key, path in paths.items()},
        "git_commit": git_commit(),
    }


def load_gate(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing external gate JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing external predictions NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = ["y_true", "y_prob", "record_id", "class_names"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Missing prediction keys: {missing}")
        payload = {key: data[key] for key in data.files}
    y_true = np.asarray(payload["y_true"], dtype=np.float32)
    y_prob = np.asarray(payload["y_prob"], dtype=np.float32)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if np.any(~np.isfinite(y_prob)) or np.min(y_prob) < 0.0 or np.max(y_prob) > 1.0:
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    payload["path"] = path
    payload["sha256"] = sha256_file(path)
    return payload


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def calibrate_scores(
    y_train: np.ndarray,
    p_train: np.ndarray,
    p_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int, list[str]]:
    """Fit per-class Platt-style score calibration where labels permit it."""
    from sklearn.linear_model import LogisticRegression

    out = np.asarray(p_test, dtype=np.float64).copy()
    adapted = 0
    skipped: list[str] = []
    x_train_all = logit(p_train)
    x_test_all = logit(p_test)
    for c in range(y_train.shape[1]):
        labels = y_train[:, c].astype(int)
        if len(np.unique(labels)) < 2:
            skipped.append(f"class_{c}:single_label")
            continue
        try:
            model = LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            )
            model.fit(x_train_all[:, [c]], labels)
            out[:, c] = model.predict_proba(x_test_all[:, [c]])[:, 1]
            adapted += 1
        except Exception as exc:  # pragma: no cover - defensive artifact trace
            skipped.append(f"class_{c}:{type(exc).__name__}")
    return np.clip(out, 0.0, 1.0).astype(np.float32), adapted, skipped


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bins: int) -> dict[str, float]:
    metrics = multilabel_metrics(y_true, y_prob, threshold=threshold)
    calib = calibration_summary(y_true, y_prob, n_bins=n_bins)
    return {**metrics, **calib}


def bootstrap_bundle(y_true: np.ndarray, y_prob: np.ndarray, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    return {
        "macro_pr_auc": bootstrap_ci(y_true, y_prob, macro_pr_auc, n_boot=args.n_boot, seed=seed),
        "macro_roc_auc": bootstrap_ci(y_true, y_prob, macro_roc_auc, n_boot=args.n_boot, seed=seed + 1),
        "f1_macro": bootstrap_ci(
            y_true,
            y_prob,
            lambda yt, yp: multilabel_metrics(yt, yp, threshold=args.threshold)["f1_macro"],
            n_boot=args.n_boot,
            seed=seed + 2,
        ),
        "brier_macro": bootstrap_ci(
            y_true,
            y_prob,
            lambda yt, yp: calibration_summary(yt, yp, n_bins=args.n_bins)["brier_macro"],
            n_boot=args.n_boot,
            seed=seed + 3,
        ),
        "ece_macro": bootstrap_ci(
            y_true,
            y_prob,
            lambda yt, yp: calibration_summary(yt, yp, n_bins=args.n_bins)["ece_macro"],
            n_boot=args.n_boot,
            seed=seed + 4,
        ),
    }


def make_seed_pool(
    n: int,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    test_n = int(round(n * test_fraction))
    test_n = min(max(test_n, 1), n - 1)
    test_idx = np.sort(order[:test_n])
    pool = order[test_n:].astype(np.int64)
    return pool, test_idx.astype(np.int64)


def train_subset_from_pool(
    pool: np.ndarray,
    n_total: int,
    fraction: float,
) -> np.ndarray:
    if fraction <= 0:
        train_idx = np.asarray([], dtype=np.int64)
    else:
        train_n = int(round(n_total * fraction))
        train_n = min(max(train_n, 1), len(pool))
        train_idx = np.sort(pool[:train_n])
    return train_idx.astype(np.int64)


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    paths = default_paths(args)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("FEW-SHOT SCORE-CALIBRATION GATE", flush=True)
    print("=" * 80, flush=True)
    print(f"dataset={args.dataset} protocol={PROTOCOL}", flush=True)

    try:
        gate = load_gate(paths["gate_json"])
        if gate.get("protocol_gate_passed") is not True or gate.get("manuscript_ready") is not True:
            raise RuntimeError(f"External protocol gate has not passed: {paths['gate_json']}")
        pred = load_predictions(paths["predictions"])
    except Exception as exc:
        payload = blocked_payload(args, paths, str(exc))
        save_json(paths["manifest"], payload)
        save_json(paths["bootstrap"], {"status": "blocked_precondition", "reason": str(exc)})
        save_csv(paths["summary"], [payload])
        save_csv(paths["table"], [payload])
        print(json.dumps(payload, indent=2), flush=True)
        if args.strict:
            raise
        return

    y_true = np.asarray(pred["y_true"], dtype=np.float32)
    y_prob = np.asarray(pred["y_prob"], dtype=np.float32)
    record_id = np.asarray(pred["record_id"]).astype(str)
    class_names = np.asarray(pred["class_names"]).astype(str)
    fractions = sorted(set(parse_float_list(args.fractions)))
    seeds = parse_int_list(args.seeds)
    if not fractions:
        raise ValueError("At least one few-shot fraction is required.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    if any(fraction < 0.0 or fraction > 1.0 for fraction in fractions):
        raise ValueError(f"Few-shot fractions must be in [0, 1], got {fractions}")
    if args.test_fraction <= 0.0 or args.test_fraction >= 1.0:
        raise ValueError(f"--test-fraction must be in (0, 1), got {args.test_fraction}")

    rows: list[dict[str, Any]] = []
    bootstrap_payload: dict[str, Any] = {
        "status": "complete",
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "n_boot": args.n_boot,
        "items": {},
    }
    split_arrays: dict[str, np.ndarray] = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        pool_idx, test_idx = make_seed_pool(len(y_true), args.test_fraction, rng)
        seed_key = f"seed{seed}"
        split_arrays[f"{seed_key}_candidate_pool_index_order"] = pool_idx
        split_arrays[f"{seed_key}_candidate_pool_record_id_order"] = record_id[pool_idx]
        split_arrays[f"{seed_key}_fixed_test_index"] = test_idx
        split_arrays[f"{seed_key}_fixed_test_record_id"] = record_id[test_idx]
        for fraction in fractions:
            train_idx = train_subset_from_pool(pool_idx, len(y_true), fraction)
            split_key = f"seed{seed}_frac{fraction:g}".replace(".", "p")
            split_arrays[f"{split_key}_train_index"] = train_idx
            split_arrays[f"{split_key}_test_index"] = test_idx
            split_arrays[f"{split_key}_train_record_id"] = record_id[train_idx]
            split_arrays[f"{split_key}_test_record_id"] = record_id[test_idx]

            if len(train_idx) == 0:
                adapted_prob = y_prob[test_idx]
                adapted_classes = 0
                skipped = ["zero_shot_identity"]
                mode = "zero_shot_identity"
            else:
                adapted_prob, adapted_classes, skipped = calibrate_scores(
                    y_true[train_idx],
                    y_prob[train_idx],
                    y_prob[test_idx],
                    seed,
                )
                mode = "fewshot_score_calibration"

            bundle = metric_bundle(y_true[test_idx], adapted_prob, args.threshold, args.n_bins)
            row: dict[str, Any] = {
                "dataset": args.dataset,
                "protocol": PROTOCOL,
                "mode": mode,
                "seed": seed,
                "fraction": fraction,
                "train_records": int(len(train_idx)),
                "test_records": int(len(test_idx)),
                "split_policy": "fixed_test_per_seed_nested_train_pool_prefix",
                "n_classes": int(y_true.shape[1]),
                "adapted_classes": int(adapted_classes),
                "skipped_class_count": int(len(skipped)),
                **bundle,
            }
            rows.append(row)
            print(
                f"{split_key}: mode={mode} train={len(train_idx)} test={len(test_idx)} "
                f"F1={bundle['f1_macro']:.4f} PR={bundle['pr_auc_macro']:.4f} "
                f"ROC={bundle['roc_auc_macro']:.4f}",
                flush=True,
            )
            bootstrap_payload["items"][split_key] = {
                "seed": seed,
                "fraction": fraction,
                "mode": mode,
                "train_records": int(len(train_idx)),
                "test_records": int(len(test_idx)),
                "split_policy": "fixed_test_per_seed_nested_train_pool_prefix",
                "adapted_classes": int(adapted_classes),
                "skipped_classes": skipped[:20],
                "bootstrap_ci": bootstrap_bundle(y_true[test_idx], adapted_prob, args, seed),
            }

    np.savez_compressed(
        paths["splits"],
        protocol=np.asarray(PROTOCOL),
        dataset=np.asarray(args.dataset),
        prediction_sha256=np.asarray(pred["sha256"]),
        class_names=class_names,
        **split_arrays,
    )

    manifest = {
        "status": "complete",
        "protocol": PROTOCOL,
        "created_utc": now_utc(),
        "dataset": args.dataset,
        "adaptation_kind": "score_calibration_only",
        "safe_wording": (
            "This is few-shot score calibration on frozen external predictions, "
            "not model-weight fine-tuning and not evidence of general transfer superiority."
        ),
        "gate": {
            "path": project_relative(paths["gate_json"]),
            "sha256": sha256_file(resolve(paths["gate_json"])),
            "gate_cache_key": gate.get("gate_cache_key"),
        },
        "predictions": {
            "path": project_relative(paths["predictions"]),
            "sha256": pred["sha256"],
            "shape": list(y_true.shape),
        },
        "fractions": fractions,
        "seeds": seeds,
        "test_fraction": args.test_fraction,
        "split_policy": "For each seed, the test split is frozen once before any fraction is selected; few-shot train sets are nested prefixes of the remaining shuffled target-domain pool.",
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "class_names": class_names.tolist(),
        "outputs": {key: project_relative(path) for key, path in paths.items()},
        "git_commit": git_commit(),
    }
    save_csv(paths["summary"], rows)
    save_csv(paths["table"], rows)
    save_json(paths["bootstrap"], bootstrap_payload)
    save_json(paths["manifest"], manifest)
    print(json.dumps({"status": True, "rows": len(rows), "manifest": project_relative(paths["manifest"])}, indent=2))


if __name__ == "__main__":
    main()
