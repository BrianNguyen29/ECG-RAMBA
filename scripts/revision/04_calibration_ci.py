"""Compute calibration metrics and bootstrap CI from saved prediction files.

Expected NPZ keys:
    y_true: shape (N, C)
    y_prob: shape (N, C)

Example:
    python scripts/revision/04_calibration_ci.py \
      --predictions reports/revision/oof_predictions.npz
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    FIGURE_DIR,
    REVISION_DIR,
    TABLE_DIR,
    bootstrap_ci,
    calibration_summary,
    ece_binary,
    macro_pr_auc,
    macro_roc_auc,
    mce_binary,
    multilabel_metrics,
    save_json,
    sha256_file,
)


def scalar_from_npz(
    data: np.lib.npyio.NpzFile | dict[str, np.ndarray],
    key: str,
    default: str | None = None,
) -> str | None:
    if key not in data:
        return default
    value = data[key]
    if np.ndim(value) == 0:
        return str(value.item())
    return str(value)


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_prediction_payload(pred_path: Path) -> dict[str, np.ndarray]:
    with np.load(pred_path, allow_pickle=False) as data:
        missing_keys = {"y_true", "y_prob"} - set(data.files)
        if missing_keys:
            raise KeyError(
                f"{pred_path} is not a metric prediction file; missing keys: {sorted(missing_keys)}"
            )
        return {key: data[key].copy() for key in data.files}


def validate_prediction_payload(
    payload: dict[str, np.ndarray],
    pred_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    y_true = np.asarray(payload["y_true"], dtype=np.float32)
    y_prob = np.asarray(payload["y_prob"], dtype=np.float32)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_prob {y_prob.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D multi-label arrays, found shape {y_true.shape}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError(f"Prediction artifact contains NaN/Inf values: {pred_path}")
    if not np.isin(y_true, [0.0, 1.0]).all():
        raise ValueError(f"y_true must be binary 0/1 labels: {pred_path}")
    if float(np.min(y_prob)) < 0.0 or float(np.max(y_prob)) > 1.0:
        raise ValueError(f"y_prob must be probabilities in [0, 1]: {pred_path}")

    class_names = None
    if "class_names" in payload:
        class_names = np.asarray(payload["class_names"]).astype(str).tolist()
        if len(class_names) != y_true.shape[1]:
            raise ValueError(
                f"class_names length {len(class_names)} does not match prediction width {y_true.shape[1]}"
            )
    return y_true, y_prob, class_names


def validate_freeze_manifest(
    *,
    freeze_path: Path,
    pred_path: Path,
    y_true: np.ndarray,
    class_names: list[str] | None,
) -> str:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("manuscript_ready") is not True:
        raise ValueError(f"Invalid OOF freeze manifest: {freeze_path}")
    relative = pred_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    frozen = {row["path"]: row for row in freeze.get("artifacts", [])}
    if relative not in frozen:
        raise ValueError(f"Freeze manifest does not contain prediction file: {relative}")
    if sha256_file(pred_path) != frozen[relative]["sha256"]:
        raise RuntimeError(f"Prediction checksum changed after freeze: {relative}")
    if int(freeze.get("validated_records", y_true.shape[0])) != y_true.shape[0]:
        raise ValueError("Prediction record count differs from freeze manifest")
    if int(freeze.get("n_classes", y_true.shape[1])) != y_true.shape[1]:
        raise ValueError("Prediction class count differs from freeze manifest")
    if class_names is not None and freeze.get("class_names") and freeze["class_names"] != class_names:
        raise ValueError("Prediction class order differs from freeze manifest")
    return sha256_file(freeze_path)


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> list[dict]:
    y_true_flat = np.asarray(y_true, dtype=float).ravel()
    y_prob_flat = np.asarray(y_prob, dtype=float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_prob_flat >= lo) & (y_prob_flat < hi if hi < 1.0 else y_prob_flat <= hi)
        if not np.any(mask):
            rows.append(
                {
                    "bin": i,
                    "lo": float(lo),
                    "hi": float(hi),
                    "count": 0,
                    "confidence": float("nan"),
                    "empirical_rate": float("nan"),
                    "abs_gap": float("nan"),
                }
            )
            continue
        confidence = float(np.mean(y_prob_flat[mask]))
        empirical_rate = float(np.mean(y_true_flat[mask]))
        rows.append(
            {
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "count": int(np.sum(mask)),
                "confidence": confidence,
                "empirical_rate": empirical_rate,
                "abs_gap": abs(empirical_rate - confidence),
            }
        )
    return rows


def calibration_micro_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> dict:
    from sklearn.metrics import brier_score_loss

    yt = np.asarray(y_true, dtype=np.float32).ravel()
    yp = np.asarray(y_prob, dtype=np.float32).ravel()
    return {
        "ece_micro": ece_binary(yt, yp, n_bins=n_bins),
        "mce_micro": mce_binary(yt, yp, n_bins=n_bins),
        "brier_micro": float(brier_score_loss(yt, yp)),
        "n_label_record_pairs": int(len(yt)),
    }


def per_class_calibration_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None,
    n_bins: int,
) -> list[dict]:
    from sklearn.metrics import brier_score_loss

    names = class_names or [f"class_{idx}" for idx in range(y_true.shape[1])]
    rows = []
    for idx, name in enumerate(names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        has_both = len(np.unique(yt)) >= 2
        rows.append(
            {
                "class_index": idx,
                "class_name": str(name),
                "n_records": int(len(yt)),
                "n_positive": int(np.sum(yt)),
                "prevalence": float(np.mean(yt)),
                "ece": ece_binary(yt, yp, n_bins=n_bins) if has_both else float("nan"),
                "mce": mce_binary(yt, yp, n_bins=n_bins) if has_both else float("nan"),
                "brier": float(brier_score_loss(yt, yp)) if has_both else float("nan"),
                "evaluated": bool(has_both),
            }
        )
    return rows


def save_csv_rows(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_reliability_figure(rows: list[dict], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plotted = [r for r in rows if r["count"] > 0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1, label="Perfect calibration")
    if plotted:
        ax.plot(
            [r["confidence"] for r in plotted],
            [r["empirical_rate"] for r in plotted],
            marker="o",
            linewidth=1.5,
            color="#1f77b4",
            label="Observed",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="NPZ with y_true and y_prob arrays.")
    parser.add_argument("--out", default=str(REVISION_DIR / "calibration_ci.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        help="Require the prediction checksum to match this freeze manifest.",
    )
    parser.add_argument(
        "--require-manuscript-ready",
        action="store_true",
        help="Reject prediction files explicitly marked manuscript_ready=false.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    data = load_prediction_payload(pred_path)
    if args.require_manuscript_ready and "manuscript_ready" in data:
        if not bool(data["manuscript_ready"].item()):
            raise ValueError(f"Prediction artifact is not manuscript-ready: {pred_path}")

    y_true, y_prob, class_names = validate_prediction_payload(data, pred_path)
    freeze_manifest_sha256 = None
    if args.freeze_manifest:
        freeze_manifest_sha256 = validate_freeze_manifest(
            freeze_path=args.freeze_manifest,
            pred_path=pred_path,
            y_true=y_true,
            class_names=class_names,
        )

    metrics = multilabel_metrics(y_true, y_prob, threshold=args.threshold)
    calibration = calibration_summary(y_true, y_prob, n_bins=args.n_bins)
    calibration_micro = calibration_micro_summary(y_true, y_prob, n_bins=args.n_bins)
    ci = {
        "macro_pr_auc": bootstrap_ci(
            y_true, y_prob, macro_pr_auc, n_boot=args.n_boot, seed=args.seed
        ),
        "macro_roc_auc": bootstrap_ci(
            y_true, y_prob, macro_roc_auc, n_boot=args.n_boot, seed=args.seed
        ),
        "f1_macro": bootstrap_ci(
            y_true,
            y_prob,
            lambda yt, yp: multilabel_metrics(yt, yp, threshold=args.threshold)["f1_macro"],
            n_boot=args.n_boot,
            seed=args.seed,
        ),
        "ece_macro": bootstrap_ci(
            y_true,
            y_prob,
            lambda yt, yp: calibration_summary(yt, yp, n_bins=args.n_bins)["ece_macro"],
            n_boot=args.n_boot,
            seed=args.seed,
        ),
        "brier_macro": bootstrap_ci(
            y_true,
            y_prob,
            lambda yt, yp: calibration_summary(yt, yp, n_bins=args.n_bins)["brier_macro"],
            n_boot=args.n_boot,
            seed=args.seed,
        ),
    }
    dataset = scalar_from_npz(data, "dataset", pred_path.stem)
    protocol = scalar_from_npz(data, "protocol", None)
    reliability = reliability_bins(y_true, y_prob, n_bins=args.n_bins)
    reliability_csv = TABLE_DIR / f"reliability_bins_{pred_path.stem}.csv"
    class_calibration_csv = TABLE_DIR / f"calibration_by_class_{pred_path.stem}.csv"
    reliability_fig = FIGURE_DIR / f"reliability_{pred_path.stem}.png"
    save_csv_rows(reliability_csv, reliability)
    save_csv_rows(
        class_calibration_csv,
        per_class_calibration_rows(y_true, y_prob, class_names, n_bins=args.n_bins),
    )
    write_reliability_figure(reliability, reliability_fig, title=f"Reliability: {dataset}")

    payload = {
        "predictions": str(pred_path),
        "predictions_sha256": sha256_file(pred_path),
        "freeze_manifest": str(args.freeze_manifest) if args.freeze_manifest else None,
        "freeze_manifest_sha256": freeze_manifest_sha256,
        "dataset": dataset,
        "protocol": protocol,
        "class_names": class_names,
        "git_commit": current_git_commit(),
        "shape": {"y_true": list(y_true.shape), "y_prob": list(y_prob.shape)},
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "metrics": metrics,
        "calibration": calibration,
        "calibration_micro": calibration_micro,
        "reliability": {
            "scope": "micro_flattened_label_record_pairs",
            "description": "All record-class probability pairs are binned together for the reliability curve.",
            "n_bins": args.n_bins,
        },
        "bootstrap_ci": ci,
        "artifacts": {
            "reliability_bins_csv": str(reliability_csv),
            "class_calibration_csv": str(class_calibration_csv),
            "reliability_figure": str(reliability_fig),
        },
    }

    save_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()

