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
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_json,
    sha256_file,
)


def scalar_from_npz(data: np.lib.npyio.NpzFile, key: str, default: str | None = None) -> str | None:
    if key not in data.files:
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

    data = np.load(pred_path, allow_pickle=True)
    missing_keys = {"y_true", "y_prob"} - set(data.files)
    if missing_keys:
        raise KeyError(f"{pred_path} is not a metric prediction file; missing keys: {sorted(missing_keys)}")
    if args.require_manuscript_ready and "manuscript_ready" in data.files:
        if not bool(data["manuscript_ready"].item()):
            raise ValueError(f"Prediction artifact is not manuscript-ready: {pred_path}")
    freeze_manifest_sha256 = None
    if args.freeze_manifest:
        freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
        if freeze.get("status") != "frozen" or freeze.get("manuscript_ready") is not True:
            raise ValueError(f"Invalid OOF freeze manifest: {args.freeze_manifest}")
        relative = pred_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        frozen = {row["path"]: row for row in freeze.get("artifacts", [])}
        if relative not in frozen:
            raise ValueError(f"Freeze manifest does not contain prediction file: {relative}")
        if sha256_file(pred_path) != frozen[relative]["sha256"]:
            raise RuntimeError(f"Prediction checksum changed after freeze: {relative}")
        freeze_manifest_sha256 = sha256_file(args.freeze_manifest)

    y_true = data["y_true"].astype(np.float32)
    y_prob = data["y_prob"].astype(np.float32)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_prob {y_prob.shape}")

    metrics = multilabel_metrics(y_true, y_prob, threshold=args.threshold)
    calibration = calibration_summary(y_true, y_prob, n_bins=args.n_bins)
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
    }
    dataset = scalar_from_npz(data, "dataset", pred_path.stem)
    protocol = scalar_from_npz(data, "protocol", None)
    class_names = data["class_names"].astype(str).tolist() if "class_names" in data.files else None
    reliability = reliability_bins(y_true, y_prob, n_bins=args.n_bins)
    reliability_csv = TABLE_DIR / f"reliability_bins_{pred_path.stem}.csv"
    reliability_fig = FIGURE_DIR / f"reliability_{pred_path.stem}.png"
    save_csv_rows(reliability_csv, reliability)
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
        "metrics": metrics,
        "calibration": calibration,
        "bootstrap_ci": ci,
        "artifacts": {
            "reliability_bins_csv": str(reliability_csv),
            "reliability_figure": str(reliability_fig),
        },
    }

    save_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()

