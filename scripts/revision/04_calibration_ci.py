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
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    REVISION_DIR,
    bootstrap_ci,
    calibration_summary,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="NPZ with y_true and y_prob arrays.")
    parser.add_argument("--out", default=str(REVISION_DIR / "calibration_ci.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    data = np.load(pred_path, allow_pickle=True)
    y_true = data["y_true"].astype(np.float32)
    y_prob = data["y_prob"].astype(np.float32)

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

    payload = {
        "predictions": str(pred_path),
        "shape": {"y_true": list(y_true.shape), "y_prob": list(y_prob.shape)},
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "metrics": metrics,
        "calibration": calibration,
        "bootstrap_ci": ci,
    }

    save_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()

