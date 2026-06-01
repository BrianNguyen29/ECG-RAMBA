"""Generate reviewer-ready prediction NPZ files.

Current scope:
    - oof: Chapman out-of-fold predictions using fold checkpoints.

Run from repo root on Colab:
    python scripts/revision/01_generate_predictions.py --dataset oof

Outputs:
    reports/revision/predictions/oof_full_predictions.npz
    reports/revision/predictions/oof_full_slice_predictions.npz
    reports/revision/metrics/oof_full_prediction_summary.json

Prediction files follow the artifact contract in docs/revision_plan/.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG, PATHS, DEVICE  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    METRIC_DIR,
    PREDICTION_DIR,
    ensure_revision_dirs,
    multilabel_metrics,
    power_mean,
    save_json,
)

class ECGSliceDatasetInfer(Dataset):
    def __init__(self, xs: np.ndarray, xh: np.ndarray, xhr: np.ndarray, rids: np.ndarray):
        self.xs = xs
        self.xh = xh
        self.xhr = xhr
        self.rids = rids

    def __len__(self) -> int:
        return len(self.rids)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.xs[idx], dtype=torch.float32),
            torch.tensor(self.xh[idx], dtype=torch.float32),
            torch.tensor(self.xhr[idx], dtype=torch.float32),
            torch.tensor(self.rids[idx], dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["oof"], default="oof")
    parser.add_argument("--checkpoint-kind", choices=["best", "final"], default="best")
    parser.add_argument("--batch-size", type=int, default=max(1, int(CONFIG["batch_size"])))
    parser.add_argument("--limit-records", type=int, default=0, help="Debug only. 0 means all records.")
    parser.add_argument("--save-slice-probs", action="store_true", default=True)
    parser.add_argument("--no-save-slice-probs", dest="save_slice_probs", action="store_false")
    return parser.parse_args()


def slice_record(x: np.ndarray) -> list[np.ndarray]:
    if x.shape[-1] < CONFIG["slice_length"]:
        return []
    slices = []
    for start in range(
        0,
        x.shape[-1] - CONFIG["slice_length"] + 1,
        CONFIG["slice_stride"],
    ):
        slices.append(x[..., start : start + CONFIG["slice_length"]])
        if len(slices) >= CONFIG["max_slices_per_record"]:
            break
    return slices


def load_folds(y: np.ndarray, subjects: np.ndarray) -> list[dict[str, np.ndarray]]:
    folds_path = Path(PATHS["model_dir"]) / "folds.pkl"
    if folds_path.exists():
        folds = joblib.load(folds_path)
        normalized = []
        for fold in folds:
            normalized.append(
                {
                    "tr_idx": np.asarray(fold["tr_idx"], dtype=np.int64),
                    "va_idx": np.asarray(fold["va_idx"], dtype=np.int64),
                }
            )
        print(f"Loaded folds from: {folds_path}")
        return normalized

    print("folds.pkl not found. Recomputing StratifiedGroupKFold from config.")
    from sklearn.model_selection import StratifiedGroupKFold

    y_strat = y.sum(axis=1).clip(max=3).astype(int)
    sgkf = StratifiedGroupKFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seeds"][0],
    )
    return [
        {"tr_idx": tr_idx.astype(np.int64), "va_idx": va_idx.astype(np.int64)}
        for tr_idx, va_idx in sgkf.split(np.zeros(len(y)), y_strat, groups=subjects)
    ]


def checkpoint_path(fold: int, checkpoint_kind: str) -> Path:
    preferred = Path(PATHS["model_dir"]) / f"fold{fold}_{checkpoint_kind}.pt"
    fallback_kind = "final" if checkpoint_kind == "best" else "best"
    fallback = Path(PATHS["model_dir"]) / f"fold{fold}_{fallback_kind}.pt"
    if preferred.exists():
        return preferred
    if fallback.exists():
        print(f"Checkpoint fallback for fold {fold}: {preferred.name} missing, using {fallback.name}")
        return fallback
    raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {preferred} or {fallback}")


def load_model_for_fold(fold: int, checkpoint_kind: str) -> torch.nn.Module:
    from src.model import ECGRambaV7Advanced

    path = checkpoint_path(fold, checkpoint_kind)
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model = ECGRambaV7Advanced(cfg=CONFIG).to(DEVICE)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"Strict checkpoint load failed; retrying strict=False. Reason: {exc}")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def prepare_clean_chapman(limit_records: int = 0):
    from src.data_loader import load_chapman_multilabel

    print("Loading Chapman data")
    X, y, X_raw_amp, subjects = load_chapman_multilabel()
    subjects = np.asarray(subjects)

    min_samples = 5
    class_counts = y.sum(axis=0)
    keep_mask = class_counts >= min_samples
    if not keep_mask.all():
        print(f"Dropping {np.sum(~keep_mask)} classes with <{min_samples} samples")
        y = y[:, keep_mask]
        valid = y.sum(axis=1) > 0
        X, y, X_raw_amp, subjects = X[valid], y[valid], X_raw_amp[valid], subjects[valid]

    if y.shape[1] != len(CLASSES):
        raise ValueError(
            f"Class dimension mismatch after cleaning: y has {y.shape[1]}, config has {len(CLASSES)}"
        )

    if limit_records > 0:
        print(f"Debug limit enabled: first {limit_records} records only")
        X = X[:limit_records]
        y = y[:limit_records]
        X_raw_amp = X_raw_amp[:limit_records]
        subjects = subjects[:limit_records]

    print(f"Chapman records: {len(y)} | classes: {y.shape[1]}")
    return X, y.astype(np.float32), X_raw_amp, subjects


def build_fold_slices(
    va_idx: np.ndarray,
    X: np.ndarray,
    X_hrv: np.ndarray,
    hydra_va_by_record: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    xs, xh, xhr, rids = [], [], [], []
    slice_counts: dict[int, int] = {}

    for rid in va_idx:
        slices = slice_record(X[rid])
        slice_counts[int(rid)] = len(slices)
        for signal_slice in slices:
            xs.append(signal_slice)
            xh.append(hydra_va_by_record[int(rid)])
            xhr.append(X_hrv[rid])
            rids.append(int(rid))

    if not xs:
        return (
            np.empty((0, 12, CONFIG["slice_length"]), dtype=np.float32),
            np.empty((0, CONFIG["hydra_dim"]), dtype=np.float32),
            np.empty((0, CONFIG["hrv_dim"]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            slice_counts,
        )

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(xh, dtype=np.float32),
        np.asarray(xhr, dtype=np.float32),
        np.asarray(rids, dtype=np.int64),
        slice_counts,
    )


def infer_slices(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_probs, all_rids = [], []
    with torch.no_grad():
        for x, xh, xhr, rid in tqdm(loader, desc="Inference", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            xh = xh.to(DEVICE, non_blocking=True)
            xhr = xhr.to(DEVICE, non_blocking=True)

            if DEVICE == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(x, xh, xhr)
            else:
                logits = model(x, xh, xhr)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_probs.append(probs)
            all_rids.append(rid.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_rids, axis=0)


def generate_oof(args: argparse.Namespace) -> None:
    from src.features import (
        apply_pca,
        fit_pca_on_train,
        generate_hrv_cache,
        generate_raw_rocket_cache,
    )
    from src.utils import set_seed

    ensure_revision_dirs()
    set_seed(CONFIG["seeds"][0])

    X, y, X_raw_amp, subjects = prepare_clean_chapman(limit_records=args.limit_records)
    n_records, n_classes = y.shape

    X_rocket_raw = generate_raw_rocket_cache(X)
    X_hrv = generate_hrv_cache(X, X_raw_amp) if CONFIG["use_hrv"] else np.zeros(
        (n_records, CONFIG["hrv_dim"]), dtype=np.float32
    )
    folds = load_folds(y, subjects)

    if args.limit_records > 0:
        allowed = set(range(n_records))
        folds = [
            {
                "tr_idx": np.asarray([idx for idx in fold["tr_idx"] if int(idx) in allowed], dtype=np.int64),
                "va_idx": np.asarray([idx for idx in fold["va_idx"] if int(idx) in allowed], dtype=np.int64),
            }
            for fold in folds
        ]

    oof_probs = np.zeros((n_records, n_classes), dtype=np.float32)
    fold_id = np.zeros(n_records, dtype=np.int16) - 1
    record_slice_count = np.zeros(n_records, dtype=np.int16)
    pca_variance = []
    slice_probs_all = []
    slice_record_index_all = []

    for fold_num, fold in enumerate(folds, start=1):
        tr_idx = fold["tr_idx"]
        va_idx = fold["va_idx"]
        if len(va_idx) == 0:
            print(f"Fold {fold_num}: no validation records after filtering; skipping")
            continue

        print("\n" + "=" * 80)
        print(f"Fold {fold_num}/{len(folds)} | train={len(tr_idx)} | val={len(va_idx)}")
        print("=" * 80)

        pca = fit_pca_on_train(X_rocket_raw[tr_idx], CONFIG["hydra_dim"])
        hydra_va = apply_pca(pca, X_rocket_raw[va_idx])
        pca_var = float(pca.explained_variance_ratio_.sum())
        pca_variance.append({"fold": fold_num, "explained_variance": pca_var})
        print(f"PCA variance retained: {pca_var:.6f}")

        hydra_va_by_record = {
            int(record_idx): hydra_va[offset]
            for offset, record_idx in enumerate(va_idx)
        }
        xs, xh, xhr, rids, slice_counts = build_fold_slices(
            va_idx=va_idx,
            X=X,
            X_hrv=X_hrv,
            hydra_va_by_record=hydra_va_by_record,
        )
        for rid, count in slice_counts.items():
            record_slice_count[rid] = count

        if len(rids) == 0:
            print(f"Fold {fold_num}: no slices generated; skipping")
            continue

        loader = DataLoader(
            ECGSliceDatasetInfer(xs, xh, xhr, rids),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=DEVICE == "cuda",
        )

        model = load_model_for_fold(fold_num, args.checkpoint_kind)
        slice_probs, slice_rids = infer_slices(model, loader)

        for rid in va_idx:
            mask = slice_rids == int(rid)
            if not np.any(mask):
                continue
            preds = slice_probs[mask]
            preds = preds[np.isfinite(preds).all(axis=1)]
            if len(preds) == 0:
                continue
            oof_probs[int(rid)] = power_mean(preds, q=3.0, axis=0)
            fold_id[int(rid)] = fold_num

        if args.save_slice_probs:
            slice_probs_all.append(slice_probs.astype(np.float16))
            slice_record_index_all.append(slice_rids.astype(np.int64))

        del model
        torch.cuda.empty_cache()
        gc.collect()

    valid_records = fold_id >= 0
    if not np.all(valid_records):
        missing = int(np.sum(~valid_records))
        print(f"Warning: {missing} records did not receive predictions")

    out_path = PREDICTION_DIR / "oof_full_predictions.npz"
    np.savez_compressed(
        out_path,
        y_true=y,
        y_prob=oof_probs,
        record_id=np.arange(n_records, dtype=np.int64),
        class_names=np.asarray(CLASSES),
        dataset=np.asarray("chapman_oof"),
        protocol=np.asarray("fold_best_power_mean_q3_threshold_0.5"),
        fold_id=fold_id,
        slice_count=record_slice_count,
    )
    print(f"Wrote: {out_path}")

    slice_out_path = None
    if args.save_slice_probs and slice_probs_all:
        slice_out_path = PREDICTION_DIR / "oof_full_slice_predictions.npz"
        np.savez_compressed(
            slice_out_path,
            slice_prob=np.concatenate(slice_probs_all, axis=0),
            record_id=np.concatenate(slice_record_index_all, axis=0),
            class_names=np.asarray(CLASSES),
            dataset=np.asarray("chapman_oof"),
            protocol=np.asarray("slice_level_fold_best"),
        )
        print(f"Wrote: {slice_out_path}")

    metrics = multilabel_metrics(y[valid_records], oof_probs[valid_records], threshold=0.5)
    summary = {
        "dataset": "chapman_oof",
        "prediction_file": str(out_path),
        "slice_prediction_file": str(slice_out_path) if slice_out_path else None,
        "n_records": int(n_records),
        "n_valid_predictions": int(np.sum(valid_records)),
        "n_classes": int(n_classes),
        "class_names": CLASSES,
        "checkpoint_kind_requested": args.checkpoint_kind,
        "aggregation": {"method": "power_mean", "q": 3.0},
        "threshold": 0.5,
        "pca_variance": pca_variance,
        "metrics": metrics,
    }
    summary_path = METRIC_DIR / "oof_full_prediction_summary.json"
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.dataset == "oof":
        generate_oof(args)
    else:
        raise ValueError(args.dataset)


if __name__ == "__main__":
    main()
