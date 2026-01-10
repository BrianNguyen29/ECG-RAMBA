
"""
ECG RAMBA - OOF Evaluation (Clean Core)
==================================================================================
Purpose:
- Compute honest Out-of-Fold (OOF) predictions
- EXACTLY replay training logic (folds, PCA, slicing)
- NO ensemble tricks, NO global PCA, NO calibration
- Fixed threshold evaluation (0.5)
"""

import os
import sys
import gc
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import CONFIG, PATHS, DEVICE
from src.data_loader import load_chapman_multilabel
from src.features import (
    generate_raw_rocket_cache,
    generate_hrv_cache,
    fit_pca_on_train,
    apply_pca,
)
from src.model import ECGRambaV7Advanced
from src.utils import compute_metrics, set_seed

# Suppress warnings
warnings.filterwarnings("ignore", message="The least populated class in y")

# ==================================================================================
# DATASET HELPERS
# ==================================================================================

def slice_record(x):
    slices = []
    # Note: Using slicing logic from Cell 9 which might slightly differ or be consistent with Cell 8
    # Cell 9 loop:
    # range(0, x.shape[-1] - CONFIG["slice_length"] + 1, CONFIG["slice_stride"])
    # This matches train.py logic.
    for s in range(
        0,
        x.shape[-1] - CONFIG["slice_length"] + 1,
        CONFIG["slice_stride"],
    ):
        slices.append(x[..., s:s + CONFIG["slice_length"]])
        if len(slices) >= CONFIG["max_slices_per_record"]:
            break
    return slices

class ECGSliceDatasetInfer(Dataset):
    def __init__(self, Xs, Xh, Xhr, rids):
        self.Xs, self.Xh, self.Xhr, self.rids = Xs, Xh, Xhr, rids
    def __len__(self):
        return len(self.rids)
    def __getitem__(self, i):
        return (
            torch.tensor(self.Xs[i], dtype=torch.float32),
            torch.tensor(self.Xh[i], dtype=torch.float32),
            torch.tensor(self.Xhr[i], dtype=torch.float32),
            self.rids[i],
        )

# ==================================================================================
# MAIN EVALUATION FUNCTION
# ==================================================================================

def main():
    set_seed(CONFIG["seeds"][0])

    print("\n🧠 ECG RAMBA | OOF EVALUATION")
    print("=" * 80)

    # ==================================================================================
    # 1️⃣ LOAD DATA & RE-APPLY CLEANING
    # ==================================================================================
    print("🔄 [STEP 1] Loading data & re-applying clean rules...")

    X, y, X_raw_amp, subjects = load_chapman_multilabel()
    print(f"Original: {len(y)} records | {y.shape[1]} classes")

    MIN_SAMPLES = 5
    class_counts = y.sum(axis=0)
    keep_mask = class_counts >= MIN_SAMPLES

    if not keep_mask.all():
        print(f"🧹 Dropping {np.sum(~keep_mask)} classes (<{MIN_SAMPLES} samples)")
        y = y[:, keep_mask]
        valid = y.sum(axis=1) > 0
        X, y, X_raw_amp, subjects = (
            X[valid], y[valid], X_raw_amp[valid], subjects[valid]
        )

    # NUM_CLASSES local var, though global config classes are fixed.
    # We proceed with loaded y.
    NUM_CLASSES = y.shape[1]
    print(f"✅ Cleaned → {len(y)} records | {NUM_CLASSES} classes")

    N = len(y)

    # ==================================================================================
    # 2️⃣ RAW FEATURE GENERATION (CACHE-SAFE, NO PCA)
    # ==================================================================================
    print("\n🔄 [STEP 2] Preparing RAW features...")

    X_rocket_raw = generate_raw_rocket_cache(X)
    X_hrv = generate_hrv_cache(X, X_raw_amp) if CONFIG["use_hrv"] else None

    print(f"✅ RAW MiniRocket: {X_rocket_raw.shape}")
    if X_hrv is not None:
        print(f"✅ HRV features  : {X_hrv.shape}")

    # ==================================================================================
    # 3️⃣ OOF INFERENCE (SUBJECT-AWARE, FOLD-WISE PCA)
    # ==================================================================================
    print("\n🔄 [STEP 3] Running OOF inference (fold-wise replay)...")

    oof_probs = np.zeros((N, NUM_CLASSES), dtype=np.float32)

    y_strat = y.sum(axis=1).clip(max=3).astype(int)

    sgkf = StratifiedGroupKFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seeds"][0],
    )

    PM_Q = CONFIG.get("power_mean_q", 3.0)
    PM_EPS = 1e-6

    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(X, y_strat, groups=subjects), start=1
    ):
        print(f"\n▶ Fold {fold}/{CONFIG['n_folds']} | Val records: {len(va_idx)}")

        # ---------- PCA (TRAIN ONLY) ----------
        pca = fit_pca_on_train(X_rocket_raw[tr_idx], CONFIG["hydra_dim"])
        hydra_tr = apply_pca(pca, X_rocket_raw[tr_idx])
        hydra_va = apply_pca(pca, X_rocket_raw[va_idx])

        print(f"   🛡️ PCA variance retained: {pca.explained_variance_ratio_.sum():.3f}")

        hydra_dict = {i: f for i, f in zip(tr_idx, hydra_tr)}
        hydra_dict.update({i: f for i, f in zip(va_idx, hydra_va)})

        # ---------- BUILD SLICED INFERENCE SET ----------
        xs, xh, xhr, rids = [], [], [], []
        for rid in va_idx:
            for s in slice_record(X[rid]):
                xs.append(s)
                xh.append(hydra_dict[rid])
                xhr.append(X_hrv[rid] if X_hrv is not None else np.zeros(1))
                rids.append(rid)

        if len(xs) == 0:
            print("   ⚠️ No slices found for validation set in this fold.")
            continue

        infer_ds = ECGSliceDatasetInfer(
            np.asarray(xs), np.asarray(xh), np.asarray(xhr), np.asarray(rids)
        )
        infer_loader = DataLoader(
            infer_ds,
            batch_size=CONFIG["batch_size"] * 2,
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
        )

        # ---------- LOAD FINAL MODEL (FIXED) ----------
        ckpt_path = os.path.join(PATHS["model_dir"], f"fold{fold}_final.pt")
        if not os.path.exists(ckpt_path):
             print(f"❌ Missing checkpoint: {ckpt_path}. Skipping fold.")
             continue

        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        model = ECGRambaV7Advanced(cfg=CONFIG).to(DEVICE)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        # ---------- SLICE-LEVEL INFERENCE ----------
        all_probs, all_rids = [], []

        with torch.no_grad():
            for x, xh_, xhr_, rid_ in infer_loader:
                x, xh_, xhr_ = x.to(DEVICE), xh_.to(DEVICE), xhr_.to(DEVICE)

                if DEVICE == 'cuda':
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x, xh_, xhr_)
                        probs = torch.sigmoid(logits).float().cpu().numpy()
                else:
                    logits = model(x, xh_, xhr_)
                    probs = torch.sigmoid(logits).float().cpu().numpy()

                all_probs.append(probs)
                all_rids.append(rid_.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_rids  = np.concatenate(all_rids, axis=0)

        # ---------- AGGREGATE SLICES → RECORD ----------
        fold_skipped_nan = 0

        for rid in va_idx:
            mask = (all_rids == rid)
            if not mask.any():
                continue

            preds = all_probs[mask]
            preds = preds[np.isfinite(preds).all(axis=1)]

            if len(preds) == 0:
                fold_skipped_nan += 1
                continue

            preds = np.clip(preds, PM_EPS, 1.0 - PM_EPS)
            oof_probs[rid] = np.exp(
                np.mean(PM_Q * np.log(preds), axis=0) / PM_Q
            )

        if fold_skipped_nan > 0:
            print(
                f"⚠️ Fold {fold} | Skipped {fold_skipped_nan}/{len(va_idx)} "
                f"records ({fold_skipped_nan/len(va_idx):.3%}) due to NaN"
            )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ==================================================================================
    # 5️⃣ FINAL METRICS (OOF, FIXED THRESHOLD)
    # ==================================================================================
    print("\n✅ OOF inference complete.")

    metrics = compute_metrics(y, oof_probs, threshold=CONFIG["default_threshold"])

    print("\n🏆 FINAL OOF RESULTS")
    print("=" * 60)
    print(f"Macro F1       : {metrics['f1_macro']:.4f}")
    print(f"Micro F1       : {metrics['f1_micro']:.4f}")
    print(f"Recall (Macro) : {metrics['recall_macro']:.4f}")
    print(f"Precision      : {metrics['precision_macro']:.4f}")
    print(f"AUPRC          : {metrics['auprc_macro']:.4f}")
    print("=" * 60)

    # ==================================================================================
    # 6️⃣ SAVE RESULTS (REPRODUCIBLE)
    # ==================================================================================
    # Ensure model directory exists
    os.makedirs(PATHS["model_dir"], exist_ok=True)
    save_path = os.path.join(PATHS["model_dir"], "oof_results_clean_core.npz")
    np.savez(
        save_path,
        probs=oof_probs,
        targets=y,
        metrics=metrics,
    )

    print(f"💾 OOF results saved to: {save_path}")
    print("\n✅ OOF EVALUATION FINISHED.")


if __name__ == "__main__":
    main()
