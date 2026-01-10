
"""
ECG RAMBA - Training Pipeline
==================================================================================
Principles:
- Subject-aware CV (no leakage)
- Fold-wise PCA only
- BCE warmup → ONE-TIME switch to FIXED Asymmetric Loss
- NO early stopping (full-epoch training)
- EMA for evaluation only (AFTER warmup)
- Fixed threshold evaluation
- Quiet logging (NaN aggregated per fold only)
"""

import os
import sys
import gc
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict

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
from src.utils import (
    compute_metrics,
    AsymmetricLossMultiLabel,
    EMA,
    set_seed,
)

# Suppress specific warning
warnings.filterwarnings("ignore", message="The least populated class in y")

# ==================================================================================
# 🔪 DATASET HELPERS
# ==================================================================================

def slice_record(x):
    if x.shape[-1] < CONFIG["slice_length"]:
        return [], []
    slices, positions = [], []
    T = x.shape[-1]
    for s in range(
        0,
        T - CONFIG["slice_length"] + 1,
        CONFIG["slice_stride"],
    ):
        slices.append(x[..., s : s + CONFIG["slice_length"]])
        positions.append((s + CONFIG["slice_length"] / 2) / T)
        if len(slices) >= CONFIG["max_slices_per_record"]:
            break
    return slices, positions


class ECGSliceDataset(Dataset):
    def __init__(self, Xs, Xh, Xhr, y, rids, pos):
        self.Xs, self.Xh, self.Xhr, self.y, self.rids, self.pos = Xs, Xh, Xhr, y, rids, pos

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.Xs[i], dtype=torch.float32),
            torch.tensor(self.Xh[i], dtype=torch.float32),
            torch.tensor(self.Xhr[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
            self.rids[i],
            self.pos[i],
        )


def build_slice_dataset(indices, hydra_dict, X, y, X_hrv_base):
    xs, xh, xhr, ys, rids, pos = [], [], [], [], [], []
    skipped = 0
    for rid in indices:
        slices, positions = slice_record(X[rid])
        if len(slices) == 0:
            skipped += 1
            continue

        hrv_ext = X_hrv_base[rid] if X_hrv_base is not None else np.zeros(1)

        for s, p in zip(slices, positions):
            xs.append(s)
            xh.append(hydra_dict[rid])
            xhr.append(hrv_ext)
            ys.append(y[rid])
            rids.append(rid)
            pos.append(p)

    return map(np.array, (xs, xh, xhr, ys, rids, pos)), skipped


# ==================================================================================
# MAIN TRAINING FUNCTION
# ==================================================================================

def main():
    set_seed(CONFIG["seeds"][0])

    # ==================================================================================
    # 🔧 RUN HEADER
    # ==================================================================================
    print("🔧 ECG RAMBA ")
    print(f"   D={CONFIG['d_model']} | Gamma={CONFIG['asym_gamma_neg']} | LR={CONFIG['lr_max']}")
    print(f"   Warmup={CONFIG['asym_start_epoch']} | Epochs={CONFIG['epochs']} | Folds={CONFIG['n_folds']}")

    # ==================================================================================
    # 🛡️ PHASE 1 | LOAD DATA & AUTO-CLEAN
    # ==================================================================================
    print("\n" + "=" * 80)
    print("PHASE 1 | LOAD DATA & AUTO-CLEAN")
    print("=" * 80)

    X, y, X_raw_amp, subjects = load_chapman_multilabel()
    print(f"Original: {len(y)} records | {y.shape[1]} classes")

    MIN_SAMPLES = 5
    class_counts = y.sum(axis=0)
    keep_mask = class_counts >= MIN_SAMPLES

    if not keep_mask.all():
        print(f"🧹 Dropping {np.sum(~keep_mask)} classes (<{MIN_SAMPLES} samples)")
        y = y[:, keep_mask]
        valid = y.sum(axis=1) > 0
        X, y, X_raw_amp, subjects = X[valid], y[valid], X_raw_amp[valid], subjects[valid]

    # NUM_CLASSES is effectively updated by valid y shape, but global config likely remains fixed
    # We should ensure model output matches this if it changed, but usually we stick to fixed classes
    print(f"✅ Cleaned → {len(y)} records | {y.shape[1]} classes")

    # ==================================================================================
    # 🧬 PHASE 2 | RAW FEATURE GENERATION (ANTI-LEAKAGE)
    # ==================================================================================
    print("\n" + "=" * 80)
    print("PHASE 2 | RAW FEATURE GENERATION (ANTI-LEAKAGE)")
    print("=" * 80)

    X_rocket_raw = generate_raw_rocket_cache(X)
    X_hrv_base = generate_hrv_cache(X, X_raw_amp) if CONFIG["use_hrv"] else None

    print(f"✅ RAW MiniRocket shape: {X_rocket_raw.shape}")
    if X_hrv_base is not None:
        print(f"✅ HRV feature shape  : {X_hrv_base.shape}")

    # ==================================================================================
    # 🚀 PHASE 3 | STRATIFIED GROUP K-FOLD TRAINING
    # ==================================================================================
    print("\n" + "=" * 80)
    print("PHASE 3 | TRAINING WITH FOLD-AWARE PCA")
    print("=" * 80)

    y_strat = y.sum(axis=1).clip(max=3).astype(int)
    sgkf = StratifiedGroupKFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seeds"][0],
    )

    PM_Q = 3.0
    PM_EPS = 1e-6
    TEMPORAL_BINS = np.array([0.0, 0.33, 0.66, 1.01])

    fold_results, epoch_logs = [], []

    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(X, y_strat, groups=subjects), start=1
    ):
        print(f"\n⚡ FOLD {fold}/{CONFIG['n_folds']}")

        pca = fit_pca_on_train(X_rocket_raw[tr_idx], CONFIG["hydra_dim"])
        hydra_tr = apply_pca(pca, X_rocket_raw[tr_idx])
        hydra_va = apply_pca(pca, X_rocket_raw[va_idx])
        print(f"   🛡️ PCA variance retained: {pca.explained_variance_ratio_.sum():.3f}")

        hydra_dict = {i: f for i, f in zip(tr_idx, hydra_tr)}
        hydra_dict.update({i: f for i, f in zip(va_idx, hydra_va)})

        (Xs_tr, Xh_tr, Xhr_tr, y_tr, rid_tr, pos_tr), skipped_tr = build_slice_dataset(
            tr_idx, hydra_dict, X, y, X_hrv_base
        )
        (Xs_va, Xh_va, Xhr_va, y_va, rid_va, pos_va), skipped_va = build_slice_dataset(
            va_idx, hydra_dict, X, y, X_hrv_base
        )

        n_val_records_expected = len(np.unique(va_idx))
        n_val_records_with_slice = len(np.unique(rid_va))
        total_val_slices = len(Xs_va)
        avg_slices_per_record = total_val_slices / max(n_val_records_with_slice, 1)

        print(
          f"   🧪 Fold {fold} | EARLY CHECK | "
          f"val_records_with_slice={n_val_records_with_slice}/{n_val_records_expected} | "
          f"total_val_slices={total_val_slices} | "
          f"avg_slices/record={avg_slices_per_record:.2f} | "
          f"skipped_no_slice={skipped_va}"
        )

        train_loader = DataLoader(
            ECGSliceDataset(Xs_tr, Xh_tr, Xhr_tr, y_tr, rid_tr, pos_tr),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
        )

        val_loader = DataLoader(
            ECGSliceDataset(Xs_va, Xh_va, Xhr_va, y_va, rid_va, pos_va),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
        )

        model = ECGRambaV7Advanced(cfg=CONFIG).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr_max"],
            weight_decay=CONFIG["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["epochs"],
            eta_min=CONFIG["lr_min"],
        )

        bce_criterion = torch.nn.BCEWithLogitsLoss()
        asym_criterion = AsymmetricLossMultiLabel(
            gamma_neg=CONFIG["asym_gamma_neg"],
            gamma_pos=CONFIG["asym_gamma_pos"],
            clip=CONFIG["asym_clip"],
        )

        ema = EMA(model, decay=CONFIG["ema_decay"])

        best_f1, best_epoch = 0.0, -1
        best_metrics = None
        fold_skipped_nan = 0

        # Ensure model directory exists
        os.makedirs(PATHS["model_dir"], exist_ok=True)
        best_ckpt_path = os.path.join(PATHS["model_dir"], f"fold{fold}_best.pt")
        final_ckpt_path = os.path.join(PATHS["model_dir"], f"fold{fold}_final.pt")

        for epoch in range(CONFIG["epochs"]):
            model.train()
            loss_sum = 0.0

            use_bce = epoch < CONFIG["asym_start_epoch"]
            loss_name = "BCE" if use_bce else "ASYM"

            for x, xh, xhr, tgt, _, _ in train_loader:
                x, xh, xhr, tgt = x.to(DEVICE), xh.to(DEVICE), xhr.to(DEVICE), tgt.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed Precision usually requires CUDA
                if DEVICE == 'cuda':
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x, xh, xhr)
                        loss = bce_criterion(logits, tgt) if use_bce else asym_criterion(logits, tgt)
                else:
                    logits = model(x, xh, xhr)
                    loss = bce_criterion(logits, tgt) if use_bce else asym_criterion(logits, tgt)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                optimizer.step()
                ema.update(model)
                loss_sum += loss.item()

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            model.eval()
            if epoch >= CONFIG["asym_start_epoch"]:
                ema.apply_shadow(model)

            record_bins = defaultdict(lambda: defaultdict(list))
            with torch.no_grad():
                for x, xh, xhr, _, rids, pos in val_loader:
                    x, xh, xhr = x.to(DEVICE), xh.to(DEVICE), xhr.to(DEVICE)
                    
                    if DEVICE == 'cuda':
                        with torch.amp.autocast("cuda"):
                            probs = torch.sigmoid(model(x, xh, xhr)).cpu().numpy()
                    else:
                        probs = torch.sigmoid(model(x, xh, xhr)).cpu().numpy()
                        
                    for p, rid, t in zip(probs, rids, pos):
                        b = int(np.digitize(t, TEMPORAL_BINS) - 1)
                        record_bins[int(rid)][b].append(p)

            if epoch >= CONFIG["asym_start_epoch"]:
                ema.restore(model)

            y_true_list, y_pred_list = [], []

            for rid, bins in record_bins.items():
                preds = []
                for b in bins.values():
                    p = np.clip(np.stack(b), PM_EPS, 1 - PM_EPS)
                    preds.append(np.exp(np.mean(PM_Q * np.log(p), axis=0) / PM_Q))
                pred = np.mean(preds, axis=0)
                if not np.isfinite(pred).all():
                    fold_skipped_nan += 1
                    continue
                y_true_list.append(y[rid])
                y_pred_list.append(pred)

            if len(y_true_list) == 0:
                continue

            metrics = compute_metrics(
                np.vstack(y_true_list),
                np.vstack(y_pred_list),
                threshold=CONFIG["default_threshold"],
            )

            f1m = metrics["f1_macro"]
            if f1m > best_f1:
                best_f1 = f1m
                best_epoch = epoch + 1
                best_metrics = metrics.copy()
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": best_epoch,
                        "f1_macro": best_f1,
                    },
                    best_ckpt_path,
                )

            print(
                f"Ep {epoch+1:03d} | {loss_name} | LR {lr:.2e} | "
                f"Loss {loss_sum/len(train_loader):.4f} | "
                f"F1m {metrics['f1_macro']:.4f} | "
                f"P {metrics['precision_macro']:.4f} | "
                f"R {metrics['recall_macro']:.4f} | "
                f"AP {metrics['auprc_macro']:.4f} | "
                f"Best {best_f1:.4f}"
            )

            epoch_logs.append(
                dict(
                    fold=fold,
                    epoch=epoch + 1,
                    loss=loss_name,
                    lr=lr,
                    loss_value=loss_sum / len(train_loader),
                    is_best_epoch=(epoch + 1 == best_epoch),
                    **metrics,
                )
            )

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": CONFIG["epochs"],
            },
            final_ckpt_path,
        )

        fold_results.append(dict(fold=fold, best_epoch=best_epoch, **best_metrics))

        del model, optimizer, scheduler, ema
        torch.cuda.empty_cache()
        gc.collect()

    # ==================================================================================
    # 🏁 FINAL REPORT
    # ==================================================================================
    print("\n" + "=" * 80)
    print("FINAL CROSS-VALIDATION RESULTS (OOF)")
    print("=" * 80)

    df_folds = pd.DataFrame(fold_results).set_index("fold")
    df_epochs = pd.DataFrame(epoch_logs)

    print(df_folds.round(4))

    for m in ["f1_macro", "f1_micro", "precision_macro", "recall_macro", "auprc_macro"]:
        v = df_folds[m].values
        mean, std = v.mean(), v.std(ddof=1)
        ci = stats.t.interval(0.95, len(v) - 1, loc=mean, scale=std / np.sqrt(len(v)))
        print(f"{m:18s} {mean:.4f} ± {std:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    df_epochs.to_csv(f"{PATHS['model_dir']}/training_log_epochs.csv", index=False)
    df_folds.to_csv(f"{PATHS['model_dir']}/cv_results_clean_core.csv")

    print("\n✅ PIPELINE FINISHED.")


if __name__ == "__main__":
    main()
