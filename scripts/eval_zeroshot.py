
"""
ECG RAMBA - Zero-Shot Evaluation (PTB-XL & CPSC-2021)
==================================================================================
Purpose:
- Evaluate model generalization on datasets NOT seen during training (PTB-XL, CPSC).
- Protocol:
  - Model source: Chapman ONLY
  - PCA source: Chapman (global_pca_zeroshot.pkl) - fitted on Chapman, applied to external data.
  - Features: MiniRocket + HRV (enabled)
  - Threshold tuning: NONE (fixed)
"""

import os
import sys
import glob
import zipfile
import joblib
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from torch.amp import autocast
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.signal import find_peaks
import scipy.stats as scipy_stats
from tqdm.auto import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import CONFIG, PATHS, DEVICE, CLASSES, SNOMED_MAPPING
from src.features import MiniRocketNative, extract_amplitude_features, extract_global_record_stats
from src.model import ECGRambaV7Advanced
from src.utils import normalize_signal

# Target Superclasses for PTB-XL
TARGET_CLASSES = ["NORM", "MI", "STTC", "CD"]

# ==================================================================================
# FEATURE EXTRACTION HELPERS (Zero-Shot Specific)
# ==================================================================================
# Note: Re-implementing specific helpers if strictly necessary to match notebook's exact behavior,
# but ideally we reuse src/features.py.
# The notebook implementation of extract_hrv36 matches src.features.generate_hrv_cache logic.
# We will use src.features specific functions to compose it.

def extract_hrv36(signal, fs=500):
    # This matches the composition in generate_hrv_cache, but operates on single signal
    # and returns the 36-dim vector directly.
    # We can reuse the granular functions from src.features.
    
    # HRV (25) - Note: src.features.extract_hrv_features is 25 dim.
    # The notebook implementation (Lines 5346-5364) seems to populate:
    # feats[:5] = [rr.mean(), rr.std(), np.median(rr), rr.min(), rr.max()]
    # src.features.extract_hrv_features populates feats[:5] similarly.
    # AND leaves the rest as zeros/nans if not found.
    # Wait, src.features.extract_hrv_features returns 25 dims?
    # Let's check src/features.py content I wrote.
    # Def: feats = np.zeros(25, dtype=np.float32) ... feats[:5] = ... return np.nan_to_num(feats)
    # So src.features.extract_hrv_features returns 25 dims, but mostly zeros except first 5?
    # Notebook: feats[:5] = ... (rest zero).
    # Then Notebook: amps (5 dims), global stats (6 dims). Total 36?
    # 5 (HRV) + 5 (Amp) + 6 (Global) = 16?
    # Wait. Notebook line 5348: feats = np.zeros(36).
    # feats[25:30] = amps
    # feats[30:36] = global
    # So indices 5-25 are unused/zeros?
    # src.features.extract_hrv_features returns 25 features.
    # So if I concatenate [hrv(25), amp(5), global(6)] -> 36.
    # This aligns 25+5+6 = 36.
    
    # So I can just import and use them.
    from src.features import extract_hrv_features, extract_amplitude_features, extract_global_record_stats
    
    hrv = extract_hrv_features(signal, fs)
    amp = extract_amplitude_features(signal) # Note: takes raw signal? function def says signal_raw.
    # Notebook line 4926: extract_hrv36(sig). sig is normalized?
    # Notebook line 4891: sig = normalize_signal(sig).
    # src.features.extract_amplitude_features: "amps = np.ptp(signal_raw, axis=-1)".
    # If we pass normalized signal, amplitude features might be affected.
    # Notebook line 5358: amps = np.ptp(signal, axis=-1). Passed 'signal'.
    # In 'main loop', 'sig' is passed to extract_hrv36. And 'sig' was normalized.
    # So we should pass the normalized signal to amplitude extraction if we want to match notebook exactly.
    
    gstat = extract_global_record_stats(signal)
    
    return np.concatenate([hrv, amp, gstat])

def parse_af_label(rec_path):
    try:
        # Check .atr for AF/AFL annotations
        ann = wfdb.rdann(rec_path, "atr")
        for note in ann.aux_note:
            if "(AFIB" in note or "(AFL" in note:
                return 1
    except:
        pass
    return 0


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================
def main():
    print("\n" + "=" * 100)
    print("🌍 PTB-XL ZERO-SHOT EVALUATION (FULL FEATURES)")
    print("=" * 100)
    
    # 0. CONFIG PATHS
    # These might need adjustment depending on where data is located in the user's workspace
    # or if we are just setting up the script for future execution.
    # Assuming standard project structure or config paths.
    
    # For now, we use placeholders or paths from config if available.
    PTB_ZIP = os.path.join(PATHS["data_dir"], "ptbxl.zip") # Example
    PTB_DIR = os.path.join(PATHS["data_dir"], "ptbxl")
    GLOBAL_PCA_PATH = os.path.join(PATHS["model_dir"], "global_pca_zeroshot.pkl")
    PTB_HYDRA_CACHE = os.path.join(PATHS["data_cache"], "ptbxl_hydra.npz")
    PTB_HRV_CACHE = os.path.join(PATHS["data_cache"], "ptbxl_hrv36.npz")
    
    # 1. LOAD PCA
    if os.path.exists(GLOBAL_PCA_PATH):
        global_pca = joblib.load(GLOBAL_PCA_PATH)
        print(f"  ✅ Global PCA: {GLOBAL_PCA_PATH}")
        print(f"     Components: {global_pca.n_components_}, Var: {global_pca.explained_variance_ratio_.sum():.3f}")
    else:
        # If not found, we can't run zero-shot without the Chapman-fitted PCA
        print(f"❌ Global PCA not found at {GLOBAL_PCA_PATH}. Please run training/PCA fitting first.")
        # We allow script to continue if just testing logic, but ideally return
        return

    # 2. EXTRACT PTB-XL (Placeholder logic - expects data to be present or zipped)
    if os.path.exists(PTB_ZIP) and not os.path.exists(PTB_DIR):
        with zipfile.ZipFile(PTB_ZIP, "r") as zf:
            zf.extractall(PTB_DIR)
        print(f"  ✅ Extracted PTB-XL to {PTB_DIR}")
    
    # Check for metadata
    csv_db_path = glob.glob(os.path.join(PTB_DIR, "**", "ptbxl_database.csv"), recursive=True)
    csv_scp_path = glob.glob(os.path.join(PTB_DIR, "**", "scp_statements.csv"), recursive=True)
    
    if not csv_db_path or not csv_scp_path:
        print("⚠️ PTB-XL metadata not found. Skipping PTB-XL evaluation.")
    else:
        CSV_DB, CSV_SCP = csv_db_path[0], csv_scp_path[0]
        DATA_ROOT = os.path.dirname(CSV_DB)
        
        df = pd.read_csv(CSV_DB, index_col="ecg_id")
        df_scp = pd.read_csv(CSV_SCP, index_col=0)
        
        # Test on Fold 10 (Standard PTB-XL Split)
        test_df = df[df.strat_fold == 10].copy()
        test_df.scp_codes = test_df.scp_codes.apply(ast.literal_eval)
        df_scp["diagnostic_class"] = df_scp.diagnostic_class.str.upper()
        
        print(f"  ✅ Test set (Fold 10): {len(test_df)} records")
        
        # 3. BUILD GROUND TRUTH
        y_true_ptb = np.zeros((len(test_df), len(TARGET_CLASSES)))
        for i, (_, row) in enumerate(test_df.iterrows()):
            record_classes = set()
            for code, likelihood in row.scp_codes.items():
                if likelihood >= 100.0 and code in df_scp.index:
                    dclass = df_scp.loc[code].diagnostic_class
                    if isinstance(dclass, str):
                        record_classes.add(dclass)
            for k, target in enumerate(TARGET_CLASSES):
                if target in record_classes:
                    y_true_ptb[i, k] = 1
        
        # 4. LOAD SIGNALS
        X_signals = []
        valid_mask = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Loading PTB"):
            try:
                rec_path = os.path.join(DATA_ROOT, row.filename_hr)
                record = wfdb.rdrecord(rec_path)
                sig = record.p_signal.T
                
                if sig.shape[0] != 12:
                     if sig.shape[0] > 12: sig = sig[:12, :]
                     else: raise ValueError
                
                # Center crop to 5000
                T = sig.shape[1]
                if T > 5000:
                    start = (T - 5000) // 2
                    sig = sig[:, start:start+5000]
                elif T < 5000:
                    sig = np.pad(sig, ((0, 0), (0, 5000 - T)))
                
                sig = normalize_signal(sig).astype(np.float32)
                X_signals.append(sig)
                valid_mask.append(True)
            except:
                valid_mask.append(False)
        
        X_signals = np.stack(X_signals)
        valid_mask = np.array(valid_mask)
        print(f"  ✅ Loaded: {len(X_signals)} / {len(test_df)} records")
        
        # 5. FEATURES
        # Hydra
        if os.path.exists(PTB_HYDRA_CACHE):
            X_hydra = np.load(PTB_HYDRA_CACHE)["X"].astype(np.float32)
        else:
            print("  🚀 Computing MiniRocket features (PTB)...")
            rocket = MiniRocketNative(c_in=12, seq_len=5000, seed=42).cpu().eval()
            feats = []
            with torch.no_grad():
                for i in tqdm(range(0, len(X_signals), 64), desc="MiniRocket"):
                    xb = torch.tensor(X_signals[i:i+64], dtype=torch.float32)
                    feats.append(rocket(xb).numpy())
            X_rocket = np.vstack(feats)
            X_hydra = global_pca.transform(X_rocket).astype(np.float32)
            np.savez_compressed(PTB_HYDRA_CACHE, X=X_hydra.astype(np.float16))
        
        # HRV
        if os.path.exists(PTB_HRV_CACHE):
            X_hrv = np.load(PTB_HRV_CACHE)["X"].astype(np.float32)
        else:
            print("  💓 Computing HRV features (PTB)...")
            X_hrv = np.zeros((len(X_signals), 36), dtype=np.float32)
            for i, sig in enumerate(tqdm(X_signals, desc="HRV36")):
                X_hrv[i] = extract_hrv36(sig)
            np.savez_compressed(PTB_HRV_CACHE, X=X_hrv.astype(np.float16))

        # 6. INFERENCE (Ensemble)
        print("\n🧠 MODEL INFERENCE")
        ckpts = sorted(glob.glob(os.path.join(PATHS["model_dir"], "fold*_best.pt")))
        models = []
        for p in ckpts:
            m = ECGRambaV7Advanced(cfg=CONFIG).to(DEVICE)
            sd = torch.load(p, map_location=DEVICE)
            m.load_state_dict(sd["model"] if isinstance(sd, dict) else sd, strict=False)
            m.eval()
            models.append(m)
        print(f"  ✅ Ensemble: {len(models)} folds")
        
        BATCH = 32
        probs_ptb = []
        with torch.no_grad():
             for i in tqdm(range(0, len(X_signals), BATCH), desc="Inference"):
                xb = torch.tensor(X_signals[i:i+BATCH], dtype=torch.float32, device=DEVICE)
                zh = torch.tensor(X_hydra[i:i+BATCH], dtype=torch.float32, device=DEVICE)
                zhrv = torch.tensor(X_hrv[i:i+BATCH], dtype=torch.float32, device=DEVICE)
                
                fold_preds = []
                for m in models:
                    if DEVICE == 'cuda':
                        with autocast("cuda"):
                            fold_preds.append(torch.sigmoid(m(xb, zh, zhrv)).cpu().numpy())
                    else:
                        fold_preds.append(torch.sigmoid(m(xb, zh, zhrv)).cpu().numpy())
                probs_ptb.append(np.mean(fold_preds, axis=0))
        
        probs_ptb = np.concatenate(probs_ptb, axis=0)
        
        # 7. RESULTS
        y_true_valid = y_true_ptb[valid_mask]
        
        print("\n" + "=" * 100)
        print("📊 PTB-XL ZERO-SHOT RESULTS")
        print("=" * 100)
        print(f"{'CLASS':^8} │ {'PR-AUC':^8} │ {'ROC-AUC':^8} │ {'N+':^6}")
        print("-" * 100)
        
        results = []
        for k, target in enumerate(TARGET_CLASSES):
            # Map PTB superclass to Chapman Classes -> Max prob
            src_codes = SNOMED_MAPPING.get(target, {}).get("codes", [])
            # Find indices in Chapman model
            src_idxs = [CLASSES.index(c) for c in src_codes if c in CLASSES]
            
            if not src_idxs:
                y_pred = np.zeros(len(probs_ptb))
            else:
                y_pred = np.max(probs_ptb[:, src_idxs], axis=1)
                
            y_true = y_true_valid[:, k]
            n_pos = int(y_true.sum())
            
            if n_pos > 0:
                prauc = average_precision_score(y_true, y_pred)
                try: rocauc = roc_auc_score(y_true, y_pred)
                except: rocauc = 0.5
            else:
                prauc, rocauc = 0.0, 0.5
            
            results.append({"class": target, "prauc": prauc, "n": n_pos})
            print(f"{target:^8} │ {prauc:^8.4f} │ {rocauc:^8.4f} │ {n_pos:^6}")
            
        print("-" * 100)

    # ==================================================================================
    # CPSC-2021 EVALUATION (Optional Section)
    # ==================================================================================
    # ... (Logic similar to PTB-XL but for CPSC AF detection)
    # ... Can be added here following similar patterns.

if __name__ == "__main__":
    main()
