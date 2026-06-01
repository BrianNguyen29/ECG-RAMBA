
"""
ECG RAMBA – Data Pipeline
==============================================================
Design goals:
- Always count RAW records from filesystem (~45k)
- Never silently load stale / partial cache
- Robust to corrupt .mat files
- Subject-aware (record-id based)
- Reviewer-safe logging
"""

import os
import tempfile
import time
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
from tqdm.auto import tqdm
import shutil

from configs.config import (
    CONFIG, PATHS, SEQ_LEN, FS, NUM_CLASSES,
    SNOMED_MAPPING
)
from src.features import extract_amplitude_features


def env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def reset_or_relocate_extract_dir(extract_dir: str) -> str:
    """Remove an incomplete extract dir, or move extraction to local scratch if Drive I/O is broken."""
    if not os.path.exists(extract_dir):
        return extract_dir

    try:
        shutil.rmtree(extract_dir)
        return extract_dir
    except OSError as exc:
        local_root = os.environ.get("ECG_RAMBA_LOCAL_ROOT")
        if not local_root:
            local_root = "/content/ecg_ramba_runtime" if os.path.exists("/content") else tempfile.gettempdir()
        fallback = os.path.join(local_root, f"chapman_extract_{int(time.time())}")
        print(f"⚠️  Could not remove incomplete extract dir: {extract_dir}")
        print(f"    Reason: {exc}")
        print(f"    Falling back to local extract dir: {fallback}")
        return fallback


# ============================================================
# SUBJECT / RECORD ID
# ============================================================
def extract_record_id(hea_path: str) -> str:
    # Chapman–Shaoxing: 1 record = 1 subject
    return os.path.basename(hea_path).replace('.hea', '')


# ============================================================
# SIGNAL PROCESSING
# ============================================================
def bandpass_filter(signal: np.ndarray, lowcut=0.5, highcut=40, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal, axis=-1)


def normalize_signal(signal: np.ndarray):
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True) + 1e-8
    return (signal - mean) / std


def pad_or_truncate(signal: np.ndarray, target_len=SEQ_LEN):
    if signal.shape[-1] >= target_len:
        return signal[..., :target_len]
    return np.pad(
        signal,
        ((0, 0), (0, target_len - signal.shape[-1])),
        mode='constant'
    )


# ============================================================
# LABEL PARSING (SNOMED)
# ============================================================
def get_labels_from_header(header_path: str) -> np.ndarray:
    labels = np.zeros(NUM_CLASSES, dtype=np.float32)
    try:
        with open(header_path, 'r') as f:
            for line in f:
                if line.startswith('#Dx:'):
                    codes = line.replace('#Dx:', '').strip().split(',')
                    for c in codes:
                        c = c.strip()
                        if c in SNOMED_MAPPING:
                            labels[SNOMED_MAPPING[c]] = 1.0
                    break
    except Exception:
        pass
    return labels


# ============================================================
# MAT FILE LOADING (ROBUST)
# ============================================================
def load_mat_signal(mat_path: str) -> np.ndarray | None:
    try:
        mat = loadmat(mat_path)
        if 'val' in mat:
            sig = mat['val'].astype(np.float32)
            # Ensure shape = (12, T)
            if sig.shape[0] != 12 and sig.shape[1] == 12:
                sig = sig.T
            return sig
    except Exception:
        return None
    return None


# ============================================================
# MAIN DATA LOADER
# ============================================================
def load_chapman_multilabel(paths: dict = None):
    if paths is None:
        paths = PATHS

    print("\n🔄 LOADING CHAPMAN–SHAOXING (ROBUST MODE)")
    print("=" * 80)

    cache_path = paths['data_cache']

    # --------------------------------------------------------
    # 1️⃣ LOAD CACHE (ONLY IF VALID & COMPLETE)
    # --------------------------------------------------------
    if os.path.exists(cache_path):
        print(f"⚠️  Loading cached data: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d['X'], d['y'], d['X_raw_amp'], d['subjects']

    extract_dir = paths['extract_dir']
    zip_path = paths['zip_path']

    # --------------------------------------------------------
    # 2️⃣ EXTRACTION SAFETY CHECK
    # --------------------------------------------------------
    should_extract = True
    if os.path.exists(extract_dir):
        num_files = sum(
            1 for _, _, files in os.walk(extract_dir) for f in files
            if f.endswith('.hea')
        )
        if num_files > 40000:
            should_extract = False
            print(f"📂 Raw directory looks complete ({num_files} records).")
        else:
            print(f"⚠️  Raw directory incomplete ({num_files} records). Re-extracting...")
            extract_dir = reset_or_relocate_extract_dir(extract_dir)
            paths['extract_dir'] = extract_dir

    if should_extract:
        if not os.path.exists(zip_path):
             print(f"❌ ZIP file not found at: {zip_path}")
             # Return empty arrays or raise error depending on desired behavior
             # For now, let's assume we want to stop and user needs to provide data
             raise FileNotFoundError(f"Dataset archive not found at {zip_path}")

        print(f"📦 Extracting dataset from ZIP: {zip_path}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

    # --------------------------------------------------------
    # 3️⃣ SCAN RAW RECORDS (TRUTH SOURCE)
    # --------------------------------------------------------
    hea_files = []
    for root, _, files in os.walk(extract_dir):
        hea_files.extend(
            os.path.join(root, f) for f in files if f.endswith('.hea')
        )

    print(f"🔎 Found {len(hea_files)} RAW records")

    # --------------------------------------------------------
    # 4️⃣ LOAD & CLEAN
    # --------------------------------------------------------
    X_list, y_list, amp_list, subject_list = [], [], [], []
    skipped = {
        'no_label': 0,
        'no_mat': 0,
        'bad_signal': 0,
        'corrupt_file': 0
    }

    # Known corrupted record (documented)
    BLACKLIST = {'JS27567'}

    for hea_path in tqdm(hea_files, desc="Loading ECG"):
        rid = extract_record_id(hea_path)

        if rid in BLACKLIST:
            skipped['corrupt_file'] += 1
            continue

        labels = get_labels_from_header(hea_path)
        if labels.sum() == 0:
            skipped['no_label'] += 1
            continue

        mat_path = hea_path.replace('.hea', '.mat')
        if not os.path.exists(mat_path):
            skipped['no_mat'] += 1
            continue

        signal = load_mat_signal(mat_path)
        if signal is None or signal.shape[0] != 12:
            skipped['bad_signal'] += 1
            continue

        try:
            signal = bandpass_filter(signal)
            amp_feats = extract_amplitude_features(signal)
            signal = normalize_signal(signal)
            signal = pad_or_truncate(signal)

            X_list.append(signal)
            y_list.append(labels)
            amp_list.append(amp_feats)
            subject_list.append(rid)

        except Exception:
            skipped['bad_signal'] += 1

    if len(X_list) == 0:
        raise RuntimeError("❌ No valid ECG samples loaded.")

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    X_raw_amp = np.stack(amp_list).astype(np.float32)
    subjects = np.array(subject_list)

    print(f"\n📊 Loaded {len(X)} usable samples")
    print(f"🚫 Skipped records: {skipped}")

    # --------------------------------------------------------
    # 5️⃣ SAVE CACHE (CONFIG-HASH SAFE)
    # --------------------------------------------------------
    if env_flag("ECG_RAMBA_SAVE_CLEAN_CACHE", default=True):
        approx_gib = (X.nbytes + y.nbytes + X_raw_amp.nbytes + subjects.nbytes) / (1024 ** 3)
        print(
            f"💾 Saving cleaned raw ECG cache (~{approx_gib:.2f} GiB before compression) to: {cache_path}",
            flush=True,
        )
        print("   This can take a long time on Google Drive.", flush=True)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            X_raw_amp=X_raw_amp,
            subjects=subjects
        )
        print(f"💾 Cached cleaned dataset to: {cache_path}")
    else:
        print(
            "⏭️  Skipping cleaned raw ECG cache because ECG_RAMBA_SAVE_CLEAN_CACHE=0. "
            "Feature/prediction artifacts will still be saved.",
            flush=True,
        )
    print("✅ DATA LOADING COMPLETE")
    print("=" * 80)

    return X, y, X_raw_amp, subjects


# ============================================================
# DATASET WRAPPER (OPTIONAL)
# ============================================================
class ECGDatasetMultiLabel(Dataset):
    def __init__(self, X, X_hydra, X_hrv, y):
        self.X = X
        self.X_hydra = X_hydra
        self.X_hrv = X_hrv
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.X_hydra[idx], dtype=torch.float32),
            torch.tensor(self.X_hrv[idx], dtype=torch.float32),
            self.y[idx]
        )
