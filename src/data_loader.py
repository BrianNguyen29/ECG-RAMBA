
"""
ECG RAMBA – Data Pipeline
==============================================================
Design goals:
- Always count RAW records from filesystem (~45k)
- Never silently load stale / partial cache
- Robust to corrupt .mat files
- Source-patient-aware under the reviewed PhysioNet one-patient-per-record contract
- Reviewer-safe logging
"""

import os
import tempfile
import time
import zipfile
from pathlib import Path
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
from src.provenance import (
    canonical_json_sha256,
    file_sha256,
    record_order_fingerprint,
    save_npz_atomic,
    source_bundle_sha256,
)


CLEAN_CACHE_SCHEMA_VERSION = 3


def clean_cache_source_contract(zip_path: str) -> dict:
    """Bind a clean cache to the archive, preprocessing code, and parameters."""

    explicit_archive_sha = os.environ.get("ECG_RAMBA_CHAPMAN_ARCHIVE_SHA256", "").strip()
    archive_path = Path(zip_path)
    archive_sha = explicit_archive_sha or (
        file_sha256(archive_path) if archive_path.is_file() else ""
    )
    if len(archive_sha) != 64:
        raise FileNotFoundError(
            "A full Chapman archive SHA256 is required to authenticate the clean cache. "
            f"Archive not available at {archive_path}; provide the archive or set "
            "ECG_RAMBA_CHAPMAN_ARCHIVE_SHA256 to its reviewed 64-character digest."
        )
    source_sha = source_bundle_sha256(
        [
            Path(__file__),
            Path(__file__).with_name("features.py"),
            Path(__file__).resolve().parents[1] / "configs" / "config.py",
        ]
    )
    preprocessing = {
        "fs_hz": int(FS),
        "sequence_length": int(SEQ_LEN),
        "n_leads": 12,
        "bandpass_hz": [0.5, 40.0],
        "bandpass_order": 4,
        "normalization": "per_lead_zscore_eps_1e-8",
        "padding": "right_zero_or_head_truncate",
        "label_mapping": sorted((str(code), int(index)) for code, index in SNOMED_MAPPING.items()),
    }
    return {
        "cache_schema_version": CLEAN_CACHE_SCHEMA_VERSION,
        "archive_sha256": archive_sha,
        "preprocessing_source_sha256": source_sha,
        "preprocessing_config_sha256": canonical_json_sha256(preprocessing),
        "preprocessing_config": preprocessing,
    }


def env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def quarantine_file(path: str, reason: str) -> None:
    if not os.path.exists(path):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = f"{path}.corrupt_{stamp}"
    try:
        os.replace(path, target)
        print(f"⚠️  Quarantined corrupt cache: {path}")
        print(f"    -> {target}")
        print(f"    Reason: {reason}")
    except OSError as exc:
        print(f"⚠️  Could not quarantine corrupt cache: {path}")
        print(f"    Reason: {reason}")
        print(f"    Rename error: {exc}")


def validate_clean_cache_arrays(
    X: np.ndarray,
    y: np.ndarray,
    X_raw_amp: np.ndarray,
    subjects: np.ndarray,
) -> str:
    """Validate the clean-cache contract without allocating a full-size mask."""
    n_records = len(X)
    expected = {
        "X": (n_records, 12, SEQ_LEN),
        "y": (n_records, NUM_CLASSES),
        "X_raw_amp": (n_records, 5),
        "subjects": (n_records,),
    }
    actual = {
        "X": X.shape,
        "y": y.shape,
        "X_raw_amp": X_raw_amp.shape,
        "subjects": subjects.shape,
    }
    if actual != expected:
        raise ValueError(f"Clean cache shape mismatch: actual={actual}, expected={expected}")
    if n_records == 0:
        raise ValueError("Clean cache contains no records")
    if len(np.unique(subjects.astype(str))) != n_records:
        raise ValueError("Clean cache record/subject identifiers are not unique")
    if not np.isfinite(y).all() or not np.isfinite(X_raw_amp).all():
        raise ValueError("Clean cache labels or amplitude features are non-finite")
    if not np.logical_or(y == 0, y == 1).all():
        raise ValueError("Clean cache labels must be binary")
    for start in range(0, n_records, 512):
        if not np.isfinite(X[start:start + 512]).all():
            raise ValueError(f"Clean cache ECG contains non-finite values near row {start}")
    return record_order_fingerprint(subjects)


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
    # The combined Chapman-Shaoxing/Ningbo source reports one ECG record per patient.
    # Freeze-time sidecars bind this identifier to the reviewed PhysioNet source contract.
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
    extract_dir = paths['extract_dir']
    zip_path = paths['zip_path']
    clean_contract = clean_cache_source_contract(zip_path)

    # --------------------------------------------------------
    # 1️⃣ LOAD CACHE (ONLY IF VALID & COMPLETE)
    # --------------------------------------------------------
    if os.path.exists(cache_path):
        if env_flag("ECG_RAMBA_USE_CLEAN_CACHE", default=True):
            print(f"⚠️  Loading cached data: {cache_path}")
            try:
                with np.load(cache_path, allow_pickle=True) as d:
                    X_cached = d['X']
                    y_cached = d['y']
                    X_raw_amp_cached = d['X_raw_amp']
                    subjects_cached = d['subjects']
                    stored_record_fingerprint = (
                        str(d["record_order_fingerprint"].item())
                        if "record_order_fingerprint" in d.files
                        else None
                    )
                    stored_contract = {
                        key: (
                            d[key].item()
                            if np.ndim(d[key]) == 0
                            else d[key].tolist()
                        )
                        for key in [
                            "cache_schema_version",
                            "archive_sha256",
                            "preprocessing_source_sha256",
                            "preprocessing_config_sha256",
                        ]
                        if key in d.files
                    }
                record_fingerprint = validate_clean_cache_arrays(
                    X_cached,
                    y_cached,
                    X_raw_amp_cached,
                    subjects_cached,
                )
                if (
                    stored_record_fingerprint is not None
                    and stored_record_fingerprint != record_fingerprint
                ):
                    raise ValueError(
                        "Clean cache record-order fingerprint does not match its subjects array"
                    )
                expected_contract = {
                    key: clean_contract[key]
                    for key in [
                        "cache_schema_version",
                        "archive_sha256",
                        "preprocessing_source_sha256",
                        "preprocessing_config_sha256",
                    ]
                }
                if stored_contract != expected_contract:
                    raise ValueError(
                        "Clean cache provenance contract mismatch: "
                        f"stored={stored_contract}, expected={expected_contract}"
                    )
                print(f"✅ Loaded cleaned data cache: {X_cached.shape}")
                print(f"🔐 Record-order fingerprint: {record_fingerprint}")
                return X_cached, y_cached, X_raw_amp_cached, subjects_cached
            except Exception as exc:
                quarantine_file(cache_path, repr(exc))
                print("⚠️  Cache load failed. Rebuilding from ZIP/raw files.")
        else:
            print(f"⏭️  Ignoring cleaned data cache because ECG_RAMBA_USE_CLEAN_CACHE=0: {cache_path}")

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
            members = z.infolist()
            print(
                f"📦 Extracting {len(members)} archive members to: {extract_dir}",
                flush=True,
            )
            for member in tqdm(members, desc="Extract ZIP", unit="file"):
                z.extract(member, extract_dir)

    # --------------------------------------------------------
    # 3️⃣ SCAN RAW RECORDS (TRUTH SOURCE)
    # --------------------------------------------------------
    hea_files = []
    for root, _, files in os.walk(extract_dir):
        hea_files.extend(
            os.path.join(root, f) for f in files if f.endswith('.hea')
        )
    hea_files.sort(key=lambda path: extract_record_id(path))

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
    record_fingerprint = validate_clean_cache_arrays(X, y, X_raw_amp, subjects)

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
        save_npz_atomic(
            cache_path,
            X=X,
            y=y,
            X_raw_amp=X_raw_amp,
            subjects=subjects,
            record_order_fingerprint=np.asarray(record_fingerprint),
            cache_schema_version=np.asarray(CLEAN_CACHE_SCHEMA_VERSION, dtype=np.int16),
            archive_sha256=np.asarray(clean_contract["archive_sha256"]),
            preprocessing_source_sha256=np.asarray(
                clean_contract["preprocessing_source_sha256"]
            ),
            preprocessing_config_sha256=np.asarray(
                clean_contract["preprocessing_config_sha256"]
            ),
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
