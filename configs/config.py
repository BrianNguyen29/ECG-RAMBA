
import os
import json
import hashlib
import torch

# ============================================================
# DEVICE & SIGNAL SETUP
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 5000        # Chapman–Shaoxing ECG length
FS = 500              # Sampling rate (Hz)

# ============================================================
# CORE CONFIGURATION
# ============================================================
BASE_CONFIG = {
    # ================= MODEL CAPACITY =================
    'd_model': 384,
    'n_layers': 16,
    'hydra_dim': 3072,
    'n_latents': 64,
    'hrv_dim': 36,
    'drop_path_rate': 0.3,

    # ================= ARCHITECTURE ==================
    'use_cross_attention_fusion': True,
    'fusion_heads': 8,
    'use_spatial_attention': True,
    'use_hrv': True,
    'use_final_perceiver': True,

    # ============== EVENT-CENTRIC SLICING =============
    'use_event_slicing': True,
    'slice_length': 2500,
    'slice_stride': 1250,
    'max_slices_per_record': 6,
    'aggregation_method': 'mean',

    # ================= TRAINING ======================
    'batch_size': 192,
    'accum_iter': 1,
    'epochs': 20,
    'lr_max': 9e-4,
    'lr_min': 1e-6,
    'weight_decay': 0.05,
    'grad_clip': 1.0,
    'num_workers': 8,

    # ================= WARMUP ========================
    'warmup_epochs': 8,
    'warmup_lr_scale': 1.0,

    # ================= LOSS STRATEGY =================
    'loss_type': 'bce_then_asymmetric',
    'asym_start_epoch': 8,

    # ASYMMETRIC LOSS PARAMS
    'asym_gamma_neg': 2.5,
    'asym_gamma_pos': 0.0,
    'asym_clip': 0.05,
    'use_negative_reweighting': False,

    # ================= OPTIMIZATION ==================
    'use_gradient_centralization': False,

    # ================= WEIGHT AVERAGING ===============
    # EMA allowed for evaluation only
    'use_ema': True,
    'ema_decay': 0.999,

    # SWA
    'use_swa': False,

    # ================= CALIBRATION ===================
    'use_calibration': False,
    'default_threshold': 0.5,

    # ================= CLEAN GUARANTEES ===============
    'optimize_thresholds': False,
    'label_smoothing': 0.0,
    'use_soft_clinical_constraints': False,

    # ================= EVALUATION =====================
    'training_mode': 'kfold',
    'cv_strategy': 'StratifiedGroupKFold',
    'group_key': 'subject_id',
    'n_folds': 5,
    'seeds': [42],

    # ================= FEATURE LEARNING ===============
    'hydra_pca_mode': 'fold_aware',
    'seq_len_after_tokenizer': 625,
}

CONFIG = BASE_CONFIG
CONFIG['_profile'] = ''

# ============================================================
# ABLATION CONFIG (MODEL COMPATIBILITY)
# ============================================================
ABLATION_CONFIG = {
    'use_multiscale': True,
    'use_rocket': True,
    'use_hrv': True,
}

# ============================================================
# FIXED TAXONOMY (CHAPMAN–SHAOXING / SNOMED CT)
# ============================================================
CLASSES = [
    'AF', 'AFL', 'Brady', 'CRBBB', 'IAVB', 'IRBBB', 'LAD', 'LBBB',
    'LAnFB', 'LQT', 'LQRSV', 'NSIVCB', 'PR', 'PAC', 'PVC', 'QAb',
    'RAD', 'RBBB', 'SA', 'SB', 'SNR', 'STach', 'SVPB', 'TAb',
    'TInv', 'VPB', 'LPR'
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Required by data.py (metadata only, not used in training)
SNOMED_MAPPING = {
    '164889003': 0, '164890007': 1, '426627000': 2, '713427006': 3,
    '270492004': 4, '713426002': 5, '39732003': 6, '164909002': 7,
    '445118002': 8, '111975006': 9, '251146004': 10, '698252002': 11,
    '10370003': 12, '284470004': 13, '427172004': 14, '164917005': 15,
    '47665007': 16, '59118001': 17, '427393009': 18, '426177001': 19,
    '426783006': 20, '427084000': 21, '63593006': 22, '164934002': 23,
    '59931005': 24, '17338001': 25, '164947007': 26
}

# ============================================================
# CONFIG HASH & PATHS
# ============================================================
CONFIG_HASH = hashlib.md5(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()[:8]

def setup_paths(num_classes: int, hydra_dim: int, cfg_hash: str, drive_mounted: bool = True) -> dict:
    # Update cache dir to be relative to the project if not in colab
    base_dir = './ecg_cache' 
    if os.path.exists('/content/drive/MyDrive/ECG'):
        base_dir = '/content/drive/MyDrive/ECG'
        
    paths = {
        'cache_dir': base_dir,
        'zip_path': f'{base_dir}/archive.zip',
        'extract_dir': './data/chapman',
        'data_cache': f'{base_dir}/ecg_data_{num_classes}c_subject.npz',
        'model_dir': './models',
    }
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(paths['model_dir'], exist_ok=True)
    os.makedirs(paths['extract_dir'], exist_ok=True)
    return paths

PATHS = setup_paths(NUM_CLASSES, CONFIG['hydra_dim'], CONFIG_HASH, drive_mounted=False)
