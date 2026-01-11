"""
ECG Signal Processing Module - Complete Implementation
=======================================================
Comprehensive preprocessing pipeline with:
- Auto-detect sampling rate from metadata
- Smart lead mapping from headers/keys
- Artifact rejection (flatline, noise detection)
- WFDB ZIP support
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample
from typing import Tuple, Optional, Dict, Any, List
import zipfile
import tempfile
import os

# =============================================================================
# Constants matching training pipeline (Chapman-Shaoxing)
# =============================================================================
TARGET_FS = 500       # Target sampling rate (Hz)
TARGET_LEN = 5000     # Target signal length (10 seconds @ 500 Hz)
NUM_LEADS = 12        # Standard 12-lead ECG

# Standard 12-lead order
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_INDICES = {name.upper(): i for i, name in enumerate(LEAD_NAMES)}

# Alternative lead name mappings
LEAD_ALIASES = {
    'LEAD_I': 0, 'LEAD_1': 0, 'LEAD1': 0, 'L1': 0,
    'LEAD_II': 1, 'LEAD_2': 1, 'LEAD2': 1, 'L2': 1,
    'LEAD_III': 2, 'LEAD_3': 2, 'LEAD3': 2, 'L3': 2,
    'AVR': 3, 'LEAD_AVR': 3,
    'AVL': 4, 'LEAD_AVL': 4,
    'AVF': 5, 'LEAD_AVF': 5,
    'V1': 6, 'LEAD_V1': 6, 'CHEST1': 6,
    'V2': 7, 'LEAD_V2': 7, 'CHEST2': 7,
    'V3': 8, 'LEAD_V3': 8, 'CHEST3': 8,
    'V4': 9, 'LEAD_V4': 9, 'CHEST4': 9,
    'V5': 10, 'LEAD_V5': 10, 'CHEST5': 10,
    'V6': 11, 'LEAD_V6': 11, 'CHEST6': 11,
}

# Lead groups for accuracy warnings
LIMB_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
CHEST_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Common sampling rates
COMMON_SAMPLE_RATES = [100, 125, 250, 256, 360, 500, 512, 1000, 2000]

# Supported file formats
SUPPORTED_FORMATS = {
    '.mat': 'MATLAB format (Chapman-Shaoxing, CPSC)',
    '.hea': 'PhysioNet WFDB header (requires .dat file)',
    '.dat': 'PhysioNet WFDB data (requires .hea file)',
    '.zip': 'ZIP archive containing WFDB files (.hea + .dat)',
    '.csv': 'CSV format (Universal - personal devices)',
    '.json': 'JSON format (Web-friendly)',
    '.npy': 'NumPy array',
    '.npz': 'NumPy compressed'
}


# =============================================================================
# A. Auto-detect Sampling Rate
# =============================================================================

def detect_sample_rate_from_metadata(metadata: Dict) -> Optional[int]:
    """
    Extract sampling rate from file metadata.
    
    Args:
        metadata: Dict containing file metadata
        
    Returns:
        Detected sampling rate or None if not found
    """
    # Common metadata keys for sampling rate
    fs_keys = ['fs', 'Fs', 'FS', 'sample_rate', 'sampling_rate', 
               'sfreq', 'samplingFrequency', 'frequency', 'freq']
    
    for key in fs_keys:
        if key in metadata:
            try:
                value = metadata[key]
                if hasattr(value, 'flat'):  # numpy array
                    value = value.flat[0]
                fs = int(float(value))
                if 50 <= fs <= 10000:  # Sanity check
                    return fs
            except:
                pass
    
    return None


def estimate_sample_rate_from_signal(signal: np.ndarray, expected_duration: Optional[float] = None) -> int:
    """
    Estimate sampling rate from signal characteristics.
    
    Args:
        signal: ECG signal
        expected_duration: Known duration in seconds if available
        
    Returns:
        Estimated sampling rate (defaults to 500 Hz if unknown)
    """
    num_samples = signal.shape[-1] if signal.ndim == 2 else len(signal)
    
    if expected_duration and expected_duration > 0:
        estimated_fs = int(round(num_samples / expected_duration))
        # Round to nearest common rate
        closest = min(COMMON_SAMPLE_RATES, key=lambda x: abs(x - estimated_fs))
        return closest
    
    # Heuristic: if samples > 10000, likely 1000Hz; if < 2500, likely 250Hz
    if num_samples > 10000:
        return 1000
    elif num_samples > 5000:
        return 500
    elif num_samples > 2500:
        return 250
    elif num_samples > 1000:
        return 250
    else:
        return 500  # Default


# =============================================================================
# B. Smart Lead Mapping
# =============================================================================

def smart_lead_mapping(
    signal: np.ndarray,
    lead_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Intelligently map leads to standard 12-lead positions.
    
    Args:
        signal: Input signal (N_leads, T) or (T,) for single lead
        lead_names: List of lead names from file header/metadata
        
    Returns:
        Tuple of (12-lead signal, mapping info)
    """
    info = {
        'input_leads': signal.shape[0] if signal.ndim == 2 else 1,
        'mapped_leads': [],
        'missing_leads': [],
        'mapping_method': 'unknown',
        'warnings': [],
        'accuracy_notes': []
    }
    
    # Create empty 12-lead tensor
    num_samples = signal.shape[-1] if signal.ndim == 2 else len(signal)
    output = np.zeros((NUM_LEADS, num_samples), dtype=np.float32)
    
    # Case 1: Single lead
    if signal.ndim == 1 or (signal.ndim == 2 and signal.shape[0] == 1):
        sig_1d = signal.flatten() if signal.ndim == 1 else signal[0]
        output[1, :] = sig_1d  # Map to Lead II
        info['mapped_leads'] = ['II']
        info['missing_leads'] = [l for l in LEAD_NAMES if l != 'II']
        info['mapping_method'] = 'single_lead_to_lead_II'
        info['warnings'].append("Single-lead input → mapped to Lead II")
        
    # Case 2: Lead names provided (from header/metadata)
    elif lead_names is not None and len(lead_names) > 0:
        info['mapping_method'] = 'smart_mapping_from_names'
        mapped_count = 0
        
        for i, name in enumerate(lead_names):
            if i >= signal.shape[0]:
                break
            
            # Normalize lead name
            normalized = name.upper().strip().replace(' ', '_').replace('-', '_')
            
            # Try standard names first
            if normalized in LEAD_INDICES:
                idx = LEAD_INDICES[normalized]
                output[idx, :] = signal[i, :]
                info['mapped_leads'].append(LEAD_NAMES[idx])
                mapped_count += 1
            # Try aliases
            elif normalized in LEAD_ALIASES:
                idx = LEAD_ALIASES[normalized]
                output[idx, :] = signal[i, :]
                info['mapped_leads'].append(LEAD_NAMES[idx])
                mapped_count += 1
            # Try partial match
            else:
                matched = False
                for std_name in LEAD_NAMES:
                    if std_name in normalized or normalized in std_name:
                        idx = LEAD_NAMES.index(std_name)
                        output[idx, :] = signal[i, :]
                        info['mapped_leads'].append(std_name)
                        mapped_count += 1
                        matched = True
                        break
                
                if not matched:
                    info['warnings'].append(f"Unknown lead name '{name}' at position {i}")
        
        info['missing_leads'] = [l for l in LEAD_NAMES if l not in info['mapped_leads']]
        
        if mapped_count == 0:
            # Fallback to sequential mapping
            info['mapping_method'] = 'fallback_sequential'
            info['warnings'].append("Could not match lead names, using sequential order")
            for i in range(min(signal.shape[0], NUM_LEADS)):
                output[i, :] = signal[i, :]
                info['mapped_leads'].append(LEAD_NAMES[i])
            info['missing_leads'] = LEAD_NAMES[signal.shape[0]:]
    
    # Case 3: No lead names - assume sequential standard order
    elif signal.shape[0] <= NUM_LEADS:
        info['mapping_method'] = 'sequential_standard_order'
        for i in range(signal.shape[0]):
            output[i, :] = signal[i, :]
            info['mapped_leads'].append(LEAD_NAMES[i])
        info['missing_leads'] = LEAD_NAMES[signal.shape[0]:]
        if signal.shape[0] < NUM_LEADS:
            info['warnings'].append(f"Assumed standard order for {signal.shape[0]} leads")
    
    # Case 4: Already 12 leads
    elif signal.shape[0] == NUM_LEADS:
        output = signal.astype(np.float32)
        info['mapped_leads'] = LEAD_NAMES.copy()
        info['mapping_method'] = 'full_12_lead'
    
    # Case 5: More than 12 leads
    else:
        output = signal[:NUM_LEADS, :].astype(np.float32)
        info['mapped_leads'] = LEAD_NAMES.copy()
        info['mapping_method'] = 'truncated'
        info['warnings'].append(f"Truncated from {signal.shape[0]} to 12 leads")
    
    # Generate accuracy warnings
    info['accuracy_notes'] = generate_lead_dropout_warnings(info['missing_leads'])
    
    return output, info


def generate_lead_dropout_warnings(missing_leads: List[str]) -> List[str]:
    """
    Generate accuracy warnings based on Lead Dropout experiment (Figure 5).
    """
    warnings = []
    
    if not missing_leads:
        return warnings
    
    missing_set = set(missing_leads)
    missing_chest = missing_set.intersection(set(CHEST_LEADS))
    
    if len(missing_chest) == 6:
        warnings.append(
            "⚠️ CHÚ Ý: Thiếu tất cả chuyển đạo ngực (V1-V6). "
            "Các bệnh về HÌNH THÁI (MI, LBBB, RBBB) sẽ GIẢM ĐỘ CHÍNH XÁC ĐÁNG KỂ."
        )
        warnings.append(
            "✓ Các bệnh về NHỊP (Rung nhĩ AFIB, Nhịp nhanh/chậm) vẫn nhận diện tốt."
        )
    elif len(missing_chest) >= 3:
        warnings.append(
            f"⚠️ Thiếu {len(missing_chest)}/6 chuyển đạo ngực. "
            "Độ chính xác cho bệnh hình thái có thể giảm."
        )
    
    if len(missing_leads) >= 11:
        warnings.append(
            "ℹ️ Dữ liệu 1 kênh: Tốt nhất cho Rung nhĩ (AFIB) và bất thường nhịp."
        )
    elif len(missing_leads) >= 9:
        warnings.append(
            f"ℹ️ Dữ liệu {12 - len(missing_leads)} kênh: Độ chính xác ~70-85% so với 12-lead."
        )
    
    return warnings


# =============================================================================
# C. Artifact Rejection
# =============================================================================

def detect_artifacts(signal: np.ndarray, fs: int = TARGET_FS) -> Dict[str, Any]:
    """
    Detect signal artifacts and quality issues.
    
    Checks for:
    - Flatline (dead signal)
    - Excessive noise
    - Clipping
    - Very short duration
    
    Args:
        signal: ECG signal (leads, samples)
        fs: Sampling frequency
        
    Returns:
        Dict with artifact detection results
    """
    result = {
        'is_valid': True,
        'quality_score': 1.0,
        'issues': [],
        'warnings': []
    }
    
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    num_leads, num_samples = signal.shape
    duration_s = num_samples / fs
    
    # Check 1: Duration too short
    if duration_s < 1.0:
        result['issues'].append(f"Tín hiệu quá ngắn: {duration_s:.2f}s (tối thiểu 1s)")
        result['quality_score'] *= 0.5
    elif duration_s < 5.0:
        result['warnings'].append(f"Tín hiệu ngắn: {duration_s:.1f}s (khuyến nghị ≥10s)")
        result['quality_score'] *= 0.8
    
    # Check 2: Flatline detection (lead by lead)
    flatline_leads = []
    for i in range(num_leads):
        lead_data = signal[i, :]
        std = np.std(lead_data)
        
        if std < 1e-6:  # Completely flat
            flatline_leads.append(LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"Lead_{i}")
    
    if len(flatline_leads) == num_leads:
        result['is_valid'] = False
        result['issues'].append("Tất cả các leads đều là flatline (tín hiệu chết)")
        result['quality_score'] = 0.0
    elif len(flatline_leads) > 0:
        result['warnings'].append(f"Các leads flatline: {', '.join(flatline_leads)}")
        result['quality_score'] *= (1 - len(flatline_leads) / num_leads * 0.5)
    
    # Check 3: Excessive noise detection
    noisy_leads = []
    for i in range(num_leads):
        lead_data = signal[i, :]
        
        # Check if signal looks like white noise (high-frequency content dominates)
        if len(lead_data) > 100:
            diff = np.diff(lead_data)
            diff_std = np.std(diff)
            sig_std = np.std(lead_data)
            
            if sig_std > 0 and diff_std / sig_std > 2.0:
                noisy_leads.append(LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"Lead_{i}")
    
    if len(noisy_leads) > num_leads // 2:
        result['warnings'].append(f"Nhiều leads có nhiễu cao: {', '.join(noisy_leads[:3])}...")
        result['quality_score'] *= 0.7
    elif len(noisy_leads) > 0:
        result['warnings'].append(f"Có nhiễu ở: {', '.join(noisy_leads)}")
        result['quality_score'] *= 0.9
    
    # Check 4: Clipping detection
    clipped_leads = []
    for i in range(num_leads):
        lead_data = signal[i, :]
        max_val = np.max(np.abs(lead_data))
        
        if max_val > 0:
            near_max = np.sum(np.abs(lead_data) > 0.99 * max_val)
            if near_max > len(lead_data) * 0.01:  # >1% samples clipped
                clipped_leads.append(LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"Lead_{i}")
    
    if len(clipped_leads) > 0:
        result['warnings'].append(f"Tín hiệu bị cắt (clipping) ở: {', '.join(clipped_leads)}")
        result['quality_score'] *= 0.85
    
    # Final validity check
    if result['quality_score'] < 0.3:
        result['is_valid'] = False
        result['issues'].append("Chất lượng tín hiệu quá thấp để phân tích")
    
    return result


# =============================================================================
# D. WFDB ZIP Support
# =============================================================================

def parse_wfdb_zip(zip_content: bytes) -> Tuple[Optional[np.ndarray], int, List[str], List[str]]:
    """
    Parse WFDB files (.hea + .dat) from a ZIP archive.
    
    Args:
        zip_content: Raw bytes of ZIP file
        
    Returns:
        Tuple of (signal, sample_rate, lead_names, warnings)
    """
    warnings = []
    
    try:
        import wfdb
        HAS_WFDB = True
    except ImportError:
        HAS_WFDB = False
        return None, TARGET_FS, [], ["WFDB package không được cài đặt. Chạy: pip install wfdb"]
    
    try:
        # Create temp directory for extraction
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract ZIP
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Find .hea file
            hea_files = []
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith('.hea'):
                        hea_files.append(os.path.join(root, f))
            
            if not hea_files:
                return None, TARGET_FS, [], ["Không tìm thấy file .hea trong ZIP"]
            
            # Use first .hea file found
            hea_path = hea_files[0]
            record_name = hea_path[:-4]  # Remove .hea extension
            
            # Read WFDB record
            record = wfdb.rdrecord(record_name)
            
            signal = record.p_signal.T  # Convert to (leads, samples)
            sample_rate = record.fs
            lead_names = record.sig_name if hasattr(record, 'sig_name') else None
            
            if len(hea_files) > 1:
                warnings.append(f"Tìm thấy {len(hea_files)} file .hea, sử dụng: {os.path.basename(hea_files[0])}")
            
            return signal, sample_rate, lead_names, warnings
            
    except Exception as e:
        return None, TARGET_FS, [], [f"Lỗi đọc WFDB: {str(e)}"]


# Need io for BytesIO
import io


# =============================================================================
# Complete Preprocessing Pipeline
# =============================================================================

def preprocess_ecg(
    signal: np.ndarray, 
    fs: int = None,
    lead_names: Optional[List[str]] = None,
    skip_artifact_check: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete ECG preprocessing pipeline.
    
    Pipeline:
    1. Artifact detection
    2. Smart lead mapping
    3. Resampling to 500 Hz
    4. Bandpass filter (0.5-40 Hz)
    5. Z-score normalization
    6. Pad/truncate to 5000 samples
    
    Args:
        signal: Raw ECG signal
        fs: Sampling frequency (auto-detected if None)
        lead_names: Lead names from file header
        skip_artifact_check: Skip artifact detection
        
    Returns:
        Tuple of (processed_signal, info_dict)
    """
    info = {
        'original_shape': signal.shape,
        'input_fs': fs,
        'pipeline_steps': [],
        'warnings': [],
        'accuracy_notes': [],
        'quality': {}
    }
    
    # Auto-detect fs if not provided
    if fs is None:
        fs = estimate_sample_rate_from_signal(signal)
        info['input_fs'] = fs
        info['warnings'].append(f"Tần số lấy mẫu được ước lượng: {fs} Hz")
    
    info['detected_fs'] = fs
    
    # Handle NaN/Inf
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        info['pipeline_steps'].append('Replaced NaN/Inf values')
    
    # Ensure 2D: (leads, samples)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    elif signal.ndim == 2 and signal.shape[1] <= 12 and signal.shape[0] > 12:
        signal = signal.T
        info['pipeline_steps'].append('Transposed (samples × leads → leads × samples)')
    
    # Artifact detection
    if not skip_artifact_check:
        artifact_result = detect_artifacts(signal, fs)
        info['quality'] = artifact_result
        
        if not artifact_result['is_valid']:
            info['warnings'].extend(artifact_result['issues'])
            # Still continue processing but warn user
        
        info['warnings'].extend(artifact_result.get('warnings', []))
    
    # Smart lead mapping
    signal, mapping_info = smart_lead_mapping(signal, lead_names)
    info['mapped_leads'] = mapping_info['mapped_leads']
    info['missing_leads'] = mapping_info['missing_leads']
    info['mapping_method'] = mapping_info['mapping_method']
    info['warnings'].extend(mapping_info['warnings'])
    info['accuracy_notes'].extend(mapping_info['accuracy_notes'])
    info['pipeline_steps'].append(f"Lead mapping: {mapping_info['mapping_method']}")
    
    # Resampling
    if fs != TARGET_FS:
        original_samples = signal.shape[1]
        new_samples = int(signal.shape[1] * TARGET_FS / fs)
        signal = resample(signal, new_samples, axis=1)
        info['pipeline_steps'].append(f"Resampled: {fs}Hz → {TARGET_FS}Hz ({original_samples} → {new_samples} samples)")
    
    # Bandpass filter
    try:
        signal = bandpass_filter(signal, lowcut=0.5, highcut=40, fs=TARGET_FS, order=4)
        info['pipeline_steps'].append("Bandpass filter: 0.5-40Hz (Butterworth order 4)")
    except Exception as e:
        info['warnings'].append(f"Bandpass filter failed: {str(e)}")
    
    # =========================================================================
    # CRITICAL: Save raw signal for amplitude features (BEFORE normalize)
    # This matches training pipeline in data_loader.py:
    #   signal = bandpass_filter(signal)
    #   amp_feats = extract_amplitude_features(signal)  # RAW amplitudes
    #   signal = normalize_signal(signal)
    # =========================================================================
    raw_for_amplitude = signal.copy()  # Save BEFORE normalize for amplitude extraction
    
    # Normalization
    signal = normalize_zscore(signal)
    info['pipeline_steps'].append("Normalization: Instance-wise Z-score")
    
    # Pad/truncate (apply to both normalized and raw)
    original_len = signal.shape[1]
    if signal.shape[1] < TARGET_LEN:
        pad_len = TARGET_LEN - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
        raw_for_amplitude = np.pad(raw_for_amplitude, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
        info['pipeline_steps'].append(f"Zero-padded: {original_len} → {TARGET_LEN} samples")
    elif signal.shape[1] > TARGET_LEN:
        signal = signal[:, :TARGET_LEN]
        raw_for_amplitude = raw_for_amplitude[:, :TARGET_LEN]
        info['pipeline_steps'].append(f"Truncated: {original_len} → {TARGET_LEN} samples")
    
    info['final_shape'] = signal.shape
    info['final_fs'] = TARGET_FS
    info['duration_s'] = TARGET_LEN / TARGET_FS
    info['raw_for_amplitude'] = raw_for_amplitude.astype(np.float32)  # For amplitude feature extraction
    
    return signal.astype(np.float32), info


def bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40, 
                    fs: int = TARGET_FS, order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=-1)


def normalize_zscore(signal: np.ndarray) -> np.ndarray:
    """Instance-wise Z-score normalization."""
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True) + 1e-8
    return (signal - mean) / std


def validate_ecg_signal(signal: np.ndarray) -> Dict[str, Any]:
    """Validate ECG signal before processing."""
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    if not isinstance(signal, np.ndarray):
        result['valid'] = False
        result['errors'].append("Input must be a numpy array")
        return result
    
    if signal.ndim == 1:
        result['info']['detected_type'] = 'single_lead'
        result['info']['num_samples'] = len(signal)
    elif signal.ndim == 2:
        if signal.shape[0] <= 12 and signal.shape[1] > 12:
            result['info']['detected_type'] = 'multi_lead'
            result['info']['num_leads'] = signal.shape[0]
            result['info']['num_samples'] = signal.shape[1]
        elif signal.shape[1] <= 12 and signal.shape[0] > 12:
            result['info']['detected_type'] = 'multi_lead_transposed'
            result['info']['num_leads'] = signal.shape[1]
            result['info']['num_samples'] = signal.shape[0]
            result['warnings'].append("Signal appears transposed (samples × leads)")
        else:
            result['info']['detected_type'] = 'ambiguous'
    else:
        result['valid'] = False
        result['errors'].append(f"Invalid dimensions: expected 1D or 2D, got {signal.ndim}D")
        return result
    
    num_samples = result['info'].get('num_samples', len(signal) if signal.ndim == 1 else signal.shape[-1])
    if num_samples < 250:
        result['valid'] = False
        result['errors'].append(f"Signal too short: {num_samples} samples (minimum 250)")
    
    result['info']['duration_s'] = num_samples / TARGET_FS
    
    return result


def get_format_error_message(filename: str) -> Optional[str]:
    """Generate error message for unsupported formats."""
    ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    
    if ext in SUPPORTED_FORMATS:
        return None
    
    supported_list = ", ".join(SUPPORTED_FORMATS.keys())
    return f"Định dạng '{ext}' không được hỗ trợ. Các định dạng hỗ trợ: {supported_list}"
