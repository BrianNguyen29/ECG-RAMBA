"""
ECG-RAMBA API Endpoints - Complete Implementation
==================================================
REST API with:
- Auto-detect sampling rate
- Smart lead mapping
- Artifact rejection
- WFDB ZIP upload support
- Detailed error messages in Vietnamese
- [P0] Upload size limit & structured logging
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import numpy as np
from scipy.io import loadmat
import io
import os
import zipfile
import logging

from app.core.model_loader import ecg_ramba
from app.core.signal_processing import (
    preprocess_ecg, validate_ecg_signal, detect_artifacts,
    detect_sample_rate_from_metadata, parse_wfdb_zip,
    SUPPORTED_FORMATS, get_format_error_message,
    TARGET_FS, TARGET_LEN, NUM_LEADS, LEAD_NAMES, LEAD_ALIASES
)

# =============================================================================
# Configuration
# =============================================================================
MAX_UPLOAD_SIZE_MB = 50  # P0.1: Maximum upload file size
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Setup structured logging
logger = logging.getLogger("ecg-ramba-api")
logger.setLevel(logging.INFO)

# =============================================================================
# P1.1: Rate Limiting with SlowAPI
# =============================================================================
from slowapi import Limiter
from slowapi.util import get_remote_address
import hashlib
from functools import lru_cache

limiter = Limiter(key_func=get_remote_address)

# =============================================================================
# P1.2: Request Caching with TTL
# =============================================================================
import time as _time

_prediction_cache = {}  # {hash: {"result": dict, "timestamp": float}}
CACHE_MAX_SIZE = 50
CACHE_TTL_SECONDS = 300  # 5 minutes

def get_signal_hash(signal_data: list) -> str:
    """Generate SHA256 hash for signal data for caching."""
    signal_str = json.dumps(signal_data, sort_keys=True)
    return hashlib.sha256(signal_str.encode()).hexdigest()[:16]

def cache_prediction(signal_hash: str, result: dict):
    """Store prediction result in cache with timestamp."""
    # Evict expired entries first
    current_time = _time.time()
    expired_keys = [k for k, v in _prediction_cache.items() 
                    if current_time - v.get("timestamp", 0) > CACHE_TTL_SECONDS]
    for k in expired_keys:
        del _prediction_cache[k]
    
    # Evict oldest if still over limit
    if len(_prediction_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_prediction_cache.keys(), 
                         key=lambda k: _prediction_cache[k].get("timestamp", 0))
        del _prediction_cache[oldest_key]
    
    _prediction_cache[signal_hash] = {"result": result, "timestamp": current_time}

def get_cached_prediction(signal_hash: str) -> Optional[dict]:
    """Retrieve cached prediction if exists and not expired."""
    entry = _prediction_cache.get(signal_hash)
    if entry:
        if _time.time() - entry.get("timestamp", 0) < CACHE_TTL_SECONDS:
            return entry["result"]
        else:
            # Expired, remove it
            del _prediction_cache[signal_hash]
    return None

def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    return obj

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictionRequest(BaseModel):
    model_name: str
    signal_data: List[List[float]]

class SimplePredictionRequest(BaseModel):
    model_name: str
    signal_data: List[float]

class EnsemblePredictionRequest(BaseModel):
    signal_data: List[List[float]]
    raw_signal_data: Optional[List[List[float]]] = None
    active_leads: Optional[List[bool]] = None # Deep RAMBA: Lead Dropout Simulation  # For amplitude feature extraction


# =============================================================================
# Prediction Endpoints
# =============================================================================

@router.get("/models")
async def get_models():
    """Return list of available ECG-RAMBA model checkpoints."""
    models = ecg_ramba.get_available_models()
    if not models:
        return ["fold1_best.pt", "fold2_best.pt", "fold3_best.pt", "fold4_best.pt", "fold5_best.pt"]
    return models


@router.post("/predict")
async def predict(request: PredictionRequest, explain: bool = False):
    """
    Run ECG-RAMBA inference on 12-lead ECG signal.
    Args:
        explain: If True, returns 'saliency_map' (12x5000) for the top diagnosis.
    """
    try:
        signal = np.array(request.signal_data, dtype=np.float32)
        
        if signal.ndim == 1:
            signal = np.tile(signal, (12, 1))
        
        if signal.shape[0] != 12:
            raise HTTPException(status_code=400, detail=f"Expected 12 leads, got {signal.shape[0]}")
        
        result = ecg_ramba.predict(request.model_name, signal)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        # Optional: Saliency Map & Clinical Features
        if explain:
            # Re-run forward pass with gradients for explanation
            saliency = ecg_ramba.explain_prediction(request.model_name, signal)
            result["saliency_map"] = saliency.tolist()

            # --- CLINICAL FEATURES (Deep Analysis) ---
            try:
                from app.core.clinical_measurements import ClinicalMeasurer
                
                # Use Lead II (Index 1) for measurements if available, else Lead I
                lead_idx = 1 if signal.shape[0] > 1 else 0
                lead_sig = signal[lead_idx] # Normalized signal
                
                # Use raw signal if possible? Here we only have preprocessed 'signal'
                # Peaks on normalized signal are fine for index locations
                peaks = ClinicalMeasurer.detect_r_peaks(lead_sig)
                waves = ClinicalMeasurer.delineate_waves(lead_sig, peaks)
                st_analysis = ClinicalMeasurer.analyze_st_segment(lead_sig, waves)
                
                result["clinical_features"] = {
                    "lead_used": "II" if lead_idx == 1 else "I",
                    "r_peaks": peaks.tolist(), # Indices
                    "wave_boundaries": waves, # List of dicts
                    "st_findings": st_analysis,
                    "heart_rate": len(peaks) * (60 / (signal.shape[1]/500)) if len(peaks) > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Clinical analysis failed: {e}")
                result["clinical_error"] = str(e)
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/simple")
async def predict_simple(request: SimplePredictionRequest):
    """Simplified prediction for single-lead ECG."""
    try:
        signal_1d = np.array(request.signal_data, dtype=np.float32)
        signal = np.tile(signal_1d, (12, 1))
        result = ecg_ramba.predict(request.model_name, signal)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/ensemble")
@limiter.limit("30/minute")  # P1.1: Rate limit
async def predict_ensemble(request: Request, body: EnsemblePredictionRequest, explain: bool = False, mode: str = 'accurate'):
    """
    Run ECG inference on 12-lead ECG signal with caching.
    
    Args:
        explain: If True, include saliency map and disentanglement scores.
        mode: 'fast' (single fold ~5s) or 'accurate' (5-fold parallel ensemble ~8s)
    """
    try:
        # P1.2: Check cache first
        signal_hash = get_signal_hash(body.signal_data)
        cached = get_cached_prediction(signal_hash)
        if cached and not explain:  # Don't use cache for explain mode
            logger.info(f"Cache hit for signal {signal_hash}")
            return {**cached, "cached": True}
        
        signal = np.array(body.signal_data, dtype=np.float32)
        
        if signal.ndim == 1:
            signal = np.tile(signal, (12, 1))
        
        if signal.shape[0] != 12:
            raise HTTPException(status_code=400, detail=f"Expected 12 leads, got {signal.shape[0]}")
        
        # Convert raw_signal if provided (for accurate amplitude feature extraction)
        raw_signal = None
        if body.raw_signal_data is not None:
            raw_signal = np.array(body.raw_signal_data, dtype=np.float32)
            if raw_signal.ndim == 1:
                raw_signal = np.tile(raw_signal, (12, 1))
        
        result = ecg_ramba.predict_ensemble(signal, raw_signal=raw_signal, explain=explain, active_leads=body.active_leads, mode=mode)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # P1.2: Cache result (only non-explain results)
        if not explain:
            cache_prediction(signal_hash, result)
            logger.info(f"Cached prediction for signal {signal_hash}")
        
        return sanitize_for_json(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")


# =============================================================================
# Upload Endpoint with Complete Processing
# =============================================================================

@router.post("/upload")
@limiter.limit("60/minute")  # P1.1: Rate limit for uploads
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    sample_rate: Optional[int] = Form(None),
    duration_seconds: Optional[float] = Form(None)
):
    """
    Parse and preprocess uploaded ECG file.
    
    Args:
        file: ECG data file
        sample_rate: Optional - sampling rate in Hz (required for CSV without metadata)
        duration_seconds: Optional - known duration for auto-detection
    
    Supported formats:
    - .mat: MATLAB format (auto-detect Fs from metadata)
    - .zip: WFDB format (.hea + .dat pair)
    - .csv: CSV format (requires sample_rate parameter if not in header)
    - .json: JSON format
    - .npy/.npz: NumPy arrays
    
    Processing Pipeline:
    1. Artifact detection (flatline, noise, clipping)
    2. Smart lead mapping
    3. Resampling to 500 Hz
    4. Bandpass filter (0.5-40 Hz)
    5. Z-score normalization
    6. Pad/truncate to 5000 samples
    """
    try:
        content = await file.read()
        filename = file.filename.lower() if file.filename else ""
        ext = os.path.splitext(filename)[1]
        
        # P0.1: Check file size limit
        if len(content) > MAX_UPLOAD_BYTES:
            logger.warning(f"Upload rejected: {filename} ({len(content)/1024/1024:.1f}MB > {MAX_UPLOAD_SIZE_MB}MB)")
            raise HTTPException(
                status_code=413,
                detail=f"File quá lớn: {len(content)/1024/1024:.1f}MB. Giới hạn: {MAX_UPLOAD_SIZE_MB}MB"
            )
        
        logger.info(f"Processing upload: {filename} ({len(content)/1024:.1f}KB)")
        
        # Check format support
        format_error = get_format_error_message(filename)
        if format_error:
            return {
                "error": format_error,
                "supported_formats": list(SUPPORTED_FORMATS.keys()),
                "signal": None
            }
        
        signal = None
        detected_fs = sample_rate  # Use provided, will auto-detect if None
        lead_names = None
        parse_warnings = []
        metadata = {}
        
        # ===== ZIP (WFDB) =====
        if ext == ".zip":
            signal, detected_fs, lead_names, parse_warnings = parse_wfdb_zip(content)
            if signal is None:
                return {
                    "error": "Không thể đọc file WFDB từ ZIP",
                    "details": parse_warnings,
                    "hint": "Đảm bảo ZIP chứa cả file .hea và .dat tương ứng"
                }
        
        # ===== JSON =====
        elif ext == ".json":
            try:
                data = json.loads(content)
                signal, detected_fs, lead_names, parse_warnings, metadata = parse_json_ecg(data, sample_rate)
            except json.JSONDecodeError as e:
                return {"error": f"JSON không hợp lệ: {str(e)}"}
        
        # ===== CSV =====
        elif ext == ".csv":
            signal, lead_names, parse_warnings = parse_csv_ecg(content)
            
            # CSV requires sample_rate input if not detected
            if sample_rate is None:
                # Try to detect from duration hint
                if duration_seconds and duration_seconds > 0 and signal is not None:
                    num_samples = signal.shape[-1] if signal.ndim == 2 else len(signal)
                    detected_fs = int(round(num_samples / duration_seconds))
                    parse_warnings.append(f"Tần số lấy mẫu ước lượng từ thời gian: {detected_fs} Hz")
                else:
                    parse_warnings.append(
                        "⚠️ Tần số lấy mẫu (Fs) không được cung cấp. "
                        "Sử dụng mặc định 500 Hz. Vui lòng nhập Fs chính xác để đảm bảo độ chính xác."
                    )
                    detected_fs = TARGET_FS
            else:
                detected_fs = sample_rate
        
        # ===== EDF (EEG/Multi-modal) =====
        elif ext == ".edf":
            from app.core.signal_processing import parse_edf_file
            signal, detected_fs, ch_names, parse_warnings = parse_edf_file(content)
            
            if signal is None:
                return {
                    "error": f"Lỗi đọc EDF: {parse_warnings[0] if parse_warnings else 'Unknown error'}",
                    "details": parse_warnings
                }
            
            # Return specialized EEG response (skip ECG pipeline)
            return {
                "modality": "EEG",
                "signal": signal.tolist(),
                "channels": ch_names,
                "sample_rate": detected_fs,
                "num_channels": int(signal.shape[0]),
                "samples": int(signal.shape[1]),
                "duration_s": float(signal.shape[1] / detected_fs),
                "warnings": parse_warnings
            }
        
        # ===== MAT (MATLAB) =====
        elif ext == ".mat":
            signal, detected_fs, lead_names, parse_warnings, metadata = parse_mat_ecg(content, sample_rate)
        
        # ===== NPY =====
        elif ext == ".npy":
            try:
                signal = np.load(io.BytesIO(content))
                if sample_rate is None:
                    parse_warnings.append("⚠️ File NPY không chứa metadata Fs. Sử dụng mặc định 500 Hz.")
                    detected_fs = TARGET_FS
                else:
                    detected_fs = sample_rate
            except Exception as e:
                return {"error": f"Không thể đọc file NPY: {str(e)}"}
        
        # ===== NPZ =====
        elif ext == ".npz":
            try:
                npz_data = np.load(io.BytesIO(content))
                for key in ['signal', 'ecg', 'data', 'arr_0']:
                    if key in npz_data:
                        signal = npz_data[key]
                        break
                if signal is None and len(npz_data.keys()) > 0:
                    signal = npz_data[list(npz_data.keys())[0]]
                
                # Check for fs in npz
                if 'fs' in npz_data:
                    detected_fs = int(npz_data['fs'])
                elif sample_rate is None:
                    detected_fs = TARGET_FS
                    parse_warnings.append("⚠️ File NPZ không chứa metadata Fs.")
                else:
                    detected_fs = sample_rate
                    
            except Exception as e:
                return {"error": f"Không thể đọc file NPZ: {str(e)}"}
        
        # ===== HEA/DAT (single file) =====
        elif ext in [".hea", ".dat"]:
            return {
                "error": "Định dạng WFDB yêu cầu cả file .hea và .dat",
                "solution": "Vui lòng nén cả 2 file vào 1 file .zip và upload",
                "hint": "Ví dụ: Nén record.hea + record.dat → record.zip"
            }
        
        else:
            return {
                "error": f"Định dạng không hỗ trợ: {ext}",
                "supported_formats": list(SUPPORTED_FORMATS.keys())
            }
        
        # ===== Validate =====
        if signal is None:
            return {"error": "Không thể trích xuất tín hiệu từ file"}
        
        signal = np.array(signal, dtype=np.float32)
        
        # Handle transposed data
        if signal.ndim == 2 and signal.shape[1] <= 12 and signal.shape[0] > 12:
            signal = signal.T
            parse_warnings.append("Đã xoay dữ liệu (samples × leads → leads × samples)")
        
        validation = validate_ecg_signal(signal)
        if not validation['valid']:
            return {
                "error": "Tín hiệu ECG không hợp lệ",
                "details": validation['errors'],
                "signal": None
            }
        
        # ===== Preprocess =====
        processed_signal, proc_info = preprocess_ecg(
            signal, 
            fs=detected_fs,
            lead_names=lead_names
        )
        
        # Combine all info
        all_warnings = parse_warnings + proc_info.get('warnings', [])
        
        # Get raw signal for amplitude features (required for accurate inference)
        raw_for_amplitude = proc_info.get('raw_for_amplitude')
        
        return {
            "signal": processed_signal.tolist(),
            "raw_for_amplitude": raw_for_amplitude.tolist() if raw_for_amplitude is not None else None,
            "num_leads": int(processed_signal.shape[0]),
            "samples": int(processed_signal.shape[1]),
            "duration_s": float(processed_signal.shape[1] / TARGET_FS),
            "sample_rate": TARGET_FS,
            "input_sample_rate": detected_fs,
            "original_shape": list(signal.shape),
            "mapped_leads": proc_info.get('mapped_leads', LEAD_NAMES),
            "missing_leads": proc_info.get('missing_leads', []),
            "mapping_method": proc_info.get('mapping_method', 'unknown'),
            "preprocessing": proc_info.get('pipeline_steps', []),
            "quality": proc_info.get('quality', {}),
            "warnings": all_warnings if all_warnings else None,
            "accuracy_notes": proc_info.get('accuracy_notes', []) or None
        }

    except Exception as e:
        import traceback
        return {
            "error": f"Xử lý file thất bại: {str(e)}",
            "traceback": traceback.format_exc()
        }


# =============================================================================
# File Parsing Helpers
# =============================================================================

def parse_json_ecg(data: Any, provided_fs: Optional[int] = None) -> tuple:
    """Parse ECG from JSON with smart field detection."""
    signal = None
    sample_rate = provided_fs or TARGET_FS
    lead_names = None
    warnings = []
    metadata = {}
    
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], list):
            signal = np.array(data, dtype=np.float32)
        else:
            signal = np.array([data], dtype=np.float32)
            warnings.append("Dữ liệu 1 kênh được phát hiện")
            
    elif isinstance(data, dict):
        metadata = data
        
        # Extract signal
        for key in ['leads', 'signal', 'data', 'ecg', 'ECG']:
            if key in data:
                signal = np.array(data[key], dtype=np.float32)
                break
        
        # Try individual lead keys
        if signal is None:
            leads = []
            for name in LEAD_NAMES:
                for variant in [name, name.lower(), f"lead_{name}", f"Lead_{name}"]:
                    if variant in data:
                        leads.append(data[variant])
                        break
            if leads:
                signal = np.array(leads, dtype=np.float32)
        
        # Extract sample rate
        detected_fs = detect_sample_rate_from_metadata(data)
        if detected_fs:
            sample_rate = detected_fs
            if provided_fs and provided_fs != detected_fs:
                warnings.append(f"Fs từ metadata ({detected_fs}Hz) khác với input ({provided_fs}Hz). Sử dụng metadata.")
        elif provided_fs:
            sample_rate = provided_fs
        else:
            warnings.append("⚠️ Không tìm thấy Fs trong JSON. Sử dụng mặc định 500 Hz.")
        
        # Extract lead names
        if 'lead_names' in data:
            lead_names = data['lead_names']
    
    return signal, sample_rate, lead_names, warnings, metadata


def parse_csv_ecg(content: bytes) -> tuple:
    """Parse ECG from CSV with smart header detection."""
    warnings = []
    lead_names = None
    
    try:
        text = content.decode("utf-8")
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        
        # Detect header
        first_line = lines[0]
        is_header = False
        
        # Check if first line contains non-numeric values (likely header)
        header_parts = [h.strip() for h in first_line.split(',')]
        numeric_count = sum(1 for h in header_parts if h.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit())
        
        if numeric_count < len(header_parts) / 2:
            is_header = True
            lead_names = []
            
            # Smart lead name extraction
            for h in header_parts:
                h_upper = h.upper().strip()
                matched = False
                
                # Try exact match
                if h_upper in LEAD_ALIASES:
                    lead_names.append(LEAD_NAMES[LEAD_ALIASES[h_upper]])
                    matched = True
                elif h_upper in [l.upper() for l in LEAD_NAMES]:
                    lead_names.append(h_upper)
                    matched = True
                
                # Try partial match
                if not matched:
                    for std_name in LEAD_NAMES:
                        if std_name in h_upper or h_upper in std_name:
                            lead_names.append(std_name)
                            matched = True
                            break
                
                if not matched:
                    lead_names.append(h.strip())  # Keep original
            
            lines = lines[1:]
            warnings.append(f"Header detected: {', '.join(lead_names[:6])}{'...' if len(lead_names) > 6 else ''}")
        
        # Parse numeric data
        rows = []
        for line in lines:
            values = []
            for x in line.split(","):
                x = x.strip()
                if x:
                    try:
                        values.append(float(x))
                    except ValueError:
                        pass
            if values:
                rows.append(values)
        
        if not rows:
            return None, None, ["Không tìm thấy dữ liệu số trong CSV"]
        
        signal = np.array(rows, dtype=np.float32)
        
        # Detect orientation
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
            warnings.append("CSV format: samples × leads → đã chuyển đổi")
        
    except Exception as e:
        return None, None, [f"Lỗi parse CSV: {str(e)}"]
    
    return signal, lead_names, warnings


def parse_mat_ecg(content: bytes, provided_fs: Optional[int] = None) -> tuple:
    """Parse ECG from MATLAB .mat with metadata extraction."""
    warnings = []
    sample_rate = provided_fs or TARGET_FS
    lead_names = None
    signal = None
    metadata = {}
    
    try:
        mat_data = loadmat(io.BytesIO(content))
        metadata = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        
        # Find signal
        for key in ['val', 'signal', 'ecg', 'data', 'ECG', 'Signal', 'p_signal']:
            if key in mat_data:
                signal = np.array(mat_data[key], dtype=np.float32)
                break
        
        if signal is None:
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.size > 100:
                    signal = np.array(value, dtype=np.float32)
                    warnings.append(f"Sử dụng field '{key}' từ MAT file")
                    break
        
        if signal is None:
            return None, sample_rate, None, ["Không tìm thấy tín hiệu trong MAT file"], metadata
        
        # Extract Fs from metadata
        detected_fs = detect_sample_rate_from_metadata(mat_data)
        if detected_fs:
            sample_rate = detected_fs
            warnings.append(f"Fs từ metadata: {detected_fs} Hz")
        elif provided_fs:
            sample_rate = provided_fs
        else:
            warnings.append("⚠️ Không tìm thấy Fs trong MAT. Sử dụng mặc định 500 Hz.")
        
        # Extract lead names
        for key in ['sig_name', 'lead_names', 'leads', 'channels']:
            if key in mat_data:
                names = mat_data[key]
                if hasattr(names, 'tolist'):
                    names = names.tolist()
                if isinstance(names, list):
                    lead_names = [str(n).strip() for n in names]
                    break
                    
    except Exception as e:
        return None, sample_rate, None, [f"Lỗi parse MAT: {str(e)}"], {}
    
    return signal, sample_rate, lead_names, warnings, metadata


# =============================================================================
# Info Endpoints
# =============================================================================

@router.get("/formats")
async def get_supported_formats():
    """Return detailed format information."""
    return {
        "formats": SUPPORTED_FORMATS,
        "parameters": {
            "sample_rate": {
                "description": "Tần số lấy mẫu (Hz)",
                "required_for": [".csv", ".npy"],
                "auto_detected_from": [".mat", ".zip (WFDB)", ".json"],
                "default": 500
            },
            "duration_seconds": {
                "description": "Thời gian tín hiệu (giây) để ước lượng Fs",
                "optional": True
            }
        },
        "lead_mapping": {
            "supported_names": list(LEAD_ALIASES.keys())[:20],
            "standard_order": LEAD_NAMES,
            "auto_detect": "CSV headers, MAT field names, JSON keys"
        },
        "quality_checks": [
            "Flatline detection",
            "Noise level assessment", 
            "Clipping detection",
            "Duration validation"
        ]
    }


@router.get("/classes")
async def get_classes():
    """Return ECG classification classes."""
    from app.core.model_loader import CLASSES
    return {
        "classes": list(CLASSES) if hasattr(CLASSES, '__iter__') else ["Unknown"],
        "num_classes": len(CLASSES) if hasattr(CLASSES, '__len__') else 0
    }


try:
    from src.layers import MAMBA_SOURCE
except ImportError:
    MAMBA_SOURCE = "Unknown (Check Logs)"

@router.get("/info")
async def get_info():
    """Return API information."""
    return {
        "api_version": "2.3.0",
        "model": f"ECG-RAMBA ({MAMBA_SOURCE})",
        "description": "Zero-Shot ECG Classification via Morphology-Rhythm Disentanglement",
        "features": [
            "Auto-detect sampling rate from metadata",
            "Smart lead mapping from headers/keys",
            "Artifact rejection (flatline, noise, clipping)",
            "WFDB ZIP support",
            "5-fold ensemble inference",
            "Lead dropout accuracy warnings"
        ],
        "supported_formats": list(SUPPORTED_FORMATS.keys()),
        "preprocessing": {
            "bandpass_filter": "0.5-40 Hz (Butterworth order 4)",
            "normalization": "Instance-wise Z-score",
            "target_fs": f"{TARGET_FS} Hz",
            "target_length": f"{TARGET_LEN} samples ({TARGET_LEN/TARGET_FS}s)"
        }
    }
