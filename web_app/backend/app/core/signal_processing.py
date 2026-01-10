import numpy as np
import scipy.signal as signal

def denoise_signal(data: np.ndarray, fs: int = 500) -> np.ndarray:
    """
    Apply bandpass filter to remove noise from ECG signal.
    """
    lowcut = 0.5
    highcut = 50.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(1, [low, high], btype='band')
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

def normalize_signal(data: np.ndarray) -> np.ndarray:
    """
    Normalize signal to [0, 1] range.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def process_ecg(data: list) -> np.ndarray:
    """
    Main processing pipeline.
    """
    np_data = np.array(data)
    # Simple check for dimensionality, assume 1D or (Leads, Time)
    # For this MVP, let's assume single lead or we flatten it for simple processing
    # In a real app, strict shape checking is needed.
    
    clean_data = denoise_signal(np_data)
    norm_data = normalize_signal(clean_data)
    
    return norm_data
