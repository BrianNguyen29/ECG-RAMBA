import numpy as np
from scipy import signal
from typing import Dict, List, Any

class DSPService:
    @staticmethod
    def compute_psd(data: np.ndarray, fs: float) -> Dict[str, Any]:
        """
        Compute Power Spectral Density using Welch's Method.
        Returns freqs and power arrays.
        """
        try:
            # Check dimensions (handle 1D or 2D [channels, samples])
            if data.ndim > 1:
                # Average across channels for global spectrum or take first
                # For simplicity in Phase 1 Lab, we analyze the first channel (or average)
                data_1d = np.mean(data, axis=0)
            else:
                data_1d = data

            freqs, psd = signal.welch(data_1d, fs, nperseg=min(len(data_1d), fs*2))
            
            # Extract basic band powers (relative)
            # Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30)
            bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30)
            }
            
            powers = {}
            total_power = np.sum(psd)
            if total_power == 0: total_power = 1.0

            for band, (f_min, f_max) in bands.items():
                idx = np.logical_and(freqs >= f_min, freqs <= f_max)
                power = np.sum(psd[idx])
                powers[band] = float(power / total_power)

            return {
                "freqs": freqs.tolist(),
                "psd": psd.tolist(),
                "features": powers
            }
        except Exception as e:
            print(f"DSP Error (PSD): {e}")
            return {"error": str(e)}

    @staticmethod
    def apply_filter(data: np.ndarray, fs: float, filter_type: str, 
                    low: float = None, high: float = None) -> np.ndarray:
        """
        Apply Butterworth filter (Bandpass, Lowpass, Highpass).
        """
        try:
            nyq = 0.5 * fs
            order = 4
            
            if filter_type == "bandpass":
                low = low / nyq
                high = high / nyq
                b, a = signal.butter(order, [low, high], btype='band')
            elif filter_type == "lowpass":
                high = high / nyq
                b, a = signal.butter(order, high, btype='low')
            elif filter_type == "highpass":
                low = low / nyq
                b, a = signal.butter(order, low, btype='high')
            elif filter_type == "notch":
                # Notch at specific freq (e.g. 50Hz)
                f0 = low  # Frequency to remove
                Q = 30.0  # Quality factor
                b, a = signal.iirnotch(f0, Q, fs)
            else:
                return data

            # Handle 2D data
            if data.ndim == 2:
                return signal.filtfilt(b, a, data, axis=1)
            else:
                return signal.filtfilt(b, a, data)
        except Exception as e:
            print(f"DSP Error (Filter): {e}")
            return data
