import numpy as np
from scipy.signal import find_peaks

class ClinicalMeasurer:
    @staticmethod
    def detect_r_peaks(sig, fs=500):
        """
        Simple, robust R-peak detector using adaptive thresholding.
        """
        # 1. Differentiation
        diff_sig = np.diff(sig)
        # 2. Squaring
        sq_sig = diff_sig ** 2
        # 3. Moving Window Integration (approx 150ms)
        window_size = int(0.150 * fs)
        mwa = np.convolve(sq_sig, np.ones(window_size)/window_size, mode='same')
        
        # 4. Peer Peak Finding
        peaks, _ = find_peaks(mwa, height=np.mean(mwa) * 2, distance=fs*0.4) # >200ms refractory
        return peaks

    @staticmethod
    def delineate_waves(sig, r_peaks, fs=500):
        """
        Heuristic wave delineation relative to R-peak.
        Returns dictionary of start/end indices for P, QRS, T, ST-segment.
        """
        annotations = []
        
        for r in r_peaks:
            # P-wave window: -200ms to -80ms
            p_start = max(0, r - int(0.25 * fs))
            p_end = max(0, r - int(0.10 * fs))
            
            # QRS window: -50ms to +50ms
            qrs_start = max(0, r - int(0.06 * fs))
            qrs_end = min(len(sig), r + int(0.06 * fs))
            
            # ST Segment: +60ms to +160ms
            st_start = min(len(sig), r + int(0.08 * fs))
            st_end = min(len(sig), r + int(0.16 * fs))

            annot = {
                "r_peak": int(r),
                "qrs_onset": int(qrs_start),
                "qrs_offset": int(qrs_end),
                "p_onset": int(p_start),
                "p_offset": int(p_end),
                "st_onset": int(st_start),
                "st_offset": int(st_end)
            }
            annotations.append(annot)
            
        return annotations

    @staticmethod
    def analyze_st_segment(sig, annotations):
        """
        Detects ST Elevation/Depression.
        """
        findings = []
        baseline = np.median(sig)  # Simple baseline
        
        for ann in annotations:
            # Extract ST segment
            st_seg = sig[ann['st_onset']:ann['st_offset']]
            if len(st_seg) == 0: continue
            
            # Check amplitude relative to baseline
            st_amp = np.mean(st_seg) - baseline
            
            # Threshold: 0.1mV (assuming normalized ~ 0.1)
            status = "Normal"
            if st_amp > 0.1: status = "Elevation"
            elif st_amp < -0.1: status = "Depression"
            
            findings.append({
                "index": ann['st_onset'],
                "status": status,
                "amplitude": float(st_amp)
            })
        return findings
