import sys
import os
import numpy as np

# Mocking the app structure for import
sys.path.append(os.getcwd())

try:
    from web_app.backend.app.core.clinical_measurements import ClinicalMeasurer
    print("Import Successful")
    
    # Test Data (Sine wave)
    x = np.linspace(0, 10, 5000)
    sig = np.sin(x)
    # Add fake peaks
    sig[::500] += 5.0
    
    peaks = ClinicalMeasurer.detect_r_peaks(sig)
    print(f"Detected Peaks: {len(peaks)}")
    
    waves = ClinicalMeasurer.delineate_waves(sig, peaks)
    print(f"Delineated Waves: {len(waves)}")
    
    st_analysis = ClinicalMeasurer.analyze_st_segment(sig, waves)
    print(f"ST Analysis: {len(st_analysis)}")
    
    print("VERIFICATION SUCCESS: Logic runs without error.")
    
except ImportError as e:
    print(f"VERIFICATION FAILED: Import Error - {e}")
except Exception as e:
    print(f"VERIFICATION FAILED: Runtime Error - {e}")
