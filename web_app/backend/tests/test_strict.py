import sys
import os
import numpy as np

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from app.core.model_loader import ECGRambaInference

def test_sqi_rejection():
    print("\n[TEST] SQI Rejection (Flatline)")
    inference = ECGRambaInference()
    
    # Create flatline signal (12 leads, 5000 samples)
    # Lead 0 is flat
    flat_signal = np.random.randn(12, 5000) * 0.1
    flat_signal[0, :] = 0.0 # Perfectly flat
    
    # Bypass load_model check by mocking it? 
    # Or rely on _check_signal_quality being called very early
    
    # We can test the private method directly
    is_valid = inference._check_signal_quality(flat_signal)
    if not is_valid:
        print("✅ SQI Correctly REJECTED flatline signal.")
    else:
        print("❌ SQI FAILED to reject flatline.")

def test_pmp_logic():
    print("\n[TEST] Power Mean Pooling (Q=3)")
    inference = ECGRambaInference()
    
    # Mock logits for 3 slices
    # Slice 1: High conf for Class 0
    # Slice 2: Low conf
    # Slice 3: Low conf
    # PMP should boost Class 0 higher than arithmetic mean
    
    logits_stack = np.array([
        [5.0, -5.0], # Prob ~0.99, 0.01
        [-2.0, -2.0], # Prob ~0.12, 0.12
        [-2.0, -2.0]
    ])
    
    pooled = inference._power_mean_pooling(logits_stack, Q=3.0)
    print(f"Pooled Probs: {pooled}")
    
    # Expected: High boost for index 0
    if pooled[0] > 0.8:
        print("✅ PMP Correctly preserved high confidence evidence.")
    else:
        print(f"❌ PMP Result too low: {pooled[0]}")

if __name__ == "__main__":
    print("--- STRICT PROTOCOL VERIFICATION ---")
    try:
        test_sqi_rejection()
        test_pmp_logic()
    except Exception as e:
        print(f"Test crashed: {e}")
