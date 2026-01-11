import sys
import os
import numpy as np
import torch

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from app.core.model_loader import ECGRambaInference

def test_saliency_generation():
    print("\n[TEST] Saliency Map Generation (Vanilla Gradient)")
    
    # Init inference
    inference = ECGRambaInference()
    
    # Create valid signal (random noise but shaped correctly)
    signal = np.random.randn(12, 5000).astype(np.float32)
    
    # Pick a model (first available)
    models = inference.get_available_models()
    if not models:
        print("❌ No models found. Cannot test saliency.")
        return

    model_name = models[0]
    print(f"Using model: {model_name}")
    
    # Generate Saliency
    saliency = inference.explain_prediction(model_name, signal)
    
    # Checks
    print(f"Shape: {saliency.shape}")
    print(f"Range: [{np.min(saliency):.4f}, {np.max(saliency):.4f}]")
    print(f"Mean: {np.mean(saliency):.4f}")
    
    if saliency.shape != (12, 5000):
        print("❌ Shape mismatch! Expected (12, 5000)")
        return
        
    if np.max(saliency) > 1.0 or np.min(saliency) < 0.0:
        print("❌ Normalization failed!")
        return
        
    if np.sum(saliency) == 0:
        print("❌ Saliency map is all zeros! Gradient did not flow.")
        return
        
    print("✅ Saliency Map Generated Successfully.")

if __name__ == "__main__":
    try:
        test_saliency_generation()
    except Exception as e:
        print(f"Test Crashed: {e}")
        import traceback
        traceback.print_exc()
