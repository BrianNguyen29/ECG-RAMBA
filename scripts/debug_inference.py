
import sys
import os
import torch
import numpy as np
import traceback

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(">>> Starting Debug Script")

try:
    from web_app.backend.app.core.model_loader import ecg_ramba
    print(">>> Model Loader Imported")
    
    # Check injection
    import sys
    if "mamba_ssm" in sys.modules:
        print(f">>> Mamba SSM injected: {sys.modules['mamba_ssm']}")
    else:
        print(">>> Mamba SSM NOT found in sys.modules")

    # List models
    models = ecg_ramba.get_available_models()
    if not models:
        print(">>> No models found!")
        sys.exit(1)
        
    model_name = models[0]
    print(f">>> Testing with model: {model_name}")
    
    # Test 1: Load Force PyTorch
    print("\n--- Test 1: Force PyTorch Load ---")
    try:
        model = ecg_ramba.load_model(model_name, force_pytorch=True)
        print(f">>> Model Loaded: {type(model)}")
        if model is None:
            print(">>> ERROR: Model is None")
    except Exception as e:
        print(f">>> FATAL: Load failed: {e}")
        traceback.print_exc()
        
    # Test 2: Explain Prediction (Forward + Backward)
    print("\n--- Test 2: Saliency Map (Backprop) ---")
    dummy_signal = np.random.randn(12, 5000).astype(np.float32)
    try:
        saliency = ecg_ramba.explain_prediction(model_name, dummy_signal)
        print(f">>> Saliency Shape: {saliency.shape}")
        if np.all(saliency == 0):
             print(">>> WARNING: Saliency is all zeros (Check try/except block coverage)")
        else:
             print(">>> Saliency Success (Non-zero)")
    except Exception as e:
        print(f">>> FATAL: Saliency failed: {e}")
        traceback.print_exc()

    print("\n>>> Debug Script Complete")

except Exception as e:
    print(f">>> GLOBAL FAIL: {e}")
    traceback.print_exc()
