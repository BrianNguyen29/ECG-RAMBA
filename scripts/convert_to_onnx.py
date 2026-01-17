"""
Script to convert trained PyTorch models to ONNX format.
optimized for CPU inference with ONNX Runtime.

Usage:
    python scripts/convert_to_onnx.py
"""

import os
import sys
import glob
import torch
import numpy as np

# Add project root needed for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

try:
    from web_app.backend.mambapy.mamba2 import Mamba2Simple # Pure PyTorch injection
    import types
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Inject Mamba immediately BEFORE importing src.model
try:
    import mamba_ssm
except ImportError:
    print("[SETUP] Injecting Pure PyTorch Mamba2...")
    mamba_mock = types.ModuleType("mamba_ssm")
    mamba_mock.Mamba2 = Mamba2Simple
    mamba_mock.Mamba = Mamba2Simple
    sys.modules["mamba_ssm"] = mamba_mock

try:
    from configs.config import CONFIG, DEVICE
    from src.model import ECGRambaV7Advanced
except ImportError as e:
    print(f"Project Import Error: {e}")
    sys.exit(1)

def convert_model(model_path, output_path):
    print(f"\n[Stats] Converting {os.path.basename(model_path)}...")
    
    # 1. Load PyTorch Model
    try:
        model = ECGRambaV7Advanced(cfg=CONFIG)
        checkpoint = torch.load(model_path, map_location="cpu")
        
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
    except Exception as e:
        print(f"[ERROR] Loading model failed: {e}")
        return False

    # 2. Example Inputs (Batch Size = 1)
    # Shapes based on layers.py inputs: x (B, 12, L), xh (B, 3072), xhr (B, 36)
    # Assuming L=2500 (slice length) from loader
    dummy_x = torch.randn(1, 12, 2500, dtype=torch.float32)
    dummy_xh = torch.randn(1, 3072, dtype=torch.float32)
    dummy_xhr = torch.randn(1, 36, dtype=torch.float32)

    # 3. Export to ONNX (Standard Torch Trace)
    try:
        # Note: Mamba's Selective Scan is complex for ONNX. 
        # We assume the 'Mamba2Simple' injection uses standard PyTorch ops (Conv1d, Linear, SiLU).
        # This allows accurate tracing without custom CUDA ops.
        
        torch.onnx.export(
            model,
            (dummy_x, dummy_xh, dummy_xhr),
            output_path,
            export_params=True,
            opset_version=18, # Use 18 (Native PyTorch 2.x export) to avoid version converter bugs
            do_constant_folding=True,
            input_names=['x', 'xh', 'xhr'],
            output_names=['logits', 'logits_morph', 'logits_rhythm'], 
            # dynamic_axes is deprecated for dynamo-based export (Opset 18+)
            # dynamic_shapes argument is preferred if available, but standard export uses dynamic_axes usually.
            # However, since we saw the warning, let's try the modern input format if using torch.onnx.dynamo_export?
            # Actually, standard torch.onnx.export might not accept 'dynamic_shapes' in all versions.
            # But let's try obeying the warning directly.
            # Reverting to Opset 14 might be safer if PyTorch is older, but user wants Opset 18.
            
            # ATTEMPT: Use dynamic_axes but clean up specific axes
            # dynamic_axes={
            #     'x': {0: 'batch_size'},
            #     'xh': {0: 'batch_size'},
            #     'xhr': {0: 'batch_size'},
            #     'logits': {0: 'batch_size'}
            # }
            # STATIC EXPORT ATTEMPT: Remove dynamic axes to avoid Dynamo constraints issues.
            # This yields a model fixed to Batch=1, Length=2500.

        )
        print(f"[OK] Exported to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] ONNX Export failed for {model_path}:")
        print(str(e))
        import traceback
        traceback.print_exc()
        return False

def verify_onnx(model_pt_path, model_onnx_path):
    print(f"[Verify] Checking equivalence...")
    import onnxruntime as ort
    
    # PT Inference
    model_pt = ECGRambaV7Advanced(cfg=CONFIG)
    checkpoint = torch.load(model_pt_path, map_location="cpu")
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_pt.load_state_dict(state_dict, strict=False)
    model_pt.eval()
    
    dummy_x = torch.randn(1, 12, 2500)
    dummy_xh = torch.randn(1, 3072)
    dummy_xhr = torch.randn(1, 36)
    
    with torch.no_grad():
        pt_out = model_pt(dummy_x, dummy_xh, dummy_xhr)

    # ONNX Inference
    ort_session = ort.InferenceSession(model_onnx_path)
    onnx_inputs = {
        'x': dummy_x.numpy(),
        'xh': dummy_xh.numpy(),
        'xhr': dummy_xhr.numpy()
    }
    onnx_out = ort_session.run(None, onnx_inputs)
    
    # Compare (logits is first output)
    diff = np.max(np.abs(pt_out.numpy() - onnx_out[0]))
    print(f"Max Absolute Difference: {diff}")
    
    if diff < 1e-4:
        print("[PASS] Models match!")
        return True
    else:
        print("[FAIL] Mismatch detected.")
        return False

def main():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    pt_files = glob.glob(os.path.join(models_dir, "*best.pt"))
    
    if not pt_files:
        print("No .pt files found!")
        return

    success_count = 0
    for pt_file in pt_files:
        base_name = os.path.basename(pt_file).replace(".pt", ".onnx")
        onnx_file = os.path.join(models_dir, base_name)
        
        if convert_model(pt_file, onnx_file):
            # basic verification on first one
            if success_count == 0: 
                verify_onnx(pt_file, onnx_file)
            success_count += 1
            
    print(f"\nDone. Converted {success_count}/{len(pt_files)} models.")

if __name__ == "__main__":
    main()
