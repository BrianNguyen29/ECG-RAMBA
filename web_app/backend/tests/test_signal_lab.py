import requests
import numpy as np
import json
import time

BASE_URL = "http://localhost:8003/api/lab"

def generate_synthetic_signal(fs=250, duration=2.0):
    t = np.linspace(0, duration, int(fs*duration))
    # Signal: 10Hz "Alpha" wave
    sig = np.sin(2 * np.pi * 10 * t) 
    # Noise: 50Hz Powerline
    noise = 0.5 * np.sin(2 * np.pi * 50 * t)
    return (sig + noise).tolist()

def main():
    print("🚀 Starting Signal Lab Verification...")
    fs = 250
    signal = generate_synthetic_signal(fs)
    print(f"Generated signal: {len(signal)} samples (10Hz Signal + 50Hz Noise)")

    # 1. Test Analyze (FFT)
    print("\n🔬 Testing /analyze endpoint (FFT/PSD)...")
    try:
        res = requests.post(f"{BASE_URL}/analyze", json={"signal": signal, "fs": fs})
        res.raise_for_status()
        data = res.json()
        
        freqs = data["freqs"]
        psd = data["psd"]
        features = data["features"]
        
        # Check Peak Freq
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        print(f"✅ PSD Computed. Peak Frequency: {peak_freq:.1f}Hz (Expected ~10Hz or 50Hz depending on power)")
        print(f"   Alpha Power: {features['alpha']:.3f}")
        print(f"   Beta Power: {features['beta']:.3f} (Contains 50Hz noise tail?)")
    except Exception as e:
        print(f"❌ Analyze Failed: {e}")
        return

    # 2. Test Filter (Notch 50Hz)
    print("\n🔬 Testing /process endpoint (Notch 50Hz)...")
    try:
        payload = {
            "signal": signal,
            "fs": fs,
            "type": "notch",
            "low": 50
        }
        res = requests.post(f"{BASE_URL}/process", json=payload)
        res.raise_for_status()
        filtered_signal = res.json()["filtered_signal"]
        
        # Re-Analyze filtered signal
        res2 = requests.post(f"{BASE_URL}/analyze", json={"signal": filtered_signal, "fs": fs})
        data2 = res2.json()
        
        # Check that 50Hz component is reduced
        # We look at PSD at approx 50Hz index
        # Simple check: Alpha power ratio should increase
        old_alpha = features['alpha']
        new_alpha = data2['features']['alpha']
        
        print(f"✅ Filter Applied. Length: {len(filtered_signal)}")
        print(f"   Old Alpha Ratio: {old_alpha:.3f}")
        print(f"   New Alpha Ratio: {new_alpha:.3f} (Should be higher as noise is removed)")
        
        if new_alpha > old_alpha:
            print("🏆 Filter Successfully Enhanced Signal Quality!")
        else:
            print("⚠️ Filter might not have worked as expected.")

    except Exception as e:
        print(f"❌ Process Failed: {e}")

if __name__ == "__main__":
    main()
