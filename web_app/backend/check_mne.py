try:
    import mne
    print(f"✅ MNE Available: {mne.__version__}")
except ImportError:
    print("❌ MNE Not Found")
