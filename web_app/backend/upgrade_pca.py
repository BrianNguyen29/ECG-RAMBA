import os
import sys
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PCA_PATH = os.path.join(MODELS_DIR, "global_pca_zeroshot.pkl")

def upgrade_pca_artifact():
    print(f"Target Artifact: {PCA_PATH}")
    
    if not os.path.exists(PCA_PATH):
        print("❌ Error: PCA file not found.")
        return

    print("Loading with warning suppression...")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            pca = joblib.load(PCA_PATH)
        print("✅ Conversion: File loaded successfully into memory.")
        
        print("Re-saving artifact with current scikit-learn version...")
        joblib.dump(pca, PCA_PATH, compress=3)
        print("✅ Success: Artifact upgraded to match current environment.")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    upgrade_pca_artifact()
