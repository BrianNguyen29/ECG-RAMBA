import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

print("🔍 Diagnostics: Attempting to import main.app...")

try:
    from main import app
    print("✅ Success: 'main.app' imported successfully.")
    print(f"   Registered Routes: {len(app.routes)}")
    for route in app.routes:
        print(f"   - {route.path} ({route.name})")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    # traceback
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ General Error during startup: {e}")
    import traceback
    traceback.print_exc()
