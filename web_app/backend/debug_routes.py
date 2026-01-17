import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from main import app

print("\n=== REGISTERED ROUTES ===")
for route in app.routes:
    methods = getattr(route, "methods", None)
    print(f"Path: {route.path} | Methods: {methods} | Name: {route.name}")
print("=========================\n")
