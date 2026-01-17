import requests
import time
import sys

URL = "http://localhost:5173"
MAX_RETRIES = 20

print(f"🏥 Polling {URL}...")

for i in range(MAX_RETRIES):
    try:
        res = requests.get(URL, timeout=2)
        if res.status_code == 200:
            print(f"✅ Frontend READY! Status: {res.status_code}")
            sys.exit(0)
        else:
            print(f"waiting... ({res.status_code})")
    except requests.exceptions.ConnectionError:
        print(f"waiting... (Connection Refused) - Attempt {i+1}/{MAX_RETRIES}")
    except Exception as e:
        print(f"waiting... ({e})")
    
    time.sleep(2)

print("❌ Frontend failed to start (Timeout)")
sys.exit(1)
