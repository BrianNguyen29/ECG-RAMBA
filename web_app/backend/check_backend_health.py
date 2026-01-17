import requests
import time
import sys

URL = "http://localhost:8003/health"
MAX_RETRIES = 20

print(f"🏥 Polling {URL}...")

for i in range(MAX_RETRIES):
    try:
        res = requests.get(URL, timeout=2)
        if res.status_code == 200:
            print(f"✅ Backend READY! Status: {res.status_code}")
            print(res.json())
            sys.exit(0)
        else:
            print(f"waiting... ({res.status_code})")
    except requests.exceptions.ConnectionError:
        print(f"waiting... (Connection Refused) - Attempt {i+1}/{MAX_RETRIES}")
    except Exception as e:
        print(f"waiting... ({e})")
    
    time.sleep(2)

print("❌ Backend failed to start (Timeout)")
sys.exit(1)
