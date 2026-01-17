import requests
import json
import time

URL = "http://localhost:8003/api/tutor/chat"

payload = {
    "message": "What is the frequency of Alpha waves and when do they appear?",
    "context": {"diagnosis": "EEG Analysis", "hr": "N/A", "confidence": "High"},
    "history": []
}

print(f"🧠 Test Query: {payload['message']}")
start = time.time()
try:
    response = requests.post(URL, json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Reply: {data.get('reply')}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"EXCEPTION: {e}")
print(f"⏱️ Took: {time.time() - start:.2f}s")
