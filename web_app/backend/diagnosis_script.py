import requests
import json
import sys

def test_tutor_api():
    url = "http://localhost:8003/api/tutor/chat"
    payload = {
        "message": "Hello",
        "context": {"diagnosis": "Normal"},
        "history": []
    }
    headers = {"Content-Type": "application/json"}
    
    import time
    start = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers)
        duration = time.time() - start
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"⏱️ Request took: {duration:.2f} seconds")
        
        if response.status_code == 200:
            print("\n✅ SUCCESS: API is reachable and responding.")
        elif response.status_code == 404:
            print("\n❌ FAILURE: Route not found (404). Check router registration.")
        else:
            print(f"\n⚠️ ERROR: API returned {response.status_code}.")
            
    except Exception as e:
        print(f"\n❌ EXCEPTION: Connection failed. {e}")

if __name__ == "__main__":
    test_tutor_api()
