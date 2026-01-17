import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print(f"Checking models for key: {api_key[:5]}...")

try:
    models = list(genai.list_models())
    print(f"Found {len(models)} models.")
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"AVAILABLE: {m.name}")
except Exception as e:
    print(f"ERROR: {e}")
