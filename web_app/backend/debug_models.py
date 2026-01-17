import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API Key not found in environment.")
    exit(1)

print(f"Using API Key: {api_key[:5]}...{api_key[-5:]}")
genai.configure(api_key=api_key)

print("\nFetching available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
    print("\n✅ Done.")
except Exception as e:
    print(f"\n❌ Error listing models: {e}")
