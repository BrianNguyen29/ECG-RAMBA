import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

candidates = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-thinking-exp-1219"
]

print(f"Testing candidates with key: {api_key[:5]}...\n")

async def test():
    for name in candidates:
        print(f"Testing {name}...", end=" ", flush=True)
        try:
            model = genai.GenerativeModel(name)
            # Use async to be fast
            resp = await model.generate_content_async("Hi")
            print(f"✅ WORKS! (Response: {resp.text.strip()})")
            return name
        except Exception as e:
            if "404" in str(e):
                print("❌ 404 Not Found")
            else:
                print(f"❌ Error: {e}")
    return None

if __name__ == "__main__":
    result = asyncio.run(test())
    if result:
        print(f"\n🏆 WINNER: {result}")
    else:
        print("\n💀 NO WORKING FLASH MODELS FOUND.")
