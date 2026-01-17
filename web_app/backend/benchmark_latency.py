import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio
import time

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

prompt = "A" * 2000 # 2KB prompt

async def test():
    print(f"Testing Gemini 2.0 Flash Exp with 2KB prompt...")
    start = time.time()
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        resp = await model.generate_content_async(prompt)
        dur = time.time() - start
        print(f"✅ Success in {dur:.2f}s")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
