import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

async def list_models():
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    print("Available Embedding Models:")
    print("-" * 30)
    for model in client.models.list():
        if "embedContent" in model.supported_actions:
            print(model)
            print("-" * 30)

if __name__ == "__main__":
    asyncio.run(list_models())
