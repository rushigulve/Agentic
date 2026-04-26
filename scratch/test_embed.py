import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

async def test_embed():
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    # Try text-embedding-004
    try:
        print("Testing text-embedding-004...")
        res = client.models.embed_content(
            model="text-embedding-004",
            contents="Hello world",
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        print("Success with 'text-embedding-004'!")
        print(f"Dim: {len(res.embeddings[0].values)}")
    except Exception as e:
        print(f"Failed with 'text-embedding-004': {e}")

    # Try models/text-embedding-004
    try:
        print("\nTesting models/text-embedding-004...")
        res = client.models.embed_content(
            model="models/text-embedding-004",
            contents="Hello world",
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        print("Success with 'models/text-embedding-004'!")
        print(f"Dim: {len(res.embeddings[0].values)}")
    except Exception as e:
        print(f"Failed with 'models/text-embedding-004': {e}")

    # Try models/gemini-embedding-2
    try:
        print("\nTesting models/gemini-embedding-2...")
        res = client.models.embed_content(
            model="models/gemini-embedding-2",
            contents="Hello world",
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        print("Success with 'models/gemini-embedding-2'!")
        print(f"Dim: {len(res.embeddings[0].values)}")
    except Exception as e:
        print(f"Failed with 'models/gemini-embedding-2': {e}")

if __name__ == "__main__":
    asyncio.run(test_embed())
