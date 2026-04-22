
import os

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter,
    SearchRequest
)
import uuid

# ─────────────────────────────────────────
# CONFIG — both are remote, nothing local
# ─────────────────────────────────────────

gemini_api_key = os.environ.get('GEMINI_API_KEY')
qdrant_api_key = os.environ.get('QDRANT_API_KEY')
qdrant_cluster_url = os.environ.get('QDRANT_CLUSTER_URL')
genai.configure(api_key=gemini_api_key)

qdrant = QdrantClient(
    url=qdrant_cluster_url,   # from qdrant cloud dashboard
    api_key=qdrant_api_key,
)

COLLECTION  = "agent_memory"
VECTOR_SIZE = 3072   # text-embedding-004 output size


# ─────────────────────────────────────────
# 1. EMBED — calls Google, returns vector
# ─────────────────────────────────────────

def embed(text: str) -> list[float]:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text
    )
    return result["embedding"]


# ─────────────────────────────────────────
# 2. SETUP COLLECTION (run once)
# ─────────────────────────────────────────

def setup_collection():
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE   # cosine similarity
            )
        )
        print(f"Created collection: {COLLECTION}")
    else:
        print(f"Collection already exists: {COLLECTION}")


# ─────────────────────────────────────────
# 3. STORE A MEMORY
#    vector goes to qdrant, text stored as payload (metadata)
# ─────────────────────────────────────────

def store_memory(text: str, metadata: dict = {}):
    vector = embed(text)   # Google computes this

    point = PointStruct(
        id=str(uuid.uuid4()),   # unique id for this memory
        vector=vector,
        payload={
            "text": text,       # store text alongside vector
            **metadata          # any extra fields you want
        }
    )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[point]
    )
    print(f"  stored: {text}")


# ─────────────────────────────────────────
# 4. SEARCH MEMORIES
#    embed the query, qdrant does similarity search remotely
# ─────────────────────────────────────────

def search_memories(query: str, top_k: int = 3) -> list[dict]:
    query_vector = embed(query)   # Google computes this

    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True   # return the text payload too
    )

    return [
        {
            "text":  r.payload["text"],
            "score": r.score,          # cosine similarity, higher = better
            "id":    r.id,
            "meta":  {k: v for k, v in r.payload.items() if k != "text"}
        }
        for r in results.points
    ]


# ─────────────────────────────────────────
# 5. DELETE A MEMORY (by id)
# ─────────────────────────────────────────

def delete_memory(point_id: str):
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=[point_id]
    )


# ─────────────────────────────────────────
# 6. TRY IT
# ─────────────────────────────────────────

if __name__ == "__main__":
    setup_collection()

    # store memories with metadata tags
    memories = [
        ("the user prefers dark mode",                  {"type": "preference"}),
        ("the user lives in Pune India",                {"type": "fact"}),
        ("Python is the user's favourite language",     {"type": "preference"}),
        ("the user is learning about LLM agents",       {"type": "fact"}),
        ("the user's name is Rushikesh",                {"type": "fact"}),
        ("the user does not like verbose explanations", {"type": "preference"}),
        ("the meeting is scheduled for 3pm tomorrow",   {"type": "event"}),
        ("the user wants to build something like OpenClaw", {"type": "goal"}),
    ]

    print("Storing memories in Qdrant Cloud...")
    # for text, meta in memories:
    #     store_memory(text, meta)

    # search
    print("\n--- search results ---")
    queries = [
        "what UI theme does the user like?",
        "where is the user located?",
        "what is the user currently working on?",
        "how should I respond to this user?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = search_memories(query, top_k=2)
        for r in results:
            print(f"  [score {r['score']:.4f}]  {r['text']}  ({r['meta'].get('type','')})")


## Key difference from sqlite-vec
"""
With sqlite-vec, distance was smaller = better (L2 distance). With Qdrant using cosine similarity, score is higher = better, ranging from 0 to 1.

Qdrant cosine score:
  0.9 - 1.0   almost identical meaning
  0.7 - 0.9   strongly related
  0.5 - 0.7   loosely related
  below 0.5   probably not relevant

"""