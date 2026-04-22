"""
db/vector_store.py

Qdrant wrapper purpose-built for news article chunks.
Refactored from vectors.py — adds article-aware metadata payloads.
"""

import uuid
import asyncio
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter,
    FieldCondition, MatchValue,
)
from config import (
    GEMINI_API_KEY, QDRANT_API_KEY, QDRANT_CLUSTER_URL,
    QDRANT_COLLECTION, VECTOR_SIZE, EMBEDDING_MODEL,
    EMBED_DELAY_SECONDS,
)

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_CLUSTER_URL,
    api_key=QDRANT_API_KEY,
)


def init_collection():
    """Create the news_articles collection if it doesn't exist."""
    try:
        existing = [c.name for c in qdrant.get_collections().collections]

        if QDRANT_COLLECTION not in existing:
            qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[qdrant] Created collection: {QDRANT_COLLECTION}")
        else:
            print(f"[qdrant] Collection already exists: {QDRANT_COLLECTION}")
    except Exception as e:
        print(f"[qdrant] WARNING: Could not connect to Qdrant: {e}")
        print("[qdrant] WARNING: Check QDRANT_CLUSTER_URL and QDRANT_API_KEY in .env")
        print("[qdrant] WARNING: Pipeline will run but embedding storage will fail")


# ─────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Embed a text string using Gemini embedding model."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
    )
    return result["embedding"]


async def embed_text_async(text: str) -> list[float]:
    """Async wrapper for embed_text with rate limiting."""
    loop = asyncio.get_event_loop()
    vector = await loop.run_in_executor(None, embed_text, text)
    await asyncio.sleep(EMBED_DELAY_SECONDS)
    return vector


# ─────────────────────────────────────────
# STORE CHUNKS
# ─────────────────────────────────────────

async def store_chunks(
    article_id: str,
    chunks: list[str],
    metadata: dict,
) -> int:
    """
    Embed and store article chunks in Qdrant.

    Args:
        article_id: UUID of the article in SQLite
        chunks: list of text chunks from the chunker
        metadata: dict with keys: title, source, published_at, category, entities

    Returns:
        Number of chunks stored
    """
    points = []

    for i, chunk_text in enumerate(chunks):
        vector = await embed_text_async(chunk_text)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "article_id": article_id,
                "chunk_index": i,
                "chunk_text": chunk_text,
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "published_at": metadata.get("published_at", ""),
                "category": metadata.get("category", ""),
                "entities": metadata.get("entities", []),
            },
        )
        points.append(point)

    # batch upsert all chunks for this article
    if points:
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
        )

    return len(points)


# ─────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────

def search_articles(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """
    Semantic search over article chunks.

    Returns list of dicts with: chunk_text, title, source, score, article_id, etc.
    """
    query_vector = embed_text(query)

    # build optional filters
    conditions = []
    if category:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )
    if source:
        conditions.append(
            FieldCondition(key="source", match=MatchValue(value=source))
        )

    search_filter = Filter(must=conditions) if conditions else None

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "chunk_text": r.payload.get("chunk_text", ""),
            "title": r.payload.get("title", ""),
            "source": r.payload.get("source", ""),
            "published_at": r.payload.get("published_at", ""),
            "category": r.payload.get("category", ""),
            "entities": r.payload.get("entities", []),
            "article_id": r.payload.get("article_id", ""),
            "chunk_index": r.payload.get("chunk_index", 0),
            "score": r.score,
        }
        for r in results.points
    ]


def get_collection_info() -> dict:
    """Return basic stats about the collection."""
    info = qdrant.get_collection(QDRANT_COLLECTION)
    return {
        "name": QDRANT_COLLECTION,
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
        "status": info.status.value,
    }
