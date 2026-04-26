"""
db/vector_store.py

Qdrant wrapper purpose-built for news article chunks.
Refactored from vectors.py — adds article-aware metadata payloads.
"""

import uuid
import asyncio
from google import genai
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

# New Google GenAI SDK client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_CLUSTER_URL,
    api_key=QDRANT_API_KEY,
)


def init_collection():
    """Create the news_articles collection with Named Vectors for multi-faceted search."""
    try:
        existing_collections = [c.name for c in qdrant.get_collections().collections]

        needs_recreation = False
        if QDRANT_COLLECTION in existing_collections:
            # Check if existing collection has named vectors
            collection_info = qdrant.get_collection(QDRANT_COLLECTION)
            # In Qdrant, if it's a single vector, 'vectors' is a VectorParams object.
            # If it's named vectors, it's a dict of VectorParams.
            if not isinstance(collection_info.config.params.vectors, dict):
                print(f"[qdrant] Existing collection '{QDRANT_COLLECTION}' uses a single vector. Schema upgrade required.")
                needs_recreation = True
            elif "base" not in collection_info.config.params.vectors:
                print(f"[qdrant] Existing collection '{QDRANT_COLLECTION}' missing 'base' vector. Schema upgrade required.")
                needs_recreation = True

        if needs_recreation:
            print(f"[qdrant] Deleting old collection to apply new schema...")
            qdrant.delete_collection(QDRANT_COLLECTION)
            existing_collections.remove(QDRANT_COLLECTION)

        if QDRANT_COLLECTION not in existing_collections:
            # We use Named Vectors to store base content, summary, and hypothetical questions
            vectors_config = {
                "base": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                "summary": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                "question": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            }
            
            qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=vectors_config,
            )
            print(f"[qdrant] Created collection with Named Vectors: {QDRANT_COLLECTION}")
        else:
            print(f"[qdrant] Collection already exists with correct Named Vector schema.")
    except Exception as e:
        print(f"[qdrant] WARNING: Could not initialize Qdrant collection: {e}")


# ─────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Embed a text string using Gemini embedding model."""
    if not text:
        return [0.0] * VECTOR_SIZE
        
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={
            "task_type": "RETRIEVAL_DOCUMENT"
        }
    )
    return result.embeddings[0].values


async def embed_text_async(text: str) -> list[float]:
    """Async wrapper for embed_text."""
    if not text:
        return [0.0] * VECTOR_SIZE
        
    loop = asyncio.get_event_loop()
    vector = await loop.run_in_executor(None, embed_text, text)
    await asyncio.sleep(EMBED_DELAY_SECONDS)
    return vector


# ─────────────────────────────────────────
# STORE ENRICHED CHUNKS
# ─────────────────────────────────────────

async def store_enriched_chunks(
    article_id: str,
    enriched_chunks: list[dict],
) -> int:
    """
    Embed and store enriched chunks using multiple named vectors.
    """
    points = []

    for i, ec in enumerate(enriched_chunks):
        # Generate three different embeddings for the same chunk
        # 1. Base (prepended with context)
        # 2. Summary
        # 3. Hypothetical Question
        
        base_task = embed_text_async(ec["enriched_content"])
        summary_task = embed_text_async(ec["summary"])
        question_task = embed_text_async(ec["question"])
        
        base_vec, summary_vec, question_vec = await asyncio.gather(
            base_task, summary_task, question_task
        )

        metadata = ec["metadata"]
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "base": base_vec,
                "summary": summary_vec,
                "question": question_vec,
            },
            payload={
                "article_id": article_id,
                "chunk_index": i,
                "chunk_text": ec["content"],
                "parent_text": ec["parent_content"], # Full context for LLM
                "summary_text": ec["summary"],
                "question_text": ec["question"],
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "published_at": metadata.get("published_at", ""),
                "category": metadata.get("category", ""),
                "entities": metadata.get("entities", []),
                "chunk_type": "hierarchical_child"
            },
        )
        points.append(point)

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
        using="base", # Search against the content-rich base vector
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "chunk_text": r.payload.get("chunk_text", ""),
            "parent_text": r.payload.get("parent_text", ""), # This is what the LLM should see
            "summary_text": r.payload.get("summary_text", ""),
            "question_text": r.payload.get("question_text", ""),
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
