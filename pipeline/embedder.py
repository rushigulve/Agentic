"""
pipeline/embedder.py

Orchestrates the embed + store step for processed articles.
Takes chunks and metadata, pushes them into Qdrant via vector_store.
"""

import logging
from db import vector_store, sqlite_store

logger = logging.getLogger(__name__)


async def embed_and_store(
    article_id: str,
    chunks: list[str],
    entities: list[dict],
    article: dict,
) -> int:
    """
    Embed article chunks and store them in Qdrant.
    Marks the article as processed in SQLite after success.

    Args:
        article_id: UUID from SQLite
        chunks: list of text chunks from chunker
        entities: list of entity dicts from extractor
        article: original article dict with metadata

    Returns:
        Number of chunks stored
    """
    if not chunks:
        logger.warning(f"  [embedder] No chunks to embed for article {article_id}")
        return 0

    # prepare metadata payload for Qdrant
    metadata = {
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "published_at": article.get("published_at", ""),
        "category": article.get("category", ""),
        "entities": [e["name"] for e in entities],  # store just names in payload
    }

    try:
        num_stored = await vector_store.store_chunks(
            article_id=article_id,
            chunks=chunks,
            metadata=metadata,
        )

        # mark as processed in SQLite
        await sqlite_store.mark_processed(article_id)

        logger.info(
            f"  [embedder] Stored {num_stored} chunks for "
            f"\"{article.get('title', '')[:50]}...\""
        )
        return num_stored

    except Exception as e:
        logger.error(f"  [embedder] Failed to embed article {article_id}: {e}")
        raise
