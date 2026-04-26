"""
pipeline/embedder.py

Orchestrates the embed + store step for processed articles.
Takes chunks and metadata, pushes them into Qdrant via vector_store.
"""

from db import vector_store, sqlite_store
from pipeline.enricher import enrich_all
from pipeline.chunker import HierarchicalChunk
import logging

logger = logging.getLogger(__name__)


async def embed_and_store(
    article_id: str,
    chunks: list[HierarchicalChunk],
    entities: list[dict],
    article: dict,
) -> int:
    """
    Enrich, embed, and store hierarchical chunks in Qdrant.
    """
    if not chunks:
        logger.warning(f"  [embedder] No chunks to embed for article {article_id}")
        return 0

    # 1. Prepare metadata
    # We pass entity names into the metadata for enrichment
    article_metadata = {
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "published_at": article.get("published_at", ""),
        "category": article.get("category", ""),
        "entities": [e["name"] for e in entities],
    }
    
    # Update chunk metadata
    for chunk in chunks:
        chunk.metadata.update(article_metadata)

    try:
        # 2. Enrichment Step (LLM-based summaries/questions)
        logger.info(f"  [embedder] Enriching {len(chunks)} chunks with LLM context...")
        enriched_chunks = await enrich_all(chunks)

        # 3. Store in Qdrant with Named Vectors
        num_stored = await vector_store.store_enriched_chunks(
            article_id=article_id,
            enriched_chunks=enriched_chunks,
        )

        # mark as processed in SQLite
        await sqlite_store.mark_processed(article_id)

        logger.info(
            f"  [embedder] Stored {num_stored} enriched chunks for "
            f"\"{article.get('title', '')[:50]}...\""
        )
        return num_stored

    except Exception as e:
        logger.error(f"  [embedder] Failed to embed article {article_id}: {e}")
        raise
