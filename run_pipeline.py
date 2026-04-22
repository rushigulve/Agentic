"""
run_pipeline.py

Entry point for the news ingestion pipeline.

Usage:
    py run_pipeline.py              # single run
    py run_pipeline.py --schedule   # continuous (every 15 min)
"""

import asyncio
import argparse
import logging
import sys

# ─────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# PIPELINE ORCHESTRATION
# ─────────────────────────────────────────

async def run_once():
    """
    Execute the full pipeline once:
        fetch → dedup → extract entities → chunk → embed → graph link → store
    """
    from pipeline.fetcher import fetch_all_categories
    from pipeline.dedup import deduplicate
    from pipeline.extractor import load_nlp, extract_entities
    from pipeline.chunker import chunk_article
    from pipeline.embedder import embed_and_store
    from db import sqlite_store
    from graph.graph_store import load_graph, save_graph, get_graph_stats
    from graph.linker import link_article
    from graph.story_detector import detect_stories

    logger.info("=" * 60)
    logger.info("PIPELINE RUN STARTED")
    logger.info("=" * 60)

    # ── Step 1: Fetch ──
    logger.info("\n[FETCH] Step 1: Fetching articles...")
    raw_articles = await fetch_all_categories()
    logger.info(f"   Fetched {len(raw_articles)} articles from GNews")

    if not raw_articles:
        logger.info("   No articles fetched — skipping pipeline")
        return

    # ── Step 2: Dedup ──
    logger.info("\n[DEDUP] Step 2: Deduplicating...")
    new_articles = await deduplicate(raw_articles)
    logger.info(f"   {len(new_articles)} new article(s) after dedup")

    if not new_articles:
        logger.info("   All articles already in database — done")
        return

    # ── Step 3: Load NLP model + graph ──
    logger.info("\n[NLP] Step 3: Loading NLP model...")
    nlp = load_nlp()
    graph = load_graph()

    # ── Step 4: Process each article ──
    total_chunks = 0
    stories_linked = 0

    for i, article in enumerate(new_articles, 1):
        title = article.get("title", "")[:60]
        logger.info(f"\n[PROCESS] [{i}/{len(new_articles)}] Processing: \"{title}...\"")

        # store article metadata in SQLite
        article_id = await sqlite_store.insert_article(article)
        logger.info(f"   → Stored in SQLite: {article_id[:8]}...")

        # extract entities
        full_text = f"{article.get('title', '')} {article.get('content', '')}"
        entities = extract_entities(full_text)
        entity_names = [e["name"] for e in entities]
        logger.info(f"   → Entities: {entity_names[:5]}{'...' if len(entity_names) > 5 else ''}")

        # semantic chunking
        chunks = chunk_article(full_text, nlp)
        logger.info(f"   → Chunks: {len(chunks)}")

        # embed and store in Qdrant
        num_stored = await embed_and_store(
            article_id=article_id,
            chunks=chunks,
            entities=entities,
            article=article,
        )
        total_chunks += num_stored

        # ── Step 5: Graph linking + story detection ──
        article_metadata = {
            "title": article.get("title", ""),
            "published_at": article.get("published_at", ""),
            "source": article.get("source", ""),
            "category": article.get("category", ""),
        }
        graph = link_article(graph, article_id, article_metadata, entities)

        graph, story_id = detect_stories(
            graph,
            article_id=article_id,
            entities=entities,
            published_at=article.get("published_at", ""),
            article_title=article.get("title", ""),
            article_content=article.get("content", ""),
        )
        if story_id:
            await sqlite_store.update_story_cluster(article_id, story_id)
            stories_linked += 1
            logger.info(f"   -> Story cluster: {story_id[:8]}...")

    # ── Save graph ──
    save_graph(graph)
    graph_stats = get_graph_stats(graph)

    # ── Summary ──
    article_count = await sqlite_store.get_article_count()

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE RUN COMPLETE")
    logger.info(f"   New articles:    {len(new_articles)}")
    logger.info(f"   Chunks stored:   {total_chunks}")
    logger.info(f"   Stories linked:  {stories_linked}")
    logger.info(f"   Total articles:  {article_count}")
    logger.info(f"   Graph nodes:     {graph_stats['total_nodes']}")
    logger.info(f"   Graph edges:     {graph_stats['total_edges']}")
    logger.info("=" * 60)


# ─────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────

async def init():
    """Initialize database and vector collection."""
    from db import sqlite_store, vector_store

    logger.info("Initializing storage...")
    await sqlite_store.init_db()
    vector_store.init_collection()
    logger.info("Ready.\n")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="News Ingestion Pipeline"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run continuously on a schedule (default: single run)",
    )
    args = parser.parse_args()

    await init()

    if args.schedule:
        from pipeline.scheduler import run_scheduled
        logger.info("Starting scheduled pipeline...")
        await run_scheduled(run_once)
    else:
        await run_once()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown by user.")
        sys.exit(0)
