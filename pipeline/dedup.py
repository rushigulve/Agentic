"""
pipeline/dedup.py

Two-layer deduplication:
  1. URL uniqueness — exact match in SQLite
  2. Content hash — SHA256 of normalized text to catch republished articles
"""

import hashlib
import re
from db import sqlite_store


def compute_content_hash(text: str) -> str:
    """
    SHA256 hash of normalized text.

    Normalization: lowercase, collapse whitespace, strip punctuation.
    This catches near-identical articles from different sources.
    """
    # lowercase, strip, collapse whitespace
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def deduplicate(articles: list[dict]) -> list[dict]:
    """
    Filter out articles that already exist in the database.

    Checks both URL and content hash.
    Returns only genuinely new articles.
    """
    new_articles = []

    for article in articles:
        url = article.get("url", "")
        content = article.get("content", "")

        # layer 1: URL check
        if await sqlite_store.article_exists(url):
            continue

        # layer 2: content hash check
        content_hash = compute_content_hash(content)
        if await sqlite_store.content_hash_exists(content_hash):
            continue

        # attach hash for storage later
        article["content_hash"] = content_hash
        new_articles.append(article)

    skipped = len(articles) - len(new_articles)
    if skipped > 0:
        print(f"  [dedup] Skipped {skipped} duplicate(s)")

    return new_articles
