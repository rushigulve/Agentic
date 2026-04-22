"""
pipeline/fetcher.py

Fetches articles from the GNews API.
Returns raw article dicts ready for dedup and processing.
"""

import httpx
import asyncio
import logging
from config import GNEWS_API_KEY, GNEWS_CATEGORIES, GNEWS_MAX_RESULTS, GNEWS_LANGUAGE, GNEWS_COUNTRY

logger = logging.getLogger(__name__)

GNEWS_BASE_URL = "https://gnews.io/api/v4"
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, multiplied by attempt number


async def fetch_articles(
    category: str,
    max_results: int = GNEWS_MAX_RESULTS,
) -> list[dict]:
    """
    Fetch top headlines from GNews for a given category.

    Returns list of article dicts with keys:
        title, description, content, url, image, publishedAt, source
    """
    if not GNEWS_API_KEY:
        logger.error("GNEWS_API_KEY not set — add it to .env")
        return []

    params = {
        "category": category,
        "lang": GNEWS_LANGUAGE,
        "country": GNEWS_COUNTRY,
        "max": max_results,
        "apikey": GNEWS_API_KEY,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{GNEWS_BASE_URL}/top-headlines",
                    params=params,
                )

            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                logger.info(
                    f"[fetcher] Got {len(articles)} articles for '{category}'"
                )
                return _normalize_articles(articles, category)

            elif response.status_code == 429:
                logger.warning(
                    f"[fetcher] Rate limited (attempt {attempt}/{MAX_RETRIES})"
                )
            elif response.status_code == 403:
                logger.error("[fetcher] Invalid API key or quota exceeded")
                return []
            else:
                logger.warning(
                    f"[fetcher] HTTP {response.status_code}: {response.text[:200]}"
                )

        except httpx.TimeoutException:
            logger.warning(f"[fetcher] Timeout (attempt {attempt}/{MAX_RETRIES})")
        except httpx.RequestError as e:
            logger.warning(f"[fetcher] Request error: {e}")

        # exponential backoff before retry
        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF * attempt
            logger.info(f"[fetcher] Retrying in {wait}s...")
            await asyncio.sleep(wait)

    logger.error(f"[fetcher] Failed to fetch '{category}' after {MAX_RETRIES} attempts")
    return []


async def fetch_all_categories() -> list[dict]:
    """
    Fetch articles across all configured categories.
    Deduplicates by URL across categories.
    """
    all_articles = []
    seen_urls = set()

    for category in GNEWS_CATEGORIES:
        articles = await fetch_articles(category)

        for article in articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                all_articles.append(article)

        # small delay between category fetches to be nice to the API
        await asyncio.sleep(1)

    logger.info(f"[fetcher] Total unique articles: {len(all_articles)}")
    return all_articles


def _normalize_articles(raw_articles: list[dict], category: str) -> list[dict]:
    """
    Normalize GNews response into a consistent format.

    GNews returns:
        { title, description, content, url, image, publishedAt,
          source: { name, url } }

    We flatten source.name → source
    """
    normalized = []

    for raw in raw_articles:
        # GNews truncates content with "[...chars] chars]" — use what we get
        content = raw.get("content", "") or raw.get("description", "")

        article = {
            "title": raw.get("title", "").strip(),
            "description": raw.get("description", "").strip(),
            "content": content.strip(),
            "url": raw.get("url", ""),
            "image": raw.get("image", ""),
            "published_at": raw.get("publishedAt", ""),
            "source": raw.get("source", {}).get("name", "unknown"),
            "category": category,
        }

        # skip articles with no content
        if article["content"] and article["url"]:
            normalized.append(article)

    return normalized
