"""
pipeline/exa_fetcher.py

Fetches full-text news articles using Exa.ai (neural search for LLMs).
Replaces GNews to provide high-quality, clean article bodies without 
the need for local scraping complexity.
"""

import logging
from typing import List, Dict
from exa_py import Exa
from config import EXA_API_KEY, GNEWS_CATEGORIES, EXA_MAX_RESULTS

logger = logging.getLogger(__name__)

# Initialize Exa client
exa = Exa(api_key=EXA_API_KEY)

async def fetch_exa_news(categories: List[str] = GNEWS_CATEGORIES) -> List[Dict]:
    """
    Fetch latest news for given categories using Exa.
    Returns full-text articles cleaned for RAG.
    """
    if not EXA_API_KEY:
        logger.error("[exa] EXA_API_KEY is missing!")
        return []

    all_results = []
    
    for category in categories:
        logger.info(f"[exa] Searching for high-quality {category} news...")
        
        try:
            # Using 'search' with 'contents' as 'search_and_contents' is deprecated
            # Note: use_autoprompt is no longer a keyword argument in this SDK version
            response = exa.search(
                f"latest news and high quality articles about {category}",
                type="neural",
                num_results=EXA_MAX_RESULTS,
                contents={"text": {"max_characters": 15000}},
                category="news"
            )
            
            for result in response.results:
                all_results.append({
                    "title": result.title or "No Title",
                    "url": result.url,
                    "source": _extract_domain(result.url),
                    "published_at": result.published_date or "",
                    "content": result.text or "", # This is the FULL text!
                    "category": category,
                    "author": result.author or "Unknown"
                })
                
        except Exception as e:
            logger.error(f"[exa] Error fetching {category}: {e}")

    logger.info(f"[exa] Total articles fetched: {len(all_results)}")
    return all_results

def _extract_domain(url: str) -> str:
    """Simple helper to get domain name from URL."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain.replace("www.", "")
    except:
        return "Unknown"
