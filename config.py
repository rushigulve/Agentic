"""
config.py

Centralized configuration for the Agentic News Intelligence System.
All env vars, constants, and paths live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_CLUSTER_URL = os.environ.get("QDRANT_CLUSTER_URL")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY")

# ─────────────────────────────────────────
# QDRANT
# ─────────────────────────────────────────

QDRANT_COLLECTION = "news_articles"       # separate from existing 'agent_memory'
VECTOR_SIZE = 3072                         # gemini-embedding-001 output dimensions

# ─────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.5                 # cosine similarity threshold for chunk splits
MAX_CHUNK_TOKENS = 800                     # hard cap on chunk size (word count)
MIN_CHUNK_SENTENCES = 2                    # minimum sentences per chunk

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────

FETCH_INTERVAL_MINUTES = 15                # how often the scheduler runs
GNEWS_CATEGORIES = ["general", "technology"]
GNEWS_MAX_RESULTS = 10                     # articles per category per fetch
GNEWS_LANGUAGE = "en"
GNEWS_COUNTRY = "in"                       # India; change as needed

# ─────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_DB_PATH = DATA_DIR / "news.db"

# ─────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBED_DELAY_SECONDS = 0.25                 # rate limit between embed calls

# ─────────────────────────────────────────
# KNOWLEDGE GRAPH (Phase 2)
# ─────────────────────────────────────────

GRAPH_PATH = DATA_DIR / "graph.json"       # persisted NetworkX graph
STORY_TIME_WINDOW_DAYS = 7                 # articles within N days can be same story
STORY_MIN_SHARED_ENTITIES = 2              # min shared entities to consider a link
STORY_LLM_MODEL = "openai/gpt-5.4-nano"   # model for story continuation confirmation
ENTITY_SIMILARITY_THRESHOLD = 0.75         # cosine sim threshold for entity merging
ENTITY_CACHE_PATH = DATA_DIR / "entity_embeddings.json"

