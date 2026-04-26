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
EXA_API_KEY = os.environ.get("EXA_API_KEY")

# ─────────────────────────────────────────
# QDRANT
# ─────────────────────────────────────────

QDRANT_COLLECTION = "news_articles"       # separate from existing 'agent_memory'
VECTOR_SIZE = 3072                         # gemini-embedding-001 output dimensions

# ─────────────────────────────────────────
# CHUNKING (Hierarchical)
# ─────────────────────────────────────────

PARENT_CHUNK_SIZE = 512                    # words (context window for LLM)
CHILD_CHUNK_SIZE = 128                     # words (retrieval precision)
CHUNK_OVERLAP_PERCENT = 0.15               # 15% overlap between siblings
SIMILARITY_THRESHOLD = 0.5                 # semantic boundary threshold

# ─────────────────────────────────────────
# ENRICHMENT
# ─────────────────────────────────────────

# Prompts for semantic enrichment
SUMMARY_PROMPT = "Summarize the core factual claim of this news snippet in one concise sentence."
QUESTION_PROMPT = "What is the single most likely question a user would ask that this specific text answers perfectly?"

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────

FETCH_INTERVAL_MINUTES = 15                # how often the scheduler runs
GNEWS_CATEGORIES = ["general", "technology"]
GNEWS_MAX_RESULTS = 5                      # articles per category per fetch
EXA_MAX_RESULTS = 5                        # articles per search
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

EMBEDDING_MODEL = "models/gemini-embedding-2"
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

