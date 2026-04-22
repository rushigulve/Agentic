"""
graph/entity_resolver.py

Embedding-based entity resolution.
Merges entities that refer to the same real-world thing but have different names.

Examples:
    "Donald Trump", "Trump", "President Trump" -> "Donald Trump" (canonical)
    "India", "Republic of India"               -> "India" (canonical)

How it works:
    1. Maintain a cache of {canonical_name: embedding_vector}
    2. When a new entity appears, embed its name
    3. Compare cosine similarity against all existing entities OF THE SAME TYPE
    4. If similarity > threshold (0.85), map to the existing canonical name
    5. Otherwise, register as a new canonical entity

The canonical name is the FIRST (longest) form encountered.
"""

import json
import logging
import numpy as np
from db.vector_store import embed_text
from config import (
    ENTITY_SIMILARITY_THRESHOLD,
    ENTITY_CACHE_PATH,
    DATA_DIR,
    EMBED_DELAY_SECONDS,
)
import time

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolves entity name variants to canonical forms using embedding similarity."""

    def __init__(self):
        # {entity_type: {canonical_name: embedding_vector}}
        self._cache: dict[str, dict[str, list[float]]] = {}
        # {(original_name_lower, type): canonical_name}
        self._alias_map: dict[tuple[str, str], str] = {}
        self._load_cache()

    def _load_cache(self):
        """Load persisted entity embeddings from disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if ENTITY_CACHE_PATH.exists():
            try:
                with open(ENTITY_CACHE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = data.get("embeddings", {})
                # rebuild alias map from stored aliases
                for alias_key, canonical in data.get("aliases", {}).items():
                    # alias_key is stored as "name||type"
                    parts = alias_key.split("||")
                    if len(parts) == 2:
                        self._alias_map[(parts[0], parts[1])] = canonical
                logger.info(
                    f"[resolver] Loaded {sum(len(v) for v in self._cache.values())} "
                    f"canonical entities, {len(self._alias_map)} aliases"
                )
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"[resolver] Failed to load cache: {e}")
                self._cache = {}
                self._alias_map = {}
        else:
            logger.info("[resolver] No entity cache found, starting fresh")

    def save_cache(self):
        """Persist entity embeddings and alias map to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # convert alias map keys to strings for JSON
        aliases_json = {
            f"{name}||{etype}": canonical
            for (name, etype), canonical in self._alias_map.items()
        }

        data = {
            "embeddings": self._cache,
            "aliases": aliases_json,
        }

        with open(ENTITY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def resolve(self, name: str, entity_type: str) -> str:
        """
        Resolve an entity name to its canonical form.

        Args:
            name: raw entity name (e.g., "President Trump")
            entity_type: spaCy entity type (e.g., "PERSON")

        Returns:
            Canonical name (e.g., "Donald Trump" if already seen)
        """
        name_clean = name.strip()
        name_lower = name_clean.lower()
        key = (name_lower, entity_type)

        # fast path: exact match in alias map
        if key in self._alias_map:
            return self._alias_map[key]

        # get or create type bucket
        if entity_type not in self._cache:
            self._cache[entity_type] = {}

        type_cache = self._cache[entity_type]

        # if this exact name is already canonical, return it
        if name_clean in type_cache:
            self._alias_map[key] = name_clean
            return name_clean

        # embed the new entity name
        try:
            new_embedding = embed_text(name_clean)
            time.sleep(EMBED_DELAY_SECONDS)
        except Exception as e:
            logger.warning(f"[resolver] Failed to embed '{name_clean}': {e}")
            # fallback: no resolution, use as-is
            self._alias_map[key] = name_clean
            return name_clean

        # compare against all existing entities of the same type
        best_match = None
        best_score = 0.0

        for canonical_name, cached_embedding in type_cache.items():
            sim = self._cosine_similarity(new_embedding, cached_embedding)
            if sim > best_score:
                best_score = sim
                best_match = canonical_name

        if best_match and best_score >= ENTITY_SIMILARITY_THRESHOLD:
            # match found — map to existing canonical (first-seen wins)
            canonical = best_match
            logger.info(
                f"  [resolver] Merged '{name_clean}' -> '{best_match}' "
                f"(sim={best_score:.3f})"
            )
        else:
            # no match — register as new canonical entity
            type_cache[name_clean] = new_embedding
            canonical = name_clean

        self._alias_map[key] = canonical
        return canonical

    def resolve_entities(self, entities: list[dict]) -> list[dict]:
        """
        Resolve a list of entities, returning deduplicated list with canonical names.

        Args:
            entities: list of {"name": ..., "type": ...}

        Returns:
            Deduplicated list with canonical names
        """
        resolved = []
        seen = set()

        for entity in entities:
            canonical_name = self.resolve(
                entity["name"], entity.get("type", "")
            )
            key = (canonical_name.lower(), entity.get("type", ""))

            if key not in seen:
                seen.add(key)
                resolved.append({
                    "name": canonical_name,
                    "type": entity.get("type", ""),
                    "original_name": entity["name"],
                })

        return resolved

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(dot / norm)


# ─────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────

_resolver = None


def get_resolver() -> EntityResolver:
    """Get or create the global entity resolver singleton."""
    global _resolver
    if _resolver is None:
        _resolver = EntityResolver()
    return _resolver
