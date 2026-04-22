"""
pipeline/extractor.py

Entity extraction using spaCy (en_core_web_md).
Extracts named entities (people, orgs, locations, events) from article text.
Also provides the shared spaCy NLP model for the chunker.
"""

import logging

logger = logging.getLogger(__name__)

# singleton — loaded once, shared across extractor and chunker
_nlp = None


def load_nlp():
    """
    Load the spaCy model (cached singleton).

    Uses en_core_web_md which includes 300d word vectors
    needed for semantic similarity in the chunker.
    """
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_md")
            logger.info("[extractor] Loaded spaCy en_core_web_md model")
        except OSError:
            logger.error(
                "[extractor] spaCy model not found. Run:\n"
                "  py -m spacy download en_core_web_md"
            )
            raise
    return _nlp


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from text.

    Returns deduplicated list of:
        [{"name": "India", "type": "GPE"}, {"name": "Modi", "type": "PERSON"}, ...]

    Entity types extracted:
        PERSON, ORG, GPE (countries/cities), LOC, EVENT,
        NORP (nationalities/groups), FAC (facilities)
    """
    nlp = load_nlp()
    doc = nlp(text)

    # entity types we care about
    relevant_types = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "NORP", "FAC"}

    seen = set()
    entities = []

    for ent in doc.ents:
        if ent.label_ not in relevant_types:
            continue

        # normalize: strip whitespace, title case for consistency
        name = ent.text.strip()
        key = (name.lower(), ent.label_)

        if key not in seen:
            seen.add(key)
            entities.append({
                "name": name,
                "type": ent.label_,
            })

    return entities


def extract_entity_names(text: str) -> list[str]:
    """Convenience: just the entity name strings."""
    return [e["name"] for e in extract_entities(text)]


# ─────────────────────────────────────────
# CATEGORY INFERENCE
# ─────────────────────────────────────────

# keyword → category mapping for fallback categorization
_CATEGORY_KEYWORDS = {
    "technology": ["ai", "tech", "software", "startup", "app", "robot", "cyber",
                    "algorithm", "data", "cloud", "chip", "semiconductor", "coding"],
    "politics": ["election", "minister", "parliament", "congress", "vote",
                 "president", "government", "legislation", "policy", "democrat",
                 "republican", "senator"],
    "business": ["market", "stock", "economy", "gdp", "inflation", "revenue",
                 "company", "trade", "investment", "profit", "earnings"],
    "sports": ["match", "tournament", "cricket", "football", "player", "score",
               "championship", "league", "medal", "olympics"],
    "science": ["research", "study", "discovery", "nasa", "space", "climate",
                "species", "experiment", "physics", "biology"],
    "health": ["health", "medical", "disease", "vaccine", "hospital", "patient",
               "drug", "treatment", "mental health", "who"],
}


def extract_category(text: str, title: str, default: str = "general") -> str:
    """
    Simple keyword-based category inference.
    Used as fallback if the source doesn't provide a category.
    """
    combined = f"{title} {text}".lower()

    scores = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)

    return default
