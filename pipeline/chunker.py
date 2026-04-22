"""
pipeline/chunker.py

Semantic chunking — splits articles based on context shifts,
not arbitrary token counts. Uses spaCy sentence vectors to detect
when the topic changes between consecutive sentences.

Algorithm:
    1. Parse into sentences via spaCy
    2. Compute 300d word vectors for each sentence
    3. Compare cosine similarity between consecutive sentences
    4. Split where similarity drops below threshold
    5. Merge very short chunks into neighbors
    6. Cap oversized chunks
"""

import numpy as np
import logging
from config import SIMILARITY_THRESHOLD, MAX_CHUNK_TOKENS, MIN_CHUNK_SENTENCES

logger = logging.getLogger(__name__)


def chunk_article(
    text: str,
    nlp,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Split article text into semantically coherent chunks.

    Args:
        text: full article text
        nlp: loaded spaCy model (en_core_web_md with word vectors)
        similarity_threshold: cosine sim below this triggers a split

    Returns:
        list of text chunks, each covering one coherent topic segment
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    sentences = [sent for sent in doc.sents if sent.text.strip()]

    if len(sentences) <= 1:
        return [text.strip()]

    # compute similarity between consecutive sentences
    similarities = _compute_similarities(sentences)

    # find split points where similarity drops below threshold
    split_indices = _find_split_points(similarities, similarity_threshold)

    # build chunks from split points
    chunks = _build_chunks(sentences, split_indices)

    # merge short chunks into neighbors
    chunks = _merge_short_chunks(chunks, MIN_CHUNK_SENTENCES)

    # cap oversized chunks
    chunks = _cap_large_chunks(chunks, MAX_CHUNK_TOKENS)

    # final cleanup
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.info(
        f"  [chunker] {len(sentences)} sentences → {len(chunks)} chunks "
        f"(threshold={similarity_threshold})"
    )

    return chunks


def _compute_similarities(sentences) -> list[float]:
    """
    Compute cosine similarity between consecutive sentence pairs.

    Uses spaCy's built-in 300d word vectors from en_core_web_md.
    Returns list of len(sentences) - 1 similarity scores.
    """
    similarities = []

    for i in range(len(sentences) - 1):
        sent_a = sentences[i]
        sent_b = sentences[i + 1]

        # spaCy .similarity() uses cosine similarity of averaged word vectors
        # returns 0-1 for en_core_web_md
        if sent_a.vector_norm and sent_b.vector_norm:
            sim = sent_a.similarity(sent_b)
        else:
            # fallback if vectors are zero (rare edge case)
            sim = 1.0  # don't split on unknown vectors

        similarities.append(sim)

    return similarities


def _find_split_points(
    similarities: list[float],
    threshold: float,
) -> list[int]:
    """
    Find indices where we should split (0-indexed into similarities list).

    A split happens after sentence[i] when similarity(sentence[i], sentence[i+1])
    drops below the threshold.
    """
    split_points = []

    for i, sim in enumerate(similarities):
        if sim < threshold:
            # split AFTER sentence i (so sentence i+1 starts a new chunk)
            split_points.append(i + 1)

    return split_points


def _build_chunks(sentences, split_indices: list[int]) -> list[str]:
    """Build text chunks from sentences and split points."""
    if not split_indices:
        # no splits — whole article is one chunk
        return [" ".join(sent.text for sent in sentences)]

    chunks = []
    prev = 0

    for split_idx in split_indices:
        chunk_sents = sentences[prev:split_idx]
        if chunk_sents:
            chunks.append(" ".join(sent.text for sent in chunk_sents))
        prev = split_idx

    # last chunk
    if prev < len(sentences):
        chunks.append(" ".join(sent.text for sent in sentences[prev:]))

    return chunks


def _merge_short_chunks(
    chunks: list[str],
    min_sentences: int = 2,
) -> list[str]:
    """
    Merge chunks that are too short (fewer than min_sentences)
    into their nearest neighbor.
    """
    if len(chunks) <= 1:
        return chunks

    merged = []

    for chunk in chunks:
        # rough sentence count: count periods/exclamation/question marks
        sentence_count = max(
            1,
            chunk.count(". ") + chunk.count("! ") + chunk.count("? ") + 1
        )

        if sentence_count < min_sentences and merged:
            # merge into previous chunk
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged


def _cap_large_chunks(chunks: list[str], max_tokens: int) -> list[str]:
    """
    Split oversized chunks at sentence boundaries.
    A 'token' here is approximately a whitespace-separated word.
    """
    result = []

    for chunk in chunks:
        words = chunk.split()
        if len(words) <= max_tokens:
            result.append(chunk)
            continue

        # split on sentence boundaries within the chunk
        # simple approach: split on ". " and rebuild
        sentences = chunk.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
        current = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > max_tokens and current:
                result.append(" ".join(current))
                current = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len

        if current:
            result.append(" ".join(current))

    return result
