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
import asyncio
import time
import google.generativeai as genai
from config import SIMILARITY_THRESHOLD, MAX_CHUNK_TOKENS, MIN_CHUNK_SENTENCES, EMBEDDING_MODEL, EMBED_DELAY_SECONDS

logger = logging.getLogger(__name__)


def _embed_for_similarity(text: str) -> list[float]:
    """
    Embed text using Gemini with task_type RETRIEVAL_DOCUMENT.

    RETRIEVAL_DOCUMENT tells Gemini to produce embeddings optimized
    for similarity search — the model's self-attention layers focus
    on capturing the semantic "topic" of the passage rather than
    generating a general-purpose embedding.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
    )
    return result["embedding"]


async def chunk_article(
    text: str,
    nlp,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Split article text into semantically coherent chunks using
    Gemini embeddings (self-attention, RETRIEVAL task type) on
    stop-word-filtered text.
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    sentences = [sent for sent in doc.sents if sent.text.strip()]

    if len(sentences) <= 1:
        return [text.strip()]

    # compute similarity between consecutive sentences using Gemini RETRIEVAL embeddings
    similarities = await _compute_similarities_gemini(sentences)

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


async def _compute_similarities_gemini(sentences) -> list[float]:
    """
    Compute cosine similarity between consecutive sentence pairs
    using Gemini embeddings (task_type=RETRIEVAL_DOCUMENT) on text
    with stop words removed.
    """
    # 1. Prepare cleaned text for each sentence
    cleaned_texts = []
    for sent in sentences:
        meaningful_tokens = [
            t.text for t in sent
            if not t.is_stop and not t.is_punct and t.text.strip()
        ]
        clean_text = " ".join(meaningful_tokens)
        cleaned_texts.append(clean_text if clean_text else sent.text.strip())

    # 2. Embed each sentence via Gemini with RETRIEVAL_DOCUMENT task type
    #    Run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    embeddings = []
    for text in cleaned_texts:
        vec = await loop.run_in_executor(None, _embed_for_similarity, text)
        await asyncio.sleep(EMBED_DELAY_SECONDS)
        embeddings.append(vec)

    # 3. Compute cosine similarity between consecutive pairs
    similarities = []
    for i in range(len(embeddings) - 1):
        v1 = np.array(embeddings[i])
        v2 = np.array(embeddings[i + 1])

        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm > 0:
            sim = np.dot(v1, v2) / norm
        else:
            sim = 0.0

        similarities.append(float(sim))

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

async def test_chunker():
    # load spacy 
    import spacy
    nlp = spacy.load("en_core_web_md")
    text = """
    Russian President Vladimir Putin on Friday warned the West against sending troops to Ukraine, calling it a dangerous escalation that could trigger a wider conflict. 
    Speaking at a news conference in Moscow, Putin said such a move would be perceived as direct intervention and would have devastating consequences. 
    Meanwhile, Ukraine’s foreign minister urged NATO allies to reconsider their “hesitation” and provide Kyiv with the weapons it needs to defend itself. 
    SRH won the IPL match 54 against CSK by 16 runs at Chepauk. 
    The beach is full of turtles due to breeding season.
    """
    
    doc = nlp(text)
    sentences = [s for s in doc.sents if s.text.strip()]
    
    print("="*60)
    print("SENTENCE SIMILARITIES (GEMINI RETRIEVAL + SELF-ATTENTION + NO STOP WORDS)")
    print("="*60)
    
    # Use the same logic as the chunker for similarity
    similarities = await _compute_similarities_gemini(sentences)
    
    for i, sim in enumerate(similarities):
        s1 = sentences[i].text.strip().replace("\n", " ")
        s2 = sentences[i+1].text.strip().replace("\n", " ")
        print(f"S{i+1} vs S{i+2}: {sim:.4f}")
        print(f"  S{i+1}: {s1[:60]}...")
        print(f"  S{i+2}: {s2[:60]}...")
        print("-" * 30)

    print("\n" + "="*60)
    print("FINAL CHUNKS")
    print("="*60)
    chunks = await chunk_article(text, nlp)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk.split())} words\n{chunk.strip()}\n")

if __name__ == "__main__":
    asyncio.run(test_chunker())