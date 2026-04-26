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

import logging
import asyncio
from typing import List, NamedTuple
from google import genai
from config import (
    SIMILARITY_THRESHOLD, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, 
    CHUNK_OVERLAP_PERCENT, EMBEDDING_MODEL, EMBED_DELAY_SECONDS, GEMINI_API_KEY
)

# Initialize GenAI Client
genai_client = genai.Client(api_key=GEMINI_API_KEY)
logger = logging.getLogger(__name__)

class HierarchicalChunk(NamedTuple):
    parent_text: str
    child_text: str
    metadata: dict = {}

import numpy as np

async def _embed_for_similarity(text: str) -> List[float]:
    """Embed text using Gemini with task_type RETRIEVAL_DOCUMENT."""
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"task_type": "RETRIEVAL_DOCUMENT"}
    )
    return result.embeddings[0].values

async def _compute_similarities_gemini(sentences: List[str]) -> List[float]:
    """Compute cosine similarity between consecutive sentence pairs using Gemini."""
    if len(sentences) < 2:
        return []
    
    # 1. Embed each sentence
    # We use a slight delay for rate limiting
    embeddings = []
    for text in sentences:
        vec = await _embed_for_similarity(text)
        await asyncio.sleep(EMBED_DELAY_SECONDS)
        embeddings.append(np.array(vec))

    # 2. Compute similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        v1, v2 = embeddings[i], embeddings[i+1]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        sim = np.dot(v1, v2) / norm if norm > 0 else 0.0
        similarities.append(float(sim))
    
    return similarities

async def chunk_article(
    text: str,
    nlp,
    title: str = "Unknown Title",
) -> List[HierarchicalChunk]:
    """
    Split article into hierarchical chunks using SEMANTIC VALLEYS to find 
    optimal split points near the 512/128 word targets.
    """
    if not text or not text.strip():
        return []

    # 1. Parse sentences and compute semantic similarities for the whole doc
    doc = nlp(text)
    sentence_objs = [s for s in doc.sents if s.text.strip()]
    sentence_texts = [s.text.strip() for s in sentence_objs]
    
    if not sentence_texts:
        return []
        
    logger.info(f"  [chunker] Computing semantic map for {len(sentence_texts)} sentences...")
    similarities = await _compute_similarities_gemini(sentence_texts)

    # 2. Create Overlapping Parent Chunks (Similarity-Aware)
    parent_indices = _find_semantic_split_indices(
        items=sentence_texts,
        similarities=similarities,
        max_size=PARENT_CHUNK_SIZE,
        overlap_percent=CHUNK_OVERLAP_PERCENT
    )
    
    all_hierarchical_chunks = []

    for start_idx, end_idx in parent_indices:
        parent_text = " ".join(sentence_texts[start_idx:end_idx])
        
        # 3. Create Overlapping Child Chunks for this Parent (Similarity-Aware)
        # We use the pre-computed similarities for the sub-range
        child_indices = _find_semantic_split_indices(
            items=sentence_texts[start_idx:end_idx],
            similarities=similarities[start_idx:end_idx-1],
            max_size=CHILD_CHUNK_SIZE,
            overlap_percent=CHUNK_OVERLAP_PERCENT
        )

        for c_start, c_end in child_indices:
            # Note: c_start/c_end are relative to parent, so we map back to global
            global_start = start_idx + c_start
            global_end = start_idx + c_end
            child_text = " ".join(sentence_texts[global_start:global_end])
            
            all_hierarchical_chunks.append(
                HierarchicalChunk(
                    parent_text=parent_text,
                    child_text=child_text,
                    metadata={"title": title}
                )
            )

    logger.info(
        f"  [chunker] Created {len(all_hierarchical_chunks)} hierarchical chunks "
        f"using semantic topic-shift detection."
    )

    return all_hierarchical_chunks


def _find_semantic_split_indices(
    items: List[str],
    similarities: List[float],
    max_size: int,
    overlap_percent: float
) -> List[tuple]:
    """
    Finds optimal [start, end] sentence indices for chunks by looking for 
    the 'deepest semantic valley' (lowest similarity) near the target word count.
    """
    chunks = []
    if not items:
        return []

    overlap_size = int(max_size * overlap_percent)
    i = 0
    
    while i < len(items):
        # 1. Find the hard limit (max_size words from i)
        current_weight = 0
        limit_idx = i
        while limit_idx < len(items):
            w = len(items[limit_idx].split())
            if current_weight + w > max_size and limit_idx > i:
                break
            current_weight += w
            limit_idx += 1
        
        # 2. Search for a semantic valley in the 'split zone'
        # zone = the last 20% of the chunk
        zone_start = max(i, limit_idx - max(1, int((limit_idx - i) * 0.3)))
        
        best_split_idx = limit_idx
        if zone_start < limit_idx and zone_start < len(similarities):
            # find min similarity in the zone
            zone_similarities = similarities[zone_start : limit_idx]
            if zone_similarities:
                # relative index of the minimum similarity
                min_sim_idx = zone_similarities.index(min(zone_similarities))
                # split AFTER the sentence with min similarity (so sentence + 1)
                best_split_idx = zone_start + min_sim_idx + 1
        
        chunks.append((i, best_split_idx))
        
        if best_split_idx >= len(items):
            break

        # 3. Handle Overlap: backtrack for the next chunk's start
        backtrack_weight = 0
        next_start = best_split_idx
        for k in range(best_split_idx - 1, i, -1):
            w = len(items[k].split())
            if backtrack_weight + w > overlap_size:
                break
            backtrack_weight += w
            next_start = k
            
        if next_start <= i: # Progress check
            i = best_split_idx
        else:
            i = next_start
            
    return chunks

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
    """ * 3 # Repeat to trigger splits
    
    print("\n" + "="*60)
    print("HIERARCHICAL CHUNKS TEST")
    print("="*60)
    chunks = await chunk_article(text, nlp, title="World News & Sports")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"Parent ({len(chunk.parent_text.split())} words): {chunk.parent_text[:100]}...")
        print(f"Child ({len(chunk.child_text.split())} words): {chunk.child_text}")
        print("-" * 30)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_chunker())