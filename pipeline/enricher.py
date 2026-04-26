"""
pipeline/enricher.py

Stage 3 Enrichment:
1. Context Injection: Prepend Title and Section context.
2. LLM Summary: Generate a one-sentence summary for parallel embedding.
3. Hypothetical Question: Generate a question this chunk answers.
"""

import logging
import asyncio
from typing import List, Dict
from google import genai
from config import GEMINI_API_KEY, SUMMARY_PROMPT, QUESTION_PROMPT
from pipeline.chunker import HierarchicalChunk

logger = logging.getLogger(__name__)

# Initialize client
client = genai.Client(api_key=GEMINI_API_KEY)

async def enrich_chunk(chunk: HierarchicalChunk) -> Dict:
    """
    Enrich a single chunk with context, summary, and a hypothetical question.
    """
    title = chunk.metadata.get("title", "Unknown Article")
    
    # 1. Context Injection
    # We create an enriched text for the base embedding
    # Format: Doc: [Title] | [Child Text]
    # (Parent text is stored in metadata but not necessarily prepended to the child vector 
    # to keep child vector focused, but we can prepend Title as requested)
    enriched_text = f"Doc: {title} | {chunk.child_text}"
    
    # 2 & 3. LLM Generations (Parallel)
    # We use Gemini to generate a summary and a question
    try:
        summary_task = _generate_enrichment(chunk.child_text, SUMMARY_PROMPT)
        question_task = _generate_enrichment(chunk.child_text, QUESTION_PROMPT)
        
        summary, question = await asyncio.gather(summary_task, question_task)
    except Exception as e:
        logger.error(f"[enricher] Failed to enrich chunk: {e}")
        summary = ""
        question = ""

    return {
        "content": chunk.child_text,
        "parent_content": chunk.parent_text,
        "enriched_content": enriched_text,
        "summary": summary,
        "question": question,
        "metadata": chunk.metadata
    }

async def _generate_enrichment(text: str, prompt: str) -> str:
    """Helper to call Gemini for a specific enrichment task."""
    full_prompt = f"{prompt}\n\nText: {text}"
    
    # Run in a thread if the SDK isn't natively async-friendly for this call
    # But genai.Client is modern. Let's use the standard call.
    response = client.models.generate_content(
        model="gemini-2.0-flash", # Use a fast model for enrichment
        contents=full_prompt
    )
    return response.text.strip() if response.text else ""

async def enrich_all(chunks: List[HierarchicalChunk]) -> List[Dict]:
    """Enrich a batch of chunks."""
    logger.info(f"[enricher] Enriching {len(chunks)} chunks...")
    tasks = [enrich_chunk(c) for c in chunks]
    return await asyncio.gather(*tasks)
