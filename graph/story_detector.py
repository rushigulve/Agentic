"""
graph/story_detector.py

Detects whether a new article continues an existing story.

Algorithm (hybrid rule-based + LLM):
    1. Get the new article's entity set
    2. Find existing articles sharing >= N entities within a time window
    3. Rank candidates by shared entity count
    4. Ask LLM to confirm: "Is article B a continuation of article A?"
    5. If yes: create CONTINUES edge, assign same story_cluster
    6. If no existing match: create a new story cluster
"""

import uuid
import logging
from datetime import datetime, timedelta
from openai import OpenAI
from config import (
    OPENROUTER_API_KEY, STORY_LLM_MODEL,
    STORY_TIME_WINDOW_DAYS, STORY_MIN_SHARED_ENTITIES,
)
from graph import graph_store

logger = logging.getLogger(__name__)

# OpenRouter client for LLM confirmation
_client = None


def _get_client() -> OpenAI:
    """Lazy-init OpenRouter client."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def detect_stories(
    graph,
    article_id: str,
    entities: list[dict],
    published_at: str,
    article_title: str = "",
    article_content: str = "",
):
    """
    Detect whether this article is part of an existing story.

    Args:
        graph: NetworkX DiGraph
        article_id: UUID of the new article
        entities: list of {"name": ..., "type": ...} from extractor
        published_at: ISO datetime string
        article_title: title for LLM context
        article_content: content snippet for LLM context

    Returns:
        Tuple of (updated graph, story_cluster_id or None)
    """
    article_node = graph_store._article_id(article_id)
    if article_node not in graph:
        logger.warning(f"  [story] Article {article_id} not in graph, skipping")
        return graph, None

    entity_names = {e["name"].lower().strip() for e in entities}

    if len(entity_names) < STORY_MIN_SHARED_ENTITIES:
        logger.info("  [story] Too few entities for story detection")
        return graph, None

    # Step 1: Find candidate articles
    candidates = _find_candidates(graph, entity_names, published_at, article_node)

    if not candidates:
        logger.info("  [story] No candidate articles found for story linking")
        return graph, None

    logger.info(f"  [story] Found {len(candidates)} candidate(s) for story linking")

    # Step 2: Try LLM confirmation on top candidates (max 3)
    for candidate in candidates[:3]:
        cand_node = candidate["node_id"]
        cand_data = candidate["data"]

        is_continuation = _confirm_continuation(
            article_a_title=cand_data.get("title", ""),
            article_b_title=article_title,
            article_b_content=article_content[:500],
            shared_entities=candidate["shared_entities"],
        )

        if is_continuation:
            # Create CONTINUES edge
            graph_store.add_edge(
                graph,
                from_id=article_node,
                to_id=cand_node,
                relation="CONTINUES",
            )

            # Assign story cluster
            story_id = _assign_story_cluster(
                graph, article_node, cand_node, article_title
            )

            logger.info(
                f"  [story] Article continues story: "
                f"\"{cand_data.get('title', '')[:40]}...\" "
                f"(cluster: {story_id[:8]}...)"
            )
            return graph, story_id

    logger.info("  [story] No story continuation confirmed by LLM")
    return graph, None


def _find_candidates(
    graph,
    entity_names: set[str],
    published_at: str,
    exclude_node: str,
) -> list[dict]:
    """
    Find existing articles that share enough entities
    and are within the time window.

    Returns candidates sorted by shared_entity_count (descending).
    """
    # parse the new article's date
    try:
        new_date = datetime.fromisoformat(
            published_at.replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        new_date = datetime.utcnow()

    time_cutoff = new_date - timedelta(days=STORY_TIME_WINDOW_DAYS)

    candidates = []

    # iterate over all article nodes in the graph
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "article":
            continue
        if node_id == exclude_node:
            continue

        # check time window
        try:
            article_date = datetime.fromisoformat(
                data.get("published_at", "").replace("Z", "+00:00")
            )
            if article_date < time_cutoff:
                continue
        except (ValueError, TypeError):
            continue

        # count shared entities
        article_entities = set()
        for _, target, edge_data in graph.out_edges(node_id, data=True):
            if edge_data.get("relation") == "MENTIONS":
                # entity node IDs are "entity:<lowercase_name>"
                ent_name = graph.nodes[target].get("name", "").lower().strip()
                article_entities.add(ent_name)

        shared = entity_names & article_entities
        if len(shared) >= STORY_MIN_SHARED_ENTITIES:
            candidates.append({
                "node_id": node_id,
                "data": data,
                "shared_entities": list(shared),
                "shared_count": len(shared),
            })

    # sort by shared count, most shared first
    candidates.sort(key=lambda c: c["shared_count"], reverse=True)
    return candidates


def _confirm_continuation(
    article_a_title: str,
    article_b_title: str,
    article_b_content: str = "",
    shared_entities: list[str] | None = None,
) -> bool:
    """
    Use LLM to confirm whether article B continues the story of article A.

    Returns True if the LLM says yes.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("  [story] No OPENROUTER_API_KEY, skipping LLM confirmation")
        # fallback: if entities match, assume continuation
        return True

    shared_str = ", ".join(shared_entities or [])

    prompt = (
        f"You are a news analyst. Determine if Article B is a continuation "
        f"or follow-up of the same news story as Article A.\n\n"
        f"Article A: \"{article_a_title}\"\n"
        f"Article B: \"{article_b_title}\"\n"
    )
    if article_b_content:
        prompt += f"Article B excerpt: \"{article_b_content[:300]}\"\n"
    if shared_str:
        prompt += f"Shared entities: {shared_str}\n"

    prompt += (
        "\nIs Article B a continuation or update of the same story as Article A? "
        "Answer ONLY 'yes' or 'no'."
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=STORY_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().lower()
        is_yes = answer.startswith("yes")
        logger.info(f"  [story] LLM says: {answer} -> {'match' if is_yes else 'no match'}")
        return is_yes

    except Exception as e:
        logger.warning(f"  [story] LLM call failed: {e}")
        # fallback: assume continuation if entities match
        return True


def _assign_story_cluster(
    graph,
    article_node: str,
    related_node: str,
    article_title: str = "",
) -> str:
    """
    Assign a story cluster. Either reuse the existing article's cluster
    or create a new one.

    Returns the story_cluster_id.
    """
    # check if the related article already belongs to a story
    existing_stories = graph_store.get_neighbors(
        graph, related_node, relation="PART_OF"
    )

    if existing_stories:
        # reuse existing story
        story_data = existing_stories[0]
        story_node = story_data["_node_id"]
        story_id = story_data.get("story_id", "")
    else:
        # create new story cluster
        story_id = str(uuid.uuid4())
        story_node = graph_store.add_story_node(
            graph,
            story_id=story_id,
            name=article_title[:100],
            summary="",
        )
        # link the related (older) article to the story too
        graph_store.add_edge(
            graph,
            from_id=related_node,
            to_id=story_node,
            relation="PART_OF",
        )

    # link the new article to the story
    graph_store.add_edge(
        graph,
        from_id=article_node,
        to_id=story_node,
        relation="PART_OF",
    )

    return story_id
