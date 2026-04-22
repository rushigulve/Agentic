"""
graph/graph_store.py

NetworkX-based knowledge graph with JSON persistence.
Stores articles, entities, and stories as nodes with typed edges.

Node ID conventions:
    article:<uuid>      — article nodes
    entity:<name>       — entity nodes (lowercase, for dedup)
    story:<uuid>        — story cluster nodes

Edge attributes always include 'relation' key:
    MENTIONS, PART_OF, CONTINUES, UPDATES, RELATED_TO
"""

import json
import logging
from datetime import datetime
import networkx as nx
from networkx.readwrite import json_graph
from config import GRAPH_PATH, DATA_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────

def load_graph() -> nx.DiGraph:
    """Load graph from JSON file, or create a new empty one."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if GRAPH_PATH.exists():
        try:
            with open(GRAPH_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            graph = json_graph.node_link_graph(data, directed=True)
            logger.info(
                f"[graph] Loaded graph: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges"
            )
            return graph
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[graph] Failed to load graph, creating new: {e}")

    graph = nx.DiGraph()
    logger.info("[graph] Created new empty graph")
    return graph


def save_graph(graph: nx.DiGraph):
    """Persist graph to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = json_graph.node_link_data(graph)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.debug(
        f"[graph] Saved: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )


# ─────────────────────────────────────────
# NODE OPERATIONS
# ─────────────────────────────────────────

def _article_id(article_id: str) -> str:
    """Canonical node ID for an article."""
    return f"article:{article_id}"


def _entity_id(name: str) -> str:
    """Canonical node ID for an entity (lowercased for dedup)."""
    return f"entity:{name.lower().strip()}"


def _story_id(story_id: str) -> str:
    """Canonical node ID for a story cluster."""
    return f"story:{story_id}"


def add_article_node(
    graph: nx.DiGraph,
    article_id: str,
    title: str = "",
    published_at: str = "",
    source: str = "",
    category: str = "",
) -> str:
    """Add an article node. Returns the canonical node ID."""
    node_id = _article_id(article_id)
    graph.add_node(
        node_id,
        node_type="article",
        article_id=article_id,
        title=title,
        published_at=published_at,
        source=source,
        category=category,
    )
    return node_id


def add_entity_node(
    graph: nx.DiGraph,
    name: str,
    entity_type: str = "",
) -> str:
    """Add an entity node (idempotent — won't overwrite existing)."""
    node_id = _entity_id(name)
    if node_id not in graph:
        graph.add_node(
            node_id,
            node_type="entity",
            name=name.strip(),
            entity_type=entity_type,
            mention_count=0,
        )
    # always increment mention count
    graph.nodes[node_id]["mention_count"] = (
        graph.nodes[node_id].get("mention_count", 0) + 1
    )
    return node_id


def add_story_node(
    graph: nx.DiGraph,
    story_id: str,
    name: str = "",
    summary: str = "",
) -> str:
    """Add a story cluster node."""
    node_id = _story_id(story_id)
    graph.add_node(
        node_id,
        node_type="story",
        story_id=story_id,
        name=name,
        summary=summary,
        created_at=datetime.utcnow().isoformat(),
    )
    return node_id


def get_node(graph: nx.DiGraph, node_id: str) -> dict | None:
    """Get node attributes by ID. Returns None if not found."""
    if node_id in graph:
        return dict(graph.nodes[node_id])
    return None


# ─────────────────────────────────────────
# EDGE OPERATIONS
# ─────────────────────────────────────────

def add_edge(
    graph: nx.DiGraph,
    from_id: str,
    to_id: str,
    relation: str,
    **attrs,
):
    """
    Add a typed edge between two nodes.

    relation: MENTIONS, PART_OF, CONTINUES, UPDATES, RELATED_TO
    """
    graph.add_edge(from_id, to_id, relation=relation, **attrs)


def get_edges(
    graph: nx.DiGraph,
    node_id: str,
    relation: str | None = None,
    direction: str = "out",
) -> list[tuple[str, str, dict]]:
    """
    Get edges for a node, optionally filtered by relation type.

    direction: 'out' (outgoing), 'in' (incoming), 'both'
    """
    edges = []

    if direction in ("out", "both"):
        for _, target, data in graph.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                edges.append((node_id, target, data))

    if direction in ("in", "both"):
        for source, _, data in graph.in_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                edges.append((source, node_id, data))

    return edges


def get_neighbors(
    graph: nx.DiGraph,
    node_id: str,
    relation: str | None = None,
) -> list[dict]:
    """
    Get neighboring nodes, optionally filtered by edge relation type.
    Returns list of node attribute dicts.
    """
    neighbors = []

    for _, target, data in graph.out_edges(node_id, data=True):
        if relation is None or data.get("relation") == relation:
            node_data = dict(graph.nodes[target])
            node_data["_node_id"] = target
            neighbors.append(node_data)

    return neighbors


# ─────────────────────────────────────────
# QUERY FUNCTIONS
# ─────────────────────────────────────────

def get_story_chain(graph: nx.DiGraph, story_id: str) -> list[dict]:
    """
    Get all articles in a story, sorted chronologically.

    Args:
        story_id: UUID of the story cluster (without 'story:' prefix)

    Returns:
        List of article dicts sorted by published_at
    """
    story_node = _story_id(story_id)
    if story_node not in graph:
        return []

    articles = []
    # find all articles that are PART_OF this story
    for source, _, data in graph.in_edges(story_node, data=True):
        if data.get("relation") == "PART_OF":
            node_data = dict(graph.nodes[source])
            node_data["_node_id"] = source
            articles.append(node_data)

    # sort by published_at
    articles.sort(key=lambda a: a.get("published_at", ""))
    return articles


def get_related_articles(graph: nx.DiGraph, article_id: str) -> list[dict]:
    """
    Get articles that share entities with this article.

    Returns articles ranked by number of shared entities (descending).
    """
    article_node = _article_id(article_id)
    if article_node not in graph:
        return []

    # find all entities this article MENTIONS
    my_entities = set()
    for _, target, data in graph.out_edges(article_node, data=True):
        if data.get("relation") == "MENTIONS":
            my_entities.add(target)

    # find other articles that mention the same entities
    article_scores = {}  # article_node_id -> shared_entity_count

    for entity_node in my_entities:
        # find articles that also MENTION this entity (incoming edges)
        for source, _, data in graph.in_edges(entity_node, data=True):
            if (
                data.get("relation") == "MENTIONS"
                and source != article_node
                and graph.nodes[source].get("node_type") == "article"
            ):
                article_scores[source] = article_scores.get(source, 0) + 1

    # build result sorted by score
    results = []
    for node_id, score in sorted(
        article_scores.items(), key=lambda x: x[1], reverse=True
    ):
        node_data = dict(graph.nodes[node_id])
        node_data["_node_id"] = node_id
        node_data["shared_entity_count"] = score
        results.append(node_data)

    return results


def get_entity_timeline(graph: nx.DiGraph, entity_name: str) -> list[dict]:
    """
    Get all articles mentioning an entity, sorted by date.

    Args:
        entity_name: Entity name (case-insensitive)
    """
    entity_node = _entity_id(entity_name)
    if entity_node not in graph:
        return []

    articles = []
    for source, _, data in graph.in_edges(entity_node, data=True):
        if data.get("relation") == "MENTIONS":
            node_data = dict(graph.nodes[source])
            node_data["_node_id"] = source
            articles.append(node_data)

    articles.sort(key=lambda a: a.get("published_at", ""))
    return articles


# ─────────────────────────────────────────
# STATS
# ─────────────────────────────────────────

def get_graph_stats(graph: nx.DiGraph) -> dict:
    """Return counts by node type and edge relation."""
    node_counts = {}
    for _, data in graph.nodes(data=True):
        ntype = data.get("node_type", "unknown")
        node_counts[ntype] = node_counts.get(ntype, 0) + 1

    edge_counts = {}
    for _, _, data in graph.edges(data=True):
        rel = data.get("relation", "unknown")
        edge_counts[rel] = edge_counts.get(rel, 0) + 1

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "nodes_by_type": node_counts,
        "edges_by_relation": edge_counts,
    }
