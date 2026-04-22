"""
graph/linker.py

Links articles to entities in the knowledge graph.
Called after entity extraction in the pipeline.

Creates:
    - Article nodes
    - Entity nodes (idempotent)
    - MENTIONS edges (Article -> Entity)
    - RELATED_TO edges (Entity -> Entity) for co-occurring entities
"""

import logging
from graph import graph_store

logger = logging.getLogger(__name__)


def link_article(
    graph,
    article_id: str,
    article_metadata: dict,
    entities: list[dict],
):
    """
    Link an article to its entities in the graph.

    Args:
        graph: NetworkX DiGraph
        article_id: UUID of the article
        article_metadata: dict with title, published_at, source, category
        entities: list of {"name": ..., "type": ...} dicts from extractor

    Returns:
        Updated graph
    """
    # 1. Add article node
    article_node = graph_store.add_article_node(
        graph,
        article_id=article_id,
        title=article_metadata.get("title", ""),
        published_at=article_metadata.get("published_at", ""),
        source=article_metadata.get("source", ""),
        category=article_metadata.get("category", ""),
    )

    if not entities:
        return graph

    # 2. Add entity nodes + MENTIONS edges
    entity_nodes = []

    for entity in entities:
        entity_node = graph_store.add_entity_node(
            graph,
            name=entity["name"],
            entity_type=entity.get("type", ""),
        )
        entity_nodes.append(entity_node)

        # Article -> MENTIONS -> Entity
        graph_store.add_edge(
            graph,
            from_id=article_node,
            to_id=entity_node,
            relation="MENTIONS",
        )

    # 3. Build RELATED_TO edges between co-occurring entities
    #    Two entities that appear in the same article are related
    for i in range(len(entity_nodes)):
        for j in range(i + 1, len(entity_nodes)):
            node_a = entity_nodes[i]
            node_b = entity_nodes[j]

            # check if edge already exists
            if graph.has_edge(node_a, node_b):
                # increment weight
                graph[node_a][node_b]["weight"] = (
                    graph[node_a][node_b].get("weight", 1) + 1
                )
            else:
                graph_store.add_edge(
                    graph,
                    from_id=node_a,
                    to_id=node_b,
                    relation="RELATED_TO",
                    weight=1,
                )

    entity_names = [e["name"] for e in entities]
    logger.info(
        f"  [linker] Linked article to {len(entities)} entities: "
        f"{entity_names[:4]}{'...' if len(entity_names) > 4 else ''}"
    )

    return graph
