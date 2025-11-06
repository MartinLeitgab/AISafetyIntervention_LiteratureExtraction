#!/usr/bin/env python3
"""
Script for creating edges between nodes based on embedding similarity.

This script finds similar nodes in the graph using vector similarity search
and creates edges between them. It converts cosine similarity to euclidean
distance for compatibility with FalkorDB's vector index.

Usage:
    uv run python -m intervention_graph_creation.src.local_graph_extraction.db.create_similarity_edges
"""

import logging

from falkordb import FalkorDB, Graph

from config import load_settings

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

# Label for the edges to create (e.g., "SIMILAR_TO", "RELATED_TO")
EDGE_LABEL = "SIMILAR_TO"

# Minimum cosine similarity threshold for creating edges (range: -1.0 to 1.0)
MIN_COSINE_SIMILARITY = 0.8

# Maximum cosine similarity threshold (range: -1.0 to 1.0)
MAX_COSINE_SIMILARITY = 1.0

# Number of similar nodes to retrieve per query
TOP_K = 100

# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SETTINGS = load_settings()


def cosine_similarity_to_euclidean_distance(cosine_similarity: float) -> float:
    """
    Convert cosine similarity to euclidean distance for normalized vectors.

    For normalized vectors, the relationship is:
    euclidean_distance = sqrt(2 - 2 * cosine_similarity)

    Args:
        cosine_similarity: Cosine similarity value in range [-1, 1]

    Returns:
        Euclidean distance corresponding to the cosine similarity

    Raises:
        ValueError: If cosine similarity is not in the range [-1, 1]
    """
    if cosine_similarity < -1.0 or cosine_similarity > 1.0:
        raise ValueError("Cosine similarity must be in the range [-1, 1]")
    return (2 - 2 * cosine_similarity) ** 0.5


def create_edges(
    graph: Graph,
    edge_label: str,
    min_cosine_similarity: float,
    max_cosine_similarity: float = 1.0,
    top_k: int = 100,
) -> dict:
    """
    Create edges between nodes based on embedding similarity.

    This function:
    1. Finds all nodes with embeddings
    2. For each node, queries similar nodes using vector similarity search
    3. Creates edges between similar nodes if they meet the similarity threshold
    4. Stores the euclidean distance score on each edge

    Args:
        graph: FalkorDB graph instance
        edge_label: Label for the edges to create (e.g., "SIMILAR_TO", "RELATED_TO")
        min_cosine_similarity: Minimum cosine similarity threshold for creating edges
        max_cosine_similarity: Maximum cosine similarity threshold (default: 1.0)
        top_k: Number of similar nodes to retrieve per query (default: 100)

    Returns:
        Query result containing seed nodes and their similar nodes
    """
    logger.info(
        f"Creating '{edge_label}' edges for similarity range [{min_cosine_similarity}, {max_cosine_similarity}]"
    )

    # Convert cosine similarity thresholds to euclidean distances
    # Note: the mapping is inverted (higher similarity = lower distance)
    max_euclidean_distance = cosine_similarity_to_euclidean_distance(
        min_cosine_similarity
    )
    min_euclidean_distance = cosine_similarity_to_euclidean_distance(
        max_cosine_similarity
    )

    logger.info(
        f"Euclidean distance range: [{min_euclidean_distance:.4f}, {max_euclidean_distance:.4f}]"
    )

    # Cypher query to find similar nodes and create edges
    query = f"""
    MATCH (seed:NODE)
    WHERE seed.is_tombstone = false AND seed.embedding IS NOT NULL
    WITH seed
    CALL db.idx.vector.queryNodes('NODE', 'embedding', {top_k}, seed.embedding)
    YIELD node, score
    // score is euclidean_distance = sqrt(2 - 2 * cosine_similarity)
    WHERE node.is_tombstone = false
      AND score >= {min_euclidean_distance}
      AND score <= {max_euclidean_distance}
      AND seed.url <> node.url
    MERGE (seed)-[r:{edge_label}]-(node)
    SET r.score = score
    RETURN seed.id AS seed_id,
           seed.name AS seed_name,
           count(node) AS edges_created,
           collect({{node_id: node.id, node_name: node.name, score: score}}) AS similar_nodes
    """

    logger.info("Executing similarity search and edge creation query...")
    result = graph.query(query)

    return result


def get_edge_stats(graph: Graph, edge_label: str) -> dict:
    """
    Get statistics about the created edges.

    Args:
        graph: FalkorDB graph instance
        edge_label: Label of the edges to analyze

    Returns:
        Dictionary with edge statistics
    """
    stats_query = f"""
    MATCH ()-[r:{edge_label}]-()
    RETURN count(r) AS total_edges,
           min(r.score) AS min_score,
           max(r.score) AS max_score,
           avg(r.score) AS avg_score
    """

    result = graph.query(stats_query)

    if result.result_set:
        row = result.result_set[0]
        return {
            "total_edges": row[0],
            "min_score": row[1],
            "max_score": row[2],
            "avg_score": row[3],
        }
    return {}


def main():
    """Main function to create similarity edges."""

    # Validate similarity range
    if not (-1.0 <= MIN_COSINE_SIMILARITY <= 1.0):
        raise ValueError("MIN_COSINE_SIMILARITY must be in range [-1, 1]")
    if not (-1.0 <= MAX_COSINE_SIMILARITY <= 1.0):
        raise ValueError("MAX_COSINE_SIMILARITY must be in range [-1, 1]")
    if MIN_COSINE_SIMILARITY >= MAX_COSINE_SIMILARITY:
        raise ValueError(
            "MIN_COSINE_SIMILARITY must be less than MAX_COSINE_SIMILARITY"
        )

    logger.info("=" * 70)
    logger.info("Creating Similarity-Based Edges")
    logger.info("=" * 70)
    logger.info(f"Database: {SETTINGS.falkordb.host}:{SETTINGS.falkordb.port}")
    logger.info(f"Graph: {SETTINGS.falkordb.graph}")
    logger.info(f"Edge label: {EDGE_LABEL}")
    logger.info(
        f"Cosine similarity range: [{MIN_COSINE_SIMILARITY}, {MAX_COSINE_SIMILARITY}]"
    )
    logger.info(f"Top K: {TOP_K}")
    logger.info("=" * 70)

    # Connect to database
    try:
        db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
        graph = db.select_graph(SETTINGS.falkordb.graph)
        logger.info("Successfully connected to FalkorDB")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return

    # Create edges
    try:
        result = create_edges(
            graph=graph,
            edge_label=EDGE_LABEL,
            min_cosine_similarity=MIN_COSINE_SIMILARITY,
            max_cosine_similarity=MAX_COSINE_SIMILARITY,
            top_k=TOP_K,
        )

        # Log results
        logger.info("=" * 70)
        logger.info("Edge Creation Results")
        logger.info("=" * 70)

        total_seeds = 0
        total_edges = 0

        if result.result_set:
            for row in result.result_set:
                seed_id = row[0]
                seed_name = row[1]
                edges_created = row[2]
                total_seeds += 1
                total_edges += edges_created

                if edges_created > 0:
                    logger.info(
                        f"Node '{seed_name}' ({seed_id}): {edges_created} edges created"
                    )

        logger.info("=" * 70)
        logger.info(f"Total seed nodes processed: {total_seeds}")
        logger.info(f"Total edges created: {total_edges}")

        # Get and display edge statistics
        stats = get_edge_stats(graph, EDGE_LABEL)
        if stats:
            logger.info("=" * 70)
            logger.info("Edge Statistics")
            logger.info("=" * 70)
            logger.info(f"Total '{EDGE_LABEL}' edges: {stats.get('total_edges', 0)}")
            logger.info(
                f"Min score (euclidean distance): {stats.get('min_score', 0):.4f}"
            )
            logger.info(
                f"Max score (euclidean distance): {stats.get('max_score', 0):.4f}"
            )
            logger.info(
                f"Avg score (euclidean distance): {stats.get('avg_score', 0):.4f}"
            )

        logger.info("=" * 70)
        logger.info("✓ Successfully completed edge creation")

    except Exception as e:
        logger.error(f"Failed to create edges: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
