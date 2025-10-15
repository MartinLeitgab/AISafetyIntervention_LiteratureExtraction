"""
Create edges between nodes based on embedding similarity.
"""
# pyright: standard

from falkordb import Graph
import fire
from config import load_settings
SETTINGS = load_settings()


def cosine_similarity_to_euclidean_distance(cosine_similarity: float) -> float:
    """
    Convert cosine similarity to euclidean distance for normalized vectors.
    """
    if cosine_similarity < -1.0 or cosine_similarity > 1.0:
        raise ValueError("Cosine similarity must be in the range [-1, 1]")
    return (2 - 2 * cosine_similarity) ** 0.5

def create_edges(graph: Graph, edge_label: str, min_cosine_similarity: float, max_cosine_similarity: float = 1.0):
    """
    Create edges between nodes based on embedding similarity.
    """

    # is flipped on purpose because queryNodes returns cosine distance = 1 - cosine similarity
    max_euclidean_distance = cosine_similarity_to_euclidean_distance(min_cosine_similarity)
    min_euclidean_distance = cosine_similarity_to_euclidean_distance(max_cosine_similarity)

    query = f"""
    MATCH (seed:NODE)
    WHERE seed.is_tombstone = false AND seed.embedding IS NOT NULL
    WITH seed
    CALL db.idx.vector.queryNodes('NODE', 'embedding', 100, seed.embedding)
    YIELD node, score
    // note that queryNodes produces euclidean_distance = sqrt(2 - 2 * cosine_similarity)
    // The id(seed) < id(node) condition is to avoid duplicate edges
    WHERE node.is_tombstone = false AND score >= {min_euclidean_distance} AND score <= {max_euclidean_distance} AND seed.url <> node.url AND id(seed) < id(node)
    MERGE (seed)-[r:{edge_label}]-(node)
    SET r.score = score
    RETURN seed.id AS seed_id, collect({{node_id: node.id, score: score}}) AS similar_nodes
    """
    return graph.query(query)

if __name__ == "__main__":
    fire.Fire(create_edges)