# pyright: standard
from dataclasses import dataclass

import pytest
import torch
from falkordb import FalkorDB, Graph

from config import load_settings

SETTINGS = load_settings()


@dataclass
class GraphFixture:
    graph: Graph


def load_test_graph() -> GraphFixture:
    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    graph = db.select_graph("TEST_GRAPH")
    return GraphFixture(graph)


@pytest.fixture
def single_test_graph():
    """A graph that is created and deleted for each test function. Use if you manipulate the graph."""
    graph = load_test_graph()
    yield graph
    graph.graph.delete()


def test_1(single_test_graph: GraphFixture):
    """
    This will test that falkordb similarity search is working.
    """
    # torch.manual_seed(89398)
    torch.manual_seed(8939834)
    d = 1536
    n_nodes = 10
    embeddings = torch.randn(n_nodes, d, dtype=torch.float32)

    dot_products = embeddings @ embeddings.T
    distance_matrix = 1 - dot_products

    normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    euclidean_distance_matrix = torch.cdist(
        normalized_embeddings, normalized_embeddings, p=2
    )

    def node_index_to_name(i: int) -> str:
        return f"node_{i}"

    for i, embedding in enumerate(embeddings):
        single_test_graph.graph.query(
            """
        MERGE (n:NODE {name: $name})
        WITH n, $embedding AS emb
        SET n.embedding = vecf32(emb)
        RETURN n
        """,
            # params={"name": node_index_to_name(i), "embedding": embedding.tolist()},
            params={
                "name": node_index_to_name(i),
                "embedding": (embedding / embedding.norm()).tolist(),
            },
        )

    single_test_graph.graph.query(
        f"""
    CREATE VECTOR INDEX FOR (n:NODE) ON (n.embedding) OPTIONS {{dimension:{d}, similarityFunction:'euclidean'}}
    """
    )

    for i, embedding in enumerate(embeddings):
        result = single_test_graph.graph.query(
            f"""
        MATCH (seed:NODE {{name: $name}})
        CALL db.idx.vector.queryNodes('NODE', 'embedding', {n_nodes}, seed.embedding)
        YIELD node, score
        RETURN node.name AS name, score
        ORDER BY name ASC
        """,
            params={"name": node_index_to_name(i)},
        )

        # sort by nodename
        # result.result_set.sort(key=lambda x: x[0])
        for k, (node_name, score) in enumerate(result.result_set):
            assert node_name == node_index_to_name(k), (
                f"Unexpected node name, got {node_name}, expected {node_index_to_name(k)}"
            )
            expected = euclidean_distance_matrix[i, k].item()
            assert abs(score - expected) < 1e-5, (
                f"Unexpected score for seed {i}, node {k}, got {score}, expected {expected}. The 1 - dot-product is {distance_matrix[i, k].item()}."
            )
