# pyright: standard
# TODO put this file in the appropriate location
from falkordb import FalkorDB, Graph, Node
import pytest
from dataclasses import dataclass
from semantic_compression_part_2 import (
    SETTINGS,
)
from typing import List
from pathlib import Path as PathLibPath
import json
import json
from typing import Any, Dict, List, Tuple, TypedDict
from falkordb.path import Path
from falkordb.node import Node
from os import environ


@dataclass
class GraphFixture:
    graph: Graph


def load_test_graph() -> GraphFixture:
    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    graph = db.select_graph(SETTINGS.falkordb.graph)
    return GraphFixture(graph)


@pytest.fixture(scope="module")
def shared_graph():
    """A graph that is created once per test module and shared among all test functions."""
    graph = load_test_graph()
    yield graph


def node_to_dict(n: Node) -> Dict[str, Any]:
    properties = n.properties or {}
    return {
        "id": n.id,
        "alias": n.alias,
        "labels": n.labels or [],
        "properties": {k: v for k, v in properties.items() if k != "embedding"},
    }


def edge_to_dict(e: Any) -> Dict[str, Any]:
    # Edge class is not shown; extract safely with best-effort fallbacks.
    def as_node_id(v: Any) -> Any:
        return v.id if isinstance(v, Node) else v

    return {
        "id": getattr(e, "id", None),
        "alias": getattr(e, "alias", None),
        # relation field name may vary; support both
        "relation": getattr(e, "relation", getattr(e, "reltype", None)),
        "src_node": as_node_id(getattr(e, "src_node", getattr(e, "src", None))),
        "dest_node": as_node_id(getattr(e, "dest_node", getattr(e, "dest", None))),
        "properties": getattr(e, "properties", {}) or {},
    }


def path_to_dict(p: Path) -> Dict[str, Any]:
    return {
        "nodes": [node_to_dict(n) for n in p.nodes()],
        "edges": [edge_to_dict(e) for e in p.edges()],
        "length": p.edge_count(),
    }


class GraphJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return path_to_dict(o)
        if isinstance(o, Node):
            return node_to_dict(o)
        return super().default(o)


def dump_paths(
    paths: List[Path], file_path: str | PathLibPath, *, indent: int = 2
) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, cls=GraphJSONEncoder, indent=indent, ensure_ascii=False)


@pytest.fixture
def single_test_graph():
    """A graph that is created and deleted for each test function. Use if you manipulate the graph."""
    graph = load_test_graph()
    yield graph


class ClusterOutput(TypedDict):
    seed: int
    cluster: List[Tuple[Node, float]]

def remove_exact_duplicate_clusters(clusters: List[ClusterOutput]) -> List[ClusterOutput]:
    unique_clusters = {}
    for cluster in clusters:
        node_ids = [n.id for n, _ in cluster["cluster"] if n.id is not None]
        node_ids.sort()
        node_ids_tuple = tuple(node_ids)
        if node_ids_tuple not in unique_clusters:
            unique_clusters[node_ids_tuple] = cluster
    return list(unique_clusters.values())

def test_1(shared_graph: GraphFixture):
    """In this test we will check all of the nodes in the test graph.
    Assuming then entire graph is a cluster.
    """
    graph = shared_graph.graph

    threshold = float(environ.get("MAX_THRESHOLD", 0.5))
    result = graph.query(
        f"""
        MATCH (seed:NODE)
        WHERE seed.is_tombstone = false AND seed.embedding IS NOT NULL
        WITH seed
        CALL db.idx.vector.queryNodes('NODE', 'embedding', 10, seed.embedding)
        YIELD node, score
        // note that queryNodes produces cosine distance = 1 - cosine similarity
        // so smaller distance means more similar
        // we filter to only very similar nodes
        WHERE score < {threshold} AND seed <> node
        RETURN seed, node, score
        // order by ascending so the most similar nodes are processed first
        ORDER BY score ASC
                 """
    )

    seed_to_nodes: Dict[int, ClusterOutput] = {}
    for seed, node, score in result.result_set:
        assert seed.id is not None, "Node ID should not be None"
        cluster_output = seed_to_nodes.setdefault(seed.id, {"seed": seed.id, "cluster": [(seed, 0.0)]})
        cluster_output["cluster"].append((node, score))

    clusters_by_size = sorted(
        seed_to_nodes.values(), key=lambda item: len(item["cluster"]), reverse=True
    )
    assert len(clusters_by_size) > 0, "no clusters"
    assert len(clusters_by_size[0]) > 0
    assert len(clusters_by_size[0]) >= len(clusters_by_size[-1])

    unique_clusters = remove_exact_duplicate_clusters(clusters_by_size)
    

    assert len(unique_clusters) > 0, "no unique clusters"
    all_clusters_path = f"./test_output_data_{threshold}/all_clusters.json"
    PathLibPath(all_clusters_path).parent.mkdir(exist_ok=True, parents=True)
    with open(all_clusters_path, "w") as f:
        json.dump(unique_clusters, f, cls=GraphJSONEncoder, indent=4)
