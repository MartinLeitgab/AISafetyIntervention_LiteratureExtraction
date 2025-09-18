# pyright: standard
# TODO put this file in the appropriate location
from falkordb import FalkorDB, Graph, Node, QueryResult
import pytest
from dataclasses import dataclass
from upload_test_graph_to_falkordb import upload_test_graph_to_falkordb
from semantic_compression_part_2 import (
    get_cluster_paths,
    SETTINGS,
    get_prompt_for_merge_llm,
    merge_llm,
)
from typing import List
from os import environ
from pathlib import Path as PathLibPath
import json
import json
from typing import Any, Dict, List, Tuple
from falkordb.path import Path
from falkordb.node import Node


@dataclass
class GraphFixture:
    graph: Graph

def should_save_debug_data() -> bool:
    return environ.get("SAVE_DEBUG_DATA") == "1"


def load_test_graph() -> GraphFixture:
    """
    Load a test graph from CSV files using the bulk loader. Then make some more nodes for corner cases.
    """
    graph_name = "TEST_merg_llm"
    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    graph = db.select_graph(graph_name)
    try:
        graph.delete()
    except:
        pass

    server_url = f"redis://{SETTINGS.falkordb.host}:{SETTINGS.falkordb.port}"
    upload_test_graph_to_falkordb(db, server_url, graph_name)
    return GraphFixture(graph)


@pytest.fixture(scope="module")
def shared_graph():
    """A graph that is created once per test module and shared among all test functions."""
    graph = load_test_graph()
    try:
        yield graph
    finally:
        # run with `SAVE_DEBUG_DATA=1 uv run pytest $ThisFile` to keep the graph
        #     To find all test nodes look use
        # MATCH p=(:TEST)-[:EDGE*0..50]-() RETURN p
        if not should_save_debug_data():
            graph.graph.delete()



def node_to_dict(n: Node) -> Dict[str, Any]:
    properties = n.properties or {}
    return {
        "id": n.id,
        "alias": n.alias,
        "labels": n.labels or [],
        "properties": {k: v for k, v in properties.items() if k != 'embedding'},
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

def dump_paths(paths: List[Path], file_path: str | PathLibPath, *, indent: int = 2) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, cls=GraphJSONEncoder, indent=indent, ensure_ascii=False)

@pytest.fixture
def single_test_graph():
    """A graph that is created and deleted for each test function. Use if you manipulate the graph."""
    graph = load_test_graph()
    try:
        yield graph
    finally:
        # run with `SAVE_DEBUG_DATA=1 uv run pytest $ThisFile` to keep the graph
        #     To find all test nodes look use
        # MATCH p=(:TEST)-[:EDGE*0..50]-() RETURN p
        if not should_save_debug_data():
            graph.graph.delete()

class Clusters:
    clusters: Dict[int, List[Node]] = {}
    node_id_to_cluster_id: Dict[int, int] = {}
    def __init__(self, max_cluster_size: int = 10) -> None:
        self.clusters = {}
        self.node_id_to_cluster_id = {}
        self.max_cluster_size = max_cluster_size

    def get_cluster_id(self, node: Node) -> int | None:
        return self.node_id_to_cluster_id.get(node.id if node.id is not None else -1, None)

    def add_node(self, cluster_id: int, node: Node) -> None:
        """Add a node to a cluster. If the cluster does not exist, create it.
        If the cluster is full, create a new cluster with the node's ID as the cluster ID.
        """
        assert self.get_cluster_id(node) is None, "Node is already in a cluster"
        if len(self.clusters.get(cluster_id, [])) >= self.max_cluster_size:
            cluster_id = node.id if node.id is not None else -1
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []
        self.clusters[cluster_id].append(node)
        if node.id is not None:
            self.node_id_to_cluster_id[node.id] = cluster_id

    def merge_clusters(self, cluster_id1: int, cluster_id2: int) -> None:
        """Merge two clusters. The nodes from cluster_id2 are added to cluster_id1.
        If the resulting cluster is too large, split the cluster into multiple clusters.
        With the last in first out order.
        """
        if cluster_id1 == cluster_id2:
            return
        if cluster_id1 not in self.clusters or cluster_id2 not in self.clusters:
            return
        nodes_to_move = self.clusters[cluster_id2]

        if len(self.clusters[cluster_id1]) + len(nodes_to_move) <= self.max_cluster_size:
            # simple case, just add all nodes to cluster_id1
            self.clusters[cluster_id1].extend(nodes_to_move)
            for node in nodes_to_move:
                if node.id is not None:
                    self.node_id_to_cluster_id[node.id] = cluster_id1
            del self.clusters[cluster_id2]
            return
        # complex case, need to split the cluster
        while len(self.clusters[cluster_id1]) < self.max_cluster_size:
            node = nodes_to_move.pop(0)
            self.clusters[cluster_id1].append(node)
            if node.id is not None:
                self.node_id_to_cluster_id[node.id] = cluster_id1
            continue
        if len(nodes_to_move) == 0:
            del self.clusters[cluster_id2]
            return
        new_cluster_id = nodes_to_move[0].id if nodes_to_move[0].id is not None else -1
        self.clusters[new_cluster_id] = nodes_to_move
        del self.clusters[cluster_id2]
        for node in nodes_to_move:
            if node.id is not None:
                self.node_id_to_cluster_id[node.id] = new_cluster_id

    def cluster_results(self, results: QueryResult) -> List[List[Node]]:
        """
        greedily build clusters of up to 10 nodes
        works semi-well because we order the result set by
        score ascending so prioritize more similar nodes first
        """
        for seed, node, score in results.result_set:
            assert isinstance(seed, Node)
            assert isinstance(node, Node)
            assert isinstance(score, float)
            if seed.id is None or node.id is None:
                assert False, "Node ID should not be None"
                continue
            if (cluster_id := self.get_cluster_id(seed)) is not None:
                if (node_cluster_id := self.get_cluster_id(node)) is not None:
                    # both seed and node are already in clusters
                    self.merge_clusters(cluster_id, node_cluster_id)
                    continue
                self.add_node(cluster_id, node)
                continue
            # seed not in a cluster
            if (cluster_id := self.get_cluster_id(node)) is not None:
                self.add_node(cluster_id, seed)
                continue
            # neither seed nor node are in a cluster
            cluster_id = seed.id
            self.add_node(cluster_id, seed)
            self.add_node(cluster_id, node)
            continue
        return list(self.clusters.values())


def test_1(shared_graph: GraphFixture):
    """In this test we will check all of the nodes in the test graph.
    Assuming then entire graph is a cluster.
    """
    graph = shared_graph.graph

    result = graph.query(
         """
        MATCH (seed:NODE)
        WHERE seed.is_tombstone = false AND seed.embedding IS NOT NULL
        WITH seed
        CALL db.idx.vector.queryNodes('NODE', 'embedding', 10, seed.embedding)
        YIELD node, score
        // note that queryNodes produces cosine distance = 1 - cosine similarity
        // so smaller distance means more similar
        // we filter to only very similar nodes
        // we also filter to id(seed) < id(node) to avoid duplicate pairs and self-matches
        WHERE score < 0.4 AND id(seed) < id(node)
        RETURN seed, node, score
        ORDER BY score ASC
                 """)
    clusters = (Clusters()).cluster_results(result)

    if should_save_debug_data():
        debug_path = PathLibPath("./test_output_data/vector_query_result.json")
        debug_path.parent.mkdir(exist_ok=True, parents=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(clusters, f, cls=GraphJSONEncoder, indent=4)

    all_nodes: List[List[Node]] = graph.query(
        """
    MATCH (n:NODE)
    RETURN n
    """
    ).result_set
    all_node_ids = {n[0].id for n in all_nodes if n[0].id is not None}
    all_node_ids = list(all_node_ids)

    # only test 100 nodes at a time
    selected_node_ids = all_node_ids[:100]

    list_of_list_of_paths = get_cluster_paths(graph, selected_node_ids)
    assert len(list_of_list_of_paths) > 0
    

    # Get cluster paths returns a list of list of paths
    # but they should only have one path in each list
    # TODO adjust get_cluster_paths to return a list of paths
    for paths in list_of_list_of_paths:
        assert len(paths) > 0

    paths = [paths[0] for paths in list_of_list_of_paths]
    if should_save_debug_data():
        paths_path = PathLibPath("./test_output_data/paths.txt")
        paths_path.parent.mkdir(exist_ok=True, parents=True)
        dump_paths(paths, paths_path)

    merge_prompt = get_prompt_for_merge_llm(paths, primary_node_ids=selected_node_ids)
    if should_save_debug_data():
        merge_prompt_path = PathLibPath("./test_output_data/merge_prompt.txt")
        merge_prompt_path.parent.mkdir(exist_ok=True, parents=True)
        with open(merge_prompt_path, "w") as f:
            f.write(merge_prompt)
    merge_set = merge_llm(merge_prompt)
    # TODO actual tests to see if merge set is correct
    # for now, just pass the test, but save the merge set
    if should_save_debug_data():
        merge_set_path = PathLibPath("test_output_data/merge_set.txt")
        with open(merge_set_path, "w") as f:
            json.dump([m.to_dict() for m in merge_set], f, indent=4)