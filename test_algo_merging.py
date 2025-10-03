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
from typing import Any, Dict, List
from falkordb.path import Path
from falkordb.node import Node


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


def test_1(shared_graph: GraphFixture):
    """In this test we will check all of the nodes in the test graph.
    Assuming then entire graph is a cluster.
    """
    graph = shared_graph.graph

    threshold = 0.6
    result = graph.query(
        f"""
        MATCH (seed:NODE)
        // jeff's graph does not have tombstone nodes
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

    seed_to_nodes: Dict[int, List[Node]] = {}
    for seed, node, _score in result.result_set:
        assert seed.id is not None, "Node ID should not be None"
        seed_to_nodes.setdefault(seed.id, [seed]).append(node)

    clusters_by_size = sorted(
        seed_to_nodes.values(), key=lambda item: len(item), reverse=True
    )
    assert len(clusters_by_size) > 0, "no clusters"
    assert len(clusters_by_size[0]) > 0
    assert len(clusters_by_size[0]) >= len(clusters_by_size[-1])



    # # target_cluster = clusters_by_size[5]
    # # for target_cluster_i in [15,25,35,45]:
    # for target_cluster_i in [0]:
    #     target_cluster = clusters_by_size[target_cluster_i]
    #     out_folder = f"./test_output_data_0.98/cluster_{target_cluster_i}"
    #     if use_jeff_graph():
    #         selected_cluster = target_cluster
    #         selected_node_ids = [node.id for node in selected_cluster if node.id is not None]
    #         assert len(selected_node_ids) > 0
    #         if should_save_debug_data():
    #             debug_path = PathLibPath(f"{out_folder}/cluster_being_tested.json")
    #             debug_path.parent.mkdir(exist_ok=True, parents=True)
    #             with open(debug_path, "w", encoding="utf-8") as f:
    #                 json.dump(selected_cluster, f, cls=GraphJSONEncoder, indent=4)
    #     else:
    #         all_nodes: List[List[Node]] = graph.query(
    #             """
    #         MATCH (n:NODE)
    #         RETURN n
    #         """
    #         ).result_set
    #         all_node_ids = {n[0].id for n in all_nodes if n[0].id is not None}
    #         all_node_ids = list(all_node_ids)

    #         # only test 100 nodes at a time
    #         selected_node_ids = all_node_ids[:100]

    #     list_of_list_of_paths = get_cluster_paths(graph, selected_node_ids)
    #     assert len(list_of_list_of_paths) > 0

    #     # Get cluster paths returns a list of list of paths
    #     # but they should only have one path in each list
    #     # TODO adjust get_cluster_paths to return a list of paths
    #     for paths in list_of_list_of_paths:
    #         assert len(paths) > 0

    #     paths = [paths[0] for paths in list_of_list_of_paths]
    #     if should_save_debug_data():
    #         paths_path = PathLibPath(f"{out_folder}/paths.txt")
    #         paths_path.parent.mkdir(exist_ok=True, parents=True)
    #         dump_paths(paths, paths_path)

    #     merge_prompt = get_prompt_for_merge_llm(paths, primary_node_ids=selected_node_ids)
    #     if should_save_debug_data():
    #         merge_prompt_path = PathLibPath(f"{out_folder}/merge_prompt.txt")
    #         merge_prompt_path.parent.mkdir(exist_ok=True, parents=True)
    #         with open(merge_prompt_path, "w") as f:
    #             f.write(merge_prompt)
    #     merge_set = merge_llm(merge_prompt)
