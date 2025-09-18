# pyright: standard
# TODO put this file in the appropriate location
from falkordb import FalkorDB, Graph, Node
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
from pathlib import Path
import json


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


def test_1(shared_graph: GraphFixture):
    """In this test we will check all of the nodes in the test graph.
    Assuming then entire graph is a cluster.
    """
    graph = shared_graph.graph

    all_nodes: List[List[Node]] = graph.query(
        """
    MATCH (n:NODE)
    RETURN n
    """
    ).result_set
    all_node_ids = {n[0].id for n in all_nodes if n[0].id is not None}
    all_node_ids = list(all_node_ids)

    list_of_list_of_paths = get_cluster_paths(graph, all_node_ids)
    assert len(list_of_list_of_paths) > 0

    # Get cluster paths returns a list of list of paths
    # but they should only have one path in each list
    # TODO adjust get_cluster_paths to return a list of paths
    for paths in list_of_list_of_paths:
        assert len(paths) > 0

    paths = [paths[0] for paths in list_of_list_of_paths]

    merge_prompt = get_prompt_for_merge_llm(paths, primary_node_ids=all_node_ids)
    if should_save_debug_data():
        merge_prompt_path = Path("test_output_data/merge_prompt.txt")
        Path("test_output_data").mkdir(exist_ok=True, parents=True)
        with open(merge_prompt_path, "w") as f:
            f.write(merge_prompt)
    merge_set = merge_llm(merge_prompt)
    # TODO actual tests to see if merge set is correct
    # for now, just pass the test, but save the merge set
    if should_save_debug_data():
        merge_set_path = Path("test_output_data/merge_set.txt")
        with open(merge_set_path, "w") as f:
            json.dump([m.to_dict() for m in merge_set], f, indent=4)