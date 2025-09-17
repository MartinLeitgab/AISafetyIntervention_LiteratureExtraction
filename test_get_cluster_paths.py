# pyright: standard
# TODO put this file in the appropriate location
from falkordb import FalkorDB, Graph, Path, Edge, Node, QueryResult
import pytest
from semantic_compression_part_2 import get_cluster_paths, SETTINGS
from falkordb_bulk_loader.bulk_insert import bulk_insert
from click.testing import CliRunner
from collections import Counter
from dataclasses import dataclass
from typing import Literal, List, cast

@dataclass
class GraphFixture:
    graph: Graph
    alone_node: Node
    """A node that is isolated (no edges)"""

    self_loop_path: Path
    """An isolated node with a self-loop edge"""

    self_loop_and_edge_edge_path: Path
    """A node with a self-loop edge and another edge to another node. This is the edge path"""

    self_loop_and_edge_loop_path: Path
    """A node with a self-loop edge and another edge to another node. This is the loop path"""

    neighbor_path: Path
    """A node with an edge to a neighbor node (not in cluster)"""
    neighbor_path_2: Path
    """A node with an edge from a neighbor node (not in cluster)"""

    multi_hop_path: Path
    """A path with two edges and three nodes"""

    multi_hop_path_2: Path
    """A path with two edges and three nodes, but in reverse order"""

    alive_to_tombstone_edge_path: Path
    """A path where an alive node has a tombstoned edge"""

    alive_to_tombstone_node_path: Path
    """A path where an alive node has an edge to a tombstoned node"""

    tombstone_to_alive_path: Path
    """A path where a tombstoned node has an edge to an alive node"""


    test_node_ids: list[int]
    test_edge_ids: list[int]
    neighbor_nodes: list[int]
    """Nodes that are one hop neighbors to cluster nodes, but not in the cluster themselves"""

@dataclass
class FixtureNode:
    tombstone: bool = False
    skip: bool = False
    cluster_neighbor: bool = False
    """Don't add to the list of test nodes to verify"""
    type: Literal["node"] = "node"

@dataclass
class FixtureEdge:
    tombstone: bool = False
    skip: bool = False
    """Don't add to the list of test edges to verify"""
    type: Literal["edge"] = "edge"

FixtureResultShape = list[FixtureNode | FixtureEdge]

class FixtureNodeList():
    """Keeps track of created test nodes and edges, so we can verify them in tests."""
    def __init__(self,graph: Graph):
        self.graph = graph
        self.nodes = []
        self.neighbor_nodes = []
        self.edges = []
    def get_result(self, cypher_query: str, shape: FixtureResultShape):
        """Run a cypher query and return the results, while keeping track of created nodes and edges.
        Returns a Path assuming all returned items make a path."""
        result = self.graph.query(cypher_query)
        assert len(shape) == len(result.result_set[0]), f"Expected {len(shape)} results, got {len(result.result_set[0])}"

        out: List[Node | Edge] = []
        path_nodes = []
        path_edges = []
        for (shape_item, result_item) in zip(shape,  result.result_set[0]):
            if shape_item.type == "node":
                assert isinstance(result_item, Node), f"Expected Node, got {type(result_item)}"
                if not shape_item.tombstone and not shape_item.skip and not shape_item.cluster_neighbor:
                    self.nodes.append(result_item.id or -1)
                if shape_item.cluster_neighbor:
                    self.neighbor_nodes.append(result_item.id or -1)
                path_nodes.append(result_item)
            elif shape_item.type == "edge":
                assert isinstance(result_item, Edge), f"Expected Edge, got {type(result_item)}"
                if not shape_item.tombstone and not shape_item.skip:
                    self.edges.append(result_item.id or -1)
                path_edges.append(result_item)
            else:
                raise ValueError(f"Unknown shape item type: {shape_item.type}")
            out.append(result_item)

        return out, Path(path_nodes, path_edges)

def load_test_graph() -> GraphFixture:
    """
    Load a test graph from CSV files using the bulk loader. Then make some more nodes for corner cases.
    """
    graph_name = "TEST_cluster_paths"
    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    graph = db.select_graph(graph_name)
    try:
        graph.delete()
    except:
        pass

    server_url = f"redis://{SETTINGS.falkordb.host}:{SETTINGS.falkordb.port}"
    runner = CliRunner()
    result = runner.invoke(
        bulk_insert,
        [
            graph_name,
            "--server-url", server_url,
            "--nodes-with-label", "CONCEPT", "./test_graph/nodes_Concept.csv",
            "--nodes-with-label", "INTERVENTION", "./test_graph/nodes_Intervention.csv",
            "--nodes-with-label", "RATIONALE", "./test_graph/nodes_Rationale.csv",
            "--nodes-with-label", "SOURCE", "./test_graph/nodes_Source.csv",
         
            "--relations-with-type", "EDGE", "./test_graph/edges_EDGE.csv",
            "--relations-with-type", "FROM", "./test_graph/edges_FROM.csv",
            "--relations-with-type", "HAS_RATIONALE", "./test_graph/edges_HAS_RATIONALE.csv",
        ],
    )
    if result.exit_code != 0:
        raise RuntimeError(f"bulk_insert failed: {result.output}\n{result.exception}")
    graph = db.select_graph(graph_name)
    graph.query("""
    MATCH (n)
    SET n:NODE
    """)

    l = FixtureNodeList(graph)

    alone_node_list, _ = l.get_result("""
    MATCH (n)
    WHERE id(n) = 1
    WITH n LIMIT 1
    CREATE (m:CONCEPT)
    SET m:NODE,m:TEST, m = properties(n),
        m.name = "alone_node"
    RETURN m
    """, [FixtureNode()])
    alone_node = cast(Node, alone_node_list[0])


    _, self_loop_path = l.get_result( """
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (m:CONCEPT)-[r:EDGE]-> (m)
    SET m:NODE, m:TEST, m = properties(n), r = properties(e),
        m.name = "self_loop_node"
    RETURN m, r
    """, [FixtureNode(), FixtureEdge()])
    

    r,_ = l.get_result("""
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (m:CONCEPT)-[r:EDGE]-> (o:CONCEPT)
    CREATE (m)-[r2:EDGE]->(m)
    SET m:NODE, o:NODE, m:TEST, o:TEST, m = properties(n),
                       o = properties(n), r = properties(e),
                       r2 = properties(e),
                       m.name = "self_loop_and_edges_node",
                       o.name = "end_self_loop_and_edges_node"

    RETURN m,r,o,r2
    """, [FixtureNode(), FixtureEdge(), FixtureNode(), FixtureEdge()])

    self_loop_and_edges_node = cast(Node, r[0])
    self_loop_and_edges_edge = cast(Edge, r[1])
    self_loop_and_edges_o = cast(Node, r[2])
    self_loop_and_edges_self_loop = cast(Edge, r[3])

    self_loop_and_edges_edge = Path([self_loop_and_edges_node, self_loop_and_edges_o], [self_loop_and_edges_edge])
    self_loop_and_edges_self_loop = Path([self_loop_and_edges_node], [self_loop_and_edges_self_loop])


    _, neighbor_path = l.get_result("""
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (m:CONCEPT)-[r:EDGE]->(o:CONCEPT)
    SET m:NODE, o:NODE,
            m:TEST, o:TEST,
            m = properties(n), o = properties(n),
            r = properties(e),
            m.name = "neighbor_node",
            o.name = "end_neighbor_node"
    RETURN m,r,o
    """, [FixtureNode(), FixtureEdge(), FixtureNode(cluster_neighbor=True)])

    _, neighbor_path_2 = l.get_result("""
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (o:CONCEPT)-[r:EDGE]->(m:CONCEPT)
    SET m:NODE, o:NODE,
            m:TEST, o:TEST,
            m = properties(n), o = properties(n),
            r = properties(e),
            o.name = "neighbor_node_2",
            m.name = "end_neighbor_node_2"
    RETURN o,r,m
    """, [FixtureNode(cluster_neighbor=True), FixtureEdge(), FixtureNode()])

    _, multi_hop_path = l.get_result("""
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (m:CONCEPT)-[r:EDGE]->(q:CONCEPT)-[r2:EDGE]->(o:CONCEPT)
    SET m:NODE, q:NODE, o:NODE, m:TEST, q:TEST, o:TEST,
                                     m = properties(n), q = properties(n),
                                     o = properties(n), r = properties(e),
                                     r2 = properties(e),
        m.name = "multi_hop_node",
        q.name = "mid_multi_hop_node",
        o.name = "end_multi_hop_node"
    RETURN m,r,q,r2,o
    """, [FixtureNode(), FixtureEdge(), FixtureNode(cluster_neighbor=True), FixtureEdge(skip=True), FixtureNode(skip=True)])


    _, multi_hop_path_2 = l.get_result("""
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (o:CONCEPT)-[r:EDGE]->(q:CONCEPT)-[r2:EDGE]->(m:CONCEPT)
    SET m:NODE, q:NODE, o:NODE, m:TEST, q:TEST, o:TEST, m = properties(n),
                                       q = properties(n), o = properties(n),
                                       r = properties(e), r2 = properties(e),
        o.name = "multi_hop_node_2",
        q.name = "mid_multi_hop_node_2",
        m.name = "end_multi_hop_node_2"
    RETURN o,r,q,r2,m
    """, [FixtureNode(skip=True), FixtureEdge(skip=True), FixtureNode(cluster_neighbor=True), FixtureEdge(), FixtureNode()])

   

    _, alive_to_tombstone_edge_path = l.get_result("""
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (m:CONCEPT)-[r:EDGE]->(o:CONCEPT)
    SET m:NODE, o:NODE,
            m:TEST, o:TEST, 
            m = properties(n), o = properties(n),
            r = properties(e), 
            r.is_tombstone = true,
            m.name = "alive_to_tombstone_edge",
            o.name = "end_alive_to_tombstone_edge"
           
    RETURN m,r,o
    """, [FixtureNode(), FixtureEdge(tombstone=True), FixtureNode()])

   
    _, alive_to_tombstone_node_path = l.get_result("""
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (m:CONCEPT)-[r:EDGE]->(o:CONCEPT)
    SET m:NODE, o:NODE,
            m:TEST, o:TEST, 
            m = properties(n), o = properties(n),
            r = properties(e),
            // o is tombstone now
            o.is_tombstone = true,
            m.name = "alive_to_tombstone_node",
            o.name = "end_alive_to_tombstone_node"
            
           
    RETURN m,r,o
    """, [FixtureNode(), FixtureEdge(skip=True), FixtureNode(tombstone=True)])

    _, tombstone_to_alive_path = l.get_result("""
    MATCH (n:NODE)-[e:EDGE]-()
    WHERE id(n) = 1
    WITH n, e LIMIT 1
    CREATE (m:CONCEPT)-[r:EDGE]->(o:CONCEPT)
    SET m:NODE, o:NODE,
            m:TEST, o:TEST, 
            m = properties(n), o = properties(n),
            r = properties(e),
            // m is tombstone now
            m.is_tombstone = true,
            m.name = "tombstone_to_alive",
            o.name = "end_tombstone_to_alive"
           
    RETURN m,r,o
    """, [FixtureNode(tombstone=True), FixtureEdge(skip=True), FixtureNode()])

   
    assert len(l.nodes) > 0, "No test nodes created"
    assert len(l.edges) > 0, "No test edges created"
    return GraphFixture(
        graph=graph,
        alone_node=alone_node,
        self_loop_path=self_loop_path,
        self_loop_and_edge_edge_path=self_loop_and_edges_edge,
        self_loop_and_edge_loop_path=self_loop_and_edges_self_loop,
        neighbor_path=neighbor_path,
        neighbor_path_2=neighbor_path_2,
        multi_hop_path=multi_hop_path,
        multi_hop_path_2=multi_hop_path_2,
        alive_to_tombstone_edge_path=alive_to_tombstone_edge_path,
        alive_to_tombstone_node_path=alive_to_tombstone_node_path,
        tombstone_to_alive_path=tombstone_to_alive_path,
        test_node_ids=l.nodes,
        test_edge_ids=l.edges,
        neighbor_nodes=l.neighbor_nodes,
    )

@pytest.fixture(scope="module")
def shared_graph():
    """A graph that is created once per test module and shared among all test functions."""
    graph = load_test_graph()
    try:
        yield graph
    finally:
        # comment this out if you want to inspect the
        # graphs in the falkorDB server after tests
        #     To find all test nodes look use
        # MATCH p=(:TEST)-[:EDGE*0..50]-() RETURN p
        graph.graph.delete()

@pytest.fixture
def single_test_graph():
    """A graph that is created and deleted for each test function. Use if you manipulate the graph."""
    graph = load_test_graph()
    try:
        yield graph
    finally:
        graph.graph.delete()


def test_1(shared_graph: GraphFixture):
    """Test that all nodes in the cluster are included"""
    g = shared_graph.graph

    cluster = shared_graph.test_node_ids
    paths = get_cluster_paths(g, cluster)

    not_found_in_cluster = set(cluster)
    not_found_in_neighbor = set(shared_graph.neighbor_nodes)
    found_outside_of_cluster = set()

    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            for node in path.nodes():
                if node.id is not None:
                    not_found_in_cluster.discard(node.id)
                    not_found_in_neighbor.discard(node.id)
                    if node.id not in cluster and node.id not in shared_graph.neighbor_nodes:
                        found_outside_of_cluster.add(node.id)

    assert len(not_found_in_cluster) == 0, f"Did not find these nodes in the cluster_paths {not_found_in_cluster}"
    assert len(not_found_in_neighbor) == 0, f"Did not find these neighbor nodes in the cluster_paths {not_found_in_neighbor}"
    assert len(found_outside_of_cluster) == 0, f"Found these nodes outside of the cluster_paths {found_outside_of_cluster}"


def get_edge_id(edge: Edge, attribute: str) -> int:
    if isinstance(edge.__getattribute__(attribute), int):
        return edge.__getattribute__(attribute)
    return edge.__getattribute__(attribute).id or -1

def test_2(shared_graph: GraphFixture):
    """Test that all edges between the nodes in the cluster are included"""
    g = shared_graph.graph
    cluster = shared_graph.test_node_ids

    desired_edges = set(shared_graph.test_edge_ids)
    paths = get_cluster_paths(g, cluster)
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            for edge in path.edges():
                desired_edges.discard(edge.id or -1)
    assert len(desired_edges) == 0, f"Edges not found in paths: {desired_edges}"


def test_3(shared_graph: GraphFixture):
    """Test that there are no repeated edges"""
    g = shared_graph.graph
    cluster = shared_graph.test_node_ids

    edge_counts = Counter()
    paths = get_cluster_paths(g, cluster)
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            for edge in path.edges():
                edge_counts[edge.id or -1] += 1
    assert all(count == 1 for count in edge_counts.values()), f"Edges repeated: { {k: v for k, v in edge_counts.items() if v != 1} }"

def test_4(shared_graph: GraphFixture):
    """Test that neighbor nodes are included"""
    g = shared_graph.graph
    assert len(shared_graph.neighbor_path.nodes()) == 2, "No neighbor nodes to test"
    [cluster_node, neighbor_node] = shared_graph.neighbor_path.nodes()
    cluster_node_id = cluster_node.id or -1
    cluster = [
        cluster_node_id
    ]
    
    paths = get_cluster_paths(g, cluster)
    assert len(paths) == 1, f"Expected 1 path, got {len(paths)}"
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            nodes = path.nodes()
            assert len(nodes) == 2, f"Expected 2 nodes in the path, cluster node ID {cluster_node_id}, len paths {len(paths)} , len path_list {len(path_list)} found:  {nodes[0].id if len(nodes) > 0 else 'N/A'}, {nodes[1].id if len(nodes) > 1 else 'N/A'   }"
            [should_be_cluster_node, should_be_neighbor_node] = nodes
            assert should_be_cluster_node.id == cluster_node.id, "Cluster node ID does not match"
            assert should_be_neighbor_node.id == neighbor_node.id, "Neighbor node ID does not match"

def test_5(shared_graph: GraphFixture):
    """Test that neighbor nodes are included"""
    g = shared_graph.graph
    assert len(shared_graph.neighbor_path_2.nodes()) == 2, "No neighbor nodes to test"
    [neighbor_node, cluster_node] = shared_graph.neighbor_path_2.nodes()
    cluster_node_id = cluster_node.id or -1
    cluster = [
        cluster_node_id
    ]

    paths = get_cluster_paths(g, cluster)
    assert len(paths) == 1, f"Expected 1 path, got {len(paths)}"
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            nodes = path.nodes()
            assert len(nodes) == 2, f"Expected 2 nodes in the path, cluster node ID {cluster_node_id}, len paths {len(paths)} , len path_list {len(path_list)} found:  {nodes[0].id if len(nodes) > 0 else 'N/A'}, {nodes[1].id if len(nodes) > 1 else 'N/A'   }"
            [should_be_cluster_node, should_be_neighbor_node] = nodes
            assert should_be_cluster_node.id == cluster_node.id, "Cluster node ID does not match"
            assert should_be_neighbor_node.id == neighbor_node.id, "Neighbor node ID does not match"


def test_6(shared_graph: GraphFixture):
    """Test that multi-hop doesn't happen, only direct connections are found"""
    g = shared_graph.graph
    cluster = [
        shared_graph.multi_hop_path.nodes()[0].id or -1
    ]
    [first_edge, last_edge] = shared_graph.multi_hop_path.edges()
    paths = get_cluster_paths(g, cluster)
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            for edge in path.edges():
                assert edge.id != last_edge.id, "Multi-hop edge found"
                assert edge.id == first_edge.id, "Multi-hop edge found"

def test_7(shared_graph: GraphFixture):
    """Test that multi-hop doesn't happen, only direct connections are found (reverse order)"""
    g = shared_graph.graph
    cluster = [
        shared_graph.multi_hop_path_2.nodes()[-1].id or -1
    ]
    [first_edge, last_edge] = shared_graph.multi_hop_path_2.edges()
    paths = get_cluster_paths(g, cluster)
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            for edge in path.edges():
                assert edge.id != first_edge.id, "Multi-hop edge found"
                assert edge.id == last_edge.id, "Multi-hop edge found"

def test_8(shared_graph: GraphFixture):
    """Test that alive to tombstone edge is not included"""
    g = shared_graph.graph
    [start_node, end_node] = shared_graph.alive_to_tombstone_edge_path.nodes()
    cluster = [
        start_node.id or -1,
        end_node.id or -1
    ]
    [edge] = shared_graph.alive_to_tombstone_edge_path.edges()


    paths = get_cluster_paths(g, cluster)
    assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"
    found_start = False
    found_end = False
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            nodes = path.nodes()
            assert len(nodes) == 1, f"Expected 1 node in the path, found: {nodes[0].id if len(nodes) > 0 else 'N/A'}"
            if nodes[0].id == start_node.id:
                found_start = True
            if nodes[0].id == end_node.id:
                found_end = True
            for e in path.edges():
                assert e.id != edge.id, "Tombstone edge found"
    assert found_start, "Did not find start node"
    assert found_end, "Did not find end node"


def test_9(shared_graph: GraphFixture):
    """Test that alive to tombstone node is not included"""
    g = shared_graph.graph
    [start_node, tombstoned_end_node] = shared_graph.alive_to_tombstone_node_path.nodes()
    cluster = [
        start_node.id or -1,
    ]
    [edge] = shared_graph.alive_to_tombstone_node_path.edges()


    paths = get_cluster_paths(g, cluster)
    assert len(paths) == 1, f"Expected 1 path, got {len(paths)}"
    found_start = False
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            nodes = path.nodes()
            assert len(nodes) == 1, f"Expected 1 node in the path, found: {nodes[0].id if len(nodes) > 0 else 'N/A'}"
            if nodes[0].id == start_node.id:
                found_start = True
            if nodes[0].id == tombstoned_end_node.id:
                raise AssertionError("Tombstoned node found")
            for e in path.edges():
                assert e.id != edge.id, "Tombstone edge found"
    assert found_start, "Did not find start node"

def test_10(shared_graph: GraphFixture):
    """Test that alive to tombstone node is not included"""
    g = shared_graph.graph
    [start_node, tombstoned_end_node] = shared_graph.alive_to_tombstone_node_path.nodes()
    cluster = [
        start_node.id or -1,
    ]
    [edge] = shared_graph.alive_to_tombstone_node_path.edges()


    paths = get_cluster_paths(g, cluster)
    assert len(paths) == 1, f"Expected 1 path, got {len(paths)}"
    found_start = False
    for path_list in paths:
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, Path)
            nodes = path.nodes()
            assert len(nodes) == 1, f"Expected 1 node in the path, found: {nodes[0].id if len(nodes) > 0 else 'N/A'}"
            if nodes[0].id == start_node.id:
                found_start = True
            if nodes[0].id == tombstoned_end_node.id:
                raise AssertionError("Tombstoned node found")
            for e in path.edges():
                assert e.id != edge.id, "Tombstone edge found"
    assert found_start, "Did not find start node"

if __name__ == "__main__":
    load_test_graph()