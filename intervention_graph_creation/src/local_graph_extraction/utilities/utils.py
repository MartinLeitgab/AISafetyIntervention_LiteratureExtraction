from hashlib import sha1

from intervention_graph_creation.src.local_graph_extraction.core.edge import GraphEdge
from intervention_graph_creation.src.local_graph_extraction.core.node import GraphNode


def short_id(s: str) -> str:
    """Deterministic 10-char hex ID (lowercase)."""
    norm = " ".join(s.strip().split()).lower()
    return sha1(norm.encode("utf-8")).hexdigest()[:10]


def short_id_node(node: GraphNode) -> str:
    return short_id(f"{node.name}|{node.type}")


def short_id_edge(edge: GraphEdge) -> str:
    return short_id(f"{edge.type}|{edge.source_node}|{edge.target_node}")
