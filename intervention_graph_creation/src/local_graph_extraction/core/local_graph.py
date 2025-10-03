from typing import List
import numpy as np
from pydantic import BaseModel, ConfigDict
from pathlib import Path
import json

from intervention_graph_creation.src.local_graph_extraction.core.edge import GraphEdge
from intervention_graph_creation.src.local_graph_extraction.core.node import GraphNode
from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import PaperSchema
from intervention_graph_creation.src.utils import short_id_edge, short_id_node
from config import load_settings

SETTINGS = load_settings()


class LocalGraph(BaseModel):
    """Container for graph data with nodes and edges that have embeddings."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paper_id: str
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __len__(self) -> int:
        """Return total number of nodes and edges."""
        return len(self.nodes) + len(self.edges)

    @classmethod
    def from_paper_schema(cls, paper_schema: PaperSchema, json_path: Path) -> "tuple[LocalGraph | None, str | None]":
        """Create a LocalGraph from a PaperSchema. Logs errors and returns (None, error_msg) if invalid."""
        from intervention_graph_creation.src.local_graph_extraction.extract.utilities import write_failure
        names = [n.name for n in paper_schema.nodes]
        if len(names) != len(set(names)):
            dupes = sorted({x for x in names if names.count(x) > 1})
            msg = f"Duplicate node names in {json_path.name}: {dupes}"
            write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.graph_error_dir, json_path.name, Exception(msg))
            return None, msg

        known = set(names)


        missing = [
            (e.source_node, e.target_node)
            for e in paper_schema.edges
            if e.source_node not in known or e.target_node not in known
        ]
        if missing:
            msg = f"Edges reference unknown nodes in {json_path.name}: {missing[:5]}..."
            write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.graph_error_dir, json_path.name, Exception(msg))
            return None, msg

        local_graph = LocalGraph(nodes=paper_schema.nodes, edges=paper_schema.edges, paper_id=json_path.stem)
        return local_graph, None

    def add_embeddings_to_nodes(self, node: GraphNode, json_path: Path) -> None:
        """Load embeddings for a node from embeddings folder."""
        node_id = short_id_node(node)
        emb_path = json_path.parent / "embeddings" / f"{node_id}.json"
        if emb_path.exists():
            try:
                with open(emb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                node.embedding = np.array(data["embedding"], dtype=np.float32)
            except Exception as e:
                print(f"[WARN] Failed to load embedding for node {node.name}: {e}")
                node.embedding = np.zeros(1536, dtype=np.float32)
        else:
            print(f"[WARN] Embedding file not found for node {node.name} -> {emb_path}")
            node.embedding = np.zeros(1536, dtype=np.float32)

    def add_embeddings_to_edges(self, edge: GraphEdge, json_path: Path) -> None:
        """Load embeddings for an edge from embeddings folder."""
        edge_id = short_id_edge(edge)
        emb_path = json_path.parent / "embeddings" / f"{edge_id}.json"
        if emb_path.exists():
            try:
                with open(emb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                edge.embedding = np.array(data["embedding"], dtype=np.float32)
            except Exception as e:
                print(f"[WARN] Failed to load embedding for edge {edge.type} ({edge.source_node}->{edge.target_node}): {e}")
                edge.embedding = np.zeros(1536, dtype=np.float32)
        else:
            print(f"[WARN] Embedding file not found for edge {edge.type} ({edge.source_node}->{edge.target_node}) -> {emb_path}")
            edge.embedding = np.zeros(1536, dtype=np.float32)
