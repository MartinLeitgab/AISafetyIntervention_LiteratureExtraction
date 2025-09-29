from typing import List
import numpy as np
from pydantic import BaseModel, ConfigDict
from pathlib import Path
# from sentence_transformers import SentenceTransformer

from intervention_graph_creation.src.local_graph_extraction.core.edge import GraphEdge
from intervention_graph_creation.src.local_graph_extraction.core.node import GraphNode
from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import (
    PaperSchema,
)
from sentence_transformers import SentenceTransformer


class LocalGraph(BaseModel):
    """Container for graph data with nodes and edges that have embeddings."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paper_id: str
    # model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=False,  # Disable validation on assignment
        validate_default=False,  # Disable validation of default values
    )
    # embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    # embedding_model: Optional[SentenceTransformer] = None

    def __len__(self) -> int:
        """Return total number of nodes and edges."""
        return len(self.nodes) + len(self.edges)

    @classmethod
    def from_paper_schema(
        self, paper_schema: PaperSchema, json_path: Path
    ) -> "tuple[LocalGraph | None, str | None]":
        """Create a LocalGraph from a PaperSchema. Logs errors and returns (None, error_msg) if invalid."""
        from intervention_graph_creation.src.local_graph_extraction.extract.utilities import (
            write_failure,
        )

        names = [n.name for n in paper_schema.nodes]
        if len(names) != len(set(names)):
            dupes = sorted({x for x in names if names.count(x) > 1})
            msg = f"Duplicate node names in {json_path.name}: {dupes}"
            write_failure(json_path.parent, json_path.name, Exception(msg))
            return None, msg

        known = set(names)

        # Check for missing nodes - handle both formats
        missing = []
        if hasattr(paper_schema, "logical_chains") and paper_schema.logical_chains:
            # Original format: check edges in logical_chains
            missing = [
                (e.source_node, e.target_node)
                for ch in paper_schema.logical_chains
                for e in ch.edges
                if e.source_node not in known or e.target_node not in known
            ]
        elif hasattr(paper_schema, "edges") and paper_schema.edges:
            # New format: check edges at top level
            missing = [
                (e.source_node, e.target_node)
                for e in paper_schema.edges
                if e.source_node not in known or e.target_node not in known
            ]

        if missing:
            msg = f"Edges reference unknown nodes in {json_path.name}: {missing[:5]}..."
            write_failure(json_path.parent, json_path.name, Exception(msg))
            return None, msg

        # Convert to LocalGraph
        graph_nodes = [
            GraphNode.model_construct(**node.model_dump())
            for node in paper_schema.nodes
        ]

        # Handle both edge formats
        graph_edges = []
        if hasattr(paper_schema, "logical_chains") and paper_schema.logical_chains:
            # Original format: edges in logical_chains
            for logical_chain in paper_schema.logical_chains:
                for edge in logical_chain.edges:
                    graph_edge = GraphEdge.model_construct(**edge.model_dump())
                    graph_edges.append(graph_edge)
        elif hasattr(paper_schema, "edges") and paper_schema.edges:
            # New format: edges at top level
            for edge in paper_schema.edges:
                graph_edge = GraphEdge.model_construct(**edge.model_dump())
                graph_edges.append(graph_edge)

        # Create the LocalGraph - THIS LINE WAS MISSING OR MISPLACED
        local_graph = LocalGraph.model_construct(
            nodes=graph_nodes, edges=graph_edges, paper_id=json_path.stem
        )

        return local_graph, None

    def add_embeddings_to_nodes(self, node: GraphNode) -> None:
        """Add embeddings to all nodes in the local graph."""
        # Disabled - no embedding generation
        node.embedding = None
        return
        # Create text representation for embedding
        text_parts = []
        if node.name:
            text_parts.append(f"Name: {node.name}")
        if node.description:
            text_parts.append(f"Description: {node.description}")
        if node.aliases:
            text_parts.append(f"Aliases: {', '.join(node.aliases)}")
        if node.concept_category:
            text_parts.append(f"Category: {node.concept_category}")

        text = " | ".join(text_parts)
        node.embedding = self._get_embedding(text)

    def add_embeddings_to_edges(self, edge: GraphEdge) -> None:
        """Add embeddings to all edges in the local graph."""
        # Disabled - no embedding generation
        edge.embedding = None
        return
        # Create text representation for embedding
        text_parts = []
        if edge.type:
            text_parts.append(f"Type: {edge.type}")
        if edge.description:
            text_parts.append(f"Description: {edge.description}")
        if edge.logical_chain_title:
            text_parts.append(f"Concept: {edge.logical_chain_title}")
        if edge.source_node:
            text_parts.append(f"From: {edge.source_node}")
        if edge.target_node:
            text_parts.append(f"To: {edge.target_node}")

        text = " | ".join(text_parts)
        edge.embedding = self._get_embedding(text)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a given text using SentenceTransformers."""
        try:
            # Lazy load the model
            if self.embedding_model is None:
                print(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Get embedding
            embedding = self.embedding_model.encode(
                text, batch_size=16, convert_to_numpy=True
            )
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error getting embedding for text: {e}")
            # Return zero vector as fallback (BGE-large-v1.5 has 1024 dimensions)
            return np.zeros(1024, dtype=np.float32)
