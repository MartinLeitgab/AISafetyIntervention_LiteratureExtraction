import json
import gzip
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Flow
from ..local_graph import LocalGraph, GraphNode, GraphEdge
from ...local_graph_extraction.core import PaperSchema, Node, Edge


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate over JSONL file, handling both regular and gzipped files."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class EmbedderFlow(Flow):
    """Flow that loads JSONL data, converts to LocalGraph, and adds embeddings."""
    
    def __init__(self, next_flow: Optional[Flow] = None, model_name: str = "BAAI/bge-large-en-v1.5"):
        super().__init__(next_flow)
        self.model_name = model_name
        self.model = None  # Lazy loading
    
    def _load_jsonl_to_paper_schema(self, output_path: str) -> PaperSchema:
        """Load JSONL file and convert to PaperSchema."""
        # Load the first (and presumably only) line from the JSONL file
        for item in iter_jsonl(output_path):
            return PaperSchema(**item)
        raise ValueError(f"No valid data found in {output_path}")
    
    def _convert_node_to_graph_node(self, node: Node) -> GraphNode:
        """Convert a Node to a GraphNode."""
        return GraphNode(
            name=node.name,
            aliases=node.aliases,
            type=node.type,
            description=node.description,
            concept_category=node.concept_category,
            intervention_lifecycle=node.intervention_lifecycle,
            intervention_maturity=node.intervention_maturity
        )
    
    def _convert_edge_to_graph_edge(self, edge: Edge, concept_meta: Optional[str] = None) -> GraphEdge:
        """Convert an Edge to a GraphEdge."""
        return GraphEdge(
            type=edge.type,
            source_node=edge.source_node,
            target_node=edge.target_node,
            description=edge.description,
            edge_confidence=edge.edge_confidence,
            concept_meta=concept_meta
        )
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a given text using SentenceTransformers."""
        try:
            # Lazy load the model
            if self.model is None:
                print(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            
            # Get embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error getting embedding for text: {e}")
            # Return zero vector as fallback (BGE-large-v1.5 has 1024 dimensions)
            return np.zeros(1024, dtype=np.float32)
    
    def _add_embeddings_to_nodes(self, local_graph: LocalGraph) -> None:
        """Add embeddings to all nodes in the local graph."""
        for node in local_graph.nodes:
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
    
    def _add_embeddings_to_edges(self, local_graph: LocalGraph) -> None:
        """Add embeddings to all edges in the local graph."""
        for edge in local_graph.edges:
            # Create text representation for embedding
            text_parts = []
            if edge.type:
                text_parts.append(f"Type: {edge.type}")
            if edge.description:
                text_parts.append(f"Description: {edge.description}")
            if edge.concept_meta:
                text_parts.append(f"Concept: {edge.concept_meta}")
            if edge.source_node:
                text_parts.append(f"From: {edge.source_node}")
            if edge.target_node:
                text_parts.append(f"To: {edge.target_node}")
            
            text = " | ".join(text_parts)
            edge.embedding = self._get_embedding(text)
    
    def process(self, local_graph_or_path) -> LocalGraph:
        """
        Process the input and convert to LocalGraph with embeddings.
        
        Args:
            local_graph_or_path: Either a LocalGraph instance or path to JSONL file
            
        Returns:
            LocalGraph with embeddings added
        """
        # Handle both string path and LocalGraph input
        if isinstance(local_graph_or_path, str):
            # Load JSONL and convert to PaperSchema
            paper_schema = self._load_jsonl_to_paper_schema(local_graph_or_path)
            
            # Convert to LocalGraph
            graph_nodes = [self._convert_node_to_graph_node(node) for node in paper_schema.nodes]
            
            # Convert logical chains to edges with concept metadata
            graph_edges = []
            for logical_chain in paper_schema.logical_chains:
                for edge in logical_chain.edges:
                    graph_edge = self._convert_edge_to_graph_edge(edge, logical_chain.title)
                    graph_edges.append(graph_edge)
            
            local_graph = LocalGraph(nodes=graph_nodes, edges=graph_edges)
        else:
            # Assume it's already a LocalGraph
            local_graph = local_graph_or_path
        
        # Add embeddings
        print("Adding embeddings to nodes...")
        self._add_embeddings_to_nodes(local_graph)
        
        print("Adding embeddings to edges...")
        self._add_embeddings_to_edges(local_graph)
        
        # Pass to next flow in chain
        return self._call_next(local_graph)
