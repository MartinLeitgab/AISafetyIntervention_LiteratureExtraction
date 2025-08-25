from typing import List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict

from ..local_graph_extraction.core import Node, Edge


class GraphNode(Node):
    """Extended Node class with embedding support."""
    embedding: Optional[np.ndarray] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        # Handle embedding separately to avoid pydantic validation issues
        embedding = data.pop('embedding', None)
        super().__init__(**data)
        self.embedding = embedding


class GraphEdge(Edge):
    """Extended Edge class with embedding and concept metadata support."""
    embedding: Optional[np.ndarray] = None
    concept_meta: Optional[str] = None  # Equivalent to title in LogicalChain
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        # Handle embedding separately to avoid pydantic validation issues
        embedding = data.pop('embedding', None)
        concept_meta = data.pop('concept_meta', None)
        super().__init__(**data)
        self.embedding = embedding
        self.concept_meta = concept_meta


class LocalGraph(BaseModel):
    """Container for graph data with nodes and edges that have embeddings."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    
    def __len__(self) -> int:
        """Return total number of nodes and edges."""
        return len(self.nodes) + len(self.edges)
    
    def get_node_by_name(self, name: str) -> Optional[GraphNode]:
        """Get a node by its name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def get_edges_by_source(self, source_name: str) -> List[GraphEdge]:
        """Get all edges that have the given source node."""
        return [edge for edge in self.edges if edge.source_node == source_name]
    
    def get_edges_by_target(self, target_name: str) -> List[GraphEdge]:
        """Get all edges that have the given target node."""
        return [edge for edge in self.edges if edge.target_node == target_name]
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
    
    def remove_node(self, node_name: str) -> bool:
        """Remove a node and all its associated edges."""
        # Remove the node
        initial_node_count = len(self.nodes)
        self.nodes = [node for node in self.nodes if node.name != node_name]
        
        # Remove associated edges
        self.edges = [
            edge for edge in self.edges 
            if edge.source_node != node_name and edge.target_node != node_name
        ]
        
        return len(self.nodes) < initial_node_count
