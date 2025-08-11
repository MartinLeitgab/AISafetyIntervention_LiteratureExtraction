"""FalkorDB storage implementation for ConceptGraphs."""

from typing import Optional, Any
from enum import Enum
import json
from .falkordb_client import FalkorDBClient
from src.models.concept_graph import ConceptNode, DirectedEdge, ConceptGraph

class ConceptGraphStore:
    def __init__(self, host: str = "localhost", port: int = 6379, password: Optional[str] = None):
        """Initialize ConceptGraph storage.
        
        Args:
            host: FalkorDB host
            port: FalkorDB port
            password: Optional password for authentication
        """
        self.db = FalkorDBClient(host=host, port=port, password=password)
        self.db.select_graph("concept_graphs")
    
    @staticmethod
    def _serialize_props(props: dict[str, Any]) -> dict[str, Any]:
        """Convert complex Python types to FalkorDB-friendly primitives.

        - Enum -> str (value)
        - list/tuple/dict -> JSON string
        - None -> omitted
        """
        serialized: dict[str, Any] = {}
        for key, value in props.items():
            if value is None:
                continue
            if isinstance(value, Enum):
                serialized[key] = value.value
            elif isinstance(value, dict):
                # Maps cannot be stored as properties; JSON-encode after converting Enums
                def to_basic(v: Any) -> Any:
                    if isinstance(v, Enum):
                        return v.value
                    if isinstance(v, (list, tuple)):
                        return [to_basic(x) for x in v]
                    if isinstance(v, dict):
                        return {kk: to_basic(vv) for kk, vv in v.items()}
                    return v
                serialized[key] = json.dumps(to_basic(value), ensure_ascii=False)
            elif isinstance(value, (list, tuple)):
                # Arrays are supported if all elements are scalars; otherwise JSON-encode
                def to_basic_elem(v: Any) -> Any:
                    if isinstance(v, Enum):
                        return v.value
                    return v
                basic_list = [to_basic_elem(v) for v in value]
                if all(isinstance(v, (str, int, float, bool)) for v in basic_list):
                    serialized[key] = basic_list
                else:
                    serialized[key] = json.dumps(basic_list, ensure_ascii=False)
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _deserialize_props(props: dict[str, Any]) -> dict[str, Any]:
        """Attempt to parse JSON-encoded list/dict strings back to Python types."""
        parsed: dict[str, Any] = {}
        for key, value in props.items():
            if isinstance(value, str) and value and value[0] in "[{":
                try:
                    parsed[key] = json.loads(value)
                    continue
                except Exception:
                    pass
            parsed[key] = value
        return parsed
        
    def store_graph(self, graph: ConceptGraph, paper_id: str) -> None:
        """Store a ConceptGraph in FalkorDB.
        
        Args:
            graph: ConceptGraph to store
            paper_id: Unique identifier for the paper (e.g. arxiv ID)
        """
        # Store nodes
        node_id_map = {}  # Map concept node IDs to FalkorDB node IDs
        for node in graph.nodes:
            properties = node.dict()
            properties["paper_id"] = paper_id
            node_type = "Intervention" if node.is_intervention else "Concept"
            serialized = self._serialize_props(properties)
            db_node_id = self.db.create_node(node_type, serialized)
            node_id_map[node.id] = db_node_id
            
        # Store edges
        for edge in graph.edges:
            for source_id in edge.source_node_ids:
                for target_id in edge.target_node_ids:
                    if source_id in node_id_map and target_id in node_id_map:
                        properties = edge.dict(exclude={"source_node_ids", "target_node_ids", "relationship"})
                        properties["paper_id"] = paper_id
                        serialized = self._serialize_props(properties)
                        self.db.create_relationship(
                            node_id_map[source_id],
                            node_id_map[target_id],
                            edge.relationship.value,
                            serialized
                        )
                        
    def get_graph(self, paper_id: str) -> Optional[ConceptGraph]:
        """Retrieve a ConceptGraph from FalkorDB.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            ConceptGraph if found, None otherwise
        """
        # Get all nodes for paper
        query = """
        MATCH (n)
        WHERE n.paper_id = $paper_id
        RETURN n
        """
        result = self.db.graph.ro_query(query, {"paper_id": paper_id})
        if not result.result_set:
            return None
            
        nodes = []
        for row in result.result_set:
            node_data = row[0]
            node_data.pop("paper_id", None)  # Remove paper_id from properties
            node_data = self._deserialize_props(node_data)
            nodes.append(ConceptNode(**node_data))
            
        # Get all relationships for paper
        query = """
        MATCH (a)-[r]->(b)
        WHERE r.paper_id = $paper_id
        RETURN type(r), properties(r), ID(a), ID(b)
        """
        result = self.db.graph.ro_query(query, {"paper_id": paper_id})
        
        edges = []
        for rel_type, props, source_id, target_id in result.result_set:
            props.pop("paper_id", None)  # Remove paper_id from properties
            props = self._deserialize_props(props)
            edge = DirectedEdge(
                **props,
                relationship=rel_type,
                source_node_ids=[str(source_id)],
                target_node_ids=[str(target_id)]
            )
            edges.append(edge)
            
        return ConceptGraph(nodes=nodes, edges=edges)
        
    def delete_graph(self, paper_id: str) -> bool:
        """Delete a ConceptGraph from FalkorDB.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            True if successful
        """
        # Delete all relationships first
        query = """
        MATCH ()-[r]->()
        WHERE r.paper_id = $paper_id
        DELETE r
        """
        self.db.graph.query(query, {"paper_id": paper_id})
        
        # Then delete all nodes; determine if anything existed by counting first
        pre_count_q = """
        MATCH (n)
        WHERE n.paper_id = $paper_id
        RETURN count(n)
        """
        pre = self.db.graph.ro_query(pre_count_q, {"paper_id": paper_id})
        existed = bool(pre.result_set and pre.result_set[0][0] > 0)
        delete_q = """
        MATCH (n)
        WHERE n.paper_id = $paper_id
        DELETE n
        """
        self.db.graph.query(delete_q, {"paper_id": paper_id})
        return existed
