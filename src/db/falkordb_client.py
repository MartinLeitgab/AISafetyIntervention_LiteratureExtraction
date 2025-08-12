"""FalkorDB client implementation for graph operations."""

from typing import List, Optional, Dict, Any
import json
from falkordb import FalkorDB

class FalkorDBClient:
    @staticmethod
    def _escape_string(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\"", "\\\"")

    @classmethod
    def _cypher_value(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return f'"{cls._escape_string(value)}"'
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(cls._cypher_value(v) for v in value) + "]"
        if value is None:
            return "null"
        # Fallback: JSON-encode and store as string
        return f'"{cls._escape_string(json.dumps(value, ensure_ascii=False))}"'

    @classmethod
    def _format_properties(cls, properties: Dict[str, Any]) -> str:
        pairs: list[str] = []
        for key, value in properties.items():
            if value is None:
                continue
            # Keys should be simple property names; assume safe identifiers
            pairs.append(f"{key}: {cls._cypher_value(value)}")
        return "{" + ", ".join(pairs) + "}"
    def __init__(self, host: str = "localhost", port: int = 6379, password: Optional[str] = None):
        """Initialize FalkorDB client.
        
        Args:
            host: FalkorDB host
            port: FalkorDB port
            password: Optional password for authentication
        """
        self.client = FalkorDB(host=host, port=port, password=password)
        
    def select_graph(self, graph_name: str) -> None:
        """Select a graph to work with.
        
        Args:
            graph_name: Name of the graph
        """
        self.graph = self.client.select_graph(graph_name)
        
    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a node in the graph.
        
        Args:
            label: Node label (type)
            properties: Node properties
            
        Returns:
            Node ID
        """
        props_literal = self._format_properties(properties)
        query = f"""
        CREATE (n:{label} {props_literal})
        RETURN ID(n) as id
        """
        result = self.graph.query(query)
        return result.result_set[0][0]
        
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node properties or None if not found
        """
        query = """
        MATCH (n)
        WHERE ID(n) = $id
        RETURN n
        """
        result = self.graph.ro_query(query, {"id": node_id})
        if not result.result_set:
            return None
        return result.result_set[0][0]
        
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties.
        
        Args:
            node_id: Node ID
            properties: New properties to set
            
        Returns:
            True if successful
        """
        props_literal = self._format_properties(properties)
        query = f"""
        MATCH (n)
        WHERE ID(n) = $id
        SET n += {props_literal}
        RETURN n
        """
        result = self.graph.query(query, {"id": node_id})
        return bool(result.result_set)
        
    def delete_node(self, node_id: str) -> bool:
        """Delete a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if successful
        """
        # Determine existence first, then delete
        pre_count_q = """
        MATCH (n)
        WHERE ID(n) = $id
        RETURN count(n)
        """
        pre = self.graph.ro_query(pre_count_q, {"id": node_id})
        existed = bool(pre.result_set and pre.result_set[0][0] > 0)
        delete_q = """
        MATCH (n)
        WHERE ID(n) = $id
        DELETE n
        """
        self.graph.query(delete_q, {"id": node_id})
        return existed
        
    def create_relationship(self, from_id: str, to_id: str, rel_type: str, properties: Dict[str, Any]) -> str:
        """Create a relationship between nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            rel_type: Relationship type
            properties: Relationship properties
            
        Returns:
            Relationship ID
        """
        props_literal = self._format_properties(properties)
        query = f"""
        MATCH (a), (b)
        WHERE ID(a) = $from_id AND ID(b) = $to_id
        CREATE (a)-[r:{rel_type} {props_literal}]->(b)
        RETURN ID(r) as id
        """
        result = self.graph.query(query, {
            "from_id": from_id,
            "to_id": to_id,
        })
        return result.result_set[0][0]
        
    def get_relationships(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """Get relationships for a node.
        
        Args:
            node_id: Node ID
            direction: Relationship direction ("in", "out", or "both")
            
        Returns:
            List of relationships
        """
        if direction == "out":
            query = """
            MATCH (n)-[r]->(m)
            WHERE ID(n) = $id
            RETURN type(r) as type, properties(r) as props, ID(m) as target_id
            """
        elif direction == "in":
            query = """
            MATCH (m)-[r]->(n)
            WHERE ID(n) = $id
            RETURN type(r) as type, properties(r) as props, ID(m) as source_id
            """
        else:  # both
            query = """
            MATCH (n)-[r]-(m)
            WHERE ID(n) = $id
            RETURN type(r) as type, properties(r) as props, ID(m) as other_id
            """
        result = self.graph.ro_query(query, {"id": node_id})
        return [dict(zip(["type", "properties", "connected_id"], row)) for row in result.result_set]
        
    def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship by ID.
        
        Args:
            rel_id: Relationship ID
            
        Returns:
            True if successful
        """
        # Determine existence first, then delete
        pre_count_q = """
        MATCH ()-[r]->()
        WHERE ID(r) = $id
        RETURN count(r)
        """
        pre = self.graph.ro_query(pre_count_q, {"id": rel_id})
        existed = bool(pre.result_set and pre.result_set[0][0] > 0)
        delete_q = """
        MATCH ()-[r]->()
        WHERE ID(r) = $id
        DELETE r
        """
        self.graph.query(delete_q, {"id": rel_id})
        return existed
