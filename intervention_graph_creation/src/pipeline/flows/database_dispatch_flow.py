import json
import subprocess
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from falkordb import FalkorDB

from .embedder_flow import GraphEdge

from .base import Flow
from ..local_graph import LocalGraph


def lit(v):
    """Helper function to properly escape values for Cypher queries."""
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _execute_falkordb_query_subprocess(query: str, container_name: str = "falkordb-test") -> str:
    """
    Execute a FalkorDB query using subprocess to avoid Mac-specific Redis client issues.
    
    Args:
        query: The Cypher query to execute
        container_name: Name of the FalkorDB Docker container
        
    Returns:
        The query result as a string
    """
    try:
        cmd = ['docker', 'exec', container_name, 'redis-cli', 'GRAPH.QUERY', 'AISafetyIntervention', query]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception(f"FalkorDB query failed: {e.stderr}")
    except Exception as e:
        raise Exception(f"Failed to execute FalkorDB query: {e}")


def _execute_falkordb_query_client(query: str, graph) -> str:
    """
    Execute a FalkorDB query using the original FalkorDB client.
    
    Args:
        query: The Cypher query to execute
        graph: The FalkorDB graph instance
        
    Returns:
        The query result as a string
    """
    try:
        result = graph.query(query)
        return str(result.result_set)
    except Exception as e:
        raise Exception(f"FalkorDB client query failed: {e}")


def execute_falkordb_query(query: str, graph=None, container_name: str = "falkordb-test") -> str:
    """
    Execute a FalkorDB query using the method specified by QUERY_EXECUTION_METHOD environment variable.
    
    Args:
        query: The Cypher query to execute
        graph: The FalkorDB graph instance (required for client method)
        container_name: Name of the FalkorDB Docker container (required for subprocess method)
        
    Returns:
        The query result as a string
    """
    method = os.getenv('QUERY_EXECUTION_METHOD', 'client').lower()
    
    if method == 'subprocess':
        return _execute_falkordb_query_subprocess(query, container_name)
    elif method == 'client':
        if graph is None:
            raise ValueError("Graph instance required for client method")
        return _execute_falkordb_query_client(query, graph)
    else:
        raise ValueError(f"Unknown query execution method: {method}. Use 'client' or 'subprocess'")


class DatabaseDispatchFlow(Flow):
    """Flow that handles database operations and disk storage."""
    
    def __init__(self, next_flow: Optional[Flow] = None, 
                 storage_dir: str = "pipeline_output",
                 host: str = "localhost",
                 port: int = 6379,
                 container_name: str = "falkordb-test",
                 graph_name: str = "AISafetyIntervention"):
        super().__init__(next_flow)
        self.host = host
        self.port = port
        self.container_name = container_name
        self.graph_name = graph_name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize FalkorDB client for client method
        self.db = FalkorDB(host=host, port=port)
        self.graph = self.db.select_graph(graph_name)
    
    def _save_to_disk(self, local_graph: LocalGraph, paper_id: str = None) -> str:
        """Save the LocalGraph to disk as JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paper_id = paper_id or f"paper_{timestamp}" # TODO: @jefedigital Once LocalGraph node is updated with metadata, we can use paper_id from the node
        
        # Convert numpy arrays to lists for JSON serialization
        graph_data = local_graph.model_dump()
        
        # Ensure embeddings are serializable
        for node in graph_data["nodes"]:
            if node.get("embedding") is not None and hasattr(node["embedding"], "tolist"):
                node["embedding"] = node["embedding"].tolist()
        
        for edge in graph_data["edges"]:
            if edge.get("embedding") is not None and hasattr(edge["embedding"], "tolist"):
                edge["embedding"] = edge["embedding"].tolist()
        
        # Save to file
        output_file = self.storage_dir / f"{paper_id}_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved LocalGraph to disk: {output_file}")
        return str(output_file)
    
    def _unsafe_push_node_to_database(self, node_id: str, local_graph: LocalGraph, seen_nodes: set, method: str) -> int:
        """Pushes a node to the database and returns 1 if the node was created, 0 if it was already in the database."""
        if node_id in seen_nodes:
            return 0

        node = local_graph.get_node_by_name(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in local graph")

        node_query = (
            f"MERGE (n:NODE {{name: {lit(node.name)}}}) "
            f"SET n.aliases = {lit(node.aliases or [])}, "
            f"n.type = {lit(node.type or 'CONCEPT')}, "
            f"n.description = {lit(node.description or '')}, "
            f"n.concept_category = {lit(node.concept_category or '')}, "
            f"n.intervention_lifecycle = {lit(node.intervention_lifecycle or 0)}, "
            f"n.intervention_maturity = {lit(node.intervention_maturity or 0)}, "
            f"n.embedding = {lit(node.embedding.tolist() or [])}"
            f"RETURN n"
        )
        if method == 'subprocess':
            execute_falkordb_query(node_query, container_name=self.container_name)
        else:
            execute_falkordb_query(node_query, graph=self.graph, container_name=self.container_name)
        seen_nodes.add(node_id)
        return 1

    def _unsafe_push_edge_to_database(self, edge: GraphEdge, local_graph: LocalGraph, method: str) -> int:
        """Pushes an edge to the database and returns 1 if the edge was created, 0 if it was already in the database."""
        edge_query = (
            f"MATCH (t:NODE {{name: {lit(edge.target_node)}}}), "
            f"(s:NODE {{name: {lit(edge.source_node)}}}) "
            f"MERGE (s)-[r:{edge.description}]->(t) " #TODO: @jefedigital: I think description here should be fine?
            f"SET r.description = {lit(edge.description or '')}, "
            f"r.edge_confidence = {lit(edge.edge_confidence or 1)}, "
            f"r.concept_meta = {lit(edge.concept_meta or '')} "
            f"r.embedding = {lit(edge.embedding.tolist() or [])}"
            f"RETURN r"
        )
        if method == 'subprocess':
            execute_falkordb_query(edge_query, container_name=self.container_name)
        else:
            execute_falkordb_query(edge_query, graph=self.graph, container_name=self.container_name)
        return 1

    def _push_to_database(self, local_graph: LocalGraph, paper_id: str = None) -> None:
        """Push the LocalGraph data to the database using configurable execution method."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paper_id = paper_id or f"paper_{timestamp}"
        
        method = os.getenv('QUERY_EXECUTION_METHOD', 'client').lower()
        print(f"🔧 Using query execution method: {method}")
        
        try:
            # Upsert nodes and edges
            nodes_created = 0
            edges_created = 0
            seen_nodes = set()
            for edge in local_graph.edges:
                nodes_created += self._unsafe_push_node_to_database(
                    edge.target_node, local_graph, seen_nodes, method) #TODO: @jefedigital: Name is being used a proxy for id.
                nodes_created += self._unsafe_push_node_to_database(
                    edge.source_node, local_graph, seen_nodes, method) #TODO: @jefedigital: Name is being used a proxy for id

                edges_created += self._unsafe_push_edge_to_database(edge, local_graph, method) #TODO: @jefedigital: Edge description is being used a proxy for id.
            print(f"✓ Successfully pushed to database: {nodes_created} nodes, {edges_created} edges")
        except Exception as e:
            print(f"⚠️  Database push failed: {e}")
            print(f"   Data has been saved to disk successfully at: {self.storage_dir}")
            if method == 'client':
                print(f"   Try setting QUERY_EXECUTION_METHOD=subprocess for Mac compatibility")
    
    def process(self, local_graph: LocalGraph) -> LocalGraph:
        """
        Process the LocalGraph by pushing to database and saving to disk.
        
        Args:
            local_graph: The LocalGraph instance to process
            
        Returns:
            The LocalGraph (passed through)
        """
        print("Starting database dispatch...")
        
        # Save to disk
        disk_path = self._save_to_disk(local_graph)
        
        # Push to database
        self._push_to_database(local_graph)
        
        print("Database dispatch completed successfully")
        
        # Pass to next flow in chain
        return self._call_next(local_graph)
