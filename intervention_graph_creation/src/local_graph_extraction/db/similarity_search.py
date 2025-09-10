import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import load_settings
from intervention_graph_creation.src.local_graph_extraction.core.node import \
    GraphNode
from intervention_graph_creation.src.local_graph_extraction.db.ai_safety_graph import \
    AISafetyGraph

SETTINGS = load_settings()

# Update port to match the running FalkorDB instance (from docker ps)
FALKORDB_PORT = 6379  # Using default port as shown in docker ps output

class SimilaritySearch:
    """
    Implements similarity search functionality for the FalkorDB graph database.
    
    This class provides three levels of similarity search capabilities:
    1. Native Vector Search: Uses FalkorDB's vector index (if available)
    2. Cypher-based similarity: Uses FalkorDB's cosine similarity function (if available)
    3. Pure Python similarity: Performs similarity calculations in Python (always available)
    
    The class will automatically fall back to the next method if a more advanced one fails.
    """
    
    def __init__(self, host='localhost', port=None):
        # Use the provided port or fallback to the one in settings
        self.port = port if port is not None else SETTINGS.falkordb.port
        self.host = host
        self.graph_name = SETTINGS.falkordb.graph
        self.db = None
        self.embedding_model = None
        self.has_vector_index = False
        self.has_vector_functions = False
        self.node_embeddings_cache = {}  # Cache for embeddings to avoid re-fetching
        
        try:
            print(f"Attempting to connect to FalkorDB on {self.host}:{self.port}...")
            self.db = FalkorDB(host=self.host, port=self.port)
            # Test connection by trying to select the graph
            # This is a very basic operation that should work if connected
            try:
                g = self.db.select_graph(self.graph_name)
                # Try a simple query to verify the connection is working
                g.query("RETURN 1 as test")
                print(f" Successfully connected to FalkorDB on {self.host}:{self.port}")
                
                # Check for vector capabilities and try to create the index if needed
                try:
                    # First try to create the vector index if it doesn't exist
                    try:
                        # Create a vector index on the NODE label and embedding property
                        create_index_query = """
                        CALL db.idx.vector.create('NODE', 'embedding', 'COSINE', 1024)
                        """  # Using 1024 dimensions for bge-large-en-v1.5
                        print(" Attempting to create vector index on NODE.embedding...")
                        g.query(create_index_query)
                        print(" Vector index created successfully")
                        self.has_vector_index = True
                    except Exception as e:
                        if "Index already exists" in str(e):
                            print(" Vector index already exists")
                            self.has_vector_index = True
                        else:
                            print(f" Could not create vector index: {e}")
                            
                    # Now test if the vector index is functional
                    test_vector = [0.0] * 1024  # Create a zero vector for bge-large-en-v1.5's 1024 dimensions
                    test_query = """
                    CALL db.idx.vector.queryNodes('NODE', 'embedding', 1, $vector)
                    YIELD node, score
                    RETURN COUNT(node) as count
                    """
                    # This will fail immediately if vector index functionality isn't available
                    g.query(test_query, {"vector": test_vector})
                    self.has_vector_index = True
                    print(" Vector search capabilities: Available (db.idx.vector.queryNodes)")
                except Exception as e:
                    print(" Vector search API not detected. Using fallback methods.")
                    print(f" Error: {str(e)}")
                    self.has_vector_index = False
                
                # We don't need the gds.similarity functions if we have the vector index
                self.has_vector_functions = self.has_vector_index
                
                if not self.has_vector_index:
                    print(" Using pure Python similarity calculations")
                    
                # Check if we can at least find nodes with embeddings
                try:
                    test_query = """
                    MATCH (n:NODE) 
                    WHERE n.embedding IS NOT NULL 
                    RETURN COUNT(n) as count
                    """
                    result = g.ro_query(test_query)
                    count = result.result_set[0][0] if result.result_set else 0
                    print(f" Found {count} nodes with embeddings")
                except Exception as e:
                    print(f" Could not count nodes with embeddings: {e}")
                
            except Exception as graph_error:
                print(f" Connected to Redis but graph {self.graph_name} may not exist: {graph_error}")
                print(" Will attempt to create the graph if needed during operations")
        except Exception as e:
            print(f" Error connecting to FalkorDB: {e}")
            print("\nPossible reasons for connection failure:")
            print("1. FalkorDB is not running - start it with:")
            print(f"   docker run -p {self.port}:6379 -p 3000:3000 -it --rm falkordb/falkordb:latest")
            print("2. FalkorDB is running on a different port - check the port and try again")
            print("3. Docker service is not running - start Docker and try again")
            print("\nContinuing in offline mode with limited functionality...")

    def _load_embedding_model(self):
        """Load the embedding model if it's not already loaded"""
        if self.embedding_model is None:
            print("Loading embedding model: BAAI/bge-large-en-v1.5")
            self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        return self.embedding_model

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text"""
        model = self._load_embedding_model()
        return model.encode(text, convert_to_numpy=True)

    def check_vector_index(self) -> bool:
        """Check if vector index exists and create it if it doesn't"""
        # Early return if we already know vector operations aren't supported
        if not self.has_vector_index:
            return False
            
        if self.db is None:
            print("Cannot check vector index: Not connected to FalkorDB")
            return False
            
        try:
            g = self.db.select_graph(self.graph_name)
            
            # First, try to create a vector index if it doesn't exist
            try:
                # Create a vector index on the NODE label and embedding property
                create_index_query = """
                CALL db.idx.vector.create('NODE', 'embedding', 'COSINE', 1024)
                """
                print("Attempting to create vector index on NODE.embedding...")
                g.query(create_index_query)
                print("Vector index created or already exists")
                self.has_vector_index = True
            except Exception as e:
                if "Index already exists" in str(e):
                    print("Vector index already exists")
                    self.has_vector_index = True
                else:
                    print(f"Could not create vector index: {e}")
            
            # Now verify the index works by testing with a vector query
            try:
                test_query = """
                MATCH (n:NODE)
                WHERE n.embedding IS NOT NULL
                WITH n LIMIT 1
                RETURN COUNT(n) as count
                """
                
                # Just check if we can access at least one node with embeddings
                result = g.ro_query(test_query)
                count = result.result_set[0][0] if result.result_set else 0
                
                if count > 0:
                    # Try a vector index specific command
                    try:
                        # Simple test query that will fail if vector index not available
                        test_vector = [0.0] * 1024  # Create a zero vector with 1024 dimensions
                        test_query = """
                        CALL db.idx.vector.queryNodes('NODE', 'embedding', 1, $vector)
                        YIELD node, score
                        RETURN COUNT(node) as count
                        """
                        g.query(test_query, {"vector": test_vector})
                        self.has_vector_index = True
                        print("Vector index is functional")
                        return True
                    except Exception as e:
                        # Vector specific command failed
                        print(f"Vector index query failed: {e}")
                        self.has_vector_index = False
                        return False
                else:
                    print("No nodes with embeddings found in database")
                    return False
            except Exception as e:
                # General query failed
                print(f"Error testing vector index: {e}")
                self.has_vector_index = False
                return False
        except Exception as e:
            # Graph selection failed
            print(f"Error selecting graph: {e}")
            self.has_vector_index = False
            return False

    def find_similar_nodes(self, text: str = None, node_name: str = None, 
                          embedding: np.ndarray = None, top_k: int = 10, 
                          threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Find nodes similar to a given text, node name, or embedding
        
        Args:
            text: Text to find similar nodes for
            node_name: Name of the node to find similar nodes for
            embedding: Embedding vector to find similar nodes for
            top_k: Number of similar nodes to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar nodes with their similarity scores
        """
        if self.db is None:
            print("Cannot perform similarity search: Not connected to FalkorDB")
            if text:
                print(f"Would have searched for text: '{text}'")
            elif node_name:
                print(f"Would have searched for node: '{node_name}'")
            return []
            
        try:
            g = self.db.select_graph(self.graph_name)
        except Exception as e:
            print(f"Error selecting graph '{self.graph_name}': {e}")
            return []
        
        # Check that we have at least one query method
        if text is None and node_name is None and embedding is None:
            raise ValueError("Must provide either text, node_name, or embedding")
            
        # Get query embedding
        query_embedding = None
        if embedding is not None:
            query_embedding = embedding
        elif node_name is not None:
            # Get embedding for existing node
            result = g.ro_query("MATCH (n {name: $name}) RETURN n.embedding", {"name": node_name})
            if result.result_set and result.result_set[0][0] is not None:
                query_embedding = np.array(result.result_set[0][0])
            else:
                raise ValueError(f"Node '{node_name}' not found or has no embedding")
        elif text is not None:
            # Generate embedding for text
            query_embedding = self.get_embedding(text)
        
        # Skip vector methods entirely if we already know they're not available
        # This prevents the error messages from showing up repeatedly
        if self.has_vector_index:
            try:
                # Try the vector index method first (fastest)
                return self._vector_index_search(query_embedding, top_k, threshold)
            except Exception:
                # Quietly disable vector index for future calls
                self.has_vector_index = False
        
        if self.has_vector_functions:
            try:
                # Try the vector function method second
                return self._fallback_similarity_search(query_embedding, top_k, threshold)
            except Exception:
                # Quietly disable vector functions for future calls
                self.has_vector_functions = False
        
        # Try pure Python similarity search if we have embeddings
        try:
            return self._pure_python_similarity_search(query_embedding, top_k, threshold)
        except Exception as e:
            print(f"Pure Python similarity search failed: {e}")
            # As a last resort, try text-based search
            if text:
                return self._basic_keyword_search(text, top_k)
            else:
                return []
    
    def _vector_index_search(self, query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """Perform similarity search using FalkorDB vector index"""
        g = self.db.select_graph(self.graph_name)
        
        # Convert embedding to list for Cypher query
        emb_list = query_embedding.tolist()
        
        # Ensure vector index exists
        self.check_vector_index()
        
        # Query for similar nodes using the correct syntax for your FalkorDB instance
        cypher = """
        CALL db.idx.vector.queryNodes('NODE', 'embedding', $top_k, $embedding)
        YIELD node, score
        WHERE score > $threshold AND node.is_tombstone = false
        RETURN node.name as name, 
               node.type as type,
               node.description as description,
               score as similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        result = g.query(cypher, {
            "embedding": emb_list,
            "top_k": top_k,
            "threshold": threshold
        })
        
        similar_nodes = []
        for row in result.result_set:
            similar_nodes.append({
                "name": row[0],
                "type": row[1],
                "description": row[2],
                "similarity": row[3]
            })
        
        return similar_nodes

    def _fallback_similarity_search(self, query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """Fallback similarity search using regular Cypher when vector index fails"""
        g = self.db.select_graph(self.graph_name)
        print("Using fallback similarity search method...")
        
        # This is slower but doesn't require vector index
        cypher = """
        MATCH (n:NODE)
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(n.embedding, $embedding) AS score
        WHERE score > $threshold
        RETURN n.name as name, 
               n.type as type,
               n.description as description,
               score as similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        result = g.query(cypher, {
            "embedding": query_embedding.tolist(),
            "top_k": top_k,
            "threshold": threshold
        })
        
        similar_nodes = []
        for row in result.result_set:
            similar_nodes.append({
                "name": row[0],
                "type": row[1],
                "description": row[2],
                "similarity": row[3]
            })
        
        return similar_nodes
        
    def _pure_python_similarity_search(self, query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """
        Pure Python implementation of similarity search using cosine similarity
        Calculates similarity in Python rather than in the database
        
        This method will work even if FalkorDB doesn't support vector search functions
        """
        print("Using pure Python similarity search...")
        
        if self.db is None:
            print("Cannot perform similarity search: Not connected to FalkorDB")
            return []
            
        g = self.db.select_graph(self.graph_name)
        
        # Get all nodes with embeddings
        if not self.node_embeddings_cache:
            print("Fetching all node embeddings and literature connections (this will be cached for future searches)...")
            # First get basic node data and embeddings
            cypher = """
            MATCH (n:NODE)
            WHERE n.embedding IS NOT NULL AND 
                  (n.is_tombstone = false OR n.is_tombstone IS NULL)
            RETURN n.name as name, 
                   n.type as type,
                   n.description as description,
                   n.embedding as embedding
            """
            
            result = g.query(cypher)
            
            # Process and cache node embeddings
            for row in result.result_set:
                name = str(row[0])
                node_type = str(row[1]) if row[1] else ""
                description = str(row[2]) if row[2] else ""
                embedding = row[3]
                
                if embedding:  # Only cache nodes with embeddings
                    self.node_embeddings_cache[name] = {
                        "name": name,
                        "type": node_type, 
                        "description": description,
                        "embedding": np.array(embedding),
                        "literature": []  # Will store connected literature sources
                    }
            
            # Now get literature connections for each node
            if self.node_embeddings_cache:
                try:
                    # Find literature sources connected to each node
                    lit_cypher = """
                    MATCH (n:NODE)-[r]-(l:LITERATURE)
                    WHERE n.name IS NOT NULL AND l.title IS NOT NULL
                    RETURN n.name as node_name, 
                           l.title as lit_title,
                           l.url as lit_url,
                           l.authors as lit_authors,
                           l.year as lit_year,
                           type(r) as relation_type
                    """
                    
                    lit_result = g.query(lit_cypher)
                    
                    # Add literature to the appropriate nodes
                    for row in lit_result.result_set:
                        node_name = str(row[0])
                        if node_name in self.node_embeddings_cache:
                            lit_info = {
                                "title": str(row[1]) if row[1] else "",
                                "url": str(row[2]) if row[2] else "",
                                "authors": str(row[3]) if row[3] else "",
                                "year": str(row[4]) if row[4] else "",
                                "relation": str(row[5]) if row[5] else ""
                            }
                            self.node_embeddings_cache[node_name]["literature"].append(lit_info)
                except Exception as e:
                    print(f"Warning: Could not fetch literature connections: {e}")
            
            print(f"Cached embeddings for {len(self.node_embeddings_cache)} nodes")
        
        # Calculate cosine similarity for each node
        similar_nodes = []
        for name, node_data in self.node_embeddings_cache.items():
            # Calculate cosine similarity
            node_embedding = node_data["embedding"]
            
            # Make sure embeddings are normalized
            query_norm = np.linalg.norm(query_embedding)
            node_norm = np.linalg.norm(node_embedding)
            
            if query_norm == 0 or node_norm == 0:
                similarity = 0
            else:
                # Cosine similarity = dot product of normalized vectors
                similarity = np.dot(query_embedding, node_embedding) / (query_norm * node_norm)
            
            # Add nodes that meet the threshold
            if similarity >= threshold:
                similar_nodes.append({
                    "name": node_data["name"],
                    "type": node_data["type"],
                    "description": node_data["description"],
                    "similarity": float(similarity),  # Convert from numpy type to native Python float
                    "literature": node_data.get("literature", [])  # Include literature connections
                })
        
        # Sort by similarity and limit to top_k
        similar_nodes.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_nodes[:top_k]

    def _basic_keyword_search(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Basic keyword search as a last resort when vector methods fail
        This uses text matching without embeddings
        """
        if text is None:
            print("Cannot perform keyword search: No text provided")
            return []
            
        print("Using basic keyword search method (no embeddings)...")
        
        try:
            # Attempt to get all nodes and filter in Python instead
            # This avoids using regex operators not supported by FalkorDB
            print("Fetching all nodes for keyword filtering...")
            
            g = self.db.select_graph(self.graph_name)
            
            # Get all non-tombstone nodes
            cypher = """
            MATCH (n:NODE)
            WHERE n.is_tombstone = false OR n.is_tombstone IS NULL
            RETURN n.name as name, 
                   n.type as type,
                   n.description as description
            """
            
            result = g.query(cypher)
            
            # Process search terms
            search_terms = [term.lower() for term in text.lower().split() if len(term) > 2]
            
            # Filter nodes based on search terms
            similar_nodes = []
            for row in result.result_set:
                name = str(row[0]).lower() if row[0] else ""
                node_type = str(row[1]) if row[1] else ""
                description = str(row[2]).lower() if row[2] else ""
                original_name = str(row[0]) if row[0] else ""
                
                # Check if any search term is in name or description
                matches = 0
                for term in search_terms:
                    if term in name or term in description:
                        matches += 1
                
                # Calculate a simple similarity score based on matches
                if matches > 0:
                    similarity = matches / len(search_terms)
                    similar_nodes.append({
                        "name": original_name,
                        "type": node_type,
                        "description": row[2],
                        "similarity": similarity
                    })
            
            # Sort by similarity and limit to top_k
            similar_nodes.sort(key=lambda x: x["similarity"], reverse=True)
            result_nodes = similar_nodes[:top_k]
            
            # If no results found with strict matching, try a more lenient approach
            if not result_nodes:
                print("No exact matches found, trying with more lenient matching...")
                for row in result.result_set:
                    name = str(row[0]).lower() if row[0] else ""
                    node_type = str(row[1]) if row[1] else ""
                    description = str(row[2]).lower() if row[2] else ""
                    original_name = str(row[0]) if row[0] else ""
                    
                    # Try matching word parts
                    matches = 0
                    for term in search_terms:
                        # Count partial matches in name or description
                        if (len(term) >= 4 and 
                            (any(term[:len(term)-1] in word for word in name.split()) or
                             any(term[:len(term)-1] in word for word in description.split()))):
                            matches += 0.5  # Partial match gets half weight
                    
                    if matches > 0:
                        similarity = matches / len(search_terms)
                        similar_nodes.append({
                            "name": original_name,
                            "type": node_type,
                            "description": row[2],
                            "similarity": similarity
                        })
                
                # Sort and limit results again
                similar_nodes.sort(key=lambda x: x["similarity"], reverse=True)
                result_nodes = similar_nodes[:top_k]
            
            # Add literature information to each node
            if result_nodes:
                try:
                    # Get literature for all result nodes in a single query
                    node_names = [node["name"] for node in result_nodes]
                    lit_query = """
                    MATCH (n:NODE)-[r]-(l:LITERATURE)
                    WHERE n.name IN $names AND l.title IS NOT NULL
                    RETURN n.name as node_name, 
                           l.title as lit_title,
                           l.url as lit_url,
                           l.authors as lit_authors,
                           l.year as lit_year,
                           type(r) as relation_type
                    """
                    
                    lit_result = g.query(lit_query, {"names": node_names})
                    
                    # Create a map of node_name -> list of literature
                    lit_map = {}
                    for row in lit_result.result_set:
                        node_name = str(row[0])
                        if node_name not in lit_map:
                            lit_map[node_name] = []
                        
                        lit_map[node_name].append({
                            "title": str(row[1]) if row[1] else "",
                            "url": str(row[2]) if row[2] else "",
                            "authors": str(row[3]) if row[3] else "",
                            "year": str(row[4]) if row[4] else "",
                            "relation": str(row[5]) if row[5] else ""
                        })
                    
                    # Add literature to each node
                    for node in result_nodes:
                        node["literature"] = lit_map.get(node["name"], [])
                except Exception as lit_error:
                    print(f"Warning: Could not fetch literature connections: {lit_error}")
            
            return result_nodes
            
        except Exception as e:
            print(f"Basic keyword search failed: {e}")
            # Last resort, return empty list
            print("Unable to perform search due to database errors")
            return []

    def find_clusters(self, threshold: float = 0.85, min_cluster_size: int = 2) -> List[List[Dict[str, Any]]]:
        """
        Find clusters of similar nodes in the graph
        
        Args:
            threshold: Similarity threshold (0-1)
            min_cluster_size: Minimum number of nodes in a cluster
            
        Returns:
            List of clusters, where each cluster is a list of node dictionaries
        """
        if self.db is None:
            print("Cannot find clusters: Not connected to FalkorDB")
            return []
            
        try:
            g = self.db.select_graph(self.graph_name)
            
            # Try methods in order of preference
            if self.has_vector_index:
                try:
                    # First try using the vector-based clustering
                    return self._vector_based_clustering(threshold, min_cluster_size)
                except Exception as e:
                    print(f"Vector-based clustering failed: {e}")
                    self.has_vector_index = False
            
            # Try pure Python clustering which uses our embeddings
            try:
                print("Using pure Python clustering...")
                return self._pure_python_clustering(threshold, min_cluster_size)
            except Exception as e:
                print(f"Pure Python clustering failed: {e}")
                print("Falling back to basic text clustering...")
                return self._basic_text_clustering(min_cluster_size)
                
        except Exception as e:
            print(f"Error finding clusters: {e}")
            return []
            
    def _pure_python_clustering(self, threshold: float, min_cluster_size: int) -> List[List[Dict[str, Any]]]:
        """
        Find clusters using pure Python similarity calculations
        This works without requiring vector functions in FalkorDB
        """
        # Ensure we have embeddings cached
        if not self.node_embeddings_cache:
            # This will populate the cache
            self._pure_python_similarity_search(
                self.get_embedding("dummy text to initialize cache"), 
                1, 0
            )
            
        # Skip if we still don't have embeddings
        if not self.node_embeddings_cache:
            print("No embeddings found for clustering")
            return []
        
        clusters = []
        processed_nodes = set()
        
        # For each node, find similar nodes
        for seed_name, seed_data in self.node_embeddings_cache.items():
            if seed_name in processed_nodes:
                continue
                
            seed_embedding = seed_data["embedding"]
            
            # Find similar nodes using the seed embedding
            similar_nodes = self._pure_python_similarity_search(seed_embedding, 100, threshold)
            
            # Only keep clusters with enough nodes
            if len(similar_nodes) >= min_cluster_size:
                # Add self to beginning (may not be included if threshold is high)
                if not any(n["name"] == seed_name for n in similar_nodes):
                    similar_nodes.insert(0, {
                        "name": seed_data["name"],
                        "type": seed_data["type"],
                        "description": seed_data["description"],
                        "similarity": 1.0
                    })
                
                clusters.append(similar_nodes)
                
                # Mark all nodes in this cluster as processed
                for node in similar_nodes:
                    processed_nodes.add(node["name"])
        
        return clusters
            
    def _vector_based_clustering(self, threshold: float, min_cluster_size: int) -> List[List[Dict[str, Any]]]:
        """Find clusters using vector similarity"""
        g = self.db.select_graph(self.graph_name)
        
        # Ensure vector index exists
        self.check_vector_index()
        
        # Query for clusters of similar nodes using correct syntax for your FalkorDB instance
        cypher = """
        MATCH (seed:NODE) 
        WHERE seed.embedding IS NOT NULL AND seed.is_tombstone = false
        CALL db.idx.vector.queryNodes('NODE', 'embedding', 20, seed.embedding)
        YIELD node, score
        WHERE score > $threshold AND node <> seed
        WITH seed, collect(node) as similar_nodes
        WHERE size(similar_nodes) >= $min_cluster_size
        RETURN seed.name as seed_name, 
               seed.type as seed_type,
               seed.description as seed_description,
               [node IN similar_nodes | {
                   name: node.name,
                   type: node.type,
                   description: node.description,
                   similarity: score
               }] as cluster
        """
        
        result = g.query(cypher, {
            "threshold": threshold,
            "min_cluster_size": min_cluster_size - 1  # -1 because we're not counting the seed node
        })
        
        clusters = []
        for row in result.result_set:
            seed_node = {
                "name": row[0],
                "type": row[1],
                "description": row[2],
                "similarity": 1.0  # Self-similarity
            }
            
            # Add seed node to the beginning of the cluster
            cluster = [seed_node] + row[3]
            clusters.append(cluster)
        
        return clusters
        
    def _basic_text_clustering(self, min_cluster_size: int) -> List[List[Dict[str, Any]]]:
        """
        Find clusters based on common words in names and descriptions
        This is a fallback when vector methods aren't available
        """
        print("Using basic text clustering (no embeddings)...")
        
        try:
            g = self.db.select_graph(self.graph_name)
            
            # Get all nodes with simpler query
            cypher = """
            MATCH (n:NODE)
            WHERE n.is_tombstone = false OR n.is_tombstone IS NULL
            RETURN n.name, n.type, n.description
            """
            
            result = g.query(cypher)
            
            # Process nodes to extract key terms
            nodes = []
            for row in result.result_set:
                name = str(row[0]) if row[0] else ""
                node_type = str(row[1]) if row[1] else ""
                description = str(row[2]) if row[2] else ""
                
                # Extract key terms from name and description
                text = (name + " " + description).lower()
                # Remove common words and punctuation with minimal regex
                stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 'for', 'with', 'about'}
                # Split by non-alphanumeric characters
                words = [word for word in ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split() 
                        if len(word) > 2]
                key_terms = [w for w in words if w not in stop_words]
                
                nodes.append({
                    "name": name,
                    "type": node_type,
                    "description": description,
                    "key_terms": set(key_terms)
                })
            
            # Find clusters based on term overlap
            clusters = []
            processed_nodes = set()
            
            for i, node in enumerate(nodes):
                if node["name"] in processed_nodes:
                    continue
                    
                # Skip nodes with too few key terms
                if len(node["key_terms"]) < 3:
                    continue
                    
                cluster = [{"name": node["name"], "type": node["type"], 
                        "description": node["description"], "similarity": 1.0}]
                
                for j, other_node in enumerate(nodes):
                    if i == j or other_node["name"] in processed_nodes:
                        continue
                        
                    # Calculate term overlap (Jaccard similarity)
                    if len(other_node["key_terms"]) == 0:
                        continue
                        
                    intersection = len(node["key_terms"].intersection(other_node["key_terms"]))
                    union = len(node["key_terms"].union(other_node["key_terms"]))
                    similarity = intersection / union if union > 0 else 0
                    
                    # If similarity is high enough, add to cluster
                    if similarity > 0.3:  # Arbitrary threshold for text similarity
                        cluster.append({
                            "name": other_node["name"],
                            "type": other_node["type"],
                            "description": other_node["description"],
                            "similarity": similarity
                        })
                
                # Only keep clusters with enough nodes
                if len(cluster) >= min_cluster_size:
                    clusters.append(sorted(cluster, key=lambda x: x["similarity"], reverse=True))
                    # Mark all nodes in this cluster as processed
                    for node_dict in cluster:
                        processed_nodes.add(node_dict["name"])
            
            return clusters
        except Exception as e:
            print(f"Basic text clustering failed: {e}")
            return []

    def suggest_merges(self, threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Suggest nodes that could be merged based on high similarity
        
        Args:
            threshold: Similarity threshold (0-1), higher values mean more similar nodes
            
        Returns:
            List of suggested merges with the nodes involved
        """
        clusters = self.find_clusters(threshold=threshold, min_cluster_size=2)
        
        suggested_merges = []
        for cluster in clusters:
            # Use the first node (usually the seed node) as the primary node
            primary = cluster[0]
            
            # All other nodes in the cluster are candidates for merging
            candidates = cluster[1:]
            
            suggested_merges.append({
                "primary_node": primary,
                "merge_candidates": candidates
            })
            
        return suggested_merges

    def execute_merge(self, keep_name: str, remove_name: str) -> bool:
        """Execute a merge between two nodes"""
        try:
            graph = AISafetyGraph()
            graph.merge_nodes(keep_name, remove_name)
            print(f"Successfully merged '{remove_name}' into '{keep_name}'")
            return True
        except Exception as e:
            print(f"Error merging nodes: {e}")
            return False


def check_vector_capabilities():
    """Diagnose and report on vector search capabilities"""
    search = SimilaritySearch(port=FALKORDB_PORT)
    
    print("\n=== FalkorDB Vector Search Diagnostic ===")
    print(f"Connected to FalkorDB: {'Yes' if search.db else 'No'}")
    
    if search.db:
        try:
            g = search.db.select_graph(search.graph_name)
            print(f"Graph '{search.graph_name}' selected: Yes")
            
            # Check for available indexes
            try:
                print("\nListing available vector indexes:")
                list_index_query = """CALL db.idx.vector.list()"""
                result = g.query(list_index_query)
                
                if result.result_set and len(result.result_set) > 0:
                    for idx, row in enumerate(result.result_set):
                        print(f"  Index {idx+1}: {row}")
                else:
                    print("  No vector indexes found")
                    
                # Count nodes with embeddings
                count_query = """
                MATCH (n:NODE) 
                WHERE n.embedding IS NOT NULL 
                RETURN COUNT(n) as count
                """
                result = g.ro_query(count_query)
                count = result.result_set[0][0] if result.result_set else 0
                print(f"\nNodes with embeddings: {count}")
                
                if count > 0:
                    # Sample an embedding to check dimensions
                    sample_query = """
                    MATCH (n:NODE) 
                    WHERE n.embedding IS NOT NULL 
                    RETURN n.name, length(n.embedding) as dimensions
                    LIMIT 1
                    """
                    result = g.ro_query(sample_query)
                    if result.result_set and len(result.result_set) > 0:
                        node_name, dimensions = result.result_set[0]
                        print(f"Sample node: {node_name}")
                        print(f"Embedding dimensions: {dimensions}")
                    
                # Try to create a vector index
                try:
                    create_index_query = """
                    CALL db.idx.vector.create('NODE', 'embedding', 'COSINE', 1024)
                    """
                    g.query(create_index_query)
                    print("\nVector index created successfully")
                except Exception as e:
                    if "Index already exists" in str(e):
                        print("\nVector index already exists")
                    else:
                        print(f"\nCould not create vector index: {e}")
            except Exception as e:
                print(f"Error checking vector capabilities: {e}")
        except Exception as e:
            print(f"Error selecting graph: {e}")
    
    print("\nVector search ready: ", "Yes" if search.has_vector_index else "No")
    print("Fallback to Python similarity calculations:", "Yes" if not search.has_vector_index else "No")
    print("=============================================")
    
    return search.has_vector_index

def test_similarity_search():
    """Test the similarity search functionality"""
    # First check vector capabilities
    check_vector_capabilities()
    
    # Initialize search with the correct port (change if needed)
    search = SimilaritySearch(port=FALKORDB_PORT)
    
    # Try text-based search
    test_query = "AI alignment methods for large language models"
    print(f"\nSearching for nodes similar to: '{test_query}'")
    similar_nodes = search.find_similar_nodes(text=test_query, top_k=5, threshold=0.7)
    
    if similar_nodes:
        print(f"Found {len(similar_nodes)} similar nodes:")
        for i, node in enumerate(similar_nodes):
            print(f"{i+1}. {node['name']} (Similarity: {node['similarity']:.3f})")
            print(f"   Type: {node['type']}")
            description = node['description']
            if description:
                print(f"   Description: {description[:100]}..." if len(description) > 100 else f"   Description: {description}")
    else:
        print("No similar nodes found. Try lowering the threshold or adding more nodes to the database.")

    # Try finding clusters
    print("\nLooking for clusters of similar nodes...")
    clusters = search.find_clusters(threshold=0.8, min_cluster_size=2)
    
    if clusters:
        print(f"Found {len(clusters)} clusters:")
        for i, cluster in enumerate(clusters):
            print(f"\nCluster {i+1} with {len(cluster)} nodes:")
            for j, node in enumerate(cluster):
                print(f"  {j+1}. {node['name']} (Similarity: {node['similarity']:.3f})")
    else:
        print("No clusters found. Try lowering the threshold or adding more nodes to the database.")
    
    # Try suggesting merges
    print("\nSuggesting nodes to merge...")
    merges = search.suggest_merges(threshold=0.9)
    
    if merges:
        print(f"Found {len(merges)} potential merges:")
        for i, merge in enumerate(merges):
            primary = merge['primary_node']
            candidates = merge['merge_candidates']
            print(f"\nMerge {i+1}: Keep '{primary['name']}' and merge:")
            for j, candidate in enumerate(candidates):
                print(f"  {j+1}. '{candidate['name']}' (Similarity: {candidate['similarity']:.3f})")
    else:
        print("No merge suggestions found. Try lowering the threshold.")


if __name__ == "__main__":
    test_similarity_search()
