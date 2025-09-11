import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import load_settings
from intervention_graph_creation.src.local_graph_extraction.core.node import \
    GraphNode
from intervention_graph_creation.src.local_graph_extraction.db.ai_safety_graph import \
    AISafetyGraph

SETTINGS = load_settings()
# FalkorDB connection settings
FALKORDB_HOST = SETTINGS.falkordb.host
FALKORDB_PORT = SETTINGS.falkordb.port
FALKORDB_GRAPH = SETTINGS.falkordb.graph

class SimilaritySearch:
    """
    Implements similarity search functionality for FalkorDB graph database.
    
    This class provides three levels of similarity search capabilities:
    1. Redis/FalkorDB-based similarity: Uses FalkorDB's vector functionality (if available)
    2. Redis/FalkorDB-based manual similarity: Uses manual cosine similarity in Cypher (if vector functions unavailable)
    3. Pure Python similarity: Performs similarity calculations in Python (always available)
    
    The class will automatically fall back to the next method if a more advanced one fails.
    """
    
    def __init__(self, host=FALKORDB_HOST, port=FALKORDB_PORT, graph_name=FALKORDB_GRAPH):
        # FalkorDB connection parameters
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.db = None
        self.embedding_model = None
        self.has_vector_index = False
        self.has_vector_functions = False
        self.node_embeddings_cache = {}  # Cache for embeddings to avoid re-fetching
        
        try:
            # Import FalkorDB here to make it optional
            try:
                from falkordb import FalkorDB
            except ImportError:
                print("FalkorDB Python client not installed. Run: pip install falkordb")
                return
                
            print(f"Attempting to connect to FalkorDB on {self.host}:{self.port}...")
            self.db = FalkorDB(host=self.host, port=self.port)
            
            # Test connection by selecting the graph
            try:
                g = self.db.select_graph(self.graph_name)
                # Try a simple query to verify the connection is working
                result = g.query("RETURN 1")
                if result.result_set and result.result_set[0][0] == 1:
                    print(f" Successfully connected to FalkorDB graph '{self.graph_name}'")
                    
                    # Check for vector similarity capabilities
                    try:
                        # Check if we can at least find nodes with embeddings
                        try:
                            test_query = """
                            MATCH (n) 
                            WHERE n.embedding IS NOT NULL 
                            RETURN COUNT(n)
                            """
                            result = g.query(test_query)
                            count = result.result_set[0][0] if result.result_set else 0
                            print(f" Found {count} nodes with embeddings")
                            
                            if count > 0:
                                # Try to test if vector operations are available in FalkorDB
                                try:
                                    # Try a manual cosine similarity calculation to see if it works
                                    manual_cosine_query = """
                                    MATCH (n)
                                    WHERE n.embedding IS NOT NULL
                                    WITH n LIMIT 1
                                    MATCH (m)
                                    WHERE m.embedding IS NOT NULL AND id(n) <> id(m)
                                    WITH n, m LIMIT 1
                                    RETURN m.name, n.name
                                    """
                                    
                                    # Just fetch the nodes first, we'll do calculation in Python
                                    cosine_result = g.query(manual_cosine_query)
                                    
                                    if cosine_result.result_set and len(cosine_result.result_set) > 0:
                                        # We found two nodes with embeddings, now get the embeddings
                                        get_emb_query = """
                                        MATCH (n {name: $name})
                                        RETURN n.embedding
                                        """
                                        # Get first node embedding
                                        first_node_name = cosine_result.result_set[0][1]
                                        emb1_result = g.query(get_emb_query, {"name": first_node_name})
                                        
                                        # Get second node embedding
                                        second_node_name = cosine_result.result_set[0][0]
                                        emb2_result = g.query(get_emb_query, {"name": second_node_name})
                                        
                                        if (emb1_result.result_set and emb2_result.result_set and
                                            emb1_result.result_set[0][0] is not None and
                                            emb2_result.result_set[0][0] is not None):
                                            
                                            # Try to use manual cosine calculation
                                            try:
                                                # Try calculating cosine similarity in Python
                                                emb1 = emb1_result.result_set[0][0]
                                                emb2 = emb2_result.result_set[0][0]
                                                
                                                # Convert to numpy arrays
                                                emb1_arr = np.array(emb1)
                                                emb2_arr = np.array(emb2)
                                                
                                                # Calculate cosine similarity
                                                dot_product = np.dot(emb1_arr, emb2_arr)
                                                norm1 = np.linalg.norm(emb1_arr)
                                                norm2 = np.linalg.norm(emb2_arr)
                                                
                                                if norm1 > 0 and norm2 > 0:
                                                    similarity = dot_product / (norm1 * norm2)
                                                    self.has_vector_functions = True
                                                    print(f" Vector similarity: Available (manual cosine calculation)")
                                                    print(f" Sample similarity between '{first_node_name}' and '{second_node_name}': {similarity:.4f}")
                                                else:
                                                    self.has_vector_functions = False
                                                    print(" Vector similarity: Unavailable (embeddings have zero norm)")
                                            except Exception as e:
                                                print(f" Vector calculation failed: {e}")
                                                self.has_vector_functions = False
                                        else:
                                            print(" Could not retrieve embeddings from nodes")
                                            self.has_vector_functions = False
                                    else:
                                        print(" Could not find two nodes with embeddings to test similarity")
                                        self.has_vector_functions = False
                                except Exception as e:
                                    print(" Vector function check failed. Using fallback methods.")
                                    print(f" Error: {str(e)}")
                                    self.has_vector_index = False
                                    self.has_vector_functions = False
                            else:
                                print(" No nodes with embeddings found. Vector search will be unavailable.")
                                self.has_vector_index = False
                                self.has_vector_functions = False
                        except Exception as e:
                            print(f" Could not count nodes with embeddings: {e}")
                            self.has_vector_index = False
                            self.has_vector_functions = False
                    
                    except Exception as e:
                        print(" Vector similarity capabilities check failed. Using fallback methods.")
                        print(f" Error: {str(e)}")
                        self.has_vector_index = False
                        self.has_vector_functions = False
                    
                    if not self.has_vector_index:
                        print(" Using pure Python similarity calculations")
                
            except Exception as graph_error:
                print(f" Connected to FalkorDB but query failed: {graph_error}")
                print(" Will attempt to recover during operations")
        except Exception as e:
            print(f" Error connecting to FalkorDB: {e}")
            print("\nPossible reasons for connection failure:")
            print(f"1. FalkorDB is not running on {self.host}:{self.port}")
            print("2. FalkorDB graph does not exist")
            print("3. Network connectivity issue")
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
        """Check if cosine similarity function is available in Neo4j"""
        # Early return if we already know vector operations aren't supported
        if not self.has_vector_index:
            return False
            
        if self.db is None:
            print("Cannot check vector capabilities: Not connected to Neo4j")
            return False
            
        try:
            if self.session is None:
                self.session = self.db.session()
            
            # First, check if there are nodes with embeddings
            try:
                test_query = """
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                RETURN COUNT(n) as count
                """
                
                # Just check if we can access at least one node with embeddings
                result = self.session.run(test_query)
                record = result.single()
                count = record["count"] if record else 0
                
                if count > 0:
                    # Try Neo4j vector capabilities first (Neo4j 5.15+ or with Graph Data Science plugin)
                    try:
                        # Check if Neo4j has vector capabilities
                        vector_test_query = """
                        CALL dbms.functions() YIELD name
                        WHERE name CONTAINS 'vector' OR name CONTAINS 'similarity'
                        RETURN count(*) > 0 as has_vector
                        """
                        vector_result = self.session.run(vector_test_query)
                        vector_record = vector_result.single()
                        
                        if vector_record and vector_record["has_vector"]:
                            # Try to use the similarity function directly
                            try:
                                # Get a sample embedding
                                sample_query = """
                                MATCH (n)
                                WHERE n.embedding IS NOT NULL
                                RETURN n.embedding as embedding
                                LIMIT 1
                                """
                                sample_result = self.session.run(sample_query)
                                sample_record = sample_result.single()
                                
                                if sample_record:
                                    sample_emb = sample_record["embedding"]
                                    
                                    # Test with the similarity function
                                    similarity_test = """
                                    WITH $emb as emb1, $emb as emb2
                                    RETURN gds.similarity.cosine(emb1, emb2) as similarity
                                    """
                                    sim_result = self.session.run(similarity_test, {"emb": sample_emb})
                                    sim_record = sim_result.single()
                                    
                                    if sim_record and sim_record["similarity"] is not None:
                                        self.has_vector_index = True
                                        self.has_vector_functions = True
                                        print("Neo4j vector functions available: gds.similarity.cosine")
                                        return True
                            except Exception:
                                # Function exists but failed, maybe syntax is different
                                pass
                    
                    except Exception:
                        # No explicit vector functions, try manual calculation
                        pass
                    
                    # Try manual cosine calculation as fallback
                    try:
                        # Get a sample embedding
                        sample_query = """
                        MATCH (n)
                        WHERE n.embedding IS NOT NULL
                        RETURN n.embedding as embedding
                        LIMIT 1
                        """
                        sample_result = self.session.run(sample_query)
                        sample_record = sample_result.single()
                        
                        if sample_record:
                            sample_emb = sample_record["embedding"]
                            
                            # Test manual cosine similarity calculation
                            manual_query = """
                            WITH $emb as emb1, $emb as emb2
                            WITH 
                                reduce(dot = 0.0, i in range(0, size(emb1)-1) | 
                                    dot + emb1[i] * emb2[i]) as dotProduct,
                                sqrt(reduce(norm1 = 0.0, i in range(0, size(emb1)-1) | 
                                    norm1 + emb1[i] * emb1[i])) as norm1,
                                sqrt(reduce(norm2 = 0.0, i in range(0, size(emb2)-1) | 
                                    norm2 + emb2[i] * emb2[i])) as norm2
                            RETURN dotProduct / (norm1 * norm2) as similarity
                            """
                            manual_result = self.session.run(manual_query, {"emb": sample_emb})
                            manual_record = manual_result.single()
                            
                            if manual_record and manual_record["similarity"] is not None:
                                self.has_vector_functions = True
                                print("Manual cosine similarity calculation is available")
                                return True
                    except Exception as e:
                        print(f"Manual similarity calculation failed: {e}")
                        self.has_vector_index = False
                        self.has_vector_functions = False
                        return False
                else:
                    print("No nodes with embeddings found in database")
                    return False
            except Exception as e:
                # General query failed
                print(f"Error testing vector capabilities: {e}")
                self.has_vector_index = False
                self.has_vector_functions = False
                return False
        except Exception as e:
            # Session creation failed
            print(f"Error creating Neo4j session: {e}")
            self.has_vector_index = False
            self.has_vector_functions = False
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
        """Perform similarity search using Neo4j vector capabilities"""
        if self.session is None:
            self.session = self.db.session()
        
        # Convert embedding to list for Cypher query
        emb_list = query_embedding.tolist()
        
        # Ensure vector capabilities exist
        self.check_vector_index()
        
        # Vector search query for Neo4j
        if self.has_vector_index:
            # Use Neo4j vector operations if available (Neo4j 5.15+ or with Graph Data Science plugin)
            cypher = """
            WITH $embedding AS query_vector
            MATCH (node)
            WHERE node.embedding IS NOT NULL AND (node.is_tombstone = false OR node.is_tombstone IS NULL)
            WITH node, gds.similarity.cosine(node.embedding, query_vector) AS score
            WHERE score > $threshold
            RETURN node.name as name, 
                   node.type as type,
                   node.description as description,
                   score as similarity
            ORDER BY similarity DESC
            LIMIT $top_k
            """
        else:
            # Use manual cosine similarity calculation
            cypher = """
            WITH $embedding AS query_vector
            MATCH (node)
            WHERE node.embedding IS NOT NULL AND (node.is_tombstone = false OR node.is_tombstone IS NULL)
            WITH node, 
                reduce(dot = 0.0, i in range(0, size(query_vector)-1) | 
                    dot + query_vector[i] * node.embedding[i]) as dotProduct,
                sqrt(reduce(norm1 = 0.0, i in range(0, size(query_vector)-1) | 
                    norm1 + query_vector[i] * query_vector[i])) as norm1,
                sqrt(reduce(norm2 = 0.0, i in range(0, size(node.embedding)-1) | 
                    norm2 + node.embedding[i] * node.embedding[i])) as norm2
            WITH node, dotProduct / (norm1 * norm2) as score
            WHERE score > $threshold
            RETURN node.name as name, 
                   node.type as type,
                   node.description as description,
                   score as similarity
            ORDER BY similarity DESC
            LIMIT $top_k
            """
        
        # Execute the query
        similar_nodes = []
        try:
            result = self.session.run(cypher, {
                "embedding": emb_list,
                "threshold": threshold
            })
            
            # Process results
            for record in result:
                similar_nodes.append({
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "similarity": record["similarity"]
                })
                
        except Exception as e:
            print(f"Vector search failed: {e}")
            # Fall back to pure Python similarity
            return self._pure_python_similarity_search(query_embedding, top_k, threshold)
            
        return similar_nodes
        
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
        """Fallback similarity search using manual cosine calculation in Cypher"""
        if self.session is None:
            self.session = self.db.session()
            
        print("Using fallback similarity search method...")
        
        # This is slower but doesn't require vector index
        # Manual cosine similarity calculation in Cypher
        cypher = """
        WITH $embedding AS query_vector
        MATCH (n)
        WHERE n.embedding IS NOT NULL AND (n.is_tombstone = false OR n.is_tombstone IS NULL)
        WITH n, 
            reduce(dot = 0.0, i in range(0, size(query_vector)-1) | 
                dot + query_vector[i] * n.embedding[i]) as dotProduct,
            sqrt(reduce(norm1 = 0.0, i in range(0, size(query_vector)-1) | 
                norm1 + query_vector[i] * query_vector[i])) as norm1,
            sqrt(reduce(norm2 = 0.0, i in range(0, size(n.embedding)-1) | 
                norm2 + n.embedding[i] * n.embedding[i])) as norm2
        WITH n, dotProduct / (norm1 * norm2) as score
        WHERE score > $threshold
        RETURN n.name as name, 
               n.type as type,
               n.description as description,
               score as similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        similar_nodes = []
        try:
            result = self.session.run(cypher, {
                "embedding": query_embedding.tolist(),
                "threshold": threshold
            })
            
            # Process results
            for record in result:
                similar_nodes.append({
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "similarity": record["similarity"]
                })
        except Exception as e:
            print(f"Fallback similarity search failed: {e}")
            # Fall back to pure Python similarity
            return self._pure_python_similarity_search(query_embedding, top_k, threshold)
        
        return similar_nodes
        
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors in a consistent way
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Make sure inputs are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        
        # Check if vectors are same length
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector dimensions don't match: {vec1.shape} vs {vec2.shape}")
        
        # Calculate norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # Ensure the result is within bounds (may slightly exceed 1.0 due to floating point errors)
        return max(min(float(similarity), 1.0), -1.0)
            
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
            # Calculate cosine similarity using our unified method
            node_embedding = node_data["embedding"]
            similarity = self._calculate_cosine_similarity(query_embedding, node_embedding)
            
            # Add nodes that meet the threshold
            if similarity >= threshold:
                similar_nodes.append({
                    "name": node_data["name"],
                    "type": node_data["type"],
                    "description": node_data["description"],
                    "similarity": similarity,  # Already converted to float in _calculate_cosine_similarity
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

    def load_all_embeddings(self, force_reload=False):
        """
        Load all node embeddings from the database into the cache
        
        Args:
            force_reload: Whether to reload all embeddings even if they're already cached
        
        Returns:
            Number of embeddings loaded
        """
        if self.db is None:
            print("Cannot load embeddings: Not connected to FalkorDB")
            return 0
        
        if not force_reload and self.node_embeddings_cache:
            print(f"Using cached embeddings for {len(self.node_embeddings_cache)} nodes")
            return len(self.node_embeddings_cache)
        
        print("Loading all node embeddings from database...")
        g = self.db.select_graph(self.graph_name)
        
        # Clear the cache if we're reloading
        if force_reload:
            self.node_embeddings_cache = {}
        
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
        return len(self.node_embeddings_cache)
            
    def find_clusters(self, threshold: float = 0.85, min_cluster_size: int = 2, max_cluster_size: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Find clusters of similar nodes in the graph
        
        Args:
            threshold: Similarity threshold (0-1)
            min_cluster_size: Minimum number of nodes in a cluster
            max_cluster_size: Maximum number of nodes in a cluster
            
        Returns:
            List of clusters, where each cluster is a list of node dictionaries
        """
        if self.db is None:
            print("Cannot find clusters: Not connected to FalkorDB")
            return []
            
        try:
            g = self.db.select_graph(self.graph_name)
            
            # Pre-load all embeddings for efficiency
            self.load_all_embeddings()
            
            # Try methods in order of preference
            if self.has_vector_index:
                try:
                    # First try using the vector-based clustering
                    return self._vector_based_clustering(threshold, min_cluster_size, max_cluster_size)
                except Exception as e:
                    print(f"Vector-based clustering failed: {e}")
                    self.has_vector_index = False
            
            # Try pure Python clustering which uses our embeddings
            try:
                print("Using pure Python clustering...")
                return self._pure_python_clustering(threshold, min_cluster_size, max_cluster_size)
            except Exception as e:
                print(f"Pure Python clustering failed: {e}")
                print("Falling back to basic text clustering...")
                return self._basic_text_clustering(min_cluster_size, max_cluster_size)
                
        except Exception as e:
            print(f"Error finding clusters: {e}")
            return []
            
    def _pure_python_clustering(self, threshold: float, min_cluster_size: int, max_cluster_size: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Find clusters using pure Python similarity calculations
        This works without requiring vector functions in FalkorDB
        """
        # Pre-load all embeddings for efficiency
        self.load_all_embeddings()
            
        # Skip if we still don't have embeddings
        if not self.node_embeddings_cache:
            print("No embeddings found for clustering")
            return []
        
        print(f"Finding clusters among {len(self.node_embeddings_cache)} nodes...")
        
        # Find similar node pairs
        # This is O(n²), but unavoidable for clustering
        node_names = list(self.node_embeddings_cache.keys())
        
        # To avoid O(n³) complexity in building clusters,
        # we'll first create an adjacency list of similar nodes
        similarity_graph = {}
        
        # Initialize similarity graph
        for name in node_names:
            similarity_graph[name] = {}
            
        # Build similarity graph
        print("Computing similarity scores between all nodes...")
        total_comparisons = len(node_names) * (len(node_names) - 1) // 2
        processed = 0
        
        for i, name1 in enumerate(node_names):
            node1 = self.node_embeddings_cache[name1]
            embedding1 = node1["embedding"]
            
            for j in range(i + 1, len(node_names)):
                name2 = node_names[j]
                node2 = self.node_embeddings_cache[name2]
                embedding2 = node2["embedding"]
                
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(embedding1, embedding2)
                
                # If similarity exceeds threshold, add to graph
                if similarity >= threshold:
                    similarity_graph[name1][name2] = similarity
                    similarity_graph[name2][name1] = similarity
                
                # Update progress periodically
                processed += 1
                if processed % 10000 == 0:
                    progress = (processed / total_comparisons) * 100
                    print(f"  {progress:.1f}% complete, {processed} comparisons done")
        
        # Now find clusters using the similarity graph
        print("Building clusters from similarity graph...")
        clusters = []
        processed_nodes = set()
        
        # Sort nodes by number of connections for better cluster seeds
        sorted_nodes = sorted([(name, len(similarity_graph[name])) 
                             for name in similarity_graph], 
                             key=lambda x: x[1], reverse=True)
        
        # For each unprocessed node with connections
        for seed_name, connection_count in sorted_nodes:
            # Skip if already in a cluster
            if seed_name in processed_nodes:
                continue
            
            # Skip seeds with no connections above threshold
            if connection_count == 0:
                continue
                
            # Get similar nodes from the similarity graph
            similar_nodes = [
                {
                    "name": neighbor_name,
                    "similarity": similarity,
                    "type": self.node_embeddings_cache[neighbor_name]["type"],
                    "description": self.node_embeddings_cache[neighbor_name]["description"]
                }
                for neighbor_name, similarity in similarity_graph[seed_name].items()
                if neighbor_name not in processed_nodes
            ]
            
            # Add the seed node itself
            similar_nodes.append({
                "name": seed_name,
                "type": self.node_embeddings_cache[seed_name]["type"],
                "description": self.node_embeddings_cache[seed_name]["description"],
                "similarity": 1.0
            })
            
            # Sort by similarity and limit size
            similar_nodes.sort(key=lambda x: x["similarity"], reverse=True)
            if len(similar_nodes) > max_cluster_size:
                similar_nodes = similar_nodes[:max_cluster_size]
            
            # Only create clusters with enough nodes
            if len(similar_nodes) >= min_cluster_size:
                clusters.append(similar_nodes)
                
                # Mark all nodes as processed
                for node in similar_nodes:
                    processed_nodes.add(node["name"])
        
        print(f"Found {len(clusters)} clusters")
        return clusters
            
    def _vector_based_clustering(self, threshold: float, min_cluster_size: int, max_cluster_size: int = 10) -> List[List[Dict[str, Any]]]:
        """Find clusters using vector similarity"""
        g = self.db.select_graph(self.graph_name)
        
        # Ensure vector index exists
        self.check_vector_index()
        
        # Query for clusters of similar nodes using correct syntax for your FalkorDB instance
        # Using a larger initial search limit to find more candidate clusters
        search_limit = max(20, max_cluster_size * 2)
        
        # Updated query to use gds.similarity.cosine instead of db.idx.vector.queryNodes
        cypher = """
        MATCH (seed:NODE) 
        WHERE seed.embedding IS NOT NULL AND 
              (seed.is_tombstone = false OR seed.is_tombstone IS NULL)
        MATCH (node:NODE)
        WHERE node.embedding IS NOT NULL AND 
              node <> seed AND
              (node.is_tombstone = false OR node.is_tombstone IS NULL)
        WITH seed, node, gds.similarity.cosine(seed.embedding, node.embedding) AS score
        WHERE score > $threshold
        WITH seed, collect({node: node, score: score}) as similar_node_data
        WHERE size(similar_node_data) >= $min_cluster_size AND 
              size(similar_node_data) <= $max_cluster_size
        RETURN seed.name as seed_name, 
               seed.type as seed_type,
               seed.description as seed_description,
               [item IN similar_node_data | {
                   name: item.node.name,
                   type: item.node.type,
                   description: item.node.description,
                   similarity: item.score
               }] as cluster
        LIMIT $search_limit
        """
        
        result = g.query(cypher, {
            "threshold": threshold,
            "min_cluster_size": min_cluster_size - 1,  # -1 because we're not counting the seed node
            "max_cluster_size": max_cluster_size - 1,  # -1 because we're not counting the seed node
            "search_limit": search_limit
        })
        
        clusters = []
        processed_nodes = set()  # Keep track of nodes we've already included in clusters
        
        for row in result.result_set:
            seed_name = row[0]
            
            # Skip if this seed node is already in another cluster
            if seed_name in processed_nodes:
                continue
                
            seed_node = {
                "name": seed_name,
                "type": row[1],
                "description": row[2],
                "similarity": 1.0  # Self-similarity
            }
            
            # Filter out any nodes that are already in other clusters
            cluster_candidates = row[3]
            new_cluster = [seed_node]
            
            for candidate in cluster_candidates:
                if candidate["name"] not in processed_nodes:
                    new_cluster.append(candidate)
                    processed_nodes.add(candidate["name"])
            
            # Add seed node to processed nodes to avoid duplicate clusters
            processed_nodes.add(seed_name)
            
            # Only keep clusters that still meet the minimum size
            if len(new_cluster) >= min_cluster_size:
                clusters.append(new_cluster)
        
        return clusters
        
    def _basic_text_clustering(self, min_cluster_size: int, max_cluster_size: int = 10) -> List[List[Dict[str, Any]]]:
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
                
                # Collect similarities to all other nodes
                similarity_scores = []
                for j, other_node in enumerate(nodes):
                    if i == j or other_node["name"] in processed_nodes:
                        continue
                        
                    # Calculate term overlap (Jaccard similarity)
                    if len(other_node["key_terms"]) == 0:
                        continue
                        
                    intersection = len(node["key_terms"].intersection(other_node["key_terms"]))
                    union = len(node["key_terms"].union(other_node["key_terms"]))
                    similarity = intersection / union if union > 0 else 0
                    
                    # If similarity is high enough, add to candidates
                    if similarity > 0.3:  # Arbitrary threshold for text similarity
                        similarity_scores.append({
                            "name": other_node["name"],
                            "type": other_node["type"],
                            "description": other_node["description"],
                            "similarity": similarity
                        })
                
                # Sort by similarity and limit to max_cluster_size - 1 (to account for seed node)
                similarity_scores.sort(key=lambda x: x["similarity"], reverse=True)
                if len(similarity_scores) > (max_cluster_size - 1):
                    similarity_scores = similarity_scores[:max_cluster_size - 1]
                
                # Add candidates to cluster
                cluster.extend(similarity_scores)
                
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

    def get_node_neighborhood(self, node_name: str) -> Dict[str, Any]:
        """
        Get all edge information and immediate neighbor nodes for a given node
        
        Args:
            node_name: Name of the node to get neighborhood for
            
        Returns:
            Dictionary with node info, edges, and neighbors
        """
        if self.db is None:
            print(f"Cannot get neighborhood for node '{node_name}': Not connected to FalkorDB")
            return {"error": "Not connected to database"}
            
        try:
            g = self.db.select_graph(self.graph_name)
                
            # First get the node itself
            node_query = """
            MATCH (n {name: $name})
            RETURN n, labels(n) as labels
            """
            
            result = g.query(node_query, {"name": node_name})
            if not result.result_set or not result.result_set[0]:
                print(f"Node '{node_name}' not found")
                return {"error": f"Node '{node_name}' not found"}
                
            # Get node properties and labels
            node_row = result.result_set[0][0]
            node_labels = result.result_set[0][1] if len(result.result_set[0]) > 1 else ["NODE"]
            
            # Create a dictionary of node properties
            if isinstance(node_row, dict):
                node = node_row
            else:
                # For FalkorDB nodes (which might not be dictionaries)
                node = {"name": node_name}  # Ensure name is included
                
                # Check if it's a FalkorDB node object
                # Try different access methods to get properties
                try:
                    # If it has properties() method
                    if hasattr(node_row, "properties") and callable(getattr(node_row, "properties")):
                        props = node_row.properties()
                        if isinstance(props, dict):
                            node.update(props)
                    # If it has __dict__ attribute
                    elif hasattr(node_row, "__dict__"):
                        props = node_row.__dict__
                        if isinstance(props, dict):
                            node.update(props)
                    # Try direct attribute access for common properties
                    else:
                        for prop in ["type", "description", "embedding", "is_tombstone"]:
                            try:
                                if hasattr(node_row, prop):
                                    node[prop] = getattr(node_row, prop)
                            except:
                                pass
                except Exception as e:
                    print(f"Warning: Could not extract properties from node: {e}")
            
            # Now get all edges and neighbors
            edges_query = """
            MATCH (n {name: $name})-[r]-(neighbor)
            RETURN type(r) as rel_type, 
                   properties(r) as rel_props,
                   neighbor.name as neighbor_name,
                   labels(neighbor) as neighbor_labels
            """
            
            edges_result = g.query(edges_query, {"name": node_name})
            
            # Process results
            edges = []
            neighbors = {}
            
            for row in edges_result.result_set:
                rel_type = str(row[0]) if len(row) > 0 and row[0] else "RELATED_TO"
                rel_props = row[1] if len(row) > 1 and row[1] else {}
                neighbor_name = str(row[2]) if len(row) > 2 and row[2] else "Unknown"
                neighbor_labels = row[3] if len(row) > 3 and row[3] else ["NODE"]
                
                # Create a clean version of edge properties (excluding embedding if present)
                clean_rel_props = {k: v for k, v in rel_props.items() if k != 'embedding'}
                
                edges.append({
                    "type": rel_type,
                    "properties": clean_rel_props,
                    "target": neighbor_name
                })
                
                # Only store neighbor info once
                if neighbor_name not in neighbors:
                    # Get neighbor properties
                    neighbor_query = """
                    MATCH (n {name: $name})
                    RETURN n
                    """
                    neighbor_result = g.query(neighbor_query, {"name": neighbor_name})
                    neighbor_row = neighbor_result.result_set[0][0] if neighbor_result.result_set else None
                    
                    # Create a dictionary of neighbor properties
                    if isinstance(neighbor_row, dict):
                        neighbor_props = neighbor_row
                    else:
                        # For FalkorDB node objects (which might not be dictionaries)
                        neighbor_props = {"name": neighbor_name}
                        
                        # Check if it's a FalkorDB node object and extract properties
                        if neighbor_row is not None:
                            try:
                                # If it has properties() method
                                if hasattr(neighbor_row, "properties") and callable(getattr(neighbor_row, "properties")):
                                    props = neighbor_row.properties()
                                    if isinstance(props, dict):
                                        neighbor_props.update(props)
                                # If it has __dict__ attribute
                                elif hasattr(neighbor_row, "__dict__"):
                                    props = neighbor_row.__dict__
                                    if isinstance(props, dict):
                                        neighbor_props.update(props)
                                # Try direct attribute access for common properties
                                else:
                                    for prop in ["type", "description", "is_tombstone"]:
                                        try:
                                            if hasattr(neighbor_row, prop):
                                                neighbor_props[prop] = getattr(neighbor_row, prop)
                                        except:
                                            pass
                            except Exception as e:
                                print(f"Warning: Could not extract properties from neighbor: {e}")
                    
                    # Create a clean version of neighbor properties (excluding embedding)
                    clean_neighbor_props = {}
                    if isinstance(neighbor_props, dict):
                        clean_neighbor_props = {k: v for k, v in neighbor_props.items() if k != 'embedding'}
                    
                    neighbors[neighbor_name] = {
                        "labels": neighbor_labels,
                        "properties": clean_neighbor_props
                    }
            
            # Create the neighborhood data structure
            node_properties = {}
            if isinstance(node, dict):
                node_properties = {k: v for k, v in node.items() if k != 'embedding'}
            
            neighborhood = {
                "node": {
                    "name": node_name,
                    "labels": node_labels,
                    "properties": node_properties
                },
                "edges": edges,
                "neighbors": neighbors
            }
            
            return neighborhood
            
        except Exception as e:
            print(f"Error getting neighborhood for node '{node_name}': {e}")
            return {"error": str(e)}
    
    def get_cluster_with_neighborhoods(self, threshold: float = 0.9, max_cluster_size: int = 10) -> List[Dict[str, Any]]:
        """
        Find clusters of similar nodes and include their neighborhood information
        
        Args:
            threshold: Similarity threshold (0-1)
            max_cluster_size: Maximum size of clusters to return
            
        Returns:
            List of clusters with node and neighborhood information
        """
        # First find the clusters
        clusters = self.find_clusters(threshold=threshold, min_cluster_size=2, max_cluster_size=max_cluster_size)
        
        enriched_clusters = []
        for cluster in clusters:
            # For each node in the cluster, get its neighborhood
            nodes_with_neighborhoods = []
            for node in cluster:
                # Get basic node info
                node_info = {
                    "name": node["name"],
                    "type": node["type"],
                    "description": node["description"],
                    "similarity": node["similarity"]
                }
                
                # Get neighborhood data
                neighborhood = self.get_node_neighborhood(node["name"])
                
                # Combine node info with neighborhood data
                enriched_node = {
                    **node_info,
                    "neighborhood": neighborhood
                }
                
                nodes_with_neighborhoods.append(enriched_node)
            
            enriched_clusters.append(nodes_with_neighborhoods)
        
        return enriched_clusters

    def suggest_merges(self, threshold: float = 0.9, max_cluster_size: int = 10) -> List[Dict[str, Any]]:
        """
        Suggest nodes that could be merged based on high similarity
        
        Args:
            threshold: Similarity threshold (0-1), higher values mean more similar nodes
            max_cluster_size: Maximum number of nodes in a cluster to consider for merging
            
        Returns:
            List of suggested merges with the nodes involved
        """
        clusters = self.find_clusters(threshold=threshold, min_cluster_size=2, max_cluster_size=max_cluster_size)
        
        suggested_merges = []
        for cluster in clusters:
            # Use the first node (usually the seed node) as the primary node
            primary = cluster[0]
            
            # All other nodes in the cluster are candidates for merging
            candidates = cluster[1:]
            
            # Get neighborhood information for the primary and all candidates
            primary_neighborhood = self.get_node_neighborhood(primary["name"])
            
            candidates_with_neighborhoods = []
            for candidate in candidates:
                candidate_neighborhood = self.get_node_neighborhood(candidate["name"])
                candidates_with_neighborhoods.append({
                    **candidate,
                    "neighborhood": candidate_neighborhood
                })
            
            suggested_merges.append({
                "primary_node": {
                    **primary,
                    "neighborhood": primary_neighborhood
                },
                "merge_candidates": candidates_with_neighborhoods
            })
            
        return suggested_merges

    def update_node_description(self, node_name: str, new_description: str) -> bool:
        """
        Update a node's description
        
        Args:
            node_name: Name of the node to update
            new_description: New description to set for the node
            
        Returns:
            True if the update was successful, False otherwise
        """
        if self.db is None:
            print(f"Cannot update node '{node_name}': Not connected to FalkorDB")
            return False
            
        try:
            g = self.db.select_graph(self.graph_name)
            
            # Update the node's description
            cypher = """
            MATCH (n {name: $name})
            SET n.description = $description
            RETURN count(n) as updated
            """
            
            result = g.query(cypher, {
                "name": node_name,
                "description": new_description
            })
            
            # Check if any nodes were updated
            updated = result.result_set[0][0] if result.result_set else 0
            
            if updated > 0:
                print(f"Successfully updated description for node '{node_name}'")
                # Clear this node from the cache since it's been updated
                if node_name in self.node_embeddings_cache:
                    del self.node_embeddings_cache[node_name]
                return True
            else:
                print(f"Node '{node_name}' not found")
                return False
            
        except Exception as e:
            print(f"Error updating node description: {e}")
            return False
            
    def execute_merge(self, keep_name: str, remove_name: str, new_description: str = None) -> bool:
        """
        Execute a merge between two nodes
        
        Args:
            keep_name: Name of the node to keep
            remove_name: Name of the node to merge into the kept node
            new_description: Optional new description for the kept node
            
        Returns:
            True if the merge was successful, False otherwise
        """
        try:
            # First update the description if provided
            if new_description:
                success = self.update_node_description(keep_name, new_description)
                if not success:
                    print(f"Failed to update description for node '{keep_name}'")
                    # Continue with the merge anyway
            
            # Now merge the nodes
            graph = AISafetyGraph()
            graph.merge_nodes(keep_name, remove_name)
            print(f"Successfully merged '{remove_name}' into '{keep_name}'")
            
            # Clear both nodes from the cache
            if keep_name in self.node_embeddings_cache:
                del self.node_embeddings_cache[keep_name]
            if remove_name in self.node_embeddings_cache:
                del self.node_embeddings_cache[remove_name]
                
            return True
        except Exception as e:
            print(f"Error merging nodes: {e}")
            return False


def check_vector_capabilities():
    """Diagnose and report on vector search capabilities"""
    search = SimilaritySearch()  # Using default connection parameters
    
    print("\n=== Neo4j Vector Search Diagnostic ===")
    print(f"Connected to Neo4j: {'Yes' if search.db else 'No'}")
    
    if search.db:
        try:
            with search.db.session() as session:
                print(f"Neo4j session created: Yes")
                
                # Check for vector functions
                try:
                    # Test if vector functions are available
                    vector_test_query = """
                    CALL dbms.functions() YIELD name
                    WHERE name CONTAINS 'vector' OR name CONTAINS 'similarity'
                    RETURN name
                    LIMIT 5
                    """
                    try:
                        vector_results = list(session.run(vector_test_query))
                        if vector_results:
                            print("\nVector/Similarity functions available:")
                            for i, record in enumerate(vector_results):
                                print(f"  {i+1}. {record['name']}")
                        else:
                            print("\nNo vector/similarity functions found in Neo4j")
                    except Exception as e:
                        print(f"\nFunction check failed: {e}")
                        
                    # Test manual cosine similarity
                    test_cosine = """
                    RETURN reduce(dot = 0.0, i in range(0, size([0.1, 0.2])-1) | 
                           dot + [0.1, 0.2][i] * [0.2, 0.3][i]) / 
                           (sqrt(reduce(norm1 = 0.0, i in range(0, size([0.1, 0.2])-1) | 
                                norm1 + [0.1, 0.2][i] * [0.1, 0.2][i])) * 
                            sqrt(reduce(norm2 = 0.0, i in range(0, size([0.2, 0.3])-1) | 
                                norm2 + [0.2, 0.3][i] * [0.2, 0.3][i]))) as similarity
                    """
                    try:
                        cosine_result = session.run(test_cosine).single()
                        similarity = cosine_result["similarity"] if cosine_result else None
                        print(f"\nManual cosine calculation test: {'Available' if similarity is not None else 'Unavailable'}")
                    except Exception as e:
                        print(f"\nManual cosine calculation failed: {e}")
                    
                    # Count nodes with embeddings
                    count_query = """
                    MATCH (n) 
                    WHERE n.embedding IS NOT NULL 
                    RETURN COUNT(n) as count
                    """
                    result = session.run(count_query).single()
                    count = result["count"] if result else 0
                    print(f"\nNodes with embeddings: {count}")
                    
                    # Count relationships with embeddings
                    edge_count_query = """
                    MATCH ()-[r]->()
                    WHERE r.embedding IS NOT NULL 
                    RETURN COUNT(r) as count
                    """
                    try:
                        edge_result = session.run(edge_count_query).single()
                        edge_count = edge_result["count"] if edge_result else 0
                        print(f"Relationships with embeddings: {edge_count}")
                    except Exception as e:
                        print(f"Could not count relationships with embeddings: {e}")
                    
                    if count > 0:
                        # Sample a node embedding to check dimensions
                        sample_query = """
                        MATCH (n) 
                        WHERE n.embedding IS NOT NULL 
                        RETURN labels(n) as labels, size(n.embedding) as dimensions
                        LIMIT 1
                        """
                        result = session.run(sample_query).single()
                        if result:
                            node_labels = result["labels"]
                            dimensions = result["dimensions"]
                            print(f"Sample node labels: {node_labels}")
                            print(f"Node embedding dimensions: {dimensions}")
                    
                    # Test a similarity search if we have embeddings
                    if count > 0:
                        try:
                            # Get a sample node embedding
                            sample_query = """
                            MATCH (n) 
                            WHERE n.embedding IS NOT NULL 
                            RETURN n.embedding as embedding
                            LIMIT 1
                            """
                            result = session.run(sample_query).single()
                            if result and result["embedding"]:
                                sample_embedding = result["embedding"]
                                
                                # Try a manual cosine similarity search with the sample embedding
                                test_search = """
                                WITH $embedding AS query_vec
                                MATCH (node)
                                WHERE node.embedding IS NOT NULL
                                WITH node, 
                                    reduce(dot = 0.0, i in range(0, size(query_vec)-1) | 
                                        dot + query_vec[i] * node.embedding[i]) as dotProduct,
                                    sqrt(reduce(norm1 = 0.0, i in range(0, size(query_vec)-1) | 
                                        norm1 + query_vec[i] * query_vec[i])) as norm1,
                                    sqrt(reduce(norm2 = 0.0, i in range(0, size(node.embedding)-1) | 
                                        norm2 + node.embedding[i] * node.embedding[i])) as norm2
                                WITH node, dotProduct / (norm1 * norm2) as score
                                WHERE score > 0.5
                                RETURN labels(node) as labels, score
                                ORDER BY score DESC
                                LIMIT 3
                                """
                                search_result = list(session.run(test_search, {"embedding": sample_embedding}))
                                if search_result:
                                    print("\nManual cosine similarity search test: Successful")
                                    print("Sample results:")
                                    for i, record in enumerate(search_result):
                                        print(f"  Result {i+1}: {record['labels']} (score: {record['score']:.4f})")
                                else:
                                    print("\nCosine similarity search test: No results found")
                        except Exception as e:
                            print(f"\nSimilarity search test failed: {e}")
                except Exception as e:
                    print(f"Error checking vector capabilities: {e}")
        except Exception as e:
            print(f"Error creating Neo4j session: {e}")
    
    print("\nVector search ready: ", "Yes" if search.has_vector_index else "No")
    print("Vector functions available:", "Yes" if search.has_vector_functions else "No")
    print("Fallback to Python similarity calculations:", "Yes" if not search.has_vector_functions else "No")
    print("=============================================")
    
    return search.has_vector_functions

def test_similarity_search():
    """Test the similarity search functionality"""
    # First check vector capabilities
    check_vector_capabilities()
    
    # Initialize search with default Neo4j connection
    search = SimilaritySearch()
    
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
    clusters = search.find_clusters(threshold=0.8, min_cluster_size=2, max_cluster_size=10)
    
    if clusters:
        print(f"Found {len(clusters)} clusters:")
        for i, cluster in enumerate(clusters):
            print(f"\nCluster {i+1} with {len(cluster)} nodes:")
            for j, node in enumerate(cluster):
                print(f"  {j+1}. {node['name']} (Similarity: {node['similarity']:.3f})")
    else:
        print("No clusters found. Try lowering the threshold or adding more nodes to the database.")
    
    # Test neighborhood retrieval for a node if clusters were found
    if clusters and len(clusters) > 0 and len(clusters[0]) > 0:
        test_node = clusters[0][0]['name']
        print(f"\nTesting neighborhood retrieval for node '{test_node}'...")
        neighborhood = search.get_node_neighborhood(test_node)
        
        if "error" not in neighborhood:
            edge_count = len(neighborhood.get('edges', []))
            neighbor_count = len(neighborhood.get('neighbors', {}))
            print(f"Neighborhood: {edge_count} edges, {neighbor_count} neighbors")
            
            if edge_count > 0:
                print("Sample edges:")
                for i, edge in enumerate(neighborhood.get('edges', [])[:3]):  # Show up to 3 edges
                    print(f"  - {edge.get('type', 'UNKNOWN')} → {edge.get('target', 'UNKNOWN')}")
        else:
            print(f"Error getting neighborhood: {neighborhood.get('error', 'Unknown error')}")
    
    # Try getting clusters with neighborhoods
    print("\nTesting clusters with neighborhood information...")
    enriched_clusters = search.get_cluster_with_neighborhoods(threshold=0.8, max_cluster_size=5)
    
    if enriched_clusters:
        print(f"Found {len(enriched_clusters)} clusters with neighborhood information")
        
        # Just print summary of first cluster
        if len(enriched_clusters) > 0:
            first_cluster = enriched_clusters[0]
            print(f"First cluster has {len(first_cluster)} nodes with neighborhoods")
    else:
        print("No clusters with neighborhood information found")
    
    # Try suggesting merges
    print("\nSuggesting nodes to merge...")
    merges = search.suggest_merges(threshold=0.9, max_cluster_size=5)
    
    if merges:
        print(f"Found {len(merges)} potential merges:")
        for i, merge in enumerate(merges):
            primary = merge['primary_node']
            candidates = merge['merge_candidates']
            print(f"\nMerge {i+1}: Keep '{primary['name']}' and merge:")
            for j, candidate in enumerate(candidates):
                print(f"  {j+1}. '{candidate['name']}' (Similarity: {candidate['similarity']:.3f})")
                
            # Just test with one merge suggestion
            if i == 0:
                break
    else:
        print("No merge suggestions found. Try lowering the threshold.")


if __name__ == "__main__":
    test_similarity_search()
