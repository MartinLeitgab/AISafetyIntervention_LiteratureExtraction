import networkx as nx
from typing import Dict, List, Set, Optional
from collections import defaultdict
from datetime import datetime
import hashlib
import re
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from models.models import CausalNode, CausalEdge
from utilities.utils import MATPLOTLIB_AVAILABLE, logger
from dataclasses import dataclass, asdict, field
import json

class EnhancedGlobalCausalGraph:
    """scalable causal graph merging system"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, List[CausalEdge]] = defaultdict(list)
        self.node_text_index: Dict[str, str] = {}
        self.node_hash_index: Dict[str, str] = {}
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.edge_semantic_index: Dict[str, List[str]] = defaultdict(list)
        self.similarity_threshold = similarity_threshold
        self.tfidf_vectorizer = None
        self.concept_vectors = None
        self.confidence_threshold = 2
        self.merge_stats = {
            'nodes_merged': 0, 'edges_merged': 0, 'similarity_computations': 0,
            'exact_matches': 0, 'semantic_matches': 0, 'stage_1_matches': 0,
            'stage_2_matches': 0, 'stage_3_matches': 0, 'stage_4_matches': 0,
            'stage_4_skipped': 0
        }
        self.performance_stats = {
            'stage_timings': defaultdict(list),
            'total_matching_time': 0,
            'average_match_time_by_stage': {}
        }

    def merge_local_graphs(self, local_graphs: List[dict], batch_size: int = 100):
        """Scalable merging with enhanced batching and progress tracking"""
        logger.info(f"Starting enhanced multi-stage merge of {len(local_graphs)} local graphs...")
        total_graphs = len(local_graphs)
        if len(self.nodes) > 50000:
            batch_size = min(batch_size, 50)
        elif len(self.nodes) > 10000:
            batch_size = min(batch_size, 75)
        
        for batch_start in range(0, total_graphs, batch_size):
            batch_end = min(batch_start + batch_size, total_graphs)
            batch = local_graphs[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (total_graphs - 1) // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} (Adaptive batch size: {len(batch)})")
            self._merge_nodes_batch_optimized(batch)
            self._merge_edges_batch(batch)
            if batch_start % (batch_size * 5) == 0 and len(self.nodes) > 10000:
                self._optimize_indices_for_scale()
        
        self._aggregate_evidence()
        self._filter_low_confidence_edges()
        self._compute_final_statistics()
        logger.info("Enhanced multi-stage merge completed successfully!")
        self._print_merge_summary()

    # ... (Include all methods from EnhancedGlobalCausalGraph except visualization and analysis)
    # Methods to include:
    def _update_indices(self, batch: List[dict]):
        """Update all indices with new batch data"""
        for local_graph_data in batch:
            # Extract nodes and build indices
            for node_data in local_graph_data.get('nodes', []):
                node = self._dict_to_node(node_data)
                
                # Update text indices
                self.node_text_index[node.canonical_text] = node.node_id
                self.node_hash_index[node.text_hash] = node.node_id
                
                # Update keyword index
                for keyword in node.semantic_keywords:
                    self.keyword_index[keyword].add(node.node_id)
    
    def _update_indices_optimized(self, batch: List[dict]):
        """Memory-efficient index updating for massive scale"""
        # Batch collect all nodes first to minimize index operations
        nodes_to_process = []
        
        for local_graph_data in batch:
            for node_data in local_graph_data.get('nodes', []):
                node = self._dict_to_node(node_data)
                nodes_to_process.append(node)
        
        # Batch update indices
        for node in nodes_to_process:
            # Update text indices
            self.node_text_index[node.canonical_text] = node.node_id
            self.node_hash_index[node.text_hash] = node.node_id
            
            # Bulk update keyword index
            for keyword in node.semantic_keywords:
                self.keyword_index[keyword].add(node.node_id)
    
    def _merge_nodes_batch_optimized(self, batch: List[dict]):
        """Enhanced node merging with optimized multi-stage matching"""
        for local_graph_data in batch:
            paper_id = local_graph_data.get('paper_id', 'unknown')
            
            for node_data in local_graph_data.get('nodes', []):
                node = self._dict_to_node(node_data)
                node.source_papers = [paper_id]
                
                # Enhanced multi-stage matching with performance tracking
                canonical_id = self._find_best_node_match(node)
                
                if canonical_id:
                    # Validate that the canonical_id exists before merging
                    if canonical_id in self.nodes:
                        # Merge with existing node
                        self._merge_node_data(canonical_id, node)
                        self.merge_stats['nodes_merged'] += 1
                    else:
                        # Index inconsistency - log warning and add as new node
                        logger.warning(f"Index inconsistency: canonical_id '{canonical_id}' not found in nodes. Adding as new node.")
                        self._add_new_node(node)
                        # Clean up invalid index entries
                        self._cleanup_invalid_indices(canonical_id)
                else:
                    # Add as new node
                    self._add_new_node(node)
    
    def _optimize_indices_for_scale(self):
        """Periodic index optimization for massive scale operations"""
        logger.info("Optimizing indices for massive scale...")
        
        # Clean up empty keyword entries
        empty_keywords = [k for k, v in self.keyword_index.items() if not v]
        for keyword in empty_keywords:
            del self.keyword_index[keyword]
        
        # Log index sizes for monitoring
        logger.debug(f"Index sizes: hash={len(self.node_hash_index)}, "
                    f"text={len(self.node_text_index)}, "
                    f"keywords={len(self.keyword_index)}")
        
        # Memory optimization: limit keyword index size if too large
        if len(self.keyword_index) > 100000:
            # Keep only high-frequency keywords for performance
            keyword_counts = {k: len(v) for k, v in self.keyword_index.items()}
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Keep top 50,000 most frequent keywords
            keywords_to_keep = set(k for k, _ in sorted_keywords[:50000])
            self.keyword_index = {
                k: v for k, v in self.keyword_index.items() 
                if k in keywords_to_keep
            }
            
            logger.info(f"Optimized keyword index: kept {len(self.keyword_index)} high-frequency keywords")
    
    def _cleanup_invalid_indices(self, invalid_id: str):
        """Clean up invalid entries from indices"""
        # Remove from text indices
        keys_to_remove = []
        for key, value in self.node_text_index.items():
            if value == invalid_id:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.node_text_index[key]
        
        # Remove from hash indices
        keys_to_remove = []
        for key, value in self.node_hash_index.items():
            if value == invalid_id:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.node_hash_index[key]
        
        # Remove from keyword indices
        for keyword_set in self.keyword_index.values():
            keyword_set.discard(invalid_id)
    
    def _merge_nodes_batch(self, batch: List[dict]):
        """Advanced node merging with multiple similarity methods"""
        for local_graph_data in batch:
            paper_id = local_graph_data.get('paper_id', 'unknown')
            
            for node_data in local_graph_data.get('nodes', []):
                node = self._dict_to_node(node_data)
                node.source_papers = [paper_id]
                
                # Find potential matches using multiple methods
                canonical_id = self._find_best_node_match(node)
                
                if canonical_id:
                    # Validate that the canonical_id exists before merging
                    if canonical_id in self.nodes:
                        # Merge with existing node
                        self._merge_node_data(canonical_id, node)
                        self.merge_stats['nodes_merged'] += 1
                    else:
                        # Index inconsistency - log warning and add as new node
                        logger.warning(f"Index inconsistency: canonical_id '{canonical_id}' not found in nodes. Adding as new node.")
                        self._add_new_node(node)
                        # Clean up invalid index entries
                        self._cleanup_invalid_indices(canonical_id)
                else:
                    # Add as new node
                    self._add_new_node(node)
    
    def _find_best_node_match(self, node: CausalNode) -> Optional[str]:
        """Enhanced Multi-Stage Node Matching for Massive Scale
        
        Stage 1: Exact Hash Matching (O(1) - fastest)
        Stage 2: Canonical Text Matching (O(1) with indexing)  
        Stage 3: Keyword-Based Similarity (O(k) where k = keyword matches)
        Stage 4: Semantic Similarity (O(n) - only for smaller graphs)
        """
        import time
        start_time = time.time()
        
        # Stage 1: Exact Hash Matching (O(1)) - Fastest possible lookup
        if node.text_hash in self.node_hash_index:
            self.merge_stats['exact_matches'] += 1
            self.merge_stats['stage_1_matches'] = self.merge_stats.get('stage_1_matches', 0) + 1
            self._log_match_timing('stage_1_hash', time.time() - start_time)
            return self.node_hash_index[node.text_hash]
        
        # Stage 2: Canonical Text Matching (O(1) with indexing) - Direct text lookup
        if node.canonical_text in self.node_text_index:
            self.merge_stats['exact_matches'] += 1
            self.merge_stats['stage_2_matches'] = self.merge_stats.get('stage_2_matches', 0) + 1
            self._log_match_timing('stage_2_canonical', time.time() - start_time)
            return self.node_text_index[node.canonical_text]
        
        # Stage 3: Keyword-Based Similarity (O(k)) - Limited by keyword matches
        keyword_matches = self._find_keyword_matches_optimized(node)
        if keyword_matches:
            best_match = self._compute_best_similarity_match_fast(node, keyword_matches)
            if best_match:
                self.merge_stats['semantic_matches'] += 1
                self.merge_stats['stage_3_matches'] = self.merge_stats.get('stage_3_matches', 0) + 1
                self._log_match_timing('stage_3_keyword', time.time() - start_time)
                return best_match
        
        # Stage 4: Semantic Similarity (O(n)) - Most expensive, use sparingly
        # Only run for graphs under 10,000 nodes to maintain performance
        if len(self.nodes) < 10000:
            semantic_match = self._find_semantic_match_optimized(node)
            if semantic_match:
                self.merge_stats['semantic_matches'] += 1
                self.merge_stats['stage_4_matches'] = self.merge_stats.get('stage_4_matches', 0) + 1
                self._log_match_timing('stage_4_semantic', time.time() - start_time)
                return semantic_match
        else:
            # For massive graphs, skip semantic matching to maintain O(1) average performance
            self.merge_stats['stage_4_skipped'] = self.merge_stats.get('stage_4_skipped', 0) + 1
            logger.debug(f"Skipped semantic matching for massive graph (nodes: {len(self.nodes)})")
        
        # No match found at any stage
        self._log_match_timing('no_match', time.time() - start_time)
        return None
    
    def _find_keyword_matches(self, node: CausalNode) -> List[str]:
        """Find potential matches based on shared keywords"""
        candidate_nodes = set()
        
        for keyword in node.semantic_keywords:
            candidate_nodes.update(self.keyword_index[keyword])
        
        # Filter candidates that share significant keywords
        matches = []
        for candidate_id in candidate_nodes:
            if candidate_id in self.nodes:
                candidate_node = self.nodes[candidate_id]
                shared_keywords = set(node.semantic_keywords) & set(candidate_node.semantic_keywords)
                
                # Require significant keyword overlap
                if len(shared_keywords) >= min(2, len(node.semantic_keywords) * 0.5):
                    matches.append(candidate_id)
        
        return matches
    
    def _find_keyword_matches_optimized(self, node: CausalNode) -> List[str]:
        """Optimized keyword matching with early termination and ranking"""
        if not node.semantic_keywords:
            return []
        
        # Count keyword overlaps for efficient ranking
        candidate_scores = defaultdict(int)
        
        # Prioritize high-value keywords (more specific terms)
        keyword_weights = self._compute_keyword_weights(node.semantic_keywords)
        
        for keyword in node.semantic_keywords:
            weight = keyword_weights.get(keyword, 1.0)
            for candidate_id in self.keyword_index[keyword]:
                candidate_scores[candidate_id] += weight
        
        # Early termination: only consider top candidates
        min_score = max(2.0, len(node.semantic_keywords) * 0.4)
        promising_candidates = [
            candidate_id for candidate_id, score in candidate_scores.items() 
            if score >= min_score and candidate_id in self.nodes
        ]
        
        # Sort by score for better matching order
        promising_candidates.sort(
            key=lambda x: candidate_scores[x], 
            reverse=True
        )
        
        return promising_candidates[:50]  # Limit to top 50 for performance
    
    def _compute_keyword_weights(self, keywords: List[str]) -> Dict[str, float]:
        """Compute weights for keywords based on specificity"""
        weights = {}
        
        for keyword in keywords:
            # Weight by inverse frequency (rarer keywords are more specific)
            frequency = len(self.keyword_index[keyword])
            if frequency == 0:
                weights[keyword] = 2.0  # New keyword, high weight
            else:
                # Inverse log frequency with minimum weight of 0.5
                weights[keyword] = max(0.5, 2.0 - np.log10(frequency + 1))
        
        return weights
    
    def _compute_best_similarity_match(self, node: CausalNode, candidates: List[str]) -> Optional[str]:
        """Compute similarity scores and return best match above threshold"""
        best_match = None
        best_score = 0
        
        for candidate_id in candidates:
            if candidate_id not in self.nodes:
                continue
                
            candidate_node = self.nodes[candidate_id]
            score = self._compute_node_similarity(node, candidate_node)
            
            self.merge_stats['similarity_computations'] += 1
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = candidate_id
        
        return best_match
    
    def _compute_best_similarity_match_fast(self, node: CausalNode, candidates: List[str]) -> Optional[str]:
        """Fast similarity computation with early termination optimizations"""
        if not candidates:
            return None
        
        best_match = None
        best_score = 0
        
        # Early termination if we find a very high confidence match
        high_confidence_threshold = 0.95
        
        for candidate_id in candidates[:20]:  # Limit comparisons for performance
            if candidate_id not in self.nodes:
                continue
                
            candidate_node = self.nodes[candidate_id]
            
            # Fast pre-screening: check type compatibility first
            if node.isIntervention != candidate_node.isIntervention:
                continue
            
            # Quick keyword overlap check
            keyword_overlap = len(set(node.semantic_keywords) & set(candidate_node.semantic_keywords))
            if keyword_overlap == 0:
                continue
                
            # Full similarity computation only for promising candidates
            score = self._compute_node_similarity_fast(node, candidate_node)
            
            self.merge_stats['similarity_computations'] += 1
            
            if score >= high_confidence_threshold:
                # Early termination for very high confidence
                return candidate_id
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = candidate_id
        
        return best_match
    
    def _compute_node_similarity(self, node1: CausalNode, node2: CausalNode) -> float:
        """Comprehensive node similarity computation"""
        
        # Text similarity (primary)
        text_sim = SequenceMatcher(None, node1.canonical_text, node2.canonical_text).ratio()
        
        # Keyword similarity
        keywords1 = set(node1.semantic_keywords)
        keywords2 = set(node2.semantic_keywords)
        if keywords1 or keywords2:
            keyword_sim = len(keywords1 & keywords2) / len(keywords1 | keywords2)
        else:
            keyword_sim = 0
        
        # Type similarity (intervention vs problem)
        type_sim = 1.0 if node1.isIntervention == node2.isIntervention else 0.5
        
        # Combined similarity with weights
        similarity = (0.6 * text_sim + 0.3 * keyword_sim + 0.1 * type_sim)
        
        return similarity
    
    def _compute_node_similarity_fast(self, node1: CausalNode, node2: CausalNode) -> float:
        """Optimized similarity computation for high-performance matching"""
        
        # Quick keyword similarity (most discriminative for AI safety concepts)
        keywords1 = set(node1.semantic_keywords)
        keywords2 = set(node2.semantic_keywords)
        
        if not keywords1 and not keywords2:
            keyword_sim = 0
        elif keywords1 or keywords2:
            keyword_sim = len(keywords1 & keywords2) / len(keywords1 | keywords2)
        else:
            keyword_sim = 0
        
        # Early exit if keyword similarity is very low
        if keyword_sim < 0.1:
            return keyword_sim
        
        # Fast text similarity using length and character overlap
        text1, text2 = node1.canonical_text, node2.canonical_text
        
        # Length-based similarity (fast approximation)
        len_sim = 1.0 - abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1)
        
        # Character overlap similarity (faster than full sequence matching)
        chars1, chars2 = set(text1), set(text2)
        char_sim = len(chars1 & chars2) / len(chars1 | chars2) if (chars1 or chars2) else 0
        
        # Type similarity
        type_sim = 1.0 if node1.isIntervention == node2.isIntervention else 0.5
        
        # Weighted combination optimized for speed
        similarity = (0.5 * keyword_sim + 0.3 * char_sim + 0.1 * len_sim + 0.1 * type_sim)
        
        return similarity
    
    def _find_semantic_match(self, node: CausalNode) -> Optional[str]:
        """Use TF-IDF for semantic matching (expensive, use sparingly)"""
        if not self.concept_vectors is not None:
            return None
        
        # This would require maintaining TF-IDF vectors - implement if needed for very high precision
        # For now, return None to avoid O(n²) complexity
        return None
    
    def _find_semantic_match_optimized(self, node: CausalNode) -> Optional[str]:
        """Optimized semantic matching with sampling and early termination"""
        
        # Only use for small graphs to maintain performance
        if len(self.nodes) > 1000:
            return None
        
        # Sample-based semantic matching for better performance
        import random
        node_sample = list(self.nodes.keys())
        
        # Sample up to 100 nodes for semantic comparison
        if len(node_sample) > 100:
            node_sample = random.sample(node_sample, 100)
        
        best_match = None
        best_score = 0
        
        for candidate_id in node_sample:
            candidate_node = self.nodes[candidate_id]
            
            # Skip if basic compatibility fails
            if node.isIntervention != candidate_node.isIntervention:
                continue
            
            # Use fast similarity first as a filter
            quick_score = self._compute_node_similarity_fast(node, candidate_node)
            if quick_score < 0.6:  # Only do expensive comparison if promising
                continue
            
            # Full similarity computation for promising candidates
            score = self._compute_node_similarity(node, candidate_node)
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = candidate_id
        
        return best_match
    
    def _log_match_timing(self, stage: str, elapsed_time: float):
        """Log timing information for performance analysis"""
        self.performance_stats['stage_timings'][stage].append(elapsed_time)
        self.performance_stats['total_matching_time'] += elapsed_time
    
    def _merge_node_data(self, canonical_id: str, new_node: CausalNode):
        """Intelligently merge node data"""
        if canonical_id not in self.nodes:
            raise ValueError(f"Cannot merge: canonical_id '{canonical_id}' not found in nodes dictionary")
        
        existing_node = self.nodes[canonical_id]
        
        # Merge metadata lists (avoid duplicates)
        existing_node.DOI_URL.extend([url for url in new_node.DOI_URL if url not in existing_node.DOI_URL])
        existing_node.authors.extend([auth for auth in new_node.authors if auth not in existing_node.authors])
        existing_node.institutions.extend([inst for inst in new_node.institutions if inst not in existing_node.institutions])
        existing_node.timestamp.extend([ts for ts in new_node.timestamp if ts not in existing_node.timestamp])
        existing_node.source_papers.extend(new_node.source_papers)
        
        # Merge aliases
        new_aliases = set(new_node.aliases + [new_node.concept_text])
        existing_aliases = set(existing_node.aliases)
        existing_node.aliases = list(existing_aliases | new_aliases)
        
        # Update semantic keywords
        existing_node.semantic_keywords = list(set(existing_node.semantic_keywords + new_node.semantic_keywords))
        
        # Update intervention information if more specific
        if new_node.isIntervention == 1 and existing_node.isIntervention == 0:
            existing_node.isIntervention = 1
            existing_node.stage_in_pipeline = new_node.stage_in_pipeline
            existing_node.maturity_level = new_node.maturity_level
            existing_node.implemented = new_node.implemented
        
        # Track merge history
        existing_node.merge_history.append(f"Merged with {new_node.node_id}")
        
        # Update confidence based on number of sources (keep within 0-1 range)
        existing_node.confidence_score = min(1.0, 0.5 + 0.1 * len(existing_node.source_papers))
        
        # Update indices
        self.node_text_index[new_node.canonical_text] = canonical_id
        self.node_hash_index[new_node.text_hash] = canonical_id
        for keyword in new_node.semantic_keywords:
            self.keyword_index[keyword].add(canonical_id)
    
    def _add_new_node(self, node: CausalNode):
        """Add a completely new node"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **asdict(node))
        
        # Update all indices
        self.node_text_index[node.canonical_text] = node.node_id
        self.node_hash_index[node.text_hash] = node.node_id
        for keyword in node.semantic_keywords:
            self.keyword_index[keyword].add(node.node_id)
    
    def _merge_edges_batch(self, batch: List[dict]):
        """Enhanced edge merging with relationship analysis"""
        for local_graph_data in batch:
            paper_id = local_graph_data.get('paper_id', 'unknown')
            
            for edge_data in local_graph_data.get('edges', []):
                edge = self._dict_to_edge(edge_data)
                edge.source_papers = [paper_id]
                
                # Resolve node references
                resolved_edge = self._resolve_edge_nodes(edge)
                if resolved_edge:
                    self._add_or_merge_edge(resolved_edge)
    
    def _resolve_edge_nodes(self, edge: CausalEdge) -> Optional[CausalEdge]:
        """Resolve edge node references to canonical node IDs"""
        resolved_sources = []
        resolved_targets = []
        
        for source in edge.source_nodes:
            source_canonical = self._resolve_node_reference(source)
            if source_canonical:
                resolved_sources.append(source_canonical)
        
        for target in edge.target_nodes:
            target_canonical = self._resolve_node_reference(target)
            if target_canonical:
                resolved_targets.append(target_canonical)
        
        if not resolved_sources or not resolved_targets:
            return None
        
        # Create new edge with resolved references
        resolved_edge = CausalEdge(
            DOI_URL=edge.DOI_URL,
            authors=edge.authors,
            institutions=edge.institutions,
            timestamp=edge.timestamp,
            edge_text=edge.edge_text,
            source_nodes=resolved_sources,
            target_nodes=resolved_targets,
            confidence=edge.confidence
        )
        
        resolved_edge.source_papers = edge.source_papers
        resolved_edge.relationship_type = edge.relationship_type
        resolved_edge.relationship_strength = edge.relationship_strength
        
        return resolved_edge
    
    def _resolve_node_reference(self, node_ref: str) -> Optional[str]:
        """Resolve a node reference to canonical ID"""
        # Try exact match first
        node_id = node_ref.replace(" ", "_").upper()
        if node_id in self.nodes:
            return node_id
        
        # Try hash lookup
        canonical_text = re.sub(r'[^\w\s]', '', node_ref.lower().strip())
        text_hash = hashlib.md5(canonical_text.encode()).hexdigest()[:8]
        if text_hash in self.node_hash_index:
            return self.node_hash_index[text_hash]
        
        # Try text lookup
        if canonical_text in self.node_text_index:
            return self.node_text_index[canonical_text]
        
        return None
    
    def _add_or_merge_edge(self, edge: CausalEdge):
        """Add edge or merge with existing similar edge"""
        for source in edge.source_nodes:
            for target in edge.target_nodes:
                edge_key = f"{source}→{target}:{edge.relationship_type}"
                
                # Check for existing similar edges
                existing_edges = self.edges[edge_key]
                merged = False
                
                for existing_edge in existing_edges:
                    if self._edges_are_similar(edge, existing_edge):
                        self._merge_edge_data(existing_edge, edge)
                        merged = True
                        self.merge_stats['edges_merged'] += 1
                        break
                
                if not merged:
                    self.edges[edge_key].append(edge)
                    
                    # Add to NetworkX graph
                    if not self.graph.has_edge(source, target):
                        self.graph.add_edge(source, target, relationships=[])
                    self.graph[source][target]['relationships'].append(edge)
                
                # Update semantic index
                self.edge_semantic_index[edge.semantic_hash].append(edge_key)
    
    def _edges_are_similar(self, edge1: CausalEdge, edge2: CausalEdge) -> bool:
        """Determine if two edges represent the same relationship"""
        
        # Same relationship type
        if edge1.relationship_type != edge2.relationship_type:
            return False
        
        # Similar semantic content
        if edge1.semantic_hash == edge2.semantic_hash:
            return True
        
        # Text similarity
        text_sim = SequenceMatcher(None, edge1.canonical_relationship, edge2.canonical_relationship).ratio()
        return text_sim >= 0.7
    
    def _merge_edge_data(self, existing_edge: CausalEdge, new_edge: CausalEdge):
        """Merge data from two similar edges"""
        # Merge metadata
        existing_edge.DOI_URL.extend([url for url in new_edge.DOI_URL if url not in existing_edge.DOI_URL])
        existing_edge.authors.extend([auth for auth in new_edge.authors if auth not in existing_edge.authors])
        existing_edge.institutions.extend([inst for inst in new_edge.institutions if inst not in existing_edge.institutions])
        existing_edge.timestamp.extend([ts for ts in new_edge.timestamp if ts not in existing_edge.timestamp])
        existing_edge.source_papers.extend(new_edge.source_papers)
        
        # Update confidence (weighted average)
        total_evidence = existing_edge.evidence_count + new_edge.evidence_count
        existing_edge.confidence = int(
            (existing_edge.confidence * existing_edge.evidence_count + 
             new_edge.confidence * new_edge.evidence_count) / total_evidence
        )
        existing_edge.evidence_count = total_evidence
        
        # Update relationship strength
        existing_edge.relationship_strength = (
            existing_edge.relationship_strength + new_edge.relationship_strength
        ) / 2
    
    def _aggregate_evidence(self):
        """Advanced evidence aggregation with domain-specific weighting"""
        logger.info("Aggregating evidence across merged edges...")
        
        evidence_weights = {
            'experimental': 1.2,
            'observational': 1.0, 
            'theoretical': 0.8,
            'conceptual': 0.6
        }
        
        for edge_key, edge_list in self.edges.items():
            if len(edge_list) > 1:
                # Compute weighted evidence
                total_weight = 0
                weighted_confidence = 0
                
                for edge in edge_list:
                    # Infer evidence type from paper metadata and confidence
                    evidence_type = self._infer_evidence_type(edge)
                    weight = evidence_weights.get(evidence_type, 1.0)
                    
                    total_weight += weight * edge.evidence_count
                    weighted_confidence += edge.confidence * weight * edge.evidence_count
                
                # Update primary edge with aggregated confidence
                primary_edge = edge_list[0]
                primary_edge.confidence = int(weighted_confidence / total_weight) if total_weight > 0 else primary_edge.confidence
                primary_edge.evidence_count = sum(e.evidence_count for e in edge_list)
    
    def _infer_evidence_type(self, edge: CausalEdge) -> str:
        """Infer evidence type from edge characteristics"""
        # Simple heuristic - in production, use more sophisticated analysis
        if edge.confidence >= 4 and any(inst in ['MIT', 'Stanford', 'OpenAI'] for inst in edge.institutions):
            return 'experimental'
        elif edge.confidence >= 3:
            return 'observational'
        else:
            return 'theoretical'
    
    def _filter_low_confidence_edges(self):
        """Remove edges below confidence threshold"""
        initial_count = sum(len(edges) for edges in self.edges.values())
        
        filtered_edges = {}
        for edge_key, edge_list in self.edges.items():
            high_conf_edges = [edge for edge in edge_list if edge.confidence >= self.confidence_threshold]
            if high_conf_edges:
                filtered_edges[edge_key] = high_conf_edges
        
        self.edges = filtered_edges
        final_count = sum(len(edges) for edges in self.edges.values())
        
        logger.info(f"Filtered {initial_count - final_count} low-confidence edges")
    
    def _compute_final_statistics(self):
        """Compute comprehensive final statistics"""
        # Compute average timing by stage
        for stage, timings in self.performance_stats['stage_timings'].items():
            if timings:
                self.performance_stats['average_match_time_by_stage'][stage] = np.mean(timings) * 1000  # ms
        
        self.final_stats = {
            'total_nodes': len(self.nodes),
            'total_unique_edges': len(self.edges),
            'total_evidence_pieces': sum(sum(e.evidence_count for e in edges) for edges in self.edges.values()),
            'merge_efficiency': {
                'nodes_merged': self.merge_stats['nodes_merged'],
                'edges_merged': self.merge_stats['edges_merged'],
                'similarity_computations': self.merge_stats['similarity_computations'],
                'exact_matches': self.merge_stats['exact_matches'],
                'semantic_matches': self.merge_stats['semantic_matches']
            },
            'stage_performance': {
                'stage_1_hash_matches': self.merge_stats['stage_1_matches'],
                'stage_2_canonical_matches': self.merge_stats['stage_2_matches'],
                'stage_3_keyword_matches': self.merge_stats['stage_3_matches'],
                'stage_4_semantic_matches': self.merge_stats['stage_4_matches'],
                'stage_4_skipped_for_scale': self.merge_stats['stage_4_skipped'],
                'total_matching_time_seconds': self.performance_stats['total_matching_time'],
                'average_timings_ms': self.performance_stats['average_match_time_by_stage']
            }
        }
    
    def _print_merge_summary(self):
        """Print comprehensive merge summary with enhanced performance metrics"""
        print("\n" + "="*80)
        print("ENHANCED MULTI-STAGE CAUSAL GRAPH MERGE SUMMARY")
        print("="*80)
        print(f"Final Graph Size:")
        print(f"  - Nodes: {self.final_stats['total_nodes']:,}")
        print(f"  - Unique Relationships: {self.final_stats['total_unique_edges']:,}")
        print(f"  - Total Evidence Pieces: {self.final_stats['total_evidence_pieces']:,}")
        
        print(f"\nMerge Efficiency:")
        merge_eff = self.final_stats['merge_efficiency']
        print(f"  - Nodes Merged: {merge_eff['nodes_merged']:,}")
        print(f"  - Edges Merged: {merge_eff['edges_merged']:,}")
        print(f"  - Total Similarity Computations: {merge_eff['similarity_computations']:,}")
        
        print(f"\nMulti-Stage Matching Performance:")
        stage_perf = self.final_stats['stage_performance']
        print(f"  - Stage 1 (Hash O(1)): {stage_perf['stage_1_hash_matches']:,} matches")
        print(f"  - Stage 2 (Canonical O(1)): {stage_perf['stage_2_canonical_matches']:,} matches")
        print(f"  - Stage 3 (Keyword O(k)): {stage_perf['stage_3_keyword_matches']:,} matches")
        print(f"  - Stage 4 (Semantic O(n)): {stage_perf['stage_4_semantic_matches']:,} matches")
        print(f"  - Stage 4 Skipped (Scale): {stage_perf['stage_4_skipped_for_scale']:,} nodes")
        
        # Calculate stage efficiency
        total_matches = (stage_perf['stage_1_hash_matches'] + stage_perf['stage_2_canonical_matches'] + 
                        stage_perf['stage_3_keyword_matches'] + stage_perf['stage_4_semantic_matches'])
        if total_matches > 0:
            stage1_pct = (stage_perf['stage_1_hash_matches'] / total_matches) * 100
            stage2_pct = (stage_perf['stage_2_canonical_matches'] / total_matches) * 100
            stage3_pct = (stage_perf['stage_3_keyword_matches'] / total_matches) * 100
            stage4_pct = (stage_perf['stage_4_semantic_matches'] / total_matches) * 100
            
            print(f"\nMatching Efficiency Distribution:")
            print(f"  - Fast O(1) Hash Matches: {stage1_pct:.1f}%")
            print(f"  - Fast O(1) Text Matches: {stage2_pct:.1f}%")
            print(f"  - Medium O(k) Keyword Matches: {stage3_pct:.1f}%")
            print(f"  - Slow O(n) Semantic Matches: {stage4_pct:.1f}%")
        
        print(f"\nPerformance Timings:")
        print(f"  - Total Matching Time: {stage_perf['total_matching_time_seconds']:.3f} seconds")
        
        avg_timings = stage_perf['average_timings_ms']
        if avg_timings:
            print(f"  - Average Stage Timings (ms):")
            for stage, time_ms in avg_timings.items():
                print(f"    • {stage}: {time_ms:.3f}ms")
        
        if self.nodes:
            interventions = sum(1 for node in self.nodes.values() if node.isIntervention == 1)
            print(f"\nContent Analysis:")
            print(f"  - Intervention Nodes: {interventions}")
            print(f"  - Problem/Concept Nodes: {len(self.nodes) - interventions}")
        
        print("="*80)
    
    def _dict_to_node(self, node_data: dict) -> CausalNode:
        """Convert dictionary to CausalNode object"""
        return CausalNode(**{k: v for k, v in node_data.items() if k in CausalNode.__dataclass_fields__})
    
    def _dict_to_edge(self, edge_data: dict) -> CausalEdge:
        """Convert dictionary to CausalEdge object"""
        return CausalEdge(**{k: v for k, v in edge_data.items() if k in CausalEdge.__dataclass_fields__})

    def export_enhanced_json(self, filename: str = None) -> dict:
        """Export with enhanced metadata and merge information"""
        export_data = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "merge_statistics": self.final_stats,
                "similarity_threshold": self.similarity_threshold,
                "confidence_threshold": self.confidence_threshold,
                "graph_properties": {
                    "density": nx.density(self.graph) if self.graph.nodes() else 0,
                    "strongly_connected_components": nx.number_strongly_connected_components(self.graph) if self.graph.nodes() else 0,
                    "weakly_connected_components": nx.number_weakly_connected_components(self.graph) if self.graph.nodes() else 0
                }
            },
            "nodes": [],
            "edges": [],
            "merge_provenance": {
                "node_merge_history": {},
                "edge_evidence_aggregation": {}
            }
        }
        
        # Export nodes with enhanced metadata
        for node_id, node in self.nodes.items():
            node_dict = asdict(node)
            node_dict['graph_metrics'] = {
                'degree': self.graph.degree(node_id) if node_id in self.graph else 0,
                'betweenness_centrality': 0,  # Computed separately if needed
                'evidence_strength': len(node.source_papers)
            }
            export_data["nodes"].append(node_dict)
            
            if node.merge_history:
                export_data["merge_provenance"]["node_merge_history"][node_id] = node.merge_history
        
        # Export edges with aggregation information
        for edge_key, edge_list in self.edges.items():
            for edge in edge_list:
                edge_dict = asdict(edge)
                edge_dict['aggregation_info'] = {
                    'total_evidence_pieces': edge.evidence_count,
                    'relationship_strength': edge.relationship_strength,
                    'semantic_hash': edge.semantic_hash
                }
                export_data["edges"].append(edge_dict)
            
            if len(edge_list) > 1:
                export_data["merge_provenance"]["edge_evidence_aggregation"][edge_key] = {
                    'total_sources': len(set().union(*[e.source_papers for e in edge_list])),
                    'evidence_pieces': len(edge_list)
                }
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Enhanced global graph exported to {filename}")
        
        return export_data
    
    def get_merge_quality_metrics(self) -> dict:
        """Compute quality metrics for the merge process"""
        metrics = {
            'merge_efficiency': {
                'node_reduction_ratio': self.merge_stats['nodes_merged'] / max(1, len(self.nodes) + self.merge_stats['nodes_merged']),
                'edge_consolidation_ratio': self.merge_stats['edges_merged'] / max(1, len(self.edges)),
                'exact_match_percentage': self.merge_stats['exact_matches'] / max(1, self.merge_stats['exact_matches'] + self.merge_stats['semantic_matches']) * 100
            },
            'graph_quality': {
                'average_evidence_per_edge': np.mean([sum(e.evidence_count for e in edges) for edges in self.edges.values()]) if self.edges else 0,
                'concept_coverage': len(set().union(*[node.semantic_keywords for node in self.nodes.values()])) if self.nodes else 0,
                'intervention_ratio': sum(1 for node in self.nodes.values() if node.isIntervention == 1) / max(1, len(self.nodes))
            },
            'semantic_coherence': {
                'avg_confidence': np.mean([edge.confidence for edges in self.edges.values() for edge in edges]) if self.edges else 0,
                'relationship_diversity': len(set(edge.relationship_type for edges in self.edges.values() for edge in edges)) if self.edges else 0
            }
        }
        
        return metrics

    # Note: Remove visualize_with_edge_details, _draw_relationship_edges, _draw_comprehensive_legend,
    # analyze_graph_metrics, _calculate_comprehensive_metrics, _print_analysis_summary,
    # _create_analysis_visualizations, _export_analysis_metrics