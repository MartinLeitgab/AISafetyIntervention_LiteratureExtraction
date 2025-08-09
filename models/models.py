# models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import re
import hashlib

@dataclass
class CausalNode:
    """Enhanced node structure with better merging support"""
    DOI_URL: List[str]
    authors: List[str]
    institutions: List[str]
    timestamp: List[str]
    concept_text: str
    incoming_edges: List[str] = field(default_factory=list)
    outgoing_edges: List[str] = field(default_factory=list)
    isIntervention: int = 0
    stage_in_pipeline: Optional[int] = None
    maturity_level: Optional[int] = None
    implemented: Optional[int] = None
    
    # Enhanced fields for better merging
    node_id: str = field(default="", init=False)
    aliases: List[str] = field(default_factory=list)
    canonical_text: str = field(default="", init=False)
    text_hash: str = field(default="", init=False)
    semantic_keywords: List[str] = field(default_factory=list)
    confidence_score: float = field(default=1.0)
    source_papers: List[str] = field(default_factory=list)
    merge_history: List[str] = field(default_factory=list)
    
    # Class-level keyword cache for performance
    _AI_SAFETY_TERMS: Set[str] = field(default_factory=lambda: {
        'alignment', 'misalignment', 'safety', 'oversight', 'reward', 'hacking',
        'deception', 'constitutional', 'interpretability', 'robustness', 'scaling',
        'capability', 'control', 'evaluation', 'training', 'optimization', 'mesa',
        'inner', 'outer', 'objective', 'gradient', 'intervention', 'monitoring',
        'detection', 'prevention', 'mitigation', 'framework', 'protocol', 'system',
        'adversarial', 'goodhart', 'corrigibility', 'value', 'human', 'preference',
        'instrumental', 'goal', 'specification', 'proxy', 'distributional', 'shift'
    }, init=False, repr=False)
    
    def __post_init__(self):
        if not self.concept_text.strip():
            raise ValueError("concept_text cannot be empty")
        
        if not self.node_id:
            self.node_id = self._generate_node_id()
        self.canonical_text = self._canonicalize_text(self.concept_text)
        self.text_hash = self._generate_text_hash()
        self.semantic_keywords = self._extract_keywords()
    
    def _generate_node_id(self) -> str:
        """Generate a stable, unique node ID"""
        # Remove special characters and normalize
        clean_text = re.sub(r'[^\w\s]', '', self.concept_text.strip())
        clean_text = re.sub(r'\s+', '_', clean_text.upper())
        
        # Limit length and ensure uniqueness with hash if needed
        if len(clean_text) > 50:
            # Use first 40 chars + hash suffix for very long texts
            text_hash = hashlib.md5(clean_text.encode()).hexdigest()[:8]
            return f"{clean_text[:40]}_{text_hash}"
        
        return clean_text or f"NODE_{hashlib.md5(self.concept_text.encode()).hexdigest()[:8]}"
    
    def _canonicalize_text(self, text: str) -> str:
        """Create canonical version of text for comparison"""
        # More aggressive normalization for better matching
        canonical = re.sub(r'[^\w\s]', '', text.lower().strip())
        canonical = re.sub(r'\s+', ' ', canonical)
        
        # Remove common stop words that don't affect meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in canonical.split() if word not in stop_words]
        
        return ' '.join(words)
    
    def _generate_text_hash(self) -> str:
        """Generate hash of canonical text for O(1) comparison"""
        if not self.canonical_text:
            return hashlib.md5(self.concept_text.encode()).hexdigest()[:8]
        return hashlib.md5(self.canonical_text.encode()).hexdigest()[:8]
    
    def _extract_keywords(self) -> List[str]:
        """Extract semantic keywords optimized for AI safety domain"""
        if not self.canonical_text:
            return []
        
        # Extract AI safety terms efficiently
        text_words = set(self.canonical_text.split())
        keywords = list(text_words.intersection(self._AI_SAFETY_TERMS))
        
        # Add multi-word terms
        text_lower = self.canonical_text
        multi_word_terms = [
            'reward hacking', 'mesa optimization', 'inner alignment', 'outer alignment',
            'capability control', 'ai safety', 'machine learning', 'deep learning',
            'reinforcement learning', 'language model', 'foundation model',
            'distributional shift', 'goodhart law', 'instrumental convergence'
        ]
        
        for term in multi_word_terms:
            if term in text_lower:
                keywords.append(term.replace(' ', '_'))
        
        return list(set(keywords))  # Remove duplicates
    
    def update_confidence(self) -> None:
        """Update confidence based on evidence sources"""
        # Keep confidence_score separate from evidence-based scoring
        # Don't modify the core confidence logic
        
    def is_similar_to(self, other: 'CausalNode', threshold: float = 0.8) -> bool:
        """Quick similarity check for deduplication"""
        if self.text_hash == other.text_hash:
            return True
        
        if self.isIntervention != other.isIntervention:
            return False
            
        # Quick keyword overlap check
        if self.semantic_keywords and other.semantic_keywords:
            overlap = len(set(self.semantic_keywords) & set(other.semantic_keywords))
            return overlap >= len(self.semantic_keywords) * 0.6
        
        return False


@dataclass
class CausalEdge:
    """Enhanced edge structure with better relationship modeling"""
    DOI_URL: List[str]
    authors: List[str]
    institutions: List[str]
    timestamp: List[str]
    edge_text: str
    source_nodes: List[str]
    target_nodes: List[str]
    confidence: int
    
    # Enhanced fields
    edge_id: str = field(default="", init=False)
    relationship_type: str = field(default="causes")
    canonical_relationship: str = field(default="", init=False)
    semantic_hash: str = field(default="", init=False)
    source_papers: List[str] = field(default_factory=list)
    evidence_count: int = field(default=1)
    
    # Class-level relationship patterns for performance
    _RELATIONSHIP_PATTERNS: Dict[str, List[str]] = field(default_factory=lambda: {
        'prevents': ['prevent', 'mitigate', 'reduce', 'alleviate', 'block', 'stop', 'avoid', 'counter'],
        'enables': ['enable', 'facilitate', 'allow', 'support', 'help', 'assist', 'promote'],
        'causes': ['cause', 'lead', 'result', 'trigger', 'produce', 'generate', 'create', 'induce'],
        'moderates': ['moderate', 'influence', 'affect', 'modify', 'adjust', 'regulate'],
        'mediates': ['mediate', 'through', 'via', 'by means of', 'intermediary'],
        'necessitates': ['necessitate', 'require', 'demand', 'need', 'mandate', 'compel'],
        'correlates': ['correlate', 'associate', 'relate', 'connect', 'link']
    }, init=False, repr=False)
    
    def __post_init__(self):
        if not self.edge_text.strip():
            raise ValueError("edge_text cannot be empty")
        if not self.source_nodes or not self.target_nodes:
            raise ValueError("source_nodes and target_nodes cannot be empty")
        
        if not self.edge_id:
            self.edge_id = self._generate_edge_id()
        self.relationship_type = self._infer_relationship_type()
        self.canonical_relationship = self._canonicalize_relationship()
        self.semantic_hash = self._generate_semantic_hash()
    
    def _generate_edge_id(self) -> str:
        """Generate stable edge ID with better handling"""
        # Create cleaner source and target representations
        source_str = "_".join([s.replace(" ", "_")[:20] for s in self.source_nodes])
        target_str = "_".join([t.replace(" ", "_")[:20] for t in self.target_nodes])
        
        edge_base = f"{source_str}_TO_{target_str}"
        
        # If too long, use hash
        if len(edge_base) > 100:
            return f"EDGE_{hashlib.md5(edge_base.encode()).hexdigest()[:12]}"
        
        return edge_base
    
    def _infer_relationship_type(self) -> str:
        """Optimized relationship type inference"""
        text_lower = self.edge_text.lower().strip()
        
        # Quick pattern matching
        for rel_type, patterns in self._RELATIONSHIP_PATTERNS.items():
            if any(pattern in text_lower for pattern in patterns):
                return rel_type
        
        # Default fallback
        return 'causes'
    
    def _canonicalize_relationship(self) -> str:
        """Create canonical representation of the relationship"""
        canonical = re.sub(r'[^\w\s]', '', self.edge_text.lower().strip())
        canonical = re.sub(r'\s+', ' ', canonical)
        
        # Remove common connecting words that don't affect semantic meaning
        connecting_words = {'that', 'which', 'this', 'these', 'those', 'when', 'where', 'how'}
        words = [word for word in canonical.split() if word not in connecting_words]
        
        return ' '.join(words)
    
    def _generate_semantic_hash(self) -> str:
        """Generate semantic hash for efficient relationship comparison"""
        # Include relationship type and key semantic content
        semantic_key = f"{self.relationship_type}|{self.canonical_relationship[:50]}"
        return hashlib.md5(semantic_key.encode()).hexdigest()[:8]
    
    def is_similar_to(self, other: 'CausalEdge') -> bool:
        """Quick similarity check for edge deduplication"""
        # Same relationship type required
        if self.relationship_type != other.relationship_type:
            return False
        
        # Semantic hash match
        if self.semantic_hash == other.semantic_hash:
            return True
        
        # Text similarity as fallback
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, self.canonical_relationship, other.canonical_relationship).ratio()
        return similarity >= 0.7
    
    def get_confidence_score(self) -> float:
        """Convert integer confidence to normalized float score if needed elsewhere"""
        
        return self.confidence / 5.0
    
    def update_evidence(self, new_confidence: int, new_papers: List[str]) -> None:
        """Update evidence with new information"""
        # Simple evidence count increment
        self.evidence_count += 1
        
        # Add new papers without modifying confidence
        for paper in new_papers:
            if paper not in self.source_papers:
                self.source_papers.append(paper)