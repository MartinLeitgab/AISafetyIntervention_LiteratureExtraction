# models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict
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
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = self._generate_node_id()
        self.canonical_text = self._canonicalize_text(self.concept_text)
        self.text_hash = self._generate_text_hash()
        self.semantic_keywords = self._extract_keywords()
    
    def _generate_node_id(self) -> str:
        """Generate a stable node ID"""
        clean_text = re.sub(r'[^\w\s]', '', self.concept_text.upper())
        return clean_text.replace(" ", "_")[:50]  # Limit length
    
    def _canonicalize_text(self, text: str) -> str:
        """Create canonical version of text for comparison"""
        canonical = re.sub(r'[^\w\s]', '', text.lower())
        return re.sub(r'\s+', ' ', canonical).strip()
    
    def _generate_text_hash(self) -> str:
        """Generate hash of canonical text for quick comparison"""
        return hashlib.md5(self.canonical_text.encode()).hexdigest()[:8]
    
    def _extract_keywords(self) -> List[str]:
        """Extract semantic keywords for similarity comparison"""
        keywords = []
        text_lower = self.canonical_text
        ai_safety_terms = {
            'alignment', 'misalignment', 'safety', 'oversight', 'reward', 'hacking',
            'deception', 'constitutional', 'interpretability', 'robustness', 'scaling',
            'capability', 'control', 'evaluation', 'training', 'optimization', 'mesa',
            'inner', 'outer', 'objective', 'gradient', 'intervention', 'monitoring',
            'detection', 'prevention', 'mitigation', 'framework', 'protocol', 'system'
        }
        for term in ai_safety_terms:
            if term in text_lower:
                keywords.append(term)
        return list(set(keywords))

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
    relationship_strength: float = field(default=1.0)
    canonical_relationship: str = field(default="", init=False)
    semantic_hash: str = field(default="", init=False)
    source_papers: List[str] = field(default_factory=list)
    evidence_count: int = field(default=1)
    
    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = self._generate_edge_id()
        self.relationship_type = self._infer_relationship_type()
        self.relationship_strength = self._calculate_strength()
        self.canonical_relationship = self._canonicalize_relationship()
        self.semantic_hash = self._generate_semantic_hash()
    
    def _generate_edge_id(self) -> str:
        """Generate stable edge ID"""
        source_clean = "_".join(self.source_nodes).replace(" ", "_")[:30]
        target_clean = "_".join(self.target_nodes).replace(" ", "_")[:30]
        return f"{source_clean}_TO_{target_clean}"
    
    def _infer_relationship_type(self) -> str:
        """Enhanced relationship type inference"""
        text_lower = self.edge_text.lower()
        relationship_patterns = {
            'prevents': ['prevents', 'mitigates', 'reduces', 'alleviates', 'blocks', 'stops'],
            'enables': ['enables', 'facilitates', 'allows', 'supports', 'helps'],
            'causes': ['causes', 'leads to', 'results in', 'triggers', 'produces'],
            'moderates': ['moderates', 'influences', 'affects', 'modifies'],
            'mediates': ['mediates', 'through', 'via', 'by means of'],
            'necessitates': ['necessitates', 'requires', 'demands', 'needs'],
            'correlates': ['correlates', 'associated with', 'related to']
        }
        for rel_type, patterns in relationship_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return rel_type
        return 'causes'
    
    def _calculate_strength(self) -> float:
        """Calculate relationship strength based on confidence and text analysis"""
        base_strength = self.confidence / 5.0
        strength_multipliers = {
            'causes': 1.0,
            'prevents': 0.9,
            'enables': 0.8,
            'necessitates': 1.1,
            'correlates': 0.6
        }
        multiplier = strength_multipliers.get(self.relationship_type, 1.0)
        return min(base_strength * multiplier, 1.0)
    
    def _canonicalize_relationship(self) -> str:
        """Create canonical representation of the relationship"""
        canonical = re.sub(r'[^\w\s]', '', self.edge_text.lower())
        return re.sub(r'\s+', ' ', canonical).strip()
    
    def _generate_semantic_hash(self) -> str:
        """Generate semantic hash for relationship comparison"""
        semantic_content = f"{self.relationship_type}_{self.canonical_relationship}"
        return hashlib.md5(semantic_content.encode()).hexdigest()[:8]