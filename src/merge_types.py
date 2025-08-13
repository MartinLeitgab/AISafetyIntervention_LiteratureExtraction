from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    """Minimal source metadata available from extractions.

    Fields are optional because many are not available from the current pipeline.
    """

    paper_id: str = Field(
        ..., description="Identifier derived from the output filename (stem)"
    )
    title: Optional[str] = Field(default=None, description="Paper title if known")
    section: Optional[str] = Field(default=None, description="Section heading if known")
    paragraph_id: Optional[str] = Field(
        default=None, description="Paragraph identifier if known"
    )


class LinkedEdgeSummary(BaseModel):
    """Summarized view of an edge touching a node for LLM context."""

    edge_type: str
    rationale: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: SourceMetadata


class NodeAggregate(BaseModel):
    """Aggregate information for a node across multiple papers/edges."""

    node_key: str = Field(
        ...,
        description="Stable key used to identify this aggregate (e.g., canonical_name lowercased)",
    )
    text: str = Field(..., description="Primary surface name for the node (name)")
    canonical_text: str = Field(
        ..., description="Canonical name provided by extraction"
    )
    aliases: List[str] = Field(default_factory=list)
    notes: List[str] = Field(
        default_factory=list, description="Collected node notes across occurrences"
    )
    confidence_samples: List[float] = Field(default_factory=list)
    linked_edges: List[LinkedEdgeSummary] = Field(default_factory=list)
    sources: List[SourceMetadata] = Field(default_factory=list)


class EdgeAggregate(BaseModel):
    """Aggregate information for an edge across multiple papers."""

    edge_key: str
    edge_type: str
    text: str = Field(..., description="Edge-level description if available")
    source_nodes: List[str]
    target_nodes: List[str]
    rationales: List[str] = Field(default_factory=list)
    confidence_samples: List[float] = Field(default_factory=list)
    sources: List[SourceMetadata] = Field(default_factory=list)


class NodeViewForComparison(BaseModel):
    text: str
    aliases: List[str]
    context: List[str] = Field(
        default_factory=list,
        description="Context strings: node notes and related rationales",
    )
    source_metadata: List[SourceMetadata]
    linked_edges: List[LinkedEdgeSummary]


class NodeComparisonInput(BaseModel):
    """Input payload for the LLM to compare two nodes (data only)."""

    node_a: NodeViewForComparison
    node_b: NodeViewForComparison


class EdgeViewForComparison(BaseModel):
    text: str
    source_nodes: List[str]
    target_nodes: List[str]
    context: List[str] = Field(default_factory=list)
    source_metadata: List[SourceMetadata]


class EdgeComparisonInput(BaseModel):
    edge_a: EdgeViewForComparison
    edge_b: EdgeViewForComparison
