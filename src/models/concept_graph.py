from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class LifecyclePhase(str, Enum):
    pre_training = "pre_training"
    fine_tuning = "fine_tuning"
    rlhf = "rlhf"
    pre_deployment_testing = "pre_deployment_testing"
    deployment = "deployment"
    other = "other"


class InterventionMaturity(str, Enum):
    inferred_theoretical = "inferred_theoretical"
    theoretical = "theoretical"
    proposed = "proposed"
    tested = "tested"
    deployed = "deployed"


class RelationshipType(str, Enum):
    # Causal
    causes = "causes"
    produces = "produces"
    triggers = "triggers"
    contributes_to = "contributes_to"
    # Conditional
    requires = "requires"
    depends_on = "depends_on"
    implies = "implies"
    enables = "enables"
    # Sequential
    follows = "follows"
    precedes = "precedes"
    builds_upon = "builds_upon"
    # Refinement
    refined_by = "refined_by"
    specified_by = "specified_by"
    detailed_by = "detailed_by"
    # Solution (final edges should be one of these)
    addressed_by = "addressed_by"
    mitigated_by = "mitigated_by"
    resolved_by = "resolved_by"
    protected_against_by = "protected_against_by"
    # Correlation
    correlates_with = "correlates_with"
    associated_with = "associated_with"


class EdgeConfidence(str, Enum):
    speculative = "speculative"
    supported = "supported"
    validated = "validated"
    established = "established"
    proven = "proven"


class ConceptNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the node for edge references")
    source_doi_urls: list[str] = Field(default_factory=list, description="DOI/URLs of source publication")
    source_authors: list[str] = Field(default_factory=list, description="Authors of source publication")
    source_institutions: list[str] = Field(default_factory=list, description="Institutions of authors of source publication")
    source_timestamps: list[str] = Field(default_factory=list, description="Timestamps of DOI/URL (e.g., arXiv submitted/updated)")
    name: str = Field(..., description="Text/name of concept")
    is_intervention: bool = Field(..., description="Whether this concept is an intervention (no outgoing edges)")
    stage_phases: list[LifecyclePhase] = Field(
        default_factory=list,
        description="Intervention maturity per prompt scale; only set for intervention nodes",
    )
    maturity: Optional[InterventionMaturity] = Field(
        default=None,
        description="Intervention maturity per prompt scale; only set for intervention nodes",
    )
    implemented_in_model_development: bool | None = Field(
        default=None,
        description="Was this concept/intervention implemented in model development vs reference papers",
    )


class DirectedEdge(BaseModel):
    source_doi_urls: list[str] = Field(default_factory=list, description="DOI/URLs of source publication")
    source_authors: list[str] = Field(default_factory=list, description="Authors of source publication")
    source_institutions: list[str] = Field(default_factory=list, description="Institutions of authors of source publication")
    source_timestamps: list[str] = Field(default_factory=list, description="Timestamps of DOI/URL")
    relationship: RelationshipType = Field(..., description="Relationship type per prompt taxonomy")
    source_node_ids: list[str] = Field(default_factory=list, description="IDs of source node(s)")
    target_node_ids: list[str] = Field(default_factory=list, description="IDs of target node(s)")
    confidence: Optional[EdgeConfidence] = Field(
        default=None,
        description="Confidence of connection per prompt scale",
    )


class ConceptGraph(BaseModel):
    nodes: list[ConceptNode] = Field(default_factory=list)
    edges: list[DirectedEdge] = Field(default_factory=list)


