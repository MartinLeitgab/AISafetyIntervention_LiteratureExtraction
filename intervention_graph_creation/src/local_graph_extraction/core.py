from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class Node(BaseModel):
    name: str = Field(description="concise natural-language description of node")
    type: Literal["concept", "intervention"]
    description: str = Field(description="detailed technical description of node (1-2 sentences only)")

    aliases: Optional[List[str]] = Field(default=None, description="2-3 alternative concise descriptions of node")
    concept_category: Optional[str] = Field(default=None, description="from examples or create a new category ("
                                                                      "concept nodes only, otherwise null)")
    intervention_lifecycle: Optional[int] = Field(default=None, ge=1, le=6,
                                                  description="1-6 (only for intervention nodes)")
    intervention_maturity: Optional[int] = Field(default=None, ge=1, le=4,
                                                 description="1-4 (only for intervention nodes)")
    meta: Optional[List['Meta']] = Field(default_factory=list, description="metadata key/value pairs for this node")

    model_config = ConfigDict(extra="forbid")

    @field_validator("name", "description")
    @classmethod
    def _non_empty_str(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("must be non-empty")
        return v2

    @field_validator("concept_category")
    @classmethod
    def _trim_optional_str(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v

    @field_validator("aliases")
    @classmethod
    def _validate_aliases(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if len(cleaned) != len(v):
            raise ValueError("aliases contain empty/invalid strings")
        if not (1 <= len(cleaned) <= 3):
            raise ValueError("aliases must have 1–3 items")
        for s in cleaned:
            if len(s) > 200:
                raise ValueError("alias too long (>200 chars)")
        return cleaned

    @model_validator(mode="after")
    def _cross_field_rules(self):
        if self.type == "concept":
            if self.intervention_lifecycle is not None or self.intervention_maturity is not None:
                raise ValueError("intervention_* must be null for concept nodes")
        else:
            if self.concept_category is not None:
                raise ValueError("concept_category must be null for intervention nodes")
            if self.intervention_lifecycle is None or self.intervention_maturity is None:
                raise ValueError("intervention_lifecycle and intervention_maturity are required for intervention nodes")
        return self


class Edge(BaseModel):
    type: str = Field(min_length=1, max_length=64, description="relationship label verb")
    source_node: str = Field(min_length=1, description="source node name")
    target_node: str = Field(min_length=1, description="target node name")
    description: str = Field(min_length=1, description="concise description of logical connection")
    edge_confidence: int = Field(ge=1, le=5, description="1-5")
    meta: Optional[List['Meta']] = Field(default_factory=list, description="metadata key/value pairs for this edge")

    model_config = ConfigDict(extra="forbid")

    @field_validator("type", "source_node", "target_node", "description")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("must be non-empty")
        return v2

    @model_validator(mode="after")
    def _no_self_loop(self):
        if self.source_node == self.target_node:
            raise ValueError("self-loop edges are not allowed (source_node == target_node)")
        return self
    

class Meta(BaseModel):
    key: str = Field(min_length=1, max_length=64, description="metadata key")
    value: Any = Field(description="metadata value (string, number, bool, or array)")

    model_config = ConfigDict(extra="forbid")

    @field_validator("key")
    @classmethod
    def _strip_nonempty_key(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("must be non-empty")
        return v2


class LogicalChain(BaseModel):
    title: Optional[str] = Field(default=None, description="concise natural-language description of logical chain")
    edges: List[Edge] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class PaperSchema(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    logical_chains: List[LogicalChain] = Field(default_factory=list)
    meta: List[Meta] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")
