from typing import Optional, Literal, List
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import numpy as np


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
    
    @model_validator(mode='before')
    @classmethod
    def remove_unwanted_fields(cls, values):
        """Rationale fields are not stored in the database"""
        if isinstance(values, dict):
            ignored_fields = ["intervention_maturity_rationale", "intervention_lifecycle_rationale",
                              "node_rationale"]
            for field in ignored_fields:
                values.pop(field, None)
        return values

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


class GraphNode(Node):
    """Extended Node class with embedding support."""
    embedding: Optional[np.ndarray] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
