from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import numpy as np


class Edge(BaseModel):
    type: str = Field(min_length=1, max_length=64, description="relationship label verb")
    source_node: str = Field(min_length=1, description="source node name")
    target_node: str = Field(min_length=1, description="target node name")
    description: str = Field(min_length=1, description="concise description of logical connection")
    edge_confidence: int = Field(ge=1, le=5, description="1-5")

    model_config = ConfigDict(extra="forbid")

    @field_validator("type", "source_node", "target_node", "description")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("must be non-empty")
        return v2
    
    @model_validator(mode='before')
    @classmethod
    def remove_unwanted_fields(cls, values):
        """Rationale fields are not stored in the database"""
        if isinstance(values, dict):
            ignored_fields = ["edge_confidence_rationale", "edge_rationale"]
            for field in ignored_fields:
                values.pop(field, None)
        return values

    @model_validator(mode="after")
    def _no_self_loop(self):
        if self.source_node == self.target_node:
            raise ValueError("self-loop edges are not allowed (source_node == target_node)")
        return self


class GraphEdge(Edge):
    """Extended Edge class with embedding and concept metadata support."""
    embedding: Optional[np.ndarray] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        # Handle embedding separately to avoid pydantic validation issues
        embedding = data.pop('embedding', None)
        super().__init__(**data)
        self.embedding = embedding
