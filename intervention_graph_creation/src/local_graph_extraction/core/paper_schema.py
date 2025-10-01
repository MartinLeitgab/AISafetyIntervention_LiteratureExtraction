from typing import List, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

from intervention_graph_creation.src.local_graph_extraction.core.edge import GraphEdge
from intervention_graph_creation.src.local_graph_extraction.core.node import GraphNode


class Meta(BaseModel):
    key: str = Field(min_length=1, max_length=64, description="metadata key")
    value: Union[str, List[str]] = Field(min_length=1, max_length=256, description="metadata value")

    model_config = ConfigDict(extra="forbid")

    @field_validator("key", "value", mode="before")
    @classmethod
    def _strip_nonempty(cls, v):
        if isinstance(v, str):
            v2 = v.strip()
            if not v2:
                raise ValueError("must be non-empty")
            return v2
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if not isinstance(item, str):
                    raise ValueError("all items must be strings")
                item2 = item.strip()
                if not item2:
                    raise ValueError("list items must be non-empty")
                new_list.append(item2)
            return new_list
        return v

class PaperSchema(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    meta: List[Meta] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")
