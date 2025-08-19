from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

class Node(BaseModel):
    name: str = Field(default=None, description="concise natural-language description of node")
    aliases: List[str] = Field(default=None, description="2-3 alternative concise descriptions of node")
    type: Literal["concept", "intervention"]
    description: str= Field(default=None, description="detailed technical description of node (1-2 sentences only)")
    concept_category: Optional[str] = Field(default=None, description="from examples or create a new category (concept nodes only, otherwise null)")
    intervention_lifecycle: Optional[int] = Field(default=None, ge=1, le=6, description="1-6 (only for intervention nodes)")
    intervention_maturity: Optional[int] = Field(default=None, ge=1, le=4, description="1-4 (only for intervention nodes)")
    model_config = ConfigDict(extra="forbid")

class Edge(BaseModel):
    type: str = Field(default=None, description="relationship label verb")
    source_node: str = Field(default=None, description="source node name")
    target_node: str = Field(default=None, description="target node name")
    description: str = Field(default=None, description="concise description of logical connection")
    edge_confidence: int = Field(default=None, ge=1, le=5, description="1-5")
    model_config = ConfigDict(extra="forbid")
    
class LogicalChain(BaseModel):
    title: str = Field(default=None, description="concise natural-language description of logical chain")
    edges: List[Edge]
    model_config = ConfigDict(extra="forbid")

class PaperSchema(BaseModel):
    nodes: List[Node]
    logical_chains: List[LogicalChain1]
    model_config = ConfigDict(extra="forbid")