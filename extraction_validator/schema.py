# pyright: strict
"""
In this file are the validation prompt and the expected output schema. It uses Pydantic models to define the structure of the validation report, in a very lenient way to allow for partial completions from GPT. Then it generates a JSON schema from these models to include in the prompt.
"""
from typing import Any, List, Literal, Optional, TypeVar
from pydantic import BaseModel, Field, TypeAdapter, field_validator
from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import PaperSchema
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT  # type: ignore[reportMissingImports, reportMissingTypeStubs]



T = TypeVar("T")
def tolerant_list(v: Optional[List[T]], adapter: TypeAdapter[T]) -> Optional[List[T]]:
    """If there is an inner validation error in a list, drop the invalid item instead of failing the whole list."""
    out: List[T] = []
    if v is None:
        return None
    for item in v:
        try:
            out.append(adapter.validate_python(item))
        except Exception:
            # drop invalid item
            continue
    return out


class AddEdgeFix(BaseModel):
    type: str = Field(..., description="edge type")

    target_node: str = Field(..., description="Name of the target node")

    description: str = Field(..., description="Description of the edge")

    edge_confidence: Literal[1, 2, 3, 4, 5] = Field(..., description="Confidence score of the edge")


class AddNodeFix(BaseModel):
    type: Literal["concept", "intervention"]
    name: str = Field(..., description="Name of the node to be added")
    edges: List[AddEdgeFix]
    intervention_lifecycle: Optional[Literal[1,2,3,4,5,6]] = Field(..., description="Only for intervention nodes: Lifecycle stage of the intervention")
    intervention_maturity: Optional[Literal[1,2,3,4]] = Field(..., description="Only for intervention nodes: Maturity level of the intervention")

    @field_validator("edges", mode="before")
    @classmethod
    def _tolerant_edges(cls, v: Optional[List[AddEdgeFix]]) -> Optional[List[AddEdgeFix]]:
        return tolerant_list(v, TypeAdapter(AddEdgeFix))


class MergeFix(BaseModel):
    new_node_name: str = Field(description="Name of the merged node")
    nodes_to_merge: List[str] = Field(description="List of node names to merge")


class DeleteFix(BaseModel):
    node_name: str = Field(description="Name of the node or edge to be deleted")
    reason: str = Field(description="explanation")

class DeleteEdgeFix(BaseModel):
    source_node_name: str = Field(..., description="Name of the source node")
    target_node_name: str = Field(..., description="Name of the target node")
    reason: str = Field(..., description="explanation")


class ChangeNodeFieldFix(BaseModel):
    node_name: str = Field(..., description="Name of the node to change")
    field: str = Field(..., description="Field name to change")
    json_new_value: str = Field(..., description="Json string of the New value for the field")
    reason: str = Field(..., description="explanation")

class ProposedFixes(BaseModel):
    add_nodes: Optional[List[AddNodeFix]] = None
    merges: Optional[List[MergeFix]] = None
    deletions: Optional[List[DeleteFix]] = None
    edge_deletions: Optional[List[DeleteEdgeFix]] = None
    change_node_fields: Optional[List[ChangeNodeFieldFix]] = None

    @field_validator("add_nodes", mode="before")
    @classmethod
    def _tolerant_add_nodes(cls, v: Optional[List[AddNodeFix]]) -> Optional[List[AddNodeFix]]:
        return tolerant_list(v, TypeAdapter(AddNodeFix))
    
    @field_validator("merges", mode="before")
    @classmethod
    def _tolerant_merges(cls, v: Optional[List[MergeFix]]) -> Optional[List[MergeFix]]:
        return tolerant_list(v, TypeAdapter(MergeFix))
    
    @field_validator("deletions", mode="before")
    @classmethod
    def _tolerant_deletions(cls, v: Optional[List[DeleteFix]]) -> Optional[List[DeleteFix]]:
        return tolerant_list(v, TypeAdapter(DeleteFix))
    
    @field_validator("edge_deletions", mode="before")
    @classmethod
    def _tolerant_edge_deletions(cls, v: Optional[List[DeleteEdgeFix]]) -> Optional[List[DeleteEdgeFix]]:
        return tolerant_list(v, TypeAdapter(DeleteEdgeFix))
    
    @field_validator("change_node_fields", mode="before")
    @classmethod
    def _tolerant_change_node_fields(cls, v: Optional[List[ChangeNodeFieldFix]]) -> Optional[List[ChangeNodeFieldFix]]:
        return tolerant_list(v, TypeAdapter(ChangeNodeFieldFix))
    
class SchemaIssue(BaseModel):
    severity: Optional[Literal["BLOCKER", "MAJOR", "MINOR", "STYLE"] | str] = None
    

    issue: Optional[str] = Field(default=None, description="Description")

    where: Optional[str] = Field(default=None, description="Path to the field")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix")

class ReferentialIssue(BaseModel):
    severity: Optional[Literal["BLOCKER", "MAJOR", "MINOR"] | str] = None
    issue: Optional[str] = Field(default=None, description="Description")
    names: Optional[List[str]] = Field(default=None, description="List of related node names")

class OrphanIssue(BaseModel):
    node_name: Optional[str] = Field(default=None, description="Name of the orphaned node")
    reason: Optional[str] = Field(default=None, description="Explanation for orphan status")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested fix for orphan issue")

class DuplicateIssue(BaseModel):
    kind: Optional[Literal["node", "edge"] | str] = None
    names: Optional[List[str]] = Field(default=None, description="List of duplicate names")
    merge_strategy: Optional[str] = Field(default=None, description="keep X, merge props, retarget edges")

class RationaleMismatch(BaseModel):
    issue: Optional[str] = Field(default=None, description="Description")
    evidence: Optional[str] = Field(default=None, description="exact quote from DATA_SOURCE")
    fix: Optional[str] = Field(default=None, description="suggested fix")

class CoverageExpectedEdge(BaseModel):
    title: Optional[str] = Field(default=None, description="Title of the expected edge")
    evidence: Optional[str | List[str]] = Field(default=None, description="quote from source")
    status: Optional[Literal["covered", "partially_covered", "missing"] | str] = None
    expected_source_node_name: Optional[List[str]] = None
    expected_target_node_name: Optional[List[str]] = None

class CoverageIssue(BaseModel):
    expected_edges_from_source: Optional[List[CoverageExpectedEdge]] = None

    @field_validator("expected_edges_from_source", mode="before")
    @classmethod
    def _tolerant_expected_edges(cls, v: Optional[List[CoverageExpectedEdge]]) -> Optional[List[CoverageExpectedEdge]]:
        return tolerant_list(v, TypeAdapter(CoverageExpectedEdge))

class ValidationReport(BaseModel):
    schema_check: Optional[List[SchemaIssue]] = None
    referential_check: Optional[List[ReferentialIssue]] = None
    orphans: Optional[List[OrphanIssue]] = None
    duplicates: Optional[List[DuplicateIssue]] = None
    rationale_mismatches: Optional[List[RationaleMismatch]] = None
    coverage: Optional[CoverageIssue] = None

    @field_validator("schema_check", mode="before")
    @classmethod
    def _tolerant_schema_check(cls, v: Any) -> Optional[List[SchemaIssue]]:
        return tolerant_list(v, TypeAdapter(SchemaIssue))

    @field_validator("referential_check", mode="before")
    @classmethod
    def _tolerant_referential_check(cls, v: Any) -> Optional[List[ReferentialIssue]]:
        return tolerant_list(v, TypeAdapter(ReferentialIssue))

    @field_validator("orphans", mode="before")
    @classmethod
    def _tolerant_orphans(cls, v: Any) -> Optional[List[OrphanIssue]]:
        return tolerant_list(v, TypeAdapter(OrphanIssue))

    @field_validator("duplicates", mode="before")
    @classmethod
    def _tolerant_duplicates(cls, v: Any) -> Optional[List[DuplicateIssue]]:
        return tolerant_list(v, TypeAdapter(DuplicateIssue))

    @field_validator("rationale_mismatches", mode="before")
    @classmethod
    def _tolerant_rationale_mismatches(cls, v: Any) -> Optional[List[RationaleMismatch]]:
        return tolerant_list(v, TypeAdapter(RationaleMismatch))

class Decision(BaseModel):
    is_valid_json: Optional[bool] = None
    has_blockers: Optional[bool] = Field(default=None, description="Whether there are BLOCKER issues, that have not been solved by proposed fixes")
    flag_underperformance: Optional[bool] = None
    valid_and_mergeable_after_fixes: Optional[bool] = Field(default=None, description="Whether the KG would be valid after applying the proposed fixes")
    summary: Optional[str] = Field(default=None, description="One-paragraph executive summary of validation results")

class RationaleRecord(BaseModel):
    method: Optional[str] = Field(default=None, description="systematic_validation")
    notes: Optional[List[str]] = Field(default=None, description="Key validation decisions with specific citations to DATA_SOURCE")


class GPT_Assessment(BaseModel):
    validation_report: Optional[ValidationReport] = None
    rationale_record: Optional[RationaleRecord] = None
    proposed_fixes: Optional[ProposedFixes] = None
    decision: Optional[Decision] = Field(default=None, description="Overall validation decision")


def create_validation_prompt(
         original_text: str, kg_output: PaperSchema
    ) -> str:
        """Create a comprehensive validation prompt for GPT."""

        return f"""You are KG-Judge, a precise and rigorous auditor for knowledge graphs. 

Your task is to validate this knowledge graph extraction against the source text and return a structured validation report.

DATA_SOURCE:
{original_text}

ORIGINAL_PROMPT:
{PROMPT_EXTRACT}

EXTRACTED_KNOWLEDGE_GRAPH:
{kg_output.model_dump_json(indent=2)}

SCHEMA_REQUIREMENTS:
- Nodes must have: name, aliases (2-3 items but not a BLOCKER), type (concept|intervention), description (1-2 sentences amount of sentences not a BLOCKER)
- If type=concept: must have concept_category, must not have intervention_lifecycle or intervention_maturity
- If type=intervention: must have intervention_lifecycle (1-6) and intervention_maturity (1-4), must not have concept_category
- Edges must have: type, source_node, target_node, description, edge_confidence (1-5)
- All node names referenced in edges must exist as nodes

VALIDATION_TASKS:
1. Check JSON structure and schema compliance
2. Verify all referenced nodes exist (referential integrity)
3. Identify orphaned nodes (not connected to any edges)
4. Find duplicate nodes/edges that should be merged
5. Check if extracted knowledge matches source text evidence
6. Assess coverage - are important edges from source missing?
7. Propose specific fixes for any issues found
8. Output a decision on overall validity, taking into account the proposed fixes. (Ie is it valid if fixes are applied?)

Return your analysis in this EXACT JSON format:
{{
  "validation_report": {{
    "schema_check": [
      {{ "severity": "BLOCKER|MAJOR|MINOR|STYLE", "issue": "description", "where": "path.to.field", "suggestion": "fix suggestion" }}
    ],
    "referential_check": [
      {{ "severity": "BLOCKER|MAJOR|MINOR", "issue": "description", "names": ["related node name 1","related node name 2"] }}
    ],
    "orphans": [
      {{ "node_name": "node name", "reason": "explanation", "suggested_fix": "what to do" }}
    ],
    "duplicates": [
      {{ "kind": "node|edge", "names": ["duplicate name 1","duplicate name 2"], "merge_strategy": "keep X, merge props, retarget edges" }}
    ],
    "rationale_mismatches": [
      {{ "issue": "description", "evidence": "exact quote from DATA_SOURCE", "fix": "suggested fix" }}
    ],
    "coverage": {{
      "expected_edges_from_source": [
        {{
          "title": "edge name",
          "evidence": "quote from source or list of quotes",
          "status": "covered|partially_covered|missing",
          "expected_source_node_name": ["expected source node name"],
          "expected_target_node_name": ["expected target node name"]
        }}
      ]
    }}
  }},
  "proposed_fixes": {{
    "add_nodes": [
      {{
        "type": "concept|intervention",
        "name": "node name",
        "edges": [
          {{
            "type": "edge type",
            "target_node": "target node name",
            "description": "edge description",
            "edge_confidence": 1-5
          }}
        ],
        "intervention_lifecycle": 1-6,
        "intervention_maturity": 1-4
      }}
    ],
    "merges": [
      {{ "new_node_name": "name of the new node", 
      "nodes_to_merge": ["node name 1","node name 2"] }}
    ],
    "deletions": [
      {{ "node_name": "name to delete", "reason": "explanation" }}
    ],
    "edge_deletions": [
      {{ "source_node_name": "source node name", "target_node_name": "target node name", "reason": "explanation" }}
    ],
    "change_node_fields": [
      {{ "node_name": "node name", "field": "field name", "json_new_value": "new value as a json string", "reason": "explanation" }}
    ]
  }},
  "decision": {{
    "summary": "One-paragraph executive summary of validation results"
    "is_valid_json": true/false,
    "has_blockers": false/false,
    "valid_and_mergeable_after_fixes": true/false,
    "flag_underperformance": true/false,
  
  }},
  "rationale_record": {{
    "method": "systematic_validation",
    "notes": [
      "Key validation decisions with specific citations to DATA_SOURCE"
    ]
  }}
}}

Be surgical and precise. Cite specific evidence from DATA_SOURCE. No extra text - return only valid JSON."""


if __name__ == "__main__":
    print(create_validation_prompt("Example_text", PaperSchema()))