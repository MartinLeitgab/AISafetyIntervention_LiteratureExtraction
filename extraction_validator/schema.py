# pyright: strict
"""
In this file are the validation prompt and the expected output schema. It uses Pydantic models to define the structure of the validation report, in a very lenient way to allow for partial completions from GPT. Then it generates a JSON schema from these models to include in the prompt.
"""
from typing import Any, Generic, List, Literal, Optional, TypeVar
from pydantic import BaseModel, Field, TypeAdapter, field_validator
from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import PaperSchema
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT  # type: ignore[reportMissingImports, reportMissingTypeStubs]


T = TypeVar("T")

class ValidatedDataOrOriginalOnError( BaseModel, Generic[T],):
    """Wrapper type that tries to validate data of type T, but if validation fails, keeps the original data for inspection."""
    data: T | None
    original: Any
    is_VDoOoE: Literal[True] = True



def tolerant_list(v: Optional[List[T] | ValidatedDataOrOriginalOnError[T]], adapter: TypeAdapter[T]) -> Optional[List[ValidatedDataOrOriginalOnError[T]]]:
    """If there is an inner validation error in a list, drop the invalid item instead of failing the whole list."""
    out: List[ValidatedDataOrOriginalOnError[T]] = []
    if v is None:
        return None
    for item in v:
        try:
            if isinstance(item, dict) and "is_VDoOoE" in item:
                out.append(ValidatedDataOrOriginalOnError(data=adapter.validate_python(item.get("data")), original=None))  # type: ignore
                continue
            out.append(ValidatedDataOrOriginalOnError(data=adapter.validate_python(item), original=None))
        except Exception:
            if isinstance(item, dict) and "is_VDoOoE" in item:
                out.append(ValidatedDataOrOriginalOnError(data=None, original=item.get("original")))  # type: ignore
                continue
            out.append(ValidatedDataOrOriginalOnError(data=None, original=item))
    return out


class AddEdgeFix(BaseModel):
    type: str = Field( description="edge type")
    source_node: Optional[str] = Field(None, description="Name of the source node")
    """This field is optional because when adding an edge from a new node, the source node is implicit, but the llm may get confused and include it anyway."""

    target_node: Optional[str] = Field(None, description="Name of the target node")

    description: str = Field(description="Description of the edge")

    edge_confidence: Literal[1, 2, 3, 4, 5] = Field( description="Confidence score of the edge")

    def get_other_node_name(self, new_node_name: str) -> Optional[str]:
        """The LLM may get confused and include the source_node even when adding an edge from a new node. Or swap source and target. This function helps retrieve the other node name."""
        if self.target_node is not None and self.target_node != new_node_name:
            return self.target_node
        if self.source_node is not None and self.source_node != new_node_name:
            return self.source_node
        return None

class NewNode(BaseModel):
    """Based off of Node, but with different validation rules for the fields."""
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

class AddNodeFix(BaseModel):
    new_node: NewNode
    new_edges: List[ValidatedDataOrOriginalOnError[AddEdgeFix]] = Field( description="List of edges from this node to existing nodes")


    @field_validator("new_edges", mode="before")
    @classmethod
    def _tolerant_new_edges(cls, v: Optional[List[AddEdgeFix]]) -> Optional[List[ValidatedDataOrOriginalOnError[AddEdgeFix]]]:
        return tolerant_list(v, TypeAdapter(AddEdgeFix))
    

class MergeFix(BaseModel):
    new_node_name: str = Field(description="Name of the merged node")
    nodes_to_merge: List[str] = Field(description="List of node names to merge")


class DeleteFix(BaseModel):
    node_name: str = Field(description="Name of the node or edge to be deleted")
    reason: str = Field(description="explanation")

class DeleteEdgeFix(BaseModel):
    source_node_name: str = Field( description="Name of the source node")
    target_node_name: str = Field( description="Name of the target node")
    reason: str = Field( description="explanation")


class ChangeNodeFieldFix(BaseModel):
    node_name: str = Field( description="Name of the node to change")
    field: str = Field( description="Field name to change")
    json_new_value: str = Field( description="Json string of the New value for the field")
    reason: str = Field( description="explanation")

class ProposedFixes(BaseModel):
    add_nodes: Optional[List[ValidatedDataOrOriginalOnError[AddNodeFix]]] = None
    merges: Optional[List[ValidatedDataOrOriginalOnError[MergeFix]]] = None
    deletions: Optional[List[ValidatedDataOrOriginalOnError[DeleteFix]]] = None
    edge_deletions: Optional[List[ValidatedDataOrOriginalOnError[DeleteEdgeFix]]] = None
    change_node_fields: Optional[List[ValidatedDataOrOriginalOnError[ChangeNodeFieldFix]]] = None
    @field_validator("add_nodes", mode="before")
    @classmethod
    def _tolerant_add_nodes(cls, v: Optional[List[AddNodeFix]]) -> Optional[List[ValidatedDataOrOriginalOnError[AddNodeFix]]]:
        return tolerant_list(v, TypeAdapter(AddNodeFix))
    
    @field_validator("merges", mode="before")
    @classmethod
    def _tolerant_merges(cls, v: Optional[List[MergeFix]]) -> Optional[List[ValidatedDataOrOriginalOnError[MergeFix]]]:
        return tolerant_list(v, TypeAdapter(MergeFix))
    
    @field_validator("deletions", mode="before")
    @classmethod
    def _tolerant_deletions(cls, v: Optional[List[DeleteFix]]) -> Optional[List[ValidatedDataOrOriginalOnError[DeleteFix]]]:
        return tolerant_list(v, TypeAdapter(DeleteFix))
    
    @field_validator("edge_deletions", mode="before")
    @classmethod
    def _tolerant_edge_deletions(cls, v: Optional[List[DeleteEdgeFix]]) -> Optional[List[ValidatedDataOrOriginalOnError[DeleteEdgeFix]]]:
        return tolerant_list(v, TypeAdapter(DeleteEdgeFix))
    
    @field_validator("change_node_fields", mode="before")
    @classmethod
    def _tolerant_change_node_fields(cls, v: Optional[List[ChangeNodeFieldFix]]) -> Optional[List[ValidatedDataOrOriginalOnError[ChangeNodeFieldFix]]]:
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
    expected_edges_from_source: Optional[List[ValidatedDataOrOriginalOnError[CoverageExpectedEdge]]] = None

    @field_validator("expected_edges_from_source", mode="before")
    @classmethod
    def _tolerant_expected_edges(cls, v: Optional[List[CoverageExpectedEdge]]) -> Optional[List[ValidatedDataOrOriginalOnError[CoverageExpectedEdge]]]:
        return tolerant_list(v, TypeAdapter(CoverageExpectedEdge))

class ValidationReport(BaseModel):
    schema_check: Optional[List[ValidatedDataOrOriginalOnError[SchemaIssue]]] = None
    referential_check: Optional[List[ValidatedDataOrOriginalOnError[ReferentialIssue]]] = None
    orphans: Optional[List[ValidatedDataOrOriginalOnError[OrphanIssue]]] = None
    duplicates: Optional[List[ValidatedDataOrOriginalOnError[DuplicateIssue]]] = None
    rationale_mismatches: Optional[List[ValidatedDataOrOriginalOnError[RationaleMismatch]]] = None
    coverage: Optional[CoverageIssue] = None

    @field_validator("schema_check", mode="before")
    @classmethod
    def _tolerant_schema_check(cls, v: Any) -> Optional[List[ValidatedDataOrOriginalOnError[SchemaIssue]]]:
        return tolerant_list(v, TypeAdapter(SchemaIssue))

    @field_validator("referential_check", mode="before")
    @classmethod
    def _tolerant_referential_check(cls, v: Any) -> Optional[List[ValidatedDataOrOriginalOnError[ReferentialIssue]]]:
        return tolerant_list(v, TypeAdapter(ReferentialIssue))

    @field_validator("orphans", mode="before")
    @classmethod
    def _tolerant_orphans(cls, v: Any) -> Optional[List[ValidatedDataOrOriginalOnError[OrphanIssue]]]:
        return tolerant_list(v, TypeAdapter(OrphanIssue))

    @field_validator("duplicates", mode="before")
    @classmethod
    def _tolerant_duplicates(cls, v: Any) -> Optional[List[ValidatedDataOrOriginalOnError[DuplicateIssue]]]:
        return tolerant_list(v, TypeAdapter(DuplicateIssue))

    @field_validator("rationale_mismatches", mode="before")
    @classmethod
    def _tolerant_rationale_mismatches(cls, v: Any) -> Optional[List[ValidatedDataOrOriginalOnError[RationaleMismatch]]]:
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
8. Find any missing concept or intervention nodes that should be added
9. Output a decision on overall validity, taking into account the proposed fixes. (Ie is it valid if fixes are applied?)

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
        
        "new_node": {{
          "name": "precise description following category pattern (concept nodes) or implementation description (intervention nodes)",
          "aliases": ["2-3 canonical alternative phrasings"],
          "type": "concept|intervention",
          "description": "detailed context with 2-3 sentences for concepts, maximum implementation detail for interventions",
          "concept_category": "risk|problem analysis|theoretical insight|design rationale|implementation mechanism|validation evidence (concepts only, null for interventions)",
          "intervention_lifecycle": "1-6 (interventions only, null for concepts)",
          "intervention_lifecycle_rationale": "justification with closest preceding data source section title reference (interventions only, null for concepts)",
          "intervention_maturity": "1-4 (interventions only, null for concepts)", 
          "intervention_maturity_rationale": "evidence-based reasoning with closest preceding data source section title reference (interventions only, null for concepts)",
          "node_rationale": "why essential to fabric with closest preceding data source section title reference"
        }},
        "new_edges": [
             {{
              "type": "preferred relationship verb or closely related",
              "target_node": "exact node name match",
              "description": "clear connection explanation",
              "edge_confidence": "1-5 evidence strength",
              "edge_confidence_rationale": "evidence assessment with closest preceding data source section title reference", 
              "edge_rationale": "connection justification with closest preceding data source section title reference"
            }}
        ]
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