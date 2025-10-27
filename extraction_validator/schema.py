# pyright: strict
"""
In this file are the validation prompt and the expected output scheme
as TypedDicts.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel
from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import PaperSchema
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT  # type: ignore[reportMissingImports, reportMissingTypeStubs]



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
- Nodes must have: name, aliases (2-3 items), type (concept|intervention), description (1-2 sentences)
- If type=concept: must have concept_category
- If type=intervention: must have intervention_lifecycle (1-6) and intervention_maturity (1-4)
- Edges must have: type, source_node, target_node, description, edge_confidence (1-5)
- All node names referenced in edges must exist as nodes

VALIDATION_TASKS:
1. Check JSON structure and schema compliance
2. Verify all referenced nodes exist (referential integrity)
3. Identify orphaned nodes (not connected to any edges)
4. Find duplicate nodes/edges that should be merged
5. Check if extracted knowledge matches source text evidence
6. Assess coverage - are important edges from source missing?

Return your analysis in this EXACT JSON format:

{{
  "decision": {{
    "is_valid_json": true/false,
    "has_blockers": true/false,
    "flag_underperformance": true/false,
    "summary": "One-paragraph executive summary of validation results"
  }},
  "validation_report": {{
    "schema_check": [
      {{ "severity": "BLOCKER|MAJOR|MINOR|STYLE", "issue": "description", "where": "path.to.field", "suggestion": "fix suggestion" }}
    ],
    "referential_check": [
      {{ "severity": "BLOCKER|MAJOR|MINOR", "issue": "description", "ids": ["id1","id2"] }}
    ],
    "orphans": [
      {{ "node_id": "node_name", "reason": "explanation", "suggested_fix": "what to do" }}
    ],
    "duplicates": [
      {{ "kind": "node|edge", "ids": ["name1","name2"], "merge_strategy": "keep X, merge props, retarget edges" }}
    ],
    "rationale_mismatches": [
      {{ "issue": "description", "evidence": "exact quote from DATA_SOURCE", "fix": "suggested fix" }}
    ],
    "coverage": {{
      "expected_edges_from_source": [
        {{ "title": "edge name", "evidence": "quote from source", "status": "covered|partially_covered|missing", "mapped_edge_ids": ["edge1"] }}
      ]
    }}
  }},
  "rationale_record": {{
    "method": "systematic_validation",
    "notes": [
      "Key validation decisions with specific citations to DATA_SOURCE"
    ]
  }}
}}

Be surgical and precise. Cite specific evidence from DATA_SOURCE. No extra text - return only valid JSON."""

class GeneratedFixProps(BaseModel):
    stable_key: Optional[str] = None

class FixProps(BaseModel):
    stable_key: str

class GeneratedAddNodeFix(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    props: Optional[GeneratedFixProps] = None

class AddNodeFix(BaseModel):
    id: str
    type: str
    name: str
    props: FixProps

class GeneratedMergFix(BaseModel):
    target_id: Optional[str] = None
    absorbed_ids: Optional[List[str]] = None
    retargeted_edge_ids: Optional[List[str]] = None

class MergeFix(BaseModel):
    target_id: str
    absorbed_ids: List[str]
    retargeted_edge_ids: List[str]

class GeneratedDeleteFix(BaseModel):
    kind: Optional[Literal["node", "edge"] | str] = None
    id: Optional[str] = None
    reason: Optional[str] = None

class DeleteFix(BaseModel):
    kind: Literal["node", "edge"]
    id: str
    reason: str

class GeneratedProposedFixes(BaseModel):
    add_nodes: Optional[List[GeneratedAddNodeFix]] = None
    merges: Optional[List[GeneratedMergFix]] = None
    deletions: Optional[List[GeneratedDeleteFix]] = None

class ProposedFixes(BaseModel):
    add_nodes: List[AddNodeFix]
    merges: List[MergeFix]
    deletions: List[DeleteFix]


  

class SchemaIssue(BaseModel):
    severity: Optional[Literal["BLOCKER", "MAJOR", "MINOR", "STYLE"] | str] = None
    issue: Optional[str] = None
    where: Optional[str] = None
    suggestion: Optional[str] = None

class ReferentialIssue(BaseModel):
    severity: Optional[Literal["BLOCKER", "MAJOR", "MINOR"] | str] = None
    issue: Optional[str] = None
    ids: Optional[List[str]] = None

class OrphanIssue(BaseModel):
    node_id: Optional[str] = None
    reason: Optional[str] = None
    suggested_fix: Optional[str] = None

class DuplicateIssue(BaseModel):
    kind: Optional[Literal["node", "edge"] | str] = None
    ids: Optional[List[str]] = None
    merge_strategy: Optional[str] = None

class RationaleMismatch(BaseModel):
    issue: Optional[str] = None
    evidence: Optional[str] = None
    fix: Optional[str] = None

class CoverageExpectedEdge(BaseModel):
    title: Optional[str] = None
    evidence: Optional[str] = None
    status: Optional[Literal["covered", "partially_covered", "missing"] | str] = None
    mapped_edge_ids: Optional[List[str]] = None

class CoverageIssue(BaseModel):
    expected_edges_from_source: Optional[List[CoverageExpectedEdge]] = None

class ValidationReport(BaseModel):
    schema_check: Optional[List[SchemaIssue]] = None
    referential_check: Optional[List[ReferentialIssue]] = None
    orphans: Optional[List[OrphanIssue]] = None
    duplicates: Optional[List[DuplicateIssue]] = None
    rationale_mismatches: Optional[List[RationaleMismatch]] = None
    coverage: Optional[CoverageIssue] = None

class Decision(BaseModel):
    is_valid_json: Optional[bool] = None
    has_blockers: Optional[bool] = None
    flag_underperformance: Optional[bool] = None
    summary: Optional[str] = None

class RationaleRecord(BaseModel):
    method: Optional[str] = None
    notes: Optional[List[str]] = None

class GPT_Assessment(BaseModel):
    decision: Optional[Decision] = None
    validation_report: Optional[ValidationReport] = None
    proposed_fixes: Optional[GeneratedProposedFixes] = None
    rationale_record: Optional[RationaleRecord] = None
