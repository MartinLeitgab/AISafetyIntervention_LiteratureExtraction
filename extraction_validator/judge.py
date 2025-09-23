#!/usr/bin/env python3
"""
KG-Judge: Knowledge Graph Validation and Improvement System
A precise and rigorous auditor for knowledge graphs extracted by LLMs
"""

import json
import re
import hashlib
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from openai import AsyncOpenAI
import os


class ValidationSeverity(Enum):
    BLOCKER = "BLOCKER"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    STYLE = "STYLE"


@dataclass
class Node:
    name: str
    aliases: List[str]
    type: str  # concept|intervention
    description: str
    concept_category: Optional[str] = None
    intervention_lifecycle: Optional[int] = None
    intervention_maturity: Optional[int] = None


@dataclass
class Edge:
    type: str
    source_node: str
    target_node: str
    description: str
    edge_confidence: int


@dataclass
class LogicalChain:
    title: str
    edges: List[Edge]


@dataclass
class KnowledgeGraph:
    nodes: List[Node]
    logical_chains: List[LogicalChain]


@dataclass
class ValidationIssue:
    severity: str
    issue: str
    where: str
    suggestion: str


@dataclass
class ReferentialIssue:
    severity: str
    issue: str
    ids: List[str]


@dataclass
class OrphanNode:
    node_id: str
    reason: str
    suggested_fix: str


@dataclass
class Duplicate:
    kind: str
    ids: List[str]
    merge_strategy: str


@dataclass
class RationaleMismatch:
    issue: str
    evidence: str
    fix: str


@dataclass
class ExpectedChain:
    title: str
    evidence: str
    status: str
    mapped_chain_ids: List[str]


@dataclass
class JudgeReport:
    decision: Dict[str, Any]
    validation_report: Dict[str, Any]
    proposed_fixes: Dict[str, Any]
    final_graph: Dict[str, Any]
    rationale_record: Dict[str, Any]


class KGJudge:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize KG-Judge with OpenAI API client."""
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.schema_config = self._get_schema_config()

    def _get_schema_config(self) -> Dict:
        """Knowledge graph schema configuration."""
        return {
            "node_types": ["concept", "intervention"],
            "concept_categories": [
                "technology",
                "methodology",
                "principle",
                "framework",
            ],
            "intervention_lifecycle_range": (1, 6),
            "intervention_maturity_range": (1, 4),
            "confidence_range": (1, 5),
            "required_node_fields": ["name", "aliases", "type", "description"],
            "required_edge_fields": [
                "type",
                "source_node",
                "target_node",
                "description",
                "edge_confidence",
            ],
        }

    async def judge_knowledge_graph_batch(
        self,
        original_texts: List[str],
        kg_outputs: List[Dict],
        original_prompts: List[str],
    ) -> List[JudgeReport]:
        """
        Process multiple knowledge graphs in batch using OpenAI API.

        Args:
            original_texts: List of source texts
            kg_outputs: List of knowledge graph outputs from LLM
            original_prompts: List of original prompts used for KG extraction

        Returns:
            List of JudgeReports for each input
        """
        if len(original_texts) != len(kg_outputs) != len(original_prompts):
            raise ValueError("All input lists must have the same length")

        tasks = []
        for i, (text, kg_output, prompt) in enumerate(
            zip(original_texts, kg_outputs, original_prompts)
        ):
            task = self._judge_single_graph(text, kg_output, prompt, i)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        reports = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                reports.append(
                    self._create_error_report(
                        f"Error processing item {i}: {str(result)}"
                    )
                )
            else:
                reports.append(result)

        return reports

    async def _judge_single_graph(
        self,
        original_text: str,
        kg_output: Dict,
        original_prompt: str,
        batch_index: int,
    ) -> JudgeReport:
        """Judge a single knowledge graph using OpenAI API."""

        try:
            # Parse the knowledge graph
            knowledge_graph = self._parse_knowledge_graph(kg_output)

            # Generate validation prompt for GPT
            validation_prompt = self._create_validation_prompt(
                original_text, kg_output, original_prompt
            )

            # Get GPT's validation assessment
            gpt_assessment = await self._get_gpt_validation(validation_prompt)

            # Perform local validation checks
            local_validation = self._perform_local_validation(
                knowledge_graph, original_text, original_prompt
            )

            # Combine GPT's assessment with local validation
            combined_report = self._combine_validations(
                gpt_assessment, local_validation, knowledge_graph, original_text
            )

            return combined_report

        except Exception as e:
            return self._create_error_report(
                f"Failed to validate knowledge graph: {str(e)}"
            )

    def _parse_knowledge_graph(self, kg_output: Dict) -> KnowledgeGraph:
        """Parse the knowledge graph output into structured format."""
        nodes = []
        logical_chains = []

        # Parse nodes
        for node_data in kg_output.get("nodes", []):
            node = Node(
                name=node_data.get("name", ""),
                aliases=node_data.get("aliases", []),
                type=node_data.get("type", ""),
                description=node_data.get("description", ""),
                concept_category=node_data.get("concept_category"),
                intervention_lifecycle=node_data.get("intervention_lifecycle"),
                intervention_maturity=node_data.get("intervention_maturity"),
            )
            nodes.append(node)

        # Parse logical chains
        for chain_data in kg_output.get("logical_chains", []):
            edges = []
            for edge_data in chain_data.get("edges", []):
                edge = Edge(
                    type=edge_data.get("type", ""),
                    source_node=edge_data.get("source_node", ""),
                    target_node=edge_data.get("target_node", ""),
                    description=edge_data.get("description", ""),
                    edge_confidence=edge_data.get("edge_confidence", 1),
                )
                edges.append(edge)

            chain = LogicalChain(title=chain_data.get("title", ""), edges=edges)
            logical_chains.append(chain)

        return KnowledgeGraph(nodes=nodes, logical_chains=logical_chains)

    def _create_validation_prompt(
        self, original_text: str, kg_output: Dict, original_prompt: str
    ) -> str:
        """Create a comprehensive validation prompt for GPT."""

        return f"""You are KG-Judge, a precise and rigorous auditor for knowledge graphs. 

Your task is to validate this knowledge graph extraction against the source text and return a structured validation report.

DATA_SOURCE:
{original_text}

ORIGINAL_PROMPT:
{original_prompt}

EXTRACTED_KNOWLEDGE_GRAPH:
{json.dumps(kg_output, indent=2)}

SCHEMA_REQUIREMENTS:
- Nodes must have: name, aliases (2-3 items), type (concept|intervention), description (1-2 sentences)
- If type=concept: must have concept_category
- If type=intervention: must have intervention_lifecycle (1-6) and intervention_maturity (1-4)
- Edges must have: type, source_node, target_node, description, edge_confidence (1-5)
- All node names referenced in edges must exist as nodes
- Logical chains group related edges with meaningful titles

VALIDATION_TASKS:
1. Check JSON structure and schema compliance
2. Verify all referenced nodes exist (referential integrity)
3. Identify orphaned nodes (not connected to any edges)
4. Find duplicate nodes/edges that should be merged
5. Check if extracted knowledge matches source text evidence
6. Assess coverage - are important logical chains from source missing?

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
      "expected_chains_from_source": [
        {{ "title": "chain name", "evidence": "quote from source", "status": "covered|partially_covered|missing", "mapped_chain_ids": ["chain1"] }}
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

    async def _get_gpt_validation(self, validation_prompt: str) -> Dict[str, Any]:
        """Get validation assessment from GPT via OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are KG-Judge, a precise and rigorous auditor for knowledge graphs. Always return valid JSON in the exact format requested.",
                    },
                    {"role": "user", "content": validation_prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )

            # Parse GPT's response
            response_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's not pure JSON
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {
                        "error": "Could not extract JSON from GPT response",
                        "raw_response": response_text,
                    }

        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}

    def _perform_local_validation(
        self, kg: KnowledgeGraph, original_text: str, original_prompt: str
    ) -> Dict[str, Any]:
        """Perform local validation checks as backup/supplement to Claude."""

        issues = []

        # Basic structure checks
        if not kg.nodes:
            issues.append(
                ValidationIssue(
                    "BLOCKER",
                    "No nodes found in knowledge graph",
                    "nodes",
                    "Extract at least one node from source text",
                )
            )

        # Node validation
        node_names = set()
        for i, node in enumerate(kg.nodes):
            # Check required fields
            if not node.name:
                issues.append(
                    ValidationIssue(
                        "BLOCKER",
                        f"Node {i} missing name",
                        f"nodes[{i}].name",
                        "Provide descriptive name",
                    )
                )

            if not node.type or node.type not in ["concept", "intervention"]:
                issues.append(
                    ValidationIssue(
                        "BLOCKER",
                        f"Invalid node type: {node.type}",
                        f"nodes[{i}].type",
                        "Use 'concept' or 'intervention'",
                    )
                )

            # Type-specific validation
            if node.type == "concept" and not node.concept_category:
                issues.append(
                    ValidationIssue(
                        "MAJOR",
                        f"Concept node '{node.name}' missing category",
                        f"nodes[{i}].concept_category",
                        "Specify concept category",
                    )
                )

            if node.type == "intervention":
                if not node.intervention_lifecycle or not (
                    1 <= node.intervention_lifecycle <= 6
                ):
                    issues.append(
                        ValidationIssue(
                            "MAJOR",
                            f"Invalid intervention_lifecycle for '{node.name}'",
                            f"nodes[{i}].intervention_lifecycle",
                            "Set value between 1-6",
                        )
                    )
                if not node.intervention_maturity or not (
                    1 <= node.intervention_maturity <= 4
                ):
                    issues.append(
                        ValidationIssue(
                            "MAJOR",
                            f"Invalid intervention_maturity for '{node.name}'",
                            f"nodes[{i}].intervention_maturity",
                            "Set value between 1-4",
                        )
                    )

            node_names.add(node.name)

        # Edge validation
        for chain_i, chain in enumerate(kg.logical_chains):
            for edge_i, edge in enumerate(chain.edges):
                if edge.source_node not in node_names:
                    issues.append(
                        ValidationIssue(
                            "BLOCKER",
                            f"Edge references non-existent source node: {edge.source_node}",
                            f"logical_chains[{chain_i}].edges[{edge_i}].source_node",
                            f"Create node '{edge.source_node}' or fix reference",
                        )
                    )

                if edge.target_node not in node_names:
                    issues.append(
                        ValidationIssue(
                            "BLOCKER",
                            f"Edge references non-existent target node: {edge.target_node}",
                            f"logical_chains[{chain_i}].edges[{edge_i}].target_node",
                            f"Create node '{edge.target_node}' or fix reference",
                        )
                    )

                if not (1 <= edge.edge_confidence <= 5):
                    issues.append(
                        ValidationIssue(
                            "MINOR",
                            f"Invalid edge confidence: {edge.edge_confidence}",
                            f"logical_chains[{chain_i}].edges[{edge_i}].edge_confidence",
                            "Set value between 1-5",
                        )
                    )

        return {"local_validation": True, "issues": [asdict(issue) for issue in issues]}

    def _combine_validations(
        self,
        gpt_assessment: Dict[str, Any],
        local_validation: Dict[str, Any],
        kg: KnowledgeGraph,
        original_text: str,
    ) -> JudgeReport:
        """Combine GPT's assessment with local validation results."""

        if "error" in gpt_assessment:
            # Fallback to local validation if GPT fails
            return self._create_fallback_report(local_validation, kg, original_text)

        # Generate proposed fixes based on issues found
        proposed_fixes = self._generate_proposed_fixes(
            gpt_assessment.get("validation_report", {}), kg
        )

        # Apply fixes to create final graph
        final_graph = self._apply_fixes_to_graph(kg, proposed_fixes, original_text)

        # Create complete report
        return JudgeReport(
            decision=gpt_assessment.get("decision", {}),
            validation_report=gpt_assessment.get("validation_report", {}),
            proposed_fixes=proposed_fixes,
            final_graph=final_graph,
            rationale_record=gpt_assessment.get("rationale_record", {}),
        )

    def _generate_proposed_fixes(
        self, validation_report: Dict[str, Any], kg: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Generate proposed fixes based on validation issues."""

        fixes = {
            "add_nodes": [],
            "add_edges": [],
            "add_chains": [],
            "edits": [],
            "merges": [],
            "deletions": [],
        }

        # Process schema issues
        for issue in validation_report.get("schema_check", []):
            if "missing" in issue.get("issue", "").lower():
                if "node" in issue.get("where", ""):
                    fixes["add_nodes"].append(
                        {
                            "id": f"generated_node_{len(fixes['add_nodes'])}",
                            "type": "concept",
                            "name": "Generated Node",
                            "props": {
                                "stable_key": hashlib.md5(
                                    f"gen_node_{len(fixes['add_nodes'])}".encode()
                                ).hexdigest()[:8]
                            },
                        }
                    )

        # Process referential issues
        for issue in validation_report.get("referential_check", []):
            if issue.get("severity") == "BLOCKER":
                for node_id in issue.get("ids", []):
                    fixes["add_nodes"].append(
                        {
                            "id": node_id,
                            "type": "concept",
                            "name": node_id,
                            "props": {
                                "stable_key": hashlib.md5(node_id.encode()).hexdigest()[
                                    :8
                                ]
                            },
                        }
                    )

        # Process duplicates
        for duplicate in validation_report.get("duplicates", []):
            if duplicate.get("kind") == "node" and len(duplicate.get("ids", [])) > 1:
                target_id = duplicate["ids"][0]
                absorbed_ids = duplicate["ids"][1:]
                fixes["merges"].append(
                    {
                        "target_id": target_id,
                        "absorbed_ids": absorbed_ids,
                        "retargeted_edge_ids": [],
                    }
                )

        # Process orphans
        for orphan in validation_report.get("orphans", []):
            if "delete" in orphan.get("suggested_fix", "").lower():
                fixes["deletions"].append(
                    {
                        "kind": "node",
                        "id": orphan.get("node_id"),
                        "reason": "orphaned_node",
                    }
                )

        return fixes

    def _apply_fixes_to_graph(
        self, kg: KnowledgeGraph, proposed_fixes: Dict[str, Any], original_text: str
    ) -> Dict[str, Any]:
        """Apply fixes to create the final improved knowledge graph."""

        # Start with original graph
        final_nodes = []
        for node in kg.nodes:
            final_nodes.append(
                {
                    "name": node.name,
                    "aliases": node.aliases,
                    "type": node.type,
                    "description": node.description,
                    "concept_category": node.concept_category,
                    "intervention_lifecycle": node.intervention_lifecycle,
                    "intervention_maturity": node.intervention_maturity,
                }
            )

        final_chains = []
        for chain in kg.logical_chains:
            chain_edges = []
            for edge in chain.edges:
                chain_edges.append(
                    {
                        "type": edge.type,
                        "source_node": edge.source_node,
                        "target_node": edge.target_node,
                        "description": edge.description,
                        "edge_confidence": edge.edge_confidence,
                    }
                )

            final_chains.append({"title": chain.title, "edges": chain_edges})

        # Apply add_nodes fixes
        for add_node in proposed_fixes.get("add_nodes", []):
            final_nodes.append(
                {
                    "name": add_node.get("name", add_node.get("id")),
                    "aliases": [add_node.get("name", add_node.get("id"))],
                    "type": add_node.get("type", "concept"),
                    "description": "Auto-generated node based on validation",
                    "concept_category": "framework"
                    if add_node.get("type") == "concept"
                    else None,
                    "intervention_lifecycle": None,
                    "intervention_maturity": None,
                }
            )

        # Apply deletions
        nodes_to_delete = {
            d.get("id")
            for d in proposed_fixes.get("deletions", [])
            if d.get("kind") == "node"
        }
        final_nodes = [n for n in final_nodes if n.get("name") not in nodes_to_delete]

        return {
            "nodes": final_nodes,
            "logical_chains": final_chains,
            "meta": {
                "version": "1.0",
                "source_hash": hashlib.md5(original_text.encode()).hexdigest(),
                "validation_timestamp": datetime.now().isoformat(),
            },
        }

    def _create_fallback_report(
        self, local_validation: Dict[str, Any], kg: KnowledgeGraph, original_text: str
    ) -> JudgeReport:
        """Create a fallback report when GPT API fails."""

        local_issues = local_validation.get("issues", [])
        has_blockers = any(issue.get("severity") == "BLOCKER" for issue in local_issues)

        return JudgeReport(
            decision={
                "is_valid_json": True,
                "has_blockers": has_blockers,
                "flag_underperformance": len(local_issues) > 5,
                "summary": f"Local validation completed. Found {len(local_issues)} issues, {sum(1 for i in local_issues if i.get('severity') == 'BLOCKER')} blockers.",
            },
            validation_report={
                "schema_check": local_issues,
                "referential_check": [],
                "orphans": [],
                "duplicates": [],
                "rationale_mismatches": [],
                "coverage": {"expected_chains_from_source": []},
            },
            proposed_fixes={
                "add_nodes": [],
                "add_edges": [],
                "add_chains": [],
                "edits": [],
                "merges": [],
                "deletions": [],
            },
            final_graph=self._kg_to_dict(kg, original_text),
            rationale_record={
                "method": "local_fallback",
                "notes": ["Used local validation due to API failure"],
            },
        )

    def _kg_to_dict(self, kg: KnowledgeGraph, original_text: str) -> Dict[str, Any]:
        """Convert KnowledgeGraph to dictionary format."""
        return {
            "nodes": [asdict(node) for node in kg.nodes],
            "logical_chains": [asdict(chain) for chain in kg.logical_chains],
            "meta": {
                "version": "1.0",
                "source_hash": hashlib.md5(original_text.encode()).hexdigest(),
            },
        }

    def _create_error_report(self, error_message: str) -> JudgeReport:
        """Create an error report when validation fails completely."""
        return JudgeReport(
            decision={
                "is_valid_json": False,
                "has_blockers": True,
                "flag_underperformance": True,
                "summary": f"Validation failed: {error_message}",
            },
            validation_report={
                "schema_check": [
                    {
                        "severity": "BLOCKER",
                        "issue": error_message,
                        "where": "system",
                        "suggestion": "Manual review required",
                    }
                ],
                "referential_check": [],
                "orphans": [],
                "duplicates": [],
                "rationale_mismatches": [],
                "coverage": {"expected_chains_from_source": []},
            },
            proposed_fixes={
                "add_nodes": [],
                "add_edges": [],
                "add_chains": [],
                "edits": [],
                "merges": [],
                "deletions": [],
            },
            final_graph={
                "nodes": [],
                "logical_chains": [],
                "meta": {"version": "1.0", "error": error_message},
            },
            rationale_record={
                "method": "error_handling",
                "notes": [f"System error: {error_message}"],
            },
        )

    def export_reports_json(self, reports: List[JudgeReport]) -> str:
        """Export all judge reports as clean JSON."""
        report_dicts = []

        for report in reports:
            report_dict = {
                "decision": report.decision,
                "validation_report": report.validation_report,
                "proposed_fixes": report.proposed_fixes,
                "final_graph": report.final_graph,
                "rationale_record": report.rationale_record,
            }
            report_dicts.append(report_dict)

        return json.dumps(report_dicts, indent=2, ensure_ascii=False)


# Example usage
async def main():
    """Example usage of the KG-Judge system with batch processing."""

    # Sample inputs
    original_texts = [
        """
        Machine learning models require careful feature engineering to achieve optimal performance. 
        The data preprocessing pipeline involves cleaning, normalization, and feature selection steps.
        Cross-validation helps prevent overfitting and ensures model generalizability.
        """,
        """
        Agile methodology emphasizes iterative development and continuous feedback.
        Sprint planning sessions help teams define deliverable increments.
        Retrospectives enable process improvement and team learning.
        """,
    ]

    # TODO: extract from prompt.md
    original_prompts = [
        "Extract a knowledge graph focusing on machine learning concepts and their relationships.",
        "Extract a knowledge graph about agile methodology and its practices.",
    ]

    # Sample knowledge graph outputs
    kg_outputs = [
        {
            "nodes": [
                {
                    "name": "Machine Learning Models",
                    "aliases": ["ML Models", "Predictive Models"],
                    "type": "concept",
                    "description": "Algorithms that learn patterns from data to make predictions.",
                    "concept_category": "technology",
                },
                {
                    "name": "Feature Engineering",
                    "aliases": ["Feature Selection", "Feature Extraction"],
                    "type": "intervention",
                    "description": "Process of selecting and transforming variables for ML models.",
                    "intervention_lifecycle": 3,
                    "intervention_maturity": 4,
                },
            ],
            "logical_chains": [
                {
                    "title": "ML Model Development Process",
                    "edges": [
                        {
                            "type": "requires",
                            "source_node": "Machine Learning Models",
                            "target_node": "Feature Engineering",
                            "description": "ML models need proper feature engineering for optimal performance",
                            "edge_confidence": 5,
                        }
                    ],
                }
            ],
        },
        {
            "nodes": [
                {
                    "name": "Agile Methodology",
                    "aliases": ["Agile", "Agile Development"],
                    "type": "concept",
                    "description": "Iterative approach to software development emphasizing flexibility.",
                    "concept_category": "methodology",
                }
            ],
            "logical_chains": [],
        },
    ]

    # Initialize judge with API key
    judge = KGJudge(api_key=os.getenv("OPENAI_API_KEY"))

    # Run batch validation
    reports = await judge.judge_knowledge_graph_batch(
        original_texts, kg_outputs, original_prompts
    )

    # Export results
    json_reports = judge.export_reports_json(reports)
    print(json_reports)


# Synchronous wrapper for easier usage
def run_kg_judge_batch(
    original_texts: List[str],
    kg_outputs: List[Dict],
    original_prompts: List[str],
    api_key: Optional[str] = None,
) -> List[JudgeReport]:
    """Synchronous wrapper for batch KG validation."""
    judge = KGJudge(api_key=api_key)
    return asyncio.run(
        judge.judge_knowledge_graph_batch(original_texts, kg_outputs, original_prompts)
    )


if __name__ == "__main__":
    asyncio.run(main())
