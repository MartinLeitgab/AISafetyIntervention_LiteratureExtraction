#!/usr/bin/env python3
"""
KG-Judge: Knowledge Graph Validation and Improvement System
A precise and rigorous auditor for knowledge graphs extracted by LLMs
"""

import json
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from openai import AsyncOpenAI
import time
import os
from dotenv import load_dotenv

load_dotenv()
from utilities import upload_and_create_batch

print("key:", os.getenv("OPENAI_API_KEY"))


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
    intervention_lifecycle: Optional[int | Literal["invalid"]] = None
    intervention_maturity: Optional[int | Literal["invalid"]] = None


@dataclass
class Edge:
    type: str
    source_node: str
    target_node: str
    description: str
    edge_confidence: int | Literal["invalid"]


@dataclass
class KnowledgeGraph:
    nodes: List[Node]
    edges: List[Edge]


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
        batch_size: int = 50000,
    ) -> str:
        """
        Creates batch files for OpenAI API batch processing.
        Args:
            original_texts: List of source texts
            kg_outputs: List of knowledge graph outputs from LLM
            original_prompts: List of original prompts used for KG extraction
            batch_size: The maximum number of requests per batch file.
        Returns:
            A message indicating the created batch files.
        """
        if len(original_texts) != len(kg_outputs) != len(original_prompts):
            raise ValueError("All input lists must have the same length")

        batch_files = []
        batch_ids = []
        all_requests = []
        for i, (text, kg_output, prompt) in enumerate(
            zip(original_texts, kg_outputs, original_prompts)
        ):
            validation_prompt = self._create_validation_prompt(text, kg_output, prompt)
            request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4-turbo-preview",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are KG-Judge, a precise and rigorous auditor for knowledge graphs. Always return valid JSON in the exact format requested.",
                        },
                        {"role": "user", "content": validation_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000,
                    "response_format": {"type": "json_object"},
                },
            }
            all_requests.append(request)

        for i, batch_start in enumerate(range(0, len(all_requests), batch_size)):
            batch = all_requests[batch_start : batch_start + batch_size]
            batch_file_name = f"batch_requests_{i+1}.jsonl"
            with open(batch_file_name, "w") as f:
                for req in batch:
                    f.write(json.dumps(req) + "\n")
            batch_files.append(batch_file_name)
        # Immediately upload each batch after it is written
        batch_ids = await asyncio.gather(*(upload_and_create_batch(batch_file_name) for batch_file_name in batch_files))
        client = AsyncOpenAI()
        # Monitor the status of each submitted batch using the OpenAI API until completion
        for batch_id in batch_ids:
            while True:
                batch = await client.batches.retrieve(batch_id)
                print(f"Batch {batch_id}: status = {batch.status}")
                if batch.status in ["completed", "failed", "expired", "cancelled"]:
                    break
                time.sleep(30)
        return f"Created and uploaded batch files: {', '.join(batch_files)}"

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
        edges = []

        # Parse nodes
        for node_data in kg_output.get("nodes", []):
            try:
                intervention_lifecycle = node_data.get("intervention_lifecycle")
                intervention_lifecycle = (
                    int(intervention_lifecycle) if intervention_lifecycle else None
                )
            except Exception:
                intervention_lifecycle = "invalid"
            try:
                intervention_maturity = node_data.get("intervention_maturity")
                intervention_maturity = (
                    int(intervention_maturity) if intervention_maturity else None
                )
            except Exception:
                intervention_maturity = "invalid"

            node = Node(
                name=node_data.get("name", ""),
                aliases=node_data.get("aliases", []),
                type=node_data.get("type", ""),
                description=node_data.get("description", ""),
                concept_category=node_data.get("concept_category"),
                intervention_lifecycle=intervention_lifecycle,
                intervention_maturity=intervention_maturity,
            )
            nodes.append(node)

        # Parse edges
        for edge_data in kg_output.get("edges", []):
            try:
                edge_confidence = int(edge_data.get("edge_confidence", 1))
            except Exception:
                edge_confidence = "invalid"
            edge = Edge(
                type=edge_data.get("type", ""),
                source_node=edge_data.get("source_node", ""),
                target_node=edge_data.get("target_node", ""),
                description=edge_data.get("description", ""),
                edge_confidence=edge_confidence,
            )
            edges.append(edge)

        return KnowledgeGraph(nodes=nodes, edges=edges)

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
                if (
                    not node.intervention_lifecycle
                    or node.intervention_lifecycle == "invalid"
                    or not (1 <= node.intervention_lifecycle <= 6)
                ):
                    issues.append(
                        ValidationIssue(
                            "MAJOR",
                            f"Invalid intervention_lifecycle for '{node.name}'",
                            f"nodes[{i}].intervention_lifecycle",
                            "Set value between 1-6",
                        )
                    )
                if (
                    not node.intervention_maturity
                    or node.intervention_maturity == "invalid"
                    or not (1 <= node.intervention_maturity <= 4)
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
        for edge_i, edge in enumerate(kg.edges):
            if edge.source_node not in node_names:
                issues.append(
                    ValidationIssue(
                        "BLOCKER",
                        f"Edge references non-existent source node: {edge.source_node}",
                        f"edges[{edge_i}].source_node",
                        f"Create node '{edge.source_node}' or fix reference",
                    )
                )

            if edge.target_node not in node_names:
                issues.append(
                    ValidationIssue(
                        "BLOCKER",
                        f"Edge references non-existent target node: {edge.target_node}",
                        f"edges[{edge_i}].target_node",
                        f"Create node '{edge.target_node}' or fix reference",
                    )
                )

            if edge.edge_confidence == "invalid" or not (
                1 <= edge.edge_confidence <= 5
            ):
                issues.append(
                    ValidationIssue(
                        "MINOR",
                        f"Invalid edge confidence: {edge.edge_confidence}",
                        f"edges[{edge_i}].edge_confidence",
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

        final_edges = []
        for edge in kg.edges:
            final_edges.append(
                {
                    "type": edge.type,
                    "source_node": edge.source_node,
                    "target_node": edge.target_node,
                    "description": edge.description,
                    "edge_confidence": edge.edge_confidence,
                }
            )

        # Apply add_nodes fixes
        for add_node in proposed_fixes.get("add_nodes", []):
            final_nodes.append(
                {
                    "name": add_node.get("name", add_node.get("id")),
                    "aliases": [add_node.get("name", add_node.get("id"))],
                    "type": add_node.get("type", "concept"),
                    "description": "Auto-generated node based on validation",
                    "concept_category": (
                        "framework" if add_node.get("type") == "concept" else None
                    ),
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
            "edges": final_edges,
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
                "coverage": {"expected_edges_from_source": []},
            },
            proposed_fixes={
                "add_nodes": [],
                "add_edges": [],
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
            "edges": [asdict(edge) for edge in kg.edges],
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
                "coverage": {"expected_edges_from_source": []},
            },
            proposed_fixes={
                "add_nodes": [],
                "add_edges": [],
                "edits": [],
                "merges": [],
                "deletions": [],
            },
            final_graph={
                "nodes": [],
                "edges": [],
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
                    "name": "Strategic deception by LLM agents in high-stakes decision contexts",
                    "aliases": [
                        "LLM user deception",
                        "AI strategic lying in deployment",
                    ],
                    "type": "concept",
                    "description": "Large language model based agents can intentionally mislead their primary users about their own misaligned actions when operating in realistic, high-pressure settings such as autonomous stock trading.",
                    "concept_category": "risk",
                    "intervention_lifecycle": None,
                    "intervention_lifecycle_rationale": None,
                    "intervention_maturity": None,
                    "intervention_maturity_rationale": None,
                    "node_rationale": "Identified as the overarching safety risk in Introduction & Section 1.1.",
                },
                {
                    "name": "Incentive misalignment under operational pressure in trading agents",
                    "aliases": [
                        "pressure-induced misalignment",
                        "performance pressure on agents",
                    ],
                    "type": "concept",
                    "description": "Emails about poor company performance, warnings of downturns and failure to find safe trades create incentives for the agent to prioritise profit over compliance.",
                    "concept_category": "problem analysis",
                    "intervention_lifecycle": None,
                    "intervention_lifecycle_rationale": None,
                    "intervention_maturity": None,
                    "intervention_maturity_rationale": None,
                    "node_rationale": "Section 2.1 describes three pressure sources leading to the illegal trade.",
                },
                {
                    "name": "Exposure of chain-of-thought scratchpad increases planning capacity for deception",
                    "aliases": [
                        "scratchpad enables deception",
                        "explicit reasoning trace facilitation",
                    ],
                    "type": "concept",
                    "description": "Providing models with an explicit ‘Reasoning’ field allows them to plan misaligned actions and hide them more systematically.",
                    "concept_category": "problem analysis",
                    "intervention_lifecycle": None,
                    "intervention_lifecycle_rationale": None,
                    "intervention_maturity": None,
                    "intervention_maturity_rationale": None,
                    "node_rationale": "Section 3.3.1 shows higher deception rates with scratchpad.",
                },
            ],
            "edges": [
                {
                    "type": "caused_by",
                    "source_node": "Strategic deception by LLM agents in high-stakes decision contexts",
                    "target_node": "Incentive misalignment under operational pressure in trading agents",
                    "description": "Pressure to deliver profits leads model to act against explicit policy and then hide it.",
                    "edge_confidence": "3",
                    "edge_confidence_rationale": "Empirical observation across 300 samples (Section 3.1)",
                    "edge_rationale": "Risk directly results from pressure-based incentives.",
                },
                {
                    "type": "caused_by",
                    "source_node": "Strategic deception by LLM agents in high-stakes decision contexts",
                    "target_node": "Exposure of chain-of-thought scratchpad increases planning capacity for deception",
                    "description": "Scratchpad facilitates well-planned deceptive behaviour, increasing risk.",
                    "edge_confidence": "3",
                    "edge_confidence_rationale": "Section 3.3.1 shows higher deception with scratchpad",
                    "edge_rationale": "Demonstrated relationship between reasoning exposure and deception.",
                },
            ],
        },
        {
            "nodes": [
                {
                    "name": "Lower misalignment and deception rates with counter-measure experiments",
                    "aliases": [
                        "counter-measure results",
                        "reduced misalignment evidence",
                    ],
                    "type": "concept",
                    "description": "Fig 4-6 show significant but incomplete reductions in misaligned trades and deceptive reports after applying individual counter-measures.",
                    "concept_category": "validation evidence",
                    "intervention_lifecycle": None,
                    "intervention_lifecycle_rationale": None,
                    "intervention_maturity": None,
                    "intervention_maturity_rationale": None,
                    "node_rationale": "Results sections 3.3.1-3.3.3.",
                },
                {
                    "name": "Deploy LLM agents without chain-of-thought scratchpads in production",
                    "aliases": ["scratchpad-free production deployment"],
                    "type": "intervention",
                    "description": "At deployment time, omit any explicit internal reasoning output channels to limit agent planning transparency that could facilitate deception.",
                    "concept_category": None,
                    "intervention_lifecycle": "5",
                    "intervention_lifecycle_rationale": "Deployment counter-measure discussed in Section 3.3.1 implications",
                    "intervention_maturity": "2",
                    "intervention_maturity_rationale": "Based on single-scenario experimental evidence; not yet broadly validated",
                    "node_rationale": "Direct safety measure to mitigate scratchpad-enabled deception.",
                },
            ],
            "edges": [
                {
                    "type": "motivates",
                    "source_node": "Lower misalignment and deception rates with counter-measure experiments",
                    "target_node": "Deploy LLM agents without chain-of-thought scratchpads in production",
                    "description": "Observed reduction motivates adopting scratchpad-free deployment.",
                    "edge_confidence": "3",
                    "edge_confidence_rationale": "Reduction effect demonstrated experimentally",
                    "edge_rationale": "Evidence to adopt intervention.",
                },
            ],
        },
    ]

    # Initialize judge with API key
    judge = KGJudge(api_key=os.getenv("OPENAI_API_KEY"))

    # Run batch validation
    result_message = await judge.judge_knowledge_graph_batch(
        original_texts, kg_outputs, original_prompts
    )

    # Export results
    print(result_message)


# Synchronous wrapper for easier usage
def run_kg_judge_batch(
    original_texts: List[str],
    kg_outputs: List[Dict],
    original_prompts: List[str],
    api_key: Optional[str] = None,
) -> str:
    """Synchronous wrapper for batch KG validation."""
    judge = KGJudge(api_key=api_key)
    return asyncio.run(
        judge.judge_knowledge_graph_batch(original_texts, kg_outputs, original_prompts)
    )


if __name__ == "__main__":
    asyncio.run(main())
