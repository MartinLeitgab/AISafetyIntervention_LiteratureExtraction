from typing import List
from pydantic import BaseModel, Field

# Sorted alphabetically
# TODO: make this dynamic and nested, use only a subset based on the paper type
NODE_TYPES = [
    "ASSUMPTION",
    "BEHAVIOR",
    "BENCHMARK",
    "CLAIM",
    "CONCEPT",
    "DATASET",
    "METRIC",
    "METHOD",
    "MITIGATION",
    "MODEL",
    "PROMPT_TECHNIQUE",
    "PROTOCOL",
    "RESULT",
    "RISK_TYPE",
    "TASK",
    "THREAT",
]

EDGE_TYPES = [
    "ASSUMES",
    "CAUSES",
    "CHANGES",
    "CORRELATES_WITH",
    "DERIVES_FROM",
    "ENABLES",
    "EVALUATES_ON",
    "EVIDENCES",
    "EXPLAINS",
    "IDENTIFIES",
    "IMPROVES_OVER",
    "MITIGATES",
    "PREVENTS",
    "PROPOSES",
    "REPORTS",
    "VARIES_WITH",
]


# https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
class Node(BaseModel):
    type: str
    name: str
    canonical_name: str
    aliases: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str


class Edge(BaseModel):
    type: str
    rationale: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    target_node: Node


class SuggestedType(BaseModel):
    type: str
    definition: str
    example_instance: str


class OutputSchema(BaseModel):
    new_node_types: List[SuggestedType]
    new_edge_types: List[SuggestedType]
    edges: List[Edge]


EXTRACTION_PROMPT_TEMPLATE = f"""
You are an expert AI safety researcher tasked with extracting structured knowledge from academic papers to build a comprehensive knowledge graph. Your goal is to identify key concepts (nodes) and their relationships (edges) that will help researchers understand the landscape of AI safety research.

You will be provided with the text of a research paper. Using this paper as a source node, extract edges and nodes from this paper, adhering strictly to the following guidelines and the required JSON output schema.

---
**Extraction Guidelines**

**1. Core Ontology (Primary Schema):**
You MUST use the following predefined node and edge types for your primary extractions.
*   **Node Types:** {", ".join(NODE_TYPES)}
*   **Edge Types:** {", ".join(EDGE_TYPES)}

**2. Creative & Causal Extraction (Suggesting New Types):**
While adhering to the core ontology, you should also identify important causal or domain-specific relationships and node types not listed above (e.g., `CAUSES`, `PREVENTS`, `RISK_TYPE`). If you find a recurring, important type that is not in the core list, you MUST add it to the `new_node_types` or `new_edge_types` field in your output, then you can use them in your edge types.
For example, PROPOSES_MITIGATION is an incorrect edge type, and should be split into PROPOSES edge and MITIGATION node types.

**3. Strict Requirements (Mandatory for all extractions):**
*   **Conservative Policy:** Extract only what the paper explicitly states or strongly implies. Do not invent connections or attributes.
*   **Confidence:** Every extraction **must** have a `confidence` score (0.0-1.0). Use lower confidence (<0.6) for extractions that rely on inference rather than explicit text.
*   **Normalization:** Canonicalize method and dataset names (e.g., "RLHF" -> "Reinforcement Learning from Human Feedback"). The `canonical_name` should be the full name. List original text in `aliases`. If uncertain, copy the surface form into `canonical_name` and explain your reasoning in the `notes` field. This field will be used to merge graphs together, so it is important to be consistent.
*   **Node Naming:** Create concise but descriptive node `name`s (use snake_case for multi-word concepts, e.g., "strategic_deception").
*   **Exclusions:** Do NOT extract authors, organizations, or bibliographic citations. The source of all extractions and edges is implicitly the current paper.

---

**Your Task:**
Based on the paper content provided to you, generate a single JSON object with the final knowledge graph. The JSON object must conform to the schema provided.
"""
