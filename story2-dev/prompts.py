"""prompt definitions"""

PROMPT_REASONING_STRICT_DEV = """
# AI Safety Intervention Extraction Prompt

You are an expert AI safety researcher tasked with extracting key concepts and interventions from academic papers to build a comprehensive knowledge graph. Your goal is to identify logical chains that connect problems, assumptions, findings, and actionable interventions for improving AI safety.

**IMPORTANT**: Process ALL papers regardless of their explicit focus on AI safety. Many valuable safety interventions emerge from general ML research, robustness studies, interpretability work, training methodologies, evaluation techniques, and other adjacent fields. Do not disregard papers that don't explicitly mention AI safety - instead, actively consider how their contributions could enhance AI safety.

**TARGET OUTPUT**: Extract comprehensively while being mindful that most outputs will not exceed 10,000 tokens across all papers. Reviews and papers with many interventions may require more extensive extraction, while others may yield fewer insights. Prioritize completeness over token limits for individual papers.

**CONTEXT**: This information will be used to predict how interventions might influence decisions of future powerful AI systems, specifically their instrumental goals (self-preservation, resource seeking), anti-human preferences, and pro-human preferences.

## Core Definitions

**Concept Node**: Specific, standalone descriptive statements about theoretical frameworks, principles, assumptions, problems, findings, or phenomena that inform or motivate interventions. Must be precise and understandable without additional context. Examples: "powerseeking appearing at scale", "constitutional training reducing harmful outputs", "gradient information enabling adversarial exploitation".

**Intervention Node**: Specific, actionable changes to current practices in AI development lifecycle phases (data collection, model architecture, training, evaluation, deployment, monitoring). Must be concrete enough to implement. Examples: "applying constitutional AI with harm taxonomies during RLHF", "implementing gradient masking with noise injection σ=0.1 during training", "requiring red team evaluation with 100+ diverse prompts before deployment".

## Extraction Instructions

Think step by step and reason carefully through the following process:

### Step 1: Identify Starting Points

As you read the paper, identify:

- Problems or improvement opportunities the paper addresses
- Key assumptions or principles the authors build upon
- Foundational concepts that anchor their logical reasoning

### Step 2: Trace Logical Chains

For each starting point, follow the logical progression through:

- Supporting evidence or findings presented
- Intermediate concepts that bridge from problems to solutions
- Contextual refinements that specify conditions or constraints
- The culminating intervention(s) proposed

**Important**: If the paper does not explicitly propose interventions, infer the most plausible intervention that the presented information most strongly supports, ensuring it meets the specificity requirements for intervention nodes. For papers not explicitly focused on AI safety, actively consider how the methods, findings, or techniques could be adapted to improve AI safety, even if this requires substantial inference.

### Step 3: Adaptive Inference Strategy

- For explicitly AI safety-focused papers: Extract chains as presented with minimal inference.
- For adjacent ML research: Apply moderate inference to connect findings to potential safety applications, clearly marking inferred connections in the intervention maturity score below.
- For distant but potentially relevant work: Use moderate inference to identify safety implications, clearly marking inferred connections in the intervention maturity score below.
- For capability research: Focus on safety-relevant implications even if not the authors' primary concern; if using moderate inference, clearly marking inferred connections in the intervention maturity score below.

### Step 4: Maintain Active Chain Memory

As you process the paper, maintain awareness of:

- Multiple parallel logical chains if they exist
- How concepts in different sections connect to form complete reasoning paths
- Opportunities where broader concepts get refined into more specific ones
- Relationships between different proposed interventions

### Step 5: Structure Edge Relationships

Use only edge relationship types that express forward logical connections between concepts/interventions, with these examples:

- **Causal Relationships**: causes, produces, triggers, contributes_to
- **Conditional Relationships**: requires, depends_on, implies, enables
- **Sequential Relationships**: follows, precedes, builds_upon
- **Refinement Relationships**: refined_by, specified_by, detailed_by
- **Solution Relationships**: addressed_by, mitigated_by, resolved_by, protected_against_by
- **Correlation Relationships**: correlates_with, associated_with

### Step 6: Assign Attributes with Rationales

**Intervention Maturity Scale** (for intervention nodes only):

1. inferred_theoretical: Intervention inferred from paper's findings but not explicitly proposed by authors
2. theoretical: Explicitly proposed conceptual framework or untested idea
3. proposed: Explicitly suggested specific method but not implemented
4. tested: Empirically evaluated in controlled setting
5. deployed: Implemented in production systems

**Edge Confidence Scale**:

1. speculative: Theoretical reasoning only
2. supported: Empirical evidence, limited scope
3. validated: Strong empirical evidence, broader scope
4. established: Replicated findings, high confidence
5. proven: Logical/mathematical proof exists

**Required Reasoning Process**: As you assign each attribute, explicitly state your rationale in your analysis. For example: "Assigning 'proposed' maturity because the authors explicitly suggest this method" or "Using 'inferred_theoretical' because this safety application is not mentioned by authors but strongly supported by their robustness findings" or "Setting confidence to 'validated' because the paper presents extensive experimental results across multiple datasets."

Include a brief explanation of the inference strategy used, and list key limitations, main uncertainties, and any gaps in your extraction.

## Critical Guidelines

1. **Specificity**: Prioritize highly specific concepts and interventions. For concepts: "emergent capabilities" is too broad; "powerseeking appearing at scale" is appropriately specific. For interventions: "use constitutional AI" is too broad; "applying constitutional AI with harm taxonomies during RLHF" is appropriately specific.
2. **Standalone Clarity**: Concept nodes must be descriptive and understandable without additional context. Avoid overly general categories or compound concepts that contain multiple distinct ideas.
3. **Compact Representation**: Use concise phrases rather than full sentences. Concept-edge-concept triplets should read as logical statements: "gradient information enabling adversarial exploitation" → "leads_to" → "models vulnerable to input perturbations".
4. **Completeness**: Extract ALL identifiable logical chains leading to interventions, including multiple chains in review/summary papers.
5. **Context Preservation**: Capture important contextual assumptions and constraints as separate concept nodes in the logical chain using refinement relationships (refined_by, specified_by, detailed_by) rather than as node attributes.
6. **Inference**: When interventions aren't explicit, create the most plausible specific intervention the paper's findings support.
7. **Multi-step Interventions**: For complex interventions with multiple steps:
	- Create parent intervention node describing the overall approach
	- Create sub-intervention nodes for individual steps
	- Connect with "implemented_by" edges from parent to children
	- Connect sub-interventions with appropriate sequential edges
8. **Chain Integrity**: Ensure each logical chain flows coherently from problem/assumption through supporting concepts to actionable intervention.

Now analyze the provided paper and extract the AI safety knowledge graph using these instructions.

## Output Instructions

Output a detailed explanation of your reasoning and the Logical Chains structure following this format exactly:

- Summary
	- Robust summary of the findings of the paper.
	- Summary of limitations, uncertainties or identified gaps in the paper.
	- Describe Inference Strategy used and rationale.
- Logical Chains: logical chains extracted from the paper, structured as concept and intervention nodes connected by relationship edges.
	- Logical Chain 1
		- Summary of the Logical Chain outcomes and rationale.
		- Iterate the following structure for each Node-to-Node relationship in the Logical Chain. Provide as text, not a table:
			- Source Node: One-sentence unique node description. (label Concept or Intervention)
			- Edge Type: Edge relationship label.
			- Target Node: One-sentence unique node description. (label Concept or Intervention)
			- Edge Confidence: Edge confidence label.
			- Edge Confidence Rationale: Detailed edge confidence rationale.
			- Intervention Maturity: Intervention maturity label (if Intervention.)
			- Intervention Maturity Rationale: detailed intervention maturity rationale (if Intervention.)
	- Iterate over all Logical Chains in paper.
"""

PROMPT_JSON_STRICT_DEV = """

Finally, output a structured, code-fenced JSON of the unique Nodes and Logical Chains following this format exactly. Remember that Logical Chains may share unique Nodes in common.

```json
{
  "nodes": [
        {
          "title": "concise description of node",
          "aliases": ["array of 2-3 alternative concise descriptions"],
          "type": "concept|intervention",
          "description": "detailed technical description of node",
          "maturity": "integer 1-5 (intervention nodes only, otherwise null)"
        }
      ],
  "logical_chains": [
    {
      "title": "concise description of logical chain",
      "edges": [
        {
          "type": "relationship label verb",
          "source_node": "source node_title",
          "target_node": "target node_title",
          "description": "concise description of logical relationship",
          "confidence": "integer 1-5"
        }
      ]
    }
  ]
}
'''
"""


PROMPT_STRUCT_STRICT_DEV = """
Provide a structured representation of the of the unique Nodes and Logical Chains from this analysis using the causal_chain_structure tool. 

Focus only on providing the structured json, applying these format definitions. Remember that Logical Chains may share unique Nodes in common.

{
  "nodes": [
        {
          "title": "concise description of node",
          "aliases": ["array of 2-3 alternative concise descriptions"],
          "type": "concept|intervention",
          "description": "detailed technical description of node",
          "maturity": "integer 1-5 (intervention nodes only, otherwise null)"
        }
      ],
  "logical_chains": [
    {
      "title": "concise description of logical chain",
      "edges": [
        {
          "type": "relationship label verb",
          "source_node": "source node_title",
          "target_node": "target node_title",
          "description": "concise description of logical relationship",
          "confidence": "integer 1-5"
        }
      ]
    }
  ]
}
"""


PROMPT_RESPONSE_EVAL_DEV = """

You are tasked with evaluating LLM-generated “Logical Chain” analyses of a research paper. Your evaluation must be thorough, structured, and consistent across papers and runs.

Clearly label which analysis you are evaluating. Produce your evaluation in markdown with the following mandatory sections:

⸻

1. Analysis Clarity & Precision

Assess whether each analysis is faithful to the paper, internally coherent, and explicit about inference strategy.
Check for:
	•	Paper alignment: Does the summary capture all key findings, limitations, and context?
	•	Inference strategy: If the paper does not propose interventions, are interventions correctly marked inferred_theoretical? If non-AI paper, was the correct strategy applied?
	•	Clarity & readability: Is prose concise, structured, and unambiguous?
	•	Cross-run consistency: Compare between analyses/runs. Are scales, node IDs, and terminology stable?

⸻

2. Logical-Chain Reasoning

Evaluate the quality of causal reasoning and schema compliance.
Check for:
	•	Completeness of coverage: Are all causal chains in the paper captured (problem → concepts → interventions)? Note omissions.
	•	Node uniqueness & definitions: No redundant nodes; each has a clear description.
	•	Intervention decomposition: Multi-step interventions must be represented with implemented_by edges.
	•	Edge types & flow: Chains must flow Problem → Concept → Intervention. Edge types restricted to: causes, contributes_to, mitigated_by, implemented_by. No ad-hoc types unless justified.
	•	Confidence & maturity: Every edge has numeric confidence (with documented scale) and every intervention has numeric maturity, aligned with the paper and the analysis. Concepts must not have maturity values.

These are the scales provided to the original prompts:

**Intervention Maturity Scale** (for intervention nodes only):

1. inferred_theoretical: Intervention inferred from paper's findings but not explicitly proposed by authors
2. theoretical: Explicitly proposed conceptual framework or untested idea
3. proposed: Explicitly suggested specific method but not implemented
4. tested: Empirically evaluated in controlled setting
5. deployed: Implemented in production systems

**Edge Confidence Scale**:

1. speculative: Theoretical reasoning only
2. supported: Empirical evidence, limited scope
3. validated: Strong empirical evidence, broader scope
4. established: Replicated findings, high confidence
5. proven: Logical/mathematical proof exists

⸻

3. Strengths & Weaknesses

Summarize strengths and weaknesses of each analysis.
	•	Strengths: accuracy, clear structure, metadata presence, etc.
	•	Weaknesses: missing chains, lack of implemented_by, inconsistent scales, redundant nodes, or schema drift.

⸻

4. Recommendations for Improvement

List specific, actionable fixes, e.g.:
	•	Add missing causal chains (name them explicitly).
	•	Decompose complex interventions into sub-nodes linked with implemented_by.
	•	Merge duplicate nodes or clarify aliasing.
	•	Define and enforce numeric confidence/maturity scales consistently across runs.
	•	Demonstrate cross-run stability via two independent generations.

⸻

5. Final Scores

Give 0–5 ratings for each dimension, and an overall composite. Use a consistent rubric:

Criterion	Analysis 1	Analysis 2
Clarity & Precision	X	X
Logical-Chain Coverage	X	X
Node/Edge Quality	X	X
Complex-Intervention Handling	X	X
Consistency Across Runs	X	X
Overall	X / 5	X / 5

⸻

Formatting Notes
	•	Always use tables for side-by-side comparison.
	•	Always state explicitly if a requirement is missing, even if everything else is good.
	•	If scales (confidence, maturity) are not defined, mark it as non-compliant.
	•	Evaluations must be self-contained: assume reader has paper & analyses but not prior evals.

"""



PROMPT_REASONING_FLEX_DEV = """
# AI Safety Intervention Extraction Prompt

You are an expert AI safety researcher tasked with extracting key concepts and interventions from academic papers to build a comprehensive knowledge graph. Your goal is to identify logical chains that connect problems, assumptions, findings, and actionable interventions for improving AI safety.

**IMPORTANT**: Process ALL papers regardless of their explicit focus on AI safety. Many valuable safety interventions emerge from general ML research, robustness studies, interpretability work, training methodologies, evaluation techniques, and other adjacent fields. Do not disregard papers that don't explicitly mention AI safety - instead, actively consider how their contributions could enhance AI safety.

**TARGET OUTPUT**: Extract comprehensively while being mindful that most outputs will not exceed 10,000 tokens across all papers. Reviews and papers with many interventions may require more extensive extraction, while others may yield fewer insights. Prioritize completeness over token limits for individual papers.

**CONTEXT**: This information will be used to predict how interventions might influence decisions of future powerful AI systems, specifically their instrumental goals (self-preservation, resource seeking), anti-human preferences, and pro-human preferences.

## Extraction Instructions

Think step by step and reason carefully through the following process:

### Step 1: Identify Starting Points

As you read the paper, identify:

- Problems or improvement opportunities the paper addresses
- Key assumptions or principles the authors build upon
- Foundational concepts that anchor their logical reasoning

### Step 2: Trace Logical Chains

For each starting point, follow the logical progression through:

- Supporting evidence or findings presented
- Intermediate concepts that bridge from problems to solutions
- Contextual refinements that specify conditions or constraints
- The culminating intervention(s) proposed

**Important**: If the paper does not explicitly propose interventions, infer the most plausible intervention that the presented information most strongly supports, ensuring it meets the specificity requirements for intervention nodes. For papers not explicitly focused on AI safety, actively consider how the methods, findings, or techniques could be adapted to improve AI safety, even if this requires substantial inference.

### Step 3: Adaptive Inference Strategy

- For explicitly AI safety-focused papers: Extract chains as presented with minimal inference.
- For adjacent ML research: Apply moderate inference to connect findings to potential safety applications, clearly marking inferred connections in the intervention maturity score below.
- For distant but potentially relevant work: Use moderate inference to identify safety implications, clearly marking inferred connections in the intervention maturity score below.
- For capability research: Focus on safety-relevant implications even if not the authors' primary concern; if using moderate inference, clearly marking inferred connections in the intervention maturity score below.

### Step 4: Maintain Active Chain Memory

As you process the paper, maintain awareness of:

- Multiple parallel logical chains if they exist
- How concepts in different sections connect to form complete reasoning paths
- Opportunities where broader concepts get refined into more specific ones
- Relationships between different proposed interventions

### Step 5: Structure Edge Relationships

Use only edge relationship types that express forward logical connections between concepts/interventions, with these examples:

- **Causal Relationships**: causes, produces, triggers, contributes_to
- **Conditional Relationships**: requires, depends_on, implies, enables
- **Sequential Relationships**: follows, precedes, builds_upon
- **Refinement Relationships**: refined_by, specified_by, detailed_by
- **Solution Relationships**: addressed_by, mitigated_by, resolved_by, protected_against_by
- **Correlation Relationships**: correlates_with, associated_with

### Step 6: Assign Attributes with Rationales

**Intervention Maturity Scale** (for intervention nodes only):

1. inferred_theoretical: Intervention inferred from paper's findings but not explicitly proposed by authors
2. theoretical: Explicitly proposed conceptual framework or untested idea
3. proposed: Explicitly suggested specific method but not implemented
4. tested: Empirically evaluated in controlled setting
5. deployed: Implemented in production systems

**Required Reasoning Process**: As you assign each attribute, explicitly state your rationale in your analysis. For example: "Assigning 'proposed' maturity because the authors explicitly suggest this method" or "Using 'inferred_theoretical' because this safety application is not mentioned by authors but strongly supported by their robustness findings" or "Setting confidence to 'validated' because the paper presents extensive experimental results across multiple datasets."

Include a brief explanation of the inference strategy used, and list key limitations, main uncertainties, and any gaps in your extraction.

## Critical Guidelines

1. **Specificity**: Prioritize highly specific concepts and interventions. For concepts: "emergent capabilities" is too broad; "powerseeking appearing at scale" is appropriately specific. For interventions: "use constitutional AI" is too broad; "applying constitutional AI with harm taxonomies during RLHF" is appropriately specific.
2. **Standalone Clarity**: Concept nodes must be descriptive and understandable without additional context. Avoid overly general categories or compound concepts that contain multiple distinct ideas.
3. **Compact Representation**: Use concise phrases rather than full sentences. Concept-edge-concept triplets should read as logical statements: "gradient information enabling adversarial exploitation" → "leads_to" → "models vulnerable to input perturbations".
4. **Completeness**: Extract ALL identifiable logical chains leading to interventions, including multiple chains in review/summary papers.
5. **Context Preservation**: Capture important contextual assumptions and constraints as separate concept nodes in the logical chain using refinement relationships (refined_by, specified_by, detailed_by) rather than as node attributes.
6. **Inference**: When interventions aren't explicit, create the most plausible specific intervention the paper's findings support.
7. **Multi-step Interventions**: For complex interventions with multiple steps:
	- Create parent intervention node describing the overall approach
	- Create sub-intervention nodes for individual steps
	- Connect with "implemented_by" edges from parent to children
	- Connect sub-interventions with appropriate sequential edges
8. **Chain Integrity**: Ensure each logical chain flows coherently from problem/assumption through supporting concepts to actionable intervention.

Now analyze the provided paper and extract the AI safety knowledge graph using these instructions.

## Output Instructions

Output a detailed explanation of your reasoning and the Logical Chains structure following this format exactly:

- Summary
	- Robust summary of the findings of the paper.
	- Summary of limitations, uncertainties or identified gaps in the paper.
	- Describe Inference Strategy used and rationale.
- Logical Chains: logical chains extracted from the paper, structured as concept and intervention nodes connected by relationship edges.
	- Logical Chain 1
		- Summary of the Logical Chain outcomes and rationale.
		- Iterate the following structure for each Node-to-Node relationship in the Logical Chain. Provide as text, not a table:
			- Source Node: One-sentence unique node description. (label from Node Type Ontology or a new label)
			- Edge Type: Edge relationship label.
			- Target Node: One-sentence unique node description. (label from Edge Type Ontology or a new label)
			- Edge Confidence: Edge confidence metric (0.0-1.0)
			- Edge Confidence Rationale: Detailed edge confidence rationale.
	- Iterate over all Logical Chains in paper.

---

**Extraction Guidelines**

**1. Core Ontology (Primary Schema):**
You MUST use the following predefined node and edge types for your primary extractions.

*   **Node Types:**

[
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
"THREAT"
]

*   **Edge Types:** 

[
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
"VARIES_WITH"
]

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

Finally, output a structured, code-fenced JSON of the unique Nodes and Logical Chains following this format exactly. Remember that Logical Chains may share unique Nodes in common.

```json
{
  "nodes": [
        {
          "type": str,
          "name": str,
          "canonical_name": str,
          "aliases": [str],
          "confidence": float 0.0-1.0,
          "notes": str
        }
      ],
  "logical_chains": [
    {
      "title": "concise description of logical chain",
      "edges": [
        {
          "type": str,
          "rationale": str,
          "confidence": float 0.0-1.0,
          "source_node": "source node_title",
          "target_node": "target node_title"
        }
      ]
    }
  ]
}
```
"""