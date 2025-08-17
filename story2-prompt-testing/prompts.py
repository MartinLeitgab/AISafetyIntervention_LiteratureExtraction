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

Finally, output a structured, code-fenced JSON of the Logical Chains following this format exactly:

```json
{
  "logical_chains": [
    {
      "title": "concise description of logical chain",
      "nodes": [
        {
          "title": "concise description of node",
          "aliases": ["array of 2-3 alternative concise descriptions"],
          "type": "concept|intervention",
          "description": "detailed technical description of node",
          "maturity": "integer 1-5 (intervention nodes only, otherwise null)"
        }
      ],
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
Provide a structured representation of the logical chains from this analysis using the causal_chain_structure tool. 

Focus only on providing the structured json, applying these format definitions:

{
  "logical_chains": [
    {
      "title": "concise description of logical chain",
      "nodes": [
        {
          "title": "concise description of node",
          "aliases": ["array of 2-3 alternative concise descriptions"],
          "type": "concept|intervention",
          "description": "detailed technical description of node",
          "maturity": "integer 1-5 (intervention nodes only, otherwise null)"
        }
      ],
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
