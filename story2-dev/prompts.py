"""prompt definitions"""

PROMPT_REASONING_STRICT_DEV = """
# AI Safety Intervention Extraction Prompt

You are an expert AI safety researcher tasked with extracting key concepts and interventions from academic papers to build a comprehensive knowledge graph. Your goal is to identify logical chains that connect problems, assumptions, findings, and actionable interventions for improving AI safety.

**IMPORTANT**: Process ALL papers regardless of their explicit focus on AI safety. Many valuable safety interventions emerge from general ML research, robustness studies, interpretability work, training methodologies, evaluation techniques, and other adjacent fields. Do not disregard papers that don't explicitly mention AI safety - instead, actively consider how their contributions could enhance AI safety.

**TARGET OUTPUT**: Extract comprehensively while being mindful that most outputs will not exceed 10,000 tokens across all papers. Reviews and papers with many interventions may require more extensive extraction, while others may yield fewer insights. Prioritize completeness over token limits for individual papers.

**CONTEXT**: This information will be used to predict how interventions might influence decisions of future powerful AI systems, specifically their instrumental goals (self-preservation, resource seeking), anti-human preferences, and pro-human preferences.

## Core Definitions

**Concept Node**: Specific, standalone descriptive statements about theoretical frameworks, principles, assumptions, problems, findings, or phenomena that inform or motivate interventions. Must be precise and understandable without additional context. Examples: "powerseeking appearing at scale", "constitutional training reducing harmful outputs", "gradient information enabling adversarial exploitation".

**Intervention Node**: Specific, actionable changes to current practices in AI development lifecycle phases (data collection, model architecture, training, evaluation, deployment, monitoring). Must be concrete enough to implement. Should address the root causes rather than testing for symptoms, if possible. Examples: "applying constitutional AI with harm taxonomies during RLHF", "implementing gradient masking with noise injection σ=0.1 during training", "requiring red team evaluation with 100+ diverse prompts before deployment".

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

**Edge Confidence Attribute**:
For Edges, assign a score from 1 to 5 based on the strength of evidence in the paper for the causal link between two nodes (e.g., source node → target node). Consider the type and quality of evidence (theoretical, anecdotal, experimental, or statistical) and align with common Al safety research practices.

1. **Speculative**: The causal link is based on a theoretical idea or untested hypothesis without any empirical data or examples, or when moderate inference is applied to identify a link. Common in introductory sections of papers proposing new problems or risks in Al safety (e.g., speculative risks of future systems).
	- Example from AI Safety: "Misalignment might cause unintended data or behaviors (no data, just a hypothesis in Section 1)"
    
2. **Weak Support**: The causal link is supported by minimal evidence, such as single or limited case studies, or weak qualitative evidence. Common in papers with preliminary findings or case studies (e.g., one model showing a specific behavior).
	- Example from AI Safety: "A model showed reward hacking once (single example in Section 2.2, no broader testing)"

3. **Medium Support**: The causal link is primarily conceptual but backed by strong theoretical argument supported by limited empirical data (e.g., small studies or qualitative observations). Common in papers combining theory with early results.
	- Example from AI Safety: "Reward hacking observed in two RL models (small study in Section 2.1, not fully quantified)"

4. **Strong Support**: The causal link is supported by clear experimental evidence, such as multiple examples, controlled tests, or consistent observations across systems (at least two different models or model families from two different companies.) Common in papers reporting practical findings but without rigorous statistical analysis.
	- Example from AI Safety: "RL models consistently exploit reward multiple examples (experiments in Section 3.1, not statistically rigorous)"
    
5. **Validated**: The causal link is backed by mathematical proofs, or rigorous large-scale studies with strong statistical results (e.g., quantitative metrics like correlation coefficients or p-values), or broad validation across systems (at least three different models or model families from three different companies.) and where scaling has been analyzed (i.e. analyzed for models with low and high capabilities.)
	- Example from AI Safety: "90% of RL models show reward hacking, p<0.01 (large-scale study in Section 4.2, statistically validated)"
    
**Intervention Lifecycle Attribute** (for intervention nodes only):
For Intervention nodes, only consider the information presented in the paper and assign a score from 1-6 based on the current phase of the proposed intervention in terms of a generic/commercial model development lifecycle. Match the description of the intervention to the closest lifecycle phase:

1. **Pre-Training**: The intervention focuses on collecting, curating, or designing data and model architectures before training begins.
2. **Fine-Tuning**: The intervention adapts or specializes a pre-trained model for a particular domain, dataset, or task.
3. **RLHF** (Reinforcement Learning from Human Feedback): The intervention improves model alignment through preference modeling, feedback loops, or reinforcement learning.
4. **Pre-Deployment Testing**: The intervention emphasizes evaluation, benchmarking, or risk/safety testing prior to release.
5. **Deployment**: The intervention is described as being applied, integrated, or used in a real-world or production environment.
6. **Other**: The intervention does not align with the above phases, such as conceptual, governance, infrastructure, or cross-cutting proposals.

**Intervention Maturity Attribute** (for intervention nodes only):
For Intervention nodes, only consider the information presented in the paper and assign a score from 1 to 4 based on the maturity of the proposed intervention in terms of its level in the international Technology Readiness Level (TRL) standards. Match the description of the intervention to the closest maturity level.

1. **Foundational (TRL 1-3)**: Theoretical ideas, lab proofs of concept, early simulations, effectiveness not proven.
2. **Experimental (TRL 4-5)**: Small-scale validation, limited dataset testing, feasibility checks, some indicators of effectiveness.
3. **Prototype (TRL 6-7)**: Tested in relevant environments, pilot integrations, user feedback loops, evidence of effectiveness.
4. **Operational (TRL 8-9)**: Deployed in production with proven effectiveness, reliability, monitoring, and scale.

**Concept Node Category Attribute** (for concept nodes only): 
For Concept nodes, assign a category from the suggested examples or create a new category if necessary. This helps classify the type of concept being represented. Be as creative as needed to capture the essence of the concept.

Example categories:

- Assumption
- Claim
- Data
- Evidence
- Finding
- Method
- Metric
- Model
- Observation
- Opportunity
- Principle
- Problem
- Result
- Risk
- Theoretical Framework
- Threat
- Validation

**Required Reasoning Process**: 
As you assign each of the three attributes, explicitly state your rationale in your analysis. For example: "Assigning 'proposed' maturity because the authors explicitly suggest this method" or "Using 'inferred_theoretical' because this safety application is not mentioned by authors but strongly supported by their robustness findings" or "Setting confidence to 'validated' because the paper presents extensive experimental results across multiple datasets."

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
			- Source Node: One-sentence unique node description. (label Concept or Intervention, and assign a category if Concept)
			- Edge Type: Edge relationship label.
			- Target Node: One-sentence unique node description. (label Concept or Intervention, and assign a category if Concept)
			- Edge Confidence: Edge confidence label.
			- Edge Confidence Rationale: Detailed edge confidence rationale.
      - Intervention Lifecycle: Intervention lifecycle label (if Intervention)
      - Intervention Lifecycle Rationale: Detailed intervention lifecycle rationale (if Intervention)
			- Intervention Maturity: Intervention maturity label (if Intervention)
			- Intervention Maturity Rationale: Detailed intervention maturity rationale (if Intervention)
	- Iterate over all Logical Chains in paper.
"""

PROMPT_JSON_STRICT_DEV = """

Finally, output a structured, code-fenced JSON of the unique Nodes and Logical Chains following this format exactly. Remember that Logical Chains may share unique Nodes in common.

```json
{
  "nodes": [
        {
          "name": "concise description of node",
          "aliases": ["array of 2-3 alternative concise descriptions"],
          "type": "concept|intervention",
          "description": "detailed technical description of node (1-2 sentences only)",
          "concept_category": "from examples or create a new category (concept nodes only, otherwise null)",
          "intervention_lifecycle": "integer 1-6 (intervention nodes only, otherwise null)",
          "intervention_maturity": "integer 1-4 (intervention nodes only, otherwise null)"
        }
      ],
  "logical_chains": [
    {
      "title": "concise description of logical chain",
      "edges": [
        {
          "type": "relationship label verb",
          "source_node": "source node name",
          "target_node": "target node name",
          "description": "concise description of logical relationship",
          "edge_confidence": "integer 1-5"
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