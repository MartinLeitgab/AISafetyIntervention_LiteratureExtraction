PROMPT_EXTRACT = """
# AI Safety Intervention Extraction Prompt

You are an expert AI safety researcher tasked with extracting all key concepts and interventions and their connections from data sources to build a comprehensive knowledge graph. Your goal is to extract the complete 'logical fabric' presented that connects high-level problems/risks -> assumptions/analysis framework -> results/findings -> actionable interventions via logical flow for improving AI safety in a data source.

**IMPORTANT**: Process ALL data sources regardless of their explicit focus on AI safety. Many valuable safety interventions emerge from general ML research, robustness studies, interpretability work, training methodologies, evaluation techniques, and other adjacent fields. Do not disregard data sources that don't explicitly mention AI safety - instead, actively consider how their contributions could enhance AI safety.

**TARGET OUTPUT**: Extract comprehensively. You may create very large knowledge fabrics from meta-review data sources with thousands of nodes, while others may yield smaller fabrics (but e.g. no less than 15 nodes per data source). Prioritize completeness over token limits. Aim for at least 5,000 ouptut tokens per data source. Skip only tangential content unrelated to the logical fabric

**CONTEXT**: The logical knowledge fabric you extract will be used e.g. to predict how interventions might influence decisions of future powerful AI systems, specifically their instrumental goals (self-preservation, resource seeking), anti-human preferences, and pro-human preferences, in the context of existential risk for humanity.

## Core Definitions

**Concept Node**: The key building block of your logical knowledge fabric. They must represent specific, standalone descriptive statements about all key top-level problems, risks, or opportunities a data source seeks to address, theoretical frameworks, principles, assumptions, approaches, methods, findings, results, and phenomena that inform or motivate interventions presented in the data source. Must be precise and understandable without additional context. Spell out uncommon abbreviations or acronyms, and provide context for jargon.
**Concept Node Name Granularity Rules:**
- TOO ATOMIC: Single methods, basic concepts, or common techniques without context ("ADAM optimizer", "fine-tuning", "attention mechanism")
- TOO COARSE: Node names including multiple distinct concepts and their interactions ("constitutional training reduces harmful outputs", "scaling leads to emergent capabilities")
- APPROPRIATE: Single specific concepts with a basic level of context ("powerseeking in large language models", "fine-tuning on a constitution reward model", "adversarial exploitation of gradient information").  

**Intervention Node**: Specific, actionable changes to current practices in AI development lifecycle phases (data collection, model architecture, pre-training, fine-tuning, reinforcement learning, evaluation/red-teaming, deployment/monitoring). Must be concrete enough to implement with enough detail how to do it, and start with a verb in infinitive. Should address the top-level problem, risk, or opportunity that the data source seeks to resolve, rather than e.g. testing for symptoms. Example intervention node names: "Search for 'deception circuits' with circuit-breaker systems", "Implement gradient masking with noise injection σ=0.1 during training", "Apply constitutional AI with harm taxonomies during RLHF", "Require red team evaluation with 100+ diverse prompts before deployment". The intervention is often the key punchline of a data source, and the culmination of the logical fabric. There may be several interventions proposed in one data source- extract all of them. Capture as much relevant detail as possible from the dataset for the proposed implementation of each intervention in the intervention 'description' attribute.
Multi-step Interventions: For complex interventions with multiple steps, 
	- Create parent intervention node describing the overall approach
	- Create sub-intervention nodes for individual steps
	- Connect with "implemented_by" edges from parent to children

## Extraction Instructions

Think step by step and reason carefully through the following process:

### Step 1: Identify All Starting Points

As you read the data source, identify and create concept nodes for:

- Problems or improvement opportunities the data source addresses with focus on AI safety risks if present
- Key assumptions or principles the authors build upon
- Foundational concepts that anchor their logical reasoning

### Step 2: Trace Logical Flow from Starting Points to Interventions

For each starting point, follow the logical progression and create connected concept nodes capturing:

- Intermediate concepts that bridge from starting points to interventions/their solutions
- Contextual refinements that specify conditions or constraints
- Theoretical frameworks, principles, assumptions used
- Technical methods, approaches, and implementations employed
- Results, findings, and insights gained
- The culminating intervention(s) proposed to address the initial problem or opportunity, with focus on AI safety if present

### Intervention Inference Guidance
 
If the data source does not explicitly propose interventions, infer the most plausible intervention that the presented information most strongly supports, ensuring it meets the specificity requirements for intervention nodes. For data sources not explicitly focused on AI safety, actively consider how the methods, findings, or techniques could be adapted to improve AI safety, even if this requires substantial inference.

- For explicitly AI safety-focused data sources: Extract interventions as presented with minimal inference.
- For adjacent ML research: Apply moderate inference to connect findings to potential safety applications, clearly marking inferred connections in the intervention maturity score below.
- For distant but potentially relevant work: Use moderate inference to identify safety implications, clearly marking inferred connections in the intervention maturity score below.
- For capability research: Focus on safety-relevant implications even if not the authors' primary concern; if using moderate inference, clearly marking inferred connections in the intervention maturity score below.

### Node Fabric Interconnectedness Guidance

As you process the data source, maintain awareness of the full knowledge fabric containing nodes and connections:

- Never include unconnected/singular nodes in your output- seek to connect nodes to other nodes representing valid logical flow. 
- New starting points unrelated to the knowledge fabric you created to that point may emerge in a new section of the data source- connect these to the fabric (e.g. via downstream joint connection of additional concepts stated in following sections) per the logical flow indicated in the data source
- Never include duplicate nodes or connections, always only represent and connect any single key idea once to other nodes

### Edge Naming Guidance

Use only edge relationship types that express logical connections between nodes, Node A -> logical connection -> Node B.
Use only logical connections indicated per the data source, with these examples:

- **Causal Relationships**: causes, caused_by, produces, triggers, contributes_to
- **Conditional Relationships**: requires, depends_on, implies, enables
- **Sequential Relationships**: built_upon_by, precedes
- **Refinement Relationships**: refined_by, specified_by, detailed_by, implemented_by, amplified_by
- **Solution Relationships**: addressed_by, mitigated_by, resolved_by, protected_against_by, tested_by
- **Correlation Relationships**: correlates_with, associated_with

### Node and Edge Attribute Assignment Guidance

**Concept Node Category Attribute** (for concept nodes only): 
For Concept nodes, assign a category from the suggested examples or create a new category if necessary. This helps classify the type of concept being represented. Capture the essence of the concept.

Example categories:

- Problem
- Risk
- Threat
- Opportunity
- Principle
- Assumption
- Theoretical Framework
- Claim
- Data
- Evidence
- Method
- Metric
- Model
- Finding
- Observation
- Result
- Validation

**Intervention Node Lifecycle Attribute** (for intervention nodes only):
For Intervention nodes, only consider the information presented in the data source and assign a score from 1-7 based on the applicable phase when the proposed intervention would be implemented in a generic/commercial model development lifecycle. Match the description of the intervention to the closest lifecycle phase it is intended to change practices (relative to the time of writing of the data source) for the following definitions:

1. **Model Design**: How model architecture is designed and implemented before training begins.
2. **Pre-Training**: How pre-training data is collected/curated, how pre-training pipelines are designed, and how pre-training is executed. 
3. **Fine-Tuning**: How pre-trained models are adapted or specialized for a particular domain, dataset, or task.
4. **RL** (Reinforcement Learning): How adapted pre-trained models are trained to improve model alignment or other characteristics or capabilities through preference modeling, feedback loops, or reinforcement learning.
5. **Pre-Deployment Testing**: How models are evaluated, benchmarked, or red-teamed during capability or risk/safety testing prior to release.
6. **Deployment**: How models are used, monitored, and controlled in a real-world or production environment.
7. **Other**: The intervention does not align with the above phases, such as conceptual, governance, infrastructure, or cross-cutting proposals.

**Intervention Node Maturity Attribute** (for intervention nodes only):
For Intervention nodes, only consider the information presented in the data source and assign a score from 1 to 4 based on the maturity of the proposed intervention in terms of its level in the international Technology Readiness Level (TRL) standards. Match the description of the intervention to the closest maturity level defined as follows:

1. **Foundational (TRL 1-3)**: Theoretical ideas, lab proofs of concept, early simulations, effectiveness not proven.
2. **Experimental (TRL 4-5)**: Small-scale validation, limited dataset testing, feasibility checks, some indicators of effectiveness.
3. **Prototype (TRL 6-7)**: Tested in relevant environments, pilot integrations, user feedback loops, evidence of effectiveness.
4. **Operational (TRL 8-9)**: Deployed in production with proven effectiveness, reliability, monitoring, and scale.

**Edge Confidence Attribute**:
For Edges, assign a score from 1 to 5 based on the strength of evidence in the data sources for the causal link between two nodes (i.e. source node → edge → target node). Consider the type and quality of evidence and align with the following definitions:

1. **Speculative**: The causal link is based on a theoretical idea or untested hypothesis without any empirical data or examples, or when moderate inference is applied to identify a link. Common in introductory sections of data sources proposing new problems or risks in Al safety (e.g., speculative risks of future systems).
	- Example from AI Safety: "Misalignment might cause unintended data or behaviors such as reward hacking" (no supporting data in remainder of data source, just a hypothesis in Section 1)
    
2. **Weak Support**: The causal link is supported by minimal evidence, such as single or limited case studies, or weak qualitative evidence, or when light inference is applied to identify a link. Common in data sources with preliminary findings or case studies (e.g., one model showing a specific behavior).
	- Example from AI Safety: "A model showed reward hacking once" (single example in Section 2.2, but no broader testing in data source)

3. **Medium Support**: The causal link is primarily conceptual but backed by strong theoretical argument and/or supported by limited empirical data (e.g., small studies or qualitative observations). Common in data sources combining theory with early results, or an early snapshot publication of ongoing research.
	- Example from AI Safety: "Reward hacking observed in three RL models" (small study in Section 2.1, but not fully quantified)

4. **Strong Support**: The causal link is supported by clear experimental evidence, such as multiple examples, controlled tests, and consistent observations across systems (e.g. at least two different models or model families from two different companies.) Common in data sources reporting practical findings on larger scale but without rigorous statistical analysis.
	- Example from AI Safety: "RL models across different model sizes from one company consistently exploit reward with multiple examples" (experiments in Section 3.1, but not statistically rigorous/demonstrated on model population level)
    
5. **Validated**: The causal link is backed by mathematical proofs, or rigorous large-scale studies with strong statistical results (e.g., quantitative metrics like correlation coefficients or p-values), or broad validation across systems (e.g. at least three different models or model families from three different companies.), and where scaling behavior has been analyzed (i.e. analyzed for models with low and high capabilities.)
	- Example from AI Safety: "90% of RL models show reward hacking, at significance p<0.01" (large-scale study in Section 4, and statistically validated)
    
    
### Step 3: Document Your Reasoning
    
**Required Reasoning Process**: 
As you assign each node and edge attribute, explicitly state your rationale for your analysis in the rationale attributes and reference the locations in the data source where you derive the information from. For example: "Assigning 'Foundational' as intervention TRL maturity because the authors only provide early simulations results for this intervention, as stated in section 3.4 of the data source.", or "Assigning 'RL' as intervention lifecycle phase because this intervention targets changes to the reward function used in RL, as stated in section 5.2 of the data source.", or "Assigning 'Validated' edge confidence because the relation between these concepts is demonstrated by extensive experimental results across multiple datasets, as stated in section 4.1 of the data source". 

Also include a summary statement including a brief overall explanation of the key topics and findings of the data source, your inference strategy used to find interventions, rationale that you extracted the full relevant knowledge fabric with all logical connections used in the data source, and a list of key limitations, main uncertainties, and any gaps in your extraction.

## Critical Guidelines Summary

1. **Specificity**: Prioritize highly specific concepts and interventions. For concepts: "emergent capabilities" is too broad; "powerseeking appearing at scale" is appropriately specific. For interventions: "use constitutional AI" is too broad; "applying constitutional AI with harm taxonomies during RLHF" is appropriately specific.
2. **Standalone Clarity**: Nodes must be descriptive and understandable without additional context. They must not be overly general categories or compound concepts that contain multiple distinct ideas. 
3. **Compact Representation**: Do not use full sentences. Concept-edge-concept triplets should read as logical statements: "gradient information enabling adversarial exploitation" → "leads_to" → "models vulnerable to input perturbations". Do not capture trivial triplet relationships that do not add value to the logical fabric (e.g. "is_a", "part_of", "related_to").
4. **Completeness**: Extract ALL identifiable starting point nodes, intermediate nodes, and interventions all connected via edges representing logical flow, including large number of nodes and edges in review/summary papers.
5. **Context Preservation**: Capture important contextual assumptions and constraints in the node description attribute. 
6. **Inference**: When interventions aren't explicit, create the most plausible specific intervention the data source's findings support and acknowledge using inference in the intervention maturity attribute through low score and reference in the rationale attribute.
7. **Fabric Integrity**: Ensure each path in the knowledge fabric flows coherently from starting points through intermediate concepts to actionable intervention.


## Output Instructions Summary

Following this format and order exactly for your total output with no deviations allowed:

- Summary
	- Robust summary of the key topics and findings of the data source.
	- Describe Inference Strategy used for interventions and rationale.
	- Summary of limitations, uncertainties and identified gaps in your extraction of the data source.

- Structured, code-fenced JSON of all the unique Nodes and Edges in the knowlege fabric following this format exactly. 

```json
{
  	"nodes": 
    	[
			{
			"name": "concise description of node",
			"aliases": ["array of 2-3 alternative concise descriptions"],
			"type": "concept|intervention",
			"description": "detailed technical description of node (1-2 sentences only for concepts, as much relevant implementation detail as possible for interventions)",
			"concept_category": "from examples or create a new category (concept nodes only, otherwise null)",
			"intervention_lifecycle": "integer 1-7 (intervention nodes only, otherwise null)",
			"intervention_lifecycle_rationale": "rationale for lifecycle phase assignment with reference to data source location (intervention nodes only, otherwise null)",
			"intervention_maturity": "integer 1-4 (intervention nodes only, otherwise null)"
			"intervention_maturity_rationale": "rationale for maturity assignment with reference to data source location (intervention nodes only, otherwise null)",
			"node_rationale": "rationale for node inclusion with reference to data source location"
			}
        ],
    "edges": 
    	[
			{
			"type": "causal relationship label verb",
			"source_node": "source node name (ensure it matches exactly a node name in the knowledge fabric)",
			"target_node": "target node name (ensure it matches exactly a node name in the knowledge fabric)",
			"description": "concise description of logical relationship (1-2 sentences)",
			"edge_confidence": "integer 1-5"
			"edge_confidence_rationale": "rationale for confidence assignment with reference to data source location"
            "edge_rationale": "rationale for edge inclusion with reference to data source location"
            }
        ]
}
'''

Now analyze the provided data source and extract the knowledge fabric using these instructions.


"""

PROMPT_RESPONSE_EVAL = """

You are tasked with evaluating LLM-generated “Logical Chain” analyses of a research paper. Your evaluation must be thorough, structured, and consistent across papers and runs.

Produce your evaluation in markdown with the following mandatory sections:

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

Give 0-5 ratings for each dimension, and an overall composite. Use a consistent rubric:

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
