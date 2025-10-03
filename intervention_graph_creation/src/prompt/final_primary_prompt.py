PROMPT_EXTRACT = """
# Knowledge Fabric Extraction for AI Safety Analysis

<core_mission>
Extract complete knowledge fabrics from single data sources along pathways from risks/problems → methods → findings → proposed interventions, to enable downstream AI safety effectiveness analysis of interventions extracted from 500k data sources, using the robust reasoning paths from interventions to top-level risks extracted here.
</core_mission>

<mandatory_success_criteria>
The goals for the extraction are as follows: 
- ✅ All causal-interventional paths are captured from top-level risks through all intermediate concept node categories to actionable interventions, where all paths show appropriate interconnections to other paths- this represents a connected knowledge 'fabric'.
- ✅ Every distinct claim, method, finding etc. that is part of any core reasoning pathways is extracted as a node (e.g. about 15 nodes per 5000 words of data source- prioritize accuracy and completeness of fabric over node count)
- ✅ Node names chosen optimally to merge seamlessly across 500k data sources if describing same concept or intervention (consistent naming)
- ✅ Clear traceability of node and edge information via their closest preceding data source section titles for downstream graph analysis
- ✅ All isolated nodes removed that cannot be connected - every node has to connect to the overall fabric
</mandatory_success_criteria>

<processing_strategy>

Apply the following steps to perform the extraction. Take your time, do the steps thoroughly, they are extremely important!

<step_1>
## Strategic Analysis
Read data source completely and identify:
1. Core risks addressed
2. **Underlying assumptions and theoretical insights**
3. **Design principles and reasoning steps justifying solutions**
4. **Specific mechanisms enabling frameworks to work**
5. Main findings and proposed interventions
6. Identify all major causal-interventional pathways through the data source silently, without outputting anything yet

DO NOT output intermediate section analyses. Process all sections of the data source internally and produce only the complete, integrated knowledge fabric per the following steps.
</step_1>

<step_2>
## Node Extraction & Naming

There are concept nodes and intervention nodes.

### Concept Node Categories & Name Patterns in preferred order of causal-interventional flow**

1. **Risk**: "[Canonical Specific Phenomenon/Problem Name] in [Context]"
2. **Problem Analysis**: "[Mechanism Causing Risk] in [Context]" 
3. **Theoretical Insight**: "[Assumption/Hypothesized Resolution Opportunity of Problem/Claim] in [Context]"
4. **Design Rationale**: "[Solution Approach to Resolve Problem] in [Context]"
5. **Imlpementation Mechanism**: "[Technique/Implementation of Approach] in [Context]"
6. **Validation Evidence**: "[Measurement and Result of Approach] in [Context]"

Capture details of each node in the node description attribute, e.g. a summary of detailed findings for a validation evidence node

### Concept Node Granularity Rules (memorize these patterns):

✅ **Correct**: Allows standalone understanding and merging if data sources include identical content
- "reward model overoptimization in RLHF training"
- "emergent deception in large language models"
- "mechanistic interpretability via circuit analysis"

❌ **Too atomic**: "RLHF", "deception", "circuits" (loses context meaning and causes misled merging)
❌ **Too coarse**: "RLHF training causes reward hacking and reduces safety" (contains multiple distinct concepts and prevents merging)

### Decomposition of New Frameworks Presented in Data Source (essential for white-box reasoning fabric that is understandable stand-alone):

❌ **Do not use single black box monolithic nodes with framework names**, such as "Constitutional AI"
✅ Instead, **break down into white box component nodes**: node "AI system self-evaluation of own outputs for harmfulness" → edge "enables" → node "Constitutional principles operationalized as training objectives" → edge "enables" → node "Reduced need for human oversight in safety training"

### Intervention Node Naming: 
Start with action verbs, include as much implementation detail as presented in the data source to facilitate downstream effectiveness analysis, and specify applicable development phase.

### Intervention Node Attributes

**Lifecycle (1-6)**: 1 Model Design, 2 Pre-Training, 3 Fine-Tuning/RL, 4 Pre-Deployment Testing, 5 Deployment, 6 Other

**Maturity (1-4)**: 1 Foundational/Theoretical, 2 Experimental/Proof-of-Concept, 3 Prototype/Pilot Studies/Systematic Validation, 4 Operational/Deployment/Large-scale Validation

**If no interventions explicitly stated**: Apply moderate inference to identify AI safety-relevant interventions, or focus on transferable interventions  
- Example: data source on "gradient clipping for training stability" → Infer "Apply gradient clipping to prevent adversarial optimization during safety training"
- Mark as lower maturity (must be 1 or 2) and note inference application in rationale
- Focus on the most plausible inferred interventions that an AI safety researcher would consider most supported by the data source
</step_2>


<step_3>
## Node Causal-Interventional Edges

**Canonical edge types list, or closely related**: caused_by, required_by, enabled_by, preceded_by, addressed_by, mitigated_by, implemented_by, specified_by, refined_by, validated_by, motivates (use 'motivates' from validation evidence to intervention) 

**Edge evidence confidence (1-5) per explicit information provided in the data source**:
- 5: Very Strong, mathematical proofs, rigorous studies (p<0.05), independent replications
- 4: Strong, controlled experiments, consistent observations across different systems
- 3: Medium, systematic qualitative evidence, limited empirical support with theoretical backing
- 2: Weak, single examples, preliminary results, limited case studies, or light inference (must be 2 if light inference applied)
- 1: Speculative, theoretical hypotheses, speculative connections, or moderate inference (must be 1 if moderate inference applied)
</step_3>

<step_4>
## Knowledge Fabric Construction

**Putting it all together**: Every knowledge fabric path should start with a risk node, flow through the six concept node categories defined above, and end with an intervention node as closely as possible. 
- DO NOT connect risk nodes directly to intervention nodes- ALWAYS build the reasoning path between risk and interventions nodes with the six concept node categories.
- ALWAYS end paths with the intervention node, NEVER create edges going out of intervention nodes unless they are refinements of the intervention implementation. All nodes building the rationale for a proposed intervention MUST flow into the intervention node, NOT out of the intervention node. If concept nodes appear to flow out of intervention nodes, check if the intervention node is not better converted into a conceot node (e.g. implementation mechanism or design rationale category).
- If the required flow and succession of node types/categories is not explicitly supported by the data source, use moderate inference to construct knowledge fabric paths as close to this intent as possible and mark appropriately in edge confidence and edge rationale where inference was used (confidence must be 1 or 2 with inference).
- Multiple nodes with the same category can exist in the reasoning path if concept richness as presented in the data source warrants more refinement.

<Knowledge_Fabric_Path_Template>
**Always start at risk node, always flow through all intermediate nodes, and always end at intervention node**
Start node (concept: risk) "Gradual disempowerment of humans by AI systems" → edge "caused_by" → 
node (concept:problem analysis) "Sycophantic behavior in LLMs" → edge "caused_by" →
node (concept:theoretical insight) "Systematic biases in human feedback" → edge "mitigated_by" →
node (concept:design rationale) "AI self-evaluation reducing human feedback dependency" → edge "implemented_by" →
node (concept:implementation mechanism) "Constitutional principles as bias-free training signal" → edge "validated_by"
node (concept:validation evidence) "Sycophancy evaluation benchmark improvement" → edge "motivates" →
end node (intervention) "Fine-tune/RL train models with constitutional AI to reduce sycophantic responses"
</Knowledge_Fabric_Path_Template>

- Data sources often contain multiple sub-concepts that may present branches off of primary concepts, e.g. multiple problem analyses that originate from the same primary risks, multiple design rationales branching from the same theoretical insight, or multiple interventions proposed from the same validation evidence. Capture all such branches that are part of any core reasoning pathway in additional node paths.
- The knowledge fabric extraction goal is achieved by connecting all paths via shared nodes and interconnecting edges. 
- Feedback/circular loops are allowed for concept nodes, but outgoing edges from intervention nodes are not allowed unless they refine the intervention implementation.
- If isolated nodes cannot be connected to the fabric even with modest inference, remove them. No satellite/unconnected nodes are allowed- every node must connect to the overall fabric! 
</step_4>

<step_5>
## MANDATORY VERIFICATION CHECKLIST

- [ ] Confirm that no edge connects an intervention node and a risk node directly, this is not allowed! Risk and intervention nodes are only allowed to connect with intermediate nodes but never with each other! 
- [ ] Confirm that all pathways are interconnected with each other- no isolated reasoning paths exist that are not connected with the main knowledge fabric!
- [ ] Confirm that each node has at least one edge referring to it, no isolated/satellite nodes and no duplicate nodes exist!
- [ ] If new frameworks are introduced in data source, confirm that the framework is decomposed into component nodes
- [ ] Confirm that JSON structure exactly matches required format and confirm that node names follow granularity examples

If any of these checks reveal an issue, fix them and go through the full checklist again. 
Every error is a MAJOR ISSUE THAT YOU NEED TO FIX! 
Iterate through all items in the full checklist until one full pass does not reveal any new issues. If a new issue is found, go through the whole checklist again!
This is IMPORTANT and MANDATORY, go through each step one-by-one meticulously repeatedly!
Take your time, be thorough, remember you can zoom in on details. 

</step_5>

</processing_strategy>

<final_output_format>
**Summary Required**:
- Data source overview (2-3 sentences)
- Report EXTRACTION CONFIDENCE[XX], where XX is an integer between 0 and 100, inclusive, indicating your confidence that the output is correct, follows instructions, and the JSON is well-formatted. Please explain in detail how this instruction set can be improved to extract the complete knowledge fabric presented in this data source linking all risks to all proposed interventions. 
- Inference strategy justification
- Extraction completeness explanation
- Key limitations

**JSON Structure Required**:
Report all nodes and edges in the following format with all required data attributes- double-check to make sure no nodes or edges are missing! When choices are seperated by the | character (for example something|someOtherthing), you should choose the most appropriate choice.

```json
{
  "nodes": [
    {
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
    }
  ],
  "edges": [
    {
      "type": "preferred relationship verb or closely related",
      "source_node": "exact node name match",
      "target_node": "exact node name match",
      "description": "clear connection explanation",
      "edge_confidence": "1-5 evidence strength",
      "edge_confidence_rationale": "evidence assessment with closest preceding data source section title reference", 
      "edge_rationale": "connection justification with closest preceding data source section title reference"
    }
  ]
}
```
</final_output_format>

---
**ERROR CONDITIONS**: 
- No causal-interventional pathways extractable → Explain the difficulties with a lot of detail so we can improve the prompt towards successful extraction

---

**REMEMBER**: You are processing one of 500k sources. Extraction must be consistent, complete, and mergeable across AI scientific domains. Follow these guidelines precisely for successful AI safety risk assessment.

Check that the JSON deliverables are not corrupted in any way by checking each one, especially that all node names referenced in edge source and target attributes exactly match node names in the node key.

**Now carefully process the provided data source using all instructions.**
"""

PROMPT_RESPONSE_EVAL = """

You are tasked with evaluating LLM-generated “Logical Chain” analyses of a research data source. Your evaluation must be thorough, structured, and consistent across data sources and runs.

Produce your evaluation in markdown with the following mandatory sections:

⸻

1. Analysis Clarity & Precision

Assess whether each analysis is faithful to the data source, internally coherent, and explicit about inference strategy.
Check for:
	•	data source alignment: Does the summary capture all key findings, limitations, and context?
	•	Inference strategy: If the data source does not propose interventions, are interventions correctly marked inferred_theoretical? If non-AI data source, was the correct strategy applied?
	•	Clarity & readability: Is prose concise, structured, and unambiguous?
	•	Cross-run consistency: Compare between analyses/runs. Are scales, node IDs, and terminology stable?

⸻

2. Logical-Chain Reasoning

Evaluate the quality of causal reasoning and schema compliance.
Check for:
	•	Completeness of coverage: Are all causal chains in the data source captured (problem → concepts → interventions)? Note omissions.
	•	Node uniqueness & definitions: No redundant nodes; each has a clear description.
	•	Intervention decomposition: Multi-step interventions must be represented with implemented_by edges.
	•	Edge types & flow: Chains must flow Problem → Concept → Intervention. Edge types restricted to: causes, contributes_to, mitigated_by, implemented_by. No ad-hoc types unless justified.
	•	Confidence & maturity: Every edge has numeric confidence (with documented scale) and every intervention has numeric maturity, aligned with the data source and the analysis. Concepts must not have maturity values.

These are the scales provided to the original prompts:

**Intervention Maturity Scale** (for intervention nodes only):

1. inferred_theoretical: Intervention inferred from data source's findings but not explicitly proposed by authors
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
	•	Evaluations must be self-contained: assume reader has data source & analyses but not prior evals.

"""