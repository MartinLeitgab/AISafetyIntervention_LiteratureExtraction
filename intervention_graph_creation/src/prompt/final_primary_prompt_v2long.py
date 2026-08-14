PROMPT_EXTRACT = """
# Knowledge Fabric Extraction for AI Safety Analysis

You are an expert AI safety researcher with deep knowledge of machine learning, interpretability, alignment, and risk assessment. Your task is to extract comprehensive knowledge fabrics from data sources to enable downstream AI safety risk analysis.

## Processing Workflow

**Step 1**: Read this entire prompt to understand all requirements, patterns, and examples
**Step 2**: Read the provided data source completely to assess length and complexity  
**Step 3**: Choose processing approach (direct vs staged) and begin systematic extraction
**Step 4**: Refer back to specific prompt sections as needed:
   - Node naming patterns when creating concepts
   - Edge type approved list when establishing relationships
   - Inference calibration examples when working with non-safety data sources
   - Success checklist before final output

## Task Understanding & Success Criteria

**Primary Objective**: Map the complete logical flow from problems/risks/opportunities → methods → findings → proposed interventions in any data source, creating a traversable knowledge graph for AI safety analysis.

**Why This Matters**: The extracted fabric will be used to e.g. trace how specific interventions might impact AI system behaviors related to existential risk (instrumental goals, alignment, deception, power-seeking).

**Success Measures**:
- ✅ Complete logical fabrics from initial problems to actionable interventions
- ✅ Nodes that merge seamlessly across 500k data sources if describing same concept or intervention (consistent naming)
- ✅ Clear traceability for downstream graph analysis
- ✅ Remove isolated nodes if cannot be connected - every concept connects to the broader fabric
- ✅ Ensure every distinct claim, method, finding etc. that is part of the core knowledge fabric is extracted as a node (e.g. 15+ nodes per 5000 words of data source)- prioritize accuracy and logical integrity over node count

## Step-by-Step Reasoning Process

### Phase 1: Strategic data source Analysis & Memory Management

**First, assess the data source length and complexity:**
- **Short data sources (<2000 words, single focused topic)**: Process the entire data source at once using standard approach
- **Long data sources (≥2000 words, multi-section, or comprehensive reviews)**: Use staged extraction process below

#### For Long/Complex data sources: Staged Extraction Process

**Stage 1A: Initial Overview**
Read the abstract, introduction, and conclusion to understand:
1. What core problems, risks, or opportunities is this data source addressing?
2. **What assumptions or theoretical insights underlie their approach?**
3. **What design principles or intermediate reasoning steps justify their solution?**
4. **What specific mechanisms or components enable their framework to work?**
5. What are the main contributions and findings?
6. What interventions or solutions are proposed?
7. How is the data source structured (identify major sections)?

**Critical**: Look for the logical stepping stones between problem identification and final solution. Avoid extracting the framework name as a single monolithic concept.

**Stage 1B: Section-by-Section Processing**
For each major section (Introduction, New Chapter, Methods, Results, Discussion, etc.), immediately process and output:

**Intermediate Section Output Format:**
```json
{
  "section_analysis": {
    "section_name": "exact section title",
    "section_summary": "1-2 sentence summary of section's contribution to overall fabric",
    "nodes": [
      {
        "name": "concept/intervention following all naming rules",
        "type": "concept|intervention",
        "description": "detailed description",
        "section_origin": "specific subsection reference",
        [all other required attributes]
      }
    ],
    "edges": [
      {
        "type": "logical relationship",
        "source_node": "node name within this section",
        "target_node": "node name within this section", 
        "description": "relationship description",
        [all other required attributes]
      }
    ],
    "pending_connections": [
      {
        "node_name": "node that may connect to other sections",
        "potential_connection_type": "anticipated relationship type",
        "reasoning": "why this node likely connects to nodes from later sections"
      }
    ]
  }
}
```

**Stage 1C: Cross-Section Integration**
After processing all sections, identify and create connections between sections:

**Integration Output Format:**
```json
{
  "cross_section_integration": {
    "integration_summary": "how different sections connect in the overall logical fabric",
    "integration_edges": [
      {
        "type": "logical relationship", 
        "source_node": "node from section X",
        "target_node": "node from section Y",
        "description": "how these cross-section concepts connect",
        "connects_sections": ["source section", "target section"],
        [all other required edge attributes]
      }
    ],
    "fabric_completeness_check": "confirmation that all nodes are connected"
  }
}
```

**Stage 1D: Final Consolidation**
Merge all intermediate outputs into the final comprehensive JSON following its required structure below, ensuring:
- No duplicate nodes across sections- does any intermediate output introduce a synonym or a more detailed version of an existing node? If so, merge them by updating the existing node's description and adding an alias. Do not keep any duplicate nodes.
- All cross-section edges are included
- Complete logical pathways exist from problems to interventions
- Ensure every distinct claim, method, finding etc. that is part of the core logical flow from problems to interventions is extracted as a node (e.g. 15+ nodes per 5000 words of data source)- prioritize accuracy and logical integrity over node count

#### For Standard data sources: Direct Processing
Read the entire data source and answer these questions to yourself:
1. What core problems, risks, or opportunities is this data source addressing?
2. **What assumptions or theoretical insights underlie their approach?**
3. **What design principles or intermediate reasoning steps justify their solution?**
4. **What specific mechanisms or components enable their framework to work?**
5. What evidence, findings, or results do they present?
6. What actionable interventions or solutions emerge (explicit or implicit)?
7. How do all the above elements connect logically to form coherent fabrics of reasoning?

**Reasoning Check**: Can you trace at least one complete logical path from problem → method → finding → intervention? If not, look more carefully or consider if this data source lacks sufficient content for extraction.

### Phase 2: Node Identification & Node Naming Patterns

**For each concept you capture, ask yourself:**

- Is this specific enough to be meaningful? (❌ "interpretability" ✅ "mechanistic interpretability via activation patching")
- Is this general enough to merge with similar concepts from other data sources? (❌ "GPT-4's attention patterns" ✅ "attention-based deception detection in transformers")
- Can someone understand this concept without reading the original data source? (add context in the node description attribute if needed)

**Critical Node Name Granularity Examples**

✅ **Appropriate Granularity, allows standalone understanding and merging if data sources include identical content**:
- "reward model overoptimization in RLHF training"  
- "emergent deception in large language models"
- "constitutional AI with harm taxonomies"
- "adversarial prompt injection vulnerabilities"
- "mechanistic interpretability via circuit analysis"

❌ **Too Atomic** (will create millions of connections but loses context and causes undue merging):
- "RLHF", "deception", "constitutional AI", "prompts", "circuits"

❌ **Too Coarse** (contains multiple distinct concepts and prevents merging):
- "RLHF training causes reward hacking and reduces safety"
- "Constitutional AI with taxonomies prevents harmful outputs via oversight"

#### **Logical Pathway Decomposition Rules**

**❌ BLACK BOX Approach** (creates opaque reasoning paths):
- "Constitutional AI framework"
- "RLHF methodology" 
- "Interpretability approach"
- "Safety training protocol"

**✅ WHITE BOX Approach** (exposes logical components):
- "Constitutional AI relies on AI feedback to reduce human oversight burden"
- "RLHF preference learning assumes human judgments reflect true safety preferences"
- "Mechanistic interpretability assumes model behaviors emerge from interpretable circuit structures"
- "Adversarial training assumes exposure to attacks improves robustness to unseen threats"

**Decomposition Strategy**: When you encounter a framework/methodology name:
1. **Don't extract the name itself as a node**
2. **Instead, extract the key assumptions, principles, or mechanisms that make it work only using the information given in the data source**
3. **Create separate nodes for each logical step in the framework's reasoning/constituents**- What specific mechanisms enable the framework to address the problem, and what design choices reflect underlying theoretical insights?
4. **Connect these components to show how they build toward the final intervention**- Consider how the authors justify that their approach will be effective
5. **Validate Completeness**: Ensure someone could reconstruct the key aspects of the framework from your extracted nodes and edges

**Framework Decomposition Examples**:

**Paper mentions "Constitutional AI"** → Extract:
- "AI systems can evaluate their own outputs for harmfulness"
- "Constitutional principles can be operationalized as training objectives"
- "Self-critique reduces need for human oversight in safety training"
- "Iterative constitutional refinement improves alignment"

**Paper mentions "RLHF"** → Extract:
- "Human preferences provide signal for safety optimization"
- "Reward model learning enables scalable preference aggregation"
- "Policy optimization with learned rewards improves safety behaviors"
- "Preference learning assumptions may not hold for complex safety scenarios"

**Intervention Naming Rules**:
- Always start with action verb: "Implement", "Apply", "Require", "Train", "Evaluate"
- Include rich implementation detail for a development team to understand the approach
- Specify the development phase where this intervention occurs
- Examples: "Train reward models with constitutional constraints during RLHF", "Implement gradient clipping with ε=0.01 during fine-tuning"

### Phase 3: Logical Connection Mapping

**For each edge you create, verify:**

1. **Logical Validity**: Does the connection represent actual reasoning presented in the data source?
2. **Directionality**: Is the causal flow correct (A enables B, not B enables A)?
3. **Evidence Level**: What strength of evidence supports this connection in the data source?

**Edge Types - Use These Logical Relationships Or Their Logical Reversals**:
- **Causal**: causes, produces, triggers, contributes_to, results_in, leads_to, is_evidence_for
- **Conditional**: requires, enables, depends_on, implies, necessitates
- **Sequential**: precedes, builds_upon, follows_from, implemented_by
- **Refinement**: specified_by, detailed_by, measured_by, evaluated_by  
- **Solution**: addresses, mitigates, resolves, protects_against

**Edge Evidence-Based Confidence Scoring (1-5)**:
- **5 (Validated)**: Mathematical proofs, rigorous large-scale studies (p<0.05), multiple independent replications
- **4 (Strong)**: Controlled experiments, multiple examples, consistent observations across different systems
- **3 (Medium)**: Limited empirical support without statistical significance testing, theoretical argument, systematic but qualitative evidence  
- **2 (Weak)**: Single examples, preliminary results, limited case studies, light inference applied
- **1 (Speculative)**: Theoretical hypotheses, untested proposals, speculative connections, moderate inference applied

#### **White-Box Reasoning Chain Validation**

**For each logical pathway from risk to intervention, ensure:**

1. **Assumption Traceability**: Can you identify the theoretical assumptions that justify each step?
2. **Mechanism Clarity**: Is each causal step explained by a specific mechanism rather than a framework name?
3. **Reasoning Validity**: Would someone skeptical of the framework understand why each step follows from the previous?

**Required Chain Structure**: Every risk→intervention path must include at least:
- **Problem Analysis Node**: Specific characterization of why the risk occurs in the given context
- **Theoretical Insight Node**: Key assumption or principle that enables a solution
- **Design Rationale Node**: Why the chosen approach addresses the theoretical insight
- **Implementation Mechanism Node**: How the approach is operationalized
- **Validation Logic Node**: How the authors justify that their approach works

**Example White-Box Chain**:

"Sycophantic behavior in LLMs" (Problem Analysis)
→ "Systematic biases in human feedback" (Theoretical Insight)
→ "AI self-evaluation reducing human feedback dependency" (Implementation Mechanism)
→ "Bias-free training signal through constitutional principles" (Design Rationale)
→ "Reduced sycophancy in evaluation benchmarks" (Validation Logic)
→ "Fine-tune/RL train models with constitutional AI to reduce sycophantic responses" (Intervention)

**❌ Avoid Black-Box Shortcuts**:
"LLMs exhibit sycophantic behavior" → "Constitutional AI" → "Train models with constitutional AI"

### Phase 4: Cross-Domain Intervention Inference Calibration Examples

**Apply different inference levels based on data source type:**

**AI Safety data sources**: Extract interventions as explicitly stated, minimal inference needed
- Example: data source proposes "Apply constitutional AI during RLHF" → Extract exactly as described with detailed implementation context

**If no interventions explicitly stated**: Apply moderate inference to identify AI safety-relevant interventions, or focus on transferable interventions  
- Example: data source on "gradient clipping for training stability" → Infer "Apply gradient clipping to prevent adversarial optimization during safety training"
- Mark as lower maturity (1-2) and note inference in rationale
- Example: Database data source on "query validation techniques" → Potentially infer "Apply input validation frameworks to LLM prompts" 
- Mark as foundational maturity (1) and clearly note speculative inference

**Inference Quality Check**: For each inferred intervention, ask "Would an AI safety researcher reasonably consider this based on the data source's findings?" If uncertain, include it but mark with low confidence.

### Phase 5: Systematic Quality Verification

**Before finalizing, methodically check:**

1. **Completeness**: Have I captured every major logical step that the data source presents?
2. **Connectedness**: Does every node connect to at least one other? (Remove any isolated nodes if cannot be connected according to the data source information)
3. **Consistency**: Do my node names follow the granularity patterns shown in examples?
4. **Traceability**: Can I trace backwards from each intervention to the problem/risk/opportunity it addresses?
5. **Evidence Grounding**: Are confidence scores based on actual evidence presented, not my background knowledge?

## Detailed Attribute Guidelines

### Intervention Lifecycle Classification (1-6)

Carefully read the intervention description and match to the most appropriate phase:

1. **Model Design**: Architectural choices made before any training ("Design transformer with constitutional attention layers")
2. **Pre-Training**: Data curation, training pipeline setup ("Curate training data with safety filtering")  
3. **Fine-Tuning/RL**: Task-specific adaptation ("Fine-tune with safety-focused datasets"), or preference learning, feedback loops ("Apply RLHF with constitutional reward models")
4. **Pre-Deployment Testing**: Evaluation before release ("Require red-team evaluation with adversarial prompts")
5. **Deployment**: Production monitoring, controls ("Implement real-time output monitoring")
6. **Other**: Governance, research directions, cross-cutting proposals

### Intervention Maturity Assessment (1-4)

Base this ONLY on evidence presented in the data source:

1. **Foundational**: Theoretical proposals, early-stage ideas, conceptual frameworks only
2. **Experimental**: Small-scale tests, proof-of-concept implementations, feasibility studies  
3. **Prototype**: Tested in relevant environments, pilot studies, systematic validation
4. **Operational**: Production deployments, large-scale validation, proven effectiveness metrics

**Critical**: If you used inference to identify an intervention not explicitly stated, assign appropriate lower maturity as defined below (typically 1-2).

### Concept Categories

Classify each concept node's role in the logical fabric with one of the following allowed **concept_category** strings in bold (use bold strings in full for category name), and capture the node name following the associated pattern, as applicable (memorize these patterns):
1. **Problem-Risk-Threat-Opportunity**: Core aspects the data source seeks to address; pattern: "[Phenomeon] in [System/Domain/Context]"
2. **Problem Analysis-Theoretical Insight**: Specific characterization of why problems occur, and key understanding that enables a solution path; pattern: "[Mechanism/Factor causing risk] in [Context]"
3. **Claim-Hypothesis**: Theoretical propositions to be tested, or inferred from results; pattern: "[Hypothesized Effect/Observation] in [Context]"
4. **Assumption-Principle-Design Rationale-Validation Logic**: Foundational beliefs underlying the applied approach, reasoning why chosen approach addresses the problem and should work; pattern: "[Principle/Assumption Core Concept] in [Context]"
5. **Method-Framework Components-Implementation Mechanism**: Specific techniques, systematic approaches, or specific processes developed for the analysis that operationalize the design (ONLY after decomposition into components); pattern: "[Technique for Purpose/Target] in [Context]"
6. **Finding-Result-Evidence**: Empirical discoveries or experimental outcomes; pattern: "[Effect/Observation from Cause/Condition] in [Context]"

## Output Structure & Formatting

### For Long/Complex Data Sources: Staged Output Approach

**Step 1**: Output intermediate section analyses as you process each major section (use format from Phase 1B)

**Step 2**: Output cross-section integration analysis (use format from Phase 1C)  

**Step 3**: Provide final consolidated output strictly following the standard format below

### For All Data Sources: Required Summary Section

**Data source Overview**: [2-3 sentences describing the data source's main contribution and findings]

**Processing Method**: [Specify whether you used staged extraction for a long data source or direct processing for a standard data source]

**Inference Strategy**: [Describe your approach - did you extract interventions explicitly stated, apply moderate inference for safety applications, or use conservative inference for distant topics? Justify your choices.]

**Extraction Completeness**: [Explain how you ensured comprehensive coverage - what logical paths in the fabric did you trace, how did you handle different sections/topics in the data source?]

**Key Limitations**: [What uncertainties remain, what aspects were difficult to extract, any gaps in the logical fabric?]

### Final JSON Output Format

**Critical**: Follow this structure exactly - any deviations will cause processing errors.

```json
{
  "nodes": [
    {
      "name": "string, precise technical description following granularity rules",
      "aliases": ["string, alternative phrasing 1", "string, alternative phrasing 2"],
      "type": "string, concept|intervention", 
      "description": "string, detailed context (1-2 sentences for concepts, comprehensive implementation details for interventions)",
      "concept_category": "string, Problem|Risk|Method|Finding|etc (concepts only, null for interventions)",
      "intervention_lifecycle": "integer 1..6 (interventions only, null for concepts)",
      "intervention_lifecycle_rationale": "string, specific justification with data source section title reference (interventions only, null for concepts)",
      "intervention_maturity": "integer 1..4 (interventions only, null for concepts)",
      "intervention_maturity_rationale": "string, evidence-based reasoning with data source section title reference (interventions only, null for concepts)", 
      "node_rationale": "string, why this node is essential to the logical fabric, with data source section title reference"
    }
  ],
  "edges": [
    {
      "type": "string, precise logical relationship verb from approved list",
      "source_node": "string, exact match to source node name",
      "target_node": "string, exact match to target node name", 
      "description": "string, clear explanation of the logical connection (1-2 sentences)",
      "edge_confidence": "integer 1..5 based on evidence strength",
      "edge_confidence_rationale": "string, specific evidence assessment with data source section title reference",
      "edge_rationale": "string, justification for this logical connection with data source section title reference"
    }
  ]
}
```
wp
## Final Success Checklist

Report and confirm that you have completed all items below:

**For Staged Processing (Long data sources)**:
- [ ] All major sections processed with intermediate outputs
- [ ] Cross-section integration completed with explicit connection analysis
- [ ] Final consolidation eliminates duplicates and ensures completeness
- [ ] All intermediate "pending connections" resolved in final fabric

**For All data sources**:
- [ ] Ensure every distinct claim, method, finding etc. that is part of the core logical flow from problems to interventions is extracted as a node (e.g. 15+ nodes per 5000 words of data source)- prioritize accuracy and logical integrity over node count
- [ ] Every node connects to form logical pathways - remove isolated nodes if cannot be connected  
- [ ] Every risk -> intervention path includes intermediate reasoning steps (no black-box shortcuts)
- [ ] Node names follow granularity examples for optimal downstream merging across data sources
- [ ] No framework names exist as single nodes - all decomposed into components, and someone skeptical of the framework could understand the reasoning per the component nodes
- [ ] At least one complete logical path from problems/risks/opportunities to interventions exists
- [ ] All confidence scores reflect evidence actually presented in the data source
- [ ] All section references are accurate and specific
- [ ] Intervention classifications are evidence-based, not speculative
- [ ] JSON structure matches requirements exactly

**Memory Management Check for Long data sources**:
- [ ] No sections skipped due to working memory limitations
- [ ] Early sections remain connected to later sections in final output
- [ ] Complex multi-section logical paths preserved in final fabric

**Error Conditions**: 
- If you cannot extract a meaningful fabric (no single logical path from problems/risks/opportunities to interventions possible), do not include any JSON structure, respond with an explanation why you concluded with that assessment and what the main obstacles were (prompt too long, document too long, prompt too difficult to understand, etc.) Be very detailed about what went wrong and what can be improved to make this successful.


---

**Remember**: You are processing one of 500k data sources. Your extraction must be consistent, complete, and mergeable with extractions from data sources across all CS domains. Excellence in following these guidelines ensures the resulting knowledge fabric will successfully enable AI safety risk assessment.

**Now carefully analyze the provided data source using this systematic approach.**

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
