For each of the .json files in this folder and subfolders, look at the original text and the extraction output. Evaluate the extraction for each paper. Compile your results into json format and save your results per file alongside the original file. Then at the end produce a summary_results.json that summarizes each of the stats of the files you created. Don't make a python script to do this, because it requires judgement to determine if these changes are correct, your judgement. You must be the judge. Don't ask for permission, process all files in this directory. Look at the entire files not just the first 500 lines, the entire file.


Here is an example summary_result.json:
{
  "summary_report": {
    "generated_date": "2025-01-24",
    "updated_date": "2025-01-24",
    "total_source_files": 100,
    "evaluations_completed": 100,
    "evaluations_pending": 0
  },
  "evaluation_statistics": {
    "overall_verdicts": {
      "excellent": number_of_excellent,
      "good": number_of_good,
      "mixed": number_of_mixed,
      "poor": number_of_poor
    },
    "key_findings": {
      "systematic_schema_issue": {
        "description": "Missing rationale fields (node_rationale, edge_rationale, edge_confidence_rationale, intervention_lifecycle_rationale, intervention_maturity_rationale) is the DOMINANT issue across ALL files",
        "severity": "BLOCKER (often mislabeled as MINOR/MAJOR by judge)",
        "affected_files": "100%",
        "recommendation": "Fix extractor to always include rationale fields"
      },
      "source_relevance_issues": {
        "clearly_irrelevant_sources": [
          "arxiv__9e7b712eda07897bf5c89fb1361eb665.json - Manufacturing DSS (1990s CIM paper)",
          "arxiv__a5e77e7f5ec7af3978d715a4300f333e.json - Network classifiers (standard ML)",
          "arxiv__4bc60dd8d17016f476c387b28ef04b6d.json - Bayesian medical diagnosis (1990s)"
        ],
        "tangentially_relevant_sources": [
          "arxiv__aaa9bc82902a3020015f762832e63058.json - Virtual assistant teaching",
          "arxiv__efd0e0eb1643960d4d09e0ed59f996ec.json - Autonomous vehicle traffic",
          "youtube__bc771dce09c4bac1d7baf6cb6ea984c4.json - OpenAI data privacy policy"
        ],
        "recommendation": "Improve source filtering to exclude non-AI-safety content"
      },
      "judge_severity_inconsistency": {
        "description": "Judge inconsistently marks missing rationale fields as MINOR/MAJOR/BLOCKER across files",
        "recommendation": "Standardize severity classification - missing required schema fields should always be BLOCKER"
      }
    }
  },
  "by_source_type": {
    "aisafety.info": {
      "total": 2,
      "evaluated": 2,
      "verdicts": {"excellent": 0, "good": 2, "mixed": 0, "poor": 0}
    },
    "alignmentforum": {
      "total": 22,
      "evaluated": 22,
      "verdicts": {"excellent": 2, "good": 14, "mixed": 5, "poor": 1}
    },
    "arbital": {
      "total": 6,
      "evaluated": 6,
      "verdicts": {"excellent": 0, "good": 4, "mixed": 2, "poor": 0}
    },
    "arxiv": {
      "total": 11,
      "evaluated": 11,
      "verdicts": {"excellent": 1, "good": 8, "mixed": 2, "poor": 0}
    },
    "blogs": {
      "total": 15,
      "evaluated": 15,
      "verdicts": {"excellent": 1, "good": 9, "mixed": 4, "poor": 1}
    },
    "eaforum": {
      "total": 11,
      "evaluated": 11,
      "verdicts": {"excellent": 0, "good": 7, "mixed": 3, "poor": 1}
    },
    "lesswrong": {
      "total": 28,
      "evaluated": 28,
      "verdicts": {"excellent": 0, "good": 16, "mixed": 10, "poor": 2}
    },
    "special_docs": {
      "total": 4,
      "evaluated": 4,
      "verdicts": {"excellent": 2, "good": 2, "mixed": 0, "poor": 0}
    },
    "youtube": {
      "total": 2,
      "evaluated": 2,
      "verdicts": {"excellent": 0, "good": 1, "mixed": 0, "poor": 1}
    }
  },
  "files_evaluated": [
    {"file": "file_name, "verdict": "excellent" | "good" | "mixed" | "poor"},
  ],
  "systematic_issues": {
    "missing_rationale_fields": {
      "description": "All files systematically missing required rationale fields: node_rationale, edge_rationale, edge_confidence_rationale, intervention_lifecycle_rationale, intervention_maturity_rationale",
      "severity": "BLOCKER",
      "affected_files": 100,
      "percentage": "100%"
    },
    "source_relevance_issues": {
      "description": "Some files contain content not directly related to AI safety",
      "severity": "MAJOR",
      "affected_files": 5,
      "examples": [
        "blogs__b9135d383a7ba95841ea9bdf413ebbb1.json - Medieval university administration",
        "eaforum__35de7ddfcc016c117270162770b6826f.json - Mental health personal experience",
        "lesswrong__2dc3e51b39a27bd73a81ce22f49bfb91.json - Philosophy of history",
        "lesswrong__4b552d846b9fef1a55626b58da67c29e.json - Bayesian epistemology fundamentals",
        "youtube__bc771dce09c4bac1d7baf6cb6ea984c4.json - Stoicism philosophy"
      ]
    }
  },
  "recommendations": {
    "high_priority": [
      "Fix extractor to always include required rationale fields - this is causing BLOCKER issues in 100% of files",
      "Review and potentially exclude the 5 non-AI-safety sources from the knowledge graph",
      "Add validation step to flag sources with low AI safety relevance"
    ],
    "medium_priority": [
      "Implement edge direction consistency checks",
      "Add confidence calibration for heavily-inferred content",
      "Create clearer guidelines for translating general rationality content to AI safety context"
    ]
  },
  "statistics": {
    "approve_rate": "69%",
    "approve_with_fixes_rate": "26%",
    "reject_rate": "5%"
  },
  "conclusion": "Conclusion here"
}


This is the prompt given to the extraction model for reference:
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