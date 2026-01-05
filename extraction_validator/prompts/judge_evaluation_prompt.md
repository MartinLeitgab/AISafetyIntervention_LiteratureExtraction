For each of the .json files in this folder, look at the original text and the judge output, especially the proposed fixes. Do they proposed fixed fix the problems? Are the added nodes necessary? Other questions like that. Compile your results into json format and save your results per file alongside the original file. Then at the end produce a summary_results.json that summarizes each of the stats of the files you created. Don't make a python script to do this, because it requires judgement to determine if these changes are correct, your judgement. You must be the judge.


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
      "excellent": 6,
      "good": 63,
      "mixed": 26,
      "poor": 5
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
    {"file": "aisafety.info__ebca764bf2f76577414efeba418a9471.json", "verdict": "good"},
    {"file": "aisafety.info__ffdb6214adf77c3aea4a4a7eb69efdf1.json", "verdict": "good"},
    {"file": "alignmentforum__02162c14d93d9d532142c0d358770b98.json", "verdict": "good"},
    {"file": "alignmentforum__027e6792b8d8c40ca0c6426714e36bb9.json", "verdict": "good"},
    {"file": "alignmentforum__095363af842e6a5d064fa0d26169641d.json", "verdict": "good"},
    {"file": "alignmentforum__0a217579b12f8d8556dbb5c3d905bf71.json", "verdict": "good"},
    {"file": "alignmentforum__323ce3592e2c04b495c096fddbdeef54.json", "verdict": "excellent"},
    {"file": "alignmentforum__43648ab491d45709537b7724a98b6a04.json", "verdict": "good"},
    {"file": "alignmentforum__519674794af0138b515f38ae8597dc3e.json", "verdict": "good"},
    {"file": "alignmentforum__5c9bb6829c48a3243e480619fc901c27.json", "verdict": "good"},
    {"file": "alignmentforum__6a445b2e04ed4a58f23c17d9a70b46b1.json", "verdict": "good"},
    {"file": "alignmentforum__793e84aad000bfb1541c899bc0298a9f.json", "verdict": "mixed"},
    {"file": "alignmentforum__7aa637006027ee604a29c0cdaf75b1fe.json", "verdict": "good"},
    {"file": "alignmentforum__83da468d940d7ff893d344ab7e37e0b5.json", "verdict": "good"},
    {"file": "alignmentforum__861d710315a821383ea7095621403834.json", "verdict": "mixed"},
    {"file": "alignmentforum__a21422e2cd97e530d627f3ad46f508ec.json", "verdict": "excellent"},
    {"file": "alignmentforum__b500e397b3445b037f426617291f4f9e.json", "verdict": "good"},
    {"file": "alignmentforum__ceb3105e38ad3dbe88d80a0ac65bcea9.json", "verdict": "mixed"},
    {"file": "alignmentforum__cfee12360b9369c20351362f10c416cc.json", "verdict": "good"},
    {"file": "alignmentforum__d21884707c3727c08be06de927cc5f2b.json", "verdict": "mixed"},
    {"file": "alignmentforum__f5c80eb22cac0f8a97b23b64aeb2953c.json", "verdict": "mixed"},
    {"file": "alignmentforum__f902c839536a971c8822284b5db37489.json", "verdict": "good"},
    {"file": "alignmentforum__fcac709693663bd332778a18e0cad6f5.json", "verdict": "poor"},
    {"file": "arbital__218bd258a3a481419afe5a018f11af23.json", "verdict": "good"},
    {"file": "arbital__25c7f517afdca99579b9ab3057048519.json", "verdict": "good"},
    {"file": "arbital__2d6655831374be979d66edb62deabcdb.json", "verdict": "mixed"},
    {"file": "arbital__6a5e818eeaab6c05f38a46ae10f53d28.json", "verdict": "good"},
    {"file": "arbital__97b4dfdc1ecf05800f5d070f45f39d8c.json", "verdict": "mixed"},
    {"file": "arbital__cd0edd240c32362ae487191097ea0579.json", "verdict": "good"},
    {"file": "arxiv__288011b7096ca1a2e2e7519fe780c1d5.json", "verdict": "good"},
    {"file": "arxiv__411d1d6df2f587c3ae7ee32c64885b47.json", "verdict": "good"},
    {"file": "arxiv__4bc60dd8d17016f476c387b28ef04b6d.json", "verdict": "good"},
    {"file": "arxiv__94ab1f40803fbccc68e4fa697e27edbc.json", "verdict": "good"},
    {"file": "arxiv__99cdcef67c70f9a8e03165eeee68a3f4.json", "verdict": "good"},
    {"file": "arxiv__9e7b712eda07897bf5c89fb1361eb665.json", "verdict": "mixed"},
    {"file": "arxiv__a5e77e7f5ec7af3978d715a4300f333e.json", "verdict": "excellent"},
    {"file": "arxiv__aaa9bc82902a3020015f762832e63058.json", "verdict": "good"},
    {"file": "arxiv__e7c9368771751d5dc8acc6751147d114.json", "verdict": "good"},
    {"file": "arxiv__ea675f7cbbe3fc9bb2c73591fc897efc.json", "verdict": "mixed"},
    {"file": "arxiv__efd0e0eb1643960d4d09e0ed59f996ec.json", "verdict": "good"},
    {"file": "blogs__1ae39c70b786a55a3148dc579b8ad707.json", "verdict": "good"},
    {"file": "blogs__267847230852557f6e0d50444fe4ea48.json", "verdict": "good"},
    {"file": "blogs__2987138bfabf5e002bc32ecddb1fd90b.json", "verdict": "good"},
    {"file": "blogs__36120ebd557e460a3dcba485c5fa97fe.json", "verdict": "good"},
    {"file": "blogs__50659ce045281bb2a6d93b01148c3f99.json", "verdict": "good"},
    {"file": "blogs__81c9d8d4423fac24b235c0207e127960.json", "verdict": "mixed"},
    {"file": "blogs__8fabccb9c9ecb3a62ee2a5d0c30e66d0.json", "verdict": "good"},
    {"file": "blogs__94a432823e0715b91c18475bdcef2f5b.json", "verdict": "excellent"},
    {"file": "blogs__a73707b9a5c2982fde5cd2b0a1d5a3e5.json", "verdict": "good"},
    {"file": "blogs__a8548fa398208f216bed82174317835c.json", "verdict": "good"},
    {"file": "blogs__b9135d383a7ba95841ea9bdf413ebbb1.json", "verdict": "poor"},
    {"file": "blogs__d36e342ba961869c82f6076456baa980.json", "verdict": "mixed"},
    {"file": "blogs__e4695f3aaf1b119f1036930e2b2b5580.json", "verdict": "mixed"},
    {"file": "blogs__faa1226fe640de7a6e64e34afc011d7a.json", "verdict": "good"},
    {"file": "blogs__ffeb1e6b5c14e661f8e53a73ffdf445b.json", "verdict": "mixed"},
    {"file": "eaforum__0736ae14938d6f6c8e7be7f5f00a1537.json", "verdict": "good"},
    {"file": "eaforum__14c9ddec1516cb412b28b00afab719c0.json", "verdict": "good"},
    {"file": "eaforum__2ce01e48c7b79bfa9e2e29254e3c7159.json", "verdict": "good"},
    {"file": "eaforum__35de7ddfcc016c117270162770b6826f.json", "verdict": "poor"},
    {"file": "eaforum__37c7d6402b9e3bb68402a4f601245343.json", "verdict": "good"},
    {"file": "eaforum__67526220b0714403edda537da32c7fff.json", "verdict": "good"},
    {"file": "eaforum__ad8bbfbaba0039af3b5ae3a41d368522.json", "verdict": "mixed"},
    {"file": "eaforum__c6b78a8e52b59cd5f20a47a5583627ef.json", "verdict": "good"},
    {"file": "eaforum__db2ba6ceb347165edbd04b78cd5a502a.json", "verdict": "mixed"},
    {"file": "eaforum__f27b8518cea2b3dffc78bd677942993d.json", "verdict": "good"},
    {"file": "eaforum__f58b8431fd6b620238b6896ca4876cee.json", "verdict": "mixed"},
    {"file": "lesswrong__0112814813e04493c14403b0042b30e1.json", "verdict": "good"},
    {"file": "lesswrong__15c3cbeded8804737dee3ebadc77845e.json", "verdict": "good"},
    {"file": "lesswrong__1ae7f37f1935dabe93de4748d6b90ff4.json", "verdict": "good"},
    {"file": "lesswrong__24d8be9adee47a7bb7d625eea16712e9.json", "verdict": "mixed"},
    {"file": "lesswrong__292c4eb3196207d1e1058704084aa8c8.json", "verdict": "good"},
    {"file": "lesswrong__2abdaa4fe424ab8d8d1b42bc74ab7ad6.json", "verdict": "good"},
    {"file": "lesswrong__2dc3e51b39a27bd73a81ce22f49bfb91.json", "verdict": "poor"},
    {"file": "lesswrong__3425ddd596beb73f1c1184eef75c7042.json", "verdict": "good"},
    {"file": "lesswrong__4b552d846b9fef1a55626b58da67c29e.json", "verdict": "poor"},
    {"file": "lesswrong__4c4fc0f646d9c809f61fc16eacc6fb68.json", "verdict": "mixed"},
    {"file": "lesswrong__4eff454bda809e5b14059e0f270e2d81.json", "verdict": "good"},
    {"file": "lesswrong__53c509ef49c3ffe35a6c1aeba28b6f73.json", "verdict": "mixed"},
    {"file": "lesswrong__575725fe6b4cea8c9fcb40d5392915d0.json", "verdict": "good"},
    {"file": "lesswrong__620e72ce09fce0b40146cb135ec1d698.json", "verdict": "mixed"},
    {"file": "lesswrong__6674da358a41e8168b9483f3b98abe2d.json", "verdict": "good"},
    {"file": "lesswrong__745cc9d67b5712356e8574c2be963f11.json", "verdict": "good"},
    {"file": "lesswrong__8375f98a7f2caa6681fa0573cc0e0975.json", "verdict": "good"},
    {"file": "lesswrong__8891844ebd5574a738714be61d792c0d.json", "verdict": "good"},
    {"file": "lesswrong__90d0dd77da011447879b58431034b685.json", "verdict": "good"},
    {"file": "lesswrong__b1e89738a76a56f90b0c38941ec9ff88.json", "verdict": "good"},
    {"file": "lesswrong__bb175b37c218bcef97e791daeabf7582.json", "verdict": "good"},
    {"file": "lesswrong__d15bf2e42ce3ca2c1ac4635f914e2903.json", "verdict": "good"},
    {"file": "lesswrong__d831e204dc8491f9b1f5795c5d7ed94f.json", "verdict": "mixed"},
    {"file": "lesswrong__eb4a824a8c34993ee38a7cb583d09aed.json", "verdict": "good"},
    {"file": "lesswrong__f334ae393d2e6c5ac66a21aa7a528d46.json", "verdict": "good"},
    {"file": "lesswrong__f3d9c756283d0252bd5e605b05cdf33e.json", "verdict": "mixed"},
    {"file": "lesswrong__f4a69a1de2e77201d9e2a4d0ed74510b.json", "verdict": "mixed"},
    {"file": "lesswrong__fb96af8834064a15c400a38657c3d63e.json", "verdict": "mixed"},
    {"file": "special_docs__035cac34995159d820cd0302cca6a491.json", "verdict": "good"},
    {"file": "special_docs__21f064ab9f06b815067d3f3d432dc549.json", "verdict": "excellent"},
    {"file": "special_docs__5756ff12431150bf1f973a4a1fd51172.json", "verdict": "good"},
    {"file": "special_docs__c76fd4181b98abb64650d6ca16be3233.json", "verdict": "good"},
    {"file": "youtube__1c150dd085a28e83151de32b786e6d3b.json", "verdict": "good"},
    {"file": "youtube__bc771dce09c4bac1d7baf6cb6ea984c4.json", "verdict": "poor"}
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
  "conclusion": "All 100 evaluations complete. The extraction pipeline produces generally good quality knowledge graph content (69% directly approvable, 95% salvageable). The dominant issue is systematic schema non-compliance with missing rationale fields - this is a pipeline configuration issue rather than extraction quality problem. A small number (5%) of sources have questionable AI safety relevance and should be filtered. Notable strengths include excellent handling of technical AI safety papers (arxiv, special_docs) and alignment forum content."
}


This is the prompt given to the judge model for reference:

You are KG-Judge, a precise and rigorous auditor for knowledge graphs. 

Your task is to validate this knowledge graph extraction against the source text and return a structured validation report.

DATA_SOURCE:
{original_text}

ORIGINAL_PROMPT:
{PROMPT_EXTRACT}

EXTRACTED_KNOWLEDGE_GRAPH:
{kg_output.model_dump_json(indent=2)}

SCHEMA_REQUIREMENTS:
- Nodes must have: name, aliases (2-3 items but not a BLOCKER), type (concept|intervention), description (1-2 sentences amount of sentences not a BLOCKER)
- If type=concept: must have concept_category, must not have intervention_lifecycle or intervention_maturity
- If type=intervention: must have intervention_lifecycle (1-6) and intervention_maturity (1-4), must not have concept_category
- Edges must have: type, source_node, target_node, description, edge_confidence (1-5)
- All node names referenced in edges must exist as nodes

VALIDATION_TASKS:
1. Check JSON structure and schema compliance
2. Verify all referenced nodes exist (referential integrity)
3. Identify orphaned nodes (not connected to any edges)
4. Find duplicate nodes/edges that should be merged
5. Check if extracted knowledge matches source text evidence
6. Assess coverage - are important edges from source missing?
7. Propose specific fixes for any issues found
8. Output a decision on overall validity, taking into account the proposed fixes. (Ie is it valid if fixes are applied?)

Return your analysis in this EXACT JSON format:
{{
  "validation_report": {{
    "schema_check": [
      {{ "severity": "BLOCKER|MAJOR|MINOR|STYLE", "issue": "description", "where": "path.to.field", "suggestion": "fix suggestion" }}
    ],
    "referential_check": [
      {{ "severity": "BLOCKER|MAJOR|MINOR", "issue": "description", "names": ["related node name 1","related node name 2"] }}
    ],
    "orphans": [
      {{ "node_name": "node name", "reason": "explanation", "suggested_fix": "what to do" }}
    ],
    "duplicates": [
      {{ "kind": "node|edge", "names": ["duplicate name 1","duplicate name 2"], "merge_strategy": "keep X, merge props, retarget edges" }}
    ],
    "rationale_mismatches": [
      {{ "issue": "description", "evidence": "exact quote from DATA_SOURCE", "fix": "suggested fix" }}
    ],
    "coverage": {{
      "expected_edges_from_source": [
        {{
          "title": "edge name",
          "evidence": "quote from source or list of quotes",
          "status": "covered|partially_covered|missing",
          "expected_source_node_name": ["expected source node name"],
          "expected_target_node_name": ["expected target node name"]
        }}
      ]
    }}
  }},
  "proposed_fixes": {{
    "add_nodes": [
      {{
        "type": "concept|intervention",
        "name": "node name",
        "edges": [
          {{
            "type": "edge type",
            "target_node": "target node name",
            "description": "edge description",
            "edge_confidence": 1-5
          }}
        ],
        "intervention_lifecycle": 1-6,
        "intervention_maturity": 1-4
      }}
    ],
    "merges": [
      {{ "new_node_name": "name of the new node", 
      "nodes_to_merge": ["node name 1","node name 2"] }}
    ],
    "deletions": [
      {{ "node_name": "name to delete", "reason": "explanation" }}
    ],
    "edge_deletions": [
      {{ "source_node_name": "source node name", "target_node_name": "target node name", "reason": "explanation" }}
    ],
    "change_node_fields": [
      {{ "node_name": "node name", "field": "field name", "json_new_value": "new value as a json string", "reason": "explanation" }}
    ]
  }},
  "decision": {{
    "summary": "One-paragraph executive summary of validation results"
    "is_valid_json": true/false,
    "has_blockers": false/false,
    "valid_and_mergeable_after_fixes": true/false,
    "flag_underperformance": true/false,
  
  }},
  "rationale_record": {{
    "method": "systematic_validation",
    "notes": [
      "Key validation decisions with specific citations to DATA_SOURCE"
    ]
  }}
}}