# APPENDIX B: Phase 2 Analysis Substep Mapping to Workshop Goals

**Document Version:** 1.0  
**Companion to:** Phase 2 Comprehensive Clustering Analysis Plan v2.0  
**Last Updated:** February 2026

---

## PART 1: SUBSTEP-TO-WORKSHOP GOAL MAPPING

This appendix maps each of the 34 Phase 2 analysis substeps to the 9 workshop-critical goals identified for publication acceptance.

### Workshop Critical Goals (Reference)

1. **⭐ Cross-Threshold Stability (ARI)** – Demonstrates robustness of mechanisms across similarity thresholds
2. **⭐ EDGE-Validation Rate** – Shows grounding in single-source literature evidence vs. similarity artifacts
3. **⭐ Algorithm Comparison** – Method justification with performance metrics
4. **⭐ Optimal Config Selection with Full Transparency** – Multi-criteria scoring showing all base data
5. **⭐ Path Length vs. Quality Relationship** – Validates ≥5 hop claim for mechanistic detail
6. **⭐ Mechanism Taxonomy with Manual Validation** – Core deliverable (40-60 families) with coherence scoring
7. **⭐ Intervention Hub Quality Assessment** – Addresses artifact concerns with EDGE-validation and source diversity
8. **⭐ Temporal Coverage Analysis** – Shows scope of literature (publication dates per cluster)
9. **⭐ Bias Documentation** – Critical for limitations section

---

### Substep Mapping Table

| Substep | Analysis Name | Primary Goal(s) | Secondary Goal(s) | Step |
|---------|---------------|-----------------|-------------------|------|
| **1** | Silhouette Score | Goal 3 (Algorithm) | Goal 4 (Selection) | Step 2 |
| **2** | Cluster Size Distribution | Goal 6 (Taxonomy) | Goal 4 (Selection) | Step 2 |
| **3** | Cluster Cohesion Metrics | Goal 3 (Algorithm) | Goal 4 (Selection) | Step 2 |
| **4** | EDGE Validation Rate | **Goal 2** (EDGE) | Goal 4 (Selection) | Step 2 |
| **5** | EDGE Purity per Cluster | **Goal 2** (EDGE) | Goal 6 (Taxonomy) | Step 2 |
| **6** | Source Diversity per Cluster | Goal 7 (Hub Quality) | **Goal 2** (EDGE) | Step 2 |
| **7** | Cross-Threshold Stability (ARI) | **Goal 1** (Stability) | Goal 4 (Selection) | Step 2 |
| **8** | Node Migration Analysis | **Goal 1** (Stability) | Goal 9 (Bias) | Step 2 |
| **9** | Mode Impact on Clustering Quality | Goal 9 (Bias) | Goal 4 (Selection) | Step 2 |
| **10** | Multi-Risk Cluster Characterization | Goal 6 (Taxonomy) | Goal 9 (Bias) | Step 2 |
| **11** | Risk Diversity per Configuration | Goal 6 (Taxonomy) | Goal 9 (Bias) | Step 2 |
| **12** | Risk→Intervention Connectivity | Goal 6 (Taxonomy) | - | Step 2 |
| **13** | Intervention Maturity Distribution | Goal 6 (Taxonomy) | Goal 9 (Bias) | Step 2 |
| **14** | Intervention Hub Quality Assessment | **Goal 7** (Hub Quality) | **Goal 2** (EDGE) | Step 2 |
| **15** | Category-Specific Mechanism Families | Goal 6 (Taxonomy) | - | Step 2 |
| **16** | Mechanism Transfer Enablers | Goal 6 (Taxonomy) | - | Step 2 |
| **17** | Risk→Concept→Intervention Triplets | Goal 6 (Taxonomy) | Goal 4 (Selection) | Step 4 |
| **18** | Exemplar Path Extraction | Goal 6 (Taxonomy) | **Goal 5** (Path Length) | Step 4 |
| **19** | Pathway Signature Validation | **Goal 5** (Path Length) | Goal 9 (Bias) | Step 2 |
| **20** | Algorithm Performance Comparison | **Goal 3** (Algorithm) | Goal 4 (Selection) | Step 3 |
| **21** | Edge Threshold Sensitivity Test | **Goal 1** (Stability) | Goal 4 (Selection) | Step 3 |
| **22** | EDGE-Only Configuration Validation | **Goal 2** (EDGE) | Goal 4 (Selection) | Step 3 |
| **23** | Multi-Criteria Scoring Transparency | **Goal 4** (Selection) | Goal 9 (Bias) | Step 3 |
| **24** | Held-Out Test Set Validation | Goal 6 (Taxonomy) | Goal 3 (Algorithm) | Step 3 |
| **25** | EDGE Subgraph Consistency Check | **Goal 2** (EDGE) | Goal 6 (Taxonomy) | Step 3 |
| **26** | Cluster Naming & Manual Validation | **Goal 6** (Taxonomy) | - | Step 4 |
| **27** | Exemplar Quality Assessment | Goal 6 (Taxonomy) | **Goal 5** (Path Length) | Step 4 |
| **28** | Risk-Intervention Connectivity Matrix | Goal 6 (Taxonomy) | - | Step 4 |
| **29** | Pathway Length vs Cluster Quality | **Goal 5** (Path Length) | Goal 9 (Bias) | Step 2 |
| **30** | Temporal Coverage Analysis | **Goal 8** (Temporal) | Goal 9 (Bias) | Step 2 |
| **31** | Intervention Lifecycle Distribution | Goal 6 (Taxonomy) | Goal 9 (Bias) | Step 2 |
| **32** | Inter-Rater Agreement (Manual) | **Goal 6** (Taxonomy) | - | Step 4 |
| **33** | Path→Prompt Engineering | Goal 6 (Taxonomy) | - | Step 4 |
| **34** | Expected Output Formatting | Goal 6 (Taxonomy) | - | Step 4 |

---

### Coverage Analysis by Workshop Goal

| Workshop Goal | Primary Substeps | Secondary Substeps | Total Coverage |
|---------------|------------------|--------------------|----------------|
| **Goal 1: Cross-Threshold Stability** | 7, 8, 21 | - | 3 substeps |
| **Goal 2: EDGE-Validation Rate** | 4, 5, 22, 25 | 6, 14 | 6 substeps |
| **Goal 3: Algorithm Comparison** | 1, 3, 20 | 24 | 4 substeps |
| **Goal 4: Optimal Config Selection** | 23 | 1, 2, 7, 9, 17, 20, 21, 22 | 9 substeps |
| **Goal 5: Path Length vs Quality** | 19, 29 | 18, 27 | 4 substeps |
| **Goal 6: Mechanism Taxonomy** | 2, 10, 11, 12, 13, 15, 16, 17, 18, 24, 26, 27, 28, 31, 32, 33, 34 | 5 | 18 substeps |
| **Goal 7: Hub Quality Assessment** | 14 | 6 | 2 substeps |
| **Goal 8: Temporal Coverage** | 30 | - | 1 substep |
| **Goal 9: Bias Documentation** | 9 | 8, 10, 11, 13, 19, 23, 29, 30, 31 | 10 substeps |

**Key Observations:**
- Goal 6 (Mechanism Taxonomy) is the most comprehensive, involving 18 substeps – reflects core deliverable
- Goal 2 (EDGE-Validation) has strong coverage (6 substeps) – critical for literature grounding
- Goal 4 (Optimal Selection) touches 9 substeps – emphasizes transparency requirement
- Goal 8 (Temporal Coverage) has minimal direct analysis – secondary priority
- All 9 workshop goals are addressed by at least 1 primary substep

---

## PART 2: ANALYSIS SUBSTEPS BY STEP (DETAILED TRACEABILITY)

This section provides comprehensive traceability tables organized by analysis step, ordered by criticality within each step.

**Table Format:**
- Three tables: Step 2, Step 3, Step 4
- Ordered by: Criticality (CRITICAL → ESSENTIAL → ENRICHMENT)
- **Criticality levels:** 
  - **CRITICAL:** Must-have for workshop acceptance
  - **ESSENTIAL:** Required for complete story
  - **ENRICHMENT:** Strengthens paper but not mandatory

---

### STEP 2: CORE METRICS & STABILITY ANALYSIS

**19 substeps total:** 4 CRITICAL, 8 ESSENTIAL, 7 ENRICHMENT

| # | Substep Name | Theme | Goals | Question Answered | Output Files/Plots | Criticality |
|---|--------------|-------|-------|-------------------|-------------------|-------------|
| **7** | Cross-Threshold Stability (ARI) | Method Rigor | **Goal 1**: Cross-Threshold Stability | Do identified mechanisms persist across different similarity thresholds (0.8→0.85→0.9→0.95→EDGE-only)? What is the ARI between adjacent threshold pairs? | `cross_threshold_stability_heatmap.png` (Plot 5)<br>`stability_metrics.csv` | **CRITICAL** |
| **4** | EDGE Validation Rate | Lit Grounding | **Goal 2**: EDGE-Validation Rate<br>Goal 4: Optimal Config Selection | What % of clusters contain ≥1 node from EDGE-only pathways? Do we meet the >60% literature grounding threshold? | `edge_validation_breakdown.png` (Plot 6)<br>`all_cluster_metrics.csv` | **CRITICAL** |
| **14** | Intervention Hub Quality Assessment | Lit Grounding | **Goal 7**: Hub Quality Assessment<br>**Goal 2**: EDGE-Validation Rate | For top-20 intervention hubs: What is EDGE-only degree vs total degree? How many unique sources cite each hub? How many risk categories does each hub address? Are hubs genuine convergence points or similarity artifacts? | `hub_quality_scatter.png` (Plot 7)<br>`hub_quality_metrics.csv` | **CRITICAL** |
| **19** | Pathway Signature Validation | Interpretability | **Goal 5**: Path Length vs Quality<br>Goal 9: Bias Documentation | Do pathways follow coherent category sequences (Risk → Problem Analysis → Theory → Design → Implementation → Validation → Intervention)? What is the length distribution (1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13+ hop bins)? Does the ≥5 hop claim hold for sufficient mechanistic detail? | `path_length_sensitivity.png` (Plot 4)<br>`pathway_signature_stats.csv` | **CRITICAL** |
| **1** | Silhouette Score | Method Rigor | **Goal 3**: Algorithm Comparison<br>Goal 4: Optimal Config Selection | How well-separated are clusters in embedding space for each algorithm (Agglomerative/Louvain/HDBSCAN)? Which configs achieve silhouette >0.3? | `cluster_quality_overview.png` (Plot 1)<br>`silhouette_distribution.png` (Plot 2)<br>`all_cluster_metrics.csv` | **ESSENTIAL** |
| **3** | Cluster Cohesion Metrics | Method Rigor | **Goal 3**: Algorithm Comparison<br>Goal 4: Optimal Config Selection | What are intra-cluster distances vs inter-cluster separation ratios? How does this vary by algorithm? | `all_cluster_metrics.csv`<br>`cohesion_analysis.csv` | **ESSENTIAL** |
| **8** | Node Migration Analysis | Method Rigor / Bias | **Goal 1**: Cross-Threshold Stability<br>Goal 9: Bias Documentation | Which nodes are "core" (stable across all thresholds) vs "peripheral" (migrate frequently)? What % of nodes migrate at each threshold transition? | `node_migration_heatmap.png` (Plot 20)<br>`migration_stats.csv` | **ESSENTIAL** |
| **5** | EDGE Purity per Cluster | Lit Grounding | **Goal 2**: EDGE-Validation Rate<br>Goal 6: Mechanism Taxonomy | What is the distribution of EDGE purity across clusters? How many "gold standard" clusters have >80% EDGE membership? | `edge_validation_breakdown.png` (Plot 6)<br>`cluster_edge_purity.csv` | **ESSENTIAL** |
| **6** | Source Diversity per Cluster | Lit Grounding | **Goal 7**: Hub Quality Assessment<br>**Goal 2**: EDGE-Validation Rate | How many unique source documents contribute to each cluster? Are there single-source clusters (potential extraction artifacts)? Do clusters meet ≥3 source threshold? | `source_diversity_heatmap.png` (Plot 18)<br>`cluster_source_stats.csv` | **ESSENTIAL** |
| **30** | Temporal Coverage Analysis | Lit Grounding / Bias | **Goal 8**: Temporal Coverage<br>Goal 9: Bias Documentation | What is the distribution of publication dates per cluster? Do clusters span multiple years or concentrate in specific periods? What is the training cutoff impact (April 2024)? | `temporal_coverage.png` (Plot 9)<br>`temporal_stats.csv` | **ESSENTIAL** |
| **2** | Cluster Size Distribution | Mechanism | **Goal 6**: Mechanism Taxonomy<br>Goal 4: Optimal Config Selection | What is the distribution of cluster sizes (min, max, mean, median, std)? Do configs produce 40-60 interpretable clusters or suffer from over-fragmentation (<5 members/cluster) or over-aggregation (>500 members)? | `cluster_quality_overview.png` (Plot 1)<br>`all_cluster_metrics.csv` | **ESSENTIAL** |
| **29** | Pathway Length vs Cluster Quality | Interpretability / Bias | **Goal 5**: Path Length vs Quality<br>Goal 9: Bias Documentation | How does silhouette score vary across pathway length bins? Do longer paths produce better-quality clusters? Is there evidence of the 6-hop peak from extraction prompt bias? | `path_length_sensitivity.png` (Plot 4)<br>`silhouette_distribution.png` (Plot 2) | **ESSENTIAL** |
| **10** | Multi-Risk Cluster Characterization | Mechanism / Bias | **Goal 6**: Mechanism Taxonomy<br>Goal 9: Bias Documentation | In unconstrained mode, what % of clusters contain multiple risk categories? Are multi-risk clusters mechanistically coherent or graph artifacts? | `multi_risk_cluster_analysis.csv`<br>Manual inspection notes | **ENRICHMENT** |
| **11** | Risk Diversity per Configuration | Mechanism / Bias | **Goal 6**: Mechanism Taxonomy<br>Goal 9: Bias Documentation | How many unique risk categories are represented across clusters? Is distribution balanced or skewed? Which risks serve as cluster exemplars? | `umap_risks.png` (Plot 15)<br>`risk_diversity_stats.csv` | **ENRICHMENT** |
| **13** | Intervention Maturity Distribution | Mechanism / Bias | **Goal 6**: Mechanism Taxonomy<br>Goal 9: Bias Documentation | What is the maturity 3 vs 4 distribution per cluster? What is the intervention lifecycle stage distribution (design/training/deployment) per cluster? Which lifecycle stage dominates each cluster? | `lifecycle_distribution.png` (Plot 8)<br>`maturity_distribution_heatmap.png` (Plot 19) | **ENRICHMENT** |
| **15** | Category-Specific Mechanism Families | Mechanism | **Goal 6**: Mechanism Taxonomy | For each concept category (problem_analysis, theoretical_foundation, etc.), what transferable mechanisms emerge? Which concepts bridge multiple categories? | `umap_concepts.png` (Plot 17)<br>`category_mechanism_families.csv` | **ENRICHMENT** |
| **16** | Mechanism Transfer Enablers (Betweenness) | Mechanism | **Goal 6**: Mechanism Taxonomy | Which high-betweenness concepts bridge multiple risk-intervention pairs? What are the top-20 transfer enablers per concept category? How many distinct risk→intervention paths pass through each? | `mechanism_transfer_betweenness.png` (Plot 10)<br>`transfer_enablers.csv` | **ENRICHMENT** |
| **31** | Intervention Lifecycle Distribution | Mechanism / Bias | **Goal 6**: Mechanism Taxonomy<br>Goal 9: Bias Documentation | How do interventions distribute across lifecycle stages within each cluster? Is there clustering by development stage? | `lifecycle_distribution.png` (Plot 8)<br>`maturity_distribution_heatmap.png` (Plot 19) | **ENRICHMENT** |
| **9** | Mode Impact on Clustering Quality | Interpretability / Bias | Goal 9: Bias Documentation<br>Goal 4: Optimal Config Selection | How do pathway constraints (unconstrained vs single-risk vs monotonic vs both) affect cluster count, silhouette score, and EDGE-validation rate? Which mode produces most interpretable results? | `edge_density_heatmap.png` (Plot 21)<br>`mode_stability_heatmap.png` (Plot 22)<br>`mode_comparison_stats.csv` | **ENRICHMENT** |

**Step 2 Key Deliverables:**
- ARI stability matrix (target: >0.7 for adjacent thresholds)
- EDGE validation rate >60% demonstration
- Hub quality assessment (top-20 interventions)
- Path length distribution with quality correlation
- Comprehensive bias documentation (10 biases identified)

---

### STEP 3: VALIDATION & OPTIMAL CONFIGURATION SELECTION

**6 substeps total:** 2 CRITICAL, 2 ESSENTIAL, 2 ENRICHMENT

| # | Substep Name | Theme | Goals | Question Answered | Output Files/Plots | Criticality |
|---|--------------|-------|-------|-------------------|-------------------|-------------|
| **20** | Algorithm Performance Comparison | Method Rigor | **Goal 3**: Algorithm Comparison<br>Goal 4: Optimal Config Selection | Which algorithm (Agglomerative/Louvain/HDBSCAN) produces highest quality clusters? How do they compare on silhouette, EDGE%, cluster count? | `algorithm_performance.png` (Plot 3)<br>`algorithm_comparison.csv` | **CRITICAL** |
| **23** | Multi-Criteria Scoring Transparency | Method Rigor / Bias | **Goal 4**: Optimal Config Selection<br>Goal 9: Bias Documentation | What are the complete scores for all 160 configs across all criteria? Can reviewers see the base data behind selection decisions? | `optimal_configs_ranked.csv`<br>`multi_criteria_parallel.png` (Plot 14)<br>`selection_justification.md` | **CRITICAL** |
| **21** | Edge Threshold Sensitivity Test | Method Rigor | **Goal 1**: Cross-Threshold Stability<br>Goal 4: Optimal Config Selection | How sensitive is clustering quality to edge threshold choice? Is there a "sweet spot" threshold? | `validation_test_results.json`<br>`threshold_sensitivity_analysis.csv` | **ESSENTIAL** |
| **22** | EDGE-Only Configuration Validation | Lit Grounding | **Goal 2**: EDGE-Validation Rate<br>Goal 4: Optimal Config Selection | How does clustering on EDGE-only pathways compare to SIM≥0.95? Does EDGE-only produce coherent mechanisms despite smaller dataset? | `validation_test_results.json`<br>`edge_only_comparison.csv` | **ESSENTIAL** |
| **24** | Held-Out Test Set Validation | Method Rigor | **Goal 3**: Algorithm Comparison<br>Goal 6: Mechanism Taxonomy | Do cluster assignments generalize to held-out pathways? What is the prediction accuracy? | `validation_test_results.json`<br>`holdout_validation_stats.csv` | **ENRICHMENT** |
| **25** | EDGE Subgraph Consistency Check | Lit Grounding | **Goal 2**: EDGE-Validation Rate<br>Goal 6: Mechanism Taxonomy | Do nodes clustered together in full graph remain together when using only EDGE connections? What is the consistency rate? | `validation_test_results.json`<br>`edge_consistency_stats.csv` | **ENRICHMENT** |

**Step 3 Key Deliverables:**
- Algorithm comparison table with performance metrics
- Complete ranked list of 160 configs with all base data
- Final optimal configuration selection with justification
- Validation test results (held-out, EDGE consistency)

---

### STEP 4: TAXONOMY CONSTRUCTION & NETWORK VISUALIZATION

**9 substeps total:** 1 CRITICAL, 0 ESSENTIAL, 8 ENRICHMENT

| # | Substep Name | Theme | Goals | Question Answered | Output Files/Plots | Criticality |
|---|--------------|-------|-------|-------------------|-------------------|-------------|
| **26** | Cluster Naming & Manual Validation | Mechanism | **Goal 6**: Mechanism Taxonomy | What are human-interpretable names for each of the 40-60 mechanism families? Do manual annotators agree these clusters represent coherent mechanisms (1-5 coherence scale)? | `mechanism_taxonomy_summary.csv`<br>`cluster_naming_validation.csv`<br>`mechanism_families.json` | **CRITICAL** |
| **32** | Inter-Rater Agreement (Manual) | Method Rigor | **Goal 6**: Mechanism Taxonomy | Do independent annotators agree on mechanism coherence? What is Fleiss' kappa for 10 risk + 10 intervention + 10 concept clusters? | `cluster_naming_validation.csv`<br>`inter_rater_agreement.json` | **ENRICHMENT** |
| **17** | Risk→Concept→Intervention Triplets | Mechanism | **Goal 6**: Mechanism Taxonomy<br>Goal 4: Optimal Config Selection | Using all_concepts clusters OR 6 individual concept categories, how many valid triplets connect risk→concept→intervention clusters? How many actual pathways support each triplet? | `cluster_level_network.png` (Plot 11)<br>`sankey_risk_concept_intervention.png` (Plot 12)<br>`cluster_level_network_data.json` | **ENRICHMENT** |
| **18** | Exemplar Path Extraction | Mechanism | **Goal 6**: Mechanism Taxonomy<br>**Goal 5**: Path Length vs Quality | For each mechanism family, what is the exemplar pathway (node closest to cluster centroid)? Do synthetic paths (connecting exemplar nodes directly) preserve mechanism semantics vs actual pathways? | `representative_pathways.jsonl`<br>`exemplar_quality_comparison.csv` | **ENRICHMENT** |
| **27** | Exemplar Quality Assessment | Mechanism | **Goal 6**: Mechanism Taxonomy<br>**Goal 5**: Path Length vs Quality | How representative are exemplar nodes of their clusters? What is the average distance from exemplar to all cluster members? | `cluster_naming_validation.csv`<br>`exemplar_quality_stats.csv` | **ENRICHMENT** |
| **28** | Risk-Intervention Connectivity Matrix | Mechanism | **Goal 6**: Mechanism Taxonomy | Which risk clusters connect to which intervention clusters via pathways? What is the pathway count strength for each risk-intervention pair? | `risk_intervention_matrix.csv`<br>`connectivity_matrix_heatmap.png` (Plot 13) | **ENRICHMENT** |
| **33** | Path→Prompt Engineering | Mechanism | **Goal 6**: Mechanism Taxonomy | Can pathways be converted to natural language descriptions for LLM simulation? Do prompts comprehensibly describe: risk → mechanism steps → intervention? | `simulation_prompts_sample.json`<br>Sample prompt validation | **ENRICHMENT** |
| **34** | Expected Output Formatting | Mechanism | **Goal 6**: Mechanism Taxonomy | How will simulation results be aggregated per mechanism family? What metrics will be reported (Δ instrumental goals, Δ pro-human goals, Δ anti-human goals with 95% CI)? | `simulation_prompts_sample.json`<br>Output format specification | **ENRICHMENT** |

**Note:** Substep #12 (Risk→Intervention Connectivity) is covered by #28.

**Step 4 Key Deliverables:**
- 40-60 mechanism families with manual validation
- Mechanism taxonomy summary with names, exemplars, coherence scores
- Representative pathways (500-1000) for simulation
- Risk-intervention connectivity matrix
- Cluster-level network visualization
- Simulation-ready prompt templates

---

### CRITICALITY SUMMARY ACROSS ALL STEPS

| Criticality | Step 2 | Step 3 | Step 4 | Total |
|-------------|--------|--------|--------|-------|
| **CRITICAL** | 4 | 2 | 1 | **7 substeps** |
| **ESSENTIAL** | 8 | 2 | 0 | **10 substeps** |
| **ENRICHMENT** | 7 | 2 | 8 | **17 substeps** |
| **Total** | **19** | **6** | **9** | **34 substeps** |

**Critical Path for Workshop (7 substeps, ~6.5 hours):**
- Step 2: #7 (ARI), #4 (EDGE%), #14 (Hub Quality), #19 (Path Length)
- Step 3: #20 (Algorithm), #23 (Selection Transparency)
- Step 4: #26 (Manual Validation)

**Essential for Complete Story (10 substeps, ~5.5 hours):**
- Step 2: #1, #3, #8, #5, #6, #30, #2, #29
- Step 3: #21, #22

**Enrichment Substeps (17 substeps, ~13.5 hours):**
- Step 2: #10, #11, #13, #15, #16, #31, #9
- Step 3: #24, #25
- Step 4: #32, #17, #18, #27, #28, #33, #34

---

---

## PART 3: WORKSHOP PAPER SECTION ALIGNMENT

### Methods Section (3.4 Mechanism Clustering)

**Primary Substeps:** 1, 2, 3, 7, 20, 23  
**Theme:** Methodological Rigor & Validation  
**Content:**
- Clustering algorithms (Agglomerative, Louvain, HDBSCAN) with hyperparameters
- Silhouette score calculation and interpretation
- Cross-threshold stability (ARI) methodology
- Multi-criteria scoring framework (transparency requirement)

---

### Methods Section (3.5 Validation Framework)

**Primary Substeps:** 4, 5, 6, 14, 22, 24, 25, 26, 32  
**Theme:** Literature Grounding & Evidence Quality  
**Content:**
- EDGE validation rate definition and calculation
- Hub quality assessment (3 metrics: EDGE%, source diversity, risk diversity)
- Manual annotation protocol (inter-rater agreement)
- Held-out test set procedure

---

### Results Section (4.5 Mechanism Taxonomy)

**Primary Substeps:** 2, 10, 11, 12, 15, 16, 17, 18, 26, 27, 28  
**Theme:** Mechanism Discovery & Taxonomy  
**Content:**
- 40-60 mechanism families identified
- Category-specific mechanism transfer enablers
- Risk-intervention connectivity patterns
- Exemplar pathway examples with quality scores

---

### Results Section (4.4 Threshold Optimization)

**Primary Substeps:** 7, 8, 9, 19, 21, 29  
**Theme:** Interpretability & Coherence  
**Content:**
- SIM≥0.85-0.9 identified as optimal (ARI stability + pathway volume)
- Path length vs quality relationship (≥5 hop minimum validated)
- Mode impact on clustering (unconstrained preserves diversity)

---

### Discussion Section (5. Limitations)

**Primary Substeps:** 8, 9, 10, 11, 13, 19, 23, 29, 30, 31  
**Theme:** Bias Documentation & Transparency  
**Content:**
- Path length bias (6-hop peak from extraction prompt)
- Category balance bias (prompt-enforced distribution)
- Maturity threshold bias (excludes 85% of interventions)
- Temporal coverage limitations (training cutoff April 2024)
- Clustering parameter sensitivity

---

## PART 4: PRIORITY MATRIX FOR IMPLEMENTATION

### Critical Path (Must-Have for Workshop Acceptance)

**Week 1 Priority:**
| Substep | Theme | Goal | Estimated Time | Step |
|---------|-------|------|----------------|------|
| #4 | Lit Grounding | Goal 2 ⭐ | 30 min | Step 2 |
| #7 | Method Rigor | Goal 1 ⭐ | 1 hour | Step 2 |
| #14 | Lit Grounding | Goal 7 ⭐ | 1 hour | Step 2 |
| #19 | Interpretability | Goal 5 ⭐ | 1 hour | Step 2 |
| #20 | Method Rigor | Goal 3 ⭐ | 1 hour | Step 3 |
| #23 | Method Rigor | Goal 4 ⭐ | 2 hours | Step 3 |
| **Total** | | | **6.5 hours** | |

---

### Essential (Required for Complete Story)

**Week 1 Priority:**
| Substep | Theme | Goal | Estimated Time | Step |
|---------|-------|------|----------------|------|
| #1, #3 | Method Rigor | Goal 3 | 1 hour | Step 2 |
| #2 | Mechanism | Goal 6 | 30 min | Step 2 |
| #5, #6 | Lit Grounding | Goal 2 | 1 hour | Step 2 |
| #8 | Method Rigor | Goal 1 | 1 hour | Step 2 |
| #21, #22 | Lit Grounding | Goal 2 | 1 hour | Step 3 |
| #29, #30 | Interpretability | Goal 5, 8 | 1 hour | Step 2 |
| **Total** | | | **5.5 hours** | |

---

### Enrichment (Strengthens Paper, Not Critical)

**Week 2 Priority:**
| Substep | Theme | Goal | Estimated Time | Step |
|---------|-------|------|----------------|------|
| #9, #10, #11, #13 | Mechanism | Goal 6 | 2 hours | Step 2 |
| #15, #16 | Mechanism | Goal 6 | 2 hours | Step 2 |
| #17, #18, #27, #28 | Mechanism | Goal 6 | 3 hours | Step 4 |
| #24, #25 | Lit Grounding | Goal 2 | 1 hour | Step 3 |
| #26, #32 | Mechanism | Goal 6 ⭐ | 4 hours (manual) | Step 4 |
| #31 | Mechanism | Goal 6 | 30 min | Step 2 |
| #33, #34 | Mechanism | Goal 6 | 1 hour | Step 4 |
| **Total** | | | **13.5 hours** | |

**Note:** Manual validation (#26, #32) can be performed in parallel with computational tasks

---

### Total Implementation Estimate

- **Critical Path:** 6.5 hours (computational)
- **Essential:** 5.5 hours (computational)
- **Enrichment:** 9.5 hours (computational) + 4 hours (manual)
- **Total Computational:** 21.5 hours
- **Total Manual:** 4 hours
- **Grand Total:** 25.5 hours

**Matches Plan Estimate:** 26 hours (18 computational + 8 manual validation/documentation)

---

## PART 5: MISSING SUBSTEP IDENTIFICATION

### Potential Gaps in Current Plan

**Gap 1: Error Rate Quantification**  
- **Issue:** No substep explicitly calculates extraction error rates (hallucinations, missing connections)
- **Relevant to:** Goal 9 (Bias), Goal 2 (EDGE Validation)
- **Recommendation:** Add substep for sampling pathways and LLM-based plausibility scoring
- **Priority:** MEDIUM (strengthens limitations section)

**Gap 2: Cluster Stability Under Bootstrapping**  
- **Issue:** No substep tests clustering stability via resampling
- **Relevant to:** Goal 1 (Stability), Goal 3 (Algorithm)
- **Recommendation:** Add substep for bootstrap ARI calculation
- **Priority:** LOW (ARI across thresholds already demonstrates stability)

**Gap 3: Cross-Category Mechanism Analysis**  
- **Issue:** Substep #15 mentions category-specific families but doesn't analyze cross-category patterns
- **Relevant to:** Goal 6 (Taxonomy)
- **Recommendation:** Add substep for identifying concepts bridging multiple categories
- **Priority:** LOW (already partially covered by #16 Betweenness Analysis)

**Gap 4: Simulation Readiness Validation**  
- **Issue:** Substeps #33-34 prepare for simulation but don't validate prompt→LLM output quality
- **Relevant to:** Goal 6 (Taxonomy)
- **Recommendation:** Add substep for LLM pilot testing on sample pathways
- **Priority:** MEDIUM (Phase 3 scope, but sample testing useful)

---

## APPENDIX: QUICK REFERENCE TABLES

### Substep-to-Step Mapping

| Step | Substeps | Count | Focus |
|------|----------|-------|-------|
| **Step 1** | Load & Parse | 0 | Data preparation (no analysis substeps) |
| **Step 2** | 1-16, 19, 29-31 | 19 | Core metrics & stability |
| **Step 3** | 20-25 | 6 | Validation & selection |
| **Step 4** | 17-18, 26-28, 32-34 | 9 | Taxonomy, network, visualization |
| **Total** | | **34** | |

---

### Goal-to-Theme Mapping

| Theme | Goals Covered | Substep Count |
|-------|---------------|---------------|
| **Methodological Rigor** | 1, 3, 4 | 9 |
| **Literature Grounding** | 2, 7, 8 | 7 |
| **Mechanism Discovery** | 6 | 15 |
| **Interpretability** | 5, 9 | 3 |
| **Bias Documentation** | 9 (cross-cutting) | 10 |

---

### Plot-to-Substep Mapping

| Plot # | Plot Name | Primary Substeps | Theme |
|--------|-----------|------------------|-------|
| 1 | Cluster Quality Overview | 1, 2 | Mechanism |
| 2 | Silhouette Distribution | 1 | Interpretability |
| 3 | Algorithm Performance | 20 | Method Rigor |
| 4 | Path Length Sensitivity | 29 | Interpretability |
| 5 | Cross-Threshold Stability | 7 | Method Rigor |
| 6 | EDGE Validation Breakdown | 4, 5 | Lit Grounding |
| 7 | Hub Quality Scatter | 14 | Lit Grounding |
| 8 | Lifecycle Distribution | 31 | Mechanism |
| 9 | Temporal Coverage | 30 | Lit Grounding |
| 10 | Mechanism Transfer Betweenness | 16 | Mechanism |
| 11 | Cluster Network | 17, 28 | Mechanism |
| 12 | Sankey Diagram | 17, 28 | Mechanism |
| 13 | Connectivity Heatmap | 28 | Mechanism |
| 14 | Multi-Criteria Parallel | 23 | Method Rigor |
| 15-17 | UMAP Projections | 11, 15 | Mechanism |
| 18 | Source Diversity Heatmap | 6 | Lit Grounding |
| 19 | Maturity Distribution Heatmap | 13 | Mechanism |
| 20 | Node Migration Heatmap | 8 | Bias |
| 21 | Edge Density Heatmap | 9 | Bias |
| 22 | Mode Stability Heatmap | 9 | Bias |

---

**END OF APPENDIX**

**Status:** Ready for integration with Phase 2 Comprehensive Analysis Plan  
**Next Action:** Use this mapping to prioritize substep implementation and structure workshop paper sections
