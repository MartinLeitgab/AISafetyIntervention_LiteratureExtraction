> 🔴 **SUPERSEDED 2026-06-10** by `paper_split_plan_2026_06_10.md` §3 (Paper A work items) + `paper_B_plan_2026_06_10.md` (Paper B work items).
> Of the 9 workshop-critical goals in this document, Goals 1-6 are now Paper B scope (the path-level substrate work). Goals 7, 9 partially apply to Paper A (hub-merge analysis within the frozen preprocessing + Limitations section). Goal 8 (temporal coverage) is reduced to an ARD-corpus-level pub-date appendix histogram in Paper A. **Cross-check against the Drive workshop-minimum list (file ID 1Z8f7MJuWIvjWvcWvPSe3_r3g32z5n9j7KcV9vNawh6I) is recorded in `paper_split_plan_2026_06_10.md` §4.**
> Do not use this document as a current planning reference; it is preserved for historical traceability only.

# APPENDIX B: Phase 2 Analysis Substep Mapping to Workshop Goals

**Document Version:** 2.0  
**Companion to:** Phase 2 Comprehensive Clustering Analysis Plan v2.0  
**Last Updated:** 2026-03-29 (updated after Steps 2–4 planning complete)

## Status Summary (as of 2026-03-29)

**Steps 2 and 3 are COMPLETE.** All 9 issues in Phase2_Step2_Issues.md resolved. All Step 3 re-runs complete (path-filtered betweenness, hub quality fix, reproducible calculations).

**Key insights that shift priorities:**
- Algorithm comparison (substep #20): ✅ DONE — agglomerative k=40 is definitively best (Louvain silhouette=0.011, HDBSCAN 80.7% noise — both unsuitable)
- ARI stability (#7): ✅ DONE — ARI 0.49–0.72 for risk/intervention at high thresholds; confirmed stable taxonomy
- EDGE validation (#4, #5, #22): ✅ DONE — SIM≥0.9 "both": 90.8% EDGE validation; SIM≥0.85 "both": 73%; SIM≥0.8 unconstrained: 27%
- Hub quality (#14): ✅ DONE (corrected) — top-5 hubs are "existential catastrophe" variants with 617 SIM≥0.9 edges each; RLHF hub is a SIM≥0.8 artifact only (18 edges at SIM≥0.9)
- Multi-criteria scoring (#23): ✅ DONE — SIM≥0.9 selected; see Step 3 Section A optimal_configs_ranked.csv
- Betweenness centrality (#16): ✅ DONE (path-filtered) — 49/50 top bridge nodes = existential catastrophe variants; FDT ABSENT (full-graph artifact)
- Path-filtered subgraph (#25): ✅ DONE — 42,870 valid-pathway nodes; 38,054 in betweenness subgraph
- Threshold sensitivity (#21): ✅ DONE — SIM≥0.9 selected as optimal
- EDGE-only validation (#22): ✅ DONE

**Step 4 is now the active step.** The taxonomy construction uses a three-level hierarchy: Risk → Connection Concept Chain → Intervention (see Phase2_Step4_Analysis_Plan.md). "Connection concept chain" replaces "mechanism" throughout as Level 2 layer name. A single connection concept chain = the body of one qualifying path; a connection concept chain family = a cluster of similar chains.

**Terminology update:** "Mechanism taxonomy" (Goal 6 and throughout) now refers to the three-level taxonomy with named connection concept chain families as Level 2.

---

## PART 1: SUBSTEP-TO-WORKSHOP GOAL MAPPING

This appendix maps each of the 34 Phase 2 analysis substeps to the 9 workshop-critical goals identified for publication acceptance.

### Workshop Critical Goals (Reference)

1. **⭐ Cross-Threshold Stability (ARI)** – Demonstrates robustness of mechanisms across similarity thresholds
2. **⭐ EDGE-Validation Rate** – Shows grounding in single-source literature evidence vs. similarity artifacts
3. **⭐ Algorithm Comparison** – Method justification with performance metrics
4. **⭐ Optimal Config Selection with Full Transparency** – Multi-criteria scoring showing all base data
5. **⭐ Path Length vs. Quality Relationship** – Validates ≥5 hop claim for mechanistic detail
6. **⭐ Connection Concept Chain Taxonomy with Manual Validation** – Core deliverable: three-level risk→connection concept chain→intervention hierarchy with named families and coherence scoring
7. **⭐ Intervention Hub Quality Assessment** – Addresses artifact concerns with EDGE-validation and source diversity
8. **⭐ Temporal Coverage Analysis** – Shows scope of literature (publication dates per cluster)
9. **⭐ Bias Documentation** – Critical for limitations section

---

### Substep Mapping Table

| Substep | Analysis Name | Primary Goal(s) | Secondary Goal(s) | Step |
|---------|---------------|-----------------|-------------------|------|
| **1** ✅ DONE | Silhouette Score | Goal 3 (Algorithm) | Goal 4 (Selection) | Step 2 |
| **2** ✅ DONE | Cluster Size Distribution | Goal 6 (Taxonomy) | Goal 4 (Selection) | Step 2 |
| **3** ✅ DONE | Cluster Cohesion Metrics | Goal 3 (Algorithm) | Goal 4 (Selection) | Step 2 |
| **4** ✅ DONE | EDGE Validation Rate | **Goal 2** (EDGE) | Goal 4 (Selection) | Step 2 |
| **5** ✅ DONE | EDGE Purity per Cluster | **Goal 2** (EDGE) | Goal 6 (Taxonomy) | Step 2 |
| **6** ✅ DONE | Source Diversity per Cluster | Goal 7 (Hub Quality) | **Goal 2** (EDGE) | Step 2 |
| **7** ✅ DONE | Cross-Threshold Stability (ARI) | **Goal 1** (Stability) | Goal 4 (Selection) | Step 2 |
| **8** ✅ DONE | Node Migration Analysis | **Goal 1** (Stability) | Goal 9 (Bias) | Step 2 |
| **9** | Mode Impact on Clustering Quality | Goal 9 (Bias) | Goal 4 (Selection) | Step 2 |
| **10** | Multi-Risk Cluster Characterization | Goal 6 (Taxonomy) | Goal 9 (Bias) | Step 2 |
| **11** | Risk Diversity per Configuration | Goal 6 (Taxonomy) | Goal 9 (Bias) | Step 2 |
| **12** | Risk→Intervention Connectivity | Goal 6 (Taxonomy) | - | Step 2 |
| **13** | Intervention Maturity Distribution | Goal 6 (Taxonomy) | Goal 9 (Bias) | Step 2 |
| **14** ✅ DONE | Intervention Hub Quality Assessment | **Goal 7** (Hub Quality) | **Goal 2** (EDGE) | Step 2 |
| **15** | Category-Specific Mechanism Families | Goal 6 (Taxonomy) | - | Step 2 |
| **16** ✅ DONE | Mechanism Transfer Enablers | Goal 6 (Taxonomy) | - | Step 2 |
| **17** | Risk→Concept→Intervention Triplets | Goal 6 (Taxonomy) | Goal 4 (Selection) | Step 4 |
| **18** | Exemplar Path Extraction | Goal 6 (Taxonomy) | **Goal 5** (Path Length) | Step 4 |
| **19** ✅ DONE | Pathway Signature Validation | **Goal 5** (Path Length) | Goal 9 (Bias) | Step 2 |
| **20** ✅ DONE | Algorithm Performance Comparison | **Goal 3** (Algorithm) | Goal 4 (Selection) | Step 3 |
| **21** ✅ DONE | Edge Threshold Sensitivity Test | **Goal 1** (Stability) | Goal 4 (Selection) | Step 3 |
| **22** ✅ DONE | EDGE-Only Configuration Validation | **Goal 2** (EDGE) | Goal 4 (Selection) | Step 3 |
| **23** ✅ DONE | Multi-Criteria Scoring Transparency | **Goal 4** (Selection) | Goal 9 (Bias) | Step 3 |
| **24** | Held-Out Test Set Validation | Goal 6 (Taxonomy) | Goal 3 (Algorithm) | Step 3 |
| **25** ✅ DONE | EDGE Subgraph Consistency Check | **Goal 2** (EDGE) | Goal 6 (Taxonomy) | Step 3 |
| **26** | Cluster Naming & Manual Validation | **Goal 6** (Taxonomy) | - | Step 4 |
| **27** | Exemplar Quality Assessment | Goal 6 (Taxonomy) | **Goal 5** (Path Length) | Step 4 |
| **28** | Risk-Intervention Connectivity Matrix | Goal 6 (Taxonomy) | - | Step 4 |
| **29** ✅ DONE | Pathway Length vs Cluster Quality | **Goal 5** (Path Length) | Goal 9 (Bias) | Step 2 |
| **30** ✅ DONE | Temporal Coverage Analysis | **Goal 8** (Temporal) | Goal 9 (Bias) | Step 2 |
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
| **Goal 6: Connection Concept Chain Taxonomy** | 2, 10, 11, 12, 13, 15, 16, 17, 18, 24, 26, 27, 28, 31, 32, 33, 34 | 5 | 18 substeps (most now complete or absorbed into S4-25–S4-28) |
| **Goal 7: Hub Quality Assessment** | 14 | 6 | 2 substeps |
| **Goal 8: Temporal Coverage** | 30 | - | 1 substep |
| **Goal 9: Bias Documentation** | 9 | 8, 10, 11, 13, 19, 23, 29, 30, 31 | 10 substeps |

**Key Observations:**
- Goal 6 (Connection Concept Chain Taxonomy) is the most comprehensive, involving 18 substeps – reflects core deliverable (3-level hierarchy: risk → connection concept chain → intervention)
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

**Active step as of 2026-03-29.** Plan in Phase2_Step4_Analysis_Plan.md.

**Revised substep set (renumbered to avoid collision with prior substeps):**

| New # | Substep Name | Theme | Goals | Question Answered | Output | Criticality |
|-------|--------------|-------|-------|-------------------|--------|-------------|
| **S4-25** | Build Connection Concept Chain Cluster Tables | Taxonomy | **Goal 6**: Taxonomy | Run Option A (path-body clustering) and Option B (pkl subtype co-occurrence) to produce Level 2 connection concept chain families; produce cluster summary tables for all three levels | `step4_cluster_tables/` | **CRITICAL** |
| **S4-29** | Consecutive SIM ARI Test | Method Rigor | **Goal 1**: Stability | Does applying the consecutive SIM ≤2 cut change cluster assignments (ARI)? If ARI < 0.7, recluster before naming. Gate decision before naming. | `step4_paths/consecutive_sim_ari_test.json` | **CRITICAL** |
| **S4-26** | Cluster Naming & Human Review | Taxonomy | **Goal 6**: Taxonomy | LLM-generated names for all ~120 clusters (risk + connection concept chain + intervention); naming test for Level 2 chains; human review of risk clusters and cited findings | `step4_cluster_tables/taxonomy_names.csv` | **CRITICAL** |
| **S4-27** | Three-Level Connectivity Network | Taxonomy | **Goal 6**: Taxonomy | EDGE connectivity between levels; gap analysis (6 gap types); three-layer Sankey/node-link visualization (primary workshop figure) | `step4_connectivity/` + `three_layer_network.png` | **CRITICAL** |
| **S4-28** | Subcluster Analysis | Taxonomy | **Goal 6**: Taxonomy | Triggered subclustering for heterogeneous clusters; refine large/ambiguous families | `step4_subclusters/` | **ESSENTIAL** |
| **S4-path** | Path Sampling | Taxonomy/Sim Prep | **Goal 5**: Path Length | Sample top paths per connection concept chain cluster (EDGE purity + path length) for Step 5 simulation | `step4_paths/` | **ESSENTIAL** |

**Original Step 4 substep disposition:**
| Old # | Original Name | Disposition |
|-------|--------------|-------------|
| 26 | Cluster Naming & Manual Validation | → S4-26 (active, enhanced with connection concept chain naming test) |
| 17 | Risk→Concept→Intervention Triplets | → Absorbed into S4-27 (connectivity network); Option A/B triplet formation = connection concept chain Options A/B |
| 28 | Risk-Intervention Connectivity Matrix | → Absorbed into S4-27 |
| 18 | Exemplar Path Extraction | → Absorbed into S4-path (path sampling); full exemplar quality = Step 5 scope |
| 27 | Exemplar Quality Assessment | → Step 5 scope |
| 32 | Inter-Rater Agreement (Fleiss' kappa) | → Absorbed into S4-26 human review protocol; full kappa calculation deprioritized |
| 33 | Path→Prompt Engineering | → Step 5 scope |
| 34 | Expected Output Formatting | → Step 5 scope |

**Step 4 Key Deliverables (revised):**
- Named three-level taxonomy: risk families × connection concept chain families × intervention families
- Three-layer Sankey/network visualization (primary workshop figure)
- Gap analysis: 6 gap types across all layer-pair combinations
- Representative path sample per connection concept chain family (for Step 5 simulation)
- Subcluster refinements for heterogeneous families

---

### CRITICALITY SUMMARY ACROSS ALL STEPS

**As of 2026-03-29: Steps 2 and 3 are COMPLETE. Step 4 is active.**

| Criticality | Step 2 | Step 3 | Step 4 | Total |
|-------------|--------|--------|--------|-------|
| **COMPLETE** | 19 | 6 | 0 | **25 substeps done** |
| **CRITICAL (remaining)** | 0 | 0 | 4 (S4-25, S4-29, S4-26, S4-27) | **4 substeps** |
| **ESSENTIAL (remaining)** | 0 | 0 | 2 (S4-28, S4-path) | **2 substeps** |
| **Step 5 scope** | — | — | 4 (old #27, #32, #33, #34) | deferred |

**Remaining Critical Path for Workshop (Step 4 only, ~1 day compute + naming):**
1. S4-25: Build cluster tables (Option A + B) — new computation
2. S4-29: Consecutive SIM ARI test — gate decision
3. S4-26: LLM naming pass + human review of risk clusters
4. S4-27: Three-level connectivity + gap analysis + Sankey visualization

**Essential (Step 4):**
5. S4-28: Subcluster analysis for heterogeneous families
6. S4-path: Path sampling per connection concept chain cluster for simulation

**Previously critical — now complete:**
- ✅ #7 ARI stability — confirmed 0.49–0.72 for risk/intervention at high thresholds
- ✅ #4 EDGE validation — SIM≥0.9 "both" = 90.8%; selection justified
- ✅ #14 Hub quality — top-5 = existential catastrophe variants (617 SIM≥0.9 edges); RLHF = SIM≥0.8 artifact
- ✅ #19 Path length validation — 0.82% of paths below 5 hops overall; 8.2% EDGE-only
- ✅ #20 Algorithm comparison — agglomerative k=40 definitively best
- ✅ #23 Multi-criteria scoring transparency — SIM≥0.9 selected; all 160 configs ranked

**Upward priority revisions:**
- Gap analysis (new, in S4-27): now CRITICAL — forms the core narrative of what AI safety literature covers and misses
- Three-layer visualization (new, in S4-27): now CRITICAL — primary workshop figure
- Connection concept chain construction option A/B (new, in S4-25): now CRITICAL — Level 2 layer is the novel contribution

**Downward priority revisions:**
- #32 Inter-rater agreement: absorbed into S4-26 human review; Fleiss' kappa not needed
- #17/#18 Triplet/exemplar extraction: absorbed into S4-27/S4-path; full exemplar analysis moved to Step 5
- #33/#34 Simulation prep: Step 5 scope

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

### Results Section (4.5 Connection Concept Chain Taxonomy)

**Primary Substeps:** 2, 10, 11, 12, 15, 16, 17, 18, 26, 27, 28  
**Theme:** Mechanism Discovery & Taxonomy  
**Content:**
- 40-60 connection concept chain families identified (three-level hierarchy: risk → chain → intervention)
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
