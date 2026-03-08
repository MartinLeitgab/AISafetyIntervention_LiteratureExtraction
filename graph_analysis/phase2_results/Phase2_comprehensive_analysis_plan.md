Phase_2_Comprehensive_Analysis_Plan.md
36.47 KB •939 lines
•
Formatting may be inconsistent from source

# Phase 2 Comprehensive Clustering Analysis Plan
## AI Safety Intervention Pathway Analysis

**Document Version:** 2.0  
**Last Updated:** January 2026  
**Status:** Implementation Ready

---

## Executive Summary

This Phase 2 analysis evaluates 160 clustering configurations (5 edge configs Ã— 4 pathway modes Ã— 8 node types) to identify distinct mechanisms connecting AI safety risks to interventions. The analysis supports preference simulation by extracting mechanistically coherent pathway families grounded in literature evidence.

**Key Objectives:**
1. Validate clustering quality across all configurations
2. Identify optimal configurations per node type
3. Extract 40-60 distinct mechanism families
4. Prepare 500-1000 representative pathways for simulation
5. Document biases and limitations for workshop presentation

---

## I. CLUSTERING QUALITY METRICS (All 160 Configurations)

### A. Internal Validity Metrics

**Per Configuration Ã— Per Algorithm** (3 algorithms Ã— 160 configs = 480 measurements):

#### 1. Silhouette Score
- **Measurement:** Cosine distance in embedding space
- **Target:** >0.3 (reasonable cluster separation)
- **Grouping:** By node_type, edge_config, mode
- **Interpretation:** Higher = better-defined clusters
- **Bias Note:** Sensitive to cluster count; artificially high for over-fragmented clusters

#### 2. Cluster Size Distribution
- **Metrics:** Mean, median, min, max, standard deviation
- **Target Range:** 40-60 clusters (interpretable for mechanism taxonomy)
- **Warning Flags:**
  - Over-fragmentation: <5 members/cluster average
  - Over-aggregation: >500 members in single cluster
- **Output:** Distribution statistics per configuration

#### 3. Cluster Cohesion Metrics
- **Intra-cluster Distance:** Average pairwise cosine distance within clusters
- **Inter-cluster Separation:** Minimum distance between cluster centroids
- **Separation Ratio:** Inter/Intra (higher = better separation)
- **Purpose:** Quantify cluster compactness and distinctiveness

### B. EDGE Validation Metrics

#### 4. EDGE Validation Rate
- **Definition:** % clusters containing â‰¥1 node from EDGE-only pathways
- **Target:** >60% (grounded in single-source literature evidence)
- **Breakdown:** By similarity threshold (0.8, 0.85, 0.9, 0.95, EDGE-only)
- **Critical for:** Establishing literature grounding vs. similarity artifacts

#### 5. EDGE Purity per Cluster
- **Measurement:** % of cluster members from EDGE-only pathways
- **Gold Standard:** Clusters with >80% EDGE membership
- **Analysis:** Distribution of EDGE purity across all clusters
- **Use Case:** Prioritizing high-confidence clusters for simulation

#### 6. Source Diversity per Cluster
- **Metric:** Count of unique source documents contributing to cluster
- **Warning:** Single-source clusters may be extraction artifacts
- **Target:** â‰¥3 sources for robust mechanisms
- **Extraction:** From graph node attributes (source_file)

---

## II. CROSS-CONFIGURATION STABILITY ANALYSIS

### C. Threshold Sensitivity (Edge Dimension)

#### 7. Cross-Threshold Stability (Adjusted Rand Index)
**Comparison Pairs:**
- EDGE-only â†” SIMâ‰¥0.95
- SIMâ‰¥0.95 â†” SIMâ‰¥0.9  
- SIMâ‰¥0.9 â†” SIMâ‰¥0.85
- SIMâ‰¥0.85 â†” SIMâ‰¥0.8

**Target:** ARI >0.7 for adjacent thresholds (stable mechanisms)  
**Interpretation:** High ARI = clustering structure persists across thresholds  
**Workshop Critical:** Demonstrates robustness of identified mechanisms

#### 8. Node Migration Analysis
- **Track:** Which nodes change cluster assignments across thresholds
- **Identify:**
  - **Core nodes:** Same cluster across all 5 thresholds
  - **Peripheral nodes:** Migrate frequently (boundary cases)
- **Output:** Migration frequency heatmap per node type

### D. Mode Sensitivity (Constraint Dimension)

#### 9. Mode Impact on Clustering Quality
**Comparison:** Unconstrained vs. single-risk vs. monotonic vs. both  
**Metrics:**
- Cluster count change
- Silhouette score change
- EDGE-validation rate change

**Research Question:** Do pathway constraints improve clustering interpretability?

#### 10. Multi-Risk Cluster Characterization
- **In Unconstrained Mode:** % clusters containing multiple risk categories
- **Analysis:** Manual inspection of 10 multi-risk clusters
- **Decision:** Are these mechanistically coherent or graph artifacts?

---

## III. NODE-TYPE SPECIFIC ANALYSIS (8 Types Ã— 5 Edge Ã— 4 Mode = 160)

### E. Risk Node Clustering

#### 11. Risk Diversity per Configuration
- **Metric:** Count unique risk categories represented
- **Distribution:** Balanced vs. skewed across clusters
- **Exemplar Analysis:** Which risks are cluster centers?
- **Bias Note:** Category balance enforced by extraction prompt

#### 12. Riskâ†’Intervention Connectivity
- **Per Risk Cluster:** Which intervention clusters are reachable via pathways?
- **Pathway Count Distribution:** Quantify connection strength
- **Note:** No orphaned risks possible (all risks extracted from complete pathways)

### F. Intervention Node Clustering

#### 13. Intervention Maturity Distribution
**Critical Correction:** All interventions already filtered to maturity â‰¥3  
**Updated Analysis:**
- Maturity 3 vs. 4 distribution per cluster
- More insightful: **Intervention lifecycle stage** distribution
  - Design stage
  - Training stage  
  - Deployment stage
- **Per Cluster:** Dominant lifecycle stage identification

#### 14. Intervention Hub Quality Assessment
**Top-20 Hubs per Configuration (by degree):**
- **EDGE-only degree** vs. **total degree**
- **Source diversity:** # unique papers citing hub
- **Risk diversity:** # unique risk categories connected
- **Categorization:**
  - **Convergence:** Genuine multi-risk solution (e.g., "interpretability")
  - **Framework:** Broad umbrella term (e.g., "AI alignment")
  - **Artifact:** Similarity-induced false hub
- **Validation:** Manual inspection of 5 sample hubs, examine 3-5 neighbors each

### G. Concept Node Clustering (6 Categories)

#### 15. Category-Specific Mechanism Families
- **Per Concept Category:** Identify transferable mechanisms
  - Example: "problem_analysis" â†’ "neglected research directions"
  - Example: "theoretical_foundation" â†’ "formal verification frameworks"
- **Cross-Category Bridges:** Which concepts connect multiple categories?

#### 16. Mechanism Transfer Enablers (Betweenness Analysis)
- **Identify:** High-betweenness concepts bridging multiple risk-intervention pairs
- **Quantify:** # distinct riskâ†’intervention paths passing through each concept cluster
- **Output:** Top-20 transfer enablers per concept category
- **Use Case:** Prioritizing concepts for simulation focus

---

## IV. META-PATHWAY CONSTRUCTION (For Simulation)

### H. Cluster Triplet Formation

#### 17. Riskâ†’Conceptâ†’Intervention Triplets
**Critical Decision Required:**
- **Option A:** Use `all_concepts` clusters for triplet formation
  - Simpler: 1 mechanism type per risk-intervention pair
  - Faster analysis
- **Option B:** Use 6 individual concept categories separately
  - Richer: 6 mechanisms per pair showing different mechanistic steps
  - More comprehensive mechanistic detail
- **Recommendation:** **Option B** for mechanistic richness, validate consistency with Option A

**Per Triplet:**
- Count actual pathways connecting exemplar nodes
- Validate pathway coherence via sampling
- Build triplet connectivity matrix

#### 18. Exemplar Path Extraction
**Exemplar Definition (Confirmed):** Node with minimum average cosine distance to all cluster members (closest to centroid)

**Methods:**
- **Method 1:** Actual pathway closest to cluster centroids
- **Method 2:** Synthetic pathway (connect exemplar nodes directly)
- **Comparison:** ARI between Method 1 and Method 2 cluster assignments
- **Validation:** Does synthetic path preserve mechanism semantics?

#### 19. Pathway Signature Validation
- **Category Sequence Coherence:**  
  Risk â†’ Problem Analysis â†’ Theory â†’ Design â†’ Implementation â†’ Validation â†’ Intervention
- **Length Distribution Analysis:**  
  - Changed from binary to **continuous bins**: [1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13+]
  - **Updated Analysis:** Plot silhouette score vs. length bins
  - **Target:** Validate â‰¥5 hop claim for sufficient mechanistic detail
- **Filter Rationale:** Document path length bias (6-hop peak from extraction prompt)

---

## V. ALGORITHM COMPARISON & VALIDATION

### I. Algorithm Performance Comparison (Workshop Critical)

#### 20. Agglomerative vs. Louvain vs. HDBSCAN
**Per Node Type Ã— Per Edge Config Ã— Per Mode:**
- Silhouette scores
- Cluster count
- EDGE-validation rate
- Computational time

**Output:** Algorithm performance matrix  
**Purpose:** Method justification for paper

### J. Earmarked Validation Tests

#### Test 1: Risk Node Frequency Analysis
- **Within-Cluster:** Risk repetition (mechanism-based grouping)
- **Across-Cluster:** Risk diversity (broad coverage)
- **Metric:** Entropy of risk distribution per configuration

#### Test 2: Intervention Hub EDGE-Only Evidence â­
**Sample:** Top-20 intervention hubs  
**Metrics:**
- EDGE-only % (fraction of degree from EDGE-only pathways)
- Source diversity (# unique papers)
- Risk diversity (# unique risk categories)
**Categorize:** Convergence vs. Framework vs. Artifact

#### Test 3: Zig-Zag Path Validity
**Sample:** 10 pathways with category backtracking (rejected by monotonic constraint)  
**Manual Inspection:** Are intermediate concepts identical or similar?  
**Score:** Coherent (valid multi-mechanism) vs. Incoherent (similarity artifact)

#### Test 4: Path Length Filtering Impact â­
**Updated Analysis:**
- Plot silhouette vs. path length bins [1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13+]
- Compare cluster quality with/without â‰¥5 hop filter
- **Question:** Does length filter improve clustering quality?

#### Test 5: Node Category Type Clustering âœ“ COMPLETE
**Status:** 6 concept categories already clustered in 160 configurations  
**Pending:** Extract insights from completed clusters  
**Question:** Which concept categories yield best mechanism families?

#### Tests 6-8: EDGE-Only Validation Suite

**Test 6: EDGE-Only as Baseline**
- Compare EDGE-only vs. SIMâ‰¥0.85 cluster quality
- Metric: ARI overlap in cluster assignments
- Target: >0.5 (substantial agreement)

**Test 7: EDGE-Only Simulation Sampling**
- Sample 100 EDGE-only pathways (stratified by lifecycle stage)
- High-confidence test set for LLM preference simulation
- Validate simulation methodology before scaling

**Test 8: EDGE-Only Coverage Analysis**
- **Interventions:** % appearing in EDGE-only vs. similarity-augmented
- **Risks:** % appearing in EDGE-only vs. similarity-augmented
- **Characterize:** Are similarity-only nodes foundational or niche?

#### Test 9: Temporal Coverage Analysis â­
- **Publication dates** per cluster (from graph node attributes)
- **Timeline visualization:** Show literature coverage
- **Identify:** Emerging vs. established mechanisms
- **Bias Documentation:** April 2024 cutoff â†’ missing recent developments

---

## VI. OPTIMAL CONFIGURATION SELECTION

### K. Selection Criteria Framework

#### 21. Tier 1 Configurations (Priority Evaluation)
1. **EDGE unconstrained** â€“ Validation baseline (single-source evidence)
2. **SIMâ‰¥0.85 unconstrained** â€“ Recommended optimal (balanced coverage + quality)
3. **SIMâ‰¥0.85 single-risk** â€“ Focused mechanisms (single risk per pathway)
4. **SIMâ‰¥0.9 monotonic** â€“ High precision (no category backtracking)
5. **SIMâ‰¥0.8 unconstrained** â€“ Maximal coverage (broadest similarity inclusion)

#### 22. Multi-Criteria Scoring (Per Node Type) â­
**Weights:**
- Silhouette Score: 25%
- EDGE-Validation Rate: 30%
- Cluster Count (40-60 target): 20%
- Cross-Threshold Stability (ARI): 15%
- Manual Interpretability: 10%

**Process:**
1. Normalize each metric to [0,1] scale
2. Apply weights
3. Rank all 160 configurations per node type
4. **Full Transparency:** Show all base data, not just final scores

#### 23. Final Configuration Selection
- **For Risks:** Config maximizing risk diversity + EDGE-validation
- **For Interventions:** Config maximizing maturity distribution + hub quality
- **For Concepts:** Config maximizing mechanism transfer (betweenness centrality)

**Deliverable:** Optimal configs table with full justification

---

## VII. GRAPH TOPOLOGY ANALYSIS (Using FalkorDB Data)

### L. Node Attribute Enrichment

#### 24. Cluster-Level Aggregations
**Per Cluster, Extract from Graph:**
- Average node degree
- Category distribution
- Source file distribution
- Maturity distribution (interventions only)
- Aliases/description text (for cluster naming)

#### 25. Temporal Coverage
- Publication dates per cluster
- Identify emerging vs. established mechanisms
- **Bias Note:** English-only corpus (ARD) â†’ misses non-English literature

### M. Edge Pattern Analysis

#### 26. Within-Cluster Edge Density
- Count EDGE edges connecting cluster members
- Count similarity edges connecting cluster members
- **Ratio:** Within-cluster / Between-cluster edges
- **Purpose:** Quantify cluster internal connectivity

#### 27. Cross-Cluster Connectivity
- Build cluster-level graph (nodes = clusters, edges = pathway connections)
- Identify isolated mechanism families vs. interconnected mechanisms
- **Output:** Cluster-level network visualization

---

## VIII. MECHANISM TAXONOMY CONSTRUCTION

### N. Mechanism Family Identification

#### 28. Final Mechanism Count
**From Optimal Configurations:**
- Total distinct mechanism families
- **Target:** 40-60 interpretable families (workshop requirement)
- Hierarchical organization: Group related mechanisms

#### 29. Riskâ†’Intervention Mapping Matrix
- **Sparse Matrix:** (n_risks Ã— n_interventions)
- **Values:** # mechanisms connecting each pair
- **Highlight:** Well-connected vs. isolated pairs
- **Coverage Analysis:** % of (risk, intervention) pairs with â‰¥1 mechanism

#### 30. Pathway Selection for Simulation
**Per Mechanism Family:** Select representative pathways  
**Stratification:**
- Path length (5-12 hops recommended)
- Intervention lifecycle stage (design/training/deployment)
- Risk category (balanced representation)

**Target:** 500-1000 pathways for initial simulation batch

---

## IX. INTERPRETABILITY & NAMING

### O. Cluster Naming and Validation

#### 31. Exemplar Quality Check
- Sample 5 clusters per optimal configuration
- Verify: Does exemplar name represent cluster semantics?
- Alternative: LLM-generated cluster descriptions from top-5 members

#### 32. Top-Terms Coherence
- Evaluate: Do top-5 terms capture cluster theme?
- Manual annotation: Mechanism type classification
- **Workshop Prep:** Sample 10 risk + 10 intervention + 10 concept clusters
- **Annotators:** Score coherence (1-5 scale)
- **Inter-Rater Agreement:** Fleiss' kappa

---

## X. SIMULATION READINESS ASSESSMENT

### P. Preference Simulation Preparation

#### 33. Pathâ†’Prompt Engineering
**Template:** Convert pathway to natural language description  
**Include:**
- Risk description â†’ mechanism steps â†’ intervention details
- Test: Sample 10 paths, generate prompts, verify comprehensibility

#### 34. Expected Output Formatting
**Per Mechanism Family:** Aggregate simulation results  
**Metrics:**
- Î” Instrumental goals (power-seeking, self-preservation)
- Î” Pro-human goals (cooperation, safety, alignment)
- Î” Anti-human goals (deception, manipulation, harm)
**Statistical Testing:** Bootstrapped 95% confidence intervals

---

## WORKSHOP QUALITY CRITICAL ITEMS â­

### Must-Have for Publication:

1. â­ **Cross-Threshold Stability (ARI)** â€“ Demonstrates robustness of mechanisms across similarity thresholds
2. â­ **EDGE-Validation Rate** â€“ Shows grounding in single-source literature evidence vs. similarity artifacts
3. â­ **Algorithm Comparison** (Agglomerative vs. Louvain vs. HDBSCAN) â€“ Method justification with performance metrics
4. â­ **Optimal Config Selection with Full Transparency** â€“ Multi-criteria scoring showing all base data, not just final scores
5. â­ **Path Length vs. Quality Relationship** â€“ Validates â‰¥5 hop claim for sufficient mechanistic detail
6. â­ **Mechanism Taxonomy with Manual Validation** â€“ Core deliverable (40-60 families) with coherence scoring
7. â­ **Intervention Hub Quality Assessment** â€“ Addresses potential artifact concerns with EDGE-validation and source diversity
8. â­ **Temporal Coverage Analysis** â€“ Shows scope of literature (publication dates per cluster)
9. â­ **Bias Documentation** â€“ Critical for limitations section (see checklist below)

### Secondary (Nice-to-Have):

- Cluster-level network visualization
- Embedding UMAP projections (supplementary material)
- Riskâ†’Intervention connectivity matrix heatmap

---

## BIAS MITIGATION CHECKLIST

### Document Throughout Analysis:

1. **Path Length Bias:** 6-hop peak from extraction prompt â†’ Flag in methods section
2. **Category Balance Bias:** Prompt enforces balanced distribution â†’ Affects frequency claims
3. **Maturity Threshold:** Excludes 85% of interventions (maturity <3) â†’ May miss emerging approaches
4. **EDGE Confidence â‰¥3:** Excludes speculative connections â†’ Conservative mechanism set
5. **Similarity Threshold Choice:** Impacts cross-source transfer â†’ Justify with stability analysis (ARI)
6. **Clustering Algorithm Choice:** Different algorithms yield different k â†’ Show all 3, justify selection
7. **Node-Type Separation:** Risks/interventions/concepts clustered separately â†’ May miss cross-type patterns
8. **Temporal Cutoff:** April 2024 knowledge cutoff â†’ Missing recent developments
9. **Exemplar Selection:** Centroid-based (min avg distance) â†’ May not represent most "important" node
10. **English-Only Corpus:** ARD language bias â†’ Misses non-English AI safety literature

**Presentation:** Include bias disclosure in limitations section of all deliverables

---

## CONSOLIDATED VISUALIZATION STRATEGY

### Maximum ~20 Main Plots (Not 160 Individual Configs)

**Format:** Each plot = 2Ã—2 or 2Ã—3 subplots by primary dimension, overlay others via color/line style

#### Core Quality Metrics (5 plots)

1. **Cluster Quality by Node Type** (1 plot, 8 subplots)
   - Subplots: 8 node types (risk, intervention, all_concepts, 6 concept categories)
   - Overlay: 5 edge configs (color) Ã— 4 modes (line style) = 20 lines/subplot
   - Metrics: Silhouette score, EDGE-validation rate

2. **Cross-Threshold Stability ARI** (1 plot, 8 subplots)
   - Subplots: 8 node types
   - Heatmap: 5Ã—5 ARI matrix per subplot
   - Diagonal = 1.0 (self-comparison)

3. **Algorithm Comparison** (1 plot, 8 subplots)
   - Subplots: 8 node types
   - Overlay: 3 algorithms (color) Ã— 5 edge configs (x-axis) Ã— 4 modes (facet)
   - Metric: Silhouette score

4. **Path Length vs. Quality** (1 plot, 4 subplots) â­
   - Subplots: 4 pathway modes
   - X-axis: Length bins [1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13+]
   - Y-axis: Silhouette score
   - Overlay: 5 edge configs (color)

5. **Cluster Size Distributions** (1 plot, 8 subplots)
   - Subplots: 8 node types
   - Violin plots: Distribution across all configs
   - Overlay: Target range (40-60 clusters)

#### EDGE Validation (2 plots)

6. **EDGE-Validation Breakdown** (1 plot)
   - Stacked bar chart
   - Categories: 0%, 1-25%, 26-50%, 51-75%, 76-100% EDGE members per cluster
   - Facet by: 5 edge configs

7. **Hub Quality Assessment** (1 plot) â­
   - Scatter plot: EDGE% (x-axis) vs. Source diversity (y-axis)
   - Point size: Node degree
   - Color: Convergence / Framework / Artifact category
   - Sample: Top-20 intervention hubs

#### Intervention & Temporal Analysis (2 plots)

8. **Intervention Lifecycle Distribution** (1 plot, 5 subplots)
   - Subplots: 5 edge configs
   - Stacked bar: Design / Training / Deployment per cluster
   - Highlight dominant lifecycle stage

9. **Temporal Coverage** (1 plot) â­
   - Timeline: Publication years (2000-2024)
   - Violin plots overlaid on timeline (cluster publication date distributions)
   - Identify: Emerging vs. established mechanisms

#### Mechanism Transfer & Networks (3 plots)

10. **Mechanism Transfer Betweenness** (1 plot, 6 subplots)
    - Subplots: 6 concept categories
    - Top-20 transfer enablers (nodes bridging many risk-intervention pairs)
    - Bar chart: Betweenness centrality scores

11. **Cluster-Level Network** (1 plot)
    - Force-directed layout
    - Nodes = clusters (size = cluster size)
    - Edges = pathway connectivity (width = # paths)
    - Color = EDGE-validation rate

12. **Sankey: Riskâ†’Conceptâ†’Intervention** (1 plot)
    - From optimal configuration only
    - Top-20 flows by pathway count
    - Color-coded by risk category

#### Connectivity & Optimization (2 plots)

13. **Riskâ†’Intervention Connectivity Matrix** (1 heatmap)
    - Sparse matrix visualization
    - From optimal config only
    - Values: # mechanisms connecting each pair

14. **Multi-Criteria Scoring Parallel Coordinates** (1 plot) â­
    - Per node type: show all 160 configs
    - Axes: Silhouette Ã— EDGE-val Ã— Cluster count Ã— Stability Ã— Interpretability
    - Highlight: Pareto front (optimal configs)

#### Embedding Visualizations (3 plots)

15-17. **UMAP Projections** (3 plots: Risks, Interventions, All_Concepts)
    - From optimal config only
    - 2D UMAP projection colored by cluster assignment
    - Overlay: Highlight EDGE-only nodes (different marker)

#### Additional Heatmaps (~5 plots)

18. **Cluster Ã— Source Diversity Heatmap**
19. **Cluster Ã— Maturity Distribution Heatmap** (interventions only)
20. **Node Migration Across Thresholds Heatmap**
21. **Within-Cluster Edge Density Heatmap**
22. **Cross-Mode Stability Heatmap** (ARI between modes)

**Total: ~22 visualizations (well within target)**

---

## IMPLEMENTATION ARCHITECTURE

### Four-Step Consolidated Pipeline

#### **STEP 1: Data Loading & Parsing (With Checkpoints)**
**Script:** `phase2_step1_load_and_parse.py`  
**Outputs Folder:** `phase2_results/step1_load_and_parse/`

**Tasks:**
1. Load all 160 cluster JSONL files
2. Extract metrics into unified dataframe
3. **Checkpoint 1:** Save `all_cluster_metrics.csv` (full dataframe)
4. Load graph node/edge data from FalkorDB
5. **Checkpoint 2:** Save `graph_node_attributes.pkl` (node data)
6. **Checkpoint 3:** Save `graph_edge_data.pkl` (edge data)
7. Generate `load_summary.txt` (data inventory)

**Key Feature:** Intermediate checkpoints allow iterating on visualization/analysis without re-extracting from FalkorDB

**Outputs:**
- `all_cluster_metrics.csv` â† Primary checkpoint
- `graph_node_attributes.pkl` â† FalkorDB node data
- `graph_edge_data.pkl` â† FalkorDB edge data
- `load_summary.txt` â† Data inventory

**Estimated Time:** 2 hours (with FalkorDB queries)

---

#### **STEP 2: Core Metrics & Stability Analysis**
**Script:** `phase2_step2_metrics_and_stability.py`  
**Outputs Folder:** `phase2_results/step2_metrics_and_stability/`

**Tasks:**

**A. Quality Metrics (Section I-III):**
1. Silhouette scores per config
2. Cluster size distributions
3. EDGE-validation rates
4. Source diversity per cluster
5. Intervention lifecycle distributions
6. Risk diversity metrics
7. Mechanism transfer betweenness (concept nodes)

**B. Stability Analysis (Section II):**
8. Cross-threshold ARI (all pairs)
9. Node migration analysis (track cluster changes)
10. Mode sensitivity analysis (constraint impact)

**Outputs:**
- `quality_metrics_summary.csv` â† All metrics in tabular form
- `stability_ari_matrix.csv` â† Pairwise ARI values
- `node_migration_frequencies.csv` â† Migration tracking
- `silhouette_by_nodetype.png` â† Plot 1
- `cluster_size_distributions.png` â† Plot 5
- `cross_threshold_ari.png` â† Plot 2
- `edge_validation_breakdown.png` â† Plot 6
- `lifecycle_distribution.png` â† Plot 8
- `temporal_coverage.png` â† Plot 9
- `mechanism_transfer_betweenness.png` â† Plot 10

**Estimated Time:** 4 hours (includes all stability calculations)

---

#### **STEP 2b: Extended Analysis** ✅ COMPLETE
**Script:** `phase2_step2b_extended_analysis.py`
**Outputs Folder:** `phase2_results/step2_metrics_and_stability/` (same as Step 2)

**Status:** Implemented all high+medium priority items from Code_Changes_Tracker.md (#1–#4, #6–#10, #12, #14–#15).

**What Step 2b covers (beyond original Step 2):**
- ✅ Change #1 — ARI pairwise + cross-threshold line plot → `stability_ari_pairwise.csv`, `cross_threshold_ari_lineplot.png`
- ✅ Change #2 — Source diversity v2 (url fallback, bug fix) → `cluster_source_diversity_v2.csv`
- ✅ Change #3 — Hub quality scatter → `hub_quality_metrics.csv`, `hub_quality_scatter.png`
- ✅ Change #4 — Node migration heatmap → `node_migration_heatmap.png`
- ✅ Change #6 — EDGE validation per-mode 2×4 grid → `edge_validation_per_mode.png`
- ✅ Change #7 — Algorithm comparison Agglomerative vs Louvain → `algorithm_comparison.csv`, `algorithm_comparison_silhouette.png`
- ✅ Change #8 — Cluster cohesion (intra/inter distances) → `cohesion_analysis.csv`
- ✅ Change #9 — Centroid similarity (semantic stability) → `cluster_centroid_similarity.csv`, `centroid_similarity_heatmap.png`
- ✅ Change #10 — EDGE purity per cluster → `cluster_edge_purity.csv`, `edge_purity_histograms.png`
- ✅ Change #12 — Cluster size y-axis fix → `cluster_size_distributions_v2.png`
- ✅ Change #14 — Path length sensitivity scatter → `path_length_sensitivity.png`
- ✅ Change #15/Sub#9 — Mode impact analysis → `mode_comparison_stats.csv`, `edge_density_heatmap.png`, `mode_stability_heatmap.png`
- ✅ Change #15/Sub#10 — Multi-risk clusters → `multi_risk_clusters.csv`
- ✅ Change #15/Sub#11 — Risk diversity (Gini) → `risk_diversity_stats.csv`
- ✅ Change #15/Sub#13 — Maturity per cluster → `maturity_per_cluster.csv`, `maturity_distribution_heatmap.png`
- ✅ Change #15/Sub#15 — Category mechanism families (preliminary) → `category_mechanism_families.csv`
- ✅ Change #15/Sub#16 — Betweenness v2 (5-panel, descending sort) → `mechanism_transfer_betweenness_v2.png`

**Deferred from Step 2b → Step 3:**
- HDBSCAN clustering (requires re-running `phase2_clustering.py`)
- Multi-criteria scoring with weighted composite score
- Final optimal config selection → `optimal_configs_final.csv`
- Change #5: UMAP visualizations (deferred to Step 4)

---

#### **STEP 3: Algorithm Completion, Validation & Selection**
**Script:** `phase2_step3_validation_and_selection.py`
**Outputs Folder:** `phase2_results/step3_validation_and_selection/`

**Tasks:**

**A. Algorithm Completion (Section V):**
1. ✅ DONE in Step 2b: Agglomerative vs Louvain silhouette comparison → `algorithm_comparison.csv`
2. **REMAINING:** Run HDBSCAN clustering (re-run `phase2_clustering.py` with HDBSCAN algorithm)
3. **REMAINING:** Extend `algorithm_comparison.csv` with HDBSCAN silhouette + cluster count

**B. Validation Tests (Section V):**
4. Test 1: Risk frequency analysis (entropy) — use `risk_diversity_stats.csv` from Step 2b as input
5. ✅ DONE in Step 2b: Test 2 — Hub quality (EDGE%, source diversity) → `hub_quality_metrics.csv`
6. Test 3: Zig-zag path validity (manual sample of 10 backtracking paths) — **still needed**
7. ✅ DONE in Step 2b: Test 4 — Path length sensitivity → `path_length_sensitivity.png`
8. ✅ DONE in Step 2b: Test 5 — Category mechanism families → `category_mechanism_families.csv`
9. Tests 6-8: EDGE-only validation suite (ARI overlap, coverage analysis) — **still needed**

**C. Optimal Selection (Section VI):**
10. Multi-criteria scoring (silhouette + EDGE% + ARI + cluster count — 5 weighted metrics)
11. Rank all 160 configs per node type → `optimal_configs_ranked.csv`
12. Select winners with full transparency → `optimal_configs_final.csv`
13. Generate justification report → `selection_justification.md`

**Outputs (Step 3 still needs to produce):**
- `algorithm_comparison_with_hdbscan.csv` — extended from Step 2b CSV
- `validation_test_results.json` — Tests 3, 6-8 outcomes
- `optimal_configs_ranked.csv` — full multi-criteria rankings
- `optimal_configs_final.csv` — selected winners + justification
- `algorithm_performance_full.png` — 3-algorithm comparison plot
- `multi_criteria_parallel.png` — Plot 14
- `selection_justification.md` — full explanation

**Estimated Time:** 4 hours (reduced; many items completed in Step 2b)

---


#### **STEP 4: Mechanism Taxonomy, Network Analysis & Visualization**
**Script:** `phase2_step4_taxonomy_network_viz.py`  
**Outputs Folder:** `phase2_results/step4_taxonomy_network_viz/`

**Tasks:**

**Note — Items deferred from Step 2b to Step 4:**
- Change #5: UMAP 2D projections (deferred from Step 2b)
- Sub#15 full version: cluster naming + coherence scoring (Step 2b produced preliminary `category_mechanism_families.csv` with exemplars; full naming requires human curation)
- Risk→Intervention connectivity matrix (Sub#28)
- Exemplar path extraction per named cluster (Sub#26)

**A. Mechanism Taxonomy (Section VIII):**
1. Extract 40-60 mechanism families from optimal configs (use `category_mechanism_families.csv` from Step 2b as seed input)
2. Build riskâ†’intervention mapping matrix
3. Select 500-1000 representative pathways (stratified)
4. Generate mechanism family metadata (exemplars, top members)

**B. Network Analysis (Section VII):**
5. Cluster-level aggregations (degree, category distribution)
6. Within-cluster edge density
7. Cross-cluster connectivity graph
8. Build cluster-level network

**C. Embeddings & Visualization (Section IX-X):**
9. UMAP 2D projections (risks, interventions, concepts)
10. Cluster naming and coherence validation
11. Generate all remaining plots

**D. Simulation Prep (Section X):**
12. Pathâ†’Prompt template engineering
13. Sample prompt generation (10 test cases)
14. Export simulation-ready pathway database

**Outputs:**
- `mechanism_families.json` â† 40-60 families with metadata
- `mechanism_taxonomy_summary.csv` â† Tabular overview
- `risk_intervention_matrix.csv` â† Connectivity matrix
- `representative_pathways.jsonl` â† 500-1000 pathways for simulation
- `cluster_level_network_data.json` â† Network topology
- `cluster_naming_validation.csv` â† Coherence scores
- `simulation_prompts_sample.json` â† 10 test prompts
- `cluster_level_network.png` â† Plot 11
- `sankey_risk_concept_intervention.png` â† Plot 12
- `connectivity_matrix_heatmap.png` â† Plot 13
- `umap_risks.png` â† Plot 15
- `umap_interventions.png` â† Plot 16
- `umap_concepts.png` â† Plot 17
- `source_diversity_heatmap.png` â† Plot 18
- `maturity_distribution_heatmap.png` â† Plot 19
- `node_migration_heatmap.png` â† Plot 20
- `edge_density_heatmap.png` â† Plot 21
- `mode_stability_heatmap.png` â† Plot 22

**Estimated Time:** 6 hours (includes UMAP computations and manual naming)

---

## FOLDER ORGANIZATION

```
phase2_results/
â”œâ”€â”€ step1_load_and_parse/
â”‚   â”œâ”€â”€ all_cluster_metrics.csv          â† PRIMARY CHECKPOINT
â”‚   â”œâ”€â”€ graph_node_attributes.pkl        â† FalkorDB checkpoint
â”‚   â”œâ”€â”€ graph_edge_data.pkl              â† FalkorDB checkpoint
â”‚   â”œâ”€â”€ load_summary.txt
â”‚   â””â”€â”€ phase2_step1_load_and_parse.py
â”‚
â”œâ”€â”€ step2_metrics_and_stability/
â”‚   â”œâ”€â”€ quality_metrics_summary.csv
â”‚   â”œâ”€â”€ stability_ari_matrix.csv
â”‚   â”œâ”€â”€ node_migration_frequencies.csv
â”‚   â”œâ”€â”€ silhouette_by_nodetype.png       â† Plot 1
â”‚   â”œâ”€â”€ cross_threshold_ari.png          â† Plot 2
â”‚   â”œâ”€â”€ cluster_size_distributions.png   â† Plot 5
â”‚   â”œâ”€â”€ edge_validation_breakdown.png    â† Plot 6
â”‚   â”œâ”€â”€ lifecycle_distribution.png       â† Plot 8
â”‚   â”œâ”€â”€ temporal_coverage.png            â† Plot 9
â”‚   â”œâ”€â”€ mechanism_transfer_betweenness.png â† Plot 10
â”‚   â””â”€â”€ phase2_step2_metrics_and_stability.py
â”‚
â”œâ”€â”€ step3_validation_and_selection/
â”‚   â”œâ”€â”€ algorithm_comparison.csv
â”‚   â”œâ”€â”€ validation_test_results.json
â”‚   â”œâ”€â”€ optimal_configs_ranked.csv       â† FULL RANKINGS with base data
â”‚   â”œâ”€â”€ optimal_configs_final.csv        â† WINNERS ONLY
â”‚   â”œâ”€â”€ selection_justification.md
â”‚   â”œâ”€â”€ algorithm_performance.png        â† Plot 3
â”‚   â”œâ”€â”€ path_length_sensitivity.png      â† Plot 4
â”‚   â”œâ”€â”€ hub_quality_scatter.png          â† Plot 7
â”‚   â”œâ”€â”€ multi_criteria_parallel.png      â† Plot 14
â”‚   â””â”€â”€ phase2_step3_validation_and_selection.py
â”‚
â””â”€â”€ step4_taxonomy_network_viz/
    â”œâ”€â”€ mechanism_families.json          â† 40-60 families
    â”œâ”€â”€ mechanism_taxonomy_summary.csv
    â”œâ”€â”€ risk_intervention_matrix.csv
    â”œâ”€â”€ representative_pathways.jsonl    â† 500-1000 paths
    â”œâ”€â”€ cluster_level_network_data.json
    â”œâ”€â”€ cluster_naming_validation.csv
    â”œâ”€â”€ simulation_prompts_sample.json
    â”œâ”€â”€ cluster_level_network.png        â† Plot 11
    â”œâ”€â”€ sankey_risk_concept_intervention.png â† Plot 12
    â”œâ”€â”€ connectivity_matrix_heatmap.png  â† Plot 13
    â”œâ”€â”€ umap_risks.png                   â† Plot 15
    â”œâ”€â”€ umap_interventions.png           â† Plot 16
    â”œâ”€â”€ umap_concepts.png                â† Plot 17
    â”œâ”€â”€ source_diversity_heatmap.png     â† Plot 18
    â”œâ”€â”€ maturity_distribution_heatmap.png â† Plot 19
    â”œâ”€â”€ node_migration_heatmap.png       â† Plot 20
    â”œâ”€â”€ edge_density_heatmap.png         â† Plot 21
    â”œâ”€â”€ mode_stability_heatmap.png       â† Plot 22
    â””â”€â”€ phase2_step4_taxonomy_network_viz.py
```

**Key Features:**
- Each step reads checkpoints from previous steps
- Intermediate `.csv` and `.pkl` files enable iteration without full re-computation
- Scripts are self-contained with clear dependencies
- All plots saved as high-resolution PNG (300 DPI for publication)

---

## DELIVERABLES PRESENTATION FORMAT

### Standard Structure for Each Analysis:

```markdown
## Analysis: [Name]

### Base Data
- **Input:** [Specific metrics/files used]
- **Sample Size:** [N configs/clusters/nodes]
- **Filters Applied:** [Any selections made]
- **Data Source:** [Step 1 checkpoint file]

### Methodology
- **Calculation:** [Exact formula/algorithm]
- **Parameters:** [All choices documented]
- **Potential Biases:** [What could skew results - reference checklist]
- **Assumptions:** [Explicitly state what we assume to be true]

### Results
[Plot/table with clear labels, legend, axis units]
[Quantitative summary: mean, median, range, outliers]

### Observations
1. [Objective fact from data - no interpretation]
2. [Objective fact from data - no interpretation]
3. [Objective fact from data - no interpretation]

### Interpretation
- [What this means for mechanism extraction]
- [Caveats and limitations]
- [Confidence level: high/medium/low]
- [Connection to workshop goals]

### Recommendation (if applicable)
- [Based on evidence above]
- [Uncertainty acknowledged]
- [Alternative approaches considered]
```

**Principle:** No premature conclusions. Present evidence first, interpret second. Show all base data for transparency.

---

## IMPLEMENTATION TIMELINE

### Week 1 (Critical Path):
- **Day 1:** Step 1 â€“ Data loading with checkpoints (2 hours)
- **Day 2-3:** Step 2 â€“ Core metrics & stability (4 hours)
- **Day 4-5:** Step 3 â€“ Validation & selection (6 hours)

### Week 2 (Enrichment):
- **Day 6-7:** Step 4 â€“ Taxonomy, network analysis, visualization (6 hours)
- **Day 8:** Final review, bias documentation, deliverable assembly

**Total Estimated Time:** 18 hours of computation + 8 hours of manual validation/documentation = 26 hours

---

## CHECKPOINT-BASED ITERATION STRATEGY

### Key Feature: Load Once, Iterate Many Times

**After Step 1 completion:**
- `all_cluster_metrics.csv` contains ALL clustering results
- `graph_node_attributes.pkl` contains ALL node data
- `graph_edge_data.pkl` contains ALL edge data

**Subsequent scripts can:**
1. Load checkpoints instantly (seconds, not hours)
2. Experiment with different:
   - Plot styles and layouts
   - Statistical tests
   - Filtering criteria
   - Visualization parameters
3. Re-run Steps 2-4 without re-querying FalkorDB

**Version Control:**
- Checkpoint files timestamped: `all_cluster_metrics_2026-01-18.csv`
- Scripts reference latest checkpoint automatically
- Preserves reproducibility across analysis iterations

---

## CRITICAL SUCCESS CRITERIA

### For Workshop Acceptance:

âœ… **Methodological Rigor:**
- All 160 configs analyzed (no sampling)
- 3 algorithms compared (method justification)
- Cross-threshold stability demonstrated (ARI >0.7)
- Bias checklist fully documented

âœ… **Literature Grounding:**
- EDGE-validation rate >60% for optimal configs
- Hub quality assessment (EDGE% vs. total degree)
- Source diversity quantified per cluster

âœ… **Interpretability:**
- 40-60 mechanism families with manual validation
- Coherence scoring (inter-rater agreement)
- Clear cluster naming with exemplar quality checks

âœ… **Transparency:**
- Full multi-criteria scoring with base data shown
- All filtering decisions justified
- Limitations section with bias disclosure

âœ… **Simulation Readiness:**
- 500-1000 representative pathways extracted
- Pathâ†’Prompt templates validated
- Stratified sampling (length, lifecycle, risk category)

---

## APPENDIX: KEY DEFINITIONS

**EDGE-Only Pathway:** Pathway where all connections derived from single source document (confidence â‰¥3)

**Similarity Edge:** Connection between nodes based on cosine similarity â‰¥ threshold in embedding space

**Exemplar Node:** Node with minimum average cosine distance to all cluster members (closest to centroid)

**Mechanism Family:** Cluster of pathways sharing similar riskâ†’conceptâ†’intervention structure

**Betweenness Centrality:** Count of shortest paths passing through a node (transfer enabler metric)

**ARI (Adjusted Rand Index):** Measures agreement between two clustering solutions (1.0 = perfect agreement, 0.0 = random)

**Hub Quality:** Intervention node with high degree, assessed by EDGE% and source diversity

**Lifecycle Stage:** Intervention maturity dimension (design â†’ training â†’ deployment)

---

## DOCUMENT CHANGELOG

**Version 2.0 (January 2026):**
- Consolidated from 9 steps to 4 for efficiency
- Added checkpoint-based iteration strategy
- Incorporated all critical corrections:
  - Maturity analysis updated to lifecycle stages
  - Path length changed from binary to continuous bins
  - Removed orphaned risks analysis (impossible)
  - Added Option A/B decision for concept clustering
- Added workshop quality critical items section
- Added comprehensive bias mitigation checklist
- Updated visualization strategy (22 plots total)
- Added deliverables presentation format template
- Updated folder organization for 4-step structure
- Added implementation timeline and success criteria

**Version 1.0 (January 2026):**
- Initial comprehensive plan (Sections I-XII)
- 9-step implementation architecture
- 56 metrics defined

---

**END OF DOCUMENT**