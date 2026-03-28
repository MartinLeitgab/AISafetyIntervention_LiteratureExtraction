# Phase 2 Step 4 Plan
**Created:** 2026-03-28
**Status:** Ready to implement
**Preceding steps:** Step 3 complete ✅ (`step3_validation_and_selection/`)
**Script to write:** `graph_analysis/phase2_step4_taxonomy_network_viz.py`
**Output folder:** `phase2_results/step4_taxonomy_network_viz/`

---

## Corrections and clarifications from Step 3 post-analysis

### Path generation constraints (Phase 1)
Paths in `phase1_rawpathsfiles/` were generated with the following mandatory filters applied at graph load time (`final_pathway_analysis_modes.py`):
- **`edge_confidence >= 3`** on all EDGE edges (FalkorDB property `edge_confidence`)
- **`intervention_maturity >= 3`** on all intervention endpoint nodes
- Only sources that have at least one mature (conf≥1) intervention in `source_pathways_final.json`

These constraints are applied BEFORE path traversal. The edge_data.pkl (which includes all EDGE edges from FalkorDB) was NOT filtered by confidence — it contains all edges. The path files use a filtered subgraph.

### Filter consistency across pipeline stages

| Stage | conf≥3 EDGE | maturity≥3 intervention |
|-------|-------------|------------------------|
| Path generation (`final_pathway_analysis_modes.py`) | ✅ | ✅ endpoint only |
| Clustering (`phase2_clustering.py`) | ✅ inherited via path files | ✅ inherited via path files |
| Step 1 PKL (`graph_edge_data.pkl`) | ❌ all confidences stored | N/A (stored as attribute) |
| Step 2/3 metrics (use PKL) | ❌ not filtered | reads raw attribute |
| Step 3 betweenness (Section D) | ❌ all EDGE edges used — inconsistency | N/A |
| **Step 4 cluster connectivity (#27)** | ✅ **must apply explicitly** (see task) | N/A |
| Step 4 path sampling | ✅ inherited from path files | ✅ inherited from path files |

**Note:** Step 3 betweenness used ALL EDGE edges from edge_data.pkl without a conf≥3 filter. This is a methodological inconsistency with path generation. The betweenness results are still valid for identifying high-centrality nodes (adding low-conf edges changes betweenness magnitude but not top-node identity significantly), but the methods section should note this distinction.

### Path count structure and "overcounting"
The path files are NOT overcounted in the sense of duplicate paths. Each path is a unique (start_node, end_node) pair. However the fan-out is large:
- 3,751 unique start (risk) nodes × up to 648 reachable intervention endpoints each = 1,054,527 total unconstrained sim0.9 paths
- High-connectivity risk nodes (x-risk hubs) generate 600+ paths each because SIM≥0.9 edges connect them to many interventions across papers

For comparison, the EDGE-only file (`paths_unconstrained_edge_only.jsonl`) has **3,473 paths** — these are the within-source paper logic chains (risk→...→intervention where ALL edges are structural EDGE edges with confidence≥3). This is the true "per-source EDGE backbone."

| Path file | Total paths | Structural interpretation |
|-----------|------------|--------------------------|
| `paths_unconstrained_edge_only.jsonl` | 3,473 | Within-source structural chains only |
| `paths_unconstrained_sim0.9.jsonl` | 1,054,527 | All cross-paper SIM-augmented chains |
| 100% EDGE throughout (in sim0.9 file) | **3,405** | Same as EDGE-only, slightly different due to filtering |

The 1M sim0.9 paths represent cross-paper connections via SIM≥0.9 edges. Each is a distinct (start, end) pair but many share identical body structure — one x-risk hub connecting to many distinct intervention endpoints.

### Node coverage in path files
The Phase 1 path files cover **42,767 unique nodes** across all 20 (mode × edge_config) combinations, and **21,553 nodes** for unconstrained sim0.9 alone. Total nodes in node_attrs.pkl = 200,525.

The remaining ~158K nodes are NOT on any valid complete risk→intervention path:
- Partial chains (paper contributes risk+mechanism nodes but no intervention endpoint, or mechanism+intervention but no risk root)
- Nodes in EDGE components that don't span from a risk category to an intervention category

The Step 3 betweenness (Section D) correctly used the **full 200,568-node graph** (from edge_data.pkl + SIM≥0.9 filter), NOT the path-file-covered subset. Cluster assignments from cluster_memberships.pkl cover 42,767 nodes across all configs.

### Path quality for Step 4 simulation
For the 1,054,527 unconstrained sim0.9 paths, the consecutive-SIM filter is applied to the **full path** (risk preamble + body + any intervention hub). The risk preamble consists of 4–5 x-risk hub nodes connected almost exclusively via SIM edges, so body-only filtering significantly understates the true SIM run lengths.

**Full-path consecutive SIM distribution (confirmed on all 1,054,527 paths):**

| Filter (full path) | Paths kept | % | Interpretation |
|---------------------|-----------|---|----------------|
| 100% EDGE (max_run=0) | 3,405 | 0.3% | Pure within-source structural chains; matches source_pathways_final conf≥3 (3,498) |
| max_consec_SIM ≤ 1 | 75,008 | 7.1% | Conservative: no two consecutive SIM hops anywhere in path |
| max_consec_SIM ≤ 2 | 432,776 | 41.0% | Moderate: allows brief cross-paper bridging via ≤2 SIM hops |
| max_consec_SIM ≤ 3 | 772,423 | 73.2% | Broader: allows 3-hop SIM chains |
| ≥50% EDGE full path | 967,730 | 91.8% | Minimal structural requirement |

**Step 4 will produce results for BOTH ≤1 and ≤2 consecutive SIM options** applied to the full path. ≤1 = 75K paths (conservative, requires every SIM hop surrounded by EDGE); ≤2 = 433K paths (allows 2-hop SIM bridging including in risk preamble).

**Why EDGE-only = 3,405 and not ~33–55K:** The `load_graph(min_conf=3)` filter removes 38.7% of EDGE edges (all conf<3). At conf≥3, only 1,917/3,703 mature sources have complete risk→intervention chains. This matches source_pathways_final conf≥3 total of 3,498. The user's estimate of 11K×3–5 paths assumed all EDGE edges and all sources; the conf≥3 filter is the primary restriction.

### Clustering mode for Step 4
**Use unconstrained** for the 3-level cluster connectivity analysis (not single_risk or both).

Rationale: single_risk removes the ~3,000 x-risk hub nodes that appear in multi-risk paths. These nodes form the large shared x-risk cluster that sits at the top of the risk hierarchy. For Step 4, this hierarchy is valuable: x-risk cluster → secondary risk clusters → mechanism clusters → intervention clusters. Keeping them is desired, not a problem to filter.

Single_risk produces 7,103 paths covering 18,590 nodes. Unconstrained produces 1,054,527 paths covering 21,553 nodes. The 2,963-node difference is primarily the x-risk hub nodes — exactly what to preserve.

**Note on monotonic:** monotonic does NOT prevent multi-risk paths. Monotonic paths have mean 4.58 risk nodes/path (nearly identical to unconstrained 5.02). The single_risk constraint is what filters multi-risk paths (99.3% reduction). If directional ordering is desired for the body, `monotonic + EDGE_fraction_filter` is an option, but is not the primary cut for Step 4.

---

## Step 4 Task List

### A. Mechanism Taxonomy — CRITICAL (human-in-loop)

**#26 Manual Cluster Naming**
- Input: `category_mechanism_families.csv` (Step 2b seed), `betweenness_both09_bridge_clusters.csv` (mechanism-space bridge seeds), `cluster_memberships.pkl`
- For each cluster in the final optimal config: examine top-5 members by name + description
- Assign mechanism family name + coherence score (1–5)
- Target: 40–60 named mechanism families per node type
- Output: human-curated `mechanism_families.json`

**Script-generated taxonomy metadata:**
- `mechanism_taxonomy_summary.csv` — cluster_id, name, coherence, n_members, exemplar_node, top5_members
- `cluster_naming_validation.csv` — coherence scores, coverage

**Seeds from Step 3:**
- Full-corpus bridges: existential catastrophe, FDT, opacity, reward misspecification (`betweenness_bridge_clusters.csv`)
- Mechanism-space bridges: catastrophic misalignment, adversarial vulnerability, reward misspecification, opacity, RLHF (`betweenness_both09_bridge_clusters.csv`)

---

### B. 3-Level Cluster Connectivity — PRIMARY new analysis

**#27 Cross-Cluster Connectivity (cluster-level graph)**
- Build cluster-level directed graph: nodes = clusters, edges = structural EDGE connections between members of different clusters
- **Filter EDGE edges: `confidence >= 3`** (consistent with path generation; edge_data.pkl contains ALL confidence levels — must filter explicitly)
- For each EDGE edge (u → v) with confidence≥3, where u ∈ cluster C1 and v ∈ cluster C2 (C1 ≠ C2): record (C1_node_type, C2_node_type, C1_id, C2_id, edge_subtype)
- Levels: risk clusters (Level 1) → body clusters (Level 2: problem_analysis, theoretical_insight, design_rationale, implementation_mechanism, validation_evidence) → intervention clusters (Level 3)
- Config: unconstrained (to preserve x-risk hierarchy)

**3-level decomposition:**
- Identify which Level 1 risk clusters connect to which Level 2 body clusters via EDGE edges
- Identify which Level 2 body clusters connect to which Level 3 intervention clusters
- Within Level 1: x-risk cluster → secondary risk clusters (via EDGE and SIM edges)
- Output: `cluster_level_network_data.json`

**Cluster-level betweenness:**
- Compute betweenness on cluster-level graph (nodes = clusters, not individual nodes)
- Identifies which mechanism (body) clusters bridge the most risk↔intervention cluster pairs
- Fast (~seconds on ~200-node cluster graph)
- Output: `cluster_level_betweenness.csv`

**#29 Risk→Intervention Mapping Matrix:**
- Sparse matrix: (n_risk_clusters × n_intervention_clusters)
- Values: number of body/mechanism clusters on paths connecting each pair
- Identifies well-connected vs isolated risk↔intervention combinations
- Output: `risk_intervention_matrix.csv`, `connectivity_matrix_heatmap.png` (Plot 13)

---

### C. Network Visualisations

| # | Plot | Description | Output file |
|---|------|-------------|-------------|
| 11 | Cluster-level network | Force-directed layout, nodes=clusters (size=cluster_size, color=node_type), edges=EDGE connectivity (width=n_connections), 3-level layout (risk top, intervention bottom) | `cluster_level_network.png` |
| 12 | Sankey: Risk → Mechanism → Intervention | Top-20 flows by EDGE connection count, color by risk category | `sankey_risk_concept_intervention.png` |
| 13 | Risk→Intervention connectivity heatmap | From risk_intervention_matrix.csv | `connectivity_matrix_heatmap.png` |

---

### D. UMAP Projections (deferred from Step 2b)

| # | Description | Output |
|---|-------------|--------|
| 15 | UMAP: Risk nodes colored by cluster, overlay EDGE-only nodes with different marker | `umap_risks.png` |
| 16 | UMAP: Intervention nodes | `umap_interventions.png` |
| 17 | UMAP: All concepts | `umap_concepts.png` |

Config: SIM≥0.9 + both, agglomerative k=40 (from `optimal_configs_final.csv`)

---

### E. Path Sampling for Simulation

Two variants (run both, compare cluster results):

**Variant A — conservative (max_consec_SIM_full ≤ 1):** 75,008 source paths
**Variant B — broader (max_consec_SIM_full ≤ 2):** 432,776 source paths

Filter applied to the FULL path (risk preamble + body + intervention), not just the body.

For each variant:
- Sample 500–1000 representative pathways stratified by lifecycle stage (~33% design / 33% training / 33% deployment)
- Additional stratification: max 3 paths per mechanism family to avoid over-representing large clusters; sample across diverse start nodes (not all from same x-risk hub)
- Each record: node_id_sequence, node_names, categories, source_urls, cluster_assignments, EDGE_fraction, max_consec_SIM_full, mechanism_family_name (after #26)
- Export: `representative_pathways_consim1.jsonl` (variant A), `representative_pathways_consim2.jsonl` (variant B)

The EDGE-only baseline (3,405 fully structural paths) should also be extracted separately as the within-source ground truth: `representative_pathways_edgeonly.jsonl`.

---

### F. Additional Plots (from Phase2 plan)

| # | Plot | Source data |
|---|------|-------------|
| 18 | Cluster × Source diversity heatmap | `cluster_source_diversity_v2.csv` (Step 2b) |
| 19 | Cluster × Maturity distribution heatmap (interventions) | `maturity_per_cluster.csv` (Step 2b) |
| 21 | Within-cluster edge density heatmap | Built from cluster_memberships + edge_data |

---

### G. Simulation Prep

- Path→Prompt template: convert a pathway record to a structured LLM prompt
- Sample 10 test prompts: `simulation_prompts_sample.json`
- Export simulation-ready pathway database with cluster+mechanism metadata

---

## Items DROPPED from original plan

| Item | Reason |
|------|--------|
| node_migration_heatmap (Plot 20) | Migration rate is a metric artifact; explicitly excluded |
| Betweenness D3 (single_risk subgraph betweenness) | 3-level cluster connectivity analysis is the primary tool; node-level betweenness within constrained subgraph is redundant |
| Zig-zag path validation (Test 3) | single_risk no longer primary mode; unconstrained + EDGE filter is preferred |
| mode_stability_heatmap (Plot 22) | Already captured in Step 3 threshold sensitivity outputs |
| hub_quality_scatter (Plot 7, new version) | Step 2c completed; no new version needed |

---

## Output File Summary

| File | Type | Step 4 use |
|------|------|------------|
| `mechanism_families.json` | Human + script | Taxonomy: 40–60 named clusters |
| `mechanism_taxonomy_summary.csv` | Script | Cluster metadata table |
| `cluster_level_network_data.json` | Script | 3-level DAG topology |
| `cluster_level_betweenness.csv` | Script | Which mechanism clusters are structural bridges |
| `risk_intervention_matrix.csv` | Script | Risk↔intervention connectivity |
| `representative_pathways.jsonl` | Script | 500–1000 simulation-ready pathways |
| `simulation_prompts_sample.json` | Script | 10 test prompts |
| `cluster_level_network.png` | Script | Plot 11 |
| `sankey_risk_concept_intervention.png` | Script | Plot 12 |
| `connectivity_matrix_heatmap.png` | Script | Plot 13 |
| `umap_risks.png` / `umap_interventions.png` / `umap_concepts.png` | Script | Plots 15–17 |
| `cluster_naming_validation.csv` | Script | Coherence scores |
| `cluster_source_diversity_heatmap.png` | Script | Plot 18 |
| `maturity_distribution_heatmap.png` | Script | Plot 19 |
| `within_cluster_edge_density.png` | Script | Plot 21 |

---

## Inputs from Prior Steps

| File | Location | Use |
|------|----------|-----|
| `optimal_configs_final.csv` | `step3_validation_and_selection/` | Determines cluster config for naming |
| `cluster_memberships.pkl` | `step1_load_and_parse_umapwithoutlocalsatellites/` | Cluster assignments for taxonomy |
| `graph_node_attributes.pkl` | `step1_load_and_parse_umapwithoutlocalsatellites/` | Node names, descriptions, embeddings |
| `graph_edge_data.pkl` | `step1_load_and_parse_umapwithoutlocalsatellites/` | EDGE edges for connectivity |
| `category_mechanism_families.csv` | `step2_metrics_and_stability/` | Seed for manual naming |
| `betweenness_both09_bridge_clusters.csv` | `step3_validation_and_selection/` | Mechanism-space bridge seeds |
| `betweenness_bridge_clusters.csv` | `step3_validation_and_selection/` | Full-corpus bridge seeds |
| `edge_only_test_set.jsonl` | `step3_validation_and_selection/` | 100 structural ground-truth pathways |
| `paths_unconstrained_sim0.9.jsonl` | `phase1_rawpathsfiles/` | Source for representative pathway sampling |

---

## Execution Order

```
1. [Human] Manual cluster naming (#26)
   → produces mechanism_families.json

2. [Script] Build cluster-level graph (#27)
   → cluster_level_network_data.json
   → cluster_level_betweenness.csv

3. [Script] Risk→Intervention matrix (#29) + Sankey
   → risk_intervention_matrix.csv
   → sankey_risk_concept_intervention.png

4. [Script] UMAP projections
   → umap_risks/interventions/concepts.png

5. [Script] Representative pathway sampling
   → representative_pathways.jsonl
   → simulation_prompts_sample.json

6. [Script] Remaining heatmaps and plots
```

---

## Workshop Deliverables

- **Taxonomy:** 40–60 mechanism families with coherence validation → mechanism_families.json
- **Connectivity:** 3-level risk→mechanism→intervention DAG with Sankey diagram
- **Coverage:** Risk×Intervention matrix showing which risk types have intervention coverage
- **Embeddings:** UMAP projections showing cluster structure
- **Simulation DB:** 500–1000 representative pathways ready for preference simulation
