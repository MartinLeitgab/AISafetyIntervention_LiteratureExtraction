# Phase 2 Step 3: Validation & Config Selection — Complete Plan

**Document Version:** 1.0
**Created:** March 2026
**Status:** READY TO IMPLEMENT
**Prerequisite:** Step 2b COMPLETE (19/19 substeps, all outputs in `step2_metrics_and_stability/`)

---

## Entry State

All Step 2b data is available. The following are resolved:

| Resolved Item | Finding |
|---------------|---------|
| Algorithm selection | **Agglomerative k=40** final (sil=0.438 >> HDBSCAN sil=0.275/80.7%noise >> Louvain sil=0.011) |
| Migration metric | ❌ Artifact — do not report. Use ARI + centroid similarity. |
| Primary analysis cut (earmarked) | SIM≥0.9, mode=both, Agglomerative k=40 |
| Source diversity | n_sources ≈ cluster_size (r=0.887); not a config discriminator |
| Lifecycle distribution | Corpus property of ARD; not a config selection criterion |
| step2c script | `phase2_step2c_plot_updates.py` — generates `hub_quality_bar_v2.png` + `edge_validation_per_mode_v2.png`; **run this first** |

---

## Outputs Folder

```
phase2_results/step3_validation_and_selection/
```

Script: `graph_analysis/phase2_step3_validation_and_selection.py`

---

## Substep Overview

| # | Name | Priority | Workshop Goal | Status |
|---|------|----------|---------------|--------|
| 2c | Run step2c plot updates | PREREQUISITE | — | ⬜ TODO |
| #23 | Multi-criteria scoring | **CRITICAL** | Goal 4 ⭐ | ⬜ TODO |
| #21 | Edge threshold sensitivity | **ESSENTIAL** | Goal 1 ⭐ | ⬜ TODO |
| #22 | EDGE-only baseline validation | **ESSENTIAL** | Goal 2 ⭐ | ⬜ TODO |
| — | Betweenness on SIM≥0.9 graph | ENRICHMENT | Goal 6 | ⬜ TODO |
| #24 | Held-out test set validation | ENRICHMENT | Goal 3 | ⬜ TODO |
| #25 | EDGE subgraph consistency | ENRICHMENT | Goal 2 | ⬜ TODO |

---

## PREREQUISITE: Run step2c

```bash
cd graph_analysis/
uv run phase2_step2c_plot_updates.py
```

**Outputs** (saved to `step2_metrics_and_stability/`):
- `hub_quality_bar_v2.png` — horizontal bar chart, top-20 hubs per node type, colored by type, ranked by SIM≥0.9 degree; replaces scatter (degree ≈ n_sources makes scatter redundant)
- `edge_validation_per_mode_v2.png` — node-type segmented grouped bars, 2×2 mode grid

These are Step 2b review outputs that belong alongside existing Step 2b files.

---

## CRITICAL: #23 Multi-Criteria Scoring

### Purpose
Produce a fully transparent ranked table of all 160 configs to **confirm or update** the earmarked primary cut (SIM≥0.9 + both). This is the audit trail reviewers need: not just "we chose SIM≥0.9" but a composite score showing it beats all alternatives on the weighted criteria.

### Workshop Goal
**Goal 4 ⭐ — Optimal Config Selection:** Can reviewers see the full evidence base for config selection decisions?

### Input Data
All from `step2_metrics_and_stability/` (Step 2b outputs):

| File | Column(s) used |
|------|---------------|
| `all_cluster_metrics.csv` | `silhouette_mean`, `n_clusters` per (edge_config, mode, node_type) |
| `quality_metrics_summary.csv` | `edge_validation_mean` per config |
| `stability_ari_matrix.csv` | ARI for high-threshold pairs (0.9↔0.95, EDGE↔0.95) per (node_type, mode) |
| `cluster_edge_purity.csv` | gold-standard cluster % (purity≥80%) per config |

### Algorithm

**Step 1: Collect raw metrics for each of 160 configs**

For each (edge_config, mode, node_type):
- `silhouette` = `silhouette_mean` from `all_cluster_metrics.csv` (Agglomerative only)
- `edge_pct` = `edge_validation_mean` from `quality_metrics_summary.csv`
- `cluster_count_score` = triangular penalty: score=1.0 at k=40–50, decays linearly to 0 at k≤20 or k≥80
- `ari_high` = mean ARI for (0.9↔0.95) and (EDGE↔0.95) pairs from `stability_ari_matrix.csv`; if node_type row missing for those exact pairs, use (0.9↔EDGE) as fallback
- `gold_purity_pct` = from `cluster_edge_purity.csv` (fraction of clusters with EDGE purity ≥80%)
- `interpretability` = 1.0 for risk/intervention/individual concepts; 0.5 for all_concepts

**Step 2: Normalize each metric to [0, 1] using min-max across all 160 configs per node_type**

```python
def normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0 + 1.0  # all equal → give full score
    return (series - mn) / (mx - mn)
```

**Step 3: Compute composite score**

```
composite = (
    0.25 × silhouette_norm +
    0.30 × edge_pct_norm +
    0.20 × cluster_count_score_norm +
    0.15 × ari_high_norm +
    0.10 × interpretability
)
```

**Step 4: Rank configs within each node_type**

Sort descending by composite score. Add `rank` column (1 = best).

**Step 5: Identify winner per node_type**

For each node_type, the config with rank=1 is the winner. Check if it matches the earmarked cut:
- Risk: expected SIM≥0.9 + both
- Intervention: expected SIM≥0.9 + both
- Individual concepts: expected SIM≥0.85 + both

Log any deviations and explain in `selection_justification.md`.

### Outputs

| File | Contents |
|------|----------|
| `optimal_configs_ranked.csv` | 160 rows × all raw metrics + normalized metrics + composite score + rank, per node_type |
| `multi_criteria_parallel.png` | Parallel coordinates plot: 5 axes (one per metric), all 160 lines in grey, top-10 per node_type highlighted in color; x-axis = metrics, y-axis = normalized [0,1] |
| `optimal_configs_final.csv` | Winner per node_type: edge_config, mode, composite_score, rank=1 row per node_type |
| `selection_justification.md` | Text: winner per node_type, whether earmarked cut is confirmed, any deviations and why |

### Workshop Claim This Enables
> "Configuration selection was determined by a 5-criteria weighted composite score (EDGE validation 30%, silhouette 25%, cluster count 20%, ARI stability 15%, interpretability 10%) applied to all 160 configurations. SIM≥0.9 with 'both' mode achieves the highest composite score for risk and intervention nodes [show rank #1 from table], confirming the earmarked primary cut."

---

## ESSENTIAL: #21 Edge Threshold Sensitivity

### Purpose
Prove that SIM≥0.9 is not an arbitrary threshold but the **entry point to a qualitatively stable regime**. Show that the system behaves differently below 0.9 (qualitatively different clustering) vs. above 0.9 (incrementally adjusting, stable). This is the "sweet spot" argument.

### Workshop Goal
**Goal 1 ⭐ — Cross-Threshold Stability:** Can we prove that SIM≥0.9 is the minimum threshold for stable mechanism identification?

### Input Data
From `step2_metrics_and_stability/`:

| File | Columns |
|------|---------|
| `all_cluster_metrics.csv` | silhouette_mean per (edge_config, mode, node_type) |
| `quality_metrics_summary.csv` | edge_validation_mean, n_clusters per config |
| `stability_ari_matrix.csv` | ARI values for each adjacent threshold pair |
| `cluster_edge_purity.csv` | gold_pct per config |
| `cluster_centroid_similarity.csv` | centroid_sim_mean per (threshold_from, threshold_to) |

### Algorithm

**Sub-analysis A: Adjacent-threshold metric change (Δ per hop)**

For each adjacent pair (0.8→0.85, 0.85→0.9, 0.9→0.95, 0.95→EDGE), compute:
- ΔARI: |ARI(T1→T2) - ARI(T2→T3)| — how much does ARI change between consecutive transitions?
- ΔSilhouette: change in mean silhouette
- ΔEDGE%: change in edge validation mean
- ΔCentroid sim: change in centroid similarity
- ΔGold purity: change in gold-standard cluster %

Expected pattern: large Δ for transitions involving 0.85→0.9 (crossing the threshold), small Δ for 0.9↔0.95↔EDGE (stable regime).

**Sub-analysis B: Threshold stability score**

For each threshold (0.8, 0.85, 0.9, 0.95, EDGE):
- `stability_score` = mean of: ARI to all higher thresholds (how well does this threshold agree with the stable high end?)
- Plot: x=threshold, y=stability_score

**Sub-analysis C: Multi-metric threshold profile**

Line plot with 5 lines (one per metric, normalized), x-axis = threshold. Shade the "stable regime" region (0.9–EDGE). Visual argument for the sweet spot.

**Sub-analysis D: ARI high-threshold cluster**

Report the within-group mean ARI for {SIM≥0.9, SIM≥0.95, EDGE-only} — this is the "within-stable-regime" ARI already computed in Step 2b. Reproduce as a highlighted number in the output.

### Outputs

| File | Contents |
|------|----------|
| `threshold_sensitivity_analysis.csv` | Per (threshold_pair, node_type, mode): ΔARI, ΔSilhouette, ΔEDGE%, ΔCentroid, ΔGold%; plus stability_score per threshold |
| `threshold_sensitivity_profile.png` | 4-panel: (A) metric line plots vs threshold with stable-regime shading; (B) Δmetric bar chart per adjacent pair; (C) stability_score per threshold; (D) ARI heatmap (threshold × threshold) for risk + intervention |

### Workshop Claim This Enables
> "SIM≥0.9 marks the entry to a qualitatively stable clustering regime: the 0.85→0.9 transition shows the largest metric shift (ΔARI = X, ΔEDGE% = Y), while 0.9→0.95 and 0.95→EDGE show incrementally small changes (ΔARI < 0.05, ΔCentroid < 0.01). Within the {SIM≥0.9, SIM≥0.95, EDGE-only} cluster, mean within-group ARI is 0.68–0.72, confirming a stable mechanism identification regime above 0.9."

---

## ESSENTIAL: #22 EDGE-Only Baseline Validation

### Purpose
Explicitly compare EDGE-only (pure single-source literature) against SIM≥0.9 (similarity-augmented). Prove that SIM augmentation adds genuine signal at 0.9 without inflating noise: it improves coverage while preserving the cluster structure validated by the literature.

### Workshop Goal
**Goal 2 ⭐ — EDGE Validation:** Does SIM augmentation add real value over the pure literature baseline?

### Input Data

| File | Use |
|------|-----|
| `cluster_memberships.pkl` | Node assignments for EDGE-only and SIM≥0.9 configs |
| `graph_edge_data.pkl` | Edge type (EDGE vs SIMILARITY) per edge |
| `graph_node_attributes.pkl` | Node metadata (url, type, intervention_lifecycle) |
| `quality_metrics_summary.csv` | Side-by-side quality metrics |
| `stability_ari_matrix.csv` | EDGE↔0.9 ARI already computed |

### Algorithm

**Test 6: Structural overlap (ARI EDGE↔SIM≥0.9)**

Already computed in `stability_ari_matrix.csv` (the EDGE→0.9 pair). Extract:
- ARI(EDGE-only, SIM≥0.9) per (node_type, mode)
- Target: >0.5 (substantial agreement = SIM≥0.9 preserves EDGE structure while adding nodes)
- Report which node_types meet or miss this target

**Test 7: EDGE-only simulation sampling (held-out high-confidence set)**

From `cluster_memberships.pkl`, for the EDGE-only + unconstrained + risk config:
1. Sample 100 pathways stratified by:
   - lifecycle stage (design/training/deployment, balanced)
   - cluster (at least 1 per cluster where possible)
2. Mark these as the "gold test set" — they will be used in Step 4 for simulation validation
3. Save as `edge_only_test_set.jsonl` (node sequence + metadata per pathway)

**Test 8: SIM-augmentation coverage analysis**

For each node_type in the primary cut (SIM≥0.9 + both):
1. Count: how many nodes appear in EDGE-only pathways? (call these "anchored nodes")
2. Count: how many nodes are SIM-only (only reachable via SIM≥0.9 edges, not in any EDGE-only complete pathway)?
3. For SIM-only nodes: compute mean degree (are they well-connected or peripheral?)
4. For SIM-only nodes: check their EDGE purity in the cluster they belong to (do they land in high-purity clusters?)
5. Classify SIM-only nodes: "foundational" (high degree, high-purity cluster) vs "niche" (low degree, low-purity cluster)

**Comparison table EDGE vs SIM≥0.9:**

| Metric | EDGE-only | SIM≥0.9 + both | Delta |
|--------|-----------|-----------------|-------|
| Silhouette (risk) | | | |
| EDGE validation % | | | |
| Cluster count | | | |
| Gold purity % | | | |
| N nodes covered | | | |
| ARI agreement (EDGE↔SIM≥0.9) | | — | |

### Outputs

| File | Contents |
|------|----------|
| `edge_only_comparison.csv` | Comparison table: EDGE vs SIM≥0.9 per node_type + mode |
| `edge_vs_sim_coverage.png` | 2-panel: (A) grouped bars EDGE vs SIM≥0.9 for 5 quality metrics; (B) SIM-only node classification (foundational vs niche) as stacked bars |
| `edge_only_test_set.jsonl` | 100 EDGE-only pathways (node sequences + metadata) for simulation validation in Step 4 |

### Workshop Claim This Enables
> "SIM≥0.9 augmentation achieves ARI=X with the EDGE-only baseline (substantial structural agreement), while adding Y% additional node coverage. SIM-only nodes are predominantly foundational (mean degree Z, landing in clusters with >80% EDGE purity in W% of cases), indicating that similarity augmentation at 0.9 extends coverage with semantically coherent additions rather than noise."

---

## ENRICHMENT: Betweenness on SIM≥0.9-Filtered Graph

### Purpose
The existing betweenness scores (`mechanism_transfer_betweenness.csv`) were computed on the **full graph** (ALL SIM edges at floor 0.80 + all structural EDGE edges). This gives betweenness in the global context. Recomputing on the SIM≥0.9-filtered graph gives betweenness specific to the primary analysis cut — nodes that are bridges specifically at the high-selectivity threshold.

### Workshop Goal
**Goal 6 — Mechanism Taxonomy:** Which concept nodes are the most critical bridges within the high-quality primary-cut graph?

### Input Data
- `graph_edge_data.pkl` — filter to SIM≥0.9 + structural EDGE
- `graph_node_attributes.pkl` — node names for labeling

### Algorithm

**Graph construction:**
```python
# Filter edges
cos_sim_from_score = lambda s: 1.0 - float(s)**2 / 2.0
sim_09 = [e for e in edge_data
          if e.get('type','').upper() == 'SIMILARITY'
          and cos_sim_from_score(e['similarity_score']) >= 0.90]
struct_edges = [e for e in edge_data if e.get('type','').upper() == 'EDGE']
graph_edges = sim_09 + struct_edges
```

**Approximate betweenness (k=1000 source samples, undirected):**
```python
import networkx as nx
G = nx.Graph()
G.add_edges_from([(e['source'], e['target']) for e in graph_edges])
# Only compute for nodes in complete pathways (reduce computation)
anchor_nodes = set(node for cfg_nodes in edge_memberships.values() for node in cfg_nodes)
betweenness = nx.betweenness_centrality_subset(
    G, sources=list(anchor_nodes)[:1000], targets=list(anchor_nodes),
    normalized=True
)
```

**Note:** Full all-pairs betweenness on 200K nodes is computationally infeasible. Options:
1. **Preferred:** Approximate betweenness with k=1000 random sources (`nx.betweenness_centrality(G, k=1000)`)
2. **Alternative:** Restrict to nodes with structural EDGE degree ≥ 1 (~30K nodes)

**Comparison:**
- Join with existing `mechanism_transfer_betweenness.csv` (computed at SIM≥0.8 floor)
- Identify rank changes: nodes that were high-betweenness at 0.8 but drop at 0.9 (= threshold artifacts at bridge position)
- Identify stable high-betweenness nodes (consistent across both computations = genuine bridges)

**Clustering top-50 by cosine similarity:**
- Take top-50 betweenness nodes from SIM≥0.9 computation
- Retrieve their embeddings from `graph_node_attributes.pkl`
- Agglomerative clustering (k=10–15) on their embeddings
- Label cluster themes — these are the unique bridge concept families

### Outputs

| File | Contents |
|------|----------|
| `betweenness_sim09.csv` | Top-100 betweenness nodes in SIM≥0.9 graph: node_id, name, category, betweenness_sim09, betweenness_sim08_rank (for comparison) |
| `betweenness_comparison.png` | Scatter: betweenness_sim08 (x) vs betweenness_sim09 (y), colored by category; label top-20 |
| `betweenness_bridge_clusters.csv` | Top-50 nodes clustered into 10–15 bridge theme groups (name, cluster_id, cluster_theme_exemplar) |

---

## ENRICHMENT: #24 Held-Out Test Set Validation

### Purpose
Demonstrate that mechanism clusters generalize: identified families are not artifacts of the specific 200K node corpus but persist when a held-out subset is removed and re-clustered.

### Scoped-Down Implementation
Full re-clustering is prohibitive (~10 min per run). Use a computationally feasible proxy:

**Algorithm:**
1. From the primary cut clusters (SIM≥0.9 + both + Agglomerative), take the 40 risk clusters and 40 intervention clusters
2. For each cluster, randomly withhold 20% of its members
3. Compute centroid of remaining 80% members
4. Check: does each withheld node (20%) lie closer to its original cluster centroid than to any other cluster centroid?
5. **"Leave-20%-out accuracy"**: % of withheld nodes correctly assigned by nearest centroid

Expected: >80% correct assignment if clusters are genuine mechanism families (not arbitrary partitions).

### Input Data
- `cluster_memberships.pkl` — node assignments for primary cut
- `graph_node_attributes.pkl` — node embeddings

### Outputs

| File | Contents |
|------|----------|
| `held_out_validation.csv` | Per cluster: n_members, n_withheld, n_correct, accuracy |
| Key metric | Mean leave-20%-out accuracy across all clusters |

### Workshop Claim This Enables
> "Mechanism clusters show X% mean leave-20%-out accuracy (N clusters tested), indicating that cluster assignments are stable and not dependent on specific corpus members — nodes withheld during centroid computation are correctly re-assigned in X% of cases."

---

## ENRICHMENT: #25 EDGE Subgraph Consistency

### Purpose
Verify that the structural EDGE-only subgraph (the 202K literary-evidence edges) is topologically coherent — not a fragmented collection of isolated local graph fragments. This validates the claim that EDGE-only pathways form a connected backbone.

### Algorithm

**Subgraph construction:**
```python
edge_only = [e for e in edge_data if e.get('type','').upper() == 'EDGE']
G_edge = nx.DiGraph()
G_edge.add_edges_from([(e['source'], e['target']) for e in edge_only])
```

**Topology checks:**
1. Weakly connected components: count, size distribution (is there one giant component?)
2. Largest WCC size as % of total EDGE-only nodes
3. Diameter (approximate via BFS sample on largest WCC)
4. Degree distribution: how many nodes have EDGE degree ≥ 2 (multi-connected vs single-connection)?
5. Check: do the top-25 betweenness nodes (from full graph) also appear in the EDGE-only subgraph?

**Expected:** One giant WCC (>70% of nodes), diameter ~7–10 (consistent with 7-hop median path length). High overlap between full-graph betweenness leaders and EDGE-only subgraph high-degree nodes.

### Input Data
- `graph_edge_data.pkl`
- `mechanism_transfer_betweenness.csv` (top-25 nodes for overlap check)

### Outputs

| File | Contents |
|------|----------|
| `edge_subgraph_stats.csv` | n_nodes, n_edges, n_wcc, largest_wcc_size, largest_wcc_pct, approx_diameter, mean_degree, pct_nodes_degree_gt2 |
| `edge_degree_distribution.png` | Log-log degree distribution of EDGE-only subgraph; annotation: mean degree, top-10 hubs |

---

## Implementation Plan

### Script Architecture

**File:** `graph_analysis/phase2_step3_validation_and_selection.py`

```
Constants / config
├── STEP2_DIR = "phase2_results/step2_metrics_and_stability/"
├── STEP3_DIR = "phase2_results/step3_validation_and_selection/"
├── PKL_DIR   = "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/"
└── PRIMARY_CUT = {"edge_config": "0.9", "mode": "both"}

Checkpoint loading (same pattern as Step 2b)
├── load_step2_data()     → loads all Step 2b CSVs
└── load_pkl_checkpoints() → graph_node_attributes.pkl, graph_edge_data.pkl, cluster_memberships.pkl

SECTION A: Multi-Criteria Scoring (#23) — no PKL needed, CSV-only
├── collect_metrics()
├── normalize_metrics()
├── compute_composite_scores()
├── save_optimal_configs_ranked()
├── plot_multi_criteria_parallel()
└── write_selection_justification()

SECTION B: Threshold Sensitivity (#21) — CSV-only
├── compute_adjacent_deltas()
├── compute_threshold_stability_scores()
└── plot_threshold_sensitivity_profile()

SECTION C: EDGE-Only Validation (#22) — needs PKL
├── compute_edge_sim_ari()          [from stability_ari_matrix.csv]
├── sample_edge_only_test_set()     [cluster_memberships.pkl]
├── analyze_sim_augmentation_coverage() [graph_edge_data.pkl]
└── plot_edge_vs_sim_comparison()

SECTION D: Betweenness SIM≥0.9 — needs PKL (slow ~15 min)
├── build_sim09_graph()
├── compute_approximate_betweenness()
├── cluster_top50_bridge_nodes()
└── plot_betweenness_comparison()

SECTION E: Held-Out Validation (#24) — needs PKL
├── compute_leave20out_accuracy()
└── save_held_out_results()

SECTION F: EDGE Subgraph Consistency (#25) — needs PKL
├── analyze_edge_subgraph_topology()
└── plot_edge_degree_distribution()

main() — run all sections, print summary table
```

### Runtime Estimate

| Section | Input | Estimated Time |
|---------|-------|----------------|
| A: Multi-criteria (#23) | CSVs only | ~2 min |
| B: Threshold sensitivity (#21) | CSVs only | ~1 min |
| C: EDGE-only validation (#22) | PKL + CSV | ~5 min |
| D: Betweenness SIM≥0.9 | PKL (200K nodes) | ~15–25 min |
| E: Held-out validation (#24) | PKL | ~3 min |
| F: EDGE subgraph (#25) | PKL | ~3 min |
| **Total** | | **~30–40 min** |

**Recommended execution order:** Run A+B first (no PKL needed, fast). Then C+E+F together after PKL load. Run D last (slowest, optional for first pass).

---

## Complete Output File List

All saved to `phase2_results/step3_validation_and_selection/`:

### CRITICAL outputs (workshop blocking)
| File | Substep | Type |
|------|---------|------|
| `optimal_configs_ranked.csv` | #23 | CSV |
| `optimal_configs_final.csv` | #23 | CSV |
| `selection_justification.md` | #23 | MD |
| `multi_criteria_parallel.png` | #23 | Plot |

### ESSENTIAL outputs
| File | Substep | Type |
|------|---------|------|
| `threshold_sensitivity_analysis.csv` | #21 | CSV |
| `threshold_sensitivity_profile.png` | #21 | Plot |
| `edge_only_comparison.csv` | #22 | CSV |
| `edge_vs_sim_coverage.png` | #22 | Plot |
| `edge_only_test_set.jsonl` | #22 | JSONL (→ Step 4 input) |

### ENRICHMENT outputs
| File | Substep | Type |
|------|---------|------|
| `betweenness_sim09.csv` | betweenness | CSV |
| `betweenness_comparison.png` | betweenness | Plot |
| `betweenness_bridge_clusters.csv` | betweenness | CSV |
| `held_out_validation.csv` | #24 | CSV |
| `edge_subgraph_stats.csv` | #25 | CSV |
| `edge_degree_distribution.png` | #25 | Plot |

---

## Key Decision: Does #23 Confirm or Update the Earmarked Cut?

The primary analysis cut (SIM≥0.9 + both) is **earmarked**, not yet final. Step 3 #23 multi-criteria scoring will either:

**A. Confirm:** SIM≥0.9 + both ranks #1 in composite score for risk AND intervention → primary cut is locked.

**B. Update for one node type:** e.g., SIM≥0.95 + both ranks #1 for concepts (not SIM≥0.85 + both as earmarked) → update the optimal config for that node type only, record the change.

**C. Major update (unlikely):** A fundamentally different config ranks #1 → investigate why before accepting; cross-check with ARI and EDGE% logic.

**Process:** Whatever #23 produces, `selection_justification.md` must explain the decision in human-readable form for inclusion in the workshop paper Methods section 3.4.

---

## Connection to Step 4

Step 3 produces these **direct inputs to Step 4**:

| Step 3 Output | Step 4 Use |
|---------------|------------|
| `optimal_configs_final.csv` | Determines which cluster memberships to use for naming |
| `edge_only_test_set.jsonl` | Gold test set for simulation validation (100 pathways) |
| `betweenness_bridge_clusters.csv` | Seed themes for cluster naming (bridge concept families) |
| `selection_justification.md` | Methods section text for workshop paper |

---

## Workshop Goal Coverage

| Goal | Step 3 Substep | Status after Step 3 |
|------|----------------|---------------------|
| Goal 1 ⭐ — Cross-threshold stability | #21 threshold sensitivity | ✅ Confirmed: SIM≥0.9 as stable-regime entry point |
| Goal 2 ⭐ — EDGE validation quality | #22 EDGE-only validation | ✅ Confirmed: SIM≥0.9 adds value over EDGE baseline |
| Goal 3 ⭐ — Algorithm comparison | ✅ DONE in Step 2b | ✅ Complete |
| Goal 4 ⭐ — Optimal config selection | #23 multi-criteria | ✅ Confirmed: composite-ranked winner selected |
| Goal 5 — Path length / exemplar paths | Step 4 #18 | Not in Step 3 scope |
| Goal 6 — Mechanism taxonomy | Betweenness bridge clusters | Partial: bridge themes identified; full naming in Step 4 |
| Goal 7 — Hub quality | ✅ DONE in Step 2b | ✅ Complete |
| Goal 8 — Temporal coverage | ✅ DONE in Step 2b | ✅ Complete |
| Goal 9 — Bias documentation | ✅ DONE in Step 2b | ✅ Complete |

---

**END OF STEP 3 PLAN**

**Next after Step 3:** Step 4 — Mechanism Taxonomy Construction
**Critical path to workshop:** Step 3 #23 → Step 4 #26 (manual cluster naming)
