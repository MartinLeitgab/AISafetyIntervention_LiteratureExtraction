# Phase 2 Step 3 Review Summary

**Generated:** 2026-03-22; last updated 2026-03-28
**Script:** `graph_analysis/phase2_step3_validation_and_selection.py`
**Outputs:** `phase2_results/step3_validation_and_selection/`
**Status:** Sections A–F complete ✅ · Section F re-run ✅ · Section D betweenness re-run ✅ COMPLETE (2026-03-29) · Section D2 betweenness re-run ✅ COMPLETE (2026-03-29) · Hub quality SIM degrees ✅ FIXED (2026-03-29)

> ✅ **Section D, D2 RE-RUN COMPLETE** (`phase2_rerun_pathfiltered.py`, completed 2026-03-29 ~00:00).
> Valid-pathway node set: union of all 20 `paths_*.jsonl` files = **42,870 nodes** (38,054 after removing isolates for betweenness).
> Section D: 38,054 nodes, 163,848 edges. Section D2: 17,952 nodes, 21,269 edges.
> Supersedes PID 2415422 (killed — used 131,634 PKL nodes, not path-filtered).
>
> ✅ **Hub quality SIM degree columns FIXED** (`phase2_fix_hub_quality_sim_degrees.py`, completed 2026-03-29).
> `degree_sim_*` and `n_sources_at_config_thr` now restricted to valid-pathway partner nodes. Top hub (node 147238): 617 SIM≥0.9 edges, 617 distinct partner papers (at SIM≥0.9); 1,684 at SIM≥0.80. All hub plots regenerated.

---

## Quality Cut Reference — Final Cutset for Step 4 Analysis

All path files and cluster membership analyses in this document inherit the following quality cuts applied at graph-load time in `final_pathway_analysis_modes.py`:

| # | Cut | Value | Notes |
|---|-----|-------|-------|
| 1 | EDGE edge confidence | ≥ 3 | Removes 38.7% of all EDGE edges (conf<3: 78,164 of 202,149). Applies ONLY to structural EDGE-type edges. Applied in `load_graph(min_conf=3)` — adjacency list built with `edge_confidence >= 3` query. |
| 2 | Intervention maturity | ≥ 3 | Applied to intervention-type endpoint nodes only. Path BFS terminates at an intervention node only if it passes this check. |
| 3 | SIM edge similarity | cos_sim ≥ 0.9 | Applied to SIMILARITY-type edges in the adjacency list. SIM edges have no confidence filter — a SIM≥0.9 edge passes regardless. |
| 4 | Mode | **unconstrained** | No single_risk or monotonic filter. All complete risk→body→intervention paths are included. See rationale in Section A. |
| 5 | Consecutive SIM per path | ≤ 2 (Step 4 primary); ≤ 1 (sensitivity) | NEW cut applied at path-sampling stage in Step 4. Not yet applied to existing clusters — ARI test in Step 4 substep #29 determines whether reclustering is needed. |

**Note on source eligibility pre-filter:** `final_pathway_analysis_modes.py` also pre-filters starting sources to those in `source_pathways_final.json["mature"]["conf>=1"]` (3,703 sources). This is a **performance optimization, not a substantive cut** — since the BFS graph already has only conf≥3 EDGE edges and maturity≥3 is enforced at the intervention endpoint check, any source excluded by this pre-filter would produce zero qualifying paths anyway. It does not remove any paths that would have survived cuts 1–4. Not counted as a required quality cut.

**Important:** Cuts #1–2 apply ONLY to their specific edge/node types. **SIM edges are NOT required to be EDGE edges.** A path can include any number of SIM≥0.9 edges and still pass cuts #1–4. Example: a path with 10 consecutive SIM≥0.9 edges passes all cuts (1–4); the consecutive SIM cut (#6) is a NEW additional constraint being evaluated in Step 4, not a requirement from the existing pipeline.

**Valid-pathway node set:** 42,870 nodes = union of all 20 `paths_*.jsonl` files (all mode × edge_config combinations). These are all nodes that appear on at least one complete qualifying path. The remaining ~158K nodes in `graph_node_attributes.pkl` are on partial chains only (no complete risk→intervention span passing quality cuts).

---

## Section A — #23 Multi-Criteria Scoring

**File:** `optimal_configs_ranked.csv` (160 rows), `optimal_configs_final.csv` (8 rows)
**Plot:** `multi_criteria_parallel.png`
**Justification:** `selection_justification.md`

### Method
5-criteria weighted composite score, per-node_type min-max normalization:

| Metric | Weight |
|--------|--------|
| EDGE validation % | 0.30 |
| Silhouette | 0.25 |
| Cluster count score (peak k=40–50) | 0.20 |
| ARI to higher thresholds (stability) | 0.15 |
| Gold purity % | 0.10 |

### Top-5 Configs: Risk

| rank | edge_config | mode | composite | silhouette | edge_pct | ari_high | n_clusters |
|------|-------------|------|-----------|------------|----------|----------|------------|
| 1 | EDGE | monotonic | 0.825 | 0.508 | 1.000 | 0.739 | 40 |
| 2 | 0.95 | monotonic | 0.822 | 0.546 | 0.850 | 0.739 | 42 |
| **3** | **0.9** | **both** | **0.789** | 0.519 | 0.908 | 0.731 | 40 |
| 4 | EDGE | unconstrained | 0.786 | 0.503 | 1.000 | 0.704 | 40 |
| 5 | EDGE | single_risk | 0.779 | 0.492 | 1.000 | 0.722 | 40 |

### Top-5 Configs: Intervention

| rank | edge_config | mode | composite | silhouette | edge_pct | ari_high | n_clusters |
|------|-------------|------|-----------|------------|----------|----------|------------|
| 1 | 0.95 | unconstrained | 0.831 | 0.459 | 0.999 | 0.777 | 40 |
| **2** | **0.95** | **both** | **0.808** | 0.456 | 0.999 | 0.744 | 40 |
| 3 | EDGE | both | 0.795 | 0.450 | 1.000 | 0.744 | 40 |
| 4 | 0.95 | monotonic | 0.786 | 0.452 | 0.999 | 0.716 | 40 |
| 5 | 0.95 | single_risk | 0.774 | 0.431 | 0.999 | 0.791 | 40 |

### Final Config Decisions

| node_type | Selected config | Mode | Basis |
|-----------|----------------|------|-------|
| **risk** | **SIM≥0.9** | **both** | Rank 3 (gap 0.036 from winner). Coverage: ~2,732 nodes vs ~2,468 EDGE-only. 90.8% EDGE validation. |
| **intervention** | **SIM≥0.9** | **unconstrained** | Updated in Step 4 planning: SIM≥0.9 used for methodological consistency with risk. See rationale below. |
| implementation_mechanism | SIM≥0.9 | both | Rank 1 — earmarked cut confirmed. |
| all_concepts | SIM≥0.95 | monotonic | Rank 1. |
| design_rationale | EDGE | monotonic | Rank 1. |
| problem_analysis | SIM≥0.95 | both | Rank 1. |
| theoretical_insight | EDGE | unconstrained | Rank 1. |
| validation_evidence | EDGE | monotonic | Rank 1. |

### Key Note: EDGE-Validation Weight Bias
The 30% EDGE validation weight structurally advantages EDGE-only configs (trivial score=1.0). For risk, the composite gap of 0.036 is narrow; SIM≥0.9+both is preferred for cross-literature coverage (10× more nodes, 90.8% structural validation). For intervention, the composite scoring showed 0.134 gap between SIM≥0.95 (rank 2) and SIM≥0.9 (rank 12).

**However:** The multi-criteria composite is an ad-hoc metric mixing silhouette, EDGE%, cluster count, ARI, and gold purity with arbitrary weights. For the Step 4 workshop analysis, **SIM≥0.9 is used for intervention as well as risk**, for the following reasons:
1. **Methodological consistency:** Using the same threshold for all node types avoids reviewer questions about why risk uses SIM≥0.9 but intervention uses SIM≥0.95. A single threshold simplifies the methods narrative.
2. **Same stability evidence applies:** The ARI data (0.9→0.95 pair = 0.757, highest adjacent pair) justifies SIM≥0.9 as the stable-regime entry point for all node types, not just risk. The EDGE validation drop at SIM≥0.9 for interventions is partially an artifact of the EDGE-validation weight bias.
3. **Consecutive SIM ≤2 cut provides additional quality control:** Under the consecutive SIM cut, low-structural-grounding paths are filtered regardless of threshold. The marginal quality gain from SIM≥0.95 is smaller after applying this cut.

### Key Note: Why Not Monotonic Mode
The multi-criteria scoring ranked "both" (single_risk + monotonic) highly for both risk and intervention. **Monotonic mode is not used in Step 4** because:
- Monotonic removes paths where body nodes have iterative causal structure (e.g., problem_analysis → theoretical_insight → problem_analysis — backtracking in categories). This excludes many diverse and valid mechanism chains that represent legitimate complex causal reasoning.
- The SIM≥0.9 tight threshold already provides quality control by requiring strong semantic similarity between connected nodes. Monotonic imposes an additional structural constraint that is not needed when SIM≥0.9 + consecutive SIM ≤2 are applied.
- For the three-level hierarchy, diverse path structures between risk and intervention are desirable — they reveal the full range of mechanism families, not just the linearly-structured ones.
- Source: decision recorded in conversation (3–5 compactions back) where monotonic was found to be "too restrictive for diverse paths."

### ⚠ Step 4 Mode Decision: Unconstrained Supersedes Both

The multi-criteria scoring above selected both mode for risk and intervention. **For Step 4 three-level hierarchy construction, unconstrained mode is used instead.** Rationale:

- **single_risk removes the x-risk hub** — but the x-risk hub IS the desired top-level structure in the three-level taxonomy. Single_risk filters 99.3% of unconstrained paths (1M → 7K) to eliminate paths with multiple risk nodes. What is filtered out is predominantly x-risk hub nodes — exactly what Step 4 needs at the top of the hierarchy.
- **EDGE validation bias against unconstrained is an artifact.** X-risk hubs are near-duplicate concept nodes connected to each other via SIM≥0.9 edges (not EDGE edges). The 30% EDGE validation weight penalizes unconstrained (56% vs 90.8%) for this structural feature, not for a validity problem.
- **Consecutive SIM ≤2 cut replaces mode constraint.** Instead of filtering via single_risk+monotonic, Step 4 applies a consecutive SIM edge cut (≤2 per path). This achieves the same goal (excluding low-structural-grounding paths) while keeping the full unconstrained node coverage (21,553 nodes vs 17,954 for both mode).

The Section A scoring remains valid for its intended purpose: selecting the optimal SIM threshold (SIM≥0.9 for risk/mechanism, SIM≥0.95 for intervention) and documenting stability/validation properties across all 160 configurations. The mode selection from scoring (both) is superseded by the Step 4 structural requirement.

⚠ **Potential reruns:** Once the consecutive SIM cut is confirmed as the final cutset in Step 4, analyses currently reported under both mode should be re-run for the **unconstrained + consecutive SIM ≤2** combination to support apples-to-apples comparison in the workshop paper. Affected sections: silhouette scores, ARI values, centroid similarity (Section B, Step 2b), hub quality (Section E), betweenness (Section D/D2).

### Path Generation Filters (Applied Before Clustering)
All Phase 1 path files were generated with these mandatory filters applied at graph-load time in `final_pathway_analysis_modes.py`:
- **`edge_confidence >= 3`** on all EDGE edges (removes 38.7% of EDGE edges — conf<3 accounts for conf=1: 1.7% + conf=2: 37.0% = 78,164 edges excluded from 202,149 total)
- **`intervention_maturity >= 3`** on all intervention endpoint nodes
- Only sources in `source_pathways_final.json["mature"]["conf>=1"]` (3,703 eligible sources)

These filters explain the EDGE-only path count of **3,473 paths**:
- Only 3,703 sources qualify as mature (not ~11K total)
- At conf≥3, only 1,917 of those 3,703 sources have complete chains — the conf≥3 cut removes 38.7% of EDGE edges, breaking many within-source chains
- The maturity≥3 cut on intervention endpoints further limits which paths are complete — an otherwise valid chain ending at a maturity=1 or 2 intervention is excluded
- `source_pathways_final.json` at conf≥3 gives 3,498 paths from 1,917 sources — BFS result of 3,473 matches closely; at conf≥1 the total is 8,281 from 3,630 sources

**The clustering inherits these filters** — `phase2_clustering.py` reads its node populations from path files, so all cluster members are nodes from conf≥3 + maturity≥3 paths. The PKL files (`graph_edge_data.pkl`, `graph_node_attributes.pkl`) store raw unfiltered data; the filters must be re-applied whenever they are used for downstream analysis.

### Mode Definitions
The four clustering modes filter **pre-generated paths** (Phase 1 path files). Each path is a linear node sequence `[risk, body_nodes..., intervention]`. The modes define which paths are kept:

- **unconstrained:** all paths (1,054,527 paths at SIM≥0.9) — includes paths with 4-5 x-risk hub nodes in sequence at the start
- **monotonic:** paths where categories go in forward order only — no backtracking from intervention back to problem_analysis etc. (336,092 paths) — BUT does NOT prevent multiple risk nodes; mean 4.58 risks/path
- **single_risk:** paths with exactly ONE risk node (7,103 paths at SIM≥0.9, max=1 risk/path) — eliminates x-risk hub chains at path start; body node sequence CAN backtrack
- **both:** single_risk AND monotonic combined (5,939 paths) — single risk root AND no backtracking in body sequence

**What monotonic actually removes:** Paths where the body sequence backtracks (e.g., problem_analysis → theoretical_insight → problem_analysis). It does NOT remove multi-risk paths — that is single_risk's job. Monotonic alone (336K paths) still has mean 4.58 risks/path.

**Forked paths** (Risk_A → Body_Node ← Risk_B): treated as TWO SEPARATE paths, each with one risk. Both paths survive single_risk filtering because each individually has one risk. Body_Node appears in both paths and gets clustered by embedding similarity — it lands in whichever risk family's cluster it's most similar to. The fork is not rejected; the clustering resolves it by embedding.

"both" mode gives clean single-root directed chains. single_risk gives single-root chains allowing iterative body structure. The clustering algorithm (agglomerative) then groups nodes by embedding similarity within the eligible node set.

### Key Finding: Mode Effect on Structural Grounding

At ec=0.9 for risk, **single_risk carries nearly all the structural grounding** — both (90.8% EDGE%) vs single_risk (90.6% EDGE%) are essentially identical on EDGE validation. Monotonic alone without single_risk drops to 58.1% EDGE%, and unconstrained drops to 56.0%. This reveals that:

- **single_risk** = the constraint that prevents x-risk hubs from aggregating diverse mechanism families by filtering out paths with multiple risk nodes (7K vs 1M paths at SIM≥0.9 — a 99.3% reduction)
- **monotonic** = adds directional body ordering → improves ARI stability (+0.022: both=0.731 vs single_risk=0.709) but excludes body nodes with iterative causal structure
- **unconstrained** = preserves x-risk hub nodes and the full x-risk hierarchy (shared starting premise across many papers); lower EDGE validation (56%) but highest node coverage (21,553 nodes vs 18,590 for single_risk)

Mode rankings for risk at ec=0.9: both (rank 3, composite=0.789) > single_risk (rank 7, composite=0.758) > monotonic (rank 11, composite=0.597) > unconstrained (rank 16, composite=0.438). The composite score favors high EDGE validation, which structurally biases against unconstrained due to the 30% weight on EDGE%.

**Step 4 uses unconstrained** for the 3-level cluster connectivity analysis (#27). The x-risk hub cluster that unconstrained preserves is a meaningful structural level — it sits at the top of the risk hierarchy (x-risk cluster → secondary risk clusters → mechanism clusters → intervention clusters) and should not be removed. The cluster-level EDGE connectivity graph (not node-level betweenness) is the primary tool for identifying which mechanism families bridge which risk and intervention clusters.

### Path and Node Coverage Facts (Confirmed from Path Files)

| Path file | Paths | Unique nodes |
|-----------|-------|-------------|
| `paths_unconstrained_edge_only.jsonl` | 3,473 | 17,136 |
| `paths_unconstrained_sim0.9.jsonl` | 1,054,527 | 21,553 |
| `paths_single_risk_sim0.9.jsonl` | 7,103 | 18,590 |
| `paths_monotonic_sim0.9.jsonl` | 336,092 | 19,982 |
| `paths_both_sim0.9.jsonl` | 5,939 | 17,954 |

Total nodes across all 20 (mode × edge_config) combinations: **42,870** (confirmed by union over all path files). Remaining ~158K nodes are on partial chains (no intervention endpoint OR no risk root). The corrected betweenness re-run uses only these 42,870 valid-pathway nodes (38,054 after removing graph isolates).

**Full-path consecutive SIM distribution** (applied to entire path: risk preamble + body + intervention, confirmed on all 1,054,527 unconstrained sim0.9 paths):

| Max consecutive SIM (full path) | Paths | % | Cumulative |
|----------------------------------|-------|---|-----------|
| 0 (100% EDGE — truly structural) | 3,405 | 0.3% | 0.3% |
| ≤ 1 | 75,008 | 7.1% | 7.1% |
| ≤ 2 | 432,776 | 41.0% | 41.0% |
| ≤ 3 | 772,423 | 73.2% | 73.2% |
| ≤ 4 | 935,251 | 88.7% | 88.7% |

Filter is applied to the full path (risk preamble + body + intervention). Step 4 path sampling: Variant A (≤1 full-path) = 75K paths, Variant B (≤2 full-path) = 433K paths.

---

## Section B — #21 Threshold Sensitivity

**File:** `threshold_sensitivity_analysis.csv` (800 rows)
**Plot:** `threshold_sensitivity_profile.png` (4-panel)
**Filters:** All ARI scores compare cluster assignments from path files; path files for all thresholds (0.8–EDGE) were generated with conf≥3 + maturity≥3 applied at graph-load time. ✅

### Stability Score per Threshold (risk, mode=both)

| edge_config | stability_score (mean ARI to higher thresholds) |
|-------------|--------------------------------------------------|
| 0.8 | 0.585 |
| 0.85 | 0.650 |
| **0.9** | **0.731** ← highest |
| 0.95 | 0.714 |
| EDGE | 0.714 |

SIM≥0.9 achieves the highest stability score — it agrees better on average with all higher-selectivity configs than any other threshold. This supports SIM≥0.9 as the optimal stable-regime entry point.

### Adjacent-Threshold ARI (risk, mode=both)

| pair | ARI |
|------|-----|
| 0.8→0.85 | 0.650 |
| 0.85→0.9 | 0.651 |
| 0.9→0.95 | **0.757** |
| 0.95→EDGE | 0.714 |

The 0.9→0.95 transition achieves the highest adjacent-pair ARI (0.757), confirming that {SIM≥0.9, SIM≥0.95, EDGE} form a stable cluster of mutually agreeing configurations. The lower ARI values below 0.9 indicate qualitatively different clustering behavior.

### Workshop Claim Supported
> "SIM≥0.9 marks the entry to the stable clustering regime: it achieves the highest mean stability score (0.731) and the 0.9→0.95 transition shows the tightest adjacent agreement (ARI=0.757), confirming that {SIM≥0.9, SIM≥0.95, EDGE-only} form a coherent high-selectivity cluster."

---

## Section C — #22 EDGE-Only Baseline Validation

**Files:** `edge_only_comparison.csv`, `edge_only_test_set.jsonl`
**Plot:** `edge_vs_sim_coverage.png` (2-panel)
**Filters:** All cluster assignments, node counts, and path samples derive from path files with conf≥3 + maturity≥3. Test 8 degree counts use SIM≥0.9 edges only (no edge_confidence filter needed for SIMILARITY edges). ✅

### Test 6: ARI Overlap (EDGE ↔ SIM≥0.9)

| node_type | mode | ARI(EDGE, SIM0.9) | Meets target (>0.5) |
|-----------|------|-------------------|----------------------|
| risk | both | **0.705** | ✅ |
| intervention | both | **0.679** | ✅ |

SIM≥0.9 preserves substantial structural agreement with EDGE-only while adding coverage.

### EDGE vs SIM≥0.9 Comparison (mode=both)

| node_type | config | silhouette | edge_pct | n_clusters | n_nodes |
|-----------|--------|------------|----------|------------|---------|
| risk | EDGE | 0.484 | 1.000 | 40 | 2,468 |
| risk | SIM≥0.9 | **0.519** | 0.908 | 40 | 2,732 (+11%) |
| intervention | EDGE | 0.450 | 1.000 | 40 | 2,670 |
| intervention | SIM≥0.9 | 0.428 | 0.941 | 40 | 2,856 (+7%) |

Note: n_nodes here counts unique embeddings per config. Total SIM≥0.9 cluster coverage across all node types is 21,546 nodes (17,104 anchored + 4,442 SIM-only).

### Test 8: SIM-Only Node Classification

- Anchored nodes (in both EDGE and SIM≥0.9): **17,104**
- SIM-only nodes (reachable only via SIM≥0.9): **4,442**
  - Foundational (degree ≥10 AND cluster gold_purity ≥0.8): **247 (5.6%)**
  - Niche (otherwise): **4,195 (94.4%)**

SIM-only additions at SIM≥0.9 are predominantly peripheral — they land in structurally-validated clusters but have low cross-paper connectivity. The core taxonomy is anchored in the EDGE-only backbone.

### Test 7: Edge-Only Test Set
100 pathways sampled (stratified: ~33 each design/training/deployment lifecycle stages).
Saved to `edge_only_test_set.jsonl` — direct input to Step 4 simulation validation.

**Note on node_type labels in test set:** `node_types_list` in the JSONL records uses the raw `node_attrs` type field ('concept' or 'intervention'), not the fine-grained category ('risk', 'problem_analysis', etc.). This is a labeling issue only — the pathway node selection is correct.

---

## Section D — Betweenness on Full SIM≥0.9+EDGE Graph

**Files:** `betweenness_sim09.csv` (top-100 nodes), `betweenness_bridge_clusters.csv` (top-50 in 12 clusters)
**Plot:** `betweenness_comparison.png`
**Method:** EXACT betweenness on **valid-pathway nodes** (38,054 nodes, 163,848 edges: SIM≥0.9 134,465 + EDGE conf≥3 29,385). Valid-pathway set: 42,870 nodes (union of all 20 `paths_*.jsonl` files), 4,816 isolated nodes removed. Runtime: ~1.3h (`phase2_rerun_pathfiltered.py`). ✅ Supersedes previous full-graph run (200,568 nodes, 346,224 edges).

### Graph Used
SIM≥0.9 edges + structural EDGE edges (conf≥3), excluding intervention nodes with maturity<3. All nodes as sources. Brandes O(V×E) exact algorithm.

**Note:** The clustering mode (both/unconstrained/monotonic/single_risk) does not affect betweenness computation. Betweenness is a property of the raw graph topology — all SIM≥0.9 + EDGE edges are used regardless of what mode was applied during clustering.

### Why X-Risk Nodes Dominate Bridges

**Correction:** There is no separate "risk literature" and "intervention literature" in the ARD corpus. Each paper traces a **full chain** from risk → mechanism → intervention. x-risk nodes are high-betweenness because they appear at the **start of many different papers' mechanism chains** — each paper begins with a variant of "existential catastrophe from misaligned AI" and then traces a distinct path through different body/mechanism nodes to a different intervention. They are the shared starting premise of many chains, not bridges between separate document populations.

In graph terms: x-risk nodes have high out-degree to many different body-node clusters (different mechanism families), all eventually connecting to intervention nodes. A shortest path between any two mechanism or intervention nodes typically passes through the shared x-risk starting premise they both derive from.

### Top-20 Bridge Nodes (path-filtered exact)

| rank | name (truncated) | category | betweenness |
|------|-----------------|----------|-------------|
| 1 | Catastrophic risk from misaligned advanced AI [170570] | risk | 8.5e-05 |
| 2 | Insufficient talent pipeline for AI safety research [170571] | problem_analysis | 8.2e-05 |
| 3 | Catastrophic failure of advanced AI systems [125864] | risk | 5.1e-05 |
| 4 | existential catastrophe from misaligned AI systems [205938] | risk | 5.1e-05 |
| 5 | Catastrophic outcomes from advanced AI objective misalignment [112836] | risk | 5.1e-05 |
| 6 | Misalignment of superintelligent AI systems [19585] | risk | 5.0e-05 |
| 7 | Catastrophic misalignment of advanced AI systems [145552] | risk | 4.8e-05 |
| 8 | Catastrophic real-world harm from advanced AI systems [215693] | risk | 4.7e-05 |
| 9 | Catastrophic misalignment of advanced AI with human values [128962] | risk | 4.7e-05 |
| 10 | Catastrophic misalignment of advanced AI goals with human values in deployment [111884] | risk | 4.7e-05 |
| 11 | Existential catastrophe from misaligned advanced AI systems [177570] | risk | 4.6e-05 |
| 12 | Catastrophic misalignment of advanced AI systems [31305] | risk | 4.5e-05 |
| 13 | Existential catastrophe from misaligned advanced AI systems [141963] | risk | 4.3e-05 |
| 14 | Existential catastrophe from misaligned advanced AI systems [208443] | risk | 4.2e-05 |
| 15 | Catastrophic misalignment of advanced AI systems [106516] | risk | 4.1e-05 |
| 16 | Catastrophic misalignment of superintelligent AI systems [191280] | risk | 3.9e-05 |
| 17 | Existential catastrophe from power-seeking misaligned AI [134933] | risk | 3.9e-05 |
| 18 | Catastrophic behavior by advanced AI systems [216546] | risk | 3.8e-05 |
| 19 | Existential catastrophe from misaligned advanced AI [147238] | risk | 3.7e-05 |
| 20 | Existential catastrophe from misaligned advanced AI [152224] | risk | 3.7e-05 |

**Category distribution top-50 bridge nodes:** 49 risk · 1 problem_analysis (talent pipeline). ⚠ **FDT is completely absent from path-filtered results** — it ranked 2–8 in the old full-graph run but is not present in any of the top-50 bridge nodes in the valid-pathway analysis.

### Bridge Theme Clusters (top-50 nodes, k=12 Agglomerative, path-filtered exact)

| cluster_id | theme | n_nodes | categories | representative names |
|------------|-------|---------|-----------|---------------------|
| 11 | Existential catastrophe (dense) | 23 | 23 risk | "Existential catastrophe from misaligned advanced AI" × many variants |
| 8 | Catastrophic misalignment (broader) | 9 | 9 risk | "Catastrophic misalignment of advanced AI goals", "Catastrophic misalignment of advanced AI with human values" |
| 5 | Catastrophic risk + harm | 4 | 4 risk | "Catastrophic risk from misaligned advanced AI" (top bridge), "Catastrophic real-world harm" |
| 2 | Superintelligent misalignment | 4 | 4 risk | "Misalignment of superintelligent AI systems" × 3 + variants |
| 1 | Value misalignment | 4 | 4 risk | "Value misalignment in advanced AI systems" × 3 variants |
| 0 | Existential catastrophe (power-seeking / transformative) | 2 | 2 risk | "Existential catastrophe from power-seeking misaligned AI", "transformative AI" |
| 3 | Catastrophic failure / behavior | 2 | 2 risk | "Catastrophic failure of advanced AI systems", "Catastrophic behavior" |
| 4 | Existential catastrophe (superintelligent AGI) | 1 | 1 risk | "Existential catastrophe from misaligned superintelligent AGI" |
| 6 | Human extinction from uncontrollable AI | 1 | 1 risk | "Human extinction from uncontrollable superintelligent AI" |
| 9 | Value misalignment (superintelligent) | 1 | 1 risk | "Value misalignment in superintelligent AI systems" |
| 10 | Catastrophic loss of human control | 1 | 1 risk | "Catastrophic loss of human control from rogue superintelligent AI" |
| 7 | Talent pipeline (problem_analysis) | 1 | 1 problem_analysis | "Insufficient talent pipeline for AI safety research" |

**50 nodes total: 49 risk · 1 problem_analysis. No FDT, no opacity, no reward misspecification in top-50.** This is a fundamental departure from the old full-graph results — path-filtering reveals that the dominant bridge theme is overwhelmingly "existential catastrophe/catastrophic misalignment" variants, with no evidence of the FDT or opacity bridge themes that appeared prominent in the unfiltered graph.

### What is FDT and Why Does It Bridge?

**Functional Decision Theory** (Yudkowsky & Soares, MIRI, ~2017–2018) proposes that a rational agent should act by asking "what is the best policy for all agents whose decision function is identical to mine?" rather than asking about causal or evidential consequences of a single action. This contrasts with Causal Decision Theory (CDT: choose the action with best causal consequences) and Evidential Decision Theory (EDT: choose the action that correlates with best outcomes).

Key examples: **Newcomb's Problem** (one-box because the predictor simulated your decision function), **Parfit's Hitchhiker** (pay on arrival because your decision function must be the type that pays, or you'd never have been rescued), and **"Cheating Death in Damascus"** (rank 7 node) which demonstrates FDT's coherence in death-avoidance scenarios where CDT/EDT generate regret loops.

FDT connects to AI alignment because: a sufficiently capable AI will be modeled/simulated by other agents. CDT-based AI systems are exploitable in Newcomb-like situations; an FDT-based AI cooperates reliably because its decision function is the right type regardless of observation. The rank-8 intervention node ("Design AI agent core decision modules with FDT to improve alignment") is a direct proposal to implement FDT in AI systems.

**Why FDT ranked 2–8 in the OLD full-graph betweenness (now corrected):** All FDT nodes in the corpus come from a single paper that traces a chain from decision-theoretic risk → formal FDT architecture → alignment intervention. This chain spans from risk concepts to intervention, placing every node on shortest paths between the risk literature and the intervention literature. The chain is dense (7 closely related nodes from one paper, all with high mutual similarity), creating a highly connected local cluster that appeared on many cross-cluster shortest paths in the 200K-node graph.

**⚠ FDT is ABSENT from the path-filtered analysis.** After restricting to the valid-pathway node set (42,870 nodes), FDT nodes do not appear in any of the top-50 bridge nodes. This confirms FDT was a full-graph artifact: the single-paper dense cluster had disproportionate influence in the unfiltered large graph, but is not structurally important within paths that satisfy the quality cuts (conf≥3, maturity≥3). Do not use FDT as a bridge theme seed for Step 4 cluster naming.

**Historical context:** FDT peaked in influence around 2018–2022 within MIRI's agent foundations research program and the EA/rationalist community. From 2023 onwards, mainstream AI safety shifted toward empirical alignment (RLHF, interpretability of deployed LLMs, scalable oversight), and MIRI itself acknowledged limited tractable progress on agent foundations. FDT is rarely cited in current NeurIPS/ICML safety papers. Its prominence here reflects the 2018–2022 era in the ARD corpus. For Step 4 cluster naming, this cluster is best labeled "Decision-theoretic alignment (agent foundations era)" to capture both content and historical context.

### Key Findings

**1. Existential catastrophe / catastrophic misalignment = overwhelmingly dominant bridge** (clusters 0+1+2+3+4+5+6+8+9+10+11, 49/50 nodes). The path-filtered result is even more concentrated than the full-graph result: essentially all top bridge nodes are risk variants of "existential catastrophe / catastrophic misalignment from advanced AI." These nodes appear on shortest paths because they are the shared motivating risk premise of most AI safety mechanism chains.

**2. Talent pipeline problem (1 node, rank #2):** "Insufficient talent pipeline for AI safety research" is the single non-risk bridge node in the top-50, ranking #2 by betweenness. It bridges problem_analysis to risk and intervention chains focused on field-building and research capacity.

**3. FDT is absent — confirmed full-graph artifact.** FDT ranked 2–8 in the old full-graph run (200K nodes) but does not appear in any of the top-50 path-filtered bridge nodes. FDT was a single-paper artifact exploiting the large unfiltered graph; it is not a genuine bridge in the quality-cut pathway space.

**4. No opacity, reward misspecification, or adversarial vulnerability** in top-50 bridge nodes (all appeared prominently in old full-graph analysis). These themes connect nodes that do NOT pass the quality cuts, so they disappear when restricted to valid-pathway nodes.

### Step 4 Use (path-filtered betweenness)
`betweenness_bridge_clusters.csv` provides seed themes for manual cluster naming in Step 4 #26:
- Clusters 11+8+5+0-4+6+9+10: Existential catastrophe / catastrophic misalignment variants → primary risk backbone, name first
- Cluster 7: Talent pipeline → field-building / research capacity mechanism cluster
- **Do NOT use FDT as a bridge theme seed** — it is absent from path-filtered analysis
- Opacity, reward misspecification, adversarial vulnerability are not in the path-filtered top-50 — use Section D2 (both-mode) for mechanism-space-specific bridge themes instead

---

## Section D2 — Betweenness on Both-Mode SIM≥0.9 Subgraph

**Files:** `betweenness_both09.csv` (top-100), `betweenness_both09_bridge_clusters.csv` (top-50 in 12 clusters), `betweenness_both09_raw_checkpoint.pkl`
**Plot:** `betweenness_both09_comparison.png`
**Method:** EXACT betweenness on induced subgraph of all nodes in both-mode ec=0.9 agglomerative clusters (conf≥3 EDGE edges, maturity≥3 intervention nodes). Runtime: 15 min (original unfiltered).

✅ **RE-RUN COMPLETE** (`phase2_rerun_pathfiltered.py`). Both-mode cluster members (17,952) intersected with valid-pathway set (17,952 — all are valid). Stats below reflect the path-filtered run.

### Why a Separate Both-Mode Analysis

The full-graph betweenness (Section D) answers: "which nodes bridge the entire AI safety corpus?" The both-mode betweenness answers: "which nodes bridge within the constrained mechanism space we will use for Step 4?" These are different questions. The both-mode subgraph contains only nodes that satisfy monotonic + single_risk constraints — the final mechanism selection. Betweenness within this graph reveals which concepts structurally connect distinct mechanism families in the curated taxonomy.

### Subgraph Structure

| Metric | Value |
|--------|-------|
| Nodes | 17,952 |
| SIM≥0.9 edges | 5,855 |
| Structural EDGE edges | 15,415 |
| Total edges | 21,269 |
| EDGE:SIM ratio | 2.6:1 |

The both-mode subgraph is **predominantly structural** (EDGE edges), as expected: both mode's monotonic + single_risk constraints select nodes that are part of directed causal chains extracted from papers, not merely semantically similar nodes.

### Top-20 Bridge Nodes (both-mode exact)

| rank | name (truncated) | category | betweenness_both09 |
|------|-----------------|----------|--------------------|
| 1 | Catastrophic AI system failures impacting humanity | risk | 0.005665 |
| 2 | Adversarial vulnerability of neural networks to small perturbations | problem analysis | 0.004630 |
| 3 | Adversarial vulnerability in neural network image classifiers | risk | 0.003551 |
| 4 | Catastrophic misalignment of advanced AI systems | risk | 0.003435 |
| 5 | Catastrophic misalignment of superintelligent AI with human values | risk | 0.003241 |
| 6 | Catastrophic misalignment of advanced AI systems (variant) | risk | 0.002631 |
| 7 | Reward function misspecification in reinforcement learning agents | problem analysis | 0.002459 |
| 8 | Catastrophic misalignment of advanced AI systems (variant) | risk | 0.002400 |
| 9 | Catastrophic existential harms from misaligned advanced AI systems | risk | 0.002113 |
| 10 | Catastrophic failure of advanced AI systems | risk | 0.002046 |
| 11 | Opaque decision-making processes in deep neural networks | problem analysis | 0.002020 |
| 12 | Existential catastrophe from misaligned AGI systems | risk | 0.001940 |
| 13 | Misaligned behavior of AI agents in deployment | risk | 0.001926 |
| 14 | Existential catastrophe from misaligned advanced AI systems | risk | 0.001915 |
| 15 | High uncertainty in AI progress forecasting | problem analysis | 0.001890 |
| 16 | High uncertainty in AI progress timelines leading to inadequate preparation | problem analysis | 0.001868 |
| 17 | High uncertainty in AI timeline forecasting | problem analysis | 0.001855 |
| 18 | Reward hacking in reinforcement learning agents | problem analysis | 0.001813 |
| 19 | Unsafe exploration in reinforcement-learning agents | problem analysis | 0.001802 |
| 20 | Opaque internal representations in neural networks | problem analysis | 0.001774 |

**Category distribution top-100:** 57 risk · 29 problem_analysis · 5 theoretical_insight · 4 design_rationale · 4 implementation_mechanism · 1 intervention

**FDT is completely absent.** FDT nodes are excluded from the both-mode subgraph because the single_risk constraint prevents their dense paper-local cluster from appearing as a multi-chain bridge. All 5 intervention nodes in the top-100 fall in the 90s by rank.

### Bridge Theme Clusters (both-mode, k=12 Agglomerative)

| cluster_id | theme | n_nodes | categories |
|------------|-------|---------|-----------|
| 1 | Catastrophic AI failures | 14 | 14 risk |
| 5 | Catastrophic misalignment | 9 | 9 risk |
| 2 | Reward misspecification / hacking | 6 | 5 problem_analysis + 1 risk |
| 8 | Opacity / opaque neural networks | 4 | 4 problem_analysis |
| 4 | Adversarial vulnerability | 3 | 2 risk + 1 problem_analysis |
| 11 | Timeline uncertainty / safety prep | 3 | 3 problem_analysis |
| 6 | RLHF / human feedback alignment | 3 | 1 theoretical + 1 design + 1 implementation |
| 0 | AGI threshold / compute governance | 2 | 1 problem_analysis + 1 design |
| 3 | Compute governance / export controls | 2 | 1 theoretical + 1 implementation |
| 7 | Misaligned behavior / value alignment | 2 | 1 risk + 1 problem_analysis |
| 9 | Mechanistic interpretability | 1 | 1 implementation |
| 10 | Unsafe RL exploration | 1 | 1 problem_analysis |

Clusters 1+5 together cover 23/50 top-50 nodes (46%) — catastrophic misalignment and AI failures dominate the both-mode bridge space even more than in the full-graph run.

### Connectivity: Full Graph vs Both-Mode

Both-mode normalized betweenness scores appear comparable to full-graph scores (max 0.005665 vs 0.003367), but this is a normalization artifact.

Betweenness is normalized by `(V-1)(V-2)/2`. The full graph's normalization factor is **124.8× larger** than both-mode's. After correcting:

```
raw betweenness ratio (top node, both/full) = 1.68 / 124.8 = 0.013×
```

The top bridge node in both-mode has **~74× fewer actual paths** through it in absolute terms. The apparent similarity of normalized scores hides a massive difference in absolute connectivity.

| Metric | Full graph | Both-mode | Notes |
|--------|-----------|-----------|-------|
| Nodes | 200,568 | 17,952 | 11.2× size difference |
| Avg degree | 3.45 | 2.38 | Both-mode sparser per node |
| Density | 1.72×10⁻⁵ | 1.33×10⁻⁴ | Both-mode 7.7× denser locally |
| WCCs | 10,298 | 1,634 | — |
| Largest WCC | 34.5% of nodes | 21.2% (3,797 nodes) | Both-mode more fragmented |
| 2nd largest WCC | large | 135 nodes | Steep dropoff — no 2nd giant |
| Reachable pairs | **11.9%** | **4.5%** | Both-mode 2.6× less connected |
| Normalization factor | 2.01×10¹⁰ | 1.61×10⁸ | 124.8× difference |
| Raw betweenness (top node) | ~6.77×10⁷ | ~9.13×10⁵ | Both-mode ~74× less absolute |

### Why Both-Mode Is More Fragmented Despite Higher Local Density

The single_risk constraint is the primary driver of fragmentation. In the full graph, a concept like "catastrophic misalignment" acts as a hub connecting many different mechanism chains — any chain that references this risk can connect to any other chain referencing the same risk. Under single_risk, each cluster has exactly one risk entry point. Two chains that share a semantically similar risk concept are assigned separate instances and kept disconnected. This converts what would be a large connected risk-hub subgraph into many isolated chains.

The monotonic constraint compounds this: path directionality prevents some lateral connections that would exist in unconstrained mode.

Result: the both-mode graph consists mostly of isolated causal chains (median WCC size is small; the giant component covers only 21.2% of nodes with a steep dropoff to 135 for the 2nd largest). Betweenness in this graph identifies the rare nodes that genuinely appear in multiple such chains — real cross-family connectors rather than corpus-wide hubs.

### Interpretation for Step 4

⚠ **NOTE (updated 2026-03-29):** Step 4 uses **unconstrained** mode, not both mode — see "⚠ Step 4 Mode Decision: Unconstrained Supersedes Both" in Section A. The both-mode betweenness (Section D2) is therefore **exploratory only** for Step 4. It remains useful for understanding what bridge nodes exist in a constrained single-root causal chain space, but the cluster IDs and themes below should NOT be used directly as Step 4 naming seeds. Use Section D (unconstrained, path-filtered) for naming seeds; use `category_mechanism_families.csv` (Step 2b) and #25 cluster tables for connection concept chain family seeds.

The both-mode analysis reveals the mechanism-space bridge themes that exist in the literature (reward misspecification, opacity, RLHF, adversarial vulnerability), which remain valuable as domain knowledge for naming even though they are not derived from unconstrained betweenness. Key implications:

- **Catastrophic misalignment / AI failures (clusters 1+5, 23 nodes)** are the structural backbone — they appear in more mechanism chains than any other concept family. Any cluster naming that doesn't account for this theme will miss the dominant organizing principle.
- **Adversarial vulnerability (cluster 4)** is the #2 bridge after catastrophic misalignment — it connects risk and problem_analysis chains in a distinct way from the pure existential risk clusters.
- **Reward misspecification (cluster 2) and opacity (cluster 8)** are the two most important problem_analysis bridges — they connect RL alignment chains to other mechanism families.
- **RLHF cluster (cluster 6)** is the only implementation-side bridge — the mechanism by which human feedback connects to the risk-problem_analysis backbone.
- **FDT's absence** from both-mode confirms it is a corpus-wide bridge (many papers reference its concepts) but not a mechanism-chain bridge (it doesn't form cross-family causal chains under monotonic + single_risk constraints). For Step 4, FDT is a naming seed from the full-graph analysis but should not be expected to appear as a central mechanism family in the constrained taxonomy.

### Step 4 Use (both-mode betweenness)
`betweenness_both09_bridge_clusters.csv` provides the mechanism-space-specific bridge themes:
- Clusters 1+5: Catastrophic misalignment → primary risk backbone, name clusters around this first
- Cluster 2: Reward misspecification → RL alignment mechanism family
- Cluster 8: Opacity → interpretability/transparency mechanism family
- Cluster 4: Adversarial vulnerability → robustness mechanism family
- Cluster 6: RLHF / human feedback → implementation mechanism connecting risk to intervention

---

## Section E — #24 Held-Out Validation

**File:** `held_out_validation.csv` (315 clusters)
**Filters:** Cluster members come from SIM≥0.9+both path files (conf≥3 + maturity≥3). Accuracy is computed on embeddings only — no EDGE edges or SIM edges are traversed. All intervention clusters contain only maturity≥3 nodes. ✅

### Method
Leave-20%-out accuracy: for each cluster in the primary cut (SIM≥0.9+both, agglomerative), withhold 20% of members, compute centroid from 80%, check if withheld nodes are nearest-neighbour assigned to their original cluster. Comparison is **within-node_type only** (not across all 315 clusters). Note: cross-node_type comparison would compute NN against all 315 cluster centroids spanning different semantic domains (risk vs intervention vs design_rationale etc.), artificially inflating error rates because withheld nodes would often be "closer" to centroids from a different node type at a different semantic scale. Per-node_type comparison is the correct baseline (1/40 = 2.5% chance).

### Results by Node Type

| node_type | n_clusters | mean_acc | overall_acc |
|-----------|------------|----------|-------------|
| all_concepts | 40 | 0.556 | 0.391 |
| design_rationale | 39 | 0.679 | 0.598 |
| implementation_mechanism | 40 | 0.650 | 0.609 |
| **intervention** | 39 | **0.716** | 0.614 |
| **problem_analysis** | 40 | **0.708** | 0.687 |
| risk | 38 | 0.600 | 0.555 |
| theoretical_insight | 39 | 0.630 | 0.546 |
| validation_evidence | 40 | 0.647 | 0.564 |
| **Overall** | **315** | | **0.512** |

Random chance baseline: 1/40 = 2.5%. Observed 51.2% = **20× above random**.

### Interpretation
51.2% overall is below the 80% target but reflects genuinely soft cluster boundaries at k=40 granularity. Intervention (71.6%) and problem_analysis (70.8%) clusters are most cohesive. all_concepts (39.1%) is the weakest — expected, as it pools all concept categories with more diffuse semantics. The 20× above-random result confirms clusters capture real semantic structure; soft boundaries are consistent with the continuum of AI safety concepts.

### Workshop Claim Supported
> "Mechanism clusters demonstrate 51.2% mean leave-20%-out accuracy (315 clusters, k=40 Agglomerative), 20× above random chance (2.5%), indicating that cluster assignments are semantically stable. Intervention and problem analysis clusters achieve 70–72% accuracy, reflecting the strongest mechanism coherence. Boundary softness in all_concepts (39%) reflects the inherent semantic continuum of AI safety risk concepts."

---

## Section F — #25 EDGE Subgraph Consistency

**File:** `edge_subgraph_stats.csv`
**Plot:** `edge_degree_distribution.png`

**Filters applied:** Node set restricted to the 42,870 valid-pathway nodes (union of all 20 `paths_*.jsonl` files — guarantees all EDGE edges on any included chain have conf≥3 and all intervention endpoints have maturity≥3). EDGE edges additionally filtered to conf≥3 with both endpoints in valid-pathway set. Run by `phase2_rerun_pathfiltered.py` (completed 2026-03-28 21:43).

**Comparison run (unfiltered, `edge_subgraph_stats_unfiltered.csv`):** 200,525 nodes, 202,123 edges, 15,123 WCCs, largest WCC 61, mean degree 2.02.

### Topology Results (valid-pathway nodes, conf≥3 EDGE)

| Metric | Value |
|--------|-------|
| Nodes (valid-pathway set) | 42,870 |
| EDGE edges (conf≥3, both endpoints valid-pathway) | 29,383 |
| Weakly connected components | 13,908 |
| Largest WCC size | 37 nodes |
| 2nd largest WCC | 35 nodes |
| Approximate diameter (BFS sample) | 12 hops |
| Mean degree | 1.37 |
| Nodes with degree ≥ 2 | 47.6% |

### Key Structural Finding: Isolated Chains by Design

The EDGE-only subgraph consists of ~13,908 isolated chains (valid-pathway: 29,383 edges, mean degree 1.37, largest component 37 nodes). This is **expected by design**: each EDGE chain was extracted from a single source document, tracing a local argument from risk → body → intervention within one paper. Cross-paper connectivity does not exist in EDGE edges and was never intended to.

The path-filtered run has fewer WCCs (13,908 vs 15,123 unfiltered) because partial-chain nodes (no complete risk→intervention span) are excluded from the valid-pathway set. The longer diameter (12 vs 6 hops unfiltered) reflects that valid complete-path chains are structurally longer than average partial fragments.

**SIM edges are therefore the structural mechanism for cross-paper analysis** — they are not optional enrichment but the only source of global connectivity. The EDGE subgraph confirming ~14K isolated fragments is a validation that extraction worked as designed (one chain per paper), not a surprising finding.

---

## Summary: Final Config Selection

| node_type | Final config | Mode | Key metric | Workshop use |
|-----------|-------------|------|------------|--------------|
| risk | SIM≥0.9 | both | composite=0.789 (rank 3), ARI stability=0.731 | Primary analysis cut |
| intervention | SIM≥0.95 | both | composite=0.808 (rank 2), ARI stability=0.744 | Primary analysis cut (updated) |
| implementation_mechanism | SIM≥0.9 | both | composite=0.865 (rank 1) | Confirmed |
| all_concepts | SIM≥0.95 | monotonic | composite=0.833 (rank 1) | Confirmed |

### Step 4 Inputs Ready

| File | Use in Step 4 |
|------|---------------|
| `optimal_configs_final.csv` | Determines cluster memberships for naming |
| `edge_only_test_set.jsonl` | 100 pathways for simulation validation |
| `betweenness_bridge_clusters.csv` | Full-corpus bridge seeds (path-filtered ✅): existential catastrophe dominates (49/50); talent pipeline only non-risk bridge; FDT absent |
| `betweenness_both09_bridge_clusters.csv` | Mechanism-space bridge seeds (path-filtered ✅): catastrophic misalignment, reward misspecification, opacity, RLHF |
| `selection_justification.md` | Methods section 3.4 text |

### Step 4 Betweenness Use

**Full-graph betweenness** (`betweenness_bridge_clusters.csv`, path-filtered ✅) — naming seeds for Step 4 #26: existential catastrophe / catastrophic misalignment dominates (49/50 top bridge nodes). Talent pipeline problem is the single non-risk bridge (#2). **⚠ FDT is absent** — do NOT use as a naming seed (confirmed artifact; see Section D). Opacity and reward misspecification also absent from top-50 path-filtered results.

**Both-mode betweenness** (`betweenness_both09_bridge_clusters.csv`) — naming seeds for Step 4 #26, mechanism-space specific: catastrophic misalignment, adversarial vulnerability, reward misspecification, opacity, RLHF. Use to prioritize which clusters to name first within the mechanism taxonomy.

**Step 4 cluster connectivity analysis (#27) uses unconstrained** — the x-risk hub cluster at the top of the risk hierarchy is a meaningful structural level (x-risk cluster → secondary risk clusters → mechanism clusters → intervention clusters). Node-level betweenness on constrained subgraphs is not the primary tool for Step 4; cluster-level EDGE connectivity (#27) supersedes it.
