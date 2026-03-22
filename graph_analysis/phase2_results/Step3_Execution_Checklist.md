# Step 3 Execution Checklist
**Generated:** 2026-03-22
**Full plan:** `Phase2_Step3_Plan.md`
**Branch:** `martin/main`
**Working dir:** `graph_analysis/`

---

## Context (read before starting)

- **Step 2b is COMPLETE.** All data in `phase2_results/step2_metrics_and_stability/`.
- **Algorithm is FINAL.** Agglomerative k=40 (sil=0.438 >> HDBSCAN 0.275/80.7%noise >> Louvain 0.011). Do not revisit.
- **Primary cut is EARMARKED** (not yet confirmed): SIM≥0.9, mode=both. Step 3 #23 either confirms or updates this.
- **PKL checkpoints** in `phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/`:
  - `graph_node_attributes.pkl` — 200,525 nodes; `url` key for source; `embedding` as FalkorDB string
  - `graph_edge_data.pkl` — 1,767,833 edges; SIMILARITY score → `cos_sim = 1 − score²/2`; `source_file` NULL on all edges
  - `cluster_memberships.pkl` — 13,214 records keyed by (edge_config, mode, node_type, algo, cluster_id)
- **SIMILARITY threshold filter:**
  ```python
  def cos_sim_from_score(s): return 1.0 - float(s)**2 / 2.0
  sim_09 = [e for e in edge_data
            if str(e.get('type','')).upper() == 'SIMILARITY'
            and cos_sim_from_score(e['similarity_score']) >= 0.90]
  ```
- **Do NOT report:** node_migration_heatmap.png (metric artifact), migration rate (96.6%)
- **Source diversity** (n_sources ≈ cluster_size r=0.887): not a config discriminator — do not use for selection

---

## Step 0 — PREREQUISITE: Run step2c

```bash
cd graph_analysis/
uv run phase2_step2c_plot_updates.py
```

**Generates** (saved to `step2_metrics_and_stability/`):
- `hub_quality_bar_v2.png` — horizontal bar, top-20 hubs per node type, ranked by SIM≥0.9 degree
- `edge_validation_per_mode_v2.png` — node-type segmented bars, 2×2 mode grid

**Status:** ⬜ TODO

---

## Step 3 Script

**New script to write:** `graph_analysis/phase2_step3_validation_and_selection.py`
**Output folder:** `phase2_results/step3_validation_and_selection/`

---

## Substep A — #23 Multi-Criteria Scoring ⭐ CRITICAL

**Goal:** Rank all 160 configs with weighted composite → confirm/update earmarked cut.
**Input:** CSVs only (no PKL needed). Fast (~2 min).

**Algorithm:**
1. Load from `step2_metrics_and_stability/`:
   - `all_cluster_metrics.csv` → `silhouette_mean` per (edge_config, mode, node_type)
   - `quality_metrics_summary.csv` → `edge_validation_mean`
   - `stability_ari_matrix.csv` → ARI for pairs (0.9↔0.95) and (EDGE↔0.95); fallback to (0.9↔EDGE)
   - `cluster_edge_purity.csv` → gold_pct (fraction clusters with purity≥80%)

2. Compute 5 raw metrics per config:
   - `silhouette` = silhouette_mean (Agglomerative only)
   - `edge_pct` = edge_validation_mean
   - `cluster_count_score` = triangular: 1.0 at k=40–50, linear decay to 0 at k≤20 or k≥80
   - `ari_high` = mean ARI for high-threshold pairs (0.9↔0.95 and EDGE↔0.95)
   - `interpretability` = 1.0 for risk/intervention/individual concepts; 0.5 for all_concepts

3. Normalize each metric to [0,1] min-max **within each node_type** (not globally).

4. Composite = 0.25×sil + 0.30×edge_pct + 0.20×cluster_count + 0.15×ari + 0.10×interpret

5. Sort descending by composite within each node_type. Add rank column.

**Outputs:**
- `optimal_configs_ranked.csv` — 160 rows: all raw + normalized + composite + rank
- `optimal_configs_final.csv` — rank=1 row per node_type (the winner)
- `multi_criteria_parallel.png` — parallel coords: 5 axes, grey=all 160, color=top-10 per node_type
- `selection_justification.md` — text: winner per node_type, confirms/updates earmarked cut

**Expected result:** SIM≥0.9 + both ranks #1 for risk + intervention.

**Status:** ⬜ TODO

---

## Substep B — #21 Edge Threshold Sensitivity ⭐ ESSENTIAL

**Goal:** Prove SIM≥0.9 is the entry to stable regime (not arbitrary choice).
**Input:** CSVs only. Fast (~1 min).

**Algorithm:**
1. For each adjacent threshold pair (0.8→0.85, 0.85→0.9, 0.9→0.95, 0.95→EDGE), compute per (node_type, mode):
   - ΔARI = ARI(T2→T3) − ARI(T1→T2)  [from `stability_ari_matrix.csv`]
   - ΔSilhouette [from `all_cluster_metrics.csv`]
   - ΔEDGE% [from `quality_metrics_summary.csv`]
   - ΔCentroid sim [from `cluster_centroid_similarity.csv`]
   - ΔGold purity [from `cluster_edge_purity.csv`]

2. Compute `stability_score` per threshold = mean ARI to all higher thresholds.

3. Plot multi-metric profile: x=threshold, 5 metric lines normalized, shade 0.9–EDGE as "stable regime".

**Outputs:**
- `threshold_sensitivity_analysis.csv` — per (threshold_pair, node_type, mode): all Δ values + stability_score
- `threshold_sensitivity_profile.png` — 4-panel: metric lines vs threshold; Δmetric bars; stability_score; ARI heatmap

**Status:** ⬜ TODO

---

## Substep C — #22 EDGE-Only Baseline Validation ⭐ ESSENTIAL

**Goal:** Prove SIM≥0.9 adds real value over pure EDGE-only baseline.
**Input:** PKL + CSVs (~5 min).

**Three tests:**

**Test 6 — ARI overlap (from existing CSV):**
- Extract EDGE↔0.9 ARI from `stability_ari_matrix.csv` per (node_type, mode). Target >0.5.

**Test 7 — EDGE-only test set (100 pathways for Step 4 simulation):**
- From `cluster_memberships.pkl` for (EDGE, unconstrained) config
- Sample 100 pathways stratified: ~33 per lifecycle stage (design/training/deployment); max 3 per cluster
- Each record: node_id_sequence, node_names, node_types, source_urls, cluster_id
- Save as `edge_only_test_set.jsonl`

**Test 8 — SIM-only node coverage:**
- "Anchored nodes" = nodes in any EDGE-only complete pathway
- "SIM-only nodes" = nodes in SIM≥0.9 clusters but NOT in any EDGE-only pathway
- For SIM-only nodes: mean degree in SIM≥0.9 graph; cluster gold_purity% they land in
- Classify: "foundational" (degree≥10 AND cluster gold_purity≥0.8) vs "niche" (otherwise)

**Outputs:**
- `edge_only_comparison.csv` — EDGE vs SIM≥0.9: silhouette, EDGE%, cluster count, gold_purity, n_nodes, ARI side-by-side
- `edge_vs_sim_coverage.png` — 2-panel: (A) grouped quality bars; (B) SIM-only node classification
- `edge_only_test_set.jsonl` — 100 pathways → **direct input to Step 4**

**Status:** ⬜ TODO

---

## Substep D — Betweenness on SIM≥0.9 Graph (ENRICHMENT)

**Goal:** Recompute bridge-node betweenness at primary-cut graph; cluster top-50 into bridge themes.
**Input:** PKL. SLOW (~15–25 min). Run last / optionally skip for first pass.

**Algorithm:**
1. Build filtered graph: SIM≥0.9 edges + all structural EDGE edges
2. Approximate betweenness: `nx.betweenness_centrality(G, k=1000)` (undirected, k=1000 samples)
   - OR restrict to ~30K nodes with structural EDGE degree ≥ 1 for speed
3. Join with `mechanism_transfer_betweenness.csv` (SIM≥0.8 computation) to compare ranks
4. Cluster top-50 by cosine similarity (embeddings from `graph_node_attributes.pkl`), k=10–15 agglomerative
5. Label cluster themes by exemplar name

**Outputs:**
- `betweenness_sim09.csv` — top-100 nodes: name, category, betweenness_sim09, old_sim08_rank
- `betweenness_comparison.png` — scatter sim08 vs sim09 betweenness, colored by category
- `betweenness_bridge_clusters.csv` — top-50 bridge nodes in 10–15 clusters with exemplar names

**Status:** ⬜ TODO

---

## Substep E — #24 Held-Out Validation (ENRICHMENT)

**Goal:** Prove clusters generalize (not corpus-specific).
**Input:** PKL. ~3 min.

**Algorithm:**
- For each cluster in primary cut, withhold random 20% of members
- Compute centroid from remaining 80%
- Check: % of withheld nodes whose nearest centroid is their original cluster
- Report mean leave-20%-out accuracy across all clusters (target >80%)

**Outputs:**
- `held_out_validation.csv` — per cluster: n_members, accuracy; summary: mean accuracy

**Status:** ⬜ TODO

---

## Substep F — #25 EDGE Subgraph Consistency (ENRICHMENT)

**Goal:** Verify EDGE-only backbone is topologically coherent.
**Input:** PKL. ~3 min.

**Algorithm:**
- Build directed graph from structural EDGE edges only (202K edges)
- Check: n_weakly_connected_components, largest_WCC_pct, approx_diameter (BFS sample), mean degree
- Overlap check: are top-25 betweenness nodes (from `mechanism_transfer_betweenness.csv`) present in EDGE subgraph?

**Outputs:**
- `edge_subgraph_stats.csv` — topology summary
- `edge_degree_distribution.png` — log-log degree distribution

**Status:** ⬜ TODO

---

## Execution Order

```
1. uv run phase2_step2c_plot_updates.py          [~2 min, no PKL]
2. Write + run phase2_step3_validation_and_selection.py:
   a. Section A (#23) + Section B (#21)          [~3 min, no PKL]
   b. Load PKL once, then:
      Section C (#22) + Section E (#24) + Section F (#25)  [~11 min]
   c. Section D (betweenness) — optional/last    [~20 min]
3. Verify outputs in step3_validation_and_selection/
4. Update selection_justification.md if earmarked cut changes
5. git add + commit + push to martin/main
```

---

## Output File Summary

| File | Substep | Priority | Step 4 Input? |
|------|---------|----------|---------------|
| `optimal_configs_ranked.csv` | #23 | CRITICAL | — |
| `optimal_configs_final.csv` | #23 | CRITICAL | ✅ determines clustering to use |
| `selection_justification.md` | #23 | CRITICAL | ✅ Methods text |
| `multi_criteria_parallel.png` | #23 | CRITICAL | — |
| `threshold_sensitivity_analysis.csv` | #21 | ESSENTIAL | — |
| `threshold_sensitivity_profile.png` | #21 | ESSENTIAL | — |
| `edge_only_comparison.csv` | #22 | ESSENTIAL | — |
| `edge_vs_sim_coverage.png` | #22 | ESSENTIAL | — |
| `edge_only_test_set.jsonl` | #22 | ESSENTIAL | ✅ simulation gold set |
| `betweenness_sim09.csv` | betweenness | ENRICHMENT | ✅ bridge theme seed |
| `betweenness_bridge_clusters.csv` | betweenness | ENRICHMENT | ✅ bridge theme seed |
| `betweenness_comparison.png` | betweenness | ENRICHMENT | — |
| `held_out_validation.csv` | #24 | ENRICHMENT | — |
| `edge_subgraph_stats.csv` | #25 | ENRICHMENT | — |
| `edge_degree_distribution.png` | #25 | ENRICHMENT | — |

---

## After Step 3 → Step 4 Critical Path

1. Lock final optimal config from `optimal_configs_final.csv`
2. Run Step 4 script using that config's cluster memberships
3. **Step 4 CRITICAL (#26):** Manual cluster naming (40–60 mechanism families, coherence 1–5 scale)
   - Input seeds: `category_mechanism_families.csv` (Step 2b), `betweenness_bridge_clusters.csv` (Step 3)
   - Use `cluster_memberships.pkl` for the final config to get cluster members
   - Name each cluster by examining top-5 member names + descriptions
