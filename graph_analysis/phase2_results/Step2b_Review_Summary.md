# Step 2b Review Summary
**Generated:** 2026-03-08
**Purpose:** Focused review doc — all items that changed, were reinterpreted, or corrected after the first Step 2b run

---

## Status Legend
- ✅ Output is correct, interpretation confirmed
- ⚠️ Output was reinterpreted / needs careful reading
- 🔁 Will be regenerated in current re-run (hub quality v2, HDBSCAN added)
- ❌ Bug found, fixed in current re-run

---

## 1. Node Migration Heatmap (`node_migration_heatmap.png`)

**Status:** ⚠️ Interpretation revised
**Issue:** Migration rate metric counts ANY cluster-ID change, even if the destination cluster is semantically very similar to the source.
**Evidence:** Migrating nodes land at mean cosine similarity 0.950 to destination centroids. Random baseline = 0.733 (off-diagonal inter-cluster mean). Excess = +0.217.
**Revised interpretation:** Migration ≈ artifact. Centroid similarity >0.93 is a REAL stability signal — clusters are genuinely distinct (range 0.47–0.94 off-diagonal), not a compact domain.
**Action:** No re-run needed. Add caveat when citing this figure.

---

## 2. Centroid Similarity Heatmap (`centroid_similarity_heatmap.png`)

**Status:** ✅ Output correct, interpretation strengthened
**Key finding:** All cross-threshold transitions >0.93 centroid cosine similarity.
**Confirmed real signal (not domain artifact):** Inter-cluster baseline = 0.733 (computed from 40×40 matrix for EDGE/unconstrained/risk). Random pairs have mean 0.733; migrating nodes land at 0.950 → excess +0.217.
**Workshop claim:** "Clustering is semantically stable across SIM thresholds — transitions represent short-range reorganization, not random resampling."

---

## 3. Hub Quality Scatter (`hub_quality_scatter.png`) — `hub_quality_metrics.csv`

**Status:** 🔁 REGENERATED in current re-run (v2, threshold-aware)
**Bugs in v1:**
  1. `edge_percentage` ≈ 0 for all hubs (used to diagnose: hub 6295 had 198 SIM edges at 0.8 but only 18 at 0.9, 0 at 0.95 → was a low-threshold artifact, not a strong hub)
  2. `n_sources` used NULL `source_file` from edge_data — always 0
  3. Only included `intervention` node type — missed concept hubs
**Fixes in v2:**
  - Threshold-aware degree index: `degree_sim_0.80/0.85/0.90/0.95` computed from `score` attribute (converted via `cos_sim = 1 - score²/2`)
  - `n_sources_at_config_thr` = distinct partner node URLs at config's threshold
  - Includes intervention + concept + all other non-all_concepts node types
  - Rankings by `degree_sim_0.90` to identify genuine cross-paper hubs
**Confirmed finding:** True SIM≥0.9 hubs are "Existential catastrophe from misaligned advanced AI" **risk** nodes (~600–635 SIM≥0.9 edges each). Top hub: 635 SIM>=0.9 edges, 1792 distinct partner papers. 5 near-duplicate risk concept nodes from different papers rank top-5. 358/2800 hub records are strong cross-paper hubs (SIM>=0.9 >= 50).

**New plot axes:**
  - x = `degree_sim_0.90` (cross-paper connections at high threshold)
  - y = `n_sources_at_config_thr` (distinct partner paper URLs)
  - size = `degree_total_0.80` (total degree at storage floor)
  - color = `degree_structural` (EDGE-type degree)
  - Reference line: x=50 (strong cross-paper hub threshold)

---

## 4. Algorithm Comparison (`algorithm_comparison_silhouette.png`) — `algorithm_comparison.csv`

**Status:** ✅ REGENERATED + confirmed
**Results confirmed:**
  - Agglomerative k=40: mean silhouette = **0.438** ✅ best
  - Louvain auto-k=42.6: mean silhouette = **0.011** ❌ unsuitable
  - HDBSCAN in-memory: mean silhouette = **0.275**, noise_pct = **80.7%**, n_clusters = 3.0 ❌ unsuitable
**Conclusion:** HDBSCAN classifies 80.7% of nodes as noise and collapses to ~3 clusters — definitively ruled out.
Agglomerative k=40 is the clear winner. No Step 3 algorithm exploration needed.

---

## 5. Source Diversity v2 (`cluster_source_diversity_v2.csv`)

**Status:** ✅ Output correct (v2 already used URL-based n_sources)
**Key finding:** n_sources ≈ cluster_size (r=0.887) — each cluster member comes from a distinct paper.
**Interpretation:** Expected given no node deduplication. Source diversity is NOT a useful discriminator between configs.
**Workshop claim REVISED to:** "Clusters have high source diversity by construction — not a meaningful differentiator."

---

## 6. EDGE Validation Per Mode (`edge_validation_per_mode.png`)

**Status:** ✅ Output correct
**Key finding:**
  - EDGE-only configs: 100% validation (all 32 configs)
  - SIM≥0.9 + "both" mode: 90%+ validation
  - SIM≥0.85 + "both" mode: 73% validation
  - SIM≥0.8 unconstrained: 27% validation (high artifact risk)

---

## 7. EDGE Purity Histograms (`edge_purity_histograms.png`) — `cluster_edge_purity.csv`

**Status:** ✅ Output correct
**Key finding:**
  - Gold-standard clusters (purity ≥80%): EDGE=100%, 0.95=98.7%, 0.9=81.6%
  - SIM≥0.9 is minimum for reliable auto-labeling
  - 66.8% of all clusters are gold-standard across all configs

---

## 8. ARI Pairwise + Line Plot (`cross_threshold_ari_lineplot.png`) — `stability_ari_pairwise.csv`

**Status:** ✅ Output correct
**Key finding:**
  - Adjacent high-threshold pairs (0.9↔0.95, 0.95↔EDGE) achieve ARI >0.7
  - Mean ARI 0.55–0.65 for risk/intervention (across all pairs)
  - Lower thresholds (0.8, 0.85) produce qualitatively different clusterings
**See also:** Centroid similarity confirms these are short-range reorganizations (not random resampling)

---

## 9. Maturity Distribution Heatmap (`maturity_distribution_heatmap.png`) — `maturity_per_cluster.csv`

**Status:** ✅ Output correct (bug was fixed in Step 2b)
**Key finding:**
  - Dominant stage distribution: Deployment=462, Training=207, Design=131 clusters
  - Mean across clusters: Design 21%, Training 26%, Deployment 53%

---

## 10. Mode Impact Analysis (`edge_density_heatmap.png`, `mode_stability_heatmap.png`) — `mode_comparison_stats.csv`

**Status:** ✅ Output correct
**Key finding:**
  - "both" mode consistently achieves best EDGE validation + silhouette balance
  - "unconstrained" has highest diversity but lowest EDGE purity
  - "monotonic" has highest structural integrity but over-constrains pathway space

---

## 11. Mechanism Transfer Betweenness (`mechanism_transfer_betweenness_v2.png`)

**Status:** ✅ Output correct
**Note:** Sorted descending, 5 category panels

---

## 12. Schema Corrections (background knowledge only — no output regeneration needed)

**FalkorDB graph_edge_data.pkl composition:**
- 88.6% SIMILARITY edges (1,565,684)
- 11.4% structural EDGE edges (202,149)
- `source_file` is NULL for ALL edges (both types)
- FROM edges absent from PKL

**SIMILARITY score conversion:**
```
cos_sim = 1 - score² / 2
```
Max stored score = 0.6325 = sqrt(2×0.2) = L2 at cos_sim=0.80 (storage floor).
All stored SIMILARITY edges have cos_sim ∈ [0.80, 0.989].

**Threshold cutoffs (score → cos_sim):**
- cos_sim ≥ 0.80: score ≤ 0.6325
- cos_sim ≥ 0.85: score ≤ 0.5477
- cos_sim ≥ 0.90: score ≤ 0.4472
- cos_sim ≥ 0.95: score ≤ 0.3162

---

## Quick Summary Table

| Output | Status | Key change |
|--------|--------|------------|
| `node_migration_heatmap.png` | ⚠️ Reinterpreted | Migration = artifact; centroid sim = real signal |
| `centroid_similarity_heatmap.png` | ✅ Confirmed | >0.93 transitions are real; baseline=0.733 |
| `hub_quality_scatter.png` | 🔁 Regenerated | v2: threshold-aware, correct n_sources, concept nodes |
| `hub_quality_metrics.csv` | 🔁 Regenerated | New columns: degree_sim_0.80/0.85/0.90/0.95, n_sources_at_config_thr |
| `algorithm_comparison_silhouette.png` | 🔁 Regenerated | Added HDBSCAN 3-way comparison |
| `algorithm_comparison.csv` | 🔁 Regenerated | Added hdbscan_silhouette, hdbscan_n_clusters, hdbscan_noise_pct |
| `cluster_source_diversity_v2.csv` | ✅ Correct | n_sources ≈ cluster_size by construction; not a useful differentiator |
| All other outputs | ✅ Unchanged | No schema errors found |

---

## Next Steps (Step 3)

1. Review hub quality v2 results — confirm "Existential catastrophe" nodes are top hubs at SIM≥0.9
2. Review HDBSCAN silhouette — compare to Agglomerative 0.438 baseline
3. Multi-criteria config selection: silhouette + EDGE% + ARI + cluster count
4. HDBSCAN noise_pct assessment — if >30% noise, HDBSCAN may not be suitable
5. Optimal config recommendation: expected SIM≥0.9 + "both" mode