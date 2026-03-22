# Step 2b Review Summary
**Generated:** 2026-03-08
**Updated:** 2026-03-21
**Purpose:** Focused review doc — all items that changed, were reinterpreted, or corrected after the first Step 2b run

---

## Status Legend
- ✅ Output is correct, interpretation confirmed
- ⚠️ Output was reinterpreted / needs careful reading
- 🔁 Regenerated (new version created, original preserved)
- ❌ Bug found, fixed in current re-run

---

## Primary Analysis Cut (Earmarked)

| Dimension | Selection | Rationale |
|-----------|-----------|-----------|
| SIM threshold | **0.9** | Entry point to stable high-quality regime; ARI >0.7 with 0.95 and EDGE |
| Mode | **both** | Best EDGE validation + silhouette balance; limits multi-risk hub inflation while retaining SIM connections |
| EDGE fraction | **≥80%** | Gold-standard literature grounding; SIM≥0.9 achieves 81.6% gold-standard clusters |
| Algorithm | **Agglomerative k=40** | Silhouette 0.438 vs Louvain 0.011 vs HDBSCAN 0.275 (80.7% noise) |
| Reject | SIM 0.80/0.85 unconstrained | 27–73% EDGE validation; high artifact risk |
| Reject | HDBSCAN | 80.7% nodes classified as noise; collapses to ~3 clusters |
| Reject | Louvain | Silhouette 0.011; unsuitable |

**Note on lifecycle distribution (item 9):** The deployment-dominant lifecycle distribution in our corpus is a property of the historical ARD dataset, not a normative selection criterion. Effective interventions for x-risk may not be on the deployment side. Config selection must NOT be guided by lifecycle distribution.

---

## 1. Node Migration Heatmap (`node_migration_heatmap.png`)

**Status:** ❌ Do not cite
**Reason:** Migration rate is a metric artifact — it counts any cluster-ID change, even when destination cluster is semantically near-identical to source (mean cosine similarity 0.950 to destination centroids vs. random baseline 0.733).
**Action:** Drop from all citations. Use `centroid_similarity_heatmap.png` (item 2) exclusively for cross-threshold stability evidence.

---

## 2. Centroid Similarity Heatmap (`centroid_similarity_heatmap.png`)

**Status:** ✅ Output correct — primary stability citation
**Key finding:** All cross-threshold transitions >0.93 centroid cosine similarity.
**Confirmed real signal:** Inter-cluster baseline = 0.733 (off-diagonal mean from 40×40 matrix). Migrating nodes land at 0.950 — excess +0.217 above random. Clusters are genuinely distinct (range 0.47–0.94 off-diagonal).
**Workshop claim:** "Clustering is semantically stable across SIM thresholds — transitions represent short-range reorganization, not random resampling."

---

## 3. Hub Quality Scatter (`hub_quality_scatter.png` → v2: `hub_quality_scatter_v2.png`)

**Status:** 🔁 v2 generated (2026-03-21) — original preserved

**Why v2 was needed:**
- Original plot contained all 4 SIM threshold rows per hub (causing 4 artificial bands at y ≈ 70–100 / 500–635 / 1200–1232 / 1786–1844 sources). These bands are threshold artifacts, not hub structure.
- Structural degree z-axis irrelevant for SIM hubs (structural degree is EDGE-type connectivity, not SIM connectivity).

**v2 changes:**
- Filtered to primary analysis cut: `edge_config=0.9`, `mode=both`
- Dropped z-axis (no color encoding)
- Log-log axes for resolution across the full hub-size range
- Size encodes `degree_total_0.80` (total degree at storage floor)

**Hub size range (SIM 0.9 + both):**
- Hub #1: **635 SIM≥0.9 edges**, 1792 distinct partner papers
- Hub #100: **6 SIM≥0.9 edges**
- Strong cross-paper hubs (SIM≥0.9 ≥ 50): 140 rows in primary cut

**Confirmed finding:** Top-5 hubs are near-duplicate "Existential catastrophe from misaligned advanced AI" **risk** nodes from different papers — cross-paper convergence on the same concept, not extraction artifacts. Top hub: 635 SIM≥0.9 edges, 1792 partner papers.

**Main goal:** Demonstrate hubs are real semantic groupings, not pipeline artifacts. Evidence:
1. Top hubs are recognizable AI safety concepts shared across papers
2. Top-5 near-duplicates from different papers → cross-paper convergence
3. Hub #1 connects to 1792 distinct papers → not single-paper artifact

**TODO (Step 3):** Cluster top-100 hubs by cosine similarity to name hub themes and find unique bridge concepts despite non-deduplicated nodes.

---

## 4. Algorithm Comparison (`algorithm_comparison_silhouette.png`)

**Status:** ✅ Confirmed
**Results:**
- Agglomerative k=40: silhouette = **0.438** ✅ winner
- Louvain auto-k=42.6: silhouette = **0.011** ❌
- HDBSCAN: silhouette = **0.275**, noise_pct = **80.7%**, n_clusters = 3.0 ❌

**Conclusion:** Algorithm comparison complete. No further exploration needed in Step 3.

---

## 5. Source Diversity v2 (`cluster_source_diversity_v2.csv`)

**Status:** ✅ Correct
**Key finding:** n_sources ≈ cluster_size (r=0.887) — expected given no deduplication.
**Revised claim:** "Clusters have high source diversity by construction — not a meaningful differentiator between configs." Do not use as config selection criterion.

---

## 6. EDGE Validation Per Mode (`edge_validation_per_mode.png` → v2: `edge_validation_per_mode_v2.png`)

**Status:** 🔁 v2 generated (2026-03-21) — original preserved

**v2 changes:**
- Original bars were anonymous validation-rate buckets. v2 segments each bar by **node type** (8 types: risk, intervention, all_concepts + 5 concept categories)
- Grouped bars: x=edge config, one bar per node type, faceted 2×2 by mode
- Each bar labeled with validation rate percentage

**Key findings (unchanged):**
- EDGE-only: 100% validation for all node types
- SIM≥0.9 + "both": 90%+ for risk/intervention
- SIM≥0.85 + "both": 73% mean (varies by node type)
- SIM≥0.8 unconstrained: 27% (high artifact risk)

---

## 7. EDGE Purity Histograms (`edge_purity_histograms.png`)

**Status:** ✅ Correct
**Key finding:**
- Gold-standard clusters (purity ≥80%): EDGE=100%, SIM 0.95=98.7%, SIM 0.9=81.6%
- SIM≥0.9 is minimum threshold for reliable auto-labeling
- 66.8% of all clusters are gold-standard across all configs

---

## 8. ARI Pairwise + Line Plot (`cross_threshold_ari_lineplot.png`)

**Status:** ✅ Correct — retain as complementary evidence to centroid similarity

**What ARI measures:** Agreement between two cluster *assignment* solutions (0 = random, 1 = identical). Answers: "Are the same nodes grouped together across thresholds?" Centroid similarity answers: "Are the cluster *centers* semantically similar?" They are complementary — centroid can be high even if node assignments shuffle; ARI captures the structural membership agreement.

**Key finding:**
- SIM 0.9↔0.95 and 0.95↔EDGE: ARI >0.7 (meets stability target)
- SIM 0.85→0.9 transition: ARI 0.55–0.65 (below target)
- Lower thresholds (0.8, 0.85): qualitatively different clusterings

**Rationale for keeping ARI despite migration artifact concern:** ARI differs from node migration in a critical way. Migration rate counts raw cluster-ID changes. ARI is a probability-corrected pairwise measure — it asks whether node PAIRS are assigned together or apart, corrected for chance. The 0.85→0.9 ARI "drop" is useful and non-artifactual: it confirms that SIM 0.9 adds real discriminative power over 0.85 (the clustering is genuinely different, not just relabeled). The SIM 0.9↔0.95 ARI >0.7 is the key positive evidence: clustering is structurally stable in the high-threshold regime. **Use only high-threshold ARI pairs as evidence; flag low-threshold pairs as showing real qualitative divergence, not as stability claims.**

**Argument for SIM 0.9 despite lower 0.85→0.9 ARI:** The 0.85→0.9 drop is expected and desirable — 0.9 is the entry point to the stable high-quality regime. Below it clustering is qualitatively different; above it it stabilizes. This is the "sweet spot" argument: 0.9 provides both structural stability (ARI >0.7 above 0.9) and discriminative power (ARI shift below 0.9).

---

## 9. Maturity Distribution Heatmap (`maturity_distribution_heatmap.png`)

**Status:** ✅ Correct
**Scope clarification:** Intervention clusters only — add "Intervention Clusters Only" to plot title.

**Key finding:**
- Dominant stage distribution: Deployment=462, Training=207, Design=131 clusters
- Mean: Design 21%, Training 26%, Deployment 53%
- Only 1–2 clusters in top-25 are predominantly Design or Training stage

**Interpretation:** The deployment-dominant distribution is a corpus property of the ARD dataset, not a normative signal. **Do not use lifecycle distribution to guide config selection.** Effective interventions for existential risk may not be on the deployment side — especially given current frontier model situational awareness and capability for strategic deception. Deployment-dominance likely reflects historical research emphasis, not causal importance for x-risk reduction.

**Use as:** Enrichment / bias documentation in the limitations section only.

---

## 10. Mode Impact Analysis (`edge_density_heatmap.png`, `mode_stability_heatmap.png`)

**Status:** ✅ Correct
**Key finding:**
- "both" mode: best EDGE validation + silhouette balance → **primary cut**
- "unconstrained": highest cross-paper diversity but lowest EDGE purity
- "monotonic": highest structural integrity but over-constrains pathway space

**On over-constraining:** Some degree of over-constraining with "both" is acceptable and desirable — without it, the pathway space is too large to meaningfully analyze. "both" mode limits multi-risk hub inflation in pathways while retaining SIM-based cross-paper connections, striking the right balance between coverage and analysis tractability.

---

## 11. Mechanism Transfer Betweenness (`mechanism_transfer_betweenness_v2.png`)

**Status:** ✅ Correct output — interpretation updated

**What the score means:** Betweenness centrality = number of shortest paths between all node pairs in the full graph that pass through that node. Score of 88M ≈ 88 million shortest paths route through that node. This is **all-pairs betweenness** (not risk→intervention pairs specifically), pre-computed from FalkorDB using **all edges** (structural EDGE + SIM at storage floor SIM 0.80). The v2 plot uses this pre-computed attribute — it is NOT specific to SIM 0.9.

**Implication of all-pairs betweenness:** Scores are not biased toward path length 3 (triplets). They integrate across all path lengths. Nodes that serve as bridges between ANY frequently co-occurring node pairs score high — not just risk→intervention pairs. Top scorers are predominantly "problem analysis" nodes, suggesting problem statements are the primary structural bridges in the full graph regardless of path length.

**Near-duplicates in top-25 (non-deduplicated corpus):**
- #3 / #12: "Misalignment between AI objectives and human values in advanced AI systems" vs "Misalignment between AI objectives and human values" — near-duplicate from different papers
- #8 / #9 / #13: "reward function misspecification in advanced AI" / "Specification gaming in reinforcement learning agents" / "reward misspecification and specification gaming in RL agents" — cluster of 3 near-duplicates
- #15 / #16: "Current estimate of ~300 AI safety researchers" (validation evidence) vs "Insufficient talent pipeline for AI safety research" (problem analysis) — same real-world fact, different extraction framing

**TODO (Step 3):** Recompute betweenness on SIM≥0.9-filtered graph for a more representative analysis at the primary cut. Cluster top-50 betweenness nodes by cosine similarity to identify unique bridge themes, accounting for near-duplicates. This gives a deduplicated view without losing the distinct contextual contributions that make these nodes appear separately.

**Take-away for pathway set:** High-betweenness concept nodes are structural bridges between risk and intervention clusters. Include at least one problem_analysis node in every representative triplet for simulation.

---

## 12. Schema Corrections (background knowledge — no regeneration needed)

**FalkorDB graph_edge_data.pkl:**
- 88.6% SIMILARITY edges (1,565,684); 11.4% structural EDGE edges (202,149)
- `source_file` is NULL for ALL edges (both types); FROM edges absent from PKL

**SIMILARITY score → cosine similarity:**
```
cos_sim = 1 - score² / 2
```
Storage floor: max score = 0.6325 = SIM 0.80. All stored SIM edges have cos_sim ∈ [0.80, 0.989].

| Threshold | Score cutoff |
|-----------|-------------|
| cos_sim ≥ 0.80 | score ≤ 0.6325 |
| cos_sim ≥ 0.85 | score ≤ 0.5477 |
| cos_sim ≥ 0.90 | score ≤ 0.4472 |
| cos_sim ≥ 0.95 | score ≤ 0.3162 |

---

## Quick Summary Table

| Output | Status | Key change |
|--------|--------|------------|
| `node_migration_heatmap.png` | ❌ Drop | Metric artifact — do not cite |
| `centroid_similarity_heatmap.png` | ✅ Primary citation | >0.93 transitions = real stability signal |
| `hub_quality_scatter_v2.png` | 🔁 Regenerated | SIM 0.9 + both only; log-log; no z-axis |
| `edge_validation_per_mode_v2.png` | 🔁 Regenerated | Node-type labeled bars |
| `hub_quality_metrics.csv` | ✅ Correct | Hub #1: 635 SIM≥0.9 edges; Hub #100: 6 edges |
| `algorithm_comparison_silhouette.png` | ✅ Confirmed | Agglomerative wins; HDBSCAN/Louvain ruled out |
| `cluster_source_diversity_v2.csv` | ✅ Correct | Not a useful config discriminator |
| `cross_threshold_ari_lineplot.png` | ✅ Keep | Use only 0.9↔0.95↔EDGE pairs as stability evidence |
| `maturity_distribution_heatmap.png` | ✅ Add title note | Intervention clusters only; deployment bias = corpus artifact |
| All other outputs | ✅ Unchanged | No schema errors found |

---

## Next Steps (Step 3 — updated 2026-03-21)

**CRITICAL (must complete for workshop):**
1. Multi-criteria config scoring (substep #23): silhouette 25% + EDGE% 30% + cluster count 20% + ARI 15% + interpretability 10% — produces ranked table of all 160 configs
2. Final optimal config selection with full transparency: expected SIM≥0.9 + "both" mode confirmed, or corrected by scoring
3. Confirm "both" is the best mode for unique risk-intervention pathway grouping (not just silhouette/EDGE balance) — if another mode scores better on pathway uniqueness, use that for hub clustering analysis

**ESSENTIAL (Step 3):**
4. Cluster top-100 hubs (by SIM≥0.9 degree) by cosine similarity → name hub themes across risk, intervention, concept node types; report hub #1 and hub #100 degree per node type
5. Recompute betweenness on SIM≥0.9-filtered graph; cluster top-50 betweenness nodes to find unique bridge themes (accounts for near-duplicates)
6. Edge threshold sensitivity confirmation: document SIM 0.9 as "sweet spot" entry point using ARI evidence

**ENRICHMENT (Step 3):**
7. EDGE-only vs SIM≥0.9 cluster quality comparison (substep #22)
8. Held-out test set validation (substep #24)

**Step 4 (Taxonomy Construction — after optimal config locked):**
1. Cluster naming + manual validation (substep #26) — CRITICAL for workshop (Goal 6)
2. Risk→Concept→Intervention triplet formation (substep #17)
3. Exemplar path extraction per cluster (substep #18)
4. Risk-intervention connectivity matrix (substep #28)
5. Simulation-ready prompt templates (substeps #33–34)
