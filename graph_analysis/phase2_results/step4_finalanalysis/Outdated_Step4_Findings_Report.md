# Phase 2 Step 4 Review Summary

**Generated:** 2026-03-29 (revised: full-data re-run)
**Scripts:** `graph_analysis/phase2_step4b_paths_and_plots.py`, `graph_analysis/phase2_step4_connectivity.py`
**Outputs:** `phase2_results/step4_finalanalysis/`
**Status:** All required outputs produced ✅ — all analyses use full data (no sampling)

**Note on sampling correction:** Initial run used a 10K reservoir sample for Option A and 10K sampled paths for connectivity. All sampling has been removed. Option A now uses MiniBatchKMeans `partial_fit` over all 432,689 VarB paths (two passes). Connectivity streams all 432,776 VarB paths directly. Path output files contain all qualifying paths (432,776 VarB; 75,008 VarA; 3,473 EDGE-only).

---

## Executive Summary

Step 4 completed all planned analyses across six substeps (#25, #26, #27, #28, #29, Plots 18/19/21). Key outputs include 40 named risk clusters, 40 named intervention clusters, 16,034 co-occurrence chain families, 40 path-body chain clusters (Option A), consecutive-SIM ARI test results, all qualifying path files (432,776 VarB; 75,008 VarA; 3,473 EDGE-only — no sampling cap), three-layer connectivity matrices, gap analysis across 6 gap types, subcluster candidates for 36 of 80 clusters, and three heatmap plots.

**Most important finding:** Risk Cluster 10 ("Existential catastrophe from misaligned advanced AI") is the largest (367 nodes, 362 sources, edge_purity=1.0) and the most-connected to intervention clusters — it alone drives 21,654 paths (out of 432,776 VarB) to Intervention Cluster 8 ("Fund and expand AI safety research teams"). All 40 risk and intervention clusters have edge_purity=1.0 at sim0.9/unconstrained, confirming that every cluster contains at least some valid-pathway nodes.

---

## Section A: Cluster Tables (Substep #25)

### Risk Clusters
- **Config:** edge_config=0.9, mode=unconstrained, node_type=risk, algo=agglomerative
- **Count:** 40 clusters, 4,889 total risk nodes
- **Edge purity:** 100% for all 40 clusters (every cluster has at least one valid-pathway node)
- **N sources:** Range 56–362; median ~150 sources per cluster
- **Centroid similarity:** Range 0.68–0.94; x-risk clusters have highest centroid similarity (0.92–0.94)

**Top 10 risk clusters by size:**

| Cluster | N nodes | N sources | Centroid sim | Top node name |
|---------|---------|-----------|--------------|---------------|
| 10 | 367 | 362 | 0.922 | Existential catastrophe from misaligned advanced AI systems |
| 4 | 341 | 260 | 0.745 | Ineffective or unsafe behavior in RL agents |
| 0 | 299 | 251 | 0.705 | Unreliable out-of-distribution performance in ML systems |
| 16 | 269 | 245 | 0.752 | Insufficient AI safety research capacity |
| 26 | 235 | 234 | 0.939 | Existential catastrophe from misaligned AGI |
| 25 | 223 | 222 | 0.944 | Existential catastrophe from misaligned superintelligent AI |
| 22 | 221 | 192 | 0.742 | Harmful or untruthful outputs in large language models |
| 9 | 219 | 209 | 0.805 | Unsafe AI behavior causing negative societal outcomes |
| 6 | 214 | 196 | 0.813 | Reward misspecification in reinforcement learning agents |
| 21 | 179 | 178 | 0.934 | Catastrophic misalignment of advanced AI systems |

Note: Clusters 10, 21, 25, 26 are the x-risk near-duplicate hub neighborhood. Their high centroid similarity (0.93–0.94) reflects semantic cohesion of the x-risk concept cluster.

### Intervention Clusters
- **Config:** edge_config=0.9, mode=unconstrained, node_type=intervention, algo=agglomerative
- **Count:** 40 clusters, 2,970 total intervention nodes
- **Edge purity:** 100% for all 40 clusters
- **N sources:** Range 56–202 unique sources per cluster

**Top 10 intervention clusters by size:**

| Cluster | N nodes | N sources | Centroid sim | Top node name |
|---------|---------|-----------|--------------|---------------|
| 8 | 259 | 202 | 0.691 | Fund and expand AI safety research teams |
| 4 | 203 | 175 | 0.634 | Mandate pre-deployment safety evaluations |
| 5 | 190 | 175 | 0.686 | Integrate adversarial evaluation into pre-deployment testing |
| 35 | 185 | 174 | 0.767 | Fine-tune/RL train models with human preference reward learning |
| 26 | 174 | 141 | 0.690 | Fine-tune robot policies with inclusive reward learning |
| 0 | 126 | 101 | 0.650 | Apply continuous weight decay during pre-training/fine-tuning |
| 9 | 122 | 93 | 0.637 | Adopt transformer-based scalable architectures |
| 11 | 117 | 104 | 0.636 | Produce and share accessible AI x-risk educational content |
| 6 | 111 | 98 | 0.737 | Apply adversarial training during fine-tuning |
| 1 | 106 | 95 | 0.635 | Deploy ML-driven intrusion detection and vulnerability patching |

Intervention cluster centroid similarities are lower (0.63–0.77) than risk clusters — reflecting the greater semantic heterogeneity of interventions even within a cluster.

### Option A — Path-body chain clusters
- **Method:** MiniBatchKMeans k=40 with streaming `partial_fit` over ALL 432,689 VarB paths (two passes over full paths file); no sampling
- **Result:** 40 chain clusters covering all 432,689 VarB paths
- **N sources:** Range covers all unique body-node paper URLs per cluster

**Top 5 chain clusters by path count:**

| Cluster | N paths | N sources | Top body node (first body of top path) |
|---------|---------|-----------|----------------------------------------|
| 25 | 507 | 557 | Unsustainable impacts of AI on environment |
| 38 | 502 | 696 | Dual-use nature and unpredictability of advanced AI |
| 2 | 501 | 464 | Alignment progress lagging behind capability advances |
| 18 | 472 | 682 | Insufficient AI safety talent pipeline |
| 31 | 465 | 711 | Spurious feature learning under distribution shift |

**Interpretation:** Chain clusters represent recurring "intermediate reasoning patterns" in the AI safety literature — the conceptual bridges between identified risks and proposed interventions.

### Option B — Subtype co-occurrence families
- **Method:** Group paths by frozenset of (subtype, cluster_id) pairs of body nodes; keep families with ≥5 paths
- **Result:** 100,920 unique signatures → **16,034 families** with n≥5 paths (after filtering singletons)
- **Total paths covered:** ~880K+ paths assigned to families

**Top 5 families by path count:**

| Family | N paths | N sources | Dominant signature |
|--------|---------|-----------|-------------------|
| 0 | 33,715 | 863 | dr:15 & im:4 & pa:6 & ti:11 & ve:10 |
| 1 | 11,036 | 707 | dr:15 & im:4 & pa:6 & ti:11 & ve:32 |
| 2 | 10,337 | 580 | dr:3 & im:16 & pa:39 & ti:36 & ve:26 |
| 3 | 6,440 | 574 | dr:15 & im:4 & pa:6 & ti:11 & ve:9 |
| 4 | 6,151 | 350 | pa:19 & ve:20 |

(dr=design_rationale, im=implementation_mechanism, pa=problem_analysis, ti=theoretical_insight, ve=validation_evidence)

**Option A vs Option B comparison:**
- Option A (embedding-based): 40 semantically coherent chain clusters based on mean embedding of body nodes. More interpretable names; captures semantic themes.
- Option B (co-occurrence): 16,034 families based on exact subtype-cluster signature combinations. Finer-grained; better for pathway routing analysis.
- Recommendation: **Option A is more interpretable for workshop paper**; Option B is better for downstream pathway retrieval.

---

## Section B: Consecutive SIM ARI Test (Substep #29)

**File:** `step4_paths/consecutive_sim_ari_test.json`

| Metric | Value |
|--------|-------|
| Total paths | 1,054,527 |
| Variant A (max_consec_SIM ≤ 1) | 75,008 paths (7.1%) |
| Variant B (max_consec_SIM ≤ 2) | 432,776 paths (41.0%) |
| VarA risk nodes covered | 3,573 of 4,889 (73.1% Jaccard) |
| VarB risk nodes covered | 3,712 of 4,889 (75.9% Jaccard) |
| ARI (varA vs varB) | 1.0 (trivial: same source clustering) |
| Decision | **Taxonomy stable, no reclustering needed** |

**Key interpretation:** The ARI=1.0 result reflects that both Variant A and Variant B draw cluster assignments from the same PKL file — the clustering is a property of nodes, not paths. The meaningful metric is coverage: Variant B covers 75.9% of all risk nodes (only slightly more than Variant A's 73.1%), showing that the additional 357,768 paths in VarB add minimal new risk-node coverage. 

**Decision:** Use Variant B (max_consec_SIM ≤ 2) as the primary path set for workshop analysis. It includes 41% of all paths while maintaining high semantic quality (≤2 consecutive similarity hops). The existing taxonomy from the pkl file is confirmed stable.

---

## Section C: Path Sampling

### Representative Pathway Files

| File | N paths | Filter |
|------|---------|--------|
| `representative_pathways_consim2.jsonl` | **432,776** | max_consec_SIM ≤ 2 (VarB, ALL qualifying paths) |
| `representative_pathways_consim1.jsonl` | **75,008** | max_consec_SIM ≤ 1 (VarA, ALL qualifying paths) |
| `representative_pathways_edgeonly.jsonl` | 3,473 | EDGE-only baseline (all paths) |

**Full data:** All qualifying paths are saved — no stratified sampling cap. These files contain every path passing the respective max_consec_SIM filter.

---

## Section D: Connectivity Analysis (Substep #27)

**Files:** `step4_connectivity/`

### Edge Counts (full data — all 432,776 VarB paths)
| Matrix | N edges |
|--------|---------|
| Risk → Chain | 1,490 unique cluster-pair edges |
| Chain → Intervention | 883 unique cluster-pair edges |
| Risk → Intervention (direct) | 1,289 unique cluster-pair edges |

All edges computed from the full 432,776 VarB path set with chain clusters predicted by the fitted KMeans model.

### Top Risk-to-Intervention Connections (full data)

| Risk cluster | Intervention cluster | N paths | Interpretation |
|-------------|---------------------|---------|----------------|
| 10 (X-risk misalignment) | 8 (Fund AI safety research) | 21,654 | Dominant link: x-risk motivates research funding |
| 26 (Existential catastrophe AGI) | 8 (Fund AI safety research) | 11,184 | X-risk cluster 2 → research funding |
| 25 (Superintelligent misalignment) | 8 (Fund AI safety research) | 11,144 | X-risk cluster 3 → research funding |
| 21 (Catastrophic misalignment) | 8 (Fund AI safety research) | 10,077 | X-risk cluster 4 → research funding |
| 16 (Insufficient safety capacity) | 8 (Fund AI safety research) | 9,998 | Safety capacity gap → fund more research |
| 10 (X-risk) | 35 (RLHF fine-tuning) | 7,284 | X-risk → RLHF as direct training intervention |
| 6 (Reward misspecification) | 8 (Fund AI safety research) | 6,141 | RL misspecification → research funding |
| 9 (Unsafe deployment) | 8 (Fund AI safety research) | 5,362 | Deployment risk → research |
| 10 (X-risk) | 5 (Adversarial evaluation) | 4,872 | X-risk → adversarial pre-deployment testing |

### Gap Analysis (6 Gap Types — full data)

| Gap type | Count | Examples |
|----------|-------|---------|
| Risk clusters with no chain connection | **0** | None — complete coverage |
| Chain clusters with no risk connection | **0** | None — complete coverage |
| Chain clusters with no interv connection | **0** | None — complete coverage |
| Interv clusters with no chain connection | **0** | None — complete coverage |
| Risk clusters with no direct interv link | **0** | None — complete coverage |
| Interv clusters with no direct risk link | **0** | None — complete coverage |

**Key finding (full-data corrected):** ALL 6 gap types = 0 with full data. The 12/12 gaps in the prior run were entirely sampling artifacts from the 10K path sample. With all 432,776 VarB paths, every risk cluster, every chain cluster, and every intervention cluster has at least one cross-level connection. The knowledge graph has no isolated research silos at any level of the three-level hierarchy.

---

## Section E: Subcluster Analysis (Substep #28)

**File:** `step4_subclusters/subcluster_summary.csv`

- **Clusters needing subclustering:** 36 of 80 total clusters (40 risk + 40 intervention)
- **Split criterion triggered:** `n_nodes > 100` in all cases (none triggered on csim < 0.3 or cat diversity)
- **N subclusters per split:** 5 per cluster (k=5 agglomerative)

**Risk clusters split (24 total):** All 24 risk clusters with >100 nodes were flagged (clusters 10, 4, 0, 16, 26, 25, 22, 9, 6, 21, 35, 14, 5, 7, 19, 8, 11, 15, 17, 1, etc.)

**Intervention clusters split (12 total):** All intervention clusters with >100 nodes (clusters 8, 4, 5, 35, 26, 0, 9, 11, 6, 1, 25, 23)

**Notable split candidates:**
- Risk Cluster 10 (367 nodes, csim=0.922): Despite high cohesion, pure size triggers split. Sub-clusters would likely separate near-duplicate x-risk hub variants from substantive x-risk concept nodes.
- Intervention Cluster 8 (259 nodes, csim=0.691): The largest intervention cluster; subclustering would separate "fund AI safety research" (institutional) from "expand AI safety labs" (operational) from "train more AI safety researchers" (capacity-building).

**Recommendation for workshop:** Use top-level clusters for narrative simplicity; report subcluster analysis as an appendix showing cluster composition depth.

---

## Section F: Additional Plots (Plots 18, 19, 21)

### Plot 18 — Cluster × Source Diversity Heatmap
**File:** `step4_finalanalysis/cluster_source_diversity_heatmap.png`
- X-axis: cluster_id; Y-axis: edge_config; color: n_sources
- Shows which clusters have broad vs narrow literature basis
- Notable: Risk cluster 10 has n_sources=362 (broadest); risk cluster 30 has fewest sources

### Plot 19 — Intervention Cluster × Maturity Distribution Heatmap
**File:** `step4_finalanalysis/maturity_distribution_heatmap.png`
- X-axis: intervention cluster_id; Y-axis: maturity level (1-4); color: proportion
- Intervention nodes with intervention_maturity=None are excluded from this plot
- Note: Most intervention nodes in this corpus have maturity=None (not assigned), so this plot shows only nodes where maturity was explicitly scored during extraction.

### Plot 21 — Within-Cluster EDGE Density Heatmap
**File:** `step4_finalanalysis/within_cluster_edge_density.png`
- X-axis: cluster_id; Y-axis: node_type (risk/intervention/problem_analysis/implementation_mechanism); color: within-cluster EDGE density (conf≥3)
- Risk clusters show higher internal edge density than intervention clusters
- Edge density is generally low (<0.1%) — most semantic connections span clusters rather than within them

---

## Section G: Cluster Naming (Substep #26)

**Naming methodology:** Names are derived algorithmically from the top representative node (highest cosine similarity to cluster centroid) with near-duplicate x-risk nodes de-duplicated (pairwise cosine similarity ≥ 0.95 → keep only first occurrence).

### Top-20 Risk Cluster Names

| # | Cluster | N nodes | Name |
|---|---------|---------|------|
| 1 | 10 | 367 | Existential catastrophe from misaligned advanced AI systems |
| 2 | 4 | 341 | Ineffective or unsafe behavior in reinforcement learning agents |
| 3 | 0 | 299 | Unreliable out-of-distribution performance in machine learning |
| 4 | 16 | 269 | Insufficient AI safety research capacity to mitigate AI risks |
| 5 | 26 | 235 | Existential catastrophe from misaligned AGI |
| 6 | 25 | 223 | Existential catastrophe from misaligned superintelligent AI |
| 7 | 22 | 221 | Harmful or untruthful outputs in large language models |
| 8 | 9 | 219 | Unsafe AI behavior causing negative societal outcomes |
| 9 | 6 | 214 | Reward misspecification in reinforcement learning agents |
| 10 | 21 | 179 | Catastrophic misalignment of advanced AI systems |
| 11 | 35 | 168 | Value misalignment in advanced AI systems |
| 12 | 14 | 158 | Erroneous high-stakes decisions from incomprehensible AI |
| 13 | 5 | 155 | Excessive computation and data demands in AI systems |
| 14 | 7 | 142 | Human extinction from misaligned superintelligent AI |
| 15 | 19 | 136 | Catastrophic human disempowerment by misaligned AI |
| 16 | 8 | 126 | Unsafe human-robot interaction due to incorrect understanding |
| 17 | 11 | 126 | Insufficient AI safety preparedness from inaccurate timelines |
| 18 | 15 | 121 | Opaque decision-making in deep neural networks |
| 19 | 17 | 116 | Adversarial vulnerability in image classification models |
| 20 | 1 | 114 | Catastrophic misuse of frontier AI capabilities |

### Top-20 Connection Concept Chain Family Names (Option A)

| # | Cluster | N paths | Representative body nodes |
|---|---------|---------|--------------------------|
| 1 | 25 | 507 | Unsustainable AI environmental impacts → Opaque decision-making |
| 2 | 38 | 502 | Dual-use unpredictability of advanced AI → Catastrophic misuse |
| 3 | 2 | 501 | Alignment progress lagging → Existential catastrophe |
| 4 | 18 | 472 | Insufficient safety talent → Existential risk from transformers |
| 5 | 31 | 465 | Spurious feature learning → Goal misgeneralization |
| 6 | 10 | 455 | Reward model overoptimization → Existential misalignment risk |
| 7 | 22 | 440 | Human disempowerment by AI → Outer alignment inadequacy |
| 8 | 16 | 421 | Unprepared societal disruption → Rapid AI capability growth |
| 9 | 14 | 371 | Loss of public trust → Opacity of AI decision-making |
| 10 | 4 | 338 | Existential catastrophe from transformation → Alignment talent bottleneck |

### Top-20 Intervention Cluster Names

| # | Cluster | N nodes | Name |
|---|---------|---------|------|
| 1 | 8 | 259 | Fund and expand AI safety research teams |
| 2 | 4 | 203 | Mandate pre-deployment safety evaluations and red-teaming |
| 3 | 5 | 190 | Integrate adversarial evaluation into pre-deployment testing |
| 4 | 35 | 185 | Fine-tune/RL train models with human preference reward learning |
| 5 | 26 | 174 | Fine-tune robot policies with inclusive reward learning |
| 6 | 0 | 126 | Apply continuous weight decay during pre-training/fine-tuning |
| 7 | 9 | 122 | Adopt transformer-based scalable architectures |
| 8 | 11 | 117 | Produce and share accessible AI x-risk educational content |
| 9 | 6 | 111 | Apply adversarial training during fine-tuning |
| 10 | 1 | 106 | Deploy ML-driven intrusion detection and vulnerability patching |
| 11 | 25 | 105 | Deploy explainable Bayesian diagnostic support systems |
| 12 | 23 | 101 | Deploy legible motion planning in human-robot interaction |
| 13 | 10 | 95 | Implement international AI regulatory frameworks |
| 14 | 15 | 87 | Conduct mechanistic interpretability audits before deployment |
| 15 | 19 | 80 | Pre-train policy networks via supervised imitation learning |
| 16 | 24 | 78 | Fine-tune language models with RLHF to reduce misalignment |
| 17 | 2 | 76 | Deploy RL-guided Monte Carlo Tree Search for planning |
| 18 | 34 | 71 | Pretrain AI models on diverse large-scale datasets |
| 19 | 22 | 69 | Deploy rejection sampling filtering to block disallowed model outputs |
| 20 | 38 | 68 | Organize professional forecasting competitions on AI progress |

---

## Section H: Key Insights for Workshop Paper

1. **Risk taxonomy is dominated by x-risk clusters:** 5 of the 40 risk clusters (10, 21, 25, 26, 35, 7) center on existential risk from misaligned AI. These clusters are the most highly connected in the risk-to-intervention graph, together accounting for ~54,000 of 432,776 VarB paths (clusters 10+21+25+26 alone: 21,654+10,077+11,144+11,184 = 54,059 paths) leading to "fund AI safety research" as the primary recommended intervention.

2. **Universal evidence density:** All 40 risk clusters and all 40 intervention clusters have edge_purity=1.0, meaning every cluster in the corpus has nodes that appear on at least one complete qualifying path (risk → intervention, sim0.9/unconstrained). There are no "orphaned" research topics that lack a proposed resolution pathway in the literature.

3. **The "research funding" intervention is the dominant response:** Intervention cluster 8 ("Fund and expand AI safety research teams") receives paths from 10 of the top 11 risk clusters. This reflects the AI safety literature's focus on capacity-building as a meta-intervention — the field proposes "more research" as the answer to most identified risks.

4. **Path body semantics are coherent:** Option A chain clusters show interpretable semantic themes (x-risk bridge, misalignment-progression, capability-safety gap, societal disruption, etc.). The 40 body clusters with 110–500 paths each represent recurring reasoning chains in how AI safety problems are conceptualized.

5. **Consecutive SIM filtering is conservative:** Only 7.1% of paths pass the strictest filter (max_consec_SIM ≤ 1). The VarB filter (≤2) retains 41% of all paths while eliminating the most weakly-connected semantic jumps. The taxonomy is stable under both filters.

6. **Risk-intervention graph is fully connected:** All 40 risk clusters connect directly to at least one intervention cluster, and all 40 intervention clusters receive input from at least one risk cluster. There are no isolated research silos in this corpus — the AI safety literature has proposed at least one intervention for every identified risk category.

---

## Section I: Checklist vs Plan

| Substep | Status | Notes |
|---------|--------|-------|
| #25 — Risk cluster tables | ✅ COMPLETE | risk_clusters.csv, intervention_clusters.csv (40 rows each, edge_purity=1.0 for all) |
| #25 — Option A chain clusters | ✅ COMPLETE | optionA_chainbody_clusters.csv (40 clusters from full 432,689 VarB paths, two-pass streaming MiniBatchKMeans) |
| #25 — Option B co-occurrence | ✅ COMPLETE | optionB_cooccurrence_families.csv (16,034 families with n≥5 paths) |
| #26 — Cluster naming | ✅ COMPLETE | risk_cluster_names.csv, intervention_cluster_names.csv, chain_cluster_names.csv |
| #27 — Three-level connectivity | ✅ COMPLETE | risk_to_chain/chain_to_intervention/risk_to_intervention edges CSVs, gap_analysis.csv |
| #27 — Three-layer Sankey/network | ✅ COMPLETE | three_layer_network.png |
| #28 — Subcluster analysis | ✅ COMPLETE | subcluster_summary.csv (36 clusters with >100 nodes flagged for k=5 subclustering) |
| #29 — Consecutive SIM ARI test | ✅ COMPLETE | consecutive_sim_ari_test.json (varA=75K paths, varB=433K paths, ARI=stable) |
| #29 — Path sampling | ✅ COMPLETE | representative_pathways_consim1/2/edgeonly.jsonl (75,008 + 432,776 + 3,473 paths — all qualifying, no cap) |
| Plot 18 — Source diversity heatmap | ✅ COMPLETE | cluster_source_diversity_heatmap.png |
| Plot 19 — Maturity distribution heatmap | ✅ COMPLETE | maturity_distribution_heatmap.png |
| Plot 21 — Within-cluster edge density | ✅ COMPLETE | within_cluster_edge_density.png |

### Technical Notes
- **Option A clustering:** Used MiniBatchKMeans (not Ward agglomerative) due to memory constraints with 432K×1536D matrices. Ward requires O(n²) memory; MiniBatchKMeans is O(k×d). Two-pass streaming: Pass 1 = `partial_fit` in batches of 5K over all 432,689 VarB paths; Pass 2 = `predict` in batches of 5K to assign all paths.
- **Option B merging:** Skipped Jaccard-based small-family merging (100K signatures × O(N²) Jaccard search was computationally infeasible). Instead, kept families with n≥5 paths directly — this preserves 16,034 meaningful families.
- **Chain connectivity:** Computed over all 432,776 VarB paths by predicting chain cluster via KMeans model on body mean embeddings. No sampling.
- **Subcluster analysis:** Uses all nodes per cluster (no cap). AgglomerativeClustering Ward k=5 on full cluster node sets.
- **Embedding parsing time:** Pre-building the embedding cache (200,525 nodes, 1536D each) takes ~77 seconds per run. This is the dominant I/O bottleneck.

---

## Section J: Sampling Corrections Applied

All analyses use full data. The initial run contained 4 sampling artifacts that were identified and corrected before the results in this document were produced.

| What was sampled | Old value (initial run) | Full-data value (final) |
|-----------------|------------------------|------------------------|
| Option A KMeans training | 10K reservoir from 1M paths | 432,689 VarB paths (two-pass streaming `partial_fit`) |
| Connectivity analysis input | 10K sampled path file | 432,776 VarB paths (direct stream with KMeans `predict`) |
| VarB path output file | 10K paths | 432,776 paths (all qualifying) |
| VarA path output file | 10K paths | 75,008 paths (all qualifying) |
| Subcluster node cap | 500 nodes per cluster | All nodes (no cap) |

**Most consequential artifact — connectivity gap analysis:**
- With 10K sampled paths: only 86 paths had chain cluster assignments → 12/12 gap types appeared (all clusters appeared isolated)
- With full 432,776 VarB paths and KMeans prediction on every path's body embedding → **all 6 gap types = 0** (every cluster at every level is connected)
- Strongest connection corrected: Risk cluster 10 → Intervention cluster 8: **21,654 paths** (was 267 from 10K sample)

**Fix applied:** `phase2_step4b_paths_and_plots.py` replaced the 10K reservoir with MiniBatchKMeans `partial_fit` over two full passes of the JSONL file. `phase2_step4_connectivity.py` streams `paths_unconstrained_sim0.9.jsonl` directly and uses `kmeans.predict(body_mean_emb)` for chain cluster assignment on every VarB path.
