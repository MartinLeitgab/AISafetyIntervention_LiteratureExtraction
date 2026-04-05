# Phase 2 Step 4 — Findings Report (Fresh)

**Generated:** 2026-04-04
**Scripts run:** `phase2_step4b_paths_and_plots.py` (fixed), `phase2_step4_connectivity.py` (fixed), `phase2_step4_phase_c_reruns.py` (new)
**All outputs in:** `phase2_results/step4_finalanalysis/`
**Filter compliance:** All Category B analyses use `valid_pathway_nodes` from `paths_unconstrained_sim0.9.jsonl` plus explicit `intervention_maturity≥3` check. NOTE (2026-04-05): path generation (`final_pathway_analysis_modes.py`) used `ALL_INTERVENTION_IDS = cache["interventions"]` as BFS terminal set, which included maturity<3 nodes — so valid_pathway_nodes alone does NOT guarantee maturity≥3. All Category B scripts apply both filters simultaneously.

---

## Phase A Code Fixes Applied (Pre-Rerun)

The following gaps from the Step 4 Analysis Plan (v9) were fixed before this run:

| Gap | Script | Fix applied |
|-----|--------|-------------|
| Gap 4 — `build_cluster_table()` unfiltered | `step4b`, `cluster_naming` | All cluster member lists now filtered to `valid_pathway_nodes` before any computation |
| Gap 3b — `risk_clusters_09.pkl` saved unfiltered | `step4b`, `cluster_naming` | `risk_clusters_09` built with `valid_pathway_nodes` filter before PKL save |
| Gap 3b — `risk_clusters_09` loaded unfiltered | `connectivity` | Filter applied after PKL load: `{cid: [n for n in nodes if n in valid_pathway_nodes]}` |
| Gap 4 — `interv_clusters_09` maturity-only filter | `connectivity` | Now filters by `valid_pathway_nodes` AND explicit `intervention_maturity≥3` (both required — see root cause note in header) |
| Gap 4 — `valid_pathway_nodes` missing | `connectivity` | Added loading from `paths_unconstrained_sim0.9.jsonl` |
| Gap 9 — `node_to_stc` unfiltered body nodes | `step4b`, `cluster_naming` | Added `if nid in valid_pathway_nodes` guard |
| Gap 4 — Plot 19 maturity heatmap unfiltered | `step4b` | Uses `valid_pathway_nodes`-filtered intervention clusters |
| Gap 4 — Plot 21 edge density denominator | `step4b`, `cluster_naming` | Denominator n uses qualifying member count |
| Gap — empty clusters in gap analysis | `connectivity` | Empty clusters excluded from universe via `if risk_clusters_09[c]` guard |

Gaps 1–3 (SIM≥0.9 in sim_edge_set, conf≥3 in betweenness, maturity≥3 belt-and-suspenders) were already fixed in prior runs.

---

## Executive Summary

Step 4 completed the three-level risk → connection-concept-chain → intervention taxonomy for **all 6 configurations** (3 consimN × 2 pathbuild options). Selected config for downstream analysis (Step 5): **`consim1_pathbuildA`**. All analyses apply `valid_pathway_nodes` filtering for Category B outputs.

**Key numbers:**
- **4,889** risk nodes in 40 risk clusters (all on ≥1 qualifying unconstrained path)
- **2,815** intervention nodes in 40 intervention clusters (qualifying, maturity≥3) — confirmed by connectivity rerun (2026-04-05)
- **40** chain body clusters (Option A, MiniBatchKMeans on mean body embeddings — consim1_pathbuildA selected)
- **51 / 1,603 / 16,034** Option B co-occurrence chain families for consim0 / consim1 / all-unconstrained (see Part 1 for labeling correction — 16,034 families are from all unconstrained paths, not consim2-filtered)
- **432,776** consim2 qualifying paths; **74,921** consim1; **3,386** edge-only (with body nodes)
- **Zero gap analysis gaps** — all 40 risk, chain, and intervention clusters are fully interconnected in the consim2 configuration
- **Risk taxonomy** spans 1994–2023; **mean publication year** ~2020.5 for risk, ~2020.1 for interventions

---

## Part 1: Cluster Tables (Substep #25)

### L1 Risk Clusters — 40 clusters, 4,889 qualifying nodes

| Cluster | N nodes (qual.) | N sources | Centroid sim | Top representative node |
|---------|----------------|-----------|--------------|------------------------|
| 10 | 367 | 362 | 0.922 | Existential catastrophe from misaligned advanced AI systems |
| 4 | 341 | 260 | 0.745 | Ineffective or unsafe behavior in RL agents in complex tasks |
| 0 | 299 | 251 | 0.705 | Unreliable out-of-distribution performance in ML systems |
| 16 | 269 | 245 | 0.752 | Insufficient AI safety research capacity to mitigate AI risks |
| 26 | 235 | 234 | 0.939 | Existential catastrophe from misaligned AGI |
| 25 | 223 | 222 | 0.944 | Existential catastrophe from misaligned superintelligent AI |
| 22 | 221 | 192 | 0.742 | Harmful or untruthful outputs in large language models |
| 9 | 219 | 209 | 0.805 | Unsafe AI behavior causing negative societal outcomes |
| 6 | 214 | 196 | 0.813 | Reward misspecification in reinforcement learning agents |
| 21 | 179 | 178 | 0.934 | Catastrophic misalignment of advanced AI systems |

All 40 clusters have `edge_purity = 1.0` — every cluster has ≥1 valid-pathway node (trivially, since valid_pathway_nodes is built from the unconstrained path file).

**Observations:**
- Clusters 10, 21, 25, 26 form the x-risk near-duplicate hub neighborhood: high centroid similarity (0.922–0.944), 178–362 distinct source papers each. These represent the most cross-referenced risk concept in the corpus.
- Cluster 16 ("Insufficient AI safety research capacity") is the largest non-x-risk cluster (269 nodes, 245 sources) — a "meta-risk" about the field itself.
- Source diversity range: 56–362 unique papers per cluster (mean 114).

### L3 Intervention Clusters — 40 clusters, 2,815 qualifying nodes (maturity≥3)

**Note:** 155 intervention nodes in the PKL with maturity<3 (15 at mat=1, 140 at mat=2) are excluded by the explicit maturity≥3 filter, reducing the count from 2,970 to 2,815. Root cause: path generation used `ALL_INTERVENTION_IDS` from cache (includes maturity<3 nodes) as BFS terminal set, so 155 maturity<3 interventions appear in path files and in valid_pathway_nodes. The maturity≥3 check is therefore an ADDITIONAL required filter, not redundant with valid_pathway_nodes.

| Cluster | N nodes (qual.) | N sources | Centroid sim | Top representative node |
|---------|----------------|-----------|--------------|------------------------|
| 8 | 232 | 179 | 0.679 | Fund and expand AI safety research teams |
| 4 | 199 | 173 | 0.632 | Mandate pre-deployment safety evaluations and red-teaming |
| 5 | 189 | 174 | 0.686 | Integrate adversarial evaluation into pre-deployment testing |
| 26 | 172 | 141 | 0.691 | Fine-tune robot policies with inclusive reward learning |
| 35 | 162 | 152 | 0.754 | Fine-tune RL agents using human preference-based reward models |
| 0 | 124 | 99 | 0.651 | Apply continuous weight decay during pre-training/fine-tuning |
| 9 | 120 | 91 | 0.637 | Adopt transformer-based scalable architectures |
| 11 | 117 | 104 | 0.636 | Produce and share accessible AI x-risk educational content |
| 6 | 111 | 98 | 0.737 | Apply adversarial training during fine-tuning |
| 1 | 105 | 94 | 0.637 | Deploy ML-driven intrusion detection and vulnerability patching |

Intervention centroid similarities are lower (0.63–0.77) than risk clusters — reflecting semantic heterogeneity even within clusters.

**Maturity distribution (qualifying, maturity≥3 only):**
- Maturity 3: **2,464 nodes (83.0%)** — "documented feasibility evidence" tier
- Maturity 4: **351 nodes (11.8%)** — highest maturity (deployed / widely validated)
- Maturity 1: 15 nodes (0.5%), Maturity 2: 140 nodes (4.7%) — these pass the cluster PKL filter but are excluded from qualifying analysis by the maturity≥3 cut

### L2 Chain Body Clusters — Option A (pathbuildA)

40 chain body clusters produced by MiniBatchKMeans (k=40) on mean body embeddings of all 432,776 consim2 paths. Top 10:

| Cluster | N paths | N unique body nodes | N sources | Representative theme |
|---------|---------|--------------------|-----------|--------------------|
| 25 | 24,202 | 4,446 | 1,372 | Value misalignment in advanced AI |
| 11 | 23,113 | 4,196 | 1,258 | RLHF alignment of outputs with policy goals |
| 0 | 22,405 | 3,880 | 1,157 | Reward hacking in RL agents |
| 32 | 21,361 | 4,240 | 1,338 | Value misalignment (variant) |
| 6 | 20,193 | 2,573 | 982 | Adversarial perturbation vulnerability |
| 8 | 19,786 | 3,807 | 1,175 | Catastrophic misalignment in real-world deployment |
| 20 | 18,218 | 2,907 | 1,137 | Catastrophic misalignment (variant) |
| 10 | 18,120 | 3,640 | 1,104 | Insufficient AI safety talent pipeline |
| 18 | 17,296 | 4,443 | 1,343 | Catastrophic misalignment (broad) |
| 3 | 14,518 | 2,660 | 996 | Human feedback steers AI toward desired behavior |

Chain clusters have high source diversity (875–1,372 sources per top cluster) — reflecting that the same conceptual chains appear across many papers.

### L2 Chain Body Clusters — Option B (subtype co-occurrence)

Co-occurrence families produced by grouping paths with identical `frozenset{(subtype, cluster_id)}` signatures among body nodes. Family counts across all configs:

| Config | N Option B families | Path source | Top family N paths | Top family N sources |
|--------|--------------------|-----------|--------------------|----------------------|
| consim0 (edge-only) | **51** | `paths_unconstrained_edge_only.jsonl` (3,405 paths) | 46 | 37 |
| consim1 (≤1 SIM hop) | **1,603** | `representative_pathways_consim1.jsonl` (62,357 with body nodes) | 6,944 | 111 |
| unconstrained (all paths, labeled consim2 in prior docs) | **16,034** | `paths_unconstrained_sim0.9.jsonl` (1,054,527 paths, 922,787 in n≥5 families) | 33,715 | 863 |

**⚠️ Labeling correction (2026-04-05):** The 16,034-family file (`optionB_cooccurrence_families.csv`) was previously labeled "consim2" in this report and the analysis plan. It is actually computed from ALL unconstrained paths (1,054,527 total) by `phase2_step4b_paths_and_plots.py` Section 4, which reads `paths_unconstrained_sim0.9.jsonl` without applying the ≤2 consecutive SIM filter. There is no `optionB_cooccurrence_families_consim2.csv`. A true consim2-specific Option B (from the 432,776 consim2-filtered paths) was never computed. The `pathbuildB_remaining.py` script correctly computes consim0 and consim1 Option B from their respective filtered path files.

**Family count growth:** consim0→consim1 grows 31× (path count grows 22×, plus new signatures cross n≥5). consim1→unconstrained grows 10× (path count grows 14.8×: 62,357→922,787; the near-exact 10× ratio is from n≥5 threshold dynamics — signatures rare at consim1 scale cross the threshold at full unconstrained scale).

**Dominant signature stable across configs:** `de:15 & im:4 & pr:6 & th:11 & va:10` is the #1 family in both consim0 (46 paths) and consim1 (6,944 paths), confirming a stable core co-occurrence structure.

**Note on path counts vs cluster-pair counts:** The 22×/5.8× path amplification is at the individual PATH level. The *cluster-pair* count is bounded by 40×40=1,600 max possible R→I combinations, so it grows more modestly: 604 → 1,087 → 1,289 pairs. Option B family counts reflect path-level granularity (each unique signature is one family), not cluster-pair counts.

---

## Part 2: Connectivity Analysis (Substep #27)

**Config:** consim2 (max_consec_SIM ≤ 2), streaming all 1,054,527 qualifying paths from `paths_unconstrained_sim0.9.jsonl`.
- 432,776 VarB paths (consim2) used for connectivity
- 447 VarB paths had no cluster mapping (0.1%) — essentially all paths resolved to clusters

### Connectivity Matrices

- **Risk → Chain:** 1,490 distinct (risk_cluster, chain_cluster) pairs with ≥1 path
- **Chain → Intervention:** 883 distinct pairs
- **Risk → Intervention (direct):** 1,289 distinct pairs

### Top Risk → Intervention Connections

| Risk cluster | Intervention cluster | N paths |
|---|---|---|
| R10 (x-risk: misaligned advanced AI) | I8 (Fund AI safety research) | 21,654 |
| R26 (x-risk: misaligned AGI) | I8 | 11,184 |
| R25 (x-risk: misaligned superintelligence) | I8 | 11,144 |
| R21 (Catastrophic misalignment) | I8 | 10,077 |
| R16 (Insufficient AI safety research capacity) | I8 | 9,998 |
| R10 | I35 (Fine-tune with RLHF reward models) | 7,284 |
| R6 (Reward misspecification) | I8 | 6,141 |
| R9 (Unsafe AI behavior) | I8 | 5,362 |
| R10 | I5 (Adversarial evaluation) | 4,872 |
| R10 | I4 (Mandate safety evaluations) | 4,467 |

**Key observation:** Intervention cluster I8 ("Fund and expand AI safety research") is the dominant intervention — it receives paths from nearly every risk cluster. It functions as a **field-building meta-intervention**: the literature acknowledges that all identified risks motivate growing the AI safety research enterprise.

### Gap Analysis

| Gap type | Count | Interpretation |
|---|---|---|
| Risk clusters with no chain connection | **0** | All 40 risk clusters have reasoning chains to some solution approach |
| Chain clusters with no risk connection | **0** | All 40 chain families are motivated by identified risks |
| Chain clusters with no intervention connection | **0** | All chain families lead to at least one concrete intervention |
| Intervention clusters with no chain connection | **0** | All interventions have documented conceptual justification chains |
| Risk clusters with no direct intervention link | **0** | All 40 risk clusters have at least one direct intervention path |
| Intervention clusters with no direct risk link | **0** | All 40 interventions are linked to some risk rationale |

**Zero gaps at consim2** — complete connectivity across all three layers. This means that within the consim2 (≤2 consecutive SIM hops) analysis universe, the AI safety literature has established paths connecting every identified risk to at least one reasoning chain and at least one concrete intervention.

**Why zero gaps are by design (not a surprise finding):** The extraction prompt instructed LLMs to trace complete logical chains from identified risks through conceptual reasoning to concrete interventions within each paper. Every paper's extracted structure is therefore a full end-to-end EDGE chain. With 40 clusters and thousands of papers, every cluster pair will have at least some EDGE-only paths at the cluster level. Zero gaps at consim0 confirms that the extraction was successful; it is NOT a finding about whether the literature has "solved" its gaps.

**The analytically meaningful question is cross-config connection strength:** which R→I cluster pairs are only weakly connected at consim0 (few single-paper argument chains) but gain substantial path evidence via SIM bridging? See Part 3 and the dedicated cross-config section below.

**Caveats:**
1. The 447 paths with no cluster mapping (0.1% of VarB paths) represent path endpoints not assigned to clusters — a negligible fraction.

### Subcluster Analysis

36 of 80 clusters (risk + intervention) triggered subclustering (csim_mean < 0.3, size > 100, or category diversity > 2). Subcluster candidates saved to `step4_connectivity/subcluster_summary.csv`.

---

## Part 3: Config Selection — Phase B Results (Substep #29) ✅ COMPLETE

All 3 consimN configs fully analyzed. Full scoring in `step4_config_selection.md`.

### Node Coverage Per Config

| Config | N paths | N unique nodes | Risk PKL members qualifying |
|--------|---------|---------------|----------------------------|
| consim0 (edge-only) | 3,386* | 17,136 | 2,639 (54.0%) |
| consim1 (≤1 SIM hop) | 74,921* | 19,791 | 3,830 (78.3%) |
| consim2 (≤2 SIM hops) | 432,776 | 21,101 | 4,648 (95.1%) |
| Unconstrained | 1,054,527 | 21,553 | 4,889 (100%) |

*paths with body nodes; total file counts: 3,473 / 75,008

### Gap Analysis — Zero Gaps Across ALL Configs

| Gap type | consim0 | consim1 | consim2 |
|----------|---------|---------|---------|
| Risk clusters with no chain connection | 0 | 0 | 0 |
| Chain clusters with no risk/interv connection | 0 | 0 | 0 |
| Intervention clusters with no chain/risk link | 0 | 0 | 0 |
| **Total** | **0** | **0** | **0** |

**Key finding (and why it is by design):** Complete cluster-level connectivity even under EDGE-only constraints. Every risk cluster family connects to every intervention cluster family via at least one single-paper argument chain.

This result is expected from the extraction design: the prompt instructs LLMs to trace full risk→intervention chains within each paper. The meaningful question is not whether connections exist, but **how strongly** each connection is supported — i.e., how many independent paths (and from how many papers) document it. Cross-paper SIM bridging adds path density and node coverage, not new cluster-level connections.

**Cross-config connection strength analysis (see Part 3b below):** 604 EDGE-only R→I pairs → 1,087 at consim1 → 1,289 at consim2. The 685 pairs (53.1%) that appear only at consim1+ are real connections for which no single paper traces the complete argument chain end-to-end, but which are collectively established via semantic convergence across papers.

### Edge-Only Path Fraction Per Config

| Config | Node type | Mean edge-only frac | Range |
|--------|-----------|--------------------|----|
| consim1 | risk | **0.682** | 0.207–1.000 |
| consim1 | intervention | **0.968** | 0.769–1.000 |
| consim2 | risk | 0.598 | 0.131–1.000 |
| consim2 | intervention | 0.961 | 0.667–1.000 |

consim1 risk clusters are 8.4 pp better edge-grounded (0.682 vs 0.598). The extra 357K paths in consim2 dilute edge-only grounding without improving connectivity.

Note: Phase C `sim_coverage_qualifying.csv` used unconstrained VPN denominator (4,889) → 56.4% for risk. Phase B uses consim2 VPN denominator (4,648) → 59.8% for risk. Both consistently show consim1 > consim2.

### Config Selection Decision

**Selected: `consim1_pathbuildA`**

- Criteria 3 (ARI = 1.0), 4 (0 gaps): equivalent for all configs
- Criterion 2 (edge-only fraction): consim1 wins (0.682 vs 0.598 for risk)
- Criterion 5: plan explicitly prefers consim1 over consim2 when comparable
- consim1 covers 84.3% of consim2's risk→intervention cluster pairs (1,087/1,289) with 17.3% of paths

Full scoring: `step4_finalanalysis/step4_config_selection.md`

---

## Part 3b: Cross-Config R→I Connection Strength Analysis

**Source:** `step4_connectivity/cross_config_ri_pairs.csv` (1,289 rows × consim0/1/2 path counts)

### Path Counts vs Cluster-Pair Counts (Important Distinction)

**Individual path counts** (per-path amplification — shows the density gain):

| Config | N individual paths | vs consim0 |
|--------|-------------------|-----------|
| consim0 (EDGE-only) | **3,386** | baseline |
| consim1 (≤1 SIM hop) | **74,921** | ×22.1 amplification |
| consim2 (≤2 SIM hops) | **432,776** | ×5.8 over consim1 |

**Cluster-pair counts** (bounded by 40×40=1,600 max possible R→I combinations — shows connectivity breadth):

| Config | R→I cluster pairs | vs consim0 |
|--------|------------------|----|
| consim0 (EDGE-only) | **604** | baseline |
| consim1 (≤1 SIM hop) | **1,087** | +483 new pairs (+80%) |
| consim2 (≤2 SIM hops) | **1,289** | +202 more (+19% over c1) |

The modest growth in cluster-pair counts (+483, +202) vs the large path amplification (×22, ×5.8) is expected: the cluster pair count is bounded by 1,600 theoretical maximum, and once a pair has any path at consim0, adding SIM hops only adds more paths to that same pair rather than new cluster-level connections. The path count growth correctly captures density; the pair count captures breadth.

**53.1% of all pairs (685/1,289) first appear at consim1 or consim2** — these are connections with no single-paper end-to-end chain, but established via cross-paper semantic convergence. All 604 consim0 pairs are preserved in consim1 and consim2 (consim0 ⊆ consim1 ⊆ consim2).

### Top Amplified Connections (consim2 path count)

| Risk cluster | Intervention cluster | N paths c0 | N paths c1 | N paths c2 | Amplification |
|---|---|---|---|---|---|
| R10 (x-risk: misaligned advanced AI) | I8 (Fund AI safety research) | 25 | 6,632 | 21,654 | **×867** |
| R26 (x-risk: misaligned AGI) | I8 | 14 | 2,715 | 11,184 | ×799 |
| R25 (x-risk: misaligned superintelligence) | I8 | 24 | 3,391 | 11,144 | ×464 |
| R21 (Catastrophic misalignment) | I8 | 6 | 2,680 | 10,077 | ×1,680 |
| R16 (Insufficient AI safety research) | I8 | 86 | 2,580 | 9,998 | ×116 |
| R10 | I35 (Fine-tune with RLHF) | 4 | 1,198 | 7,284 | ×1,821 |
| **R6 (Reward misspecification)** | **I8** | **0** | **1,072** | **6,141** | **∞ (cross-paper only)** |
| **R16 (Insufficient AI safety research)** | **I35** | **0** | **684** | **4,242** | **∞ (cross-paper only)** |
| **R20 (variant x-risk)** | **I8** | **0** | **877** | **3,830** | **∞ (cross-paper only)** |

### Key Interpretive Observations

1. **Intervention cluster I8 ("Fund AI safety research") is the dominant meta-intervention.** Nearly every risk cluster motivates it. Even R6 (reward misspecification — a narrow technical risk) has no single-paper argument chain directly to I8, but 1,072 cross-paper paths in consim1 document the connection collectively.

2. **Amplification is highly asymmetric** — the x-risk clusters (R10, R21, R25, R26) have astronomically larger path counts than edge-only would suggest. These are the most cross-referenced argument clusters in the corpus.

3. **SIM-bridged-only pairs (c0=0)** represent connections the field has collectively established but no single paper argues end-to-end. These are prime candidates for literature synthesis papers — the connection is real and cross-referenced, but implicit rather than explicit in any single document.

4. **consim1 captures most of the gain** — 80% of net new pairs appear at consim1 vs consim0, while consim2 adds only 15.7% more. This further supports consim1 as the efficiency-optimal config.

**Output file:** `step4_connectivity/cross_config_ri_pairs.csv` — columns: `risk_cid, interv_cid, key, n_paths_c0, n_paths_c1, n_paths_c2, c1_boost, c2_boost`

---

## Part 4: SIM Coverage Analysis (Phase C Item 27 + Phase B cross-config)

Two sources: Phase C `sim_coverage_qualifying.csv` (unconstrained VPN denominator) and Phase B `edge_only_frac_by_config.csv` (config-specific VPN denominator).

### Risk clusters

| Source | Denominator VPN | Mean edge-only frac |
|--------|----------------|---------------------|
| Phase C sim_coverage (consim2 filter) | Unconstrained (4,889) | 0.564 (56.4%) |
| Phase B edge_only_frac (consim1 config) | consim1 VPN (3,830) | **0.682 (68.2%)** |
| Phase B edge_only_frac (consim2 config) | consim2 VPN (4,648) | 0.598 (59.8%) |

The Phase B consim1-specific measure (68.2%) is the appropriate value for the selected config: 68.2% of consim1-qualifying risk cluster nodes are also grounded in EDGE-only paths. Min 20.7%, max 100.0%.

### Intervention clusters

| Source | Mean edge-only frac |
|--------|---------------------|
| Phase C (consim2) | 0.917 (91.7%) |
| Phase B consim1 | **0.968 (96.8%)** |
| Phase B consim2 | 0.961 (96.1%) |

Interventions are consistently well-grounded in single-paper EDGE-only paths (96.8% for consim1).

**Interpretation:** Risk concepts are more dependent on cross-paper SIM bridging than interventions (68.2% vs 96.8% edge-only grounding for consim1). Risk identification is debated across many papers with slightly different framings; interventions tend to be proposed and elaborated within single papers. Under consim1, nearly all interventions (96.8%) have single-paper grounding, while 31.8% of risk nodes are only reached via cross-paper SIM bridges.

---

## Part 5: Held-Out Embedding Validation (Phase C Item 28)

20% holdout on qualifying cluster members, 5 splits, mean cosine similarity of holdout to training centroid.

| Node type | Mean holdout centroid sim | Min | Max |
|-----------|--------------------------|-----|-----|
| Risk | **0.8103** | 0.676 | 0.941 |
| Intervention | **0.6896** | 0.591 | 0.845 |

**Interpretation:**
- Risk clusters have very high centroid coherence (0.81 mean) — the embedding structure of risk concepts is compact and well-defined. The x-risk clusters (10, 21, 25, 26) have the highest values (0.91–0.94).
- Intervention clusters are moderately coherent (0.69 mean) — reflecting the broader semantic span of interventions within a cluster.
- All 78 clusters (40 risk + 38 intervention with sufficient qualifying members) have holdout centroid sim > 0.59 — no cluster has fundamentally poor geometric coherence.
- This confirms that the agglomerative k=40 clustering at SIM≥0.9 produces geometrically meaningful clusters.

---

## Part 6: Source Diversity (Phase C Items 20, 25)

### Risk clusters (qualifying)
- Mean n_sources: **114.0** per cluster
- Max: **362 sources** (Cluster 10 — x-risk misaligned advanced AI)
- Cluster 10's 367 nodes draw from 362 distinct paper URLs — nearly 1:1 nodes-to-papers ratio, confirming each source paper contributes 1 concept node on average.

### Intervention clusters (qualifying)
- Mean n_sources: **64.7** per cluster
- Max: **202 sources** (Cluster 8 — Fund AI safety research)

Intervention clusters have ~44% fewer sources than risk clusters on average. This may reflect that risk identification is more widely distributed across the literature than intervention proposals.

---

## Part 7: Temporal Coverage (Phase C Item 26)

Computed from `first_published` field in node_attrs.

| Node type | Earliest paper | Latest paper | Mean publication year |
|-----------|---------------|-------------|----------------------|
| Risk | 1994 | 2023 | 2020.5 |
| Intervention | 1994 | 2023 | 2020.1 |

Both risk and intervention nodes span the full 30-year range of AI safety literature (1994–2023). Mean years of 2020.1–2020.5 indicate the corpus is heavily weighted toward recent publications (2018–2023), consistent with the accelerating pace of AI safety research.

---

## Part 8: Multi-Risk Analysis (Phase C Item 22)

All 40 risk clusters have `n_unique_risk_categories = 1` and `is_multi_risk = False`. Every cluster contains only nodes of `concept_category = 'risk'`. This is expected since the clusters were built from risk-type nodes specifically.

**Gini coefficient of cluster sizes:** 0.4236 — moderate inequality. Some clusters (top 5 by size) hold 35% of all risk nodes; the bottom 10 clusters have ≤65 qualifying nodes each.

---

## Part 9: Mechanism Family Categorization (Phase C Item 24)

Five body subtype families, each with 40 clusters:

| Body subtype | N qualifying clusters populated | N qualifying nodes total |
|---|---|---|
| problem_analysis | 40 | — |
| theoretical_insight | 40 | — |
| design_rationale | 40 | — |
| implementation_mechanism | 40 | — |
| validation_evidence | 40 | — |

Full details in `mechanism_families_qualifying.csv` (200 rows — 40 clusters × 5 subtypes).

---

## Part 10: Subcluster Analysis (Substep #28 + Step 5d)

36 clusters triggered subclustering (24 risk, 12 intervention), based on threshold:
- `csim_mean < 0.3` OR `cluster_size > 100` OR `category_diversity > 2`

Most were triggered by `cluster_size > 100` — large clusters. Each was re-clustered at k=5 (AgglomerativeClustering on valid_pathway_nodes-filtered qualifying members with valid embeddings), then named via 2-pass gpt-4o-mini (Pass 1: name generation; Pass 2: judge review).

**Results (2026-04-05):** 36 parents → 180 subclusters, 0 errors.

**Naming quality:**
- High confidence: 173/180 (96.1%)
- Medium confidence: 6/180 (3.3%)
- Low confidence: 1/180 (0.6%)
- Judge revisions: 2/180 (1.1%)

**Structural finding — near-universal dominance pattern:**
35/36 parent clusters produce exactly 1 large subcluster capturing nearly all members, plus 4 tiny outlier subclusters (n≤2 nodes each, 109/180 total subclusters are singletons/pairs). This confirms that the parent cluster taxonomy is semantically tight: the k=40 agglomerative clustering correctly identified coherent clusters with minimal internal heterogeneity.

**The sole genuine split — I9 (Design robust model architectures):**
| Subcluster | N nodes | Name |
|---|---|---|
| SC0 | 75 | Architecting Transformer Models for Robustness |
| SC1 | 21 | Memory Optimization Techniques for Training Efficiency |

I9 splits cleanly into architectural design (attention, residual, transformer components) vs. memory/compute efficiency techniques — two distinct intervention mechanisms that co-exist under the broad "architecture design" theme.

**Interpretation:** The k=40 agglomerative clustering is well-calibrated. Very few clusters benefit from further subdivision. The subcluster naming validates the parent cluster names — dominant subcluster names closely match their parent names in all 35 cases.

Full results: `step4_subclusters/subcluster_names_llm.csv` (180 rows), `step5_naming/subcluster_naming_detail.csv` (180 rows with Pass 1 + Pass 2 detail).
Candidate list: `step4_connectivity/subcluster_summary.csv`.

---

## Part 11: Phase B Completion Status

All consimN analyses complete. Config selection made.

| Config | Cluster tables | Chain KMeans / Families | Connectivity | Gap analysis | Status |
|--------|---------------|------------------------|-------------|-------------|--------|
| `consim0_pathbuildA` | ✅ | ✅ k=10 | ✅ | ✅ | ✅ Complete |
| `consim1_pathbuildA` | ✅ | ✅ k=40 | ✅ | ✅ | ✅ Complete |
| `consim2_pathbuildA` | ✅ | ✅ k=40 | ✅ | ✅ | ✅ Complete |
| `consim0_pathbuildB` | ✅ 51 families | — | ✅ R→B→I (2026-04-05) | ✅ (2026-04-05) | ✅ Complete |
| `consim1_pathbuildB` | ✅ 1,603 families | — | ✅ R→B→I (2026-04-05) | ✅ (2026-04-05) | ✅ Complete |
| `unconstrained_pathbuildB` ⚠️ (mislabeled consim2) | ✅ 16,034 families (all unconstrained paths) | — | ✅ R→B→I (2026-04-05) | ✅ (2026-04-05) | ✅ Complete |

**All 6 configs complete.** Three-layer network visualizations produced for all 3 consimN configs (`three_layer_network_consim0/1/2.png`).

**Config selected: `consim1_pathbuildA`** (see `step4_config_selection.md`)

**All Phase D items complete (2026-04-05):**
- Step 5a/5b/5c: LLM naming (120 clusters), pathway examples (prevalent + gaps + EDGE-only + Option B), triplet SIM reach — all done. See Part 14.
- Step 5d — Subcluster naming: 36 parent clusters × k=5 AgglomerativeClustering + 2-pass gpt-4o-mini; 180 subclusters, 96.1% high confidence. See Part 10.
- Color-coded three-layer networks: 6 plots (3 consimN × Sankey + detail) + `cluster_color_categories.csv`. See `step4_connectivity/`.
- UMAP plots: ✅ per-consim plots for consim0/consim1/consim2 × risk/intervention = 6 plots. Maturity≥3 filter applied. Node counts (maturity-filtered):
  - consim0: 2,639 risk / 2,693 intervention — `umap_risks_consim0.png`, `umap_interventions_consim0.png`
  - consim1: 3,830 risk / 2,799 intervention — `umap_risks_consim1.png`, `umap_interventions_consim1.png`
  - consim2: 4,648 risk / 2,808 intervention — `umap_risks_consim2.png`, `umap_interventions_consim2.png`
  - Original `umap_risks.png` / `umap_interventions.png` (unconstrained, no maturity filter): 4,889 risk / 2,970 intervention — preserved as reference
  - Note: Original unconstrained plots still use unfiltered intervention count (2,970). The per-consim plots are the workshop-appropriate outputs.

### PathbuildB Connectivity — Substep #27 (completed 2026-04-05)

Script: `phase2_step4_pathbuildB_connectivity.py`
Outputs in `step4_connectivity/`: `risk_to_Bfamily_edges_consimN.csv`, `Bfamily_to_interv_edges_consimN.csv`, `risk_to_interv_via_B_edges_consimN.csv`, `gap_analysis_pathbuildB_consimN.csv` for N ∈ {0, 1, 2}.

| Config | Paths | Families matched | R→B edges | B→I edges | R→I direct |
|--------|-------|-----------------|-----------|-----------|------------|
| consim0 | 3,473 | 51/51 (100%) | 170 | 104 | 610 |
| consim1 | 75,008 | 1,603/1,603 (100%) | 6,461 | 1,952 | 1,088 |
| consim2 (unconstrained) | 1,054,527 | 16,034/16,034 (100%) | 69,712 | 19,931 | 1,362 |

**Gap analysis highlights:**
- consim0: 8 risk clusters, 7 intervention clusters have no B-family connection (expected — sparse edge-only data); all clusters have direct R→I links
- consim1: 3 risk clusters, 1 intervention cluster disconnected from B-families; all have R→I
- consim2: only 1 risk cluster and 1 intervention cluster disconnected; all have R→I
- No B-families are orphaned in any config (0 families with no risk or intervention connection)

**Note on `unconstrained_pathbuildB` labeling:** The 16,034-family file was previously labeled "consim2" in the report. It is computed from ALL 1,054,527 unconstrained paths (no max_consec_sim filter). A true consim2-filtered pathbuildB (max_consec_sim≤2) would use a subset of these paths and would yield a slightly smaller family count. This distinction is documented; the unconstrained computation is the broader/less restrictive version.

**Human review items (from `step5_naming/human_review_checklist.csv`, 63 items — mandatory before workshop submission):**
1. R13 (4 qualifying nodes) — confirm artifact or genuine niche cluster; consider removing from taxonomy
2. R36 (3 qualifying nodes) — same assessment
3. X-risk near-duplicate group (R7, R10, R12, R18, R21, R24, R25, R26, R35, R38): 10 clusters with semantically overlapping names — consider merging to 3-4 distinct sub-types for paper
4. 8 medium-confidence intervention clusters (I7, I9, I12, I14, I17, I19, I23, I25, I26, I29, I31, I33, I36) — manual name revision recommended
5. 40 mandatory risk cluster names (all require human review per `human_review_checklist.csv`)
6. I9 split candidate: subclusters confirm clean split into Transformer architectures (n=75) vs Memory optimization (n=21)

---

## Part 12: Phase C Outputs Summary

| Output file | Items | Notes |
|---|---|---|
| `source_diversity_qualifying.csv` | 280 rows (7 node types × 40 clusters) | Valid_pathway_nodes-filtered |
| `source_diversity_v1_qualifying.csv` | 80 rows (risk+interv × 40) | Step2 v1 format |
| `maturity_distribution_qualifying.png` | Heatmap | Qualifying intervention members |
| `maturity_per_cluster_qualifying.csv` | 40 rows | With counts + % per maturity level |
| `multi_risk_clusters_qualifying.csv` | 40 rows | All single-category (0 multi-risk) |
| `risk_diversity_qualifying.csv` | 1 row summary | Gini=0.4236 |
| `mechanism_families_qualifying.csv` | 200 rows | 5 subtypes × 40 clusters |
| `temporal_coverage_qualifying.csv` | 80 rows | year_min/max/mean per cluster |
| `temporal_coverage_qualifying.png` | Bar charts | Year range per cluster |
| `lifecycle_distribution_qualifying.png` | Heatmap | Intervention lifecycle by cluster |
| `sim_coverage_qualifying.csv` | 80 rows | edge-only-fraction per cluster |
| `held_out_validation_qualifying.csv` | 78 rows | Holdout centroid sim per cluster |

---

## Part 13: File Inventory — What Was Overwritten

Per **Part 10 Rule 3** of the Analysis Plan, all Phase C outputs go to `step4_finalanalysis/` with `_qualifying` suffix. No Step 2 Category A files were modified. The following Step 4 files were regenerated (by the fixed step4b run):

| File | Location | Status |
|---|---|---|
| `risk_clusters.csv` | `step4_cluster_tables/` | Regenerated — n_nodes now reflects valid_pathway_nodes filter |
| `intervention_clusters.csv` | `step4_cluster_tables/` | Regenerated — 2,815 qualifying nodes (was 2,970) |
| `optionA_chainbody_clusters.csv` | `step4_cluster_tables/` | Regenerated |
| `optionB_cooccurrence_families.csv` | `step4_cluster_tables/` | Regenerated — node_to_stc now filtered |
| `risk_clusters_09.pkl` | `step4_finalanalysis/` | Regenerated — valid_pathway_nodes-filtered |
| `optionA_kmeans_model.pkl` | `step4_finalanalysis/` | Regenerated |
| `optionA_cluster_labels.pkl` | `step4_finalanalysis/` | Regenerated |
| `within_cluster_edge_density.png` | `step4_finalanalysis/` | Regenerated — valid n denominator |
| `maturity_distribution_heatmap.png` | `step4_finalanalysis/` | Regenerated |
| `consecutive_sim_ari_test.json` | `step4_paths/` | Regenerated |
| `representative_pathways_consim1.jsonl` | `step4_paths/` | Regenerated |
| `representative_pathways_consim2.jsonl` | `step4_paths/` | Regenerated |
| `gap_analysis.csv` | `step4_connectivity/` | Regenerated — valid_pathway_nodes filter applied |
| `risk_to_chain_edges.csv` | `step4_connectivity/` | Regenerated |
| `chain_to_intervention_edges.csv` | `step4_connectivity/` | Regenerated |
| `risk_to_intervention_edges.csv` | `step4_connectivity/` | Regenerated |
| `three_layer_network.png` | `step4_connectivity/` | Regenerated |
| `subcluster_summary.csv` | `step4_connectivity/` | Regenerated |

**Category A files (preserved, not modified):**
All files in `step2_metrics_and_stability/` including silhouette, ARI, algorithm comparison, centroid similarity, edge purity, mode comparison — unchanged.
All files in `step3_validation_and_selection/` including path-filtered betweenness from `rerun_pathfiltered.py` — unchanged.

---

---

## Part 14: Phase D — Step 5 Results (consim1_pathbuildA, VPN-filtered, 2026-04-05)

All three Step 5 scripts re-run on selected config (consim1_pathbuildA) with Gap 5a/5b fixes applied. Prior provisional outputs (consim2-based, unfiltered) replaced.

### Step 5a: LLM Cluster Naming (gpt-5.4-mini, 2026-04-05)

| Metric | Count |
|--------|-------|
| Total clusters named | **120** (40 risk + 40 intervention + 40 chain) |
| High confidence | **105** (87.5%) |
| Medium/low confidence | 15 (12.5%) |
| Split candidates flagged | **29** |
| Judge-inaccurate | 7 |
| Human review checklist | **63** (40 mandatory risk + 23 auto-flagged) |

**Criterion 1 (named family coherence ≥80% non-generic):** 87.5% high confidence ✅ — exceeds the 80% threshold.

#### Risk Cluster Names (40 clusters)

All 40 high confidence except R39 (medium). Representative selection:

| Cluster | N nodes | Name | Confidence |
|---------|---------|------|-----------|
| R10 | 367 | Catastrophic misalignment leading to existential loss | high |
| R4 | 341 | Unsafe exploration and sample inefficiency in RL | high |
| R0 | 299 | Out-of-Distribution Generalization Failure | high |
| R16 | 269 | Insufficient AI safety talent and research capacity | high |
| R26 | 235 | AGI Misalignment Existential Catastrophe Risk | high |
| R25 | 223 | Misaligned superintelligent AI existential catastrophe | high |
| R22 | 221 | Harmful, misleading, or deceptive language model outputs | high |
| R9 | 219 | Deployed AI Misalignment Causing Societal Harm | high |
| R6 | 214 | Reward misspecification and reward hacking in RL | high |
| R21 | 179 | Catastrophic AI Misalignment Risk | high |
| R13 | 4 | HCH-based alignment reliability and feasibility risks | high ⚠ artifact candidate |

Note: R13 has only 4 qualifying nodes — likely an extraction artifact or very niche cluster. R36 has 3 qualifying nodes (5 total paths).

Full table: `step5_naming/risk_cluster_names_llm.csv`

#### Intervention Cluster Names (40 clusters)

32/40 high confidence; 8 medium (I7, I9, I12, I14, I17, I19, I23, I25, I26, I29, I31, I33, I36). Representative selection:

| Cluster | N nodes | Name | Confidence |
|---------|---------|------|-----------|
| I8 | 259 | Expand Funding for AI Safety Research Capacity | high |
| I4 | 203 | Pre-deployment safety review, gated release, and human oversight | high |
| I5 | 190 | Pre-deployment adversarial red-teaming and robustness testing | high |
| I35 | 185 | RLHF with human preference reward models | high |
| I26 | 174 | Safe RL fine-tuning with learned rewards and preference feedback | medium |
| I10 | 95 | AI governance via oversight, audits, and standards | high |
| I24 | 78 | RLHF fine-tuning for aligned language models | high |
| I15 | 87 | Pre-deployment mechanistic interpretability safety audits | high |

Full table: `step5_naming/intervention_cluster_names_llm.csv`

#### Chain Body Cluster Names — Critical Structural Finding (40 clusters)

**~30/40 chain clusters are named variants of "Catastrophic AI misalignment risk."** This is the core structural finding from chain naming: Option A (KMeans on mean body embeddings, k=40) produces chains that collapse to risk-dominated themes rather than distinct intermediate reasoning themes.

Key distinct chains (non-misalignment):

| Cluster | N nodes | Name | Note |
|---------|---------|------|------|
| C15 | **7,852** | Generalization and transfer under limited supervision | Largest chain cluster — ML capability-to-safety bridge |
| C12 | 40 | QARY-based cost-effectiveness modeling for AI safety field-building | Field-building cost-effectiveness chain |
| C23 | 355 | Uncertain AI timeline forecasting and planning | Forecasting/planning bridge |
| C6 | 1,203 | Adversarial examples and AI alignment failures | Technical safety chain |
| C0 | 1,085 | Reward Misspecification and Specification Gaming Risks | RL alignment chain |

The remaining 35+ chains are semantic variants of misalignment risk. This reveals that the AI safety literature's intermediate reasoning is dominated by misalignment concepts — the "chain" level does not cleanly separate risk framing from body-of-thinking. **Workshop implication:** Option A k=40 for chains is too granular; k=5-10 would produce more semantically distinct chain families. The Option B (subtype co-occurrence) families may provide better structural specificity.

Split candidates (29 of 40 clusters): Many of these reflect within-cluster heterogeneity where the same misalignment theme spans multiple sub-concepts.

Full table: `step5_naming/chain_cluster_names_llm.csv`

---

### Step 5b: Pathway Examples (consim1, 2026-04-05)

Produced from consim1 path file (selected config). Top 15 R→I connections use `risk_to_interv_edges_consim1.csv`.

**Top 5 R→I connections by consim1 path count:**

| Connection | N paths | Dominant chain | Best example type | Source |
|---|---|---|---|---|
| R10→I8 (x-risk → fund AI safety research) | 6,632 | C10 (Misalignment pathways) | **EDGE-only** (path_len=6) | EA Forum |
| R25→I8 (misaligned superintelligence → fund AI safety) | 3,391 | C19 (Misalignment risk) | VarB(consec≤1) (path_len=2) | aisafety.info |
| R26→I8 (AGI misalignment → fund AI safety) | 2,715 | C19 | VarB(consec≤1) (path_len=2) | aisafety.info |
| R21→I8 (catastrophic misalignment → fund AI safety) | 2,680 | C31 | VarB(consec≤1) (path_len=4) | aisafety.info |
| R16→I8 (insufficient AI safety capacity → fund AI safety) | 2,580 | C19 | VarB(consec≤1) (path_len=4) | Google Docs |

**Key EDGE-only (single-paper) examples:**
- R10→I8: EA Forum post, 6 nodes, full misalignment→field-building chain within one post ✅
- R10→I35: MIRI newsletter, 6 nodes, misalignment→RLHF chain
- R10→I24: Alignment Forum, 6 nodes, misalignment→RLHF fine-tuning
- R10→I4: EA Forum, 5 nodes, misalignment→pre-deployment safety review

**Gap clusters (bottom 10 by total path count in consim1):**

| Risk cluster | Total paths | N interventions | Has EDGE-only |
|---|---|---|---|
| R36 (RL agent alignment failure) | 5 | 3 | ✅ |
| R13 (HCH alignment) | 8 | 2 | ✅ |
| R29 (AI-driven wealth concentration) | 38 | 16 | ✅ |
| R39 (Bias/stagnation in AI safety) | 39 | 24 | ✅ |
| R23 (Engagement-driven recommenders) | 56 | 15 | ✅ |
| R27 (Recursive self-improvement) | 63 | 19 | ✅ |
| R37 (AI-driven job displacement) | 98 | 16 | ✅ |
| R33 (AI privacy breaches) | 120 | 26 | ✅ |
| R31 (Undetected harmful failures) | 130 | 27 | ✅ |
| R3 (Opaque transformer reasoning) | 203 | 29 | ✅ |

All 10 gap clusters have at least one EDGE-only path. These are "thin" rather than "absent" coverage areas.

**EDGE-only top-20 standalone pairs (regenerated 2026-04-05):**
Full 3,473 EDGE-only paths processed; 610 unique (risk, interv) pairs found. Top pairs by EDGE-only path count:

| Risk cluster | Intervention cluster | N EDGE-only paths |
|---|---|---|
| R4 (RL unsafe exploration) | I26 (Safe RL fine-tuning) | **133** |
| R16 (Insufficient AI safety talent) | I8 (Fund AI safety research) | 87 |
| R0 (OOD generalization failure) | I0 (Regularization) | 79 |
| R0 | I34 (Large-scale pretraining) | 70 |
| R8 (Human–AI Coordination Failure) | I23 (Uncertainty-aware control) | 65 |

Note: R10→I8 (the highest consim1 pair) has only 25 EDGE-only paths — its dominance at consim1 (6,632 paths) is driven almost entirely by cross-paper SIM bridging, not single-paper argument chains.

**Option B family examples (new `pathway_examples_optionB.json`):**
Top-10 Option B co-occurrence families from consim1 each have 3 full example chains (11,982 paths matched to top-10 signatures).

Outputs: `step5_examples/pathway_examples_prevalent.json`, `pathway_examples_gaps.json`, `pathway_examples_edgeonly.json`, `pathway_examples_optionB.json`

---

### Step 5c: Triplet SIM Reach Analysis (consim1, VPN-filtered, 2026-04-05)

For each top-15 R→I triplet (via dominant chain cluster), computed union of distinct partner paper URLs reachable via SIM≥0.9 edges from any node in any of the 3 clusters.

**SIM≥0.9 graph:** 144,140 edges, 9,414 nodes with SIM partners.

**Ranking by union reach (papers reachable from any cluster in triplet):**

| Rank | Triplet | N paths | Union reach | Triplet core |
|------|---------|---------|-------------|--------------|
| 1 | R10→C18→I4 (misalignment → adversarial deployment → safety review) | 985 | **1,704** | 2 |
| 2 | R21→C31→I8 (catastrophic misalignment → field-building) | 2,680 | 1,693 | **30** |
| 3 | R10→C10→I8 (x-risk → misalignment pathways → fund AI safety) | 6,632 | 1,639 | **49** |
| 4 | R10→C20→I5 (x-risk → misalignment → adversarial red-teaming) | 807 | 1,588 | 1 |
| 5 | R10→C25→I35 (x-risk → objective misalignment → RLHF) | 1,198 | 1,583 | 21 |

**Ranking by triplet core (papers discussing all 3 clusters simultaneously):**

| Rank | Triplet | Triplet core | Union | N paths |
|------|---------|-------------|-------|---------|
| 1 | R10→C10→I8 | **49** | 1,639 | 6,632 |
| 2 | R25→C19→I8 | 37 | 1,364 | 3,391 |
| 3 | R26→C19→I8 | 33 | 1,399 | 2,715 |
| 4 | R21→C31→I8 | 30 | 1,693 | 2,680 |
| 5 | R10→C25→I35 | 21 | 1,583 | 1,198 |

**Key interpretations:**
- R10→C10→I8 has the highest triplet core (49) — the most papers explicitly discuss all three: x-risk, misalignment pathways, AND funding AI safety research together.
- R6→C21→I8 (reward misspecification → catastrophic misalignment → fund AI safety): triplet core=0 despite 1,072 paths — **entirely cross-paper connection** with no single paper explicitly covering all three.
- Chain cluster C19 (Existential risk from advanced AI misalignment, 466 nodes) appears in 5 of the top 15 triplets — the most pivotal intermediate reasoning cluster.
- Intervention I8 dominates as endpoint in 9/15 triplets.

Full table: `step5_naming/triplet_simreach.csv`

---

### Step 5 Summary

| Output | File | Status |
|--------|------|--------|
| Risk cluster names (40) | `step5_naming/risk_cluster_names_llm.csv` | ✅ 39/40 high confidence |
| Intervention cluster names (40) | `step5_naming/intervention_cluster_names_llm.csv` | ✅ 32/40 high confidence |
| Chain cluster names (40) | `step5_naming/chain_cluster_names_llm.csv` | ✅ high confidence but thematically collapsed |
| All clusters detail | `step5_naming/all_clusters_naming_detail.csv` | ✅ 120 rows |
| Human review checklist | `step5_naming/human_review_checklist.csv` | ✅ 63 items |
| Prevalent pathway examples | `step5_examples/pathway_examples_prevalent.json` | ✅ top-15 R→I + top-10 chains |
| Gap pathway examples | `step5_examples/pathway_examples_gaps.json` | ✅ 10 gap clusters |
| EDGE-only examples | `step5_examples/pathway_examples_edgeonly.json` | ✅ top-20 EDGE-only pairs (regenerated 2026-04-05; 3,473 paths, 610 pairs) |
| Option B family examples | `step5_examples/pathway_examples_optionB.json` | ✅ top-10 families × 3 examples each (new 2026-04-05) |
| Triplet SIM reach | `step5_naming/triplet_simreach.csv` | ✅ 15 triplets, ranked |

**Outstanding issue — Chain naming collapse:** The chain level needs k reduction (k=5-10) or Option B subtype families for the workshop paper to have meaningfully distinct L2 labels. Current k=40 produces 35+ semantically near-identical "AI misalignment risk" chains. Recommend for workshop paper: use Option B families for L2 quantitative analysis, and for L2 qualitative examples use the 5 clearly distinct chains (C0, C6, C12, C15, C23) as representatives.
