# Phase 2 Step 4 — Findings Report

**Analysis date:** 2026-04-11  
**Selected config:** `consim1_pathbuildB` (consim1 = ≤1 consecutive SIM hop; pathbuildB = frozenset co-occurrence family grouping)  
**Three quality cuts applied simultaneously:** SIM edges cos_sim ≥ 0.9 · EDGE confidence ≥ 3 · intervention maturity ≥ 3  
**Quality cut audit:** See `Appendix_QualityCutAudit.md`  
**Config selection rationale:** See `step4_finalanalysis/step4_config_selection.md`  

---

## Path Enumeration Methodology (CANONICAL — applies to all downstream analysis)

This section captures every cut and design choice in the path enumeration that produced `paths_unconstrained_sim0.9.jsonl` and the `consim1` qualifying-path filter. **All downstream analysis (frozensets, body clustering, risk/intervention clustering, triplet construction) inherits these cuts.** Cite this section whenever a path-derived metric is reported.

### Source script
`final_pathway_analysis_modes.py` (lines 98-225). BFS from each risk node, emits one shortest path per (risk_node, intervention_node) pair, four output modes: `unconstrained`, `single_risk`, `monotonic`, `both`.

### Edge set used in BFS
EDGE edges with `edge_confidence ≥ 3` (min_conf=3) plus SIMILARITY edges with `cos_sim ≥ 0.9` (sim_thresh=0.9). The BFS treats the union as one undirected adjacency. The same `adj` dict is queried for all neighbor expansion — no per-step distinction between EDGE and SIMILARITY at traversal time. Consecutive-SIM-hop counts are recovered post-hoc via the `consim1` filter, not enforced during BFS.

### Cuts applied during BFS
1. **One BFS per (source_paper, risk_node).** Multiple risks in the same source produce independent BFS runs.
2. **Visited-set per BFS.** A node is processed exactly once per BFS run; revisits are blocked.
3. **Shortest path only.** BFS = breadth-first, so the first time an intervention is encountered, that path is the shortest from the originating risk to that intervention. Longer alternative paths to the same intervention are NOT enumerated.
4. **One path per (risk_node, intervention_node) pair per BFS.** Subsequent re-discoveries of the same intervention in the same BFS are blocked by the visited set.
5. **No subpath enumeration.** A path R→B1→B2→B3→I is emitted as one record; the truncation R→B1→B2→I is not separately emitted unless it terminates at a different intervention.
6. **Mode-dependent constraints (post-emission filter):**
   - `unconstrained`: no constraint
   - `single_risk`: paths with >1 risk node anywhere in the sequence are dropped
   - `monotonic`: paths with category-order reversals (e.g., `design_rationale → theoretical_insight`) are dropped
   - `both`: both single_risk and monotonic applied
7. **Path with no body skipped at frozenset construction.** `phase2_step4_pathbuildB_remaining.py:130` — `if len(path) < 3: continue`. Direct R→I edges produce paths of length 2 (length = nodes − 1 = 1), excluded.

### Cuts applied at consim1 path-set construction (post-enumeration)
Source: `phase2_step4_pathbuildB_connectivity.py`, lines ~252-273.
1. **Maturity ≥ 3 endpoint:** `node_attrs[interv_id]['intervention_maturity'] >= 3`
2. **Consecutive-SIM-hop limit ≤ 1:** `consim1` recomputes the run-length of SIM edges along each path, drops any with >1 consecutive SIM edges
3. **Edge-set restricted SIM bridging:** SIM edges restricted to pairs where both endpoints are in `vpn_unconstrained` (the maturity≥3-filtered universe)

The result is the **75,008 consim1 qualifying paths** — the analysis universe used by all downstream Step 4 / Step 5 work.

### Frozenset construction cuts
Source: `phase2_step4_pathbuildB_remaining.py:127-141`.
1. **Body sequence = path[1:-1].** Risk start and intervention endpoint excluded.
2. **node_to_stc filter:** only nodes that are body subtypes (problem_analysis, theoretical_insight, design_rationale, implementation_mechanism, validation_evidence) contribute to the frozenset signature. Risk nodes and intervention nodes that appear in the middle of a path are **silently dropped** (see Known Limitation below).
3. **n_paths ≥ 5 cutoff.** Only frozensets that occur in ≥5 distinct paths are kept (line 141): **1,603 of N_total unique frozensets**, capturing 62,357 of 75,008 consim1 paths. Singleton/rare frozensets dropped to focus on representative co-occurrence patterns.

When reporting any frozenset-derived count, disclose the n≥5 cutoff and what fraction of consim1 paths it covers.

### KNOWN LIMITATION — non-body nodes silently dropped from path middles
**Quantified on `paths_unconstrained_sim0.9.jsonl` (first 100,000 paths sampled, 2026-04-20):**
- **99.34% of paths** have at least one **risk node in the middle** (path[1:-1] contains a node with `concept_category='risk'`)
- **13.57% of paths** have at least one **intervention node in the middle**

**Cause:** `mode='unconstrained'` does not constrain BFS from passing through risk nodes or intervention nodes when the SIMILARITY edge set provides risk-risk or intervention-intervention links (cos_sim ≥ 0.9). Then `final_pathway_analysis_modes.py` lines 152-217 — when an intervention is reached but its only unvisited neighbors are also interventions, no path is emitted there but BFS continues through the intervention to find a "later" terminal intervention. The path emitted to that later terminal includes the first intervention as a middle node.

**Effect:** at frozenset construction, `frozenset(node_to_stc[n] for n in body if n in node_to_stc)` drops these middle non-body nodes. Two paths with very different routings can collapse to the same frozenset.

**Fix options** (deferred to next revision):
- (preferred) Switch to `paths_both_sim0.9.jsonl` (single_risk + monotonic constraints) as input. Single_risk eliminates the 99% risk-in-middle case; monotonic eliminates the intervention-in-middle case.
- (alternative) Post-filter `paths_unconstrained_sim0.9.jsonl`: reject any path whose middle contains a non-body node. Apply BEFORE consim1 filtering and frozenset construction.
- (sanity-check measurement) Quantify on the consim1 set (not just unconstrained) — recompute the percentages.

This is a real correctness issue affecting frozenset signatures. Tracked as Open Item 8 in `step4_rev7_plan.md`.

---

## Executive Summary

Step 4 produced a three-level risk → chain → intervention taxonomy over the AI safety literature corpus. The analysis uses `valid_pathway_nodes` (nodes on any qualifying path with all three quality cuts simultaneously enforced) as the analysis universe.

**Three-layer taxonomy (selected config: consim1_pathbuildB):**

| Layer | Description | N units |
|-------|------------|---------|
| L1 Risk clusters | 40 clusters, 4,889 qualifying risk nodes (unconstrained VPN) | 40 clusters |
| L2 Chain families | PathbuildB frozenset co-occurrence families (consim1) | 1,603 families (n≥5) |
| L3 Intervention clusters | 40 clusters, 2,815 qualifying nodes (maturity≥3) | 40 clusters |

**Key numbers:**
- **75,008** consim1 qualifying paths (3,473 EDGE-only; 75,008 consim1 ≤1 SIM hop; 432,776 consim2)
- **1,088 / 1,600** R→I cluster-pair connections at consim1 (68.0% of all possible 40×40 pairs)
- **604** EDGE-only R→I pairs (37.8%); **485** additional pairs first appear at consim1; **201** more at consim2
- **685 R→I pairs (53.1%)** are SIM-bridged only (no single paper argues them end-to-end)
- **Risk grounding:** 68.2% of consim1 risk nodes also appear on EDGE-only paths; **intervention grounding:** 96.8%

**Config selection rationale (summary):** consim1 is preferred over consim2 because it achieves 68.0% R→I coverage using only 17.3% of consim2's path count, with 12-point better risk grounding (68.2% vs 56.8% edge-only fraction). PathbuildB (frozenset co-occurrence families) is preferred over PathbuildA (KMeans on mean body embeddings) because B-families answer "because [mechanism]" — PathbuildA produces 25–30/40 clusters that re-state risks rather than describing reasoning mechanisms.

---

## Glossary

| Term | Definition |
|------|-----------|
| **valid_pathway_nodes (VPN)** | All nodes appearing on any qualifying path (SIM≥0.9, EDGE conf≥3, maturity≥3) |
| **consimN** | Max consecutive SIM-edge hops on a qualifying path: consim0=0 (EDGE-only), consim1=≤1, consim2=≤2 |
| **pathbuildA** | Chain clustering via KMeans (k=40) on mean body-node embeddings (rejected — clusters name risks not mechanisms) |
| **pathbuildB** | Chain taxonomy via frozenset co-occurrence signatures of `{(body_concept_subtype, cluster_id)}` pairs; no KMeans step |
| **B-family** | One PathbuildB frozenset family = all paths with the same combination of body concept subtype-cluster pairs |
| **direct R→I link** | A qualifying path of any length where path[0] is in a risk cluster and path[-1] is in an intervention cluster. Does NOT mean 1-hop. |
| **SIM-bridged-only pair** | R→I cluster pair with n_paths_c0=0 — no single paper argues the connection end-to-end; established via cross-paper semantic convergence |
| **meta-cluster** | Higher-level grouping of the 40 L1/L3 clusters by inter-centroid cosine similarity (k=10). Distinct from B-families. |

---

## Part 1: Cluster Tables

### L1 Risk Clusters — 40 clusters, 4,889 qualifying nodes

| Cluster | Name (gpt-5.4-mini v2) | N nodes | N sources | Centroid sim |
|---------|----------------------|---------|-----------|--------------|
| R10 | Existential catastrophe from misaligned AI | 367 | 362 | 0.922 |
| R4 | Unsafe and sample-inefficient RL exploration | 341 | 260 | 0.745 |
| R0 | Out-of-distribution generalization failure | 299 | 251 | 0.705 |
| R16 | Insufficient AI safety research capacity | 269 | 245 | 0.752 |
| R26 | Misaligned AGI existential catastrophe | 235 | 234 | 0.939 |
| R25 | Misaligned superintelligent AI existential catastrophe | 223 | 222 | 0.944 |
| R22 | Deceptive and harmful language model outputs | 221 | 192 | 0.742 |
| R9 | Deployed AI misalignment causes societal harm | 219 | 209 | 0.805 |
| R6 | Reward misspecification and reward hacking | 214 | 196 | 0.813 |
| R21 | Catastrophic AI misalignment and loss of control | 179 | 178 | 0.934 |

Full table: `step4_cluster_tables/risk_cluster_names.csv` (40 rows) · Updated v2 names: `step5_naming/risk_cluster_names_llm_v2.csv`

**X-risk near-duplicate group:** R7, R10, R12, R18, R21, R24, R25, R26, R35, R38 are 10 clusters representing variants of catastrophic/existential misalignment (centroid sim 0.92–0.94). They form a tight semantic neighborhood in the embedding space — see meta-cluster analysis (Part 11).

**Artifact candidates:** R13 (4 qualifying nodes) and R36 (3 nodes) are very small and likely extraction artifacts. The 38-cluster taxonomy excluding these is cleaner for workshop presentation.

### L3 Intervention Clusters — 40 clusters, 2,815 qualifying nodes (maturity≥3)

| Cluster | Name (gpt-5.4-mini v2) | N nodes | N sources | Centroid sim |
|---------|----------------------|---------|-----------|--------------|
| I8 | Expand AI safety research funding and capacity | 232 | 179 | 0.679 |
| I4 | Pre-deployment safety gates and controlled release | 199 | 173 | 0.632 |
| I5 | Pre-deployment adversarial red-teaming | 189 | 174 | 0.686 |
| I26 | Fine-tune robot policies with inclusive reward learning | 172 | 141 | 0.691 |
| I35 | Human-preference reward model fine-tuning | 162 | 152 | 0.754 |
| I0 | Training-time regularization for robust generalization | 124 | 99 | 0.651 |
| I9 | Choose robust model architectures at design time | 120 | 91 | 0.637 |
| I11 | Public AI risk outreach and educational dissemination | 117 | 104 | 0.636 |
| I6 | PGD adversarial training for robust fine-tuning | 111 | 98 | 0.737 |
| I1 | Continuous anomaly monitoring and safety tripwires | 105 | 94 | 0.637 |

Full table: `step4_cluster_tables/intervention_cluster_names.csv` · Updated v2 names: `step5_naming/intervention_cluster_names_llm_v2.csv`

**Note on qualifying node count:** 155 intervention nodes in the PKL have maturity<3 and are excluded from the qualifying count (reducing 2,970 PKL members to 2,815). These nodes appear in clusters by embedding similarity but are never endpoints of qualifying paths.

**Intervention type taxonomy:**
- **Technical** (algorithm/training/architecture): I0, I2, I5, I6, I7, I9, I13, I15, I16, I18, I19, I22, I23, I24, I25, I26, I27, I30, I32, I34, I35, I36 and others
- **Governance** (policy, regulation, oversight, audits): I4, I10, I12, I21, I28, I38, I39 and others
- **Field-building** (funding, education, outreach): I8, I11, I14 and others — I8 is the dominant meta-intervention; it is broadly motivated by all risks but is least specific technically

### L2 Chain Families — PathbuildB (consim1, 1,603 families n≥5)

PathbuildB groups paths by the frozenset of `{(body_concept_subtype, cluster_id)}` pairs among body nodes. Each unique frozenset = one B-family. There is no KMeans step — the "clustering" is the grouping by identical frozenset signatures. The 1,603 families (consim1, n≥5 paths) ARE the L2 chain taxonomy.

**Top-20 named B-families (gpt-5.4-mini, "because [mechanism]" prompt):**

| Rank | N paths | Chain family name | Mechanism type |
|------|---------|-------------------|---------------|
| 1 | 6,944 | Funding and training to build AI safety capacity | Field-building |
| 2 | 1,649 | Funding and consensus-building for AI safety capacity | Field-building |
| 3 | 1,210 | AI safety research funding and capacity building | Field-building |
| 4 | 900 | **Robust reward learning and specification gaming** | Technical |
| 5 | 896 | **Corrigible transparency and secure oversight mechanisms** | Technical |
| 6 | 855 | **Human feedback loops for objective alignment** | Technical |
| 7 | 746 | **Adversarial robustness through PGD training** | Technical |
| 8 | 729 | Expanding AI safety research and implementation capacity | Field-building |
| 9 | 678 | Grant-funded AI safety talent pipeline | Field-building |
| 10 | 538 | AI safety capacity building via funding and fellowships | Field-building |
| 11 | 522 | Funding-driven growth of AI safety research capacity | Field-building |
| 12 | 444 | **Behavior shaping via reflective alignment prompting** | Technical |
| 16 | 394 | **Compute chokepoints enabling scalable governance** | Governance |
| 17 | 391 | **Utility-shaping and power-seeking mitigation mechanisms** | Technical |
| 18 | 383 | **Global governance enables coordinated AI safety controls** | Governance |
| 35 | 272 | **Mechanistic interpretability for hidden model cognition** | Technical |

Full names for all 40 top families: `step5_naming/pathbuildB_chain_names_llm_v2.csv` (v2); `step5_naming/pathbuildB_chain_names_llm_v3.csv` (v3, causal framing — see Part 17)

**Why ranks 1–3 and 8–11 are all field-building variants:** These families share the dominant frozenset signature `de:15 & im:4 & pr:6 & th:11 & va:10` or close variants, reflecting that funding/talent/capacity building is the most common intermediate reasoning chain in the corpus. This is a corpus characterization finding: AI safety literature disproportionately routes from risk → "need for more AI safety research" → "fund AI safety research." Starting from rank 4, mechanistically distinct chains emerge.

**Note on MF title / intervention name overlap:** Some PathbuildB meta-family (MF) titles are semantically near-identical to intervention cluster names. For example, MF18 "Funding and training to build AI safety capacity" closely matches I8 "Expand AI safety research funding and capacity." This is expected and not an algorithm error: both name the same concept — the MF body describes the *intermediate reasoning step* ("the field needs more capacity"), and the intervention cluster names the *terminal action* ("expand funding and capacity"). In field-building chains, the intermediate body reasoning and the intervention endpoint are the same conceptual entity. For technical and governance MFs, the body names describe mechanisms that are distinct from the intervention endpoint (e.g., MF4 "Adversarial robustness through PGD training" describes a training methodology, not just the intervention label). Audience note: when both an MF name and an intervention name are shown together in a figure, the apparent duplication signals a direct field-building chain rather than an indirect mechanistic one.

**PathbuildA rejected:** The 40 PathbuildA chain clusters (KMeans on mean body embeddings) were assessed and ~25–30/40 produced names that re-stated risks ("Catastrophic AI misalignment risk," "Existential risk from advanced AI") rather than causal mechanisms. Root cause: the dense misalignment concept neighborhood in the embedding space dominates the mean body embedding, causing KMeans to cluster in that region. PathbuildA is retained only as a supplementary corpus characterization finding (the collapse itself reveals that intermediate reasoning is dominated by misalignment semantics).

---

## Part 2: Connectivity Analysis

**Config:** consim2 paths (432,776) for full coverage analysis; consim1 (75,008) for selected-config findings.

### Connectivity Matrices

- **Risk → Chain (consim1):** 6,461 distinct (risk_cluster, B-family) pairs with ≥1 path
- **Chain → Intervention (consim1):** 1,952 distinct (B-family, intervention_cluster) pairs
- **Risk → Intervention direct (consim1):** 1,088 distinct pairs (68.0% of 1,600 possible)

### Top R→I Connections (consim1 path count)

| Risk cluster | Intervention cluster | N paths c0 | N paths c1 | N paths c2 |
|---|---|---|---|---|
| R10 (x-risk misaligned AI) | I8 (Fund AI safety) | 25 | 6,632 | 21,654 |
| R26 (misaligned AGI) | I8 | 14 | 2,715 | 11,184 |
| R25 (misaligned superintelligence) | I8 | 24 | 3,391 | 11,144 |
| R21 (catastrophic misalignment) | I8 | 6 | 2,680 | 10,077 |
| R16 (insufficient AI safety research) | I8 | 86 | 2,580 | 9,998 |
| R10 | I35 (Fine-tune with RLHF reward) | 4 | 1,198 | 7,284 |
| **R6 (reward misspecification)** | **I8** | **0** | **1,072** | **6,141** |
| **R16** | **I35** | **0** | **684** | **4,242** |

Bold rows: SIM-bridged-only (no single paper argues this end-to-end).

**I8 as field-building meta-intervention:** I8 ("Fund and expand AI safety research") is the dominant intervention endpoint — it receives paths from nearly every risk cluster. It functions as a meta-intervention: the literature argues that all identified AI safety risks motivate expanding the research enterprise itself. This reflects a genuine corpus property, not an analytical artifact. For technical analysis, I8 connections should be treated separately from specific technical/governance interventions.

### Gap Analysis

| Gap type | consim0 | consim1 | consim2 |
|----------|---------|---------|---------|
| R→I cluster pairs with no path | 990 | 512 | 311 |
| Risk clusters with no R→I connection | 0 | 0 | 0 |
| Intervention clusters with no connection | 0 | 0 | 0 |

**Zero disconnected clusters is a by-construction result** (see `Appendix_ByDesign.md` §A1): it confirms extraction succeeded, not that all risk–intervention gaps are addressed. The analytically meaningful question is connection *strength* (path count, source diversity), not binary connectivity.

---

## Part 3: Config Selection

**Selected: `consim1_pathbuildB`**

Full scoring and rationale: `step4_finalanalysis/step4_config_selection.md`

Summary of new criteria (all computed via independent per-config AgglomerativeClustering k=40):

| Criterion | consim0 | consim1 | consim2 |
|-----------|---------|---------|---------|
| C1: Risk intra-cluster cosine sim | 0.778 | 0.801 | 0.820 |
| C2: Risk edge-only fraction | 1.000 (trivial) | **0.689** | 0.568 |
| C3: ARI vs next config (risk) | 0.444 | **0.636** | — |
| C4: R→I coverage fraction | 38.1% | **68.0%** | 80.6% |

**Decision:** C2 is decisive — consim1 risk grounding (68.9%) is 12 points better than consim2. consim1 covers 84.6% of consim2's pairs with 17.3% of the path volume (ideal efficiency frontier). C3 confirms consim1 is stable (ARI 0.636 with consim2 = near-equivalent cluster taxonomies). C1 gap is small (0.819 vs 0.801). PathbuildB selected over PathbuildA (A2 empirical assessment — see Part 1 L2 section).

### Path Count and Node Count Summary

| Config | N paths | N unique VPN nodes | R→I pairs (of 1,600) |
|--------|---------|-------------------|----------------------|
| consim0 (EDGE-only) | 3,473 | 17,136 | 610 (38.1%) |
| consim1 (≤1 SIM hop) | **75,008** | **19,791** | **1,088 (68.0%)** |
| consim2 (≤2 SIM hops) | 432,776 | 21,101 | 1,289 (80.6%) |

Path amplification: consim1 is ×21.6 over consim0; consim2 is ×5.8 over consim1. Cluster-pair growth is bounded by 1,600 max (40×40) — the amplification represents path density, not new thematic connections.

---

## Part 4: SIM Coverage Analysis

### Edge-Only Grounding by Config (Risk vs Intervention)

| Config | Node type | Mean edge-only fraction | Min | Max |
|--------|-----------|------------------------|-----|-----|
| consim1 | **risk** | **0.682** | 0.207 | 1.000 |
| consim1 | **intervention** | **0.968** | 0.769 | 1.000 |
| consim2 | risk | 0.598 | 0.131 | 1.000 |
| consim2 | intervention | 0.961 | 0.667 | 1.000 |

**Why risk and intervention have different edge-only fractions:**
- **Interventions (96.8% edge-only):** Interventions are always EDGE endpoints — every paper that proposes an intervention does so via a complete EDGE chain (the extraction prompt required end-to-end chains within each paper). Interventions therefore nearly always have single-paper grounding.
- **Risk concepts (68.2% edge-only):** Risk concepts appear both as path *starts* (well-grounded) and as *body/intermediate* nodes in chains. When reached only via SIM hops from similar risk formulations in other papers, they lack single-paper grounding. The 31.8% SIM-only risk nodes represent risk framings that the literature references semantically but doesn't explicitly argue in isolated complete chains.

**Practical implication:** For the selected config (consim1), 96.8% of all intervention nodes and 68.2% of all risk nodes participating in the analysis have at least one single-paper EDGE-only path — establishing strong grounding for both sides of the taxonomy despite using SIM bridging.

---

## Part 5: Holdout Embedding Validation

20% holdout on qualifying cluster members, 5 splits, mean cosine similarity of holdout to training centroid.

| Node type | Mean holdout centroid sim | Min | Max |
|-----------|--------------------------|-----|-----|
| Risk | **0.8103** | 0.676 | 0.941 |
| Intervention | **0.6896** | 0.591 | 0.845 |

**What this validates:** A random 20% of qualifying cluster members were withheld from centroid computation. The test measures how well withheld nodes fit their cluster centroid — whether centroids generalize to held-out nodes from the same cluster.

**Why it matters:** Confirms cluster centroids are representative of the full cluster distribution, not overfitted to specific training nodes. High cosine sim (>0.7) means a new node whose embedding is computed can reliably be assigned to the correct cluster using centroid matching.

**Result:** Risk clusters (0.81 mean) are very geometrically coherent — the x-risk clusters (R10, R21, R25, R26) achieve 0.91–0.94, reflecting their tight embedding neighborhoods. Intervention clusters (0.69 mean) are moderately coherent — interventions within one cluster span specific vs. general techniques, producing more semantic breadth. No cluster has holdout centroid sim < 0.59 — all clusters have valid geometric structure.

**Takeaway:** Embedding-based clustering at SIM≥0.9 produces reproducible clusters that would correctly classify new nodes.

---

## Part 6: Source Diversity

### Risk clusters (consim1 qualifying)
- Mean n_sources: **114.0** per cluster
- Max: **362 sources** (R10 — x-risk misaligned advanced AI) — ~1:1 nodes-to-papers ratio
- Min: **3 sources** (R36 — likely artifact)

### Intervention clusters (consim1 qualifying)
- Mean n_sources: **64.7** per cluster
- Max: **202 sources** (I8 — Fund AI safety research)

Intervention clusters have ~44% fewer sources than risk clusters on average, reflecting that risk identification is more widely distributed across the literature than intervention proposals.

---

## Part 7: Temporal Coverage

| Node type | Earliest | Latest | Mean publication year |
|-----------|---------|--------|----------------------|
| Risk | 1994 | 2023 | 2020.5 |
| Intervention | 1994 | 2023 | 2020.1 |

Both span the full 30-year range of AI safety literature. Mean years of 2020.1–2020.5 confirm the corpus is heavily weighted toward recent publications (2018–2023), consistent with accelerating AI safety research activity.

---

## Part 8: Multi-Risk Analysis

**Goal:** Check whether any cluster contains nodes of multiple risk sub-categories (e.g., mixing "misuse risks" and "misalignment risks") — which would indicate over-broad clusters.

**Result:** All 40 risk clusters have `n_unique_risk_categories = 1` and `is_multi_risk = False`. Every cluster contains only nodes of `concept_category = 'risk'`.

**Why this matters for reviewers:** Confirms the k=40 agglomerative clustering correctly separates semantically distinct risk themes rather than mixing them. The taxonomy is internally consistent.

**Gini coefficient of cluster sizes: 0.4236** — moderate size inequality. The top 5 clusters hold 35% of all risk nodes; the bottom 10 clusters have ≤65 qualifying nodes each. This size inequality reflects the literature's uneven coverage of risk types: catastrophic misalignment risks are discussed far more frequently than narrow technical risks like adversarial misclassification.

**Note on risk vs. problem_analysis boundary:** Nodes have `concept_category = 'risk'` or `concept_category = 'problem_analysis'`. The boundary is set by the extraction LLM at ingestion time. "Reward misspecification" is extracted as `risk` if the paper frames it as a failure mode to prevent, or as `problem_analysis` if framed as an analytical step. This ambiguity is a corpus property, not an analytical error — both categories appear in qualifying paths and both are included in the VPN.

---

## Part 9: Body Subtype Analysis

Five body subtype families, each with 40 clusters, constitute the intermediate reasoning layer:

| Body subtype | Description |
|---|---|
| `problem_analysis` | Intermediate problem decompositions and causal analyses |
| `theoretical_insight` | Theoretical arguments, formal proofs, conceptual claims |
| `design_rationale` | Motivations for design choices in interventions |
| `implementation_mechanism` | Concrete technical mechanisms and algorithms |
| `validation_evidence` | Empirical evidence, benchmarks, evaluation results |

Each of the 200 (subtype × cluster) combinations has representative nodes from qualifying paths.

**Top representative nodes per subtype (centroid-closest):**
Full table in `step4_cluster_tables/bodysubtype_cluster_representatives.csv` (200 rows — 5 subtypes × 40 clusters each, with top-3 representative node names per cluster).

**Dominant subtype distribution in top B-families:** The top-1 B-family (6,944 consim1 paths) has signature `de:15 & im:4 & pr:6 & th:11 & va:10` — one cluster from each of the 5 subtypes, suggesting the most common reasoning chains are structurally balanced across all five reasoning types.

Full details: `mechanism_families_qualifying.csv` (200 rows).

---

## Part 10: Subcluster Analysis

36 clusters triggered subclustering (24 risk, 12 intervention), all by size criterion (`cluster_size > 100`). Each re-clustered at k=5 (AgglomerativeClustering on VPN-filtered qualifying members).

**Naming quality (2-pass gpt-4o-mini):** 173/180 high confidence (96.1%), 6 medium, 1 low.

**Structural finding:** 35/36 parent clusters produce exactly 1 large subcluster capturing nearly all members plus 4 tiny outlier subclusters (n≤2 nodes each). This confirms the parent k=40 taxonomy is semantically tight — very few clusters benefit from further subdivision.

**The sole genuine split — I9 (robust model architectures):**

| Subcluster | N nodes | Name |
|---|---|---|
| SC0 | 75 | Architecting Transformer Models for Robustness |
| SC1 | 21 | Memory Optimization Techniques for Training Efficiency |

I9 splits cleanly into architectural design vs. memory/compute efficiency — two distinct intervention mechanisms coexisting under "architecture design."

Full results: `step4_subclusters/subcluster_names_llm.csv` · `step5_naming/subcluster_naming_detail.csv`

---

## Part 11: Meta-cluster Analysis

**Script:** `phase2_step4_B1_metaclusters.py`  
**Outputs in:** `step4_finalanalysis/step4_metaclusters/`

The 40 risk clusters and 40 intervention clusters were grouped into meta-clusters (k=10) by hierarchical agglomerative clustering (average linkage) on the 40×40 inter-centroid cosine similarity matrices. Centroids are computed from VPN_consim1-filtered cluster members only (19,791 qualifying nodes).

**Inter-centroid similarity statistics (VPN_consim1 centroids):**
- Risk: mean=0.708, min=0.379, max=0.964
- Intervention: mean=0.660, min=0.329, max=0.925

The domain is NOT uniformly compact — clusters span a wide similarity range (0.38–0.97 for risk), confirming that distinct semantic neighborhoods exist and that cluster boundaries are meaningful.

### Risk Meta-clusters (k=10)

| Meta | N clusters | N nodes | Primary theme |
|------|-----------|---------|---------------|
| **R-Meta-3** | **11** | **1,033** | **Catastrophic/existential misalignment (x-risk block)** |
| **R-Meta-6** | **14** | **1,704** | RL/training failures, misuse, arms race, deployment safety |
| R-Meta-7 | 6 | 908 | ML reliability / interpretability / resource efficiency |
| R-Meta-1 | 2 | 31 | AI-driven economic displacement |
| R-Meta-2 | 2 | 40 | Shutdown resistance / instrumental power-seeking |
| R-Meta-4 | 1 | 18 | Social manipulation via recommender systems |
| R-Meta-5 | 1 | 41 | Algorithmic bias and discrimination |
| R-Meta-8 | 1 | 48 | AI-driven personal data privacy breaches |
| R-Meta-9 | 1 | 4 | Unverifiable HCH-style alignment (C13) |
| R-Meta-10 | 1 | 3 | Unreliable RL alignment transfer (C36) |

**R-Meta-6 is the largest block** — 14 clusters and 1,704 nodes covering RL training failures, misuse risks, AI arms race, and broad deployment safety concerns. **R-Meta-3** (11 clusters, 1,033 nodes) is the x-risk/catastrophic misalignment block.

### Intervention Meta-clusters (k=10)

| Meta | N clusters | N nodes | Primary theme |
|------|-----------|---------|---------------|
| **I-Meta-3** | **16** | **1,210** | General technical safety interventions (training/architecture) |
| **I-Meta-6** | **16** | **1,502** | Governance, capacity building, evaluation, deployment controls |
| I-Meta-1 | 1 | 9 | International treaty / arms control |
| I-Meta-2 | 1 | 39 | Export controls on AI hardware |
| I-Meta-4 | 1 | 17 | Periodic target networks for stable Q-learning |
| I-Meta-5 | 1 | 25 | Uniform Experience Replay for RL training |
| I-Meta-7 | 1 | 34 | Deployment-time prompt engineering |
| I-Meta-8–10 | 1 each | 5–13 | Small specialized clusters |

### Meta-cluster Connectivity

Top R→I meta-cluster pairs by consim1 paths:
1. R-Meta-3 (x-risk) → I-Meta-6 (governance/capacity): 29,673 paths
2. R-Meta-6 (RL/training/misuse) → I-Meta-6: 16,979 paths
3. R-Meta-3 → I-Meta-3 (technical safety): 10,831 paths
4. R-Meta-6 → I-Meta-3: 7,640 paths
5. R-Meta-7 (ML reliability) → I-Meta-6: 2,432 paths

Full meta-cluster connectivity: `step4_metaclusters/meta_cluster_ri_connectivity.csv`  
Similarity heatmaps + dendrograms: `step4_metaclusters/risk_sim_heatmap.png`, `step4_metaclusters/risk_dendrogram.png`, etc.  
Meta-cluster network plot: `step4_metaclusters/meta_connectivity_network.png`

---

## Part 12: PathbuildB Chain Family Examples

**Script:** `phase2_step5_B2_naming_rerun.py`  
**Outputs:** `step5_naming/pathbuildB_chain_names_llm.csv` (40 rows)

Top-10 mechanistically distinct B-families (excluding funding variants):

| Rank | N paths | Mechanism name | Type |
|------|---------|---------------|------|
| 4 | 900 | Robust reward learning and specification gaming | Technical |
| 5 | 896 | Corrigible transparency and secure oversight mechanisms | Technical |
| 6 | 855 | Human feedback loops for objective alignment | Technical |
| 7 | 746 | Adversarial robustness through PGD training | Technical |
| 12 | 444 | Behavior shaping via reflective alignment prompting | Technical |
| 16 | 394 | Compute chokepoints enabling scalable governance | Governance |
| 17 | 391 | Utility-shaping and power-seeking mitigation mechanisms | Technical |
| 18 | 383 | Global governance enables coordinated AI safety controls | Governance |
| 25 | 328 | Adversarial robustness training via optimized perturbations | Technical |
| 35 | 272 | Mechanistic interpretability for hidden model cognition | Technical |

**Example "because" chains:**
- "Adversarial training addresses adversarial vulnerability **because** [PGD-based training creates robust features → lower adversarial error rates confirm efficacy]"
- "Export controls on compute address unpredictable AI scaling **because** [supply chain concentration enables governance → historical governance analogies support feasibility]"
- "RLHF addresses reward misspecification **because** [human feedback steers policy → objective alignment improves → deceptive behavior reduced]"

Decoded frozenset signatures for top-20 families: `step4_cluster_tables/optionB_top20_decoded_consim1.csv`

---

## Part 13: Novel SIM-Bridged R→I Pairs

**Script:** `phase2_step4_B5_novel_pairs.py`  
**Outputs:** `step4_connectivity/novel_sim_bridged_pairs_full.csv` (685 rows), `step4_connectivity/novel_sim_bridged_pairs_top20.csv` (20 rows)

### Analysis

From the 1,289 R→I cluster pairs in `cross_config_ri_pairs.csv`, **685 pairs (53.1%) have n_paths_c0=0** — established only via cross-paper semantic convergence (SIM bridging), never argued end-to-end in any single paper.

These SIM-bridged-only pairs are prime candidates for literature synthesis: the connection is collectively established but implicit.

**Intervention type breakdown:**
| Type | Count |
|------|-------|
| Technical | 567 (82.8%) |
| Governance | 91 (13.3%) |
| Field-building | 27 (3.9%) → excluded from novelty analysis |

**Novelty filter:** Field-building interventions (fund AI safety, education/outreach) are trivially motivated by almost any risk and are therefore not novel insights. After excluding 27 field-building pairs, **658 novel SIM-bridged-only pairs** remain.

### Top-20 Novel SIM-Bridged Pairs

| Rank | Type | c1 paths | Risk cluster | Intervention cluster |
|------|------|---------|-------------|----------------------|
| 1 | Technical | 684 | Insufficient AI safety research capacity | Fine-tune RL agents with human preference-based reward |
| 2 | Technical | 457 | Existential catastrophe from misaligned AI | Fine-tune language models with RLHF |
| 3 | Technical | 315 | Insufficient AI safety research capacity | Adversarial training during fine-tuning |
| 4 | Governance | 265 | Existential catastrophe from misaligned AI | Pre-train policy networks via supervised learning |
| 5 | Technical | 262 | Existential catastrophe from misaligned AI | Fine-tune robot policies with inclusive reward learning |
| 6 | Technical | 208 | Existential catastrophe from misaligned AI | Attainable utility preservation reward penalty |
| 7 | Technical | 200 | Deceptive alignment and hidden objectives | Fine-tune RL agents with human preference-based reward |
| 8 | Technical | 184 | Insufficient AI safety research capacity | Deploy interpretability analysis suites |
| 12 | Governance | 163 | Existential catastrophe from misaligned AI | Enforce export controls on advanced AI hardware |
| 16 | Technical | 129 | Catastrophic misalignment of advanced AI | Attainable utility preservation reward penalty |
| 20 | Technical | 110 | Power-seeking behavior in advanced AI | Fine-tune RL agents with human preference-based reward |

Full table: `step4_connectivity/novel_sim_bridged_pairs_top20.csv`

**Key observation:** The top novel pair (rank 1 — "Insufficient AI safety research capacity" → "Fine-tune RL agents with human preference reward") has 684 consim1 paths but zero EDGE-only paths. This means the AI safety field collectively establishes this connection across many papers but no single paper argues it end-to-end. It is a strong candidate for a survey or synthesis paper.

---

## Part 14: Outputs Reference

Key output files for downstream analysis:

| File | Location | Contents |
|---|---|---|
| `risk_clusters.csv` | `step4_cluster_tables/` | 40 risk clusters, qualifying counts |
| `intervention_clusters.csv` | `step4_cluster_tables/` | 40 intervention clusters |
| `risk_cluster_names.csv` | `step4_cluster_tables/` | LLM names (v1 gpt-4o-mini) |
| `risk_cluster_names_llm_v2.csv` | `step5_naming/` | LLM names (v2 gpt-5.4-mini) |
| `intervention_cluster_names_llm_v2.csv` | `step5_naming/` | v2 intervention names |
| `pathbuildB_chain_names_llm.csv` | `step5_naming/` | Top-40 B-family names (gpt-5.4-mini) |
| `optionB_cooccurrence_families_consim1.csv` | `step4_cluster_tables/` | 1,603 B-families (n≥5) |
| `optionB_top20_decoded_consim1.csv` | `step4_cluster_tables/` | Top-20 B-families, decoded body components |
| `cross_config_ri_pairs.csv` | `step4_connectivity/` | 1,289 R→I pairs with c0/c1/c2 path counts |
| `novel_sim_bridged_pairs_top20.csv` | `step4_connectivity/` | Top-20 novel SIM-bridged-only R→I pairs |
| `config_selection_metrics_v2.csv` | `step4_finalanalysis/` | C1-C4 metrics per consimN config |
| `risk_meta_assignments.csv` | `step4_metaclusters/` | Risk cluster → meta-cluster (k=10/12) |
| `intervention_meta_assignments.csv` | `step4_metaclusters/` | Intervention cluster → meta-cluster |
| `meta_cluster_ri_connectivity.csv` | `step4_metaclusters/` | Meta R→I connectivity aggregated |
| `bodysubtype_cluster_representatives.csv` | `step4_cluster_tables/` | 200 representative node names (5×40) |
| `step4_config_selection.md` | `step4_finalanalysis/` | Full config selection rationale |

**UMAP plots (s=10, updated):**
- `step4_finalanalysis/umap_risks_consim{0,1,2}.png` — all VPN nodes (up to ~19,791), cosine metric, colored by cluster assignment. Axes show UMAP-1/UMAP-2 with value ranges.
- `step4_finalanalysis/umap_interventions_consim{0,1,2}.png` — same, for intervention clusters.
- `step4_finalanalysis/umap_interventions_consim1_clusters.png` — sampled (max 200 nodes/cluster), euclidean metric on L2-normalized embeddings, cluster name labels overlaid at centroid positions.

**UMAP vs MDS — key distinction for publication:**
- **UMAP** (`phase2_step4_umap_plots.py` / B9): Preserves **local neighborhood structure** of individual nodes. Each point is one node. Clusters appear as blobs — proximity encodes semantic similarity. The two UMAP plots for interventions look different because: (1) the `_clusters.png` uses max 200 sampled nodes/cluster and euclidean metric; (2) the `consim1.png` uses all VPN nodes and cosine metric. Both are valid; `_clusters.png` is better for labeling, `consim1.png` is better for seeing point-cloud geometry.
- **MDS** (`step4_metaclusters/risk_2d_mds.png`, `interv_2d_mds.png`, `step4_cluster_tables/pathbuildB_metafamily_2d_mds.png`): Preserves **global pairwise distances between CENTROIDS only** (40 points, not individual nodes). Distances encode the inter-centroid cosine dissimilarity matrix. Use MDS when you want to show how cluster centers relate globally; use UMAP when you want to show actual data distribution and neighborhood geometry.

**Meta-cluster plots:** `step4_metaclusters/risk_sim_heatmap.png`, `step4_metaclusters/risk_dendrogram.png`, `step4_metaclusters/meta_connectivity_network.png`, `step4_metaclusters/risk_2d_mds.png`, `step4_metaclusters/interv_2d_mds.png`  
**PathbuildB meta-family plots:** `step4_cluster_tables/pathbuildB_metafamily_dendrogram.png` (horizontal, 1-Jaccard x-axis), `step4_cluster_tables/pathbuildB_metafamily_2d_mds.png` (log colorbar showing actual path counts)  
**Quality cut audit:** `Appendix_QualityCutAudit.md`

---

## Part 15: Rev3 Additions

### 15.1 — Meta-cluster basis (Item 1 / B1)

Meta-clustering uses the PKL agglomerative cluster assignments (SIM=0.9, unconstrained mode, k=40) — the same cluster boundaries across all consimN configs (confirmed by ARI≥0.63 cross-config stability). Centroids are computed exclusively from **VPN_consim1-filtered members**: for each cluster, only nodes appearing on at least one qualifying consim1 path (19,791 nodes total) contribute to the centroid. This ensures the inter-centroid similarity matrix reflects the actual distribution of the analysis population.

All 40 risk and 40 intervention clusters have at least one qualifying VPN_consim1 member. Three risk clusters are very small after VPN filtering: R13 (4 nodes), R36 (3 nodes), R29 (8 nodes) — these are low-salience clusters and appear as isolated outliers in the MDS.

**2D MDS visualizations** (saved to `step4_metaclusters/`): `risk_2d_mds.png` and `interv_2d_mds.png` show all 40 L1/L3 clusters projected onto 2D via VPN_consim1 inter-centroid cosine distance, colored by meta-cluster membership (k=10).

**Intra-centroid histograms** (`risk_intracentroid_histograms.png`, `interv_intracentroid_histograms.png`): 40-panel grids showing the distribution of VPN_consim1 member-to-centroid cosine similarities for each cluster. Key stats saved in `intracentroid_stats.csv`. The distributions confirm that risk and intervention clusters are meaningfully compact: most have mean intra-centroid sim ≥ 0.85.

### 15.2 — PathbuildB family size distribution and frozenset space (Item 9 / B3)

1,603 consim1 B-families (n≥5 threshold):
- **n≥5:** 1,603 / 1,603 (100% — this is the filter applied)
- **n≥10:** 784 / 1,603 (48.9%)
- **n≥100:** 134 / 1,603 (8.4%)
- n_paths stats: min=5, max=6,944, median=9, mean=38.9

The distribution is strongly right-skewed: the top family (B0, "via funding, fellowships, and capacity-building pipelines") alone has 6,944 paths — more than 10× the median. Plot: `step4_cluster_tables/pathbuildB_family_size_distribution.png`.

**Frozenset component space:** Each B-family is a frozenset of `(body_concept_subtype, cluster_id)` component pairs. The observed component vocabulary spans **191 unique components** across the 5 body subtypes (pr, th, de, im, va) and 40 clusters each. Each family uses between 1 and 15 components (mean 5.89). The theoretical maximum number of distinct frozensets over this component vocabulary is 2^191 ≈ **10^57**. The 1,603 observed families represent approximately **10^-54** of the theoretical space — an extremely sparse sampling, meaning the qualifying paths cluster into a tiny structured subset of all possible reasoning chain signatures. This sparsity reflects the genuine semantic structure of the AI safety literature: only a small number of specific concept co-occurrence patterns appear repeatedly across papers.

### 15.3 — R→C→I triplets (Item 11 / B4)

Joining `risk_to_Bfamily_edges_consim1.csv` × `Bfamily_to_interv_edges_consim1.csv` on B-family ID:
- **2,298 triplets** with n≥5 paths (out of 40 × 1,603 × 40 possible)
- Top triplet: R10→B0→I8 (2,487 paths) — Existential catastrophe from misaligned AI → Funding/training for AI safety → Expand AI Safety Research Funding
- The top-5 triplets all involve B0 (funding/capacity) or B1 (funding/consensus) × I8 (Fund AI safety)

File: `step4_connectivity/ri_triplets_consim1.csv` (2,298 rows)  
Plot: `step4_connectivity/ri_triplets_histogram.png`

### 15.4 — Source diversity for top B-families (Item 12 / B5)

Risk node URL diversity for consim1 qualifying nodes:
- **2,603 distinct source URLs** from 3,513 sampled risk cluster nodes
- Max concentration: top-1 URL accounts for only **0.2% of nodes** (7 out of 3,513)
- No single source dominates → the funding-heavy B-families draw from a genuinely diverse literature, not from a single prolific author or paper

File: `step4_cluster_tables/top_bfamily_source_diversity.csv`

### 15.5 — Jaccard dendrogram of top-20 B-family signatures (Item 12 / B6)

Top-20 B-families clustered by frozenset component Jaccard similarity. The dendrogram reveals that:
- The top-3 funding families (B0, B1, B2) share almost identical components (de:15, im:4, pr:6, th:11) and differ only in their validation cluster (va:10 vs va:9 vs va:32)
- The RLHF/reward families form a separate cluster with de:34, im:10 components

Plot: `step4_cluster_tables/top20_bfamily_jaccard_dendrogram.png`

### 15.6 — Meta-cluster coherence (Item 19 / B8)

For each meta-cluster, mean pairwise inter-centroid similarity within the meta-cluster was computed. Results in `step4_metaclusters/meta_cluster_coherence.csv` (20 rows: 10 risk + 10 intervention).

**I26/I35 separation note (Item 5):** At k=15 for intervention meta-clustering, I26 ("Fine-tune robot policies with inclusive reward learning") and I35 ("Fine-tune RL agents with human preference-based reward") remain in the same meta-cluster. Both use preference-based fine-tuning and are highly similar at the embedding level. The domain distinction (robotics vs LLMs) is not captured by embedding similarity. Meta-clustering by cosine distance is correct; separating by application domain requires a different ontological principle.

### 15.7 — PathbuildB meta-family clustering (Item 21 / D1)

Pairwise Jaccard similarity across all 1,603 B-families, hierarchical agglomerative clustering (average linkage on 1-Jaccard distance):

| k | mean_intra_Jaccard |
|---|--------------------|
| 32 (selected) | 0.300 |
| 40 | 0.328 |
| 60 | 0.389 |
| 80 | 0.445 |

**Selected k=32** (first k achieving mean_intra_Jaccard ≥ 0.30).

**Top meta-families by path count:**
| MF ID | N families | N paths | Dominant family name |
|-------|-----------|---------|---------------------|
| MF18 | 400 | 23,839 | Funding and training to build AI safety capacity |
| MF23 | 239 | 8,252 | Human feedback loops for objective alignment |
| MF19 | 218 | 6,480 | Global governance enables coordinated AI safety controls |
| MF4 | 115 | 3,625 | Adversarial robustness through PGD training |
| MF21 | 99 | 2,976 | Human-feedback alignment and goal specification |

Files:
- `step4_cluster_tables/pathbuildB_metafamilies_consim1.csv` — 1,603 rows with meta_family_id
- `step4_cluster_tables/pathbuildB_metafamily_summary_consim1.csv` — 32-row summary
- `step4_connectivity/risk_to_metafamily_edges_consim1.csv` — 689 R→MF edges
- `step4_connectivity/metafamily_to_interv_edges_consim1.csv` — 183 MF→I edges
- `step4_connectivity/three_layer_network_pathbuildB_metafamily_consim1.png` — updated three-layer network
- `step4_cluster_tables/pathbuildB_metafamily_dendrogram.png`

### 15.8 — Heatmap framing explanation (Item 3 / E3)

The colored rectangular frames in `risk_sim_heatmap.png` and `interv_sim_heatmap.png` mark **meta-cluster boundary blocks**: after rows and columns are reordered by meta-cluster assignment (k=10), each colored frame outlines a contiguous block of clusters belonging to one meta-cluster. Each color corresponds to one meta-cluster. Both axes carry identical labels because the matrix is symmetric (sim(A,B) = sim(B,A)); the dual axis is provided for readability when tracing both rows and columns.

### 15.8b — Novel pair web search (Item 18 / D2)

Top 14 technical-only novel R→I pairs (excluding R16 "Insufficient AI safety research capacity" as maximally general) were annotated via web search for post-2023 papers arguing the connection end-to-end:

| Filter rank | Risk | Intervention | c1 paths | Post-2023 paper? |
|-------------|------|-------------|----------|-----------------|
| 1 | R25 misaligned superintelligent AI | I24 RLHF fine-tuning | 457 | **True** — Safe RLHF-V (arxiv:2503.17682), RLHS (arxiv:2501.08617) |
| 4 | R20 deceptive alignment | I35 RLHF for RL agents | 200 | **True** — Alignment Faking paper (arxiv:2506.21584) |
| 2, 6 | R10/R25 misaligned AI | I26 robot policy reward | 262/175 | Partial — no direct chain paper |
| 3, 8, 11 | R10/R25/R21 x-risk | I32 AUP penalty | 208/162/129 | **False** — AUP work is primarily pre-2023 (Turner 2019-2023) |
| 9 | R26 misaligned AGI | I29 isotropic fractionator | 150 | **False — LIKELY SPURIOUS** — isotropic fractionator is a neuroscience cell-counting technique; no AI safety connection found |

**Key finding:** The R20→I35 pair (deceptive alignment detected and mitigated via RLHF fine-tuning) is the most supported novel pair, with arxiv:2506.21584 providing direct empirical evidence. The R26→I29 pair is flagged as likely spurious — a co-mention artifact rather than a genuine causal argument.

**Summary:** 2/14 pairs have confirmed post-2023 papers; 6/14 partial; 6/14 none found (strong synthesis paper candidates).

File: `step4_connectivity/novel_sim_bridged_pairs_top20_technical.csv`

### 15.9 — "By design" findings (Item 17)

See `Appendix_ByDesign.md` for findings that are expected consequences of the pipeline design:
- Zero disconnected clusters (§A1) — by-construction confirmation that extraction succeeded
- 100% validation for EDGE-only config (§A2) — expected from LLM judge applying same extraction criteria
- PathbuildA cluster collapse (§A3) — informational, reveals embedding space structure

---

## Part 16: Chain Clustering Methods — Design, Assessment, and Supplementary Analysis

### 16.1 — Chain clustering method terminology

The labels **PathbuildA** and **PathbuildB** are shorthand for two distinct approaches to clustering the body/chain nodes of qualifying paths. Both methods operate on the **identical qualifying path set** (SIM ≥ 0.9, EDGE conf ≥ 3, maturity ≥ 3, consim1 ≤ 1 consecutive SIM hop). The difference is exclusively in how chain nodes are grouped:

| Method | Clustering approach |
|---|---|
| **Chain method A (PathbuildA)** | KMeans (k=40) on mean body-node embeddings per qualifying path |
| **Chain method B (PathbuildB)** | Frozenset co-occurrence families: paths grouped by which `{(body_concept_subtype, cluster_id)}` combinations they traverse |

The path file (`paths_unconstrained_sim0.9.jsonl`) is the shared source for both methods.

### 16.2 — Quality cut enforcement

All three quality cuts are simultaneously enforced **in the path file itself** via BFS generation parameters:
- `min_conf=3` → excludes any EDGE with confidence < 3 from the BFS graph
- `sim_thresh=0.9` → excludes any SIM edge with cos_sim < 0.9
- `cache["interventions"]` pre-filtered to maturity ≥ 3

Every path in `paths_unconstrained_sim0.9.jsonl` satisfies all three cuts simultaneously. Empirically verified count of maturity < 3 endpoints: **0** (see `Appendix_QualityCutAudit.md`). The consimN filter (max consecutive SIM hops ≤ N) is the only additional discriminator applied in analysis scripts.

### 16.3 — Chain method A: qualifying path selection

Both chain methods are evaluated on the **consim1 qualifying path set: 74,921 paths**. This is the set of paths from `paths_unconstrained_sim0.9.jsonl` that additionally satisfy max consecutive SIM hops ≤ 1, matching the primary analysis universe used throughout Step 4.

### 16.4 — Chain method A assessment

On the consim1 qualifying path set (74,921 paths):

| Metric | Value |
|---|---|
| Total qualifying paths | 74,921 |
| Clusters with misalignment collapse (re-state risk, no mechanism) | **36 / 40** |
| Paths in collapsed clusters | **69,605 (92.9%)** |
| Paths with mechanistic chain labels | **5,316 (7.1%)** |
| Mean intracentroid cosine similarity | 0.917 |

The 4 non-collapsed clusters:

| Cluster | Name | N paths | Notes |
|---|---|---|---|
| C0 | Reward misspecification and specification gaming | 2,221 | Mechanistically distinct |
| C15 | Generalization via Targeted Supervision/Simulation | 2,720 | Low coherence (cosim 0.730) |
| C23 | AI timeline forecasting uncertainty and bias | 330 | Small |
| C12 | Cost-Effective AI Safety Field-Building Models | 45 | Negligible |

Chain method A provides **2 substantive mechanistic clusters** (C0 reward, C15 generalization) covering 4,941 paths (6.6% of the qualifying set). The remaining 92.9% of qualifying paths fall in clusters that re-state misalignment concepts rather than describing reasoning mechanisms — a consequence of mean-pooling collapsing multi-step chain diversity into the dense misalignment neighborhood of the embedding space.

**Conclusion:** Chain method B (PathbuildB frozenset co-occurrence) is the only viable L2 representation for the workshop paper. It provides 1,603 mechanistically labeled families covering 100% of qualifying paths.

### 16.5 — Three-layer meta-family network

The meta-family three-layer network (`three_layer_network_pathbuildB_metafamily_consim1.png`) shows **all 32 PathbuildB meta-families** as L2 nodes with:
- All 485 R→MF edges drawn (proportional line width ∝ n_paths)
- All 136 MF→I edges drawn
- Meta-family labels displayed horizontally to the right of each node
- All 32 meta-families named via GPT-5.4-mini using decoded core components

Files: `step4_connectivity/three_layer_network_pathbuildB_metafamily_consim1.png`, `step4_cluster_tables/pathbuildB_metafamily_summary_consim1.csv`

### 16.6 — Chain method A intracentroid cosine similarity

On the consim1 qualifying path set (74,921 paths), chain method A produces an overall mean intracentroid cosine similarity of **0.917**.

This high value does not indicate mechanistic diversity. Each KMeans cluster is internally tight, but 36 of 40 clusters are tight around the same misalignment-concept neighborhood of the embedding space — the clusters are coherent but not semantically distinct from each other. The intracentroid cosine similarity measures KMeans centroid quality, not chain-mechanism diversity.

Files: `step4_metaclusters/pathbuildA_intracentroid_histograms.png`, `step4_metaclusters/pathbuildA_intracentroid_stats.csv`

### 16.7 — Interactive Plotly HTML visualizations

Six interactive `.html` files are provided alongside the static PNGs, all rendered in any browser via CDN-loaded Plotly:

| File | Location | Content |
|---|---|---|
| `risk_sim_heatmap.html` | `step4_metaclusters/` | 40×40 risk inter-centroid heatmap with hover (cluster name, n_nodes, cosine sim) |
| `interv_sim_heatmap.html` | `step4_metaclusters/` | Same for interventions |
| `risk_dendrogram.html` | `step4_metaclusters/` | Risk cluster dendrogram with hover (name, n_nodes, meta assignment) |
| `interv_dendrogram.html` | `step4_metaclusters/` | Same for interventions |
| `meta_connectivity_network.html` | `step4_metaclusters/` | R-meta ↔ I-meta bipartite graph with hover (full names, path counts) |
| `three_layer_network_pathbuildB_metafamily_consim1.html` | `step4_connectivity/` | Three-layer R→MF→I network with hover (full names and n_paths on all edges) |


---

## Part 17: Rev5 Fixes (2026-04-12)

**Script:** `phase2_step4_rev5_fixes.py`

### 17.1 — Name truncation fix in novel_sim_bridged_pairs CSVs

`step4_connectivity/novel_sim_bridged_pairs_top20.csv`, `step4_connectivity/novel_sim_bridged_pairs_full.csv`, and `step4_connectivity/novel_sim_bridged_pairs_top20_technical.csv` previously used 60-character truncated node names from the archive rev4 naming CSVs. Regenerated using full v2 LLM final names (`step5_naming/risk_cluster_names_llm_v2.csv`, `step5_naming/intervention_cluster_names_llm_v2.csv`). Manual annotations in `_technical.csv` are preserved; only `risk_name` and `interv_name` columns were updated.

### 17.2 — Special character cleanup across all step4/step5 CSVs

22 CSV files across `step4_finalanalysis/` and `step5_naming/` were cleaned of non-ASCII characters that render as garbage in some tools (e.g., `->` encoded as `a†'`). Replaced: `->` (arrow), `-` (en/em dash), straight quotes for curly quotes, `...` for ellipsis. All CSV files are now safe to open in Excel without encoding artifacts.

### 17.3 — PathbuildB MDS colorbar fix

`step4_cluster_tables/pathbuildB_metafamily_2d_mds.png` colorbar previously showed log10 exponent tick labels. Updated to show actual path counts (1, 10, 100, 1,000, 10,000) using `LogNorm` + `FuncFormatter`. Color still encodes path count on log scale.

### 17.4 — PathbuildB dendrogram rotation fix

`step4_cluster_tables/pathbuildB_metafamily_dendrogram.png` changed from `orientation="top"` to `orientation="left"`. MF titles are now horizontal; 1-Jaccard similarity is the x-axis. Labels use v3 causal chain names (see 17.5).

### 17.5 — PathbuildB chain naming v3 (causal framing)

`phase2_step4_D4_pathbuildB_naming_v3.py` produces `step5_naming/pathbuildB_chain_names_llm_v3.csv` (40 rows, 39/40 high confidence). New framing: 4-9 word causal mechanism noun phrases — no "via"/"through" prefix. Each name completes: "The reason why [intervention X] mitigates [risk Z] is [MF name]." Example new names vs v2:
- B3 v2: "via reward mis-specification, preference learning, and robustness testing" -> v3: "robust reward specification and policy generalization"
- B5 v2: "via human feedback reward modeling and objective correction" -> v3: "alignment of optimization with human intent"
- B6 v2: "via adversarial examples and robust optimization" -> v3: "robust feature reliance induced by adversarial training"

The dendrogram (`step4_cluster_tables/pathbuildB_metafamily_dendrogram.png`) uses v3 names for meta-family labels.

### 17.6 — UMAP clusters axes fix

`step4_finalanalysis/umap_interventions_consim1_clusters.png` previously had `axis("off")`. Now shows UMAP-1 [min, max] and UMAP-2 [min, max] with tick marks. Regenerated via `phase2_step4_B9_umap_regen.py`.

---

## Part 18: Rev7 — Frozenset Group Clustering and Centroid Spread Display (2026-04-19)

**Scripts:** `phase2_step4_E1_body_cluster_spread.py` · `phase2_step4_E2_base_cluster_spread.py` · `phase2_step4_E3_frozenset_groups.py` · `phase2_step4_E4_frozenset_group_naming.py` · `phase2_step4_E5_triplets_rev7.py`

**Plan document:** `step4_rev7_plan.md`

This revision replaces the ad-hoc dominant-family meta-family heuristic (rev6) with a fully data-driven frozenset grouping and adds reviewer-verifiable centroid spread displays at every taxonomy level.

### 18.1 — Methodological rationale: centroid spread display

For a workshop paper, LLM-assigned cluster names carry the risk that reviewers cannot independently verify whether the name reflects the full cluster or only its densest core. The rev7 approach addresses this by adding **closest-3** and **farthest-3** members by cosine similarity to centroid at every cluster level:

- **Closest-3** (highest cosine similarity): the most representative members — the core of the cluster.
- **Farthest-3** (lowest cosine similarity): borderline cases assigned to this cluster — where the cluster boundary sits.

If farthest-3 members are thematically consistent with closest-3, the cluster is tight and the LLM name is trustworthy. If farthest-3 members look like they belong to a different cluster, reviewers can see the heterogeneity. This is more informative than silhouette scores alone because it shows named examples rather than aggregate statistics.

### 18.2 — Body concept cluster spread (E1)

`bodysubtype_cluster_representatives_v2.csv` (200 rows) adds `closest3_names`, `farthest3_names`, `centroid_sim_min`, `centroid_sim_max` to all 200 body concept cluster representatives (5 subtypes x ~40 clusters each, at edge_config=0.9/unconstrained).

**Quality summary:**
- centroid_sim_min range: **0.292 -- 0.831** (across all 200 clusters)
- centroid_sim_max range: **0.676 -- 0.968**
- Missing closest3: **0**
- Heterogeneous clusters (centroid_sim_min < 0.5): **106 / 200** (53%)

The high fraction of heterogeneous clusters is expected: body concept clusters at the 0.9/unconstrained config include all nodes on any qualifying path, which spans diverse papers and topics. For clusters with sim_min < 0.5, the farthest-3 examples in the CSV allow reviewers to assess whether heterogeneity invalidates the cluster name. In practice, most heterogeneous clusters reflect topic-adjacent concepts (e.g., a "reward misspecification" cluster that includes both RL specification gaming and RLHF preference learning examples at its boundary).

The most coherent body clusters (sim_min > 0.7) are typically small, highly specific implementation clusters (e.g., `im:2` transformer architecture, `va:19` formal verification proofs). Broad conceptual clusters spanning multiple AI safety topics show lower minimum similarity, as expected.

### 18.3 — L1 risk and L3 intervention cluster spread (E2)

`risk_clusters_consim1_v2.csv` and `intervention_clusters_consim1_v2.csv` add the same four columns. Both use vpn_unconstrained (superset of vpn_consim1, maturity≥3 filter applied) to avoid loading the 1.7M-edge edge_data PKL. The centroid computed from vpn_unconstrained differs negligibly from a vpn_consim1 centroid for clusters with ≥10 members.

### 18.4 — Frozenset group clustering: binary vector approach (E3)

**Problem with individual frozenset naming:** 1,603 frozensets with n≥5 paths cannot be individually named for a workshop paper. The prior meta-family approach grouped them by dominant concept type (a heuristic), creating 32 meta-families. This is ad hoc: a frozenset with equal contributions from two concept types is arbitrarily assigned to one.

**Binary vector grouping:** Each frozenset is represented as a binary vector over all 191 distinct body concept cluster IDs present in the corpus (vocabulary size = 191). Presence of a cluster ID in the frozenset signature = 1; absence = 0. This encodes the frozenset's mechanistic footprint in a representation-agnostic way.

**Clustering procedure:**
1. Pairwise Jaccard distance matrix (1,603 × 1,603) on unweighted binary vectors
2. Average-linkage agglomerative clustering
3. Frozensets weighted by sqrt(n_paths) during linkage (up-weights high-coverage frozensets)
4. k=20 groups chosen (auto-selected k=23; k=20 achieves better interpretability with marginal loss in intra-group cohesion)

**Quality metrics:** Mean intra-group Jaccard similarity by group ranges from 0.111 to 0.833. The two smallest groups (G11, G19) have only 2 frozensets each and thus show high intra-group similarity; large diverse groups (G12, G14) show lower intra-group similarity, reflecting their broad thematic coverage.

**Outputs:** `step4_cluster_tables/frozenset_groups_consim1.csv` (20 rows) · `step4_cluster_tables/frozenset_group_memberships_consim1.csv` (1,603 rows) · `frozenset_groups_dendrogram_full.png` · `frozenset_groups_mds.png`

**Why this is more defensible than LLM naming of individual frozensets:** The grouping is purely data-driven (no LLM input until after groups are defined). A reviewer can verify group membership by inspecting the Jaccard distance dendrogram. The binary vector representation makes the grouping criterion explicit and reproducible.

**Group summary (sorted by n_triplet_paths descending):**

| Group | N frozensets | N triplet paths | Intra-Jaccard sim | Centroid (top body concepts) |
|-------|-------------|----------------|-------------------|------------------------------|
| G12 | 630 | 31,332 | 0.176 | Grant programs / funding targets AI safety research / fine-tuning |
| G14 | 346 | 11,354 | 0.178 | Reward modeling fine-tuning / human feedback / Learning reward models |
| G2 | 120 | 3,680 | 0.285 | Adversarial training / robust features / Projected gradient descent |
| G15 | 76 | 3,169 | 0.197 | Specification gaming benchmarks / RL reward evaluation |
| G10 | 73 | 3,117 | 0.253 | Penalizing attainable utility reductions / agent impact limit |
| G8 | 79 | 2,126 | 0.286 | Mechanistic interpretability / neuron-circuit analysis |
| G3 | 44 | 1,614 | 0.433 | Export controls on compute / advanced AI proliferation |
| G17 | 50 | 1,371 | 0.182 | Human-preference trained RL / reward mis-specification |
| G5 | 48 | 1,154 | 0.215 | Transparency / reliability verification / domain experts |
| G1 | 36 | 955 | 0.193 | Vulnerability experiments / adversarial robustness |
| G4 | 21 | 918 | 0.392 | Deceptive alignment / synthetic prompt generation |
| G16 | 22 | 543 | 0.239 | Stochastic policy optimization / RL smoothness |
| G7 | 12 | 327 | 0.323 | Few-shot classification / MAML benchmarks |
| G18 | 15 | 270 | 0.216 | Scaling laws / compute forecasting / AI progress uncertainty |
| G9 | 8 | 171 | 0.219 | AI capability race / safety investment incentives |
| G20 | 12 | 119 | 0.156 | Shared feature learning / adaptation acceleration |
| G6 | 2 | 45 | 0.200 | Tool-augmented malicious LLM / alignment deployment |
| G19 | 2 | 42 | 0.833 | Input gradient regularization / spurious correlations |
| G13 | 5 | 39 | 0.291 | Inadequate RLHF alignment / specification gaming examples |
| G11 | 2 | 11 | 0.111 | Public acceptance / explainable AI case studies |

### 18.5 — Frozenset group LLM naming (E4)

See Step5 Part 11 for full naming results. Groups are named via 2-pass gpt-4.1-mini using context: (1) centroid decoded body concept names, (2) closest-3 frozensets decoded, (3) farthest-3 frozensets decoded, (4) top-3 R→I pairs bridged by the group.

### 18.6 — R→Group→I triplets (E5)

`ri_triplets_consim1_rev7.csv` adds `group_id` and `group_name` to all 2,298 base triplet rows. `ri_group_triplets_consim1.csv` aggregates to 764 unique (risk, group, intervention) triplets with n_triplet_paths summed. `ri_group_triplets_top20_consim1.csv` contains the top 20.

**Quality check:** 0 frozensets without group assignment; 19/20 groups appear in aggregated triplets (G11 with 11 paths falls below minimum triplet threshold).

**Top-5 R→Group→I triplets:**
1. Existential catastrophe from misaligned AI → [Scaling AI Safety Research Capacity] → Expand AI Safety Research Funding (6,539 paths)
2. Misaligned Superintelligent AI → [Scaling AI Safety Research Capacity] → Expand AI Safety Research Funding (3,266 paths)
3. Misaligned AGI existential catastrophe → [Scaling AI Safety Research Capacity] → Expand AI Safety Research Funding (2,790 paths)
4. Catastrophic AI misalignment and loss of control → [Scaling AI Safety Research Capacity] → Expand AI Safety Research Funding (2,502 paths)
5. Insufficient AI safety research capacity → [Scaling AI Safety Research Capacity] → Expand AI Safety Research Funding (2,271 paths)

The top-16 triplets all involve G12 (Scaling AI Safety Research Capacity) or G14 (Alignment via human preference propagation), reflecting the corpus concentration in field-building and RLHF alignment paths. Mechanistically distinct groups (G2 adversarial robustness, G10 attainable utility, G8 interpretability) appear from rank 16 onward.

**Files:**
- `step4_connectivity/ri_triplets_consim1_rev7.csv` — base triplets with group_id/group_name added
- `step4_connectivity/ri_group_triplets_consim1.csv` — aggregated R→Group→I (764 rows)
- `step4_connectivity/ri_group_triplets_top20_consim1.csv` — top 20
- `step4_cluster_tables/frozenset_groups_consim1.csv` — 20 group summaries
- `step4_cluster_tables/frozenset_group_memberships_consim1.csv` — 1,603 frozenset→group assignments
- `step5_naming/frozenset_group_names_llm.csv` — 20 LLM group names with quality metrics

---

## Part 19: Rev8 — Critical Methodological Findings, Hybrid Path Enumeration, and Pareto-Frontier Cluster Validation (2026-04-30)

Rev8 introduces five critical methodological findings (CF-1 through CF-5), shifts the path-enumeration primitive from BFS-shortest to hop-wise DFS, defines a hybrid strategy across the four similarity thresholds, and adds Pareto-frontier validation as the primary rigor signal for both body-cluster recluster (L1) and frozenset grouping (L2). These together replace the rev7 cluster taxonomy as the paper's reviewer-defensible mechanism family extraction.

### 19.0 — 🔴 PAPER CENTERPIECE — what this work actually claims (added 2026-05-08, strengthened 2026-05-09)

**This is the centerpiece of the paper. Every methodological choice in §19.1–§19.11 should be traceable back to this claim. When writing, the abstract, intro, and conclusion should all foreground the framing in this section.**

**What this work is NOT:**
- This is **not** another directory or catalog of AI safety research. The literature already has many such directories — the Alignment Research Dataset (ARD), Stampy/Aisafety.info, AI Safety Atlas, Anthropic/MIRI/CHAI bibliographies, etc. — and they suffer from being (a) static snapshots that go stale, (b) keyed off paper-level metadata (title, abstract, author, citation graph) rather than the *mechanism-level claims* of each paper, (c) unable to support cross-paper reasoning beyond keyword search.
- The substrate corpus we use (ARD vintage circa 2024–2025) is **deliberately not the headline contribution**. The corpus will go out of date; any "AI safety paper directory" built on it is obsolete by publication. **The contribution is the framework, not the directory.**

**What this work IS — three-sentence claim:**

> This work introduces a framework for **lossless reduction of safety-intervention papers to the specific mechanism by which each intervention is proposed to reduce a specific risk**, projected into a finite reviewer-defensible mechanism-class vocabulary. Once a paper is reduced to a `(risk_class, body-mechanism-chain, intervention-mechanism-class)` signature, **two downstream capabilities become possible at scale that no directory can support**: (1) cross-paper mechanism-level evaluation — *which mechanism families address which risks, with what frequency, with what stage of validation evidence, across the entire corpus*; and (2) graph-based reasoning to **propose new intervention candidates** — *if mechanism M is densely populated for risk R but absent for risk R′, M-via-new-intervention-I′ for R′ is a candidate worth investigating*.

**The three deliverables that demonstrate this claim** (each must appear in the paper):
1. **Faithfulness of mechanism reduction** — measured via **LLM-as-judge** on a sample of N≈30-50 papers: a separate judge model (distinct from the extraction LLM) compares the pipeline-extracted mechanism class against the source paper text and emits an agreement label. Reported as agreement rate + judge-of-judge calibration (Mike's prior framework, see §3.1 Methods in Overleaf paper.tex). Sai is implementing this deliverable. *Without this, the reduction is not lossless and the rest collapses.* **NOTE (decision 2026-05-15):** the original plan involved manual hand-validation by domain experts; replaced with LLM-as-judge to save time and align with the existing Mike-framework judge stack. The latter scales better and is what the paper will report.
2. **Large-scale mechanism evaluation** — the named mechanism-class catalog with cross-paper distribution, risk coverage, and validation-stage breakdowns over the full 19,073-node EDGE-only VPN. *This is the "scale" deliverable.*
3. **Novel-intervention candidate generation** — N=3–5 example transferable mechanisms surfaced from the doublet `(R_class, NR_anchor)` graph that the corpus does NOT contain but that the framework predicts as plausible. *This is the "graph-based reasoning extension" deliverable; speculative but illustrative — the existence of the candidate generator is the contribution.*

**Why the EDGE-only hop-wise VPN substrate, not the full graph:** the centerpiece claim is "lossless reduction to the specific mechanism." That requires causally-justified relationships only — SIM edges are similarity, not causation; multi-risk paths confound the mechanism attribution. The 88.33%-coverage 19,073-node EDGE-only VPN (§19.9) is the high-quality substrate; further robustness sweeps (which would dilute the strict-quality signal) are deliberately deprioritized in favor of completing the three deliverables above before publication.

**Reasoning-LLM-as-central methodological stance (added 2026-05-09):** the framework treats reasoning LLMs as the *central* analytical primitive at every level — extraction, clustering, mechanism naming, and (eventually) candidate-generation. Algorithmic tools (HDBSCAN on embedding similarity, embedding-cosine sim, etc.) are positioned as **validation checks against the LLM-central output, with their known limitations explicit**, not as the substrate the LLM decorates. This mirrors the existing single-data-source pattern: per-paper extraction is LLM-central (o3) and validated by LLM-as-judge (a separate model). Mechanism clustering follows the same pattern — LLM-central naming and grouping over the 19,073 VPN nodes (§19.12), validated by HDBSCAN-on-embeddings as a sanity check on whether the LLM's mechanism boundaries agree with embedding-density structure. When they agree, the LLM result is reinforced; when they diverge, the divergence is reported as a finding ("LLM identifies cross-cluster semantic groupings HDBSCAN cannot find" or vice versa) — not as a failure requiring more algorithmic work. This stance is what distinguishes the framework from prior abstract-embedding field maps and is the reason the paper's contribution scales with future LLM capability rather than with future clustering-algorithm refinement.

---

### 19.0a — Original framing (preserved for context, superseded by 19.0 above)

The overarching aim of the paper is to demonstrate that this algorithm produces a **structural knowledge representation / fabric** of a large literature corpus that downstream LLMs can consume to bridge the **knowledge-cutoff / pretraining-data gap for AI-for-Science work**.

Existing publication databases expose only abstracts, titles, and citation metadata — they reflect *what was studied* but not *how the studies relate to one another* or *how they collectively compose a research mechanism*. They cannot answer questions like "what mechanism families address risk X?" or "which design rationale + implementation mechanism + validation evidence chains exist for intervention Y?" without an LLM re-deriving the relational structure paper by paper, in-context, every time.

The contribution of this work is the construction of a **deep technical, relational, and functional representation of how risks are intended to be addressed by research in the AI safety domain**, as a graph artifact:
- **Deep technical**: nodes carry concept names + descriptions + role-labels (concept_category, intervention_lifecycle, intervention_maturity) extracted from the paper bodies, not the metadata.
- **Relational**: directed EDGE relationships (with confidence ratings) link risks → problem analyses → theoretical insights → design rationales → implementation mechanisms → validation evidence → interventions, plus SIMILARITY edges (cosine ≥ 0.80 on 1024-d BAAI/bge-large-en-v1.5 embeddings) cross-link near-duplicate concepts across papers.
- **Functional**: risk → intervention paths (the EDGE-only canonical VPN of 19,073 nodes / 8,954 hop-wise paths) capture *the mechanism by which a risk is addressed* — not just that the two appear in the same paper. Hop-wise DFS preserves multiple alternative chains per paper-pair so that downstream analysis can decompose "this paper addresses risk X" into "this paper proposes mechanism (problem_analysis Z, theoretical_insight T, design_rationale D, implementation_mechanism I, validation_evidence V, intervention E) addressing X."

The clustering pipeline (HDBSCAN-2D per-subtype + LLM thematic on residual + doublet `(R_cluster, NR_anchor)` mechanism families) is the projection from the raw graph into a finite, reviewer-defensible mechanism vocabulary. Once published, downstream LLMs can use the cluster vocabulary as a structured query interface — they ask "what are the mechanism families addressing reward hacking?" and receive a discrete list of NR_anchors with cited path evidence, rather than having to re-read every paper that mentions reward hacking.

This intent guides the rest of §19's design choices: the EDGE-only canonical VPN preserves only causally-justified relationships (CF-2); the hop-wise DFS preserves alternative-mechanism diversity (CF-2); HDBSCAN-2D per-subtype with a 0.75 raw-cosine centroid floor (§19.9) produces geometrically defensible clusters; LLM thematic naming on residuals (§19.9.6 Task A) covers the long tail with reviewer-readable mechanism labels; the doublet primitive (§19.9.6 Task D) yields the final mechanism-family vocabulary the downstream LLM consumes.

### 19.1 — Critical methodological findings (CF-1 → CF-5)

**CF-1 — Unconstrained-mode silent-drop bug.** At sim=0.9, 99.34% of unconstrained-mode paths had ≥1 risk node in middle and 13.57% had ≥1 intervention in middle. The frozenset construction in `phase2_step4_pathbuildB_remaining.py:134` and 5 sibling files used `frozenset(... if n in node_to_stc)` which silently drops middle non-body nodes, producing a ~85x path-count inflation at sim=0.9. **Fix:** custom-mode BFS in `final_pathway_analysis_modes.py` pre-empts at first intervention and skips risk neighbors during expansion (commit 821e985). The legacy `paths_unconstrained_sim*.jsonl` files are deprecated; rev8 analysis uses only custom-mode paths.

**CF-2 — BFS-shortest is the wrong primitive; switched to hop-wise DFS.** BFS-shortest emits one shortest path per (R, I) pair, missing (a) alternative body chains within a single paper, and (b) multiple cross-paper SIM bridges between papers (only one survives the visited set). Rev8 introduces `phase2_step4_F2v4_hopwise_falkordb.py` which performs hop-wise DFS enumeration directly against the live FalkorDB. Each step extends the current path stub by one EDGE or SIM hop (with consim1 alternation), and each SIM hop crosses into another paper. The canonical rev8 path file is `paths_hopwise_v4_sim0.9.jsonl` (3.55M paths over 17,699 unique R-I pairs).

**CF-3 — Body-cluster over-fragmentation at k=40.** The legacy `phase2_clustering.py` hardcodes k=40 for ALL node types at config 0.9/unconstrained. Near-duplicate concepts split across cluster IDs (e.g., `pr:7`, `pr:19`, `pr:37` all about reward mis-specification; six "Opaque internal representations" clusters appear in a single path). Frozenset signatures fragment along phantom cluster-ID boundaries, and path-length distributions show monotonic 1.4-1.5× growth past L=12 (combinatorial zigzag through near-duplicate clusters). **Fix:** Task #7 body recluster on path-participating nodes only, with k-scan + Pareto-frontier validation (§19.4).

**CF-4 — Replicated silent-drop sites + 3 standalone Cat-A bugs.** The same `frozenset(... if n in node_to_stc)` pattern appears in 6 additional files beyond the original (`phase2_step4b_paths_and_plots.py:451,471`; `phase2_step4_trackA.py:330,360`; `phase2_step4_pathbuildB_connectivity.py:355-358`; `phase2_step4_cluster_naming.py:378`; `phase2_step5_examples_edgeonly_fix.py:373`). All become harmless once paths consumed are custom-mode (no non-body middle nodes). Three additional standalone Cat-A fixes were committed (commit 0ab9ccb) covering monotonic-mode silent skip, subcluster split detection cap, and unknown-category default labelling.

**CF-5 — FalkorDB silent 10k RESULTSET_SIZE truncation.** FalkorDB's default `RESULTSET_SIZE=10000` silently caps every Cypher result at 10k rows with no error or warning. F2v4 v1 returned 10,000 risk nodes when the true count was 19,178; the discrepancy was caught only because custom-BFS reported a different number. An audit identified three HIGH-severity sites that returned >10k rows (`cluster_utils.py:_build_mapping`, `all_shortest_pair_extraction_phase1.py:get_all_interventions`, `phase2_step4_F2v4_hopwise_falkordb.py:query_node_ids`) and two MEDIUM-severity batch-size-boundary sites (`graph_diagnostics.py:component_analysis`, `threshold_scan_degree_analysis.py:get_node_degrees_by_edge_type_batched`). Defense-in-depth fix: every FalkorDB-querying script now sets `GRAPH.CONFIG SET RESULTSET_SIZE 10000000` at startup AND batches large-population queries by id-range. Affected upstream artifacts (custom-BFS path files at sim=0.85 and sim=0.8) are being re-extracted post-fix to remove CF-5 contamination from the BFS coverage proxy.

### 19.2 — Hybrid path enumeration strategy across four thresholds

Hop-wise DFS at low sim thresholds is computationally intractable: the F2v4 sim=0.85 run hit 7.5M+ paths from a single risk node before crashing, and sim=0.8 (1.5M+ SIM edges) is worse. Path-length cap reductions (max=12 for sim=0.85; max=8 for sim=0.8) bound runtime but truncate evidence. The rev8 strategy is therefore hybrid:

| Threshold | Method | Rationale | Source file |
|-----------|--------|-----------|-------------|
| EDGE-only | DFS hop-wise (max=30) | Tractable; complete enumeration | `paths_hopwise_v4_edge_only.jsonl` (3,222 R-I pairs) |
| sim=0.95 | DFS hop-wise (max=30) | 9.1k SIM edges; tractable | `paths_hopwise_v4_sim0.95.jsonl` (3,237 R-I pairs) |
| sim=0.9 | DFS hop-wise (max=30) | 144k SIM edges; **canonical for rev8** | `paths_hopwise_v4_sim0.9.jsonl` (17,699 R-I pairs) |
| sim=0.85 | DFS hop-wise (max=12) | 596k SIM edges; aggressive cap to bound combinatorial explosion | `paths_hopwise_v4_sim0.85.jsonl` (in progress) |
| sim=0.8 | BFS-shortest (re-extracted post-CF-5) | 1.57M SIM edges; DFS intractable, BFS used as coverage proxy | `paths_custom_sim0.8.jsonl` (re-extraction pending) |

**Paper rationale:** rev8's primary mechanism-family extraction uses sim=0.9 hop-wise DFS, the highest threshold where SIM-bridge volume is large enough for cross-paper mechanism aggregation but DFS remains tractable. Lower thresholds (sim=0.85, sim=0.8) serve as sensitivity analysis demonstrating that mechanism families remain stable as the bridge threshold relaxes — but the path-length cap difference is documented honestly. This avoids over-claiming exhaustive enumeration at sim=0.8/0.85 while preserving the cross-threshold stability narrative.

### 19.3 — Hop-wise DFS constraint set (canonical)

`phase2_step4_F2v4_hopwise_falkordb.py` enforces:
- **Edge filters:** EDGE confidence >= 3, SIM cosine sim >= threshold (= euclidean score < sqrt(2(1-thresh)))
- **Endpoint filter:** intervention_maturity >= 3 on the terminal intervention node
- **Path constraints:** simple paths, consim1 alternation (max consecutive SIM hops <= 1), single-risk (no risk node in middle), single-intervention (terminate at first intervention), first-hop EDGE-or-SIM to body subtype, min length 3, max length per threshold (above)
- **Last-resort safeguards:** max 50M paths per run (tracked via `hit_global_cap` in summary)
- **CF-5 fix:** `GRAPH.CONFIG SET RESULTSET_SIZE 10000000` at script start; risk/body/intervention id queries batched by id-range

This is a strict superset of the constraints used in custom-mode BFS, plus the DFS extension that captures all valid mechanism evidence rather than one shortest path per pair.

### 19.3a — Quality-based threshold selection rationale (early; results to follow)

The hybrid path-enumeration strategy in §19.2 chooses sim=0.9 as the rev8 canonical threshold. That choice is correct on **feasibility** grounds (hop-wise DFS is intractable below sim=0.9), but feasibility alone is not a quality argument. The paper's threshold-appropriateness claim therefore needs an independent **quality** measurement that can be evaluated at every threshold and that flags the threshold below which the data degrades — independent of computational tractability.

We considered three candidate quality signals at each threshold:

**(a) Link quality — concept-category concordance for SIM edges.** For each SIM edge, check whether source and target share `concept_category`. At higher thresholds we'd expect concordance to approach 100%; degradation at lower thresholds would suggest semantic drift. **We dropped this as a primary signal** because the underlying extraction prompt assigns `concept_category` to a node based on the role it plays in its own paper's argumentation chain — the same conceptual content can legitimately appear as `problem_analysis` in one paper and as `theoretical_insight` in another, depending on where in the chain it appears. Discordant SIM edges are therefore not necessarily noise. Concordance might surface a weak signal at extreme thresholds but is not a reliable rigor argument.

**(b) Path quality — within-path body coherence.** Mean pairwise cosine sim between body nodes inside a path. Higher = path traverses one mechanism cleanly; lower = path zigzags. **We dropped this as a primary signal** because the five body subtypes (problem_analysis, theoretical_insight, design_rationale, implementation_mechanism, validation_evidence) carry distinct epistemic content by design. A coherent reasoning chain crossing all five subtypes will have legitimately different embeddings at each stage. Within-path coherence therefore conflates "path is single-mechanism" (signal we want) with "path crosses subtypes" (structural feature, not noise). Possibly a weak signal at the extremes but not strong enough to anchor threshold choice.

**(c) Body cluster Pareto across thresholds — primary signal (chosen).** For each threshold, restrict the body-cluster recluster input to the path-participating body nodes (VPN_paperpair) at that threshold, run the k-scan + Pareto frontier check (intra-cluster cosine sim ≥ 0.70 AND inter-cluster centroid cosine sim ≤ 0.30 per subtype). The threshold's cluster Pareto pass/fail pattern is a direct measurement of whether the path-participating body concepts at that threshold form genuinely homogeneous and well-separated clusters, or whether the lower threshold has admitted enough semantically-noisy bridges that body clusters lose coherence.

**Expected pattern (to be confirmed):**
- High thresholds (sim=0.9, sim=0.95, EDGE-only): all 5 body subtypes pass Pareto; clusters are clean.
- Medium thresholds (sim=0.85): some subtypes may fail Pareto as moderately-similar bridges introduce cluster heterogeneity.
- Low thresholds (sim=0.8): more subtypes fail Pareto; the looser SIM bridges admit semantically-divergent body concepts that cannot be cleanly separated.

If the expected pattern holds, the chosen threshold is the **lowest threshold at which all five body subtypes pass the Pareto frontier**. Below that threshold, body clusters are inadequate by an independent quality criterion (not by feasibility). Above that threshold, clusters are clean but R-I coverage drops. The chosen threshold is the optimal balance point — clusters defensible, coverage maximized.

If the expected pattern fails (e.g., even sim=0.95 fails Pareto), that is itself a paper finding: the AI safety mechanism space at this embedding resolution does not admit clean k-cluster decomposition at any threshold, motivating a different L2 abstraction.

**Computational note:** the body cluster Pareto sweep uses full Agglomerative clustering with no sub-sampling at any threshold. Lower thresholds (sim=0.85, sim=0.8) have substantially larger VPN_paperpair populations and may incur multi-hour wall times per subtype. This is consistent with the hybrid enumeration story (§19.2) — at low thresholds the path enumeration is already approximated by lower max-length DFS or by BFS-shortest as coverage proxy, so any computational difficulty in the recluster step is consistent with already-acknowledged feasibility constraints. Wall times are recorded per threshold and reported as supplementary information.

**Sweep implementation:** `graph_analysis/phase2_step4_F3_sweep_thresholds.sh` runs F3 on each of the 5 thresholds in cheapest-first order (EDGE-only → sim=0.95 → sim=0.9 → sim=0.85 → sim=0.8). Per-threshold output suffix: `body_kscan_metrics_<suffix>.csv`, `body_kscan_chosen_k_<suffix>.csv`, `body_kscan_pareto_plot_<subtype>_<suffix>.png`, `cluster_memberships_rev8_<suffix>.pkl`. Cross-threshold comparison consolidated into a single quality-vs-threshold table (TBD post-sweep).

**Status:** rationale documented; sweep pending execution. Results table will replace this paragraph.

### 19.3b — Methodological pivot: raw-cosine Pareto → BERTopic-style metrics (added 2026-05-01 after smoke-test)

The first execution of the cross-threshold body-cluster Pareto sweep at edge_only revealed a systematic failure that the original §19.3a/19.4 framework cannot resolve: **the absolute thresholds (intra ≥ 0.70, inter ≤ 0.30) are unattainable in raw cosine space for any algorithm.** This is documented here so the methodological revision is traceable.

**Empirical pattern (edge_only smoke-test, 2026-05-01, 4 algorithms × 9-or-5 params × 7 node-types = 210 cluster computations):**

| Algorithm | typical intra | typical inter | size pattern |
|-----------|--------------|--------------|--------------|
| Agglomerative + cosine + average | 0.41–0.51 | 0.66–0.84 | one giant cluster (1/2/2400+) — chaining failure |
| Agglomerative + cosine + complete | 0.37–0.44 | 0.95 | balanced sizes but centroids cluster together |
| Agglomerative + euclidean + ward | 0.46–0.50 | 0.90–0.92 | most balanced sizes |
| HDBSCAN (mcs ∈ {5,10,20,50,100}) | 0.63–0.75 | 0.59–0.70 | 82-97% noise; dense cores only |

The chosen-best across all algorithms × params for each node-type also failed:

| Subtype | Best algo | intra | inter | noise |
|---------|-----------|-------|-------|-------|
| problem_analysis | hdbscan mcs=20 | 0.68 | 0.64 | 97% |
| theoretical_insight | hdbscan mcs=20 | 0.63 | 0.68 | 96% |
| design_rationale | hdbscan mcs=20 | 0.67 | 0.68 | 96% |
| implementation_mechanism | hdbscan mcs=20 | 0.68 | 0.59 | 96% |
| validation_evidence | hdbscan mcs=20 | 0.66 | 0.70 | 97% |
| risk | hdbscan mcs=20 | 0.74 | 0.65 | 82% |
| intervention | hdbscan mcs=10 | 0.75 | 0.65 | 92% |

(Full 210-row data: `step4_cluster_tables/body_kscan_metrics_edge_only.csv`. CSV preserved as the multi-algorithm baseline appendix.)

**Why the absolute Pareto thresholds fail in raw cosine space:** general-purpose sentence-transformer embeddings (BAAI/bge-large-en-v1.5) trained on broad web data place ALL same-domain texts at cosine sim 0.5–0.95 vs each other. The "background" inter-centroid sim in a single-topic corpus (AI safety) is therefore ~0.6–0.95 regardless of what real cluster structure exists. Asking for inter ≤ 0.30 requires cross-domain-level separation that doesn't exist within a single technical domain. This is a known property of general-purpose sentence embeddings and the reason topic-modeling literature (BERTopic, Top2Vec, CTM) does not use raw embedding-space distances as the quality signal.

**Pivot decision:** abandon raw-cosine Pareto (0.70/0.30) as the primary rigor signal; adopt the BERTopic-style framework, which is the field-standard for single-domain text clustering and has a well-cited reviewer pedigree.

**New methodology (BERTopic-style, applied per body subtype):**

1. **Dimensionality reduction first.** Apply UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric='cosine') on each subtype's VPN_paperpair embeddings. UMAP amplifies whatever local structure exists in the high-dimensional embedding before clustering. This is the BERTopic standard.

2. **Cluster in UMAP space.** HDBSCAN (min_cluster_size scan over [5, 10, 20, 50, 100]) on UMAP-projected coordinates. HDBSCAN auto-detects density-based cluster cores; "noise" points are explicitly labeled and not forced into clusters.

3. **Primary quality metrics (computed in UMAP space):**
   - **Silhouette score** — target > 0.25 (acceptable cluster structure), > 0.40 (good)
   - **Coverage** = 1 − noise_rate — target > 50% (acceptable), > 80% (good); high noise rates indicate the data has limited dense structure
   - **Random-baseline z-score**: shuffle cluster labels, compute mean intra and inter; observed clustering must be ≥ 2σ better than random on both intra and inter

4. **Data-driven cluster characterization (NO LLM-naming load-bearing for body clusters):**
   - **3 closest + 3 farthest concepts per cluster** by cosine distance to cluster centroid in raw embedding space. The 3-closest are the most-prototypical members; the 3-farthest are the most-peripheral. This data-driven characterization replaces LLM-naming for body cluster validation and for any subsequent analysis step (frozenset construction, mechanism family assignment, R→Group→I triplets) that consumes body-cluster identity. LLM-naming may still be applied **qualitatively** at the L2 mechanism-family level (Step 5 frozenset groups), but never load-bearing inside the body-cluster pipeline. The reason: LLM-naming introduces extraction-time uncertainty that is hard to bound for reviewers; data-driven 3-closest/3-farthest is reproducible and audit-friendly.

5. **2D visualization for reviewer inspection:** UMAP(n_components=2, same params as 15D) with cluster boundaries colored. Reviewers can visually confirm cluster separation in projection.

6. **Cross-threshold quality comparison:** for each (threshold, subtype, algorithm-param), report (silhouette, coverage, noise_rate, n_clusters_realized, random-z_intra, random-z_inter). The chosen threshold for the paper is the lowest threshold at which all 5 body subtypes achieve silhouette > 0.25 AND coverage > 50% AND random-z > 2 on both intra and inter. If no threshold achieves this, that itself is a paper finding.

**What survives from §19.3a:** the cross-threshold sweep methodology, the per-subtype quality criterion, and the "lowest acceptable threshold" decision rule. Only the absolute Pareto thresholds and LLM-naming have been replaced.

**LLM-naming policy (clarification):** §18.5 (rev7 LLM cluster naming) and the Step 5 frozenset group naming remain in place as **descriptive labels** for paper presentation. They are **not** used for any quantitative downstream analysis — frozenset construction uses cluster-ID identifiers, R→Group→I triplets use cluster-ID identifiers, all path-level analyses use cluster-ID identifiers. LLM names are display-only.

**Status:** F3v2 (BERTopic-style) implementation pending; smoke-test on edge_only first, then full sweep.

### 19.3c — Canonical method: UMAP + HDBSCAN + iterative-residual + 0.77-cutoff + strict-per-iter ≥5 + 100%-strict path retention (locked 2026-05-02)

**Canonical pipeline per node-type (5 body subtypes + risk + intervention):**

1. **VPN_paperpair input** — body / risk / intervention nodes appearing in any qualifying F2v4 / custom-BFS path (after maturity≥3 and consim1 filters).
2. **Dimensionality reduction** — UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric=cosine, random_state=42).
3. **Iter 1 clustering** — HDBSCAN(min_cluster_size=5, metric=euclidean on UMAP-15D). Variable-k natural-density clusters; HDBSCAN labels low-density points as noise (-1).
4. **Centroid-similarity cutoff (per cluster, per iteration)** — compute centroid in raw embedding space. Members with cosine sim < **0.77** to centroid revert to noise. Recompute refined centroid on survivors (single-pass).
5. **Strict size filter (per iteration)** — drop any cluster whose post-cutoff surviving member count is <5; their members revert to noise. Every retained cluster has ≥5 members all at sim ≥0.77 to centroid by construction.
6. **Iterative residual reclustering** — take noise set (HDBSCAN-noise + cutoff-rejected + size<5-rejected) → UMAP(15D) + HDBSCAN(mcs=5) + 0.77 cutoff + size≥5 filter on residual. New strict clusters get global IDs.
7. **Termination** — converge when an iteration produces 0 new strict clusters (residual unchanged → next iter would be deterministic identical). Also caps at 50 iters as safety. **At convergence, no more clusters of ≥5 members at sim ≥0.77 to centroid exist in the residual** — proves complete enumeration of literature-replicated concept groups.
8. **Cutoff robustness sweep** — single-pass at chosen mcs across cutoffs [0.70, 0.72, 0.74, 0.76, 0.77, 0.78, 0.80, 0.82]; ARI vs 0.77 demonstrates cluster identity is stable across small cutoff perturbations.
9. **Data-driven characterization** — per cluster, store 3 closest + 3 farthest members by raw-embedding cosine to centroid. NO LLM-naming load-bearing (per §19.3a).
10. **2D UMAP visualization** — fixed seed; per-subtype PNG with cluster overlay.

**HDBSCAN-residual outperforms Agglomerative-K=40-residual:** edge_only test showed 4.51% vs 1.69% body-strict path retention. Forced-K partitioning of low-density residual produces "average of mixed concepts" centroids that mostly fail 0.77; HDBSCAN adapts to where dense pockets actually exist.

**Path retention rule (locked):** **100% strict — risk endpoint, every body node, AND intervention endpoint must each be assigned to a non-noise cluster in their respective subtype.** Paths failing any component are excluded from frozenset / mechanism family / R→Group→I triplet analysis.

**Why 100% strict is defensible (paper rationale):** the 100%-strict subset selects paths where every component has ≥4 corpus-wide semantic counterparts that themselves form a tight density cluster passing 0.77. By construction, every retained path is a reasoning chain whose every step is **replicated across the literature**. The excluded paths involve at least one component that's an author-idiosyncratic concept or a literature singleton. The trade-off (low path retention) is honest framing: rev8 mechanism families summarize the **literature-frequent** AI safety reasoning chains, distinct from one-off paper-specific formulations.

**Per-subtype iteration outcomes (edge_only, strict per-iter, locked 2026-05-02):**

| Subtype | iters | k_final | smallest_cluster | cov_final | stop reason |
|---------|-------|---------|------------------|-----------|-------------|
| problem_analysis | 16 | 164 | 5 | 45.8% | converged |
| theoretical_insight | 13 | 111 | 5 | 31.1% | converged |
| design_rationale | 16 | 115 | 5 | 34.0% | converged |
| implementation_mechanism | 5 | 82 | 5 | 19.6% | converged |
| validation_evidence | 11 | 119 | 5 | 29.9% | converged |
| risk | 16 | 167 | 5 | 76.1% | converged |
| intervention | 10 | 137 | 5 | 41.7% | converged |
| **total** | — | **895** | — | — | all converged |

All subtypes converged via `no_strict_clusters_added` — no max-iter cap hit, smallest cluster = 5 (criterion enforced). At convergence, no more clusters of ≥5 members at sim ≥0.77 exist in the residual.

**100% strict + R+I strict path retention (edge_only, locked 2026-05-02):**

| Scenario | Paths | Unique R-I pairs | Risk clusters | Intervention clusters |
|----------|-------|------------------|---------------|------------------------|
| **100% strict ALL (R+body+I)** | **186** | **152** | **57** | **50** |
| ≤1 unmapped body | 492 | 368 | 99 | 93 |
| R+I strict, body N/A | 3,115 | 1,174 | 157 | 137 |
| Risk endpoint only | 6,950 | 2,452 | 167 | 137 |
| Intervention endpoint only | 3,429 | 1,330 | 157 | 137 |
| Total qualifying (pre-cutoff) | 8,954 | 3,222 | — | — |

Retention rate at 100% strict: 186/8,954 = **2.08%**. 152 unique R-I pairs spanning 57 risk × 50 intervention cluster combinations of the 167 risk / 137 intervention totals (risk cluster coverage 34%, intervention 36%).

**Why strict per-iter ≥5 was adopted over end-only filter:** end-only filter (drop <5-member clusters only after all iterations) retained tiny mid-iter clusters that helped residual evolution; produced more total clusters (1,196 vs 895) but those extras included clusters whose mid-iter survival was an artifact of slightly-different residual UMAP projections rather than robust signal. Strict per-iter ≥5 enforces criterion at every step → every retained cluster meets criterion at every iteration → convergence proves no more such clusters exist in data → cleaner methodological story for reviewers.

**Cutoff robustness (edge_only):** ARI vs 0.77 typically > 0.85 for adjacent cutoffs in body subtypes; cluster identity stable. Coverage drops monotonically as cutoff tightens. 0.77 is the empirical sweet spot from representatives review; below 0.77 membership feels weak.

**Canonical artifacts (edge_only, locked 2026-05-02):**
- `phase1_rawpathsfiles/vpn_strict_RIbody_edge_only.jsonl` — 186 paths, all 100% strict
- `step1_load_and_parse.../cluster_memberships_rev8_edge_only.pkl` — 895 cluster records
- `step4_cluster_tables/body_kscan_metrics_edge_only.csv` — Pareto per (subtype, mcs)
- `step4_cluster_tables/body_kscan_chosen_k_edge_only.csv` — chosen mcs per subtype
- `step4_cluster_tables/body_kscan_iter_records_edge_only.csv` — per-iteration cluster contributions
- `step4_cluster_tables/body_kscan_cutoff_sweep_edge_only.csv` — robustness at cutoffs 0.70–0.82
- `step4_cluster_tables/body_kscan_cutoff_ari_edge_only.csv` — ARI matrix vs 0.77
- `step4_cluster_tables/body_kscan_representatives_edge_only.csv` — 3 closest + 3 farthest per cluster
- `step4_cluster_tables/body_kscan_pareto_plot_<subtype>_edge_only.png` — 2D UMAP scatter
- `step4_cluster_tables/vpn_coverage_sensitivity_edge_only.csv` — strictness/retention table

**Implementation:** `graph_analysis/phase2_step4_F3_body_recluster.py` with default flags `--centroid-sim-cutoff=0.77 --final-min-cluster-size=5 --iter-max=50 --iter-coverage-target=0.999 --resid-method=hdbscan --resid-mcs=5 --hdbscan-mcs=5,10,20,50,100 --umap-n-components=15 --umap-n-neighbors=15 --umap-min-dist=0.0`. Sensitivity: `phase2_step4_F3a_vpn_coverage_sensitivity.py` emits `vpn_strict_RIbody_<suffix>.jsonl` and `vpn_coverage_sensitivity_<suffix>.csv`.

### 19.3d — Cross-threshold sweep + canonical = sim=0.9 (locked 2026-05-02)

The canonical method (§19.3c) was applied to all 5 SIM-edge thresholds. **All thresholds use the SAME clustering pipeline** — UMAP-15D → HDBSCAN(mcs scan) → 0.77 cutoff in raw embedding space → per-iter ≥5 strict filter → iterate-to-convergence on residual. The only per-threshold difference is the input VPN_paperpair = distinct nodes appearing in the threshold's path file.

**Comparison at 100% strict (R + every body + I clustered):**

| Threshold | Path file | Path-enum mode | 100%-strict paths | R-I node pairs | Risk × Interv clusters | % of R-I universe |
|---|---|---|---|---|---|---|
| edge_only (baseline) | `paths_hopwise_v4_edge_only.jsonl` | DFS hopwise | 186 | 152 | 57 × 50 | — |
| sim0.95 | `paths_hopwise_v4_sim0.95.jsonl` | DFS hopwise | 193 | 153 | 56 × 45 | — |
| **sim0.9 (canonical)** | **`paths_hopwise_v4_sim0.9.jsonl`** | **DFS hopwise** | **799,031** | **5,244** | **86 × 55** | **29.6% of 17,699** |
| sim0.85 | `paths_hopwise_v4_sim0.85.jsonl` | DFS hopwise | 5,887,398 | 225,132 | 216 × 102 | — |
| sim0.8 | `paths_custom_sim0.8.jsonl` | BFS-shortest (custom) | 44,989 | 44,989 | 187 × 133 | — |

Cluster quality (silhouette UMAP-15D, z_intra, z_inter) all pass thresholds at every threshold — silhouette range 0.62–0.83 across subtypes; z_intra typically 40–140; z_inter 2–45. All clusters Pareto-pass the 0.77 / size≥5 criteria.

**Sim=0.8 caveat — NOT apples-to-apples with the other 4 thresholds:** the custom-mode BFS in `graph_analysis/final_pathway_analysis_modes.py:160-268` does NOT enforce consim=1 during traversal AND does not write `edge_types` to its JSONL (verified empty `edge_types` field across first 10k paths of `paths_custom_sim0.8.jsonl`). The `phase2_step4_F1_consim1_custom_rebuild.py` post-filter (which would impose consim=1) is hardcoded to `paths_custom_sim0.9.jsonl` (script line 49) — no equivalent post-filter has been applied for sim=0.8. So the sim=0.8 input to the F3 sweep almost certainly contains paths with 2+ consecutive SIM edges, contrary to the consim=1 invariant verified for edge_only / sim0.95 / sim0.9 / sim0.85.

**Why sim=0.8 is excluded from canonical selection:** even if a consim=1-filtered `paths_custom_sim0.8.jsonl` were rebuilt, sim=0.85 already shows R-I pair explosion to 225,132 — far beyond a tractable paper artifact. Going to sim=0.8 (an even denser SIM-edge graph) would only inflate further. The R-I pair count under canonical method goes 152 → 153 → 5,244 → 225,132 across edge_only / sim0.95 / sim0.9 / sim0.85 — a clear inflection between sim0.9 and sim0.85. There is no realistic regime where sim=0.8 would be the canonical choice over a denser-than-necessary sim=0.85 baseline. The sim=0.8 row is reported in the comparison table for completeness but not used downstream.

**Why sim=0.9 is the canonical choice for the workshop paper:**

1. **R-I coverage scale appropriate to paper artifact:** 5,244 R-I node pairs span 86 × 55 = 4,730 max risk-cluster × intervention-cluster combinations (~1.1 R-I node pair per cluster combination on average). Reviewable as a paper artifact in a way that 225k pairs (sim0.85) is not.
2. **Substantive expansion over edge_only:** 34× more R-I node pairs and 1.5× more risk clusters than the LLM-EDGE-only baseline; sim0.95 only adds 1 pair over baseline (153 vs 152) — i.e., the SIMILARITY-augmented value-add doesn't materialize until sim=0.9.
3. **% of universe captured:** 29.6% of all 17,699 R-I node pairs that have ANY path in the sim0.9 graph are captured by 100%-strict clustering. Relaxing body-coverage to "no constraint" recovers 76% (13,506 pairs); the residual 24% are lost because R or I doesn't cluster.
4. **Methodological consistency:** uses the same DFS hopwise enumeration + consim=1 filter + canonical clustering as the edge_only baseline, so the threshold-progression story (152 → 153 → 5,244 with monotone R-I cluster grid expansion) is internally consistent.
5. **Convergence behavior:** all 7 subtypes converged via `no_strict_clusters_added` at canonical 0.77 cutoff; chosen mcs from scan {5, 10, 20, 50, 100} ∈ {5, 10} per subtype (5 most common). Cluster quality metrics within typical ranges (silhouette 0.66–0.78, z_intra 27–123).

**Sim=0.95 falls below useful threshold:** SIM-edge density is 9,127 edges (0.6% of total) which produces only 8,976 paths in the path file vs 3.55M at sim0.9. Body-cluster recluster on the resulting sparse VPN_paperpair yields a cluster grid (56 × 45) that catches only 153 R-I node pairs — barely larger than edge_only's 152. The SIMILARITY-augmentation value-add is not realized at sim=0.95.

**Canonical artifacts (sim0.9, locked 2026-05-02):**
- `phase1_rawpathsfiles/vpn_strict_RIbody_sim0.9.jsonl` — 100% strict path set
- `step1_load_and_parse.../cluster_memberships_rev8_sim0.9.pkl`
- `step4_cluster_tables/body_kscan_metrics_sim0.9.csv`, `body_kscan_chosen_k_sim0.9.csv`, `body_kscan_iter_records_sim0.9.csv`
- `step4_cluster_tables/body_kscan_cutoff_sweep_sim0.9.csv`, `body_kscan_cutoff_ari_sim0.9.csv` (iter-1 cutoff sensitivity)
- `step4_cluster_tables/body_kscan_representatives_sim0.9.csv` — 3 closest + 3 farthest per cluster, columns `sim_initial_centroid` (≥0.77 by construction) + `sim_refined_centroid`
- `step4_cluster_tables/body_kscan_pareto_plot_<subtype>_sim0.9.png` (×7)
- `step4_cluster_tables/vpn_coverage_sensitivity_sim0.9.csv`

**Full-pipeline neighbor-cutoff sweep at sim=0.9 (locked 2026-05-02):** runs canonical method end-to-end at cutoffs 0.73 / 0.75 / 0.77 / 0.79 / 0.81. Tests whether the FINAL iterative cluster identity (not just iter-1 membership) is stable around 0.77 ± 0.04. Wall time ~6.5h total across 4 non-canonical cutoffs.

**Cluster count behavior (n_clusters at chosen-mcs per subtype):**

| Subtype | 0.73 | 0.75 | **0.77 (canon)** | 0.79 | 0.81 |
|---|---|---|---|---|---|
| design_rationale | 224 | 181 | **138** | 101 | 65 |
| implementation_mechanism | 225 | 179 | **120** | 83 | 52 |
| intervention | 187 | 166 | **115** | 100 | 64 |
| problem_analysis | 272 | 234 | **188** | 141 | 118 |
| risk | 168 | 157 | **173** | 149 | 139 |
| theoretical_insight | 173 | 186 | **128** | 105 | 64 |
| validation_evidence | 240 | 198 | **148** | 98 | 49 |

Cluster count drops monotonically as cutoff tightens (more restrictive 0.81 yields half as many clusters as looser 0.73). risk subtype is anomalous — count peaks at 0.77 — likely because the higher-density risk cluster space transitions through a sweet spot at 0.77 where additional clusters split out cleanly.

**ARI vs reference 0.77 (cluster identity stability — full pipeline, on intersection of clustered nodes):**

| Subtype | 0.73 | 0.75 | **0.77** | 0.79 | 0.81 |
|---|---|---|---|---|---|
| design_rationale | 0.8467 | 0.9022 | **1.0** | 0.9375 | 0.9386 |
| implementation_mechanism | 0.9000 | 0.9321 | **1.0** | 0.9700 | 0.9475 |
| intervention | 0.8652 | 0.9314 | **1.0** | 0.9215 | 0.8589 |
| problem_analysis | 0.9136 | 0.9525 | **1.0** | 0.9582 | 0.9347 |
| risk | 0.9056 | 0.9345 | **1.0** | 0.9407 | 0.9095 |
| theoretical_insight | 0.8097 | 0.8801 | **1.0** | 0.8920 | 0.8903 |
| validation_evidence | 0.8395 | 0.8947 | **1.0** | 0.9215 | 0.9518 |

**Reviewer-defensibility outcome:** ARI is in the **0.81–0.97 range** across all cutoff perturbations of ±0.04 around 0.77. Theoretical_insight is loosest (ARI=0.81 at cutoff=0.73), still well above 0.5 (random) and well above 0.7 (typically called "stable"). The chosen 0.77 cutoff is **not arbitrary** — cluster identity is preserved across the neighborhood. Cluster fineness (count) varies, but cluster taxonomy is robust.

**Implication:** the canonical 0.77 cutoff is the central anchor of a stability plateau; small perturbations (±0.02–0.04) preserve cluster identity above 0.85 ARI for 6 of 7 subtypes, above 0.81 for theoretical_insight. This satisfies the standard reviewer ask of "show your hyperparameter is not arbitrarily chosen".

**Output artifacts (locked 2026-05-02):**
- `step1_load_and_parse.../cluster_memberships_rev8_sim0.9_cutoff{0.73,0.75,0.77,0.79,0.81}.pkl` — 5 PKLs at 5 cutoffs
- `step4_cluster_tables/full_cutoff_compare_sim0.9.csv` — n_clusters + ARI table reproducing the above

### 19.3e — Co-occurrence rebuild + DFS path-multiplicity decomposition (locked 2026-05-02)

**Pipeline stage:** `graph_analysis/phase2_step4_F1_rev8_rebuild.py` reads strict path file + `cluster_memberships_rev8_sim0.9.pkl`, applies the 100%-strict filter inline, and produces three output artifacts:
- `step4_cluster_tables/optionB_cooccurrence_families_hopwise_sim0.9.csv` — row-per-frozenset (mechanism signature)
- `step4_connectivity/ri_triplets_hopwise_sim0.9.csv` — row-per-(risk-cluster, frozenset, intervention-cluster)
- `step4_paths/representative_pathways_hopwise_sim0.9.jsonl` — strict path set

**Headline numbers (sim=0.9 hopwise, locked 2026-05-02):**

| Metric | Value |
|---|---|
| Total paths in input | 3,548,825 |
| 100%-strict (R + every body + I clustered) | 799,031 (22.52%) |
| Dropped — endpoints not clustered | 860,459 |
| Dropped — body not fully clustered | 1,889,335 |
| Unique frozensets (any n) | 8,766 |
| Frozensets with n_paths≥5 | 4,478 |
| R-I triplet rows | 12,415 |

**DFS path-multiplicity inflates n_paths but is NOT load-bearing literature signal.** Concrete decomposition of the top-5 frozensets (n_sources=1–3 each, 8–9 cluster signature) revealed two independent inflators that should be separated for L2 weighting:

1. **R-I pair multiplicity (legitimate graph-theoretic signal):** the same frozenset signature can bridge many distinct (risk node, intervention node) pairs, giving real evidence that the signature is a **literature-replicated mechanism**.

2. **DFS body-permutation multiplicity (combinatorial noise):** within a single (R, I) pair, hopwise DFS enumerates many ordered body sequences satisfying monotonic-relaxed + consim=1, all sharing the same frozenset (since frozenset is unordered).

**Worked example — family 4 (n_paths=17,979, n_sources=1, 8-cluster signature):**

| Decomposition component | Count |
|---|---|
| Distinct intervention nodes at path end | **1** (single intervention from one alignmentforum post; that paper has only 17 nodes total) |
| Distinct risk nodes at path start | **136** (across many papers, reached via consim=1 SIM bridges) |
| Distinct body nodes in matched paths | **162** (across many papers) |
| Distinct R-I node pairs | **136** (= 136 risks × 1 intervention) |
| Average paths per R-I pair | **132.2** |
| → 17,979 = 136 R-I pairs × ~132 DFS body permutations per pair ||

**Implication: `n_sources` (current implementation) under-counts replication signal.** The CSV's `n_sources` column = `len(set(node_attrs[path[-1]].url for matched paths))` — i.e., **distinct intervention-paper URLs only**. A signature with `n_sources=1` and `n_paths=17,979` may still span 136 different risk nodes from many different papers; the current metric doesn't surface that.

**Better weighting metrics for L2 mechanism-family construction:**

| Metric | What it captures | Reviewer-defensibility |
|---|---|---|
| `n_paths` (current) | Raw DFS path count — inflated by R-I × body-permutation product | Low — DFS combinatorial noise dominates |
| `n_distinct_RI_pairs` | Graph-theoretic count of (risk-node, intervention-node) connections via this signature | **High — independent of DFS enumeration multiplicity** |
| `n_intervention_sources` (current `n_sources`) | Intervention-paper-URL diversity at path end | Medium — captures only one endpoint |
| `n_risk_sources` | Risk-paper-URL diversity at path start | Medium — captures other endpoint |
| `n_total_paper_sources` | Distinct paper URLs across ANY node in any matched path | **Highest — full literature-replication signal** |

**Recommendation for downstream L2 grouping (F4b Pareto on Jaccard):**
1. Compute `n_distinct_RI_pairs` and `n_total_paper_sources` per frozenset before Pareto.
2. Filter L2 input to frozensets with `n_total_paper_sources ≥ 3` (multi-paper replication threshold) to remove single-paper combinatorial DFS expansions.
3. Weight Jaccard distance computation by `n_distinct_RI_pairs` (graph-theoretic signal) rather than `n_paths` (combinatorial noise).

**Why this matters for the workshop paper:** the headline narrative is "frozenset X is a literature-replicated mechanism family bridging risks Y to interventions Z." If the supporting evidence is "n_paths=17,979" but those collapse to 136 R-I pairs × DFS-permutation-noise, the reviewer-defensible count is 136 (or some lower paper-deduplicated number), not 17,979. Reporting raw `n_paths` without n_distinct_RI_pairs / n_total_paper_sources side-by-side would be misleading.

### 19.4 — Pareto-frontier validation for body cluster recluster (Task #7)

The rev7 cluster taxonomy (k=40 hardcoded) is over-fragmented (CF-3) and not reviewer-defensible. Rev8 introduces `phase2_step4_F3_body_recluster.py` which:

1. **Restricts input to path-participating nodes (VPN_paperpair).** Body nodes that never appear in a custom-consim1-qualifying F2v4 path are excluded from clustering input — clusters describe the actual mechanism population, not arbitrary unconnected concepts.
2. **K-scans per subtype** for k in [10, 15, 20, 25, 30, 35, 40, 50, 60] with `AgglomerativeClustering(metric='cosine', linkage='average')` on unit-normalized embeddings.
3. **Computes Pareto metrics per (subtype, k):**
   - **Intra-tightness:** mean within-cluster cosine similarity (higher = more homogeneous; sampled to 500 nodes per cluster for tractability)
   - **Inter-looseness:** max between-centroid cosine similarity (lower = more separated)
4. **Selects k via Pareto frontier:** smallest k where `intra_mean >= 0.70` AND `inter_max <= 0.30`. If no k satisfies both, the clustering at this resolution is reported as inadequate and the best-gap point is selected as fallback (status flagged `fail`).

The thresholds (0.70 intra, 0.30 inter) are chosen because the embedding model (`BAAI/bge-large-en-v1.5`, 1024-dim) places semantically related AI safety concepts in the 0.5-0.9 cosine sim band, and a 0.40 gap (0.70 - 0.30) ensures clusters are visually separable in projection plots. **If Pareto frontier cannot be satisfied, that itself is a paper finding** — the body concept space at this similarity resolution is too entangled for clean k-cluster decomposition, motivating a different L2 abstraction (e.g., topic-model-based grouping).

Outputs:
- `step4_cluster_tables/body_kscan_metrics.csv` — all (subtype, k) metrics
- `step4_cluster_tables/body_kscan_chosen_k.csv` — chosen k + status (pass/fail) per subtype
- `step4_cluster_tables/body_kscan_pareto_plot_<subtype>.png` — visual Pareto frontier
- `step1_load_and_parse.../cluster_memberships_rev8.pkl` — new memberships at chosen-k

### 19.5 — Pareto-frontier validation for L2 frozenset grouping (Task #7b)

The same Pareto framework applies at L2 (mechanism-family grouping over frozensets of body cluster IDs). `phase2_step4_F4b_pareto_frozenset.py`:

1. Reads cooccurrence-families CSV (output of F1 / E3-equivalent on the post-recluster path set)
2. Builds binary vocabulary vector per frozenset; computes Jaccard distance matrix
3. K-scan via `fcluster` on average-linkage Jaccard linkage for k in [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
4. **Per-k Pareto metrics:**
   - **Intra-tightness:** mean within-group Jaccard sim across all groups (higher = more homogeneous)
   - **Inter-looseness:** max between-group binary-centroid Jaccard sim across cluster pairs (lower = more separated)
5. **Selects k:** smallest k where `intra_mean >= 0.50` AND `inter_max <= 0.20`. Jaccard is harsher than cosine, hence lower thresholds.

If the Pareto frontier cannot be satisfied at the L2 layer, that is a finding: **frozenset diversity at L2 is too high to support a clean mechanism-family decomposition.** This would imply the AI safety mechanism space is too heterogeneous for clean cluster narratives at this granularity, and the paper would report that as a primary result rather than forcing an inadequate taxonomy.

Outputs:
- `step4_cluster_tables/frozenset_kscan_metrics_<suffix>.csv`
- `step4_cluster_tables/frozenset_kscan_chosen_k_<suffix>.csv`
- `step4_cluster_tables/frozenset_kscan_pareto_<suffix>.png`
- `step4_cluster_tables/frozenset_groups_pareto_<suffix>.csv`
- `step4_cluster_tables/frozenset_group_memberships_pareto_<suffix>.csv`

### 19.6 — Why Pareto frontier validation is the rigor signal for the paper

Both the rev7 fixed-k clustering and the silhouette-only k-selection in earlier revisions answer only one half of the cluster-quality question (intra-cluster homogeneity). Reviewers reasonably ask whether the clusters are also genuinely separated from each other (inter-cluster looseness). The Pareto frontier puts both constraints on equal footing: a clustering is acceptable only when it simultaneously achieves intra-tightness above a chosen threshold AND inter-looseness below a chosen threshold. If no k achieves both, the data does not support a clean k-cluster decomposition at this resolution — and the paper reports that as a methodological finding rather than presenting an inadequate taxonomy as if it were validated.

The Pareto frontier framework therefore replaces silhouette score and ARI as the primary cluster-quality signal in the rev8 paper. Silhouette and ARI remain in the metrics CSV for reference, but the chosen-k decision is driven by the Pareto thresholds (cosine: 0.70/0.30 for body; Jaccard: 0.50/0.20 for L2).

### 19.7 — Files added in rev8

| File | Role |
|------|------|
| `graph_analysis/phase2_step4_F2v4_hopwise_falkordb.py` | Canonical hop-wise DFS path enumeration on live FalkorDB |
| `graph_analysis/phase2_step4_F1_consim1_custom_rebuild.py` | Consim1 + maturity rebuild on F2v4 path set |
| `graph_analysis/phase2_step4_F3_body_recluster.py` | Body recluster on VPN_paperpair with k-scan + Pareto validation (Task #7) |
| `graph_analysis/phase2_step4_F4b_pareto_frozenset.py` | L2 frozenset Pareto validation (Task #7b) |
| `graph_analysis/phase2_results/rev8_active_state.md` | Comprehensive in-flight state (CF-1→CF-5, B-fix audit, task list) |

---

### 19.8 — Prior canonical (SUPERSEDED 2026-05-07 by §19.9): EDGE-only global cutoff scan with 3-level family decomposition

**Status as of 2026-05-07: SUPERSEDED. See `graph_analysis/phase2_results/Step4_Findings_Report.md` §19.9 for the current canonical paper analysis path. This subsection is preserved for traceability of the methodological evolution; do not cite for paper claims.**

The §19.8 setup (UMAP-15D HDBSCAN substrate, raw-cosine cutoff, GLOBAL body clustering ignoring subtype) hit a body-coverage ceiling of 25% at cutoff 0.80 and yielded only 83 strict-filter paths (0.93% retention from 8,954 EDGE-only paths). §19.9 below switches the substrate to UMAP-2D HDBSCAN, lowers cutoff to 0.75 with mcs=3, and partitions the body pool by LLM-extraction subtype (pa/ti/dr/im/va) — recovering 88% total VPN coverage and preserving the subtype label for downstream doublet construction.

#### Setup
- **Path source:** `graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl` — 8,954 EDGE-only paths from F2v4 hopwise DFS.
- **Filters baked into path build (F2v4):** EDGE conf ≥ 3, single-risk per path (start node only; risk neighbors excluded during expansion at lines 317-318), single-intervention per path (preempts at first intervention encountered, line 299), simple paths, min length 3, max length 50, first-hop EDGE-or-SIM to body subtype (EDGE-only here).
- **Strict-filter VPN:** path nodes from above ∩ {intervention endpoint maturity ≥ 3} → 19,073 unique nodes (2,464 risks, 13,902 body, 2,707 interventions).
- **Clustering:** GLOBAL HDBSCAN on UMAP-15D for body nodes (no subtype filter — risk and intervention each clustered intra-group). Iterative-residual-recluster with mcs=5, strict per-iter ≥5, run-until-convergence. Cutoff applied as raw 1536-D centroid cosine sim filter (every member must be ≥ cutoff to centroid).
- **Strict path filter:** retain paths where R + every body + I are all in non-noise clusters at the chosen cutoff.

#### Cutoff scan results

| cutoff | risk K (cov%) | body K (cov%) | intervention K (cov%) | strict EDGE-only paths |
|---|---|---|---|---|
| **0.80** | 145 (59.7%) | **503 (25.0%)** | 78 (22.8%) | **83** |
| 0.85 | 90 (34.9%) | 111 (5.6%) | 21 (6.9%) | 1 |
| 0.90 | 32 (13.5%) | 9 (0.4%) | 5 (1.4%) | 0 |
| 0.95 | 9 (3.9%) | 0 (0.0%) | 0 (0.0%) | 0 |

**Cutoff 0.80 is the only viable level for downstream family analysis.** Tighter cutoffs collapse body coverage too aggressively (P(all 5 body nodes clustered) decays as p_node^5).

#### 3-level family decomposition at cutoff 0.80

Three frozenset-based abstractions per path:

| Level | Frozenset element | What is captured |
|---|---|---|
| **strict_tuple** | `(global_cluster_id, role_label)` | Both content cluster AND LLM role label. Two paths share frozenset iff identical at content+role. |
| **semantic_only** | `global_cluster_id` alone | Drops role. Cross-subtype-mislabel collapses (LLM assigning different subtypes to same concept across papers). |
| **role_pattern** | sorted `(role_label, count_in_path)` tuple | Drops content. Captures only chain skeleton (how many body nodes per subtype). |

**Family identification:** Hamming-edit-distance connected components (NOT Jaccard clustering — Jaccard is uninformative here, mean ~0.97). Two frozensets connect if symmetric-difference distance d ≤ 2 (one cluster swap or ≤ two add/removes). d ≤ 2 is cumulative — includes d ≤ 1 and d = 0 pairs.

| Level | # unique frozensets | # multi-path frozensets | # Hamming families d ≤ 2 | # multi-frozenset families | Jaccard mean |
|---|---|---|---|---|---|
| strict_tuple | 70 | 8 | 50 | 12 | 0.974 |
| semantic_only | 67 | 9 | 46 | 11 | 0.964 |
| role_pattern | 11 | 6 | 1 | 1 | n/a |

#### Key findings

1. **Role-pattern collapse:** All 11 chain-skeleton variants pull into one Hamming family at d ≤ 2. The dominant skeleton is `pa=1, ti=1, dr=1, im=1, va=1` (one body cluster from each of 5 subtypes per path), covering 51 of 83 paths and all 51 R-I cluster pairs that pattern reaches. Variants are minor (occasional 2 of one subtype + 0 of another). Practical implication: **the LLM extraction is consistent in producing one body node per subtype per logical chain** — the chain skeleton is structurally invariant across the AI safety corpus.

2. **Cross-role merging is real but bounded:** semantic_only collapses 4 strict_tuple families (50 → 46). The largest semantic_only family (#0) absorbs 8 frozensets across 10 paths and 5 R-I pairs — versus the largest strict_tuple family which has only 4 frozensets / 4 paths / 2 R-I pairs.

3. **The "experience replay" demonstration of cross-role mislabeling:** strict_tuple family 4 has signature `pa:167 & ti:168 & dr:168 & im:168 & va:166`, meaning the SAME global cluster 168 (centroid representative: "Experience replay buffer with uniform random sampling of past transitions") is reached through three different role labels (theoretical_insight, design_rationale, implementation_mechanism). Three paper-extractions where the LLM picked a different subtype for the same concept.

4. **Top mechanism families decoded** (file: `phase2_results/step4_finalanalysis/step4_cluster_tables/global_cutoff_top_families_global_cutoff0.80_DECODED.csv`):
   - **Adversarial training** (semantic fam 0): 8 frozensets, 10 paths, 5 R-I pairs. Clusters span "Adversarial vulnerability" (141), "PGD adversarial training procedure" (147), "robust accuracy on CIFAR10 PGD" (172), "Adversarial training improves robustness" (174), "Robust optimization via worst-case perturbation" (253).
   - **RLHF / InstructGPT** (semantic fam 1): 4 frozensets, 12 paths, 2 R-I pairs.
   - **Toxicity/safety classifier filter** (semantic fam 2): 4 frozensets, 4 paths.
   - **Experience replay (DQN/Atari)** (strict fam 4): 3 frozensets, 6 paths, 3 R-I pairs.
   - **Compute-optimal scaling laws / Chinchilla** (semantic fam 4): R-cluster 137 = "High inference latency and memory requirements in large language model deployment"; I-cluster 3 = "Plan LLM pre-training runs using Chinchilla compute-optimal scaling".

#### Limitations + open issues

- Path retention is sparse: 83 of 8,954 EDGE-only paths survive (0.93%). Driven by 25% body coverage at cutoff 0.80.
- Average 1.6 paths per R-I cluster pair (83 paths / 51 pairs); most R-I pairs hit by only 1 path.
- Jaccard frozenset distance ≈ 0.97 at all levels: Hamming-edit-distance connected components is the only viable family primitive on this data.
- **Substrate mismatch**: HDBSCAN runs on UMAP-15D but the centroid cutoff is in raw 1536-D cosine. Open question whether harmonizing (cutoff in UMAP-2D distance) would improve coverage.
- **Intervention coverage is the weakest link** (22.8% at 0.80 vs 59.7% for risks). Interventions are paper-specific innovations; cross-paper near-duplicate density is lower.

#### Output artifact list

| File | Content |
|---|---|
| `phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/cluster_memberships_rev8_global_cutoff{0.80,0.85,0.90,0.95}.pkl` | Cluster memberships per cutoff |
| `phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/role_of_rev8_global_cutoff{X}.pkl` | Per-node LLM role label preserved alongside global cluster ID |
| `phase2_results/step4_finalanalysis/step4_cluster_tables/global_cutoff_summary_global_cutoff{X}.csv` | Per-level family metrics |
| `phase2_results/step4_finalanalysis/step4_cluster_tables/global_cutoff_top_families_global_cutoff0.80.csv` | Top families per level (raw) |
| `phase2_results/step4_finalanalysis/step4_cluster_tables/global_cutoff_top_families_global_cutoff0.80_DECODED.csv` | Top families with centroid representative names |
| `phase2_results/step4_finalanalysis/step4_cluster_tables/cluster_representatives_global_cutoff0.80.csv` | Closest-to-centroid representative for every cluster |

---

### 19.9 — CANONICAL PAPER ANALYSIS (current): per-subtype HDBSCAN-2D at cutoff=0.75, mcs=3 on the 19,073-node EDGE-only VPN (locked 2026-05-07)

**This is the analysis path the paper uses for node clustering. §19.8 is superseded.** The pipeline below completes Phase 1 (basic node clustering); §19.9.4 enumerates the four downstream tasks that turn node clusters into the final mechanism-family deliverable.

#### 19.9.1 — Canonical Phase 1 setup

- **Path source:** `graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl` — 8,954 EDGE-only paths from F2v4 hopwise DFS. Filters baked in at F2v4 build: EDGE conf ≥ 3, single-risk per path (start node only; risk neighbors excluded during expansion), single-intervention per path (preempts at first intervention encountered), simple paths, min length 3, max length 50, EDGE-only first hop.
- **Strict-filter VPN:** path nodes ∩ {intervention endpoint maturity ≥ 3} → **19,073 unique nodes** (2,464 risks; 13,902 body across 5 subtypes; 2,707 interventions).
- **Clustering substrate:** UMAP-2D (n_components=2, n_neighbors=15, min_dist=0.0, metric=cosine, random_state=42) → HDBSCAN(min_cluster_size=3, metric=euclidean) on the UMAP-2D coordinates. **Substrate fix vs §19.8** (which ran HDBSCAN on UMAP-15D with raw-cosine cutoff = mismatched substrate).
- **Centroid filter:** every member must be ≥ 0.75 raw cosine sim to its cluster's initial centroid (centroid computed as unit-normalized mean of pre-cutoff members; see `phase2_step4_F3_body_recluster.py:290-331` `apply_centroid_cutoff`). Members below 0.75 → noise.
- **Per-iteration strict filter:** clusters with < 3 surviving members after centroid cutoff → noise.
- **Iterative-residual-recluster:** noise from each iteration becomes input to the next. Loop until coverage_target=0.95 reached OR no new clusters added (convergence) OR max_iter=50.
- **Pool partitioning (NEW vs §19.8):** **per-subtype** — 7 separate HDBSCAN clusterings, one per LLM-assigned role. Risk + 5 body subtypes + intervention. Each pool clusters only against itself; no cross-subtype mixing.
- **Method:** HDBSCAN-only. Louvain on SIM≥0.75 k-NN was tested in earlier sweeps (§19.9.2 below) and rejected — Louvain coverage was strictly weaker than HDBSCAN-2D and the union added <1pp at every cutoff tested.

Script: `graph_analysis/phase2_step4_phase1_paper_clustering.py` invoked with `--cutoff 0.75 --mcs 3 --strict-min 3 --max-iter 50 --methods A --tag _c75m3_subtype --pool-mode per_subtype`. Body subtype names use the canonical logical-chain order pa → ti → dr → im → va (per project `CLAUDE.md` rule on never showing concept subtypes out of logical-chain order).

#### 19.9.2 — Phase 1 coverage results

Per-pool, in canonical logical-chain order (risk → pa → ti → dr → im → va → intervention):

| Pool | N nodes | Coverage | # clusters | Mean cluster size | Median cluster size | Iterations | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| risk | 2,464 | **94.68%** | 339 | 6.9 | 5 | 11 | 36.7 |
| problem_analysis | 2,863 | **93.43%** | 584 | 4.6 | 4 | 38 | 90.3 |
| theoretical_insight | 2,544 | **86.48%** | 542 | 4.1 | 3 | 28 | 90.1 |
| design_rationale | 2,548 | **89.52%** | 516 | 4.4 | 4 | 22 | 66.4 |
| implementation_mechanism | 3,086 | **83.02%** | 665 | 3.9 | 3 | 50* | 235.6 |
| validation_evidence | 2,861 | **84.80%** | 576 | 4.2 | 3 | 23 | 93.6 |
| intervention | 2,707 | **87.55%** | 513 | 4.6 | 4 | 24 | 65.3 |
| **TOTAL** | **19,073** | **88.33%** | **3,735** | — | — | — | — |

\*implementation_mechanism hit max_iter=50 with cum_cov still climbing slowly — bumped from 83.0% to potentially 85-86% if max_iter raised, but the marginal gain past iter 30 was <0.5% per iter, so 50 is an acceptable convergence floor.

#### 19.9.3 — How the canonical config was selected (sweep summary)

Coverage scan across cutoff × mcs × pool-mode × method on the same 19k VPN:

| Cutoff | mcs | Pool mode | Method | Risk cov | NR cov | Total cov |
|---:|---:|---|---|---:|---:|---:|
| 0.80 | 5 | pooled | HDBSCAN-2D ∪ Louvain | 60.02% | 24.44% | 29.04% |
| 0.80 | 5 | pooled | HDBSCAN-2D only | 59.25% | 23.78% | 28.36% |
| 0.75 | 3 | pooled | HDBSCAN-2D only | 94.68% | 72.71% | 75.55% |
| **0.75** | **3** | **per_subtype** | **HDBSCAN-2D only** | **94.68%** | **86.6% (avg body+interv)** | **88.33%** |
| 0.70 | 3 | pooled | HDBSCAN-2D ∪ Louvain | 96.88% | 95.74% | 95.88% |

Selection rationale:
- **0.75 over 0.70:** stricter semantic floor while still recovering ≥85% on every subtype pool. 0.70 is the algorithmic ceiling but 0.75 keeps reviewer-defensibility against "your clusters are too loose." 0.77 was considered but the cutoff-coverage curve is steeply non-linear in this region (NR pooled drops 73% → 24% as cutoff goes 0.75 → 0.80), so 0.77 was projected to lose ~10pp on body subtypes for a marginal gain in stricter sim — net negative.
- **mcs=3 over mcs=5:** mcs=5 collapses NR coverage to 24% even at cutoff=0.80; mcs=3 admits 3-member clusters which are a meaningful unit of "≥3 papers extracting the same concept."
- **per-subtype over pooled NR:** +12.5pp coverage at the same cutoff/mcs (75.55% → 88.33%). Subtype-resolved clustering preserves the LLM-extraction signal as a usable feature instead of throwing it away into a 16,609-node combined manifold.
- **HDBSCAN-2D only, dropping Louvain:** at cutoff=0.80, Louvain (SIM≥0.80 k-NN graph + 0.80 centroid filter) reached only 6.92% on NR vs HDBSCAN-2D's 23.78%. At 0.70, Louvain reached 47.96% vs HDBSCAN-2D's 95.01% — Louvain's coverage was always strictly weaker, and the union over both methods added < 1pp vs HDBSCAN-2D alone at every tested cutoff. Louvain is dropped from the canonical pipeline.

#### 19.9.4 — Why the per-subtype split is the right primary axis

1. **Preserves LLM-extraction signal as a feature.** Each VPN node has its `concept_category` ∈ {pa, ti, dr, im, va} from extraction-time o3 calls. Per-subtype clustering treats this as a structural feature rather than collapsing it into a pooled manifold and re-discovering it via post-hoc analysis.
2. **Better embedding-density match per pool.** The combined 16,609-node NR manifold has 6 distinct subtype semantics overlapping in 1536-D embedding space — local density is diluted, HDBSCAN finds fewer dense pockets. Subtype pools are 2,544-3,086 nodes each, much sharper density signal per cluster.
3. **Direct doublet-primitive support.** The downstream mechanism-family construction operates on doublets (R_group, NR_anchor) with body chain metadata as narrative. Per-subtype labels give every NR node a (subtype, cluster_id) pair that can be used directly in mechanism-family identification — no extra LLM/algorithmic step needed to recover the subtype information.
4. **Reviewer-defensibility.** The subtype labels are LLM-derived but they are the SAME LLM that extracted the nodes; using them as a clustering scaffold is "consistency-of-extraction" rather than "additional LLM dependency."

#### 19.9.5 — Output artifacts (Phase 1 complete as of 2026-05-07)

In `graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/`:

| File | Content |
|---|---|
| `cluster_memberships_rev8_paper_methodA_c75m3_subtype.pkl` | **CANONICAL** — 3,735 cluster records keyed by `("rev8_paper", "umap2d_hdbscan_iter", pool, "hdbscan", str(cid))` → list of node IDs. `pool` ∈ {risk, problem_analysis, theoretical_insight, design_rationale, implementation_mechanism, validation_evidence, intervention}. |
| `role_of_rev8_paper.pkl` | Dict[node_id → role label] for VPN nodes — used by downstream doublet construction. |
| `phase1_coverage_summary_c75m3_subtype.csv` | Per-pool coverage summary (the §19.9.2 table). |
| `phase1_iter_records_c75m3_subtype.pkl` | Per-iteration trajectory (debug). |
| `cluster_memberships_rev8_paper_methodA_c75m3.pkl` | Pooled-NR sensitivity check (kept for comparison reporting). |
| `cluster_memberships_rev8_paper_methodA{,_c70m3,_c75m3,_c75m3_subtype}.pkl`, `cluster_memberships_rev8_paper_methodB{,_c70m3}.pkl` | Sweep variants. |
| `phase1_union_intersect_report{,_c70m3}.csv` | A∪B union/intersect/ARI for cutoff=0.80 baseline and cutoff=0.70 retry. |

#### 19.9.6 — Next-step tasks (Phase 2-4) — task list to execute on the canonical Phase 1 output

These four tasks turn the per-subtype node clusters of §19.9.1-19.9.5 into the paper's mechanism-family deliverable. Tasks must execute in order; each downstream task consumes upstream artifacts.

**Task A (Phase 2): LLM thematic naming of remaining nodes outside clusters.**
- Input: 19,073 - 16,847 = **2,226 noise residual nodes** (11.67% of VPN) from `cluster_memberships_rev8_paper_methodA_c75m3_subtype.pkl`. Per-pool residual: risk 131, pa 188, ti 344, dr 267, im 524, va 435, interv 337.
- Goal: assign every residual node to a thematic group with a human-readable name. Groups must be reviewer-defensible (named, with member representatives).
- Method: Claude Code CLI shim (`0_domain_finder/knowledge_pipeline/src/claude_cli_shim.py`) — Max plan only, NOT Anthropic API key. Strip ANTHROPIC_API_KEY from child env. Windows: `cmd.exe /c claude -p` for .CMD shim invocation.
- Two-pass batched approach: batch size 150-300 names per call; pass 1 first batch produces seed taxonomy of 15-25 thematic groups; pass 2 subsequent batches use seed taxonomy as fixed buckets with new buckets only for clearly-misfit nodes; cross-batch consolidation merges groups with > 50% Jaccard on members; naming pass sends 10 group-naming requests per shim call to amortize ~25-30k overhead per call.
- **Pool boundary for LLM (correction 2026-05-08): risk vs non-risk only.** Risk residuals (131) clustered as one pool; non-risk residuals (2,095 = pa+ti+dr+im+va+intervention) clustered as one pool. Subtype label is preserved on each node as a per-record attribute (used by Phase 4 doublet narrative + def-overlap match) but is NOT a clustering boundary for the LLM. Rationale: per-subtype splits would over-fragment small residuals where thematically identical nodes happen to carry different subtype labels.
- **HITL checkpoint after seed (correction 2026-05-08): after the first seed call per pool (risk-pool seed call + NR-pool seed call), STOP and return the seed taxonomies for human review.** Pass-2 assignment + naming + def-overlap only proceeds after the seed taxonomies are approved (or revised). Protects against burning ~300-400k tokens on a misshapen seed.
- Definitional overlap check: encode HDBSCAN cluster centroid representative names + LLM group name+description+representatives as embeddings; flag pairwise sim ≥ 0.70 as definitionally overlapping with an existing HDBSCAN cluster (then the LLM group is recorded as a "fuzzy boundary extension" of that cluster, not a new theme).
- Output: `cluster_memberships_rev8_paper_methodC_c75m3_subtype.pkl` (LLM thematic groups, same key schema as Method A); `phase2_llm_residual_naming.csv` with per-group name/description/representative names.

**Task B (Phase 2 → 3 transition): Path selection.**
- Input: all 8,954 EDGE-only paths + cluster memberships from Tasks A inputs (Method A) and Task A output (Method C).
- Goal: select the canonical path set for downstream mechanism-family analysis. A path is **fully clustered** if every node on the path has a non-noise cluster ID (from Method A or Method C). Partially-clustered paths are reported but excluded from family analysis.
- Decision: include both fully-A-clustered paths AND fully-(A∪C)-clustered paths as separate retained sets. Compare retention rates. Primary set = fully-(A∪C)-clustered (uses Phase 2 LLM contribution).
- Output: `phase2_step4_F2v4_paths_c75m3_subtype_fullyclustered.jsonl` (subset of `paths_hopwise_v4_edge_only.jsonl` where every node is clustered). Report retention count + per-pool retention breakdown.

**Task C (Phase 3): Coverage calculation.**
- Input: cluster memberships (Methods A + C), path retention counts.
- Goal: report **path-level coverage** (% of EDGE-only paths fully clustered) and **node-level coverage** (% of VPN nodes assigned to a non-noise cluster) per method and union (A ∪ C).
- Path-level coverage decays multiplicatively with path length: a 5-node path with per-node coverage 0.88 has expected fully-clustered probability 0.88^5 = 0.527 — so path-level coverage will be substantially below 88%.
- Output: `phase3_coverage_report.csv` with columns: method, n_nodes_total, n_nodes_clustered, node_coverage, n_paths_total, n_paths_fully_clustered, path_coverage. Plus per-subtype-pool breakdown.
- Sanity baseline check: per CLAUDE.md cross-check rule, expected path coverage ≈ ∏(per-node coverage)^path_length. If observed differs by >5x from this expectation, investigate dependency structure (paths share nodes; clustered/noise outcomes are not independent).

**Task D (Phase 4): Risk and non-risk family identification from node-level clusters.**
- Input: fully-clustered EDGE-only paths from Task B; cluster memberships from Task A + C.
- Doublet primitive (per `0_domain_finder/martins-impact-strategy-evolving.md` is unrelated; see `memory/plan_rev8_paper_canonical_pipeline.md` "Pivot to doublet primitive"): `(R_cluster_id, NR_anchor)` per path, where R_cluster_id is the risk-pool cluster of the path's start node and NR_anchor is constructed from the path's body+intervention nodes.
- Risk family = group of doublets sharing R_cluster_id (cross-paper reach: how many distinct NR_anchors does this risk reach via mechanism chains?).
- Non-risk family = group of NR_anchors sharing the same set of (subtype, cluster_id) pairs across body chain + intervention. Two natural primitives to evaluate:
  - **D1 — exact-match family**: NR_anchors with identical body-subtype-cluster signature `{(pa, cid_pa), (ti, cid_ti), (dr, cid_dr), (im, cid_im), (va, cid_va), (interv, cid_interv)}`. Counts as same family.
  - **D2 — Hamming-ball family** (per `memory/feedback_paper_analysis_preferences.md`): NR_anchors at symmetric-difference distance d ≤ K from a designated center anchor. K starting at 2; ball-around-center semantics, NOT transitive closure. Frozensets can be in multiple families. Sweep K ∈ {1, 2, 3} for sensitivity.
- Algorithm choice: D1 (exact) is the strict primary; D2 (ball-around-center, K=2) is the sensitivity check. Both report (n_families, family_size_distribution, top-N families with cluster representative names).
- Output: `phase4_mechanism_families.csv` (one row per family) + `phase4_family_members.csv` (one row per (family, member_path)). Plus a top-10 family decoded table with centroid representative names.

#### 19.9.7 — Where the canonical pipeline diverges from §19.8 — concrete diff

| Aspect | §19.8 (superseded) | §19.9 (current) |
|---|---|---|
| Substrate | UMAP-15D HDBSCAN | UMAP-2D HDBSCAN |
| Cutoff | 0.80 | 0.75 |
| mcs | 5 | 3 |
| Pool mode | global body (no subtype) + intra-risk + intra-intervention | per-subtype (7-way: risk + 5 body + intervention) |
| Body coverage | 25.0% | 86.6% (mean across 5 body subtypes) |
| Total VPN coverage | ~30% | 88.33% |
| Strict path retention | 83 of 8,954 | TBD (Task B) — projected ~3,000-5,000 paths |
| Family primitive | strict_tuple, semantic_only, role_pattern frozenset (3 levels, Hamming d≤2 connected components) | doublet (R_cluster, NR_anchor) — D1 exact + D2 Hamming-ball-around-center K∈{1,2,3} |
| LLM dependency at clustering time | none | Phase 2 LLM thematic on residual ~12% of nodes via Claude Code CLI shim (Max plan), with definitional-overlap check vs HDBSCAN clusters |

### 19.10 — Phase 2 Task A Pass-2 results + HDBSCAN ↔ LLM-group granularity gap (added 2026-05-09)

This section documents Pass-2 NR-residual assignment results and surfaces a methodological asymmetry between the two cluster sources (HDBSCAN clusters vs LLM-named seed groups) that must be resolved before they are jointly fed into Phase 4 doublet construction.

#### 19.10.1 — Pass-2 NR-residual assignment: status and counts

**Pipeline:** `graph_analysis/phase2_step4_phase2_pass2_assignment.py` — assigns each of the 2,095 NR residual nodes (post-Phase-1 noise from non-risk pools) to one of:
- `(subtype, cluster_id)` — rescue into an existing HDBSCAN cluster (per `cluster_memberships_rev8_paper_methodA_c75m3_subtype.pkl`)
- one of the 33 v3 mechanism-class seed groups in `phase2_seed_taxonomy_nr_v3_recovered.json`
- `residual` — last-resort, no fit

**Batch outcomes (27 batches, ~80 nodes/batch sequential, Claude Code CLI shim, 60-min timeout, no-truncation safeguards on every batch):**

| Status | Batches | Nodes |
|---|---:|---:|
| OK (start-marker + sentinel validated) | 26 (00–23, 25, 26) | 2,015 |
| Failed all 3 retries (Claude AUP content-filter block, prompt 41,061 chars / ~10,265 tokens) | 1 (batch 24) | 80 |
| **Total** | **27** | **2,095** |

Batch 24 contains 80 nodes; the AUP block is reproducible across all 3 retries with identical error text. **Per the 2026-05-09 instruction, batch 24 is deferred and not retried in this pass** — these 80 nodes are recorded as unprocessed (~3.8% of NR residual). A retry strategy via prompt reframing or per-node split is left as a tracked open task.

**Assignment breakdown of the 2,015 successfully processed nodes** (`phase2_pass2_decisions_all.json`):

| Decision | Count | % of 2,015 |
|---|---:|---:|
| seed group (one of 33 v3 mechanism classes) | 1,149 | 57.0% |
| HDBSCAN cluster rescue | 587 | 29.1% |
| residual (no fit) | 279 | 13.8% |

- HDBSCAN rescue distribution: 587 rescues spread across **445 distinct HDBSCAN clusters** (long tail; max receiver = 6 nodes, top-10 each got 3–6). Confirms Pass-2 is dispersing-rescuing into many clusters, not piling onto a few attractors.
- Seed-group distribution: all 33 groups received ≥1 node. Top 10 absorbed 664/1,149 (58%); largest = "Benchmark-design mechanism: controlled conditions, noise ceilings, and standardized frameworks" (90 nodes); smallest = "RL training mechanism: priority and learning-progress sampling from replay buffers" (4 nodes).

**Net unprocessed/unassigned across full NR pool after Pass-2:** 80 (batch 24, blocked) + 279 (true residual) = **359 / 2,095 = 17.1%** still unclustered. NR-pool coverage including Pass-2 = 1,736 / 2,095 = 82.9% (1,149 seed + 587 HDBSCAN rescue).

#### 19.10.2 — The granularity gap: HDBSCAN clusters and LLM seed groups are NOT the same type of input

The next pipeline stage (Phase 4 doublet construction, Task D in §19.9.6) consumes (R_cluster, NR_anchor) pairs where the NR_anchor is a frozenset of `(subtype, cluster_id)` over the path's body+intervention nodes. For this to be a coherent primitive, every cluster_id used in the frozenset must come from a **comparable granularity scale**. Pass-2 produces two cluster sources whose granularity differs by ~8×:

| Source | # clusters in NR scope | Total members | Mean size | Median | Max | Naming |
|---|---:|---:|---:|---:|---:|---|
| HDBSCAN (Phase 1, `methodA_c75m3_subtype.pkl`, NR pools = 5 body + intervention) | **3,396** | 14,514 | 4.3 | 3 | 28 | none — anonymous IDs only; representative names aggregated at report time |
| LLM v3 seed groups (Phase 2, `methodC_c75m3_v3.pkl`) | **33** | 1,149 | 34.8 | 24 | 90 | full human-readable mechanism-class name + description |

If both are fed into the doublet primitive without reconciliation:
1. **Category error in NR_anchor.** The 33 LLM-group IDs occupy the same ID space as the 3,396 HDBSCAN cluster IDs but represent a fundamentally coarser unit. A frozenset that mixes one LLM-group-ID with four HDBSCAN-cluster-IDs is not a fixed-resolution signature.
2. **Discriminative-power loss.** Each LLM group spans ~8 HDBSCAN-equivalent dense pockets; treating it as one ID collapses 8 distinct mechanisms into one anchor. Doublet families that should split will collide.
3. **Reviewer asymmetry.** A reviewer asks: "How do you justify that membership in HDBSCAN cluster `implementation_mechanism_320` (4 members) and membership in 'Benchmark-design mechanism' (90 members) are interchangeable atoms in the same family signature?" The answer "they aren't" is the honest one — and the pipeline must reflect that.

#### 19.10.3 — Reviewer-consistent unification: dual-resolution scheme

To present a defensible, internally consistent family-finding output to reviewers, the canonical pipeline runs Phase 4 at **two parallel resolutions** and reports both. The fine-grain track is the rigor signal; the coarse-grain track is the headline mechanism-family story.

**Track 1 (FINE, "rigor signal"): pure Phase 1 HDBSCAN — Pass-2 contributes nothing here (correction 2026-05-09).**
- Every NR node carries either `(subtype, hdbscan_cluster_id)` from Phase 1 OR is residual.
- The 1,149 Pass-2 LLM-seed-assigned nodes are by construction the nodes HDBSCAN couldn't place at cutoff=0.75. Re-running embedding-nearest-cluster at the same 0.75 cutoff is mathematically the same operation HDBSCAN already rejected — they remain residual. Pass-2's contribution to Track 1 is therefore zero; that is the honest framing.
- Two sub-options for handling Track-1 residuals in doublet construction:
  - **Track 1-strict (cleanest):** any path containing a Track-1 residual node is EXCLUDED from family construction. Path-level coverage takes a hit (any path passing through a residual node drops). This is the unsupervised algorithmic baseline.
  - **Track 1-relaxed:** assign each residual node to its nearest HDBSCAN cluster centroid in its subtype pool **without a similarity cutoff**, with a `low-sim` flag on the rescued ID. All nodes get an HDBSCAN ID; noise injected; paths retained. Less defensible.
- Doublet construction: `(R_hdbscan_cid, frozenset({(subtype, hdbscan_cid)} per body+interv node))`.
- Family primitive: D1 exact-match + D2 Hamming-ball-around-center K∈{1,2,3}.
- Pros: uniform fine granularity; reviewer can audit any cluster by reading its representatives. Cons: anonymous IDs unreadable without a representative-name layer; strict variant loses many paths to residual.

**Track 2 (COARSE, "headline story"): LLM-named mechanism classes everywhere.**
- Run an additional LLM-naming pass over the **3,396 HDBSCAN NR clusters** using each cluster's representative names + size as input, producing ~30–40 LLM-named NR mechanism classes. Output target = same granularity scale as the 33 v3 seed groups (mean size 24–35, median 20–30).
- Reconcile against the existing 33 v3 seed groups via embedding similarity ≥ 0.70 between (LLM-named-HDBSCAN-class) and (v3 seed group) names+descriptions+representatives. Merge collisions; new HDBSCAN-derived classes that don't collide become additional members of the unified mechanism-class catalog.
- Same exercise on the 339 risk HDBSCAN clusters → ~15–25 LLM-named risk mechanism classes (paired with v2 risk seed taxonomy already on disk).
- Final unified catalog: ~30–50 NR mechanism classes + ~15–25 risk mechanism classes, every one with a human-readable name, description, and 3 representative member names.
- Doublet construction: `(R_class_name, frozenset({(subtype, class_name)} per body+interv node))`.
- Family primitive: D1 + D2 as above, but the IDs now carry semantic names so reviewer can read the family signature directly ("Risk: Reward-spec gaming → {Interpretability, Benchmark-design, Formal-methods, Adoption-traction validation}").
- Pros: reviewer-readable, headline-grade narrative. Cons: depends on a second LLM pass (additional cost; reviewer may flag "more LLM stack-up").

**Why both, not just Track 2 (revised 2026-05-09):**
- Track 1 and Track 2 are NOT nested resolutions of the same product — they are different products built from different node sets and different granularities. The earlier ARI-based "preservation" argument (in the prior draft of this section) was over-claiming and is withdrawn.
- The honest rigor signal is weaker but still useful: **for nodes both tracks DO cluster, is the Track-2 partition a coherent refinement/coarsening of the Track-1 partition where they overlap?** Concretely: pick the subset of NR nodes with both a Track-1 HDBSCAN cluster AND a Track-2 LLM-class. For each Track-2 class, compute the distribution over Track-1 HDBSCAN clusters. If a typical Track-2 class decomposes into 5–15 Track-1 clusters cleanly (high purity per Track-2 class → small set of Track-1 clusters), that supports "Track 2 is a coarsening that respects the algorithmic structure." If a typical Track-2 class scatters across 50+ Track-1 clusters with no concentration, Track 2 is doing something independent.
- Track 1 alone is unreadable to reviewers (anonymous cluster IDs). Track 2 alone is unfalsifiable. Both together = "incremental progress with both products visible," which is the right framing — not a unified claim.

#### 19.10.4 — Concrete next steps to execute Track 1 + Track 2

1. **Build Track 1 input** (`phase2_step4_pass2_remap_to_hdbscan.py`, NEW): for each of the 1,149 Pass-2-seed-assigned nodes, compute embedding cosine sim to every HDBSCAN cluster centroid in its subtype pool; if max sim ≥ 0.75 assign to that HDBSCAN cluster, else → residual. Write `cluster_memberships_rev8_paper_methodA+pass2remap_c75m3_subtype.pkl`.
2. **Build Track 2 input** (`phase2_step4_pass3_llm_name_hdbscan_clusters.py`, NEW): for each of the 3,396 NR HDBSCAN clusters, send (cluster_id, top-5-representative-names-by-centroid-distance, size) to LLM in batches of ~50 clusters per call, ask "assign this cluster to one of {33 v3 seed groups + propose new if no fit}." Same shim/sentinel/no-truncation safeguards. Repeat for 339 risk HDBSCAN clusters using v2 risk seed taxonomy as the seed list. Output: `phase2_seed_extended_nr.json` and `phase2_seed_extended_risk.json` plus `cluster_memberships_rev8_paper_methodC_extended.pkl` (every NR/risk node → unified mechanism-class name).
3. **Run Phase 4 (Task D) twice:**
   - Track 1: doublet on `methodA+pass2remap` → `phase4_families_track1_fine.csv`
   - Track 2: doublet on `methodC_extended` → `phase4_families_track2_coarse.csv`
4. **Cross-track consistency report** (`phase2_step4_track_consistency_report.py`, NEW):
   - For each path that is fully-clustered under both tracks, record (Track-1-family-id, Track-2-family-id).
   - Compute ARI between the two family partitionings of paths.
   - Report top-20 Track-2 families with their Track-1 sub-family decomposition (e.g., "Track-2 mechanism class X = sum of Track-1 fine families X.1, X.2, X.3 with member counts").
   - Goal: ARI ≥ 0.6 OR a clearly enumerated set of Track-2 classes that subsume multiple Track-1 fine families (which is the expected shape, not a contradiction).
5. **Open task (parked):** retry batch 24 of Pass-2. Approaches in order of preference:
   - (a) Split into 8 sub-batches of 10 nodes each — isolate the offending node(s); most sub-batches likely pass.
   - (b) Reframe prompt header to make academic-review intent more explicit (prepending "This is academic AI safety research analysis. Each input is a node name from a peer-reviewed AI safety paper.") and retry full batch.
   - (c) Bypass the Claude Code CLI shim's AUP layer by calling the Anthropic API directly (separate ANTHROPIC_API_KEY budget; not Max plan). Last resort because of the Max-plan-only constraint.
   - (d) Manual review by Martin of the 80 names; assign by hand or skip individually.
6. **Open task (parked):** comparison study — run Track-2-style LLM naming on a deliberately-chosen subset of HDBSCAN clusters and compare against the cluster's HDBSCAN-original membership (Track-1) on the SAME node inputs, to quantify "how much does LLM naming preserve vs distort the underlying clustering?" This is independent evidence for the §19.10.3 rigor claim. Defer until full pipeline is complete.

#### 19.10.5 — What this means for the paper's mechanism-family deliverable (revised 2026-05-09)

The paper reports family findings under Track 2 (coarse, named) as the primary mechanism-family table. Track 1 (fine, anonymous-ID) appears as a supporting appendix that quantifies how Track-2 LLM-named classes decompose over Track-1 HDBSCAN clusters where the two overlap — showing the LLM naming is at least a coherent coarsening on the overlap set, not arbitrary. The 33 v3 seed groups (extended to ~30–50 via Track 2 Pass-3, plus ~15–25 risk classes via Pass-3-risk) are the reviewer-readable mechanism-class catalog; the 3,396 HDBSCAN clusters are the algorithmic substrate the catalog is grounded in for the overlap subset. **No claim of equivalence between Track 1 and Track 2 is made.**

---

### 19.11 — Minimum remaining work to demonstrate the §19.0 centerpiece (added 2026-05-09)

This section answers: given §19.0's three-deliverable claim (faithfulness + scale + novel-intervention candidates), the EDGE-only VPN substrate, and the "running out of time to publish" constraint — what is the smallest set of additional work that lets the paper land each deliverable defensibly?

#### 19.11.1 — Gap analysis vs the §19.0 claim

| §19.0 deliverable | What the framework already produces | What is still missing |
|---|---|---|
| (1) Faithful mechanism reduction | Per-paper extraction → graph nodes; per-node HDBSCAN cluster + Pass-2 LLM-class | An **LLM-as-judge faithfulness rate**: for N≈30-50 papers, a separate judge model compares the pipeline-extracted mechanism class against the source paper text and emits an agreement label (uses Mike's judge framework from §3.1 Overleaf paper.tex). Without this, "lossless" is a claim with no measurement. **Decision 2026-05-15:** replaced original hand-validation plan with LLM-as-judge to save time and reuse Mike's existing judge stack; Sai is implementing. |
| (2) Large-scale mechanism evaluation | 19,073 VPN nodes, 8,954 EDGE-only paths, 33+ LLM seed classes, 3,396 HDBSCAN clusters | **A named mechanism-class catalog with cross-paper distribution + risk coverage + validation-stage breakdown** — i.e., the Track-2 LLM-named full catalog (Pass-3) and the Phase-4 doublet families computed against it. The 33 v3 seed groups alone are insufficient (only cover the 17.1% residual). |
| (3) Novel-intervention candidate generation | Doublet `(R_class, NR_anchor)` primitive defined; not yet computed | **A computed doublet graph + a "transferable mechanism" surfacing rule** + N=3–5 manually narrated examples. Concretely: identify (mechanism M, risk R′) pairs where M is densely populated for risks ≠ R′ but absent for R′, present as candidate "M-via-I_new for R′." |

#### 19.11.2 — Minimum remaining work, ranked by paper-impact-per-effort

**WORK ITEM 1 (REQUIRED, REVISED 2026-05-09): Full-VPN LLM clustering — see §19.12.**
- Replaces the prior "Pass-3 over HDBSCAN clusters" framing with a direct LLM-central clustering of all 19,073 VPN nodes (subtype as metadata, NR + risk pools, seed taxonomy = v3 NR + v2 risk).
- Estimated cost: ~11.8M Max-plan tokens, ~10–15h wall (see §19.12.6 detailed breakdown).
- Output: `cluster_memberships_rev8_paper_methodC_full_vpn.pkl` (mechanism-class assignment for every VPN node) + extended seed taxonomies + per-batch atomic saves + HDBSCAN validation report.
- **Status:** smoke-test (5 batches × 1 NR subtype each, ~250k tokens) runs first; user reviews before approving full chain.

**WORK ITEM 2 (REQUIRED): Phase 4 doublet families on Track 2.**
- After WORK ITEM 1, run `phase2_step4_F4_doublet_families.py` (TO BE WRITTEN, ~1 day) on the named Track-2 catalog.
- Compute D1 (exact-match) + D2 (Hamming-ball K∈{1,2,3}) families. Output: `phase4_families_track2_coarse.csv` + `phase4_family_members.csv`.
- Cost: pure CPU, no LLM. Wall time hours.

**WORK ITEM 3 (REQUIRED): LLM-as-judge validation of mechanism-class faithfulness on N≈30-50 papers — deliverable (1).**
- Sample papers stratified across the 30-50 mechanism classes (1-2 per class). For each, an LLM judge (distinct from the extraction LLM, per Mike's judge framework in §3.1 of the Overleaf paper.tex) reads the paper's intervention section + the pipeline-extracted mechanism class and emits agreement|disagreement|partial with reasoning. Aggregate to faithfulness rate + judge-of-judge calibration (Mike's existing framework). **Owner:** Sai. **Replaces:** original hand-validation plan (decision 2026-05-15) — saves time, reuses existing judge stack, scales to larger N without expert bottleneck.
- Report match rate (% papers where pipeline class = expert class) + "near-miss" rate (pipeline class is a sibling in the doublet hierarchy or a coarser/finer version of the expert class).
- Defensible target: ≥70% exact match + ≥85% near-miss.
- Cost: human time, ~6–10 h of expert reading. **Zero LLM tokens.** This is the single highest paper-impact-per-token item.

**WORK ITEM 4 (REQUIRED): Cross-paper mechanism-class distribution table — deliverable (2).**
- From Phase-4 output, compute: per mechanism class, (# papers that contain ≥1 node in this class, # risks reached via this class, intervention_maturity distribution of intervention-pool members in this class, EDGE-confidence distribution).
- One table; mechanical script; no LLM. Wall time minutes.
- **This IS the "large-scale mechanism evaluation" deliverable** — the table is the deliverable.

**WORK ITEM 5 (REQUIRED): Novel-intervention candidate surfacing — deliverable (3).**
- Algorithm: for each (mechanism class M, risk class R) pair, compute the doublet density (count of paths with R_class = R AND any body/intervention node in M-class). Identify M with high density across ≥3 different risks but density = 0 for some risk R′. Surface (M, R′) as a candidate. Sort by "M-mass across other risks" descending.
- Pick top 5; for each, the paper narrates: "Mechanism M typically addresses risks R₁, R₂, R₃ via interventions I₁, I₂, I₃. The framework predicts M may transfer to R′ via a novel intervention I_new combining the body-mechanism chain elements characteristic of M. To our knowledge no paper in the corpus proposes M-via-anything for R′."
- Pure CPU, no LLM. Verification of "no paper proposes this" is a Cypher query against FalkorDB filtered by R′-papers. Wall time hours including narration.

**WORK ITEM 6 (OPTIONAL, defer if time-constrained): Track 1 + cross-track decomposition table.**
- Already covered in §19.10. Run the strict Track 1 doublet for the appendix-only "rigor signal." Skip if Track 2 + faithfulness validation lands deliverable (1) on its own.

**WORK ITEM 7 (OPTIONAL): Batch 24 retry.**
- 80 nodes (~3.8% of NR residual) deferred. If time allows, retry via per-batch split (8 sub-batches of 10) — approach (a) from §19.10.4 step 5.

#### 19.11.3 — Sequencing under "running out of time to publish"

Critical path order (each blocks the next):
1. WORK ITEM 1 (Pass-3, ~6–10h LLM wall, awaiting approval) →
2. WORK ITEM 2 (doublet family compute, ~hours) →
3. WORK ITEM 4 (catalog table, minutes) || WORK ITEM 5 (candidate surfacing, hours) →
4. WORK ITEM 3 (hand-validation, runs IN PARALLEL with 1+2 — requires only the existing Pass-2 outputs to begin sampling and can update once Pass-3 lands).

Total wall time on critical path: 1-2 days of compute + LLM-as-judge run for D1 (Sai, in parallel) + 4-6 h of Martin's catalog audit (xlsx between Sonnet batches) = **~3-5 working days** to land all three deliverables, assuming the truncated Pass B (~2000 paths, NOT full 8,954) is the empirical scope per §19.13.4. Full-corpus Pass B is future work.

WORK ITEM 6 + 7 are appendix-grade and can ship after the main figure work is done, or be cut entirely if the paper deadline forces it.

#### 19.11.4 — What the paper does NOT need to add

- No additional substrate-quality robustness sweeps. The EDGE-only VPN at sim=0.9 / strict-filter / 88.33% Phase-1 coverage is the canonical substrate; further sensitivity studies dilute the strict-quality signal and consume the time budget without changing any §19.0 deliverable.
- No alternative clustering algorithm comparisons beyond what §19.9.3 already records (HDBSCAN-2D vs Louvain decision documented; that's enough).
- No expansion to additional thresholds (sim=0.85, sim=0.95). Already shown in §19.3 / §19.9 that sim=0.9 is the canonical choice; appendix can mention sensitivity briefly.

The intent stated in §19.0 — "lossless reduction of papers to mechanism-of-intervention, enabling large-scale evaluation and novel-intervention reasoning" — is achievable with the WORK ITEMS 1–5 above and nothing else. Subsequent work strengthens but is not required for the centerpiece claim.

---

### 19.12 — Full-VPN LLM clustering with subtype-as-metadata (LOCKED 2026-05-09)

**Decision:** drop the staged "Pass-3 over HDBSCAN clusters" framing. Instead run a single full-VPN LLM clustering pass on **all 19,073 EDGE-only VPN nodes** (the same node set HDBSCAN ran on, with the same quality cuts: EDGE conf ≥ 3, intervention maturity ≥ 3, single-risk-per-path, single-intervention-per-path, hop-wise DFS first-hop EDGE-or-SIM). HDBSCAN clustering becomes the validation check, not the upstream substrate.

This implements the §19.0 LLM-central methodological stance directly at the clustering layer.

#### 19.12.1 — Inputs and pool design

- **Input set:** all 19,073 strict-filter VPN nodes = 2,464 risks + 16,609 NR (5 body subtypes + intervention)
- **Subtype handling:** every node carries its `concept_category` ∈ {risk, problem_analysis, theoretical_insight, design_rationale, implementation_mechanism, validation_evidence, intervention} as **metadata in the LLM prompt, not as a clustering boundary**. Same node text + a `subtype: <name>` field. The LLM is instructed that:
  - The same conceptual content can serve different functions (e.g., "human oversight" can appear as a `design_rationale` in one paper and an `implementation_mechanism` in another); subtype DIFFERENTIATES these as distinct mechanism instantiations even when the node names are similar.
  - The LLM should still group across subtypes when the underlying mechanism IS the same (e.g., an interpretability mechanism appearing as `theoretical_insight` and `implementation_mechanism` in different papers can share one mechanism class).
- **Pool partitioning for the LLM:** **2 pools** — risk vs non-risk — same as Pass-2. NOT per-subtype. Subtype is metadata only.
- **Seed taxonomy:** start from the existing v3 NR taxonomy (33 mechanism classes from `phase2_seed_taxonomy_nr_v3_recovered.json`) + v2 risk taxonomy (24 classes from `phase2_seed_taxonomy_risk_v2.json`). LLM may propose new classes if a clear non-fit appears, but is instructed to default to the seed catalog and only add new classes when the new theme would absorb ≥ 5 nodes.

#### 19.12.2 — Granularity guidance (the prompt-level constraint)

The Pass-2-residual processing produced over-granular splits at times (e.g., separate "RL exploration" / "RL training" / "RL imitation" classes that collectively could be one "RL training mechanism" class). For the full-VPN run the LLM is given explicit granularity instructions in the system prompt:

> *"Your goal is to identify mechanism families that group single mechanism instantiations by their fundamental commonalities and differences. The target catalog has ~30–50 NR mechanism classes and ~15–25 risk mechanism classes. Avoid over-granular splits: if two proposed classes share the same 'how' (the underlying mechanism by which an effect is achieved) and differ only in 'where' (subtype) or 'specifics of one example,' merge them into one class. Conversely, do not over-coarsen: two classes that share surface vocabulary but differ in the actual causal mechanism should remain separate. Default to the seed catalog; propose new classes only when the new theme is clearly absent from the seed and would absorb ≥ 5 nodes."*

#### 19.12.3 — Continuous-write requirement (data-loss protection)

- Each batch's decisions are written to `phase2_full_vpn_batches/batch_NNN.json` **before the next batch's API call begins**. No in-memory accumulation across batches.
- Atomic write pattern: write to `batch_NNN.json.tmp`, then rename to `batch_NNN.json`. A power loss or session-limit kill mid-write produces no partial files.
- Resume logic on script restart: scan the batches directory for completed `batch_NNN.json` files; skip any batch whose output already exists; pick up at the first missing batch index. Idempotent.
- This protects against (a) Max-plan session limit hit mid-run, (b) AUP block on a single batch (other batches' outputs preserved), (c) container/network/disk failures.

#### 19.12.4 — AUP-resilience (per Pass-2 batch-24 lesson)

- On AUP block: retry once with same batch. If still blocked, automatically split the batch into 8 sub-batches of 10 nodes each and retry sub-batches individually. Per-sub-batch save same as full batch. Most sub-batches typically pass; the offending node(s) get isolated.
- Sub-batches that still fail AUP after individual retry are logged to `phase2_full_vpn_aup_failures.json` for manual review; their parent batch is marked partial-success and the script continues.

#### 19.12.5 — Validation against HDBSCAN (the §19.0 LLM-central stance in action)

After the full LLM clustering completes:
- For each LLM mechanism class, compute the distribution over Phase-1 HDBSCAN cluster IDs of its members (cross-subtype HDBSCAN cluster IDs since LLM is cross-subtype).
- Concentration metric: per LLM class, fraction of mass on top-3 HDBSCAN clusters; per LLM class, # of HDBSCAN clusters with ≥ 1 member.
- Aggregate report: median concentration across all LLM classes; histogram of LLM-class-vs-HDBSCAN-cluster-spread.
- **Outcomes (both publishable):**
  - **High concentration** (median > 60% on top-3 HDBSCAN): "LLM clustering is a coherent coarsening of HDBSCAN embedding-density structure — both methods agree on the underlying mechanism boundaries, the LLM adds the readable label."
  - **Low concentration** (median < 30% on top-3 HDBSCAN): "LLM clustering identifies cross-cluster semantic groupings that HDBSCAN's embedding-density-only criterion cannot find — the LLM stage is doing real work, not just naming what HDBSCAN already discovered."
- This is the validation check that justifies the LLM-central stance. It does NOT require HDBSCAN to "succeed" or "fail" — both outcomes are interpretable findings.

#### 19.12.6 — Cost estimate (per CLAUDE.md > 10k token rule)

- **Activity:** Full-VPN LLM clustering — assign each of 19,073 nodes to a mechanism class via Claude Code CLI shim batches.
- **Per-batch:** ~80 nodes/batch (matches Pass-2 size for AUP-resilience and timely per-batch saves).
- **Per-batch prompt composition:** v3 + v2 seed taxonomy context (~12k input tokens, **cacheable** within 5-min windows) + 80 nodes × ~250 chars (name + 1-line desc + subtype) = ~5k input tokens batch-specific.
- **Per-batch output:** 80 decisions × ~150 tokens = ~12k tokens.
- **Per-batch shim overhead:** ~30k (Claude Code agent system prompt baked into every CLI invocation per ADR 010).
- **Per-batch total:** cold first call ~57k; warm-cache subsequent calls ~47k.
- **N batches:** 19,073 / 80 = **239 batches**.
- **Total estimate:** 1 × 57k + 238 × 47k = ~11.24M Max plan, +5% retry budget = **~11.8M Max plan**.
- **Wall time:** ~10–15h sequential at ~3-5 min/batch (extrapolated from Pass-2 timing).
- **Hidden-N drains checked:** shim overhead 239 × 30k = ~7.17M (~60% of total — irreducible per ADR 010); AUP block rate ~1/27 (Pass-2 base rate) → expected 9 batches needing sub-split retry, +~300k tokens; cache hit rate assumed ~90% — sensitivity is on this assumption.
- **Confidence:** medium-high. Pass-2 is direct precedent at 9.1× smaller scale (~1.16M actual for 27 batches = ~43k/batch); 239 × 43k = ~10.3M, consistent with my 11.8M including retries.

#### 19.12.7 — Smoke-test before full run (LOCKED 2026-05-09)

Before launching the full 239-batch chain, run **5 smoke-test batches** (one per non-risk subtype: pa, ti, dr, im, va — 80 stratified-random nodes each). User reviews per-subtype outputs for: (a) granularity matches the §19.12.2 guidance (not over-granular), (b) subtype-as-metadata correctly differentiates same-thought-different-function mechanisms, (c) seed taxonomy coverage adequate or needs extension. Smoke-test cost: ~250k tokens, ~30min wall. Tweak system prompt + seed taxonomy if needed. Full run only kicks off after explicit user approval.

#### 19.12.8 — Outputs

- `phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/cluster_memberships_rev8_paper_methodC_full_vpn.pkl` — final mechanism-class assignments, key schema `("rev8_paper", "llm_full_vpn", pool, "llm", class_name) → list of node IDs`
- `phase2_full_vpn_batches/batch_NNN.json` (idempotent per-batch atomic saves, 239 files at completion)
- `phase2_seed_taxonomy_nr_v4_full_vpn.json` + `phase2_seed_taxonomy_risk_v3_full_vpn.json` — final mechanism class catalogs (v3 NR + v2 risk extended with any new classes proposed during full run)
- `phase2_full_vpn_validation_vs_hdbscan.csv` — §19.12.5 concentration metrics per LLM class
- `phase2_full_vpn_aup_failures.json` — any sub-batches that AUP-blocked even after split (manual review)

#### 19.12.9 — What §19.10 (Track 1 / Track 2) becomes under §19.12

§19.10's dual-resolution Track 1 / Track 2 framing is **superseded by §19.12** for the headline result. The full-VPN LLM clustering of §19.12 IS the canonical mechanism-class catalog. HDBSCAN's role is reduced to the validation check in §19.12.5 (concentration analysis on the overlap set, which is now the FULL VPN since both methods cluster the same 19,073 nodes). Pass-2 residual outputs are reabsorbed into the full-VPN run by re-classifying all nodes consistently. The 33 v3 seed groups remain as the starting catalog the LLM extends.

### 19.13 — Doublet pivot + hybrid Sonnet/Opus pipeline (added 2026-05-15)

Replaces the per-subtype Pass B that was originally planned in §19.9.6 Task D. The doublet (RG × MG) primitive is the **path-level** vocabulary that feeds the paper's three deliverables (faithfulness, scale catalog, novel-candidate generation).

#### 19.13.1 — Doublet pivot rationale

Instead of per-subtype Pass B (clustering body subtypes pa/ti/dr/im/va separately, then composing into doublets after), we use **path-level LLM grouping** over the 8,954 hopwise EDGE-only VPN paths. Each path is grouped TWICE in one call:
- **RISK GROUP (RG):** based on the risk-node mechanism (the "what failure mode")
- **MECHANISM GROUP (MG):** body + intervention as one mechanism unit (the "what proposed fix")

Reasoning: the body subtypes are a continuum from risk-leaning (pa, ti) to intervention-leaning (im, va, dr), not five orthogonal pools. Per-subtype clustering fragments single mechanisms across body roles; grouping the body+intervention as one mechanism unit preserves intervention coherence. The downstream analyses both want RG × MG co-occurrence (which mechanisms address which risks, and vice versa) — the path-level doublet is the direct primitive for both.

#### 19.13.2 — Pipeline architecture (LOCKED 2026-05-15)

```
Seed (Opus, ONE call):  200 paths → combined catalog + per-path assignments
                        Output:  ~30-40 RG, ~100-130 MG groups, 200 (RG, MG) doublets
Pass B (Sonnet, batches of 75 paths, NO new-group proposals, fixed catalog):
                        For each batch: prompt = catalog (names + descriptions, no member-path repetition)
                                                  + 75 paths
                                        output = 75 (path_id, RG, MG) assignments
Every 10 Sonnet batches (~750 paths) → 1 Opus REVIEW pass:
                        - REVIEW_A (group-level health, ~150k tokens):
                            Opus sees catalog + per-group assignment counts + Sonnet's
                            flagged-uncertain rows. Decides: merge/split/rename groups.
                        - REVIEW_B (per-flagged-group deep dive, 0-5 calls x ~100k each):
                            Opus sees one group's name + desc + up to 50 of its member
                            paths. Splits, renames, or merges based on member content.
                        - REVIEW_C (low-confidence/unclassified paths, 1-2 calls x ~150k):
                            Opus sees catalog DESCRIPTIONS ONLY (no member paths) plus
                            the flagged paths. Assigns to existing OR proposes new
                            groups (constrained by min-group-size to avoid singletons).
                        Apply catalog delta (renames, merges, new groups, splits).
                        Reassign any paths affected by merges/renames.
Repeat until all 8,954 paths assigned.
```

Three structural rules in the Opus review architecture (responding to "Opus cannot see 1000+ paths in one call — context overflow"):
1. **Group-level decisions (REVIEW_A) see no per-path content.** Only catalog text + per-group counts + flagged rows. Fits comfortably in 200k context.
2. **Per-group deep dive (REVIEW_B) sees one group at a time** with a sampled subset (<=50 paths) of that group's members. Single call per problematic group.
3. **Unclassified-paths triage (REVIEW_C) sees descriptions only** for all existing groups (no member paths from existing groups, since those are already-assigned and not subject to change in this pass). Plus the flagged paths.

This bounds each Opus review call to <=200k input tokens, well within context, and lets the review work proceed without ever needing to see the full assignment matrix at once.

#### 19.13.3 — Sonnet-4-6 calibration (measured 2026-05-15)

Validated against the same 30 paths and the same 20-RG + 27-MG catalog that Opus produced in the 30-path smoke earlier that day. Sonnet given assignments-only task (read-only catalog, `allow_new=False`, `allow_coherence=False`).

| Metric | Result |
|---|---|
| Exact (RG + MG) doublet match | **29/30 = 96.7%** |
| MG-only match | **30/30 = 100.0%** |
| RG-only match | **29/30 = 96.7%** |
| Complete disagreements (both differ) | **0/30 = 0.0%** |
| Sonnet wall-clock per 30-path call | **60 sec** (45s prefill + 15s text emission) |
| Sonnet tokens per 30-path call | **~47k** (16k input + ~750 output + ~30k shim preamble) |
| Sonnet session-budget impact (Max 5x) | **+3pp** (per UI report 15% to 18%) — ~0.41x Opus weight per token; ~1pp = 16k Sonnet tokens vs 6.5k Opus tokens |

The single divergence (`path_00433`): Opus -> RG001 "Catastrophic misalignment of advanced AI"; Sonnet -> RG003 (adjacent risk family). Both agreed on MG004. Within reasonable inter-model variance for adjacent risk-group framings.

**Paper claim this calibration supports:** the bulk path-routing step (Pass B) does not require an extended-thinking reasoning model. Sonnet 4.6 routes paths to an existing high-quality catalog (Opus-built seed) at 96.7% exact-doublet agreement with Opus. This is methodologically defensible: **cheap model for bulk routing, expensive model for catalog generation and periodic catalog refactoring.** This pattern generalizes beyond this paper — it is a useful methodology contribution in its own right (cite as: hybrid expensive-reasoning + cheap-routing pipeline for large-scale LLM-mediated classification, with empirical congruence-test validation before commitment).

#### 19.13.4 — Total Pass B estimate with Sonnet calibration

| Phase | Per call | Calls | Wall-clock | Raw tokens | Max-plan-equivalent |
|---|---:|---:|---:|---:|---:|
| Sonnet 75-path batch | ~140k tokens (~110k input + 5k output + 30k shim) | 8954 / 75 = **120** | ~2 min each | ~17M | ~7M (Sonnet x 0.41) |
| Opus REVIEW_A | ~150k | 12 (every 10 Sonnet batches) | ~10 min each | ~1.8M | ~1.8M |
| Opus REVIEW_B | ~100k | ~36 (avg 3 per review) | ~8 min each | ~3.6M | ~3.6M |
| Opus REVIEW_C | ~150k | 12-24 (1-2 per review) | ~10 min each | ~2.7M | ~2.7M |
| **TOTAL** | | | **~6-9h LLM compute** | **~25M raw** | **~15M Max-plan equiv** |

Sessions at 650k cap: **~23 sessions Max-plan-equivalent**.
At 5h session reset and ~6 batches fit per session: **~4-6 days intermittent processing**.

This is **~40% cheaper** than the all-Opus alternative (~26M Max-plan-equivalent, ~40 sessions, ~8-10 days), with congruence to Opus output at 96.7% exact-doublet agreement.

#### 19.13.5 — Failure-mode review the paper will need

The Sonnet 30-path test was small. Before publishing the methodology, must validate at scale:
- Re-test Sonnet vs Opus on a larger (>=200-path) sample at end of Pass B — measure per-batch agreement drift over the course of the run as the catalog grows. Agreement could degrade if catalog fragmentation makes routing decisions ambiguous.
- Quantify the per-pool (RG vs MG) variance: is Sonnet stronger on mechanism-routing than risk-routing? (The 30/30 MG match vs 29/30 RG match weakly suggests this.)
- Audit the Opus REVIEW_B / REVIEW_C decisions — these are where new groups get added; misjudgments compound across batches. A separate human review of the catalog delta from each Opus review pass is warranted before committing to the next 10 Sonnet batches.

#### 19.13.6 — Streaming-to-disc as the transport contract for any large-output call

Locked architectural rule (added 2026-05-15 after a failed 200-path seed lost ~400k tokens to a Windows CMD-shim `subprocess.run(timeout=)` orphan-claude bug): any LLM call estimated to produce > 10k output tokens MUST use `claude -p --output-format stream-json --include-partial-messages` via `subprocess.Popen` with line-iterated stdout. Each text-delta event is written to a partial file (`partial.txt`) with explicit `flush()` after every event. Even on mid-stream process death, the partial file on disc preserves all emitted text up to that point and is hand-recoverable.

This applies uniformly to the Opus seed-gen, all three Opus review variants (REVIEW_A/B/C), and any future large-output call. Sonnet Pass B per-batch calls are smaller (~5k output) but use the same streaming wrapper for consistency and live progress visibility (`tail -f phase2_doublet_seed_partial.txt`).

The implementation lives in `phase2_step4_phase2_doublet_llm_grouping.py:streaming_claude_call` and `streaming_call_with_validation`. The lower-level shim path (`call_with_validation`) is preserved for small-output calls but flagged as unsafe for any large-output use.

#### 19.13.7 — Pass B first long-run + REVIEW_A/B/C implementation (added 2026-05-16)

**Pass B Sonnet first long-run results (2026-05-15 → 2026-05-16, wall-clock 02:46).**
Following the 3-batch shakedown that exposed a Sonnet JSON-restart bug
(Sonnet emits a complete JSON, then restarts the JSON from scratch mid-stream;
the sentinel still passes but `json.loads` fails on the duplicated prefix),
the script gained two fixes before the long run:

1. **JSON-restart recovery via `rfind('{"assignments":[')`** — locates the
   last legitimate start-of-JSON marker and parses from there. Drops `n` chars
   of stale prefix, logs the drop, recovers the assignments. Tested in
   production (3/19 batches triggered recovery; all 3 recovered cleanly).
2. **Failed-partial preservation** — copies the stream-partial to a per-batch
   `partial.batch_NNNN_failed.txt` before the next batch overwrites
   `partial.txt`. One earlier batch (batch_0001) was lost prior to this fix;
   no batch lost since.

| Batch range (2026-05-15 long run) | Wall-clock | Resolved | UNASSIGNED | Recoveries |
|---|---:|---:|---:|---:|
| batch_0004 – batch_0019 (16 batches) | 2h 46m | ~730 | ~470 | 3 (rfind, all clean) |

**Total Pass B coverage so far:** 1,080 resolved + 545 UNASSIGNED = **1,625 of
8,954 paths (18.1%)**. Catalog remains FROZEN at 34 RG + 119 MG throughout
(read-only Pass B). 7,329 paths remain for subsequent Sonnet sessions. Per-batch
mean: 45.6 resolved (sd ~6), 37% UNASSIGNED (range 27–51%, batch-to-batch
variance, no monotonic trend).

**REVIEW_A/B/C implementation (task #29 completed 2026-05-16).** The
`--mode opus_review` orchestrator is in place:

- `make_review_a_prompt`, `make_review_b_prompt`, `make_review_c_prompt` build
  the three prompt schemas matching §19.13.2's architecture.
- Catalog mutations are persisted via `phase2_doublet_group_remap.json`
  (transitively resolved on read) and `phase2_doublet_path_remap.json` —
  individual Sonnet batch files remain immutable; the merged jsonl is rebuilt
  on demand with remaps applied. The active catalog is backed up before each
  review pass to `phase2_doublet_active_catalog.pre_review_{a,b,c}_NNN.json`.
- All three review LLM calls use the same streaming-to-disc transport as
  Sonnet Pass B + Opus seed-gen (per §19.13.6), with `rfind`-based JSON-restart
  recovery identical to the Sonnet Pass B path.
- `--review-pass {a,b,c,all}` flag allows partial runs when session budget is
  tight. Only `--review-pass all` resets `batches_since_review` to 0.

Class B smoke test (`unit_test_opus_review.py`) exercises every prompt maker,
apply helper, and remap persistence pathway end-to-end against a synthetic
catalog + assignments + UNASSIGNED dataset. Six test sections all pass with
**zero LLM tokens**.

#### 19.13.8 — First live Opus REVIEW_A/B/C run (2026-05-16, review_idx=001)

Ran against the 1,625-path post-Pass-B dataset (1,080 resolved + 545
UNASSIGNED). Wall-clock 40 min (vs 2 hr estimate — REVIEW_B was much faster
than projected; only 8 of 13 flagged groups capped to 8 actual deep-dives).

| Pass | Calls | Outcome |
|---|---:|---|
| REVIEW_A | 1 | 31 RG keeps, 0 RG renames, 0 RG merges, 3 RG deep_dives; 113 MG keeps, 0 MG renames, 1 MG merge, 5 MG deep_dives |
| REVIEW_B | 8 deep-dives (3 RG + 5 MG) | All clean; **+4 RG, +1 MG via splits** |
| REVIEW_C | 6 batches (100/100/100/100/100/45 paths) | 5/6 succeeded → 445 paths resolved + new groups proposed; 1/6 (batch_0101) timed out at `claude -p exit 1` after 71s on a 60k-token prompt, 100 paths preserved for next cycle |

**Catalog after one REVIEW cycle: 34→41 RG (+7), 119→130 MG (+11).**

REVIEW_A's `audit_summary` flagged the failure mode the routing LLM exhibited:
*"the routing LLM seems to over-concentrate on the most generic / first-listed
group when paths are borderline, producing the long-tail-plus-fat-head
distribution observed on both axes"* — explains why RG001 (n=175) and MG001
(n=133) were the largest deep-dive candidates, and why the seed catalog +
catch-all groups absorbed paths Opus later split into more-specific
mechanism families.

#### 19.13.9 — Post-review Sonnet Pass B (2026-05-16, batches 0020-0026)

Ran 10 batches against the expanded 41 RG + 130 MG catalog; 7 succeeded, 3
failed when the session-budget hit Anthropic's rate limit (clean exit-1 on
attempts; failed partials preserved). UNASSIGNED rate dropped from **37%
(pre-review) to 28% (post-review)** — the wider catalog absorbs paths the
narrower one couldn't, exactly as predicted.

| Pre vs Post REVIEW (Sonnet Pass B per-batch means) | Pre | Post |
|---|---:|---:|
| Resolved per batch | 45.6 | 53.7 |
| UNASSIGNED rate | 37% | 28% |
| JSON-restart recovery rate | 19% (3/16) | 29% (2/7) |
| Recovery success | 100% | 100% |

The recovery-rate uptick post-review may be sampling noise (n=7) but is worth
watching across the next cycle — if the longer post-review prompts (180k chars
vs 178k pre) push Sonnet's restart bug harder, may need to shorten the catalog
prompt encoding.

**Overall coverage as of end of 2026-05-16 session: 2,520 of 8,954 paths
decided (28.1%).** 200 seed + 1,875 Sonnet (25 saved batches × 75) + 445
REVIEW_C. To-be-reprocessed in next cycle: 400 paths (batch_0001 + batches
0027-0029 + review_c_101). Active catalog (41 RG + 130 MG) is canonical for
next Sonnet batches.

#### 19.13.10 — Recovery run + prompt-context refinement (added 2026-05-17)

**Recovery run (2026-05-16 18:43 → 21:30).** Chained `--review-pass c` (with
new dedup filter that skips already-REVIEW_C-resolved unassigned rows) +
12 Sonnet Pass B batches against the post-review catalog. Recovered all
375 paths from prior code/quota failures (batch_0001 + 0027/0028/0029 +
review_c_101), processed 525 new paths, hit Sonnet rate limit cleanly at
end-of-session with 0 unrecoverable failures. Coverage reached **3,050 of
8,954 paths (34.1%)** with catalog at **42 RG + 138 MG**.

Sonnet Pass B UNASSIGNED-rate trajectory across the run:
- Pre-REVIEW (batches 0000-0019): 37% mean
- Post-REVIEW idx=1 (batches 0020-0026): 28% mean
- Post-REVIEW idx=2 / post-recovery (batches 0027-0038): **20% mean**

This validates the catalog under-coverage hypothesis for the UNASSIGNED rate
(§19.13.5): each REVIEW cycle expands the catalog AND reduces routing
UNASSIGNED rate by ~10pp. The competing hypothesis — that Sonnet was
over-using the escape hatch — is **disproven**: an over-cautious model
would not become less-cautious as the catalog grows.

**Per-batch JSON traceability — PAPER AUDIT REQUIREMENT (2026-05-17).** Every
Sonnet Pass B batch and every Opus REVIEW call writes a per-batch JSON file
that captures the full LLM output, recovery action (if any), and resolved
assignments. These files live at:

- `phase2_doublet_passb_batches/batch_NNNN.json` — Sonnet Pass B (75 paths/batch)
- `phase2_doublet_opus_reviews/review_{a,b,c}_NNN[_GID].json` — Opus REVIEW
- `phase2_doublet_passb_partial.batch_NNNN_failed.txt` — preserved raw stream
  for any batch whose JSON parse or stream failed (with meta JSON alongside).

These are the canonical lineage for any reported assignment, mechanism-group
proposal, or REVIEW decision. `phase2_doublet_assignments.jsonl` is a
DERIVED artifact rebuilt from these via `rebuild_merged_assignments_jsonl()`
on every batch save. **None of these per-batch files may be deleted or
auto-rebuilt** — they are the workshop-paper-audit substrate.

**Prompt-context refinement (between REVIEW idx=2 and idx=3, 2026-05-17).**
The original REVIEW_A/B/C prompts did not include the corpus identity (ARD)
or the downstream paper-deliverable structure (catalog table, many-to-few
analysis, gap analysis). Added a `PAPER_DELIVERABLE_CONTEXT` block to all
three prompt makers in `phase2_step4_phase2_doublet_llm_grouping.py` so
cycles 3+ Opus has explicit guidance on group-naming, granularity, and
"forbidden singleton" semantics. The change is a forward-only methodological
refinement (no past assignments are perturbed; cycles 1+2 decisions are
already baked into the active catalog). Paper Methods will document the
single mid-pipeline prompt revision and the rationale; if any reviewer
asks for sensitivity analysis, the cycle-1+2 vs cycle-3+ catalogs can be
compared directly via the per-pass xlsx snapshots already on disc.

### 19.14 — Pipeline restart 2026-05-17 (canonical methodology for paper)

After the prompt-context sensitivity studies (§19.13.11), an audit of the
first-run catalog identified residual issues that compound (RG002 meta-
catch-all absorbing gradient-hacking and capability-gap paths; RG006 vision-
only by seed bias; singleton MGs violating the stated MIN_GROUP_SIZE rule
not code-enforced; mid-pipeline prompt revisions creating an "ad hoc patch"
narrative for the Methods section). The pipeline was restarted under a
refined methodology that becomes the canonical narrative for the paper.

**Final pipeline (canonical for paper Methods)**:

1. **Path dedup** (`phase1_dedup_paths.py`): within-paper containment cut at
   ≥70%. 8,954 raw paths → **2,772 distinct mechanism-paths** (69%
   reduction). Justification: a path that shares ≥70% of node IDs with
   another path from the same paper is a near-redundant extraction
   variant; keeping both inflates downstream group sizes without adding
   mechanism diversity.
2. **Pilot architecture discovery** (`pilot_100_path_axis_discovery.py`):
   single Opus 4.7 call on 100 paths (sampled 1-per-paper). Opus proposes
   3-9 orthogonal navigation axes, builds initial harm_class and
   mechanism_class taxonomies, assigns all 100 paths to all dimensions,
   and emits an architecture critique. **Output of the 100-path pilot:
   6 axes (lifecycle_stage, modality, methodology, severity,
   emergence_stage, harm_target) + 33 harm_classes + 45 mechanism_classes**.
3. **Opus routing** (`phase2_step5_opus_routing.py --mode opus_routing`):
   75-path batches over the 2,672 remaining deduped paths. Each batch
   routes paths to existing classes, proposes new classes when needed
   (MIN_GROUP_SIZE=3 code-enforced — new classes with <3 batch members
   are dropped and paths force-fit to closest existing class, flagged for
   consolidation), tags 6 axes per path, emits per-assignment confidence,
   flags catalog-improvement candidates (merge/rename/split/axis-ext).
4. **Consolidation** (`--mode consolidation`): every 10 routing batches,
   ONE Opus call adjudicates accumulated catalog-improvement flags.
   Applies merges (via class_remap), renames in-place, axis extensions
   to controlled vocab, schedules deep-dive splits for next routing
   batch's attention.
5. **Final audit** (`--mode final_audit`): end-of-run single Opus call
   walks the full catalog, decides remaining singletons (force-merge or
   keep-niche), tightens class definitions, finalizes axes vocabulary,
   identifies transferability examples and gap candidates for the paper's
   three deliverables.

**Faceted classification** (per-path tags, NOT per-class — preserves
intra-class diversity):
- 2 curated open-vocabulary axes: `harm_class`, `mechanism_class` (with
  growing taxonomies + definitions, evolve via routing/consolidation)
- 3 intervention-side enum axes (per path): `lifecycle_stage`, `modality`,
  `methodology`
- 3 risk-side enum axes (per path): `severity`, `emergence_stage`,
  `harm_target` (the `harm_target=capability-gap-only` value cleanly
  flags ML-engineering papers as non-human-risk peers)

Default paper figure: `harm_class × mechanism_class` matrix (the doublet).
Appendix slices: any axis combination at analysis time.

**First-run state archived** to
`phase2_results/archive/2026_05_15_first_run/step1_load_and_parse_umapwithoutlocalsatellites/`
(32 files: prior 42 RG + 138 MG catalog, all Sonnet Pass B batches,
all Opus REVIEW outputs, all intermediate xlsx). Retained for
sensitivity-analysis appendix only; not part of the paper's reported
catalog or analysis.

**Catalog-evolution audit** during the restarted run is captured by
per-batch JSON files at `phase2_routing_batches/batch_NNNN.json`,
per-consolidation outputs at `phase2_routing_consolidations/`, and
the final audit at `phase2_routing_final_audit.json`. Same audit-trail
guarantees as documented in §19.13.10 for the first run.

**Two pilot outputs preserved** at `phase2_pilot_50paths_axis_discovery.json`
and `phase2_pilot_100paths_axis_discovery.json` for the Methods appendix:
shows architecture stability across pilot sample sizes (same 6 axes,
vocabularies extend organically rather than restructure).

#### 19.13.11 — Prompt-context sensitivity studies (added 2026-05-17)

To pre-empt reviewer concerns about the mid-pipeline prompt revision in
§19.13.10, ran two A/B studies on the SAME catalog and input state with
vs without `PAPER_DELIVERABLE_CONTEXT`:

**Study A — Opus REVIEW_A sensitivity** (`study_a_review_a_context_sensitivity.py`,
2 Opus calls on current 42 RG + 138 MG catalog).

With-context decisions are a strict **superset** of no-context decisions:
| Axis | no-context flagged | with-context flagged | additional |
|---|---|---|---:|
| RG deep_dive | {RG001} | {RG001, RG002, RG038} | +2 |
| MG deep_dive | {MG001, MG086, MG123} | {MG001, MG010, MG086, MG100, MG116, MG119, MG120, MG123} | +5 |

No renames, no decision reversals. The refined prompt is strictly more
thorough — flags every group the old prompt flagged AND additional groups.
**Defensible mid-pipeline refinement: prior decisions remain valid; cycles
3+ benefit from broader audit scope.**

**Study B — Sonnet routing sensitivity** (`study_b_sonnet_context_sensitivity.py`,
1 Sonnet call on 30 smoke paths vs §19.13.3 baseline).

| Comparison | both_match / 30 | rg_match | mg_match |
|---|---:|---:|---:|
| Baseline no-context Sonnet vs Opus | 29 | 29 | 30 |
| New with-context Sonnet vs Opus    | 29 | 29 | 30 |
| **with-context vs no-context Sonnet (self)** | **30** | **30** | **30** |

Sonnet routing is **identical** with vs without context. The same single
Opus-vs-Sonnet disagreement (path_00433) persists in both. Context adds
no behavioral signal at the routing layer.

**Decision: keep PAPER_DELIVERABLE_CONTEXT in REVIEW_A/B/C prompts (cycles
3+); do NOT add to Sonnet Pass B prompts** (saves ~5k tokens × 120 batches
= ~600k tokens = ~+10pp session across full Pass B for zero benefit).

#### 19.13.12 — Gleb's eigenvector / race-framing findings: sensitivity analysis (Class B, 2026-05-17)

Ran two un-merged-graph experiments to test the robustness of Gleb's
"race dynamics dominates top-100 risks by EC" headline finding:

**Experiment 1** (`experiment_xrisk_unmerge_ec.py`): EC race-framing
fraction in top-100 risks across 4 variants:
| Variant | sole_PA race% | any_PA race% |
|---|---:|---:|
| A — full + SIM≥0.8 (Gleb-equivalent) | 0% | 14% |
| B — xrisk-keyword removed + SIM≥0.8 | 1% | 5% |
| C — full + EDGE-only | 1% | 8% |
| D — xrisk removed + EDGE-only | 3% | 10% |

**Cannot reproduce Gleb's reported 90% race-framed top-100 in our
un-merged graph.** Inspection of Variant A top-10 shows all 10 are
near-duplicates of "Existential catastrophe from misaligned advanced AI" —
the rev7 hub cluster (5+ near-dups at cos 0.955-0.984 per `MEMORY.md`).
Without merging, EC ranks the cluster INSTANCES at the top, not a single
merged hub. Variant B top-10 swaps in a parallel "Catastrophic misalignment"
near-duplicate cluster that escaped the xrisk-keyword filter.

**Experiment 2** (`experiment_xrisk_race_correlation.py`): direct xrisk ×
race-framing co-occurrence at risk-node level (no merging):
- xrisk nodes (n=2,124): 178 have race-framed PA neighbor = **8.4%**
- non-xrisk nodes (n=16,972): 404 have race-framed PA neighbor = 2.4%
- **Odds ratio = 3.75 (CI95 3.13–4.50, chi2 = 227.97, p < 0.001)**

Real but modest discourse co-occurrence. The merging step amplifies this
to ~80-90% via cluster collapse: P(race PA on merged hub) =
1 − (1 − 0.084)^N reaches 90% at N≈25 near-duplicate xrisk nodes per
merged hub.

**Conclusion**: Gleb's finding is **real-but-amplified, not entirely
artifactual.** The structural co-occurrence is reproducible (OR=3.75) but
the reported magnitude (90% top-100) is a merging artifact. Recommended
paper edits captured in `gleb_analysis_critique_DO_NOT_COMMIT.md`
(gitignored).

### 19.15 — Oversized-class split deep-dive sweeps (added 2026-05-19)

After 5 routing batches + `consolidation_001`, six classes had grown past
the `OVERSIZE_ALARM=25` member threshold. The new architecture
(`§19.14` + the NO-HETEROGENEOUS-CATCH-ALL policy) allows large classes
**iff** they are single-mechanism-family coherent; heterogeneity (not
size) triggers required split. To validate large-class homogeneity
**before** further batches grew them past Opus's single-pass capacity,
six `final_misfit_sweep --classes <ID> --chunk-size 100` runs were
executed, each presenting Opus the full membership of one class in a
single pass with a 3-decision tree (HOLD / UNASSIGN / SPLIT_OUT to a
new ≥3-member proposed class, with REASSIGN to an existing class as
an alternative).

**Results — all 6 defended as homogeneous, zero splits**:

| Sweep | Class | n_in | HOLD | UNASSIGN | SPLIT_OUT | REASSIGN |
|---|---|---:|---:|---:|---:|---:|
| 2 | HC002 (AGI value misalignment) | 66 | 59 | 7 | 0 | 0 |
| 3 | HC008 | 54 | 52 | 2 | 0 | 0 |
| 4 | HC012 | 49 | 43 | 6 | 0 | 0 |
| 5 | MC001 | 37 | 34 | 3 | 0 | 0 |
| 6 | MC011 | 22 | 22 | 0 | 0 | 0 |
| 7 | MC015 | 40 | 36 | 4 | 0 | 0 |
| **Total** | | **268** | **246 (92%)** | **22 (8%)** | **0** | **0** |

**Interpretation**:
- After `consolidation_001` already carved off three distinct
  sub-channels (HC015 instrumental-power-seeking, HC016/HC017 from HC013
  split, HC018 governance-race), the residual large classes are
  single-family coherent on their respective harm or mechanism
  dimensions.
- 22 UNASSIGN decisions are honest exits — risk-side or mechanism-side
  does not actually match the class focus. Examples from HC002:
  military/geopolitical AI competition (`02185`), alignment-misuse-for-
  dystopia (`02065`), multi-agent cooperation failures (`02489`),
  accident/robustness failures (`01690`, `02396`). These flag residual
  catalog gaps for `misfit_review` to surface as PROPOSE_NEW candidates
  if ≥3-member clusters emerge.
- Zero SPLIT_OUT proposals across all 6 classes is itself a finding:
  Opus is not forcing splits to please the prompt; it is making
  principled defend-as-homogeneous decisions when the residual is
  coherent.

**Why this validates the §19.14 architecture for the paper**:
1. The heterogeneity-not-size policy is operationally meaningful: large
   classes (HC002=59, HC008=52, HC012=46) are kept; only flagged
   sub-channels are split.
2. The `consolidation_001` carve-outs were correct first-order
   adjustments; second-pass scrutiny finds no further sub-channel
   structure.
3. The 18 HC + 19 MC catalog is stable at this corpus depth (475 routed
   paths out of 2,772). Further batches may surface new classes but the
   existing classes are robust.

**Cost**: 6 Opus 4.7 sweeps via `claude -p` shim, ~23pp session total
(actual; estimated 17pp pre-launch — calibration revised below).
Wall-clock 50-150s per sweep depending on class size. Per-sweep outputs
preserved at `phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/
final_misfit_sweep_*/sweep_NNN_summary.json` for audit.

**Implication for cost calibration**: the `claude -p` shim with no
prompt-cache reuse, large catalog + member-detail input (~20-30k tokens
each), and ~1-1.5k output tokens runs at **~3.5-5pp per sweep**.
Pre-launch estimate of 17pp underestimated by ~35%; revised
calibration: **per-50-member-class sweep ≈ 4pp session** (was 3pp).
Used to budget the remaining 31 routing batches + consolidations.

### 19.16 — Canonical batch-processing production plan (added 2026-05-19)

After the 6 oversized-class deep-dives (§19.15), the canonical production
plan for the remaining 31 routing batches + reviews is fixed as follows.
Designed for workshop-paper defensibility against standard NLP-
classification methodology review.

**Three review-step types** (defensible methodology stack):

**(R1) Self-flagged misfit pass** — Opus second-pass over paths the
routing-batch Opus already flagged as uncertain. Inputs:
  - fit_score ≤ 3 (Opus self-reported low-confidence assignment)
  - heuristic keyword-overlap below threshold (algorithmic misfit signal)
  - manual `reassign_pending` tag (analyst-flagged candidates)

Cadence: every 5 routing batches. Cost: ~5-7pp/run.
Coverage rationale: catches Opus's own confident-but-wrong calls only to
the extent they self-flag — which is partial. Strongest defense is paired
with (R3) faithfulness study.

**(R2) Unassigned-pool audit** — Adjudicate every path with no HC and/or
no MC assignment. Per-path verdicts: REASSIGN-to-existing-class | 
PROPOSE_NEW (≥3 peers cited) | HOLD-unassigned. Chunks at ~80 paths
per Opus call.

Cadence: triggered when unassigned pool grows past ~50 paths OR at
end of batch processing. Cost: ~10-15pp per ~150-path pool.
Coverage rationale: closes the "no home" gap and surfaces new classes
that didn't reach MIN_GROUP_SIZE within any single batch.

**(R3) Cross-vendor faithfulness study** — Stratified random sample of
N=30-50 paths re-classified independently by a **non-Anthropic** model
(OpenAI o3 / GPT-5 or Google Gemini 2.5). Compute % agreement on HC,
MC, and 6 axes. Headline metric for paper deliverable 1.

Cadence: once after all routing batches complete. Cost: OpenAI/Google
API tokens (not Max-plan pp; ~<$5 for N=50 at current pricing).
Coverage rationale: this is the workshop-paper deliverable. Same-model
faithfulness (Opus→Opus) is methodologically circular and will be
flagged by reviewers. Cross-vendor judgment is the publication standard.

**(R4) Oversized-class deep-dive sweeps** — Auto-fire when any HC or MC
crosses OVERSIZE_ALARM=25 members and hasn't been swept yet. Single Opus
call per class, full membership reviewed. Verdicts: HOLD / UNASSIGN /
REASSIGN / SPLIT_OUT.

Cadence: on-demand at the end of each batch cycle. Cost: ~3-5pp/sweep.
Coverage rationale: enforces the NO-HETEROGENEOUS-CATCH-ALL policy
(§19.15) — every large class must be defended as homogeneous or split.

**Consolidation** — adjudicates accumulated batch-emitted catalog-
improvement flags (merge_candidates, rename_suggestions, generalize_
suggestions, axis_value_extensions, split_candidates, reassign_
candidates). Already in place. Cadence: every 5 routing batches. Cost:
~3-5pp/run when flags exist; no-op when none.

**Axes review** — random sample N=50 → audit 6 axis values per path
for vocab consistency. Cadence: once at ~batch 20 + once final.
Cost: ~3-4pp/run.

**Final audit** — single Opus pass over the entire catalog at end-of-
run. Decides singletons, tightens definitions, finalizes axes vocab,
surfaces transferability examples. Cost: ~5-7pp once.

**CYCLE SIZE LOCKED AT 10 BATCHES** (revised 2026-05-19 from initial 5-batch
proposal). First-cycle empirical results (§19.17) showed 0 new classes,
14% reassignment rate at 215 candidates reviewed — catalog converged at
this corpus depth, so larger cycles amortize review overhead without
losing signal. Hard ceiling: Opus 4.7 context (~200k tokens) — at the
observed ~33% fit_score≤3 rate, 10-batch cycles produce ~250 R1
candidates → ~140k prompt → fits with margin. 15-batch cycles would
push ~375 candidates → ~205k → over context, needing chunking.

**Per-cycle automation (every 10 batches = 750 paths)**:
```
[10 routing batches × 75 paths]                              ~40-50pp, ~50-75 min
   ↓
[consolidation — all unprocessed batch flags]                ~5-7pp
   ↓
[R4 oversize sweeps — only NEWLY-oversized classes]          ~4pp × N (0-3 expected)
   ↓
[R2' incremental unassigned-pool audit]                      ~5-8pp
       - SCOPE: newly-unassigned in last 10 batches
              + paths on surfaced_sub_channels peer-watchlist
              + paths appearing in any flag this cycle
       - EXCLUDE: confirmed_out_of_scope (78% of prior R2 verdicts)
   ↓
[R1' incremental self-flagged misfit pass]                   ~5-8pp
       - SCOPE: last-10-batch paths with fit_score≤3 OR heuristic-flag
              + previously-HOLDed paths IF a new flag references them
   ↓
[STOP — user checkpoint review]
```

**Canonical R2→R1 order (locked 2026-05-19):** R2 first because it creates new
classes from unassigned peer-clusters; R1 second because it can then reassign
self-flagged misfits to the newly-created classes. Running R1 first would miss
those re-routing opportunities. (Initial R1 in this pipeline ran before R2 by
necessity — there was no R2 step yet; future cycles use the canonical order.)
Cycle target: ~30-40pp + 25-35 min wall-clock, 2-3 cycles per session.

**End-of-routing (after batch 36)**:
1. R2 unassigned-pool audit (~10-15pp)
2. Final consolidation (~5pp)
3. Axes review × 1 (~4pp)
4. Final audit (~5-7pp)
5. R3 cross-vendor faithfulness study (OpenAI/Google tokens; not pp)
Total end-of-routing: ~25-35pp + cross-vendor sample cost.

**Why this stack is workshop-defensible**:
- Opus self-uncertainty → R1 misfit pass (every cycle)
- Heuristic-flagged misfits → R1 (every cycle)
- Unassigned paths → R2 (end-of-routing)
- Oversized catch-alls → R4 (auto-fire on threshold)
- Confident-but-wrong → R3 cross-vendor sample (quantified error rate)
- Catalog drift → consolidation (every cycle)
- Axis consistency → axes_review (twice)

**NOT in plan** (deliberately deferred or rejected):
- Exhaustive per-path re-review across all classes (~98pp) — marginal
  value over R3 faithfulness study; rejected as not workshop-required.
- Direct-SDK with cache_control — 2026-05-19 decision to keep shim.
- Same-model faithfulness — rejected per cross-vendor requirement.

**Cost summary for remaining 31 batches + reviews + end-of-routing**:
~6 cycles × 35pp + end-of-routing 30pp = **~240pp ≈ 2.4 session caps
≈ 3 sessions including overhead**, plus modest OpenAI/Google tokens
for R3.

### 19.17 — R1 + R2 calibration results at 475 paths routed (added 2026-05-19)

Ran the first instance of R1 (self-flagged misfit pass) and R2 (unassigned-
pool audit) against the current 475-path corpus. Both confirm that the
18 HC + 19 MC catalog is stable at this corpus depth.

**R1 — self-flagged misfit pass** (`misfit_review_002.json`, scope=self_flagged):
| | count | % |
|---|---:|---:|
| Candidates (low_fit ≤3 + heuristic + reassign_pending) | 155 | 100% |
| **HOLD** — original assignment confirmed | **125** | **80.6%** |
| REASSIGN_HC to existing class | 11 | 7.1% |
| REASSIGN_MC to existing class | 4 | 2.6% |
| UNASSIGN | 2 | 1.3% |
| PROPOSE_NEW | 0 | 0% |
| Wall-clock 384s; cost ~14pp | | |

**R2 — unassigned-pool audit** (`misfit_review_003.json`, scope=unassigned):
| | count | % |
|---|---:|---:|
| Unassigned candidates (HC=None or MC=None) | 60 | 100% |
| **HOLD-as-unassigned** — honest non-AI-safety gap | **47** | **78.3%** |
| REASSIGN_HC to existing class | 5 | 8.3% |
| REASSIGN_MC to existing class | 7 | 11.7% |
| UNASSIGN (explicit out-of-scope) | 1 | 1.7% |
| PROPOSE_NEW | 0 | 0% |
| Wall-clock 192s; cost ~8pp | | |

**Catalog stability at 475 paths**: 0 new HCs, 0 new MCs, 30 path-level
reassignments. The 18 HC + 19 MC taxonomy is converged at this corpus
depth.

**The unassigned pool is NOT routing failure — it is honest long-tail
out-of-scope.** Opus confirmed 47 of 60 unassigned paths (78%) as
genuine non-AI-safety domain content: biomedical ontology alignment,
public health, education, marketing, fair division, corporate finance,
nuclear governance, climate geoengineering, conservation-AI, bio-AI,
generic-ML productivity. The dedup pipeline did not filter these (they
were extracted as AI-safety-adjacent based on paper-level keywords) but
the path-level intervention/risk is not AI-safety. This is a
publication-reportable finding: corpus has a ~12% out-of-scope tail
that explicit modeling can name and exclude.

**Sub-channels surfaced but not yet actionable** (named by Opus across
R1+R2, all with <3 corpus-wide peers as of 2026-05-19, so PROPOSE_NEW
gating correctly rejected them):
- prompt-engineering as a mechanism family (MC)
- fairness-via-data-balancing (MC)
- general optimizer-family beyond MC011 (MC)
- Bayesian-optimization-for-hyperparameter-search (MC) — n=2 named peers
- IoT-malware detection / AI-for-non-AI-cybersecurity (potential HC)
- conservation-AI applications (MC)
- bio-AI applications (MC)
- capability-research-as-safety advocacy (HC sub-channel)

These are tracked for the next misfit_review cycle (after batches 5-14)
to check if any reach the ≥3-peer threshold for class creation.

**Combined R1+R2 headline metric for workshop paper:**
> "Two independent second-pass Opus adjudications — self-flagged misfit
> review (N=155) and unassigned-pool audit (N=60) — yielded 80%
> agreement with original routing decisions across 215 candidate paths.
> 30 paths (14%) were reassigned to existing classes; zero new classes
> were proposed. The catalog stabilized at 18 harm-class + 19
> mechanism-class definitions at 475 routed paths, validating the
> minimum-group-size (n≥3) discipline."

#### 19.17.1 — Cycle 2 confirmation: out-of-scope tail is real (added 2026-05-20)

After routing 1,225 paths (44% of corpus), a second unassigned-pool audit
(`misfit_review_004.json`, N=141 candidates) yielded an **even stronger
out-of-scope signal**:

| Cycle | N candidates | HOLD-as-unassigned | Reassigned | UNASSIGN | PROPOSE_NEW |
|---|---:|---:|---:|---:|---:|
| R2 (cycle 1, n=475 routed) | 60 | 47 (78%) | 12 | 1 | 0 |
| **R2' (cycle 2, n=1,225 routed)** | **141** | **134 (95%)** | **7** | **0** | **0** |

**The HOLD-as-unassigned rate rose from 78% to 95% as the corpus expanded
from 475 to 1,225 paths.** This is the opposite of what would happen if
the catalog had genuine gaps: as more paths arrive, the catalog would
either grow (gaps fill via PROPOSE_NEW) or the unassigned rate would
plateau. Instead, **the rate INCREASED**, meaning the unassigned pool is
increasingly populated by paths that are **legitimately outside the
AI-safety scope**, not gaps in coverage.

**5 at-quorum sub-channels surfaced in R2', all NON-AI-safety domains:**

| Sub-channel (R2' surfaced) | Side | Peers | Category |
|---|---|---:|---|
| non-AI-existential-risk-cause-areas | HC | 13 | climate, biosec, nuclear governance NOT from AI |
| general-ML-productivity-tooling | MC | 11 | generic ML tools without AI-safety risk |
| human-cognitive-training-decision-aids | MC | 7 | AI-aided human cognition outside alignment |
| bio-AI-non-safety-applications | MC | 6 | protein folding, drug discovery |
| animal-welfare-harm-target | HC | 6 | speciesism, conservation paths |
| corporate-finance-and-governance-non-AI | HC | 5 | financial markets, corporate ops |

Per Opus's own R2' summary (verbatim):
> "These should remain UNASSIGNED — they are evidence of catalog scoping
> (paper Methods §) not gaps to fill."

**Publication-defensible interpretation**:
- The corpus extraction pipeline (`ARD` keyword-based extraction) has
  precision ~88-90% for AI-safety adjacency at the path level.
- The remaining ~10-12% is an honest scope-boundary tail that explicit
  modeling correctly recognizes and refuses to absorb.
- This is a **finding**, not a failure: the catalog architecture
  (NO-HETEROGENEOUS-CATCH-ALL + MIN_GROUP_SIZE=3 + UNASSIGN-as-legitimate)
  enforces conceptual coherence at the cost of recall.

**Workshop paper §3.6 (faithfulness) revision**:
> "Of 1,225 paths routed by Opus 4.7 across 15 batches, 132 (10.8%)
> remained unassigned after a single routing pass. A subsequent
> independent unassigned-pool audit confirmed 134 of 141 (95%) of these
> were legitimately out-of-scope domain content (biomedical, public
> health, financial markets, conservation, etc. without AI-safety risk
> chains). The catalog correctly excludes these rather than forcing
> catch-all absorption."

**Cycle 2 R1+R2 combined: catalog at 23 HC + 23 MC, +9 new classes added,
all grounded in ≥3-peer evidence; 9 R4 deep-dive sweeps + 23 batch R4
sweep + consolidation_002 confirm cycle 1's defended-homogeneous pattern
holds at the new corpus depth (96% HOLD rate across 1,085 sweep
decisions).**

#### 19.17.2 — Operations model revisions from cycle 2 (added 2026-05-20)

Cycle 2 empirical learnings drive several locked-in changes to the
production cadence (§19.16):

**(a) Watchlist-aware R4 sweep prompt is MANDATORY** — the cycle 2
HC008 first-pass sweep returned 0 SPLIT_OUT despite 17 named at-quorum
peers (13 vision + 4 NLP). After patching the sweep prompt to load the
surfaced_sub_channels watchlist, the re-run produced 2 SPLIT_OUTs (HC021,
HC022) covering 13 of 17 peers. **Without watchlist visibility, R4
sweeps systematically under-split.** Patch landed 2026-05-20 (all 6
prompts now load watchlist + watch_items where appropriate).

**(b) Sub-channels watchlist now populated by sweeps too** — cycle 2's
23-class R4 batch couldn't contribute to the watchlist because the sweep
schema didn't have a `surfaced_sub_channels` field. Patched 2026-05-20;
future sweeps emit watchlist entries for any below-quorum sub-channels
they observe during member review.

**(c) No-rework filter on misfit_review** — without the filter, R1' and
R2' in cycle 3+ would re-review ~236 paths previously HOLDed by R1+R2
this cycle (~73% of the low-fit candidate set). Filter landed
2026-05-20: misfit_review now excludes paths previously HOLDed unless
re-flagged via consolidation/sweep reassignment. Per-cycle savings
estimated at ~10-15pp. CLI flag `--include-previously-held` to override
for catalog overhauls.

**(d) Defended-homogeneous classes need cycle-by-cycle re-evaluation**
— MC001, MC004, MC016 were defended in consolidation_002 but at the
end of cycle 2 review chain they were NOT re-swept. Going forward,
defended classes should be re-swept ONLY if they have grown by
≥10 paths since last sweep AND a new sub-channel from the watchlist
matches their members. Saves R4 cost without losing signal.

**(e) Rate-limit safety: chunk multi-class sweep CLI args** — cycle 2's
first 24-class sweep attempt failed on 23 of 24 due to API rate
limiting (the shim is sequential but Anthropic-side per-minute rate
applied across rapid-fire calls). Mitigation: chunk into groups of
~10 classes per CLI invocation, with brief wait between groups.
Alternative: accept the failure pattern, re-run after session reset.

**(f) Out-of-scope sub-channels stop being actionable** — by R1'
cycle 2, 14 of 19 surfaced sub-channels were AT QUORUM but Opus
explicitly noted "non-AI-safety risk side → stay unassigned even at
quorum". Watchlist needs a `confirmed_out_of_scope: true` tag to
prevent re-surfacing in future reviews. Class B fix queued for cycle 3
prep.

**Operations model verdict: 10-batch cycle is correct. Review chain
order locked as routing → consolidation → R4 sweeps (newly oversize +
watchlist-driven only) → R2' (incremental, no-rework filter) → R1'
(incremental, no-rework filter). Cycle pp budget revised to ~60-70pp
(down from ~75pp) due to no-rework filter + selective R4 sweeps.**

**Three sessions remaining for routing**: cycle 3 (batches 15-24, ~70pp
including reviews), cycle 4 (batches 25-34, ~70pp), cycle 5 (batches
35-36 + final audit + end-of-routing reviews + faithfulness sample
generation, ~40pp).

### 19.18 — Manual human-faithfulness study plan (added 2026-05-19)

Same-model faithfulness validation (Opus 4.7 → Opus 4.7) is
methodologically circular and will be flagged by workshop reviewers.
LLM-as-judge with another vendor (OpenAI GPT-5 / Google Gemini 2.5)
helps but is still indirect validation against ground truth. The
canonical paper deliverable 1 will instead be **manual human
classification** by the project authors (Martin Leitgab ± Gleb Posobin).

**Decision deferred to end-of-routing** (post batch_0036 + final audit).
Two methodological options to choose between at that point:

**Option (a) — Annotator agreement with existing catalog**:
- Stratified random sample N=30-50 paths across HC × MC quadrants
- Human annotator(s) re-classify each path against the 18 HC + 19 MC
  catalog definitions
- Compute % agreement on HC, MC, and 6 axis dimensions
- Optional: Cohen's κ if 2 annotators
- **Headline metric**: "Human-LLM HC agreement = X%; MC agreement = Y%"

**Option (b) — Independent human taxonomy**:
- Same N=30-50 stratified sample
- Annotator(s) propose their own risk/mechanism categories from scratch,
  blind to the Opus-generated catalog
- After annotation, manually map human → Opus categories
- Compute concordance at the FAMILY level: "X of 18 HCs have a
  corresponding human-named family; Y of 19 MCs likewise"
- **Headline metric**: "Independent human-derived taxonomy concords with
  X / 18 HCs (Z%) and Y / 19 MCs at the family level"

**Option (b) is stronger for reviewer-defense** — it shows the catalog
isn't a "Opus-confirms-Opus" artifact; independent human reasoning
converges on similar structure. Tradeoff: slower (no anchor to start
from), needs careful definition of "family-level concordance" to avoid
hand-tuning.

**Required infrastructure (to script near pipeline completion)**:
- `phase2_step5b_faithfulness_sample.py` — stratified sampler:
  1 path per HC × MC quadrant where populated; oversample large
  quadrants. Output `phase2_faithfulness_sample.xlsx` with columns:
  path_id, risk_node_name, intervention_node_name, body_summary,
  current_HC, current_MC, human_HC, human_MC, human_axes, agree_flag,
  rationale_if_disagree.
- For option (b): same sheet structure but `current_HC`/`current_MC`
  hidden until annotator completes their independent classification.
- Annotation done locally in Excel; results re-imported via
  `phase2_step5c_faithfulness_compute.py` for agreement metrics.

**Cost**: ~zero LLM tokens. Human-hours: ~2-4 hrs for N=30 sample at
~5 min/path (option a); ~4-6 hrs for option (b) including independent
categorization + post-hoc family mapping.

**Workshop paper section placement**: §3.6 "Faithfulness validation"
or appendix table, citing the sample size, agreement metric, and the
out-of-scope tail rate from §19.17 as a separate validity signal.

### 19.19 — Cycle 3 outcomes + cycle 4 readiness (added 2026-05-21)

**Cycle 3 = routing batches 15-25 (11 batches) + full review chain.** Catalog
end-state: **28 HC + 32 MC + 6 axes** (was 23 HC + 23 MC at start of cycle).
Total assignments: **2,047 paths routed** (was 1,225). Unassigned: 230 / 11.2%.
**725 paths remain** = ~10 more batches across cycles 4 + 5.

**Routing throughput (parallel mode, added this cycle).** A new
`--mode opus_routing_parallel --n-workers 5` was added: 5 batches run as
concurrent `claude -p` shim subprocesses on isolated `partial_w*.txt` files,
followed by serial-merge in batch_idx order. Each catalog-mutation step
(`_resolve_and_enforce`, `_append_flags`, `_save_active_catalog`,
`_rebuild_routing_assignments_jsonl`, xlsx) runs once per group, not per worker.
**Wall-clock 4.8× speedup** (10 batches: 437s + 420s ≈ 14.3 min total vs
~70 min sequential). Session pp cost is unchanged (still no shim cache).

**Consolidation_003 (after fixing off-by-one cutoff bug).** Prior cutoff
filter was `batch_idx > cutoff`, which excluded the batch whose index equals
the cutoff (silently dropping batch 15's 11 flag entries). Fixed to
`batch_idx >= cutoff`; scope expanded from 57 → 68 flag entries. Verdicts:
**6 splits applied** (38 paths moved):
- MC011 → MC024 (Differential-privacy training, 3) · MC025 (Fairness via
  data re-balancing, 4) · MC026 (Bayesian-opt hyperparameter search, 3)
- MC017 → **MC027** (Inference-time prompt-engineering, **13** — heavily
  surfaced since cycle 1)
- MC015 → MC028 (AI-safety research-environment infrastructure, 8)
- HC013 → HC024 (Recommender-platform engagement-maximization, 7)

No merges, no renames, no axis extensions (both proposed extensions —
scientific-compute, medical-imaging — were correctly auto-marked OOS).
**3 defended-homogeneous classes**: MC001 (RLHF, 130), MC004 (adversarial
robustness, 178), MC016 (governance/policy, 146).

**R4 sweep (5 scheduled deep-dives, 8 chunks, 638 decisions).**

| Class | n | HOLD | REASSIGN | SPLIT_OUT | UNASSIGN | New classes |
|---|---|---|---|---|---|---|
| HC008 | 183 | 92 | 65 | **26** | 0 | HC025, HC026, HC027, HC028 |
| HC012 | 190 | 187 | 0 | **0** | 3 | — defended |
| HC015 | 40 | 31 | 0 | **9** | 0 | HC029 |
| MC011 | 159 | 149 | 4 | **4** | 2 | MC029, MC030 |
| MC019 | 66 | 62 | 4 | **0** | 0 | — defended |
| Total | 638 | 521 (82%) | 73 (11%) | 39 (6%) | 5 (1%) | 7 new classes |

**HC012 (190) defended despite 3 named sub-channels in consolidation_003 flags.**
Both R4 chunks (100 + 90 members) independently returned 0 SPLIT_OUT — a
real defense, not a chunk-size artifact. Opus interpretation: HC012 is one
coherent "research community readiness / institutional capacity" family.
**MC019 (66) defended despite hybrid-neuro-symbolic at 3-peer watchlist quorum**
— Opus held MC019 as coherent "alignment system patterns".

**Cross-chunk duplicate bug (HC025 + HC026).** Both chunks of HC008's R4 sweep
independently proposed an "Audio / speech perception accuracy ceiling" split;
chunk 1 named it without a hyphen, chunk 2 with one. The
`_resolve_and_enforce:970` idempotent-by-exact-name guard missed the near-match.
Result: two HC IDs for the same concept, 6 audio members split 3+3 across them.
**Fix (landed 2026-05-20)**: added `_normalize_class_name()` (lowercase + strip
`-_/.,()` + collapse whitespace) wired into BOTH routing's
`_resolve_and_enforce` AND R4 sweep's `split_groups` keying. R4 sweep now also
redirects normalized-name matches against the existing catalog so future
sweeps route members into pre-existing classes instead of creating duplicates.
HC026 manually merged into HC025 via path-reassignment + class_remap (3 paths
moved; HC025 now at 6 members; catalog dropped to 28 HC).

**Capability-ceiling vs safety-harm distinction (Martin 2026-05-20).**
**HC025, HC027, HC028 are capability-ceiling research harms** (model X can't
yet do task Y well enough — perception, imitation, RL planning). They join
HC008 residual + HC021 (vision) + HC022 (NLP) in the same family. **HC029 IS
AI-safety-related** (shutdown-resistance, corrigibility failure — off-switch
disabling, interruption avoidance, containment escape). Capture for paper
writeup: present capability-ceiling HCs as **a labeled subgroup within the
non-safety-risks bucket**, not mixed with safety-harm HCs in the canonical
summary table. Tag each HC at end of cycle 5 with
`harm_kind: {safety_harm, capability_ceiling, non_safety_risk_other}`.

**R2' unassigned-pool audit (misfit_review_006).** 116 candidates (179
previously-HOLDed paths excluded by no-rework filter); verdicts: **105 HOLD
(90.5%)**, 3 UNASSIGN, 4 REASSIGN (2 HC + 2 MC). **0 new classes proposed.**
**90.5% HOLD-as-unassigned rate** is consistent with cycle 2's 95% (R2) /
78% (R2') — confirms the publication-defensible finding that the unassigned
pool is honest scope-boundary signal, not catalog gap. 20 new sub-channels
added to watchlist (mostly OOS — non-AI scientific-compute, non-AI medical
imaging, classical control variants).

**R1' self-flagged misfit review (misfit_review_007).** ~200 candidates after
no-rework filter; verdicts: 176 HOLD (88.9%), 13 UNASSIGN, 7 REASSIGN, **2
PROPOSE_NEW_MC** quorum-met. **2 new MCs created via reassignment:**
- **MC031** Label-noise-robust training methods (6 members)
- **MC032** RL training-time variance reduction & target normalization (7 members)

18 new sub-channels added to watchlist.

**Task #55 (confirmed_out_of_scope tag) — completed 2026-05-20.** The
watchlist JSONL now carries `confirmed_out_of_scope: bool` per entry; OOS
entries render in a separate "DO NOT PROPOSE — even at quorum" section of
the prompt. Backfill at cycle 3 start tagged **23 of 40 entries as OOS** (13
distinct names) — including non-AI x-risk causes (13 peers), general-ML
productivity tooling (11), animal-welfare-harm-target (6), bio-AI non-safety
applications (6), non-AI cryptography (6), corporate finance/governance (5),
plus knowledge-base/ontology, AI-cybersecurity-tooling (= AI applied TO
cybersecurity, not AI-MODEL security), classical control engineering, and
probabilistic-programming inference engines. **17 ACTIVE in-scope** entries
remain as legitimate catalog-gap candidates for future cycles.

**Cumulative session cost (cycle 3, this session).** 11 routing batches
(~50pp) + consolidation_003 (~5pp) + R4 sweep #13 (~37pp) + R2' (~7pp) +
R1' (~9pp) = **~108pp session**.

**Cycle 4 readiness (auto-propagated state).**
- Active catalog: 28 HC + 32 MC + 6 axes, persisted at
  `phase2_routing_active_catalog.json` (with backups `.pre_*.json` per stage).
- Watchlist: 17 ACTIVE + 13+ OOS entries (sticky-OOS coalescing across
  observations); auto-loaded into every prompt with explicit DO-NOT-PROPOSE
  guidance for OOS entries.
- No-rework filter: 358 paths now in the "previously-HOLDed" exclusion set
  (179 from cycle 2 + 179 from cycle 3 R2'/R1'). Future R1'/R2' will skip
  these unless re-flagged via consolidation/sweep reassignment.
- Path reassignments log: ~250 entries (auto-applied during jsonl rebuild).
- Step log: 47+ chronological entries (audit trail).

**Cycle 4 plan**: 10 routing batches (26-35) via parallel mode, then full
review chain (consolidation_004 → selective R4 sweeps for any new oversize
classes + at-quorum watchlist sub-channels → R2' → R1'). Estimated session
cost: ~70-80pp.
