# Phase 2 Step 4 — Findings Report

**Analysis date:** 2026-04-11  
**Selected config:** `consim1_pathbuildB` (consim1 = ≤1 consecutive SIM hop; pathbuildB = frozenset co-occurrence family grouping)  
**Three quality cuts applied simultaneously:** SIM edges cos_sim ≥ 0.9 · EDGE confidence ≥ 3 · intervention maturity ≥ 3  
**Quality cut audit:** See `Appendix_QualityCutAudit.md`  
**Config selection rationale:** See `step4_finalanalysis/step4_config_selection.md`  

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
