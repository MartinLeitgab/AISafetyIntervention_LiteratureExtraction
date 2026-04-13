# Phase 2 Step 5 — Findings Report

**Analysis date:** 2026-04-11  
**Config:** `consim1_pathbuildB` (≤1 consecutive SIM hop; PathbuildB frozenset co-occurrence families as L2)  
**Naming model:** gpt-5.4-mini (v2 rerun, updated from gpt-4o-mini)  
**Quality cuts applied:** SIM cos_sim ≥ 0.9 · EDGE confidence ≥ 3 · intervention maturity ≥ 3  
**Scripts:** `phase2_step5_naming.py` (v1), `phase2_step5_B2_naming_rerun.py` (v2), `phase2_step5_examples.py`, `phase2_step5_triplet_simreach.py`, `phase2_step5_subcluster_naming.py`  

---

## Executive Summary

Step 5 applied the selected config (`consim1_pathbuildB`) to produce four deliverables:

1. **LLM cluster naming (gpt-5.4-mini)** — all 120 clusters (40 risk + 40 intervention + 40 PathbuildB chain families), 34/40 risk high-conf, 29/40 intervention high-conf, 40/40 chain families high-conf
2. **Pathway examples** — top-15 R→I connections, 10 gap clusters, top-20 EDGE-only pairs, top-10 B-families
3. **Triplet SIM reach** — 15 top triplets: union paper reach and triplet-core (papers covering all 3 clusters simultaneously)
4. **Subcluster naming** — 36 parent clusters × k=5 AgglomerativeClustering + 2-pass LLM → 180 subclusters, 96.1% high confidence

**Key finding — PathbuildB chain families are mechanistically informative:** All 40 top PathbuildB chain families received high-confidence names that describe causal mechanisms (not risk re-statements). Example names: "Robust reward learning and specification gaming" (rank 4), "Adversarial robustness through PGD training" (rank 7), "Compute chokepoints enabling scalable governance" (rank 16). This contrasts with PathbuildA (KMeans chain clusters), where ~30/40 clusters collapse to variants of "Catastrophic AI misalignment" — see Part 4.

**I8 is the dominant meta-intervention:** appears in 9/15 top triplets; even narrow technical risks (e.g., R6 reward misspecification) reach I8 via 1,072 consim1 paths despite zero EDGE-only paths — a collective cross-paper inference the field has established but no single paper argues explicitly.

---

## Part 1: Cluster Naming Statistics

| Metric | Risk (v2) | Intervention (v2) | PathbuildB chains (v2) |
|--------|-----------|-------------------|----------------------|
| Total named | 40 | 40 | 40 |
| High confidence | **34** | **29** | **40** |
| Medium confidence | 5 | 11 | 0 |
| Low confidence | 1 | 0 | 0 |
| Judge-inaccurate | 1 | 2 | 0 |
| Split candidates | 11 | 17 | 0 |

All cluster naming uses gpt-5.4-mini. Full v2 outputs: `step5_naming/risk_cluster_names_llm_v2.csv`, `step5_naming/intervention_cluster_names_llm_v2.csv`, `step5_naming/pathbuildB_chain_names_llm.csv`.

---

## Part 2: Risk Cluster Names (v2, gpt-5.4-mini)

39/40 high confidence. Key artifact candidates: R13 (4 qualifying nodes) and R36 (3 nodes).

| Cluster | N nodes (qual.) | Name (v2) | Confidence |
|---------|----------------|-----------|-----------|
| R0 | 299 | Out-of-distribution generalization failure | high |
| R1 | 114 | Malicious misuse of frontier AI for large-scale harm | high |
| R2 | 112 | AI arms race causing unsafe deployment | high |
| R3 | 50 | Opaque transformer reasoning undermines safety oversight | high |
| R4 | 341 | Unsafe and sample-inefficient RL exploration | medium |
| R5 | 155 | Compute and data inefficiency in AI training | high |
| R6 | 214 | Reward misspecification and reward hacking | high |
| R7 | 142 | Human extinction from misaligned superintelligent AI | high |
| R8 | 126 | Human-AI coordination failure in shared environments | high |
| R9 | 219 | Deployed AI misalignment causes societal harm | high |
| R10 | 367 | Existential catastrophe from misaligned AI | high |
| R11 | 126 | Misforecasted AI timelines undermining safety preparedness | high |
| R12 | 102 | Catastrophic risks from misaligned advanced AI | high |
| R13 | 4 | Unverified or computationally infeasible HCH-style alignment | high ⚠️ artifact (4 nodes) |
| R14 | 158 | Unsafe AI decisions in high-stakes settings | medium |
| R15 | 121 | Opaque AI decision-making undermines oversight and safety | high |
| R16 | 269 | Insufficient AI safety research capacity | high |
| R17 | 116 | Adversarial image classification misclassification | high |
| R18 | 60 | Catastrophic superintelligent AI misalignment | high |
| R19 | 136 | Misaligned AI takeover and human disempowerment | high |
| R20 | 105 | Deceptive alignment and hidden objectives | high |
| R21 | 179 | Catastrophic AI misalignment and loss of control | high |
| R22 | 221 | Deceptive and harmful language model outputs | medium |
| R23 | 20 | Engagement-driven recommendation manipulates and polarizes users | high |
| R24 | 48 | Objective misspecification and misaligned optimization | medium |
| R25 | 223 | Misaligned superintelligent AI existential catastrophe | high |
| R26 | 235 | Misaligned AGI existential catastrophe | high |
| R27 | 44 | Uncontrolled recursive self-improvement explosion | high |
| R28 | 45 | Algorithmic discrimination in high-stakes decisions | high |
| R29 | 15 | AI-driven wealth and power concentration | medium |
| R30 | 39 | Existential catastrophe causing extinction risks | high |
| R31 | 59 | Undetected unsafe behavior in deployed AI systems | high |
| R32 | 23 | AI shutdown resistance and control loss | high |
| R33 | 49 | AI-driven personal data privacy breaches | high |
| R34 | 37 | Instrumental power-seeking by advanced AI agents | high |
| R35 | 168 | Advanced AI value misalignment risk | high |
| R36 | 3 | Unreliable and costly RL alignment transfer | high ⚠️ artifact (3 nodes) |
| R37 | 33 | AI-driven labor displacement and socioeconomic instability | high |
| R38 | 102 | Catastrophic AGI misalignment risk | high |
| R39 | 10 | Bias-driven progress distortion and stagnation | low |

**X-risk near-duplicate group:** R7, R10, R12, R18, R21, R24, R25, R26, R35, R38 (10 clusters, ~1,501 nodes) are variants of catastrophic/existential AI misalignment. High centroid similarity (0.92–0.94). Workshop option: merge to 3–4 sub-types (value misalignment, deceptive alignment, catastrophic control loss, misuse) or keep as-is noting the semantic concentration.

**Artifact candidates:** R13 (4 nodes) and R36 (3 nodes) — recommend excluding from reported taxonomy. The 38-cluster taxonomy is cleaner.

---

## Part 3: Intervention Cluster Names (v2, gpt-5.4-mini)

29/40 high confidence, 11 medium. Intervention type taxonomy below.

| Cluster | N nodes (qual.) | Name (v2) | Type | Confidence |
|---------|----------------|-----------|------|-----------|
| I0 | 124 | Training-time regularization for robust generalization | Technical | high |
| I1 | 105 | Continuous anomaly monitoring and safety tripwires | Technical | high |
| I2 | 76 | MCTS-guided RL planning and training | Technical | medium |
| I3 | 22 | Efficient, local, and controllable AI hardware deployment | Technical | medium |
| I4 | 199 | Pre-deployment safety gates and controlled release | Governance | high |
| I5 | 189 | Pre-deployment adversarial red-teaming | Governance | high |
| I6 | 111 | PGD adversarial training for robust fine-tuning | Technical | high |
| I7 | 12 | Structured human and synthetic data collection pipelines | Technical | high |
| I8 | 232 | Expand AI safety research funding and capacity | **Field-building** | high |
| I9 | 120 | Choose robust model architectures at design time | Technical | medium |
| I10 | 95 | Governed AI oversight and compliance regimes | Governance | high |
| I11 | 117 | Public AI risk outreach and educational dissemination | **Field-building** | medium |
| I12 | 14 | Pre-deployment bias auditing and fairness mitigation | Governance | high |
| I13 | 26 | Multi-objective recommender ranking with safety penalties | Technical | medium |
| I14 | 15 | Beginner-friendly AGI safety knowledge infrastructure | **Field-building** | high |
| I15 | 87 | Pre-deployment mechanistic interpretability audits | Technical | high |
| I16 | 25 | Ensemble-based uncertainty and robustness mitigation | Technical | high |
| I17 | 42 | Runtime control and coordination modules | Technical | high |
| I18 | 53 | Critique-guided fine-tuning and human feedback | Technical | medium |
| I19 | 80 | Demonstration-based policy pretraining | Technical | medium |
| I20 | 25 | Uniform experience replay for RL training | Technical | medium |
| I21 | 48 | Formal verification for pre-deployment AI safety | Governance | high |
| I22 | 69 | Deployment-time confidence filtering and rejection sampling | Technical | high |
| I23 | 101 | Human-aware predictive robot control and safety planning | Technical | medium |
| I24 | 78 | RLHF fine-tuning for aligned language models | Technical | high |
| I25 | 105 | Explainable decision support with uncertainty | Technical | high |
| I26 | 172 | Reward-guided RL fine-tuning with safety constraints | Technical | high |
| I27 | 23 | KL-regularized RLHF fine-tuning | Technical | high |
| I28 | 19 | International treaty banning autonomous weapons | Governance | high |
| I29 | 5 | Neuroscience validation platforms for cognition and emulation | Technical | medium |
| I30 | 17 | Periodic target Q-network updates for stable Q-learning | Technical | high |
| I31 | 28 | Human-AI collaboration tools for skill building | Technical | medium |
| I32 | 53 | Attainable utility preservation reward penalty | Technical | high |
| I33 | 34 | Deployment-time prompt engineering for aligned behavior | Technical | high |
| I34 | 71 | Large-scale pretraining before task fine-tuning | Technical | high |
| I35 | 162 | Human-preference reward model fine-tuning | Technical | high |
| I36 | 38 | Robustness-oriented training data augmentation | Technical | high |
| I37 | 6 | ROME-based factual knowledge editing | Technical | high |
| I38 | 68 | Incentivized AI forecasting and prediction systems | Technical | high |
| I39 | 42 | Export controls on advanced AI hardware | Governance | high |

### Intervention Type Summary

| Type | N clusters | N nodes (qual.) | Description |
|------|-----------|----------------|-------------|
| **Technical** | ~28 | ~1,550 | Algorithm changes, training procedures, architecture modifications |
| **Governance** | ~7 | ~625 | Policy, regulation, oversight, audits, treaties, standards |
| **Field-building** | ~3 | ~380 | Funding, education, outreach (I8, I11, I14) |

**I8 is the dominant field-building meta-intervention** — it appears in 9/15 top triplets and receives paths from nearly every risk cluster. Even narrowly technical risks motivate it via cross-paper semantic bridging. I8 is least actionable for technical practitioners but most broadly motivated in the literature; for technical analysis, I8 connections should be analyzed separately.

**I9 split candidate:** Subclustering confirms a clean split into SC0 (n=75, transformer architecture design) vs SC1 (n=21, memory optimization). Workshop option: report as two distinct interventions.

---

## Part 4: PathbuildB Chain Family Names

**Chain method A not used:** ~30/40 chain method A (KMeans) clusters collapse to variants of "Catastrophic AI misalignment risk" — they re-state the risk rather than describing the causal mechanism. This is a structural consequence of KMeans on mean body embeddings: mean-pooling across multi-step body nodes collapses mechanistic diversity into the dense misalignment-concept neighborhood of the embedding space. See Step4 Part 16.4 for the full assessment table.

**PathbuildB selected:** B-families group paths by their frozenset of `{(body_concept_subtype, cluster_id)}` combinations. There is no KMeans step — the grouping is structural, encoding which concept subtype clusters co-occur in a path. This captures the mechanistic diversity of reasoning chains.

**Top-20 PathbuildB chain family names (gpt-5.4-mini, "because [mechanism]" prompt):**

| Rank | N paths | Chain family name | Type |
|------|---------|-------------------|------|
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
| 13 | 429 | Building AI safety capacity through funded training | Field-building |
| 14 | 403 | Grant-driven expansion of AI safety research | Field-building |
| 15 | 399 | Capacity-building through funded safety talent and oversight | Field-building |
| 16 | 394 | **Compute chokepoints enabling scalable governance** | Governance |
| 17 | 391 | **Utility-shaping and power-seeking mitigation mechanisms** | Technical |
| 18 | 383 | **Global governance enables coordinated AI safety controls** | Governance |
| 19 | 379 | Human feedback reward-model alignment learning | Technical |
| 20 | 377 | Human feedback reward-model alignment | Technical |

Full 40-family table: `step5_naming/pathbuildB_chain_names_llm_v2.csv` (v2); `step5_naming/pathbuildB_chain_names_llm_v3.csv` (v3, causal framing — see Part 9)

**Why ranks 1–3 and 8–11, 13–15 are all field-building variants:** These families all contain the dominant B-family signature or close variants, reflecting that funding/talent/capacity-building is the most common intermediate reasoning chain in the corpus. Starting from rank 4, mechanistically distinct chains emerge.

**Note on MF title / intervention name overlap:** Some B-family names are semantically near-identical to intervention cluster names. For example, B-family "Funding and training to build AI safety capacity" closely matches I8 "Expand AI safety research funding and capacity." This is expected: in field-building chains, the intermediate body reasoning and the intervention endpoint name the same concept. For technical/governance B-families, body names describe mechanisms distinct from the intervention endpoint (e.g., "robust reward specification and policy generalization" is not the same as any single intervention cluster name). Audience note: apparent duplication between an MF name and an intervention name signals a direct field-building chain.

**Corpus structure finding from PathbuildA collapse:** The fact that PathbuildA collapses is itself a finding — intermediate reasoning in AI safety literature is dominated by misalignment semantics, reflecting the field's focus on a single risk paradigm. PathbuildB reveals more diverse mechanistic structure below this surface.

---

## Part 5: Pathway Examples

All examples from consim1 qualifying paths (VPN-filtered, maturity≥3 endpoint).

### Top R→I Connections by consim1 Path Count

| Connection | N consim1 paths | N EDGE-only paths | Dominant B-family | Source |
|---|---|---|---|---|
| R10→I8 (x-risk → fund AI safety) | **6,632** | 25 | "Funding and training to build AI safety capacity" | EA Forum |
| R25→I8 (misaligned superintelligence → fund AI safety) | 3,391 | 24 | Same | aisafety.info |
| R26→I8 (AGI misalignment → fund AI safety) | 2,715 | 14 | Same | aisafety.info |
| R21→I8 (catastrophic misalignment → fund AI safety) | 2,680 | 6 | Same | aisafety.info |
| R16→I8 (insufficient AI safety capacity → fund AI safety) | 2,580 | 86 | Same | Google Docs |
| R10→I35 (x-risk → RLHF reward models) | 1,198 | 4 | "Human feedback loops for objective alignment" | MIRI newsletter |
| **R6→I8 (reward misspecification → fund AI safety)** | **1,072** | **0** | — | *SIM-bridged only* |
| **R16→I35 (insufficient safety capacity → RLHF)** | **684** | **0** | — | *SIM-bridged only* |

Bold: SIM-bridged-only connections — collectively established but not argued end-to-end in any single paper.

### EDGE-only Top Pairs (Single-Paper Argument Chains)

| Risk cluster | Intervention cluster | N EDGE-only paths |
|---|---|---|
| R4 (RL unsafe exploration) | I26 (Safe RL fine-tuning) | **133** |
| R16 (Insufficient AI safety talent) | I8 (Fund AI safety) | 87 |
| R0 (OOD generalization failure) | I0 (Regularization) | 79 |
| R0 | I34 (Large-scale pretraining) | 70 |
| R8 (Human–AI coordination failure) | I23 (Uncertainty-aware control) | 65 |
| R6 (Reward misspecification) | I26 | 58 |
| R4 | I35 (RLHF reward models) | 56 |
| R6 | I35 | 52 |
| R4 | I19 (Demonstration-based pretraining) | 50 |
| R9 (Deployed AI misalignment) | I4 (Pre-deployment safety review) | 48 |

R4→I26 (133 paths) is the field's single most frequently argued complete risk→intervention chain in any individual paper.

### Gap Clusters (10 Risk Clusters with Fewest consim1 Paths)

| Risk cluster | Total consim1 paths | Has EDGE-only path? |
|---|---|---|
| R36 (3 nodes — artifact) | **5** | ✅ |
| R13 (4 nodes — artifact) | **8** | ✅ |
| R29 (AI-driven power concentration) | 38 | ✅ |
| R39 (Bias/stagnation) | 39 | ✅ |
| R23 (Engagement-driven recommenders) | 56 | ✅ |
| R27 (Recursive self-improvement) | 63 | ✅ |
| R37 (AI-driven job displacement) | 98 | ✅ |
| R33 (AI privacy breaches) | 120 | ✅ |
| R31 (Undetected harmful failures) | 130 | ✅ |
| R3 (Opaque transformer reasoning) | 203 | ✅ |

All 10 gap clusters have at least one EDGE-only path — thin but not absent coverage.

---

## Part 6: Triplet SIM Reach Analysis

**Setup:** 15 top R→I triplets. For each: union reach (distinct paper URLs reachable via SIM≥0.9 from any node in the 3 clusters) and triplet core (papers with SIM connections to nodes in ALL three clusters).

### Ranked by Triplet Core (Papers Covering All 3 Simultaneously)

| Rank | Triplet | Triplet core | Union reach | N consim1 paths |
|------|---------|-------------|-------------|----------------|
| 1 | R10→C10→I8 (x-risk → misalignment pathway → fund AI safety) | **49** | 1,639 | 6,632 |
| 2 | R25→C19→I8 | 37 | 1,364 | 3,391 |
| 3 | R26→C19→I8 | 33 | 1,399 | 2,715 |
| 4 | R21→C31→I8 | 30 | 1,693 | 2,680 |
| 5 | R10→C25→I35 (x-risk → objective misalignment → RLHF) | 21 | 1,583 | 1,198 |

**Key interpretations:**
- R10→C10→I8 is the most comprehensively documented triplet: 49 papers discuss all 3 clusters simultaneously. The AI safety field-building argument is the backbone of the corpus.
- R6→(chain)→I8 (reward misspecification → fund AI safety): triplet core = 0 despite 1,072 consim1 paths. Entirely cross-paper implicit — a synthesis gap where no paper argues the full chain.
- High union reach with low triplet core (e.g., R10→C18→I4: union 1,704, core 2) indicates many adjacent papers but few co-citing the full chain.

Full table: `step5_naming/triplet_simreach.csv`

---

## Part 7: Subcluster Naming (Step 5d)

36 parent clusters triggered subclustering (24 risk, 12 intervention) by size criterion (`cluster_size > 100`). Re-clustered at k=5, named via 2-pass LLM.

| Metric | Value |
|--------|-------|
| Parent clusters processed | 36 |
| Total subclusters named | 180 |
| High confidence | **173** (96.1%) |
| Medium confidence | 6 (3.3%) |
| Low confidence | 1 (0.6%) |

**Structural finding:** 35/36 parents produce exactly 1 large dominant subcluster plus 4 tiny outliers. Confirms the k=40 parent taxonomy is semantically tight — very few clusters benefit from further subdivision.

**Sole genuine split — I9:** SC0 (n=75, transformer architecture design) + SC1 (n=21, memory optimization). Workshop option: report as two distinct interventions.

Full results: `step4_subclusters/subcluster_names_llm.csv` · `step5_naming/subcluster_naming_detail.csv`

---

## Part 8: Visualization Reference — UMAP vs MDS

**UMAP plots** (`phase2_step4_umap_plots.py`, Step4 `phase2_step4_B9_umap_regen.py`):
- `step4_finalanalysis/umap_risks_consim{0,1,2}.png` and `step4_finalanalysis/umap_interventions_consim{0,1,2}.png`: All VPN nodes (~19,791 for consim1), cosine metric, colored by cluster assignment. Show **local neighborhood structure** of individual nodes in embedding space.
- `step4_finalanalysis/umap_interventions_consim1_clusters.png`: Max 200 sampled nodes/cluster, euclidean metric on L2-normalized vectors, cluster name labels at centroids. Designed for readability with labels; looks different from the full-node UMAP because (1) sample, (2) euclidean vs cosine metric, (3) label overlay.

**MDS plots** (centroid-level only):
- `step4_metaclusters/risk_2d_mds.png`, `step4_metaclusters/interv_2d_mds.png`: 40×2D projection using classical MDS on inter-centroid cosine distance matrix. Each point = one cluster centroid (40 points total). Preserves **global pairwise distances between cluster centers**, not individual node layout. Use for showing macro-structure of the taxonomy.
- `step4_cluster_tables/pathbuildB_metafamily_2d_mds.png`: Same for 32 meta-families, distances = 1-Jaccard component similarity.

For publication: use UMAP figures to illustrate semantic distribution and cluster geometry; use MDS figures to show taxonomy structure and meta-cluster relationships.

---

## Part 9: Rev5 Additions (2026-04-12)

### PathbuildB chain naming v3 — causal framing

`step5_naming/pathbuildB_chain_names_llm_v3.csv` (40 rows, 39/40 high confidence) uses causal mechanism noun phrases that complete: "The reason why [intervention X] mitigates [risk Z] is [v3 name]." No "via"/"through" prefixes. Examples:
- B3: "robust reward specification and policy generalization"
- B4: "robust corrigibility and verifiable containment mechanisms"
- B5: "alignment of optimization with human intent"
- B6: "robust feature reliance induced by adversarial training"
- B15: "constrained global proliferation of advanced AI compute"

These names are more suitable for publication than v2 ("via ...") names because they describe mechanisms as noun phrases that fit naturally in explanatory prose.

### CSV encoding cleanup

All `step4_finalanalysis/` and `step5_naming/` CSV files cleaned of non-ASCII characters (arrows, dashes, curly quotes) that render as garbage in Excel/CSV readers. 22 files updated; all are now ASCII-safe.

---

## Part 10: Strategic Recommendations for Workshop Paper

### What to Report

| Paper section | What to use | Source |
|---|---|---|
| L1 Risk taxonomy | 38 clusters (exclude R13/R36 artifacts); updated v2 names | Part 2 |
| L3 Intervention taxonomy | 40 clusters with type taxonomy (technical/governance/field-building); note I9 split | Part 3 |
| L2 Chain taxonomy (qualitative) | Top-10 distinct B-family names (ranks 4–7, 12, 16–18, 25, 35) | Part 4 |
| L2 Chain taxonomy (quantitative) | 1,603 consim1 B-families; 51 EDGE-only | Step4 Part 1 |
| Zero connectivity gaps | Confirmed — every risk cluster reaches every intervention at consim2 | Step4 Part 2 |
| Cross-paper bridging | 53.1% of R→I pairs exist only via SIM bridges (685/1,289 pairs) | Step4 Part 13 |
| I8 field-building meta-intervention | Dominant; motivated by all risk clusters including narrow technical risks | Parts 3, 5, 6 |
| Novel SIM-bridged pairs | Top-20 non-field-building pairs (technical + governance) | Step4 Part 13 |
| Meta-cluster structure | 3 risk blocks (x-risk, RL/robotics, ML reliability) + 2 intervention blocks | Step4 Part 11 |
| Flagship example chain | R10→C10→I8 (49-paper triplet core, 6,632 paths, EDGE-only example from EA Forum) | Part 5 |

### Limitations to Acknowledge

- **Corpus bias toward x-risk:** 10+ near-duplicate x-risk clusters reflect source literature composition; the taxonomy is descriptive of the existing literature, not prescriptive of the actual risk distribution.
- **Field-building dominance in B-families:** Ranks 1–3 and 8–11 of B-families are funding/capacity variants, reflecting corpus bias toward I8. The mechanistically interesting families start at rank 4. Source diversity confirmed: top funding families draw from 173–245 distinct papers — not duplicates.
- **Chain clustering method A collapse:** On the consim1 qualifying path set (74,921 paths, same as PathbuildB), chain clustering method A (KMeans on mean body embeddings) places 92.9% of qualifying paths in misalignment-collapse clusters with no mechanistic label. Only 2 substantive mechanistic clusters survive (C0 reward misspecification at 2,221 paths; C15 generalization at 2,720 paths). Chain clustering method B (PathbuildB frozenset co-occurrence) is the definitive L2 representation. The collapse is itself a corpus finding: intermediate reasoning in AI safety literature is semantically dominated by misalignment concepts at the mean-embedding level; PathbuildB reveals the mechanistic diversity that exists structurally below this surface. See Step4 Part 16 for full assessment.
- **Both chain clustering methods operate on the identical qualifying path set** (SIM≥0.9, EDGE conf≥3, maturity≥3, consim1). The difference is exclusively in how body/chain nodes are clustered, not in path construction.
- **SIM bridging limitation:** 53.1% of R→I pairs established via cross-paper SIM bridging lack single-paper end-to-end arguments. These connections are valid collective inferences but their strength relative to EDGE-grounded connections should be qualified.
- **Interactive visualizations:** Six interactive Plotly HTML files are available alongside the static PNGs (heatmaps, dendrograms, meta-connectivity network, three-layer network). See Step4 Part 16.7 for the full file listing.
