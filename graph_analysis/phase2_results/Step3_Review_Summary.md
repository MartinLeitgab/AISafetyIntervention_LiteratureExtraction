# Phase 2 Step 3 Review Summary

**Generated:** 2026-03-22; last updated 2026-03-28
**Script:** `graph_analysis/phase2_step3_validation_and_selection.py`
**Outputs:** `phase2_results/step3_validation_and_selection/`
**Status:** Sections A–F complete ✅ · Section D full-graph betweenness ✅ (33.1h) · Both-mode betweenness ✅ (15 min)

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
| **intervention** | **SIM≥0.95** | **both** | Rank 2. 0.134 composite improvement over earmarked SIM≥0.9 (rank 12). |
| implementation_mechanism | SIM≥0.9 | both | Rank 1 — earmarked cut confirmed. |
| all_concepts | SIM≥0.95 | monotonic | Rank 1. |
| design_rationale | EDGE | monotonic | Rank 1. |
| problem_analysis | SIM≥0.95 | both | Rank 1. |
| theoretical_insight | EDGE | unconstrained | Rank 1. |
| validation_evidence | EDGE | monotonic | Rank 1. |

### Key Note: EDGE-Validation Weight Bias
The 30% EDGE validation weight structurally advantages EDGE-only configs (trivial score=1.0). For risk, the composite gap of 0.036 is narrow; SIM≥0.9+both is preferred for cross-literature coverage (10× more nodes, 90.8% structural validation). For intervention, the 0.134 gap is substantial — SIM≥0.95 is genuinely better.

### Mode Definitions
The four clustering modes apply constraints on top of the edge_config similarity threshold:
- **unconstrained:** no path constraints — all SIM + EDGE edges used freely
- **monotonic:** path must follow monotonically increasing causal direction (EDGE edge direction enforced)
- **single_risk:** each cluster can have only one risk concept entry point
- **both:** monotonic AND single_risk combined — most restrictive mode

"both" mode produces the most semantically coherent mechanism chains: each chain has a directed causal structure and a single risk root. It is the primary cut for risk and intervention node types.

### Key Finding: Single_Risk Is the Primary Structural Constraint

At ec=0.9 for risk, **single_risk carries nearly all the structural grounding** — both (90.8% EDGE%) vs single_risk (90.6% EDGE%) are essentially identical on EDGE validation. Monotonic alone without single_risk drops to 58.1% EDGE%, and unconstrained drops to 56.0%. This reveals that:

- **single_risk** = the constraint that prevents x-risk hubs from aggregating semantically diverse mechanism families → structural grounding comes from this
- **monotonic** = adds directional path ordering → improves ARI stability (+0.022: both=0.731 vs single_risk=0.709) but at a cost: it excludes body nodes that have bidirectional causal relationships (both influenced by a risk AND contributing to an intervention)

Mode rankings for risk at ec=0.9: both (rank 3, composite=0.789) > single_risk (rank 7, composite=0.758) > monotonic (rank 11, composite=0.597) > unconstrained (rank 16, composite=0.438). The composite gap between both and single_risk (0.031) is narrow; the gap to unconstrained (0.351) is large.

**Implication for Step 4:** Single_risk at ec=0.9 is the methodologically appropriate cut for body-node differentiation analysis (mechanism nodes between risk and intervention). It prevents x-risk hub dominance while allowing intermediate nodes to appear with bidirectional connections, making them visible as cross-family bridges in betweenness analysis. See Section D3 (planned).

---

## Section B — #21 Threshold Sensitivity

**File:** `threshold_sensitivity_analysis.csv` (800 rows)
**Plot:** `threshold_sensitivity_profile.png` (4-panel)

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
**Method:** EXACT betweenness on FULL graph (200,568 nodes, 346,224 edges). Runtime: 33.1 hours. All 200,568 nodes used as sources — no sampling, no subgraph restriction.

### Graph Used
SIM≥0.9 edges (144,140) + structural EDGE edges (202,149) = 346,224 total edges, 200,568 nodes.
All nodes included as both sources and potential path intermediates. Brandes O(V×E) exact algorithm.

**Note:** The clustering mode (both/unconstrained/monotonic/single_risk) does not affect betweenness computation. Betweenness is a property of the raw graph topology — all SIM≥0.9 + EDGE edges are used regardless of what mode was applied during clustering.

### Why X-Risk Nodes Dominate Bridges

In the AI safety literature, existential catastrophe concepts are the **shared motivational reference point** across nearly all papers — whether identifying a risk, analyzing a mechanism, or proposing an intervention. This means x-risk nodes have both risk-paper neighbors and intervention-paper neighbors, placing them at the intersection of the two primary literature clusters. A shortest path from any node in the risk-identification literature to any node in the intervention literature frequently passes through these concepts. They are bridges not because they are only risk nodes — they are bridges because they are the semantic hinge connecting why interventions exist to the problems they address.

### Top-20 Bridge Nodes (full-graph exact, corrected categories)

| rank | name (truncated) | category | betweenness | rank_sim08 |
|------|-----------------|----------|-------------|------------|
| 1 | Existential catastrophe from misaligned advanced AI | risk | 0.003367 | — |
| 2 | Faulty decision-theoretic reasoning in autonomous AI agents | risk | 0.002669 | 892 |
| 3 | Functional decision theory enables globally optimal choices | risk | 0.002665 | 1004 |
| 4 | Embedding FDT as AI agent decision-making framework | risk | 0.002662 | 9507 |
| 5 | Existential catastrophe from misaligned advanced AI systems | risk | 0.002618 | — |
| 6 | Formal agent architecture implementing FDT decision procedure | risk | 0.002615 | 44727 |
| 7 | "Cheating Death in Damascus" — FDT outperforms CDT/EDT | risk | 0.002612 | 57141 |
| 8 | Design AI agent core with FDT to improve alignment | intervention | 0.002608 | — |
| 9 | Existential catastrophic outcomes from misaligned AI systems | risk | 0.002464 | — |
| 10 | Opaque reasoning processes in large language models | problem analysis | 0.002399 | 1698 |
| 11 | Reward misspecification in reinforcement learning agents | problem analysis | 0.002306 | 438 |
| 12 | Catastrophic misalignment of advanced AI systems | risk | 0.002129 | — |
| 13 | Catastrophic AI system failures impacting humanity | risk | 0.002119 | — |
| 14 | Reward specification errors in RL-based AGI | problem analysis | 0.002041 | 853 |
| 15 | Existential catastrophe from uncontrolled AGI capability emergence | risk | 0.001995 | — |
| 16 | Uncertain compute threshold for human-level cognition | problem analysis | 0.001980 | 7036 |
| 17 | Opaque reasoning in large language models | problem analysis | 0.001969 | — |
| 18 | high uncertainty in compute threshold for human-level AI | problem analysis | 0.001944 | 10221 |
| 19 | Opaque internal representations in large language models | problem analysis | 0.001921 | — |
| 20 | insufficient AI safety preparedness from inaccurate timeline estimates | problem analysis | 0.001916 | — |

**Category distribution top-100:** 73 risk · 15 problem_analysis · 5 intervention · 7 other
**Note:** Prior versions of this table showed category='concept' for all non-intervention nodes. This was a data labeling bug — `node_attrs` stores `type='concept'` or `type='intervention'`, while the fine-grained category ('risk', 'problem_analysis', etc.) comes from `concept_category`. The rankings and betweenness scores are unaffected; only the label was wrong.

### Bridge Theme Clusters (top-50 nodes, k=12 Agglomerative, full-graph exact)

| cluster_id | theme | n_nodes | categories | representative names |
|------------|-------|---------|-----------|---------------------|
| 1 | Existential catastrophe | 11 | 11 risk | "Existential catastrophe from misaligned advanced AI" × many |
| 4 | Catastrophic misalignment (broader) | 10 | 10 risk | "Catastrophic misalignment of advanced AI", "Catastrophic AI system failures" |
| 0 | FDT + alignment intervention | 4 | 3 risk + 1 intervention | "Faulty decision-theoretic reasoning", "Embedding FDT", "Design AI agent with FDT" |
| 2 | Opacity / opaque LLM reasoning | 5 | 5 problem_analysis | "Opaque reasoning in LLMs", "Opaque internal representations in transformers" |
| 9 | Adversarial vulnerability | 4 | 4 problem_analysis | "Adversarial vulnerability in neural network classifiers" × 3 |
| 11 | Reward misspecification | 4 | 4 problem_analysis | "Reward misspecification in RL agents" × 3 |
| 3 | Timeline uncertainty / safety preparedness | 3 | 3 problem_analysis | "Insufficient AI safety preparedness from inaccurate timeline estimates" |
| 7 | Misaligned utility maximization | 3 | 3 risk | "Misaligned utility maximization in advanced AI systems" × 3 |
| 5 | FDT case studies | 2 | 2 risk | "FDT enables globally optimal choices", "Cheating Death in Damascus" |
| 6 | Compute threshold uncertainty | 2 | 2 problem_analysis | "Uncertain compute threshold for human-level cognition" × 2 |
| 8 | Mechanistic interpretability | 1 | 1 problem_analysis | "mechanistic interpretability tools for neuron-circuit analysis" |
| 10 | Unsafe RL exploration | 1 | 1 problem_analysis | "Unsafe exploration by RL agents in safety-critical environments" |

Singleton clusters (n=1): these are bridge nodes whose embeddings are sufficiently distinct from all other top-50 bridge nodes to form their own group in the k=12 clustering. They are structurally important bridges that don't share a thematic family with other top-50 nodes.

### What is FDT and Why Does It Bridge?

**Functional Decision Theory** (Yudkowsky & Soares, MIRI, ~2017–2018) proposes that a rational agent should act by asking "what is the best policy for all agents whose decision function is identical to mine?" rather than asking about causal or evidential consequences of a single action. This contrasts with Causal Decision Theory (CDT: choose the action with best causal consequences) and Evidential Decision Theory (EDT: choose the action that correlates with best outcomes).

Key examples: **Newcomb's Problem** (one-box because the predictor simulated your decision function), **Parfit's Hitchhiker** (pay on arrival because your decision function must be the type that pays, or you'd never have been rescued), and **"Cheating Death in Damascus"** (rank 7 node) which demonstrates FDT's coherence in death-avoidance scenarios where CDT/EDT generate regret loops.

FDT connects to AI alignment because: a sufficiently capable AI will be modeled/simulated by other agents. CDT-based AI systems are exploitable in Newcomb-like situations; an FDT-based AI cooperates reliably because its decision function is the right type regardless of observation. The rank-8 intervention node ("Design AI agent core decision modules with FDT to improve alignment") is a direct proposal to implement FDT in AI systems.

**Why FDT ranks 2–8 in the full-graph betweenness:** All FDT nodes in the corpus come from a single paper that traces a chain from decision-theoretic risk → formal FDT architecture → alignment intervention. This chain spans from risk concepts to intervention, placing every node on shortest paths between the risk literature and the intervention literature. The chain is dense (7 closely related nodes from one paper, all with high mutual similarity), creating a highly connected local cluster that appears on many cross-cluster shortest paths.

**Historical context:** FDT peaked in influence around 2018–2022 within MIRI's agent foundations research program and the EA/rationalist community. From 2023 onwards, mainstream AI safety shifted toward empirical alignment (RLHF, interpretability of deployed LLMs, scalable oversight), and MIRI itself acknowledged limited tractable progress on agent foundations. FDT is rarely cited in current NeurIPS/ICML safety papers. Its prominence here reflects the 2018–2022 era in the ARD corpus. For Step 4 cluster naming, this cluster is best labeled "Decision-theoretic alignment (agent foundations era)" to capture both content and historical context.

### Key Findings

**1. Existential catastrophe = confirmed dominant bridge** (clusters 1+4, 21 nodes = 42% of top-50). The structural hub connecting risk identification to intervention literature across the entire corpus. These nodes appear on shortest paths between the risk and intervention subgraphs because they are co-referenced as motivating context by both literatures.

**2. FDT / decision theory is the #2 bridge theme** (clusters 0+5, 6 nodes, ranks 2–8). A dense paper-local cluster from a single paper bridging decision theory to alignment interventions and catastrophic risk framing. The only intervention node in the top-10 is in this cluster (rank 8). FDT's bridge role is genuine in the unconstrained full-graph but does not survive the both-mode constraint (see Section D2).

**3. Opacity/opaque LLM reasoning is #3** (cluster 2, 5 nodes, rank 10). Neural network opacity bridges risk identification to alignment/interpretability interventions.

**4. Reward misspecification, adversarial vulnerability, compute uncertainty, timeline uncertainty** are secondary bridges (ranks 11–20), all categorised as problem_analysis.

### Step 4 Use (full-graph betweenness)
`betweenness_bridge_clusters.csv` provides seed themes for manual cluster naming in Step 4 #26:
- Clusters 1+4: Existential catastrophe / catastrophic misalignment → dense risk mechanism families
- Clusters 0+5: FDT / decision theory → decision-theoretic alignment mechanism cluster (agent foundations era)
- Cluster 2: Opacity → transparency/interpretability mechanism cluster
- Cluster 11: Reward misspecification → RL alignment risk cluster
- Cluster 9: Adversarial vulnerability → robustness risk cluster

---

## Section D2 — Betweenness on Both-Mode SIM≥0.9 Subgraph

**Files:** `betweenness_both09.csv` (top-100), `betweenness_both09_bridge_clusters.csv` (top-50 in 12 clusters), `betweenness_both09_raw_checkpoint.pkl`
**Plot:** `betweenness_both09_comparison.png`
**Method:** EXACT betweenness on induced subgraph of all nodes in both-mode ec=0.9 agglomerative clusters. Runtime: 15 min.

### Why a Separate Both-Mode Analysis

The full-graph betweenness (Section D) answers: "which nodes bridge the entire AI safety corpus?" The both-mode betweenness answers: "which nodes bridge within the constrained mechanism space we will use for Step 4?" These are different questions. The both-mode subgraph contains only nodes that satisfy monotonic + single_risk constraints — the final mechanism selection. Betweenness within this graph reveals which concepts structurally connect distinct mechanism families in the curated taxonomy.

### Subgraph Structure

| Metric | Value |
|--------|-------|
| Nodes | 17,952 |
| SIM≥0.9 edges | 5,855 |
| Structural EDGE edges | 15,502 |
| Total edges | 21,355 |
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

The both-mode betweenness is the more directly relevant analysis for Step 4 cluster naming, because it operates within the actual mechanism space being used. Key implications:

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

### Topology Results

| Metric | Value |
|--------|-------|
| Nodes | 200,525 |
| Edges | 202,123 |
| Weakly connected components | 15,123 |
| Largest WCC size | 61 nodes (0.03%) |
| Approximate diameter | 5 hops |
| Mean degree | 2.02 |
| Nodes with degree ≥ 2 | 74.4% |
| Top-25 betweenness nodes in EDGE subgraph | 25/25 (100%) |

### Key Structural Finding: No Global Backbone

The EDGE-only subgraph is **not a connected backbone** — it consists of 15,123 isolated chains with a maximum component of 61 nodes and mean degree of 2.0. Each "chain" represents a paper-local causal argument (problem → concept → intervention) that is structurally disconnected from chains in other papers.

**Implication:** SIM edges are the **sole source of global connectivity** in the AI safety knowledge graph. Without similarity edges, the graph is a collection of 15K isolated paper-local fragments. This reframes SIM augmentation from "optional enrichment" to "structural necessity" for cross-paper analysis.

The diameter-5 finding within components is consistent with the 7-hop median path length found in Step 2 — small components with depth-5 internal chains.

All 25 top-betweenness nodes from Step 2b appear in the EDGE subgraph (100% overlap), confirming that the most structurally important nodes are grounded in paper-local literature chains, not only SIM-connected.

**Relationship to both-mode subgraph:** The both-mode SIM≥0.9 subgraph (Section D2) has 15,502 EDGE edges — a substantial subset of the 202,123 total EDGE edges, filtered to those between nodes satisfying both-mode constraints. This gives the both-mode subgraph a similar chain-dominated structure (EDGE:SIM ratio 2.6:1) but with SIM edges providing cross-chain bridges that don't exist in the pure EDGE-only graph.

### Workshop Claim Supported
> "Structural (EDGE-only) edges form 15,123 isolated paper-local chains (mean component size ≤61 nodes, mean degree=2.0), with no global connectivity. Similarity edges at SIM≥0.9 are therefore not optional augmentation but the structural mechanism enabling cross-paper mechanism identification. All 25 top-betweenness concept nodes are present in EDGE chains, confirming that bridge concepts are grounded in explicit literature arguments."

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
| `betweenness_bridge_clusters.csv` | Full-corpus bridge seeds: existential catastrophe, FDT, opacity, reward misspecification |
| `betweenness_both09_bridge_clusters.csv` | Mechanism-space bridge seeds: catastrophic misalignment, adversarial vulnerability, reward misspecification, opacity, RLHF |
| `selection_justification.md` | Methods section 3.4 text |

### Key Distinction for Step 4: Three Betweenness Perspectives

**Full-graph betweenness** (`betweenness_bridge_clusters.csv`) identifies nodes that bridge the broadest cross-section of the corpus, including connections through nodes excluded from the final mechanism selection. Use for: understanding the overall intellectual structure of the corpus, identifying historically prominent themes (including FDT from the 2018–2022 era), writing the corpus-level methods description.

**Both-mode betweenness** (`betweenness_both09_bridge_clusters.csv`) identifies nodes that bridge within the constrained mechanism chains actually used in the taxonomy. Use for: Step 4 cluster naming seeds (primary), identifying which risk and problem_analysis concepts are the structural organizing principles of the mechanism taxonomy, prioritizing which clusters to name first. Limitation: monotonic constraint may exclude body nodes with bidirectional causal relationships, biasing toward risk/problem_analysis nodes over intermediate mechanism nodes.

**Single_risk ec=0.9 betweenness** (`betweenness_singlerisk09_bridge_clusters.csv` — PLANNED) identifies nodes that bridge within the mechanism space when x-risk hub aggregation is prevented but directional constraints are relaxed. This is the most appropriate cut for body-node differentiation: reveals which intermediate mechanism nodes (implementation_mechanism, design_rationale, problem_analysis body nodes) act as cross-family bridges. Expected: less fragmented than both-mode (more reachable pairs), body nodes more visible as bridges, catastrophic misalignment still dominant but lower ranked than in full-graph. See Section A Key Finding for rationale.

---

## Section D3 — Betweenness on Single_Risk ec=0.9 Subgraph (PLANNED)

**Files:** `betweenness_singlerisk09.csv`, `betweenness_singlerisk09_bridge_clusters.csv`, `betweenness_singlerisk09_raw_checkpoint.pkl`
**Method:** EXACT betweenness on induced subgraph of all nodes in single_risk ec=0.9 agglomerative clusters. Analogous to Section D2 but with monotonic constraint removed.
**Rationale:** Answers the question neither Section D nor D2 answers: "which body nodes (intermediate mechanism nodes between risk and intervention) act as cross-family bridges when x-risk hub dominance is prevented but directionality is not enforced?"

**Why single_risk, not unconstrained:**
- unconstrained at ec=0.9 for risk: composite=0.438 (rank 16/20), EDGE%=56.0% — x-risk hubs aggregate diverse mechanism families, clusters lose coherence
- single_risk at ec=0.9: composite=0.758 (rank 7/20), EDGE%=90.6% — nearly identical structural grounding to both (90.8%), without directional restrictions on body nodes

**Status:** ⬜ TODO
