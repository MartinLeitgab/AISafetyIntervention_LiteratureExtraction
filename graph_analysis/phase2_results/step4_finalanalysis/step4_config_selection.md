# Step 4 Config Selection (Revised v2)

**Analysis script:** `phase2_step4_trackA.py` (output: `config_selection_metrics_v2.csv`)

---

## Config Framework

| Config | Max consec SIM | N paths | N unique VPN nodes |
|--------|----------------|---------|-------------------|
| `consim0` | 0 (edge-only) | 3,473 | 17,136 |
| `consim1` | ≤1 | 75,008 | 19,791 |
| `consim2` | ≤2 | 432,776 | 21,101 |

Note: path counts are total paths with body nodes ≥1. VPN nodes are unique nodes on any qualifying path (maturity ≥3 endpoint).

**PathbuildA vs PathbuildB (chain layer):**
- `pathbuildA`: KMeans(k=40) on mean body-node embeddings per path → clusters path bodies
- `pathbuildB`: Frozenset co-occurrence families — paths grouped by which body concept subtype+cluster combinations they traverse

---

## Criteria and Results

### C1 — Mean intra-cluster cosine similarity (independent clustering quality)

**Methodology:** For each consimN, run independent AgglomerativeClustering(k=40, ward linkage) on only the VPN risk/intervention nodes for that config. Compute mean cosine similarity of cluster members to their centroid. Higher = tighter, more coherent clusters.

This is a non-trivial metric: each consimN produces different cluster assignments because the node sets differ.

| Config | Risk intra-sim | Intervention intra-sim |
|--------|---------------|----------------------|
| consim0 | 0.778 | 0.704 |
| consim1 | 0.801 | 0.709 |
| consim2 | 0.820 | 0.715 |

**Verdict:** consim2 > consim1 > consim0 (cluster quality improves with more nodes). The gain from consim1→consim2 is smaller (0.019 risk) than consim0→consim1 (0.023 risk). All three configs produce coherent clusters.

---

### C2 — Edge-only node fraction (single-paper grounding)

Fraction of qualifying VPN nodes that also appear on consim0 (edge-only) paths. Higher = nodes are better grounded in single-paper EDGE-only argument chains.

| Config | Risk EO frac | Intervention EO frac |
|--------|-------------|---------------------|
| consim0 | 1.000 (trivial) | 1.000 (trivial) |
| consim1 | **0.689** | **0.962** |
| consim2 | 0.568 | 0.959 |

**Verdict:** consim1 >> consim2 for risk grounding (0.689 vs 0.568, a 12-point gap). The 357,768 additional paths in consim2 over consim1 add risk nodes reachable only via 2 SIM hops, diluting single-paper evidence quality significantly. Intervention grounding is nearly identical (96.2% vs 95.9%) because interventions are always EDGE endpoints.

---

### C3 — True cross-config ARI (independent clustering stability)

**Methodology:** ARI computed between independent per-config cluster assignments (not the shared PKL). For nodes appearing in both consimN configs, compare their per-config cluster labels.

| Pair | Risk ARI | Intervention ARI | N shared risk nodes |
|------|---------|-----------------|---------------------|
| consim0 ↔ consim1 | 0.444 | 0.566 | ~2,639 |
| consim1 ↔ consim2 | **0.636** | **0.795** | ~3,830 |

**Verdict:** consim1 and consim2 produce highly similar cluster taxonomies (ARI 0.636/0.795), confirming that the additional consim2 nodes do not disrupt the established cluster structure. consim0↔consim1 is less stable (ARI 0.444/0.566) because consim0 has fewer qualifying nodes and the sparse sampling creates more boundary instability.

---

### C4 — R→I pair coverage (fraction of possible risk-intervention cluster pairs)

Fraction of all possible risk-cluster × intervention-cluster pairs (40×40 = 1,600 total) that have at least one qualifying path for each consimN.

| Config | N covered pairs | Coverage fraction | N paths |
|--------|----------------|-----------------|---------|
| consim0 | 610 / 1,600 | 38.1% | 3,473 |
| consim1 | 1,088 / 1,600 | **68.0%** | 75,008 |
| consim2 | 1,289 / 1,600 | 80.6% | 432,776 |

**Verdict:** consim1 covers 84.6% of consim2's pairs (1,088/1,289) using only 17.3% of consim2's path count (75,008/432,776). This is the ideal efficiency frontier: consim1 achieves high coverage with high grounding at a fraction of the path volume. consim2 adds 201 more cluster-pair connections (12.6% of total) but at the cost of 12-point worse risk grounding.

---

## PathbuildA vs PathbuildB Empirical Assessment (A2)

### PathbuildA chain clustering — fails the "because" criterion

PathbuildA applies KMeans(k=40) to the mean body-node embedding of each path. Result: **14/40 clusters** match explicit misalignment keywords ("misalign", "existential", "catastrophic"), and the full chain_cluster_names.csv shows ~25-30 clusters are effectively risk re-statements rather than mechanistic chains.

Root cause: mean-body embedding is dominated by the high-density misalignment concept neighborhood in the corpus. KMeans clusters form in this dense region, repeatedly naming the cluster by the dominant risk theme instead of the mechanistic connection from risk → intervention.

**PathbuildA verdict:** Does NOT answer "Intervention I addresses risk R **because** [chain mechanism]." Chain clusters are named as risks, not reasoning chains. PathbuildA should be used only as a supplementary finding (the collapse itself is a corpus characterization result).

### PathbuildB chain families — provides mechanistic diversity

PathbuildB groups paths by their frozenset of `{(body_concept_subtype, cluster_id)}` combinations. This captures WHICH concept subtype clusters co-occur in a path (structurally), not the embedding centroid of the path.

**Top-20 consim1 B-families (representative decoded names):**

| Rank | N paths | Chain theme (decoded) |
|------|---------|----------------------|
| 1-3 | 6,944–1,210 | Field-building: "Funding initiatives / Government AI safety grants / Insufficient AI safety research capacity / Targeted outreach" |
| 4 | 900 | RL safety: "Randomized environment training / Specification gaming benchmarks / Limited preference learning / Reward misspecification / Predictive modeling" |
| 5 | 896 | Corrigibility: "Preference-uncertain AI design / AI-box containment / Instrumental convergence / Transparency verification" |
| 6 | 855 | RLHF oversight: "Human oversight loops / Reward modeling fine-tuning / Objective misspecification → RLHF" |
| 7 | 746 | Adversarial robustness: "Adversarial training / PGD adversarial examples / Adversarial vulnerability" |
| 9–11, 13–15 | 678–399 | Funding variants (different validation evidence cluster) |
| 12 | 444 | Deceptive alignment: "Value alignment embedding / Constitutional principles / Deceptive alignment" |
| 16 | 394 | Compute governance: "Export controls / AI hardware controls / Capability spikes / Supply chain governance" |
| 17 | 391 | Utility preservation: "Impact measures / Containment / Reward misspecification / Power-seeking" |
| 18 | 383 | International governance: "Coordinated governance / International AI frameworks / Global governance gaps" |
| 19-20 | 379–377 | RLHF variants (reward modeling from feedback / goal misalignment) |

**PathbuildB verdict:** B-families DO answer the "because" question:
- "Expand AI safety research funding addresses insufficient safety research capacity **because** [funding organizations create grants + training programs that demonstrate skill transfer + outreach attracts new researchers]"
- "Adversarial training addresses adversarial vulnerability **because** [PGD-based adversarial training creates robust features + lower adversarial error rates confirm efficacy]"
- "Export controls on compute address unpredictable AI scaling **because** [supply chain concentration enables governance + historical governance analogies support feasibility]"

The dominant families (ranks 1-3) are variants of the field-building/funding chain — reflecting the corpus bias toward I8 (funding AI safety research) as the dominant intervention. Starting from rank 4, distinct mechanistic families emerge across technical safety (RL, adversarial, corrigibility, RLHF) and governance (compute controls, international governance) domains.

**PathbuildB as L2 chain layer:** Use the 1,603 B-families (consim1, n≥5 paths) as the L2 chain taxonomy. For qualitative analysis, report the top-10 mechanistically distinct families (skipping near-duplicate funding variants). For quantitative analysis, the full 1,603 families provide R→B→I connectivity structure.

---

## Config Selection Decision

**Selected config: `consim1_pathbuildB`**

| Criterion | consim0 | consim1 | consim2 | Winner |
|-----------|---------|---------|---------|--------|
| C1 (cluster quality) | 0.778/0.704 | 0.801/0.709 | 0.820/0.715 | consim2 (marginal) |
| C2 (risk grounding) | 1.000 (trivial) | **0.689** | 0.568 | consim1 |
| C3 (ARI stability) | 0.444/0.566 | **0.636/0.795** | — | consim1 (vs consim0) |
| C4 (R→I coverage) | 38.1% | 68.0% | 80.6% | consim2 (marginal) |
| Path efficiency | low | **high** | low | consim1 |
| Chain layer | pathbuildA fails | **pathbuildB** | pathbuildB | pathbuildB |

**Rationale:**
1. **C2 is the decisive criterion:** consim1 risk grounding (68.9%) is 12 points better than consim2 (56.8%). Nodes only reachable via 2 consecutive SIM hops have weaker single-paper evidence — they are semantic neighbours of nodes on qualified paths but not themselves directly argued in single papers. For a workshop paper claiming to identify grounded connections, edge-only grounding is essential.

2. **C4 efficiency frontier:** consim1 covers 84.6% of consim2's R→I cluster-pair connections with 17.3% of the path volume. The 201 additional pairs in consim2 come at the cost of 357,768 more paths, nearly all reusing existing cluster connections via weaker (2-SIM-hop) evidence.

3. **C3 confirms consim1 is stable:** ARI(consim1, consim2) = 0.636/0.795, meaning consim1 and consim2 share highly similar cluster taxonomies. Choosing consim1 does not lose cluster stability.

4. **C1 difference is small:** consim2 has marginally better intra-cluster sim (0.820 vs 0.801 for risk), but both exceed the 0.7 threshold for meaningful clustering. The difference does not justify accepting worse grounding.

5. **PathbuildB over pathbuildA:** PathbuildB provides mechanistically distinct chain families that answer "because [mechanism]." PathbuildA chain clusters name risks, not reasoning chains. PathbuildB is the correct L2 representation.

---

## Selected Config Summary for Workshop Paper

| Layer | Description | N qualifying units |
|-------|------------|-------------------|
| L1 Risk clusters | 40 clusters, consim1 VPN (3,830 risk nodes) | 40 clusters |
| L2 Chain families | PathbuildB B-families, consim1 (1,603 families n≥5) | Top-20 for paper |
| L3 Intervention clusters | 40 clusters, consim1 VPN (2,799 intervention nodes) | 40 clusters |
| R→I coverage | 1,088 / 1,600 cluster pairs (68.0%) | — |
| Risk grounding | 68.9% of risk VPN nodes on edge-only paths | — |
| Intervention grounding | 96.2% of intervention VPN nodes on edge-only paths | — |

**Note on L1/L3 cluster tables:** The cluster tables (`risk_clusters.csv`, `intervention_clusters.csv`) use the unconstrained VPN (4,889 risk / 2,970 intervention) for the full qualifying universe. The consim1 VPN is used for connectivity analysis only. Both are valid for different analysis questions: unconstrained VPN = holistic qualifying universe; consim1 VPN = well-grounded analysis subset.

---

## Outstanding: Phase E (Step 5 LLM Naming)

Apply consim1_pathbuildB config to:
1. **B2:** Rerun LLM cluster naming (120 clusters: 40 risk + 40 intervention + 40 chain body) with `gpt-5.4-mini`, using corrected chain prompt: "Intervention [I] addresses risk [R] because [chain theme]"
2. **B4:** Top-20 B-family named table for workshop paper (use decoded body component names as input to gpt-5.4-mini for synthesized chain names)
3. **B5:** Top-20 novel SIM-bridged-only R→I pairs with novelty filter (from `cross_config_ri_pairs.csv`, n_paths_c0=0)
