# Phase 2 Step 4 Analysis Plan (v9)
## Three-Level Cluster Taxonomy with Named Connection Concept Chain Families

**Created:** 2026-03-29
**Revised:** 2026-04-04 v9 — Exhaustive background agent audit of all Phase 2 scripts completed; four new Step 2b Category B gaps added (multi_risk_clusters.csv, risk_diversity_stats.csv, category_mechanism_families.csv, mechanism_transfer_betweenness_v2.png); phase2_step3_rerun_betweenness_sectionf.py audit row corrected from ✅ clean to ⚠️ superseded (uses maturity≥3 only, NOT valid_pathway_nodes; rerun_pathfiltered.py is authoritative); Gap 3 note corrected; Phase C updated with new Step 2b reruns
**Branch:** martin/main
**Goal:** Produce a named three-level risk → connection concept chain → intervention taxonomy, evaluated across 6 analysis configurations to select the best balance of cross-paper coverage vs. scientific grounding.

---

## Part 0 — Holistic Execution Principle (Topmost Priority)

**This principle overrides all individual quality cut descriptions throughout this document. Every analysis, at every step, must comply.**

---

### 0.1 — The Dividing Line: Category A (Cartography) vs Category B (Workshop Paper)

**This is the first and governing rule. Read it before anything else.**

Two fundamentally different categories of analysis exist in this project:

**Category A — Full-graph cartography and algorithm characterization**
Analyses that explore the full graph to select an algorithm or configuration, or to characterize the data distribution as background context. These intentionally operate on all nodes. Their outputs are NOT reported as workshop paper findings about the AI safety landscape.

`valid_pathway_nodes` filter is **NOT required** for Category A. Full-graph is correct.

| Analysis | Step | Why full-graph is correct |
|----------|------|--------------------------|
| Clustering pipeline (UMAP, k=40 agglomerative) | Step 1 | All nodes clustered to produce stable cluster IDs — filtering is post-hoc |
| Silhouette, ARI, EDGE validation % per config | Step 2 | Algorithm comparison requires full node sets for fair relative comparison |
| Node migration rates, cohesion, centroid similarity, algorithm comparison | Step 2 | Same — config-selection metrics, not paper findings |
| Step 1 graph metrics (betweenness, PageRank, in/out-degree, clustering coeff) | Step 1 | Full-graph characterization; only consumed by superseded Step 2 betweenness |
| Multi-criteria scoring, threshold sensitivity analysis | Step 3 A/B | Config selection rationale — not paper findings |

**Category B — Workshop paper contributions**
Analyses whose results will appear in the paper as findings about the AI safety literature — the structure, coverage, and patterns of risk→intervention reasoning chains. These must only operate on `valid_pathway_nodes`-qualified members.

`valid_pathway_nodes` filter is **REQUIRED** for Category B. Full-graph is wrong.

| Analysis | Step | Paper section |
|----------|------|---------------|
| Three-tier taxonomy (L1/L2/L3 cluster tables, member counts, representative nodes) | Step 4 | Taxonomy |
| Within-cluster EDGE density (structural coherence of qualifying members) | Step 4 | Taxonomy / Methods |
| Connectivity and gap analysis (path counts, missing connections) | Step 4 | Coverage gaps |
| Per-cluster source diversity | Step 4 (rerun of Step 2b) | Cluster characterization |
| Per-cluster maturity distribution | Step 4 (rerun of Step 2b) | Cluster characterization |
| Hub quality (top cross-paper hubs within qualifying set) | Step 4 (rerun of Step 2b) | Cluster characterization |
| SIM coverage: anchored vs SIM-only nodes (consim0 vs consim2) | Step 4 (rerun of Step 3 C) | Cross-paper inference |
| Held-out validation (cluster centroid coherence on qualifying members) | Step 4 (rerun of Step 3 E) | Validation |
| Betweenness centrality of bridge concepts | Step 3/4 | Key bridge concepts |
| Subcluster identification and naming | Step 4 | Taxonomy |
| Cluster naming (LLM labels based on representative nodes) | Step 5 | Named families |
| Pathway examples (technical chains, field-building chains) | Step 5 | Example chains |
| Triplet SIM reach (papers reached via SIM from each triplet) | Step 5 | Cross-paper coverage |

**Rule:** If the result will appear in the workshop paper as a finding about the AI safety landscape, `valid_pathway_nodes` is required. If it is about algorithm or config selection, full-graph is correct and intentional.

---

### 0.2 — The Three Quality Cuts Are Path-Level Constraints, Not Node Properties

The three quality cuts — EDGE confidence≥3, SIM cos_sim≥0.9, intervention maturity≥3 — are **not independent node-level properties**. They are constraints on the specific edges and endpoints used in a complete risk→intervention path.

**Independent application of these cuts is incorrect** because:

- A node may have some EDGE edges with conf<3 and others with conf≥3. Filtering nodes by "all EDGE edges have conf≥3" excludes valid nodes; filtering by "any EDGE edge has conf≥3" includes invalid nodes. Neither is right.
- A node that fails the EDGE conf≥3 cut in isolation may still be on qualifying paths via its SIM≥0.9 connections. Excluding it based on its EDGE edges would be a false exclusion.
- A maturity≥3 intervention node that has no reachable qualifying path from any risk node should not be included, even though it passes the maturity cut individually.
- Two cuts applied sequentially create both false inclusions and false exclusions.

### 0.3 — The Pathway-First Solution

The path files generated by `phase2_rerun_pathfiltered.py` (confirmed clean — all three cuts applied simultaneously during BFS path generation) encode the holistic constraint. Every path satisfies all three conditions at once:
- Every EDGE hop has confidence≥3
- Every SIM hop has cos_sim≥0.9
- Every intervention endpoint has maturity≥3

**`valid_pathway_nodes` = nodes appearing in any path in the appropriate path file = the authoritative holistic qualifying set.**

A node is in `valid_pathway_nodes` if and only if there exists at least one complete risk→intervention path containing it where all three constraints are satisfied simultaneously.

```python
# Build valid_pathway_nodes from the config-appropriate path file
valid_pathway_nodes = set()
with open(paths_file, 'r') as f:
    for line in f:
        obj = json.loads(line)
        for nid in obj['path']:
            valid_pathway_nodes.add(int(nid))

# Apply to ALL cluster member access (replaces all isolated cuts):
def get_qualifying_clusters(edge_config, mode, node_type, algo='agglomerative'):
    raw = get_clusters(edge_config, mode, node_type, algo)
    return {cid: [n for n in nodes if n in valid_pathway_nodes]
            for cid, nodes in raw.items()}
```

### 0.4 — Scope of This Principle

**Every Category B analysis must use only `valid_pathway_nodes`-qualified members.** This applies to:
- All cluster membership tables (risk, intervention, body)
- All connectivity and gap analyses
- All betweenness and graph structure analyses
- All per-cluster statistics reported in the workshop paper (source diversity, maturity distribution, hub quality)
- All naming and example selection analyses
- All within-cluster edge density computations

**Intentional exceptions (Category A — full-graph is correct):**
- Config-selection quality metrics from Step 2 (silhouette, ARI, EDGE validation %) — algorithm evaluation, not paper findings
- Clustering pipeline (Step 1 UMAP) — must use all nodes to produce stable cluster IDs; filtering is post-hoc
- Step 1 graph metrics stored in `node_attrs.pkl` (PageRank, betweenness, degree) — used only by superseded Step 2 betweenness; ⚠️ must NOT be used in any Category B analysis

### 0.5 — Step 4 Is the Authoritative Rerun Point

For any Category B analysis from Steps 2 or 3 that violates the holistic principle, **the corrected version is produced as part of Step 4 execution**. Step 4 scripts may re-implement any analysis from earlier steps using the pathway-first approach. The corrected Step 4 output supersedes the earlier step's output for workshop reporting.

---

## Status Entering Step 4

| Prerequisite | Status |
|---|---|
| Phase2_Step2_Issues.md: all 9 issues | ✅ FIXED |
| Step 3 re-run: path-filtered betweenness (Section D, unconstrained) | ✅ COMPLETE |
| Step 3 re-run: both-mode betweenness (Section D2, exploratory) | ✅ COMPLETE |
| Hub quality SIM degrees fix | ✅ COMPLETE |
| FDT artifact confirmed absent | ✅ CONFIRMED |

---

## Part 1 — Definitions

All terms used in this plan are defined here. These definitions are authoritative for the workshop paper methods section.

### Edge Types

**EDGE edge:** A structural relationship extracted by the LLM from a single source paper. Represents an explicit causal, logical, or dependency relationship between two concepts or interventions within one paper's argument. Stored in FalkorDB with label `EDGE` and property `edge_confidence` (integer 1–5).

**SIM edge:** A cross-paper semantic similarity connection. Stored with label `SIMILARITY_ABOVE_POINT_EIGHT_*`. Similarity score stored as L2 distance between unit-normalized embeddings: `cos_sim = 1 − score²/2`. A SIM edge between two nodes means they appear in different papers but describe semantically equivalent or highly overlapping concepts (cos_sim ≥ threshold). SIM edges do not assert a causal or logical relationship — they assert semantic correspondence.

### Quality Cuts (all apply universally — see Part 3 for implementation status per script)

| Cut | Value | Scope | Applied at | Rationale |
|-----|-------|-------|------------|-----------|
| EDGE confidence | ≥ 3 | All structural EDGE-type edges in any analysis | Path generation (inherited by path files); must be applied explicitly when reading `graph_edge_data.pkl` | Removes LLM-uncertain extractions (scores 1–2 = low model confidence) |
| Intervention maturity | ≥ 3 | All intervention-type nodes used as cluster members or path endpoints | Path generation (BFS endpoint check); downstream via `valid_pathway_nodes` filter (Gap 4) — which subsumes the maturity constraint because path generation already enforces maturity≥3 at every endpoint | Restricts to interventions with documented feasibility evidence; maturity 1–2 = speculative proposals only |
| SIM similarity | cos_sim ≥ 0.9 | SIM-type edges in paths and cluster assignments | Path generation + clustering edge_config=0.9 | Strong semantic similarity threshold; removes weaker cross-paper associations |
| Max consecutive SIM | 0 / ≤1 / ≤2 | Full path (risk → body → intervention) | Path sampling (consimN config) | Limits consecutive cross-paper SIM hops without an EDGE anchor |
| Mode | Unconstrained | Path structure | Path generation | No restriction on risk node count or body monotonicity; preserves x-risk hub structure at L1 |

**Implementation status — consistent with Part 0 (Gap 4 is the primary mechanism):**

1. ✅ **Path generation** — BFS only terminates at maturity≥3 intervention nodes. Every node in any path file necessarily has maturity≥3 if it is an intervention endpoint. This means the `valid_pathway_nodes` filter (Gap 4) automatically enforces the maturity constraint; no separate maturity check is needed downstream.

2. ✅ **Belt-and-suspenders maturity filter (Gap 2)** — explicitly applied to `build_cluster_table` (cluster_naming.py, step4b.py) and `interv_clusters_09` in connectivity.py. This is a necessary condition sanity check but NOT sufficient as a standalone mechanism (see Gap 2 — it does not guarantee the node is on any qualifying path).

3. ❌ **Primary control: valid_pathway_nodes filter (Gap 4)** — not yet implemented. `valid_pathway_nodes` from the config-appropriate path file is the correct and complete mechanism (see Part 0, Gap 4). It subsumes the maturity filter. The `cluster_memberships.pkl` contains 155 intervention nodes with maturity < 3 (15 at maturity=1, 140 at maturity=2) that must be excluded — but these are excluded automatically by the valid_pathway_nodes filter without any explicit maturity check needed.

### Path Types

**EDGE-only path (consim0):** A complete risk→intervention path where every connection is a structural EDGE edge (conf≥3). Every step in the path is documented in a single paper's argument chain. Maximum grounding, minimum cross-paper coverage (3,405 paths).

**Cross-paper SIM-bridged path (consim1 or consim2):** A path that includes one or more SIM edges connecting semantically equivalent nodes from different papers. These paths represent connections that are supported by *multiple papers converging on the same concepts* — not hallucinated connections, but inferred connections that no single paper may have argued explicitly end-to-end.

**SIM-bridged connection:** A connection between two clusters that exists in consim1 or consim2 but not in consim0. This means the connection is supported by cross-paper semantic convergence (multiple papers discussing related concepts that are semantically equivalent by embedding) but not by a single paper's explicit argument chain. These are legitimate findings about how the AI safety literature collectively constructs arguments across publications — they should be reported as "cross-paper convergent connections" not as artifacts or hallucinations.

### Node Types (concept_category / type field in node_attrs)

| Type | Description |
|------|-------------|
| `risk` | A harm, failure mode, or undesirable outcome described in the literature |
| `problem_analysis` | Analysis of why a risk occurs or its mechanisms |
| `theoretical_insight` | Theoretical result or principle relevant to a risk or solution |
| `design_rationale` | Reasoning about why a particular intervention approach is appropriate |
| `implementation_mechanism` | Technical or institutional mechanism by which an intervention operates |
| `validation_evidence` | Empirical or theoretical evidence supporting an intervention |
| `intervention` | A proposed action, method, or policy to reduce a risk |

Body node subtypes are: problem_analysis, theoretical_insight, design_rationale, implementation_mechanism, validation_evidence. Risk and intervention are the path endpoints.

### Cluster Types

**L1 Risk cluster:** k=40 agglomerative cluster of risk-type nodes at SIM≥0.9/unconstrained. Represents a semantically coherent family of related risks discussed across the literature.

**L2 Chain body family:** A cluster of connection concept chain bodies — the intermediate reasoning (body nodes) between a risk and an intervention. Two methods to build:
- **pathbuildA:** Mean embedding of all body nodes per path → KMeans k=40 on path-body vectors. Groups paths with holistically similar intermediate content.
- **pathbuildB:** Frozenset of (subtype, cluster_id) pairs per path → group by exact signature match. Groups paths with the same combination of reasoning step types.

**L3 Intervention cluster:** k=40 agglomerative cluster of intervention-type nodes at SIM≥0.9/unconstrained. Represents a semantically coherent family of intervention approaches.

### Connectivity Analysis

For each ordered layer pair (L1→L2, L2→L3, L1→L3 direct), and for each cluster pair:
- **Path count:** number of qualifying paths passing through both clusters (path start in L1 cluster, a body node in L2 cluster, path end in L3 cluster)
- **EDGE edge count:** number of EDGE edges (conf≥3 applied explicitly) between the two clusters' member nodes
- A cluster pair is "connected" if it has ≥5 EDGE edges between members

Three matrices: L1→L2 (which risks connect to which chain families), L2→L3 (which chain families connect to which interventions), L1→L3 direct (risks to interventions bypassing chain layer).

### Gap Analysis

For each of 6 directed gap types, count clusters with zero connections at that layer boundary:

| # | Gap | Interpretation |
|---|-----|----------------|
| 1 | L1 cluster with no L2 connection | Risk identified but no documented chain of reasoning toward solutions |
| 2 | L2 cluster with no L1 connection | Reasoning chain exists but no risk motivating it — framing gap |
| 3 | L2 cluster with no L3 connection | Reasoning chain exists but no concrete intervention proposed — theory without practice |
| 4 | L3 cluster with no L2 connection | Intervention proposed with no documented conceptual justification |
| 5 | L1 cluster with no L3 connection | Risk with no intervention at all — highest-priority research gap |
| 6 | L3 cluster with no L1 connection | Orphan intervention — proposed with no stated safety rationale |

**Zero gaps are by design:** The extraction prompt instructs LLMs to trace complete logical chains from identified risks through conceptual reasoning to concrete interventions within each paper. Every paper's extracted structure is therefore a full end-to-end EDGE chain. With 40 clusters and thousands of papers, every cluster pair will have at least some EDGE-only paths at the cluster level. Zero gaps at consim0 confirms that the extraction design succeeded — it does NOT mean the literature has "solved" its research gaps.

**The analytically meaningful question is cross-config connection strength:** which R→I cluster pairs are strongly supported in edge-only (many papers document the full chain) vs. weakly supported (few edge-only paths, but many cross-paper SIM-bridged paths)? Gaps in consim0 that persist to consim1/consim2 (N=0 for this corpus at cluster level) would represent connections the field has NOT established even collectively. Pairs with c0=0 but c1/c2>0 are connections that require cross-paper semantic bridging — real connections that are implicit across papers but explicit in none.

**Observation (2026-04-04):** Cross-config pair analysis on 1,289 R→I pairs shows 604 edge-only, 483 new at consim1, 202 more at consim2. 53.1% of all pairs (685/1,289) are cross-paper-only. All 604 edge-only pairs are preserved at higher configs (consim0 ⊆ consim1 ⊆ consim2). See `step4_connectivity/cross_config_ri_pairs.csv`.

### Edge-Only Path Fraction

**Definition:** For a given cluster and config, the fraction of cluster member nodes that appear on at least one consim0 (edge-only) path.

- For consim0: undefined/trivial — the node set IS defined as nodes appearing on consim0 paths
- For consim1 and consim2: informative — higher fraction = more cluster nodes are grounded in single-paper EDGE-only argument chains, not just cross-paper SIM-bridged paths

**Important:** cluster member nodes are NOT pre-filtered to qualifying path nodes (see Part 4 — Cluster Membership Scope). The edge-only path fraction is a genuine quality signal, not circular.

---

## Part 2 — Intent

### Why 3 Path Configs

The consecutive SIM threshold is the cross-paper grounding dial:
- **consim0** = every connection documented in a single paper — maximum grounding, minimum coverage
- **consim1** = allows one SIM hop between papers — moderate cross-paper bridging
- **consim2** = allows two consecutive SIM hops — broadest coverage, most cross-paper inference

Goal: run all 3 configs and select the one where the named body families are coherent, stable, and traceable to actual paper arguments without being dominated by SIM-only paths with no EDGE anchor.

### Why Option A and Option B

**Option A (pathbuildA):** Holistic chain semantics via mean body embedding. Easier to name, may wash out subtype structure.

**Option B (pathbuildB):** Preserves argumentative structure via subtype co-occurrence signatures. Finer-grained, produces many small families (16,034 at consim2).

### Config Selection Criterion (after all 6 versions produced)

1. Named family coherence: ≥80% of L2 families receive a crisp, non-generic label?
2. Edge-only path fraction: mean across all 40 L2 clusters (higher = better single-paper grounding)
3. Cross-config stability: ARI(consim1, consim2) ≥ 0.7?
4. Gap analysis sensitivity: how many gaps disappear between consim0 and consim2? (SIM-bridged connections)
5. Prefer consim1 over consim2 if coherence and gap analysis are comparable

---

## Part 3 — Quality Cut Audit Across All Steps

Full audit completed (2026-03-30, extended 2026-04-04) across all Steps 1–5 scripts. The audit applies the Part 0 dividing line: **Category A** (full-graph cartography, algorithm comparison) does not require valid_pathway_nodes; **Category B** (workshop paper contributions) does.

### Audit Results Per Script

| Script | Category | EDGE conf≥3 | SIM≥0.9 | Pathway-first (valid_pathway_nodes) | Overall |
|---|---|---|---|---|---|
| `phase2_step1_loadandparse.py` — `compute_graph_metrics` | A (full-graph characterization) | ❌ no filter | ❌ no filter | N/A (intentional — full-graph cartography) | ✅ acceptable; ⚠️ stored metrics (pagerank/betweenness/degree in node_attrs.pkl) MUST NOT be used in Category B analyses |
| `phase2_step2_metrics_stability.py` — silhouette, ARI, EDGE validation, node migration | A (config selection) | N/A | N/A | N/A (intentional — algorithm comparison) | ✅ clean for purpose |
| `phase2_step2_metrics_stability.py` — `cluster_source_diversity.csv` | B (workshop output) | N/A | N/A | ❌ unfiltered — reads all cluster members without valid_pathway_nodes | ❌ Required Step 4 rerun; superseded by qualifying version |
| `phase2_step2_metrics_stability.py` — `cluster_temporal_coverage.csv` | B (workshop output — **NEW GAP**) | N/A | N/A | ❌ unfiltered — reads all cluster members without valid_pathway_nodes | ❌ Required Step 4 rerun; this output not previously documented in plan |
| `phase2_step2_metrics_stability.py` — `mechanism_transfer_betweenness.csv` | B (superseded) | ❌ unfiltered | N/A | ❌ unfiltered | ⚠️ superseded by `rerun_pathfiltered.py` betweenness output |
| `phase2_step2b_extended_analysis.py` — source_diversity, maturity, multi_risk, risk_diversity, mechanism_families | B (workshop outputs) | N/A | N/A | ❌ all cluster members, no path filter | ❌ Required Step 4 rerun |
| `phase2_step2b_extended_analysis.py` — hub quality | B (workshop output) | N/A | ✅ SIM≥0.9 applied | ✅ **ALREADY DONE** — `phase2_fix_hub_quality_degree.py` + `phase2_fix_hub_quality_sim_degrees.py` applied valid_pathway_nodes | ✅ hub_quality_metrics.csv, hub_quality_scatter*.png, hub_quality_bar_v2.png are **already clean** — do NOT rerun from Step 4 |
| `phase2_fix_hub_quality_degree.py` | B (hub quality fix) | ✅ conf≥3 | ✅ | ✅ builds valid_nodes from path files; degree_structural restricted to endpoints in valid_nodes | ✅ clean — produces valid_pathway_nodes-filtered hub_quality_metrics.csv |
| `phase2_fix_hub_quality_sim_degrees.py` | B (hub quality fix) | N/A | ✅ SIM≥0.9 | ✅ builds valid_nodes from path files; ALL SIM edges counted only if BOTH endpoints in valid_nodes; n_sources restricted to valid-pathway partners | ✅ clean — produces final valid_pathway_nodes-filtered hub_quality_metrics.csv |
| `phase2_step3_validation_and_selection.py` — Sections D, F | B (workshop outputs) | ✅ fixed | ✅ | ❌ D2 `both_nodes` not restricted (Gap 3c); C uses cluster proxy; E unfiltered | ❌ Gap 3c needs fix; C/E need Step 4 rerun |
| `phase2_step3_rerun_betweenness_sectionf.py` | B | ✅ | ✅ | ⚠️ maturity≥3 filter only — NOT valid_pathway_nodes | ⚠️ SUPERSEDED by `rerun_pathfiltered.py`; do NOT rerun this script; `rerun_pathfiltered.py` overwrites its output files with correct path-filtered versions |
| `phase2_rerun_pathfiltered.py` | — (path generator) | ✅ | ✅ | ✅ IS the path filter source | ✅ clean |
| `phase2_step4_cluster_naming.py` | B | ✅ | ✅ | ❌ no valid_pathway_nodes filter; risk PKL saved unfiltered | ❌ needs Gaps 3b+4 |
| `phase2_step4_connectivity.py` | B | ✅ | ✅ | ❌ valid_pathway_nodes not loaded; risk PKL loaded unfiltered | ❌ needs Gaps 3b+4 |
| `phase2_step4b_paths_and_plots.py` | B | ✅ | ✅ | ❌ no valid_pathway_nodes filter; risk PKL saved unfiltered | ❌ needs Gaps 3b+4 |
| `phase2_step5_naming.py` | B | N/A | N/A | ❌ `get_cluster_dict()` unfiltered (Gap 5a) | ❌ needs Gap 5a fix before Step 5 rerun |
| `phase2_step5_examples.py` | B | N/A | N/A | ✅ lookup-only — qualifying path endpoints only | ✅ clean |
| `phase2_step5_triplet_simreach.py` | B | N/A | ✅ applied | ❌ cluster node sets unfiltered (Gap 5b) | ❌ needs Gap 5b fix before Step 5 rerun |

---

### Gap 1 — CRITICAL: SIM≥0.9 missing from sim_edge_set (all three Step 4 scripts)

All three Step 4 scripts build `sim_edge_set` from ALL SIMILARITY edges without the SIM≥0.9 threshold filter:

```python
# CURRENT (WRONG) — in phase2_step4_cluster_naming.py lines 101-109,
#   phase2_step4_connectivity.py lines 80-87,
#   phase2_step4b_paths_and_plots.py lines 98-104:
for e in edge_data:
    if str(e.get('type', '')).upper() == 'SIMILARITY':
        sim_edge_set.add((min(s, t), max(s, t)))  # no similarity_score check
```

**Impact:** `max_consec_sim()` uses this set to classify every consecutive path node pair as either SIM-connected or EDGE-connected. With the buggy set:
- Buggy sim_edge_set: **1,565,684** edges (all SIMILARITY)
- Correct sim_edge_set (SIM≥0.9): **144,140** edges
- Over-inclusion: **10.9×** — 1,421,544 extra SIM<0.9 edges

If two consecutive path nodes (i, i+1) are connected by an EDGE in the path but also have a SIM<0.9 edge in `graph_edge_data.pkl`, `max_consec_sim()` incorrectly classifies that hop as a SIM hop. This over-counts SIM hops, causing EDGE-grounded path segments to appear as SIM hops.

**Downstream corruption:** every path file generated by these scripts has incorrect `max_consec_SIM` values and incorrect consimN classification. The files `representative_pathways_consim1.jsonl`, `representative_pathways_consim2.jsonl`, and their counts (VarA=75,008, VarB=432,776) must all be treated as **invalid and requiring regeneration**.

**Fix:**
```python
# CORRECT:
def cos_sim_from_score(s):
    return 1.0 - float(s)**2 / 2.0

for e in edge_data:
    if str(e.get('type', '')).upper() == 'SIMILARITY':
        score = e.get('similarity_score')
        if score is not None and cos_sim_from_score(score) >= 0.9:
            try:
                s, t = int(e['source']), int(e['target'])
                sim_edge_set.add((min(s, t), max(s, t)))
            except (ValueError, TypeError):
                pass
```

Apply this fix in all three scripts before any rerun.

---

### Gap 2 — Intervention maturity≥3 not applied to cluster members (partial fix — superseded by Gap 4)

All Step 4 scripts load intervention cluster members from `cluster_memberships.pkl` without filtering by maturity. The PKL contains 155 intervention nodes with maturity < 3 (15 at maturity=1, 140 at maturity=2) that must be excluded.

**Status:** ✅ maturity≥3 filter has been applied to `build_cluster_table` (cluster_naming.py, step4b.py) and to `interv_clusters_09` in connectivity.py. This is a correct necessary condition for intervention cluster membership.

**However this is not sufficient — see Gap 4.** The maturity≥3 filter alone does not:
- Remove risk cluster nodes that have no qualifying outgoing paths
- Remove intervention nodes that are maturity≥3 but unreachable from any risk node via the current config's qualifying edges
- Ensure that any node included is actually on a complete risk→intervention qualifying path

Gap 4 (pathway-first filter) supersedes this gap and provides the complete holistic solution.

---

### Gap 3 — EDGE conf≥3 not applied in connectivity and betweenness

**Step 3** (`phase2_step3_validation_and_selection.py`):
- Section D (line ~1109): `elif etype == "EDGE": G.add_edge(src, tgt)` — no conf check
- Section D2 (line ~1384): same pattern
- Section F (line ~1664): same pattern

**Step 4** (`phase2_step4_connectivity.py`):
- EDGE edge counts between cluster members: no conf filter applied

**Fix** (add before every `G.add_edge()` or EDGE count for EDGE-type edges):
```python
conf = e.get('confidence')
try:
    if float(conf) >= 3:
        G.add_edge(src, tgt)
except (TypeError, ValueError):
    pass
```

**Note:** Only `phase2_rerun_pathfiltered.py` applies all three quality cuts correctly and is the authoritative betweenness rerun. `phase2_step3_rerun_betweenness_sectionf.py` applies maturity≥3 but does NOT apply valid_pathway_nodes — it is SUPERSEDED by `phase2_rerun_pathfiltered.py`, which writes to the same output filenames. Do NOT rerun `step3_rerun_betweenness_sectionf.py` as a fix; `rerun_pathfiltered.py` is the correct replacement.

---

### Gap 3c — Step 3 Section D2 `both_nodes` not restricted to valid_pathway_nodes

`run_betweenness_both` (lines 1354–1541) builds `both_nodes` from ALL both-mode, ec=0.9 cluster members:

```python
for key, members in cluster_memberships.items():
    ec, mode, ntype, algo, cid = key
    if str(ec) != "0.9" or mode != "both" or algo != "agglomerative":
        continue
    both_nodes.update(str(m) for m in members)
```

This includes ~24% non-qualifying risk nodes and 155 maturity<3 intervention nodes. The betweenness graph is then built on `both_nodes ∩ edge_data`. The conf≥3 fix (Gap 3) is now applied, but the node universe is still too broad.

**Fix:** After building `both_nodes` from PKL, intersect with `valid_pathway_nodes`:
```python
# Load valid_pathway_nodes once (at top of function or passed in)
both_nodes = {str(n) for n in both_nodes if int(n) in valid_pathway_nodes}
```

This requires loading `valid_pathway_nodes` in or before `run_betweenness_both`. The path file is available at `PATHS_DIR / 'paths_unconstrained_sim0.9.jsonl'`.

---

### Step 2 Audit — Holistic Quality Cut Assessment

Step 2 is a **characterization and config-selection** step. It operates on the full clustering output (all cluster members) to compare clustering algorithms and edge configs. This is appropriate for its intended purpose: selecting which config to use in Step 4. The config selection conclusion (SIM≥0.9/unconstrained/agglomerative k=40) is robust and does not change with path filtering.

**Step 2 does NOT feed into Step 4 execution** — it only feeds Step 3 Sections A and B (multi-criteria scoring/threshold sensitivity), which are retrospective analyses that document the selection rationale.

However, several Step 2 outputs used in **workshop reporting** should be computed on valid_pathway_nodes-filtered members to be accurate for the analysis universe:

| Analysis | Script | Uses unfiltered? | Impact on reporting | Action |
|----------|--------|-----------------|--------------------|-|
| Silhouette, ARI, EDGE validation % | `step2_metrics_stability.py` — reads from pre-computed `all_cluster_metrics.csv` | Yes (computed in Step 1 pipeline on all nodes) | Used for algorithm/config selection only — correct as-is | No rerun needed; these are clustering QA metrics, not workshop analysis outputs |
| Node migration rates | `step2_metrics_stability.py` — CSV-based | Yes | Config selection rationale — correct as-is | No rerun needed |
| Cluster source diversity v1 (`cluster_source_diversity.csv`) | `step2_metrics_stability.py` — `analyze_cluster_attributes` | Yes — all cluster members | Workshop reports inflated n_sources; different from step2b's `_v2` file | **Rerun in Step 4** using valid_pathway_nodes-filtered members → `step4_finalanalysis/source_diversity_qualifying.csv` |
| Cluster temporal coverage (`cluster_temporal_coverage.csv` + `temporal_coverage.png`) | `step2_metrics_stability.py` — `analyze_cluster_attributes` + plot | Yes — all cluster members | **NEW GAP (not previously documented)**: year-range distribution per cluster includes non-qualifying nodes | **Rerun in Step 4** using valid_pathway_nodes-filtered members → `step4_finalanalysis/temporal_coverage_qualifying.{csv,png}` |
| Lifecycle distribution per cluster (`lifecycle_distribution.png`) | `step2_metrics_stability.py` — plot | Yes — all cluster members | **NEW GAP**: intervention lifecycle distribution per cluster includes non-qualifying nodes | **Rerun in Step 4** using valid_pathway_nodes-filtered members → `step4_finalanalysis/lifecycle_distribution_qualifying.png` |
| Betweenness original (`mechanism_transfer_betweenness.csv` + `mechanism_transfer_betweenness.png`) | `step2_metrics_stability.py` — `analyze_betweenness` + plot | No conf≥3; no valid_pathway_nodes | Superseded by Step 3 rerun | **Superseded** by `rerun_pathfiltered.py` outputs — do not report step2 betweenness in workshop |
| Cluster source diversity v2 (`cluster_source_diversity_v2.csv`) | `step2b_extended_analysis.py` — `analyze_source_diversity_v2` | Yes — all cluster members | Workshop reports inflated n_sources per cluster | **Rerun in Step 4** using valid_pathway_nodes-filtered members |
| Maturity distribution per cluster | `step2b_extended_analysis.py` — `analyze_maturity_per_cluster` | Yes — all cluster members | Workshop reports maturity for non-qualifying nodes | **Rerun in Step 4** using valid_pathway_nodes-filtered members |
| Hub quality | `step2b_extended_analysis.py` — `analyze_hub_quality` + fix scripts | ✅ Already fixed by `phase2_fix_hub_quality_degree.py` + `phase2_fix_hub_quality_sim_degrees.py` — both scripts build valid_nodes from path files | Current `hub_quality_metrics.csv`, scatter, and bar plots are the correct valid_pathway_nodes-filtered versions | ✅ **ALREADY DONE** — do NOT rerun from Step 4; the fix scripts have produced the authoritative clean versions |
| Multi-risk cluster analysis (`multi_risk_clusters.csv`) | `step2b_extended_analysis.py` — `analyze_multi_risk_clusters` | Yes — all cluster members, no path filter | Reports which nodes appear in multiple risk clusters; includes non-qualifying nodes that inflate overlap counts | **Rerun in Step 4** using valid_pathway_nodes-filtered members |
| Risk diversity stats (`risk_diversity_stats.csv`) | `step2b_extended_analysis.py` — `analyze_risk_diversity` | Yes — all cluster members | Characterizes risk diversity per cluster but includes non-qualifying risk nodes | **Rerun in Step 4** using valid_pathway_nodes-filtered members |
| Mechanism family categorization (`category_mechanism_families.csv`) | `step2b_extended_analysis.py` — mechanism family analysis | Yes — all cluster members | Groups clusters into named mechanism families; includes non-qualifying members | **Rerun in Step 4** using valid_pathway_nodes-filtered members |
| Betweenness visualization (`mechanism_transfer_betweenness_v2.png`) | `step2b_extended_analysis.py` — betweenness plot | All cluster members (from Step 1 unfiltered betweenness scores in node_attrs) | Uses pre-computed betweenness from unfiltered Step 1 graph; not valid_pathway_nodes-restricted | **Superseded** by `rerun_pathfiltered.py` betweenness outputs — do not use step2b betweenness for workshop reporting |
| Edge purity (CHANGE #10) | `step2b_extended_analysis.py` — `analyze_edge_purity` | Uses cluster membership proxy | Circular metric — already dropped (Part 10) | Confirmed dropped; superseded by edge-only path fraction in Step 4 |
| Betweenness (Step 2) | `step2_metrics_stability.py` — `analyze_betweenness` | No conf≥3; no valid_pathway_nodes | Duplicate of Step 3 Section D analysis | Results from `phase2_rerun_pathfiltered.py` supersede this; not used in Step 4 |

**Key Step 2 rules:**
1. Config-selection metrics (silhouette, ARI, EDGE validation from CSVs): computed on full node sets, correct for algorithm comparison (Category A). **No rerun needed.**
2. Workshop-reported per-cluster statistics (source diversity, maturity, hub quality): Category B — must be recomputed on valid_pathway_nodes-filtered members in Step 4. **The Step 2b versions are superseded by Step 4 outputs.**
3. Betweenness: superseded by Step 3 rerun. **No rerun needed.**
4. `phase2_step2c_plot_updates.py` (downstream visualization): generates plots from Step 2b CSVs. Since Step 2b Category B CSVs will be superseded by Step 4 reruns, Step 2c plots of those metrics will also be outdated. **Step 2c does not need a standalone rerun — Step 4 produces the corrected versions directly.**

---

### Step 3 Sections C and E — Step 4 Rerun Scope

**Section C `_analyze_sim_coverage`**: Classifies nodes as "anchored" (in EDGE-config clusters) vs "SIM-only" (in SIM≥0.9 clusters but not EDGE clusters). Uses cluster membership as a **proxy** for path participation rather than actual path files. This is incorrect for workshop reporting: the correct distinction is nodes in `valid_pathway_nodes_consim0` (appear on EDGE-only paths) vs nodes in `valid_pathway_nodes_consim2` but NOT in `valid_pathway_nodes_consim0` (appear only on SIM-bridged paths). **Rerun in Step 4** using path file membership, not cluster membership proxy. The goal of this analysis IS part of the workshop dataset characterization — it should be run on the final qualifying node set.

**Section C `_sample_edge_pathways`**: Loads from `paths_unconstrained_edge_only.jsonl` (already quality-filtered). ✅ Clean — no change needed.

**Section E held-out validation**: Embedding coherence test (leave-20%-out centroid similarity) on all cluster members. The goal is to verify that cluster centroids are representative of their members. For workshop reporting, this should be run on valid_pathway_nodes-filtered members so the result reflects the qualifying analysis universe. **Rerun in Step 4** using valid_pathway_nodes-filtered cluster members. The conclusion (clusters are geometrically coherent) is expected to remain robust.

---

### Gap 3b — risk_clusters_09.pkl saved without valid_pathway_nodes filter

`cluster_naming.py` (line 583) and `step4b.py` (line 620) both save `risk_clusters_09` directly from `get_clusters('0.9','unconstrained','risk')` — without any path-level filter:

```python
risk_clusters_09 = get_clusters('0.9', 'unconstrained', 'risk')
# ... (no filter applied)
with open(os.path.join(STEP4_DIR,'risk_clusters_09.pkl'), 'wb') as f:
    pickle.dump(risk_clusters_09, f)
```

`connectivity.py` loads this PKL at startup and uses it for:
- Building `node_to_risk` mapping (line 144)
- Gap analysis cluster counts (line 272): `all_risk_clusters = set(str(c) for c in risk_clusters_09.keys())`
- Subcluster analysis (line 349)

Result: even though we added maturity≥3 filtering to `interv_clusters_09` in connectivity.py, `risk_clusters_09` still contains ~24% non-qualifying risk nodes from the saved PKL. The PKL is the unfiltered ground truth that connectivity.py can't override.

**Fix:** After loading `risk_clusters_09.pkl` in connectivity.py, apply `valid_pathway_nodes` filter:
```python
with open(...'risk_clusters_09.pkl', 'rb') as f:
    risk_clusters_09_raw = pickle.load(f)
# Apply pathway-first filter
risk_clusters_09 = {cid: [n for n in nodes if n in valid_pathway_nodes]
                    for cid, nodes in risk_clusters_09_raw.items()}
```

And in `cluster_naming.py` / `step4b.py`, apply valid_pathway_nodes before saving:
```python
risk_clusters_09 = {cid: [n for n in nodes if n in valid_pathway_nodes]
                    for cid, nodes in get_clusters('0.9','unconstrained','risk').items()}
```

This must happen after `valid_pathway_nodes` is built (Section 1) in those scripts.

---

### Gap 4 — CRITICAL: Quality cuts must be applied HOLISTICALLY (pathway-first), not independently

**The problem with independent cuts:**

The three quality cuts are path-level constraints, not node-level properties:

- A concept node may have both EDGE edges with conf<3 AND EDGE edges with conf≥3. Filtering to "nodes where all EDGE edges have conf≥3" excludes valid nodes; filtering to "nodes where any EDGE edge has conf≥3" includes invalid nodes. Neither is correct.
- A node that fails the "EDGE conf≥3" cut may still be on qualifying paths via SIM≥0.9 connections from the same node. Excluding it based on its EDGE edges would be wrong.
- A maturity≥3 intervention node that is not reachable from any risk node via currently qualifying edges (EDGE conf≥3 + SIM≥0.9) should NOT be included, even though it passes the maturity cut.
- Two independent cuts applied sequentially can create both false inclusions (nodes that pass each cut individually but are not on any complete qualifying path) and false exclusions (nodes that fail one individual cut but ARE on qualifying paths via another edge type).

**The correct approach: pathway-first membership filter**

The path files (`paths_unconstrained_sim0.9.jsonl` and the derived consimN files) are generated by `phase2_rerun_pathfiltered.py` which is confirmed clean — it applies all three quality cuts simultaneously during BFS path generation:
- Every EDGE hop in every path has confidence≥3
- Every SIM hop in every path has cos_sim≥0.9
- Every intervention endpoint has maturity≥3

Therefore: **`valid_pathway_nodes` = nodes appearing in any path in the path file = the holistic qualifying node set.**

A node is in `valid_pathway_nodes` if and only if there exists at least one complete risk→intervention path containing it where all three quality constraints are satisfied simultaneously. This is exactly the correct membership criterion.

**What valid_pathway_nodes membership means per node type:**

| Node type | What it guarantees |
|-----------|-------------------|
| Risk node | Has at least one qualifying outgoing path reaching a maturity≥3 intervention via conf≥3 EDGE and/or SIM≥0.9 edges |
| Body (concept) node | Appears in at least one such path's intermediate steps |
| Intervention node | Has maturity≥3 AND is reachable from at least one risk node via qualifying edges |

Note: the maturity≥3 filter we applied in Gap 2 is a strict subset of the `valid_pathway_nodes` filter for intervention nodes (any node in valid_pathway_nodes with type=intervention necessarily has maturity≥3). For risk and body nodes, valid_pathway_nodes adds additional restriction beyond what maturity alone can provide.

**Config-specific valid_pathway_nodes:**

Each consimN config has its own valid path set. A node qualifying for consim2 may not qualify for consim0 (requires an EDGE-only path). Use config-specific sets:

| Config | Path file | valid_pathway_nodes semantics |
|--------|-----------|------------------------------|
| consim0 | `paths_unconstrained_edge_only.jsonl` | On ≥1 path with zero SIM hops (EDGE-only) |
| consim1 | `paths_unconstrained_consim1.jsonl` | On ≥1 path with max 1 consecutive SIM hop |
| consim2 | `paths_unconstrained_consim2.jsonl` | On ≥1 path with max 2 consecutive SIM hops |
| unconstrained | `paths_unconstrained_sim0.9.jsonl` | On ≥1 qualifying path (any consec SIM count) |

**Node counts (approximate from Part 9):**
- consim0 valid nodes: ~3,500
- consim1 valid nodes: ~21,000
- consim2 valid nodes: ~21,553
- Risk cluster members (unfiltered): 4,889 — of which ~75.9% appear on consim2 paths

**Required code change — `get_qualifying_clusters()` wrapper:**

Add this function to all Step 4 scripts and use it everywhere a cluster member list is accessed:

```python
# Build this once, after loading valid_pathway_nodes from the config's path file:
def get_qualifying_clusters(edge_config, mode, node_type, algo='agglomerative'):
    """Return cluster members filtered to valid_pathway_nodes (holistic quality cut)."""
    raw = get_clusters(edge_config, mode, node_type, algo)
    return {cid: [n for n in nodes if n in valid_pathway_nodes]
            for cid, nodes in raw.items()}
```

**Every place in Steps 3 and 4 that accesses cluster members must use `get_qualifying_clusters()` or equivalent filtering.**

**Affected locations:**

| Script | Section | Current access | Fix |
|--------|---------|----------------|-----|
| `step4_cluster_naming.py` | Section 2 cluster tables | `get_clusters('0.9','unconstrained','risk/intervention')` | → `get_qualifying_clusters(...)` |
| `step4_cluster_naming.py` | Section 4 Option B signatures | `get_clusters('0.9','unconstrained',subtype)` for body subtypes | → filter `node_to_stc` to valid_pathway_nodes |
| `step4_cluster_naming.py` | Section 5 ARI Jaccard | `rn_unconstrained = all risk cluster nodes` | → restrict denominator to `valid_pathway_nodes ∩ risk nodes` |
| `step4_connectivity.py` | risk_clusters_09 loading | unfiltered (PKL) | → filter to valid_pathway_nodes after loading |
| `step4_connectivity.py` | interv_clusters_09 | maturity≥3 filter only | → replace with valid_pathway_nodes filter |
| `step4_connectivity.py` | Gap analysis cluster counts | raw cluster sizes | → valid_pathway_nodes-filtered sizes |
| `step4b_paths_and_plots.py` | Section 2 cluster tables | `get_clusters(...)` | → `get_qualifying_clusters(...)` |
| `step4b_paths_and_plots.py` | Section 4 Option B signatures | `get_clusters('0.9','unconstrained',subtype)` | → filter `node_to_stc` to valid_pathway_nodes |
| `step4_cluster_naming.py` | Plot 21 within-cluster edge density (lines 557–563) | `get_clusters('0.9','unconstrained',nt)` — unfiltered member denominator | → `get_qualifying_clusters(...)` (edge set `hce_set` has conf≥3 correct already) |
| `step4b_paths_and_plots.py` | Plot 21 within-cluster edge density (lines 599–604) | `get_clusters('0.9','unconstrained',nt)` — unfiltered member denominator | → `get_qualifying_clusters(...)` |
| `step3_validation_and_selection.py` | Sections D/D2 — betweenness graph | node universe = full edge graph | → restrict graph nodes to valid_pathway_nodes (add node filter before G.add_edge) |

**Note on betweenness graph restriction:** Betweenness centrality is a topological property. Restricting the graph to valid_pathway_nodes computes "bridge nodes within the qualifying analysis universe," which is the appropriate framing. Nodes outside valid_pathway_nodes are not part of the analysis; their topological role in the broader graph is irrelevant to the workshop findings.

**Note on the maturity≥3 filter (Gap 2):** Keep the maturity≥3 filter as a belt-and-suspenders check, but it is no longer the primary mechanism. The valid_pathway_nodes filter subsumes it. The order should be: (1) load valid_pathway_nodes, (2) filter cluster members to valid_pathway_nodes, (3) optionally assert that all resulting intervention nodes have maturity≥3 as a sanity check.

**Implementation order:**
1. In connectivity.py: add valid_pathway_nodes loading at script start (currently missing entirely)
2. In all three Step 4 scripts: replace `get_clusters(...)` calls for risk/intervention cluster membership with `get_qualifying_clusters(...)`
3. In Option B body subtype lookup: add `and n in valid_pathway_nodes` to `node_to_stc` building
4. In cluster_naming.py Section 5 ARI: fix denominator to `valid_pathway_nodes ∩ risk_cluster_nodes`
5. In step3_validation_and_selection.py Sections D/D2: add valid_pathway_nodes filter to graph node set

---

### Step 3 Sections Unaffected

| Section | Data source | Status | Reason |
|---|---|---|---|
| Multi-criteria scoring (A) | CSV files only | ✅ clean | No raw edge access |
| EDGE validation % (B) | Path files | ✅ clean | Path files inherit all cuts |
| SIM coverage (C, SIM path only) | edge_data SIM only | ✅ clean | conf not applicable to SIM |

---

### Step 5 Audit

Step 5 scripts were also audited. **Correction from prior version:** the Step 5 audit previously stated "no quality cut gaps." This is incorrect — two of three Step 5 scripts access `cluster_memberships.pkl` without `valid_pathway_nodes` filter.

| Script | SIM≥0.9 | valid_pathway_nodes filter | Status | Detail |
|---|---|---|---|---|
| `phase2_step5_naming.py` | N/A | ❌ not applied | ❌ Gap 5a | `get_cluster_dict()` (line 133) returns all cluster members unfiltered; centroid and representative nodes include non-qualifying members |
| `phase2_step5_examples.py` | N/A | ✅ (lookup-only) | ✅ clean | `node_to_risk`/`node_to_interv` used only to look up cluster IDs for qualifying path endpoints — non-qualifying entries in the dict are never accessed |
| `phase2_step5_triplet_simreach.py` | ✅ applied | ❌ not applied | ❌ Gap 5b | `risk_cluster_nodes`, `interv_cluster_nodes` built from unfiltered `cluster_memberships.pkl` (lines 61–102); SIM reach computed over ALL cluster members → `r_reach`, `c_reach`, `i_reach`, `union_reach`, `triplet_core` are all inflated by non-qualifying nodes |

**Step 5 has two quality cut gaps (5a and 5b) — see Gap 5 below.**

In addition, Step 5 results are currently provisional (based on corrupted consim2 path files from Gap 1 above) and must be rerun after path files are regenerated with the correct sim_edge_set. The Gap 5 cluster membership fixes must be applied before that Step 5 rerun.

---

### Step 1 Audit — Graph Metrics Warning

`phase2_step1_loadandparse.py` calls `compute_graph_metrics()` (line 1226) which builds a NetworkX graph from **all** edges (no conf≥3, no SIM threshold) and stores `betweenness`, `pagerank`, `in_degree`, `out_degree`, `clustering_coefficient` into `node_attrs`, which are then saved to `graph_node_attributes.pkl`.

These unfiltered graph metrics are **only consumed by `analyze_betweenness()` in `phase2_step2_metrics_stability.py`** (line 822 reads `pagerank` from `node_attrs`). The Step 2 betweenness analysis is already superseded by the Step 3 rerun. **No active workshop gap from Step 1 graph metrics.**

⚠️ **WARNING:** `graph_node_attributes.pkl` contains `pagerank`, `betweenness`, `in_degree`, `out_degree`, and `clustering_coefficient` fields computed from an unfiltered graph (all 1,767,833 edges without quality cuts). These fields MUST NOT be used in any Step 4 or Step 5 analyses. Any new analysis requiring graph centrality or degree must recompute from `graph_edge_data.pkl` with appropriate quality cuts applied.

---

### Summary: What Must Be Fixed Before Any Rerun

| Fix | Scripts | Status | Priority |
|-----|---------|--------|----------|
| Add SIM≥0.9 filter to sim_edge_set | `step4_cluster_naming.py`, `step4_connectivity.py`, `step4b_paths_and_plots.py` | ✅ Applied | Done |
| Add maturity≥3 filter to intervention cluster members | All Step 4 scripts | ✅ Applied (partial — superseded by Gap 4) | Done (partial) |
| Add conf≥3 filter to EDGE in betweenness Sections D, D2, F | `step3_validation_and_selection.py` | ✅ Applied | Done |
| **PATHWAY-FIRST: Filter ALL cluster members to valid_pathway_nodes** | All Step 4 scripts + Step 3 Section D2 | ❌ Not yet done | 🔴 **Critical** |
| Load valid_pathway_nodes in connectivity.py | `step4_connectivity.py` | ❌ Not yet done | 🔴 **Critical (prerequisite for above)** |
| Fix risk_clusters_09 loading in connectivity.py (loaded from unfiltered PKL) | `step4_connectivity.py` line 128 | ❌ Not yet done | 🔴 **Critical** |
| Apply valid_pathway_nodes before saving risk_clusters_09.pkl | `step4_cluster_naming.py` line 583, `step4b_paths_and_plots.py` line 620 | ❌ Not yet done | 🔴 **Critical** |
| Filter Step 3 D2 `both_nodes` to valid_pathway_nodes | `step3_validation_and_selection.py` line 1369 | ❌ Not yet done | 🔴 **Required — betweenness graph scope** |
| **Filter within_cluster_edge_density cluster member sets to valid_pathway_nodes** | `step4_cluster_naming.py` lines 557–563, `step4b_paths_and_plots.py` lines 599–604 | ❌ Not yet done | 🔴 **Required — edge density denominator uses unfiltered cluster size; conf≥3 on edge set is correct but member node universe is wrong** |
| Filter Option B body subtype lookup to valid_pathway_nodes | `step4_cluster_naming.py`, `step4b_paths_and_plots.py` Section 4 | ❌ Not yet done | 🟡 Low (labels only, path bodies already filtered) |
| Fix ARI Jaccard denominator to valid_pathway_nodes risk nodes | `step4_cluster_naming.py` Section 5 | ❌ Not yet done | 🟡 Secondary metric |
| Recompute source diversity + maturity per cluster on valid_pathway_nodes | Step 4 rerun of `step2b` logic | ❌ Not yet done | 🟡 Required for workshop reporting |
| Recompute `cluster_source_diversity.csv` (step2 v1) on valid_pathway_nodes | Step 4 rerun of `step2_metrics_stability` source diversity logic | ❌ Not yet done | 🟡 Required — different file from step2b `_v2` |
| Recompute `cluster_temporal_coverage.csv` (NEW GAP) on valid_pathway_nodes | Step 4 rerun of `step2_metrics_stability` temporal coverage logic | ❌ Not yet done | 🟡 Required — not previously documented; year-range distributions must use qualifying nodes only |
| Hub quality (`hub_quality_metrics.csv`, scatter, bar plots) | Fix scripts `phase2_fix_hub_quality_degree.py` + `phase2_fix_hub_quality_sim_degrees.py` already produced valid_pathway_nodes-filtered versions | ✅ **ALREADY DONE** | No Step 4 rerun needed — current files are authoritative clean versions |
| Recompute multi_risk_clusters.csv on valid_pathway_nodes | Step 4 rerun of `step2b` multi-risk logic | ❌ Not yet done | 🟡 Required for workshop reporting |
| Recompute risk_diversity_stats.csv on valid_pathway_nodes | Step 4 rerun of `step2b` risk diversity logic | ❌ Not yet done | 🟡 Required for workshop reporting |
| Recompute category_mechanism_families.csv on valid_pathway_nodes | Step 4 rerun of `step2b` mechanism family logic | ❌ Not yet done | 🟡 Required for workshop reporting |
| mechanism_transfer_betweenness_v2.png: superseded by rerun_pathfiltered.py betweenness outputs | Replace with path-filtered betweenness from `rerun_pathfiltered.py` | ❌ Not yet done | 🟡 Superseded — do not cite step2b version |
| Step 3 Section C sim_coverage: replace cluster proxy with valid_pathway_nodes (consim0 vs consim2) | Step 4 rerun of section C logic | ❌ Not yet done | 🟡 Required — analysis IS workshop dataset characterization |
| Step 3 Section E held-out validation: use valid_pathway_nodes-filtered members | Step 4 rerun of section E logic | ❌ Not yet done | 🟡 Required for accurate workshop reporting |
| **Gap 5a: Filter cluster members to valid_pathway_nodes in naming.py** | `step5_naming.py` `get_cluster_dict()` line 133 | ❌ Not yet done | 🟡 Required before Step 5 rerun — representative nodes must be from qualifying set |
| **Gap 5b: Filter cluster members to valid_pathway_nodes in triplet_simreach.py** | `step5_triplet_simreach.py` lines 61–102 | ❌ Not yet done | 🔴 Required before Step 5 rerun — SIM reach counts are workshop-reported numbers, inflation = incorrect results |
| ⚠️ Step 1 graph metrics warning: do NOT use node_attrs pagerank/betweenness/degree fields in new analyses | `graph_node_attributes.pkl` | Documented | 🔴 Warning (not a code fix — a constraint on new code) |

---

### Gap 5 — CRITICAL: Step 5 cluster membership not filtered to valid_pathway_nodes

**Gap 5a — `phase2_step5_naming.py`**

`get_cluster_dict()` (line 133) iterates `cluster_memberships` and returns all members of risk/intervention clusters without filtering:

```python
def get_cluster_dict(node_type):
    result = {}
    for (ec, mode, nt, algo, cid), members in cluster_memberships.items():
        if ec == 0.9 and mode == "unconstrained" and nt == node_type and algo == "agglomerative":
            result[int(cid)] = list(members)   # ← unfiltered
    return result
```

This unfiltered member list is used for cluster centroid computation and top-5 representative node selection (the nodes whose names/descriptions feed into the LLM naming prompt). If non-qualifying nodes (up to ~24% of risk cluster members) are included, the centroid may be shifted and the representative nodes may not be from the workshop analysis universe.

**Fix:** After building the result dict, intersect all member lists with `valid_pathway_nodes` (config-appropriate):
```python
return {cid: [n for n in members if n in valid_pathway_nodes]
        for cid, members in result.items()}
```

---

**Gap 5b — `phase2_step5_triplet_simreach.py`**

`node_to_risk` and `node_to_interv` are built from raw `cluster_memberships` (lines 61–70) without any valid_pathway_nodes filter. These are expanded into `risk_cluster_nodes` and `interv_cluster_nodes` (lines 92–98) which include ALL cluster members. The SIM reach (`cluster_sim_reach()`) is then computed over ALL cluster members including non-qualifying nodes:

```python
for nid, cid in node_to_risk.items():         # ← includes non-qualifying risk nodes
    risk_cluster_nodes[cid].add(nid)
# ...
r_urls = cluster_sim_reach(r_nodes)           # ← inflated by non-qualifying nodes
```

**Impact:** `r_reach`, `c_reach`, `i_reach`, `union_reach`, `triplet_core` are all inflated. The triplet SIM reach metric is a workshop-reported number (how many papers does this risk→chain→intervention triplet collectively reference via semantic similarity). If non-qualifying nodes contribute partners that qualifying nodes would not, the count is wrong.

**Fix:** After building `node_to_risk` and `node_to_interv`, intersect with `valid_pathway_nodes`:
```python
node_to_risk   = {nid: cid for nid, cid in node_to_risk.items() if nid in valid_pathway_nodes}
node_to_interv = {nid: cid for nid, cid in node_to_interv.items() if nid in valid_pathway_nodes}
```

This requires loading `valid_pathway_nodes` from the selected config's path file at script startup.

---

**`phase2_step5_examples.py` — no fix needed:**

`node_to_risk` and `node_to_interv` are used only for `node_to_risk.get(start)` / `node_to_interv.get(end)` where `start` and `end` are endpoints from a qualifying path file. These endpoints are already in valid_pathway_nodes. The non-qualifying entries in the dict are never looked up and do not affect the output. ✅

---

**Implementation note:** All Step 5 scripts are provisional and will be rerun after Step 4 config selection and Gap 1 path file regeneration. The Gap 5 fixes (5a and 5b) must be applied before the Step 5 rerun. They do not require separate reruns of earlier steps.

---

## Part 4 — Cluster Membership Scope

### What Nodes Are In cluster_memberships.pkl

`cluster_memberships.pkl` contains nodes assigned by the UMAP clustering pipeline (SIM≥0.9/unconstrained/agglomerative). The SIM≥0.9 quality cut is correctly applied as the clustering edge_config. However, **the PKL is NOT filtered by intervention maturity or by path participation**. It represents the full node set used for embedding-based clustering.

**Confirmed node counts:**

| Node type | In cluster_memberships.pkl | In graph_node_attributes.pkl |
|---|---|---|
| risk | 4,889 | 19,096 |
| intervention (all) | 2,970 | 36,959 |
| intervention maturity≥3 | **2,815** | — |
| intervention maturity<3 | **155** (15 mat=1, 140 mat=2) | — |

**The authoritative membership filter for all analyses: valid_pathway_nodes**

All Step 4 analyses must filter cluster members to `valid_pathway_nodes` (nodes appearing on qualifying paths in the path file). This is the holistic quality cut — see Gap 4 for full rationale.

```python
# Correct filter — replace all isolated maturity/conf/sim cuts with this:
def get_qualifying_clusters(edge_config, mode, node_type, algo='agglomerative'):
    raw = get_clusters(edge_config, mode, node_type, algo)
    return {cid: [n for n in nodes if n in valid_pathway_nodes]
            for cid, nodes in raw.items()}
```

The maturity≥3 filter (Gap 2) is a necessary condition for intervention endpoints and is kept as a belt-and-suspenders check, but `valid_pathway_nodes` is the primary filter.

### Cluster Membership Filtration Hierarchy

```
All graph nodes (200,525 in node_attrs)
  ↓ UMAP pipeline (embedding quality, non-satellite nodes)
Cluster members in PKL (4,889 risk / 2,970 intervention at SIM≥0.9/unconstrained)
  ↓ valid_pathway_nodes filter (holistic: EDGE conf≥3 + SIM≥0.9 + maturity≥3, all simultaneous)
Qualifying cluster members (~3,712 risk / ~2,815 intervention on consim2)
  ↓ consimN filter (config-specific valid_pathway_nodes subset)
Members for consimN analysis
  └── consim0: ~3,500 total nodes (strictest)
  └── consim1: ~21,000 total nodes
  └── consim2: ~21,553 total nodes (broadest)
```

**Key point:** The valid_pathway_nodes filter is applied AFTER loading cluster_memberships.pkl. It is NOT a re-clustering — cluster IDs are unchanged. We are simply restricting which PKL-assigned nodes participate in the analysis.

**Implications:**
1. **Edge-only path fraction is a genuine quality signal** — fraction of qualifying cluster members (already in valid_pathway_nodes_consimN) that ALSO appear in valid_pathway_nodes_consim0. Not circular.
2. **consim0 edge-only path fraction = 1.0 by definition** — the consim0 member set IS valid_pathway_nodes_consim0. Trivial, not reported.
3. **Naming uses top-representative nodes by centroid similarity** — correct since cluster semantic identity reflects the full embedding neighborhood. Naming inputs use qualifying members only.
4. **Gap analysis cluster sizes** — must use valid_pathway_nodes-filtered sizes, not raw PKL sizes. A cluster with zero qualifying members is an "empty" cluster in this config and should not count as a gap source.

### consim0/1/2 Purpose

The three configs serve two simultaneous purposes:

1. **Show which connections arise from cross-paper inference:** consim0 contains only connections a single paper argues end-to-end. consim1/2 add connections where the field has converged across papers via semantic equivalence. The diff between configs reveals the cross-paper structure of the literature.

2. **Constrain unjustified SIM-only connections:** paths with too many consecutive SIM hops have no EDGE anchor — they travel entirely via semantic similarity with no single paper's explicit argument grounding any step. The consimN limit prevents taxonomy families from being built on chains that are purely embedding-similar with no structural logical grounding. This is not about hallucination per se, but about ensuring every qualifying path has at least some EDGE-grounded structural backbone.

---

## Part 5 — Config Naming Convention

| Config name | Max consecutive SIM | Path body method | Paths | Prior name |
|-------------|---------------------|-----------------|-------|------------|
| `consim0_pathbuildA` | 0 (edge-only) | Option A (KMeans on mean body emb) | 3,405 | — |
| `consim0_pathbuildB` | 0 (edge-only) | Option B (subtype co-occurrence) | 3,405 | — |
| `consim1_pathbuildA` | ≤1 | Option A | 75,008 | VarA |
| `consim1_pathbuildB` | ≤1 | Option B | 75,008 | VarA |
| `consim2_pathbuildA` | ≤2 | Option A | 432,776 | VarB |
| `consim2_pathbuildB` | ≤2 | Option B | 432,776 | VarB |

`consimN` = max consecutive SIM hops in the full path. `pathbuildA/B` = method to build L2 chain body families.

**Output file naming:** all outputs suffixed with full config name:
`risk_clusters_consim2_pathbuildA.csv`, `three_layer_network_consim1_pathbuildA.png`, etc.

**Note on consim0:** `paths_unconstrained_edge_only.jsonl` contains 3,405 unique EDGE-throughout paths. KMeans k=40 may yield very small clusters — reduce to k=10 if mean cluster size < 5, and report adjusted k.

---

## Part 6 — Substep Execution Plan

### Substep #25 — Build Cluster Tables (all 6 configs)

**Category B — all cluster table fields must use `get_qualifying_clusters()` (valid_pathway_nodes-filtered members), not raw PKL counts.**

**L1 and L3 per config:**
- N nodes total in cluster (from PKL — reference count, Category A context only — not a workshop-reported finding)
- **N nodes qualifying for this config** (`valid_pathway_nodes`-filtered — this is the Category B workshop-reported cluster size)
- Edge-only path fraction (consim1 and consim2 only — fraction of qualifying cluster nodes also on consim0 paths)
- N unique source paper URLs for qualifying nodes only
- Cluster centroid computed over qualifying members; per-node cosine similarity to centroid
- Top-5 representative nodes by centroid similarity from qualifying members only (x-risk near-duplicates deduplicated: pairwise cos_sim ≥0.95 → keep one)
- Within-cluster EDGE density: conf≥3 EDGE edge count / (n_qualifying × (n_qualifying − 1)), where n_qualifying = qualifying member count

**L2 pathbuildA per config:**
1. Load path file → extract body nodes (problem_analysis, theoretical_insight, design_rationale, implementation_mechanism, validation_evidence)
2. Parse embeddings from node_attrs (FalkorDB string `'<v1,...,v1536>'` → np.float32)
3. Mean embedding per path body → KMeans k=40 (k=10 for consim0)
4. Per cluster: n_paths, n_unique_source_urls, top-5 paths by centroid sim, subtype distribution, edge-only path fraction

**L2 pathbuildB per config:**
1. Load path file → for each body node, look up (subtype, cluster_id) from `cluster_memberships.pkl`
2. Per-path signature = frozenset of (subtype, cluster_id) pairs
3. Group by exact signature; keep n_paths ≥ 5
4. Per family: n_paths, dominant signature, n_unique_source_urls, top-3 representative paths

### Substep #26 — Cluster Naming (all 6 configs)

LLM naming for all ~80 clusters per config. Input: top-5 representative node names/descriptions, cluster size, edge-only path fraction, n_sources. For L2: connected L1 and L3 clusters.

**L2 naming test:** good label reads as "the body of thinking about [X] that connects [risk] to [intervention]." Generic labels → flag as incoherent.

**Config selection scoring** documented in `step4_config_selection.md` after all 6 versions complete.

### Substep #27 — Three-Level Connectivity + Visualization (all 6 configs)

See Part 1 definitions for connectivity analysis and gap analysis.

**Category B — all connectivity computations must use `get_qualifying_clusters()` for cluster member sets. Gap analysis cluster counts must use valid_pathway_nodes-filtered sizes. A cluster with zero qualifying members is "empty" for this config and must be excluded from the gap analysis universe.**

**EDGE connectivity fix required:** explicitly apply `(e.get('confidence') or 0) >= 3` filter when reading EDGE edges from `graph_edge_data.pkl`. This must be in ALL connectivity scripts for ALL configs.

**Gap interpretation:** gaps that exist in consim0 but disappear in consim1/consim2 represent **cross-paper convergent connections** — the field has established these connections via semantic convergence across papers, but no single paper traces the full argument chain end-to-end. Report as "cross-paper convergent connections" in the paper, not as artifacts.

**Three-layer network visualization (per config, full color coding):**

Node color coding:
- L1 Risk: x-risk (dark red), near-term capability risk (orange), governance/oversight failure (purple), other (grey)
- L2 Chain: technical alignment (blue), oversight & governance (green), near-term safety (yellow), field-building (teal), other (grey)
- L3 Intervention: design lifecycle (light blue), training lifecycle (medium blue), deployment lifecycle (dark blue)

Node size ∝ N nodes in cluster. Edge width ∝ N qualifying paths.

Outputs per config: `three_layer_network_[config].png` (Sankey), `three_layer_network_detail_[config].png` (labeled node-link).

### Substep #28 — Subcluster Analysis (all 6 configs)

**Category B — subcluster analysis operates on valid_pathway_nodes-filtered cluster members. All trigger thresholds (silhouette, size, category diversity) are evaluated on qualifying members only.**

Triggers (evaluated on qualifying member count): within-cluster silhouette < 0.3, cluster size > 100 qualifying nodes, top-5 qualifying nodes span > 2 concept_categories, LLM naming produces > 1 candidate. For triggered clusters: agglomerative k=5 subclustering on qualifying members, then LLM naming per subcluster.

### Substep #29 — Cross-Config Stability

1. ARI(consim0, consim1) for L1/L3 nodes in both path sets
2. ARI(consim1, consim2) for L1/L3 nodes in both path sets
3. ARI > 0.7 = taxonomy stable; < 0.7 = config choice materially changes taxonomy

### Conf≥3 Gap Reruns (all steps)

All scripts that read EDGE edges directly from `graph_edge_data.pkl` must be fixed and rerun. The fix in every case is one filter line: `and (e.get('confidence') or 0) >= 3` applied when iterating edges of type EDGE.

**Step 3 — Betweenness scripts:**

| Script | Sections affected | Fix location |
|--------|------------------|--------------|
| `phase2_step3_validation_and_selection.py` | Section D (unconstrained betweenness) | line ~1109: `elif etype == "EDGE": G.add_edge(...)` → add conf≥3 guard |
| `phase2_step3_validation_and_selection.py` | Section D2 (both-mode betweenness) | line ~1380: same pattern |
| `phase2_step3_validation_and_selection.py` | Section F (edge subgraph stats) | line ~1664: same pattern |
| `phase2_step3_rerun_betweenness_sectionf.py` | Section F standalone rerun | same pattern |

Expected outcome: x-risk hubs still dominate betweenness top-50. Numbers will shift slightly (removing ~38% of EDGE edges with conf<3) but node identity is robust.

**Step 4 — Connectivity scripts:**

| Script | Fix location |
|--------|--------------|
| `phase2_step4_connectivity.py` (existing, for consim2) | EDGE edge filter for all three connectivity matrices |
| New consim0/consim1 connectivity scripts | Apply from the start — do not inherit the unfixed pattern |

All consimN connectivity scripts must apply conf≥3 before computing any EDGE edge count between cluster members.

After all reruns: update Step 3 review summary (Section D, D2, F numbers) and Step 4 review summary (connectivity matrices, gap analysis). Note in methods section that all betweenness and connectivity analyses apply EDGE conf≥3 consistently.

---

## Part 7 — Step 5 Sequencing

**Recommendation: complete config selection before doing Step 5.**

Current Step 5 outputs (`step5_naming/`, `step5_examples/`) are provisional — produced on `consim2_pathbuildA` and `consim2_pathbuildB` only. If a different config is selected, Step 5 must be redone on the selected config.

Step 5 adds on top of Step 4 naming:
- gpt-5.4-mini naming pass (richer descriptions than centroid-representative names)
- Pathway examples (technical chains, field-building chains)
- Extraction artifact deep-dive per cluster
- Per-cluster path participation analysis (consim1 vs consim2 ratio)
- Cross-paper SIM reach per cluster
- Triplet cross-layer SIM reach analysis
- Human review checklist

**Simulation prep** (path→prompt templates, simulation-ready database) is deferred beyond Step 5 to a separate Phase 3 effort. Not in scope for Step 4 or Step 5.

---

## Part 8 — Scope Audit

**Last updated:** 2026-04-05. **Selected config: consim1_pathbuildA.**
Note: L1/L3 cluster assignments are the SAME across all 6 configs (same PKL, same SIM≥0.9 agglomerative k=40). Only L2 chain body clustering differs per config. L1/L3 tables, connectivity, and gap analysis are therefore config-independent at the cluster taxonomy level; only path counts change.

| Analysis | consim2_A | consim2_B | consim1_A ★SELECTED | consim1_B | consim0_A | consim0_B |
|---|---|---|---|---|---|---|
| L1/L3 cluster tables | ✅ | same | ✅ | same | ✅ | same |
| Edge-only path fraction | ✅ | — | ✅ | — | n/a (trivial=1.0) | — |
| L2 chain body clustering (A) | ✅ k=40 | — | ✅ k=40 | — | ✅ k=10 | — |
| L2 chain body families (B) | — | ✅ 16,034 fam | — | ✅ 1,603 fam | — | ✅ 51 fam |
| Connectivity pathbuildA | ✅ | — | ✅ | — | ✅ | — |
| Connectivity pathbuildB (R→B→I) | — | ✅ (2026-04-05) | — | ✅ (2026-04-05) | — | ✅ (2026-04-05) |
| Gap analysis pathbuildA | ✅ | — | ✅ | — | ✅ | — |
| Gap analysis pathbuildB | — | ✅ (2026-04-05) | — | ✅ (2026-04-05) | — | ✅ (2026-04-05) |
| Three-layer network (basic) | ✅ | — | ✅ (2026-04-05) | — | ✅ (2026-04-05) | — |
| Three-layer network (color) | ✅ (2026-04-05) | — | ✅ (2026-04-05) | — | ✅ (2026-04-05) | — |
| Subcluster identification | ✅ 36 flagged | — | same PKL | — | same PKL | — |
| Subcluster naming (k=5+LLM) | ✅ (2026-04-05) | — | ✅ | — | ✅ | — |
| Cross-config ARI | ✅ ARI=1.0 | — | ✅ | — | — | — |
| Cross-config R→I pair analysis | ✅ | — | ✅ | — | ✅ | — |
| Config selection decision | — | — | ✅ consim1_pathbuildA | — | — | — |
| Step 3 betweenness rerun | ✅ | — | — | — | — | — |
| Phase C qualifying reruns | ✅ all 12 | — | — | — | — | — |
| Step 5 LLM naming | — | — | ✅ 105/120 high | n/a (sel. config) | n/a | n/a |
| Step 5 pathway examples | — | — | ✅ consim1 paths | n/a | n/a | n/a |
| Step 5 triplet SIM reach | — | — | ✅ VPN-filtered | n/a | n/a | n/a |

**All planned items complete (last updated 2026-04-05):**
- ✅ Color-coded three-layer networks — 6 plots + cluster_color_categories.csv produced
- ✅ Subcluster naming — 180 subclusters, 96.1% high confidence; only I9 splits meaningfully
- ✅ PathbuildB R→B→I connectivity (substep 27) — `phase2_step4_pathbuildB_connectivity.py` (2026-04-05); 100% family match for all 3 configs; no orphaned families
- ✅ UMAP per-consim config — 6 plots (consim0/1/2 × risk/intervention) with maturity≥3 fix (2026-04-05); see `step4_finalanalysis/umap_*_consimN.png`
- ✅ Maturity filter root cause fix — `valid_pathway_nodes` alone does not guarantee maturity≥3 (path generator used unfiltered `ALL_INTERVENTION_IDS` cache); all Category B scripts now apply both filters (fixed in `connectivity.py` and `umap_plots.py`, 2026-04-05)

**Priority order:**
**Phase A — Code fixes (before any rerun):**

1. ✅ SIM≥0.9 in sim_edge_set — all Step 4 scripts
2. ✅ conf≥3 in betweenness Sections D, F — step3_validation_and_selection.py
3. ✅ maturity≥3 on intervention cluster members — all Step 4 scripts (belt-and-suspenders)
4. ✅ **Add valid_pathway_nodes loading to `connectivity.py`** (Gap 4 fix)
5. ✅ **Filter `risk_clusters_09` after loading PKL in `connectivity.py`** (Gap 3b fix)
6. ✅ **Filter `risk_clusters_09` before saving PKL in `cluster_naming.py` and `step4b.py`** (Gap 3b fix)
7. ✅ **Replace all `get_clusters()` membership calls with `get_qualifying_clusters()`** in all Step 4 scripts (Gap 4 fix)
8. ⚠️ **Restrict Step 3 D2 `both_nodes` to valid_pathway_nodes** (Gap 3c) — Step 3 output superseded by path-filtered betweenness; Step 4 does not use D2 output
9. ✅ Filter Option B body subtype lookup (`node_to_stc`) to valid_pathway_nodes
10. ✅ Fix ARI Jaccard denominator in `cluster_naming.py` Section 5
11. ✅ **Root cause fix: maturity≥3 NOT guaranteed by valid_pathway_nodes** — path generation used `ALL_INTERVENTION_IDS` cache (includes maturity<3 nodes). Added explicit `intervention_maturity≥3` filter on top of valid_pathway_nodes in `connectivity.py` and `umap_plots.py` (2026-04-05). All existing analyses already had maturity≥3 as belt-and-suspenders in earlier fix — this note documents the root cause.

**Phase B — Core Step 4 analyses (all 6 configs):**

11. ✅ consim0 pathbuildA + pathbuildB; consim1 pathbuildA + pathbuildB; consim2 pathbuildA + pathbuildB
12. ✅ cluster tables (risk, intervention, chain) on VPN-filtered members for all configs
13. ✅ edge-only path fraction for consim1 and consim2
14. ✅ Connectivity + gap analysis for all configs (pathbuildA) ✅ PathbuildB R→B→I connectivity for all 3 consims (substep 27 — `phase2_step4_pathbuildB_connectivity.py`, 2026-04-05)
15. ✅ Subcluster identification (36 flagged); ✅ subcluster naming (180 subclusters, 2026-04-05)
16. ✅ Three-layer network (basic + color-coded) for all 3 consimN configs (2026-04-05)
17. ✅ Cross-config ARI stability test (ARI=1.0)
18. ✅ Config selection decision → `step4_config_selection.md`

**Phase C — Step 2/3 analyses re-run in Step 4 on qualifying node set:**

19. ✅ **Hub quality: ALREADY DONE** — `phase2_fix_hub_quality_degree.py` + `phase2_fix_hub_quality_sim_degrees.py` already produced valid_pathway_nodes-filtered `hub_quality_metrics.csv`, scatter, and bar plots. Do NOT rerun from Step 4.
20. Source diversity per cluster (step2b `cluster_source_diversity_v2.csv`): recompute on valid_pathway_nodes members → `step4_finalanalysis/source_diversity_qualifying.csv`
21. Maturity distribution per cluster (step2b `maturity_distribution_heatmap.png`): recompute on valid_pathway_nodes members → `step4_finalanalysis/maturity_distribution_qualifying.png`
22. Multi-risk cluster analysis (`multi_risk_clusters.csv`): recompute on valid_pathway_nodes members → `step4_finalanalysis/multi_risk_clusters_qualifying.csv`
23. Risk diversity stats (`risk_diversity_stats.csv`): recompute on valid_pathway_nodes members → `step4_finalanalysis/risk_diversity_qualifying.csv`
24. Mechanism family categorization (`category_mechanism_families.csv`): recompute on valid_pathway_nodes members → `step4_finalanalysis/mechanism_families_qualifying.csv`
25. Source diversity v1 (`cluster_source_diversity.csv` from step2 not step2b — **new gap**): recompute on valid_pathway_nodes → `step4_finalanalysis/source_diversity_v1_qualifying.csv`
26. Temporal coverage per cluster (`cluster_temporal_coverage.csv` + `temporal_coverage.png` — **new gap, not previously documented**): recompute on valid_pathway_nodes → `step4_finalanalysis/temporal_coverage_qualifying.{csv,png}`
26b. Lifecycle distribution per cluster (`lifecycle_distribution.png` — **new gap**): recompute on valid_pathway_nodes → `step4_finalanalysis/lifecycle_distribution_qualifying.png`
27. SIM coverage (anchored vs SIM-only): reimplement using valid_pathway_nodes_consim0 vs valid_pathway_nodes_consim2 as the anchor/SIM-only criterion (not cluster proxy)
28. Held-out validation: rerun on valid_pathway_nodes-filtered cluster members

**Phase D — Step 5 (after config selection) — code fixes required before rerun:**

Pre-rerun code fixes (apply before running any Step 5 script):
- ✅ **Gap 5a:** `step5_naming.py` `get_cluster_dict()` — filter returned members to valid_pathway_nodes (fixed 2026-04-05)
- ✅ **Gap 5a (chain):** `step5_naming.py` `derive_chain_clusters_from_paths()` — now uses consim1 path file (selected config) instead of consim2 (fixed 2026-04-05)
- ✅ **Gap 5b:** `step5_triplet_simreach.py` — `node_to_risk` and `node_to_interv` now filtered to valid_pathway_nodes (fixed 2026-04-05)
- ✅ **step5_examples.py** — now uses consim1 path file and consim1 ri_edges CSV (fixed 2026-04-05); VPN filter on node_to_risk/interv
- ⚠️ **Step 1 graph metric fields warning:** Do NOT use `node_attrs[nid]['pagerank']`, `['betweenness']`, `['in_degree']`, `['out_degree']`, `['clustering_coefficient']` in any Step 5 code — these are computed from an unfiltered graph in Step 1 and are only valid for the superseded Step 2 betweenness

After fixes:

29. ✅ LLM naming on selected config (step5_naming.py with Gap 5a fix) — 2026-04-05; 105/120 high confidence
30. ✅ Pathway examples (step5_examples.py — fixed for consim1) — 2026-04-05; top-15 R→I + gap clusters
31. ✅ Triplet SIM reach analysis (step5_triplet_simreach.py with Gap 5b fix) — 2026-04-05; R10→C10→I8 core=49
32. ✅ Artifacts, SIM reach, and human review checklist — Step5_Review_Summary.md produced
33. ✅ EDGE-only top-20 examples regenerated (phase2_step5_examples_edgeonly_fix.py) — 2026-04-05; 3,473 paths, 610 pairs, top-20 with 3 examples each; root cause: old script skipped length-2 paths during KMeans predict
34. ✅ Option B family examples (pathway_examples_optionB.json) — 2026-04-05; top-10 families × 3 examples each from consim1
35. ✅ Code fix: phase2_step4_cluster_naming.py — removed MAX_VECTORS=100,000 sampling for KMeans and path output (replaced with MiniBatchKMeans streaming); note: current on-disk outputs (produced by step4b) were already correct (no sampling)

---

## Part 9 — Reference: Path Counts and Files

| Config | Path file | N paths |
|--------|-----------|---------|
| consim0 | `paths_unconstrained_edge_only.jsonl` | 3,405 |
| consim1 | `paths_unconstrained_consim1.jsonl` | 75,008 |
| consim2 | `paths_unconstrained_consim2.jsonl` | 432,776 |

All files: EDGE conf≥3, intervention maturity≥3, SIM cos_sim≥0.9, unconstrained mode, full-path consecutive SIM measurement.

### Node Coverage

| Config | Unique nodes | % of node_attrs |
|--------|-------------|-----------------|
| consim0 | ~3,500 | ~1.7% |
| consim1 | ~21,000 | ~10.5% |
| consim2 | ~21,553 | ~10.7% |

consim1 and consim2 have nearly identical node coverage — the extra 360K paths in consim2 reuse the same nodes via different routes.

---

## Part 10 — File Overwrite Safety Rules

Every script rerun must preserve all Category A files exactly as they are. The rules below govern which scripts may be rerun, under what conditions, and where their outputs must go.

### Rule 1 — NEVER rerun `phase2_step3_rerun_betweenness_sectionf.py`

This script writes `betweenness_sim09.csv`, `betweenness_bridge_clusters.csv`, `betweenness_comparison.png`, `betweenness_both09.csv`, `betweenness_both09_bridge_clusters.csv`, `betweenness_both09_comparison.png` to `step3_validation_and_selection/` — the same filenames that `phase2_rerun_pathfiltered.py` writes. The current files in that directory are the clean path-filtered versions from `rerun_pathfiltered.py`. **Rerunning `step3_rerun_betweenness_sectionf.py` would silently overwrite the clean outputs with the maturity-filtered (non-compliant) versions.** This script must never be rerun. If betweenness needs to be regenerated, rerun `phase2_rerun_pathfiltered.py` only.

### Rule 2 — `phase2_step3_validation_and_selection.py`: Use `--skip-betweenness` flag whenever rerunning

Section D and D2 of this script write to the same `betweenness_sim09.csv`, `betweenness_both09.csv`, etc. filenames (and their PNG/CSV companions). **The script supports a `--skip-betweenness` command-line flag** that suppresses Section D. If this script is rerun for any reason (e.g., Section C or E fixes), ALWAYS pass `--skip-betweenness` to prevent overwriting the clean path-filtered outputs from `phase2_rerun_pathfiltered.py`.

Available flags:
- `--skip-betweenness`: skips Section D (run A/B/C/E/F safely)
- `--csv-only`: runs only Sections A+B
- `--betweenness-only`: runs ONLY Section D (never use without valid_pathway_nodes fix in place)
- `--betweenness-both`: runs ONLY both-mode betweenness (never use without valid_pathway_nodes fix in place)

| Script | Sections safe to rerun | Command to use | Sections that MUST be skipped |
|--------|------------------------|----------------|-------------------------------|
| `phase2_step3_validation_and_selection.py` | A, B, C, E, F | `python script.py --skip-betweenness` | **D and D2** — would overwrite `betweenness_sim09.csv` and related files |
| `phase2_step3_rerun_betweenness_sectionf.py` | — | DO NOT RUN | **ALL** — entire script is superseded; running it overwrites clean outputs |
| `phase2_rerun_pathfiltered.py` | All sections | Run normally | None — this is the authoritative clean betweenness source |

### Rule 2a — `phase2_rerun_pathfiltered.py` creates `_unfiltered` backups before overwriting

When `phase2_rerun_pathfiltered.py` runs, it renames existing files to `*_unfiltered.*` before writing the new path-filtered versions. This backup mechanism is already implemented in the script (lines 109–110 and 222–224). The `_unfiltered` variants are reference copies of the pre-rerun outputs and are safe to keep. They confirm that the current files ARE the clean path-filtered versions.

### Rule 3 — Phase C reruns (Step 2b Category B analyses) must write to `step4_finalanalysis/`

When Step 4 reimplements the Category B analyses from Step 2b (source diversity, maturity, hub quality, multi-risk clusters, risk diversity, mechanism families), **the output files must go to `step4_finalanalysis/` with `_pathfiltered` or `_qualifying` suffix**, NOT back into `step2_metrics_and_stability/`.

| Step 2/2b file | Status | Step 4 output location |
|----------------|--------|------------------------|
| `step2_metrics_and_stability/cluster_source_diversity_v2.csv` | ❌ Unfiltered — needs rerun | `step4_finalanalysis/source_diversity_qualifying.csv` |
| `step2_metrics_and_stability/cluster_source_diversity.csv` (v1) | ❌ Unfiltered — needs rerun | `step4_finalanalysis/source_diversity_v1_qualifying.csv` |
| `step2_metrics_and_stability/cluster_temporal_coverage.csv` | ❌ Unfiltered — **new gap** | `step4_finalanalysis/temporal_coverage_qualifying.csv` |
| `step2_metrics_and_stability/maturity_distribution_heatmap.png` | ❌ Unfiltered — needs rerun | `step4_finalanalysis/maturity_distribution_qualifying.png` |
| `step2_metrics_and_stability/hub_quality_metrics.csv` | ✅ **ALREADY CLEAN** — fix scripts applied valid_pathway_nodes | Preserve as-is; no overwrite |
| `step2_metrics_and_stability/hub_quality_scatter*.png`, `hub_quality_bar_v2.png` | ✅ **ALREADY CLEAN** — fix scripts applied valid_pathway_nodes | Preserve as-is; no overwrite |
| `step2_metrics_and_stability/multi_risk_clusters.csv` | ❌ Unfiltered — needs rerun | `step4_finalanalysis/multi_risk_clusters_qualifying.csv` |
| `step2_metrics_and_stability/risk_diversity_stats.csv` | ❌ Unfiltered — needs rerun | `step4_finalanalysis/risk_diversity_qualifying.csv` |
| `step2_metrics_and_stability/category_mechanism_families.csv` | ❌ Unfiltered — needs rerun | `step4_finalanalysis/mechanism_families_qualifying.csv` |

The step2_metrics_and_stability/ Category A files must remain untouched:

| Preserved Category A file | Why preserved |
|---------------------------|---------------|
| `stability_ari_pairwise.csv` | Cross-threshold ARI — algorithm selection metric |
| `cohesion_analysis.csv` | Cluster cohesion — algorithm selection metric |
| `cluster_centroid_similarity.csv` | Centroid similarity — config selection metric |
| `cluster_edge_purity.csv` | Edge purity — algorithm selection metric |
| `cross_threshold_ari_lineplot.png` | ARI plot — config selection |
| `edge_validation_per_mode.png` | Edge validation — mode comparison |
| `silhouette_by_nodetype_v2.png` | Silhouette — algorithm comparison |
| `cluster_size_distributions_v2.png` | Cluster sizes — algorithm characterization |
| `centroid_similarity_heatmap.png` | Centroid sim heatmap — config selection |
| `edge_purity_histograms.png` | Edge purity histograms — algorithm comparison |
| `path_length_sensitivity.png` | Path length sensitivity — config selection |
| `mode_comparison_stats.csv` | Mode comparison — config selection |
| `edge_density_heatmap.png` | Edge density — algorithm characterization |
| `mode_stability_heatmap.png` | Mode stability — config selection |
| `algorithm_comparison.csv` | Algorithm comparison — definitively Category A |
| `algorithm_comparison_silhouette.png` | Algorithm comparison silhouette plot |
| `edge_validation_per_mode_v2.png` | Updated edge validation — config selection |

The following are **Category B files already cleaned by fix scripts** — preserve but do NOT include in any "Category A" claim:

| Preserved Category B (clean) file | Status |
|-----------------------------------|--------|
| `hub_quality_metrics.csv` | ✅ Clean — `phase2_fix_hub_quality_sim_degrees.py` applied valid_pathway_nodes (built from path files); this is the authoritative qualifying hub ranking |
| `hub_quality_scatter.png` | ✅ Clean — generated from clean `hub_quality_metrics.csv` |
| `hub_quality_scatter_v2.png` | ✅ Clean — same |
| `hub_quality_bar_v2.png` | ✅ Clean — same |

### Rule 4 — Step 4 scripts write to `step4_finalanalysis/`; do not write back to step2/ or step3/ directories

All new Step 4 script outputs (cluster tables, connectivity matrices, gap analysis, maturity/source/hub reruns, network visualizations) must use `step4_finalanalysis/` as their output directory. Outputs from corrected within_cluster_edge_density and maturity plots go to `step4_finalanalysis/`, not to the directory the Step 4 cluster_naming.py currently uses.

### Rule 5 — `phase2_step2_metrics_stability.py` must not be rerun with modifications

This script produced the foundational Category A outputs used for algorithm selection. It must not be modified or rerun. All its outputs are Category A and are final.

### Rule 6 — Verify output file inventory before any rerun

Before running any script, compare the current file listing of `step3_validation_and_selection/` and `step2_metrics_and_stability/` against the Category A preservation lists above. After each rerun, verify that all preserved files still have their pre-rerun modification timestamps or identical content.

---

## Part 12 — Items Dropped

| Item | Reason |
|------|--------|
| node_migration_heatmap (Plot 20) | Migration rate is a metric artifact |
| Betweenness D3 (single_risk subgraph) | Superseded by 3-level connectivity |
| Zig-zag path validation (Test 3) | Superseded by consec SIM config framework |
| mode_stability_heatmap (Plot 22) | Captured in Step 3 threshold sensitivity |
| Fleiss' kappa inter-rater reliability | Absorbed into naming protocol |
| ARI=1.0 claim as taxonomy stability | Was circular — replaced by cross-config ARI test |
| "edge_purity=1.0 for all clusters" | Was circular — replaced by edge-only path fraction |
| Simulation prep | Deferred beyond Step 5 to Phase 3 |

---

## Document Changelog

**v9 (2026-04-04):**
- Two exhaustive background agent audits of all Phase 2 scripts (first audit: 8 scripts; second audit: 6 remaining scripts including step2_metrics_stability.py and fix scripts).
- **Corrected `phase2_step3_rerun_betweenness_sectionf.py` audit row**: was incorrectly marked ✅ clean; agent confirmed it applies maturity≥3 only (NOT valid_pathway_nodes). Audit table and Gap 3 note both corrected. `phase2_rerun_pathfiltered.py` is the sole authoritative path-filtered betweenness rerun.
- **New Part 10 — File Overwrite Safety Rules**: Documents which scripts must not be rerun, `--skip-betweenness` flag for step3_validation_and_selection.py, `_unfiltered` backup mechanism in rerun_pathfiltered.py, Step 4 output path conventions, Category A preservation list, and corrected hub quality status.
- **Hub quality already clean**: `phase2_fix_hub_quality_degree.py` + `phase2_fix_hub_quality_sim_degrees.py` both correctly build valid_nodes from path files and restrict all degree/SIM computations to qualifying nodes. The current `hub_quality_metrics.csv` and scatter/bar plots ARE the valid_pathway_nodes-filtered authoritative outputs. Phase C item 19 marked ✅ ALREADY DONE; Step 2 audit table updated; hub quality moved from "Category A preserve" to "Category B (already clean) preserve."
- **Three new Step 2 (`step2_metrics_stability.py`) Category B gaps added**:
  - `cluster_source_diversity.csv` (v1) — unfiltered source diversity, different from step2b's `_v2` file; rerun in Step 4
  - `cluster_temporal_coverage.csv` — **entirely new gap not previously documented anywhere**; year-range distributions per cluster without valid_pathway_nodes; rerun in Step 4
  - `mechanism_transfer_betweenness.csv` (step2 original) — superseded by rerun_pathfiltered.py
- **Four new Step 2b Category B gaps added**: `multi_risk_clusters.csv`, `risk_diversity_stats.csv`, `category_mechanism_families.csv`, `mechanism_transfer_betweenness_v2.png`
- Phase C renumbered to items 19–28 (hub quality marked done; items 20–26 are reruns; 27–28 are SIM coverage and held-out validation); Phase D renumbered to 29–32.
- Main audit table updated with fix scripts (phase2_fix_hub_quality_degree.py, phase2_fix_hub_quality_sim_degrees.py) as new clean entries.

**v8 (2026-04-04):**
- Part 0 restructured: Category A vs Category B dividing line promoted to first subsection (0.1); original content reorganized into 0.2–0.5. Part 0 now leads with the governing rule before the technical rationale.
- Full consistency pass: Substep #25 (cluster tables) now explicitly requires `get_qualifying_clusters()` for all Category B fields and distinguishes "N total from PKL" (reference only) from "N qualifying" (workshop-reported); Substep #27 (connectivity) adds Category B requirement for cluster member sets and gap analysis denominator; Substep #28 (subcluster triggers) now evaluated on valid_pathway_nodes-filtered member counts.
- Gap 4 affected locations table extended with `within_cluster_edge_density.png` (steps 4_cluster_naming.py lines 557–563 and step4b_paths_and_plots.py lines 599–604): edge set conf≥3 is correct, but cluster member denominator uses `get_clusters()` (unfiltered) — Gap 4 violation confirmed by code inspection.
- Summary fix table updated with `within_cluster_edge_density.png` as 🔴 critical.
- Step 2 audit rule 4 added: step2c_plot_updates.py is downstream of step2b CSVs; its Category B plots are superseded by Step 4 outputs; no standalone step2c rerun needed.

**v7 (2026-04-04):**
- Extended audit to full Steps 1–5 chain; introduced explicit Category A (full-graph cartography) vs Category B (workshop paper output) dividing line in Part 0.
- Part 1 Quality Cuts table and "Critical" note corrected to be consistent with Part 0: valid_pathway_nodes is the primary mechanism for cluster member filtering; maturity≥3 filter is belt-and-suspenders only; the ❌ status now correctly refers to Gap 4 (valid_pathway_nodes) not just the maturity filter.
- Part 3 audit table extended to include Steps 1 and 5, with Category column distinguishing cartography vs workshop outputs.
- **New Gap 5**: Discovered Step 5 audit was incorrect — `phase2_step5_naming.py` (`get_cluster_dict()` line 133) and `phase2_step5_triplet_simreach.py` (lines 61–102) both use unfiltered cluster_memberships.pkl. SIM reach counts (r_reach, c_reach, i_reach, union_reach, triplet_core) are inflated; representative nodes for naming may include non-qualifying members. `step5_examples.py` is clean (lookup-only). Gap 5a (naming) and Gap 5b (triplet SIM reach) added to Part 3, summary fix table, and Phase D in Part 8.
- Step 1 `compute_graph_metrics()` confirmed called (line 1226): stores unfiltered betweenness/pagerank/degree/clustering_coefficient in node_attrs.pkl. Category A (intentional full-graph). ⚠️ Warning added: these fields MUST NOT be used in any new Category B analysis.
- Phase D in Part 8 updated with explicit pre-rerun Gap 5 code fixes and Step 1 graph metrics warning.

**v6 (2026-04-04):**
- Added Part 0: Holistic Execution Principle as the topmost document section. States that quality cuts are path-level constraints, not node properties; documents why independent application is incorrect; defines valid_pathway_nodes as the authoritative qualifying set; declares Step 4 as the authoritative rerun point for all prior analyses that violate holistic intent.
- Hub quality upgraded from "acceptable as-is" to required Step 4 rerun: hub rankings must be restricted to valid_pathway_nodes members.
- Step 3 Section C `_analyze_sim_coverage` upgraded from low-priority to required Step 4 rerun: path files are the correct data source (not cluster membership proxy), and this IS workshop dataset characterization.
- Step 3 Section E held-out validation upgraded to required Step 4 rerun.
- Part 8 priority order restructured into Phases A (code fixes), B (core Step 4), C (Step 2/3 reruns in Step 4), D (Step 5).
- Summary fix table updated with hub quality, sim_coverage, held-out validation as 🟡 required for workshop reporting.

**v5 (2026-04-04):**
- Full audit of Steps 2, 3, and 4 for holistic quality cut compliance.
- Added Gap 3b, 3c, Gap 4, Step 2 audit section, Step 3 Sections C/E gaps.

**v4 (2026-03-30):**
- All configs renamed to `consimN_pathbuildX` — `conf_` prefix removed since conf≥3 is a universal quality cut, not a config dimension
- Part 1 (Definitions) rewritten: all terms defined including edge types, node types, cluster types, quality cuts, connectivity, gap analysis, edge-only path fraction, SIM-bridged connections
- Language corrected throughout: "SIM-bridge artifact" → "cross-paper convergent connection" (SIM-bridged connections are legitimate multi-paper findings, not hallucinations)
- Added conf≥3 audit table for Steps 3, 4, 5 (Part 3): Step 3 betweenness and Step 4 connectivity both need fix; Step 5 is clean
- Added Part 4 (Cluster Membership Scope): clarified that cluster_memberships.pkl is NOT pre-filtered to qualifying path nodes (4,889 risk nodes in clusters vs ~3,712 on qualifying paths) — edge-only path fraction metric is genuine, not circular
- All quality cuts listed in one place in Part 1

**v3 (2026-03-30):**
- Rewrote to document 6-config framework, intent, design architecture

**v2 (2026-03-30):**
- Added 6-config framework, edge-only path fraction metric

**v1 (2026-03-29):**
- Initial plan: single config (consim2), Option A+B comparison only
