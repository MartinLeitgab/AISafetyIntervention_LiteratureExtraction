# Step 4 Revision 7 — Plan

**Created:** 2026-04-19  
**Goal:** Replace ad-hoc meta-family naming with a fully data-driven L2 taxonomy.  
Add centroid spread (closest-3, farthest-3) at every level for reviewer verifiability.  
**Principle:** Never overwrite old results. All new outputs use `_v2` suffix or `_rev7` suffix.

---

## Motivation

Reviewer concern: LLM-named clusters cannot be independently verified.  
Fix: Show closest-3 (most representative) and farthest-3 (borderline) nodes/families at every level.  
If farthest-3 remain thematically consistent with closest-3, the cluster is tight. If not, the LLM name is flagged as partial.

Additional fix: current L2 chain grouping (PathbuildB "meta-families") was built by ad-hoc dominant-family selection.  
Replace with principled binary-vector Jaccard clustering of frozensets.

---

## What Changes vs What Is Preserved

| Layer | Old | New (rev7) |
|-------|-----|-----------|
| Body concept clusters | `rep_name`, `top3_names` | + `closest3_names`, `farthest3_names` |
| L1 Risk clusters | `top_node_name`, `top5_names` | + `closest3_names`, `farthest3_names` |
| L3 Intervention clusters | same as risk | same |
| L2 grouping | Jaccard meta-families via dominant family heuristic | Binary vector agglomerative clustering of frozensets |
| L2 names | LLM from body signatures only | LLM from centroid components + closest/farthest frozensets |
| Triplets | `ri_meta_triplets_consim1.csv` | `ri_meta_triplets_consim1_rev7.csv` (old file untouched) |

Old files are NEVER deleted or overwritten.

---

## Data Sources (confirmed available)

| Data needed | Source file |
|-------------|-------------|
| Node embeddings | `step1_load.../graph_node_attributes.pkl` — key `embedding` (np.float32 1024-dim) |
| Node names | `graph_node_attributes.pkl` — key `name` |
| Body concept cluster memberships | `cluster_memberships.pkl` — config "0.9", mode "unconstrained", subtypes below |
| L1/L3 cluster memberships | `cluster_memberships.pkl` — config "0.9", mode chosen per subtype |
| Frozenset families | `step4_cluster_tables/optionB_cooccurrence_families_consim1.csv` — cols: `family_id, n_paths, signature_str` |
| Body cluster reps (existing) | `step4_cluster_tables/bodysubtype_cluster_representatives.csv` — cols: `subtype, cluster_id, prefix_key, rep_name, top3_names, n_members, n_vpn_members` |
| Risk/Interv clusters (existing) | `step4_cluster_tables/risk_clusters_consim1.csv`, `intervention_clusters_consim1.csv` |

**Body subtypes and their prefix codes** (from `phase2_step4_pathbuildB_remaining.py`):
- `problem_analysis` → pr
- `theoretical_insight` → th
- `design_rationale` → de
- `implementation_mechanism` → im
- `validation_evidence` → va

Body concept cluster memberships loaded via: `get_clusters("0.9", "unconstrained", subtype)` from `cluster_memberships.pkl`.

---

## Scripts (all new, prefix `phase2_step4_E`)

### E1 — Body concept cluster spread
**Script:** `graph_analysis/phase2_step4_E1_body_cluster_spread.py`  
**Input:** `cluster_memberships.pkl`, `graph_node_attributes.pkl`, `bodysubtype_cluster_representatives.csv`  
**Method:**
1. Load body cluster members per subtype via `get_clusters("0.9", "unconstrained", subtype)`
2. Filter to VPN nodes (use valid_pathway_nodes from `all_paths_consim1.pkl` or recompute)
3. For each body cluster: compute centroid embedding (mean of member embeddings)
4. Find closest-3 members (highest cosine sim to centroid) and farthest-3 (lowest)
5. Get their `name` field from node_attrs

**Output:** `step4_cluster_tables/bodysubtype_cluster_representatives_v2.csv`  
New cols: `closest3_names`, `farthest3_names`, `centroid_sim_min`, `centroid_sim_max`

---

### E2 — L1/L3 base cluster spread
**Script:** `graph_analysis/phase2_step4_E2_base_cluster_spread.py`  
**Input:** `cluster_memberships.pkl`, `graph_node_attributes.pkl`, existing cluster tables  
**Method:**
1. Load risk cluster members: `get_clusters("0.9", "unconstrained", "risk")` (or equivalent consim1 config)
2. Load intervention cluster members similarly
3. For each cluster: centroid, closest-3, farthest-3 by cosine similarity
4. Get node names

**Note:** The existing `risk_clusters_consim1.csv` was built from VPN_consim1-filtered members.  
Use same VPN filter here (nodes on any qualifying consim1 path).

**Output:**
- `step4_cluster_tables/risk_clusters_consim1_v2.csv` — adds `closest3_names`, `farthest3_names`
- `step4_cluster_tables/intervention_clusters_consim1_v2.csv` — same

---

### E3 — Frozenset binary-vector clustering (new L2)
**Script:** `graph_analysis/phase2_step4_E3_frozenset_groups.py`  
**Input:** `step4_cluster_tables/optionB_cooccurrence_families_consim1.csv`, `bodysubtype_cluster_representatives_v2.csv`

**Method:**
1. Build vocabulary: all distinct `prefix_key` values across all frozensets (e.g. "de:15", "im:4") — expected ~100-200 unique IDs
2. For each frozenset: build binary vector of length |vocab|, entry=1 if component present, 0 otherwise
3. Weight each frozenset row by `sqrt(n_paths)` (square root dampens outliers)
4. Compute pairwise Jaccard distance matrix: `d(A,B) = 1 - |A∩B| / |A∪B|` on binary vectors (unweighted Jaccard; weight used only for clustering objective)
5. Agglomerative clustering (Ward linkage on binary matrix, or average linkage on Jaccard distances) — target k=20 to 30 groups
6. Choose k by dendrogram inspection: cut where merge distance increases sharply
7. For each group:
   - Centroid = mean binary vector → top component IDs by mean presence score
   - Decode centroid top-5 components to readable names via `bodysubtype_cluster_representatives_v2.csv`
   - Find closest-3 frozensets (highest Jaccard similarity to group centroid) and farthest-3 (lowest)
   - Decode their signatures to readable component names

**Output:** `step4_cluster_tables/frozenset_groups_consim1.csv`  
Cols: `group_id, n_frozensets, n_paths_total, centroid_components, centroid_decoded, closest3_signatures, closest3_decoded, farthest3_signatures, farthest3_decoded, intra_jaccard_mean`

Also: `step4_cluster_tables/frozenset_group_memberships_consim1.csv`  
Cols: `family_id, group_id, jaccard_sim_to_centroid` (one row per frozenset)

Plots:
- Dendrogram of frozenset groups
- 2D MDS scatter of frozensets colored by group

---

### E4 — Frozenset group LLM naming
**Script:** `graph_analysis/phase2_step4_E4_frozenset_group_naming.py`  
**Input:** `frozenset_groups_consim1.csv`, `bodysubtype_cluster_representatives_v2.csv`, `ri_triplets_consim1.csv` (for R→I context), risk/intervention v2 names

**Method:** 2-pass gpt-4.1-mini naming (same causal framing as v3):
- Pass 1 (naming): provide centroid components (decoded), closest-3 frozensets (decoded), farthest-3 frozensets (decoded), top R→I pairs this group connects
- Requirement: name must complete "The reason why [intervention] mitigates [risk] is [NAME]"
- Pass 2 (judge): verify no via/through prefix, consistency with closest/farthest examples

**Output:** `step5_naming/frozenset_group_names_llm.csv`  
Cols: `group_id, n_frozensets, n_paths_total, centroid_decoded, llm_name, description, test_sentence, test_sentence_ok, judge_accurate, judge_starts_via, suggested_revision, final_name`

---

### E5 — Rebuild triplets with new L2 groups
**Script:** `graph_analysis/phase2_step4_E5_triplets_rev7.py`  
**Input:** `frozenset_group_memberships_consim1.csv`, `ri_triplets_consim1.csv` (base triplets), `frozenset_group_names_llm.csv`, risk/interv v2 names

**Method:**
1. For each row in `ri_triplets_consim1.csv`, look up which group the `bfamily_id` belongs to
2. Aggregate: sum `n_triplet_paths` across all frozensets in each group for each (risk_cid, interv_cid) pair
3. Join group names, risk names, intervention names
4. Sort by total paths descending

**Output:**
- `step4_connectivity/ri_triplets_consim1_rev7.csv` — base triplets with group_id column added
- `step4_connectivity/ri_group_triplets_consim1.csv` — aggregated: `risk_cid, group_id, interv_cid, n_paths, risk_name, group_name, interv_name`
- `step4_connectivity/ri_group_triplets_top20_consim1.csv` — top 20 rows

---

## Execution Order

```
E1 → E2 → E3 → E4 → E5 → report updates
```

E1 and E2 can run in parallel (independent inputs).  
E3 depends on E1 (for decoded component names in output).  
E4 depends on E3.  
E5 depends on E3 and E4.

---

## Report Updates

After E5 completes, append to both findings reports:

**Step4_Findings_Report.md — new Part 18:**
- Methodology: spread display rationale
- Body cluster spread results (any heterogeneous clusters flagged)
- Frozenset group clustering: k chosen, dendrogram, group summary table
- Comparison: old meta-family approach vs new group approach

**Step5_Findings_Report.md — new section:**
- Updated L2 naming: group names with quality metrics
- Closest/farthest verification table
- Flagged groups (farthest-3 inconsistent with closest-3)

---

## Quality Checks per Script

| Script | Check |
|--------|-------|
| E1 | All 5 subtypes produce output; no group has 0 VPN members |
| E2 | N clusters = 40 risk, 40 intervention; centroid_sim_min > 0 for all |
| E3 | No frozenset assigned to >1 group; k chosen ≥ 15 and ≤ 35; all groups n≥3 frozensets |
| E4 | 0 names starting with "via"/"through"; all test_sentence_ok=True |
| E5 | Total n_paths in rev7 triplets = total in original (no paths lost); 0 missing group_ids |

---

## Files NOT Changed

- `optionB_cooccurrence_families_consim1.csv` — frozenset statistics unchanged
- `ri_meta_triplets_consim1.csv` — old triplets preserved (rev7 goes to new file)
- `pathbuildB_chain_names_llm_v3.csv` — old individual frozenset names preserved
- `pathbuildB_metafamily_names_llm.csv` — old meta-family names preserved
- All PKL checkpoints

---

## Git Commit Plan

1. After E1+E2: commit `feat: rev7 — add centroid spread (closest/farthest-3) to body and base clusters`
2. After E3: commit `feat: rev7 — frozenset binary-vector Jaccard grouping, k=N groups, dendrogram`
3. After E4: commit `feat: rev7 — LLM naming of frozenset groups (causal framing)`
4. After E5 + report updates: commit `feat: rev7 — rebuilt triplets and findings report update`

---

## Open Items / Future Work (added 2026-04-20)

These items were discussed in the rev7 review but **not yet implemented**. They depend on prior items being resolved first; all are noted here so they can be resumed later.

### Open Item 1 — Document n>=5 frozenset cutoff in findings reports
**Source:** `phase2_step4_pathbuildB_remaining.py` line 141 (`large_sigs_set = {s for s, c in sig_counts.items() if c >= 5}`).
**Action:** Whenever the figure "1,603 frozensets" or "62,357 paths" or any n_paths-aggregated number is reported, accompany with explicit cutoff disclosure: "n_paths >= 5 frozensets retained for downstream analysis (1,603 of N_total unique frozensets, capturing M of M_total qualifying paths). Singleton/rare frozensets dropped to focus analysis on representative co-occurrence patterns; raw counts available in optionB_cooccurrence_families_consim1.csv."
**Where to apply:** Step4 Part 1, Part 18.4, Step5 Part 11.1, Appendix_QualityCutAudit.md.

### Open Item 2 — Body cluster homogeneity audit and k-scan
**Problem:** k=40 hardcoded in `phase2_clustering.py:532-534` for all node types (risk, intervention, all 5 body subtypes). No homogeneity-vs-separation optimization. This is the upstream root cause of frozenset over-fragmentation symptoms (e.g., G13/G15 split, pr:7/pr:19/pr:37 near-duplicates).
**Action:** Per-subtype k-scan in [10, 60] optimizing the (max_intra_homogeneity, min_inter_homogeneity) Pareto frontier. Metrics:
  - Intra: mean cosine sim of each cluster's members to its centroid (per cluster, then aggregate)
  - Inter: max cosine sim between any two cluster centroids (lower = more separated)
  - Silhouette as a baseline reference
**Output:** `step4_cluster_tables/body_kscan_metrics.csv` with per-(subtype, k) intra/inter/silhouette. Choose k per subtype where intra is high and inter is low.
**Apply same scan to:** L1 risk, L3 intervention. Also to E3 frozenset grouping (currently k=20 hardcoded).
**Verification:** Re-run E3 on rederived body clusters; confirm G13/G15-style splits collapse where appropriate.

### Open Item 3 — Body clustering must be done on qualifying-path nodes only
**Problem:** `phase2_step4_pathbuildB_connectivity.py:174-188` and rev7 E1/E2 both load body clusters from `cm[(0.9, "unconstrained", subtype, "agglomerative", cid)]` — these clusters were built on ALL body nodes in the graph, not only qualifying-path body nodes. VPN filtering is applied AFTER cluster definition, leading to the "9 body cluster IDs that never appear on a qualifying path" anomaly.
**Action:** Restrict body clustering input to body nodes that appear on at least one qualifying path (consim1 + maturity≥3 + EDGE conf≥3 + SIM≥0.9). Re-cluster per-subtype on this VPN-restricted population.
**Order of operations:** must be done BEFORE Open Item 2 k-scan (the k-scan should be over the VPN-restricted body node population).
**Verification:** every body cluster ID appearing in cluster_memberships output must have ≥1 member on a qualifying path.

### Open Item 4 — Recursive frozenset group reclustering for homogeneity floor
**Problem:** k=20 frozenset groups have intra_jaccard_mean ranging 0.111 to 0.833. Many groups (G12, G14) are heterogeneous (intra<0.3) and would benefit from being split.
**Action:** After Open Items 2 and 3 are resolved (so body cluster IDs are no longer over-fragmented), run E3 with recursive splitting: for each group with intra_jaccard_mean < threshold (proposed: 0.5), re-run agglomerative clustering on its members; repeat until all leaf groups meet threshold or hit min-size floor (proposed: 3 frozensets).
**Output:** Hierarchical taxonomy. Each leaf group gets its own E4 LLM name; intermediate nodes get summary names from path-weighted leaf merging.
**Why deferred:** Recursive splitting on the current over-fragmented body cluster IDs would produce phantom splits driven by ID-level differences, not real mechanism diversity.

### Open Item 5 — Triplet-level grouping
**Problem:** Top-20 R→Group→I triplets sorted purely by path count = corpus volume, not mechanism diversity. The top-16 are dominated by G12+G14 routes; mechanistically distinct groups (G2, G10, G8) appear only at rank 16+.
**Three options noted, not yet chosen:**
  - **Option A (simplest):** Group the 764 triplets by `group_id` (already a partitioning). Within each of the 19 mechanism groups, list top-3 risk-intervention pairs. Output: 57 triplets covering all mechanism groups, ranked within each by paths.
  - **Option B:** Compute combined embedding per triplet (concatenate risk centroid + group binary vector + intervention centroid), cluster, take top-N per cluster.
  - **Option C:** Run Open Item 4 first (recursive group homogeneity), then aggregate triplets at finer-grained mechanism level. Diversity emerges naturally.
**Recommended:** Option C if Open Items 2-4 are done; otherwise Option A.

### Open Item 6 — Interactive hierarchical visualization (public-facing)
**Goal:** A clickable web UI for the full 3-layer taxonomy + triplets, hosted publicly (GitHub Pages on a personal/org domain like `martin.github.io` — confirm with user which domain).
**Specification:**
  - **Top level:** all triplets visualization (ranked by paths or by mechanism group cluster). Each triplet is clickable.
  - **Click-through to entity:** clicking a Risk cluster, Frozenset Group, or Intervention cluster opens its detail page.
  - **Detail pages show:** centroid name, n_members, n_paths, closest-3 / farthest-3 members with metadata.
  - **Click-through to nodes:** clicking a member node shows: name, description, aliases, intervention_lifecycle/maturity (if applicable), source URL (hyperlinked to original paper), local extraction audit record (the original LLM extraction JSON for that paper).
  - **Hierarchy breadcrumb:** persistent top bar like `Triplets > Group G14 > Intervention I35 > Node 12345` so user can navigate up/down.
  - **Frozenset group page:** shows constituent frozensets as nodes; click a frozenset to see its constituent paths and their counts.
**Hosting candidates:**
  - GitHub Pages on `MartinLeitgab.github.io/<repo>` (free, static-only)
  - Fly.io / Render free tier if dynamic backend is needed
  - Static HTML + JSON data files is preferred (no backend needed)
**Tech stack candidate:** static HTML + Plotly/D3 for the triplet network plot + JSON-driven detail panes. All data already in CSVs/PKLs.
**Apply public-deployment-readiness checks:** any private-only audit records (raw LLM extraction with internal notes) must be filtered out before public deploy.
**Why deferred:** depends on stable taxonomy (Open Items 2-4) so the visualization doesn't need to be rebuilt after every regrouping.

### Open Item 7 — Step 4 / Step 5 report consolidation
Currently Step 4 (~830 lines) and Step 5 (~430 lines) overlap at the boundary "structure vs naming." After Open Items 2-4 resolve and rev7 LLM naming becomes the only LLM-driven step, most of Step 5 (Parts 1-7 covering v1/v2 cluster naming, examples, subclusters) may be archivable. Decision deferred until taxonomy stabilizes.

### Open Item 8 — Reject paths with non-body middle nodes (CRITICAL — correctness issue)
**Discovered:** 2026-04-20 path enumeration audit (first 100k of `paths_unconstrained_sim0.9.jsonl`):
- 99.34% of paths have at least one risk node in the middle
- 13.57% of paths have at least one intervention node in the middle

**Cause:** `mode='unconstrained'` BFS uses SIMILARITY edges (cos_sim ≥ 0.9) which include risk-risk and intervention-intervention links. BFS walks freely through these. `final_pathway_analysis_modes.py:152-217` only emits a path when an intervention has a non-intervention unvisited neighbor — when it doesn't, BFS continues THROUGH the intervention. Frozenset construction (`phase2_step4_pathbuildB_remaining.py:134`) silently drops these middle non-body nodes via the `if n in node_to_stc` filter.

**Effect:** frozenset signatures are partial. Two paths with very different mechanism routings collapse to the same frozenset. The L2 taxonomy reflects body-cluster co-occurrence with non-body waypoints invisibly stripped.

**Fix options:**
- (preferred) Switch path-source from `paths_unconstrained_sim0.9.jsonl` to `paths_both_sim0.9.jsonl` (single_risk + monotonic constraints). Single_risk eliminates the 99% risk-in-middle case; monotonic eliminates the intervention-in-middle case (interventions can only appear as the terminal node, since reversing from intervention back to body is forbidden).
- (alternative) Post-filter `paths_unconstrained_sim0.9.jsonl` to reject any path whose `categories[1:-1]` contains a non-body category. Apply BEFORE consim1 filtering.

**Verification step:**
1. Re-run path audit on filtered set → 0 paths with non-body middle
2. Recompute n_paths totals — expect substantial drop from 75,008 consim1 paths to fewer cleaner paths
3. Re-run E3 frozenset grouping → expect different group structure since underlying data changes
4. Document % of corpus retained vs dropped

**Order in pipeline:** must be done BEFORE Open Items 2-4 (body re-clustering should happen on body nodes that appear on TRULY body-only middle paths, not the polluted current set).
