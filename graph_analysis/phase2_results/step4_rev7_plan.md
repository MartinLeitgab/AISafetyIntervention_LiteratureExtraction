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
