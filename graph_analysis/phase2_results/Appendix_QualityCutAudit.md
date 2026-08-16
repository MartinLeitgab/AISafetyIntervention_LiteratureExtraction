# Appendix: Pathway-Level Quality Cut Audit

**Audited:** 2026-04-06  
**Purpose:** Documents the definitive audit of Steps 2–5 for holistic simultaneous application of the three pathway-level quality cuts. This content was moved from Step4_Findings_Report.md Part 14 to this appendix to keep the main report focused on findings.

---

## Governing Principle

Every analysis must ONLY use nodes and edges that are part of qualifying paths. A qualifying path must **simultaneously** satisfy:

1. Every SIM edge: cos_sim ≥ 0.9
2. Every EDGE (structural LLM-extracted) edge: confidence ≥ 3
3. Intervention endpoint: maturity ≥ 3

---

## Audit Findings by Step

| Step | Scripts | Category | Path filter required? | Status | Notes |
|------|---------|----------|----------------------|--------|-------|
| Step 1 (load & parse) | `phase2_step1_loadandparse.py` | A | NO | ✅ | Graph loading only; no path-level analysis |
| Step 2 (metrics/stability) | `phase2_step2_metrics_stability.py`, `phase2_step2b_extended_analysis.py` | A | NO | ✅ | Algorithm selection — full graph acceptable per plan. Does not use path files. |
| Step 3 (validation/selection) | `phase2_step3*.py` | B (examples) | YES | ✅ Fixed (2026-04-07) | `_sample_edge_pathways()` now builds VPN and filters; 16/3,473 edge-only paths excluded; `edge_only_test_set.jsonl` regenerated |
| Step 4b (paths, plots, families) | `phase2_step4b_paths_and_plots.py` | B | YES | ✅ Fixed | VPN filtering applied; PKL members intersected with VPN |
| Step 4 cluster naming | `phase2_step4_cluster_naming.py` | B | YES | ✅ Fixed | VPN applied to all cluster member lists |
| Step 4 connectivity | `phase2_step4_connectivity.py` | B | YES | ✅ Fixed | VPN + sim_edge_set VPN restriction |
| Step 4 pathbuildB connectivity | `phase2_step4_pathbuildB_connectivity.py` | B | YES | ✅ Fixed | VPN built per-config from correct path files |
| Step 4 B1 meta-clustering | `phase2_step4_B1_metaclusters.py` | B | YES | ✅ Fixed (2026-04-12) | VPN_consim1 built via 2-pass sim0.9 scan + sim_edge_set; cluster centroids, sim matrices, meta-assignments, all plots regenerated from VPN-filtered members only |
| Step 4 rev3 computations | `phase2_step4_rev3_computations.py` | B | YES | ✅ Fixed (2026-04-12) | VPN construction updated from unconstrained to VPN_consim1; intra-centroid histograms and 2D MDS now use VPN-filtered members |
| Step 4 UMAP plots | `phase2_step4_umap_plots.py` | B | YES | ✅ Fixed | VPN + sim_edge_set restriction; maturity≥3 on all plots |
| Step 4 phase C reruns | `phase2_step4_phase_c_reruns.py` | B | YES | ✅ Fixed | VPN applied |
| Step 5 naming | `phase2_step5_naming.py` | B | YES | ✅ Fixed | VPN applied to all cluster member sampling |
| Step 5 triplet simreach | `phase2_step5_triplet_simreach.py` | B | YES | ✅ Fixed | VPN + sim_edge_set restriction |
| Step 5 examples | `phase2_step5_examples.py` | B | YES | ✅ Fixed | VPN applied; example paths from correct path files |

---

## Path File Correctness (Empirically Verified)

**Result:** Both raw path files are fully correct. No regeneration needed.

| File | Total paths | Maturity<3 endpoints | Verdict |
|------|-------------|----------------------|---------|
| `paths_unconstrained_sim0.9.jsonl` | 1,054,527 | **0** | ✅ Clean |
| `paths_unconstrained_edge_only.jsonl` | 3,473 | **0** | ✅ Clean |

All three quality cuts are simultaneously enforced in path generation:
- `min_conf=3` passed to `load_graph()` — excludes EDGE edges with confidence<3 from BFS graph
- `sim_thresh=0.9` passed to `load_graph()` — excludes SIM edges with cos_sim<0.9
- `cache["interventions"]` pre-filtered to maturity≥3 — BFS terminal set contains only qualifying interventions

The consimN derived files (`representative_pathways_consim1.jsonl`, `representative_pathways_consim2.jsonl`) inherit this correctness.

---

## Root Cause: 2,970→2,815 Intervention Count Discrepancy

| Source | Count | Explanation |
|--------|-------|-------------|
| PKL cluster memberships (raw) | 2,970 intervention nodes | Agglomerative clustering run on ALL node embeddings; 155 maturity<3 nodes fall into intervention clusters by embedding proximity |
| valid_pathway_nodes (from path files) | 2,815 intervention nodes | Path files only contain maturity≥3 endpoints; 155 maturity<3 nodes are in clusters but never on any qualifying path |
| After VPN filter (correct analysis) | **2,815** | ✅ |

The 155 maturity<3 nodes are structurally in the graph and share embeddings with genuine interventions, but no qualifying path (R→I with all three cuts) ever terminates at them. The fix (filter PKL by VPN) is the correct and sufficient solution.

---

## sim_edge_set VPN Restriction — Justification

The `max_consec_sim()` function classifies each hop in a path as SIM or EDGE by looking up `(min(a,b), max(a,b))` in `sim_edge_set`. If `sim_edge_set` includes SIM edges between non-path-participating nodes, it can misclassify hops. Restricting to VPN-pair SIM edges ensures only edges that are part of qualifying paths contribute to hop classification.

---

## Category A vs. Category B Boundary

- **Category A** (algorithm selection — full graph acceptable): Steps 1, 2, 3. These characterize the clustering algorithm performance over the full embedding space, not the workshop-level analysis universe.
- **Category B** (workshop findings — must use path-participating nodes/edges): Steps 4, 5. All analyses here correctly filter to VPN-intersected cluster members with simultaneous three-cut compliance.

---

**No remaining correctness gaps** — all Category B analyses confirmed VPN-filtered and path-file-clean as of 2026-04-12. B1 meta-clustering and rev3 computations updated 2026-04-12 to use VPN_consim1 (19,791 nodes, 75,008 paths) for all centroid computations and intra-cluster distributions.
