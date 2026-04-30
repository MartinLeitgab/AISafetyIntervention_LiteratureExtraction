# rev8 Active State — Open Items and Decisions

**Last updated:** 2026-04-30
**Branch:** `martin/main`
**Project root:** `/mnt/c/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/AISafetyIntervention_LiteratureExtraction`
**Container:** FalkorDB `:edge` running in WSL Ubuntu, mounted from `intervention_graph_creation/data/`. The :edge image required `-it` (foreground) — `-d` exits 255 silently mid-index-construction. See issue #129 comment 4354839797.

This file persists rev8 work-in-progress state across context compactions. Update as items resolve.

---

## Critical methodological findings (rev8 — 2026-04-30)

These are the corrections discovered during the rev7→rev8 transition. They shape the entire downstream analysis.

### CF-1: Silent-drop bug in unconstrained-mode path enumeration
**Discovered:** 2026-04-30 audit on `graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl`.
**Quantified:** 99.34% of unconstrained-mode paths have ≥1 risk node in the middle; 13.57% have ≥1 intervention in the middle. Root cause: SIMILARITY edges (cos_sim ≥ 0.9) connect risk-risk and intervention-intervention near-duplicates across papers; the unconstrained BFS walks freely through these.
**Effect:** `phase2_step4_pathbuildB_remaining.py:134` builds frozensets via `frozenset(node_to_stc[n] for n in body if n in node_to_stc)` — non-body middle nodes are silently dropped. Frozensets are partial signatures, multi-path inflation by ~85x at sim=0.9.
**Fix:** Custom mode (added to `graph_analysis/final_pathway_analysis_modes.py`) — single risk at start, single intervention at end, body↔body order unconstrained. BFS-level pre-emption (skip risk-cat neighbors, always emit at first intervention).
**Status:** Custom-mode paths generated for all 5 thresholds (commit 821e985). Audit confirms 0 invariant violations. See `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/custom_mode_audit_report.csv`.

### CF-2: BFS shortest-path is the wrong primitive
**Discovered:** 2026-04-30 (post-custom-mode review).
**Issue:** BFS emits ONE shortest path per (risk_node, intervention_node) pair. Misses:
  - Alternative body chains within a single paper for the same (R, I) pair
  - Multiple cross-paper SIM bridges between papers (only one bridge per (R, I) survives the visited set)
**Quantified:** 1,683 (R, I) pairs in `paths_single_risk_sim0.9.jsonl` are NOT in `paths_custom_sim0.9.jsonl`. These are paths ending `…→I_a→I_b` (multi-intervention) — SR allowed by validation, custom rejected by design. SR ⊄ Custom established as wrong; the relationship is set-difference, not subset, due to BFS exploration variance + intervention-end policy.
**Recommendation:** Switch to paper-pair enumeration: for each paper, enumerate local EDGE-only R→body→I chains; extend with cross-paper SIM bridges (consim1 cap = 1 bridge per path). This mirrors graph construction and counts all valid corpus-level evidence per (R, I) pair, not just one shortest route.
**Status:** Not yet implemented. Open question: implement paper-pair enumeration vs k-shortest-paths fallback.

### CF-3: Body cluster over-fragmentation (separate from CF-2)
**Discovered:** 2026-04-19.
**Issue:** body subtype clustering uses k=40 hardcoded in `graph_analysis/phase2_clustering.py:532-534`, applied to ALL body nodes regardless of qualifying-path membership. Near-duplicate concepts split across cluster IDs (e.g., `pr:7`, `pr:19`, `pr:37` all about "reward mis-specification" but different IDs).
**Effect:** Frozenset signatures fragment along phantom cluster-ID boundaries. Same mechanism appears as multiple frozensets. Path-length distributions show zigzag through near-duplicate body nodes.
**Fix:** Per-subtype k-scan in [10,60], optimize Pareto frontier of intra-cluster cosine sim (high) vs inter-cluster centroid sim (low). Restrict input to qualifying-path body nodes only (VPN-restricted, not full unconstrained).
**Status:** Task #7. BLOCKED by Task #9 (need threshold choice to define VPN).

### CF-4: Six other replicated silent-drop sites + 3 standalone Category-A bugs
**Discovered:** 2026-04-30 silent-drop audit.
**Same `frozenset(... if n in node_to_stc)` pattern in 6 files** (only `phase2_step4_pathbuildB_remaining.py` was originally identified). All become harmless once paths consumed are custom-mode (no non-body middle nodes by design).
**Three standalone fixes already committed (commit 0ab9ccb):**
  1. `final_pathway_analysis_modes.py:184-218` — monotonic mode no longer silently skips unknown-category nodes
  2. `phase2_step4_connectivity.py:471-482` — removed [:50] cap on subcluster split detection
  3. `phase2_step4_B5_novel_pairs.py` — replaced silent "technical" default with "unknown_no_name" + warning + v2 input migration
**Status:** Bug fixes committed. Reruns deferred until rev8 path generation locks in.

---

## Task list (rev8)

| # | Status | Task | Depends on | Reference |
|---|---|---|---|---|
| 1 | ✅ done | Fix CLAUDE.md and README.md docker command (`-it` not `-d`) | — | commit c8ad228, fd77c72 |
| 2 | ✅ done | FalkorDB indices loaded (`:edge` image, container `reverent_faraday`) | — | runtime state |
| 3 | ✅ done | Custom-mode re-BFS for all 5 thresholds | #2 | commit 821e985 |
| 4 | ✅ done | Audit custom-mode path output (zero invariant violations) | #3 | commit 0526385 |
| 5 | ✅ done | Rebuild VPN_consim1 + frozenset families + ri_triplets on custom paths | #4 | commit ([F1 commit hash]) |
| 9 | 🔄 in_progress | Threshold-selection sweep on custom data | — | not yet started |
| 7 | ⏳ pending (blocked by #9) | Body recluster on VPN-only with k-scan | #9 | not yet started |
| 6 | ⏳ pending (blocked by #7) | Re-run E1-E5 chain on cleaned data | #7 | F4/F5 ran as preview only — see commit ([preview hash]) |
| 8 | ⏳ pending (blocked by #6, #7) | Update Step4 + Step5 findings reports for rev8 | #6, #7 | not yet started |
| 10 | ⏳ NEW pending | Switch path generation from BFS-shortest to paper-pair enumeration (or k-shortest fallback) | discussion needed | CF-2 above |

**Order of execution (after Task #10 design decision):**
1. **Task #10** (path enumeration algorithm decision) — paper-pair vs k-shortest
2. **Task #9** (threshold sweep) — on the new path set from #10
3. **Task #7** (body recluster) — on VPN-restricted body nodes from chosen threshold
4. **Task #6** (E1-E5 chain) — on new body clusters + new path set
5. **Task #8** (findings reports update) — final rev8 numbers + methodology documentation

---

## Open Items inherited from `step4_rev7_plan.md` (still applicable)

1. **Document n≥5 frozenset cutoff** in all reports (`phase2_step4_pathbuildB_remaining.py:141`)
2. **Body cluster k-scan** (Open Item 2; this is Task #7)
3. **Body clustering on VPN nodes only** (Open Item 3; subsumed by Task #7)
4. **Recursive frozenset group reclustering** for homogeneity floor (Open Item 4; deferred until #6 done)
5. **Triplet-level grouping** options (Open Item 5; deferred until #6 done)
6. **Hierarchical interactive visualization** for public-facing review (Open Item 6; deferred to post-rev8)
7. **Step4/Step5 report consolidation** (Open Item 7; deferred to #8)
8. **Reject paths with non-body middle nodes** (Open Item 8 — RESOLVED by custom mode + CF-1)

---

## Key file paths (rev8 outputs)

**Custom-mode path files (5 thresholds):**
- `graph_analysis/phase1_rawpathsfiles/paths_custom_edge_only.jsonl` (3,283 paths)
- `graph_analysis/phase1_rawpathsfiles/paths_custom_sim0.8.jsonl` (3,393,673 paths)
- `graph_analysis/phase1_rawpathsfiles/paths_custom_sim0.85.jsonl` (1,125,319 paths)
- `graph_analysis/phase1_rawpathsfiles/paths_custom_sim0.9.jsonl` (21,521 paths)
- `graph_analysis/phase1_rawpathsfiles/paths_custom_sim0.95.jsonl` (3,293 paths)

**Custom-mode F1 (consim1 rebuild) outputs:**
- `graph_analysis/phase2_results/step4_finalanalysis/step4_paths/representative_pathways_custom_consim1.jsonl` (4,584 qualifying paths)
- `graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/optionB_cooccurrence_families_custom_consim1.csv` (104 frozensets at n>=5; 2,581 unique frozensets total)
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/ri_triplets_custom_consim1.csv` (450 ri_triplet rows)

**Audit:**
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/custom_mode_audit_report.csv`
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/custom_consim1_summary.txt`

**Preview (NOT FOR PAPER — uses rev7 over-fragmented body clusters):**
- `graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/frozenset_groups_custom_consim1.csv` (10 groups, k=10)
- `graph_analysis/phase2_results/step5_naming/frozenset_group_names_custom_llm.csv` (gpt-5.5 names, 3/10 flagged)

**Plots regenerated 2026-04-30 with custom mode added:**
- `graph_analysis/plots/constrained_modes_path_lengths_all_with_edge.png`
- `graph_analysis/plots/constrained_modes_heatmaps_edge_only.png`
- `graph_analysis/plots/constrained_modes_heatmaps_sim0.{8,85,9,95}.png`
- `graph_analysis/plots/constrained_modes_degrees_all_with_edge.png`

---

## Key scripts (rev8)

- `graph_analysis/final_pathway_analysis_modes.py` — adds 'custom' mode (commit 821e985)
- `graph_analysis/phase2_step4_F0_custom_path_audit.py` — invariant + retention audit (commit 0526385)
- `graph_analysis/phase2_step4_F1_consim1_custom_rebuild.py` — single-pass consim1 + frozensets + ri_triplets on custom paths
- `graph_analysis/phase2_step4_F4_frozenset_groups_custom.py` — Jaccard grouping on 104 frozensets, k=10 (preview)
- `graph_analysis/phase2_step4_F5_group_naming_custom.py` — gpt-5.5 LLM naming with `max_completion_tokens=1500` (preview)
- `graph_analysis/phase2_step4_F6_triplets_custom.py` — R→Group→I triplets (preview, not run)

---

## Decision points awaiting user input

1. **Task #10 — path enumeration algorithm:**
   - (a) Paper-pair enumeration (recommended; mirrors graph construction; needs new code ~200 lines)
   - (b) k-shortest paths fallback (e.g. k=20, networkx Yen's; less faithful but simpler)
   - (c) Bounded all-simple-paths up to length L (intractable for L>~10)

2. **Task #9 — threshold-sweep metrics design:**
   - Body-cluster-independent metrics: N qualifying paths, N (R, I) pairs, % SIM-only pairs, path length distribution, mean N body subtypes per path
   - Body-cluster-dependent metric (silhouette) is provisional pre-Task #7
   - Decision criterion for threshold choice: balance R→I coverage + cross-paper bridging volume vs over-fragmentation signals

---

## Container / runtime notes

- FalkorDB started: `wsl -d Ubuntu -- docker run -p 6379:6379 -p 3000:3000 -it --rm --volume <data_path>:/var/lib/falkordb/data falkordb/falkordb:edge`
- Use `-it`, NOT `-d` (silent exit 255 with no logs).
- Container name auto-generated (e.g., `reverent_faraday`); use `docker ps` to find ID.
- Indices take ~15 min to build on first start.
- Python redis package installed in conda base env: `redis-7.4.0`.
- All E5/F1+ scripts execute via `conda run --no-capture-output -n base python -u ...` for live output.
