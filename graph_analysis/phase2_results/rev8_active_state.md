# rev8 Active State — Comprehensive Todo + Critical Findings

> 🔴 **2026-05-17 PIPELINE RESTART IN PROGRESS** — see `restart_plan_2026_05_17.md`.
> First-run doublet artifacts (200-seed → 42 RG + 138 MG → 3,050 paths) being
> archived to `archive/2026_05_15_first_run/`. New seed + Sonnet Pass B + REVIEW
> cycles will run on the deduped 2,949-path corpus under refined prompts
> (human-risk framing, continuum guidance, anti-singleton, organic multi-axial).
> Tasks #34 (prep, Class B) + #35 (execute, Class A) are the active work items.
> All other open task work paused pending restart completion.

> 🔴 **2026-05-15 STRATEGIC PIVOT — PUBLICATION-PATH PLAN LOCKED.**
> The §19.10/§19.12 full-VPN LLM clustering plan below (locked 2026-05-09) is **SUPERSEDED** for the immediate publication path by the doublet pipeline in `Step4_Findings_Report.md` §19.13 and the forward plan in `forward_plan_2026_05_15_publication_path.md`.
>
> **Active strategy:** Sonnet-only full-VPN Pass B over all 8,954 paths using the 200-path Opus seed catalog as fixed read-only input (no Opus mid-run reviews). One Opus meta-grouping pass at end. Targets ~5-7 working day timeline. Read `forward_plan_2026_05_15_publication_path.md` first; this state doc preserves history below for context.
>
> **D1 (faithfulness) is now LLM-as-judge** (validation lead owns), not hand-validation (replaces earlier plan; saves time + reuses the existing judge framework built on Mike's judge artifacts).
>
> **Why full VPN over truncated sample:** the 2026-05-15 review of the frozen Overleaf draft raised quality concerns about its algorithmic node-clustering approach (no quality cuts, hub poisoning risk, similarity-edge traps at τ≥0.80). The LLM-central path-level analysis must stand alone with full corpus coverage to justify the methodology contribution distinct from those navigational k=40 clusters.

**Last updated:** 2026-05-15 (publication-path plan locked; supersedes 2026-05-09 §19.12 plan for the immediate ship)
**Earlier last-update:** 2026-05-09 (Pass-2 complete; granularity-gap reconciliation plan locked — see Step4_Findings_Report.md §19.10) — SUPERSEDED but preserved below

**Latest Phase 2 Task A status (see Step4 §19.10 + §19.12):**
- Pass-2 complete: 26/27 batches OK; batch 24 (80 nodes) deferred — Claude AUP block, not retried.
- 2,015 of 2,095 NR residuals assigned: 1,149 seed-group (33 v3 mechanism classes) / 587 HDBSCAN rescue / 279 residual.
- **2026-05-09 strategic pivot — LLM-central, full-VPN clustering (§19.12 LOCKED):** drop the "Track 1 / Track 2" dual-resolution framing. Run a single full-VPN LLM clustering pass on all 19,073 nodes (subtype as metadata, NR + risk pools, v3 + v2 seed taxonomy as starting catalog). HDBSCAN's role downgrades from substrate to validation check on whether LLM mechanism boundaries agree with embedding-density structure. Mirrors the single-extraction LLM-as-judge pattern.
- Estimated cost: ~11.8M Max-plan tokens, ~10–15h wall. Smoke-test (5 batches, ~250k tokens, 30min) runs first; user reviews granularity + subtype-as-metadata utility before approving full chain.
- Per-batch atomic save mandatory (`phase2_full_vpn_batches/batch_NNN.json` written before next API call) so a Max-plan session-limit hit mid-run does not lose data.
- AUP-resilience: per Pass-2 batch-24 lesson, AUP-blocked batches auto-split into 8 sub-batches of 10 nodes; failed sub-batches logged for manual review, script continues.

**Earlier last-update (preserved):** 2026-04-30 23:24 CDT
**Branch:** `martin/main`
**Project root:** `/mnt/c/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/AISafetyIntervention_LiteratureExtraction`
**Container:** FalkorDB `:edge` running in WSL Ubuntu (auto-named, e.g. `reverent_faraday`), mounted from `intervention_graph_creation/data/`. Uses `-it` (foreground) — `-d` exits 255 silently. See issue #129 comment 4354839797. **CRITICAL: container has `RESULTSET_SIZE=10000000` set live; this is not persistent across container restart. Every FalkorDB-query script must SET this at startup or batch queries (see Bug Audit below).**

**Completed runs (2026-05-01 07:46 CDT):**
- F2v4 sim=0.85 DFS: COMPLETED (20,439,041 paths, 481,414 unique R-I pairs, 19,178/19,178 risks). 78 min runtime, hit_global_cap=False. 7.3 GB jsonl. Path-length histogram top-heavy at L=12 (14M of 20M).
- Custom-BFS rerun for sim=0.8 + sim=0.85 (CF-5-clean): COMPLETED 07:46 CDT (~8.5h total). Defense-in-depth `RESULTSET_SIZE=10M` bump applied to `final_pathway_analysis_modes.py`. EDGE/sim=0.9/sim=0.95 skipped via existing checkpoints. New custom-mode counts: sim=0.8=4,900,771 paths, sim=0.85=1,309,004 paths, sim=0.9=21,521 paths (from prior run, identical), sim=0.95=3,293 paths, EDGE=3,283 paths. CF-1 silent-drop fix verified: all custom-mode paths have max 1 risk/path (vs 94-99% multi-risk in unconstrained). New jsonl files moved from `phase1_otherrawdata/` to `phase1_rawpathsfiles/`; CF-5-contaminated old files (Apr 30 13:32) archived to `phase1_rawpathsfiles/archive_pre_cf5fix/`.

This file is the canonical work-in-progress state. Update as items resolve. **Persists across context compactions** by living in the project repo.

---

## Critical Methodological Findings (CF-1 through CF-5)

### CF-1: Unconstrained-mode silent-drop bug
**File:** `graph_analysis/phase2_step4_pathbuildB_remaining.py:134`
**Quantified 2026-04-30:** 99.34% of unconstrained-mode paths at sim=0.9 had ≥1 risk node in middle, 13.57% had ≥1 intervention in middle. SIM-bridged risk-cluster duplicates and intervention-intervention SIM bridges traversed by BFS. Frozenset construction silently drops middle non-body nodes (`if n in node_to_stc`). Result: ~85x path-count inflation at sim=0.9.
**Fix:** custom-mode BFS in `graph_analysis/final_pathway_analysis_modes.py` (commit 821e985) — pre-empts at first intervention, skips risk neighbors during expansion. Path file: `graph_analysis/phase1_rawpathsfiles/paths_custom_sim0.9.jsonl` (21,521 paths).
**Status:** RESOLVED at the path-source level. The original `paths_unconstrained_sim0.9.jsonl` is still there but should NOT be used for any new analysis.

### CF-2: BFS shortest-path is wrong primitive — use hop-wise DFS
**Issue:** BFS emits ONE shortest path per (R, I) pair. Misses (a) alternative body chains within a single paper, (b) multiple cross-paper SIM bridges between papers (only one survives the visited set).
**Fix:** hop-wise DFS enumeration. Each step extends the current path stub by one edge (EDGE or SIM with consim1 alternation). Each SIM hop crosses into another paper. Path traverses arbitrarily many papers.
**Implementation versions:**
- `graph_analysis/phase2_step4_F2v3_hopwise_paths.py` — uses PKL graph data (DEPRECATED if PKL turns out to be incomplete/different from FalkorDB)
- `graph_analysis/phase2_step4_F2v4_hopwise_falkordb.py` — **CANONICAL**, queries FalkorDB live, max_length=50, max_total=10M
**Constraint set (canonical):**
  - Edge filters: EDGE conf>=3, SIM cos_sim>=0.9 (= euclidean score < 0.4472)
  - Endpoint filter: intervention_maturity>=3
  - Path: simple paths, consim1 alternation, single-risk, single-intervention, first-hop EDGE-or-SIM to body subtype, min length 3, max length 50
  - Last-resort safeguards: max 10M paths total (tracked via `hit_global_cap` in summary)
**Status:** F2v4 in progress as of last session. Output: `graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_sim0.9.jsonl`.

### CF-3: Body cluster over-fragmentation
**File:** `graph_analysis/phase2_clustering.py:532-534`
**Issue:** k=40 hardcoded for ALL node types (5 body subtypes + risk + intervention + all_concepts) at config 0.9/unconstrained. Near-duplicate concepts split across cluster IDs (e.g., `pr:7`, `pr:19`, `pr:37` all about reward mis-spec; 6 different "Opaque internal representations" clusters seen in one path).
**Effect:** Frozenset signatures fragment along phantom cluster-ID boundaries. Same mechanism appears as multiple frozensets. Path-length distributions show monotonic 1.4-1.5× growth past L=12 (combinatorial zigzag through near-duplicate clusters). At L=20, 80% of paths have 3+ clusters per subtype (likely phantom splits not real diversity).
**Fix (Task #7):** per-subtype k-scan in [10,60]. Optimize Pareto frontier of intra-cluster cosine sim (high) vs inter-cluster centroid sim (low). Restrict input to qualifying-path body nodes only. Apply same scan to risk and intervention clusters.

### CF-4: Replicated silent-drop sites + 3 standalone Cat-A bugs
**Same `frozenset(... if n in node_to_stc)` pattern in 6 files** beyond the original `pathbuildB_remaining.py:134`: `phase2_step4b_paths_and_plots.py:451,471`; `phase2_step4_trackA.py:330,360`; `phase2_step4_pathbuildB_connectivity.py:355-358`; `phase2_step4_cluster_naming.py:378`; `phase2_step5_examples_edgeonly_fix.py:373`. **All become harmless once paths consumed are custom-mode (no non-body middle nodes).**
**Three standalone fixes already committed (commit 0ab9ccb):**
1. `graph_analysis/final_pathway_analysis_modes.py:184-218` — monotonic mode no longer silently skips unknown-category nodes
2. `graph_analysis/phase2_step4_connectivity.py:471-482` — removed [:50] cap on subcluster split detection
3. `graph_analysis/phase2_step4_B5_novel_pairs.py` — replaced silent "technical" default with "unknown_no_name" + warning
**Status:** committed.

### CF-5: FalkorDB silent 10k RESULTSET_SIZE truncation (DISCOVERED 2026-04-30)
**Issue:** FalkorDB default `RESULTSET_SIZE=10000`. Every Cypher query result silently capped to 10,000 rows. No error, no warning. Single-shot queries returning >10k rows are silently truncated.
**Discovered:** F2v4 v1 returned 10,000 risk nodes but custom-BFS reported 19,178. Audit of all FalkorDB query call sites identified multiple affected scripts (see Bug Audit section).
**Mitigation applied to live container:** `GRAPH.CONFIG SET RESULTSET_SIZE 10000000` (10M). NOT persistent across container restart.
**Permanent fixes needed:** see "Bug Audit (CF-5)" section below.

---

## Bug Audit (CF-5) — Silent-Truncation Fix List

Audit done 2026-04-30 across all .py files in `graph_analysis/`. Three HIGH-severity sites need batched-query fixes; two MEDIUM-severity sites need batch-size reduction.

### HIGH severity (corrupted analysis output)

**B-1: `graph_analysis/cluster_utils.py:107-113` (`_build_mapping`)**
- Query: `MATCH (n) WHERE 'Concept' IN labels(n) OR 'Intervention' IN labels(n) RETURN id(n), n.name`
- Population: ~200k nodes in Concept+Intervention. Default cap = 10k → only ~5% of nodes get a name→cluster mapping.
- Downstream impact: `ClusterMapper.get_cluster()` returns "unmapped" for ~95% of name lookups. Used in stratified sampling, cluster naming context. Hard to detect because no error — silently returns None.
- **FIX:** convert to batched id-range query (existing pattern in `phase2_step1_loadandparse.py`):
  ```python
  cur, batch = min_id, 5000
  while cur <= max_id:
      cy = f"MATCH (n) WHERE id(n) >= {cur} AND id(n) < {cur+batch} AND ('Concept' IN labels(n) OR 'Intervention' IN labels(n)) RETURN id(n), n.name"
      ...
      cur += batch
  ```
- After fix: re-build any cluster-mapping artifact that used the old code path.

**B-2: `graph_analysis/all_shortest_pair_extraction_phase1.py:40-48` (`get_all_interventions`)**
- Query: `MATCH (i:Intervention) WHERE intervention_maturity >= 3 RETURN id(i), i.name` — single-shot, no batching.
- Population: ~5,590 maturity≥3 interventions in current FalkorDB (verified by F2v4 with RESULTSET_SIZE bumped). UNDER 10k cap, so likely NOT truncated in practice. BUT lower bound on intervention count was higher previously — prior runs may have been affected.
- Downstream impact: `phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/reachability_matrix_{0.8,0.85,0.9,0.95}.npz`, `reachable_interventions_{...}.json`. Phase 1 reachability artifacts feeding `analysis_phase1.py` and threshold-comparison analyses.
- **FIX:** convert to batched id-range. Same pattern as B-1.
- After fix: re-extract reachability matrices.

**B-3: `graph_analysis/phase2_step4_F2v4_hopwise_falkordb.py:87-112` (`query_node_ids` + 4 callers)**
- Query: `MATCH (n:Concept) WHERE n.concept_category = '{subtype}' RETURN id(n)` per subtype + risk + intervention queries.
- Population: 19,178 risks (>10k!), ~144k body nodes (>>10k), ~37k interventions (>10k). All would be truncated under default RESULTSET_SIZE.
- Downstream impact: `paths_hopwise_v4_sim0.9.jsonl` — current rev8 canonical hop-wise enumeration. The current run is SAFE because RESULTSET_SIZE was bumped to 10M before the run. But future runs with default config would corrupt.
- **FIX:** add `GRAPH.CONFIG SET RESULTSET_SIZE 10000000` at script start AND batch the queries by id-range (defense-in-depth).
- After fix: no re-extraction needed for current output (it ran with bumped config).

### MEDIUM severity (diagnostic-only)

**B-4: `graph_analysis/graph_diagnostics.py:651-700` (`component_analysis`)**
- Edge queries with `batch_size=10000` over node-id ranges. With avg ~8 EDGE neighbours per node, each batch can produce >10k edge rows. Truncation possible.
- Downstream impact: NetworkX components computed on truncated edges → falsely fragmented. Diagnostic-only script, not in core pipeline.
- **FIX:** reduce `batch_size` to 2000 or 1000.

**B-5: `graph_analysis/threshold_scan_degree_analysis.py:184-191, 231-249` (`get_node_degrees_by_edge_type_batched`)**
- `batch_size=10000` exactly at the cap. Returns 1 row per node so usually safe but boundary-risky.
- Downstream impact: degree-distribution / threshold-comparison plots.
- **FIX:** reduce `batch_size` to 5000 for safety margin.

### Defense-in-depth (apply to ALL FalkorDB-querying scripts)

Add at script start:
```python
client.execute_command("GRAPH.CONFIG", "SET", "RESULTSET_SIZE", "10000000")
```
This protects against forgetting to batch a query, and against the container being restarted with default config.

Affected scripts (per audit, even those with batched queries currently):
- `graph_analysis/final_pathway_analysis_modes.py`
- `graph_analysis/phase2_step1_loadandparse.py`
- `graph_analysis/phase2_step4_F2v4_hopwise_falkordb.py`
- `graph_analysis/phase2_clustering.py`
- and all others in the audit file list

### SAFE (no fix needed, audit-confirmed)

`phase2_step1_loadandparse.py` (canonical PKL extractor; batch_size=100), `attributes_analyzed.py`, `analysis_phase1.py`, `edge_only_paths.py`, `analyze_path_confidence_contributions.py`, `degree_analysis.py`, `pathway_sampling*.py`, `single_source_node_test.py`, `final_pathway_analysis_modes.py` (batched per-source-paper), `phase2_clustering.py` (uses `RETURN count`/aggregation), `loadalledges_all_shortest_pair_extraction_phase1.py`, `degree_distributions_risks_interventions.py`, `hierarchies_analysis.py` (uses SKIP/LIMIT), `final_summary_analysis_pathways_preclustering.py`, `final_pathways_per_source_histogram.py`, `diagnose_localgraph_extraction.py`, `similaritydebug.py`, `test_phase1_reachability.py`, `test_all_paths_feasibility.py`.

---

## Open Tasks — Detailed Implementation Specs

### Task #B-fix: Apply silent-truncation fixes (NEW, from CF-5 audit)

**Order:**
1. Bump `RESULTSET_SIZE` defense-in-depth in all 25 FalkorDB-querying scripts (cosmetic 1-line addition)
2. Patch `cluster_utils.py:_build_mapping` with id-range batching (HIGH; ~10 line change)
3. Patch `all_shortest_pair_extraction_phase1.py:get_all_interventions` with id-range batching (HIGH)
4. Patch `phase2_step4_F2v4_hopwise_falkordb.py` with id-range batching for node-set queries (HIGH defensive)
5. Reduce `batch_size` in `graph_diagnostics.py` and `threshold_scan_degree_analysis.py` (MEDIUM)
6. Re-extract any artifacts that used corrupted code paths (mainly `ClusterMapper`-derived outputs and `reachability_matrix_*.npz`)

**Estimated time:** 2-3 hours for all patches + reruns.

### Task #7: Body recluster on VPN-only nodes with k-scan + Pareto frontier — CROSS-THRESHOLD QUALITY-BASED THRESHOLD SELECTION (UNBLOCKED 2026-05-01 07:46)

**Quality rationale (added 2026-05-01):** body cluster Pareto is the primary quality signal across thresholds (link/path-quality dropped as weakly informative; see Step 4 §19.3a). Sweep runs F3 on each of the 5 thresholds; chosen threshold is the lowest at which all 5 body subtypes pass intra ≥ 0.70 / inter ≤ 0.30. Full Agglomerative, no sampling. Wall times recorded per threshold.

**Files written:**
- `graph_analysis/phase2_step4_F3_body_recluster.py` ✅ DRAFTED 2026-04-30, updated 2026-05-01 (output-suffix arg, sample-cap=0 default)
- `graph_analysis/phase2_step4_F3_sweep_thresholds.sh` ✅ DRAFTED 2026-05-01
**Inputs:** F2v4 path file → defines VPN_paperpair = nodes appearing in any custom-hop-wise path. Restrict body clustering to these nodes only (also exclude any non-VPN nodes from the clustering input).
**Algorithm:**
1. Build VPN_paperpair from `paths_hopwise_v4_sim0.9.jsonl` (union of body nodes appearing in any path)
2. For each of 5 body subtypes, extract embeddings of VPN_paperpair body nodes
3. K-scan: for k in [10, 15, 20, 25, 30, 35, 40, 50, 60]:
   - AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
   - Compute mean within-cluster cosine sim (intra; high = good)
   - Compute max between-cluster centroid cosine sim (inter; low = good)
   - Compute silhouette as reference
4. **Pareto frontier check (REVIEWER-CRITICAL):** pick k where BOTH:
   - mean intra-cluster cosine sim >= **0.70** (calibrate per subtype if 0.70 unattainable)
   - max inter-cluster centroid sim <= **0.30**
   If no k achieves both thresholds, the clustering at this resolution is inadequate; report as a finding (cluster homogeneity insufficient at all k).
5. Save new cluster_memberships.pkl with the optimized clusters per subtype
6. Apply same scan to risk and intervention clusters (smaller populations)
**Output:**
- `graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/cluster_memberships_rev8.pkl`
- `graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/body_kscan_metrics.csv` (k vs intra vs inter vs silhouette per subtype)
- `graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/body_kscan_chosen_k.csv` (final k choice + Pareto justification per subtype)
- `graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/body_kscan_pareto_plot_<subtype>.png` (visual Pareto frontier per subtype, for paper)
**Estimated time:** 3-5 hours (k-scan is fast; Pareto rigor + plotting per subtype is the main work).
**Sweep run command (when ready, requires user authorization — heavy multi-hour run):**
```
bash graph_analysis/phase2_step4_F3_sweep_thresholds.sh
```
Default order: edge_only → sim0.95 → sim0.9 → sim0.85 → sim0.8 (cheapest-first; failures surface early).
Override order: `bash phase2_step4_F3_sweep_thresholds.sh --order sim0.9 sim0.95 ...`

**Single-threshold dry-run (to validate before full sweep):**
```
conda run --no-capture-output -n base python -u graph_analysis/phase2_step4_F3_body_recluster.py \
  --paths-file graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl \
  --output-suffix edge_only \
  --intra-threshold 0.70 --inter-threshold 0.30 \
  --k-values 10,15,20,25,30,35,40,50,60 --sample-cap 0
```

### Task #7b: Pareto-frontier check for L2 Jaccard frozenset grouping (extension of E3 / F4)

**File written:** `graph_analysis/phase2_step4_F4b_pareto_frozenset.py` ✅ DRAFTED 2026-04-30 end-of-session

After Task #7, when E3-equivalent (Jaccard binary-vector grouping of frozensets) reruns on the new body clusters:
- **Intra-tight:** mean within-group Jaccard sim >= **0.50** (Jaccard is harsher than cosine; lower threshold)
- **Inter-loose:** max between-group centroid Jaccard sim <= **0.20**
- Use same k-scan + Pareto framework
- If Pareto cannot be satisfied, report as a finding (frozenset diversity at L2 too high)

This is the L2 (mechanism family) coherence test, parallel to the L1 (body cluster) coherence test in Task #7. **Both are required for reviewer-defensible mechanism family extraction.**

**Run command (after F1 cooccurrence_families CSV is rebuilt on rev8 body clusters):**
```
conda run --no-capture-output -n base python -u graph_analysis/phase2_step4_F4b_pareto_frozenset.py \
  --cooccurrence-csv graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/optionB_cooccurrence_families_custom_consim1.csv \
  --suffix custom_consim1 --intra-threshold 0.50 --inter-threshold 0.20 \
  --k-values 3,5,8,10,12,15,18,20,25,30
```

### Task #6: Re-run E1-E5 chain on cleaned data (BLOCKED → unblocked once #7 done)

After Task #7 produces new body clusters:
- F1 equivalent: rerun consim1 rebuild on F2v4 paths with new body clusters
- F4: rerun frozenset binary-vector grouping on the new frozensets (k-scan + dynamic K)
- F5: rerun gpt-5.5 LLM naming
- F6: rerun R→Group→I triplets
- E1, E2: regenerate body cluster spread + risk/interv cluster spread on new clusters

### Task #8: Findings reports rev8 (BLOCKED → after Task #6, #7)

Update `Step4_Findings_Report.md` and `Step5_Findings_Report.md` with:
- Path enumeration methodology section: hop-wise DFS rationale, constraint set, max_length cap discussion, comparison table vs custom-BFS
- CF-5 documentation: FalkorDB silent truncation found in this revision; document the audit + permanent fixes
- New rev8 numbers: paths, frozensets, mechanism families, R-I triplets

### Task #9: Threshold-selection sweep on custom data (PENDING → unblocked once F2v4 stable)

Run F2v4 across 5 thresholds (EDGE-only, sim=0.8, 0.85, 0.9, 0.95). For each threshold:
- N qualifying paths (custom + hop-wise + maturity≥3 + consim1)
- N unique (R, I) pairs covered
- % SIM-bridged-only pairs (no EDGE-only path)
- Path length distribution (median, p95)
- Mean N body subtypes per path
- % within-path body-subtype duplicates (over-fragmentation indicator)
**Decision criterion:** maximize R→I coverage and cross-paper bridging volume while keeping over-fragmentation indicators in check.
**Output:** `phase2_results/step4_finalanalysis/step4_config_selection_rev8.md`

### Task #11: Multi-mechanism path detection (PENDING — can run on F2v4 output now)

Two complementary checks for whether each path represents single mechanism vs multi:

**(A) Within-subtype embedding sim check** — runs NOW on F2v4:
- For each path, for each body subtype that appears multiple times, compute pairwise cosine sim of those node embeddings
- High (≥0.85): same concept across papers (over-fragmentation, single-mechanism)
- Low (<0.5): genuinely different concepts in same subtype (multi-mechanism)
- Path is single-mechanism iff ALL multi-instance subtypes show high pairwise sim

**(B) Frozenset coherence check** — runs after Task #7 + E3-equivalent:
- Each path's frozenset assigned to primary L2 mechanism family (by Jaccard nearest centroid)
- Compute distance to primary vs second-nearest family
- Tightly-belonging frozensets = single mechanism; borderline = multi-mechanism

**Combined rule:** path is single-mechanism iff (A passes) AND (B passes). Multi-mechanism paths reported separately as "cross-family connections."

**Output:** `phase2_results/step4_finalanalysis/step4_connectivity/multi_mechanism_classification.csv` with per-path classification.

---

## Decision queue (awaiting user input)

1. After F2v4 finishes: compare R-I pair count vs custom-BFS (21,521). If F2v4 matches or exceeds, the 10,752-pair gap is resolved by RESULTSET_SIZE fix. If still gap, deeper investigation needed.
2. Order of execution: B-fix first (1-2h) → Task #7 (2-4h) → Task #11(A) (~30min, on F2v4 output) → Task #9 (~2h, F2v4 across 5 thresholds) → Task #6 (rerun E-chain, several hours) → Task #11(B) → Task #8 (findings reports). Total ~12-20 hours of work.
3. Cluster-level path enumeration (post Task #7): if combinatorial explosion at L=50 / 10M cap is a recurring problem, switch from node-level to cluster-level path enumeration. Cluster space is ~200 (vs ~200k nodes), so all-paths enumeration becomes tractable without caps.

---

## Key file paths reference

**Path enumeration outputs:**
- `graph_analysis/phase1_rawpathsfiles/paths_custom_sim0.9.jsonl` (custom-BFS, 21,521 paths)
- `graph_analysis/phase1_rawpathsfiles/paths_hopwise_v3_sim0.9.jsonl` (PKL-based hop-wise; deprecated if PKL has issues)
- `graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_sim0.9.jsonl` (FalkorDB live; canonical for rev8)

**F1 consim1 rebuild outputs (uses custom-BFS paths):**
- `graph_analysis/phase2_results/step4_finalanalysis/step4_paths/representative_pathways_custom_consim1.jsonl`
- `graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/optionB_cooccurrence_families_custom_consim1.csv`
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/ri_triplets_custom_consim1.csv`

**F4/F5/F6 preview (NOT FOR PAPER — uses rev7 over-fragmented body clusters):**
- `graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/frozenset_groups_custom_consim1.csv`
- `graph_analysis/phase2_results/step5_naming/frozenset_group_names_custom_llm.csv`

**Plots:**
- `graph_analysis/plots/constrained_modes_path_lengths_all_with_edge.png`
- `graph_analysis/plots/constrained_modes_heatmaps_{edge_only,sim0.8,sim0.85,sim0.9,sim0.95}.png`
- `graph_analysis/plots/constrained_modes_degrees_all_with_edge.png`

**Audit and summaries:**
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/custom_mode_audit_report.csv`
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/custom_consim1_summary.txt`
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/paperpair_v2_summary.txt`
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/hopwise_v3_summary.txt`
- `graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/hopwise_v4_summary.txt` (when F2v4 completes)

---

## Container / runtime notes

- FalkorDB started: `wsl -d Ubuntu -- docker run -p 6379:6379 -p 3000:3000 -it --rm --volume <data_path>:/var/lib/falkordb/data falkordb/falkordb:edge`
- Use `-it`, NOT `-d` (silent exit 255 with no logs).
- Container name auto-generated; use `docker ps` to find ID.
- Indices take ~15 min to build on first start.
- After start, **set `GRAPH.CONFIG SET RESULTSET_SIZE 10000000`** before running any analysis script (or include in scripts as defense-in-depth).
- Python `redis` package installed in conda base env (`redis==7.4.0`).
- All scripts execute via `conda run --no-capture-output -n base python -u ...` for live output.
