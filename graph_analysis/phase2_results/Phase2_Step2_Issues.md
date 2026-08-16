# Phase 2 Step 2 — Open Issues Tracker
**Created:** 2026-03-29
**Last updated:** 2026-03-29 (all computable issues resolved)
**Source:** Filter compliance audit of Phase2_Step2_Comprehensive_Findings.md

---

## Issue 1 — "1792 distinct partner papers" internal inconsistency (Substep #14)
**Type:** Internal inconsistency + known violation
**Status:** ✅ FIXED

**Problem:** Executive summary and Workshop Claim both stated "Hub #1: 635 SIM≥0.9 edges, 1792 distinct partner papers." The Substep #14 top-5 table showed 635 distinct partner papers. A hub with 635 SIM≥0.9 edges cannot have 1792 distinct SIM≥0.9 partner papers — contradictory. "1792" referred to n_sources at SIM≥0.80, mislabeled.

**Fix applied:** `phase2_fix_hub_quality_sim_degrees.py` corrected all degree columns with valid-pathway partner restriction. Corrected values:
- Hub #1 (node 147238): **617 SIM≥0.9 edges**, 617 distinct partner papers (at SIM≥0.9); 1,684 at SIM≥0.80
- Hub #100: 6 SIM≥0.9 edges
- Top-5 hub table updated with corrected node IDs and degrees (valid-pathway restriction changed ranking)
- "1792" removed; replaced with correct threshold-labelled values throughout

---

## Issue 2 — SIMILARITY edge count table presents full-graph PKL stats without scope flag (Substep #14)
**Type:** Missing scope flag
**Status:** ✅ FIXED (added valid-pathway column)

**Problem:** The SIMILARITY score conversion table listed edge counts from the full `graph_edge_data.pkl` (all 200K+ nodes) without flagging this as a full-graph statistic. The valid-pathway-restricted counts are substantially smaller.

**Fix:** Added a fourth column "N edges (valid-pathway both endpoints)" to the table, with counts computed by `phase2_reproducible_calculations.py`. Results:
| Threshold | Full PKL | Valid-pathway | Valid % |
|-----------|----------|---------------|---------|
| ≥0.80 | 1,565,684 | 1,302,549 | 83.2% |
| ≥0.85 | 596,313 | 535,046 | 89.7% |
| ≥0.90 | 144,140 | 134,465 | 93.3% |
| ≥0.95 | 9,127 | 8,850 | 97.0% |

---

## Issue 3 — Betweenness cross-reference in Substep #14 cites full-graph result (Substep #14)
**Type:** Full-graph result cited as corroboration for valid-pathway claim
**Status:** ✅ FIXED

**Problem:** Substep #14 Interpretation cited full 200K-node graph betweenness without conf≥3/maturity≥3 filters.

**Fix applied:** `phase2_rerun_pathfiltered.py` completed betweenness re-run on valid-pathway nodes (38,054 nodes, 163,848 edges). Result: 49/50 top bridge nodes are risk variants of "existential catastrophe / catastrophic misalignment." Cross-reference updated in findings doc to cite path-filtered result.

---

## Issue 4 — "107×" claim conflates different thresholds (Substep #8)
**Type:** Misleading statistic
**Status:** ✅ FIXED (recalculated at consistent threshold, computed by `phase2_reproducible_calculations.py`)

**Problem:** The "107×" claim compared "96.1% of migrating nodes achieve centroid sim >0.8" vs "0.9% of random pairs exceed >0.9 sim" — two different thresholds (0.8 vs 0.9). Not a fair comparison.

**Fix applied:** Recalculated at consistent threshold:
- At >0.8: 96.1% migration vs 22.1% random = **4.4× more directional**
- At >0.9: 0.9% of random pairs (correct baseline context preserved separately)
- Workshop Claim updated; "107×" removed throughout; summary bullet updated

---

## Issue 5 — 0.733 baseline and "0.9% of random pairs" not traceable to any CSV output (Substep #8)
**Type:** Undocumented ad-hoc calculation
**Status:** ✅ FIXED (reproducible calculation in `phase2_reproducible_calculations.py`)

**Problem:** The inter-cluster similarity baseline of 0.733 (range 0.47–0.94) and the "0.9% of random cluster pairs exceed 0.9 similarity" figure were cited as fact but no script or stored CSV produced them directly.

**Fix applied:** `phase2_reproducible_calculations.py` computes these values from `cluster_memberships.pkl` + `graph_node_attributes.pkl` for EDGE/unconstrained/risk/agglomerative. Confirmed values:
- mean_sim = **0.733** ✓ (exact match)
- range = [0.472, 0.942] ✓
- % > 0.9 = **0.9%** ✓ (exact match)
- 40 clusters, 780 unique off-diagonal pairs
- Confidence note updated with reproducible citation

---

## Issue 6 — "1024-dimensional embedding space" factual error (Substep #3)
**Type:** Factual error
**Status:** ✅ FIXED

**Problem:** Substep #3 stated "k=40 forced clustering in a 1024-dimensional embedding space." Incorrect: clustering used 150D UMAP projections; cohesion recomputed on 1536D raw embeddings.

**Fix:** Corrected to "150D UMAP-projected embedding space (clustering step; cohesion recomputed post-hoc on 1536D raw embeddings)."

---

## Issue 7 — Inter-hub cosine similarity (0.955–0.984) undocumented (Substep #14)
**Type:** Undocumented computation
**Status:** ✅ FIXED (reproducible calculation in `phase2_reproducible_calculations.py`)

**Problem:** "The top-5 are near-duplicates of each other (cos_sim 0.955–0.984)" — no script or CSV cited. Values also slightly inaccurate.

**Fix applied:** `phase2_reproducible_calculations.py` computes pairwise cosine similarities between top-5 hub node embeddings from `graph_node_attributes.pkl`. Verified values:
- Range: **0.954–0.979**, mean 0.966 (corrected from original 0.955–0.984)
- All top-5 are "Existential catastrophe from misaligned advanced AI" variants
- Findings doc updated with corrected range and citation

---

## Issue 8 — "10–20% of shortest paths removed" undocumented (Substep #19)
**Type:** Undocumented estimate
**Status:** ✅ FIXED (reproducible calculation in `phase2_reproducible_calculations.py`)

**Problem:** "The filter removes approximately 10–20% of the shortest paths" — no CSV or script produces this figure. Substantially wrong for overall count.

**Fix applied:** `phase2_reproducible_calculations.py` counts paths below the ≥5 hop threshold across all 20 `paths_*.jsonl` files. Result:
- **Overall: 0.82%** of paths below 5 hops (200,738 of 24,535,465 paths)
- **EDGE-only: 8.2%** below 5 hops (1,091 of 13,283)
- Findings doc updated with exact figure and citation

---

## Issue 9 — Violin plot "widest 2017–2021" is linspace approximation (Substep #30)
**Type:** Low-confidence visualization artefact
**Status:** ✅ Already flagged LOW confidence in doc — no change needed

**Problem:** The temporal violin uses linspaced year values between `year_min` and `year_max` per cluster — not individual node publication years. The density shape is a visualization approximation.

**Action:** No change needed beyond confirming the existing LOW confidence flag is present.

---

## Summary

| # | Issue | Action required | Status |
|---|-------|----------------|--------|
| 1 | 1792 partner papers inconsistency | Updated with corrected hub_quality_metrics.csv | ✅ FIXED |
| 2 | SIMILARITY table full-graph scope | Added valid-pathway column | ✅ FIXED |
| 3 | Betweenness cross-reference | Updated with path-filtered result | ✅ FIXED |
| 4 | 107× conflates thresholds | Recalculated at consistent threshold (4.4×) | ✅ FIXED |
| 5 | 0.733 / 0.9% undocumented | Reproducible script confirms exact values | ✅ FIXED |
| 6 | 1024-dimensional factual error | Corrected to 150D UMAP / 1536D raw | ✅ FIXED |
| 7 | Inter-hub cos_sim undocumented | Corrected range 0.954–0.979, cited | ✅ FIXED |
| 8 | 10-20% paths removed undocumented | 0.82% overall, 8.2% EDGE-only | ✅ FIXED |
| 9 | Violin linspace approximation | LOW confidence flag already present | ✅ NO CHANGE |

**Open items:** 0 — all issues resolved ✅
