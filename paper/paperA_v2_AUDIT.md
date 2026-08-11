# Paper A draft v2 — claim audit, 2026-08-11

Every quantitative claim in `paper/paperA_draft_v2.tex` re-derived from **raw** sources
(`graph_node_attributes.pkl`, `paths_hopwise_v4_edge_only*.jsonl`, the judge JSONs) rather than from
the receipt files, then compared to what the draft prints.

**Script:** `graph_analysis/experiment_paper_claim_audit.py` → `phase2_results/experiment_paper_claim_audit.json`
**Arithmetic result: 42 / 42 numeric claims PASS, 0 FAIL** (re-run after the fixes below). No number
in the draft was arithmetically wrong.

The problems found are **interpretation, unit-of-analysis, and provenance** — three of them serious.
All fixes below are already applied to the draft unless marked OPEN.

---

## A. Serious findings (fixed)

### A1. Pseudo-replication in the headline dataset number — FIXED

The draft led with **8,954 chains / 89.0% carry all five stages**. That set is raw path enumeration and is
dominated by a handful of prolific papers:

| | raw | de-duplicated |
|---|---:|---:|
| chains | 8,954 | 2,772 |
| max chains from ONE paper | **700** (7.8% of the set) | 7 |
| papers contributing ≥10 chains | 169 | 0 |
| source papers | 1,868 | 1,868 |
| all-five-stages | 89.0% | **87.4%** |
| length = 7 | 48.0% | 64.0% |
| distinct R→I node pairs | 3,222 | 2,643 |

Reporting per-path statistics on the raw set weights the corpus by how branched each paper's extracted
graph happens to be. **This is the same unit-of-analysis error the paper's own §Practical Guidance
criticises** ("count canonical concepts or papers, never raw nodes") — a reviewer would land on it
immediately. The project's own canonical working set is the de-duplicated one (Part 3 §21.1).

**Fix applied:** the de-duplicated 2,772 set is now the reporting unit throughout (abstract,
contributions, §Pathway Dataset, table, conclusion); the raw set is reported as a sensitivity check;
the 700-path paper is disclosed in Methods. Headline conclusion survives (87.4% vs 89.0%).

### A2. Missing corpus-level denominator — FIXED

"87.4% carry all five stages" is the share **of complete chains**, not of the corpus. Only **1,868 of
11,779 documents (15.9%)** yield any complete high-confidence chain. The draft stated both numbers but
not adjacently, which invited the misreading. A "What this is not" paragraph now states it explicitly.

### A3. `papers_with_complete_strict_chain: 1868` is a hardcoded literal — FIXED

`experiment_query_demo.py:125` writes `1868` as a constant; it is not computed in that script. The value
happens to be **correct** (independently recomputed: 1,868 distinct source URLs across the path set, on
both raw and de-duplicated sets), but the receipt was misleading. The draft now cites the audit receipt
instead, with the hardcoding noted in a `% SRC:` comment.

---

## B. Claims checked and CONFIRMED against raw data

- **Node inventory** — 200,525 total; 19,096 risks; 36,959 interventions; the five subtype counts; and
  they sum exactly to the total. 11,779 distinct source URLs. ✅
- **Quality cuts are real.** The draft's Methods description was initially unverified; it is now
  confirmed both in the builder (`phase2_step4_F2v4_hopwise_falkordb.py`: `EDGE_CONFIDENCE_MIN=3`,
  `INTERVENTION_MATURITY_MIN=3`, single-risk) **and empirically**: 100% of the 8,954 paths start at a
  risk node, **zero** contain a risk after position 0, and every endpoint has maturity 3 (8,050) or
  4 (904). So "complete, high-confidence, mature-intervention" is a supported description. ✅
- **Single-source extraction** — 100% of chains have all nodes from one source URL. ✅
- **Maturity profile** — 27.6 / 57.3 / 12.7 / 2.5%, recomputed from the PKL. ✅
- **Race prevalence** — 2.21% of problem-analysis nodes, 1.65% of risk nodes, recomputed with the
  strict regex. ✅
- **EC table** — all four conditions re-derived from the receipt's raw top-10 lists: 90.1× (merged +
  SIM-excluded), 33.6× (merged + full SIM), 31× flattening on the un-merged graph, 4,066 members /
  7,777 edges. ✅
- **The three retrieval chains** — all three exist verbatim in the raw path file, are single-source,
  their node categories run in canonical chain order, the printed names match the node names, and the
  printed maturity values match. ✅

---

## C. Provenance findings (OPEN — decisions needed)

### P1. Numbers inherited from Gleb's frozen Overleaf — structural ones RE-DERIVED, rest still OPEN

These were printed in the draft as fact with no receipt behind them. The Methods ones were cheap to
re-derive, so they were, and **every one of them disagreed with the frozen Overleaf**:

| Quantity | Frozen Overleaf | Re-derived on our substrate | Status |
|---|---:|---:|---|
| disconnected components (EDGE only) | 18,424 | **15,123** | draft updated |
| largest component | 55 nodes | **61 nodes** | draft updated |
| average degree | 2.0 | **2.02** | agrees |
| average clustering coefficient | 0.007 | **0.013** | draft updated |
| within-category SIM edges at τ≥0.80 | 169,083 | **1,435,806** (8.5× denser) | draft updated |
| components after SIM augmentation | 6,522 | **4,124** | draft updated |
| largest component after SIM | 142,772 (71.4%) | **152,753 (76.2%)** | draft updated |

The discrepancies are explained, not alarming: his counts were measured on the *merged* 200,061-node
graph with 197,542 edges, and his k-NN similarity construction is far sparser than the graph dump this
paper uses. The qualitative conclusion is unchanged in every case. But the paper describes *our* graph,
so it now prints *our* numbers, each with a `% SRC:` comment naming the discrepancy.
*(Re-derivation: union-find over `graph_node_attributes.pkl` + `graph_edge_data.pkl`, 2026-08-11.)*

**Still un-reproduced, still printed as fact:**

| Where | Numbers | Recommendation |
|---|---|---|
| ~~§Practical Guidance + abstract~~ | ~~the 88% race figure~~ | ✅ **RE-DERIVED 2026-08-11 — DOES NOT REPRODUCE. See P5 below. Removed from the abstract; guidance section rewritten around the reproduction failure.** |
| §Practical Guidance | silhouette ≈0.02 raw / 0.298 at k=40 | attribute in-text |
| Appendix B | dedup threshold table (416 / 105,390 / 4,411 / 503 pairs) | attribute in-text |
| Appendix C | clustering comparison table (silhouette / CH / DB) | attribute in-text |
| Appendix E | similarity-hop table; α-sensitivity (Spearman ρ=1.0) | attribute in-text |
| Appendix F | race classifier precision 48/52, recall 36/37 | keep — it is a manual validation Gleb performed and is correctly attributed as such |

### P2. Docstring/code mismatch in the de-duplication threshold

`phase1_dedup_paths.py` docstring says "≥80% contained"; `CONTAINMENT_THRESHOLD = 0.70`. The code is what
ran. The draft states 70%. Fix the docstring so the artifact and the paper agree.

### P5. 🔴 The 88% race figure does not reproduce

Re-derived under the frozen analysis's own definitions (importance = eigenvector centrality; path
diversity = distinct first-hop structural problem-analysis neighbours; race-framed = sole PA neighbour
name matching `/competitive|race/i`), on the closest reconstruction of its graph (merged risk block +
full within-category SIM at τ≥0.80).
**Script:** `graph_analysis/experiment_race_top100_rederive.py` →
`phase2_results/experiment_race_top100_rederive_report.json`.

| Tier | single-path risks | race-framed (frozen claim) | race-framed (re-derived) |
|---|---:|---:|---:|
| top-100 by EC | 38 (claim: 41) | 88% | **2.6%** |
| top-500 | 255 | 45% | 6.3% |
| top-1000 | 569 | 22% | 4.4% |
| all 12,638 canonical risks | 9,141 | 2% | 1.3% |
| head-vs-population gradient | | 44× | **2.0×** |

The **structural** half reproduces well (38 vs 41 single-path risks in the head). The **framing** half
does not: 2.6% against 88%, a 34-fold discrepancy. On the decontaminated graph (un-merged + risk↔risk
SIM excluded) it is 0.0% in the top-100.

Most likely cause: the similarity layer in the graph dump this paper uses is ~8.5× denser than the one
the frozen analysis ran on (1,435,806 vs 169,083 within-category edges at τ≥0.80), so the centrality
ranking — and therefore *which* 100 risks are selected — is not the same set. That is not a
reconstruction we can close without his Iteration-B pipeline.

**Action taken:** the 88% is removed from the abstract and is no longer asserted anywhere. The guidance
subsection now reports the reproduction failure itself, which makes the *stronger* point: a
selection-conditioned statistic on this kind of graph is not merely inflated but **unstable under
pipeline parameters**, so it should never be reported without the selection rule, the graph
construction, and a full-population baseline. The section's other examples (the 90× merge artefact, the
odds-ratio table, the 51→2 isolation result) are all receipt-backed and carry the argument.

**Gleb should see this before the sprint** — it directly concerns his section and he may be able to
explain the gap from his side.

### P3. The flagship retrieval example is sourced from a Google Doc

Query 1 (governance / export controls, the maturity-4 example) comes from
`https://docs.google.com/document/d/1DF31DIkwS9GONzmy1W3nuI9HRAwSKy8JcIbzKYXg-ic/...`. Fetched
2026-08-11: the document is titled **"Transformative AI and Compute - Reading List [shared]"** — a
bibliography, not a research paper. Its node metadata in the graph is entirely empty (`title`,
`authors`, `date_published`, `source`, `filename` all `None`); it contributes 19 nodes.

This is worse than a formatting nit. The paper's flagship retrieval demo shows a complete
risk→…→intervention argument with a deployed (maturity-4) intervention, extracted from a **reading
list** — a document that does not itself propose an intervention. Either the extraction inferred the
argument from listed titles/annotations, or it attributed claims made in the *cited* works to the list
itself. **Recommendation: swap Query 1 for a chain whose source is a real paper, and separately check how
many chains in the dataset originate from bibliography-like `special_docs` sources.** Queries 2 and 3 are
fine (openai.com/research/summarizing-books and aisafety.info).

### P4. Correction to a prior working assumption — meta-grader scores are NOT bug-contaminated

Earlier project notes assumed the judge-of-judge run scored the buggy applied `final_graph`, so its
scores inherited judge.py bugs 5/6. **Verified false.** The rubric prompt asks for a score "before and
after the judge's *proposed* fixes", and `combine_judge_and_extraction_with_original_text.py` bundles
Original Text + Extraction Output + Judge Output for the grader. The graders never saw an applied graph.
Consequence: the post-repair score is *hypothetical* (quality of proposed repairs as described, not a
re-extracted graph) — now stated in the draft — but it is free of the fix-application bugs, which
**removes the need to fix those bugs before writing the paper**.

---

## D. What this audit did not cover

- Prose claims with no number attached (framing, related-work characterisations).
- The judge receipt's internals — audited separately when it was built (see `paperA_v2_GAPS.md` G1).
- Anything in the sections still marked `[GAP: ...]` (Related Work, Appendix A, author list).
