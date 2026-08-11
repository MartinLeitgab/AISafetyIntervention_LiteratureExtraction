# Paper A draft v2 — provenance + gap list

**Built 2026-08-11** from `inputs/Gleb_overleaf_paper.txt` (Iteration B, frozen 2026-01-26)
restructured per `graph_analysis/phase2_results/paperA_positive_outline_2026-06-26.md`.
Draft: `paper/paperA_draft_v2.tex`. Every number in the draft carries a `% SRC:` comment
naming the on-disk receipt it came from.

---

## 1. Where the draft came from (reuse vs new)

| Draft section | Source | Status |
|---|---|---|
| Abstract | new | written |
| Introduction | Overleaf bullets → new prose | written |
| Related Work | Overleaf bullets ("Axel started") | **stub + CITE markers** |
| Methods / Corpus + Extraction | new (schema from `core/node.py`, `edge.py`) | written |
| Methods / Dedup + edge norm | Overleaf L143–168 **verbatim** + honest-scope ¶ new | written |
| Methods / Structural diagnostics | Overleaf L172–180 verbatim, **moved Results→Methods** | written |
| Methods / SIM layer + quality cuts | Overleaf L217–239 reworked | written |
| Methods / Validation | — | **BLOCKED on Sai** |
| §Pathway Dataset | new, from `experiment_dataset_strength_report.json` + `experiment_query_demo_report.json` | written |
| §Mechanism-Level Retrieval | new, from `experiment_query_demo_report.json` | written |
| §Practical Guidance | new prose over Gleb's material + our 5 experiment receipts | written |
| Limitations / Outlook / Conclusion | new | written |
| App. B dedup thresholds | Overleaf App. D verbatim | written |
| App. C clustering comparison | Overleaf App. F verbatim | written |
| App. D cluster reference | Overleaf App. E, reframed as browsing index | written |
| App. E τ + hops + α sensitivity | Overleaf App. G + H merged | written |
| App. F race classifier validation | Overleaf App. J, prevalence claims stripped out | written |
| App. A extraction prompt | — | **paste needed** |
| App. G worked chains | — | **needs 2–3 chains** |

Net: ~35% reused text, ~65% new. All new quantitative content is receipt-backed today.

---

## 2. Gaps — what is actually missing

### BLOCKING (paper cannot be submitted without it)

**G1. Judge validation — SUBSTANTIALLY CLOSED 2026-08-11.** All of Mike's data is local
(`Final-archive-from-Mike/`, `Mike2/judge_recovery_bundle/`) plus the 100 judge reports in git at
`origin/anthropic_judge_test :: extraction_validator/extend_try_1/`.

**Canonical receipt: `graph_analysis/phase2_results/experiment_judge_full_report.json`**, produced by
`graph_analysis/experiment_judge_full_receipt.py` (no LLM calls, fails fast, ~seconds). The paper cites
the receipt; nobody re-derives from raw. Supersedes the narrower `experiment_judge_item2_report.json`.
Validation of the extraction: the script reproduces all three of Mike's published `results.md` aggregates
(69.65/73.13, 62.69/95.77, 66.96/80.18) exactly.

Now in the draft: judge audit (referential/orphan/duplicate/coverage), meta-grader table with honest $n$,
Fleiss κ + pairwise Spearman, auto-derived error profile, and the failed-extraction recovery result.

*🔴 Three data findings that change what may be claimed:*

1. **`has_blockers = 87%` and `is_valid_json = 85%` are unusable.** 184 of 218 blocker-severity schema
   flags (84%) are the judge demanding inline `*_rationale` fields the pipeline stores as separate
   `:Rationale` nodes. Substantive residue: 34 blockers / 100 papers.
2. **The three meta-graders scored unequal, partly disjoint subsets** — Opus n=95, GPT-5.1 n=95,
   **Gemini n=13** — because each directory mixes rubric-prompt iterations and only some emit a pre/post
   pair. Only **13 papers** carry all three scores. `results.md`'s three-row table is *not* a like-for-like
   comparison and must not be presented as one. Gemini's 95.8±1.8 is n=13 saturation.
3. **🔴 "~60 recovered of ~400 processable ≈ 15%" is WRONG** and is repeated in the standup log, in
   `extraction_validator/STATUS.md`, in the recovery-bundle README, and in project memory. The 65 files in
   `recovered_errors_graph/` have **zero overlap** with the 441 attempts in `recovered_errors/`, and
   include source types absent (`agisf`) or near-absent (`arxiv`: 14 files vs 2 candidates) from the
   candidate set — they are from a different failure population (most likely the `graph_error` set of 91
   dirs the README says it excluded), with a denominator that is not in the bundle.
   **Defensible number: 23 of 441 attempts (5.2%) produced a non-empty graph, mean 2.4 nodes / 0.9 edges.**
   The paper now states this, and draws the useful conclusion (a judge repairs existing extractions well
   and reconstructs failed ones poorly).

*Still genuinely blocked on Sai (ticket #147):*
- **Manual 50-instance error taxonomy** with human adjudication and distribution by source type.
- **Human-anchored spot-check** — no human adjudicated any judge or meta-grader verdict, so the entire
  validation chain is LLM-internal. This is the weakest point a reviewer will press.
- **Confirm the third grader's model id** — `results.md` says GPT5.1, `STATUS.md` says GPT-5.2.
- *Optional, would strengthen:* re-run Gemini (or a third grader) on the full 100 so the meta-grader
  comparison is like-for-like; establish the provenance/denominator of the 65-file recovered set.

### NON-BLOCKING, small (hours)

**G2. Figure 1** — two panels: chain-length histogram + maturity stacked bar. Plots directly
from `experiment_dataset_strength_report.json`. White background. ~1h.

**G3. Source citations for the 3 retrieved chains** (Table `tab:query`) —
lookup `node_attrs[nid]['url']` for the `path_node_ids` already in
`experiment_query_demo_report.json`. ~15 min script.

**G4. Appendix A** — paste `PROMPT_EXTRACT` from
`intervention_graph_creation/src/prompt/final_primary_prompt.py` + node/edge schema summary, trim to 1 page.

**G5. Appendix G** — 2–3 additional worked chains from
`paths_hopwise_v4_edge_only_deduped.jsonl`, **including one deliberately imperfect chain**
(missing stage / low-confidence edge) so reviewers see failure cases.

**G6. Author list, affiliations, acknowledgments.** Contributors on record:
Martin Leitgab, Gleb Maksimov, Sai, Mike, Axel, Jeffrey Parks. Order + affiliations unconfirmed.

### NON-BLOCKING, medium (a day)

**G7. Related Work** — 4 strands scoped in the draft with `[CITE: ...]` markers; needs ~3 paragraphs
of prose and ~12 bib entries in `aaai25.bib`. Axel started it; Jeff Parks offered review bandwidth.

**G8. Dataset/code release artifact.** The abstract and Outlook both say "we release". There is no
artifact yet — needs a hosted dump (graph + path set + extraction code) and a stable link.

**G9. Multi-model consistency (n=20, o3 / GPT-5 / Claude-4)** — Mike ran this; if recoverable from the
ticket-#147 recovery bundle, one sentence in Limitations materially strengthens the single-extractor
caveat. If not recoverable, delete the marker.

**G10. Cluster representatives** — publish `cluster_representatives_20.json` alongside the dataset so
reviewers can audit cluster naming (Appendix D). *Owner: Gleb.*

### DECISIONS, not work

**G11. Venue + page limit — not committed.** Current draft is AAAI 2-column, ~8 pages main + 7 appendices.
Trimming target depends entirely on this. Decide before polishing.

**G12. Does §Practical Guidance get a figure?** It is currently table-only (EC table + race OR table).
Gleb's `figure2_race_dynamics_combined.png` panel A (the 88%→2% concentration gradient) would work as an
illustration of the selection artefact *if* recaptioned as a failure mode. His
`figure1_hub_gaps.png` should **not** be reused — it plots the cut coverage-ratio finding.

**G13. Optional qualitative landscape figure (outline §8)** — still optional, still not built. It needs the
quality-cut path VPN with separate risk/intervention axes. Recommend: skip for the workshop bar; it adds a
figure and a paragraph of guardrail caption for no load-bearing claim.

---

## 3. What was cut from the Overleaf draft (so nothing is silently dropped)

Removed as artefact-driven, by-design, or non-load-bearing:

- Eigenvector-centrality "top central risks are existential" **as a finding** → reframed into
  §Practical Guidance as the merge artefact (with the 4-condition EC table).
- Research-coverage ratios (326:1, 89:1, 31:1) and the "theoretical vs empirical content" correlation.
- Understudied score (Eq. 2) and the top-10 understudied risk table (old App. I).
- Path completion (13.6%) and the "98% of risks have a complete path" stat.
- Single-path-as-structural-fragility; path diversity as a finding.
- Root / bridge / leaf typing; low-silhouette-as-hub correlation (r = −0.34).
- Bridge-concept betweenness analysis; the 66%-intra-cluster siloing result.
- Race as a headline: 88% of top-100, 51/100 isolation, 44× gradient, micro-narratives,
  intervention-type bias (χ²=66.24). Race survives only as a **measured selection artefact** plus a
  minor-but-robust OR table.
- Old App. I (top-risk tables) dropped entirely.

Kept and reframed: dedup methodology, structural diagnostics, clustering comparison, τ/hop/α
sensitivity, cluster list, race classifier validation.

---

## 4. Critical-path read

One blocking item (G1, Sai). G2–G6 are a single focused day. G7 is the only sizeable writing task left
and it is the most delegable. That fits the **Aug 17–28 sprint window** Gleb proposed on 2026-08-07 —
which is still unanswered.
