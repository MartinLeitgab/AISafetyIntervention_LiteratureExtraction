# Issue #147 — proposed rewrite (drafted 2026-08-11, NOT yet posted)

Current ticket body last updated **2026-06-10**; it is now substantially stale. Below: what changed,
what to drop, and a replacement body scoped to the minimum workshop-paper threshold.

---

## 1. Why the ticket needs rewriting

The ticket was written when the data was believed to be on Mike's local disk and unanalysed. Since then
all of it has been located and analysed (`graph_analysis/experiment_judge_full_receipt.py` →
`phase2_results/experiment_judge_full_report.json`), and the paper section is drafted
(`paper/paperA_draft_v2.tex` §Methods/Validation). Most of the ticket's 22 checkboxes are done, moot, or
now known to be based on wrong premises.

**Three premises in the current ticket body are factually wrong and must be corrected in place:**

1. **"~60 records recovered / ~400 processable (~15%)"** (§1 table row A, and §2 Item-3 last row). The 65
   files in `recovered_errors_graph/` have **zero overlap** with the 441 attempts in `recovered_errors/`
   and contain source types absent from the candidate set. Correct figure: **23 of 441 (5.2%)**.
2. **Meta-grader means are presented as comparable** (§1). They are computed on unequal, partly disjoint
   subsets: Opus n=95, GPT-5.1 n=95, **Gemini n=13**; only 13 papers carry all three scores.
3. **"These [judge.py bugs] must be fixed before any new judge run is trusted"** implies they block the
   paper. They do not: the meta-graders scored the judge's *proposed* fixes from the report text, never
   an applied `final_graph`, so bugs 5/6 cannot have contaminated any number the paper reports.

---

## 2. Drop from the ticket (done, moot, or not needed for the workshop bar)

| Current item | Disposition |
|---|---|
| Martin to retrieve per-file rubric JSONs from Mike's disk → Drive | **DONE** — local in `Final-archive-from-Mike/`, now aggregated into the receipt |
| Martin to retrieve the recovered records → Drive | **DONE** — local in `Mike2/judge_recovery_bundle/` |
| Compute Fleiss' κ + pairwise agreement | **DONE** — κ=0.54 pre / 0.09 post on the 13 common papers; pairwise Spearman in the receipt |
| Error-rate breakdown by category from `extend_try_1/` | **DONE** — in the receipt, with the rationale-field artifact split out (the ticket did not anticipate that artifact) |
| Per-source-type table | **DONE** — in the receipt |
| Persist recovery summary CSV | **DONE in substance** — receipt JSON carries it; a CSV is optional packaging |
| Error patterns by source type (Item 3) | **DONE** — in the receipt |
| Bugs 1–6 + integration tests + merge reference branch | **DROP from #147** → move to a separate hygiene ticket. Required only for *future* judge runs; cannot affect any published number |
| "Finish the GPT-5.2 rubric run" | **DROP** — mis-stated. GPT-5.1 is the complete grader (n=95). The thin grader is Gemini (n=13) |
| "Re-run the Claude meta-grader with the same final prompts" | **DROP for the workshop bar** — footnote the prompt-iteration heterogeneity instead |
| Write Item-2 / Item-3 Overleaf paragraphs | **RESCOPE** — §Methods/Validation is already drafted; Sai reviews and corrects rather than writes |
| Update #125, decide on PR #145 | **KEEP** — cheap closeout |

---

## 3. Proposed replacement body

> ## Goal
>
> Close out the judge validation to the **minimum workshop-paper threshold**. Everything computable from
> existing data is done and lives in one receipt:
> `graph_analysis/phase2_results/experiment_judge_full_report.json` (built by
> `graph_analysis/experiment_judge_full_receipt.py`; no LLM calls, fails fast, seconds to run).
> The paper section is drafted at `paper/paperA_draft_v2.tex` §Methods/Validation.
>
> **All source data is local** — `Final-archive-from-Mike/` and `Mike2/judge_recovery_bundle/`. Nothing is
> blocked on Mike.
>
> ### Corrections to the previous ticket body (please read first)
>
> 1. The "~60 recovered / ~400 processable (~15%)" figure is **wrong** — those two directories are disjoint
>    populations. Correct: **23 of 441 attempts (5.2%)** produced a non-empty graph, mean 2.4 nodes /
>    0.9 edges. The 65-file `recovered_errors_graph/` set comes from a different failure population whose
>    denominator is not in the bundle.
> 2. The three meta-graders scored **unequal, partly disjoint subsets** — Opus n=95, GPT-5.1 n=95,
>    Gemini n=13, with only 13 papers common to all three. Gemini's 95.8±1.8 is n=13 saturation.
> 3. The judge.py bugs do **not** affect any number in the paper: meta-graders scored the judge's
>    *proposed* fixes as described in the report, never an applied `final_graph`.
>
> ### Already done — no action needed
>
> 100-paper judge audit (referential / orphan / duplicate / coverage, with the rationale-field schema
> artifact separated); per-source-type breakdown; per-paper pre/post scores for all three graders;
> Fleiss' κ (0.54 pre → 0.09 post) + pairwise Spearman; auto-derived error profile from the Opus
> structured fields (216 missed concepts / 20 fabrications / 15 category errors over 95 papers);
> failed-extraction recovery rate. All in the receipt.
>
> ### Open — required for the workshop bar
>
> - [ ] **Manual 50-instance error taxonomy.** Sample 50 error instances stratified by source type from
>       the BLOCKER/MAJOR-flagged subset, hand-code into 6 categories (hallucinated node, hallucinated
>       edge, missing node, missing edge, wrong edge type, granularity mismatch), report the distribution.
>       → `extraction_validator/error_taxonomy_50.csv`. **This is the one deliverable no automation can
>       replace, and it is the ticket's centre of gravity.**
> - [ ] **Human-anchored spot-check (~20 papers).** A human scores extraction faithfulness directly, so the
>       validation chain is not entirely LLM-internal. This is the single weakest point a reviewer will
>       press. → `extraction_validator/human_anchor_20.csv`.
> - [ ] **5–10 annotated examples** with source quotes, drawn from the 50 above.
> - [ ] **Confirm the third meta-grader's model id** — `results.md` says GPT5.1, the old ticket body says
>       GPT-5.2. One line.
> - [ ] **Review** `paper/paperA_draft_v2.tex` §Methods/Validation against the receipt and correct anything
>       mis-stated. (Writing is done; this is a review pass, not a drafting task.)
>
> ### Nice-to-have — only if the above lands early
>
> - [ ] Re-run Gemini (or a third grader) over all 100 papers so the meta-grader comparison is
>       like-for-like instead of n=13 vs n=95.
> - [ ] Establish the provenance and denominator of the 65-file `recovered_errors_graph/` set.
>
> ### Moved out of this ticket
>
> judge.py bugs 1–6, their integration tests, and merging the reference branch into `main` → new hygiene
> ticket. They are prerequisites for any *future* judge run, not for this paper.
>
> Closes #76. Related: #125, #77, #83.

---

## 4. Net effect

Sai's open list goes from **22 checkboxes to 5**, and the remaining five are all things that genuinely
require a human: hand-coding 50 errors, a human spot-check, annotated examples, one factual confirmation,
and a review pass. Estimated effort drops from ~10–15 h to roughly **4–6 h**.
