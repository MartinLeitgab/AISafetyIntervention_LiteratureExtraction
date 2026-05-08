## Goal

Close out **Workshop Validation Plan items 2 and 3** — the judge-analysis and error-taxonomy deliverables on the 100-paper sample. This supersedes #76 (broader "implement the judge" scope), which is being closed in favor of this scoped close-out ticket.

The Workshop Validation Plan lives in Drive sheet `1Z8f7MJuWIvjWvcWvPSe3_r3g32z5n9j7KcV9vNawh6I` ("MINIMAL WORKSHOP VALIDATION PLAN") inside the project folder https://drive.google.com/drive/u/0/folders/1u5r4OTasGRhcHRAoXFPMl6E_dr4abIrZ.

A reference branch holds the restored code and a comprehensive `STATUS.md`:
- Branch: [`judge_handoff_workshop_items_2_3`](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/tree/judge_handoff_workshop_items_2_3)
- Status doc in branch: [`extraction_validator/STATUS.md`](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/blob/judge_handoff_workshop_items_2_3/extraction_validator/STATUS.md) — mirrors this issue body, plus a few cross-refs.

The branch restores rubric prompts and combine scripts that produced data on `rubric_2_and_3` but were missing from `main`. It does NOT include data dumps (data is on Drive or on Mike's local disk — see §4).

---

## 1. Three judge sub-projects (history)

| # | Sub-project | Window | Model | Sample | Output location | Status |
|---|---|---|---|---|---|---|
| A | **Failed-extraction recovery** — does the judge salvage extractions that errored out? | Oct–Nov 2025 | OpenAI `gpt-5-nano` (batch) | ~400 processable error sources (out of ~1,400 total errors flagged in Gleb's `processed.zip`) | NOT in any git branch — Mike's local disk; planned Overleaf write-up (incomplete) | **~60 records recovered / ~400 processable (~15 %).** Per team standup 2025-11-03. |
| B | **Working-paper extension** — does the judge add missing nodes/edges to extractions that already succeeded? | 2025-11-26 onward | Anthropic `claude-sonnet-4-5` (batch) | 100 successful extractions | PR #145 (`extraction_validator/extend_try_1/`, `extraction_validator/test_extend/`) — branch `anthropic_judge_test`, OPEN, marked "don't have to merge" | Done. 100 judge reports written. No quantitative summary yet. |
| C | **Judge-of-judge / multi-grader rubric calibration** — do independent meta-graders agree on whether the judge's fixes improve the extraction? | Dec 2025 – Jan 2026, in flight | Meta-graders: Claude Opus 4.5, Gemini 3 Pro, GPT-5.2 | Same 100 papers as B | Branch `rubric_2_and_3` (`combined_extraction_with_original_text/`, `prompts/`, `results.md`). Per-file rubric JSONs only on Mike's local disk. | Opus + Gemini done; GPT-5.2 incomplete (model "lazy" / refusing the 100-paper task as of 2026-01-08). |

**Note.** The rough framing "100 error sources + 100 non-error sources" is approximate. Actual: 100 successful papers (sub-projects B and C share this sample); ~400 processable error papers (sub-project A — never sized to 100, sized to "all errors that aren't trivially missing source content").

Latest meta-grader pre/post means (`results.md` on the reference branch):

| Meta-grader | Pre mean ± std | Post mean ± std |
|---|---|---|
| Claude Opus 4.5 | 69.65 ± 18.01 | 73.13 ± 18.21 |
| Gemini 3 Pro | 62.69 ± 22.58 | 95.77 ± 1.80 |
| GPT 5.1 | 66.96 ± 16.19 | 80.18 ± 10.21 |

Gemini's post-judge std collapsing to 1.80 with mean 95.77 looks like ceiling-bias / overgrading rather than uniform excellence — flag for closer inspection during the κ analysis.

---

## 2. Workshop items 2 and 3 — what's done, what's open

### Item 2 — Judge Analysis on Extraction Quality (100 papers)

| Required deliverable | Status | Where it lives / what's missing |
|---|---|---|
| 100-paper judge-evaluated set | ✅ done | `extend_try_1/` on PR #145 (Anthropic Sonnet 4.5 outputs); same 100 in `combined_extraction_with_original_text/` on `rubric_2_and_3` |
| Pre / post extraction scores per paper, per meta-grader | ⚠️ partial | `results.md` aggregates 3 meta-graders. **Per-file rubric JSONs (raw scores) only on Mike's local disk — Martin to retrieve and upload to Drive.** |
| Inter-judge agreement (Fleiss' κ) between meta-graders | ❌ missing | Only mean ± std reported. Need κ + pairwise agreement matrix. |
| Judge-of-judge calibration | ⚠️ partial | 3 meta-graders provide raw signal; no spot-check vs human anchor. |
| Overall error rate + breakdown by type (hallucinated nodes, hallucinated edges, missing nodes/edges, wrong edge type) | ❌ missing | `extraction_stats.py` only counts `add_nodes`/`add_edges` aggregates; doesn't categorize. |
| Source-heterogeneity table (arXiv / blog / YouTube etc.) | ⚠️ partial | Prompt example shows the table; not produced from real data. |
| Summary statistics for all 100 papers | ⚠️ partial | `results.md` has 3-row pre/post mean/std table only. |

### Item 3 — Error Taxonomy from Judge Data

| Required deliverable | Status | Where it lives / what's missing |
|---|---|---|
| Filter to error-flagged docs | ⚠️ partial | Sub-project A processed ~400 error sources; output not in git. |
| Sample 50 specific error instances | ❌ missing | |
| Manual categorization (hallucinated, missing, wrong-type, granularity-mismatch) | ❌ missing | |
| Distribution % across error types | ❌ missing | |
| 5–10 illustrative annotated examples | ❌ missing | |
| Error patterns by source type | ❌ missing | |
| Improvement recommendations | ❌ missing | |
| Recovery rate of failed extractions (related, sub-project A) | ✅ partial | "60 / 400 processable, 0 / 1000 missing-source" per team standup 2025-11-03. Needs to be moved into Overleaf and into a CSV in the repo. |

---

## 3. Known bugs in `extraction_validator/judge.py` (current `main`, post PR #146 HEAD `f22e504`)

These must be fixed before any new judge run is trusted. **Each fix needs an integration test** under `extraction_validator/tests/` (per the "every bug fix needs a test that exercises the fix" rule).

| # | Severity | Where | Description |
|---|---|---|---|
| 1 | High | `judge.py:238` | **Stale model literal `gpt-4-vision-preview`.** Deprecated multimodal preview; not what was actually used. The OpenAI path is currently dead code; `--model_type=OpenAI` will likely 404. The active path is Anthropic (`claude-sonnet-4-5`, `judge.py:258`). The saved `test_output/` predates this literal — actual model used there was `gpt-5-nano`. **Fix:** replace with `gpt-5-nano` (or current OpenAI default) or delete the OpenAI path. |
| 2 | High | `judge.py:879` | **`concept_category="concept"` literal** for new concept add-nodes. Schema requires `risk \| problem analysis \| theoretical insight \| design rationale \| implementation mechanism \| validation evidence`. **Fix:** use `add_node.concept_category` from the judge's response. |
| 3 | High | `judge.py:891` | **`aliases=[add_node.aliases]`** wraps a list inside a list. **Fix:** `aliases=add_node.aliases`. |
| 4 | High | `judge.py:893` | **`concept_category=add_node.concept_category` set on intervention nodes.** Interventions must have `concept_category=None`. **Fix:** set to `None` on the intervention branch. |
| 5 | **Catastrophic** | `judge.py:917` | **`break` at end of `for add_node_fix in proposed_fixes.add_nodes` loop.** Only the first add-node is ever processed; the rest silently dropped. The two `if … break` lines inside the loop also exit on any conflict (instead of `continue`-ing). **Fix:** remove the trailing `break`; convert the inner `break`s back to `continue`. |
| 6 | Medium | `judge.py:_apply_fixes_to_graph` | **`proposed_fixes.merges` is parsed by the schema but never consumed.** Judge-recommended node merges are silently dropped. **Fix:** add a merge loop that retargets edges from to-merge nodes onto the new merged node and deletes the originals. |

---

## 4. Where every artifact actually lives

### Code

Currently on `main`:
- `extraction_validator/judge.py` — main batch runner (post PR #146; bugs 1-6 above outstanding)
- `extraction_validator/schema.py` — Pydantic schemas with the `ValidatedDataOrOriginalOnError[T]` envelope (introduced in PR #146)
- `extraction_validator/utilities.py` — Anthropic / OpenAI batch primitives
- `extraction_validator/find_judge_able_sources.py` — filter to sources with both KG output and original text
- `extraction_validator/create_directoy_for_retry.py` — re-process error files from a previous batch run
- `extraction_validator/extraction_stats.py` — aggregate `add_nodes` / `add_edges` counts (added in PR #146)

**Already restored on the reference branch** (was on `rubric_2_and_3`, missing from `main`):
- `extraction_validator/prompts/extraction_evaluation_prompt.md` — rubric 1
- `extraction_validator/prompts/judge_evaluation_prompt.md` — rubric 2
- `extraction_validator/prompts/judge_and_extraction_evaluation_prompt.md` — rubric 3 (produces `pre_judge_score` / `post_judge_score`)
- `extraction_validator/combine_extraction_with_original_text.py`
- `extraction_validator/combine_judge_with_original_text.py`
- `extraction_validator/combine_judge_and_extraction_with_original_text.py`
- `extraction_validator/get_judge_improvement.py` — pre/post score aggregator
- `extraction_validator/results.md` — current pre/post mean ± std table

### Data

| Asset | Where | Action needed |
|---|---|---|
| 100 successful-extraction sample (input) | PR #145 `test_extend/` (300 files = 100 × 3) | Keep on PR #145 OR upload one tarball to Drive |
| 100 judge reports on those (Anthropic Sonnet 4.5) | PR #145 `extend_try_1/` (102 files) | Same |
| 100 combined extraction+text bundles for meta-grader | `rubric_2_and_3` `combined_extraction_with_original_text/` | Same |
| Per-file rubric JSONs from Opus 4.5, Gemini 3 Pro, GPT-5.2 (raw `pre_judge_score` / `post_judge_score` per file) | **Mike's local disk only** | Martin to retrieve from Mike and upload to Drive sub-folder `judge_workshop_item_2_rubric_outputs/` |
| Sub-project A failed-extraction inputs | Drive folder https://drive.google.com/drive/u/0/folders/1RPbETx21KMyEATtVOBt8LOClVd56eZ4x → `processed.zip` → `extraction_error/` | Already in Drive |
| Sub-project A judge outputs (60 recovered records, ~340 attempted-but-failed) | **Mike's local disk only** | Martin to retrieve from Mike and upload to Drive sub-folder `judge_workshop_item_3_error_recovery/recovered/` and `not_recoverable/` |
| 6-paper arxiv mini-test from `gpt-5-nano` | Repo `extraction_validator/test_output/` and `test_processed/` | Already in repo |

---

## 5. TODO list

### Code cleanup (small, mechanical, all need a test under `extraction_validator/tests/`)
- [ ] **Bug 1** — replace stale `gpt-4-vision-preview` model literal in `judge.py:238` (use `gpt-5-nano` or current default), or delete the OpenAI code path if Anthropic is the only target.
- [ ] **Bug 2** — fix `concept_category="concept"` literal at `judge.py:879` to use `add_node.concept_category`.
- [ ] **Bug 3** — fix `aliases=[add_node.aliases]` at `judge.py:891` → `aliases=add_node.aliases`.
- [ ] **Bug 4** — set `concept_category=None` for the intervention branch at `judge.py:893`.
- [ ] **Bug 5** — remove trailing `break` at end of the `add_nodes` loop and convert the two early-exit `break`s to `continue` (`judge.py:867`, `:870`, `:917`).
- [ ] **Bug 6** — implement `merges` consumption in `_apply_fixes_to_graph` (currently silently dropped).
- [ ] **Merge reference branch into `main`** to permanently restore the 8 code/doc files in §4 ("Already restored on the reference branch").

### Workshop Item 2 — Judge Analysis (100 papers)
- [ ] Martin to retrieve per-file rubric JSONs (Opus 4.5, Gemini 3 Pro, GPT-5.2) for all 100 papers from Mike's local disk and upload to Drive folder `1u5r4OTasGRhcHRAoXFPMl6E_dr4abIrZ` under `judge_workshop_item_2_rubric_outputs/`.
- [ ] Finish the GPT-5.2 rubric run. Current state: model refused the 100-paper task as of 2026-01-08. Subdivide into smaller batches and re-run.
- [ ] Re-run the Claude meta-grader with the same final prompts so all 3 meta-graders use the same prompt version.
- [ ] Compute Fleiss' κ + pairwise agreement between the 3 meta-graders on per-paper verdicts (or pre/post score buckets). Add to `results.md`.
- [ ] Compute error-rate breakdown per the Workshop spec (hallucinated nodes / edges, missing nodes / edges, wrong edge types) by parsing `validation_report.schema_check` / `referential_check` / `coverage` lists in the 100 judge outputs in `extend_try_1/`. Save as `extraction_validator/error_rate_breakdown.csv`.
- [ ] Compute per-source-type table (arXiv / blog / YouTube / lesswrong / alignmentforum / eaforum / arbital / aisafety.info / special_docs) showing mean pre/post score and error-rate breakdown. Save as `extraction_validator/by_source_type.csv`.
- [ ] Write the Item-2 paragraph for the Overleaf (https://www.overleaf.com/project/6891855bebedbb8d7e5ff7f6) including κ, error-rate breakdown, and source-heterogeneity table.

### Workshop Item 3 — Error Taxonomy
- [ ] Martin to retrieve the 60-recovered records (and the ~340 attempted-but-failed processable records from sub-project A) from Mike's local disk and upload to the Drive folder under `judge_workshop_item_3_error_recovery/`.
- [ ] Persist the recovery summary as `extraction_validator/error_recovery_summary.csv` with columns `(source_type, paper_id, error_class, judge_attempt_status, recovered_node_count, recovered_edge_count, error_text_excerpt)`.
- [ ] Sample 50 error instances (from sub-project A or the BLOCKER/MAJOR-flagged subset of sub-project B), stratified by source type.
- [ ] Manually categorize into 6 categories: (a) hallucinated nodes, (b) hallucinated edges, (c) missing nodes, (d) missing edges, (e) wrong edge type, (f) granularity mismatch. Save as `extraction_validator/error_taxonomy_50.csv`.
- [ ] Compute distribution % across categories, broken out by source type. Save as `extraction_validator/error_taxonomy_distribution.csv`.
- [ ] Pick 5–10 illustrative examples with text quotes + commentary. Save as `extraction_validator/error_taxonomy_examples.md`.
- [ ] Write the Item-3 paragraph for the Overleaf with the taxonomy table, recovery rate, and improvement recommendations.

### Closeout
- [ ] Update Issue #125 with a link to the Item-3 deliverables once produced.
- [ ] Decide whether to keep or close PR #145. Recommend keeping data, closing PR (or merging the data-only commit if that fits the team's archival pattern).
- [ ] Once both items 2 and 3 are documented in Overleaf and the data is on Drive, this issue can close.

---

## 6. Reference materials

- Branch [`judge_handoff_workshop_items_2_3`](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/tree/judge_handoff_workshop_items_2_3) — restored code + `extraction_validator/STATUS.md` (mirrors this issue).
- Workshop Validation Plan sheet (Drive: `1Z8f7MJuWIvjWvcWvPSe3_r3g32z5n9j7KcV9vNawh6I`) — canonical spec.
- 100-paper judge data: PR #145 (`extend_try_1/` outputs, `test_extend/` inputs).
- 100-paper combined extraction-with-text bundles + 3 rubric prompts: branch `rubric_2_and_3` (`extraction_validator/combined_extraction_with_original_text/`, `extraction_validator/prompts/`).
- September Increment Drive doc `1fTjkA02bn3rE3JA7tk3UqNv61N1X4mOebAp46FJe964` — chronological pre-Workshop history.

---

Closes #76 (broader "implement judge" scope; this ticket scopes the workshop close-out).
Related: #125 (error analysis for processed articles), #77 (compression LLM judge), #83 (lower-cost judge investigation).
