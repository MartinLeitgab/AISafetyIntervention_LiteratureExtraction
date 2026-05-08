# Judge Effort — Status & Handoff Document

**Date:** 2026-05-08
**Branch:** `judge_handoff_workshop_items_2_3`
**Goal:** Hand the judge effort to a new collaborator with complete context to close out **Workshop Validation Plan items 2 and 3** (judge analysis + error taxonomy on the 100-paper sample).

This document reconstructs the state of three judge sub-projects that have run since October 2025, lists known bugs, points to where each output dataset lives, and gives a concrete TODO list to finish the workshop deliverables.

---

## 1. Three judge sub-projects (history at a glance)

Three separate experiments have used the same `extraction_validator/judge.py` codebase:

| # | Sub-project | Window | Model | Sample | Output location | Status |
|---|---|---|---|---|---|---|
| A | **Failed-extraction recovery** — does the judge salvage extractions that errored out? | Oct–Nov 2025 | OpenAI `gpt-5-nano` (batch) | ~400 processable error sources (out of ~1,400 total errors flagged by Gleb in `processed.zip`) | NOT in any git branch — Mike's local disk; planned Overleaf write-up | **~60 records recovered / 400 processable (~15 %)**. Discord standup 2025-11-03. Documentation incomplete. |
| B | **Working-paper extension** — does the judge add missing nodes/edges to extractions that already succeeded? | Nov 26 2025+ | Anthropic `claude-sonnet-4-5` (batch, 50 % discount) | 100 successful extractions | PR [#145](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/pull/145) (`extraction_validator/extend_try_1/`, `extraction_validator/test_extend/`). Branch `anthropic_judge_test`, OPEN, marked "don't have to merge". | Done. 100 judge reports written. No quantitative summary yet. |
| C | **Judge-of-judge / multi-grader rubric calibration** — do independent meta-graders agree on whether the judge's fixes improve the extraction? | Dec 2025 – Jan 2026, in flight | Meta-graders: Claude Opus 4.5, Gemini 3 Pro, GPT-5.2 | Same 100 papers as B | Branch `rubric_2_and_3` (`combined_extraction_with_original_text/`, `prompts/`, `results.md`) — now copied into this branch | Opus + Gemini done; GPT-5.2 incomplete (model "lazy"). Per-file rubric JSONs only on Mike's disk. |

**Important nuance.** The user-stated framing of "100 error sources + 100 non-error sources" is approximate. The actual setup is: **100 successful papers (sub-projects B and C share this sample)** and **~400 processable error papers (sub-project A)**. Sub-project A was never sized to 100; it was sized to "all that aren't trivially missing source content."

---

## 2. Workshop Validation Plan — judge-related items

Reference: Drive sheet `1Z8f7MJuWIvjWvcWvPSe3_r3g32z5n9j7KcV9vNawh6I` "MINIMAL WORKSHOP VALIDATION PLAN".

### Item 2 — Judge Analysis on Extraction Quality (100 papers)

| Required deliverable | Status | Where it lives / what's missing |
|---|---|---|
| 100-paper judge-evaluated set | ✅ DONE | `extraction_validator/extend_try_1/` on PR #145 (Anthropic Sonnet 4.5 outputs); same 100 in `combined_extraction_with_original_text/` on `rubric_2_and_3` |
| Pre / post extraction scores per paper, per meta-grader | ✅ partial | `results.md` aggregates 3 meta-graders. **Per-file rubric JSONs (raw scores) NOT in any branch — only on Mike's disk.** Need to upload to Drive. |
| Inter-judge agreement (Fleiss' κ) between meta-graders | ❌ MISSING | Only mean ± std reported. Need to add κ and pairwise agreement matrix. |
| Judge-of-judge calibration | ⚠️ partial | 3 meta-graders provide raw signal; no spot-check vs human anchor done. |
| Overall error rate + breakdown by type (hallucinated nodes, hallucinated edges, missing nodes/edges, wrong edge type) | ❌ MISSING | `extraction_stats.py` only counts proposed `add_nodes` / `add_edges` aggregates; doesn't categorize. |
| Source-heterogeneity table (arXiv / blog / YouTube quality breakdown) | ⚠️ partial | Prompt example shows the table; not produced from real data. |
| Summary statistics for all 100 papers | ⚠️ partial | `results.md` has 3-row pre/post mean/std table only. |

### Item 3 — Error Taxonomy from Judge Data

| Required deliverable | Status | Where it lives / what's missing |
|---|---|---|
| Filter to error-flagged docs | ⚠️ partial | Sub-project A processed ~400 error sources; output not in git. |
| Sample 50 specific error instances | ❌ MISSING | Not done. |
| Manual categorization (hallucinated, missing, wrong-type, granularity-mismatch) | ❌ MISSING | Not done. |
| Distribution % across error types | ❌ MISSING | |
| 5–10 illustrative annotated examples | ❌ MISSING | |
| Error patterns by source type | ❌ MISSING | |
| Improvement recommendations | ❌ MISSING | |
| Recovery rate of failed extractions (related, from sub-project A) | ✅ partial | "60 / 400 processable, 0 / 1000 missing-source" per Discord 2025-11-03. Needs to be moved from Discord standup into the Overleaf and into a CSV in this branch. |

**Net:** Item 2 has raw materials but no formal stats. Item 3 is mostly OPEN; the 60-record recovery story is captured only in Discord and needs to be promoted into a concrete artifact (CSV + write-up).

---

## 3. Known bugs in `extraction_validator/judge.py` (current `main`, post PR #146)

| # | Severity | Where | Description |
|---|---|---|---|
| 1 | High | `judge.py:238` | **Stale model literal `gpt-4-vision-preview`.** This is a deprecated multimodal preview model and is not what was actually used. The OpenAI path is currently dead code; if anyone runs `--model_type=OpenAI` it will likely 404. The active path is `--model_type=Anthropic` (default), which uses `claude-sonnet-4-5` at `judge.py:258`. The saved `test_output/` predates this literal — actual model used there was `gpt-5-nano`. **Fix:** replace with `gpt-5-nano` (or current OpenAI default), or remove the OpenAI path entirely. |
| 2 | High | `judge.py:879` | **`concept_category="concept"` literal** for new concept add-nodes. Schema requires `risk \| problem analysis \| theoretical insight \| design rationale \| implementation mechanism \| validation evidence`. **Fix:** use `add_node.concept_category` from the judge's own response. |
| 3 | High | `judge.py:891` | **`aliases=[add_node.aliases]`** wraps a list inside a list. **Fix:** `aliases=add_node.aliases`. |
| 4 | High | `judge.py:893` | **`concept_category=add_node.concept_category` set on intervention nodes.** Interventions must have `concept_category=None`. **Fix:** set to `None` for the intervention branch. |
| 5 | **Catastrophic** | `judge.py:917` | **`break` at end of `for add_node_fix in proposed_fixes.add_nodes` loop.** Only the first add-node is ever processed; the rest are silently dropped. The two `if … break` lines inside the loop also exit the loop on any conflict (instead of `continue`-ing). **Fix:** remove the trailing `break` and turn the inner `break`s back into `continue`s. |
| 6 | Medium | `judge.py:_apply_fixes_to_graph` | **`proposed_fixes.merges` is parsed by the schema but never consumed.** Judge-recommended node merges are silently dropped. **Fix:** add a merge loop that retargets edges from the to-merge nodes onto the new merged node and deletes the originals. |

All 6 bugs survive in current `main` (HEAD `f22e504`, PR #146 merged 2025-12-09).

---

## 4. Where every artifact actually lives

### Code (in this branch, all under `extraction_validator/`)
- `judge.py` — main batch runner (post PR #146, with bugs 1-6 above outstanding)
- `schema.py` — Pydantic schemas with the `ValidatedDataOrOriginalOnError[T]` envelope (introduced in #146)
- `utilities.py` — Anthropic / OpenAI batch primitives
- `find_judge_able_sources.py` — filter to sources with both KG output and original text (used to prepare sub-project A inputs)
- `create_directoy_for_retry.py` — re-process error files from a previous batch run
- `extraction_stats.py` — aggregate `add_nodes` / `add_edges` counts across judge outputs (added in #146)
- **From `rubric_2_and_3`, restored in this branch:**
  - `prompts/extraction_evaluation_prompt.md` — rubric 1 (extraction-only meta-grader)
  - `prompts/judge_evaluation_prompt.md` — rubric 2 (judge-only meta-grader)
  - `prompts/judge_and_extraction_evaluation_prompt.md` — rubric 3 (combined; produces `pre_judge_score` / `post_judge_score`)
  - `combine_extraction_with_original_text.py` — bundle text + extraction for meta-grader
  - `combine_judge_with_original_text.py` — bundle text + judge output
  - `combine_judge_and_extraction_with_original_text.py` — bundle text + extraction + judge for rubric 3
  - `get_judge_improvement.py` — aggregate per-file `pre_judge_score` / `post_judge_score` into mean ± std
  - `results.md` — current mean / std pre vs post for the 3 meta-graders

### Data (NOT all in repo — needs consolidation to Drive)
| Asset | Location | Action |
|---|---|---|
| 100 successful-extraction sample (input) | PR #145 `extraction_validator/test_extend/` (300 files = 100 × 3) | Keep on PR #145 OR upload one tarball to Drive |
| 100 judge reports on those (Anthropic) | PR #145 `extraction_validator/extend_try_1/` (102 files) | Same |
| 100 combined extraction+text bundles for meta-grader | `rubric_2_and_3` `extraction_validator/combined_extraction_with_original_text/` | Same |
| Per-file rubric JSONs from Opus 4.5, Gemini 3 Pro, GPT-5.2 (raw `pre_judge_score` / `post_judge_score` per file) | **Mike's local disk only** | **Mike to upload to Drive** |
| Sub-project A failed-extraction inputs | Drive folder `1RPbETx21KMyEATtVOBt8LOClVd56eZ4x` (Gleb), `processed.zip` extraction_error/ | Already in Drive |
| Sub-project A judge outputs (60 recovered records) | **Mike's local disk only** | **Mike to upload to Drive** |
| 6-paper arxiv mini-test from `gpt-5-nano` | Repo `extraction_validator/test_output/` and `test_processed/` | Already in repo |

### Drive
- Project root: `https://drive.google.com/drive/u/0/folders/1u5r4OTasGRhcHRAoXFPMl6E_dr4abIrZ`
  - `1Z8f7MJu…wh6I` — Workshop Validation Plan (canonical spec for items 1-6)
  - `1fTjkA02…e964` — September Increment standup journal (chronological judge timeline)
- Gleb's processed-articles + errors folder: `https://drive.google.com/drive/u/0/folders/1RPbETx21KMyEATtVOBt8LOClVd56eZ4x`
  - `processed.zip` (6.26 GB) — full processed corpus + `extraction_error/`, `embeddings_error/`, `graph_error/`
  - `instructions.txt` — describes the layout; refers to Issue [#125](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/125)

### GitHub
- Open issues: [#76](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/76) Implement Extraction LLM Judge (still OPEN), [#77](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/77) Compression LLM Judge, [#83](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/83) Lower-cost judge model investigation, [#125](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/125) Error Analysis for Processed Articles
- Judge PR chronology:

  | PR | State | Branch | Author | Date | Note |
  |---|---|---|---|---|---|
  | #41 | merged | story2/staging | jeffreyparks | 2025-08-20 | Final extraction prompt candidate |
  | #101 | merged | Axel-judge-llm | axellabs | 2025-10-27 | KG Judge LLM prompt + skeleton (gpt-5-nano default) |
  | #135 | merged | find_judge_able_sources | mmulet | 2025-11-03 | Filter to judge-able sources |
  | #142 | merged | judge_2 | mmulet | 2025-11-21 | "Judge 2" — change_node_fields fix |
  | #143 | closed | anthropic_judge | mmulet | – | Superseded by #144 |
  | #144 | merged | anthropic_judge2 | mmulet | 2025-11-26 | Anthropic batch path; default flips to Anthropic; introduces stale `gpt-4-vision-preview` literal |
  | #145 | **OPEN** | anthropic_judge_test | mmulet | 2025-11-27 | Working-paper extension experiment data (NOT for merge) |
  | #146 | merged | refined_judge_prompt | mmulet | 2025-12-09 | Refined judge prompt; `.data` envelope schema; introduces bugs 3-5 above |

---

## 5. TODO list to close Workshop items 2 and 3

### Code cleanup (small, mechanical)
- [ ] **Bug 1.** Replace stale `gpt-4-vision-preview` model literal in `judge.py:238` with the actual OpenAI model that should run there (e.g. `gpt-5-nano` or current default), OR delete the OpenAI code path if Anthropic is now the only target.
- [ ] **Bug 2.** Fix `concept_category="concept"` literal at `judge.py:879` to use `add_node.concept_category`.
- [ ] **Bug 3.** Fix `aliases=[add_node.aliases]` at `judge.py:891` to `aliases=add_node.aliases`.
- [ ] **Bug 4.** Set `concept_category=None` for the intervention branch at `judge.py:893`.
- [ ] **Bug 5.** Remove the trailing `break` at the end of the `add_nodes` loop and convert the two early-exit `break` statements to `continue` (`judge.py:867`, `:870`, `:917`).
- [ ] **Bug 6.** Implement the `merges` consumption in `_apply_fixes_to_graph` (currently silently dropped).
- [ ] Add a per-bug pytest in `extraction_validator/tests/` that creates a synthetic judge response triggering each bug and asserts the post-fix `final_graph` is correct (per global rule "Every bug fix needs a test that exercises the fix").

### Workshop Item 2 — Judge Analysis (100 papers)
- [ ] **Mike to upload the per-file rubric JSONs** (Opus 4.5, Gemini 3 Pro, GPT-5.2) for all 100 papers to the Drive folder `1u5r4OTasGRhcHRAoXFPMl6E_dr4abIrZ`. Add a sub-folder `judge_workshop_item_2_rubric_outputs/`.
- [ ] **Finish the GPT-5.2 rubric run.** Mike's last update (1/8/26): GPT-5.2 was refusing the 100-paper task; subdivide into smaller batches and re-run.
- [ ] **Add Claude rubric re-run with the same final prompts** so all 3 meta-graders are on the same prompt version (per Mike's 1/8/26 plan).
- [ ] **Compute Fleiss' κ + pairwise agreement** between the 3 meta-graders on per-paper verdicts (or pre/post score buckets). Add to `results.md`.
- [ ] **Compute error-rate breakdown** per the Workshop spec (hallucinated nodes, hallucinated edges, missing nodes, missing edges, wrong edge types) by parsing the `validation_report.schema_check` / `referential_check` / `coverage` lists in the 100 judge outputs in `extend_try_1/`. Save as `extraction_validator/error_rate_breakdown.csv`.
- [ ] **Compute per-source-type table** (arXiv / blog / YouTube / lesswrong / alignmentforum / eaforum / arbital / aisafety.info / special_docs) showing mean pre/post score and error-rate breakdown. Save as `extraction_validator/by_source_type.csv`.
- [ ] **Write the Item-2 paragraph for the Overleaf** including the κ, error-rate breakdown, and source-heterogeneity table.

### Workshop Item 3 — Error Taxonomy
- [ ] **Move the 60-recovered records from Mike's local disk to the Drive folder.** Add a sub-folder `judge_workshop_item_3_error_recovery/` with subfolders `recovered/` (60 files) and `not_recoverable/` (the 340 attempted-but-failed processable records).
- [ ] **Persist the recovery summary as a CSV** in `extraction_validator/error_recovery_summary.csv` with columns `(source_type, paper_id, error_class, judge_attempt_status, recovered_node_count, recovered_edge_count, error_text_excerpt)`.
- [ ] **Sample 50 specific error instances** from the judge outputs of either sub-project A (failed-extraction recovery) or the BLOCKER/MAJOR-flagged subset of sub-project B (working-paper extension). Stratify by source type.
- [ ] **Manually categorize each into 6 categories**: (a) hallucinated nodes, (b) hallucinated edges, (c) missing nodes, (d) missing edges, (e) wrong edge type, (f) granularity mismatch. Save as `extraction_validator/error_taxonomy_50.csv`.
- [ ] **Compute distribution % across categories**, broken out by source type. Save as `extraction_validator/error_taxonomy_distribution.csv`.
- [ ] **Pick 5–10 illustrative examples** with text quotes + commentary, write up in `extraction_validator/error_taxonomy_examples.md`.
- [ ] **Write the Item-3 paragraph for the Overleaf** including the taxonomy table, recovery rate, and improvement recommendations.

### Closeout
- [ ] Mark Issue [#76](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/76) for closure once both items are done.
- [ ] Update Issue [#125](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/125) (Error Analysis for Processed Articles) with a link to the Item-3 deliverables.
- [ ] Decide whether to keep or close PR [#145](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/pull/145) (Anthropic working-paper test). Recommend keeping data, closing PR.

---

## 6. Quick map for the next person

If you are picking this up:

1. Start with the Workshop Validation Plan sheet on Drive (`1Z8f7MJu…wh6I`) — that is the canonical spec.
2. Read `inputs/discord_messages_judge.txt` if Martin shares it (private, in `.gitignore`) for the chronological history of every decision.
3. The 100-paper judge data you need for Item 2 is on PR [#145](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/pull/145) (`extend_try_1/` for outputs, `test_extend/` for inputs).
4. The combined-with-original-text bundles you need for the meta-grader rubrics are in this branch under `extraction_validator/combined_extraction_with_original_text/` (copied here from `rubric_2_and_3`). The 3 prompt files are in `extraction_validator/prompts/`.
5. The bugs in §3 above must be fixed before any new judge run is trusted; tests under `extraction_validator/tests/` should be added at the same time.
6. The 60-record failed-extraction recovery from sub-project A lives only on Mike's disk — coordinate with him to retrieve.

Questions? Read the September Increment Drive doc (`1fTjkA02…e964`) chronologically from the bottom (oldest first) for the pre-Workshop history; read the `inputs/discord_messages_judge.txt` for Nov–Jan context.
