# Judge — Paper Findings and Plan

**Created:** 2026-05-19
**Owner:** Martin (Sai executes remaining work, asks Mike-via-Martin for data on local disk)
**Companion to:** GitHub issue #147 (open), `STATUS.md` (in this directory)
**Scope:** Workshop "Must-Have To-Do List" items 2 (judge validation on good extractions) and 3 (error taxonomy + judge recovery on failed extractions). Both close-out items for the AAAI submission per Gleb's Overleaf §1.

---

## 1. Asset consolidation (3 trees → one inventory)

Three on-disk locations and several branches hold pieces of the judge work. Map:

| Location | What it holds | Status |
|---|---|---|
| `martin/main` (this branch, main worktree) | nothing judge-related; phase 2 corpus analysis only | n/a |
| Branch `rubric_2_and_3` (=`_rubric23` worktree) | **CANONICAL grader prompts + 100 input bundles**: prompts/{extraction,judge,judge_and_extraction}_evaluation_prompt.md, combined_extraction_with_original_text/ (100 files), results.md naming graders | latest commit `ac1bfb8`, no PR open |
| Branch `judge_handoff_workshop_items_2_3` (=`_judgehandoff` worktree) | rubric23 + PR #146 refined judge prompt + `ValidatedDataOrOriginalOnError` schema envelope + `STATUS.md` + this document | latest `7029790`, no PR open (issue #147 is the discussion thread) |
| Branch `anthropic_judge_test` (PR #145, OPEN) | `test_extend/` = 100 per-paper input dirs (extraction + raw_response + summary); `extend_try_1/` = 100 judge-output bundles — committed-to-git mirror of Mike's pre-archive run | PR #145 stays open as evidence-of-run |
| `Final-archive-from-Mike/` (outside repo, 14.8 MB) | **OUTPUT DATA**: judge-run bundles (`extend_try_with_extration_and_judge_and_original_text/`, 101 files: 95 evals + 101 bundles) + 3 grader evaluation sets (Opus 4.5 / Gemini 3 Pro / GPT-5.1, 100 each) + `summary_results.json` from Opus | superset of what's in `anthropic_judge_test`; not in git |
| Gleb's Drive folder `1RPbETx21K...eZ4x` | Original extraction errors (~1000 records) — the INPUT for any failed-extraction recovery run | external |
| **Mike's local disk only** (not shared) | The ~60 records the judge actually recovered when run on the ~400 processable errors. **NOT in any branch, archive, or Drive folder.** | ask Mike |
| Workshop Drive `1Z8f7MJu...wh6I` | 6-item checklist; items 2+3 are this scope | external |
| Gleb's Overleaf `6891855beb...ff7f6` | Paper draft naming the scope: §1 "Judge validation framework + judge-of-judge calibration (n=100), Mike" + "Error taxonomy from judge data (n=50 instances), Mike" + "Judge reliability: κ=X.XX, error rate breakdown" | external |

**De-duplication finding:** `rubric_2_and_3` and `_judgehandoff` overlap heavily; `_judgehandoff` is the superset (rubric23 + PR #146 + new docs). Sai should work on `_judgehandoff` and ignore `_rubric23` except for `combined_extraction_with_original_text/` (only-in-rubric23 dataset) and `results.md` (grader naming). Mike's archive is the authoritative outputs.

**Mike's archive vs PR #145 `extend_try_1`:** Same 100-paper roster, different file sizes (~10% diff per file). Mike's archive was re-generated AFTER PR #146 (refined judge prompt). Use Mike's archive as the canonical good-extractions judge output for paper analysis; PR #145's `extend_try_1` is a stale earlier run kept for git lineage.

---

## 2. Workshop item 2 — Judge validation on 100 good extractions

### Status: **MOSTLY DONE**, ready for paper-section drafting

**What exists:**
- 100 input bundles (original_text + extraction): `_rubric23/extraction_validator/combined_extraction_with_original_text/`
- 100 judge-output bundles (original + extraction + judge output): `Final-archive-from-Mike/extend_try_with_extration_and_judge_and_original_text/`
- 3 **meta-grader** evaluations (judge-of-the-judge layer; the judge itself is **Sonnet 4.5** — these 3 models are independent meta-graders scoring the judge's output):
  - **Opus 4.5** (`test_extend_all_evaluation_opus_4_5/`) — 100 evals + verbose summary_results.json
  - **Gemini 3 Pro** (`test_extend_all_evaluation_gemini_pro_3/`) — 100 evals, concise; **post-judge σ=1.80 is suspicious — see Gemini inspection task in §6**
  - **GPT-5.1** (identity confirmed by `_rubric23/extraction_validator/results.md`) — 95 of 100 evals in `extend_try_with_extration_and_judge_and_original_text/*_evaluation.json`. **The 5 missing files are missing GPT-5.1 *meta-grader* evals, not missing judge outputs.** Per issue #147 §1: the GPT-5.1 meta-grader was refusing the 100-paper task as of 2026-01-08 ("model lazy"). Fix: retry GPT-5.1 with smaller per-batch sizes; **substitution to another model is forbidden** per locked decision §8 (model-version-drift confound).

### Grader pool — identity of the "unlabeled third grader"

**It is GPT-5.1.** Source: `_rubric23/extraction_validator/results.md`, committed by Mike in `5a31be5 "rubric 2 and 3"` 2026-01-05:

```
# Claud Opus 4.5
Pre scores - Mean: 69.65, Std: 18.01
Post scores - Mean: 73.13, Std: 18.21
# Gemini 3 Pro
Pre scores - Mean: 62.69, Std: 22.58
Post scores - Mean: 95.77, Std: 1.80
# GPT5.1
Pre scores - Mean: 66.96, Std: 16.19
Post scores - Mean: 80.18, Std: 10.21
```

The 95 `_evaluation.json` files in `extend_try_with_extration_and_judge_and_original_text/` are the GPT-5.1 grader output. The 5 missing files are paper-quality blockers (retry GPT-5.1 in smaller batches ~$0.10, see §5 — substitution to another model is forbidden per locked decision §8).

### Headline paper-ready findings (cross-grader agreement and divergence)

| Metric | Opus 4.5 (n=100) | Gemini 3 Pro (n=100) | GPT-5.1 (n=95) |
|---|---|---|---|
| Pre-judge mean | 69.65 ± 18.01 | 62.69 ± 22.58 | 66.96 ± 16.19 |
| Post-judge mean | 73.13 ± 18.21 | **95.77 ± 1.80** | 80.18 ± 10.21 |
| Δ (judge effect) | +3.48 | **+33.08** | +13.22 |

**Cross-grader divergence is the headline.** All three graders agree pre-judge extraction is in the "good" 60–70 band. They diverge sharply on whether the judge's fixes help:
- **Opus is conservative** (+3.48): judge fixes barely move the needle.
- **GPT-5.1 is moderate** (+13.22): meaningful improvement.
- **Gemini is wildly generous post-judge** (+33.08, σ collapses to 1.80): grader-bias signal, not extraction-quality signal. Likely Gemini conflates "schema now valid" with "content now correct."

**For the paper:** report all three graders side by side. The divergence is itself the calibration finding. Recommendation: emphasize Opus + GPT-5.1 (median = +8.4 pp judge effect) and flag Gemini as a known generous-grader outlier; do not average across all three without caveat.

### Opus 4.5 qualitative findings (paper-ready, from `summary_results.json`)

- **Systematic schema issue (100% of files):** missing rationale fields (`node_rationale`, `edge_rationale`, `edge_confidence_rationale`, `intervention_lifecycle_rationale`, `intervention_maturity_rationale`).
- **Structural issues:** over-inference of interventions on theoretical/question posts (~30%), causality reversal in problem-analysis edges (~15%), missing validation evidence when source is empirical (~20%).
- **Judge over-pruning:** avg 25% edge reduction after judge's proposed_fixes applied — judge removes legitimate edges in pursuit of schema compliance.
- **Judge placeholder pollution:** judge-added nodes often have auto-generated placeholder descriptions ("Auto-generated node based on validation") rather than evidence-grounded text.
- **Source-type AI-safety relevance:** alignmentforum 85.3, arbital 88.0, arxiv 82.1, lesswrong 80.2, aisafety.info ~80, eaforum 71.4, special_docs 67.3, blogs ~76 (computed from `by_source_type` table in `summary_results.json`).

**Paper figure suggestion:** Box-plot pre/post per grader, stratified by source type. Use Opus's `by_source_type` table.

### What's left for item 2 — NOT blocked, ready for Sai

| Task | Effort | Owner |
|---|---|---|
| Backfill the 5 missing GPT-5.1 *meta-grader* evals — retry GPT-5.1 in smaller batches; substitution forbidden (model-version-drift confound) | ~$0.10, ~30 min | Sai |
| **Inspect Gemini high-variance signal** (Δ=+33pp, σ=1.80 post) — random 10-eval audit, decide paper-usable vs outlier | ~2 hr manual | Sai |
| Compute Fleiss' κ + pairwise agreement across 3 meta-graders × 100 papers × 4 verdict-buckets | local Python, no LLM | Sai |
| Produce `error_rate_breakdown.csv` + `by_source_type.csv` | local Python | Sai |
| Draft Item 2 Overleaf paragraph (§1 L141) | Sai (Martin edits) | Sai |

**No re-run of the 100-good-extraction judge needed.** The Mike-archive output is the canonical dataset. **The judge itself = Sonnet 4.5 (already run).** The 3 meta-graders score the judge's output — those are the items above.

---

## 3. Workshop item 3 — Error taxonomy + judge recovery on failed extractions

### Status: **UNBLOCKED 2026-05-20** — Mike supplied `judge_recovery_bundle/` on Drive

Mike shared the complete recovery dataset as `judge_recovery_bundle/` (Drive). Inventory:

| Path inside `judge_recovery_bundle/` | Files | What it is |
|---|---|---|
| `data/extraction_error_recoverable_info/` | 441 per-paper dirs | The ~400 processable failed extractions (input set after `find_judge_able_sources.py` filter) |
| `data/recovered_errors/` | 443 files (441 attempts + `summary.json` + `errors.json`) | Full judge attempt log including the ~376 non-recoveries |
| `data/recovered_errors_graph/` | **65 files** | "~60 recovered" set — judge runs that produced a valid `final_graph`. Empirical recovery rate **65/441 = 14.7%**. |
| `data/recovered_errors/summary.json` | — | total_tokens=4.06M, prompt=2.14M, completion=1.93M, 13 technical errors out of 441 |
| `data/recovered_errors/errors.json` | — | 13 technical-failure cases (503 server errors, Pydantic schema-validation errors) — error-taxonomy seeds |
| `code/` | 9 files | Mike's pre-PR-#146 `judge.py` (SHA `5389d5a5...` = same as `origin/rubric_2_and_3:judge.py`) + same prompts/scripts already restored on this branch — no new code |

**Source-type distributions** (informative for paper table):
- **Inputs (441):** arbital 124, eaforum 101, lesswrong 75, blogs 69, alignmentforum 47, aisafety.info 10, special_docs 9, agentmodels 4, arxiv 2
- **Recovered (65):** alignmentforum 16, eaforum 15, arxiv 14, blogs 11, lesswrong 6, arbital 2, agisf 1 — note: 65-set uses `ard_file_source` field naming, not failure-bucket prefix. Sai must reconcile naming when building the per-source-type recovery rate table.

### Critical caveat — bug contamination of recovered graphs

Mike's `judge.py` for the 65 recoveries is the **pre-PR-#146 version with all 6 bugs B1–B6 present**. Impact:

- **Recovery rate 65/441 = 14.7% is empirically valid** — bugs don't affect "did a graph get produced".
- **Recovered graph *content* is partially bug-contaminated:**
  - **Bug 5** (trailing `break` in `add_nodes` loop) — only the first add-node was applied per record; remaining add-nodes silently dropped. Several of the 65 `final_graph`s are missing nodes the judge proposed.
  - **Bug 6** (`merges` ignored) — judge-proposed merges never executed in any of the 65 `final_graph`s.
- **For the paper:** recovery-rate claim is solid. Recovery-quality claim (validated by 3-grader judge-of-judge on n=30 sample) reports on Mike's as-produced bug-contaminated graphs — which is what the team actually has. Sai's bug fixes (Phase B) are needed only for a future re-run, not for closing out Items 2+3.

### Bonus datasets Mike has but did NOT include in `judge_recovery_bundle/`

Per the bundle's README, these exist on Mike's local disk but are excluded from the bundle. **None are critical for Workshop minimum acceptance** — every Item 2/3 deliverable in the Workshop spec is satisfiable from `judge_recovery_bundle/` alone. Document the omission in the Item 3 Overleaf paragraph as deferred-future-work:

1. `extraction_error_recoverable_info_graph_error/` (179 MB, 91 dirs) — analogous judge-able set from `graph_error` failure stage. Per-failure-stage recovery rate is not in the Workshop spec.
2. `recovered_errors_graph_new_prompt{,2..7,9,A..E}/` (~92 files each, ~12 prompt-iteration variants) — judge-prompt sensitivity experiments. Single-prompt recovery rate is what the spec asks for.
3. `processed_ard/extraction_error/` (1,667 unfiltered failures) — full failure denominator. The `find_judge_able_sources.py` filter (441/1,667 = 26% judge-able) already covers the spec's "filter to error-flagged docs" requirement.

**Provenance of the 400/60 numbers.** From Discord export `inputs/discord_messages_judge.txt` line 334:

> "@Mike, @soma and @Axel continued judge work and Mike ran on errors from global graph extraction — 1k errors due to missing content in ARD records (not recoverable), about **400 records may remain processable — judge recovered 60 records** (judge has difficulties producing local graph if no prior work done even if had original source). Mike will work to document error analysis on Overleaf."

**Decomposition:**
- ~1,000 total extraction errors observed during global graph extraction
- ~600 are unrecoverable (missing ARD source content — nothing for the judge to work from)
- ~400 are processable (had source content; failed for other reasons — finish_reason:length, schema-reject, JSON-parse error)
- Judge run produced **~60 successfully recovered records** = ~15% recovery rate on processable errors

### To Martin's question — "if those are recovered by the judge, then those will not be in Gleb's drive zip"

**Correct.** Gleb's Drive folder `1RPbETx21K...eZ4x` holds the **input** to recovery (the original extraction errors). The **output** (~60 recovered records + per-record fail/recover trace + error categorization) is **only on Mike's local disk** — not in any branch, not in any PR, not in Mike's Final-archive. The Discord log says "Mike will work to document error analysis on Overleaf" — that documentation has not appeared in the Overleaf draft yet (only the placeholder line 211 "Error taxonomy from judge data (n=50 instances), Mike").

### Asks from Mike — RESOLVED 2026-05-20

All asks satisfied by `judge_recovery_bundle/`:
1. ✓ Input "processable" errors set → `data/extraction_error_recoverable_info/` (441 dirs)
2. ✓ Recovered records (per-record judge-fixed graph) → `data/recovered_errors_graph/` (65 files)
3. ✓ Non-recovered processable errors → `data/recovered_errors/` minus `data/recovered_errors_graph/` (~376 records)
4. ✗ Mike did not include separate analysis notes — only the raw recovery output and `summary.json`. Error categorization is Sai's hand-classification task per Workshop Item 3 spec.

### Building the error taxonomy (n=50 instances per Gleb's Overleaf §211)

The taxonomy categorizes WHY each error occurred. Categories observable from the single `errors.json` example committed (`arxiv__a5851beb3e8d80c0bded11fa6b1f8fff`):
- `finish_reason: length` — gpt-5-nano hit the 16,000-output-token ceiling without producing valid JSON
- `JSON parse error` — model emitted malformed JSON
- `schema-reject` — Pydantic validation failed (e.g., concept node with `intervention_lifecycle` set)
- `empty content` — ARD record lacked extractable text

n=50 means hand-categorizing 50 of the ~400 processable errors. This is a Sai task; no LLM tokens required (manual review).

### What's left for item 3 — UNBLOCKED (Mike data on Drive)

| Task | Effort | Owner |
|---|---|---|
| ~~Pull Gleb's Drive zip + identify ~400 processable errors~~ | — | NOT NEEDED — Mike's `data/extraction_error_recoverable_info/` already filtered |
| ~~Re-run judge on 400 processable~~ | — | NOT NEEDED — Mike's `data/recovered_errors/` is the complete attempt log |
| Hand-categorize 50 errors into taxonomy buckets (use `data/recovered_errors/errors.json` + spot-check non-recovered records in `data/recovered_errors/`) | ~3 hr | Sai |
| Run 3-grader judge-of-judge on n=30 random `data/recovered_errors_graph/` records (same prompts as item 2) | ~$5–10 (3-vendor pool) | Sai |
| Compute κ on recovery-quality verdicts; build per-source-type recovery rate table | local Python | Sai |
| Draft Overleaf paragraph for item 3 (taxonomy + recovery rate + κ + recommendations + bonus-data-deferred note) | drafting | Sai |

---

## 4. Known bugs (from issue #147, blocking paper-quality use)

These are real bugs in current `extraction_validator/judge.py` HEAD that affect any new run. Sai must fix before running on the failed-extraction set.

| # | Location | Bug | Fix |
|---|---|---|---|
| B1 | `judge.py:238` | Stale literal `"model": "gpt-4-vision-preview"` (vision model — doesn't make sense for text judging) | Replace with `gpt-5-nano-2025-08-07` or whatever the active OpenAI judge model is; or delete the OpenAI branch entirely if Anthropic path is canonical |
| B2 | `judge.py:879` | `concept_category="concept"` literal — should be one of the 6 valid categories (risk / problem_analysis / theoretical_insight / design_rationale / implementation_mechanism / validation_evidence) | Either preserve the original concept_category from the source node or let the model fill it; never hardcode "concept" |
| B3 | `judge.py` `_apply_fixes_to_graph` | `proposed_fixes.merges` field is read but never applied — silent drop of judge's merge suggestions | Either implement merge application or remove the merges field from the schema |
| B4 | PR #146 regression | Judge output schema changed to `ValidatedDataOrOriginalOnError[T]` — downstream consumers may not parse it | Audit all judge-output readers; update or document |
| B5 | PR #146 regression | Errors now returned as raw response strings, not parsed | Decide if this is intentional; if not, parse before returning |
| B6 | PR #146 regression | One more from #147 — read issue body for details | Triage |

**Recommendation:** before running judge on the 400 failed extractions, Sai should land bug fixes for B1, B2, B3 minimum on a new branch (e.g., `judge_bugfixes_pre_workshop_3`) and PR against main. B4–B6 may not block.

---

## 5. Cost estimate + recommended model decisions

### Models in play (current public list, batch API 50% off Anthropic/OpenAI)

| Role | Recommended model | Reason | Per-call est. |
|---|---|---|---|
| Judge (extraction fixer) | **Sonnet 4.5** (`claude-sonnet-4-5`, per `judge.py:258`) | Same as Mike's Item 2 baseline; no upgrade to keep judge output methodology constant | $3 in / $15 out per MTok |
| Grader 1 of 3 | **Opus 4.5** (same model Mike used for Item 2; keeps grader methodology constant) | Frontier vendor; verbose evals already proven informative; methodology-locked to Item 2 baseline | $15 in / $75 out per MTok |
| Grader 2 of 3 | **Gemini 3 Pro** (already Mike's choice) | Vendor diversity; known generous-grader caveat | ~$1.25 in / ~$10 out per MTok (est) |
| Grader 3 of 3 | **GPT-5.1** | Vendor diversity; locked to Mike's Item 2 baseline (do not substitute GPT-5.2 — model-version-drift confound) | ~$1.25 in / ~$10 out per MTok (est) |

**Gemini API access:** Sai needs a Google AI Studio (or Vertex) API key. Cheapest path: Google AI Studio key on Martin's account, shared via the same private channel as the Anthropic key. If Gemini is too operationally heavy to provision, drop Gemini and document the decision (the cross-vendor calibration narrows to Opus + GPT-5.x — still defensible with a noted caveat).

### Cost table — what's left to do

| Task | N | Tokens in/out per call | Model | List price | Batch subtotal (-50%) |
|---|---|---|---|---|---|
| Backfill 5 missing GPT-5.1 (retry GPT-5.1 in smaller batches — substitution forbidden, see locked decision) | 5 | 16.5k / 1.3k | GPT-5.1 | ~$1.25 / ~$10 | **~$0.10** |
| ~~Judge on 100 failed~~ | — | — | — | — | NOT NEEDED — Mike's `judge_recovery_bundle/` includes the full attempt log |
| 3-grader judge-of-judge on n=30 random recovered records (`judge_recovery_bundle/data/recovered_errors_graph/`) | 30 × 3 | 16.5k / 1.3k each | Opus 4.5 + Gemini 3 Pro + GPT-5.1 | varies | **~$5–10** batch |
| **Total — Items 2+3 close-out (Mike's data on Drive)** | | | | | **~$10** |

**Recommendation:** budget **$5–10** total (down from earlier $10–14 — Mike's `judge_recovery_bundle/` eliminated the conditional 100-failed-judge re-run). Wall-clock with batch API ~1 day. All 4 models locked to Mike's Item 2 baseline (Judge = Sonnet 4.5; Meta-graders = Opus 4.5 + Gemini 3 Pro + GPT-5.1) per §8 #1.

### Re-run decision matrix

| Set | Action | Why |
|---|---|---|
| 100 good extractions — judge run | **No re-run.** Use Mike's archive. | Already done; PR #145 has a stale git mirror; Mike's archive is post-PR #146. |
| 100 good extractions — Opus 4.5 grader | **No re-run.** | Mike already did this with full summary_results.json. |
| 100 good extractions — Gemini 3 Pro grader | **No re-run.** | Mike already did this. |
| 100 good extractions — GPT-5.1 grader | **Backfill 5 missing only (retry GPT-5.1; no model substitution).** | 95 of 100 done. |
| ~400 failed extractions — judge run | **No re-run.** Use `judge_recovery_bundle/data/recovered_errors/`. | Mike supplied 441 attempts on Drive 2026-05-20. |
| 65 recovered — judge-of-judge (n=30 sample) | **NEW run required.** | Never done; needed for item 3 close-out. |

---

## 6. Sai task list (concrete, ordered, with file pointers)

### Phase A — Setup (do first)
1. **Set up env.** Clone repo, check out `judge_handoff_workshop_items_2_3` branch, install via `uv sync`. API keys: OpenAI + Anthropic per `STATUS.md` §"API Keys"; Gemini 3 Pro key (Google AI Studio) already sent by Martin on 2026-05-19.
2. **Read** `STATUS.md`, this doc end-to-end, issue #147 thread.
3. **Download two Drive folders** into a local gitignored area:
   - `Final-archive-from-Mike/` (Item 2 data — 100 good extractions + 3 meta-grader runs)
   - `judge_recovery_bundle/` (Item 3 data — 441 inputs + 65 recovered + summary/errors JSONs)

### Phase B — Bug fixes (must precede any FUTURE judge run; not blocking Items 2/3 close-out per §3 caveat)
4. **Fix all 6 bugs B1–B6** in `judge.py` on a new branch `judge_bugfixes_pre_workshop_3`. Each bug fix needs an integration test under `extraction_validator/tests/` that exercises the fix. Bug 5 is catastrophic (silently drops all add-nodes after the first). Open PR against `main`. **Note:** these fixes do NOT need to land before the Item 2/3 paper analysis — Mike's existing data is what the paper reports on. Bug fixes are for future runs only.

### Phase C — Item 2 close-out (NOT blocked; data in `Final-archive-from-Mike/`)
5. **Backfill the 5 missing GPT-5.1 meta-grader evals** on the 100 good extractions. Per issue #147 §1: GPT-5.1 was refusing the 100-paper task as of 2026-01-08 ("model lazy"). **Substitution to another model is forbidden** per locked decision §8 #1. Retry GPT-5.1 with smaller per-batch sizes (e.g., 5 papers per batch). If GPT-5.1 still refuses after 3 retry attempts: document the 5 as unrecoverable, drop from κ computation, footnote n=95 for the GPT-5.1 column. Update `summary_results.json`.
6. **Inspect Gemini 3 Pro high-variance signal.** Gemini posts Pre 62.69±22.58 → Post 95.77±**1.80** with Δ=+33.08. σ collapse + ceiling-band post-score looks like grader bias. Random-sample 10 of the 100 Gemini eval JSONs, manually compare cited strengths/weaknesses against actual judge output. Decide: paper-usable, or document as outlier and exclude from cross-grader average. Save findings to `extraction_validator/gemini_grader_variance_inspection.md`. **Paper-quality gate** — cross-vendor κ is suspect without it.
7. **Compute Fleiss' κ + pairwise agreement** on the 4-bucket verdict (excellent / good / mixed / poor) across the 3 meta-graders × 100 papers. Python only. Append table to `results.md`. **Headline calibration finding for item 2.**
8. **Parse `validation_report.{schema_check, referential_check, coverage}`** lists from `Final-archive-from-Mike/extend_try_with_extration_and_judge_and_original_text/` bundles. Produce `extraction_validator/error_rate_breakdown.csv` (hallucinated/missing nodes/edges, wrong edge types).
9. **Produce `extraction_validator/by_source_type.csv`** (mean pre/post + error rates per source). Use Opus's `by_source_type` table from `summary_results.json` as starting structure.
10. **Draft the Item 2 Overleaf paragraph** (§1 L141 placeholder). Include κ, pairwise agreement table, error-rate breakdown, source-heterogeneity table, Gemini-variance caveat. Martin edits.

### Phase D — Item 3 close-out (UNBLOCKED 2026-05-20; data in `judge_recovery_bundle/`)
11. **Persist recovery summary** as `extraction_validator/error_recovery_summary.csv` with columns `(source_type, paper_id, error_class, judge_attempt_status, recovered_node_count, recovered_edge_count, error_text_excerpt)`. Built directly from `judge_recovery_bundle/data/recovered_errors/` (status="recovered" if paper_id appears in `data/recovered_errors_graph/`, else "failed"). Reconcile the source-type naming convention difference between input dirs and recovered files (see §3 source-type distributions note).
12. **Sample 50 error instances** stratified by source type from the ~376 non-recovered records in `judge_recovery_bundle/data/recovered_errors/`. Hand-categorize into 6 buckets per Workshop spec: hallucinated nodes / hallucinated edges / missing nodes / missing edges / wrong edge type / granularity mismatch. Save as `extraction_validator/error_taxonomy_50.csv`. Seed buckets from the 13 cases already characterized in `data/recovered_errors/errors.json` (503 errors, schema validation errors).
13. **Compute distribution % across categories × source types.** Save as `extraction_validator/error_taxonomy_distribution.csv`.
14. **Pick 5–10 illustrative examples** with text quotes + commentary. Save as `extraction_validator/error_taxonomy_examples.md`.
15. **Run 3-grader judge-of-judge on n=30 random `data/recovered_errors_graph/` records** (Opus 4.5 + Gemini 3 Pro + GPT-5.1 per locked decision §8 #1). Same prompts as item 2 (`prompts/judge_and_extraction_evaluation_prompt.md`). ~$5–10 batch. Validates recovery *quality* on Mike's as-produced (bug-contaminated per §3) graphs — which is what the paper reports.
16. **Compute κ on recovery quality verdicts** (same calc as step 7, smaller n). Save to `extraction_validator/recovery_quality_kappa.csv`.
17. **Draft the Item 3 Overleaf paragraph** (§1 L211 + L297 placeholders). Include taxonomy table, recovery rate by error category (65/441 = 14.7% overall), κ on recovery quality, improvement recommendations, and a "deferred future work" note covering the 3 bonus datasets Mike has but didn't bundle (per §3). Martin edits.

### Phase E — Closeout
18. **Update this doc** (`Judge_Paper_Findings_and_Plan.md`) with all delivered numbers (κ, error-bucket distribution, recovery rate per category).
19. **Commit all CSVs + Markdown + analysis scripts** to `judge_handoff_workshop_items_2_3`. Drive folders stay on Drive; only pointers + derived analysis go into git.
20. **Open PR** consolidating everything. Link issue #147 + issue #125 (error analysis).
21. **Decision on PR #145** (anthropic_judge_test): keep open as data lineage. Note as historical lineage in PR description.
22. **Close issue #147** once Items 2 + 3 paragraphs are in Overleaf and data is on Drive.

---

## 7. Paper-section drafting notes (for Martin)

When writing the paper section on the judge:

- **Frame as a faithfulness check, not a quality claim.** The judge's value proposition is *"validate the extraction structure against the source and propose targeted fixes"*, not *"make the extraction objectively better"* — that's what the Opus-Gemini divergence captures.
- **Use Opus + GPT-5.1 as primary; flag Gemini as outlier.** Gemini's σ=1.80 post-judge is suspicious; report it but caveat.
- **Don't average across the 3 graders** without showing the per-grader breakdown — readers need to see the divergence.
- **The 25% edge-pruning finding is novel and publishable.** Frame as: "schema-strict judges trade edge coverage for schema validity; this is a tunable tradeoff in the validator design."
- **Recovery rate ~15% (60/400) is honest reporting** — frame as "lower bound on recoverability with current judge prompt; suggests path for prompt refinement rather than indictment of approach."
- **Error taxonomy is a methodology contribution** — most KG-extraction papers don't categorize failure modes. Cite this as future-work substrate.

---

## 8. Decisions (locked 2026-05-19)

1. **All models LOCKED to Mike's Item 2 baseline — no version upgrades.** Same models must be used for Item 3 work to avoid model-version-drift confounds between Items 2 and 3. The locked set:
   - **Judge: `claude-sonnet-4-5`** (Sonnet 4.5, per `judge.py:258`)
   - **Meta-grader 1: Opus 4.5** (per Mike's directory name + `results.md`; Opus 4.7 not yet released during Mike's Dec 2025 – Jan 2026 runs)
   - **Meta-grader 2: Gemini 3 Pro**
   - **Meta-grader 3: GPT-5.1** (do not substitute GPT-5.2 — would introduce within-column drift)

   Cost ~$10–14 (lower than initial $15–20 estimate because n=30 sample on the meta-grader run, not n=100).
2. **Gemini API: Martin provided the API key to Sai on 2026-05-19** via private channel.
2. **Gemini API: Martin provided the API key to Sai on 2026-05-19** via private channel.
3. **Bug fixes: Sai fixes all 6 bugs (B1–B6) before any new judge run.** Each fix gets an integration test under `extraction_validator/tests/`. Land bug-fix PR against `main` before running on failed extractions.
4. **Mike data access: RESOLVED 2026-05-20.** Mike supplied `judge_recovery_bundle/` on Drive — 441 inputs + 65 recovered + full attempt log. Phase D is UNBLOCKED. No re-run on failed extractions needed.
5. **Paper-section ownership: Sai drafts both Item 2 and Item 3 paragraphs in Overleaf; Martin edits.** Sai uses the headline-finding framing in §7 of this doc.

---

## 9. Reference materials

- Issue #147 (open) — task description + bug list as comments
- PR #145 (open) — `extend_try_1` good-extraction judge run, pre-PR-#146
- PR #146 (merged) — refined judge prompt + `ValidatedDataOrOriginalOnError` envelope
- `_rubric23/extraction_validator/results.md` — grader identity (Opus 4.5 / Gemini 3 Pro / GPT-5.1) + score statistics
- `_rubric23/extraction_validator/combined_extraction_with_original_text/` — canonical 100 input bundles
- `Final-archive-from-Mike/test_extend_all_evaluation_opus_4_5/summary_results.json` — paper-ready Opus 4.5 qualitative findings
- `Final-archive-from-Mike/prompts/{extraction,judge,judge_and_extraction}_evaluation_prompt.md` — canonical grader prompts (must reuse for item 3 to keep results comparable)
- Workshop Drive `1Z8f7MJu...wh6I` — 6-item checklist
- Gleb's Overleaf §1 + §211 — paper-plan placeholders for items 2+3
- Discord export (not committed) — line 334 = canonical 400/60 statement
