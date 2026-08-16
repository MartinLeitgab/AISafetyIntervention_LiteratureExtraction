# Paper A — open items. THE canonical list.

🔴 **This is the single place open work is tracked. Work it down here; add new items here;
do not start a parallel list.** The filename is deliberately undated — do not create a
successor file. Everything else in `paper/` is either evidence (reviews, receipts) or a
record of decisions already taken.

**One exception, by design:** authorship, the compute-donor gate, and co-author
coordination live in `paper/NEXT_STEPS_PRIVATE.md`, which is gitignored and must stay that
way. Nothing else belongs there, and nothing there is a study. This file carries a stub for
each so the two never drift apart.

**Written 2026-08-15**, renamed from `STUDY_LIST_2026-08-15.md` and made canonical the same
day. Every entry is scoped to the point where it could be started the same day.

**Amended 2026-08-15 PM** after an independent three-model review round (Claude Opus 5,
GPT-5.6 Sol, Gemini 3.1 Pro; conference and workshop bars; reviews and usage receipts in
`paper/reviews_2026-08-15/`, script `paper/review_multi_model.py`). Each model received
only the compiled PDF, the scoring guidance and a short prompt — no repo context. Verdicts:
conference 3 / 2 / 3 (all reject or borderline reject), workshop 4 / 3 / 5 (split).
Agreement counts below are over the three models, and priorities are revised where that
round disagreed with the earlier two internal reviews.

Two studies from the reviewer backlog are **done** and are not repeated here: the
extraction cost measurement (issue #152, PR #154) and the stage-separability probe
(issue #153). Both were Class B and cost nothing.

## Where everything lives — read this first in a fresh session

| What | Where | Note |
|---|---|---|
| The manuscript | `../AISafetyIntervention_PaperA_shared/paperA_altstyle.tex` | **Canonical.** Overleaf syncs to the same branch, so `git pull --ff-only` **before you Read or Edit**, not just before you commit, and push the same session. The `paper/*.tex` copies in this repo are stale duplicates. Full rule in the project `CLAUDE.md` |
| Compiled PDF the external round reviewed | `C:\Users\malei\Downloads\AISafetyIntervention_PaperA_shared.pdf` | 25 pp, 1.38 MB, as of 2026-08-15 |
| Spine decisions, disposition tables, traps | `NEXT_STEPS_REWRITE_2026-08-14.md` | §1 has the locked spine (do not restore the v3 "the graph is unreliable" thesis); §7 the settled numbers; §10 the traps |
| Internal reviews, W- and V-tags cited throughout this file | `REVIEW_neurips_scored_plus_style_shared_2026-08-14.md` (W1–W23), `REVIEW_workshop_scored_plus_style_shared_2026-08-15.md` (V1–V10, H1–H4) | What was already implemented against them: `REVIEW_RESPONSE_2026-08-14.md` |
| External three-model round | `reviews_2026-08-15/` — six `.md` + six `.meta.json` | Usage, wall-clock and stop reason per job in the meta files |
| Re-running that round | `review_multi_model.py` (`--smoke` first) | Verified model IDs `claude-opus-5`, `gpt-5.6-sol`, `models/gemini-3.1-pro-preview` (there is no non-preview `gemini-3.1-pro`). Keys read from three different projects' `.env` files; paths are constants at the top of the script. Actual cost of the full six-job run: **$2.53**, 2.5 min wall-clock, nothing near the 64k output cap. **This repo is public**, so the two machine-specific paths are environment-supplied and fail fast if unset — export `REVIEW_PAPER_PDF` (the compiled PDF) and `REVIEW_ANTHROPIC_ENV` (the `.env` holding `ANTHROPIC_API_KEY`) before running. The OpenAI and Gemini key files are repo-relative |
| Verification loop after any manuscript edit | `asciify_tex.py` → `texlint.py` → `graph_analysis/experiment_paper_claim_audit.py` | Expect **218/218**, not the 42/42 several docs still say — see C9 |

## 🔴 Locked decisions — do not revert

Carried forward from `NEXT_STEPS_REWRITE_2026-08-14.md` §1 and §10, which is gitignored and
whose disposition tables are already executed. These exist because each was re-derived
wrongly at least once.

- **The spine is the pipeline and the corpus it produces.** The graph-analytic material
  (merge-induced centrality, loose-threshold clustering, selection-conditioned
  co-occurrence) is *practical guidance for people reusing the corpus*, not the finding. An
  earlier draft made "the chain is reliable, the aggregate graph is not" the thesis; that
  overstates four controllable artifacts into a verdict. **Do not restore it.** If a
  sentence reads as "the aggregate graph does not support analysis", rewrite it as "this
  analysis needs this control."
- **Structural completeness is not evidence of fidelity.** 87.4% five-stage completion is
  what schema-filling predicts. Only the judge study bears on fidelity.
- **The reporting unit is the 2,772 de-duplicated chain set**, never the raw 8,954.
- **Intervention maturity is LLM-assigned and un-adjudicated** — composition, never a
  measured rate.
- **The extraction is one schema-constrained call per document. Not agentic.**
- **The two non-reproducing quantities** (88% race framing, 51-of-100 isolation) come from
  this project's own earlier internal pass, not from published work. No citation exists and
  none can be manufactured.
- **Paper B material stays out**: MIT risk anchoring, the HC/MC mechanism-class catalog, the
  doublet matrix, novel-intervention candidates.
- **`git blame` is useless on the analysis repo** — a 2026-03-08 bulk commit rewrote everything.
- **Keep the manuscript pure ASCII.** One astral-plane character in a comment once got the
  file typed as binary in Overleaf and locked against editing; cleaning the bytes afterwards
  did not release it.

**Ground rules carried into every estimate below.** Token counts come from the measured
corpus: median document 1,976 tokens, mean 5,648, and a mean extraction of 17 nodes and
17 edges. Dollar figures assume batch pricing at roughly USD 1–3 per million input tokens
and USD 4–15 per million output, and are order-of-magnitude only — check the rate before
committing. Every study below gets a GitHub issue and a PR carrying the analysis, per the
project rule, so the experiment trail stays auditable.

---

## Tier 1 — highest reviewer value per dollar

### S1. Null-repair grader arm — DEMOTED 2026-08-15 PM
**Answers** NeurIPS W3/Q2, workshop V4 — the single item both *internal* reviews ranked first.
The meta-grader pre/post comparison is confounded: graders saw the repairs they were asked
to score. The paper now reports that stage as a design lesson and draws nothing from it.
One batch call decides whether it can be a result at all.

🔴 **All three external models say cut it rather than rescue it (3/3, both bars).** Their
argument: the paper already draws nothing from the stage, so a page of main text plus the
ICC / Krippendorff / two-binnings-of-Fleiss / median-split appendix is space spent on a
measurement that licenses no conclusion — and five agreement statistics computed on 13
common papers cannot rescue a design the paper disavows. Recommended replacement, close to
verbatim across all three: one or two sentences in Limitations ("an attempted pre/post
repair scoring was unblinded and had no null-repair arm, so we discard it"), with the
numbers deleted. That makes running the sham arm optional rather than the top item: run it
only if the team wants the stage to become a *result*; if the plan is to cut, S1 is money
not worth spending. Decide cut-vs-run before booking the batch call.

**Design.** The same three graders, the same 100 papers, the same rubric, but an empty or
fabricated repair list in place of the judge's. Compare the pre/post movement against the
real arm. A materially smaller movement rescues the original result; a similar movement
confirms the presentation effect, which is itself publishable and is the more likely
outcome given the post-repair agreement collapse.

**Inputs, all recoverable.** Grader prompt and bundler on `origin/judge_handoff_workshop_items_2_3`
(`extraction_validator/prompts/judge_and_extraction_evaluation_prompt.md`,
`combine_judge_and_extraction_with_original_text.py`); the 100 judge reports on
`origin/anthropic_judge_test`; source text in `data/raw/ard_json_full/`.

**Cost.** ~11.5k input tokens per call (source + extraction + rubric). One grader over 100
papers ≈ 1.2M input, ~50k output ≈ **USD 4–8**. All three graders ≈ **USD 12–25**.
**Human involvement: none.** Batch job.

**Caveat to design around.** Gemini contributed only 13 paired scores in the original run
because of rubric iteration. Fix the rubric prompt once for all three graders in this arm,
which also retires the 95/95/13 denominator problem (W4).

### S2. Second judge run, stratified on chain-yielding papers
**Answers** W1/Q1 — the paper's largest single gap. The judge covers 0.6% of the reported
chain set, so no fidelity number in the paper applies to the unit it reports on. The
Limitations section says so; a stratified run closes it.

**Design.** Sample 100 documents from the 1,868 chain-yielding ones, stratified by source
type to match the chain set rather than the corpus. Same judge, same protocol, so the
result is directly comparable to the existing run.

**Cost.** ~10.3k input per call over 100 papers ≈ 1.0M input ≈ **USD 5–10**.
**Human involvement: none.**

**Why it is worth more than its price.** It converts every "the verified population is not
the analysed population" sentence into a measured statement, and both reviews price it at
+1 overall. **Confirmed 3/3 by the external round, which now makes this the top structural
item** (S1 having been demoted): Opus calls it "the single change most likely to raise my
score," and all three note it needs no new method.

### S3. Stage-assignment agreement, second model
**Answers** W20/Q6, and completes the study already in the paper. The probe in
`sec:r-stages` shows the extractor applies the stage vocabulary at 98.8% internal
consistency; it cannot show a second annotator would agree, because one call wrote both
the text and the label.

**Design.** Take 50 documents' extracted nodes, strip the stage labels, and have a
different-provider model assign a stage to each from the node text alone. Report
Cohen's kappa against the extractor's assignment, and a confusion matrix — the prediction
worth registering in advance is that disagreement concentrates on the pa/ti and dr/im
boundaries, which is where the existing probe's errors already fall.

**Cost.** ~5.3k input per document over 50 ≈ 265k input, ~50k output ≈ **USD 1–3**.
**Human involvement: none for the model arm.** Adding one human annotator over the same
50 documents (see S5) turns it into the full three-way study the reviewer asked for.

### S10. Edge-coverage reconciliation — NEW 2026-08-15 PM, and the most consequential new finding
**Answers** a gap none of the earlier reviews caught. GPT-5.6 Sol raises it at both bars:
the judge's coverage list flags **a mean of 7.8 missing relationships per paper** against a
mean audited extraction of **10.8 edges** — i.e. a possible edge-level omission signal of
the same order as the extraction itself — while the abstract and body foreground the 0.6%
*node* addition figure. Verified in the manuscript 2026-08-15: 7.8 appears only at
`app:judgeprompt` and `app:judge`, never in the body, never in the abstract, and never set
against 10.8. As it stands the paper's headline coverage framing rests on the one
instrument with the most favourable reading, and a reviewer who finds the appendix figure
will say so.

**Design.** No new inference. Re-read the existing judge receipts and report the coverage
list broken out as covered / partially covered / missing, per paper and in total, against
the extracted edge count for the same papers — the same treatment `tab:omission` already
gives the two *node*-level measurements. Then either (a) fold the edge measurement into
`tab:omission` as a third row and drop "implied coverage of 99.4%" from the abstract, or
(b) state explicitly why the coverage list is not an omission estimate. Note the judge
prompt has no `add_edges` slot (`app:judgeprompt` already says this), which is the likely
explanation and should be stated as one, not left implicit.

**Cost.** **Zero dollars, Class B, no LLM call** — the numbers are in
`experiment_judge_item2_report.json` (`per_paper_means.missing_edges_flagged`) and the
released graph. **Human involvement: none** unless the team wants the flagged relationships
manually inspected, which folds into S4/S5.

**Why it is Tier 1 despite being bookkeeping.** It is free, it is the only item on this list
that could move a headline number in the abstract, and leaving it unaddressed is the
cheapest way to lose a reviewer who reads the appendices.

---

## Tier 2 — needs a person, not a budget

### S4. Human-anchored spot-check, 20 papers
**Answers** W2/Q3. The judge says extractions omit 0.6% of what they should contain; the
Opus grader says 28.8%. The paper reports both, reconciles neither, and calls this the
clearest reason the protocol needs a human anchor.

**Design.** 20 papers, **stratified across source types within the chain-yielding
population, not the corpus** (revised 2026-08-15 PM: 3/3 external reviewers ask for the
human anchor on the *analysed* unit — sampling the corpus again would reproduce S2's
population mismatch inside the human study). For each, an annotator reads the source and
the extraction and records: nodes the extraction missed, nodes it asserts that the source
does not support, and stage assignments they would change. GPT-5.6 Sol asks for edge-level
grounding and source spans as well, and would rather see 50 documents than 20; Opus holds
that ~20 with two-annotator adjudication is enough to settle 0.6% vs 28.8%. Treat 20 as the
floor and 50 as the version that also answers S10. Adjudicate against both machine
measurements. A second annotator on 5 of the 20 gives an inter-annotator figure, without
which the anchor is one person's opinion.

**Cost.** Zero dollars. **~2–4 hours of one author's time**, plus ~1 hour for the second
annotator's subset.

**Where it is tracked.** This is issue #150's centre of gravity. The ticket is open to
whoever on the team picks it up and is unstarted as of 2026-08-15. It does **not** carry the
chain-yielding sampling change above — read D8 and `paper/TICKET_150_UPDATE_LOCAL.md` before
anyone starts, or the sample lands on the wrong population. I can generate the annotation packet — the 20 papers, their
extractions, a blank verdict sheet and the rubric — so the time spent is judgment only.

### S5. Manual 50-instance error taxonomy
**Answers** W4. Folds naturally into the same sitting as S4: while the annotator has the
sources open, classify 50 flagged instances by error type and record whether each is a
genuine error. Converts the auto-derived taxonomy over 43 papers from un-adjudicated model
output into something with a human floor under it.

**Cost.** Zero dollars, **~1–2 hours** on top of S4.

---

## Tier 3 — real experiments, real budget

### S6. Baselines on 200 documents
**Answers** W6/Q5. No design choice in the paper is shown to be load-bearing: not the
reasoning model, not full text over abstract-only, not the seven-stage schema over flat
triples. Three arms over the same 200 documents, scored by the same judge.

**Cost.** Extraction arms ~200 × 5.6k input ≈ 1.1M each; abstract-only is roughly a tenth
of that. Judge scoring adds ~200 calls per arm. All three arms plus scoring ≈
**USD 15–40**. **Human involvement: none.**

**Note.** This is the item most likely to change a reviewer's rating and the one most
likely to produce an unwelcome answer — a non-reasoning model at a fraction of the cost
may extract chains a judge scores similarly. That is worth knowing either way, and the
paper should be prepared to report it.

### S7. Repeat-extraction agreement, 300 documents
**Answers** W10. Every descriptive in the paper sits on an unmeasured noise floor: no
document was extracted twice, and a reasoning model at non-zero effort is not
deterministic.

**Cost.** 300 × 5.6k input ≈ 1.7M, plus output and reasoning tokens, which dominate on
`o3` ≈ **USD 30–50**. **Human involvement: none.**

**Cheaper variant.** 100 documents bounds the noise floor well enough for a workshop and
costs a third as much.

### S8. Schema ablation and degraded-source control
**Answers** the fidelity question the paper names in Limitations and does not answer.
Re-extract a sample with a prompt that does not name the five stages (does the structure
survive un-prompted?), and re-extract from documents whose argument is destroyed but whose
vocabulary is retained — sentence-shuffled, abstract only, reference list only (does the
model confabulate chains from topical vocabulary alone?).

**Cost.** ~200 documents per arm, four arms ≈ **USD 20–40**. **Human involvement:** a
judgment call on what counts as "the emergent chain maps onto the five stages", which is
either a rubric for a model or an hour of annotation.

**Why it is Tier 3 despite being the most scientifically interesting item here.** It is
the only study that would let the paper make a fidelity claim about the schema itself,
rather than about the extractor's consistency. If budget appears for exactly one Tier 3
study, this is the one with the most upside — and the most risk.

### S9. Retrieval evaluation
**Answers** W7/S-W1. The retrieval use case rests on two worked queries. A 50-query set
with three-way relevance labels, compared against embedding search over abstracts and a
retrieve-then-read baseline, would convert a demonstration into a result.

**Cost.** Query generation and baseline runs are cheap (**under USD 10**); the binding
constraint is **relevance labelling, ~3–5 hours of human time**, or a model-labelled
proxy that a reviewer will discount.

---

---

## Not studies, but owed: manuscript corrections from the external round

Each of these is an edit, not an experiment, and each was verified against
`paperA_altstyle.tex` on 2026-08-15 rather than taken on the reviewer's word. Ordered by
how badly a reviewer reacts to finding it.

| # | Correction | Agreement | Verified |
|---|---|---|---|
| C1 | The eight rendered `\OPEN{[GAP: ...]}` blocks (enumerated below), four of which are notes addressed to co-authors ("Open for the team to decide", "Do not populate from git history alone"). Every model at both bars calls the submission unfinished on this basis alone, and two say nothing else matters until it is fixed | **3/3** | 8 blocks render |
| C9 | **Five stale "42/42" claim-audit references** — `REPRODUCE.md:44`, `NEXT_STEPS_REWRITE_2026-08-14.md:17` and `:369`, `NEXT_STEPS_2026-08-11.md:20` and `:173` — against the manuscript's current 218/218. `REPRODUCE.md` is the file a reviewer actually runs, so the mismatch reads as a broken audit. Not raised by any model (they never saw the repo); found in-session 2026-08-15 | in-session | 5 references confirmed |

**C1 in full — the eight blocks, and who can close each:**

| Where | Open item | Closes by |
|---|---|---|
| Abstract | release URL | team: hosting decision |
| `sec:m-repro` | licence pair + ARD redistribution terms | team: legal call (see the licence-fallback note below) |
| `sec:limitations` | human spot-check + manual error taxonomy, NOT performed | doing S4/S5, or rewording to drop the forward lean |
| `sec:limitations` | n=20 multi-model consistency check run earlier in the project | a co-author recovering the numbers; else delete |
| Acknowledgments | compute acknowledgment — blocked on donor consent (gate G14) | team: send the built PDF to the donor, default anonymous |
| Acknowledgments | author list, affiliations, contribution statement (gate G15) | team |
| Use of AI Assistance | scope of the drafting claim | settles with G15, same conversation |
| `app:clusters` | publish 20 representative nodes per cluster | ships with the release, or moots itself if `app:clusters` is cut per the cut list |
| C2 | "schema-constrained" (×4, incl. abstract and Fig. 1 caption) while `sec:m-extraction` concedes conformance is prompt-enforced with no structured-output constraint. Use "schema-prompted" | 1/3 | 4 occurrences, contradiction confirmed |
| C3 | `sec:m-repro` "a re-run reproduces the same model generation" (L627) reads against `sec:m-extraction` "runs are not bit-reproducible" (L339). Both render; the first invites the wrong reading | 1/3 | both lines confirmed |
| C4 | `tab:gates` marks the maturity-$\geq 3$ row "(deployed)" while the rubric reserves *deployed* for maturity 4. The intent is "the deployed setting"; relabel so it cannot be read as the maturity band | 1/3 | confirmed |
| C5 | `refs.bib` carries 29 non-standard annotation fields, including "Verified 2026-08-15" and per-entry mini-summaries. Move to a source-verification file | 1/3 | 29 fields |
| C6 | `sec:m-recovery` says the judge cannot recover failed extractions "at a useful rate" with no number in the body. The figure exists (23 of 441, 5.2%) — print it or cut the sentence to "failed extractions are corpus loss" | 2/3 | no number in body |
| C7 | "One call per document" vs a client that retries up to three times — distinguish logical requests from API attempts | 1/3 | `max_retries=3` in Methods |
| C8 | The release's defect status is undocumented: the judge found 108 referential-integrity findings, 42 orphans and 56 duplicate pairs, and no repaired graph was rebuilt. State whether the released dump carries them | 2/3 | consistent with `app:judge` |

## Language, register and typography — the low-hanging fruit

Every reviewer at both bars flagged writing style, and until 2026-08-15 PM none of it was
recorded here — it lived only in the review files. Collected below so it can be worked down
rather than re-read. Instances are quoted so they are greppable in `paperA_altstyle.tex`.
None of this requires a decision; all of it is a prose pass.

| # | Item | Agreement | Action |
|---|---|---|---|
| L1 | **`\author{Author List Placeholder}` (L102) renders on page 1** — a *ninth* visible placeholder, separate from the eight `\OPEN{}` blocks of C1 | 1/3 | goes with G15; until then it is the first thing a reviewer sees |
| L2 | **Aphoristic paragraph-enders.** "We add the layer under it."; "A corpus of $N$ short chains is exactly $N$ components until something links them."; "A faithful extraction from a weak paper is a successful extraction."; "The objective above is what the step is for; the counts below are what this approximation to it produced."; "The counts require no control, being direct tallies." | **3/3** | the internal reviews counted eight and said keep two. Opus: "deployed forty times they read as generated polish and displace information" |
| L3 | **Contrastive correction** — "X is not Y", "read this as A and never as B". ~30 instances by the internal count, "well over a dozen" by Opus's | **3/3** | keep where a plausible misreading exists *and* the paper has evidence about it; target under 8 |
| L4 | **"honest" as editorial** — "the honest statement of yield", "the honest positive residue" | 2/3 | GPT-5.6 Sol: "implicitly characterizes alternative summaries as dishonest" |
| L5 | **Promotional / advocacy** — "The corpus is a snapshot of one dataset and will date. The paired extract-and-verify design will not."; "would make research coordination ... tractable"; "the natural agentic use"; "The extension this work most needs"; "What makes a mechanism layer worth building" | 2/3 | "will not [date]" is unsupported and absolute — models, prompts and schemas date too. "tractable" → "could support" |
| L6 | **Conversational / blog register** — "What the release contains. Five things:"; "A closing note: some statistics are true by design."; "All three duly record an improvement"; "the reader who takes the release and does something with it" | 2/3 | "duly" reads as sarcasm |
| L7 | **Legalistic meta-formulations** — "what licenses reading the other rows"; "which is what settles it"; "A reader would otherwise misread a number" | 1/3 | state the assumption and its implication directly |
| L8 | **Formulaic openers** — "Three things follow"; "Two properties bear on"; "What this does not show" | 1/3 | frequency is the tell, not any single instance |
| L9 | **Over-attribution of importance** (the flag Martin asked reviewers for) — "The verification stage is half the contribution" (`app:judgeprompt`); "what makes the extraction checkable rather than merely large" (`sec:r-judge`); "This is the single licensing gate"; "That qualifier is essential" (`sec:r-corpus`); "The single most consequential row is the sixth" (`tab:populations-master` caption); "the choice of extractor moves the bill by about a factor of five — more than any other decision in the pipeline" (`sec:m-repro`); "The sharpest is a merge-manufactured centrality hub at 90x" (Conclusion); "218 of 218 numeric claims passing" as a quality badge; "fifty documents ... would settle it"; the two worked queries as "the precondition for the cross-paper analysis" | **3/3** | the verification-stage claims are the load-bearing ones: the stage ran on 0.85% of documents and 0.6% of the analysed chains. The 90x hub in the Conclusion is an artifact of a step **not applied** to the released substrate, elevated to a headline |
| L10 | **Same three caveats repeated across 6–8 sections** (verified ≠ analysed; yield is a gate property; completeness is schema-filling) | 2/3 | one clear statement each plus cross-references |
| L11 | **Acknowledgments carry project-management detail** (Discord stand-ups, working threads) | 1/3 | not scholarly acknowledgment |
| L12 | **Terminology overstates the evidence** — "verification", "implied coverage", documents "argue a complete mechanism" | 1/3 | → "the extractor produced a chain judged to pass the model-assigned gates"; "auditable" or "subject to an LLM diagnostic pass" rather than "verified" |
| L13 | **Mechanical sweeps** (from the internal reviews, not re-raised externally): mixed British/American spelling — "randomisation", "neighbourhood", "specialised", "favourable" against "normalization", "labeling", "colored"; number-words inconsistent — "Twelve of the 100" vs "12 of the 100"; `\emph{}` 40+ times, mostly on ordinary words | internal | one spelling variety, one number rule, `\emph{}` for term introductions only |
| L14 | **Source-file editorial trail** — "REMOVED 2026-08-14", "Moved out of sec:r-hub", "the frozen Overleaf reported...", "the module docstring says 80%, the code uses 70%", the compute-donor gate block. Several disclose internal disagreement, an Overleaf workflow and a private donor | internal | strip or move to a NOTES file before any public posting. **Not** the same as C1: these are comments and never render |

**Checked and clear:** GPT-5.6 Sol asked that future-dated bibliography entries be verified
against the submission date. Checked 2026-08-15 — `refs.bib` years top out at 2025, so there
is nothing to fix. The four `urldate` / "Verified" annotations are C5's business.

Two further items are judgement calls for the team rather than corrections. **Ungated
release (Opus, both bars):** make the ungated chain set the primary released unit with the
gates exposed as a user-side filter — this defuses "the reporting unit is selected by two
unvalidated attributes" structurally instead of by measurement, and is worth weighing
against S2/S4. **Licence fallback (Opus, workshop):** treat the unresolved licence as a
limitation with legal exposure and state a fallback position (release structure only for
sources whose terms permit it) rather than deferring it to a gate. Opus also notes ARD's
own selection bias — who curates it, what it excludes, English-only — is inherited by every
number in the paper and is discussed nowhere beyond the source-type mix.

## Decisions owed by the team

| # | Decision | Blocked on | Notes |
|---|---|---|---|
| D1 | **Venue.** Nothing committed. The draft is venue-neutral two-column `article`, so switching is a preamble-only change | team | The external round prices the choice: conference 3 / 2 / 3 across the three models (all reject or borderline reject), workshop 4 / 3 / 5 (split). On this evidence a main-track submission is not currently viable and a workshop is borderline-to-positive. Re-check the AI-disclosure wording against the choice: ICLR 2026 desk-rejects undisclosed LLM use; ICML 2026 permits assistance but forbids crediting an LLM |
| D2 | **Cut vs run S1** (null-repair arm) | team | See S1. 3/3 external reviewers say cut; running it only makes sense if the stage is to become a result |
| D3 | **Ungated vs gated release** as the primary unit | team | See the ungated-release note above; interacts with S2 and S4 |
| D4 | **Licence pair + ARD redistribution position** | team, possibly legal | = C1 row 2; Opus asks for a stated fallback rather than a deferral |
| D5 | **Release hosting + URL** | team | = C1 row 1. Blocks C1 and every reviewer's first question |
| D6 | Compute-donor consent (G14), author list + contribution statement (G15), AI-drafting scope | team | 🔒 **Detail in `NEXT_STEPS_PRIVATE.md`** — these three are tracked there, not here, and they gate four of the eight `\OPEN{}` blocks |
| D7 | Co-author coordination: draft send, the outstanding contribution question, the #150 refresh (D8), PR #151 (#149 was closed unmerged) | team | 🔒 **Named detail in `NEXT_STEPS_PRIVATE.md`** — who owes what stays off the remote, per the `paper/` gitignore policy |
| D8 | **Issue #150 refresh — priority, and whether to send it now.** Verified 2026-08-15: all five open items and all three nice-to-haves unstarted, no ticket activity since 2026-08-11, neither target CSV exists. Four things changed underneath the ticket, one of which would waste the work for whoever picks it up (the human anchor must sample **chain-yielding** documents, not the judged 100) | team; change 3 of it waits on D2 | 🔒 **Full write-up and a ready-to-send draft comment in `paper/TICKET_150_UPDATE_LOCAL.md`** (local, gitignored). Nothing has been posted to the ticket |

---

## Not a study, but still owed: the body must reach 10 pages

**This is a hard requirement regardless of venue, and it is not yet met.** Ten pages
excluding references is the ceiling at conferences as well as workshops, so there is no
target under which the current draft fits.

Estimated body after the 2026-08-15 pass: **~12.5–13 pages**, down from ~15–16. That
estimate is a character-count heuristic, not a build — **compile on Overleaf and read the
real number before deciding what else to cut.**

Where the remaining ~3 pages would have to come from, in the order I would take them:

1. **The three artifact use cases** (clustering, centrality, co-occurrence) — roughly 110
   source lines. Keeping each finding plus its control inline and moving the derivations to
   an appendix recovers most of it. Both reviews call this material the paper's strongest
   evidence, so this is an author call, not an editorial one.
2. **Methods §From path enumeration to the reporting unit** — the containment sweep and its
   loss measurement can sit in the appendix behind one sentence.
3. **Figure 2 (the funnel)** — a full-width figure whose job is disambiguating four
   operations that now have four distinct names. Cutting it recovers about half a page.
4. **The judge subsection** — the structural-findings detail can follow the grader mechanics
   into Appendix I.

Items 2–4 are mechanical and cost little. Item 1 costs prominence, which is why it is
listed first by yield and last by preference.

**The external round supplies a cut list that reaches the same page count without touching
item 1**, and its top three are unanimous. Opus estimates items 1–3 below alone remove
about a third of the appendix and a page of main text "without touching a single
load-bearing claim":

| Cut | Agreement | Disposition |
|---|---|---|
| Pre/post repair scoring + the agreement-instrument appendix (ICC, Krippendorff, two Fleiss binnings, median split) | **3/3** | → two sentences in Limitations; delete the statistics (see S1) |
| `sec:r-selection` race-framing non-reproduction + `app:race` (52-node classifier validation, odds-ratio table) | **3/3** | → one paragraph carrying the general control: selection-conditioned statistics are unstable under merge and threshold choices |
| Everything quoted from the earlier internal substrate — `tab:clustering-methods`, the first four rows of `tab:dedup-thresholds`, `app:sensitivity` hop counts | **3/3** | → cut, or re-derive on the released graph. Opus additionally asks that the recurring "earlier pass on a merged 200,061-node graph with an 8.5× sparser similarity layer" thread be consolidated into one footnote |
| `app:clusters` 40-cluster name list | 2/3 | → size range plus 3–4 examples; ship the list with the release |
| `sec:m-recovery` + the 441-row of `tab:populations-master` | 2/3 | → one clause (see C6) |
| Meta-grader agent-session operational detail (13 JSON shapes, folder-agent behaviour, uneven denominators) in `app:judgeprompt` | 1/3 | → cut, or re-run the graders on a fixed schema |
| The "218 of 218 numeric claims" audit narrative | 1/3 | → mention the reproducibility scripts once; it demonstrates manuscript consistency, not empirical validity |

**Explicitly keep**, named by every model that raised the topic: the merge/centrality
artifact (`sec:r-hub`), `tab:gates`, `tab:populations-master`, the source-type skew table,
and the Euclid failure case (`app:failure`) — Gemini calls the last one worth a slide by
itself. Opus adds a closing warning worth weighing against the whole cut list: with so many
numbers labelled "not evidence of anything wider", it becomes hard for a reader to say what
the paper *does* establish, and a draft stating two or three defensible claims and cutting
the rest would be stronger, not less honest.

## What is deliberately not on this list

- **Extending the corpus past 2023.** Both reviews note ARD stops at 2023. Running the
  pipeline over a 2024–2026 arXiv safety slice is now costed by S1's sibling measurement:
  at USD 32–118 per 1,000 documents, a 5,000-document slice is **USD 160–590**. That is a
  scope decision for the team, not a study.
- **MIT risk-anchoring of a chain subset.** Paper B material, deliberately out of scope.
- **Anything requiring the frozen co-author substrate.** Three appendix tables still come
  from it; re-deriving them on the released graph is bookkeeping, not a study, and is
  tracked in the manuscript's own carve-out in `sec:m-repro`.
