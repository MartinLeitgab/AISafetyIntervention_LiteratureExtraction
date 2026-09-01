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

🔴 **Framing decision, 2026-08-16 PM, and it governs how the rest of this list is read.**
The judge is a proof of principle over the population we *release* -- 11,779 extracted
documents, 200,525 nodes -- and that is deliberate. The gate-selected chain set is an
**exemplary analysis**: a demonstration of what the released dataset supports, not a claim
of novel scientific insight about the literature. Stated that way, the "verified population
is not the analysed population" objection loses most of its force, because the stage we
audit is the stage we release and the chain work sits on top of it as a worked example.

✅ **EXECUTED in manuscript commit `e620d71`.** Six passages carried the old framing, each
phrased as a deficiency ("does not constitute an audit of", "very little about", "nothing in
this subsection is"). All six now state what the design *is*: the Introduction (both halves,
before any number), Methods `sec:m-validation`, the `sec:r-judge` opener, the `sec:usecases`
opener, Limitations, and the Conclusion sentence that used to say the corpus "profiles a
literature", which was the one line contradicting all the others. No number moved.

🔴 **Correction to earlier advice in this file: S2 is NOT better than the reframe, and the
line saying "prefer measuring over reframing" was wrong.** S2 would buy a fidelity
measurement for the gate-selected chain set. Under this framing the paper does not claim
scientific validity for that set, so S2 measures something the paper deliberately does not
assert — it is evidence for a claim we are not making. It stays on the list as optional and
genuinely cheap, and it is the right purchase only if the team later decides to promote the
chain analysis from demonstration to finding. **Do not run it to pre-empt a reviewer
objection the framing already answers.**

## 🔴 UNIT DOCTRINE, set 2026-08-26. Read this before quoting any quality number.

**The unit of this paper is the CHAIN.** Every quality, fidelity and stability number must
count chains — or say plainly that it does not. A concept added mid-argument does not change
which risk reaches which intervention, so a node count is not an omission rate for the
artifact, and printing one invites a reviewer to price the contribution on a measurement of
something else.

Node counts are fine where the claim is about nodes: the artifact inventory (200,525 nodes,
subtype counts, release hygiene), the stage-vocabulary evidence (98.8% probe, kappa 0.84 —
those ARE claims about node attributes), the merge/centrality/clustering use cases (whose
entire point is that node-level analysis distorts), and the edge-orientation finding. Leave
them.

**What was wrong and is now fixed** (manuscript `9583f77`, `a52e32c`, `bfd8789`): the
abstract printed five omission rates and "we report them all and reconcile none"; the
Conclusion repeated it; `sec:r-judge`'s three-measurement summary never said what the rates
were a rate *of*; Limitations' "reads worse, not better" drew a chain verdict from node
counts; the re-run paragraph's 46.5% read as pipeline stability when it measures name
strings; and both loss paragraphs led with node figures (6.1%, 78.3%) ahead of the
argument-level ones (18.0%, 28.0%).

**Two receipts underwrite the doctrine.** `experiment_review_omission_is_chain_level.py`:
the flagged omissions are NOT inert — 91.5% name a node the extraction lacks, matcher
validated by its 95.1/43.0/8.5% resolution gradient. `experiment_review_omission_chain_
impact.py`: granting all 270 flagged relationships makes **1 new risk-to-intervention pair
reachable out of 86 available** (1.2% of headroom). 🔴 Headroom is the denominator — 86 of
the 100 papers are a single connected component, so quoting against the 408 reachable pairs
reports a ceiling as a finding. The claim audit fails if anyone does.

**All three chain-level studies are DONE. Closed record with every trap:
`paper/FINDINGS_2026-08-26_chain_unit.md`.** Read that before re-deriving any of it.

1. ✅ **Is a dropped path a distinct argument?** (R22/R23) — #182 / PR #185. 50.8% of 118
   drops different, 40 of those 60 by endpoint, null arm 15/15. Report as a hierarchy:
   **18.0%** (579 of 3,222 distinct pairs) is the headline; 33.9% and 50.8% are shares of
   sampled drops and are NOT competing estimates.
2. ✅ **Chain recall** — #181 / PR #184. Gate-free 69.9% carried / 10.6% material,
   sensitivity 9/20. Gated 32.9% carried, **which is selection and not loss**: 471 of 597
   graph-present pairs (79%) are gated out, so the document was extracted and the chain
   scored below a gate.
3. ⛔ **Does a re-run produce the same PAIR? DROPPED 2026-08-26**, and not for cost. The
   reporting unit keeps 1.30 of ~6.15 candidate pairs per document, so which chain survives
   turns on a narrow scoring margin and a re-run flipping membership is expected gate
   behaviour — already reported as 9 of 18. The question it proxied for is answered better
   by the recall study. Do not revive it without a new reason.
4. ⛔ **Retrieval** (R28/S9) — still skipped, and now lower value: the substrate is known to
   be selective by design, so a retrieval evaluation would measure that selection first.

What still follows from the framing, for a session working the list:
- Any sentence treating a chain-set descriptive as a finding about the literature must be
  re-read against it. 87.4% completeness, the maturity profile and the one-in-six yield are
  properties of a worked example under our gates. The draft now says so in all six places,
  but a fresh pass over `sec:r-stages` and `sec:r-corpus` is worth doing once.
- This is also why S4 shrank to "do nothing" and S5 was dropped: a human anchor on an
  explicitly exemplary analysis is not load-bearing evidence.

## 🔴 RUNBOOK — check-off list

Worked top to bottom on **2026-08-16 PM**. Each row names the issue, the PR, what it found
and where it landed in `paperA_altstyle.tex`. Rules that govern all of it: `git pull
--ff-only` the manuscript repo **before you Read**, push the same session, keep the `.tex`
pure ASCII, and after any manuscript edit run `asciify_tex.py` → `texlint.py` →
`graph_analysis/experiment_paper_claim_audit.py` (**re-run it; do not quote a count from
this file** -- see the verification-loop row of "Where everything lives". If a number leaves
the manuscript, delete its `check(...)` line, and if one arrives, add one). Every study gets
a GitHub issue and a PR.

| | Item | State | Issue / PR | Landed in the manuscript |
|---|---|---|---|---|
| R0 | OpenAI key on this machine | ✅ used | — | not a manuscript item |
| R1 | Download ARD | ✅ **was never needed** | — | — |
| R2 | S6+S8 schema ablation + degraded source | ✅ done | #165 / PR #169 | `sec:r-stages`, `sec:limitations`, `app:failure`, new `app:ablation` |
| R3 | S11 multi-model consistency, n=20 | ✅ arms A/B done, C running | #168 / PR #170 | `sec:limitations`, rewritten around the run-to-run floor |
| R3b | S12 comparison against an existing artifact | ✅ done | #166 / PR #167 | `sec:related`, first paragraph |
| R4 | L3 contrastive corrections | ✅ done, target missed | — | throughout; manuscript `99a20ec` |
| R5 | L10 repeated caveats, L12 verification→audit | ✅ done | — | throughout; manuscript `99a20ec` |
| R6 | Strip L14 before posting | ⏸ deliberately not now | — | source comments only |
| R7 | Ungated chain file + CLI flags | ⛔ retired | — | — |
| + | S6 remainder: baselines B/C/D + unconditioned repeat arm | ✅ done | #171 / PR #170 | `sec:r-stages`, `sec:limitations`, `app:ablation` |
| + | S2: judge the chain-yielding population | ✅ done | #172 / PR #170 | `sec:r-judge`, `sec:limitations`, abstract, conclusion |
| + | What a chain count is a function of | ✅ done | #168 follow-up | `sec:limitations` |
| + | Chain order + edge orientation: does an undirected enumerator yield a directed argument? | ✅ done | #173 / PR #174 | `sec:m-paths`, `tab:cuts` |
| + | Third review round (3 models x 2 bars) + the cut list built from it | ✅ done | PR #174 | nine `\CUTNOTE{}` scaffold blocks; `paper/section_cut_list.md` |
| + | The four unsupported claims the third round rejected, and the audit-count drift | ✅ done 2026-08-26 | manuscript `7d5e1d6` | `sec:r-judge` no longer reads "a direction and a floor" off a three-way-confounded comparison; the "document-level resources **cannot** serve this query" absolute is a representation claim in all three places it appears (`sec:intro`, `sec:r-retrieval`, `sec:conclusion`); the audit count is re-run, not quoted |

🔴 **S2 changed a claim rather than confirming one.** Node-level omission on the
chain-yielding population is **26.4%** against the corpus-sampled run's 0.6%, edge-level
21.7% against 18.1%. Eight passages described that population as unaudited and have been
reconciled (manuscript `b745108`). Two confounds ride with the population and the paper
states both: those documents are about twice the corpus mean length, and their extractions
were rebuilt from the released graph without the four rationale fields.

🔴 **"594 chains per document" was raw enumeration, not the reporting unit.** After the 70%
sub-path collapse the three models read 0.5, 2 and 4 chains per document and the worst
document falls from 57,007 paths to 44. Never compare pre-collapse counts across
extractors; the driver is mean degree crossing 2, not extraction quality.

🔴 **The evidence program is complete.** What remains needs a person (S4 human anchor, S9
relevance labels) or a decision (venue, release URL, authorship, and the page cut).

## 🔴 MEASURED BUT NOT IN THE MANUSCRIPT — check this list before writing any new number

Everything below is committed, receipted and ticketed, and **none of it appears in
`paperA_altstyle.tex`.** That is a live decision in each case, not an oversight, and each row
says what would have to be true first. A future session must not quote any of these numbers
into the paper on the strength of the receipt alone.

| Finding | Where | Why it is not in the paper |
|---|---|---|
| **52.0%** of 200 reporting-unit chains judged not a fair summary of an argument their source makes; **16.5%** carry a risk framing the judge could not quote | #175 / PR #177 | One model's opinion. It over-flags (8 of 200 correctly-paired chains called a different document) and should also under-detect, sharing the extractor's priors. Needs #176 |
| **17.8 pp** gate discrimination: 52.0% against 69.8% on gate-rejected chains | #175 / PR #177 | Same instrument. This is the half most likely to survive #176, and the half a reviewer would find most useful, but it rests on the same unadjudicated verdicts |
| All 33 invented risk framings **cleared the confidence gate** (27 at exactly 3), and neither gate discriminates on that failure mode (16.5% against 15.6%) | #175 / PR #177 | Depends on the 52% verdicts being real. If #176 overturns them this row dissolves |
| Blind re-labelling of 66 first hops: **the label discriminates (24.2% vs 78.8% on "does the document assert this link") but its level is unusable** — a second annotator puts 60.6% of *faithful* links below the gate too | #178 / PR #179 | One model applying a rubric to a task shape it was not written for, and the same model family as #175, so this is consistency rather than correctness. It also carries a design lesson worth more than the numbers: ask the binary question, not the five-point scale |
| **Collapse adjudication: 50.8% of drops judged a different argument; 33.9% change an ENDPOINT; null arm 15/15** | #182 / PR #185 | One model. Report as a hierarchy strictest-first — 18.0% (pair in no kept chain, already in the paper) then 33.9% (endpoints differ from the container) then 50.8% (plus same-endpoint re-routings). Do NOT headline 50.8%: a third of it is a different mechanism between the same two endpoints, which is a loss only if mechanism is the unit |
| **Chain-level recall: 10.6% of arguments materially uncaptured, 23.6% corrected, sensitivity 0.45 [0.26, 0.66]** | #181 / PR #184 | 🔴 **Gate-FREE baseline**, so it measures whether the PIPELINE captured the argument, not whether the 2,772 reporting unit carries it. The gated version is a different and larger number and has not been run. Quote the sensitivity beside the rate or neither |
| Human adjudication of 30 chains, stratified by #175's reason codes | #176 | **Packet rebuilt and pre-registered 2026-08-26 (`7e0be0e`); annotation not started.** This is the arm that would license any of the above |

🔴 **#176 is ready to annotate, and three things about it were settled 2026-08-26.**
Commit `7e0be0e` on `study/confidence-relabel`. Start at
`graph_analysis/phase2_results/human_review_packet/README.md`.
- **It returns a population rate, and only because of how #175 was sampled.** #175's real
  arm is 200 chains stratified by URL host *proportional* to the reporting unit, so its
  reason-code shares are population estimates. The 30 human cells sit inside those codes
  and post-stratify back up. The strata were rebuilt to make this work: `intermediate_
  unsupported` went 0 -> 3 (it was 7.5% of the unit and blind) and the gate-rejected arm
  1 -> 6, paid for by `risk_framing_invented_both_agree` 6 -> 3 and the known judge false
  positives 2 -> 1. Arm-A coverage is now **100.0%**, up from 92.5%.
- **The estimand is pre-registered.** `experiment_review_human_adjudication.py` was written
  before annotation on purpose, because a reason-code-stratified sample supports several
  different rates. Do not quote the pooled rate over the 30 (it runs high by construction),
  do not renormalise the weighted shares to 1.0, and do not print a human-anchored 17.8 pp
  — arm B carries six observations and is directional only.
- **One annotator, so R11 stays open.** No inter-annotator figure will exist. The eight
  double-coded rows still ship in case someone is found. Re-judging one's own rows is
  test-retest, not agreement, and must not be reported as agreement.
- **What #176 does NOT close**, so nothing here gets over-sold: recall and omission (R10,
  R16 — the 0.6% vs 28.8% discrepancy is untouched, since the packet only shows chains that
  *were* emitted), maturity labels, the 70% collapse (R22/R23), non-chain-yielding
  documents, retrieval relevance (R28).

**What IS in the paper from this line of work:** one row of `tab:gates` — the asymmetric
first-hop $\geq 4$ cut at 1,148 chains and 7.2% yield. It is there as a **price**, with the
caption and the source comment both stating that the threshold was chosen on the sample that
measured it and that nothing claims the surviving chains are more faithful. Do not add the
benefit side without #176.

🔴 **The Claude Code CLI is NOT a free execution path, and every note in this file that
said so was wrong (corrected 2026-08-16 PM).** `claude -p` bills the interactive session's
usage allowance, which is shared with everything else this project does and is exhaustible.
Arm C of #168 consumed it at roughly **one percent per minute** while emitting 43k output
tokens per document, and it had to be killed mid-run. **Use a vendor API and spend dollars**
-- `paper/review_multi_model.py` has always done this and is the pattern to copy: three
vendors, three SDKs, keys from `.env` files. Metered Opus 5 on the same work costs **USD
0.70 per document**, which is cheaper than the session time it would otherwise eat. S3
(#161/#162) also ran on the CLI and its "USD 0" is wrong in the same way; the study is done
and does not need re-running, but do not cite it as evidence that the CLI is free.

**R0. Credentials that exist on this machine.** An OpenAI key is at
`~/0_project_work/ExistentialRiskBenchmark/.env` (`OPENAI_API_KEY`, alongside `GOOGLE_API_KEY`
and `ANTHROPIC_API_KEY`). It is NOT in this repo and must never be copied into it — read it
from that path at run time and keep it out of logs, receipts and commit messages. Metered
spend applies, so estimate and confirm before using it. **Spent 2026-08-16: 85 `o3` calls
for #165 plus 40 `o3`/`gpt-5` calls for #168, against a pre-run estimate of USD 9–30.**
`gpt-5.6-sol`, the id this file used to quote, is **not** in this account's model list;
`gpt-5` is.

**R1. — RESOLVED WITHOUT WORK, and the old text was wrong.** ARD has been on this machine
since 2025-08-31 at `intervention_graph_creation/data/raw/ard_json_full/` (12 `.jsonl`,
440 MB, one record per document with `url` and `text`). `experiment_review_extraction_cost.py`
has been reading it all along — that is where the 100% URL match on 11,779 documents comes
from. The claim that a missing `data/raw/ard_json_full/` blocked S2, S6 and S8 was right
about the repo-root path and wrong about the machine. **Nothing was ever blocked on this.**

**R2. S6+S8, done — and it split the claim rather than settling it.** Issue #165, PR #169.
85 `o3` calls, arms E/F/G, every raw response committed. Removing the five stage names from
the prompt takes chain yield from 30 of 30 documents to 11 of 28 and the five-stage share
from 65.0% to **zero**, so the seven-node chain is prompt-supplied. But the model invents
the vocabulary anyway: **144 emergent labels, 138 of which map onto one of the five stages**
under a rubric that offers "unmappable" as a real answer. Sentence-shuffled sources still
yield a chain for **30.0%** of documents (46.3% of those carrying all five stages), which is
the confabulation rate `app:failure` previously lacked. Reference lists alone yield 2 chains
in 25 documents, and for 6 the model returns no graph and says so explicitly.

🔴 **Substrate finding from the same run, and it belongs in every future session's head: the
released enumerator traverses edges UNDIRECTED.** `phase2_step4_F2v4_hopwise_falkordb.py`
queries `MATCH (n)-[e:EDGE]-(m)` and fills `adj[s].add(t); adj[t].add(s)`. Three of 55
sampled documents have a released chain that is unreachable if edge direction is honoured. A
chain's risk-to-intervention order comes from where the traversal starts and stops, never
from the direction of the edges it walks — which is defensible, since `caused_by` points
from effect to cause, but was undocumented. Now in `sec:m-paths` and as a tenth row of
`tab:cuts`.

**R3. S11, multi-model consistency.** Issue #168. Arms: `o3` re-run, `gpt-5`, and
`claude-opus-5` through the Claude Code CLI on subscription auth. Arm A re-runs the corpus
extractor on documents it already extracted, so released-vs-rerun is a repeat-extraction
noise floor at n=20 — part of S7, free. That floor is what the #165 arm differences have to
clear, so the two studies are written to be read together.

**R3b. S12, done.** Issue #166, PR #167. The AI Safety Graph publishes its input (7,011
arXiv records from ARD, title and abstract only) and its clustering (554 papers, 160
subcategories). 534 are in our corpus, 216 yield a chain. Across the 79 subcategories
holding one, **79 topic labels against 325 distinct risk-to-intervention pairs**; their
largest, *Adversarial Machine Learning*, holds 18 distinct risks and 23 distinct
interventions under one string. The converse ships with it: their clustering asserts 2,986
cross-document co-membership pairs and ours asserts none. Unplanned check: chain yield on
their paper list is **40.4%** against the 40.5% the manuscript reports for arXiv documents.

**R4. L3, done, and the target was missed on purpose.** 70 contrastive constructions down to
38, not the under-15 the reviewers asked for. What went: "rather than trusting the builder
configuration", "rather than a figure", "and not in Results", "rather than a synthesized
answer". What stayed, in three groups — locked decisions the paper must keep visible
(composition never a measured rate; corpus description not fidelity; a batched job and not
an agent), distinctions with a measurement behind them (the 4-node floor is imposed not
observed; omission rather than fabrication), and definitions that exist only as a contrast
(confidence scores the source's evidence, not the model's certainty). Cutting further would
delete evidence.

**R5. L10 and L12, done.** L12 all-or-nothing as required: "verification" as a defined term
is gone from all five rendered places and the paper says "audit" throughout. L10: the three
repeated caveats now state once and cross-reference.

**R6. Strip L14 immediately before any public posting, and not before.** See L14.

🔴 **What "gated" means, since the wording here caused a real misreading.** It is NOT the
raw graph, which has no gates at all and is the primary release. It is NOT unconstrained
enumeration either: five *structural* constraints (simple paths, first hop on an intermediate
subtype, stop at the first qualifying intervention, a three-hop floor, a thirty-hop ceiling)
always apply and are part of what a chain *is*. "Gated" refers only to the two model-assigned
*quality* attributes, edge confidence and intervention maturity, at $\geq 3$ rather than
$\geq 1$ — the two reviewers called unvalidated.

**What a fresh session must NOT do alone:** anything in "Decisions owed by the team", the
four rendered `\OPEN{}` blocks, and the human studies. Those are listed further down with
their blockers.

**Amended 2026-08-16** after an execution pass that closed S3, S10, C2-C9, every
unanimous cut-list row and the language items: manuscript commit `337d033` in
`AISafetyIntervention_PaperA_shared`, analysis issues #156 / #157 with PRs #158 /
#159, and S3 as #161 / #162.

🔴 **Rendered open blocks: FOUR, counted 2026-08-16 PM** (`grep -c 'OPEN{\[GAP:'` on
`paperA_altstyle.tex`). Earlier counts of eight, nine or six are all stale, and the
difference is not bookkeeping -- five of the eight closed as work landed. What renders now:
the release URL (abstract, D5), the compute acknowledgment (D6/G14), the author list
(D6/G15) and the AI-drafting scope (D6/G15). **Every one is a team decision; none is a
writing task.** Do not count with `grep -c 'OPEN{'` -- that also matches three source
comments discussing the mechanism. `\author{Author List Placeholder}` (L1) still renders on
page 1 and is a fifth visible placeholder that is not an `\OPEN{}` block; it closes with
G15.

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
| Open PRs from this work, stacked | #158 (edge coverage, #156) -> #159 (substrate audits, #157) -> #160 (audit re-point) -> #162 (stage agreement, #161) -> #164 (null-repair, #163) -> **#169 (schema ablation, #165) -> #167 (artifact comparison, #166) -> the #168 PR (multi-model), pending** | Each branches off the previous, so **merging a later one merges the earlier ones**. Review in number order. All target `experiment/extraction-cost` -> `paper/receipts-clean` (#151) -> `main`. The three newest were merged forward on 2026-08-16 so the tip carries every check; re-run the audit to see the count there |
| Verification loop after any manuscript edit | `asciify_tex.py` → `texlint.py` → `graph_analysis/experiment_paper_claim_audit.py` (then `experiment_paper_finding_coverage.py`, which greps the rendered text for retired phrasings) | 🔴 **Re-run the audit and read what it prints; never carry its count forward by hand.** It read **385/385 PASS, 0 FAIL** on `study/confidence-relabel` on 2026-08-26, and it differs per branch in the stack. This row said 285 and the runbook said 302 while the manuscript comments said 257 and `paper/REPRODUCE.md` said 257 — four figures, one true value, and the same failure the 42/42 sweep was supposed to end. The canonical home of the number is `REPRODUCE.md` **in the manuscript repo**; it now says re-run rather than quote, and nothing else should print a count at all |

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
- **The extraction is one schema-*prompted* request per document. Not agentic.** Conformance
  is prompt-enforced with no structured-output constraint, so "schema-constrained" was an
  overclaim and left the manuscript on 2026-08-16 (C2). A request may cost up to three API
  attempts; that is a retry, not a second request, and never a conversation.
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

### S1. Null-repair grader arm — CUT 2026-08-16; preserved in issue #163 / PR #164
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

**Decision, 2026-08-16 (this is D2, now resolved): cut the stage, do not run the arm.**
Two of the three graders are Gemini and GPT-5.1, so a subscription-billed re-run cannot
reproduce the three-grader design and a Claude-only arm is not comparable to what it must
be scored against. More importantly the stage is second-order. **What replaces it: nothing,
and nothing needs to.** The judge is not claimed as a validated instrument. It is a
diagnostic pass whose outputs are reported as its own un-adjudicated opinions with the
denominator attached to every one, and the paper says so in `sec:m-validation`, in
`sec:r-judge` and in `tab:omission`. The meta-grader stage *was* the attempt to validate the
judge, and it failed by construction. Validating the validator is a second-order rabbit
hole: if human time is ever spent, it belongs on the extraction, not on the judge's opinion
of the extraction. The manuscript already carries the one paragraph this leaves behind.

### S2. Second judge run, stratified on chain-yielding papers - OPTIONAL under the framing decision
**Answers** W1/Q1 — the paper's largest single gap. The judge covers 0.6% of the reported
chain set, so no fidelity number in the paper applies to the unit it reports on. The
Limitations section says so; a stratified run closes it.

**Design.** Sample 100 documents from the 1,868 chain-yielding ones, stratified by source
type to match the chain set rather than the corpus. Same judge, same protocol, so the
result is directly comparable to the existing run.

**Cost.** ~10.3k input per call over 100 papers ≈ 1.0M input ≈ **USD 5–10**.
**Human involvement: none.**

**Unblocked 2026-08-16.** The earlier note that this was blocked on missing source text was
wrong in one direction: `data/raw/ard_json_full/` is indeed absent from this machine, but
ARD is a public HuggingFace dataset that downloads without credentials (runbook R1). What
remains true is that the judge ran as `claude-sonnet-4-5` through the Anthropic **batch**
API, so reproducing it through the subscription CLI changes model version and transport and
destroys the like-for-like comparison that is the study's whole point. **Run it on the batch
API or not at all.** Under the framing decision at the top of this file S2 is optional; if
the team would rather measure than reframe, this is the item to fund.

**Why it is worth more than its price.** It converts every "the verified population is not
the analysed population" sentence into a measured statement, and both reviews price it at
+1 overall. **Confirmed 3/3 by the external round, which now makes this the top structural
item** (S1 having been demoted): Opus calls it "the single change most likely to raise my
score," and all three note it needs no new method.

### S3. Stage-assignment agreement, second model — DONE 2026-08-16 (issue #161, PR #162)
**Answers** W20/Q6, and completes the study already in the paper. The probe in
`sec:r-stages` shows the extractor applies the stage vocabulary at 98.8% internal
consistency; it cannot show a second annotator would agree, because one call wrote both
the text and the label.

**Design.** Take 50 documents' extracted nodes, strip the stage labels, and have a
different-provider model assign a stage to each from the node text alone. Report
Cohen's kappa against the extractor's assignment, and a confusion matrix — the prediction
worth registering in advance is that disagreement concentrates on the pa/ti and dr/im
boundaries, which is where the existing probe's errors already fall.

**Result.** Cohen's kappa **0.838**, raw agreement 87.1% against a 20.4% chance rate, over
653 intermediate-stage nodes from 50 documents (25 chain-yielding, 25 not; kappa 0.835 vs
0.844 by stratum, so no population effect). 0 unusable responses.

**The pre-registered prediction was half right and the paper says so.** Predicted: pa/ti
and dr/im dominate. Actual: dr/im is the largest single confusion (26 of 84), pa/ti is only
7, and the *unpredicted* ti/dr pair is 19. The two predicted boundaries carry 39.3% of
disagreements. The disagreement is concentrated in one stage rather than spread along the
chain: **theoretical insight** is the weakest class (F1 0.756, recall 0.707), bleeding
mostly into design rationale. That is the boundary to reword in any reuse of this schema,
and it is a more useful output than the kappa.

**Cost.** ~98k tokens on subscription auth via the Claude Code CLI (`--safe-mode`, explicit
`--system-prompt`, `ANTHROPIC_API_KEY` stripped from the child environment). **USD 0** —
the projected USD 1–3 assumed metered API, which was not used.

**Still open, and unchanged by this:** two model assignments are not a human anchor. Adding
one human annotator over the same 50 documents (S4/S5) turns it into the three-way study
the reviewer asked for, and only that arm can say whether the five stages are the right
five rather than merely reproducible.

### S10. Edge-coverage reconciliation — DONE 2026-08-16 (issue #156, PR #158)
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

**Result.** The 7.8 was never an omission count: it is `len(coverage list)` from
`experiment_judge_item2_summary.py:144`, and 42.2% of its 777 rows are marked *covered*.
By status: 328 covered / 146 partially / 302 missing = **3.02 missing per paper**, in 90
of 100 papers. Against the 1,667 structural edges the released graph holds for those
papers, **18.1%** — where node-level omission reads 0.6%. So the alarm rested on a
mislabelled quantity **and** the corrected figure is still the paper's largest
unreported coverage signal. In the manuscript: three omission rates in the abstract,
a third row and a unit column in `tab:omission`, the no-`add_edges`-slot explanation
stated rather than implied, and "implied coverage of 99.4%" deleted.

---

## Tier 2 — needs a person, not a budget

### S4. Human-anchored spot-check - DECIDED 2026-08-16: do nothing; use the six graphs as illustration
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

🔴 **Rescoped 2026-08-16, and the earlier estimate was wrong.** Two to four hours for 20
papers assumed skimming. Reproducing a chain honestly means reading the source in enough
depth to say what it argues, which is nearer **3 hours per paper**: 20 papers is three to
four weeks of full-time work, which the project does not have. Treat the 2-4 hour figure as
retracted.

**What to do instead, in order of preference.**
1. **Nothing**, under the framing decision at the top of this file. If the chain set is an
   exemplary analysis rather than a validated sub-corpus, a human anchor on it is not the
   load-bearing evidence a reviewer needs, and the paper already states that no human
   adjudicated anything.
2. **n=3 to 5, one author, as an existence check** rather than a rate. Enough to say whether
   the extraction is recognisable to a domain reader; not enough for a fidelity number, and
   it must be reported as an illustration, never as a rate.
3. ✅ **DECIDED 2026-08-16: do nothing on S4, and use the six team-reviewed graphs as an
   illustration only.** Drive folder `15HQtkJuYNO96a15GM96qEzg9Zf1uZ_yu`.

   🔴 **The provenance question is now answered, and it constrains the use.** From
   `git log --follow` on `intervention_graph_creation/src/prompt/final_primary_prompt.py`:
   the extraction prompt's content **froze on 2025-09-30** (`7526bc0`, `302291e`, `8227d33`,
   then `b9e4bbb` aligning the extractor to the new schema). The only later commit touching
   the file, `b50ef5d` on 2025-10-26, comments out `PROMPT_RESPONSE_EVAL` and does not change
   `PROMPT_EXTRACT`. The team's reviews begin **2025-08-29**, one month and at least four
   revisions earlier — and one of those revisions, `302291e`, changed a structural rule
   ("inhibiting direct risk-intervention connections"). **The reviewers were therefore
   looking at output from a materially different schema than the released corpus.**

   **Usable for:** a qualitative appendix example showing that domain readers trace chains
   this way, with the 2025-08-29 date and the schema difference stated in the caption.
   **Not usable for:** any agreement rate, any fidelity claim, or any number the claim audit
   would check. A rate computed over six graphs reviewed against a superseded schema is worse
   than no rate, because it looks like evidence. Do not compute one, and do not let a future
   session be tempted to.

**Where it is tracked.** This is issue #150's centre of gravity. The ticket is open to
whoever on the team picks it up and is unstarted as of 2026-08-15. It does **not** carry the
chain-yielding sampling change above — read D8 and `paper/TICKET_150_UPDATE_LOCAL.md` before
anyone starts, or the sample lands on the wrong population. I can generate the annotation packet — the 20 papers, their
extractions, a blank verdict sheet and the rubric — so the time spent is judgment only.

### S5. Manual 50-instance error taxonomy — DROPPED 2026-08-16
**Answers** W4. Folds naturally into the same sitting as S4: while the annotator has the
sources open, classify 50 flagged instances by error type and record whether each is a
genuine error. Converts the auto-derived taxonomy over 43 papers from un-adjudicated model
output into something with a human floor under it.

**Dropped, and why.** The scope as written needs the sources: deciding whether a flagged
instance is a *genuine* error means reading the paper, so it inherits S4's cost, not an
hour on top of it. A source-free version exists -- classify the 50 instances by error type
from the judge's own quoted evidence, with no correctness verdict -- but it is much weaker
and it only props up `tab:errorprofile`, which the reviewers already want demoted. If S4
happens at any size the classification comes free with it. **Do not schedule this
separately.**

---

## Tier 3 — real experiments, real budget

### S6 + S8. One ablation experiment — ✅ **ARMS E/F/G DONE 2026-08-16 (issue #165, PR #169)**

**Answers** W6/Q5 (no baseline shows any design choice is load-bearing) and the fidelity
question Limitations names and does not answer. These were two entries; they are one
machine with different arms, they share a document sample and a scorer, and running them
separately would pay the setup cost twice.

| Arm | Question it answers | From |
|---|---|---|
| A full text + reasoning model + seven-stage schema | the released pipeline — already have it | — |
| B abstract only | does full text earn its cost? | S6 |
| C smaller / non-reasoning model | does reasoning earn its cost? | S6 |
| D flat triple extraction, no stage schema | does the schema earn its cost? | S6 |
| E prompt that does not name the five stages | does the structure survive un-prompted? | S8 |
| F sentence-shuffled source | confabulation from topical vocabulary? | S8 |
| G reference-list-only source | the same, harder | S8 |

**Scoring.** E/F/G are scored *structurally* and need no judge: the question is whether a
complete chain still appears and whether the emergent stages map onto the five. B/C/D need
the judge for a quality comparison. Run the structural arms first — they are the cheaper
half and they carry the more interesting claim.

🔴 **In-session feasible, and the caveat is now smaller than it was.** An OpenAI key exists
at `~/0_project_work/ExistentialRiskBenchmark/.env` (runbook R0), so the arms **can** run on
`o3` — the corpus extractor — making arm C a real reasoning-vs-non-reasoning comparison and
every other arm a true ablation of the released pipeline. That is the preferred way to run
it, and it costs metered dollars, so estimate before booking. The subscription-CLI fallback
remains available for D/E/F/G, which are internally controlled claims about the *schema and
the inputs* where every arm shares one extractor; if that fallback is used, the caption must
say the ablation extractor is not the corpus extractor. Arm C cannot be run that way at all
— on Claude it stops being the comparison it is for.

**Budget, computed rather than guessed.** Per document: full text ~5.6k input, abstract-only
~0.6k, reference-list-only ~1.0k, shuffled and schema-blind ~5.6k each. At **n = 30
documents** over arms E/F/G: ~30 x (5.6 + 5.6 + 1.0)k ≈ **370k input**, plus ~2k output per
call x 90 calls ≈ **180k output**. Call it **~600k tokens** and about 90 minutes of
wall-clock at the observed rate. Adding B/D and a judge pass on a 30-document subset roughly
triples it to **~2M tokens and three hours**, which needs chunking across sessions — the
per-batch atomic-save pattern in `experiment_review_stage_agreement.py` is the template.
**Start with E/F/G at n = 30.** If the structure does not survive arm E, that is the single
most publishable negative result available to this paper and B/C/D matter much less.

**Human involvement:** one judgement call on what counts as "the emergent chain maps onto
the five stages". Write it as a rubric for a model and state the rubric in the appendix.

✅ **Result, and it is not the binary either reviewer expected.** Arms E/F/G ran at n=30,
30 and 25 on `o3`, 85 calls, USD within the 5.64-20.22 pre-run band. The **vocabulary
survives un-prompted and the chain does not**: 144 emergent category labels, 138 mapping
onto one of the five stages, against 0.0% of ablated chains carrying all five where the
released extraction carries 65.0%. Shuffled sources yield a chain for 30.0% of documents,
which is the confabulation rate the paper lacked. Reference lists alone yield 2 in 25, with
6 explicit refusals. Arms B/C/D (abstract-only, non-reasoning model, flat triples) are
**still unrun** and are what a "does the design earn its cost" reviewer asks for; E/F/G
answer "is the structure real", which is the question with the fidelity claim attached.

### S6-OLD (superseded by the merged experiment above). Baselines on 200 documents
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

**Not the same as S11.** S7 re-runs the *same* model to get a noise floor; S11 runs
*different* models to get cross-model stability. S11 is in-session and cheap, S7 needs the
`o3` key. If only one happens, S11 answers more of what the reviewers asked.

### S8-OLD (superseded by the merged experiment above). Schema ablation and degraded-source control
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

### S11. Multi-model extraction consistency — ✅ **ARMS A/B DONE 2026-08-16 (issue #168, PR #170)**

**Answers** the rendered gap in Limitations that currently asks a co-author to recover an
n=20 o3 / GPT-5 / Claude-4 check run earlier in the project. That data was produced manually,
outside source control, and is presumed gone. **Do not keep waiting for it — re-run it.**

**Design.** Take 20 documents. Extract each with two or three models under the identical
released prompt, then report node-count and edge-count agreement, stage-distribution
agreement, and whether the same risk-to-intervention endpoints appear. This is the
"single extractor, single run" limitation turned into a measurement, and unlike the original
it will be reproducible.

**Cost.** 20 documents x ~5.6k input x 3 models ≈ 340k input, plus output. Runnable on the
subscription CLI, but **prefer the OpenAI key at `~/0_project_work/ExistentialRiskBenchmark/.env`
(runbook R0)** so one arm is the corpus extractor itself — cross-model stability measured
against `o3` is the claim Limitations needs, and a Claude-only version measures prompt
stability across models that never produced the corpus. Say which was run either way.

**Either outcome closes the gap.** Numbers replace the `\OPEN{}` block; a failed run means
deleting it and keeping the limitation as stated.

✅ **Result: the counts survive a re-run and the gate does not.** `o3` re-extracting
documents it already extracted gives 21.1 nodes and 21.0 edges against the shipped 21.0 and
20.9, with the five stages within two points — but a chain for **9 of 18** where the shipped
run has 18 of 18, because maturity >= 3 lands on an intervention in 11 documents rather than
18. Edge confidence is stable. **The attribute that selects the reporting unit is the
unstable one**, which is the sharpest form of reviewer R21 anywhere in this file.

🔴 **Read it as an upper bound.** The sample is #165's chain-yielding sample, so the shipped
side is 18 of 18 by construction and a re-run can only lose. **The next study on this list
should be the unconditioned arm: 20 documents drawn from the corpus rather than from the
chain-yielding subset, ~USD 2, and it turns this into a symmetric rate.**

Also measured: node *identity* is unstable where counts are not — a shipped risk node has a
counterpart in the re-run 46.5% of the time at cosine 0.80 over name embeddings, 27.8% at
0.85, 19.0% lexically. And `gpt-5` under the identical prompt extracts 1.8x the nodes, 2x
the edges and yields a mean of 97.8 chains per document against `o3`'s 1.4, which is what
denser graphs do to a path enumerator.

### S12. Comparison against existing artifacts — ✅ **DONE 2026-08-16 (issue #166, PR #167)**

**Distinct from S6/S8, which compare us against simpler versions of ourselves.** This
compares the released graph against artifacts that already exist over the same literature —
the AI Safety Graph's clustering of ~5,000 ARD documents, and ARD's own unsupervised
analysis \citep{kirchner2022ard}. The claim it would support is the one the Introduction
makes and never tests: that full-text reasoning extraction recovers something those
document-level artifacts do not.

**Design sketch, needs tightening before it is run.** Take documents present in both. Ask
what our chains assert that a topical clustering cannot express, and quantify it — for
instance the share of our risk-to-intervention pairs whose two endpoints fall in the same
topical cluster, which is where a topic model can say nothing about direction. Cheap, Class
B if the other artifact is downloadable.

**Why it may be worth more than S6.** It answers a reviewer's "why is this needed at all"
rather than "why is this design needed", and it is the only item on the list that engages
the Related Work stack the Introduction leans on.

✅ **Result.** 79 of their topic labels stand against 325 distinct risk-to-intervention
pairs in ours, 4.1 per label. Their clustering asserts 2,986 cross-document co-membership
pairs where ours asserts none, and that half is reported in the same paragraph. Chain yield
on their paper list, 40.4%, independently reproduces the 40.5% arXiv row of
`tab:populations`.

### S9. Retrieval evaluation - needs human relevance labels, lowest priority
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
| C9 **DONE** | **Five stale "42/42" claim-audit references** — `REPRODUCE.md:44`, `NEXT_STEPS_REWRITE_2026-08-14.md:17` and `:369`, `NEXT_STEPS_2026-08-11.md:20` and `:173` — against the manuscript's current 218/218. `REPRODUCE.md` is the file a reviewer actually runs, so the mismatch reads as a broken audit. Not raised by any model (they never saw the repo); found in-session 2026-08-15 | in-session | 5 references confirmed |

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
| C2 **DONE** | "schema-constrained" (×4, incl. abstract and Fig. 1 caption) while `sec:m-extraction` concedes conformance is prompt-enforced with no structured-output constraint. Use "schema-prompted" | 1/3 | 4 occurrences, contradiction confirmed |
| C3 **DONE** | `sec:m-repro` "a re-run reproduces the same model generation" (L627) reads against `sec:m-extraction` "runs are not bit-reproducible" (L339). Both render; the first invites the wrong reading | 1/3 | both lines confirmed |
| C4 **DONE** (now "as run") | `tab:gates` marks the maturity-$\geq 3$ row "(deployed)" while the rubric reserves *deployed* for maturity 4. The intent is "the deployed setting"; relabel so it cannot be read as the maturity band | 1/3 | confirmed |
| C5 **DONE** (25 moved to `paper/refs_verification_notes.md`) | `refs.bib` carries 29 non-standard annotation fields, including "Verified 2026-08-15" and per-entry mini-summaries. Move to a source-verification file | 1/3 | 29 fields |
| C6 **DONE** (23 of 441, 5.2% now printed) | `sec:m-recovery` says the judge cannot recover failed extractions "at a useful rate" with no number in the body. The figure exists (23 of 441, 5.2%) — print it or cut the sentence to "failed extractions are corpus loss" | 2/3 | no number in body |
| C7 **DONE** | "One call per document" vs a client that retries up to three times — distinguish logical requests from API attempts | 1/3 | `max_retries=3` in Methods |
| C8 **DONE** (PR #159: 0 orphans / 0 dangling / 0 self-loops / 1 duplicate edge; the judge's classes were pre-ingest and do not reach the dump) | The release's defect status is undocumented: the judge found 108 referential-integrity findings, 42 orphans and 56 duplicate pairs, and no repaired graph was rebuilt. State whether the released dump carries them | 2/3 | consistent with `app:judge` |

## Language, register and typography — the low-hanging fruit

Every reviewer at both bars flagged writing style, and until 2026-08-15 PM none of it was
recorded here — it lived only in the review files. Collected below so it can be worked down
rather than re-read. Instances are quoted so they are greppable in `paperA_altstyle.tex`.
None of this requires a decision; all of it is a prose pass.

| # | Item | Agreement | Action |
|---|---|---|---|
| L1 (team, G15) | **`\author{Author List Placeholder}` (L102) renders on page 1** — a *ninth* visible placeholder, separate from the eight `\OPEN{}` blocks of C1 | 1/3 | goes with G15; until then it is the first thing a reviewer sees |
| L2 **DONE** | **Aphoristic paragraph-enders.** "We add the layer under it."; "A corpus of $N$ short chains is exactly $N$ components until something links them."; "A faithful extraction from a weak paper is a successful extraction."; "The objective above is what the step is for; the counts below are what this approximation to it produced."; "The counts require no control, being direct tallies." | **3/3** | the internal reviews counted eight and said keep two. Opus: "deployed forty times they read as generated polish and displace information" |
| L3 **PARTIAL** | **Contrastive correction** — "X is not Y", "read this as A and never as B". ~30 instances by the internal count, "well over a dozen" by Opus's | **3/3** | keep where a plausible misreading exists *and* the paper has evidence about it; target under 8 |
| L4 **DONE** | **"honest" as editorial** — "the honest statement of yield", "the honest positive residue" | 2/3 | GPT-5.6 Sol: "implicitly characterizes alternative summaries as dishonest" |
| L5 **DONE** | **Promotional / advocacy** — "The corpus is a snapshot of one dataset and will date. The paired extract-and-verify design will not."; "would make research coordination ... tractable"; "the natural agentic use"; "The extension this work most needs"; "What makes a mechanism layer worth building" | 2/3 | "will not [date]" is unsupported and absolute — models, prompts and schemas date too. "tractable" → "could support" |
| L6 **DONE** | **Conversational / blog register** — "What the release contains. Five things:"; "A closing note: some statistics are true by design."; "All three duly record an improvement"; "the reader who takes the release and does something with it" | 2/3 | "duly" reads as sarcasm |
| L7 **DONE** (7 of 11; the rest are ordinary prose) | **Legalistic meta-formulations** — "what licenses reading the other rows"; "which is what settles it"; "A reader would otherwise misread a number" | 1/3 | state the assumption and its implication directly |
| L8 **DONE 2026-08-16** | **Formulaic openers** — "Three things follow"; "Two properties bear on"; "What this does not show" | 1/3 | frequency is the tell, not any single instance |
| L9 **DONE** | **Over-attribution of importance** (the flag Martin asked reviewers for) — "The verification stage is half the contribution" (`app:judgeprompt`); "what makes the extraction checkable rather than merely large" (`sec:r-judge`); "This is the single licensing gate"; "That qualifier is essential" (`sec:r-corpus`); "The single most consequential row is the sixth" (`tab:populations-master` caption); "the choice of extractor moves the bill by about a factor of five — more than any other decision in the pipeline" (`sec:m-repro`); "The sharpest is a merge-manufactured centrality hub at 90x" (Conclusion); "218 of 218 numeric claims passing" as a quality badge; "fifty documents ... would settle it"; the two worked queries as "the precondition for the cross-paper analysis" | **3/3** | the verification-stage claims are the load-bearing ones: the stage ran on 0.85% of documents and 0.6% of the analysed chains. The 90x hub in the Conclusion is an artifact of a step **not applied** to the released substrate, elevated to a headline |
| L10 **PARTIAL** | **Same three caveats repeated across 6–8 sections** (verified ≠ analysed; yield is a gate property; completeness is schema-filling) | 2/3 | one clear statement each plus cross-references |
| L11 **DONE** | **Acknowledgments carry project-management detail** (Discord stand-ups, working threads) | 1/3 | not scholarly acknowledgment |
| L12 **PARTIAL** (abstract + contribution bullet say *audit*; the defined term "verification stage" is unchanged, and renaming it is all-or-nothing) | **Terminology overstates the evidence** — "verification", "implied coverage", documents "argue a complete mechanism" | 1/3 | → "the extractor produced a chain judged to pass the model-assigned gates"; "auditable" or "subject to an LLM diagnostic pass" rather than "verified" |
| L13 **DONE** (see also the sentence-length note below) | **Mechanical sweeps** (from the internal reviews, not re-raised externally): mixed British/American spelling — "randomisation", "neighbourhood", "specialised", "favourable" against "normalization", "labeling", "colored"; number-words inconsistent — "Twelve of the 100" vs "12 of the 100"; `\emph{}` 40+ times, mostly on ordinary words | internal | one spelling variety, one number rule, `\emph{}` for term introductions only |
| L14 🔴 **STRIP BEFORE RELEASE — last action before posting** | **Source-file editorial trail** — "REMOVED 2026-08-14", "Moved out of sec:r-hub", "the frozen Overleaf reported...", "the module docstring says 80%, the code uses 70%", the compute-donor gate block. Several disclose internal disagreement, an Overleaf workflow and a private donor | internal | strip or move to a NOTES file before any public posting. **Not** the same as C1: these are comments and never render |

**Sentence length, measured 2026-08-16.** Mean 20.6 words over 588 body sentences; 33
exceed 40 words (5.6%) and 12 exceed 55. That is ordinary for this kind of paper and a
blanket split would add length to a draft that must lose three pages. 🔴 Only **two** were
genuine offenders, not the five a first pass reported: the sentence splitter breaks on
LaTeX, so the abstract's apparent 82-word sentence is three sentences separated by `(1)` and
`(2)`, and the 69-word practitioner sentence is two separated by a question mark inside
`\emph{}`. The two real ones — the nine-constraint sentence at 91 words and the
edge-confidence rubric at 69 — are split. **Do not re-run a naive splitter and conclude
there are five.**

**Prose pass, 2026-08-16.** Seventeen targeted edits, listed in the manuscript's own
source comments. Fully done: the three named aphoristic enders (L2), both "honest"-as-
editorial instances (L4), the five promotional phrases including "will not date" (L5),
the blog-register openers (L6), all ten over-attribution instances the reviewers named
(L9), the Acknowledgments project-management sentence (L11), and one spelling variety
with `\emph{}` left alone (L13). Partial: L3 contrastive corrections were reduced where
the rewrite touched them but were not counted down to the under-8 target; L7 and L12
were reduced at the instances the reviewers quoted, not swept; L10's repeated caveats
are fewer after the cuts but not consolidated to one statement each. Not done: L8
formulaic openers, and L14 **on purpose** -- the source-comment trail is what stops a
future session re-deriving a corrected number wrongly, and it never renders. Strip it in
one pass immediately before posting, not now.


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
| D2 **RESOLVED 2026-08-16: cut** | Cut vs run S1 | — | 3/3 reviewers, and two of the three graders are unreachable on subscription auth. Nothing replaces the stage and nothing needs to: the judge is a diagnostic pass, not a validated instrument. See S1 |
| D3 **DISSOLVED 2026-08-16 — there was no decision to make** | Ungated vs gated release as the primary unit | — | The primary delivery is the **raw graph**; every chain set is an exemplary analysis on top of it. The quality gates were therefore never a property of the release to choose between, and the question was malformed rather than open. `sec:m-repro` now states what ships and argues nothing: the dump carries maturity on every node and confidence on every edge, so any gate setting is reproducible from it. No convenience file and no CLI flag are promised or needed (R7 retired). Manuscript `541b022` |
| D4 **RESOLVED 2026-08-16: MIT + CC-BY-4.0** | Our own licence pair | — | = C1 row 2. **Narrowed 2026-08-16**: ARD is published under MIT (dataset card, verified; now cited as `stampyai2023ardataset`), so our use of the collection is unambiguously permitted and the manuscript says so. The card is silent on the terms of the individual documents ARD aggregates, which is why we release derived structure and not source text. What is left is picking our pair -- MIT for code, CC-BY-4.0 for the derived data is the natural one -- which is a decision, not a question of fact. **Taken 2026-08-16**: MIT for code, CC-BY-4.0 for the derived data, so anyone may reuse the framework subject only to the terms their own sources impose. Written into `sec:m-repro`; one rendered gap closed |
| D5 | **Release hosting + URL** | team | = C1 row 1. Blocks C1 and every reviewer's first question |
| D6 | Compute-donor consent (G14), author list + contribution statement (G15), AI-drafting scope | team | 🔒 **Detail in `NEXT_STEPS_PRIVATE.md`** — these three are tracked there, not here, and they gate four of the eight `\OPEN{}` blocks |
| D7 | Co-author coordination: draft send, the outstanding contribution question, the #150 refresh (D8), PR #151 (#149 was closed unmerged) | team | 🔒 **Named detail in `NEXT_STEPS_PRIVATE.md`** — who owes what stays off the remote, per the `paper/` gitignore policy |
| D8 **CLOSEABLE 2026-08-16** | Issue #150 | — | All five open items are resolved or retired. Human-anchored spot-check → S4 "do nothing". Manual 50-instance taxonomy → S5 dropped. Re-run Gemini over all 100 → dies with D2. Edge-coverage reconciliation → done (#156 / PR #158). Confirm the third meta-grader's model id → **resolved without a human**: its output records no model, so the manuscript no longer asserts one (`ed016c9`). The remaining co-author read of `sec:m-validation` / `sec:r-judge` is not tracked as a ticket item by decision. **Action: post the dispositions as a closing comment and close #150.** The draft in `paper/TICKET_150_UPDATE_LOCAL.md` predates all of this and should be replaced by that comment, not sent |

---

## Not a study, but still owed: the body must reach 10 pages

**This is a hard requirement regardless of venue, and it is not yet met.** Ten pages
excluding references is the ceiling at conferences as well as workshops, so there is no
target under which the current draft fits.

🔴 **MEASURED 2026-08-17 on the compiled PDF, and the estimate below was optimistic by two
pages.** `AISafetyIntervention_PaperA_shared_long.pdf` is 27 pages; the body runs pages 1 to
15 with References beginning 89% down page 15, so the main body excluding references is
**~14.9 pages and the gap is ~5 pages, not ~3.** The scoped, agreement-ranked cut package
that closes it lives in `paper/section_cut_list.md`, built from the 2026-08-17 six-review
round — work the cuts there, not here.

Earlier estimate, superseded: body after the 2026-08-15 pass **~12.5–13 pages**, down from
~15–16. That estimate was a character-count heuristic, not a build.

🔴 **The 2026-08-16 pass did not close this gate, and the arithmetic says why.** Measured on
non-comment source lines at commit `337d033` against `06c3440`: the appendices lost **90
lines** to the unanimous cuts, and the body **gained 17** because the three new findings
(edge coverage, what the collapse drops, what the release ships) had to be stated. Net for
the whole file: −73 lines, roughly −1.7% of body characters. Every cut on the list below
that a reviewer agreed on has now been taken, so the remaining ~3 pages have to come from
material the reviewers wanted **kept** — which makes it an author's call, not an editorial
one. The realistic options are the four in the ordered list below, and the first is the
only one that yields a page on its own.

🔴 **The 2026-08-16 PM pass made the gate worse, deliberately, and the number is measured.**
Non-comment characters from `dbf88fd` to `ef5c23f`: **+3,686 characters, +60 lines**, about
+2.8%, call it three quarters of a page. What bought it: the ablation paragraph and
`app:ablation` (#165), the Related Work measurement (#166), the undirected-traversal
sentence and `tab:cuts` row. The language pass (L3/L10/L12) gave back roughly a fifth of
that; the rest is new evidence.

**This is the right trade and it still has to be paid for.** Every one of the four reviewer
rounds asked for a baseline and a fidelity control, and the paper now has both. But it is
now ~13 pages against a 10-page ceiling, so the page budget is the single largest unclosed
item on this list and it is now an author's call. The candidates, in the order I would take
them: the three artifact use cases to an appendix (item 1 below, ~110 lines), `sec:r-selection`
compressed to its control paragraph, and `app:race` cut to its two-line prevalence. None of
those touches the new evidence.

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
| ~~Pre/post repair scoring + the agreement-instrument appendix~~ **CUT 2026-08-16** | **3/3** | done: `sec:r-judge` keeps one paragraph naming the confound and the null-repair control; ICC / ICC(2,k) / Krippendorff / two Fleiss binnings / median split all deleted, receipt still ships |
| ~~`sec:r-selection` race-framing + `app:race` odds-ratio table~~ **CUT 2026-08-16** | **3/3** | done: three paragraphs to one, `tab:race` gone, the 2.7-5.2 OR range kept in a clause, classifier precision kept |
| ~~Everything quoted from the earlier internal substrate~~ **CUT 2026-08-16** | **3/3** | done: `tab:clustering-methods` and `tab:dedup-thresholds` deleted, `app:sensitivity` hop counts reduced to the growth ratio. `sec:m-repro`'s carve-out now names only the merge sweep and that ratio |
| ~~`app:clusters` 40-cluster name list~~ **CUT 2026-08-16** | 2/3 | done: size range plus four examples; the full list ships with the release, which also closed one `\OPEN{}` block |
| `sec:m-recovery` + the 441-row of `tab:populations-master` | 2/3 | **partly done**: the number is printed (C6) rather than the section cut. Cutting to one clause is still available if the page budget needs it |
| ~~Meta-grader agent-session operational detail~~ **CUT 2026-08-16** | 1/3 | done: the consequence for the denominators is kept, the JSON-shape counts are gone |
| ~~The "218 of 218 numeric claims" audit narrative~~ **CUT 2026-08-16** | 1/3 | done: the body prints no count; the audit figure survives only in source comments and `REPRODUCE.md`, and is re-run rather than quoted |

**Explicitly keep**, named by every model that raised the topic: the merge/centrality
artifact (`sec:r-hub`), `tab:gates`, `tab:populations-master`, the source-type skew table,
and the Euclid failure case (`app:failure`) — Gemini calls the last one worth a slide by
itself. Opus adds a closing warning worth weighing against the whole cut list: with so many
numbers labelled "not evidence of anything wider", it becomes hard for a reader to say what
the paper *does* establish, and a draft stating two or three defensible claims and cutting
the rest would be stronger, not less honest.

## Reviewer-comment register — every fixable comment, one row each

Built 2026-08-15 PM by stepping through all six external review files line by line. **This
is the master; the S / C / L / cut entries above are the scoped subset.** Every row is one
distinct comment. Rows already scoped elsewhere carry a cross-reference; rows marked **NEW**
were not captured anywhere until this pass. Source codes: **OC/OW** = Opus 5
conference/workshop, **GC/GW** = GPT-5.6 Sol, **MC/MW** = Gemini 3.1 Pro.

Work them down by editing this table's Status column. Strengths and praise are not
registered — only things that change the paper.

### A. Submission-blocking

| # | Comment | Src | Maps to |
|---|---|---|---|
| R1 | Eight `\OPEN{[GAP:]}` blocks render in the PDF | all 6 | C1 |
| R2 | `\author{Author List Placeholder}` renders on page 1 | GW | L1 |
| R3 | Release URL absent — the central contribution cannot be assessed at review time | all 6 | C1, D5 |
| R4 | Licence unnamed; ARD redistribution terms unresolved | all 6 | C1, D4 |
| R5 | Four GAP blocks are notes addressed to co-authors, not manuscript content | OC OW GC GW | C1 |
| R6 | Appendix K's GAP asks to publish cluster representatives "so reviewers can audit the cluster naming" — i.e. the audit the authors themselves consider necessary was not enabled | OC | **NEW** |
| R7 | Provide an anonymised artifact URL during review, not just at camera-ready | GC | **NEW** |
| R8 | Ship the artifacts needed to reproduce the intake counts (the failure directories are not in the release) | GC | **NEW** |

### B. Evidence gaps

| # | Comment | Src | Maps to |
|---|---|---|---|
| R9 | No human validates any node, edge, chain, maturity label, confidence label or judge verdict | all 6 | S4 |
| R10 | Human anchor should cover node-level precision *and* recall against human annotation | OC | S4 |
| R11 | Human anchor needs inter-annotator agreement or it is one person's opinion | OC GC | S4 |
| R12 | Human anchor should capture source spans / edge-level grounding | GC GW | S4 |
| R13 | Prefer 50–100 chain-yielding documents over 20 | GC | S4 |
| R14 | Verified population ≈ disjoint from analysed population (12/100, 17/2,772) | all 6 | S2 |
| R15 | Second judge run stratified on chain-yielding papers — needs no new method | OC OW GC | S2 |
| R16 | 0.6% vs 28.8% omission discrepancy unreconciled; paper reports no usable fidelity number | all 6 | S4, S10 |
| R17 | **Judge flags a mean 7.8 missing relationships/paper against a mean 10.8 extracted edges — edge recall may be poor while node recall looks high** | GC GW | S10 |
| R18 | Report the coverage list as covered / partially covered / missing, and say whether those were manually inspected | GW | S10 |
| R19 | Explain why the abstract emphasises 9 added nodes rather than the edge-coverage findings | GW | S10 |
| R20 | The repair schema had no add-edge slot — state that as the explanation rather than leaving it implicit | GC | S10 |
| R21 | Both gates (edge confidence, intervention maturity) are unvalidated model self-assessments; sensitivity is not validation | all 6 | S4, D3 |
| R22 | The 70% containment rule ignores edge identity, order and semantics; no annotation study shows retained paths are distinct arguments | GC GW | **NEW** |
| R23 | Validate the gates and the collapse rule on a small hand-checked sample | GW | **NEW** — **ADDRESSED 2026-08-26**: `experiment_review_collapse_adjudication.py`, 120 displaced (kept, dropped) pairs + 15 mismatched-document nulls, batch `msgbatch_01SRNWJKUcan5ukvuQoRHiE7`, USD 3.83. Judge rather than hand, and the null arm is what makes it readable |

🔴 **The register missed the sharpest methodological comment in the external round, and it
sat unregistered for eleven days.** The rows below are from the **2026-08-17** round; the
register was built on 2026-08-15 and only the *cut list* was re-swept afterwards, so the
newer round's evidence comments were never entered. GPT-5.6 Sol gave the paper its lowest
scores and attached an explicit score impact to this one: *"a corrected analysis would
improve Quality and Clarity. Retaining the current interpretation would reinforce my
borderline-reject assessment."* Four of its five instructions were in fact already
satisfied; two were not, and one of those needed a study nobody had scoped.

| # | Comment | Src | Status |
|---|---|---|---|
| R23a | "The current 0.6%, 18.1%, 28.8%, 26.4% and 21.7% quantities divide proposed missing items by the size of an existing graph, even though **the numerator and denominator are produced by different instruments**" — stop calling them omission rates unless a common universe is defined | GC GW | **PARTLY DONE 2026-08-26.** The prose caveats landed (`9583f77`, `a52e32c`); `tab:omission` still carries the word *omission* in its name and the section heading still says "Three omission measurements". Rename both, or say once in the caption what the table is a table of |
| R23b | **"Distinguish additions per extracted item from recall"** | GC GW | **THE STUDY THAT DID NOT EXIST.** Prose now says the rates bound material and not arguments. The measurement is `experiment_review_chain_recall.py`, batch `msgbatch_01S117PieDEXocGorph6d5RB`, 127 documents, USD 3.22, with a 20-document ablation arm giving it a measured sensitivity |
| R23c | "'The three sort by instrument rather than by unit.' **Opaque.** Explicitly define the numerator, denominator, and instrument" | GC | ✅ **DONE 2026-08-26**, manuscript `b11f7ca` |
| R23d | Explain why the second judge run marks 95 of 97 documents for node additions | GC | ✅ already done — `sec:r-judge` names document length and the rationale-field-free rebuild, and says the comparison bounds nothing |
| R23e | Avoid describing any quantity as an upper bound, lower bound, direction or floor without assumptions that justify it | GC GW | ✅ already done — the paper explicitly disavows a floor and a direction for the two-run comparison |
| R23f | "The expected-list statuses sum to 776, although the paper says there are 777 rows" | GW | ✅ **reviewer error, no change.** There is a fourth status: one row the judge marked covered only in the abstract. 328 + 146 + 302 + 1 = 777, and `sec:r-judge` already names it. The audit checks the sum |
| R24 | Report sensitivity of the substantive retrieval examples to the gates, not only aggregate counts | GW | **NEW** |
| R25 **CLOSED** | Stage probe is circular — one call wrote both text and label; TF-IDF on the name alone reaches 69.4% | OC OW GC GW | S3 done: kappa 0.838 across providers |
| R26 | No baseline: flat triples, abstract-only, non-reasoning model, sentence-level argument mining, retrieval over chunks | all 6 | S6 |
| R27 | Baselines should demonstrate *why a reasoning model is necessary* for this schema | MC | S6 |
| R28 | Retrieval use case rests on two hand-picked arXiv examples; no query set, relevance judgements or faithfulness evaluation | all 6 | S9 |
| R29 | Schema ablation (prompt without the five stages) not run | OC OW GC GW | S8 |
| R30 | Degraded-source control (sentence-shuffled / abstract-only / reference-list-only) not run | OC OW GC GW | S8 |
| R31 | Add an out-of-scope-documents arm (problem/solution structure, non-safety) to the ablation | GC | S8 |
| R32 | No repeat-extraction run; descriptives sit on an unmeasured noise floor | OC OW GW | S7 |
| R33 | Meta-grader denominators are run artifacts (95/95/13; taxonomy 43/100) — re-run on one fixed schema or drop the taxonomy | GC GW | S1 note, #150 change 3 |
| R34 | Prime-number chain passes every filter *including* confidence ≥3 and maturity 4 — the gates do not reject a clear negative | GC GW | **NEW** |
| R35 | How prevalent is the Euclid-style invented framing in the corpus at large? | MC | S8 |
| R36 | Consider a safety-relevance pre-filter classifier to stop the schema hallucinating relevance | MC | **NEW** |
| R37 | Does the similarity layer have any use case that is not confounded, or should downstream users be told structural-edges-only? | MW | **NEW** |

### C. Factual and internal-consistency corrections

| # | Comment | Src | Maps to |
|---|---|---|---|
| R38 | "schema-constrained" vs the paper's own "prompt-enforced, no structured-output constraint" | GC GW | C2 |
| R39 | "a re-run reproduces the same model generation" vs "not bit-reproducible" | GC GW | C3 |
| R40 | `tab:gates` labels the maturity-≥3 row "(deployed)" though the rubric reserves that for maturity 4 | GC GW | C4 |
| R41 | `refs.bib` carries non-standard annotations, mini-summaries and "Verified" notes | GC GW | C5 |
| R42 | Check for future-dated publication/retrieval entries | GC | **checked, clear** |
| R43 | "One call per document" vs a client that retries three times | GC | C7 |
| R44 | Recovery result quoted as "cannot at a useful rate" with no number in the body | OC OW GC GW | C6 |
| R45 | Judge found 108 referential-integrity findings, 42 orphans, 56 duplicate pairs; no repaired graph was rebuilt — does the release ship them? | GC GW | C8 |
| R46 | Users need a way to distinguish audited from unaudited content in the release | GW | **NEW** |
| R47 | Output-token and cost figures extrapolated from a single surviving response, reasoning tokens assumed — label as estimate and move to appendix, or give actual invoices | GC | **NEW** |
| R48 | Several failure counts come from an earlier substrate or unreleased directories | GC | cut list row 3 |
| R49 | Appendix tables from the earlier substrate are mixed with released-substrate tables, the distinction relegated to captions | OC | cut list row 3 |

### D. Overstated claims to soften

| # | Comment | Src | Maps to |
|---|---|---|---|
| R50 | "The verification stage is half the contribution" | OC OW | L9 |
| R51 | "what makes the extraction checkable rather than merely large" | OC OW GC GW | L9 |
| R52 | "builds and verifies this missing layer" — it is an audit/diagnostic stage, not a verified corpus | GC | L12 |
| R53 | Use "auditable" / "subject to an LLM diagnostic pass" rather than "verified" | GC | L12 |
| R54 | "implied coverage of 99.4%" should go if the edge findings stand | GW | S10, L12 |
| R55 | Documents "argue a complete mechanism" → "the extractor produced a chain judged to pass the model-assigned gates" | GC | L12 |
| R56 | "This is the single licensing gate" | OC OW GC | L9 |
| R57 | "That qualifier is essential" | OC | L9 |
| R58 | "The single most consequential row is the sixth" (table caption) | OC OW GW | L9 |
| R59 | "Two constraints deserve naming here rather than only in the table" | OC OW | L9 |
| R60 | "the choice of extractor moves the bill by about a factor of five — more than any other decision in the pipeline" | OC GC GW | L9 |
| R61 | "The sharpest is a merge-manufactured centrality hub at 90×" in the Conclusion — an artifact of a step not applied to the released substrate | OC OW GC GW | L9 |
| R62 | "218 of 218 numeric claims passing" reads as a quality badge; it shows manuscript consistency, not validity | GC GW | L9, cut list |
| R63 | "fifty documents would settle it" — would inform, not settle | GW | L9 |
| R64 | The two worked queries called "the precondition for the cross-paper analysis of §6.2" — frames an unevaluated demo as foundational | OW | L9 |
| R65 | "would make research coordination and the search for under-addressed pairs tractable" → "could support" | GW | L9 |
| R66 | "The corpus will date. The paired extract-and-verify design will not." — unsupported and absolute | GC GW | L5 |
| R67 | "the weakest point in the evidence" / "the clearest reason the protocol requires a human anchor" | OC | L9 |
| R68 | Structural completeness (87.4%) and the stage probe (98.8%) are near-vacuous yet carry the abstract, §1.1, §3.1 and the Conclusion — remove from the abstract | OW | **NEW** |
| R69 | §4.4/§4.5 give outsized prominence to debunking an internal artifact, with deep statistical tables | MC MW | cut list |
| R70 | "Two controls are needed and their order matters" — pedantry about a manufactured artifact | MW | L9 |
| R71 | Compress the merge artifact substantially unless graph-analysis controls are elevated to a stated secondary contribution | GW | **NEW** |

### E. Cuts

| # | Comment | Src | Maps to |
|---|---|---|---|
| R72 | Pre/post repair scoring → two sentences in Limitations | all 6 | cut list 1, S1 |
| R73 | Delete the ICC / ICC(2,k) / Krippendorff / two-binning Fleiss / median-split paragraph | OC OW GC | cut list 1 |
| R74 | Race-framing §4.5 → one paragraph carrying the general control | all 6 | cut list 2 |
| R75 | Cut `app:race` except the two-line corpus prevalence | OC OW GC GW | cut list 2 |
| R76 | Table 15 odds ratios belong in the release, not the paper | OC | cut list 2 |
| R77 | Delete `tab:clustering-methods` (Table 13); keep the same-space table | OC OW GC | cut list 3 |
| R78 | Cut the first four rows of `tab:dedup-thresholds` (Table 11) or re-derive them | OW | cut list 3 |
| R79 | Cut `app:sensitivity` similarity-hop counts | OC OW GC | cut list 3 |
| R80 | Consolidate the recurring "earlier internal pass / 8.5× sparser layer" thread into one footnote | OW | cut list 3 |
| R81 | Cut `app:clusters` 40-name list to the size range plus 3–4 examples | OW GC GW | cut list 4 |
| R82 | Cut §2.2.2 recovery experiment and the 441-row of the populations table to one clause | OC OW GC GW | cut list 5, C6 |
| R83 | Cut the meta-grader agent-session operational detail (13 JSON shapes, folder-agent behaviour, uneven denominators) | GC GW | cut list 6 |
| R84 | Cut the "218 of 218" audit narrative from the main text | GC GW | cut list 7 |
| R85 | **Cut the §2.5 non-monotonicity aside** ("0.90 keeps 5,460 where 1.00 keeps 5,427") — an artifact of a greedy heuristic that changes no reported number; keep the 0.60/0.70/0.90 row | OC | **NEW** |
| R86 | **Cut the narration of how the deployed pipeline came to compute an unsound silhouette** — state the correct comparison and the conclusion only | OC | **NEW** |
| R86b | **Cut the debug-history narrative from §4.4 too** (what the pipeline *used* to do), keeping the finding. Suggested replacement, verbatim: "Naive node deduplication using transitive closure over similarity thresholds manufactures artificial centrality hubs (see App H). Therefore, our released dataset and primary analyses avoid this by..." — then present the correct results. Note the tension with R121: keep the artifact, cut the story of how we hit it | MW | **NEW** |
| R87 | Move the speculative output-token/cost reconstruction to an appendix or label it explicitly as an estimate | GC | R47 |

### F. Language and register

| # | Comment | Src | Maps to |
|---|---|---|---|
| R88 | Aphoristic paragraph-enders, ~8 instances | OC OW | L2 |
| R89 | Contrastive "X is not Y" / "read as A never B", ~30 instances | OC OW GC GW | L3 |
| R90 | "the honest statement of yield" / "the honest positive residue" | GC GW MW | L4 |
| R91 | "All three duly record an improvement" — "duly" reads sarcastic | GC GW | L6 |
| R92 | "We add the layer under it." | OW GW | L2 |
| R93 | "What the release contains. Five things:" — blog register | MW | L6 |
| R94 | "A closing note: some statistics are true by design." | MW | L6 |
| R95 | Second-person / instructional register ("A reader reusing the release then knows...") | OC | L6 |
| R96 | Confessional first-person process narration ("we abandoned the attempt", "we draw nothing from that", "we did not run it") | OC | L6 |
| R97 | Internal-memo tone in §3.2 ("Scoring the repairs is a design lesson, not a result...") | MC | L6 |
| R98 | Legalistic meta-formulations ("what licenses reading the other rows", "which is what settles it", "a reader would otherwise misread a number") | GC | L7 |
| R99 | Advocacy phrasing ("The extension this work most needs", "What makes a mechanism layer worth building", "the natural agentic use") | GC | L5 |
| R100 | Formulaic openers ("Three things follow", "Two properties bear on", "What this does not show") | GC | L8 |
| R101 | The same three caveats repeated across six to eight sections | OW GC GW | L10 |
| R102 | The "Use of AI Assistance" section is itself imprecise about its own scope | OW | C1, L12 |
| R103 | Appendix K ends in a bracketed instruction printed to the reader | OW | C1 |
| R104 | Acknowledgments carry Discord/stand-up project-management detail | OC | L11 |
| R105 | Mixed British/American spelling; inconsistent number-words; `\emph{}` on ordinary words 40+ times | internal | L13 |
| R106 | Source-file editorial trail (REMOVED/Moved-out comments, frozen-Overleaf notes, donor gate block) | internal | L14 |
| R107 | Reads as an internal audit log rather than a finished article | GC GW | L6, cut list |

### G. Release, licensing and ethics

| # | Comment | Src | Maps to |
|---|---|---|---|
| R108 | Name the licence pair and document the legal basis per source type | GC GW | C1, D4 |
| R109 | The unresolved licence is a limitation with legal exposure — state a fallback position rather than deferring | OW | D4 |
| R110 | Provide a documented correction/removal procedure | GC | Impact Statement (present; confirm it is in the release docs too) |
| R111 | Distinguish unverified model assertions from source quotations in the documentation **and in any user interface** | GC | **NEW** |
| R112 | Misattribution rate to named authors is unknown because no human validates extractions | GW | S4 |
| R113 | ARD's own selection bias (who curates, what it excludes, English-only) is inherited by every number and is not discussed | OW | D-note |
| R114 | Does the release let a reuser re-enumerate at any gate setting from the dump, or only consume the two path files? | OW | **NEW** |

### H. Structure, length, framing

| # | Comment | Src | Maps to |
|---|---|---|---|
| R115 | Body is far too long and repetitive for its central contribution | GC GW | page budget |
| R116 | Eleven shifting denominators load the reader | GC | `tab:populations-master` (done) |
| R117 | Main text is not readable standalone — 30+ forward references, several carrying the actual evidence | internal | page budget |
| R118 | Make the ungated chain set the primary release unit, gates as a user-side filter | OC OW | D3 |
| R119 | Over-hedging makes it hard to say what the paper *does* establish; state two or three defensible claims and cut the rest | OW | page budget note |
| R120 | Corpus stops at 2023 — caps significance; a 2024–2026 slice would change it | all 6 | "deliberately not on this list" |

### I. Explicit keeps — do not cut these while working the list

| # | Item | Src |
|---|---|---|
| R121 | The merge/centrality artifact and its EC table (compress, do not remove) | OC OW GC GW |
| R122 | The gate-sensitivity grid `tab:gates` | OC OW GC |
| R123 | The populations table `tab:populations-master` | OC OW GC |
| R124 | The source-type skew table | OC |
| R125 | The Euclid/prime-number failure case `app:failure` — "worth a slide by itself" | OC OW GC GW MC |
| R126 | The same-space silhouette comparison (one clean version) | OC GC |

## What is deliberately not on this list

- **Extending the corpus past 2023.** Both reviews note ARD stops at 2023. Running the
  pipeline over a 2024–2026 arXiv safety slice is now costed by S1's sibling measurement:
  at USD 32–118 per 1,000 documents, a 5,000-document slice is **USD 160–590**. That is a
  scope decision for the team, not a study.
- **MIT risk-anchoring of a chain subset.** Paper B material, deliberately out of scope.
- **Anything requiring the frozen co-author substrate.** Three appendix tables still come
  from it; re-deriving them on the released graph is bookkeeping, not a study, and is
  tracked in the manuscript's own carve-out in `sec:m-repro`.
