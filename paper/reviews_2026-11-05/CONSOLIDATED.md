# Review round 2026-11-05, manuscript v4 (tag `paperA-v4`)

Input: anonymized review copy built by `paper/make_review_copy.py`, 27 pages, body ends on page 8.

| Reviewer | Input tokens / job | Notes |
|---|---|---|
| `claude-opus-5-5`, effort high | 88.9k | Prompt flag 4 still said "body is 15 pages, must reach 10" |
| `gpt-6.1-sol`, effort high | 43.0k | Flag 4 corrected to "about 8 of 10 pages"; run with `REVIEW_OPENAI_KEY_VAR=newopenai_api_key` (the first key had no credits) |

Flag 4 is not scored, so the scores of the two reviewers stay comparable.

## Scores against the 2026-08-17 round (v3)

| Reviewer | Conference | Workshop | Conference Q / C / S / O |
|---|---|---|---|
| Opus 5.5, v4 | 3 | 4 | 2 / 2 / 2 / 2 |
| GPT-6.1 Sol, v4 | **2** | 4 | 2 / 3 / 2 / 2 |
| Opus 5, v3 | 3 | 4 | 2 / 2 / 2 / 3 |
| GPT-5.6 Sol, v3 | 3 | 4 | -- |
| Gemini 3.1 Pro, v3 | 5 | 5 | -- |

Reviewer models changed between rounds, so no row is a paired comparison. Quality is 2 in every
review and is set by fidelity evidence. GPT-6.1 Sol's conference drop to 2 cites the two
soundness inconsistencies under B below; it says a reconciliation "would remove a substantial
soundness concern".

Human-time decision (2026-11-05): no further human annotation. Requests that need it are listed
under D and not pursued.

## A. Claim and wording changes (session work, no new data)

| # | Change | Raised by |
|---|---|---|
| A1 | Gate described as admitting "links the source evidences" (sec:m-extraction, app:rubrics, tab:cuts "source-evidenced"): rescope to what the prompt instructs; contradicted by 30% shuffled-source yield, the Euclid case, C27 | all four |
| A2 | Post-cutoff claim (Introduction "reads the source rather than recalling it", Outlook): restate as "conditioned on the source, can introduce content from parametric memory"; move to Limitations | all four |
| A3 | Human reading: report as 8 in-unit chains + 2 gate-rejected; present as a qualitative case study; drop "18.9 / 77.1 / 4.0%" to one decimal; delete "Three observations hold regardless of sample size" (line 614) | all four |
| A4 | Separate, from the existing verdict sheet, endpoint support / intermediate-step support / link-sourced content in tab:human | GPT both |
| A5 | "1 of 86" out of the abstract headline, or framed as an endpoint-reachability diagnostic; define the 86 consistently (abstract says "structurally available", app:judge says "previously unreachable") | Opus conf, GPT both |
| A6 | "10.6% is a floor" -> "judge-reported share"; the floor argument ignores false positives | GPT both |
| A7 | Shuffled-source 30%: remove "puts a rate on the invented framing"; call it a perturbation test | GPT both |
| A8 | Repeat extraction 9 of 18: report as conditional retention and promote near the chain-set definition | all four |
| A9 | Contribution bullet 1 (87.4% five-stage): demote to a descriptive result; it is schema compliance | Opus conf, GPT conf |
| A10 | Sec. 4 opening "need no control": retrieval and profiling still depend on extraction and label accuracy | Opus conf, GPT both |
| A11 | Flat-triple "no chain" is structural incompatibility, not a performance result | GPT workshop |
| A12 | "the pinned model identifier is the reproducibility guarantee" -> partial mitigation | Opus workshop, GPT workshop |
| A13 | Bold aphoristic lead sentences and "X, not Y" constructions; "the layer beneath that stack"; "defenders, who are the constituency that lacks it"; "unambiguously permitted" | all four |
| A14 | Long argumentative captions (tab:gates, tab:omission): cut to definitions | Opus both |

## B. Soundness reconciliation (session work; investigate before writing)

| # | Issue | Verified 2026-11-05 |
|---|---|---|
| B1 | sec:r-hub: "removes 2,385 nodes" with a 4,066-member group is impossible on one graph | Yes. 2,385 is from the threshold sweep on an earlier graph (app:dedup line 2247-2252); 4,066 is from the risk-block merge in `experiment_merge_vs_simexcl_ec.py` (includes an alias pre-merge step). Fix: report this run's own removal count (risk nodes minus groups; script prints it, 27 s run) |
| B2 | app:judge line 2108 says nodes are keyed by name, so same-named nodes from different documents become one node; the body claims per-paper isolation | Partly. `ai_safety_graph.py:46` MERGEs on `{name, type}`, which would fold names across documents in FalkorDB. Against that, the analysis substrate has 15,123 components (largest 61) and 1,140 exact-name duplicates, which global name keying would not leave. Need: which loader built `graph_node_attributes.pkl` and the released dump, then correct whichever text is wrong |

## C. New evidence obtainable without human annotation

| # | Item | Cost | Expected effect, per the reviews |
|---|---|---|---|
| C1 | Coverage table: released graph vs raw paths vs reporting unit, endpoint-pair vs argument level, from existing receipts | session work | Clarity; GPT conference asks for it |
| C2 | Retrieval comparison judged by a cross-provider LLM: 20-30 risk-intervention questions, chain lookup vs passage retrieval plus synthesis, scored for source support and citation correctness | LLM tokens, estimate needed | All four name a retrieval evaluation as the Significance lever; reviewers asked for independent (human) judgments, so an LLM-judged version is weaker |
| C3 | Corpus-wide flag for link-only content (named work not string-present in the source) and degenerate sources (body below a length threshold), shipped as a node attribute | script, no LLM | Opus workshop; addresses misattribution risk |

## D. Requires human annotation (not pursued, decision 2026-11-05)

- Random sample of 50-100 chains with readers who are not authors, agreement, intervals (all four).
- Second reader on the 10 existing chains (Opus workshop, GPT both).
- Human ratings of edge confidence and maturity near the thresholds (Opus conf, GPT conf).
- Relevance judgments for retrieval by people (all four; C2 is the LLM substitute).

## E. Cuts (both reviewers' lists agree)

Race-framing narrative (app:race and the sec:r-hub sentence); meta-grader material (app:judgeprompt
second half, grader scores, populations rows, the "216 vs 20" sentence); schema-version flag
accounting; the 23-of-441 recovery; the 30 -> 20 -> 10 packet history; the threshold-sweep
chronology in app:dedup; the asymmetric row of tab:gates (Opus both; GPT silent).

Move to appendix or compress: sec:r-clustering and sec:r-hub to one result each (all four, since
the merge is not applied to the release); the stage probe and TF-IDF detail (all four); Outlook
(Opus both, GPT workshop); Impact Statement licensing and dual-use paragraphs (Opus both).

## F. Defects (verified in paper_v4.tex 2026-11-05)

- Figure 1 labels Stage 2 "Verification"; the text says "audit".
- Duplicated sentence "Extraction fails outright on..." (lines 1431, 1440).
- 12,638 risks (line 2464) against 19,096 risk nodes in sec:m-extraction.
- app:query heading prints empty; its table floats elsewhere.
- Appendices cited as "Section M/S/T"; app:clustering and app:ablation cite themselves.

## G. Figures (session assessment 2026-11-05, before the reviews)

Figure 1 prints at 3.6-4.0 pt; rebuild at >= 7 pt and fill panel B with the real chain of
tab:query. tab:human full width with short stratum labels. Replace Figure 2A (length) with a
five-stage-share dot plot (released, nine gate settings, stage-name ablation). Row totals on
Figure 2B. Vector PDFs instead of PNGs. Opus conference independently proposes a single-column
Figure 1 panel B and moving Figure 2A to the appendix.
