# Paper A — consolidated cut list, to reach a 10-page main body

**Built 2026-08-17** from the six-review round in `paper/reviews_2026-08-17/` (Claude Opus 5,
GPT-5.6 Sol, Gemini 3.1 Pro; conference and workshop bars). Each model saw only the compiled
PDF, the NeurIPS scoring guidance and the prompt — no repo context. The prompt carried a
fourth item this round asking each reviewer to name the material whose removal is least
likely to move its own scores, with per-item page estimates and a running total. Script:
`paper/review_multi_model.py`; receipts `<model>_<mode>.md` + `.meta.json`.

🔴 **This file is the cut list only.** Everything else — studies, decisions, the reviewer
register — stays in `paper/OPEN_ITEMS.md`. Do not start a second open-items list here.

## The measured starting point, which is worse than the estimate on record

`AISafetyIntervention_PaperA_shared_long.pdf`, 27 pages, 1.37 MB, compiled 2026-08-17.
Measured with PyMuPDF: the body runs pages 1 to 15, References begins 89% down page 15, and
the appendices occupy pages 16 to 27. **Main body excluding references is therefore ~14.9
pages, and the gap to close is ~5 pages, not the ~3 that `OPEN_ITEMS.md` carries.** That
entry was explicitly a character-count heuristic with an instruction to compile and read the
real number; this is the real number. All six reviewers independently read the body as
"roughly 15 pages", so the target in the prompt was not leading them.

## Scores this round, against the 2026-08-15 round

| Model | Conference | vs 08-15 | Workshop | vs 08-15 |
|---|---|---|---|---|
| Claude Opus 5 | 3 borderline reject | = | 4 borderline accept | = |
| GPT-5.6 Sol | 3 borderline reject | **+1** | 4 borderline accept | **+1** |
| Gemini 3.1 Pro | **5 accept** | **+2** | 5 accept | = |

The evidence added between the two rounds (#165, #166, #168, #171, #172) moved two of the
three models and neither of the two that moved cited anything else as the reason. Conference
is still not viable on two of three; workshop is 4/4/5 and is the defensible target.

## The consolidated cut table

Votes are over the six reviews (OC/OW = Opus conference/workshop, GC/GW = GPT-5.6 Sol,
MC/MW = Gemini). Pages are the reviewers' own estimates; where they bundled items the bundle
is split proportionally and marked. "Keep-collision" flags a row that touches the
explicit-keep list of `OPEN_ITEMS.md` (R121-R126) and states how the collision resolves.

| # | Item | Where | Action | Votes | Pages (median) | Keep-collision |
|---|---|---|---|---|---|---|
| 1 | Limitations, compressed to ~0.8 pp — one paragraph per distinct limitation, no re-derivation of numbers already in tables | `sec:limitations` | compress | 5/6 (OC OW MC MW GW) | **1.1** (1.0-1.5) | none. Every reviewer that names it also says the *content* is a credit; the objection is that it restates §3 |
| 2 | Merge/centrality derivation + Figure 2 (the two-panel hub) + the ANN candidate-recall paragraph | `sec:r-hub`, `fig:hub` | move to appendix, keep the headline and the control inline | 6/6 | **0.9** (0.45-1.45) | **R121.** Resolves: all six keep the 90x finding and its control in the body; only the derivation, the four-condition detail and the figure move |
| 3 | Probe decomposition, second-annotator per-stage confusion, arm-by-arm ablation detail | `sec:r-stages` | move to appendix (G, C, ablation) | 5/6 (OC OW MW GC GW) | **0.9** (0.5-0.9) | **R122-adjacent.** Resolves: keep 98.8% vs 20%, kappa 0.84, arm E and arm F headlines, and `tab:gates` inline |
| 4 | Framing analysis, the race-framing non-reproduction | `sec:r-selection` | **delete**; the control survives as two sentences under `sec:r-hub` | 6/6 | **0.55** (0.3-0.65) | none. Called "postmortem of an unpublished internal result" by four of six |
| 5 | The confounded pre/post repair design, "Two sevens in this subsection are unrelated", the schema-mismatch paragraph, and the three-omission prose that duplicates `tab:omission` | `sec:r-judge` | **delete**; one sentence in Limitations | 6/6 | **0.5** (0.35-0.65) | none. Unanimous in this round and in the 08-15 round |
| 6 | Token-bill reconstruction and the cross-model repricing band | `sec:m-repro` | move to appendix; keep one sentence (one call per document, 122.4M input, USD 32-118/1,000) | 4/6 (OC OW GC GW) | **0.5** (0.45-0.55) | none |
| 7 | Clustering narrative, the UMAP-vs-original-space silhouette argument | `sec:r-clustering` | compress to the control plus one number, detail to `app:clustering` | 6/6 | **0.45** (0.25-0.55) | **R126.** Resolves: the one clean same-space comparison stays as a sentence; the argument moves |
| 8 | "What the step drops" — chords, 78.3%, 579 pairs, threshold sweep | `sec:m-reporting` | move to appendix; keep 6.1% node loss and 18.0% pair loss inline | 6/6 | **0.4** (0.3-0.5) | none. Every reviewer insists the two headline losses stay in the body |
| 9 | "Two things follow for how the rest of this paper should be read" + the `sec:goals` bullets that restate the abstract | `sec:intro`, `sec:goals` | delete | 2/6 as a page item, **6/6 as a style flag** | 0.55 (0.2-1.0) | none |
| 10 | Intake / failure-count / structural-diagnostics prose | `sec:m-corpus`, `sec:m-recovery`, `sec:m-structural` | move to `app:composition` | 3/6 (OC OW GC) | 0.35 (0.15-0.7) | none |
| 11 | Related Work: GraphRAG and hypothesis-generation paragraphs | `sec:related` | compress by half | 2/6 (OC OW) | 0.38 | **Inverse collision:** OC wants the AI Safety Graph paragraph *lengthened*; it is the only external validation in the paper |
| 12 | Second retrieval example (Query 2, Ngo et al.) | `tab:query` | move to appendix, keep one worked query | 2/6 (OC MC) | 0.33 | five of six require at least one example to stay |
| 13 | Judge recovery of failed extractions, 23 of 441 | `sec:m-recovery` | move to appendix | 2/6 (GC GW) | 0.2 | matches R82 from the 08-15 round (2/3 then) |
| 14 | Figure 1 panel B (maturity x confidence) | `fig:dataset` | move to appendix | 1/6 (OW) | 0.2 | do not act on one vote |
| 15 | Impact Statement, compressed | `sec:impact` | compress | 1/6 (OW) | 0.3 | GC/GW both list it as un-cuttable. **Do not act** |
| 16 | Arm G (reference-list-only) and arm D (flat triples) | `app:ablation` | delete | 1/6 (GC) | 0.1 | contradicted by four reviewers who cite arm G as evidence. **Do not act** |

## The package that closes the gap

Rows 1-8, every one of them at 4/6 agreement or better, and none of them touching a claim:

| Row | Item | Pages | Running |
|---|---|---|---|
| 5 | `sec:r-judge` confounded design + editorial paragraphs, deleted | 0.50 | 0.50 |
| 4 | `sec:r-selection` deleted, control preserved | 0.55 | 1.05 |
| 8 | `sec:m-reporting` collapse forensics to appendix | 0.40 | 1.45 |
| 7 | `sec:r-clustering` compressed | 0.45 | 1.90 |
| 6 | `sec:m-repro` cost reconstruction to appendix | 0.50 | 2.40 |
| 2 | `sec:r-hub` derivation + Figure 2 to appendix | 0.90 | 3.30 |
| 3 | `sec:r-stages` probe/agreement/ablation detail to appendix | 0.90 | 4.20 |
| 1 | `sec:limitations` compressed to ~0.8 pp | 1.10 | **5.30** |

Row 9 (0.55) and row 10 (0.35) are the buffer if the estimates run optimistic, and both are
free: row 9 is a style fix every reviewer asked for independently, and row 10 is
reproducibility detail nobody defends.

**Nothing in this package requires cutting the three use cases wholesale**, which is what
`OPEN_ITEMS.md` listed first by yield and last by preference and which was flagged as a
co-author's call. Rows 2 and 7 move the derivations and keep each finding with its control
inline — the shape the June outline already specified.

## Explicitly do not move or cut

Union of the six "would refuse" lists, with the score each reviewer attached:

| Item | Named by | Score at risk |
|---|---|---|
| Figure 4 (pipeline + schema) and `sec:m-extraction` | 6/6 | Originality, Clarity |
| `tab:gates` and the paragraph reading it | 6/6 | Quality — "every chain-level number is unreadable without it" |
| Both audit runs' headline numbers and the instrument explanation | 6/6 | Quality; also the paper's main honesty credit |
| Arm E and arm F headlines (`tab:ablation`) | 6/6 | Quality, Originality — arm E is what converts 87.4% from claim to measured schema effect |
| `app:failure` (the Euclid chain) and its body pointer | 6/6 | Quality, limitations adequacy |
| At least one chain from `tab:query` | 5/6 | Significance — "without it the paper builds a dataset and never shows it can be used" |
| `tab:populations-master` | 3/6 (OC OW GC) | Clarity and Quality jointly |
| The nine enumerator constraints, at least in condensed form | 3/6 (OW GC GW) | Quality, reproducibility |
| The AI Safety Graph comparison (#166) | 2/6, and OC wants it *expanded* | Originality, Significance — the only external validation |
| A working release URL and the licence pair | 6/6 | Significance; currently a rendered placeholder |

## Not cuts — issues this round raised that are new to the register

None of these is a length item, and three of them are cheap. Ranked by what a reviewer does
with them.

1. 🔴 **Stage order is not enforced and traversal is undirected, so a "chain" may not be a
   directed argument** (GC, GW; both make it score-moving, GC says a bad result would take it
   to 2). Verified against `app:cuts`: monotonic stage order is not among the ten constraints,
   and "edge direction ignored" is row 10. The asks are all Class B over the released path
   file: the share of chains with monotonic stage order, the share whose edges are all
   traversed with their stored orientation, and the counts under directed traversal. **Cheap,
   and the paper cannot answer it today.**
2. 🔴 **The omission denominators are misaligned with their instruments** (GC, GW). Within the
   coverage list the missing share is 302/777 = 38.9%, and on the second run 476/627 = 75.9%;
   the paper divides by released edges instead and prints 18.1% and 21.7%. The 38.9% already
   exists in a source comment at the abstract and in `app:judge` as "42.2% of rows were
   covered", but no body sentence carries it as a rate. Both reviewers also reject
   `tab:omission`'s "upper bounds on omission" wording, since an unadjudicated judge can miss
   omissions as well as invent them. **Wording plus one table column; no new inference.**
3. 🔴 **`sec:r-judge` prints "777 rows --- 328 covered, 146 partially covered, 302 missing",
   which sums to 776** (GW). Verified in the receipt: the 777th row is
   `unlabelled_or_other = 1`, a "covered abstractly" status. One clause fixes it. A reviewer
   who checks the arithmetic and finds it wrong discounts every other number.
4. **"The stage we audit is the stage we ship" is in tension with `app:judge`** (GW), which
   says the judge audited pre-ingestion extraction JSON whose defect classes do not reach the
   released graph, and whose re-serialised edge count (10.8/paper) is well below the released
   one (16.7). Both statements are in the manuscript and both are true; the framing sentence
   needs the qualifier, or the tension reads as a contradiction.
5. **No precision or spurious-chain rate anywhere** (OC, OW, and implied by GC/GW). The
   sharpest form: *what share of 50 randomly sampled released chains would an author call
   spurious in the Euclid sense?* OC calls this the number it would weight above every graph
   statistic in the paper. This is the S4-shaped human item the team decided not to do,
   restated at n=50 and on precision rather than omission — and unlike the S4 design it does
   not need the annotator to reproduce a chain, only to judge one, which is minutes rather
   than three hours per paper. **Worth reopening the S4 decision for this one variant.**
6. **"122.4M input tokens" is called measured and is reconstructed** (GW): logs did not
   survive, retries and failed records are excluded, visible output is calibrated from one
   response. The manuscript says all of this in `sec:m-repro`; the abstract does not. Retitle
   as a reconstructed nominal estimate in the abstract.
7. **"Scalable" in the title is established for extraction only** (GC): the judge ran at
   sample scale, similarity construction can be quadratic, and the paper's own #168 follow-up
   shows path enumeration exploding with mean degree. Either scope the claim or measure the
   other three stages.

Items 1, 2, 3, 4 and 6 are all corrections rather than experiments. Item 1 is one Class B
script. Item 5 is the only one needing a person.

## Style, restated because it is now unanimous

Every one of the six flagged the same register, and four flagged the same three sentences:
"We report them all and reconcile none" (abstract), "One evaluation design that answers
nothing" (`sec:r-judge` heading), and "Two sevens in this subsection are unrelated". Rows 5
and 9 of the cut table remove two of the three. The general fix all six converge on: convert
finding-as-sentence headings to noun phrases, delete reader-instruction ("Read those rows
for portability rather than for capability"), and state each caveat once. This is L2/L3/L6 in
`OPEN_ITEMS.md`, which recorded L3 as partial; the external round says the residue is still
the dominant impression the prose leaves.

Two reviewers separately noted that the density of parallel constructions reads as
machine-drafted. Neither concluded the content was generated; both said it costs credibility.

## Provenance

Six jobs, all completed, no truncation: 307,854 input and 57,829 output tokens total,
4.7 minutes wall-clock for the slowest. Per-job usage in the `.meta.json` files. The
prompt change is the fourth item added to `PROMPT_CONFERENCE`, which the workshop mode
inherits by concatenation, so both bars answered the same length question.
