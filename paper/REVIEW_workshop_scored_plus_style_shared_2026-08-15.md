# Workshop review + writing-style audit

**Target:** `AISafetyIntervention_PaperA_shared/paperA_altstyle.tex` (2,205 lines, read in full)
**Rubric:** `AISafetyIntervention_LiteratureExtraction/paper/neurips-scoringguidance.txt`, re-weighted for a workshop track (see 0.1)
**Companion:** `REVIEW_neurips_scored_plus_style_shared_2026-08-14.md` (main-track review; findings W1-W23 referenced by tag, not repeated in full)
**Date of review:** 2026-08-15
**Not read (excluded by instruction):** `refs.bib`, the four figure images, receipt JSONs, other REVIEW_* files. Citation resolution and figure legibility remain **unassessed**.

Line numbers refer to `paperA_altstyle.tex`.

---

## 0.1 How this review differs from the main-track one

A workshop buys **early, discussable, honestly-reported work**. Significance and completeness are discounted; execution quality, calibration of claims, and presentation are not. I therefore re-weight as follows, and state it so the authors can see which of their known weaknesses I am waiving:

| Main-track finding | Workshop verdict |
|---|---|
| W6 no baselines | **Waived.** Expected at workshop scale; note it as future work, which the paper already does. |
| W7 retrieval shown by two examples | **Waived as an evaluation gap**, provided the claim is worded as a demonstration (it currently is, L1256-1261). |
| W17 corpus capped at 2023 | **Largely waived.** Materially lowers the artifact's value but not the paper's contribution to discussion. |
| W18 three of five use cases are cautions | **Inverted into a strength.** Negative results with controls are exactly what workshops exist to surface. |
| W10 single run, no noise floor | **Waived**, if stated (it is, L1268-1276). |
| W1 "quality-controlled" overclaim | **Not waived.** Wording accuracy is free. |
| W2/W3/W4 unreconciled omission figures, confounded grader design, run-artefact denominators | **Not waived as presentation.** The measurements can stay unresolved at a workshop; presenting the grader pre/post movement in a way that reads as a result cannot. |
| W5 "Scalable" in the title, never measured | **Not waived.** Title claims are cheap to fix. |
| W11/W12/W13 internal contradictions | **Not waived.** |
| W22 rendered `\OPEN{}` placeholders incl. in the abstract | **Not waived.** Blocking at any venue. |
| Style findings (Part 2) | **Not waived.** A shorter paper makes every tic more visible, not less. |

---

## 1 Assessment

### 1.1 Workshop fit and the story to lead with

The paper's best workshop identity is **not** "we built a knowledge graph of AI safety literature". It is:

> *What happens when you push a reasoning LLM through a fixed causal schema over 11,779 heterogeneous documents, and which of the graph analyses everyone reaches for afterwards are artifacts of your own preprocessing.*

That framing is well supported by content the paper already has: the merge-manufactured 90.1x centrality hub with only 111/4,066 members inside the merge threshold (L997-1019), the UMAP-silhouette non-comparability (0.281 in its own space, 0.004 in the source space, below a directly-fitted 0.014 --- L926-933), the 88% -> 2.6% and 44x -> 2.0x reproduction failure of the authors' own earlier finding (L1048-1066), and the Euclid/prime-scarcity chain that passes every structural filter (L2164-2196). Four concrete, receipted, transferable cautions. A workshop audience building LLM-extraction pipelines will take those home.

The current framing --- resource paper first, cautions in Section 5 --- puts the weakest-supported claims (an unvalidated chain set, a two-example retrieval demo) in front of the strongest-supported ones. **Recommendation: lead with the artifacts, present the corpus and pipeline as what produced them.** This costs nothing evidentially and raises the paper's score at a workshop by a full point.

Secondary fit note: the paper works at a *safety-community* workshop (audience cares about the corpus) and at an *NLP/IE or eval-methods* workshop (audience cares about schema-guided extraction failure modes) --- but the emphasis differs. For the latter, cut the AI-safety-specific motivation to a paragraph and lead with the schema-induced-chain finding.

### 1.2 Strengths (workshop-weighted)

**S1 --- Receipts.** Every number carries an on-disk source comment; a released script re-derives 112/112 numeric claims (L566-576); figures plot only from committed receipts (L583). At workshop scale this is far above the norm and should be said aloud in the talk.

**S2 --- Self-refutation is published.** Section 5.5 reports that the authors' own earlier headline (88% race-framing in the top-100, 44-fold gradient) does not reproduce under re-derivation, and that the keyword classifier was sound at 92% precision, so the failure sits in the selection step (L1056-1060). Publishing this is the single most credible thing in the paper.

**S3 --- Gate sensitivity as a full grid.** Table 1 (L666-696) re-runs the entire enumeration at all nine gate settings and shows corpus yield moving 15.9% -> 99.4% while five-stage completeness holds inside eight points, with the deployed row reproducing the released files exactly. This is the right way to report a threshold-dependent dataset size.

**S4 --- Population honesty.** 12 of 100 judged papers reach the chain set, covering 17 of 2,772 chains, stated three times where it matters (L216-219, L520-528, L1196-1202).

**S5 --- A demonstrated failure, not an asserted one.** The prime-number chain is worth a slide by itself.

**S6 --- Impact Statement.** Misattribution-to-identifiable-authors treated as distinct from misreading, with a takedown route (L1392-1401). Better than most workshop submissions bother with.

### 1.3 Weaknesses that still count at a workshop

**V1 (blocking) --- Rendered placeholders, including in the abstract.** `\OPEN{}` blocks render highlighted in the built PDF: release URL in the **abstract** (L123), licence (L584-587), human spot-check (L1226-1228), multi-model check (L1277-1279), acknowledgments and author list (L1345-1350), cluster representatives (L1855-1856), AI-use scope (L1438-1439). A workshop reviewer reads this as a draft submitted by accident. Nothing else in this review matters until these are closed. `texlint.py`'s GAP count is already the checklist (L61-63).

**V2 (blocking) --- "quality-controlled" is not supported and is free to fix.** The judge covers 0.6% of the chain set and scores neither gate that selects it (L554-562). Use "gate-selected" or "quality-gated" in the abstract, the contribution bullets, Table 1's caption and the conclusion. This is a five-instance find-and-replace, and leaving it is the kind of unearned wording that makes a reviewer distrust the receipted numbers too.

**V3 (blocking) --- "Scalable" in the title with zero cost measurement.** Linearity is asserted three times (L245-247, L303-307, L1293-1297); no tokens, wall-clock, dollars or throughput anywhere, and L595-596 concedes the run's totals may be unrecoverable. Either measure it on a 200-document re-run (cheap, and useful for the talk) or drop the word. See D5 below for a retitle.

**V4 --- The grader pre/post result is presented as a finding and cannot be one as run.** Graders were shown the repairs they were asked to score; no blinding, no order randomisation, no null-repair arm (L540-548). The authors name the confound, then still report unanimity on direction in a `\paragraph` headed "Graders agree on the extraction and not on the repair" (L763-776) and carry a +3.47/+13.22/+33.08 table (L2126-2145). At a workshop the honest move is available and costs nothing: **demote the pre/post comparison to a methods-lesson** --- "here is a judge-evaluation design that looks reasonable and answers nothing, and here is the one-line control that would fix it". That is a better workshop contribution than the number ever was.

**V5 --- Denominator hygiene.** 43 papers profiled "determined by which rubric iteration a grader run used" (L736-739); graders at n=95/95/13 on partly disjoint subsets; agreement statistics on the 13 common papers; Gemini saturated at 95.77 +/- 1.80. Nobody expects a workshop paper to re-run everything, but the paper should stop quoting rates off these subsets in prose (28.8%, 16.3%, 32.6% at L724-734) and instead present them once, in one table, with n and how-drawn columns. Table 11 (L2082-2100) already does this correctly for the two omission measurements --- extend that treatment.

**V6 --- Two rendered internal contradictions.** (a) Similarity-layer edge count at tau=0.80: 1,435,806 (L1680-1681) vs 169,083 (L1908-1913). The reconciliation exists only in a LaTeX comment (L1684-1685), which does not render. (b) Centrality method: eigenvector centrality under four conditions (L1014, Table 6) vs an alpha-mixed structural/similarity transition matrix, i.e. PageRank, described as "the centrality analyses in `sec:r-hub`" (L1938-1944). Appendix E is inherited from the frozen draft and was not updated to this substrate; the fix is to delete it (see H2) rather than repair it.

**V7 --- One phrase, two numbers.** "The chain set is 38.1% arXiv" (L674, L1199, and 1,055/2,772 at L1654) is a share of *chains*; Table 9's chain-set column gives 35.2% (L2038), a share of *chain-yielding documents*. Both correct, unit disclosed only in the caption. Name the unit in the prose. Same for LessWrong (9.6% vs 10.2%) and Alignment Forum (11.8% vs 13.9%).

**V8 --- Four confusable reduction operations.** Quality gates / containment de-duplication / node merge / similarity layer. The funnel caption exists mainly to disambiguate them (L488-493), a subsection was renamed for the same reason (L444-447), and a removal note records readers conflating two (L378-384). In a shorter paper this gets worse, not better, because the disambiguating scaffolding is the first thing cut. Fix the nouns: *gate*, *sub-path collapse*, *node coreference merge*, *semantic layer* --- and use nothing else.

**V9 --- The de-duplication threshold is undefended and non-monotone.** 0.60 -> 2,658 chains, 0.70 -> 2,772, 0.80 -> 3,356, 0.90 -> 5,460, and 0.90 exceeds 1.00 (5,427) because the keep-set is built greedily longest-first (L473-483). Also docstring 80% vs code 70%. Honestly reported; still means the headline count is arbitrary within a factor of two. At workshop scale, one sentence stating the objective the step is meant to achieve would settle it.

**V10 --- Table 5 assigns verdicts the paper later contradicts.** The merge threshold table labels 0.88/0.05 "**Selected**" on precision reasoning with no recall column (L1711-1730), while Section 5.4 reports the same specification returns 4,411 candidate pairs against 54,282 exhaustive --- a 12x miss concentrated in the risk block --- plus 1,140 surviving exact-name duplicates (L1031-1035). Add the recall column or drop the verdicts.

### 1.4 Length: this is not yet a workshop paper

Uncompiled estimate (**not verified --- I did not build the PDF**): body text through the Conclusion is roughly 1,100 source lines of two-column 10pt prose plus four full-width figures and two full-width tables, i.e. of the order of 9-10 pages, with eight appendices adding several more. Typical workshop limits are 4 pages (+ unlimited appendix) or 8-9 pages. The paper must lose most of its body, not trim it.

**Suggested 4-page cut** (keeps the strongest evidence, loses nothing receipted):

| Keep | Compress | Cut to appendix / delete |
|---|---|---|
| Abstract (minus the closing promissory sentence, D4) | Intro to ~2 paragraphs; delete "Research Goals" (L199-228) entirely --- it restates the abstract and pre-announces the section structure | §3.1 corpus prose -> two sentences + Table 7 pointer |
| Figure 1, with the middle panel swapped (H4) | Methods §3.2 schema to ~half; keep the model config paragraph verbatim (L333-341), it is the reproducibility core | §3.4, §3.5, §3.6 mechanics -> appendix, leaving one sentence each in a numbered pipeline list |
| Table 1 (gate grid) --- the paper's best single object | §4.1 to ~2 paragraphs + Table 1 | §5.1 retrieval demo: keep 3 sentences, move Table 2 to appendix |
| §4.2 judge, restructured per V4, with Table 11 (omission) moved **inline** | §5.3/5.4/5.5 to one paragraph each, keeping the 0.004-vs-0.014 silhouette number, the 90.1x hub, the 88%->2.6% failure | Related Work to 5 sentences; §5.2 corpus profiling to 2 sentences |
| Appendix J (failure case) --- or a 5-line version inline | Limitations to the five that bear on a claim | Outlook to 3 sentences |

Appendices B, D, I (composition, centrality conditions, judge protocol) are worth keeping in full. Appendices F, G, H should go regardless of page limit (H1-H3 below).

### 1.5 Ratings

Scored on the guidance file's four dimensions with workshop expectations substituted for main-track ones (i.e. Significance is scored as *"is this worth an audience's hour"*, not *"does this advance the field"*).

| Dimension | Rating | Basis |
|---|---|---|
| **Quality** | **3 (good)** | Receipt discipline and self-refutation are strong; sample-scale verification and missing baselines are acceptable here. Held at 3 rather than 4 by V4 (a confounded comparison presented as a result) and V5 (rates quoted off run-artefact subsets). |
| **Clarity** | **3 (good)** | Well organized and unusually explicit about scope; costs points for V6-V8 and for being roughly twice a workshop's length with no evident compression plan. |
| **Significance** | **3 (good)** | Under workshop weighting: four transferable, receipted cautions about LLM-extracted graphs, plus a released corpus. The 2023 cap and the cautionary character of most use cases stop mattering much here. |
| **Originality** | **3 (good)** | The seven-stage causal-interventional schema and the artifact measurements are new; the pipeline itself is standard practice and the schema is still not defended against argumentative zoning or PICO, which the paper's own Related Work names (W20). |
| **Overall** | **4 --- Accept (poster), and a credible spotlight** if V1-V4 are closed and the paper is re-led per 1.1. As submitted, with rendered `\OPEN{}` blocks in the abstract, **3 --- borderline**. | The evidence that exists is well-made and the negative results are the kind a workshop should want. What holds it back is presentation choices the authors can fix in a day, not missing experiments. |
| **Confidence** | **4** | Full source read; familiar with LLM-as-judge and scientific-IE literature. `refs.bib`, receipts and figure images unread, so citation resolution and figure legibility are unassessed. |

### 1.6 Questions to the authors (workshop-scale, ranked by score impact per hour of work)

**Q1.** Will you re-lead the paper on the four measured construction artifacts, with the corpus as what produced them? **+1 overall; costs a rewrite of the abstract and intro, no new experiments.**

**Q2.** Will you demote the grader pre/post comparison from result to methods-lesson, and state the null-repair arm as the control that would make it interpretable? **+1 Quality; costs three paragraphs.** If you can also *run* the sham arm (one batch call, same 100 papers), report it --- either outcome is publishable and it is the most interesting thing you could add before the deadline.

**Q3.** What did the extraction cost per document (tokens, wall-clock, dollars)? Measure on 200 documents if the original batch logs are gone. **Required for "Scalable"; also the number the audience will ask for in Q&A.**

**Q4.** Can the 20-paper human-anchored spot-check land before camera-ready? It reconciles 0.6% vs 28.8%, which the paper itself calls the clearest reason it needs a human anchor (L757-758). **+1 Quality.** At workshop scale, 20 papers annotated by one author with a stated protocol and a second-annotator subset is enough.

**Q5.** Please state the objective the 0.70 containment step optimizes, given 0.90 yields more chains than 1.00 (V9). **Clarity only, one sentence.**

**Q6.** Which workshop audience is this for --- safety-corpus users or extraction-pipeline builders? The intro currently serves the first and the results serve the second. **Affects framing, not score.**

### 1.7 Limitations and societal impact

**Yes, adequately addressed --- among the better-handled I have reviewed at this scale.** Nine named limitations each stating the control not run; Impact Statement separates misreading from misattribution and ships a takedown path.

Two items to add, both cheap:
- The licence remains undetermined (L584-587). For a corpus derived from LessWrong / EA Forum / transcript sources under mixed terms, this is a live redistribution question, not formatting. Name the pair before release.
- No limitation states that the gate-selected chain set is arXiv-weighted where the corpus is forum-weighted (Table 9 measures this), so any downstream "what does the field propose" reading inherits a venue bias. One sentence in Limitations.

---

## 2 Writing-style audit (workshop bar: unchanged)

Full detail is in the companion review, Part 2. Repeated here in short form because compression makes each tic more conspicuous, and because a workshop paper is read start-to-finish in one sitting.

**2.1 Contrastive correction ("X, not Y") --- ~30 instances.** L399, 634-643, 655, 706, 733, 791, 896, 1027, 1087, 1096-1104 (three consecutively), 1148, 1158, 1265, 1287, 2189. Individually correct scoping; cumulatively the register reads generated and *defensive*, and a reader told twenty times not to over-read the results concludes the results are fragile. Keep the earned ones (L655, L706, L1027); target under 8 in a 4-page version, under 5.

**2.2 Epigrammatic paragraph-enders --- 8 instances.** L400 ("A corpus of N short chains is exactly N components until something links them"), L643, L995, L1002, L1102, L1168, L1194, L1387 ("That claim would be confident and wrong"). Two are good writing; eight is a cadence tell. Keep L400 and L643, delete the rest.

**2.3 Register breaks.** L797 "the reader who takes the release and does something with it" --- colloquial. The five `\paragraph{Control.}` blocks (L869, 914, 948, 1040, 1085) are second-person README prose, two of them near-empty ("**Control.** None."). In a compressed paper, fold the controls into the sentences that state each artifact and delete the empty slots.

**2.4 Diction.** "bends a little under pressure" (L638); "the structure barely moves" (L464 --- give the delta 89.0 -> 87.4, drop the adverb); "we describe what it returned in some detail" (L702 --- delete); "no ranking exists to report" (L1017 --- overstated, the top-2 gap is 3%); "an obvious application we do not attempt" (L1160); "The extension we would most like to see" (L1334); table header "**xrisk**" (L1758); "**intv**" used before its caption defines it (L833).

**2.5 Mechanical sweeps.** Mixed spelling ("randomisation" L546, "neighbourhood" L928, "specialised" L1124, "favourable" L1416 vs "normalization" L982, "labeling" L929, "colored" L960); number-words inconsistent ("Twelve of the 100" L2022 vs "12 of the 100" L521); `\emph{}` 40+ times, mostly on ordinary words --- keep it for term introductions only.

**2.6 Dramatized verbs.** "manufactures" (L986, 1004, 1332 and the §5.4 heading), "baked" (L1023), "breaks centrality" (L985), "collapses" (L768, 1216), "disagree by a factor of fifty" (L746), "absent, not merely under-represented" (L1286), "the weakest point in the evidence" (L1224), "The sharpest" (L1332). Every one is numerically supported; the density is what reads as pitch. Trim about half, keeping the strongest instance of each finding. Note for accuracy: over-assignment of "critical"/"essential" is **not** a problem here --- exactly one emphatic "essential" (L903) and no rhetorical "critical". The dramatization runs through verbs and epigrams instead.

**2.7 Density and duplication (the main lever for a workshop cut).**
- **D1** Headline numbers (200,525 / 2,772 / 1,868 / 87.4% / 15.9%) appear five times: abstract (L104-126), intro (L189-190), Research Goals (L209-213), §4.1 (L612-620), Conclusion (L1320-1323). Delete Research Goals and the Results preamble (L602-607) outright.
- **D2** §5's opener says the same thing three ways in one paragraph (L797-806). Two sentences.
- **D3** Figure 1's caption is ~190 words and carries hedging and interpretation ("Neither reading is measured here"). Captions say what is plotted; the hedges already exist at L634-643.
- **D4** The abstract's closing 40-word promissory clause (L124-126) --- "acceleration of AI safety progress ... graph-based reasoning over stored arguments to explore new AI safety solutions" --- measures nothing in the paper and reads as slogan. Cut; the Outlook carries it (L1299-1312).
- **D5** Title: "Chains in Graphs: Scalable Framework for Extraction of AI Safety Mechanisms from Published Literature" --- "Chains in Graphs" is near-tautological, the article before "Scalable Framework" is missing, and "Scalable" is the unmeasured claim of V3. For the workshop framing of 1.1: *"Mechanism Chains from the AI Safety Literature: What an LLM-Extracted Graph Does and Does Not Support"*.
- **D6** The "maturity is LLM-assigned and unscored" caveat appears five times (L147-151, 554-558, 881-886, 914-916, 1263-1266). Once, with cross-references.

**2.8 Over-highlighted low-relevance material (all three should go in a workshop version).**
- **H1** Appendix F: a full column listing 40 cluster names (L1845-1902) for an analysis the paper demotes to "a browsing index, not a taxonomy" and whose silhouette in the real embedding space is 0.004. Eight examples in a sentence; ship the list with the release.
- **H2** Appendix G (L1908-1936): inherited from the frozen draft, carries the stale 169,083 figure (V6a), and supports no main-text claim since every quantitative claim uses EDGE-only paths. Table 8's ~20x-per-hop growth is arithmetic. Delete.
- **H3** Appendix H (L1946-2004): validates a classifier for a finding the paper withdrew as unsupportive ("supports no claim about the discourse structure of the field", L1981), yet keeps precision/recall strata, a prevalence paragraph and an odds-ratio table with CIs. Reduce to one paragraph: the earlier result did not reproduce, and the classifier was sound at 92% precision.
- **H4** Figure 1's middle panel spends a third of the page-1 figure on intervention maturity, which the paper says four times is unvalidated and descriptive. Replace with the gate-yield curve from Table 1 or the two-measurement omission comparison --- both load-bearing, both already receipted.

**2.9 Source hygiene before posting.** The .tex carries an internal editorial trail --- "REMOVED 2026-08-14" (L378-384), "Moved out of sec:r-hub" (L1777), "the frozen Overleaf reported..." (L395, 425, 1677, 1684), "the module docstring says 80%, the code uses 70%" (L459, 482), the compute-donor gate block (L1352-1375), the WITHDRAWN-paragraph note (L889-893). Several disclose internal disagreement, an Overleaf workflow and a private compute donor. Move to a separate NOTES file.

---

## 3 Prioritized fix list for a workshop submission

**Day 1 --- blocking, no new results needed**
1. Close every `\OPEN{}`; release URL and licence first (V1).
2. "quality-controlled" -> "gate-selected" everywhere (V2).
3. Retitle; drop or measure "Scalable" (V3, D5).
4. Fix the two rendered contradictions --- deleting Appendix G resolves one of them (V6, H2).
5. Rename the four reduction operations to distinct nouns (V8).

**Day 2 --- reframing, which is where the score is**
6. Re-lead on the four construction artifacts; corpus and pipeline become the apparatus (Q1).
7. Demote the grader pre/post comparison to a methods-lesson with the null-repair control named (V4, Q2).
8. Execute the 4-page cut plan in 1.4; move Table 11 (omission) inline.
9. Swap Figure 1's middle panel (H4); shorten its caption (D3).
10. Cut Appendices F, G, H (H1-H3).

**Day 3 --- prose pass**
11. Contrastive corrections down to <8; six of eight epigrams deleted (2.1, 2.2).
12. Control blocks folded into prose; empty ones deleted (2.3).
13. One spelling variety, one number-word rule, `\emph{}` restricted to term introductions (2.5).
14. Cut the abstract's closing sentence (D4); delete Research Goals and the Results preamble (D1).
15. Strip the editorial trail from the .tex (2.9).

**If time allows before camera-ready, in this order**
16. Null-repair grader arm --- one batch call, and the best Q&A material in the paper (Q2).
17. Cost measurement on 200 documents (Q3).
18. 20-paper human-anchored spot-check (Q4).
