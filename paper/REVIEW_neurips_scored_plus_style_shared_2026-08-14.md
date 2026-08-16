# NeurIPS-style review + writing-style audit

**Target:** `AISafetyIntervention_PaperA_shared/paperA_altstyle.tex` (2,205 lines, read in full)
**Rubric:** `AISafetyIntervention_LiteratureExtraction/paper/neurips-scoringguidance.txt`, applied section by section
**Date of review:** 2026-08-14
**Not read (excluded by instruction):** `refs.bib`, the four figure image files, all receipt JSONs, prior REVIEW_* files. Citation resolution and figure legibility are therefore **unassessed**; every numeric cross-check below is internal to the .tex.

Line numbers refer to `paperA_altstyle.tex` as read.

---

# PART 1 --- NeurIPS scoring guidance, applied

## 1.1 Summary of the submission

Two-stage pipeline over the Alignment Research Dataset (ARD): (1) one schema-constrained `o3` call per document extracts a typed chain risk -> problem analysis -> theoretical insight -> design rationale -> implementation mechanism -> validation evidence -> intervention; (2) a cross-provider judge (`claude-sonnet-4-5`) re-reads 100 extractions against sources, with three meta-graders scoring pre/post proposed repairs. Output: 200,525 nodes over 11,779 documents; 2,772 quality-gated chains from 1,868 documents. Five use cases are reported, two as capabilities and three as artifact-dominated cautions with controls.

## 1.2 Strengths

**S1 --- Claims-to-receipts discipline is unusually strong.** Every number carries an on-disk SRC comment; one released script re-derives 112/112 numeric claims (L566-576). Figures plot only from committed receipts so they rebuild without the 3.2 GB checkpoints (L583). This is better provenance hygiene than most accepted dataset papers.

**S2 --- The negative results are the most valuable content.** Three findings are genuinely useful to anyone building graphs over LLM-extracted scientific text: (a) transitive-closure node merging manufactures a 90.1x centrality super-hub from a group where only 111/4,066 members are within the merge threshold of the canonical node (L997-1019); (b) a silhouette computed in UMAP space is not commensurable with one in the source space, and the deployed reduced-space clustering scores *worse* (0.004) in the original space than a clustering fitted there directly (0.014) (L926-933); (c) a centrality-selected-head co-occurrence statistic collapses from 88% to 2.6% and 44x to 2.0x under re-derivation (L1048-1066). (b) and (c) are corrections the authors are publishing against their own earlier pipeline. That is rare and creditable.

**S3 --- Population honesty.** The paper states plainly that the verified population is not the analysed population: 12 of 100 judged papers reach the chain set, covering 17 of 2,772 chains = 0.6% (L216-219, L520-528, L1196-1202). Most papers with this structure would not surface it.

**S4 --- Gate sensitivity is reported, not hidden.** Table 1 (L666-696) re-runs the full enumeration at all nine gate combinations and shows corpus yield moving 15.9% -> 99.4% while five-stage completeness stays inside eight points. The deployed row reproduces the released path files exactly, which licenses the other rows.

**S5 --- A real failure case is exhibited, not described.** The Euclid/prime-scarcity chain (L2164-2196) passes every structural filter and is spurious; the authors correctly identify it as invention of a risk framing rather than an out-of-domain read.

**S6 --- Impact Statement is above the norm.** Misattribution-to-identifiable-authors is treated as a distinct harm from misreading, with three concrete mitigations including a takedown route (L1392-1401).

## 1.3 Weaknesses

### Quality

**W1 (major) --- The headline artifact is essentially unvalidated.** The judge stage validates *extraction over ARD documents*. The paper's reporting unit is the 2,772-chain set, of which 0.6% falls inside the judged sample. No fidelity number in this paper applies to the released chain set. The abstract's phrase "2,772 quality-controlled chains" therefore rests on two LLM-assigned attributes (edge confidence, maturity) that the validation protocol explicitly does not score (L554-562). "Quality-controlled" is doing work the evidence does not support; "quality-gated" or "gate-selected" is the defensible word.

**W2 (major) --- The two omission measurements differ by 50x and are unreconciled (L746-758).** 0.6% vs 28.8% is not a limitation, it is the absence of a measurement. As submitted, the paper reports no usable omission or coverage rate. The authors say so honestly, but honesty does not substitute for the number, and a 20-paper human-anchored spot-check --- named in the paper as not performed (L1226-1228) --- would settle it at trivial cost relative to the extraction run already paid for.

**W3 (major) --- The meta-grader design is confounded beyond repair as run, and the cheap control was skipped.** Graders saw the proposed repairs they were asked to score; no blinding, no order randomisation, no null-repair arm (L540-548). The authors name the confound. The fix --- same graders, sham or empty repair list --- costs one batch call. Until it exists, "graders agree the judge improved the extraction" (L763-776) carries no evidential weight, and the post-repair ICC(2,1) collapse from 0.92 to 0.15 is consistent with the presentation-effect reading rather than the improvement reading.

**W4 (major) --- Denominators are set by artefacts of the run, not by sampling.** The structured error profile exists for 43 of 100 papers "determined by which rubric iteration a grader run used" (L736-739); meta-graders scored n=95, 95, 13 partly disjoint subsets (L2126-2145); agreement statistics rest on the 13 papers all three scored (L2113-2124); Gemini's post-repair mean is 95.77 +/- 1.80, i.e. saturated. A reviewer will discount the entire grader analysis on these grounds. Rerun the rubric once, at a fixed prompt, over all 100.

**W5 (major) --- "Scalable" is in the title and is never measured.** Cost linearity is asserted three times (L245-247, L303-307, L1293-1297) and never quantified: no token totals, no wall-clock, no dollar cost, no throughput, and the GAP at L595-596 concedes the extraction run's wall-clock and token totals may not be recoverable. A scalability claim in a title requires at minimum a cost-per-document figure and a projection to the arXiv+LessWrong corpus the Outlook proposes.

**W6 (major) --- No baseline of any kind.** The authors state this (L1253-1261). The consequence is that no design choice is shown to matter: not the reasoning model over a cheap one, not full text over abstract-only, not the seven-stage schema over flat triples. For a pipeline paper this is the difference between "we built a thing" and "we built the right thing". A single 200-document ablation on all three axes, scored by the same judge, would move my rating.

**W7 (major) --- The retrieval capability rests on two hand-picked examples.** Table 2 (L820-855) shows two chains, both from arXiv, both selected because "the extraction input is structured and the citation is unambiguous" (L816-817). There is no query set, no relevance judgement, no comparison against embedding search over abstracts. The paper's motivating question ("If I care about risk R, what has been proposed...", L166-167) is answered by anecdote. A 50-query set with three-way relevance labels against a BM25/embedding baseline is the smallest thing that would convert this into a result.

**W8 --- The reporting unit is arbitrary within a factor of two.** The 0.70 containment threshold gives 2,772 chains; 0.60 gives 2,658, 0.80 gives 3,356, 0.90 gives 5,460 (L473-477). Worse, the SRC note concedes 0.90 (5,460) exceeds 1.00 (5,427) because the keep-set is built greedily longest-first, i.e. the procedure is non-monotone in its own threshold. The code/docstring mismatch (docstring 80%, code 70%, L459-460) is reported honestly but signals the step was never specified deliberately. Specify the de-duplication as an algorithm with a stated objective, or report the raw 8,954 set as primary.

**W9 --- Reproducibility has an unverifiable hole at intake.** The three failure counts (1,667 / 128 / 58) come from a packaging record; the directories are not in the release (L273-277). The 13,632 intake total is "reconstructed" (L1619) and is the sum of the corpus and those unverifiable counts, so the 86.4% pipeline-completion figure cannot be checked by a reader. Also "Most of the first group are ARD entries carrying a title and URL but an empty source body" (L269-272) is unquantified for the same reason.

**W10 --- Single extractor, single run, non-deterministic, and retiring.** No repeat-run agreement, so every descriptive in Section 4 sits on an unmeasured noise floor (L1268-1276). `o3` at reasoning effort medium is not reproducible bit-for-bit and will be retired. Re-extracting 300 documents and reporting node/edge agreement is cheap and would bound this; the recovered n=20 multi-model check flagged at L1277-1279 should be recovered or dropped.

### Clarity

**W11 --- Internal contradiction in the similarity-layer edge count (rendered text, not comments).** L1680-1681: the layer at tau >= 0.80 "adds 1,435,806 within-category edges and reduces the component count from 15,123 to 4,124". L1908-1913: "At tau=0.80, 169,083 similarity edges connect conceptually related nodes across documents." Both render. The reconciliation (the frozen draft's k-NN construction is ~8.5x sparser) exists only in a LaTeX comment (L1684-1685). Appendix E is inherited from the frozen draft and was not updated to the current substrate. Fix or delete the appendix.

**W12 --- The centrality method is described two different ways.** L1014 and Table 6 (L1745-1776) report **eigenvector centrality** under four merge/SIM conditions with no mixing parameter. L1938-1944 states "For the centrality analyses in `sec:r-hub` we combine structural and similarity transition matrices as P = alpha P_struct + (1-alpha) P_sim" and reports rank stability over alpha --- that is a PageRank construction. A reader cannot tell what was computed.

**W13 --- The same quantity is quoted with two values under the same name.** "The chain set is 38.1% arXiv" (L674, L1199, and 1,055/2,772 at L1654) is a share of *chains*; Table 9's chain-set column gives arXiv 35.2% (L2038), a share of *chain-yielding documents*. Same for LessWrong (9.6% vs 10.2%) and Alignment Forum (11.8% vs 13.9%). Both are correct and the caption discloses the unit, but the main text uses one phrase for both. Name the units in the prose.

**W14 --- Four confusable reduction operations, and the paper knows it.** Quality gates, containment de-duplication, node merge, similarity layer --- the funnel caption exists mainly to disambiguate them (L488-493), a Methods subsection was renamed for the same reason (L444-447), and a removal comment at L378-384 records readers conflating two of them. That is evidence the vocabulary itself needs fixing, not more signposting. Rename to distinct nouns (gate / sub-path collapse / node coreference merge / semantic layer) and use them nowhere else.

**W15 --- Denominator load on the reader.** 11,779 / 8,954 / 2,772 / 1,868 / 100 / 95 / 43 / 13 / 441 / 36 / 30,000. Section 5 alone changes denominator six times. One small table of populations with n, how drawn, and what it licenses, placed at the head of Section 4, would carry most of the paper's honesty at a fraction of the prose.

**W16 --- The main text is not readable standalone.** 30+ forward references to appendices, several of which carry the actual evidence (Table 6 centrality, Table 9 populations, Table 11 omission). At least the omission table should be inline; it is the paper's central unresolved measurement.

### Significance

**W17 (major) --- The corpus stops at 2023 (L283-288, L1284-1288).** Zero post-2023 documents, median 2021. The practitioner question the introduction poses is, in 2026, overwhelmingly a question about 2024-2026 literature. The authors are right that the pipeline outlives the corpus, but a resource paper is judged partly on the resource, and this one is unusable for the use case that motivates it. Running the released pipeline over even a 5,000-document 2024-2026 arXiv safety slice would change my significance rating materially --- the paper argues it is one batched call per document, so this is budget, not method.

**W18 --- Three of five use cases are cautions, and they are post-hoc (L802-806).** The section is, in substance, an erratum on the authors' own earlier internal analyses. That is honest and useful, but it means the released graph's advertised analytic surface is: two operations that need no control (one of which is a lookup) and three that are artifact-dominated. State the net capability claim in one sentence in the introduction so the reader is not left computing it.

**W19 --- The merge recall problem undercuts the dedup appendix.** Section 5.4 reports that the deployed merge specification returns 4,411 candidate pairs against 54,282 under exhaustive search --- a 12x miss concentrated in the risk block (L1031-1035) --- and that 1,140 exact-name duplicates survive. Table 5 (L1711-1730) nonetheless labels the 0.88/0.05 configuration "**Selected**" on precision reasoning alone, with no recall column. The table reads as a validated choice; the body says it catches ~8% of candidates.

### Originality

**W20 --- The pipeline is standard; the schema is the novel object and is not defended.** One structured prompt plus one cross-provider judge is 2024-vintage practice. The contribution that could be novel is the seven-stage causal-interventional schema, and it is never justified against the obvious alternatives the Related Work itself names --- argumentative zoning labels (Teufel), claim-premise argument mining, PICO. No evidence is offered that the five intermediate stages are separable, that two models assign them consistently, or that a human would. A stage-assignment agreement study on 50 documents (two models + one human) is the single experiment that would establish the schema as a contribution rather than a design choice.

**W21 --- Related Work is complete on adjacent fields and thin on the nearest one.** Missing: schema-induced hallucination and faithfulness of structured extraction (the paper's own failure case is an instance), and any LLM-KG-extraction work that reports a measured fidelity rate against human annotation --- which is exactly the comparison the paper's judge study substitutes for. The GAP at L1109-1110 concedes the section is unfinished.

### Submission-readiness (weighs on the overall score as submitted)

**W22 --- The manuscript contains rendered placeholders.** `\OPEN{}` blocks are yellow-highlighted in the built PDF and appear in the **abstract** (release URL, L123), Methods (licence, L584-587), Limitations (human spot-check, L1226-1228; multi-model check, L1277-1279), Acknowledgments (compute, author list, L1345-1350), Cluster appendix (L1855-1856), and the AI-use statement (L1438-1439). A reviewer sees an unfinished submission. The release URL and the licence are not cosmetic: a dataset paper with neither is not reviewable on reproducibility or ethics.

**W23 --- Anonymity and venue fit.** The Acknowledgments name EleutherAI and SOAR (L1340-1343) --- incompatible with a double-blind submission as written. The paper is a Datasets & Benchmarks submission, not a main-track one: at D&B its artifact-plus-controls shape is the expected shape, and it would additionally need a datasheet, structured metadata, a named licence, and a persistent DOI, none of which are present.

## 1.4 Numeric internal-consistency check (performed)

Checked and **consistent**: node inventory sums to 200,525 (L1647-1652); maturity counts sum to 36,959 and match the percentages (L1691-1694 vs L877-879); chain-length shares sum to 100.0% (L627-629); gate-table yields match 11,779 as denominator (all nine rows); mean 17.0 nodes/document x 11,779 ~= 200,243 vs 200,525 reported; 1,055/2,772 = 38.1% arXiv matches the deployed gate row; schema flag split 184+34 = 218 (L790-791, L2065-2067).

Checked and **inconsistent or ambiguous**: W11 (169,083 vs 1,435,806 similarity edges), W12 (eigenvector vs PageRank-alpha), W13 (38.1% vs 35.2% under one phrase).

## 1.5 Ratings

| Dimension | Rating | One-line basis |
|---|---|---|
| **Quality** | **2 (fair)** | Provenance discipline is excellent; the evidence chain is not. No human ground truth, validation population disjoint from the analysed unit, confounded pre/post design, run-artefact denominators, zero baselines. |
| **Clarity** | **3 (good)** | Well organized and unusually explicit about scope; costs points for two rendered contradictions, four confusable reduction operations, eleven shifting denominators, and appendix-dependence. |
| **Significance** | **2 (fair)** | Real value in the measured artifacts and the release; blunted by a 2023-capped corpus, three of five use cases being cautions, and the retrieval claim resting on two examples. |
| **Originality** | **3 (good)** | The stage schema and the artifact measurements are new; the pipeline is standard and the schema is not defended against the alternatives the paper itself cites. |
| **Overall** | **3 (borderline reject)** as a main-track submission; **4 (borderline accept)** if moved to Datasets & Benchmarks with W22/W23 closed. | Reasons to reject (no validated fidelity number for the released unit, no baseline, unmeasured scalability claim, placeholders in the abstract) currently outweigh a genuinely useful artifact and a set of correct, well-evidenced negative results. |
| **Confidence** | **4** | Read the full source; familiar with LLM-as-judge and scientific-IE literature. Did not read `refs.bib`, the receipt JSONs or the figure images, so citation resolution and figure legibility are unassessed. |

## 1.6 Questions to the authors (rebuttal-actionable, ranked)

**Q1.** What fidelity claim, if any, do you make about the 2,772-chain reporting unit? The judge covers 0.6% of it and does not score either selecting gate. If the answer is "none", say so in the abstract and drop "quality-controlled". **Score impact: +1 overall if a stratified second judge run on chain-yielding papers (n>=100) lands, or if the wording is corrected.**

**Q2.** Will you run the null-repair arm (same three graders, empty or sham repair list, same 100 papers)? This is one batch call and it decides whether Section 4.2's direction result survives at all. **Score impact: Quality 2 -> 3 if the sham arm shows a materially smaller delta than the real one; the current design supports no conclusion either way.**

**Q3.** The 20-paper human-anchored spot-check is named as not performed. Given it would reconcile 0.6% vs 28.8% --- the paper's own "clearest reason the protocol requires a human anchor" --- what blocks it? **Score impact: +1 Quality if it lands with an inter-annotator figure.**

**Q4.** What did extraction cost, in tokens, wall-clock and dollars, per document and in total? Without this, "Scalable" in the title and "cost is linear in corpus size" are unsupported. If the batch logs are gone, measure it on a 200-document re-run. **Score impact: required for the title claim to stand.**

**Q5.** Can you provide any baseline on a 200-document sample --- abstract-only extraction, a non-reasoning model, or flat triple extraction --- scored by the same judge? Currently no design choice in the paper is shown to be load-bearing. **Score impact: +1 overall.**

**Q6.** Are the five intermediate stages separable? Report agreement between two models (and ideally one human) on stage assignment for 50 documents. This is what would make the schema a contribution rather than a convention. **Score impact: +1 Originality.**

**Q7.** Please reconcile the rendered contradictions: 169,083 vs 1,435,806 similarity edges at tau=0.80 (Appendix E vs Appendix B), and eigenvector centrality (Table 6) vs the alpha-mixed transition matrix (Appendix E). **Score impact: none upward; unresolved, it costs Clarity.**

**Q8.** Will the release URL, licence pair, and datasheet exist at camera-ready? A dataset paper cannot be accepted with `\OPEN{[GAP: release URL]}` in its abstract. **Score impact: blocking.**

## 1.7 Limitations and societal impact --- assessment

**Adequately addressed, with two gaps.** The Limitations section is the strongest part of the paper: nine named limitations, each stating the control that was not run and what it would have shown. The Impact Statement correctly separates misreading from misattribution, ships three mitigations, and reaches a defensible dual-use conclusion.

Missing: (i) the licence is undetermined (L584-587), which is a live redistribution risk for a corpus derived from LessWrong/EA Forum/transcript sources under mixed terms, not a formatting gap; (ii) no limitation acknowledges that the corpus's arXiv-weighted chain set systematically under-represents the forum discourse that is a large share of the AI-safety literature, so any downstream "what does the field propose" reading inherits a venue bias the paper measures (Table 9) but never names as a societal-reading risk.

---

# PART 2 --- Writing-style audit

Scope as requested: informality/sloppiness below conference register; passages that read artificial in word choice, sentence rhythm and paragraph cohesion; low information density, fluff, and over-highlighting of low-relevance material.

## 2.1 The dominant artificial-cadence tell: contrastive correction ("X, not Y")

The paper's default sentence shape is *assert-then-negate*. Non-exhaustive, all rendered text:

- L399-400 "These figures belong in Methods and not in Results"
- L634-643 "Both figures describe the corpus and neither is evidence of fidelity"; "It is whether the extraction reflects what a paper argued, never whether that argument is correct"
- L655-656 "It should be read as 'one in six under these cuts', never as a fact about the literature"
- L706-707 "Nothing in this subsection should be read as a quality figure for..."
- L733-734 "the dominant failure mode is *omission, not hallucination*. ... It does not support reading the absence of a stage as evidence that..."
- L791-792 "measures a schema version difference, not extraction quality"
- L896-897 "That is a property of the extraction schema, not a measurement of the literature"
- L1027-1029 "is a statement about graph construction rather than about the field"
- L1087-1088 "not merely inflated but *unstable*"
- L1096-1104 three consecutive instances: "which is extraction structure and not fragility"; "measures the traversal, not the corpus"; "siloed by construction"
- L1148-1149, L1158-1160, L1265-1266, L1287-1288, L2189-2192 ("Note which failure this is. The source is a real mathematical argument, so the failure is not that... It is that...")

Roughly 30 instances. Individually each is a correct scoping statement. Cumulatively the register reads as generated, and --- worse for the review outcome --- as **pre-emptively defensive**: a reader who is told twenty times not to over-read a result concludes the results are fragile. **Fix:** keep the negation where a plausible misreading exists and the paper has evidence about it (L655, L706, L1027 are earned). Convert the rest to plain declaratives. Target: fewer than 8 in the whole paper.

## 2.2 Epigrammatic paragraph-enders

Short aphoristic closers, one per paragraph, in a regular rhythm --- the strongest single "unhuman" signal in the manuscript:

- L400 "A corpus of N short chains is exactly N components until something links them."
- L643 "A faithful extraction from a weak paper is a successful extraction."
- L995 "The layout, the colors and the chains are identical in the two panels, so the star is one preprocessing step and nothing else."
- L1002-1004 "The group is a connected component in similarity space, not a set of restatements of one risk."
- L1168-1169 "A risk taxonomy says what can go wrong and a directory says who wrote about it."
- L1387 "That claim would be confident and wrong."
- L1194 "Verifying the whole corpus is a matter of inference budget and not of method."
- L1102-1104 "The general form of the control is one question."

Two or three of these are good writing. Eight is a tic. **Fix:** keep L400 and L643 (both do real explanatory work); delete or fold the rest into the preceding sentence.

## 2.3 Register breaks --- documentation voice inside a paper

- L797 "This section is for the reader who takes the release and does something with it." --- colloquial and vague; "does something with it" is below conference register.
- Every `\paragraph{Control.}` block (L869, L914, L948, L1040, L1085) is second-person imperative README prose: "Do not merge near-duplicates...", "Treat node-level clusters as a browsing index...", "Read the maturity distribution as composition...". Useful content, wrong register, and it repeats what the release documentation is said to carry anyway (L1388-1389).
- L869 "**Control.** None." and L914 "**Control.** None on the counts" --- two near-empty paragraphs occupying a structural slot purely for parallelism.
- **Fix:** convert Control blocks to declarative sentences inside the subsection ("Centrality on this graph requires the un-merged substrate and the exclusion of within-category similarity edges"), and delete the two empty ones.

## 2.4 Informal or imprecise diction

| Line | Text | Problem |
|---|---|---|
| 638 | "a schema that bends a little under pressure" | metaphor, unquantified |
| 464 | "the structure barely moves" | "barely" is doing quantitative work; give the delta (89.0 -> 87.4) and drop the adverb |
| 702 | "we describe what it returned in some detail" | meta-narration; delete |
| 1017 | "no ranking exists to report" | overstated: a ranking exists, its top-2 gap is 3% |
| 1160 | "an obvious application we do not attempt" | "obvious" is editorializing |
| 1334 | "The extension we would most like to see" | personal preference in a conclusion |
| 1758-1764 | column header "**xrisk**", row label "un-merged" | jargon abbreviation in a table header |
| 833, 845 | "**intv**" | abbreviation not defined in the table it appears in (defined in the caption below) |

## 2.5 Anglicisation is mixed

British: "randomisation" (546), "neighbourhood" (928), "domain-specialised" (1124), "favourable" (1416), "colour"-adjacent "capitalise" (73, unavoidable, it is a package option). American: "normalization" (982), "labeling"/"labelings" (929, 1832), "colored" (960), "modeled" (15), "de-duplication" hyphenation varying against "deduplication" (1698 heading vs 454 body). Pick one variety and one hyphenation and sweep.

## 2.6 Number-word inconsistency

"Twelve of the 100 judged papers" (L2022) vs "12 of the 100 judged papers" (L521, L1198). "Seven chains (0.3%...)" (L1657) vs "7 of the 100" (L217). Fix to one rule.

## 2.7 Over-emphasis

`\emph{}` appears 40+ times in the body, frequently on ordinary words ("*not*", "*read*", "*documents*", "*unstable*", "*post-hoc*", "*proposed*", "*same*"). Italic emphasis at this density stops functioning. Keep it for term-of-art introductions (EDGE layer, SIM layer, the five stage names) and for the two or three genuine scope-critical negations; strip the rest.

## 2.8 Dramatic and absolute language

Accurate finding, dramatized verbs: "**manufactures** a centrality super-hub" (title of 5.4, and L986, L1004, L1332), "the merge **baked** the concentration into the structural topology" (L1023), "it **breaks** centrality" (L985), post-repair agreement "**collapses**" (L768, L1216), the two measurements "**disagree by a factor of fifty**" (L746), "Frontier-era discourse is therefore **absent, not merely under-represented**" (L1286-1287), "which is **the weakest point in the evidence**" (L1224), "**The sharpest** is a merge-manufactured centrality hub" (L1332). Each is individually defensible and the underlying numbers support them; the density is what reads as pitch rather than report. Trim roughly half, keeping the strongest instance of each finding.

Attribute over-assignment is **less** of a problem than the brief anticipated: I found exactly one emphatic "essential" ("That qualifier is essential", L903) and no rhetorical use of "critical". The dramatization runs through verbs and epigrams instead, per above.

## 2.9 Low information density and duplication

**D1 --- The headline numbers appear five times.** 200,525 / 2,772 / 1,868 / 87.4% / 15.9% appear in the abstract (L104-126), the introduction (L189-190), Research Goals (L209-213), Results 4.1 (L612-620), and the Conclusion (L1320-1323). Research Goals and the Results-section preamble (L602-607) are the two that can go almost entirely; the goals bullets restate the section structure the reader is about to encounter.

**D2 --- Section 5's opening paragraph (L797-806) says the same thing three ways:** "Two of the five need no control and are contributions; three are artifact-dominated"; then "The first two need nothing beyond the stored chains"; then "The last three are the analyses a graph of this shape invites". Cut to two sentences.

**D3 --- Figure 1's caption is ~190 words and carries interpretation, hedging and a forward reference** ("Neither reading is measured here"; "we read this panel as coarse composition, not as a measured rate"). Captions should say what is plotted. Move the hedges to L634-643 where the same point is already made.

**D4 --- The abstract's closing sentence is a 40-word unbroken promissory clause** (L124-126): "A mechanism-indexed corpus of the field could support acceleration of AI safety progress through research coordination, systematic search for under-addressed risk--mechanism pairs, and graph-based reasoning over stored arguments to explore new AI safety solutions." Nothing in the paper measures any of the three. "Acceleration of AI safety progress" is slogan diction. Cut to one clause or delete; the Outlook already carries it (L1299-1312) at length.

**D5 --- The title is weak.** "Chains in Graphs: Scalable Framework for Extraction of AI Safety Mechanisms from Published Literature" --- "Chains in Graphs" is uninformative and near-tautological, the article is missing before "Scalable Framework", and "Scalable" is the unmeasured claim of W5. Consider: *"Extracting Risk-to-Intervention Mechanism Chains from the AI Safety Literature, and What the Resulting Graph Does Not Support"* --- or any title that names the unit (mechanism chain) and does not assert scalability.

**D6 --- Redundant restatement of the same caution in adjacent sections.** The "maturity is LLM-assigned and unscored" caveat appears at L147-151 (figure caption), L554-558 (Methods), L881-886 (Results), L914-916 (Control), and L1263-1266 (Limitations) --- five times. Once in Methods with a single cross-reference is enough.

## 2.10 Over-highlighted low-relevance material

**H1 --- Appendix F (Topical Cluster Reference, L1845-1902) spends a full column on a bare 40-item list** of cluster names, for an analysis the paper itself demotes to "a browsing index, not a taxonomy" and whose silhouette in the real embedding space is 0.004. This is the clearest instance of length signalling importance that the content does not have. Cut to 8 illustrative names in a sentence, and ship the full list with the release.

**H2 --- Appendix G's threshold and hop paragraphs (L1908-1936)** are inherited from the frozen draft, use stale numbers (W11), and support no main-text claim, since every quantitative claim uses EDGE-only paths. Table 8 (sim-hops) shows reach growing ~20x per hop, which is arithmetic. Delete, or update and reduce to one sentence.

**H3 --- Appendix H (Race-Framing Classifier Validation, L1946-2004)** validates a classifier for a finding the paper withdrew from the main text as unsupportive ("supports no claim about the discourse structure of the field", L1981-1982), and still carries precision/recall strata, a corpus prevalence paragraph and a three-row odds-ratio table with CIs and p-values. Either the OR residue is a finding --- in which case state it in the main text --- or the appendix should be one paragraph reporting that the earlier 88%/44x result did not reproduce and the classifier was sound.

**H4 --- Figure 1's middle panel (a third of the page-1 figure) plots intervention maturity**, an attribute the paper says four times is unvalidated and descriptive only. A page-1 figure should carry the paper's strongest evidence. Replace that panel with the gate-sensitivity curve (Table 1's yield column) or with the omission-measurement comparison, both of which are load-bearing.

**H5 --- Table 5's "Result" column** ("Too conservative", "False clusters", "**Selected**") assigns verdicts without the recall evidence that Section 5.4 later shows contradicts the verdict (W19). Add the recall column or drop the verdicts.

## 2.11 Source-file hygiene (pre-submission)

Not a style issue in the built PDF, but it will be read by anyone who gets the .tex: the file carries an extensive internal editorial trail addressed to co-authors --- "REMOVED 2026-08-14..." (L378-384), "Moved out of sec:r-hub 2026-08-14..." (L1777-1778), "NOTE: the frozen Overleaf reported..." (L395-397, L425-427, L1677-1678, L1684-1685), "the module docstring says 80%, the code uses 70%" (L459-460, L482-483), the compute-acknowledgment gate block (L1352-1375), and the WITHDRAWN-paragraph note (L889-893). Several disclose internal disagreement, an Overleaf workflow, and the existence of a private compute donor. Strip or move to a separate NOTES file before any public posting, and re-run `texlint.py`'s GAP count as the removal checklist the header describes (L61-63).

---

# PART 3 --- Prioritized fix list

**Blocking for submission (do first):**
1. Close all `\OPEN{}` GAPs, above all the release URL in the abstract and the licence pair (W22).
2. Remove EleutherAI/SOAR from Acknowledgments if double-blind; decide main track vs D&B (W23).
3. Fix the two rendered contradictions: similarity edge count, centrality method (W11, W12).
4. Drop "quality-controlled" for the chain set, or qualify it at every occurrence (W1).

**Highest score-per-effort experiments (in order):**
5. Null-repair grader arm --- one batch call, decides whether Section 4.2 stands (W3, Q2).
6. Human-anchored 20-paper spot-check --- reconciles 0.6% vs 28.8% (W2, Q3).
7. Re-run the grader rubric at one fixed prompt over all 100 papers --- removes the 43/95/13 denominator problem (W4).
8. Cost measurement on a 200-document re-run --- makes "Scalable" defensible (W5, Q4).
9. Abstract-only + non-reasoning-model baseline on 200 documents (W6, Q5).
10. Second judge run stratified on chain-yielding papers (W1, Q1).
11. Repeat-run agreement on 300 documents --- gives the descriptives a noise floor (W10).
12. Stage-assignment agreement study, 50 documents (W20, Q6).

**Writing (mechanical, one pass each):**
13. Cut contrastive-correction constructions from ~30 to under 8 (2.1).
14. Delete six of the eight epigrammatic closers (2.2).
15. Convert the five Control blocks to declarative prose; delete the two empty ones (2.3).
16. One spelling variety, one number-word rule, `\emph{}` down to term introductions (2.5-2.7).
17. Delete Research Goals and the Results preamble; cut Section 5's opener to two sentences; shorten Figure 1's caption (D1-D3).
18. Cut the abstract's closing promissory sentence; retitle (D4, D5).
19. Cut Appendix F to 8 examples; delete or update Appendix G; reduce Appendix H to one paragraph (H1-H3).
20. Swap Figure 1's maturity panel for gate sensitivity or the omission comparison (H4).
21. Strip the internal editorial trail from the .tex (2.11).
