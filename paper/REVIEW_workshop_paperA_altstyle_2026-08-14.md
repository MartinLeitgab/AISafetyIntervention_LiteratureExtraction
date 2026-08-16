# Simulated Workshop Review — `paperA_altstyle.tex`

**Reviewed file:** `../AISafetyIntervention_PaperA_shared/paperA_altstyle.tex` (1,537 lines; repo `main` pulled 2026-08-14, up to date)
**Companion review:** `paper/REVIEW_neurips_paperA_altstyle_2026-08-14.md` (main-track calibration, Overall 3 / borderline reject)
**Rubric:** `paper/neurips-scoringguidance.txt` dimensions, re-calibrated to workshop norms
**Calibration used:** a workshop accepts *lighter contributions* — work in progress, negative results, position and resource papers — but holds the same bar on **rigor of what is actually claimed, honesty of scoping, writing quality, and completeness of the submitted artifact**. Lower novelty ceiling, identical integrity floor.

---

## VERDICT

| | |
|---|---|
| **As submitted today** | **Reject — incomplete submission.** Ten `\OPEN{[GAP: ...]}` markers render as yellow highlights in the PDF, including a missing release URL *in the abstract*, an empty Appendix A where the extraction prompt should be, and "Author List Placeholder" on the title. This is a draft, not a submission. |
| **On content, assuming the placeholders are closed** | **Accept (poster).** Good fit for a workshop, with two required corrections (W-1, W-2 below) that cost editing time, not experiments. |
| **Overall recommendation** | **Weak Accept — conditional on W-1, W-2, W-3 and closing the placeholders** |
| **Confidence** | **4** — read the full manuscript and appendices, re-derived the arithmetic, ran a style-frequency scan; did not run the released code. |

**Sub-scores (workshop calibration; main-track scores in brackets for contrast):**

| Dimension | Workshop | [Main track] | Why the calibration moved — or did not |
|---|---|---|---|
| Quality | **3 (good)** | [2] | Missing baselines and human validation are legitimate work-in-progress at a workshop. Two *mis-scoped claims* are not, and they hold the score at 3 rather than 4. |
| Clarity | **3 (good)** | [3] | Unchanged. Genuinely well-written prose, held down by a missing funnel figure, four confusable reduction operations, and measurable stylistic repetition (see Section D). |
| Significance | **3 (good)** | [2] | Rises. At a workshop the artifact catalogue and the failed self-reproduction are exactly the contribution the venue exists to circulate. |
| Originality | **2 (fair)** | [2] | Does **not** rise. Omitting the argument-mining literature is a positioning error, not a scope limitation; four citations fix it and cost nothing. |

---

## A. What workshop standards waive here — do not spend the rebuttal on these

Listing these explicitly so the authors do not over-correct. Each is a real limitation; none should block a workshop acceptance, **provided it is named in Limitations and not contradicted elsewhere in the paper**:

1. **No comparative baseline** (abstract-only, non-reasoning model, flat triples). Already declared. One sentence of future work is sufficient here.
2. **No extrinsic/downstream evaluation** of the corpus against a retrieval baseline.
3. **Verification at n=100 rather than corpus scale.** Explicitly framed as a proof of principle, which is a workshop-appropriate claim.
4. **No human adjudication** — *conditionally* waived. Waived as a missing experiment; **not** waived as a scoping obligation (see W-2).
5. **Single extractor, single run**, no repeat-extraction stability check.
6. **Proprietary, deprecating extractor** (`o3`) with no open-weights replication.
7. **Corpus recency** — ARD thins after 2023.
8. **Deferred capabilities** (mechanism-family clustering, under-addressed risk-mechanism search) left to Outlook, with the reason given.
9. **Reference count (16)** as a raw number — a short paper does not need 60 references. What is *not* waived is the specific missing literature (W-5).

Everything below this line is something a workshop reviewer should still hold you to.

---

## B. Strengths

- **B-1. The paper is honest in a way that is rare and directly useful to a workshop audience.** It publishes two of its own prior headline findings failing to reproduce (88% race-framing -> 2.6%; 44-fold gradient -> 2.0-fold; 51/100 -> 2/100), and an appendix exhibiting a chain that passes every structural filter and is nonetheless spurious (Euclid's infinitude-of-primes proof cast as a cryptographic-risk intervention). Workshops exist partly to circulate exactly this. Keep both, prominently.
- **B-2. Claim-to-receipt discipline.** Every number carries an on-disk receipt and one released script re-derives 42/42 numeric claims. I re-derived roughly twenty figures by hand; the arithmetic holds except for five small items (Section E). This is above the norm for the venue class.
- **B-3. Correct refusal to read structure as fidelity.** Sec. 3.1 states that 87.4% five-stage completion is what schema-filling predicts, names the two controls that would separate the hypotheses (schema ablation, degraded-source), and explicitly rejects an out-of-domain run as a substitute. That reasoning is worth more than the number it disclaims.
- **B-4. The "Control." paragraph after each use case is an excellent device** and the most reusable thing in the paper. "Do not merge near-duplicates before computing centrality," backed by a measured 90.1x manufactured super-hub, is a warning other literature-KG projects need.
- **B-5. The "some statistics are true by design" closing note** (path diversity, path completion, siloing as pipeline consequences) generalizes beyond this corpus and would survive as a standalone workshop talk.
- **B-6. Related Work positions against GraphRAG cleanly** — instrumental graph vs. graph-as-object-of-study, provenance-carrying edges as the consequence. The best paragraph in that section.

---

## C. Weaknesses that survive workshop standards

### W-1 (required fix). The verification sample does not cover the analysed substrate, and the paper states the opposite.

The 100 judged papers are sampled from *successfully extracted* documents: LessWrong 28, Alignment Forum 21, blogs 15, EA Forum 11, arXiv 11, Arbital 6, special 4, YouTube 2, aisafety.info 2 (App. G). The 2,772-chain reporting unit is a different mix: arXiv 1,055 (38%), Alignment Forum 326, LessWrong 267 (9.6%), EA Forum 265, YouTube 238, MIRI 120 (App. B). Only 15.9% of documents yield a chain at all, so the judged sample plausibly contains on the order of sixteen chain-yielding papers.

App. G nonetheless asserts: *"The audited sample is forum-weighted in the same way the corpus is."* That is true of the 11,779-document corpus and false of the chain set every headline number is computed over — and the error runs in the flattering direction.

This is not a missing experiment; it is a claim that does not hold as written, so a workshop does not waive it. **Minimum fix, no new compute:** report how many of the 100 judged papers are among the 1,868 chain-yielding ones, delete or correct the representativeness sentence, and scope Sec. 3.2 to "extraction over documents" rather than implying it validates the analysed chains.

### W-2 (required fix). The abstract selects the flattering measurement that the body declines to prefer.

Abstract: *"with the judge proposing additional nodes for **only** 7 of the 100 papers."* Sec. 3.2 reports, three paragraphs later, 216 missed concepts (5.02 per profiled paper) and concludes omission is the dominant error mode — then states, admirably, *"We report both rather than the more flattering one, and we do not know which is closer to the truth."*

The abstract does the opposite of what that sentence promises. "Only" editorializes a 50x-contested number into evidence of completeness. Delete "only", or state both figures. This is a one-word edit and it is the first thing a skeptical reader will notice, precisely because the body is otherwise so scrupulous.

### W-3 (required fix). The reporting unit is gated by an attribute the paper says is unvalidated, and this is never listed as a limitation.

The 2,772-chain set requires `intervention_maturity >= 3` and `edge_confidence >= 3`, both LLM-assigned. Fig. 1 (middle) shows ~85% of interventions are maturity 1-2, so the gate discards the large majority of the corpus on an uncalibrated four-point scale that Sec. 2.6 states the judge does not score. The paper flags maturity as unvalidated *for the Fig. 1 panel* but not for the fact that it **selects the entire analysis set** — which makes "one document in six yields a complete chain" a function of an unmeasured judgement rather than a property of the literature.

**Minimum fix, no new compute:** add one Limitations paragraph naming the gate as a selection mechanism. **Better, cheap:** recompute the four headline descriptives at maturity >= 2 and >= 1 from data you already have and report them as a three-row sensitivity table. If they are stable, the concern mostly dissolves and you gain a result; a workshop reviewer would read that table as a strength.

### W-4. The meta-grader "improvement" is close to non-informative by construction, and the paper stops one sentence short of saying so.

Graders receive source text + extraction + the judge report *including its proposed fixes*, and score before/after. Sec. 2.6 already states, to the authors' credit, that no repaired graph was built or re-scored. The unstated consequence: a grader shown a list of plausible repairs and asked whether things are better after them will say yes. There is no blinding, no order randomization, and no sham-repair control. So "every grader records an improvement on every paper" is what a demand characteristic predicts, not independent evidence of a quality gain.

A workshop waives *running* the sham control. It does not waive naming the confound. Add one sentence: the pre/post design measures whether proposed repairs read as improvements, and cannot separate that from a presentation effect; a null-repair control is the fix and was not run.

Related, and cheap: Fleiss' kappa is computed on 0-100 scores binned into four ordinal bands (cut-points never stated), at n=13, with one grader saturated at 95.8 +/- 1.8. Report ICC or Krippendorff's alpha on the raw scores alongside it, state the cut-points, and consider that the post-repair "collapse to kappa = 0.09" may be a ceiling artifact rather than a disagreement finding.

### W-5. Argument mining is not cited at all — a positioning error, not a scope limitation.

Extracting claim-premise structure, argumentative zoning, and discourse-role labels from scientific text is an established field. This paper's task is a domain-specialized instance of it, and Related Work covers scientific IE, agentic KGs, GraphRAG, safety resources and LLM-as-judge without ever naming it. A reader who knows that literature will ask what is new beyond a domain and a seven-label scheme, and the paper cannot answer because it does not raise the question.

Also unresolved: two live `[CITE:]` placeholders in Related Work (AI Safety Atlas, named in the text but uncited; PICO/evidence-synthesis extraction, where the analogy is drawn with no reference). Evidence-synthesis automation is doubly worth citing because it has the human-validated fidelity designs this paper lacks — it gives you both a citation and a template for Q2.

Four to six references and a two-sentence differentiation paragraph close this. Until then Originality stays at 2 regardless of venue.

### W-6. Cross-dimensionality silhouette comparison is not valid.

Sec. 4.3 and Table 5 contrast direct-1536D silhouette (0.022) with UMAP-150D silhouette (0.298), and the caption calls it a *"13-17x silhouette improvement."* UMAP explicitly optimizes local neighbourhood structure, so silhouette computed on UMAP coordinates is inflated by construction and is not commensurable with silhouette in the original space; the Kaufman-Rousseeuw interpretation thresholds cited alongside do not transfer to a learned embedding either. Either evaluate both label sets in the original space, or drop the multiplier and present the UMAP row as "usable for browsing" without the comparison. Your conclusion — clustering is too weak to carry mechanism families — is unaffected and arguably strengthened.

### W-7. Model configuration is under-specified for a paper that releases a pipeline.

Not stated anywhere: the embedding model and version behind "1536-dimensional OpenAI embeddings"; `o3` temperature / reasoning-effort / seed; whether schema constraint used enforced structured output or post-hoc JSON parsing; retry policy; JSON parse-failure rate; UMAP hyperparameters and the procedure that chose k=40. The Compute paragraph carries no wall-clock, token count or cost — an open `[GAP]` in the source. Every item is a sentence you already know the answer to, and their absence is disproportionately costly for a paper whose pitch is "run this pipeline on a larger corpus."

### W-8. Extraction-failure accounting is missing.

Sec. 2.2.2 says extraction "fails outright on a minority of ARD records" — no count, no denominator, no breakdown between empty-body records, parse failures and schema violations. App. G later implies at least 441 failures repairable in principle. The paper reports 11,779 as the working corpus without saying what it was filtered from. A five-row intake table fixes this and feeds directly into the funnel figure recommended in W-9.

### W-9. Two structural presentation gaps do more damage than any single missing experiment.

- **No data-reduction funnel.** Four confusable operations are described only in prose: quality cuts (confidence / maturity / single-risk), containment de-duplication of paths, the node merge (measured but **not applied** to this substrate), and similarity thresholding. The manuscript's own source comments record that readers already conflated two of them and that a subsection was renamed for that reason. One small figure — raw ARD records -> 11,779 parseable -> 200,525 nodes -> 8,954 raw paths -> 2,772 chains -> 1,868 papers, each arrow labelled with its operation and marked applied vs. measured-only — is the highest-value single addition to the paper.
- **No pipeline or schema figure.** The sole figure is a three-panel dataset descriptive. A reader assembles the node/edge schema, the seven-stage chain and the extract-then-judge flow entirely from prose. For a resource paper this is the figure the audience expects, and at a workshop — where many readers will only see the poster — its absence is costly.

### W-10. Which graph the reader receives is never stated.

Multiple (non-rendering) source comments record that earlier internal numbers came from a merged 200,061-node graph with a similarity layer roughly 8.5x sparser than the dump used here. That substrate difference is part of why Sec. 4.5's reproduction failed — so a reader of the PDF cannot follow the reproduction narrative, because they cannot see that two substrates exist. One sentence in Methods naming the released dump and noting that prior internal analyses ran on a different one makes Sec. 4.5 interpretable instead of merely candid.

### W-11. The containment de-duplication is asserted lossless rather than shown.

Sec. 2.5 drops any path whose node set is >=70% contained in a longer same-document path, on the ground that it "carries no chain that the longer one does not already carry." At 70% containment a 10-node path can carry three nodes absent from its container — a different branch of the argument. Either give the argument in one more sentence, or spot-check twenty dropped paths and report what fraction carried nothing new. Worth doing because this step defines the reporting unit; the source also records that the module docstring says 80% while the code used 70%.

---

## D. Writing and presentation

The prose is genuinely good — active voice, little filler, disciplined verbs, and headings that let a reader reconstruct the argument from the table of contents. What follows is calibrated to that standard, not against a lower one. Counts are from a frequency scan over the manuscript's non-comment body.

### D-1. One construction carries too much of the paper: "X rather than Y" appears **44 times**.

At roughly 9,900 words of prose that is one every ~225 words, and it clusters in exactly the passages doing the most argumentative work: *"omission rather than fabrication," "composition rather than measurement," "corpus loss rather than a recoverable pool," "a batched job rather than an agent," "measured rather than corrected," "graph construction rather than the field," "inference budget rather than method," "scale rather than method," "read from the document rather than imposed on it," "vocabulary proximity rather than a mechanistic relationship," "misreading rather than misuse,"* and a paragraph heading that is itself *"...we name it rather than reconcile it."*

The construction is the paper's core move — distinguishing what a number measures from what a reader will assume it measures — so it *should* appear. But at this density it stops registering, and by Section 4 the reader is pattern-matching rather than reading. Cut it to roughly fifteen occurrences: keep it where the contrast is the point, and elsewhere use a plain negation ("this is not X; it is Y"), a colon, or simply state Y and drop X. Two adjacent instances in one paragraph is always one too many.

### D-2. Punctuation density: **67** em-dash pairs and **47** semicolons.

Both are used correctly, which is why the effect is cumulative rather than jarring — but several sentences carry a dashed aside *and* a semicolon *and* a trailing subordinate clause. Sec. 4.4's centrality paragraph and Sec. 4.5's control paragraph are the worst cases. Target: no sentence with more than one dashed interruption; convert roughly a third of the semicolons into full stops. This costs no content and buys noticeably faster reading.

### D-3. The abstract is 275 words across 7 sentences — mean 39 words per sentence.

Three sentences are 44, 58 and 66 words. The 66-word third sentence carries the entire pipeline description (extraction stage + five stage names + judge stage) in one breath; the 58-word closing sentence carries the release, the invitation, and three future applications at once. Split both. Target 200-220 words and a mean under 28. Most workshops cap abstracts at 200-250 words, so this is a submission-mechanics issue as well as a style one.

### D-4. Paragraph headings mix three grammatical forms.

Within a single section the reader encounters noun phrases (*"Judge configuration."*, *"Coverage."*, *"Compute."*), full declarative sentences with a first-person editorial stance (*"The two measurements disagree by a factor of fifty, and we name it rather than reconcile it."*), a bare fragment (*"One measurement artifact to separate before quoting any error rate."*), and a semicolon-joined clause pair (*"Direction of the judge effect is robust; magnitude is not."*). Pick one register per section. The declarative-claim style works well at subsection level in Sections 3 and 4 and should stay there; inside a section, short noun-phrase heads read better and scan faster.

Also: **`\paragraph{Coverage.}` appears twice** with different content — once in Sec. 3.2 (judge coverage findings) and once in App. G. Rename one.

### D-5. Register drift in the control paragraphs.

The paper is first-person plural throughout (**52** instances of "we") and then switches to second person in exactly three places, all in Sec. 4.4: *"If you do merge, report the recall..."*, *"...the node block to which your downstream metric is most sensitive."* The imperative control voice is effective and I would keep it — but make it consistently imperative across all five Control paragraphs rather than second-person in one and impersonal in the others.

### D-6. Redundancy costs roughly a page.

The core numbers (11,779 / 200,525 / 2,772 / 1,868 / 87.4% / 2.5% / 90.1x) appear in the abstract, again in the Research Goals bullets, again in Sec. 3.1, again in Sec. 4.2, and again in the Conclusion — which is close to a verbatim restatement of the abstract. Strip the numbers from the Research Goals bullets (state the claims, cross-reference the sections) and cut the Conclusion to three sentences. That alone funds the funnel figure from W-9.

### D-7. Smaller items.

- Stage abbreviations (pa / ti / dr / im / va) are defined only inside the Table 1 caption, but used in the table body and again in App. I. Define at first use in Sec. 2.2.
- Table 1 consumes a full-width `table*` for two examples while the paper has one figure. Compress to one column, or trade one query for the schema diagram.
- Sec. 4's dual role is unsignposted: it holds two working use cases and three negative results, and a skimming reader cannot tell contribution from caution. One lead sentence — "two of the five need no control; three are artifact-dominated and we report them as cautions" — fixes it.
- The released artifact is never described: no file manifest, no schema listing, no license name ("a permissive license"), no pointer to `REPRODUCE.md`. For a resource paper this belongs in the main text, not only in the repository.
- Sec. 2.2 points forward to App. A for the prompt; App. A is empty. Any reviewer who follows the pointer hits a placeholder.
- Fig. 1's caption interprets rather than reports: chains longer than seven are "longer because they carry several nodes for one or more stages" and shorter ones "carry genuine gaps." Neither is measured. Soften to "consistent with" or measure it.

---

## E. Arithmetic and internal consistency

I re-derived every reported figure. These check out: node inventory (19,096 + 27,748 + 26,361 + 27,543 + 34,222 + 28,596 + 36,959 = 200,525); intermediates 144,470; maturity counts summing to 36,959 with all four percentages correct; length histogram 1,775 + 715 + 282 = 2,772 at 64.0 / 25.8 / 10.2%; 2,423/2,772 = 87.4%; 1,868/11,779 = 15.9%; 9/100 = 0.09; 216/43 = 5.02; 7/43 = 16.3%; 14/43 = 32.6%; 184/218 = 84%; 48/52 = 92%; 614/27,748 = 2.21%; 23/441 = 5.2%; EC ratios 33.6x and 90.1x; App. G source types summing to 100.

Five items to fix:

1. **Table 8:** Opus 69.7 -> 73.1 is +3.4, printed as **+3.5**. Rounding from unrounded means, presumably — but print the difference of the displayed values, or add a decimal.
2. **Sec. 4.4 vs Table 4:** "flattening ... by a factor of **31**." The table's own printed values give 0.0328 / 0.0011 = 29.8x; the receipt's unrounded values give 30.7x. Make the printed ratio consistent with the printed inputs.
3. **Table 4, row 2:** EC1/EC2 shown as **1.0x** where the receipt records 1.03. Minor, but this row carries the "no ranking exists to report" claim.
4. **Sec. 3.2 uses "7" for two different quantities two paragraphs apart** — 7 of 100 papers with proposed added nodes, and 7 of 43 papers containing fabrications. Disambiguate in prose.
5. **Abstract length**: 275 words by my count (see D-3), above the 200-250 cap most workshops impose.

---

## F. Fit, framing and page budget

**F-1. Lead with the artifact catalogue, not the corpus size.** The main text runs ~7,500 words plus ~2,300 in appendices — roughly 8-9 two-column pages before figures. Most workshops cap at 4, 6, 8 or 9 pages including references. If the cap is 4-6, the current structure cannot survive proportional trimming, and the right cut is not uniform: **the strongest workshop paper here is "we built a mechanism-extraction pipeline, and here are four ways graph analysis over it goes wrong, including two of our own findings that did not survive re-derivation."** That framing (a) makes the negative results the contribution rather than an apology, (b) makes the absent baseline and absent human validation *appropriate* rather than missing, since the claims no longer depend on them, and (c) is more useful to a workshop audience than another resource announcement.

**F-2. Concrete cut plan to ~6 pages**, in the order I would cut:
1. Conclusion -> three sentences (currently a restatement of the abstract). **-0.4 pp**
2. Numbers out of the Research Goals bullets; claims and cross-references only. **-0.2 pp**
3. Table 1 from full-width to single-column, one query kept in full and one compressed. **-0.4 pp**
4. Sec. 4.5 (framing analysis) compressed to one paragraph plus the control, with the odds-ratio table left in App. F where it already lives. **-0.5 pp**
5. Sec. 2.4 / 2.5 merged and tightened once the funnel figure (W-9) carries the structure visually. **-0.4 pp**
6. Apply D-1 and D-2 throughout. **-0.3 pp**

Then **spend** roughly 0.6 pp buying back the funnel figure and the pipeline/schema figure. Net: ~6 pages with both missing figures added, and nothing of substance lost.

**F-3. Keep, under any page limit:** the Euclid failure case (App. I), the failed self-reproduction (Sec. 4.5), the five Control paragraphs, and the "true by design" closing note. These are the paper's identity. If the page limit forces a choice between the corpus descriptives and these, cut the descriptives.

---

## G. Questions for the authors (3)

**Q1.** How many of the 100 judged papers are among the 1,868 that yield a quality-cut chain, and what is that overlap's source-type composition? This number should already exist in your receipts. It determines whether the verification stage bears on the analysed chains at all, and it decides whether App. G's representativeness sentence stands or must be deleted (W-1). *Answering this is the single largest factor in my final score.*

**Q2.** Can you report the four headline descriptives (five-stage completion, yield, length distribution, source-type mix) at `maturity >= 2` and `>= 1`, and at `edge_confidence >= 2`? No new inference required — the paths already exist. If they are stable, say so and W-3 becomes a strength; if the 15.9% yield moves materially, report it as a range across the gate.

**Q3.** App. I shows a chain passing every structural filter while being wholly spurious. Even without a full human audit, could one author read a random 20 chains against their sources and report a rough spurious rate with a caveat that it is indicative and unadjudicated? A workshop does not require a validated protocol here — but "we hand-checked 20 and n were spurious" transforms App. I from an anecdote into a bound, and it is a couple of hours of work.

---

## H. Limitations and impact assessment

**Limitations: strong, with three specific omissions.** The section is better than most I see at any venue — sample-scale verification, LLM-internal validation, the non-stratified 43-paper subset, structural completeness not being fidelity, no baseline, LLM-assigned attributes, single extractor, corpus staleness. Missing: (i) the maturity gate as a selection mechanism on the reporting unit (W-3); (ii) the mismatch between judged and analysed populations (W-1); (iii) the pre/post grader confound (W-4).

**Impact Statement: the thinnest section in the paper, and one omission is substantive.** It covers misreading — a reader quoting cluster sizes or centrality as facts about the field — and Sec. 4 is explicitly built to make that harder. Good. But the released graph asserts, *with attribution to named documents by identifiable authors*, that those documents argue particular things, and App. I proves the pipeline sometimes invents a framing the source never asserts. A citable graph that mis-attributes an argument to a real researcher's post is a distinct and concrete harm, and it needs a stated mitigation: per-node provenance back to the source text, a documented correction path, and a shipped warning that extracted claims are model assertions *about* a document, not quotations *from* it. Two further gaps: licensing and redistribution terms are unaddressed (the corpus includes LessWrong, EA Forum, Alignment Forum and YouTube-transcript content; the release license is called "permissive" but never named), and dual-use is not considered at all — a map of which risks have only weak or unvalidated interventions is also a map of under-defended surfaces. I do not think that blocks release; the statement should say the authors weighed it.

---

## I. What changes my score

- **To Accept (oral-worthy):** W-1 answered with the overlap number and the representativeness sentence corrected; W-2's "only" removed; W-3's sensitivity table added; the funnel and pipeline figures added; placeholders closed. All editing and re-tabulation, no new compute.
- **To Weak Reject:** the representativeness claim in App. G left standing, or the 0.09-vs-5.02 discrepancy resolved by dropping the unflattering number rather than adjudicating it. The paper's whole credibility rests on not doing that.
- **Unchanged by:** adding baselines, corpus-scale verification, or downstream evaluation. Those are main-track requirements; at this venue they belong in Outlook, where they already are.

---

## Reviewer's closing note

The unusual thing about this submission is that its self-criticism is more rigorous than its central claim. It publishes its own failed reproductions and a spurious extraction, then states in the abstract that the judge proposed additional nodes for "only" 7 of 100 papers. Fix that asymmetry — scope the verification claim to the population it actually covers, drop the editorializing adverb, and disclose the maturity gate — and this is a good workshop paper whose negative results are worth more to the audience than its headline counts.
