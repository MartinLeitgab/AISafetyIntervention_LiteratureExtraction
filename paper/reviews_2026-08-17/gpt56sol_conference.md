## Summary and recommendation

This paper introduces a large-scale, full-text LLM pipeline for extracting typed risk-to-intervention argument chains from 11,779 documents in the Alignment Research Dataset. It produces a 200,525-node provenance-bearing graph and a filtered reporting set of 2,772 chains. A cross-provider LLM judge audits two 100-document samples, and the paper presents several downstream analyses and extensive diagnostics of graph-construction artifacts.

The task and resulting artifact are potentially valuable, and the paper is unusually candid about its negative results, instability, and limitations. However, the central object—the extracted mechanism chain—has not been evaluated against human annotations. The LLM-only audit is internally inconsistent, uses non-comparable denominators, and does not evaluate the two attributes that determine the main reporting set. In addition, path extraction ignores edge direction and apparently does not enforce monotonic stage order, so an enumerated “chain” need not be a logically directed risk-to-intervention argument. The arbitrary path-collapse rule removes 18% of distinct risk–intervention pairs, and chain membership is unstable across reruns. These issues substantially weaken the technical validity of the released chain set and its claimed uses.

I lean **borderline reject**. A modest, stratified human evaluation and clarification/correction of the path semantics and audit estimands could materially change my assessment.

---

# Strengths and weaknesses

## Strengths

### Quality

1. **A useful and well-motivated extraction target.**  
   Extracting the internal argument connecting a risk to an intervention is meaningfully different from clustering papers by title, abstract, or topic. The typed stages—problem analysis, theoretical insight, design rationale, implementation mechanism, and validation evidence—provide a potentially useful representation for literature navigation.

2. **Substantial scale and provenance.**  
   Processing 11,779 full documents and retaining source-level provenance is a nontrivial engineering contribution. The separation between single-document structural edges and cross-document similarity edges is conceptually clean and important.

3. **Strong sensitivity and artifact analysis.**  
   The gate sweep in Table 7 is particularly informative: it shows that document yield moves from 15.9% to 99.4% depending on two model-assigned thresholds. The analysis of merge-induced centrality artifacts is also useful and demonstrates that seemingly substantive graph findings can be consequences of graph construction.

4. **Unusually honest reporting.**  
   The paper reports schema forcing, shuffled-input failures, run-to-run instability, confounded grader experiments, and a concrete spurious chain. This candor is a major strength. The authors frequently distinguish corpus composition from claims about the field.

5. **Considerable reproducibility detail in the manuscript.**  
   Prompts, rubrics, traversal constraints, model configurations, and population definitions are documented more thoroughly than in most LLM extraction papers.

### Clarity

1. The main pipeline is easy to understand, and Figures 1–3 communicate the extraction and filtering process effectively.
2. The EDGE/SIM distinction and provenance model are explained clearly.
3. The appendices provide unusually complete methodological documentation.
4. The paper repeatedly states which analyses are descriptive and which should not be interpreted as findings about the AI safety literature.

### Significance

1. A trustworthy mechanism-indexed corpus could be useful for literature review, research coordination, evidence mapping, and identifying repeated or missing intervention mechanisms.
2. The graph-construction diagnostics are relevant beyond AI safety. In particular, the merge/centrality and UMAP/silhouette results are useful warnings for knowledge-graph and literature-mapping work.
3. The released extraction schema and provenance-bearing graph could become useful research infrastructure if fidelity is established.

### Originality

1. The specific risk-to-intervention mechanism-chain task is original and well motivated.
2. The contribution is a novel combination of full-text schema-guided extraction, provenance-preserving graph construction, and downstream artifact diagnostics.
3. The paper contains several useful negative insights about schema forcing and graph post-processing, even though the underlying methods—prompted extraction, LLM judging, embeddings, DFS, and clustering—are individually standard.

---

## Weaknesses

### Quality

#### 1. The central extraction has no human-grounded fidelity evaluation

This is the main weakness. The extractor is judged by another LLM, and the meta-evaluation is performed by further LLMs. Cross-provider agreement reduces self-preference concerns but does not establish source faithfulness. The stage relabeling result, \(\kappa=0.84\), establishes that two models can apply the same definitions consistently; it does not show that the extracted nodes are supported by the source or that the five-stage ontology is appropriate.

This concern is not hypothetical. The paper presents a fully gate-passing Euclid/prime-number chain whose risk framing and intervention are invented. The shuffled-sentence control also produces chains for 30% of documents. These results imply that structural conformance and model-assigned edge confidence are inadequate substitutes for fidelity evaluation.

The paper deserves credit for acknowledging this, but acknowledgment does not make the corpus technically validated.

#### 2. The headline “omission rates” are not well-defined omission rates

The three reported quantities use flagged additions divided by the size of the existing extraction:

- \(9/1617=0.6\%\) proposed added nodes,
- \(302/1667=18.1\%\) missing relationships relative to existing graph edges,
- \(216/751=28.8\%\) grader-recorded missed concepts relative to existing nodes.

These denominators are not aligned with the judged items. In particular, the judge’s coverage list contains 777 expected relationships, of which 302 are marked missing; within that instrument, the missing fraction is \(302/777=38.9\%\), not 18.1%. On the second run it is \(476/627=75.9\%\), although the paper instead reports \(476/2192=21.7\%\) by dividing by graph edges that were not necessarily enumerated by the coverage instrument.

Because the expected-edge list is itself neither exhaustive nor human-adjudicated, even 38.9% is not a corpus omission rate. The quantities are better described as “judge-proposed additions per extracted item.” They are neither demonstrated upper bounds nor lower bounds: the judge can both invent omissions and fail to notice omissions.

This is especially problematic because these figures appear prominently in the abstract and conclusion.

#### 3. The two audit runs are not comparable

The change from added nodes in 7/100 documents to 95/97 documents is enormous. The second run differs in document length, source mixture, and extraction serialization, including omitted rationale fields. Therefore, the paper cannot infer a population “direction” or a “floor” from the comparison. The result instead indicates that the audit instrument is highly sensitive to input representation or population.

Before these results can support extraction quality claims, the authors need either a matched comparison or a diagnosis of why nominally the same protocol changed so dramatically.

#### 4. The path enumerator does not establish a directed logical mechanism chain

The enumerator ignores edge direction because relation types use mixed orientations. However, the extracted object is described as a logical risk-to-intervention mechanism chain. An undirected traversal through typed relations establishes connectivity, not necessarily a directed argument.

It is also unclear whether stage order is enforced. The listed constraints require a risk root, an intermediate first hop, and an intervention endpoint, but do not require categories to progress monotonically through the canonical order. A path could apparently move from problem analysis to validation evidence to theoretical insight and still count as containing all stages. Similarly, a low-maturity intervention can occur inside the path.

The paper should report:

- the fraction of paths with monotonic stage order;
- the fraction whose edge semantics are consistent with the claimed risk-to-intervention reading;
- results after canonicalizing relation direction;
- how many retained chains contain internal intervention nodes.

Without this, “chain” is stronger terminology than the construction supports.

#### 5. The reporting unit is arbitrary and lossy

The 70% node-containment collapse reduces 8,954 paths to 2,772 chains, but:

- 78.3% of dropped paths contain a node absent from their container;
- 28.0% terminate at a different intervention;
- 18.0% of distinct risk–intervention pairs disappear;
- the count changes from 2,658 to 5,460 as the threshold moves from 0.60 to 0.90.

The paper explicitly states that it does not know whether these are distinct arguments. Therefore, the 2,772-chain set should not be described as one record per distinct argument without manual validation. Reporting document-level or endpoint-pair-level statistics alongside the collapse would be safer.

#### 6. The two gates are unvalidated and unstable

Edge confidence and intervention maturity select the main chain set, but neither is evaluated by the judge. Moreover:

- the gates change document yield from 15.9% to 99.4%;
- they strongly change venue composition;
- on a rerun, only 9 of 18 parseable extractions from previously chain-yielding documents still yield a chain;
- the paper’s confidence rubric conflates source support with evidentiary strength.

An explicit conceptual claim may receive confidence 2 because it is not empirically supported, while a systematic qualitative claim may receive 3. Thus, “confidence \(\ge 3\)” is not simply a fidelity filter. It selects a particular kind of evidence and venue.

The main results should either be presented across gates or based on calibrated human labels. A single thresholded set is currently too contingent to be the primary reporting unit.

#### 7. The baselines and use-case evaluation are limited

The abstract-only, non-reasoning-model, schema ablation, and flat-triple comparisons are run on small internal samples, mostly \(n=25\) or \(30\). There is no independent extraction baseline, no annotated benchmark, and no statistical uncertainty.

The retrieval use case consists of two hand-selected examples with no query set, relevance judgments, source-faithfulness evaluation, or comparison against BM25, dense retrieval, or document-level RAG. The paper establishes addressability, not retrieval superiority.

#### 8. The “scalable pipeline” claim is only partly established

The extraction stage is linear in the number of input documents, but the full pipeline includes:

- a judge that is only run at sample scale;
- similarity construction that can be quadratic without approximate search;
- path enumeration that the paper itself shows can explode with modest increases in mean degree;
- output and reasoning token costs reconstructed from incomplete logs.

The title would be more defensible as “A large-scale extraction pipeline” unless end-to-end scaling behavior is measured.

#### 9. The submitted artifact is incomplete as presented

The manuscript contains a missing release URL and unresolved editorial placeholders for authorship, acknowledgments, and AI-assistance scope. The claimed release is central to the paper’s significance and reproducibility. An anonymized artifact should be available during review, and all internal `[GAP: ...]` notes must be removed from a final submission.

### Clarity

1. **The paper is much too long and repetitive.**  
   The same caveats—different populations, unvalidated gates, “the stage we ship,” and construction artifacts—are restated multiple times. The transparency is welcome, but the repetition obscures the main empirical message.

2. **Too much project history is included.**  
   Earlier internal substrates, failed prior analyses, schema-version history, grader-session drift, and stale internal passes are narrated at length. Much of this belongs in an appendix, repository changelog, or postmortem.

3. **The prose is often rhetorically stylized rather than scientific.**  
   Examples include “We report them all and reconcile none,” “One evaluation design that answers nothing,” and “Two sevens in this subsection are unrelated.” These formulations call attention to the writing rather than clarifying the estimand.

4. **Terminology is occasionally stronger than the evidence.**  
   “Chain,” “quality gate,” “complete mechanism,” and “omission rate” suggest semantic validation that has not been established.

5. **Twelve populations are difficult to track.**  
   Table 6 helps, but the main text should organize results around three primary units: all extracted documents, chain-yielding documents, and retained chains.

### Significance

1. The artifact could be significant, but its utility depends strongly on fidelity. A large graph with unknown precision and recall may create additional verification work rather than reduce it.
2. The corpus ends in 2023, which is a substantial limitation for a rapidly changing field.
3. Only 15.9% of documents enter the main chain set under the chosen gates, with strong venue bias.
4. The paper does not evaluate whether researchers actually retrieve better mechanisms, synthesize literature more accurately, or save time using the resource.
5. The scope is currently niche. The general argument-extraction methodology could broaden the impact, but the present evidence is specific to one safety corpus and one proprietary extractor.

### Originality

1. The task and artifact are original, but the technical pipeline is mostly a composition of standard components.
2. The five-stage ontology is imposed by the prompt rather than learned or derived from human argument analysis.
3. The relation to argument mining is discussed, but the paper does not compare against sentence-level argument mining, scientific claim extraction, or structured summarization baselines.
4. Some of the most interesting findings concern common analysis pitfalls rather than the proposed extraction method itself.

---

# Scores

| Dimension | Score | Rationale |
|---|---:|---|
| **Quality** | **2 / 4 — Fair** | Large and carefully documented engineering effort, but central fidelity is not human-evaluated; audit rates are not valid omission estimands; chain semantics and gates are insufficiently validated. |
| **Clarity** | **2 / 4 — Fair** | Generally intelligible and well organized locally, but excessively long, repetitive, rhetorically stylized, and burdened by internal project history and unresolved placeholders. |
| **Significance** | **3 / 4 — Good** | A validated mechanism-indexed corpus could be useful to AI safety and literature-mining researchers; current impact is limited by fidelity uncertainty, corpus age, and lack of user/retrieval evaluation. |
| **Originality** | **3 / 4 — Good** | Novel task, schema, and dataset combination, though the technical components are standard and the ontology is prompt-imposed. |

## Overall score

**3 / 6 — Borderline Reject**

The artifact and task are promising, and the paper’s transparency is exemplary. However, the principal claims depend on a chain set whose faithfulness, directionality, gate labels, and reporting-unit construction have not been adequately validated. The reasons to reject currently outweigh the reasons to accept, but this could change with a relatively focused human evaluation and correction of the audit/path semantics.

## Confidence

**4 / 5**

I am confident in the assessment. The central methodological and evaluation issues are visible from the paper’s own reported results, although I have not inspected the unreleased graph or code.

---

# Questions and actionable requests

## 1. Can the authors provide a human-annotated fidelity evaluation?

Please annotate a stratified sample covering at least arXiv, LessWrong/Alignment Forum, transcripts, chain-yielding documents, and non-chain-yielding documents. At minimum, report:

- node precision and recall against source-supported concepts;
- edge precision and recall;
- validity of the complete risk-to-intervention path;
- whether each intervention is explicitly proposed;
- human ratings of edge confidence and maturity;
- inter-annotator agreement and an adjudication protocol.

A smaller but rigorous human study would be more informative than additional LLM graders.

**Score impact:** convincing human evidence of reasonable chain-level precision and useful recall could raise my overall score from **3 to 4 or 5**. Evidence of substantial schema-induced fabrication would lower it to **2** unless the release is reframed as an unvalidated candidate graph.

## 2. What exactly is the audit estimand, and can the headline rates be corrected?

The current 0.6%, 18.1%, 28.8%, 26.4%, and 21.7% quantities divide proposed missing items by the size of an existing graph, even though the numerator and denominator are produced by different instruments. Please:

- stop calling these omission rates unless a common universe is defined;
- report the judge coverage statuses within the coverage list;
- distinguish additions per extracted item from recall;
- explain why the second judge run marks 95/97 documents for node additions;
- avoid describing any quantity as an upper bound, lower bound, direction, or floor without assumptions that justify it.

**Score impact:** a corrected analysis would improve Quality and Clarity. Retaining the current interpretation would reinforce my borderline-reject assessment.

## 3. Do retained paths preserve the semantics of a directed argument?

Please report the fraction of chains that:

- obey monotonic stage order;
- have all edge relations oriented consistently after relation-specific direction canonicalization;
- contain an intervention before the endpoint;
- require traversing at least one edge against its semantic direction;
- remain valid under directed rather than undirected traversal.

A manually evaluated sample should determine whether the resulting paths are genuinely coherent mechanisms rather than connected node sequences.

**Score impact:** if a large fraction fails this test, I would lower the overall score to **2**. If nearly all paths remain semantically coherent after canonicalization, this would materially improve Quality.

## 4. Can the gates and 70% collapse be validated or replaced with less brittle reporting?

Please evaluate maturity and edge-confidence labels against human judgments, and report chain-set stability across repeated runs or extractors. For the collapse, manually label whether dropped paths are distinct arguments, particularly those with different interventions.

I suggest making documents and risk–intervention endpoint pairs the primary stable units, with the 2,772 collapsed paths treated as one optional view. Soft weighting by maturity/confidence may also be preferable to a hard gate.

**Score impact:** evidence that the primary findings are stable to reruns and reporting-unit choice could raise Quality. Continued dependence on an unvalidated, unstable gate and lossy collapse is a major reason for my current score.

## 5. Can the retrieval claim be evaluated against realistic baselines, and can the artifact be made available?

Please provide:

- an anonymized release URL during review;
- a query set with risk/intervention/mechanism information needs;
- human relevance and source-faithfulness judgments;
- comparisons against BM25, dense full-text retrieval, and a document-level RAG baseline;
- latency/cost measurements if scalability is retained as a headline claim.

Also clarify how documents exceeding the model context limit were handled and distinguish extraction-stage scaling from similarity construction, path enumeration, and audit scaling.

**Score impact:** a working artifact is necessary for the paper’s resource contribution. If it remains unavailable, I would consider an overall score of **2**. A useful retrieval benchmark could raise Significance from **3 to 4**.

---

# Limitations

**yes**

---

# Writing-style flags

The following passages are unusually rhetorical, informal, editorial, or read like machine-polished parallel constructions rather than conventional scientific prose.

| Location | Passage or pattern | Concern and suggested revision |
|---|---|---|
| Abstract | “We report them all and reconcile none.” | Defiant/rhetorical. Replace with: “Because these instruments are not directly comparable and none is human-adjudicated, we report them separately.” |
| Introduction | “The stage we audit is the stage we ship…” / “What we analyse is an example…” | Slogan-like repetition. State the two populations once in neutral terms. |
| Throughout | “the thing we ship,” “the unit it releases,” “reads 26.4%,” “what the gates decide” | Informal or unnatural phrasing. Use “released artifact,” “estimated value,” and “depends on the gate thresholds.” |
| Section 3.1 heading | “Removing the five stages from the prompt costs the chain, not the vocabulary.” | Rhetorical headline. Use “Ablation of the stage schema reduces complete-chain yield.” |
| Section 3.1 heading | “Three simpler pipelines, and what each of them loses.” | Promotional and overly categorical for \(n=25\)–30 experiments. Use “Small-sample comparisons with simplified extraction configurations.” |
| Section 3.2 | “Two sevens in this subsection are unrelated.” | Editorial note that should be removed; rename variables or rewrite the preceding sentences. |
| Section 3.2 heading | “One evaluation design that answers nothing.” | Informal and absolute. Replace with one sentence in Limitations: “The unblinded pre/post repair scores are not interpretable and are not analyzed.” |
| Section 3.2 | “The three sort by instrument rather than by unit.” | Opaque. Explicitly define the numerator, denominator, and instrument. |
| Section 4 | “each with the control it needs” repeated in the introduction, goals, and section opening | Repetitive framing. State this once. |
| Section 4.4 | “merging near-duplicate nodes manufactures a centrality super-hub” | “Manufactures” is rhetorically loaded. “Induces a dominant centrality hub” is more neutral. |
| Section 6.1 heading | “Single extractor, and a run-to-run floor that only the counts clear.” | Opaque and stylized. Use “Run-to-run stability of extraction counts and identities.” |
| Conclusion | “the extension we would prioritize…” | First-person roadmap language is unnecessary. State the highest-priority future work directly. |
| Page 15 | All `[GAP: ...]` comments and “Author List Placeholder” | These are internal editorial notes and make the submission look unfinished. Remove before submission; provide an anonymized artifact link. |

The paper’s density of parallel constructions—“what X does and what it does not,” “one measurement…,” “two controls…,” “three omissions…”—also creates an LLM-generated stylistic impression. This does not imply that the content is machine-generated, but substantial copy-editing would make the prose more natural and scientific.

---

# Passages assigning outsized importance

1. **Abstract/Introduction: “the paired extract-and-audit design outlasts [the corpus].”**  
   This is not established. The audit design currently produces incompatible measurements and lacks human calibration. Rephrase as an intended reusable design.

2. **Sections 1 and 4.1: “a resource whose unit is the document cannot represent it” and “document-level resources cannot serve” the query.**  
   Too categorical. Document-level retrieval systems can answer mechanism questions through passage retrieval and synthesis, even if they do not store the mechanism as a first-class structured object. The defensible claim is that the proposed graph makes mechanisms directly queryable and comparable.

3. **Section 4.1: a shared stage vocabulary is “the precondition” for cross-paper analysis.**  
   It is one useful representation, not a necessary precondition. Cross-document comparison can be performed with embeddings, argument mining, or open relation extraction.

4. **Section 2.7: extractor choice moves cost “more than any other decision in the pipeline.”**  
   Only a limited set of model prices is compared, while audit coverage, similarity construction, retries, and path explosion are not comprehensively costed. Restrict the claim to the measured configurations.

5. **Section 3.2: the second audit establishes “a direction and a floor.”**  
   This is not supported because the population, document length, and serialized inputs differ. The result establishes instrument sensitivity, not a lower bound on omission.

6. **Section 6.2: “The first extension is of scale rather than of method.”**  
   This reverses the paper’s own evidence. Human-grounded fidelity evaluation and path-semantic validation should precede scaling to a larger corpus.

7. **Conclusion: “The corpus answers mechanism-level queries that document-level resources cannot.”**  
   Replace with the narrower claim that the corpus stores mechanism paths explicitly and returns them with source provenance.

---

# Non-load-bearing “sausage-making” to cut

1. **The confounded before/after repair-grader experiment**  
   - **Where:** Section 3.2, “One evaluation design that answers nothing”; related meta-grader score discussion in Sections 2.6 and B.  
   - **Cut:** Delete the narrative entirely. Retain at most one limitation sentence saying the scores were excluded because the evaluation was unblinded.  
   - **Why:** It contributes no usable result.

2. **The prior internal substrate and stale-analysis history**  
   - **Where:** Section 2.2, discussion of a “different substrate”; Section 2.7, numbers “carried over from an earlier internal pass”; Section 4 opening.  
   - **Cut:** Remove from the main paper. Repository release notes can document version history.  
   - **Why:** Readers need the final substrate and provenance of current results, not the project’s internal sequence of analyses.

3. **The failed prior race-framing result**  
   - **Where:** Section 4.5 and Appendix M.  
   - **Cut:** Delete the section and appendix, or reduce to one general warning in the graph-artifact discussion.  
   - **Why:** This is a postmortem of an unpublished internal result. It does not validate the proposed extraction pipeline or establish a substantive finding.

4. **The pre-registered confusion-pair prediction that was “half right”**  
   - **Where:** Section 3.1, second-model stage agreement discussion.  
   - **Cut:** Remove the paragraph beginning “Where the two models disagree…” except for the theoretical-insight class metrics.  
   - **Why:** The prediction is not central, and narrating which confusion pair was anticipated adds little.

5. **Reference-list-only extraction arm**  
   - **Where:** Section 3.1 and Table 16, arm G.  
   - **Cut:** Delete.  
   - **Why:** A bibliography is not a plausible competing input to the pipeline. The result is not informative beyond showing that the model sometimes refuses irrelevant input.

6. **Flat-triple arm scored on chain yield**  
   - **Where:** Section 3.1 and Table 16, arm D.  
   - **Cut:** Delete unless it is evaluated on a task that flat triples can express.  
   - **Why:** The paper notes that no chain is structurally expressible from this baseline; therefore failure on chain yield is predetermined by the representation.

7. **“Two sevens are unrelated” and similar denominator narration**  
   - **Where:** Section 3.2.  
   - **Cut:** Delete and rewrite the statistics with unambiguous variable names.  
   - **Why:** This is editing residue, not analysis.

8. **Judge recovery of failed extractions**  
   - **Where:** Section 2.2.2, 23/441 recovery result.  
   - **Cut:** Move to appendix or repository documentation.  
   - **Why:** Judge-based reconstruction of failed parser outputs is not part of the released pipeline or main evaluation.

9. **Current-model repricing and hypothetical 10× extrapolation**  
   - **Where:** Section 2.7.  
   - **Cut:** Retain measured input tokens and estimated original cost; remove speculative repricing and tenfold-corpus discussion.  
   - **Why:** Prices change rapidly and the extrapolation adds little scientific insight.

Important failures that **should not** be cut include the shuffled-input result, the spurious Euclid chain, gate instability, and the merge-induced centrality artifact. These directly reveal limitations of the method or common downstream misinterpretations.

---

# Length triage: reducing the main paper from 15 to 10 pages

The following list is ordered by estimated pages recovered per unit of score risk. “Move” assumes appendices are outside the hard 10-page main-body budget.

| Order | Material | Action and reason | Estimated recovery | Running total | Score effect |
|---:|---|---|---:|---:|---|
| 1 | **Section 4.5, framing analysis that does not reproduce** | **Delete outright.** It is an internal-analysis postmortem rather than evidence for the extraction contribution. | 0.55 pages | 0.55 | None; Clarity may improve |
| 2 | **Section 3.2 editorial/confounded material:** “Two sevens…,” repeated three-rate reconciliation, and the unblinded pre/post grader experiment | **Delete outright.** Keep a compact table defining each audit output and one limitation sentence. | 0.50 | 1.05 | None; Clarity may improve |
| 3 | **Section 2.7 detailed compute/cost reconstruction and current-model repricing** | **Move to appendix.** Keep one main-text sentence with input tokens, model calls, and the caveat that output/reasoning cost was reconstructed. | 0.55 | 1.60 | None |
| 4 | **Sections 2.2.2 and 2.3, plus detailed release plumbing in Section 2.7** | **Move to appendix.** Failure taxonomy, component counts, receipt-file architecture, retry behavior, and checkpoint sizes are reproducibility details rather than main findings. | 0.70 | 2.30 | None if the artifact and core configuration remain specified |
| 5 | **Section 4.3 detailed clustering comparison** | **Move to appendix.** Keep one paragraph: node-level clusters are suitable for navigation, not inference; UMAP-space silhouettes are not comparable. | 0.55 | 2.85 | None |
| 6 | **Section 4.4 detailed centrality case study, including Figure 4 and ANN candidate-recall discussion** | **Move Figure 4 and detailed four-condition analysis to appendix.** Keep one compact main-text table or paragraph stating the induced 90× hub and required control. | 0.90 | 3.75 | Minor; no numerical score change if the central result remains |
| 7 | **Section 2.5 detailed collapse-loss narration** | **Move to appendix.** Keep the algorithm, 8,954→2,772 count, threshold sensitivity, 18% endpoint-pair loss, and explicit statement that “distinct argument” is unvalidated. | 0.50 | 4.25 | Minor; Quality would fall if the caveats were removed rather than moved |
| 8 | **Section 3.1 detailed small-sample ablation and probe narrative** | **Move full arm-by-arm results, endpoint-recovery details, and confusion analysis to appendix.** In the main text retain: no-schema result, shuffled-input result, \(\kappa=0.84\), and one compact summary table. | 0.90 | **5.15** | Minor; no numerical score change if those four headline controls remain |

This reaches the required five-page reduction without removing the core contribution.

## Material I would refuse to move or cut

1. **Pipeline/schema overview and Figure 1**  
   - **Why:** Central to Originality and Clarity.  
   - **Score at risk:** Originality 3→2 and Clarity 2→1 if the task representation becomes unclear.

2. **Corpus and filtering funnel, preferably Figure 2 or an equivalent compact table**  
   - **Why:** Readers must see the reduction from intake to the reporting set.  
   - **Score at risk:** Quality and reproducibility.

3. **The gate definitions and full gate-sensitivity result from Table 7**  
   - **Why:** The selected set changes from 15.9% to 99.4% document yield; this is essential for interpreting every chain-level number.  
   - **Score at risk:** Quality 2→1 if hidden.

4. **The core audit results for both populations, together with the lack of human adjudication**  
   - **Why:** This is the only direct fidelity evidence offered. It should be corrected, not omitted.  
   - **Score at risk:** Quality 2→1.

5. **Evidence of schema-induced fabrication and instability**  
   - **Keep:** a concise shuffled-input result, the spurious-chain example or equivalent, and rerun gate instability.  
   - **Why:** Removing these would materially misrepresent method reliability.  
   - **Score at risk:** Quality and ethical assessment.

6. **At least one provenance-bearing retrieval example, currently Table 1**  
   - **Why:** This is the clearest demonstration of why the proposed representation could be useful.  
   - **Score at risk:** Significance 3→2 if no concrete use case remains.

7. **A concise related-work comparison with argument mining and document-level AI-safety resources**  
   - **Why:** Necessary to establish the novelty of the task.  
   - **Score at risk:** Originality 3→2.

8. **Limitations, impact, licensing, misattribution risk, and correction/removal policy**  
   - **Why:** The release can misattribute claims to identifiable authors, and the source licensing is heterogeneous.  
   - **Score at risk:** Quality and overall acceptability.

9. **A functioning anonymized artifact link and exact final release specification**  
   - **Why:** The dataset and code are a primary contribution.  
   - **Score at risk:** Overall 3→2 if unavailable.