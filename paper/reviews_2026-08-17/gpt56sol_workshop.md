## Summary

This paper introduces a large-scale, schema-guided LLM pipeline for extracting risk-to-intervention argument chains from 11,779 documents in the Alignment Research Dataset. It produces a 200,525-node provenance-bearing graph and a filtered reporting set of 2,772 chains. A second-provider LLM audits samples of the extraction, while several ablations and graph-analysis case studies expose schema-completion, gating, merging, clustering, and centrality artifacts.

The task is useful and the proposed artifact could be valuable. The paper is unusually candid about negative results and construction artifacts. However, the central scientific question—whether the extracted chains faithfully represent what the source documents argue—remains unresolved. Evaluation is entirely LLM-internal, the audit statistics are not conventional omission rates, the shipped and audited representations may differ, and paths are enumerated while ignoring edge direction and without enforcing stage monotonicity. Thus, the work is best viewed as an interesting, openly reported workshop-stage resource and pipeline, not yet a validated scientific information-extraction dataset.

For a workshop, I lean **borderline accept**, conditional on making the artifact available and materially narrowing the claims.

---

# Strengths and Weaknesses

## Strengths

### Quality

1. **The paper tackles an important and genuinely difficult extraction target.**  
   Extracting a multi-stage, document-internal argument is more ambitious than metadata indexing, topical clustering, or isolated entity/relation extraction. Representing provenance-bearing paths from risks through rationales and mechanisms to interventions is potentially useful.

2. **The engineering effort and scale are substantial.**  
   Processing 11,779 heterogeneous full-text documents and producing a graph with 200,525 typed nodes is a meaningful resource contribution. The one-request-per-document architecture is operationally simple and plausibly scalable, although the actual call and cost claims need correction for retries and failures.

3. **The authors report many important sensitivities rather than hiding them.**  
   Particularly valuable observations include:
   - chain yield changes from 15.9% to 99.4% under different LLM-assigned gates;
   - the prompt strongly induces the canonical chain;
   - sentence-shuffled and out-of-domain material can still yield apparently valid chains;
   - node merging can manufacture a centrality super-hub;
   - UMAP-space silhouette scores are not comparable to scores in the original embedding space;
   - individual chain membership is unstable across repeated model runs.

   These are useful warnings for researchers constructing LLM-generated knowledge graphs.

4. **The provenance and release design are thoughtful in principle.**  
   Retaining an unmerged graph, source URLs, node/edge attributes, raw enumerations, receipts, and scripts is preferable to releasing only a highly processed graph. The claim-to-script-to-receipt design would be a strong reproducibility feature if the artifact is actually available.

5. **The paper is commendably frank about limitations and ethics.**  
   It explicitly discusses misattribution, licensing uncertainty, venue skew, schema-induced completion, run-to-run instability, non-human validation, and the danger of treating graph-construction artifacts as findings about the field.

### Clarity

1. **The overall pipeline and data reductions are explained in considerable detail.**  
   Figures 1–3, the enumerator constraints, the population table, and the gate sensitivity analysis help disentangle several otherwise confusable datasets.

2. **The paper generally distinguishes extraction fidelity from truth of the source claim.**  
   This is an important conceptual distinction: a faithful extraction of a weak argument is still a successful extraction.

3. **The appendices contain unusually extensive implementation and prompt details.**  
   An expert reader can understand the intended schema, model settings, path filters, and most downstream analyses.

### Significance

1. **A trustworthy mechanism-indexed safety corpus could be useful.**  
   It could support direct retrieval, literature navigation, mapping of interventions to risks, and eventually evidence-gap analysis.

2. **The negative methodological lessons generalize beyond AI safety.**  
   The findings about schema completion, model-assigned gates, transitive deduplication, and graph metrics are relevant to LLM-generated scientific knowledge graphs more broadly.

3. **The work is appropriate for workshop discussion.**  
   It is early, empirical, and openly reports unresolved failures. That is a reasonable workshop contribution even though the resource is not yet sufficiently validated for strong substantive conclusions.

### Originality

1. **The task formulation and resulting dataset are relatively novel.**  
   The method itself—schema prompting, embeddings, graph ingestion, and LLM auditing—is not algorithmically novel, but the risk-to-intervention mechanism-chain representation is a useful domain-specific synthesis.

2. **The paper offers original empirical observations about its own construction pipeline.**  
   The merge-manufactured centrality hub and the strong dependence on LLM-assigned gates are more original and convincing than several of the proposed downstream literature findings.

---

## Weaknesses

### Quality

1. **There is no human-anchored fidelity evaluation. This is the central limitation.**  
   The extractor, judge, meta-graders, maturity labels, confidence labels, and stage labels are all produced or assessed by LLMs. Cross-provider agreement does not establish correctness because both models use the same definitions and similar linguistic priors.

   The paper therefore does not establish:
   - node precision or recall against source documents;
   - edge precision or recall;
   - whether a complete path is a coherent argument actually made by the source;
   - calibration of edge confidence;
   - calibration of intervention maturity;
   - whether the 70% path-collapse heuristic preserves distinct arguments.

   The Euclid failure case is especially consequential: a chain can pass all gates while inventing a safety framing. This demonstrates that the current gates are not reliable fidelity filters.

2. **The extraction prompt conflicts with the stated objective of recording what a paper argues.**  
   The prompt says to use “moderate inference” when the source lacks a required flow and to infer an intervention when none is explicit. It also requires every path to begin at a risk and end at an intervention. This encourages completion of the schema rather than extraction of source-supported arguments.

   The confidence gate is not a sufficient remedy because confidence is assigned by the same model that invented the chain. The paper’s own failure case shows that unsupported framing can receive confidence and maturity values high enough to pass.

3. **The path definition does not guarantee a semantically valid mechanism chain.**  
   Enumeration ignores stored edge direction and allows all relation types. It apparently does not require stage order to be monotonic. Consequently, a path can:
   - traverse a causal relation backward;
   - visit stages in a non-canonical order;
   - combine relations that do not jointly express a risk-reduction mechanism;
   - reach an intervention through graph connectivity rather than an argumentatively coherent chain.

   Reporting that 87.4% of paths contain all five stage labels does not answer this problem. A path can contain all labels while being directionally or logically incoherent.

4. **The “omission rates” are not well-defined omission rates.**  
   For example, 302 missing relationships divided by 1,667 graph edges gives 18.1%, but the 302 items come from a separate “expected relationships” list. A recall-like denominator would instead involve the expected items, not all extracted graph edges. The expected-list statuses sum to 776, although the paper says there are 777 rows:
   \[
   328 + 146 + 302 = 776.
   \]
   Similarly, 557 proposed nodes divided by 2,110 existing nodes is a relative addition ratio, not the fraction of relevant nodes omitted.

   Table 15’s statement that all three measurements are “upper bounds on omission” is also unjustified. An unadjudicated LLM can both invent missing items and fail to identify actual omissions; these quantities are neither guaranteed upper nor lower bounds.

5. **The comparison between the two audit samples is confounded, yet the paper draws a directional conclusion.**  
   The second audit uses longer documents and graph-reconstructed extractions lacking fields present in the first audit. The paper acknowledges this but then states that omission in the analyzed population is “at least as high” and calls the result a “direction and a floor.” That conclusion does not follow from the design. The difference could arise from representation, document length, population, model instability, or their interaction.

6. **It is unclear whether the audited artifact is the shipped artifact.**  
   Section N says the judge audited pre-ingestion extraction JSON containing referential-integrity and orphan issues, while those defects do not survive into the released graph. The judge’s re-serialized graphs also have substantially fewer edges than the released graph for the same documents. This conflicts with the claim that “the stage we audit is the stage we ship.” The mapping from audited JSON to final graph needs to be explicit, and ideally the final released representation should itself be audited.

7. **The 70% containment collapse is insufficiently validated.**  
   It removes:
   - 18.0% of distinct raw risk-to-intervention pairs;
   - 6.1% of enumerated nodes entirely;
   - many paths reaching interventions absent from their retained container.

   The authors correctly acknowledge that it is unknown whether these are duplicates or distinct arguments. Given that the 2,772 chains are the main reporting unit, this heuristic needs at least a human annotation study or a stronger sensitivity analysis.

8. **Several baselines are weak or structurally unfair.**
   - The flat-triple baseline cannot express the target chain by construction.
   - Abstract-only and non-reasoning comparisons use very small, selected samples.
   - The baseline documents are often conditioned on yielding a chain under the shipped system.
   - Endpoint recovery against an earlier stochastic run of the same system is not a gold-standard fidelity metric.

   These experiments are informative diagnostics, but they do not establish superiority over a competing extraction or retrieval system.

9. **The compute accounting is presented too strongly.**  
   The abstract calls 122.4M input tokens “measured,” but the logs did not survive and the value is reconstructed for successful documents under one nominal request each. Retries and failed records may have generated additional calls and tokens. Likewise, visible output is extrapolated from one surviving response, and the reasoning-token band is assumed. This should be called a reconstructed nominal estimate, not a measured bill.

10. **The submission is incomplete as presented.**  
    The release URL is a placeholder, and the acknowledgments and AI-assistance statement contain unresolved GAP markers. An anonymous author placeholder is understandable during review, but an unavailable artifact prevents verification of the main resource contribution.

### Clarity

1. **The main paper is far too long and repetitive.**  
   It repeatedly restates the same distinctions:
   - released graph versus analyzed chain set;
   - structural completeness versus fidelity;
   - node versus edge omission instruments;
   - gate-selected results versus claims about the literature;
   - merged versus unmerged graph artifacts.

   These are important once, but the repetition obscures the central result.

2. **The prose often uses rhetorical, slogan-like constructions rather than direct scientific exposition.**  
   Examples include “We report them all and reconcile none,” “One evaluation design that answers nothing,” and “Two sevens in this subsection are unrelated.” These may be memorable, but they make the paper read like an analysis diary rather than a finished scientific report.

3. **Important results are interleaved with project history and failed internal analyses.**  
   References to “earlier drafts,” “an earlier internal pass,” grader sessions that drifted, and prior non-reproducing findings make it difficult to identify the final experimental design.

4. **Some terminology overstates what is measured.**  
   “High-confidence,” “mature,” “omission rate,” and “complete mechanism” are stronger than the underlying LLM-assigned or structurally defined quantities warrant.

### Significance

1. **Utility depends strongly on fidelity, which is currently unknown.**  
   A mechanism-indexed graph can be valuable only if users can trust that a source actually makes the represented argument. Provenance helps users check claims, but it does not substitute for dataset-level quality measurement.

2. **The corpus stops at 2023.**  
   This substantially limits immediate usefulness for a fast-moving AI-safety literature. The pipeline may be more valuable than the current snapshot.

3. **The retrieval demonstration lacks an evaluation.**  
   Two worked examples show addressability, not retrieval quality. There is no query set, relevance judgment, user study, or comparison against full-text semantic retrieval/RAG.

4. **Many graph-level use cases are shown to be unreliable.**  
   This is a useful methodological lesson, but it narrows the demonstrated practical impact of the released graph.

### Originality

1. **The pipeline is mainly a composition of existing techniques.**  
   The novelty lies in the schema, scale, artifact, and diagnostics—not in a new extraction, auditing, or graph-learning method.

2. **The paper needs a stronger comparison to modern LLM information-extraction and knowledge-graph construction work.**  
   The related-work section covers scientific IE, argument mining, and GraphRAG, but the empirical comparison is almost entirely against internally designed degraded variants rather than external extraction systems or benchmarks.

---

# Scores

| Dimension | Score | Rationale |
|---|---:|---|
| **Quality** | **2 / 4 — Fair** | Large and carefully documented pipeline, but central source fidelity is not human-validated; path semantics and audit metrics have material validity problems. |
| **Clarity** | **3 / 4 — Good** | Well organized and unusually explicit, but substantially overlong, repetitive, and occasionally confusing due to multiple incompatible audit instruments. |
| **Significance** | **3 / 4 — Good** | Potentially useful workshop resource and broadly relevant methodological cautions, though present utility is limited by unknown fidelity and a pre-2024 corpus. |
| **Originality** | **3 / 4 — Good** | Novel task/schema/resource and useful artifact analyses, despite relying mainly on standard LLM extraction and graph-processing components. |

## Overall

**4 / 6 — Borderline Accept**

For a workshop, the reasons to accept narrowly outweigh the reasons to reject: the task and artifact are interesting, the scale is meaningful, and the authors report failures with unusual honesty. The acceptance case depends on presenting this as an early, unvalidated resource rather than a measured map of the safety literature.

I would increase this to **5 (Accept)** if the authors provide an accessible artifact and a credible blinded human fidelity evaluation showing useful node-, edge-, and path-level precision. I would decrease it to **3 (Borderline Reject)** if the artifact remains unavailable or if the audited representation is materially different from the shipped graph without a final-graph audit.

## Confidence

**4 / 5**

I am confident in the assessment of the experimental design and evaluation limitations. Some uncertainty remains about the exact contents of the unavailable release and about conventions in this domain-specific literature.

---

# Questions and Actionable Suggestions

1. **Can the authors provide human-anchored node-, edge-, and path-level evaluation?**  
   Please have at least two blinded human annotators inspect a stratified sample of source documents and report:
   - precision of extracted nodes and edges;
   - recall against independently annotated source claims;
   - whether each retained path is a coherent argument made by the source;
   - agreement and adjudication;
   - accuracy/calibration of edge confidence and intervention maturity.

   Sampling should include chain-yielding and non-chain-yielding documents, major source types, and document-length strata. Actual human results showing useful fidelity would increase my Quality score from 2 to 3 and likely the Overall score from 4 to 5. Merely promising this as future work would not.

2. **How many reported chains are directionally and logically valid?**  
   Please quantify:
   - stage-order violations;
   - edges traversed against their semantically intended direction;
   - paths using each relation type in each direction;
   - paths whose intervention is connected through a structurally valid but argumentatively incoherent route.

   A useful comparison would be directed versus undirected traversal, and traversal with versus without monotonic stage-order constraints. If a substantial fraction of the 2,772 chains are invalid under these checks, the paper should stop calling them mechanism chains and my Overall score would decrease.

3. **Please redefine and rerun the omission analysis.**  
   The current ratios are not standard omission rates. Please:
   - audit exactly the final released representation;
   - use the same schema and fields for both populations;
   - give the judge explicit node- and edge-repair slots;
   - define precision/recall denominators before evaluation;
   - report confidence intervals;
   - resolve the 777-versus-776 arithmetic discrepancy;
   - avoid describing unadjudicated LLM findings as upper bounds.

   At minimum, rename the current quantities as “judge-proposed additions relative to extracted size” and “judge-listed missing relationships relative to extracted edge count.”

4. **Can the 70% path-collapse heuristic be validated or replaced?**  
   Please annotate a sample of retained/dropped path pairs as:
   - duplicate rendering of the same argument;
   - granularity variant;
   - distinct argument;
   - invalid path.

   Alternatively, use paper-level sets of risk–intervention pairs as the primary reporting unit and retain all paths as evidence. The present loss of 18% of raw endpoint pairs is too large to treat as incidental.

5. **Will the complete anonymous artifact be available during review?**  
   The release URL is currently a GAP marker. Please provide the graph dump, sample IDs, extraction JSON, prompts, model identifiers, path files, receipts, and scripts through an anonymous link. Also distinguish nominal requests from API attempts and reconstruct cost over both successful and failed records. If the artifact cannot be inspected, the paper’s main resource and reproducibility claims are not verifiable.

---

# Limitations

yes

---

# Additional Writing and Presentation Flags

## 1. Unscientific, unacademic, overly informal, or LLM-like writing

The paper is generally grammatical, but it repeatedly uses an LLM-like cadence of emphatic fragments, binary contrasts, and self-commentary. This contributes heavily to the excessive length.

1. **Abstract:**  
   > “We report them all and reconcile none.”

   This is rhetorically dramatic but scientifically unhelpful. Replace it with a direct statement that the instruments are not comparable and no aggregate fidelity estimate is claimed.

2. **Section 3.2:**  
   > “One evaluation design that answers nothing.”

   Too informal and absolute. Prefer: “The pre/post repair comparison is confounded and is therefore excluded from the analysis.”

3. **Section 3.2:**  
   > “Two sevens in this subsection are unrelated.”

   This is editorial housekeeping that should not appear in a final paper. Rename the quantities or rewrite the paragraph so no disambiguation is necessary.

4. **Section 3.2:**  
   > “Three omission measurements, spanning a factor of thirty, and what separates them.”

   This reads like a blog heading. Use a conventional heading such as “Comparison of audit instruments.”

5. **Section 3.1:**  
   > “The categories are ones the model reaches for unprompted; the seven-node chain that traverses all five is what the prompt supplies.”

   “Reaches for” anthropomorphizes the model. State instead that free-form labels can be mapped to the proposed categories, while full-chain completion depends strongly on the prompt.

6. **Section 4.4:**  
   > “what follows measures what the step costs a reader who does.”

   Awkward and conversational. Use: “We therefore evaluate the downstream effect of applying this merge.”

7. **Section 4.5 and elsewhere:**  
   Frequent references to “an earlier pass,” “earlier drafts,” and findings that “did not survive” make the paper read as a project log. The final paper should report the final protocol and results; development history belongs in version control, not the scientific narrative.

8. **Section 6.3:**  
   > “The extension we would prioritize…”

   This is unnecessarily first-person and subjective. Use a concise future-work statement.

9. **Unresolved placeholders:**  
   The release URL, compute acknowledgment, contribution statement, and AI-assistance scope remain marked as GAPs. These make the submission visibly unfinished. The author-list placeholder may be appropriate for anonymization, but the artifact and substantive declarations must be finalized.

10. **General style:**  
    The repeated pattern “X, and what it does/does not mean,” “the control each one needs,” “not X but Y,” and “two things follow” is overused. It resembles generated rhetorical scaffolding and should be replaced with shorter declarative prose.

## 2. Outsized-importance language

There are few literal uses of “critical” or “essential,” but several passages serve the same rhetorical function.

1. **Introduction:**  
   > “the paired extract-and-audit design outlasts it”

   This overstates a routine cross-provider LLM audit that is not human-calibrated and produces incompatible measurements. Suggested revision: “The pipeline may be reusable across newer corpora, subject to domain- and model-specific validation.”

2. **Sections 1 and 4.1:**  
   > “a resource whose unit is the document cannot represent it, query it, or compare it across papers”  
   > “The dataset supports a query that document-level resources cannot serve”

   Too absolute. Full-text retrieval and RAG systems can answer such questions, although they do not natively expose a structured path. Suggested revision: “Document-level indexes do not directly expose this relation as a structured, comparable field.”

3. **Section 3.2:**  
   > “What the run establishes is a direction and a floor”

   The confounded comparison establishes neither. Delete this conclusion.

4. **Section 4.2:**  
   > “The skew is a four-bucket call with a very large margin and would survive a substantial per-item error rate”

   This is unsupported without a quantified sensitivity analysis. Delete or provide an explicit worst-case calculation.

5. **Section 4.4:**  
   > “Centrality on this graph requires the un-merged substrate and the exclusion of within-category similarity edges, in that order”

   The evidence supports this recommendation for the tested merge and eigenvector-centrality configuration, not as a universal requirement. Scope the statement accordingly.

6. **Conclusion:**  
   > “The corpus answers mechanism-level queries that document-level resources cannot.”

   Again, use “supports direct structured mechanism-level queries not natively represented by document-level indexes.”

---

# 3. Non-load-bearing “sausage-making” to cut

1. **Confounded pre/post repair scoring — Section 3.2, “One evaluation design that answers nothing.”**  
   **Cut outright.** The paper draws no result from it, and the experiment was knowingly unblinded and uncontrolled. One sentence in the audit limitations is enough: “We exclude unblinded pre/post repair scores.”

2. **Failed race-framing result — Section 4.5 and Appendix M.**  
   **Cut the main-body section outright.** It reports an earlier 88%/44-fold finding that does not reproduce, followed by a generic lesson about selected heads. This does not validate the extraction pipeline or establish a substantive corpus result. If retained at all, one sentence can appear in an appendix on analysis pitfalls.

3. **History of the earlier internal substrate — Section 2.2 final paragraph, Section 2.7 exceptions, and the setup to Sections 4.3–4.5.**  
   **Cut references to the earlier pass.** Report only the released substrate and the final controlled experiments. The internal graph’s node count and previous similarity density are development history.

4. **Uninterpretable meta-grader pre/post scores and session mechanics — Sections 2.6, 2.7, 3.2, and Appendix B.**  
   **Cut the scoring component unless it is rerun under a standardized protocol.** Interactive sessions with schema drift and denominators of 95/95/13 do not support a scientific result. If the 43-document taxonomy is retained, state its limited provenance once.

5. **Judge recovery of failed extractions — Section 2.2.2.**  
   **Cut the 441-document/23-recovery experiment.** It is peripheral to the released successful-extraction graph and does not inform the quality of the reported chains. Retain only the extraction-failure counts and causes.

6. **Detailed rejected merge configurations and ANN implementation history in the main text — Section 4.4.**  
   **Move to the appendix.** The central insight—that transitive merging can manufacture a hub—is useful. The exact rejected configurations, candidate counts, and engineering history are not needed in the main narrative.

7. **Detailed catalog of clustering attempts — Section 4.3.**  
   **Move to the appendix, not delete.** The UMAP silhouette pitfall is useful, but the list of configurations “tried and abandoned” is not central.

I would **not** classify the sentence-shuffling control or the Euclid failure case as sausage-making. They are load-bearing evidence that structural validity and gate passage do not imply source fidelity.

---

# 4. Length triage: reducing the main body from approximately 15 to 10 pages

Estimates below are approximate and assume that concise one-paragraph summaries remain in the main paper where necessary. The list is ordered by pages recovered per unit of score risk.

| Rank | Material and location | Action and rationale | Estimated recovery | Running total | Effect on scores |
|---:|---|---|---:|---:|---|
| 1 | Repetitive limitations, **Section 6.1** | **Compress heavily** to approximately 0.7–0.8 pages. Move detailed run-to-run numbers, model comparisons, source-type tables, and gate-specific repetitions to the existing appendices. Keep the lack of human validation, gate instability, venue skew, collapse loss, and corpus cutoff. | **1.1 pages** | **1.1** | None; likely improves Clarity |
| 2 | Failed framing analysis, **Section 4.5** | **Delete outright.** The non-reproducing race-framing result is not part of the final contribution. Retain at most one general warning about selection-conditioned statistics. | **0.65 pages** | **1.75** | None |
| 3 | Audit editorial material, **Section 3.2**: “two sevens,” failed pre/post design, repeated three-way reconciliation, schema-mismatch detail | **Delete the failed design and compress/move diagnostic detail to the appendix.** Keep one compact table or paragraph reporting both audit populations, the incompatible instruments, and the absence of human adjudication. | **0.65 pages** | **2.40** | None |
| 4 | Detailed cost reconstruction and model repricing, **Section 2.7** | **Move to appendix.** Keep one sentence with nominal input scale and an explicitly qualified cost range. Current derivation is too uncertain for main-body emphasis. | **0.45 pages** | **2.85** | None |
| 5 | Detailed graph-analysis case studies, **Sections 4.3–4.4 and Figure 4** | **Move details and Figure 4 to appendix.** Retain approximately half a page summarizing two reusable cautions: UMAP-space evaluation inflation and transitive-merge centrality artifacts. These are useful but secondary to the extraction resource. | **1.45 pages** | **4.30** | Minor risk only; no numerical score drop if preserved in appendix |
| 6 | Auxiliary small-sample experiments, **Section 3.1**: three simpler pipelines, probe decomposition, second-annotator confusion details | **Move to appendix.** Keep one paragraph with the key conclusions and numbers: prompt-induced chain completion, shuffled-text chains, and cross-provider stage agreement. Keep gate sensitivity in the main body. | **0.9 pages** | **5.20** | Minor; no numerical score drop if preserved in appendix |
| 7 | Detailed drop-versus-container diagnostics, **Section 2.5** | **Move to appendix as a buffer.** Keep the collapse rule and the headline losses—18% endpoint-pair loss and 6.1% node loss—in the main text. | **0.3 pages** | **5.50** | None |

This reaches the five-page target without removing the central method or its most important limitations.

## Material I would refuse to move or cut

1. **The task definition and canonical schema, including a compact Figure 1.**  
   The **Originality** and **Clarity** scores depend on understanding what is being extracted.

2. **Corpus funnel and released artifact definition, including the distinction between 200,525 nodes, 8,954 raw paths, and 2,772 retained chains.**  
   The **Quality** and **reproducibility** assessment depends on this.

3. **Model configuration and prompt-induced inference policy.**  
   The instruction to infer missing stages and interventions is central to interpreting fidelity. Removing it would materially reduce **Quality** and **Clarity**.

4. **Path constraints, especially ignored edge direction, stopping rules, maturity/confidence gates, and the 70% collapse.**  
   These determine what a “chain” means. The **Quality** score depends directly on them.

5. **Gate sensitivity headline results.**  
   The movement from 15.9% to 99.4% document yield is one of the paper’s most important and credible findings. Cutting it would reduce **Quality**.

6. **A concise core audit section covering both populations, the incompatible results, and absence of human adjudication.**  
   The paper’s acceptance case depends heavily on its honesty about unknown fidelity. Removing these caveats would reduce **Quality** and could lower the Overall score.

7. **At least one explicit schema-invention control or failure case.**  
   The shuffled-source result or Euclid example must remain visible, even if the full details are in an appendix. The **Quality** assessment depends on acknowledging that gate-passing chains can be spurious.

8. **A compact related-work section.**  
   The **Originality** score depends on distinguishing this task from argument mining, scientific IE, knowledge-graph construction, and document-level literature maps.

9. **Artifact availability, reproducibility statement, impact statement, licensing discussion, and AI-assistance disclosure.**  
   The **Quality**, ethics, and Overall scores depend on these. They should be finalized and shortened, not removed.