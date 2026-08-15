# Outstanding studies — scoped, costed, waiting on a decision

**Written 2026-08-15.** For the next reviewer round to prioritize against. Every entry is
scoped to the point where it could be started the same day.

Two studies from the reviewer backlog are **done** and are not repeated here: the
extraction cost measurement (issue #152, PR #154) and the stage-separability probe
(issue #153). Both were Class B and cost nothing.

**Ground rules carried into every estimate below.** Token counts come from the measured
corpus: median document 1,976 tokens, mean 5,648, and a mean extraction of 17 nodes and
17 edges. Dollar figures assume batch pricing at roughly USD 1–3 per million input tokens
and USD 4–15 per million output, and are order-of-magnitude only — check the rate before
committing. Every study below gets a GitHub issue and a PR carrying the analysis, per the
project rule, so the experiment trail stays auditable.

---

## Tier 1 — highest reviewer value per dollar

### S1. Null-repair grader arm
**Answers** NeurIPS W3/Q2, workshop V4 — the single item both reviews rank first.
The meta-grader pre/post comparison is confounded: graders saw the repairs they were asked
to score. The paper now reports that stage as a design lesson and draws nothing from it.
One batch call decides whether it can be a result at all.

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
+1 overall.

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

---

## Tier 2 — needs a person, not a budget

### S4. Human-anchored spot-check, 20 papers
**Answers** W2/Q3. The judge says extractions omit 0.6% of what they should contain; the
Opus grader says 28.8%. The paper reports both, reconciles neither, and calls this the
clearest reason the protocol needs a human anchor.

**Design.** 20 papers, stratified across source types. For each, an annotator reads the
source and the extraction and records: nodes the extraction missed, nodes it asserts that
the source does not support, and stage assignments they would change. Adjudicate against
both machine measurements. A second annotator on 5 of the 20 gives an inter-annotator
figure, without which the anchor is one person's opinion.

**Cost.** Zero dollars. **~2–4 hours of one author's time**, plus ~1 hour for the second
annotator's subset. I can generate the annotation packet — the 20 papers, their
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

## What is deliberately not on this list

- **Extending the corpus past 2023.** Both reviews note ARD stops at 2023. Running the
  pipeline over a 2024–2026 arXiv safety slice is now costed by S1's sibling measurement:
  at USD 32–118 per 1,000 documents, a 5,000-document slice is **USD 160–590**. That is a
  scope decision for the team, not a study.
- **MIT risk-anchoring of a chain subset.** Paper B material, deliberately out of scope.
- **Anything requiring the frozen co-author substrate.** Three appendix tables still come
  from it; re-deriving them on the released graph is bookkeeping, not a study, and is
  tracked in the manuscript's own carve-out in `sec:m-repro`.
