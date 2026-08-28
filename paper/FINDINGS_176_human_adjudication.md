# #176 human adjudication — run record

Started 2026-08-27. One annotator, 30 chains. This file is the **run record**: predictions
registered before they can be checked, rules clarified mid-run, packet defects found while
working, and per-chain observations worth carrying into the write-up. It is not an open list
— that is `paper/OPEN_ITEMS.md` — and it is not annotator-facing, so nothing here should be
read while judging.

## 🔴 Pre-registered predictions

Registered before the chain named in each row was judged, so each reports honestly either
way. The project has done this once before and reported the result half-wrong on purpose
(S3's stage-agreement prediction); same discipline here.

| # | Registered | Before | Prediction |
|---|---|---|---|
| P1 | 2026-08-27 | chain 2 of 30 | **Level 3 (`unsupported`) will be rare in these 30**, because the extraction prompt licenses inference only where the source does not supply a stage, and caps it at edge confidence 1–2. The annotator's own statement of what inference should be allowed — "only if no clear pair is given, coupled with low confidence" — matches the prompt's rule, so the two agree on what a defect is before any defect has been counted. |

**What P1 does not license.** A low `unsupported` count is not evidence the extraction is
faithful: the sample is stratified by #175's reason codes, not drawn at random, and the
strata were built to over-sample the classes a machine judge flagged. If level 3 stays rare
across strata *chosen because a machine called them suspect*, that is the finding. If it
stays rare only in the faithful-both-agree stratum, that is nearly tautological. Report by
stratum, never pooled — the pooled rate over the 30 runs high by construction and
`OPEN_ITEMS.md` already forbids quoting it.

## Rules clarified mid-run

Both were ambiguities in the shipped rubric, not changes of criterion. Both are in the
generator and patched into the shipped README in place; the packet was never regenerated,
which would have reshuffled a pre-registered sample.

| Date | Rule | Trigger |
|---|---|---|
| 2026-08-27 | **Endpoints govern the verdict.** Level 3 anywhere → `unsupported`; both endpoints 0 → `faithful` regardless of body inference; an endpoint at 1–2 → `inferred_but_reasonable`. | C03. The old wording defined `faithful` by endpoints and `inferred_but_reasonable` by the worst level anywhere, so both fired on level-0 endpoints with an inferred body — an ordinary shape, so the clash would have recurred on most of the thirty. |
| 2026-08-27 | **Sub-interventions are carried, not missed.** The recall unit is the (risk, intervention) pair *as the extraction names it*; an argument refining a listed intervention is carried. | C03 produced two, both refining "get better at predicting real-life capability". |

🔴 **The second rule's limit, and the write-up must carry it:** intervention granularity is
the extractor's choice, not a property of the document. Had the extraction emitted "predict
real-world capability" as its own node, the identical argument would have been a distinct
pair. The rule measures something well-defined and cannot separate a real recall gap from a
coarser naming decision. This is the same pair-versus-mechanism choice `sec:m-reporting`
already faces one level up, where 33.9% of dropped chains differ in an endpoint and the rest
are same-endpoint re-routings called "a loss only if mechanism is the unit."

## Packet defects found while working

| Found | Defect | Disposition |
|---|---|---|
| 2026-08-27 | The shipped README printed `{N_RECALL_ARM} of the {30}` verbatim — `RUBRIC` was a plain string where the header block beside it was an f-string. | Generator fixed, README patched in place. |
| 2026-08-27 | **The chain rendering shows node names only** — no edge type, description, or confidence — while `risk_link_asserted` asks whether the document asserts "the link". The annotator cannot see what relation the extraction claims. | **Not fixed, deliberately.** Fixing means regenerating (forbidden) or shipping a supplementary sheet, which changes the instrument between chain 1 and chains 2–30. Field is read as "does the document assert *any* direct relation between these two concepts". **Report as a limitation.** |
| 2026-08-27 | `recall_enumeration.csv` had no field for how far an enumerated candidate sits from the text, so the first annotator recorded levels in free text. | Column `argument_inference_level` (0–3) added to generator and shipped CSV; C03 backfilled; the nine unstarted chains keep it blank. |

## Cost, measured

The README budgets 20-30 min per chain plus ~20 for a recall chain, so 13-20 hours overall.

| Chain | Arm | Chars | Minutes |
|---|---|---|---|
| C03 | verdict + recall enumeration | 14,627 | 60 |
| C01 | verdict only | 84,526 | 30 |

🔴 **Correction to what this file said after one chain.** On C03 alone the projection was
20-25 hours, "not the 13-20 the README states". C01 lands at 30 minutes for a verdict-only
chain six times longer, which is squarely inside the README's own per-chain estimate. The
projection is now 20 verdict-only chains at ~30 min plus 10 recall chains at ~60 min = about
**20 hours** -- the top of the stated range, not beyond it. Two points is still a projection,
not a rate; re-check at chain 5.

Note what the two rows do **not** show: any relationship between document length and time.
C01 is 5.8x the characters of C03 and took half as long. The recall enumeration, not the
reading, is what costs.

## 🔴 Pattern to watch: verdicts turning on the node NAME, not the argument

Registered after chain 2 of 30, while it is still a pattern and not yet a claim.

C01's verdict is `inferred_but_reasonable` rather than `faithful` because of **one qualifier
in one node label**: the intervention reads "perform training story sensitivity analysis
**pre-deployment**", and `pre-deployment` occurs zero times in an 84,526-character source.
Every other node in the chain is the author's own wording, twice his own coinage. The
argument the chain traces is not in dispute; a word in the name is.

If this recurs, it is a finding about **node naming** rather than about extraction fidelity,
and the paper already has the substrate to say so: `app:prompt` shows the prompt prescribes a
name template per stage, and `sec:r-stages` reports that TF-IDF on the node name alone
recovers the stage at 69.4% against a 20.0% chance rate. Names carry a lot in this pipeline,
so it should not surprise anyone if they also carry the verdicts.

Two things follow if it holds up. The fidelity rate would be partly a measurement of how
much unsupported specificity the naming template invites -- a fixable prompt property, not a
limit of the approach. And it would sharpen an already-recorded packet defect: the annotator
sees node *names* and no edge type or description, so the name is disproportionately what
they can judge.

**Do not report this from two chains.** Count the verdicts that turn on a name qualifier
across all thirty, and report the count, not the impression.

## Per-chain observations

**C03** — `faithful`, endpoints both level 0, body 1, confidence high, 60 min.
- The annotator's own framing of the risk was broader than the extraction's and they judged
  the extraction's node *closer to the as-written text* than their own. Worth reporting: the
  disagreement ran in the direction of the extraction being more conservative, not less.
- Maturity 1 accepted because the author asserts only enthusiasm ("I'd be really excited"),
  no mandate — an instance of the maturity label tracking the source's own hedging.
- Validation evidence is backward-looking (a survey's past failure) rather than evidence for
  the proposed intervention's design. The annotator judged that framing accurate here. See
  the reframed note below — this is not a defect signal.
- Recall: three arguments enumerated, all carried. One of them — forecasting training-run
  spend — the annotator raised independently and the extraction already had as pair 1. That
  collision is the enumerate-then-check discipline working.

**C01** -- `inferred_but_reasonable`, risk 0, intervention 1, body 0. Not a recall chain.
- Source is Hubinger's training-stories post, 84,526 characters -- 2.4x C03 and well above
  the packet's 35,618 median.
- Nodes 1-6 are supported by the author's own wording, and two are his coinages: "mechanistic
  training stories" and the section heading "Training story sensitivity analysis". Node 4's
  "four" is his word. Node 5 compresses two separately named criteria (training goal
  alignment, training goal competitiveness) into one label -- both stated, the joint phrase
  is not.
- The whole verdict rests on `pre-deployment` in node 7, which has **zero occurrences** in
  the source. The annotator scored it 1, not 2: a training story is a requirements document
  written before execution, so pre-deployment follows in a short step, and the document does
  define "training setup" as everything before the model is deployed. See the naming pattern
  above.
- The intervention node fuses training-story *implementation* with *sensitivity analysis*.
  The annotator judged the fusion fair, since a sensitivity analysis can only run on an
  existing training story. Worth counting how often composite intervention nodes appear --
  they interact directly with the sub-intervention rule from C03.
- **Two design-rationale nodes, no validation-evidence node**, and the document does contain
  VA-shaped material the extraction did not use: the author falsifying his own worked example
  through Ilyas et al. This is the second chain in a row where the validation-evidence slot
  behaved oddly (C03's was backward-looking at a survey's failure rather than evidence for
  the intervention). **Watch it.**

## 🔴 An absent stage node is usually the extractor being right, not missing something

Scribe's earlier framing was wrong and the annotator corrected it. This file flagged
"validation-evidence slot behaving oddly, two for two" after C03 and C01. That framing
treated absence as a defect signal. It is not.

The annotator's reading: **the schema prescribes stage slots that the source often cannot
fill**, so an absent node is frequently correct behaviour. C01 is the clean case -- the
document's only validation-shaped material *falsifies* its own worked example (Ilyas et al.
on cat-detection heuristics) rather than validating the proposed intervention, so the
extractor was right to emit no validation-evidence node.

This sharpens a claim the manuscript already makes rather than contradicting it.
`sec:r-stages` says a schema suggesting five stages will tend to produce five-stage chains,
and ablation arm E measured it: strip the stage names from the prompt and five-stage
completion goes from 65.0% to 0.0% on the same documents. The stage structure is
prompt-supplied. The annotator's point is the other half of that coin -- when the source
cannot support a stage, the honest output is a gap, and the 87.4% completion figure is
therefore an upper bound on how often the source actually supported all five.

**What to measure instead of "is a stage missing":** when a stage node is absent, does the
source contain material that would have filled it? That is answerable per chain and it is
the question that separates a correct gap from a real omission. Record it in `notes` as the
thirty are worked; do not add a field mid-run.
