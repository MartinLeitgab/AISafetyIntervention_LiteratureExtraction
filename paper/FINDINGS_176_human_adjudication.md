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

## RETRACTED: the risk-to-risk causal reading on C01

**Logged 2026-08-27, retracted 2026-08-28 by the annotator, one turn later.**

The original entry: the annotator remarked in passing that risk A (*inadequate evaluation of
advanced AI safety proposals*) is "to some extent the cause of" risk B (*unanticipated unsafe
behaviour of trained models*), and this file elevated that to a finding -- that C01's
saturated 2x6 pair list flattens a two-step causal structure, and that `tab:cuts` row 4
("exactly one risk, at the root -- no risk-to-risk hop") therefore cost the document its
argument shape.

**The annotator's revised reading, which supersedes it:** all six interventions map to both
risks reasonably, so no hierarchy needs claiming. Safety *proposals* and trained *models* are
not the same set of objects; the two risks stand on their own. The cross-product is
justified, not a flattening.

What survives and what does not. The enumerator constraint is real and documented -- a chain
cannot traverse risk-to-risk, whatever the released graph holds. **C01 does not demonstrate
that this costs anything**, and nothing else has yet. Leave the constraint where it belongs,
in `tab:cuts`, and do not attach a cost to it without a document where the annotator holds
that two extracted risks stand in a causal relation *on reflection*, not in passing.

🔴 **The lesson is about method, and it is worth more than the retracted finding.** An aside
the annotator volunteered while answering a different question was promoted to a logged
finding, then withdrawn on reflection one turn later. The structured fields survived the
same reflection unchanged. **Weight volunteered asides below filled fields**, give the
annotator a turn before logging one as a finding, and expect this to recur -- a careful
annotator revising a first impression is the process working, not a defect in it.

## Saturated pair lists make the recall floor easier to pass

C03's pair list was 3 pairs over 1 risk. C01's is 12 over 2, every intervention against every
risk. A saturated cross-product is maximally generous, so `chain_recall_missed = yes` is
mechanically harder to reach on C01 than on C03 -- not because the document is better
covered, but because the extraction cross-connected more densely.

If this varies across the thirty, the recall floor is partly a function of extraction
density rather than of what documents argue. Same root as the sub-intervention rule's stated
limit: **granularity and connection density are the extractor's choices, not properties of
the source.** Record the pair-list size per chain so the write-up can say whether the two
correlate.

**C01's own recall answer is `unclear`, not `no`** -- the annotator recognised all twelve
pairs in the document and noticed no thirteenth, but stated plainly that the post was not
read end to end. The rubric prefers `unclear` to a guessed `no` for exactly this case, and
the distinction matters for the write-up: a floor built from `no`s that are really `unclear`s
would read as evidence of coverage it does not have.

## 🔴 `chain_recall_missed` has a readable polarity trap

The annotator who filled C03's row asked, a day later, whether `no` meant recall had
*failed*. It means the opposite: `no` = no missed pair = no recall gap found; `yes` = a gap
found. The field name parses both ways -- "missed? no" and "chain recall: no".

That is a live risk to the study's headline, not a cosmetic one. A sign flip between the
sheet and the analysis inverts the result, and the person most likely to flip it is whoever
reads the CSV months later without the rubric open.

Three protections, none of which touch the instrument mid-run:
1. The analysis script asserts the polarity in its own code and prints it beside the rate --
   e.g. `assert row["chain_recall_missed"] in {"yes","no","unclear"}` plus an explicit
   `n_gaps_found = count(yes)`, never `count(no)`.
2. The write-up states the direction in words ("no gap was found in N of M chains") and never
   quotes the raw field name.
3. Any future packet renames the field to something unambiguous -- `missed_pair_found`
   would have avoided this entirely. Not renamed now: renaming a column mid-run is an
   instrument change, and the C03 and C01 rows are already written under the current name.

Running tally, stated in the unambiguous direction: **zero recall gaps found in two chains**
-- C03 exhaustively (`no`), C01 opportunistically and inconclusively (`unclear`).

## 🔴 The low-confidence flag did not fire on the chain we call spurious

Receipt: `experiment_review_failure_case_confidence.py` ->
`..._report.json`. Class B, no LLM call. Prompted by the annotator asking, mid-C02, whether
the Euclid chain really does carry confidence >= 3 on every link.

It does, and more. Stored confidences along `app:failure`'s chain are **3, 4, 4, 4, 4, 3**.
Four hops sit at 4, which `tab:rubric-conf` defines as **"a controlled experiment"** -- on an
expository mathematics article about prime numbers, which contains none.

The extraction prompt is explicit that this should not have happened: *"Where the source does
not support the required flow, use moderate inference and mark it in edge confidence (1 or
2)."* The flag exists and did not fire. That is a stronger statement than the manuscript
currently makes -- `app:failure` says only "edge confidence >= 3", and reviewer row R34 says
only that the gates fail to reject a clear negative.

**The nuance cuts the other way and belongs in any write-up of this.** The invented hop is the
*lowest*: risk->pa, the one asserting prime scarcity threatens cryptographic key space, sits
at exactly 3, while the 4s are on the mathematical steps, which are genuinely rigorous. The
extractor discriminated partially; it simply never went below the gate.

**Consequence for `tab:gates`.** The asymmetric first-hop >= 4 row would have excluded this
chain, and that is independent evidence -- `app:failure` predates the #175 sample on which
that threshold was chosen. 🔴 **One case. It does NOT license adding a benefit side to that
row.** The caption's warning stands, and #176 is still the arm that could license it.

**Consequence for `sec:m-extraction`.** That section says the >= 3 gate "admits links the
source evidences systematically and excludes those the model supplied." This chain's risk->pa
link was supplied by the model and was admitted at 3. The sentence is not true of the
artifact as stated, and this is a claim-level issue rather than a packet one -- route it to
`OPEN_ITEMS.md`, not to this file's disposition.

## 🔴 The 1/2 boundary is where the judgements actually live, and the scale has no room there

Twice now the annotator has asked for a value between 1 and 2 -- C03's recall candidates
("maybe between 1 and 2", recorded 2) and C02's safety mapping ("1.5", recorded 2). Both were
rounded UP, which is the conservative direction: rounding toward 1 would systematically
flatter the extraction.

This is a finding about the instrument, and it is mostly a *good* one. The boundary the
annotator strains against is 1-vs-2, and **neither the verdict nor the defect count depends
on it** -- both land in `inferred_but_reasonable`, and the only defect class is level 3, which
has not been close on either chain. The scale is coarse exactly where coarseness is cheap and
sharp exactly where it matters.

Two things to carry into the write-up. Report the 1/2 split as an annotator-judgement
distribution and never as a rate. And record the rounding rule (up, conservative) explicitly,
because a reader who assumes rounding-to-nearest would read the level-2 count as larger than
the annotator intended.

## 🔴 WATCH: the distribution of the safety mapping may be the study's real output

Raised after C02 and agreed as a watch item. Two chains in, the pattern to test is not a
defect count but a *distribution*.

C02 maps a non-safety source -- a DeepMind lecture on attention and memory -- onto a safety
frame, scoring the mapping at inference level 2, which the rubric explicitly licenses as "the
design working, not a failure". `app:failure`'s Euclid chain does the same move on an
expository mathematics article, and the paper calls that one spurious. The difference is
real: C02's underlying concept (attention makes a network's decisions inspectable) is
developed at length by the speaker, while Euclid's "prime scarcity" appears nowhere in its
source. But the *move* is the same, and the instrument separates them by one step on a
four-point scale.

**If several more chains land at level 2 on this same move, the defensible output of #176 is
the distribution of that mapping, not a count of defects.** A study that reports "0 of 30
unsupported" while 12 of 30 sit at level 2 on a lecture-to-safety mapping has reported the
less interesting half of what it found.

Two things to record per chain from here, both cheap and neither a new field:
1. Whether the source is a safety document at all, in `notes`. C02 is not; C01 and C03 are.
2. Whether the level-1-or-2 inference *is* the safety mapping, as opposed to some other
   inferential step. C01's was a timing qualifier (`pre-deployment`), not a safety mapping --
   a different thing that happens to score the same.

Without that split the write-up cannot tell a lecture-mapped-to-safety from a
safety-paper-with-a-loose-qualifier, and they are not the same claim about the corpus.

## Scale-rounding rules, applied twice each and stated once here

The annotator has twice given a value the scale does not carry, in opposite directions, and
both were resolved conservatively -- meaning toward the reading less flattering to the
extraction and less flattering to the study's own confidence:

| Field | Given | Recorded | Rule |
|---|---|---|---|
| inference levels | "between 1 and 2" (C03), "1.5" (C02) | **2** | round UP -- rounding toward 1 would understate how far the extraction reached |
| `annotator_confidence` | "medium to high" (C01), "low to medium" (C02) | **medium**, **low** | round DOWN -- overstating the annotator's certainty would overstate the study's |

🔴 **Both rules must appear in the write-up.** A reader assuming round-to-nearest would read
the level-2 count as larger than intended and the confidence distribution as lower than
intended, in the same document.

## RESOLVED: `no` vs `unclear` turns on the risk of being wrong, not on coverage

Flagged as an inconsistency after C02, resolved by the annotator the same day. It was never
an inconsistency -- the two answers came from a rule that had not yet been written down.

🔴 **The rule, and it governs the remaining 27: how much risk is there in being wrong?**

- **C02 -> `no`.** A non-safety source. There was little safety content to miss in the first
  place, so a `no` carries little risk of being wrong even unread in full.
- **C01 -> `unclear`.** A definitive safety piece. Smaller risks and interventions could
  plausibly be hidden anywhere in it, so without a full read `unclear` is the honest answer.

So the discriminator is **the source's density of things-that-could-be-missed**, not the
annotator's coverage. Coverage is the same in both cases; the exposure is not.

**No instrument change follows.** The annotator's position: the recall answer can be read
alongside the verdict and the source type to inform the result with context before anything
enters the paper. That is the right call at n=30 -- adding a confidence dimension to the
recall field mid-run would change the instrument for a distinction the write-up can make from
data already recorded.

What the write-up must therefore do: report `no` and `unclear` separately, and report each
against whether the source is a safety document. A pooled recall figure that mixes a `no`
from a lecture on attention with a `no` from a safety paper is not measuring one thing.

## Performance risks as upstream safety risks -- the annotator's scope argument

C02's five extracted risks are engineering problems, not safety risks: computational
inefficiency, RNN forgetting, limited relational reasoning, memory saturation, opaque
decision-making. The annotator judged them probably in scope anyway, on a specific mechanism:
**memory saturation could drop a system prompt including its safety guidance, and forgetting
long-term dependencies could be safety-relevant for the same reason.** Performance risk as
the upstream cause of a safety risk.

This is a real argument and it is also exactly the boundary this project has already had to
draw once -- see [[project_capability_ceiling_hcs]], where HC025-028 are held to be
capability-ceiling rather than safety risks and reported as a separate group, while HC029 is
safety-related. The same question, arriving from the annotation side.

Worth recording per chain rather than deciding globally: when a source is not a safety
document, does the chain's risk reach safety through a stated mechanism, an argued one, or
only a conceivable one? C02's is conceivable -- the lecture says nothing about system prompts
or safety guidance, and the annotator supplied that link themselves while judging.

