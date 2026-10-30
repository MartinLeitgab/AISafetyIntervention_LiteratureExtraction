# #176 human adjudication — run record

Started 2026-08-27. One annotator, 30 chains. This file is the **run record**: predictions
registered before they can be checked, rules clarified mid-run, packet defects found while
working, and per-chain observations worth carrying into the write-up. It is not an open list
— that is `paper/OPEN_ITEMS.md` — and it is not annotator-facing, so nothing here should be
read while judging.

## 🔴 REGISTERED CHANGE 2026-10-29: the packet is cut again, from 20 chains to 10

**Reason: annotator time; about 4 hours is what remains for the paper.** Registered before
any of C07-C30 was read.

🔴 **Disclosure the write-up must carry.** Six chains were judged when this was decided --
C01-C06, 2 `faithful`, 4 `inferred_but_reasonable`, **0 `unsupported`**. Unlike the first
cut, this one comes after a run of favorable verdicts, so it looks like stopping on good
news. The guard: the four chains kept are exactly the arm-A strata #175's judge flagged
and the human has not yet seen, which are the chains most likely to yield an
`unsupported`. The dropped chains sit mostly in strata already covered.

> **JUDGE these 4, in this order:** C12 (`intermediate_unsupported`, 7.5k chars), C15
> (`risk_framing_invented_both_agree`, 26.7k), C24 (`model_disagreement_invented_but_link_asserted`,
> 5.8k), C27 (`known_judge_false_positive`, 215 chars -- see the packet-defect note below).
>
> **DROP these 10:** C07, C09, C13, C16, C18, C20, C23, C28, C29, C30.

Rule: keep every judged chain; keep the lowest packet id in each arm-A stratum still
uncovered; drop the rest. Ids are shuffled, so lowest-id is not a choice by content.

| | 20 | 10 |
|---|---|---|
| arm-A population coverage | 100.0% | **100.0%** |
| arm-A chains per stratum | 1-4 | 1-2 |
| arm B (gate-rejected) in scope | 4 | 2, both `gate_rejected_faithful` -- report as cases |
| **ceiling if `unsupported` is zero** (Wilson 95%) | 16.1% | **27.8%** |
| recall arm | 7 documents, a rate | **retired as a rate**: C03 and C06 reported as enumerated cases; C15 and C24 are verdict-only |
| hours remaining (50 min/chain measured over C01-C06) | ~12-14 | **~3-4** |

**What the study can still claim at 10:** a human judged a sample covering every stratum of
the 2,772-chain reporting unit, quoting a source span for each endpoint, and found N
`unsupported`. If N = 0, the sentence is "none of 10; 95% upper bound 27.8%". Plus the
defect-shape findings, which do not depend on n: verdicts turning on lifecycle-label
qualifiers in node names (measured corpus-wide), and C06's recalled-not-extracted node.
Not available at 10: a recall rate, an arm-B direction, anything per stratum.

Implemented in `experiment_review_human_adjudication.py` as `DROPPED_2026_10_29` and an
extended `RECALL_DROPPED`; the packet is not regenerated and the dropped rows stay blank.

## REGISTERED CHANGE 2026-08-30: the packet is cut from 30 chains to 20

**Reason: annotator time, and nothing else.** Receipt:
`experiment_review_packet_resize_assessment.py` ->
`..._report.json` (Class B, no LLM call), which reproduces the pre-registered weights,
strata and variance formula of `experiment_review_human_adjudication.py` and crashes rather
than reporting a design that loses population coverage.

🔴 **Why this is not optional stopping.** Three chains were judged when the decision was
taken and they are 1 `faithful` + 2 `inferred_but_reasonable`, **zero `unsupported`** -- no
defect has been observed, so nothing seen so far bears on the rate the study reports. All
three are retained; dropping a judged chain would be selection on an observed outcome. The
drop list is fixed below *before* C04 is read.

**What survives, and it is the thing that matters.** Arm-A population coverage stays at
**100.0%**. Coverage is what licenses post-stratification -- it depends on which strata hold
a chain, not on how many sit in each -- so the estimand is exactly as legitimate at 20 as at
30. Arm B stays at its usual 89.58% (it was never 100%). The heaviest stratum,
`faithful_both_agree` at 37.8% of population weight, only ever held 4 chains in the packet and
all 4 survive, so the dominant variance term is unchanged.

**What it costs, priced rather than asserted.**

| | 30 | 20 |
|---|---|---|
| arm A / arm B / recall | 24 / 6 / 10 | 16 / 4 / 8 |
| arm-A coverage | 100.0% | **100.0%** |
| 95% half-width, weighted share @ p=0.10 | 14.0 pp | 15.5 pp |
| @ p=0.20 | 18.7 pp | 20.6 pp |
| @ p=0.50 | 23.3 pp | 25.8 pp |
| **ceiling if `unsupported` is zero** | **11.4%** | **16.1%** |
| recall arm, ceiling if no document has a miss | 27.8% | 32.4% |
| hours (45 min verdict-only; 60-90 recall) | 25-30 | **17-21** |

🔴 **The zero-count ceiling is the real loss and the write-up must own it.** If P1 holds and
no chain is `unsupported`, the sharpest sentence available is "no unsupported chain in n",
whose 95% upper bound is set by n alone: 11.4% at 30, **16.1% at 20**. Everything else moves
by about two points, which is noise against intervals this wide.

**The drop list, by rule.** Retain every judged chain; keep at least one chain in every
stratum; within a stratum prefer recall-arm and double-coded chains; take the surplus off the
strata the build over-sampled relative to their population weight. That gives:

> **DROP these 10:** C08, C10, C11, C14, C17, C19, C21, C22, C25, C26.
>
> **JUDGE these 20:** C01, C02, C03 (done), then C04, C05, C06, C07, C09, C12, C13, C15,
> C16, C18, C20, C23, C24, C27, C28, C29, C30 -- **17 remaining**, of which 8 are recall-arm
> chains (C05, C06, C09, C15, C18, C24, C29, and C03 already done).

Do **not** regenerate the packet. The ids, the chains and the weights are untouched; the
dropped rows stay in `verdict_sheet.csv` unfilled, and the analysis script already reports an
unfilled row as outstanding rather than imputing it.

**Against the reviewers, 20 changes nothing that 30 answered.** R9 (no human validates
anything) and R12 (source spans) are answered by the design, not by n. R21 keeps its
direction from arm B, which was never allowed a pp figure at 6 either. R11 was already open
with one annotator and 7 of the 8 double-coded rows survive. R13 asked for 50-100 documents
and cut-list item 5 for 50 -- **30 already failed both**, so the defence is unchanged at 20:
a stratified sample with full population coverage and a post-stratified estimator beats a
simple random sample of the same size.

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

All were ambiguities in the shipped rubric, not changes of criterion. Each is in the
generator and patched into the shipped README in place; the packet was never regenerated,
which would have reshuffled a pre-registered sample.

| Date | Rule | Trigger |
|---|---|---|
| 2026-08-27 | **Endpoints govern the verdict.** Level 3 anywhere → `unsupported`; both endpoints 0 → `faithful` regardless of body inference; an endpoint at 1–2 → `inferred_but_reasonable`. | C03. The old wording defined `faithful` by endpoints and `inferred_but_reasonable` by the worst level anywhere, so both fired on level-0 endpoints with an inferred body — an ordinary shape, so the clash would have recurred on most of the thirty. |
| 2026-08-27 | **Sub-interventions are carried, not missed.** The recall unit is the (risk, intervention) pair *as the extraction names it*; an argument refining a listed intervention is carried. | C03 produced two, both refining "get better at predicting real-life capability". |
| 2026-09-01 | **Future work is not an argued chain.** The recall question asks for every risk-to-intervention argument the document *makes*. A stated limitation with a proposed direction ("we intend to...") is not one, so it is not an uncaptured pair. | C05's stated limitations -- partial observability, uniform action cost -- and its one-sentence plan-repair aside. Counting them would inflate the miss count with claims the paper never made. |

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
| 2026-09-03 | 🔴 **The analysis script encoded the SUPERSEDED verdict rule.** `validate()` required `worst == 0` for `faithful`, so it demoted any chain with level-0 endpoints and an inferred body -- exactly what the 2026-08-27 clarification says must not happen. Masked rather than harmless: the mismatch is reported only when `notes` is empty, and every row carries notes, so C03 (0/0/body 1, `faithful`) passed while being flagged internally as contradicting its own levels. Surfaced by C05, the second row of that shape. | Fixed in `experiment_review_human_adjudication.py`; the three rules now apply in the rubric's own order -- level 3 anywhere wins, then endpoints. **No verdict changes**: C03 and C05 were both judged by the human under the correct rule. |
| 2026-09-03 | **C05's recall-arm duty dropped for cost.** Its source is 86,655 characters, 5.9x the only document enumerated so far, and the annotator read only the framing sections in full. An enumeration from a partial read is not the exhaustive pass the recall *rate* claims. | `RECALL_DROPPED` in the analysis script, mirroring the 30-to-20 `DROPPED` pattern; the packet is untouched. Arm goes 8 -> 7 documents, and the no-miss ceiling 32.4% -> 35.4%. **Dropped, not swapped** -- every recall-arm chain is already in the arm, so nothing can be swapped in, and picking a replacement by length would add a second size bias on top of the arm's existing skew toward chain-rich documents. 🔴 The 1.19x size-bias figure in the script docstring is now stale (computed on the original 10-document arm) and must be re-derived before it is quoted. |

## Cost, measured

The README budgets 20-30 min per chain plus ~20 for a recall chain, so 13-20 hours overall.

| Chain | Arm | Chars | Minutes |
|---|---|---|---|
| C03 | verdict + recall enumeration | 14,627 | 60 |
| C01 | verdict only | 84,526 | 30 |
| C02 | verdict only | 101,344 | 60 |

🔴 **The projection has now moved twice and the third point put it back above the README.**
On C03 alone it read 20-25 hours, "not the 13-20 the README states"; C01's 30-minute
verdict-only chain pulled it to about 20 hours. C02 is also verdict-only and took 60 minutes,
so the two verdict-only points are 30 and 60 and their mean is 45. The arms are fixed at 20
verdict-only and 10 recall chains (the recall arm is C03, C05, C06, C08, C09, C14, C15, C18,
C24, C29 in `recall_enumeration.csv`), giving 20 x 45 min + 10 x 60 min = about **25 hours**,
above the README's 13-20 again. Three points is still a projection, not a rate; re-check at
chain 5, and expect it to keep moving while the spread within one arm is 2x.

Note what the three rows do **not** show: any relationship between document length and time.
C01 is 5.8x the characters of C03 and took half as long; C02 is 1.2x C01 and took twice as
long, and its source is ASR-garbled auto-captions rather than prose. Neither the reading nor
the length is what costs -- on the evidence so far it is the recall enumeration and the
condition of the source.

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

### MEASURED 2026-09-01: the qualifier is the prompt's LIFECYCLE label

The pattern above has a mechanism, and it is narrower and more fixable than "the prompt
pushes toward safety". Receipt: `experiment_review_lifecycle_name_absorption.py` (Class B).

C01's name says `pre-deployment` and its stored `intervention_lifecycle` is **4 =
"pre-deployment testing"**. C04's says `RL fine-tuning`; its lifecycle is **3 =
"fine-tuning/RL"**. Both qualifiers are the wording of the rubric value the extractor
assigned, taken from `app:prompt` step 2.

Measured over all **28,889** intervention nodes in the released graph carrying a lifecycle
with distinctive wording, against a control asking how often a name carries some *other*
lifecycle's marker:

| lifecycle | n | own label in name | other label in name | lift |
|---|---|---|---|---|
| 1 model design | 7,183 | 24.1% | 1.3% | +22.9 pp |
| 2 pre-training | 2,279 | 37.1% | 2.1% | +35.0 pp |
| 3 fine-tuning/RL | 7,198 | 48.6% | 1.8% | +46.8 pp |
| 4 pre-deployment testing | 5,486 | 35.4% | 7.8% | +27.5 pp |
| 5 deployment | 6,743 | 43.7% | 0.3% | +43.4 pp |
| **pooled** | **28,889** | **38.0%** | **2.5%** | **+35.5 pp** |

And on the packet: **9 of the 20 in-scope chains** carry their lifecycle label in the
intervention name. Of the **7** whose source is long enough to test (>=10k chars), **7 never
state that marker anywhere in the source** -- C01, C02, C04, C07, C15, C18, C28.

🔴 **What this does and does not license.** It is lexical, not semantic: a marker in a name
can be the document's own word, and a name can carry the lifecycle idea without the string.
It bounds how often the qualifier *could* have come from the rubric; it does not prove it
did. It is also not a defect rate -- a level-1 qualifier is the design working. What it does
say is that if the human verdicts keep turning on these qualifiers, the fidelity rate is
partly measuring a **naming convention the prompt supplies**, which is a fixable prompt
property rather than a limit of the approach. Two of the three judged chains have turned on
one so far. Keep counting.

## 🔴 PACKET DEFECT, found 2026-09-01: C27's source is 215 characters

`C27.md` -- stratum `known_judge_false_positive`, still unjudged -- carries a **215-character**
source: an Italian EA Forum post whose entire body is a Google Translate link to a Vox
article. The extraction nonetheless produced a full seven-node chain ending in *"Fine-tune
models with reinforcement learning from human feedback"*, content that cannot have come from
a bare link.

🔴 **This may mean the stratum is mislabeled.** That stratum exists because #175's judge said
the chain belongs to a *different document* and the project recorded the judge as **wrong**.
On this evidence the judge looks right. If so, a "known judge false positive" is a genuine
judge *hit*, which moves in the opposite direction from everything else this study is set up
to find.

**Do not act on it yet.** Judge C27 on its merits when it comes up in order, record the
verdict, and only then decide whether the stratum label is wrong. Its weight is 0.04, so it
moves the estimand very little either way -- the interest is in what it says about the
machine instrument, not in the rate. Also note it for the write-up as a corpus-hygiene
observation: a document with no body still produced a gate-passing chain.

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

**C04** -- `inferred_but_reasonable`, risk 0, intervention 1, body 0, medium, 45 min. Not a
recall chain. Pathak et al., self-supervised exploration via ensemble disagreement.
- Non-safety source, but unlike C02 the extraction did **not** map it to safety: the risk
  node is an ML-engineering risk in the paper's own terms. Whether such a chain belongs in
  the corpus is a scope question, not a fidelity one.
- The whole verdict rests on `RL fine-tuning` in the intervention name. `fine-tun*` has zero
  occurrences in the source, **and** it mis-describes the regime -- the paper trains on
  intrinsic plus sparse extrinsic reward jointly, or purely intrinsically with extrinsic
  reward used only for evaluation, and has no pretrain-then-finetune stage anywhere.
- Scored 1 on the argument/modifier split: not 0, which would launder an unsourced and
  inapplicable modifier; not 3, because the argument itself is asserted verbatim.

**C05** -- **`faithful`**, risk 0, intervention 0, body 1, medium, 60 min. Recall duty
dropped for cost. Classical planning, no ML model anywhere in the source.
- **The first `faithful` in arm A**, and the first row to exercise the endpoints-govern rule
  in anger: body 1 does not demote it. That is what surfaced the analysis-script bug above.
- The annotator scored the intervention 0 where C04's got 1, **deliberately**: `real-time`
  has zero occurrences but `runtime` has two, and the method is online by construction, so
  the modifier glosses what the paper does rather than contradicting it. C04's did contradict
  it. Record this as a drawn distinction, not an inconsistency -- it is the 0-vs-1 boundary
  the study exists to capture, and the annotator drew it on the substance both times.
- `MonitorPlanOptimality` is the paper's own function name. Node 6's "eight domains" is never
  written in the source but is correct -- the results tables carry exactly eight rows where
  section 5.1 describes six in prose. Arithmetic over stated data, not fabrication.
- **Second failure mode for the lifecycle rubric, and it is not lexical.** Stored lifecycle
  is 5 = "deployment" while the name says "execution runtime", so the absorption detector
  correctly did not fire. But the lifecycle scale is an ML training pipeline and this paper
  has no model, so "deployment" is the only bucket that fits and the value is a **forced
  choice, not an assessment**. This is exactly why the manuscript's "lexical rather than
  semantic" caveat on the absorption measurement is load-bearing.

## 🔴 C06 node 5: the extractor recalled, it did not extract

Verified 2026-09-05, and it is the sharpest single instance the study has produced.

C06's validation-evidence node reads "meta-analyses of deliberate practice effectiveness".
`meta-analys*` has **zero** occurrences in the source. The source carries two bare markdown
links whose anchor text is literally **"here"** and **"here"**, and whose URLs
(`frontiersin.org/articles/10.3389/fpsyg.2014.00646/full`,
`Macnamara-et-al.-2014.pdf`) contain no such string. Both linked works *are* in fact
meta-analyses -- their titles say so -- but **their titles are not in the document**.

The pipeline is one schema-prompted call per document with **no tool use and no retrieval
loop** (`sec:m-extraction`, a locked decision). It therefore could not have followed those
links. The only remaining route is the extractor's own parametric knowledge of those two
papers.

🔴 **This bears directly on a claim in `sec:outlook`**: "Extraction here is grounded rather
than recalled, so the same pipeline indexes sources that postdate any extractor's training."
Here the extraction is recalled, not grounded, and the recall happens to be correct. Do NOT
soften the outlook claim from one case -- but it is a counter-example to the property that
paragraph asserts, and unlike the Euclid chain it is a case where parametric recall produced
a *true* statement that the source does not support. Count how often node names resolve
citations the document only links; that is the measurable version.

## 🔴 An absent stage node is usually the extractor being right, not missing something

Scribe's earlier framing was wrong and the annotator corrected it. This file flagged
"validation-evidence slot behaving oddly, two for two" after C03 and C01. That framing
treated absence as a defect signal. It is not.

The annotator's reading: **the schema prescribes stage slots that the source often cannot
fill**, so an absent node is frequently correct behavior. C01 is the clean case -- the
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
behavior of trained models*), and this file elevated that to a finding -- that C01's
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

🔴 **CONFIRMED 2026-09-03 by a second, independent instance.** C05's extraction carries the
same shape -- a 2x2 cross-product where "failure to honour commitments" reads as a parent of
"unmet delegated goals", and where optimality monitoring is demonstrably a component of
abandonment detection (section 4.2 builds the latter from the former). The annotator raised
the hierarchy, considered it, and again concluded the risks stand alone as separate risks.
**Two documents, two rejections on reflection.** That is now positive evidence that the
cross-product is justified rather than a flattening, and the retraction above is not a
one-off. The finding to report is the pair of rejections; the no-risk-to-risk-hop constraint
still has no demonstrated cost after five chains.

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

**C01's own recall answer is `unclear`, not `no`** -- the annotator recognized all twelve
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

Running tally, stated in the unambiguous direction: **zero recall gaps found in three chains**
-- C03 exhaustively (`no`), C02 on a non-safety source where little could be missed (`no`),
C01 opportunistically and inconclusively (`unclear`). Only C03's `no` rests on a full
enumeration; C02's rests on the exposure rule below, which is why the write-up reports `no`
and `unclear` separately and against whether the source is a safety document.

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
link was supplied by the model and was admitted at 3, so the sentence overstates what the gate
does. Routed to `OPEN_ITEMS.md` as **C10**, a one-sentence wording fix.

🔴 **Keep this at its true size: one chain of 2,772, from a corpus of 11,779 documents.** It
is an existence proof that the wording is too strong and nothing else -- not a gate failure
rate, not evidence about prevalence, and not grounds to promote the first-hop >= 4 row. The
project already has a measured prevalence figure for invention-from-vocabulary, the
shuffled-source 30.0% in `sec:r-stages`, and that is the number any prevalence claim uses.

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

### 🔴 RESOLVED 2026-09-05: the non-safety sources are the design, and the sign is inverted

Raised from the top down after five chains: a majority of the sampled sources are not AI
safety research, which sits badly under a title reading "from the AI Safety Literature".
Two facts sharpened it. The packet is **7 of 20 arXiv (35%)**, matching the chain set's
35.2% share, so the sample is correctly proportional and the design is not at fault. And
the non-safety sources are **not** confined to the forum tail -- C04 (Pathak, exploration)
and C05 (plan monitoring) are ordinary arXiv papers. Filtering to "real papers" would not
have touched it. Tally so far: C01 safety, C02 not, C03 safety, C04 not, C05 not, C06 not.

**The resolution is that this is intentional and the corpus is built for it.** A mechanism
argued in another field -- densely attested against one class of risk, absent against
another -- is the shape of a transferable candidate, and that is what the graph-reasoning
extension draws on. Landed in the manuscript as `c76769c`: `sec:m-corpus` now states the
choice and refers forward to `sec:outlook`, which refers back.

**Three consequences for this study.**
- **Reviewer R36 (a safety-relevance pre-filter) is now a considered REJECTION**, not an
  open item. A filter at intake deletes the transfer candidates. The reviewer's underlying
  observation stands and the paper concedes it; only the remedy is refused.
- **The per-chain relevance note keeps being recorded, with its meaning inverted.** It is
  not a defect count. It measures how much of the corpus is transfer material rather than
  native safety work, which is the input to the graph-reasoning claim. C01 needs a backfill
  to complete the series; C02 onward already carry it.
- **C02's level-2 safety mapping is the design working**, exactly as the rubric says, and
  the watch item above stands unchanged -- the *distribution* of that mapping may still be
  the study's real output. What changes is that a high level-2 count is now a finding about
  corpus breadth rather than a hint of over-reach.

🔴 **Do not use this to relax a verdict.** Relevance and fidelity remain separate: a chain
may be a faithful reading of a document that is not safety research, and the rubric asks
only the first question. Nothing here licenses scoring a chain more leniently because its
source turned out to be transfer material.

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

