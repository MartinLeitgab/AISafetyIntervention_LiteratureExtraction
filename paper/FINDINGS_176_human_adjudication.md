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

The README budgets 20–30 min per chain plus ~20 for a recall chain. **C03 took 60 minutes** —
and C03 is a recall chain at the short end of the length distribution (14,627 characters
against a 35,618 median). If that rate holds, the thirty are **20–25 hours**, not the 13–20
the README states. Re-check after chain 5 before committing to all thirty.

## Per-chain observations

**C03** — `faithful`, endpoints both level 0, body 1, confidence high, 60 min.
- The annotator's own framing of the risk was broader than the extraction's and they judged
  the extraction's node *closer to the as-written text* than their own. Worth reporting: the
  disagreement ran in the direction of the extraction being more conservative, not less.
- Maturity 1 accepted because the author asserts only enthusiasm ("I'd be really excited"),
  no mandate — an instance of the maturity label tracking the source's own hedging.
- Validation evidence is backward-looking (a survey's past failure) rather than evidence for
  the proposed intervention's design. The annotator judged that framing accurate here, but
  the schema's intent was the latter. **Watch whether this recurs** — if it does, it is a
  finding about what `validation evidence` means in practice, not about this chain.
- Recall: three arguments enumerated, all carried. One of them — forecasting training-run
  spend — the annotator raised independently and the extraction already had as pair 1. That
  collision is the enumerate-then-check discipline working.
