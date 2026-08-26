# Human adjudication packet -- issue #176

30 chains from the released corpus. Judge each against its source. **~20-30 minutes each**,
so budget 10-15 hours; the sheet has a `minutes_spent` column because knowing the real cost
is worth as much as the verdicts. The estimate went up from 15-25 when the recall question
was added on 2026-08-26 -- it is the one field that needs you to think about what the
document argues *beyond* the chain in front of you, and it is the reason to do this at all.

## What is in here

| File | What |
|---|---|
| `README.md` | this, including the rubric |
| `chains/C01.md` ... `C30.md` | one chain plus its full source text |
| `verdict_sheet.csv` | the sheet to fill in, one row per chain |
| `verdict_sheet_annotator2.csv` | 8 chains pre-selected for a second annotator **if one is ever found**. None is planned -- see below |
| `manifest.json` | which packet id maps to which chain -- for the analysis afterwards, not needed while judging |
| `REVEAL_stage1_verdicts.md` | 🔴 **do not open until the sheet is filled in** |

## One annotator, and what that costs

Decided 2026-08-26: one person judges all 30. So this study reports **no inter-annotator
agreement figure**, and the reviewer objection behind it -- that a human anchor without
agreement is one person's opinion, raised by two of the three external models at both bars
-- stays open. That is a known, accepted cost, not an oversight.

Two things follow while you work. Re-judging your own rows later is *not* a substitute: it
measures whether you are consistent, not whether the judgement is shared, and reporting it
as agreement would be a relabelled measurement. And if a second annotator does turn up,
they must not see your filled sheet -- keep it somewhere they will not open, because the
8 rows in `verdict_sheet_annotator2.csv` are only worth anything blind.

## The order is shuffled and the ids are opaque, deliberately

The 30 were selected by strata (a machine flagged some and not others) but they are shuffled
and numbered `C01`-`C30`, so nothing in the reading order tells you which is which. If you
can infer a stratum you have lost the property the packet is built to protect.

## Document lengths

Median 35,618 characters, longest 101,342. The full
text ships for every chain because deciding a document does NOT assert something requires
being able to search all of it. For the long ones the URL in each file is often easier to
read than the plain-text dump.

## Composition

Deliberately not stated here. The 30 are not a random sample and knowing the mix would tell
you roughly how many to expect in each verdict, which is an anchor as strong as seeing the
verdicts themselves. It is recorded in `manifest.json` and in issue #176, both of which are
for the analysis afterwards.

---

# How to judge a chain

You are deciding one thing: **does this document make this argument?** You are NOT judging
whether the argument is correct, whether the intervention would work, or whether the document
is good research. A faithful record of a weak argument is faithful.

Fill in the fields below per chain. Quote spans verbatim from the source; if you cannot find one,
that is itself the answer and the field is left empty.

| Field | Question | Values |
|---|---|---|
| `risk_link_asserted` | **Does the document assert the link from the risk to the next node at all?** A plain yes or no, before any judgement of degree. | `yes` / `no` |
| `risk_inference_level` | How far from the document is the risk at the head of the chain? | `0`-`3`, below |
| `risk_quote` | The span that states or best supports it | verbatim, or empty |
| `intervention_inference_level` | How far from the document is the intervention at the tail? Describing or citing a technique is not proposing it. | `0`-`3`, below |
| `intervention_quote` | The span in which it is proposed | verbatim, or empty |
| `body_inference_level` | The intermediate nodes together, scored at their worst step | `0`-`3`, below |
| `verdict` | Overall | see below |
| `chain_recall_missed` | **Does this document argue a materially different risk-to-intervention chain that is NOT in the extraction?** See below -- this is the only recall question anyone asks in this project | `yes` / `no` / `unclear` |
| `chain_recall_note` | If yes: name the risk and the intervention, in a few words each | free text, or empty |
| `annotator_confidence` | How sure are you of the `verdict` on this chain? | `high` / `medium` / `low` |
| `notes` | Anything the fields above cannot carry | free text |
| `minutes_spent` | How long this one took | a number |

## The inference scale, and why it is not a five-point quality rating

The extraction prompt **deliberately licenses inference** where the document does not supply a
stage, and it ties each degree to a label the extractor was then required to store:

| Level | Meaning | What the extraction prompt says |
|---|---|---|
| **0** | stated in the document | no inference applied |
| **1** | light inference: a short step a domain reader takes without effort | "must be 2 if light inference applied" |
| **2** | moderate inference: a real reading, but one this literature would accept | "must be 1 if moderate inference applied" |
| **3** | beyond moderate: not reasonably inferrable from this document | **licensed nowhere. This is the defect** |

A 1 or a 2 is **not a complaint**. It is the design working, and the extractor was supposed to
record it by storing a low confidence on that edge. **Level 3 is the only value that says the
extraction asserted something it had no licence to assert.**

Four values anchored to the prompt's own words, rather than five points of "how much",
because an earlier run of this project asked a model to place these same links on a
five-point evidence scale: its ordering was informative and its absolute level was useless,
and it put 61% of the links a different instrument called faithful below the project's own
threshold. An anchored level is checkable against a rule; a five-point feel is not.

## The verdict values, and the one that matters most

The verdict follows the levels; where it does not, say why in `notes`.

- **`faithful`** -- level 0 on the risk and the intervention. The document makes this argument.
- **`inferred_but_reasonable`** -- the worst level is 1 or 2. The document does not state part
  of the chain, but the reading is one a domain reader would accept. **The design working, not
  a failure**, and the judgement no model can make for us.
- **`unsupported`** -- level 3 anywhere. The extraction asserted something the document does
  not support and could not license.

The 2-versus-3 boundary is what the whole exercise turns on, and it is exactly what an
automated judge cannot draw, because it requires knowing what a reader of this literature
would accept.

## Why the first field is a plain yes/no

Because the graded version of this question turned out to be the weak instrument. A second
model applying the project's own five-point evidence rubric to these same links separated the
good from the bad by 0.7 of a point, and marked 61% of the *good* ones below the project's own
threshold -- its ordering was informative and its level was not. The same call answered the
binary "does the document assert this link" and separated the two groups by a factor of three.
So the binary is asked first, on its own, before any graded field can anchor it.

It is also the one field that is worded identically to what the machine was asked, which is
what makes it possible to compute how often the machine was right rather than merely how often
it was confident.

## The recall question, and why it is worth the extra minutes

Everything above judges a chain the extraction *produced*. That is precision. It cannot see
what the extraction *missed*, and nothing else in this project can either -- so while the
source is open in front of you, you are the only instrument that will ever answer it.

Ask it at the chain level, never at the node level. **A concept the extraction did not name
is not a miss.** A second pass over any document will always find more nameable concepts,
and a denser middle does not change which risk is connected to which intervention. What
counts as a miss is a materially different *argument*:

- a **different risk** the document argues against, with its own intervention;
- the **same risk** routed to a **different intervention**;
- the **same intervention** offered against a **different risk**.

If the document argues one of those and the extraction does not carry it, that is `yes`,
and `chain_recall_note` gets the risk and the intervention in a few words each. If the only
thing you can point at is a stage the chain states thinly or skips, that is `no` -- the
inference levels above already record it. `unclear` is a real answer for a long document
you could not search exhaustively, and it is far better than a guessed `no`.

Two warnings. This field will read low by construction, because you are looking at one
chain and its source rather than at every chain the document produced -- so it is a **floor
on recall failure, never a rate**. And it is the one field where being unsure is common:
use `unclear` freely.

## Two things to resist

1. **Do not repair the chain.** If a node is nearly right, it is not right. Judge what is
   written, not the best version of it.
2. **Do not calibrate to the other chains.** Each is judged against its own source only.
   Some of these were selected because a machine flagged them and some because it did not;
   you are not being asked to reproduce or to contradict any earlier verdict, and you will
   not see one until you are done. There is no target rate. Two machine annotators on this
   exact task differed enormously in overall strictness while agreeing on the ranking, so a
   verdict distribution that feels too harsh or too lenient is not evidence you are doing it
   wrong.

