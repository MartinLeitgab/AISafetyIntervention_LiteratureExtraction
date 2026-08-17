# Human adjudication packet -- issue #176

30 chains from the released corpus. Judge each against its source. **~15-25 minutes each**,
so budget 8-12 hours; the sheet has a `minutes_spent` column because knowing the real cost
is worth as much as the verdicts.

## What is in here

| File | What |
|---|---|
| `README.md` | this, including the rubric |
| `chains/C01.md` ... `C30.md` | one chain plus its full source text |
| `verdict_sheet.csv` | the sheet to fill in, one row per chain |
| `verdict_sheet_annotator2.csv` | 8 chains for a second annotator, for the inter-annotator figure |
| `manifest.json` | which packet id maps to which chain -- for the analysis afterwards, not needed while judging |
| `REVEAL_stage1_verdicts.md` | 🔴 **do not open until the sheet is filled in** |

## The order is shuffled and the ids are opaque, deliberately

The 30 were selected by strata (a machine flagged some and not others) but they are shuffled
and numbered `C01`-`C30`, so nothing in the reading order tells you which is which. If you
can infer a stratum you have lost the property the packet is built to protect.

## Document lengths

Median 35,039 characters, longest 150,520. The full
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

Fill in five fields per chain. Quote spans verbatim from the source; if you cannot find one,
that is itself the answer and the field is left empty.

| Field | Question | Values |
|---|---|---|
| `risk_supported` | Does the source assert this risk, or something a domain reader would accept as it? | `yes` / `partial` / `no` |
| `risk_quote` | The span that asserts it | verbatim, or empty |
| `intervention_supported` | Does the source **propose** this intervention against that risk? Merely describing or citing the technique is **not** proposing it. | `yes` / `partial` / `no` |
| `intervention_quote` | The span in which it is proposed | verbatim, or empty |
| `body_supported` | Is each intermediate node's content present in the source? | `yes` / `partial` / `no` |
| `verdict` | Overall | see below |
| `notes` | Anything the fields above cannot carry | free text |

## The verdict values, and the one that matters most

- **`faithful`** -- the document makes this argument. Quotes exist for the risk and the
  intervention.
- **`inferred_but_reasonable`** -- the document does not state part of the chain, but the
  extraction's reading is one a domain reader would accept as a fair inference from what the
  document does say. **This category exists because the extraction prompt deliberately
  licenses moderate inference**, and it is the judgement no model can make for us. Use it
  freely; it is not a failure verdict.
- **`unsupported`** -- the document does not support this, and a domain reader would not get
  here from it. This is the verdict that means the extraction asserted something about the
  document that is not there.

`inferred_but_reasonable` versus `unsupported` is the distinction the whole exercise turns
on. An automated judge cannot draw it, because it requires knowing what a reader of this
literature would accept.

## Two things to resist

1. **Do not repair the chain.** If a node is nearly right, it is not right. Judge what is
   written, not the best version of it.
2. **Do not calibrate to the other chains.** Each is judged against its own source only.
   Some of these were selected because a machine flagged them and some because it did not;
   you are not being asked to reproduce or to contradict any earlier verdict, and you will
   not see one until you are done.

