# The chain-unit session, 2026-08-26

One day's work, four new studies, USD 13.45. It started from one observation — **every
quality number in the paper counted a unit the paper is not about** — and everything below
follows from chasing that.

This file is the durable record. `paper/OPEN_ITEMS.md` remains the canonical *open* list;
this is the *closed* one, written because several of these findings are the kind a future
session would otherwise re-derive wrongly or, worse, re-derive in the same broken way.

---

## 1. The doctrine

**The unit of this paper is the chain.** A concept added mid-argument does not change which
risk reaches which intervention. So a node count is not a quality figure for the artifact,
and printing one invites a reviewer to price the contribution on a measurement of something
else.

Node counts stay where the claim is about nodes: the artifact inventory, the
stage-vocabulary evidence (98.8% probe, kappa 0.84 — those *are* claims about node
attributes), the merge/centrality/clustering use cases whose entire point is that node-level
analysis distorts, and the edge-orientation finding. **Do not over-apply the rule and start
deleting correct numbers.**

## 2. What the studies found

| Study | Issue / PR | Result |
|---|---|---|
| Are the flagged omissions inert? | #180 / #183 | **No.** 3.7% inert, 4.8% re-routing, **91.5% name a node the extraction lacks** |
| Do they change the chain set? | #180 / #183 | Granting all 270 + 307 invented nodes: **1 new pair of 86 available (1.2% of headroom)** |
| Does the 70% collapse drop arguments? | #182 / #185 | **Yes.** 50.8% of 118 drops judged a different argument; 40 of those 60 change an endpoint. Null arm **15/15** |
| Chain recall, released graph | #181 / #184 | **69.9% carried, 10.6% materially uncaptured**, sensitivity **9/20 = 0.45** |
| Chain recall, gated reporting unit | #181 / #184 | 32.9% carried — **but 79% of the gap is gate selectivity, not loss** |

## 3. Five traps, each of which nearly produced a wrong number

### 3.1 Headroom, not the total, is the denominator
86 of the 100 audited papers extract as a **single connected component**, so no added edge
*can* create a risk-to-intervention pair there. The chain-impact result against all 408
reachable pairs reads 0.2%; against the 86 that were structurally available it reads 1.2%.
Quoting the first would report a ceiling effect as a finding. `experiment_paper_claim_audit.py`
fails if a future edit does this.

### 3.2 The gated recall rate is SELECTION, not loss
For the same 95 documents the released graph supports **6.15** risk-to-intervention pairs
each; the reporting unit keeps **1.30**. **471 of 597 graph-present pairs (79%) are absent
from the reporting unit because a chain scored below a gate**, not because the document was
poorly extracted. An argument the pipeline captured and the gates declined is not a recall
failure of the pipeline. Reporting 36.5% as a miss rate would invent a failure mode that
does not exist.

### 3.3 The naive sensitivity correction breaks at high rates
`observed / sensitivity` is valid on the gate-free 10.6% (giving 23.6%). On the gated 36.5%
it gives **91%**, which exceeds any sensible bound and double-counts: the enumeration
supplying the denominator and the detection supplying the numerator are the **same step of
the same judge**.

### 3.4 The population must change with the instrument
Only **1,868 of 11,779** documents yield a gated chain, and just **12 of the 99** audited
ones do. Running the gated recall on the audited population would leave 87 of 99 pair lists
empty, every argument would read as uncaptured, and the answer would be trivially 100% — a
measurement of the sampling frame. S2 hit the same wall for the same reason.

### 3.5 Different denominators are not competing estimates
The collapse produces three figures and they answer three questions:
- **18.0%** = 579 of **3,222 distinct raw pairs**. Corpus-level. **The headline.**
- **33.9%** = 40 of **118 sampled drops** (of 6,182). A property of drops.
- **50.8%** = 60 of the same 118, adding same-endpoint re-routings.

The adjudication's contribution is **not a rival rate**. It kills the benign reading that
the collapse removes repeats, which makes 18.0% a **floor** rather than an artifact of
counting traversals.

## 4. String similarity over generated names is not an instrument

**Third time this project has hit this wall**, and the first two are already in the paper:

1. The **46.5%** node re-identification figure — cosine over name embeddings, now labelled
   as a measurement of *names* rather than of pipeline stability.
2. **#179's** five-point rubric — ordering informative, absolute level useless.
3. **This session's ablation arm.** Detection scored by token overlap is a pure function of
   the threshold: **0/20 at jaccard 0.40, 8/20 at 0.15, 10/20 at 0.10.** *"Design AI models
   with cryptographic shutdown backdoors activated by secret trigger"* against *"Insert
   cryptographic backdoors (off-switches) that are hard to detect"* scores **0.12** and is
   obviously the same intervention.

**The fix that works:** ask a model directly — one name against a short list, no source
text, no enumeration. A far easier task than the one it referees, which is the only reason
it may referee. That settled the sensitivity at 9/20.

## 5. The ablation arm took three attempts and two failures were ours

1. **Deleted a `(risk, intervention)` CELL.** Void. The pair list is a reachability
   **cross-product** — 2 risks and 6 interventions give 12 pairs — so removing one cell
   rarely removes an endpoint. Of 20 deletions: **8 left BOTH endpoints visible**, 11 left
   the risk visible, **exactly 1** removed an endpoint. Returned 0/20 and meant nothing.
2. **Deleted a whole INTERVENTION** (20/20 genuinely absent). Still 0/20 — see §4.
3. **Semantic adjudication.** 9/20 = 0.45.

The control caught a bug in the control, twice, which is the cheap version of finding out.
**Any future ablation on a cross-product list must delete an endpoint.**

## 6. The reviewer comment that fell through

The reviewer register was built **2026-08-15** and only the **cut list** was re-swept after
the 2026-08-17 round — so that round's **evidence** comments were never entered. Among them
the sharpest methodological criticism the paper received, from the reviewer who scored it
lowest, with an explicit score impact attached:

> *"The current 0.6%, 18.1%, 28.8%, 26.4%, and 21.7% quantities divide proposed missing
> items by the size of an existing graph, even though the numerator and denominator are
> produced by different instruments… distinguish additions per extracted item from recall."*

Four of its five instructions turned out to be **already satisfied** — worth recording as
loudly as the misses. One was opaque wording (fixed, `b11f7ca`). One needed a study nobody
had scoped (#181). Now registered as R23a–R23f.

**Process lesson: when a new review round lands, re-sweep the register for evidence
comments, not only for cuts.**

One row is a reviewer arithmetic error rather than a paper error: 328+146+302=776 misses the
fourth status (one row covered only in the abstract). **777 is right** and the audit checks
the sum.

## 7. Manuscript changes

`9583f77` abstract + Limitations + `sec:r-judge` · `a52e32c` Conclusion, the 46.5% naming
figure, three more · `bfd8789` containment reordered so argument-level leads · `b11f7ca`
numerator/denominator/instrument named · `cfa5c2f` three headings that still said *omission*
· `d3890db` the collapse adjudication and both recall arms.

Claim audit **408/408**.

## 8. What this changed about the open work

- **Same-pair re-run: dropped.** The reporting unit keeps 1.30 of ~6.15 candidate pairs, so
  which chain survives turns on a narrow scoring margin. A re-run flipping chain membership
  is expected gate behaviour, already reported as 9 of 18. The question it proxied for is
  answered better by the recall study.
- **Retrieval evaluation: still skipped.** Lower value than before, since the substrate is
  now known to be selective by design.
- **#176 human study: amended, not rescoped.** Its value went *up* — machine recall
  sensitivity is 0.45, so the 10-document human enumeration is the only unbiased recall
  instrument in the project. The rubric now sets the expectation that several uncaptured
  arguments per document is *normal*, so the volume does not read as alarming.
- **Page budget.** The three node-level use-case subsections total **1.50 pp** but four of
  six reviewers said keep the merge/centrality artifact; compressing to finding-plus-control
  yields ~0.9 pp. **Limitations at 1.70 pp against a 0.8 pp target is the larger win** and
  costs no reviewer-valued content.
