#!/usr/bin/env python3
"""Build the annotation packet for #176, with stage 1's verdicts withheld.

30 chains, stratified by the reason codes stage 1 (#175) produced, so the annotator's hours
land where a human changes the answer rather than where a random sample would put them. The
strata and their sizes are argued in #176; this script only implements them.

Two properties matter more than anything else here:

  BLIND     No file the annotator opens while working carries stage 1's verdict, its reason
            code, or which stratum a chain came from. Chains are shuffled and given opaque
            ids. The verdicts live in one clearly-named reveal file to be opened only after
            the verdict sheet is filled in, so that "did you agree with the judge" is a
            question the annotator answers afterwards rather than a prior they work under.
  FULL TEXT The whole source document ships, never an excerpt. Deciding whether a document
            asserts a risk means being able to find that it does not, which an excerpt
            cannot support. The source URL ships too, for anyone who would rather read the
            original rendering.

Class B: no LLM call, no network.

    cd graph_analysis
    python -u experiment_build_human_review_packet.py
"""

from __future__ import annotations

import csv
import glob
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "phase2_results" / "chain_precision_raw"
NODE_ATTRS = HERE / "phase2_results" / "node_attrs_slim.pkl"
GRAPH_EDGES = (
    HERE
    / "phase2_results"
    / "step1_load_and_parse_umapwithoutlocalsatellites"
    / "graph_edge_data.pkl"
)
ARD_DIR = ROOT / "intervention_graph_creation" / "data" / "raw" / "ard_json_full"
OUT = HERE / "phase2_results" / "human_review_packet"

SEED = 42

# REVISED 2026-08-17 after #178/#179 re-labelled 66 of the same chains with a second, blind
# instrument. Three things in that result change where an hour of human attention is worth
# most, and the strata below are the response:
#
#   1. Two independent instruments now agree on the invented-risk class -- #175 called them
#      invented, #179 says the document does not assert the link in 25 of 33. Agreement is
#      not correctness (same model family), but the marginal value of a human label on a
#      case both machines already call the same way is lower than it was.
#   2. Nothing corroborates intervention_not_proposed, and nothing can: it turns on whether
#      the schema's imperative rendering plus a development-stage maturity label amount to
#      the source "proposing" the intervention. That is a question about what the schema
#      means, and it governs 24% of the reporting unit. It goes UP.
#   3. The two instruments DISAGREE on 15 chains -- 7 the precision judge called faithful
#      where the re-labeller says the link is not asserted, 8 the reverse. These are the
#      highest-information items in the study and did not exist as a stratum before, because
#      the second instrument did not exist. They go in.
#
# Total stays 30: the constraint is the annotator's hours, not the sample frame.
#
# REVISED AGAIN 2026-08-26, and this revision is about what the study can RETURN rather
# than about where attention is worth most. The 2026-08-17 frame was built to adjudicate
# the classes #175 flagged. Reading it against what the reviewers actually ask for
# exposed two holes, both of which freeze the moment the first verdict is written:
#
#   A. NO POPULATION RATE WAS REACHABLE. #175's real arm is 200 chains stratified by URL
#      host PROPORTIONAL to the 2,772-chain reporting unit, so its reason-code shares are
#      population estimates (faithful 48.0%, intervention_not_proposed 24.0%,
#      risk_framing_invented 16.5%, intermediate_unsupported 7.5%, wrong-document 4.0%).
#      That makes post-stratification the whole game: human verdicts per code, reweighted
#      by those shares, give a human-anchored rate for the reporting unit rather than a
#      rate for a hand-picked 30. But intermediate_unsupported had ZERO cells, so 7.5% of
#      the population was blind and no reweighted rate could be quoted for it. It goes in.
#   B. THE GATE ARM WAS n=1. The 17.8 pp gate discrimination (52.0% arm A against 69.8%
#      arm B) is the finding OPEN_ITEMS calls the half most likely to survive #176 and the
#      half a reviewer would find most useful, and one human observation cannot license
#      it. Six can make it directional. The six are drawn across arm B's own top three
#      reason codes (intervention_not_proposed 43.8%, faithful 30.2%, invented 15.6%),
#      so arm B reweights the same way arm A does and the two rates are comparable.
#      🔴 SIX IS A DIRECTION, NOT A CONFIRMATION. Do not print 17.8 pp as human-anchored.
#
# The two slots that paid for this, and why these two:
#   - risk_framing_invented_both_agree 6 -> 3. This is the class the 2026-08-17 note
#     itself argued DOWN: two instruments already call it the same way, so a human label
#     on it is worth less than it was. Taking the note's own logic one step further.
#   - known_judge_false_positive 2 -> 1. 4.0% of the population and both machines already
#     know these are judge false positives; the human label confirms a known answer.
#
# Real-arm cells now cover 100.0% of the reporting unit's reason codes (was 92.5%).
STRATA = [
    # (label, n, predicate over the joined record)
    # --- arm A: the 2,772-chain reporting unit. Weights come from #175's real arm. ---
    (
        "intervention_not_proposed",
        7,
        lambda r: r["arm"] == "real" and r["code"] == "intervention_not_proposed",
    ),
    (
        "faithful_both_agree",
        4,
        lambda r: r["arm"] == "real"
        and r["code"] == "faithful"
        and r["asserted"] is True,
    ),
    (
        "model_disagreement_faithful_but_link_not_asserted",
        3,
        lambda r: r["disagree"] and r["code"] == "faithful",
    ),
    (
        "model_disagreement_invented_but_link_asserted",
        3,
        lambda r: r["disagree"] and r["code"] == "risk_framing_invented",
    ),
    (
        "risk_framing_invented_both_agree",
        3,
        lambda r: r["arm"] == "real"
        and r["code"] == "risk_framing_invented"
        and r["asserted"] is False,
    ),
    (
        "intermediate_unsupported",
        3,
        lambda r: r["arm"] == "real" and r["code"] == "intermediate_unsupported",
    ),
    (
        "known_judge_false_positive",
        1,
        lambda r: r["arm"] == "real"
        and r["code"] == "chain_belongs_to_a_different_document",
    ),
    # --- arm B: chains the same enumerator emits that the two quality gates reject. ---
    # Not part of the reporting unit. These exist only so gate discrimination has a human
    # observation on both sides. Drawn across arm B's top three codes, which cover 89.6%
    # of it; the two tail codes are left uncovered rather than filled with cells of one.
    (
        "gate_rejected_intervention_not_proposed",
        3,
        lambda r: r["arm"] == "gate_rejected"
        and r["code"] == "intervention_not_proposed",
    ),
    (
        "gate_rejected_faithful",
        2,
        lambda r: r["arm"] == "gate_rejected" and r["code"] == "faithful",
    ),
    (
        "gate_rejected_invented",
        1,
        lambda r: r["arm"] == "gate_rejected" and r["code"] == "risk_framing_invented",
    ),
]

# Double-coded for an inter-annotator figure. #179 showed the strictness spread on this exact
# judgement is enormous -- a second annotator put 60.6% of links the first called faithful
# below the gate -- so an agreement figure computed on easy cases would flatter the protocol.
# The 8 are therefore CONCENTRATED on the two hardest strata rather than spread at random,
# which makes the resulting figure a lower bound on agreement and says so in the README.
#
# 🔴 DECIDED 2026-08-26: THERE IS NO SECOND ANNOTATOR. One person judges all 30. The sheet
# below is still emitted, pre-selected and ready, because recruiting one later costs 2-3
# hours and nothing else -- but nothing in the study currently consumes it. The consequence
# is not cosmetic and must be carried into the write-up: reviewer R11 (Opus 5 and GPT-5.6
# Sol, both bars) says a human anchor without inter-annotator agreement is one person's
# opinion, and that objection stays OPEN. Do not report an agreement figure, do not imply
# one exists, and do not substitute the annotator re-judging their own rows -- that is
# test-retest consistency, a different and weaker instrument, and calling it agreement
# would be the kind of relabelled measurement this project has already been caught doing.
N_DOUBLE_CODED = 8

# The recall arm, added 2026-08-26. Everything else in this packet measures PRECISION -- it
# judges chains the extraction produced. Recall needs the opposite move: read the document
# exhaustively, enumerate every risk-to-intervention argument it makes, and check each
# against what the extraction holds. `chain_recall_missed` alone is only a floor, because
# noticing a missing argument while judging a chain is not the same as looking for all of
# them.
#
# Ten of the thirty, drawn with the same seed and fixed BEFORE any verdict is written, so
# the subset cannot be chosen after seeing which documents look bad. Drawn at random rather
# than by length: picking short documents would bias recall downward, since a short document
# has fewer arguments to miss.
#
# 🔴 The resulting rate is document-level, and the document sample is size-biased by 1.19x
# -- these documents were reached through a CHAIN-proportional sample, so a document with
# more chains had proportionally more chances to be drawn (packet mean 1.76 chains per
# document against 1.48 across the 2,772-chain reporting unit; median 1.0 in both). Small,
# but it must be stated with the rate, and it cannot be corrected by the precision weights:
# those are reason-code weights over chains and have nothing to do with recall.
N_RECALL_ARM = 10
DOUBLE_CODE_STRATA = {
    "model_disagreement_faithful_but_link_not_asserted",
    "model_disagreement_invented_but_link_asserted",
    "intervention_not_proposed",
}

RUBRIC = """\
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
| `chain_recall_missed` | **Does this document argue a risk-to-intervention pair that appears NOWHERE in the list at the top of the file?** Additive only -- see below | `yes` / `no` / `unclear` |
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

## The recall question, and the list it is asked against

🔴 **This question is ADDITIVE, not corrective.** It does not ask whether the chain in front
of you is wrong -- `verdict` and the inference levels already record that, and a chain can
be `unsupported` while this field is `no`. It asks a separate thing: **is there an argument
in this document that the extraction did not capture at all?**

Everything above judges a chain the extraction *produced*. That is precision. It cannot see
what the extraction *missed*, and nothing else in this project can either -- so while the
source is open in front of you, you are the only instrument that will ever answer it.

**Ask it against the list at the top of the file, never against the chain alone.** Each
chain file opens with *every* risk-to-intervention pair this document's extraction holds --
typically six, sometimes seventeen, never just the one you are judging. That list is what
"the extraction" means for this field. Without it the question would be unanswerable, since
a pair absent from your chain is usually present in the document's other chains.

Ask it at the chain level, never at the node level. **A concept the extraction did not name
is not a miss.** A second pass over any document will always find more nameable concepts,
and a denser middle does not change which risk is connected to which intervention. What
counts as a miss is a risk-to-intervention pair that appears **nowhere in that list**:

- a **different risk** the document argues against, with its own intervention;
- the **same risk** routed to a **different intervention** than any listed;
- the **same intervention** offered against a **different risk** than any listed.

If the document argues one of those and no listed pair covers it, that is `yes`, and
`chain_recall_note` gets the risk and the intervention in a few words each. If the only
thing you can point at is a stage the chain states thinly or skips, that is `no` -- the
inference levels above already record it. `unclear` is a real answer for a long document
you could not search exhaustively, and it is far better than a guessed `no`.

Two warnings. The list is built by reachability over the document's extracted graph, so it
is **generous**: it includes pairs that the quality gates later reject, which is deliberate,
because a pair the extraction found and then gated out is not a recall failure. And this
field will read low by construction -- you are reading one document's argument closely
rather than auditing it exhaustively -- so it is a **floor on recall failure, never a
rate**.

## What a normal document looks like, so the volume does not alarm you

Expect to find **several arguments per document that no listed pair carries**, and expect
that most of them are `thin` rather than material. A machine run over 95 documents of this
corpus enumerated about **6.5** risk-to-intervention arguments per document, and roughly a
third were carried, a third thin, a third material. Finding four or five uncaptured things
in a long paper is the normal case, not a sign the extraction failed.

Two reasons it looks that way, and neither is a defect:

- **Documents argue more than they conclude.** A paper's discussion section will gesture at
  half a dozen risk-intervention links it never develops. Those are `thin` -- fewer than two
  supporting reasoning steps -- and the materiality bar exists precisely to keep them out of
  the number.
- **The extraction is selective on purpose.** Over the same documents the released graph
  supports about **6.15** risk-to-intervention pairs each while the gated reporting unit
  keeps **1.30**. You are judging against the generous list, so that second reduction is not
  yours to worry about -- but it is why the extraction is not trying to be exhaustive.

So the question is never "is there anything else in here", because there always is. It is
**"does the document develop a materially different argument that the list does not carry"**
-- new endpoint, two or more supporting steps. If you find none in a document, `no` is a
perfectly ordinary answer and you should not go hunting for one.

## The recall subset: {N_RECALL_ARM} of the {30} get one extra pass

`chain_recall_missed` above is a **floor**: it records misses you happened to notice while
judging a chain, which is not the same as having looked for all of them. A recall *rate*
needs the opposite move, and {N_RECALL_ARM} of the thirty chain files ask for it explicitly.
Which ten was fixed before any verdict existed, and drawn at random rather than by length --
picking the short documents would bias the answer, because a short document has fewer
arguments to miss.

On those, after the verdict: re-read the document and list **every** risk-to-intervention
argument it makes, one row each in `recall_enumeration.csv`, marking for each whether the
pair list at the top of that chain file already carries it.

**Enumerate from the document, then check the list. Never the reverse.** Reading the list
first and asking "is this one in the document" cannot find anything missing, which is the
only thing this pass exists to find. About 20 extra minutes each, so 3-5 hours over the ten.

Two honest limits to carry into the write-up. The rate is **document-level**, so it does not
combine with the chain-level precision weights -- those are reason-code weights and have
nothing to say about recall. And these ten documents were reached through a chain-
proportional sample, so they are size-biased toward chain-rich documents by **1.19x**
(1.76 chains per document against 1.48 across the reporting unit; median 1.0 in both).
Small, and it gets stated rather than corrected.

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
"""


def pairs_for_document(nids, attrs, adj_all):
    """Every risk-to-intervention pair this document's extraction supports.

    Added 2026-08-26, and the recall field is unanswerable without it. An annotator sees ONE
    chain, but every document in this packet yields more than one pair -- median six, one of
    them seventeen. Asking "is there a pair the extraction does not have" while showing a
    single chain would collect pairs the extraction DOES have, filed as recall failures.

    Reachability, undirected, gate-free, matching the released enumerator's traversal
    (sec:m-paths). Gate-free is deliberate: a pair the extraction found and the quality
    gates then rejected is not a recall failure, so including it keeps the annotator from
    reporting one.
    """
    inside = set(nids)
    risks, ivs = [], []
    for n in nids:
        a = attrs.get(n, {})
        if a.get("type") == "intervention":
            ivs.append(n)
        elif (a.get("concept_category") or "").lower() == "risk":
            risks.append(n)
    out = []
    for r in risks:
        seen, q = {r}, deque([r])
        while q:
            x = q.popleft()
            for m in adj_all.get(x, ()):
                if m in inside and m not in seen:
                    seen.add(m)
                    q.append(m)
        for i in ivs:
            if i in seen:
                out.append((attrs.get(r, {}).get("name"), attrs.get(i, {}).get("name")))
    return out


def die(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def host_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return (m.group(1) if m else "unknown").lower().replace("www.", "")


def load_sources() -> dict[str, dict]:
    files = sorted(glob.glob(str(ARD_DIR / "*.jsonl")))
    if not files:
        die(f"ARD source text not found: {ARD_DIR}/*.jsonl")
    by_url: dict[str, dict] = {}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                u = (r.get("url") or "").strip()
                if u and (r.get("text") or "").strip():
                    by_url.setdefault(u, r)
    return by_url


def render_chain(nodes: list[int], attrs: dict) -> str:
    out = []
    for i, n in enumerate(nodes):
        a = attrs.get(n, {})
        if a.get("type") == "intervention":
            label = "INTERVENTION"
            extra = f"  (model-assigned maturity {a.get('intervention_maturity')})"
        else:
            label = (a.get("concept_category") or "?").upper()
            extra = ""
        out.append(f"{i + 1}. **{label}** -- {a.get('name')}{extra}")
    return "\n".join(out)


def preflight_outputs_are_writable() -> None:
    """Refuse to start unless every output can be replaced.

    Learned 2026-08-26 the expensive way. `verdict_sheet.csv` was open in Excel, which
    holds an exclusive lock on Windows. The run rewrote all 30 `chains/*.md` with the new
    sample, then died on the first sheet write -- leaving the packet INCONSISTENT: new
    chain files against an old manifest, so every packet id pointed at a different chain
    than the manifest claimed. That state is silently wrong rather than obviously broken,
    which is the worst kind, and an annotator who started work in it would have produced
    verdicts attached to the wrong chains.

    Checking first costs nothing. Half-writing the packet costs a rebuild at best and
    corrupt verdicts at worst.
    """
    blocked = []
    for name in (
        "verdict_sheet.csv",
        "verdict_sheet_annotator2.csv",
        "manifest.json",
        "README.md",
        "REVEAL_stage1_verdicts.md",
    ):
        p = OUT / name
        if not p.exists():
            continue
        try:
            with p.open("a", encoding="utf-8"):
                pass
        except OSError as e:
            blocked.append(f"{name}: {e.strerror or e}")
    if blocked:
        die(
            "output files are locked by another process, so the packet was NOT touched:\n  "
            + "\n  ".join(blocked)
            + "\n\n  On Windows this is almost always Excel holding the CSV. Close it and\n"
            "  re-run. Nothing has been written, so the existing packet is still coherent."
        )


def main() -> int:
    for p in (
        NODE_ATTRS,
        GRAPH_EDGES,
        RAW / "results.jsonl",
        RAW / "results_contrast.jsonl",
    ):
        if not p.is_file():
            die(
                f"missing input: {p}\n"
                "  results files come from experiment_review_chain_precision.py;\n"
                "  node_attrs_slim.pkl from experiment_review_prep_slim_nodes.py."
            )
    preflight_outputs_are_writable()

    attrs = pickle.load(NODE_ATTRS.open("rb"))
    sources = load_sources()

    nodes_by_url = defaultdict(list)
    for nid, rec in attrs.items():
        if rec.get("url"):
            nodes_by_url[rec["url"]].append(nid)
    adj_all = defaultdict(set)
    for e in pickle.load(GRAPH_EDGES.open("rb")):
        if e.get("type") != "EDGE":
            continue
        adj_all[e["source"]].add(e["target"])
        adj_all[e["target"]].add(e["source"])

    rows = []
    for fn in ("results.jsonl", "results_contrast.jsonl"):
        rows += [
            json.loads(x)
            for x in (RAW / fn).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    rows = [r for r in rows if "verdict" in r]

    # Join the second instrument (#178/#179) where it exists. Only the 66 chains it
    # re-labelled carry `asserted`; everything else is None, which the predicates treat as
    # "no second opinion" rather than as disagreement.
    relabel_fp = HERE / "phase2_results" / "confidence_relabel_raw" / "results.jsonl"
    asserted_by_prec_id: dict[str, bool | None] = {}
    if relabel_fp.is_file():
        for x in relabel_fp.read_text(encoding="utf-8").splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            if "verdict" in r:
                asserted_by_prec_id[r["precision_custom_id"]] = r["verdict"].get(
                    "is_this_link_asserted_by_the_document"
                )
    else:
        print("  NOTE: no #179 results on disk; disagreement strata will be unfillable")

    joined = []
    for r in rows:
        code = r["verdict"]["reason_code"]
        asserted = asserted_by_prec_id.get(r["custom_id"])
        joined.append(
            {
                "row": r,
                "arm": r["arm"],
                "code": code,
                "asserted": asserted,
                "disagree": (code == "faithful" and asserted is False)
                or (code == "risk_framing_invented" and asserted is True),
            }
        )

    rng = random.Random(SEED)
    picked, used = [], set()
    for label, n, pred in STRATA:
        cands = sorted(
            (j for j in joined if j["row"]["custom_id"] not in used and pred(j)),
            key=lambda j: j["row"]["custom_id"],
        )
        if len(cands) < n:
            die(
                f"stratum '{label}' has {len(cands)} chains, needs {n}. Adjust STRATA "
                "rather than silently taking fewer -- the sizes are argued in #176 and a "
                "quiet shortfall would misreport the design."
            )
        rng.shuffle(cands)
        for j in cands[:n]:
            used.add(j["row"]["custom_id"])
            picked.append({"row": j["row"], "stratum_code": label, "arm": j["arm"]})

    # Double-code the hardest strata, then top up in shuffle order if they run short.
    hard = [p for p in picked if p["stratum_code"] in DOUBLE_CODE_STRATA]
    rng.shuffle(hard)
    double_ids = {p["row"]["custom_id"] for p in hard[:N_DOUBLE_CODED]}

    rng.shuffle(picked)  # so strata are not clustered in the reading order

    # Recall arm: fixed here, before any verdict exists. Drawn after the shuffle so it is
    # independent of stratum, and recorded in the manifest so the choice is auditable.
    recall_ids = {
        p["row"]["custom_id"]
        for p in rng.sample(picked, min(N_RECALL_ARM, len(picked)))
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chains").mkdir(exist_ok=True)

    manifest, sheet, reveal = [], [], []
    lengths = []

    for i, item in enumerate(picked):
        r = item["row"]
        pid = f"C{i + 1:02d}"
        src = sources.get(r["source_url"])
        if src is None:
            die(f"source text vanished for {r['source_url']}")
        text = src["text"]
        lengths.append(len(text))
        chain_md = render_chain(r["nodes"], attrs)
        prs = pairs_for_document(nodes_by_url.get(r["source_url"], []), attrs, adj_all)
        pairs_md = "\n".join(
            f"{k + 1}. **{a}**  ->  *{b}*" for k, (a, b) in enumerate(sorted(prs))
        ) or (
            "(none -- this document's extraction supports no complete "
            "risk-to-intervention pair)"
        )

        (OUT / "chains" / f"{pid}.md").write_text(
            f"# {pid}\n\n"
            f"**Source:** {src.get('title') or '(no title)'}\n\n"
            f"**URL:** {r['source_url']}\n\n"
            f"**Document length:** {len(text):,} characters "
            f"(~{len(text) // 5:,} words)\n\n"
            f"---\n\n## Every risk-to-intervention pair this document's extraction "
            f"already holds\n\n"
            f"Read this BEFORE the chain below. `chain_recall_missed` asks whether the "
            f"document argues a pair that appears NOWHERE in this list. The chain you are "
            f"judging is one of them; the others are here so that you do not report a pair "
            f"the extraction already has. The list is gate-free on purpose -- a pair the "
            f"extraction found and the quality gates then rejected is not a recall "
            f"failure.\n\n{pairs_md}\n\n"
            f"---\n\n## The chain you are judging\n\n{chain_md}\n\n"
            f"---\n\n## Your verdict\n\n"
            f"Fill the row for {pid} in `verdict_sheet.csv`. Read the rubric in "
            f"`README.md` first if you have not.\n\n"
            + (
                f"### {pid} IS IN THE RECALL SUBSET -- one extra pass\n\n"
                f"After the verdict, go back over the whole document and list **every** "
                f"risk-to-intervention argument it makes, not only the one above. Add one "
                f"row per argument to `recall_enumeration.csv`, and for each mark whether "
                f"the pair list at the top of this file already carries it.\n\n"
                f"Enumerate what the document argues, then check it against the list -- "
                f'never the other way round. Reading the list first and asking "is this '
                f'one in the document" finds nothing that is missing, which is the whole '
                f"point of the pass.\n\n"
                f"Ten of the thirty are in this subset, fixed before any verdict was "
                f"written. Budget about 20 extra minutes.\n\n"
                if r["custom_id"] in recall_ids
                else ""
            )
            + "Do NOT open `REVEAL_stage1_verdicts.md` until the whole sheet is filled in.\n\n"
            + f"---\n\n## Full source text\n\n```\n{text}\n```\n",
            encoding="utf-8",
        )

        manifest.append(
            {
                "packet_id": pid,
                "custom_id": r["custom_id"],
                "arm": r["arm"],
                "stratum_code": item["stratum_code"],
                "source_url": r["source_url"],
                "host": host_of(r["source_url"]),
                "nodes": r["nodes"],
                "double_coded": r["custom_id"] in double_ids,
                "recall_arm": r["custom_id"] in recall_ids,
            }
        )
        sheet.append(
            {
                "packet_id": pid,
                "risk_link_asserted": "",
                "risk_inference_level": "",
                "risk_quote": "",
                "intervention_inference_level": "",
                "intervention_quote": "",
                "body_inference_level": "",
                "verdict": "",
                "chain_recall_missed": "",
                "chain_recall_note": "",
                "annotator_confidence": "",
                "notes": "",
                "minutes_spent": "",
            }
        )
        v = r["verdict"]
        reveal.append(
            f"### {pid}  ({r['custom_id']}, arm {r['arm']})\n\n"
            f"- stage-1 verdict: **{'fair summary' if v.get('chain_is_a_fair_summary_of_an_argument_the_document_makes') else 'NOT a fair summary'}**\n"
            f"- reason code: `{v.get('reason_code')}`  (judge confidence {v.get('confidence')})\n"
            f"- risk framing: {v.get('risk_framing', {}).get('verdict')} "
            f"| quote: {(v.get('risk_framing', {}).get('quote') or '(none)')[:200]}\n"
            f"- intervention: {v.get('intervention', {}).get('verdict')} "
            f"| quote: {(v.get('intervention', {}).get('quote') or '(none)')[:200]}\n"
            f"- intermediate: {v.get('intermediate_stages', {}).get('verdict')} "
            f"| {(v.get('intermediate_stages', {}).get('note') or '')[:300]}\n"
        )

    double_pids = {m["packet_id"] for m in manifest if m["double_coded"]}

    with (OUT / "verdict_sheet.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0].keys()))
        w.writeheader()
        w.writerows(sheet)

    with (OUT / "verdict_sheet_annotator2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0].keys()))
        w.writeheader()
        w.writerows([s for s in sheet if s["packet_id"] in double_pids])

    # Recall enumeration sheet: one row per argument the annotator finds, NOT one per
    # document. Pre-seeded with eight blank rows for each of the ten so the shape is
    # obvious; add or delete rows freely, since the number of arguments a document makes is
    # exactly what is being measured and must not be capped by the sheet.
    recall_pids = [m["packet_id"] for m in manifest if m["recall_arm"]]
    recall_cols = [
        "packet_id",
        "argument_index",
        "risk",
        "intervention",
        "carried_by_the_pair_list",
        "evidence_quote",
        "notes",
    ]
    with (OUT / "recall_enumeration.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=recall_cols)
        w.writeheader()
        for pid in recall_pids:
            for k in range(8):
                w.writerow(
                    {
                        "packet_id": pid,
                        "argument_index": k + 1,
                        **{c: "" for c in recall_cols[2:]},
                    }
                )

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    (OUT / "REVEAL_stage1_verdicts.md").write_text(
        "# Stage-1 verdicts -- DO NOT OPEN UNTIL THE VERDICT SHEET IS FILLED IN\n\n"
        "These are one model's opinions, produced by `experiment_review_chain_precision.py` "
        "(#175). They are not ground truth and they are not what you are checking against. "
        "They are here so that after you have judged independently, the disagreements can be "
        "counted -- which is the actual output of this exercise.\n\n"
        "Reading these first destroys the study. There is no way to un-anchor.\n\n"
        + "\n".join(reveal),
        encoding="utf-8",
    )

    strata_counts = Counter((m["arm"], m["stratum_code"]) for m in manifest)
    (OUT / "README.md").write_text(
        f"""# Human adjudication packet -- issue #176

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
| `recall_enumeration.csv` | the recall pass, for the {N_RECALL_ARM} chains whose file says so. One row per argument you find |
| `verdict_sheet_annotator2.csv` | {N_DOUBLE_CODED} chains pre-selected for a second annotator **if one is ever found**. None is planned -- see below |
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
{N_DOUBLE_CODED} rows in `verdict_sheet_annotator2.csv` are only worth anything blind.

## The order is shuffled and the ids are opaque, deliberately

The 30 were selected by strata (a machine flagged some and not others) but they are shuffled
and numbered `C01`-`C30`, so nothing in the reading order tells you which is which. If you
can infer a stratum you have lost the property the packet is built to protect.

## Document lengths

Median {sorted(lengths)[len(lengths) // 2]:,} characters, longest {max(lengths):,}. The full
text ships for every chain because deciding a document does NOT assert something requires
being able to search all of it. For the long ones the URL in each file is often easier to
read than the plain-text dump.

## Composition

Deliberately not stated here. The 30 are not a random sample and knowing the mix would tell
you roughly how many to expect in each verdict, which is an anchor as strong as seeing the
verdicts themselves. It is recorded in `manifest.json` and in issue #176, both of which are
for the analysis afterwards.

---

{RUBRIC}
""",
        encoding="utf-8",
    )

    print(f"wrote {OUT}")
    print(f"  30 chains, strata: {dict(strata_counts)}")
    print(
        f"  document length: median {sorted(lengths)[len(lengths) // 2]:,} chars, "
        f"max {max(lengths):,}, total {sum(lengths):,}"
    )
    print(f"  {N_DOUBLE_CODED} chains double-coded for inter-annotator agreement")
    print("  stage-1 verdicts are in REVEAL_stage1_verdicts.md and nowhere else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
