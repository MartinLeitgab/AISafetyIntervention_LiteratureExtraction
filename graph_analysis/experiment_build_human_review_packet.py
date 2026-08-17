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
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "phase2_results" / "chain_precision_raw"
NODE_ATTRS = HERE / "phase2_results" / "node_attrs_slim.pkl"
ARD_DIR = ROOT / "intervention_graph_creation" / "data" / "raw" / "ard_json_full"
OUT = HERE / "phase2_results" / "human_review_packet"

SEED = 42

# (arm, reason_code, n). Sizes and their justification are in issue #176.
STRATA = [
    ("real", "risk_framing_invented", 10),
    ("real", "intervention_not_proposed", 8),
    ("real", "faithful", 7),
    ("real", "chain_belongs_to_a_different_document", 3),
    ("gate_rejected", "risk_framing_invented", 2),
]

# Double-coded for an inter-annotator figure. Without one the result is one person's opinion,
# which three of the six external reviewers say in as many words.
N_DOUBLE_CODED = 8

RUBRIC = """\
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
"""


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


def main() -> int:
    for p in (NODE_ATTRS, RAW / "results.jsonl", RAW / "results_contrast.jsonl"):
        if not p.is_file():
            die(
                f"missing input: {p}\n"
                "  results files come from experiment_review_chain_precision.py;\n"
                "  node_attrs_slim.pkl from experiment_review_prep_slim_nodes.py."
            )

    attrs = pickle.load(NODE_ATTRS.open("rb"))
    sources = load_sources()

    rows = []
    for fn in ("results.jsonl", "results_contrast.jsonl"):
        rows += [
            json.loads(x)
            for x in (RAW / fn).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    rows = [r for r in rows if "verdict" in r]

    pool: dict[tuple, list] = defaultdict(list)
    for r in rows:
        pool[(r["arm"], r["verdict"]["reason_code"])].append(r)

    rng = random.Random(SEED)
    picked = []
    for arm, code, n in STRATA:
        cands = sorted(pool.get((arm, code), []), key=lambda r: r["custom_id"])
        if len(cands) < n:
            die(
                f"stratum ({arm}, {code}) has {len(cands)} chains, needs {n}. "
                "Adjust STRATA rather than silently taking fewer -- the sizes are argued "
                "in #176 and a quiet shortfall would misreport the design."
            )
        rng.shuffle(cands)
        for r in cands[:n]:
            picked.append({"row": r, "stratum_arm": arm, "stratum_code": code})

    rng.shuffle(picked)  # so strata are not clustered in the reading order

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chains").mkdir(exist_ok=True)

    manifest, sheet, reveal = [], [], []
    double = {f"C{i + 1:02d}" for i in range(N_DOUBLE_CODED)}
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

        (OUT / "chains" / f"{pid}.md").write_text(
            f"# {pid}\n\n"
            f"**Source:** {src.get('title') or '(no title)'}\n\n"
            f"**URL:** {r['source_url']}\n\n"
            f"**Document length:** {len(text):,} characters "
            f"(~{len(text) // 5:,} words)\n\n"
            f"---\n\n## The extracted chain\n\n{chain_md}\n\n"
            f"---\n\n## Your verdict\n\n"
            f"Fill the row for {pid} in `verdict_sheet.csv`. Read the rubric in "
            f"`README.md` first if you have not.\n\n"
            f"Do NOT open `REVEAL_stage1_verdicts.md` until the whole sheet is filled in.\n\n"
            f"---\n\n## Full source text\n\n```\n{text}\n```\n",
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
                "double_coded": pid in double,
            }
        )
        sheet.append(
            {
                "packet_id": pid,
                "risk_supported": "",
                "risk_quote": "",
                "intervention_supported": "",
                "intervention_quote": "",
                "body_supported": "",
                "verdict": "",
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

    with (OUT / "verdict_sheet.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0].keys()))
        w.writeheader()
        w.writerows(sheet)

    with (OUT / "verdict_sheet_annotator2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0].keys()))
        w.writeheader()
        w.writerows([s for s in sheet if s["packet_id"] in double])

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

30 chains from the released corpus. Judge each against its source. **~15-25 minutes each**,
so budget 8-12 hours; the sheet has a `minutes_spent` column because knowing the real cost
is worth as much as the verdicts.

## What is in here

| File | What |
|---|---|
| `README.md` | this, including the rubric |
| `chains/C01.md` ... `C30.md` | one chain plus its full source text |
| `verdict_sheet.csv` | the sheet to fill in, one row per chain |
| `verdict_sheet_annotator2.csv` | {N_DOUBLE_CODED} chains for a second annotator, for the inter-annotator figure |
| `manifest.json` | which packet id maps to which chain -- for the analysis afterwards, not needed while judging |
| `REVEAL_stage1_verdicts.md` | 🔴 **do not open until the sheet is filled in** |

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
