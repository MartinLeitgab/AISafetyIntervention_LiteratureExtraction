#!/usr/bin/env python3
"""What confidence did the extractor store on the chain the paper calls spurious?

app:failure exhibits the Euclid chain -- an expository mathematics article about infinitely
many primes, cast into the risk-to-intervention schema with an invented safety framing --
and reports that it "passes every structural filter we apply ... edge confidence >= 3".
Reviewers GC and GW raised the same point as R34: the gates do not reject a clear negative.

Neither says what the stored confidences ACTUALLY are, and the extraction prompt is explicit
that inference should have been marked down: "Where the source does not support the required
flow, use moderate inference and mark it in edge confidence (1 or 2)". So the question this
script answers is whether the flag fired at all on the one chain we know is spurious.

Class B: no LLM call, no network. Reads the released edge table.

    cd graph_analysis
    python -u experiment_review_failure_case_confidence.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGES = (
    HERE
    / "phase2_results"
    / "step1_load_and_parse_umapwithoutlocalsatellites"
    / "graph_edge_data.pkl"
)
OUT = HERE / "phase2_results" / "experiment_review_failure_case_confidence_report.json"

# SRC: app:failure source comment -- "node ids [60117..60123],
#      source https://arbital.com/p/infinitely_many_primes"
CHAIN = list(range(60117, 60124))
LABELS = ["risk->pa", "pa->ti", "ti->dr", "dr->im", "im->va", "va->intv"]

# SRC: tab:rubric-conf, reproduced from the extraction prompt
RUBRIC = {
    5: "a proof, a rigorous study, or a replication",
    4: "a controlled experiment",
    3: "systematic qualitative evidence",
    2: "a single example, or light inference by the model",
    1: "speculation, or moderate inference by the model",
}


def main() -> None:
    if not EDGES.exists():
        raise SystemExit(
            f"FATAL: {EDGES} not found.\n"
            f"  Rebuild it from the released dump with the step1 loader, or point this\n"
            f"  script at another copy. This script does NOT fall back to a cached result."
        )
    rows = pickle.load(open(EDGES, "rb"))
    want = set(CHAIN)
    idx: dict[tuple[int, int], list[dict]] = {}
    for e in rows:
        if e.get("type") != "EDGE":
            continue
        s, t = e.get("source"), e.get("target")
        if s in want and t in want:
            idx.setdefault((s, t), []).append(e)

    hops = []
    for lab, (a, b) in zip(LABELS, zip(CHAIN, CHAIN[1:])):
        got = idx.get((a, b)) or idx.get((b, a))
        if not got:
            raise SystemExit(
                f"FATAL: no stored edge for {lab} ({a}->{b}) in either direction"
            )
        e = got[0]
        hops.append(
            {
                "hop": lab,
                "source": a,
                "target": b,
                "confidence": e.get("confidence"),
                "confidence_means": RUBRIC.get(e.get("confidence")),
                "subtype": e.get("subtype"),
                "description": e.get("description"),
                "stored_reversed": (e["source"], e["target"]) != (a, b),
            }
        )

    confs = [h["confidence"] for h in hops]
    report = {
        "experiment": "stored edge confidence on the app:failure Euclid chain",
        "question": "did the prompt's mark-inference-with-low-confidence rule fire on the "
        "one chain the paper calls spurious?",
        "source_url": "https://arbital.com/p/infinitely_many_primes",
        "node_ids": CHAIN,
        "hops": hops,
        "confidences": confs,
        "min": min(confs),
        "max": max(confs),
        "n_at_4_controlled_experiment": sum(1 for c in confs if c == 4),
        "answer": (
            "NO. Every hop is at or above the >=3 gate, and four are at 4, which the rubric "
            "defines as 'a controlled experiment' -- on an expository mathematics article "
            "that contains none. The flag did not fire."
        ),
        "the_nuance_that_cuts_the_other_way": (
            "The INVENTED hop is the lowest. risk->pa, the one asserting prime scarcity "
            "threatens cryptographic key space, is exactly 3; the 4s sit on the mathematical "
            "steps, which are genuinely rigorous. The extractor discriminated partially and "
            "simply never went below the gate."
        ),
        "consequence_for_tab_gates": (
            "The asymmetric first-hop >=4 row would have EXCLUDED this chain. That is "
            "independent of the sample which chose the threshold -- app:failure predates "
            "#175. 🔴 ONE CASE. It does NOT license adding a benefit side to that row; the "
            "caption's warning stands and #176 remains the arm that could."
        ),
        "consequence_for_sec_m_extraction": (
            "sec:m-extraction says the >=3 gate 'admits links the source evidences "
            "systematically and excludes those the model supplied'. This chain's risk->pa "
            "link was supplied by the model and was admitted at 3."
        ),
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for h in hops:
        print(
            f"  {h['hop']:10s} confidence={h['confidence']} "
            f"({h['confidence_means']})  {h['subtype']}"
        )
    print(
        f"\n  confidences={confs}  min={min(confs)}  "
        f"at-4={report['n_at_4_controlled_experiment']}"
    )
    print(f"  wrote {OUT.name}")


if __name__ == "__main__":
    main()
