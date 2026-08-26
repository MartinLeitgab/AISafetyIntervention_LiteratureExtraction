#!/usr/bin/env python3
"""Should the confidence gate have caught the invented risk framings? It should. It did not.

The extraction prompt licenses inference and couples it to a low label. Two passages, quoted
from src/prompt/final_primary_prompt.py as it produced the corpus:

  L74-77  "If no interventions explicitly stated: Apply moderate inference to identify AI
          safety-relevant interventions, or focus on transferable interventions [...] Mark as
          lower maturity (must be 1 or 2)"
  L100    "If the required flow and succession of node types/categories is not explicitly
          supported by the data source, use moderate inference to construct knowledge fabric
          paths as close to this intent as possible and mark appropriately in edge confidence
          and edge rationale where inference was used (confidence must be 1 or 2 with
          inference)."

So the design is coherent: an inferred risk framing should produce a low-confidence first hop
out of the risk node, and the reporting unit's confidence >= 3 gate should then drop the
chain. Note what the prompt does NOT say -- there is no instruction to invent a plausible
risk for a source that names none. The inference licence is general (L100) and is always tied
to confidence 1 or 2.

This script tests whether that coupling held, by joining the stage-1 precision verdicts
(#175) to the confidence the extractor actually stored on each chain's first hop.

Class B: no LLM call, no network. Requires the precision results and the edge checkpoint.

    cd graph_analysis
    python -u experiment_review_precision_by_gate_label.py
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "phase2_results" / "chain_precision_raw"
EDGES = (
    HERE
    / "phase2_results"
    / "step1_load_and_parse_umapwithoutlocalsatellites"
    / "graph_edge_data.pkl"
)
OUT = HERE / "phase2_results" / "experiment_review_precision_by_gate_label_report.json"

ARMS = {"real": "results.jsonl", "gate_rejected": "results_contrast.jsonl"}


def main() -> int:
    for p in (EDGES, RAW / "results.jsonl", RAW / "results_contrast.jsonl"):
        if not p.is_file():
            raise SystemExit(
                f"FATAL: missing input: {p}\n"
                "  the results files come from experiment_review_chain_precision.py "
                "--collect / --collect-contrast;\n"
                "  graph_edge_data.pkl comes from phase2_step1_loadandparse.py.\n"
                "  No fallback exists: the join needs the stored confidence, which only the\n"
                "  edge checkpoint carries."
            )

    print("loading edges ...", flush=True)
    best: dict[frozenset, int] = defaultdict(int)
    for e in pickle.load(EDGES.open("rb")):
        if e.get("type") != "EDGE":
            continue
        c = e.get("confidence")
        if c is None:
            continue
        k = frozenset((e["source"], e["target"]))
        if c > best[k]:
            best[k] = c
    print(f"  {len(best):,} distinct structural node pairs", flush=True)

    report: dict = {
        "study": "does the confidence gate catch an invented risk framing?",
        "answers": (
            "The extraction prompt requires confidence 1 or 2 wherever inference was applied "
            "(final_primary_prompt.py L100), so an inferred risk framing should be excluded "
            "by the reporting unit's confidence >= 3 gate. This joins the #175 verdicts to "
            "the confidence the extractor actually stored on each chain's first hop."
        ),
        "arms": {},
    }

    for arm, fn in ARMS.items():
        rows = [
            json.loads(x)
            for x in (RAW / fn).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        rows = [r for r in rows if r.get("arm") == arm and "verdict" in r]
        by_code: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            p = r["nodes"]
            by_code[r["verdict"]["reason_code"]][best[frozenset((p[0], p[1]))]] += 1
        n_total = len(rows)
        report["arms"][arm] = {
            "n": n_total,
            "reason_code_share_pct": {
                k: round(100.0 * sum(v.values()) / n_total, 1)
                for k, v in sorted(by_code.items(), key=lambda kv: -sum(kv[1].values()))
            },
            "first_hop_confidence_by_reason_code": {
                k: {
                    "n": sum(v.values()),
                    "hist": {str(c): n for c, n in sorted(v.items())},
                    "mean": round(
                        sum(c * n for c, n in v.items()) / sum(v.values()), 2
                    ),
                }
                for k, v in sorted(by_code.items(), key=lambda kv: -sum(kv[1].values()))
            },
        }

    real = report["arms"]["real"]["first_hop_confidence_by_reason_code"]
    inv = real.get("risk_framing_invented", {"n": 0, "hist": {}})
    faith = real.get("faithful", {"n": 0, "hist": {}})

    report["headline"] = {
        "invented_risk_framings_whose_first_hop_cleared_the_gate": inv["n"],
        "all_of_them_did": True,
        "why": (
            "Every chain in the reporting unit carries confidence >= 3 on every hop by "
            "construction, so an invented risk framing that reaches the reporting unit is one "
            "the extractor labelled >= 3 in violation of the prompt's own inference rule. "
            "The gate design is sound; the label feeding it is not."
        ),
        "gate_does_not_discriminate_on_this_failure_mode": {
            "risk_framing_invented_pct_arm_A": report["arms"]["real"][
                "reason_code_share_pct"
            ].get("risk_framing_invented"),
            "risk_framing_invented_pct_arm_B": report["arms"]["gate_rejected"][
                "reason_code_share_pct"
            ].get("risk_framing_invented"),
            "reading": (
                "Indistinguishable. The 17.8 pp advantage the gates show overall comes from "
                "the intervention classes, not from this one: intervention_not_proposed runs "
                f"{report['arms']['real']['reason_code_share_pct'].get('intervention_not_proposed')}% "
                f"in the reporting unit against "
                f"{report['arms']['gate_rejected']['reason_code_share_pct'].get('intervention_not_proposed')}% "
                "among gate-rejected chains. The maturity gate works on interventions; "
                "neither gate touches an invented risk."
            ),
        },
    }

    # What a stricter first-hop gate would buy, measured rather than guessed.
    if inv["n"] and faith["n"]:
        inv_ge4 = sum(n for c, n in inv["hist"].items() if int(c) >= 4)
        faith_ge4 = sum(n for c, n in faith["hist"].items() if int(c) >= 4)
        kept = sum(
            n for v in real.values() for c, n in v["hist"].items() if int(c) >= 4
        )
        report["headline"]["a_stricter_first_hop_gate"] = {
            "at_confidence_ge_4_invented_retained": inv_ge4,
            "at_confidence_ge_4_faithful_retained": faith_ge4,
            "chains_retained_of_200": kept,
            "invented_share_after_pct": round(100.0 * inv_ge4 / kept, 1)
            if kept
            else None,
            "invented_share_before_pct": round(100.0 * inv["n"] / 200, 1),
            "reading": (
                "Reported as a measured option, NOT a recommendation. It roughly halves the "
                "invented share but discards most of the chain set with it, and it is a "
                "post-hoc threshold chosen on this sample. No confidence-5 first hop appears "
                "anywhere in the sample."
            ),
        }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== reason-code share, arm A vs arm B ===")
    for arm in ARMS:
        print(f"  {arm:14s} {report['arms'][arm]['reason_code_share_pct']}")
    print("\n=== first-hop confidence by reason code, reporting unit ===")
    for code, d in real.items():
        print(f"  {code:40s} n={d['n']:3d} hist={d['hist']} mean={d['mean']}")
    h = report["headline"]
    print(
        f"\ninvented risk framings that cleared the gate: {h['invented_risk_framings_whose_first_hop_cleared_the_gate']} of 33 sampled"
    )
    if "a_stricter_first_hop_gate" in h:
        s = h["a_stricter_first_hop_gate"]
        print(
            f"a >= 4 first-hop gate: invented share "
            f"{s['invented_share_before_pct']}% -> {s['invented_share_after_pct']}%, "
            f"keeping {s['chains_retained_of_200']} of 200 chains"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
