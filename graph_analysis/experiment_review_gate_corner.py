#!/usr/bin/env python
"""Joint distribution of the two attributes that gate the chain set.

The chain set is selected by two LLM-assigned attributes: every edge on a path must carry
edge confidence >= 3, and the intervention that ends it must carry maturity >= 3. The
paper reports each marginal separately (maturity in the dataset descriptives, confidence
nowhere), so a reader cannot see how much of the extracted corpus the pair admits, or
whether the two are correlated.

This computes the joint over every extracted intervention:

    x = intervention_maturity (1..4)
    y = the highest edge confidence among the structural edges incident to it (1..5)

Max, not min: a path needs ONE sufficiently confident edge to reach the intervention, so
the maximum is the right question for "could a confidence-gated path end here". The min is
reported alongside as the stricter reading.

The gate corner is (maturity >= 3) x (best incident confidence >= 3): the interventions
eligible to terminate a gate-selected chain. Its mass is the honest denominator behind
"one document in six".

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_review_gate_corner.py

Output: graph_analysis/phase2_results/experiment_review_gate_corner_report.json
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = STEP1 / "graph_edge_data.pkl"
OUT = ROOT / "phase2_results/experiment_review_gate_corner_report.json"

MAT = [1, 2, 3, 4]
CONF = [1, 2, 3, 4, 5]


def fail(artifact: Path, produced_by: str) -> None:
    raise SystemExit(
        f"FATAL: missing {artifact}\n"
        f"  produced by: {produced_by}\n"
        "  this script does NOT fall back to a marginal distribution: the point of it is "
        "the joint."
    )


def main():
    for p, how in [
        (SLIM, "experiment_review_prep_slim_nodes.py"),
        (EDGES, "phase2_step1_loadandparse.py"),
    ]:
        if not p.exists():
            fail(p, how)

    na = pickle.load(open(SLIM, "rb"))
    interventions = {
        n: a
        for n, a in na.items()
        if (a.get("type") or "").lower() == "intervention"
        and a.get("intervention_maturity") in MAT
    }
    print(f"interventions with a maturity label: {len(interventions)}", flush=True)

    edges = pickle.load(open(EDGES, "rb"))
    best = defaultdict(int)
    worst = defaultdict(lambda: 99)
    n_incident = defaultdict(int)
    for e in edges:
        if e.get("type") != "EDGE":
            continue
        c = e.get("confidence")
        if c not in CONF:
            continue
        for endpoint in (e["source"], e["target"]):
            if endpoint in interventions:
                best[endpoint] = max(best[endpoint], c)
                worst[endpoint] = min(worst[endpoint], c)
                n_incident[endpoint] += 1
    del edges

    joint_best = {m: {c: 0 for c in CONF} for m in MAT}
    joint_worst = {m: {c: 0 for c in CONF} for m in MAT}
    isolated = 0
    for n, a in interventions.items():
        m = a["intervention_maturity"]
        if not n_incident[n]:
            isolated += 1
            continue
        joint_best[m][best[n]] += 1
        joint_worst[m][worst[n]] += 1

    placed = sum(sum(r.values()) for r in joint_best.values())
    corner = sum(joint_best[m][c] for m in (3, 4) for c in (3, 4, 5))

    report = {
        "experiment": "joint distribution of the two chain-set gates over extracted interventions",
        "SCOPE_NOTE": (
            "Both axes are LLM assignments that the judge protocol does not score. This is "
            "the composition of the extracted corpus with respect to the two gates, not a "
            "measured property of the literature. The corner is a necessary condition for "
            "an intervention to end a gate-selected chain, not a sufficient one: the path "
            "reaching it must also clear the confidence gate at every hop."
        ),
        "definitions": {
            "x": "intervention_maturity 1..4 as assigned by the extractor",
            "y_primary": "highest edge_confidence among structural edges incident to the "
            "intervention (max: a path needs one sufficiently confident edge to arrive)",
            "y_secondary": "lowest such confidence, the stricter reading",
        },
        "n_interventions_with_maturity": len(interventions),
        "n_with_at_least_one_structural_edge": placed,
        "n_isolated_excluded": isolated,
        "joint_maturity_by_best_incident_confidence": joint_best,
        "joint_maturity_by_worst_incident_confidence": joint_worst,
        "gate_corner_maturity_ge3_and_best_conf_ge3": {
            "n": corner,
            "pct_of_placed": round(100 * corner / placed, 2),
        },
        "marginals": {
            "maturity": {m: sum(joint_best[m].values()) for m in MAT},
            "best_incident_confidence": {
                c: sum(joint_best[m][c] for m in MAT) for c in CONF
            },
        },
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\nmaturity (rows) x best incident edge confidence (cols)")
    print(f"{'':>10}" + "".join(f"{c:>9}" for c in CONF))
    for m in MAT:
        print(f"{m:>10}" + "".join(f"{joint_best[m][c]:>9,}" for c in CONF))
    print(
        f"\ngate corner (mat>=3, best conf>=3): {corner:,} = {report['gate_corner_maturity_ge3_and_best_conf_ge3']['pct_of_placed']}% of {placed:,}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
