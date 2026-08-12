#!/usr/bin/env python
"""Re-derive the "88% of top-100 single-path risks are race-framed" figure.

That number is the load-bearing example of the selection artefact in the paper's
guidance section, but it originates from the frozen Overleaf analysis and we held no
receipt for it. This reproduces it on our substrate under the frozen analysis's own definitions, and
then repeats the measurement on the decontaminated graph to show it is a selection
artefact rather than a corpus property.

Definitions taken verbatim from the frozen Overleaf:
  - importance      = eigenvector centrality on the (merged, SIM-augmented) graph
  - path diversity  = number of distinct first-hop STRUCTURAL (EDGE) neighbours of a
                      risk that are problem-analysis nodes
  - single-path     = path diversity == 1
  - race-framed     = the sole problem-analysis neighbour's name contains
                      "competitive" or "race"

Conditions:
  A  merged risk block + full within-category SIM   (reproduces the frozen setup)
  B  un-merged + risk<->risk SIM excluded           (decontaminated, per our controls)

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_race_top100_rederive.py
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path


import experiment_merge_vs_simexcl_ec as M

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
OUT = ROOT / "phase2_results/experiment_race_top100_rederive_report.json"

# The frozen analysis's classifier, verbatim: name contains "competitive" or "race".
FROZEN_RACE = re.compile(r"competitive|race", re.I)
# Our broader strict pattern, for comparison with the corpus-prevalence receipt.
OUR_RACE = re.compile(r"\brac(?:e|es|ed|ing)\b|competi|arms.?race", re.I)


def pa_neighbours(ed, na, cn, idx_ok):
    """risk canonical id -> set of canonical problem-analysis first-hop EDGE neighbours."""
    out = defaultdict(set)
    for e in ed:
        if (e.get("type") or "").upper() != "EDGE":
            continue
        s, t = e.get("source"), e.get("target")
        if s is None or t is None:
            continue
        cs, ct = cn(s), cn(t)
        if cs == ct or cs not in idx_ok or ct not in idx_ok:
            continue
        for a, b in ((cs, ct), (ct, cs)):
            if (na.get(a, {}).get("concept_category") or "").lower() != "risk":
                continue
            if (
                na.get(b, {}).get("concept_category") or ""
            ).lower() == "problem analysis":
                out[a].add(b)
    return out


def measure(label, na, ed, risk_map, exclude_risk_sim):
    A, nids, idx, _ = M.build_A(na, ed, risk_map, exclude_risk_sim)
    vec, n_iter = M.ec(A)  # ec() returns (eigenvector, iterations)

    def cn(x):
        return risk_map.get(x, x)

    idx_ok = set(nids)
    pa = pa_neighbours(ed, na, cn, idx_ok)

    risks = [
        n
        for n in nids
        if (na.get(n, {}).get("concept_category") or "").lower() == "risk"
    ]
    risks.sort(key=lambda n: -vec[idx[n]])

    def block(subset, name):
        single = [r for r in subset if len(pa.get(r, ())) == 1]
        race_g, race_o = 0, 0
        for r in single:
            nm = na.get(next(iter(pa[r])), {}).get("name") or ""
            if FROZEN_RACE.search(nm):
                race_g += 1
            if OUR_RACE.search(nm):
                race_o += 1
        return {
            "tier": name,
            "n_risks": len(subset),
            "n_single_path": len(single),
            "pct_single_path": round(100 * len(single) / len(subset), 1)
            if subset
            else None,
            "n_race_framed_gleb_keywords": race_g,
            "pct_of_single_path_race_framed": round(100 * race_g / len(single), 1)
            if single
            else None,
            "pct_of_single_path_race_framed_our_regex": round(
                100 * race_o / len(single), 1
            )
            if single
            else None,
        }

    tiers = [
        block(risks[:100], "top-100 by EC"),
        block(risks[:500], "top-500 by EC"),
        block(risks[:1000], "top-1000 by EC"),
        block(risks, f"all {len(risks)} risks"),
    ]
    return {
        "condition": label,
        "n_risk_nodes": len(risks),
        "top10_ec": [
            {"name": (na.get(n, {}).get("name") or "")[:70], "ec": float(vec[idx[n]])}
            for n in risks[:10]
        ],
        "tiers": tiers,
        "gradient_ratio_top100_vs_all": (
            round(
                tiers[0]["pct_of_single_path_race_framed"]
                / tiers[3]["pct_of_single_path_race_framed"],
                1,
            )
            if tiers[3]["pct_of_single_path_race_framed"]
            else None
        ),
    }


def main():
    print("loading substrate ...", flush=True)
    na = pickle.load(open(STEP1 / "graph_node_attributes.pkl", "rb"))
    ed = pickle.load(open(STEP1 / "graph_edge_data.pkl", "rb"))
    print(f"  {len(na)} nodes, {len(ed)} edges", flush=True)

    print(
        "merging risk block (frozen rule: alias P0 + cos>=0.88 AND Jaccard>=0.05) ...",
        flush=True,
    )
    risk_map, member_count, n_risk, n_groups = M.merge_risk_block(na)
    print(f"  {n_risk} risk nodes -> {n_groups} canonical", flush=True)

    print(
        "\ncondition A: merged + full within-category SIM (the frozen setup) ...",
        flush=True,
    )
    a = measure(
        "A merged + full SIM (reproduces frozen Overleaf setup)",
        na,
        ed,
        risk_map,
        False,
    )
    print(
        "condition B: un-merged + risk<->risk SIM excluded (decontaminated) ...",
        flush=True,
    )
    b = measure(
        "B un-merged + risk<->risk SIM excluded (decontaminated)", na, ed, {}, True
    )

    out = {
        "experiment": "re-derivation of the 88% race-framing figure on the top-100-by-EC head",
        "definitions": {
            "importance": "eigenvector centrality",
            "path_diversity": "distinct first-hop EDGE neighbours of concept_category='problem analysis'",
            "single_path": "path diversity == 1",
            "race_framed": "sole problem-analysis neighbour name matches /competitive|race/i (frozen rule)",
        },
        "frozen_overleaf_claim": {
            "top100_single_path_risks": 41,
            "of_which_race_framed": 36,
            "pct": 88,
            "gradient": "88% top-100 -> 45% top-500 -> 22% top-1000 -> 2% all (44x)",
        },
        "condition_A": a,
        "condition_B": b,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")

    for cond in (a, b):
        print(f"\n=== {cond['condition']}")
        for t in cond["tiers"]:
            print(
                f"  {t['tier']:>22}: single-path {t['n_single_path']:>5}/{t['n_risks']:<6}"
                f" race-framed {t['pct_of_single_path_race_framed']}%"
                f"  (our regex {t['pct_of_single_path_race_framed_our_regex']}%)"
            )
        print(f"  gradient top-100 vs all: {cond['gradient_ratio_top100_vs_all']}x")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
