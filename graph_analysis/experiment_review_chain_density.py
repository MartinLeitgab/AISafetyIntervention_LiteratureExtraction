#!/usr/bin/env python
"""Why one model's extraction yields 594 enumerated chains and another's yields 2.

Follow-up inside GitHub issue #168, prompted by a fair objection: 594 chains per document
cannot be a property of an extraction, and it is not. The model does not emit chains. It
emits nodes and edges; a chain is what OUR enumerator counts when it walks simple paths
from a risk root to a maturity->=3 intervention. This script profiles the emitted graphs to
show which of their properties the path count is a function of.

Three multiply together, and all three are properties of the graph rather than of the
extraction's quality:

  * risk roots            each one starts its own enumeration
  * mature interventions  each one is a valid terminal
  * mean degree           below 2 a graph is nearly a path; above 2 it carries cycles, and
                          the number of distinct SIMPLE paths grows exponentially in the
                          number of independent cycles

The released corpus is subject to the same arithmetic -- one paper contributes 700 of the
8,954 raw chains -- which is why the enumerator carries a path cap at all.

CLASS B: no LLM call, no network. Run from graph_analysis/:
    python -u experiment_review_chain_density.py

Output: phase2_results/experiment_review_chain_density_report.json
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import experiment_review_schema_ablation as ABL

EDGE_CONFIDENCE_MIN = ABL.EDGE_CONFIDENCE_MIN
INTERVENTION_MATURITY_MIN = ABL.INTERVENTION_MATURITY_MIN
MIN_HOPS, MAX_HOPS = ABL.MIN_HOPS, ABL.MAX_HOPS
CONTAINMENT = 0.70  # phase1_dedup_paths.py, the value that ran


def enumerate_chains(nodes: dict, edges: list, directed: bool) -> list:
    """The released enumerator, with edge direction optional.

    The released run is undirected (`MATCH (n)-[e:EDGE]-(m)`); `directed=True` is the
    counterfactual this script reports beside it.
    """
    adj = {}
    for e in edges:
        try:
            conf = int(e.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0
        if conf < EDGE_CONFIDENCE_MIN:
            continue
        if e["source"] in nodes and e["target"] in nodes:
            adj.setdefault(e["source"], []).append(e["target"])
            if not directed:
                adj.setdefault(e["target"], []).append(e["source"])

    def is_risk(n):
        return (
            nodes[n]["type"] == "concept"
            and (nodes[n].get("category") or "").lower() == "risk"
        )

    def is_intv(n):
        return nodes[n]["type"] == "intervention"

    chains = []
    for root in [n for n in nodes if is_risk(n)]:
        stack = [(root, [root], {root})]
        while stack:
            cur, path, seen = stack.pop()
            for nb in adj.get(cur, ()):
                if nb in seen or is_risk(nb):
                    continue
                if len(path) == 1 and is_intv(nb):
                    continue
                new = path + [nb]
                if is_intv(nb):
                    if (
                        ABL._mat(nodes[nb]) >= INTERVENTION_MATURITY_MIN
                        and MIN_HOPS <= len(new) - 1 <= MAX_HOPS
                    ):
                        chains.append(new)
                    continue
                if len(new) - 1 < MAX_HOPS:
                    stack.append((nb, new, seen | {nb}))
    return chains


def collapse(chains: list) -> list:
    """The released sub-path collapse: greedy, longest first, drop anything whose node set
    is >=70% inside a chain already kept from the same document."""
    kept = []
    for c in sorted(chains, key=len, reverse=True):
        s = set(c)
        if not any(len(s & set(k)) / len(s) >= CONTAINMENT for k in kept):
            kept.append(c)
    return kept


ROOT = Path(__file__).parent
OUT = ROOT / "phase2_results/experiment_review_chain_density_report.json"
ARMS = {
    "A_o3": "phase2_results/multimodel_raw/A_o3/*.json",
    "B_gpt5": "phase2_results/multimodel_raw/B_gpt5/*.json",
    "C_opus5": "phase2_results/multimodel_raw/C_opus5/*.json",
}


def profile(nodes: dict, edges: list) -> dict:
    deg = defaultdict(int)
    e3 = 0
    for e in edges:
        if int(e.get("confidence") or 0) >= 3:
            e3 += 1
            deg[e["source"]] += 1
            deg[e["target"]] += 1
    risks = [
        k
        for k, v in nodes.items()
        if v["type"] == "concept" and (v.get("category") or "").lower() == "risk"
    ]
    intv = [k for k, v in nodes.items() if v["type"] == "intervention"]
    und = enumerate_chains(nodes, edges, directed=False)
    dir_ = enumerate_chains(nodes, edges, directed=True)
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "edges_conf_ge3": e3,
        "mean_degree_conf_ge3": round(2 * e3 / max(1, len(nodes)), 2),
        "max_degree": max(deg.values()) if deg else 0,
        "risk_roots": len(risks),
        "interventions": len(intv),
        "mature_interventions": sum(1 for k in intv if ABL._mat(nodes[k]) >= 3),
        "chains_undirected_raw": len(und),
        "chains_undirected_collapsed": len(collapse(und)),
        "chains_directed_raw": len(dir_),
        "chains_directed_collapsed": len(collapse(dir_)),
    }


def main() -> None:
    report = {
        "study": "what the enumerated chain count is a function of (issue #168 follow-up)",
        "what_a_chain_is": (
            "a simple path this project's enumerator walks from a risk root to a "
            "maturity>=3 intervention. The extractor emits nodes and edges and never a "
            "chain, so a chain count is a property of the graph plus our traversal."
        ),
        "arms": {},
    }
    for arm, pat in ARMS.items():
        rows = []
        for fp in sorted(glob.glob(str(ROOT / pat))):
            rec = json.loads(Path(fp).read_text(encoding="utf-8"))
            ext = ABL.parse_extraction(rec.get("text") or "")
            if not ext:
                continue
            nodes, edges = ABL.graph_from_extraction(ext)
            rows.append((rec["url"], profile(nodes, edges)))
        if not rows:
            continue
        keys = [k for k in rows[0][1]]
        med = {k: statistics.median([r[1][k] for r in rows]) for k in keys}
        worst = max(rows, key=lambda r: r[1]["chains_undirected_raw"])
        report["arms"][arm] = {
            "n_documents": len(rows),
            "median": med,
            "worst_document": {"url": worst[0], **worst[1]},
        }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for arm, a in report["arms"].items():
        m = a["median"]
        print(
            f"{arm:9s} n={a['n_documents']:2d} degree {m['mean_degree_conf_ge3']:.2f}"
            f" roots {m['risk_roots']:.0f} mature {m['mature_interventions']:.0f}"
            f" | median chains: undirected raw {m['chains_undirected_raw']:.0f},"
            f" collapsed {m['chains_undirected_collapsed']:.0f};"
            f" directed raw {m['chains_directed_raw']:.0f},"
            f" collapsed {m['chains_directed_collapsed']:.0f}"
            f" | worst doc raw {a['worst_document']['chains_undirected_raw']}"
            f" -> collapsed {a['worst_document']['chains_undirected_collapsed']}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
