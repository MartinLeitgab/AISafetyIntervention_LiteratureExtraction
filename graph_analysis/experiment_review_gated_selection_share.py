#!/usr/bin/env python
"""Receipt for "471 of 597 graph pairs are not in the reporting unit" (review C1, v5).

The figure was quoted in paper/FINDINGS_2026-08-26_chain_unit.md section 3.2 without a
receipt file. This re-derives it on the 97-document gated recall sample: for each document,
the risk-to-intervention node pairs connected in its structural-edge subgraph (undirected,
no gates, no length rule), against the pairs the 2,772-chain reporting unit carries for the
same document.

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_review_gated_selection_share.py
Output: phase2_results/experiment_review_gated_selection_share_report.json
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
SAMPLE = ROOT / "phase2_results/chain_recall_gated_raw/sample.json"
NODES = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl"
)
DEDUP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
OUT = ROOT / "phase2_results/experiment_review_gated_selection_share_report.json"


def main() -> int:
    for p in (SAMPLE, NODES, EDGES, DEDUP):
        if not p.exists():
            raise SystemExit(f"FATAL: missing input {p}")
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    urls = {s["url"] for s in sample}
    na = pickle.load(open(NODES, "rb"))
    by_url = defaultdict(set)
    for n, a in na.items():
        if a.get("url") in urls:
            by_url[a["url"]].add(n)
    adj = defaultdict(set)
    for e in pickle.load(open(EDGES, "rb")):
        if e["type"] != "EDGE":
            continue
        s, t = e["source"], e["target"]
        if na.get(s, {}).get("url") in urls:
            adj[s].add(t)
            adj[t].add(s)

    def is_risk(n):
        return (na[n].get("concept_category") or "").lower() == "risk"

    def is_intv(n):
        return na[n].get("type") == "intervention"

    graph_pairs = {}
    for u in urls:
        pairs = set()
        for r in (n for n in by_url[u] if is_risk(n)):
            seen, stack = {r}, [r]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            pairs |= {(r, i) for i in seen if is_intv(i)}
        graph_pairs[u] = pairs

    unit_pairs = defaultdict(set)
    for line in open(DEDUP, encoding="utf-8"):
        p = json.loads(line)["path"]
        u = na[p[0]]["url"]
        if u in urls:
            unit_pairs[u].add((p[0], p[-1]))

    n_graph = sum(len(v) for v in graph_pairs.values())
    n_unit = sum(len(unit_pairs[u]) for u in urls)
    n_unit_not_in_graph = sum(len(unit_pairs[u] - graph_pairs[u]) for u in urls)
    n_absent = sum(len(graph_pairs[u] - unit_pairs[u]) for u in urls)
    report = {
        "study": "share of graph risk-intervention pairs absent from the reporting unit",
        "documents": len(urls),
        "graph_pairs": n_graph,
        "unit_pairs": n_unit,
        "unit_pairs_not_in_graph_pairs": n_unit_not_in_graph,
        "graph_pairs_absent_from_unit": n_absent,
        "absent_pct": round(100 * n_absent / n_graph, 1),
        "WHY_ABSENT": "A pair connected in the document's graph is absent from the unit because "
        "a chain between them fails a gate (confidence or maturity), a structural rule (first "
        "hop, minimum length, stop at first qualifying intervention), or the 0.70 collapse.",
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    if n_unit_not_in_graph:
        raise SystemExit(
            "FATAL: a reporting-unit pair is not connected in its own graph"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
