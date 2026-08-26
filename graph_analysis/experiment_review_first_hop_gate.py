#!/usr/bin/env python3
"""What the chain set becomes if the first hop out of the risk must carry confidence >= 4.

tab:gates already re-enumerates the corpus under all nine combinations of the two symmetric
gates. This adds an asymmetric one: every hop >= 3 as deployed, EXCEPT the risk-to-first-body
hop, which must be >= 4. It exists because #175 found that all 33 invented risk framings in
its sample cleared the >= 3 gate, and the invented group's first hops sit lower (mean 3.18)
than the faithful group's (3.47).

Reported as a sensitivity row, NOT as a new operating point, and the row's own caption has to
say why: the threshold was chosen on the sample that measured it, so its apparent benefit is
selection on the outcome until something independent confirms it. Producing corpus-level
counts for the table is the whole reason this script exists -- the #175 figures are properties
of 200 sampled chains and belong nowhere near tab:gates.

Class B: no LLM call. Reuses the enumerator from experiment_review_gate_sensitivity.py, which
asserts it reproduces the released 8,954-path and 2,772-chain files before anything else is
believed.

    cd graph_analysis
    python -u experiment_review_first_hop_gate.py
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "phase2_results" / "experiment_review_first_hop_gate_report.json"
FIRST_HOP_MIN = 4
REST_MIN = 3
CONTAINMENT = 0.70


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "gate_sens", HERE / "experiment_review_gate_sensitivity.py"
    )
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    sys.setrecursionlimit(50000)

    print("loading ...", flush=True)
    slim = pickle.load(gs.SLIM.open("rb"))
    edge_rows = pickle.load(gs.EDGES.open("rb"))

    risk_nodes = {
        n
        for n, a in slim.items()
        if (a.get("concept_category") or "").lower() == "risk"
    }
    body_nodes = {
        n
        for n, a in slim.items()
        if (a.get("concept_category") or "").lower() in set(gs.BODY)
    }
    interventions = {
        n for n, a in slim.items() if (a.get("type") or "").lower() == "intervention"
    }
    mat3 = {
        n for n in interventions if (slim[n].get("intervention_maturity") or 0) >= 3
    }
    url_of = {n: a.get("url") for n, a in slim.items()}

    best: dict[frozenset, int] = defaultdict(int)
    for e in edge_rows:
        if e.get("type") != "EDGE":
            continue
        c = e.get("confidence")
        if c is None:
            continue
        k = frozenset((e["source"], e["target"]))
        if c > best[k]:
            best[k] = c

    released_raw = [
        json.loads(x)["path"]
        for x in (HERE / "phase1_rawpathsfiles" / "paths_hopwise_v4_edge_only.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    adj3, _ = gs.build_adjacency(edge_rows, REST_MIN)
    base, _ = gs.enumerate_paths(adj3, risk_nodes, body_nodes, mat3)
    if Counter(map(tuple, base)) != Counter(map(tuple, released_raw)):
        raise SystemExit(
            "FATAL: the in-memory enumerator no longer reproduces the released 8,954-path "
            "file at the deployed setting. Every number below would be unverifiable; fix "
            "that before trusting anything here."
        )
    print(
        f"  guard OK: reproduced the released {len(base):,}-path enumeration",
        flush=True,
    )

    # The asymmetric gate is a filter over the deployed enumeration rather than a new
    # traversal: the constraint only tightens the first hop, so any qualifying path is
    # already in the >= 3 set. Filtering is exact here and avoids re-running the DFS.
    kept = [p for p in base if best[frozenset((p[0], p[1]))] >= FIRST_HOP_MIN]
    print(
        f"  raw paths surviving a >= {FIRST_HOP_MIN} first hop: {len(kept):,}",
        flush=True,
    )

    def url_list(paths):
        out = []
        for p in paths:
            u = {url_of.get(x) for x in p}
            u.discard(None)
            out.append(sorted(u)[0] if u else None)
        return out

    keep_idx, _ = gs.dedupe(kept, url_list(kept), CONTAINMENT)
    chains = [kept[i] for i in keep_idx]
    papers = {u for u in url_list(chains) if u}

    body_set = set(gs.BODY)
    all5 = 0
    len7 = 0
    for p in chains:
        cats = {
            (slim.get(n, {}).get("concept_category") or "").lower() for n in p[1:-1]
        }
        if body_set <= cats:
            all5 += 1
        if len(p) == 7:
            len7 += 1
    arxiv = sum(1 for u in url_list(chains) if u and "arxiv.org" in u)

    n = len(chains)
    row = {
        "edge_confidence": f">= {REST_MIN} (first hop >= {FIRST_HOP_MIN})",
        "maturity": ">= 3",
        "chains": n,
        "papers": len(papers),
        "corpus_yield_pct": round(100.0 * len(papers) / 11779, 1),
        "all_five_pct": round(100.0 * all5 / n, 1) if n else None,
        "length_7_pct": round(100.0 * len7 / n, 1) if n else None,
        "arxiv_pct": round(100.0 * arxiv / n, 1) if n else None,
    }

    deployed = {
        "chains": 2772,
        "papers": 1868,
        "corpus_yield_pct": 15.9,
        "all_five_pct": 87.4,
        "length_7_pct": 64.0,
        "arxiv_pct": 38.1,
    }

    report = {
        "study": "asymmetric first-hop confidence gate, corpus-wide",
        "definition": (
            f"every traversed hop >= {REST_MIN} as deployed, except the risk-to-first-body "
            f"hop which must be >= {FIRST_HOP_MIN}; then the same {CONTAINMENT} sub-path "
            "collapse."
        ),
        "why_it_is_a_sensitivity_row_and_not_an_operating_point": (
            "The threshold was chosen on the 200-chain sample of #175, which is the same "
            "sample whose invented-risk share it appears to improve. Adopting it as the "
            "reported setting would be selection on the outcome. It is reported the way "
            "tab:gates reports its other rows: as what the corpus becomes under a different "
            "cut, with no claim that the cut is better."
        ),
        "row": row,
        "deployed_for_comparison": deployed,
        "cost_of_the_cut": {
            "chains_lost": deployed["chains"] - n,
            "chains_lost_pct": round(
                100.0 * (deployed["chains"] - n) / deployed["chains"], 1
            ),
            "papers_lost": deployed["papers"] - len(papers),
            "corpus_yield_falls_from_to": [
                deployed["corpus_yield_pct"],
                row["corpus_yield_pct"],
            ],
        },
        "what_it_does_NOT_establish": (
            "That the surviving chains are more faithful. #175 measured a lower invented "
            "share among high-first-hop chains on 200 sampled chains judged by one model; "
            "nothing here re-judges the surviving corpus. The row shows the price of the "
            "cut, not its benefit."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  chains {n:,} (deployed 2,772) | papers {len(papers):,} (1,868)")
    print(
        f"  yield {row['corpus_yield_pct']}% (15.9%) | all-five {row['all_five_pct']}% (87.4%)"
    )
    print(
        f"  length-7 {row['length_7_pct']}% (64.0%) | arXiv {row['arxiv_pct']}% (38.1%)"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
