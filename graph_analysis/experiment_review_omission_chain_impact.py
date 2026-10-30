#!/usr/bin/env python
"""If we granted the judge every omission it flagged, how many CHAINS would appear?

experiment_review_omission_is_chain_level.py established that the judge's missing rows are
not inert -- 91.5% name a node the extraction lacks. That answers "are they real" and
leaves the question that decides whether they matter: a chain is a connected
risk-to-intervention argument, so an omission only costs the artifact something if
repairing it would put a risk in touch with an intervention it could not previously reach.

This grants the judge everything and measures what follows. For each of the 100 audited
papers it builds the released per-paper subgraph, then adds every relationship the judge
marked `missing` -- inventing a placeholder node wherever an endpoint does not exist -- and
compares the set of reachable (risk, intervention) pairs before and after.

Two design choices, both deliberately generous to the judge:

  REACHABILITY, NOT ENUMERATION. A pair is counted if a path exists at all, ignoring the
  first-hop subtype rule, the three-hop floor, the thirty-hop ceiling, and both quality
  gates. Every one of those can only REMOVE pairs, so the delta here is an upper bound on
  new chains and the "no impact" reading, if it survives, survives a fortiori.

  UNDIRECTED. The released enumerator traverses edges undirected (sec:m-paths), so this
  does too. Honouring direction would only reduce reachability.

Placeholder nodes are typed `unknown`, so they can never themselves be a risk or an
intervention endpoint. That is not a limitation being hidden: 98.7% of the absent names
carry no category marker, so there is nothing to type them from. It means the test measures
whether the missing relationships act as BRIDGES between endpoints the extraction already
holds. New endpoints that the judge never typed cannot be conjured, and the report says so.

Class B: no LLM call, no network. Run from graph_analysis/:

    python -u experiment_review_omission_chain_impact.py --judge-reports <dir>

    git archive origin/anthropic_judge_test extraction_validator/extend_try_1 \\
        | tar -x -C /tmp/judge
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl"
)
OUT = ROOT / "phase2_results/experiment_review_omission_chain_impact_report.json"
SKIP_FILES = {"summary.json", "errors.json"}

# Reuse the validated name matcher rather than writing a second one that could drift.
_spec = importlib.util.spec_from_file_location(
    "_oicl", ROOT / "experiment_review_omission_is_chain_level.py"
)
_oicl = importlib.util.module_from_spec(_spec)
sys.modules["_oicl"] = _oicl
_spec.loader.exec_module(_oicl)
norm, match, is_endpoint = _oicl.norm, _oicl.match, _oicl.is_endpoint


def reachable_pairs(adj: dict, risks: set, interventions: set) -> set:
    """Every (risk, intervention) pair joined by any path. Undirected, gate-free."""
    pairs = set()
    for r in risks:
        seen, q = {r}, deque([r])
        while q:
            n = q.popleft()
            for m in adj.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    q.append(m)
        for i in interventions & seen:
            pairs.add((r, i))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-reports", required=True)
    a = ap.parse_args()

    jdir = Path(a.judge_reports)
    if not jdir.is_dir():
        raise SystemExit(
            f"FATAL: judge report directory not found: {jdir}\n"
            "  git archive origin/anthropic_judge_test extraction_validator/extend_try_1"
            " | tar -x -C /tmp/judge\n"
            "  this script does NOT fall back to a cached copy."
        )
    for p, how in (
        (SLIM, "experiment_review_prep_slim_nodes.py"),
        (EDGES, "phase2_step1_loadandparse.py against the FalkorDB dump"),
    ):
        if not p.exists():
            raise SystemExit(f"FATAL: {p} not found. Produce it with {how}.")

    slim = pickle.load(SLIM.open("rb"))
    nodes_by_url: dict[str, list] = defaultdict(list)
    for nid, rec in slim.items():
        if rec.get("url"):
            nodes_by_url[rec["url"]].append(nid)

    adj_all: dict[int, set] = defaultdict(set)
    for e in pickle.load(EDGES.open("rb")):
        if e.get("type") != "EDGE":
            continue
        adj_all[e["source"]].add(e["target"])
        adj_all[e["target"]].add(e["source"])

    tot = Counter()
    per_paper, papers_with_new = [], []

    for p in sorted(jdir.glob("*.json")):
        if p.name in SKIP_FILES:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        url = r.get("url")
        nids = nodes_by_url.get(url or "")
        if not nids:
            continue

        pool = {
            norm(slim[n]["name"]): {**slim[n], "_nid": n}
            for n in nids
            if slim[n].get("name")
        }
        risks = {n for n in nids if is_endpoint(slim[n]) == "risk"}
        ivs = {n for n in nids if is_endpoint(slim[n]) == "intervention"}
        inside = set(nids)
        adj = {n: (adj_all.get(n, set()) & inside) for n in nids}

        before = reachable_pairs(adj, risks, ivs)

        adj2 = {n: set(v) for n, v in adj.items()}
        ph = 0
        added_edges = 0
        for row in ((r.get("validation_report") or {}).get("coverage") or {}).get(
            "expected_edges_from_source"
        ) or []:
            if (row.get("status") or "").strip().lower() != "missing":
                continue
            s = (row.get("expected_source_node_name") or [None])[0]
            t = (row.get("expected_target_node_name") or [None])[0]
            if not s or not t:
                continue
            ends = []
            for name in (s, t):
                rec = match(name, pool)
                if rec is not None:
                    ends.append(rec["_nid"])
                else:
                    ph += 1
                    key = f"PH::{p.name}::{ph}"
                    adj2.setdefault(key, set())
                    ends.append(key)
            adj2.setdefault(ends[0], set()).add(ends[1])
            adj2.setdefault(ends[1], set()).add(ends[0])
            added_edges += 1

        after = reachable_pairs(adj2, risks, ivs)
        new = after - before

        # HEADROOM, and it is the denominator this result must be read against. Most
        # audited papers extract as ONE connected component, so every risk in them already
        # reaches every intervention and no added edge can possibly create a pair. Quoting
        # new pairs against all pairs would therefore report a ceiling effect as a finding.
        # The pairs that were structurally available to gain are (risks x interventions)
        # minus the ones already reachable, and that is what the delta is measured on.
        possible = len(risks) * len(ivs)
        tot["pairs_possible"] += possible
        tot["headroom"] += possible - len(before)
        if possible and len(before) < possible:
            tot["papers_with_headroom"] += 1

        tot["papers"] += 1
        tot["missing_edges_granted"] += added_edges
        tot["placeholders_invented"] += ph
        tot["pairs_before"] += len(before)
        tot["pairs_after"] += len(after)
        tot["pairs_new"] += len(new)
        if new:
            tot["papers_with_a_new_pair"] += 1
            papers_with_new.append({"paper": p.name, "new_pairs": len(new)})
        per_paper.append(
            {
                "paper": p.name,
                "risks": len(risks),
                "interventions": len(ivs),
                "pairs_before": len(before),
                "pairs_after": len(after),
                "new_pairs": len(new),
                "missing_edges_granted": added_edges,
            }
        )

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    report = {
        "experiment": "chain-level impact of granting the judge every omission it flagged",
        "question": (
            "A chain is a connected risk-to-intervention argument. If every relationship "
            "the judge marked missing were added, how many risk-intervention pairs would "
            "become reachable that were not reachable before?"
        ),
        "method": (
            "Per audited paper: released subgraph, undirected, EDGE-type only. Add every "
            "`missing` coverage row as an edge, inventing a placeholder node for any "
            "endpoint the extraction lacks. Compare reachable (risk, intervention) pairs "
            "before and after. Reachability ignores the first-hop rule, the hop floor and "
            "ceiling and both gates, all of which can only remove pairs -- so new_pairs is "
            "an UPPER BOUND on new chains."
        ),
        "papers": tot["papers"],
        "missing_edges_granted": tot["missing_edges_granted"],
        "placeholder_nodes_invented": tot["placeholders_invented"],
        "risk_intervention_pairs": {
            "possible": tot["pairs_possible"],
            "before": tot["pairs_before"],
            "after": tot["pairs_after"],
            "new": tot["pairs_new"],
            "new_as_pct_of_before": pct(tot["pairs_new"], tot["pairs_before"]),
        },
        "headroom": {
            "why": (
                "Most audited papers extract as ONE connected component, so every risk in "
                "them already reaches every intervention and no added edge CAN create a "
                "pair. Reading new pairs against `before` would report that ceiling as a "
                "finding. Headroom is the number of pairs structurally available to gain."
            ),
            "pairs_unreachable_before": tot["headroom"],
            "papers_with_any_headroom": tot["papers_with_headroom"],
            "saturation_pct": pct(tot["pairs_before"], tot["pairs_possible"]),
            "new_as_pct_of_headroom": pct(tot["pairs_new"], tot["headroom"]),
            "READ_THIS_ONE": (
                "new_as_pct_of_headroom is the honest figure. The raw share of `before` is "
                "ceiling-inflated and must not be quoted on its own."
            ),
        },
        "papers_with_at_least_one_new_pair": tot["papers_with_a_new_pair"],
        "papers_with_a_new_pair_pct": pct(tot["papers_with_a_new_pair"], tot["papers"]),
        "papers_with_new_detail": sorted(
            papers_with_new, key=lambda x: -x["new_pairs"]
        )[:20],
        "WHAT_THIS_CANNOT_SHOW": (
            "A missing relationship whose absent endpoint IS a risk or an intervention "
            "would add an endpoint, not just a bridge, and this test cannot see it: 98.7% "
            "of the absent names carry no category marker, so placeholders are typed "
            "`unknown` and can only bridge. Settling that needs a judge re-run with a typed "
            "add-node slot, or the human packet's chain_recall_missed field. Read this "
            "result as: the flagged omissions do not RE-WIRE the chain set among the "
            "endpoints the extraction already found."
        ),
        "LIMITS": (
            "Every input row is the judge's unadjudicated opinion, granted here without "
            "checking. Node-name resolution is the fuzzy matcher validated in "
            "experiment_review_omission_is_chain_level.py (95.1/43.0/8.5% resolution on "
            "covered/partial/missing rows)."
        ),
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"papers: {tot['papers']}")
    print(f"missing relationships granted: {tot['missing_edges_granted']}")
    print(f"placeholder nodes invented:    {tot['placeholders_invented']}")
    print(
        f"risk-intervention pairs  before {tot['pairs_before']}  after {tot['pairs_after']}"
    )
    print(
        f"  saturation before: {pct(tot['pairs_before'], tot['pairs_possible'])}% "
        f"({tot['pairs_possible']} possible), headroom {tot['headroom']} pairs "
        f"in {tot['papers_with_headroom']} papers"
    )
    print(
        f"  NEW pairs: {tot['pairs_new']}  = "
        f"{pct(tot['pairs_new'], tot['headroom'])}% OF HEADROOM "
        f"(the honest denominator), in {tot['papers_with_a_new_pair']} of {tot['papers']} papers"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
