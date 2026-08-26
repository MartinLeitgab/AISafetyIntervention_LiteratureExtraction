#!/usr/bin/env python3
"""Do the released chains obey the stage order they are described in, and are their
edges walked with or against the orientation the extractor stored?

Answers the 2026-08-17 external-review objection (reviewers GC and GW, both bars):
the enumerator traverses edges undirected and the constraint list in app:cuts does NOT
require the concept categories to progress monotonically along the canonical chain, so a
"risk-to-intervention mechanism chain" might be a connected node sequence rather than a
directed argument. Nothing in the manuscript answers that today.

Class B: no LLM call, no network. Reads the released path files, the slim node attribute
table and the structural edge list, and writes one receipt.

Definitions, all reported separately because they answer different questions:

  monotonic (non-decreasing)  the stage rank never decreases along the chain. Repeats of
                              one stage are allowed, which the schema explicitly permits.
  strictly ordered            no repeats either: rank increases at every hop.
  inversions                  hops where the rank decreases, counted and located.
  forward hop                 a structural edge exists in the stored orientation
                              source -> target matching the traversal direction.
  backward hop                the edge exists only as target -> source, i.e. the
                              traversal walks it against the orientation the extractor
                              wrote.
  against-majority hop        a backward hop whose relation type, measured over the whole
                              graph, predominantly runs from lower to higher stage rank.
                              This is the data-derived version of the reviewers' "after
                              canonicalizing relation direction" request; no orientation is
                              assumed a priori.

Usage
-----
    cd graph_analysis
    python -u experiment_review_chain_order_semantics.py
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATHS_RAW = HERE / "phase1_rawpathsfiles" / "paths_hopwise_v4_edge_only.jsonl"
PATHS_DEDUPED = (
    HERE / "phase1_rawpathsfiles" / "paths_hopwise_v4_edge_only_deduped.jsonl"
)
NODE_ATTRS = HERE / "phase2_results" / "node_attrs_slim.pkl"
EDGE_DATA = (
    HERE
    / "phase2_results"
    / "step1_load_and_parse_umapwithoutlocalsatellites"
    / "graph_edge_data.pkl"
)
OUT = HERE / "phase2_results" / "experiment_review_chain_order_semantics_report.json"

# The logical chain, in the order CLAUDE.md fixes and the paper never reorders.
STAGE_RANK = {
    "risk": 0,
    "problem analysis": 1,
    "theoretical insight": 2,
    "design rationale": 3,
    "implementation mechanism": 4,
    "validation evidence": 5,
}
INTERVENTION_RANK = 6


def die(what: str, expected: Path, produced_by: str) -> None:
    raise SystemExit(
        f"FATAL: {what} not found.\n"
        f"  expected: {expected}\n"
        f"  produced by: {produced_by}\n"
        f"  This script does NOT rebuild it and does NOT fall back to another source. It\n"
        f"  measures the RELEASED chain set only; there is no substitute input."
    )


def load_paths(path: Path) -> list[list[int]]:
    if not path.is_file():
        die("path file", path, "phase2_step4_F2v4_hopwise_falkordb.py")
    out: list[list[int]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # The released files key the node sequence as "path"; "nodes" is accepted only
            # so a hand-made fixture does not need re-keying.
            if isinstance(rec, dict):
                nodes = rec.get("path") or rec.get("nodes")
            else:
                nodes = rec
            if not isinstance(nodes, list) or not nodes:
                raise SystemExit(
                    f"FATAL: {path.name} carries a record with no node list: {rec!r:.200}"
                )
            out.append([int(n) for n in nodes])
    return out


def rank_of(node_id: int, attrs: dict) -> int | None:
    a = attrs.get(node_id)
    if a is None:
        return None
    if a.get("type") == "intervention":
        return INTERVENTION_RANK
    return STAGE_RANK.get(a.get("concept_category"))


def analyse(
    paths: list[list[int]],
    attrs: dict,
    fwd: dict,
    rev: dict,
    majority: dict[str, str] | None = None,
) -> dict:
    n = len(paths)
    monotonic = strict = 0
    all_forward = any_backward = 0
    unresolved_hops = 0
    hops_total = hops_backward = 0
    inversion_pairs: Counter = Counter()
    backward_types: Counter = Counter()
    forward_types: Counter = Counter()
    inversions_per_chain: Counter = Counter()
    backward_per_chain: Counter = Counter()
    unranked_nodes = 0
    against_majority = 0
    majority = majority or {}

    for nodes in paths:
        ranks = [rank_of(x, attrs) for x in nodes]
        unranked_nodes += sum(1 for r in ranks if r is None)
        known = [r for r in ranks if r is not None]
        inv = 0
        for a, b in zip(known, known[1:]):
            if b < a:
                inv += 1
                inversion_pairs[f"{a}->{b}"] += 1
        inversions_per_chain[inv] += 1
        if inv == 0:
            monotonic += 1
            if all(b > a for a, b in zip(known, known[1:])):
                strict += 1

        back = 0
        for u, v in zip(nodes, nodes[1:]):
            hops_total += 1
            f = fwd.get((u, v))
            r = rev.get((u, v))
            if f is not None:
                forward_types[f] += 1
            elif r is not None:
                back += 1
                hops_backward += 1
                backward_types[r] += 1
                # A backward hop matters most where the relation type is otherwise
                # consistent: if the type predominantly ascends the stage rank across the
                # whole graph, walking one instance of it backward is a traversal against
                # the orientation the extractor itself uses for that type.
                if majority.get(r) == "up":
                    against_majority += 1
            else:
                unresolved_hops += 1
        backward_per_chain[back] += 1
        if back == 0:
            all_forward += 1
        else:
            any_backward += 1

    return {
        "n_chains": n,
        "n_hops": hops_total,
        "stage_order": {
            "monotonic_non_decreasing": monotonic,
            "monotonic_pct": round(100.0 * monotonic / n, 1) if n else None,
            "strictly_increasing": strict,
            "strictly_increasing_pct": round(100.0 * strict / n, 1) if n else None,
            "chains_with_an_inversion": n - monotonic,
            "chains_with_an_inversion_pct": (
                round(100.0 * (n - monotonic) / n, 1) if n else None
            ),
            "inversions_per_chain_hist": dict(sorted(inversions_per_chain.items())),
            "most_common_inversions_rank_pairs": inversion_pairs.most_common(10),
        },
        "edge_direction": {
            "chains_all_hops_forward": all_forward,
            "chains_all_hops_forward_pct": (
                round(100.0 * all_forward / n, 1) if n else None
            ),
            "chains_with_a_backward_hop": any_backward,
            "chains_with_a_backward_hop_pct": (
                round(100.0 * any_backward / n, 1) if n else None
            ),
            "hops_backward": hops_backward,
            "hops_backward_pct": (
                round(100.0 * hops_backward / hops_total, 1) if hops_total else None
            ),
            "backward_hops_per_chain_hist": dict(sorted(backward_per_chain.items())),
            "relation_types_walked_forward": forward_types.most_common(),
            "relation_types_walked_backward": backward_types.most_common(),
            "hops_against_type_majority_orientation": against_majority,
            "hops_against_type_majority_orientation_pct": (
                round(100.0 * against_majority / hops_total, 1) if hops_total else None
            ),
            "unresolved_hops": unresolved_hops,
        },
        "unranked_nodes_encountered": unranked_nodes,
    }


def main() -> int:
    if not NODE_ATTRS.is_file():
        die("slim node attributes", NODE_ATTRS, "experiment_review_prep_slim_nodes.py")
    if not EDGE_DATA.is_file():
        die("structural edge list", EDGE_DATA, "phase2_step1_loadandparse.py")

    print("loading node attributes ...", flush=True)
    attrs = pickle.load(NODE_ATTRS.open("rb"))
    print(f"  {len(attrs):,} nodes", flush=True)

    print("loading edges ...", flush=True)
    edges = pickle.load(EDGE_DATA.open("rb"))
    fwd: dict[tuple[int, int], str] = {}
    rev: dict[tuple[int, int], str] = {}
    type_orientation: dict[str, Counter] = defaultdict(Counter)
    n_struct = 0
    for e in edges:
        if e.get("type") != "EDGE":
            continue
        n_struct += 1
        s, t = int(e["source"]), int(e["target"])
        sub = e.get("subtype") or "unknown"
        fwd.setdefault((s, t), sub)
        rev.setdefault((t, s), sub)
        rs, rt = rank_of(s, attrs), rank_of(t, attrs)
        if rs is not None and rt is not None and rs != rt:
            type_orientation[sub]["up" if rt > rs else "down"] += 1
    print(f"  {n_struct:,} structural edges", flush=True)

    majority_up = {
        k: {
            "up": v["up"],
            "down": v["down"],
            "majority": "up" if v["up"] >= v["down"] else "down",
            "majority_share_pct": round(
                100.0 * max(v["up"], v["down"]) / (v["up"] + v["down"]), 1
            ),
        }
        for k, v in sorted(type_orientation.items())
    }

    report = {
        "study": "chain order and edge orientation of the released chain set",
        "answers": (
            "2026-08-17 external review, reviewers GC and GW: the enumerator is undirected "
            "and app:cuts imposes no monotonic stage-order constraint, so 'chain' may "
            "describe a connected node sequence rather than a directed argument."
        ),
        "inputs": {
            "raw_paths": str(PATHS_RAW.name),
            "deduped_paths": str(PATHS_DEDUPED.name),
            "node_attrs": str(NODE_ATTRS.name),
            "edges": str(EDGE_DATA.name),
            "n_structural_edges": n_struct,
        },
        "relation_type_orientation_over_the_whole_graph": majority_up,
    }

    for label, path in (("deduped_2772", PATHS_DEDUPED), ("raw_8954", PATHS_RAW)):
        print(f"analysing {label} ...", flush=True)
        paths = load_paths(path)
        report[label] = analyse(
            paths, attrs, fwd, rev, {k: v["majority"] for k, v in majority_up.items()}
        )
        s = report[label]["stage_order"]
        d = report[label]["edge_direction"]
        print(
            f"  {report[label]['n_chains']:,} chains | monotonic "
            f"{s['monotonic_pct']}% | all hops forward {d['chains_all_hops_forward_pct']}%"
            f" | backward hops {d['hops_backward_pct']}%",
            flush=True,
        )

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)

    d = report["deduped_2772"]
    print("\n=== headline, reporting unit (2,772 chains) ===")
    print(f"  monotonic stage order      : {d['stage_order']['monotonic_pct']}%")
    print(
        f"  strictly increasing        : {d['stage_order']['strictly_increasing_pct']}%"
    )
    print(
        f"  chains with every hop stored forward: "
        f"{d['edge_direction']['chains_all_hops_forward_pct']}%"
    )
    print(f"  hops walked backward       : {d['edge_direction']['hops_backward_pct']}%")
    if d["edge_direction"]["unresolved_hops"]:
        print(
            f"  WARNING unresolved hops    : {d['edge_direction']['unresolved_hops']}"
            " (no structural edge found in either orientation)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
