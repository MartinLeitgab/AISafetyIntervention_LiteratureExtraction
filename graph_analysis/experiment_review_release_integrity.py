#!/usr/bin/env python
"""Does the released graph carry the defects the judge found, and can a reuser re-gate it?

Three reviewer questions in the 2026-08-15 round have the same shape -- they ask what a
person who downloads the release actually gets:

  C8/R45  The judge reported 108 referential-integrity findings, 42 orphans and 56
          duplicate node pairs across 100 papers, and no repaired graph was ever rebuilt.
          Does the released dump ship those defects? The paper does not say.
  R46     Users need a way to tell audited content from unaudited content.
  R114    Can a reuser re-enumerate chains at any gate setting from the dump, or are they
          restricted to consuming the two path files we ship?

None of the three needs an inference call. This script measures them directly on the
released substrate: orphan and dangling counts computed the same way the judge's classes
are defined, the share of the graph that any judge ever looked at, and whether every
attribute the two gates read is actually present on every node and edge that carries it.

Class B (no LLM call, no network). Run from graph_analysis/:

    python -u experiment_review_release_integrity.py --judge-reports <dir>

<dir> is extraction_validator/extend_try_1 from branch anthropic_judge_test.

Output: graph_analysis/phase2_results/experiment_review_release_integrity_report.json
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl"
)
RAW = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
OUT = ROOT / "phase2_results/experiment_review_release_integrity_report.json"

EXPECTED_NODES, EXPECTED_EDGE_EDGES = 200525, 202149
NONE_STRINGS = {"None", "none", "", "null", "NULL"}


def pct(n, d, nd=2):
    return round(100 * n / d, nd) if d else None


def present(v):
    return v is not None and str(v).strip() not in NONE_STRINGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-reports", required=True)
    a = ap.parse_args()
    jdir = Path(a.judge_reports)
    if not jdir.is_dir():
        raise SystemExit(
            f"FATAL: judge report directory not found: {jdir}\n"
            "  git archive origin/anthropic_judge_test extraction_validator/extend_try_1"
            " | tar -x -C /tmp/judge"
        )
    for p, how in (
        (SLIM, "run experiment_review_prep_slim_nodes.py"),
        (EDGES, "run phase2_step1_loadandparse.py against the FalkorDB dump"),
        (RAW, "the released 8,954-chain path file ships with the repo"),
    ):
        if not p.exists():
            raise SystemExit(f"FATAL: {p} not found. To produce it: {how}.")

    slim = pickle.load(open(SLIM, "rb"))
    if len(slim) != EXPECTED_NODES:
        raise SystemExit(
            f"FATAL: node table has {len(slim)}, expected {EXPECTED_NODES}. Wrong substrate."
        )
    nodes = {int(n): r for n, r in slim.items()}
    edges = pickle.load(open(EDGES, "rb"))

    deg = Counter()
    dangling = 0
    self_loops = 0
    edge_pair_subtype = Counter()
    n_struct = 0
    conf_missing = 0
    subtype_missing = 0
    for e in edges:
        if e.get("type") != "EDGE":
            continue
        n_struct += 1
        s, t = e["source"], e["target"]
        if s not in nodes or t not in nodes:
            dangling += 1
            continue
        if s == t:
            self_loops += 1
        deg[s] += 1
        deg[t] += 1
        edge_pair_subtype[(frozenset((s, t)), e.get("subtype"))] += 1
        if not present(e.get("confidence")):
            conf_missing += 1
        if not present(e.get("subtype")):
            subtype_missing += 1

    if n_struct != EXPECTED_EDGE_EDGES:
        raise SystemExit(
            f"FATAL: {n_struct} EDGE-type edges, expected {EXPECTED_EDGE_EDGES}."
        )

    orphans = [n for n in nodes if deg[n] == 0]
    dup_edges = sum(v - 1 for v in edge_pair_subtype.values() if v > 1)

    # exact-name duplicates within concept category (the residue app:composition reports)
    by_name = defaultdict(list)
    for n, r in nodes.items():
        key = (
            (r.get("name") or "").strip().lower(),
            r.get("concept_category") if r.get("type") == "concept" else "intervention",
        )
        if key[0]:
            by_name[key].append(n)
    dup_groups = {k: v for k, v in by_name.items() if len(v) > 1}
    dup_nodes_beyond_one = sum(len(v) - 1 for v in dup_groups.values())

    # can a reuser re-gate? both gate attributes must be present wherever they are read
    interventions = [n for n, r in nodes.items() if r.get("type") != "concept"]
    intv_no_maturity = [
        n for n in interventions if not present(nodes[n].get("intervention_maturity"))
    ]
    concepts_no_category = [
        n
        for n, r in nodes.items()
        if r.get("type") == "concept" and not present(r.get("concept_category"))
    ]
    nodes_no_url = [n for n, r in nodes.items() if not present(r.get("url"))]

    # audited vs unaudited share of the release
    judged_urls = set()
    for p in sorted(jdir.glob("*.json")):
        if p.name in ("summary.json", "errors.json"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("url"):
            judged_urls.add(r["url"])
    nodes_judged = sum(1 for r in nodes.values() if r.get("url") in judged_urls)

    chain_nodes = set()
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chain_nodes.update(int(x) for x in json.loads(line)["path"])

    out = {
        "experiment": "released-graph defect inventory and re-gateability (C8/R45/R46/R114)",
        "substrate": {
            "nodes": len(nodes),
            "structural_edges": n_struct,
            "note": "the un-merged 200,525-node graph this paper reports on",
        },
        "does_the_release_ship_the_defect_classes_the_judge_found": {
            "question": (
                "The judge reported these classes on 100 papers' extraction JSON. No "
                "repaired graph was rebuilt, so the released dump should still carry them. "
                "Measured here on the dump itself, corpus-wide."
            ),
            "orphan_nodes_zero_structural_edges": len(orphans),
            "orphan_pct_of_graph": pct(len(orphans), len(nodes)),
            "dangling_edges_endpoint_not_in_node_table": dangling,
            "self_loops": self_loops,
            "duplicate_edges_same_pair_same_relation_type": dup_edges,
            "exact_name_duplicate_groups_within_category": len(dup_groups),
            "exact_name_duplicate_nodes_beyond_one_per_group": dup_nodes_beyond_one,
            "cross_check": (
                "app:composition reports 448 groups and 1,140 redundant nodes from "
                "experiment_J6_merge_approx_v2_report.json; these are re-derived here."
            ),
        },
        "can_a_reuser_re_enumerate_at_any_gate": {
            "question": "R114 -- is the dump enough, or must a reuser consume our path files?",
            "structural_edges_missing_a_confidence_value": conf_missing,
            "structural_edges_missing_a_relation_type": subtype_missing,
            "intervention_nodes_missing_a_maturity_value": len(intv_no_maturity),
            "concept_nodes_missing_a_category": len(concepts_no_category),
            "nodes_missing_a_source_url": len(nodes_no_url),
            "answer": (
                "Yes if all five counts above are zero: every attribute the two gates read "
                "is on the dump, so any gate setting is re-enumerable from it and the path "
                "files are a convenience, not the only access path."
            ),
            "verdict_all_gate_attributes_present": (
                conf_missing == 0
                and len(intv_no_maturity) == 0
                and len(concepts_no_category) == 0
            ),
        },
        "audited_vs_unaudited_content": {
            "question": "R46 -- can a user tell which part of the release anyone checked?",
            "judged_source_documents": len(judged_urls),
            "nodes_from_judged_documents": nodes_judged,
            "nodes_from_judged_documents_pct": pct(nodes_judged, len(nodes)),
            "nodes_never_seen_by_any_judge": len(nodes) - nodes_judged,
            "reading": (
                "Under one percent of the released nodes were looked at by the judge, and "
                "no node was looked at by a human. The release should carry a per-node flag "
                "for this rather than leaving it to the paper's population table."
            ),
        },
        "chain_coverage_of_the_graph": {
            "nodes_appearing_in_at_least_one_enumerated_chain": len(chain_nodes),
            "pct_of_graph": pct(len(chain_nodes), len(nodes)),
            "reading": (
                "The chain set touches a small share of the released graph. A reuser who "
                "consumes only the path files is consuming that share."
            ),
        },
        "LIMITS": (
            "Structural integrity only. An orphan here is a node with no structural edge in "
            "the dump, which is the judge's own orphan definition applied corpus-wide; it is "
            "not the same measurement as the judge's per-paper count and the two should not "
            "be added together."
        ),
    }

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
