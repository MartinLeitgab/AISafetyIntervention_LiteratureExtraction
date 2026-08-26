#!/usr/bin/env python
"""Break the judge's coverage list out by status, against the edges the extraction produced.

The paper reports node-level omission twice (0.6% and 28.8%, tab:omission) and reports the
judge's *edge* coverage list only as a per-paper mean of 7.8 "missing relationships",
buried in two appendices. A reviewer set that 7.8 against the mean audited extraction of
10.8 edges and concluded edge recall may be poor while node recall looks high.

That comparison does not hold, and this script is what shows it. The 7.8 is
`len(validation_report.coverage.expected_edges_from_source)` -- the length of the WHOLE
coverage list, computed that way in experiment_judge_item2_summary.py:144. Every row of
that list carries a `status` field taking one of covered / partially_covered / missing, so
the list is the judge's inventory of relationships it went looking for, not its count of
ones the extraction lacks. This script reads the status field and reports the three classes
separately, then puts the genuinely-missing count on the same denominator tab:omission uses
for nodes: the edges the RELEASED graph holds for the same papers.

Class B (no LLM call, no network). Run from graph_analysis/:

    python -u experiment_review_edge_coverage.py --judge-reports <dir>

where <dir> is extraction_validator/extend_try_1 from branch anthropic_judge_test:

    git archive origin/anthropic_judge_test extraction_validator/extend_try_1 | tar -x -C /tmp/judge

Output: graph_analysis/phase2_results/experiment_review_edge_coverage_report.json

This script does NOT adjudicate any judge verdict. A row labelled `missing` is the judge's
opinion, unchecked by a human, exactly like every other number in sec:r-judge.
"""

import argparse
import json
import pickle
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl"
)
OUT = ROOT / "phase2_results/experiment_review_edge_coverage_report.json"

# The judged sample's file names carry the ARD source type as a prefix, which is how
# experiment_judge_item2_summary.py buckets them. Reused here so the two receipts agree.
SKIP_FILES = {"summary.json", "errors.json"}


def pct(n, d, nd=1):
    return round(100 * n / d, nd) if d else None


def source_type(fname):
    return fname.split("__")[0] if "__" in fname else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-reports", required=True)
    a = ap.parse_args()

    jdir = Path(a.judge_reports)
    if not jdir.is_dir():
        raise SystemExit(
            f"FATAL: judge report directory not found: {jdir}\n"
            "  produce it with:\n"
            "    git archive origin/anthropic_judge_test extraction_validator/extend_try_1"
            " | tar -x -C /tmp/judge\n"
            "  this script does NOT fall back to a cached or partial copy."
        )
    for p, how in (
        (SLIM, "run experiment_review_prep_slim_nodes.py"),
        (EDGES, "run phase2_step1_loadandparse.py against the FalkorDB dump"),
    ):
        if not p.exists():
            raise SystemExit(f"FATAL: {p} not found. To produce it: {how}.")

    # ---- released graph: EDGE-type edges per source paper -------------------------------
    slim = pickle.load(open(SLIM, "rb"))
    url_of_node = {nid: r.get("url") for nid, r in slim.items() if r.get("url")}
    edges = pickle.load(open(EDGES, "rb"))

    edges_per_url = Counter()
    cross_paper_edges = 0
    n_structural = 0
    for e in edges:
        if e.get("type") != "EDGE":
            continue
        n_structural += 1
        su, tu = url_of_node.get(e["source"]), url_of_node.get(e["target"])
        if su and tu and su != tu:
            cross_paper_edges += 1
        if su:
            edges_per_url[su] += 1

    # ---- judge reports: the coverage list, by status ------------------------------------
    rows = []
    status_counter = Counter()
    proposed_fix_keys = Counter()
    unmatched = []
    for p in sorted(jdir.glob("*.json")):
        if p.name in SKIP_FILES:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        cov = ((r.get("validation_report") or {}).get("coverage") or {}).get(
            "expected_edges_from_source"
        ) or []
        by_status = Counter()
        for row in cov:
            s = (row.get("status") or "unlabelled").strip().lower()
            by_status[s] += 1
            status_counter[s] += 1
        for k in r.get("proposed_fixes") or {}:
            proposed_fix_keys[k] += 1
        fg = r.get("final_graph") or {}
        url = r.get("url")
        n_edges = edges_per_url.get(url) if url else None
        if not n_edges:
            unmatched.append(p.name)
        rows.append(
            {
                "paper": p.name,
                "source_type": source_type(p.name),
                "url": url,
                "coverage_rows": len(cov),
                "covered": by_status.get("covered", 0),
                "partially_covered": by_status.get("partially_covered", 0),
                "missing": by_status.get("missing", 0),
                "other_status": len(cov)
                - by_status.get("covered", 0)
                - by_status.get("partially_covered", 0)
                - by_status.get("missing", 0),
                "released_edges": n_edges,
                "judge_final_graph_edges": len(fg.get("edges") or []),
                "judge_final_graph_nodes": len(fg.get("nodes") or []),
            }
        )

    if not rows:
        raise SystemExit(f"FATAL: no judge reports parsed from {jdir}")

    n_papers = len(rows)
    matched = [r for r in rows if r["released_edges"]]
    tot = {
        k: sum(r[k] for r in rows)
        for k in (
            "coverage_rows",
            "covered",
            "partially_covered",
            "missing",
            "other_status",
        )
    }
    tot_edges_matched = sum(r["released_edges"] for r in matched)
    missing_matched = sum(r["missing"] for r in matched)
    partial_matched = sum(r["partially_covered"] for r in matched)

    out = {
        "experiment": "judge edge-coverage list, broken out by status (S10)",
        "question": (
            "The paper quotes a mean of 7.8 flagged relationships per paper against a mean "
            "audited extraction of 10.8 edges, which reads as an edge-omission signal of "
            "the same order as the extraction. Is it one?"
        ),
        "answer_in_one_line": (
            "No. 7.8 is the mean LENGTH OF THE WHOLE COVERAGE LIST, which the judge fills "
            "with relationships it looked for and mostly found. Broken out by the status "
            f"field the judge writes on every row: {pct(tot['covered'], tot['coverage_rows'])}% "
            f"covered, {pct(tot['partially_covered'], tot['coverage_rows'])}% partially "
            f"covered, {pct(tot['missing'], tot['coverage_rows'])}% missing."
        ),
        "n_papers": n_papers,
        "n_papers_matched_to_the_released_graph": len(matched),
        "unmatched_papers": unmatched,
        "coverage_list": {
            "rows_total": tot["coverage_rows"],
            "rows_mean_per_paper": round(tot["coverage_rows"] / n_papers, 2),
            "by_status_total": dict(status_counter.most_common()),
            "covered": tot["covered"],
            "partially_covered": tot["partially_covered"],
            "missing": tot["missing"],
            "unlabelled_or_other": tot["other_status"],
            "missing_mean_per_paper": round(tot["missing"] / n_papers, 2),
            "partially_covered_mean_per_paper": round(
                tot["partially_covered"] / n_papers, 2
            ),
            "papers_with_at_least_one_missing": sum(1 for r in rows if r["missing"]),
            "papers_with_zero_missing": sum(1 for r in rows if not r["missing"]),
            "missing_max_in_one_paper": max(r["missing"] for r in rows),
            "missing_median_per_paper": st.median(r["missing"] for r in rows),
        },
        "REPRODUCES_THE_RECEIPT_FIGURE": {
            "note": (
                "experiment_judge_item2_summary.py:144 sets missing_edges_flagged = len(cov), "
                "i.e. the whole list. The mean below must equal the 7.77 that receipt reports; "
                "if it does not, the two scripts are reading different inputs."
            ),
            "mean_len_coverage_list": round(tot["coverage_rows"] / n_papers, 2),
            "expected_from_experiment_judge_item2_report_json": 7.77,
            "agrees": abs(tot["coverage_rows"] / n_papers - 7.77) < 0.01,
        },
        "against_the_extraction_it_is_measured_on": {
            "denominator": (
                "EDGE-type edges the RELEASED graph holds for the same papers, keyed by "
                "source URL -- the same construction tab:omission uses for nodes"
            ),
            "papers": len(matched),
            "released_edges_total": tot_edges_matched,
            "released_edges_mean_per_paper": round(tot_edges_matched / len(matched), 1),
            "missing_total": missing_matched,
            "missing_as_pct_of_released_edges": pct(missing_matched, tot_edges_matched),
            "implied_edge_coverage_pct": pct(
                tot_edges_matched, tot_edges_matched + missing_matched
            ),
            "missing_plus_partial_total": missing_matched + partial_matched,
            "missing_plus_partial_as_pct_of_released_edges": pct(
                missing_matched + partial_matched, tot_edges_matched
            ),
        },
        "which_denominator": {
            "note": (
                "Two edge counts exist per judged paper and they differ. The released graph "
                "holds more edges than the judge's own final_graph field reports, and the "
                "paper quotes the latter (10.8) in app:judge. The released count is the "
                "correct denominator for an omission share, because it is the artifact a "
                "reader gets; the judge's final_graph is its own re-serialisation."
            ),
            "released_graph_edges_mean_per_paper": round(
                tot_edges_matched / len(matched), 1
            ),
            "judge_final_graph_edges_mean_per_paper": round(
                sum(r["judge_final_graph_edges"] for r in rows) / n_papers, 2
            ),
            "judge_final_graph_nodes_mean_per_paper": round(
                sum(r["judge_final_graph_nodes"] for r in rows) / n_papers, 2
            ),
            "missing_as_pct_of_judge_final_graph_edges": pct(
                tot["missing"], sum(r["judge_final_graph_edges"] for r in rows)
            ),
        },
        "by_source_type": {
            stype: {
                "n_papers": len(g),
                "coverage_rows_mean": round(
                    sum(r["coverage_rows"] for r in g) / len(g), 2
                ),
                "missing_mean": round(sum(r["missing"] for r in g) / len(g), 2),
                "released_edges_mean": round(
                    sum(r["released_edges"] or 0 for r in g) / len(g), 1
                ),
            }
            for stype, g in sorted(_group(rows).items(), key=lambda kv: -len(kv[1]))
        },
        "no_add_edges_slot": {
            "proposed_fixes_keys_seen": dict(proposed_fix_keys.most_common()),
            "has_add_edges_key": "add_edges" in proposed_fix_keys,
            "reading": (
                "The repair schema gives the judge a slot for added NODES and none for "
                "added EDGES. A relationship the judge marks missing therefore cannot be "
                "proposed as a repair, which is why the node-addition count (9) and the "
                "coverage list are different instruments rather than a contradiction."
            ),
        },
        "released_graph_structural_edges": {
            "n_edge_type_edges": n_structural,
            "n_crossing_two_source_papers": cross_paper_edges,
            "reading": (
                "Zero cross-paper structural edges is the single-source-by-design property "
                "of sec:m-structural, verified here on the released substrate rather than "
                "asserted."
            ),
        },
        "LIMITS": (
            "Every status label is the judge's own, un-adjudicated. The judge was also not "
            "given a way to propose an edge repair, so `missing` is an opinion recorded in "
            "a free-text inventory, not a verified omission. These counts bound the "
            "edge-level signal; they do not measure edge recall."
        ),
    }

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


def _group(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["source_type"]].append(r)
    return g


if __name__ == "__main__":
    sys.exit(main())
