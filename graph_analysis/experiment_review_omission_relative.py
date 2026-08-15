#!/usr/bin/env python
"""Put the two omission counts on a common denominator.

"Approximately 5 missed concepts per paper" is uninterpretable without the size of the
graph they are missing from. This script computes, for exactly the papers each measurement
covers, how many nodes the extraction actually produced, and expresses both omission
measurements as a fraction of that.

  - the judge's proposed additions: 9 nodes over the 100 audited papers
  - the Opus grader's missed concepts: 216 over the 43 papers it profiled

Node counts come from the released graph (one count per source URL), not from the judge's
own final_graph field, which already includes its proposed repairs.

Class B (no LLM, no network). Run from graph_analysis/:
    python -u experiment_review_omission_relative.py \
        --judge-reports <dir> --grader-archive <dir>

Output: graph_analysis/phase2_results/experiment_review_omission_relative_report.json
"""

import argparse
import json
import pickle
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from experiment_judge_full_receipt import load_grader

ROOT = Path(__file__).parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
OUT = ROOT / "phase2_results/experiment_review_omission_relative_report.json"


def pct(n, d, nd=1):
    return round(100 * n / d, nd) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-reports", required=True)
    ap.add_argument("--grader-archive", required=True)
    a = ap.parse_args()
    jdir, ma = Path(a.judge_reports), Path(a.grader_archive)
    for p, what in ((jdir, "judge report dir"), (ma, "grader archive")):
        if not p.is_dir():
            raise SystemExit(f"FATAL: {what} not found: {p}")
    if not SLIM.exists():
        raise SystemExit(
            f"FATAL: {SLIM} not found. Run experiment_review_prep_slim_nodes.py first."
        )

    slim = pickle.load(open(SLIM, "rb"))
    nodes_per_url = Counter()
    for r in slim.values():
        if r.get("url"):
            nodes_per_url[r["url"]] += 1

    # paper file name -> url, from the judge reports
    url_of_paper, judge_add_nodes = {}, {}
    for p in sorted(jdir.glob("*.json")):
        if p.name in ("summary.json", "errors.json"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("url"):
            url_of_paper[p.name] = r["url"]
            judge_add_nodes[p.name] = len(
                (r.get("proposed_fixes") or {}).get("add_nodes") or []
            )

    opus, _ = load_grader(ma / "test_extend_all_evaluation_opus_4_5", "*.json")
    profiled = {k: v for k, v in opus.items() if v.get("has_taxonomy_fields")}
    if not profiled:
        raise SystemExit("FATAL: no grader rows carry the taxonomy fields")

    def block(papers, label, missed_of):
        rows, unmatched = [], []
        for name in papers:
            url = url_of_paper.get(name)
            n_nodes = nodes_per_url.get(url) if url else None
            if not n_nodes:
                unmatched.append(name)
                continue
            rows.append((name, n_nodes, missed_of(name)))
        n_papers = len(rows)
        tot_nodes = sum(n for _, n, _ in rows)
        tot_missed = sum(m for _, _, m in rows)
        return {
            "label": label,
            "n_papers_matched_to_the_graph": n_papers,
            "n_papers_unmatched": len(unmatched),
            "extracted_nodes_total": tot_nodes,
            "extracted_nodes_mean_per_paper": round(tot_nodes / n_papers, 1),
            "extracted_nodes_median_per_paper": st.median(n for _, n, _ in rows),
            "omissions_total": tot_missed,
            "omissions_mean_per_paper": round(tot_missed / n_papers, 2),
            "omissions_as_pct_of_extracted_nodes": pct(tot_missed, tot_nodes),
            "implied_coverage_pct": pct(tot_nodes, tot_nodes + tot_missed),
            "unmatched_papers": unmatched,
        }

    out = {
        "experiment": "omission counts relative to extraction size",
        "question": (
            "Five missed concepts per paper is uninterpretable on its own. Against how "
            "many extracted nodes, in the same papers?"
        ),
        "judge_proposed_additions_over_the_100_audited": block(
            sorted(url_of_paper), "judge add_nodes", lambda n: judge_add_nodes.get(n, 0)
        ),
        "grader_missed_concepts_over_the_43_profiled": block(
            sorted(profiled),
            "Opus missed_concepts",
            lambda n: profiled[n]["missed_concepts"],
        ),
        "READING": (
            "implied_coverage_pct treats every flagged omission as a genuine missing node "
            "and asks what share of the implied total the extraction captured. It is an "
            "upper bound on the omission rate and a lower bound on coverage, because no "
            "flagged omission was adjudicated against human judgement and some are "
            "restatements of nodes already present."
        ),
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
