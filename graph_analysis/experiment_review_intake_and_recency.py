#!/usr/bin/env python
"""Extraction intake accounting and corpus recency (reviewer items W-8/Q-W11 and S-W4).

W-8 / Q-W11: the manuscript says extraction "fails outright on a minority of ARD records"
with no count and no denominator, and reports 11,779 as the working corpus without saying
what it was filtered from. S-W4: the manuscript asserts ARD coverage "thins after 2023"
without a number.

This script assembles both from local evidence, and marks every figure with whether it is
re-derived here or quoted from a packaging record that cannot be re-derived on this
machine (the raw processed_ard/ failure directories are not in this repo).

Class B (no LLM, no network). Run from graph_analysis/:
    python -u experiment_review_intake_and_recency.py --recovery-bundle <dir>

Output: graph_analysis/phase2_results/experiment_review_intake_recency_report.json
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
DEDUP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
OUT = ROOT / "phase2_results/experiment_review_intake_recency_report.json"

# Quoted from Mike2/judge_recovery_bundle/README.md, which is the only surviving local
# record of the failure-bucket sizes. The directories themselves are not in this repo, so
# these three numbers are DOCUMENTED, not re-derived.
DOCUMENTED_FAILURE_BUCKETS = {
    "processed_ard/extraction_error": 1667,
    "processed_ard/graph_error": 128,
    "processed_ard/embeddings_error": 58,
}


def pct(n, d, nd=1):
    return round(100 * n / d, nd) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recovery-bundle", required=True, help="judge_recovery_bundle/data"
    )
    a = ap.parse_args()
    rb = Path(a.recovery_bundle)
    if not rb.is_dir():
        raise SystemExit(f"FATAL: recovery bundle data dir not found: {rb}")
    if not SLIM.exists():
        raise SystemExit(
            f"FATAL: {SLIM} not found. Run experiment_review_prep_slim_nodes.py first."
        )

    slim = pickle.load(open(SLIM, "rb"))
    corpus_urls = {r["url"] for r in slim.values() if r.get("url")}

    chain_urls = set()
    for line in open(DEDUP, encoding="utf-8"):
        urls = {slim[n]["url"] for n in json.loads(line)["path"] if n in slim}
        if len(urls) == 1:
            chain_urls.add(next(iter(urls)))

    # ---- re-derived failure-side counts from the bundle --------------------------
    cand = rb / "extraction_error_recoverable_info"
    attempts = rb / "recovered_errors"
    recovered = rb / "recovered_errors_graph"
    bundle = {
        "judgeable_failed_extractions_candidates": sum(
            1 for p in cand.iterdir() if p.is_dir()
        )
        if cand.is_dir()
        else None,
        "judge_repair_attempts": sum(
            1
            for p in attempts.glob("*.json")
            if p.name not in ("summary.json", "errors.json")
        )
        if attempts.is_dir()
        else None,
        "files_in_recovered_errors_graph": sum(1 for _ in recovered.glob("*.json"))
        if recovered.is_dir()
        else None,
    }

    documented_failures = sum(DOCUMENTED_FAILURE_BUCKETS.values())
    intake_reconstructed = len(corpus_urls) + documented_failures

    # ---- publication years -------------------------------------------------------
    year_of = {}
    for r in slim.values():
        u, y = r.get("url"), r.get("first_published")
        if u and y and u not in year_of:
            year_of[u] = int(y)
    corpus_years = Counter(year_of[u] for u in corpus_urls if u in year_of)
    chain_years = Counter(year_of[u] for u in chain_urls if u in year_of)

    def cumulative(counter, urls):
        known = sum(counter.values())
        return {
            "n_documents_with_a_year": known,
            "n_documents_without_a_year": len(urls) - known,
            "by_year": dict(sorted(counter.items())),
            "pct_2023_or_earlier": pct(
                sum(v for k, v in counter.items() if k <= 2023), known
            ),
            "pct_2024_or_later": pct(
                sum(v for k, v in counter.items() if k >= 2024), known
            ),
            "median_year": sorted(k for k, v in counter.items() for _ in range(v))[
                known // 2
            ]
            if known
            else None,
        }

    nodes_per_doc = Counter()
    for r in slim.values():
        if r.get("url"):
            nodes_per_doc[r["url"]] += 1

    out = {
        "experiment": "extraction intake accounting and corpus recency (W-8/Q-W11, S-W4)",
        "intake_funnel": {
            "ard_records_taken_in_RECONSTRUCTED": intake_reconstructed,
            "documents_with_a_parseable_extraction_REDERIVED": len(corpus_urls),
            "documents_yielding_a_quality_cut_chain_REDERIVED": len(chain_urls),
            "extraction_yield_pct": pct(len(corpus_urls), intake_reconstructed),
            "chain_yield_pct_of_extracted": pct(len(chain_urls), len(corpus_urls)),
            "failure_buckets_DOCUMENTED_not_rederivable_here": DOCUMENTED_FAILURE_BUCKETS,
            "total_documented_failures": documented_failures,
            "PROVENANCE": (
                "The two REDERIVED rows are computed from the released graph in this run. "
                "The failure buckets are quoted from the judge recovery bundle's README, "
                "the only local record of them; the processed_ard/ directories are not in "
                "this repository, so the intake total is a reconstruction (extracted + "
                "documented failures) and not an independent count of the ARD snapshot."
            ),
        },
        "failure_repair_bundle_REDERIVED": bundle,
        "nodes_per_document": {
            "mean": round(sum(nodes_per_doc.values()) / len(nodes_per_doc), 2),
            "median": sorted(nodes_per_doc.values())[len(nodes_per_doc) // 2],
            "min": min(nodes_per_doc.values()),
            "max": max(nodes_per_doc.values()),
        },
        "recency": {
            "corpus_11779_documents": cumulative(corpus_years, corpus_urls),
            "chain_yielding_documents": cumulative(chain_years, chain_urls),
            "NOTE": (
                "first_published is carried on the extracted nodes and inherited from the "
                "ARD record. It dates the document, not the extraction."
            ),
        },
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "recency"}, indent=1))
    print(json.dumps(out["recency"]["corpus_11779_documents"], indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
