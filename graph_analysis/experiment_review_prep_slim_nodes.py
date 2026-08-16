#!/usr/bin/env python
"""Build a slim node-attribute checkpoint for the review-response experiments.

The full Step-1 checkpoint (graph_node_attributes.pkl, 3.3 GB) carries embeddings we do
not need for any of the reviewer-requested analyses. This script loads it ONCE and writes
a small pickle holding only the scalar fields those analyses use.

Class B (no LLM, no network). Run from graph_analysis/:
    python -u experiment_review_prep_slim_nodes.py

Output: graph_analysis/phase2_results/node_attrs_slim.pkl
        graph_analysis/phase2_results/experiment_review_prep_report.json
"""

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
SRC = STEP1 / "graph_node_attributes.pkl"
OUT_PKL = ROOT / "phase2_results/node_attrs_slim.pkl"
OUT_JSON = ROOT / "phase2_results/experiment_review_prep_report.json"

KEEP = (
    "type",
    "concept_category",
    "intervention_maturity",
    "intervention_lifecycle",
    "url",
    "name",
    "paper_dir",
    "semantic_cluster",
    "first_published",
    "publication_years",
)


def main():
    if not SRC.exists():
        raise SystemExit(
            f"FATAL: {SRC} not found.\n"
            "  Produced by the Step-1 load-and-parse stage of the graph pipeline.\n"
            "  This script does NOT fall back to a cached or partial node table."
        )
    t0 = time.time()
    print(f"loading {SRC} ({SRC.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    na = pickle.load(open(SRC, "rb"))
    print(f"  {len(na):,} nodes in {time.time() - t0:.0f}s", flush=True)

    sample_keys = sorted(next(iter(na.values())).keys())
    present = Counter()
    slim = {}
    for nid, a in na.items():
        row = {}
        for k in KEEP:
            if k in a:
                v = a[k]
                row[k] = v
                if v not in (None, ""):
                    present[k] += 1
        slim[nid] = row

    with open(OUT_PKL, "wb") as f:
        pickle.dump(slim, f, protocol=4)

    urls = {r.get("url") for r in slim.values() if r.get("url")}
    rep = {
        "experiment": "slim node-attribute checkpoint for review-response analyses",
        "n_nodes": len(slim),
        "full_attribute_keys_on_a_sample_node": sample_keys,
        "kept_fields_nonempty_counts": dict(present),
        "n_distinct_urls": len(urls),
        "example_urls": sorted(list(urls))[:5],
        "out_pkl": str(OUT_PKL),
        "out_pkl_bytes": OUT_PKL.stat().st_size,
        "load_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    print(f"\nwrote {OUT_PKL}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
