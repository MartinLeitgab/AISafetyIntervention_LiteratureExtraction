"""experiment_dataset_strength_descriptives.py

Clustering-FREE descriptive stats that leverage the dataset's unique asset:
complete per-paper logical chains risk -> problem_analysis -> theoretical_insight
-> design_rationale -> implementation_mechanism -> validation_evidence -> intervention,
with maturity/lifecycle attributes. These are the "what no abstract/metadata
directory can do" use cases that need NO risk/intervention clustering and are
immune to the xrisk-hub / merge / SIM contamination.

Reports:
  (1) intervention maturity (1-4) + lifecycle (1-6) distributions  -> field evidence-state
  (2) EDGE-only path profile: length dist; per-path body-subtype presence;
      fraction of complete risk->intervention chains that include each subtype
      (esp. validation_evidence = empirically grounded vs theory-only)
  (3) end-to-end single-paper risk->intervention coverage: distinct (risk,interv)
      node pairs and distinct paper-pairs argued end-to-end in ONE paper

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_dataset_strength_descriptives.py
"""

from __future__ import annotations
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PATHS = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
OUT = ROOT / "phase2_results/experiment_dataset_strength_report.json"

BODY_SUBTYPES = [
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]


def main():
    na = pickle.load(open(STEP1 / "graph_node_attributes.pkl", "rb"))
    print(f"{len(na)} nodes", flush=True)

    # ---------- (1) intervention maturity / lifecycle ----------
    mat = Counter()
    life = Counter()
    n_interv = 0
    for a in na.values():
        if (a.get("type") or "").lower() != "intervention":
            continue
        n_interv += 1
        m = a.get("intervention_maturity")
        lifecycle = a.get("intervention_lifecycle")
        mat[m] += 1
        life[lifecycle] += 1
    print(f"\n(1) interventions: {n_interv}")
    print(
        f"  maturity (1=concept..4=deployed): {dict(sorted(mat.items(), key=lambda x: (x[0] is None, x[0])))}"
    )
    print(
        f"  lifecycle (1..6):                 {dict(sorted(life.items(), key=lambda x: (x[0] is None, x[0])))}"
    )

    def pctmap(counter, tot):
        return {
            str(k): {"n": v, "pct": round(100 * v / tot, 1)} for k, v in counter.items()
        }

    # ---------- (2) EDGE-only path profile ----------
    n_paths = 0
    length_hist = Counter()
    subtype_present = Counter()  # paths containing >=1 node of this subtype
    subtype_counts_per_path = Counter()  # total nodes of subtype across paths
    n_with_all5 = 0
    n_with_validation = 0
    ri_node_pairs = set()
    ri_paper_pairs = set()
    with open(PATHS) as f:
        for line in f:
            d = json.loads(line)
            nodes = d["path"]
            n_paths += 1
            length_hist[len(nodes)] += 1
            body = nodes[1:-1]
            present = set()
            for nid in body:
                cc = (na.get(nid, {}).get("concept_category") or "").lower()
                if cc in BODY_SUBTYPES:
                    present.add(cc)
                    subtype_counts_per_path[cc] += 1
            for st in present:
                subtype_present[st] += 1
            if present.issuperset(set(BODY_SUBTYPES)):
                n_with_all5 += 1
            if "validation evidence" in present:
                n_with_validation += 1
            r, i = nodes[0], nodes[-1]
            ri_node_pairs.add((r, i))
            ru = na.get(r, {}).get("url") or f"r{r}"
            iu = na.get(i, {}).get("url") or f"i{i}"
            ri_paper_pairs.add((ru, iu))

    print(f"\n(2) EDGE-only complete chains: {n_paths}")
    print(f"  length distribution (nodes): {dict(sorted(length_hist.items()))}")
    print(f"  per-path body-subtype PRESENCE (% of {n_paths} paths):")
    sub_rows = {}
    for st in BODY_SUBTYPES:
        p = subtype_present[st]
        sub_rows[st] = {"paths_present": p, "pct": round(100 * p / n_paths, 1)}
        print(f"    {st:26s} {p:>5} ({100 * p / n_paths:4.1f}%)")
    print(
        f"  paths with ALL 5 body subtypes present: {n_with_all5} ({100 * n_with_all5 / n_paths:.1f}%)"
    )
    print(
        f"  paths with validation_evidence (empirically grounded): {n_with_validation} ({100 * n_with_validation / n_paths:.1f}%)"
    )

    # ---------- (3) end-to-end coverage ----------
    print("\n(3) single-paper end-to-end risk->intervention coverage:")
    print(f"  distinct (risk_node, intervention_node) pairs: {len(ri_node_pairs)}")
    print(f"  distinct (risk_paper, intervention_paper) pairs: {len(ri_paper_pairs)}")

    out = {
        "interventions": {
            "n": n_interv,
            "maturity": pctmap(mat, n_interv),
            "lifecycle": pctmap(life, n_interv),
        },
        "edge_only_chains": {
            "n_paths": n_paths,
            "length_distribution": {str(k): v for k, v in sorted(length_hist.items())},
            "body_subtype_presence_pct": sub_rows,
            "paths_with_all5_subtypes": n_with_all5,
            "paths_with_all5_pct": round(100 * n_with_all5 / n_paths, 1),
            "paths_with_validation_evidence": n_with_validation,
            "paths_with_validation_pct": round(100 * n_with_validation / n_paths, 1),
        },
        "end_to_end_coverage": {
            "distinct_ri_node_pairs": len(ri_node_pairs),
            "distinct_ri_paper_pairs": len(ri_paper_pairs),
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}\nDONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
