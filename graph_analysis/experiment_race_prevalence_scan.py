"""experiment_race_prevalence_scan.py

Grounds whether "race framing" is a dominant theme or a top-100-selection artefact,
by scanning RACE-keyword prevalence across:
  (1) every node type (risk / 5 body subtypes / intervention) over the full graph
  (2) our EDGE-only canonical paths (8,954) — fraction of complete single-paper
      reasoning chains that contain any race-worded node, and PA-specifically.

Two keyword variants:
  loose  = Gleb-style substring ["race","racing","competi"]  (catches embrace/trace/grace/racial too)
  strict = word-boundary regex  \\brac(e|es|ed|ing)\\b | competi | arms.?race

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_race_prevalence_scan.py
"""

from __future__ import annotations
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PATHS = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
OUT = ROOT / "phase2_results/experiment_race_prevalence_report.json"

LOOSE = ["race", "racing", "competi"]
STRICT = re.compile(r"\brac(?:e|es|ed|ing)\b|competi|arms.?race")


def loose_hit(name):
    n = (name or "").lower()
    return any(k in n for k in LOOSE)


def strict_hit(name):
    return bool(STRICT.search((name or "").lower()))


def main():
    na = pickle.load(open(STEP1 / "graph_node_attributes.pkl", "rb"))
    print(f"{len(na)} nodes", flush=True)

    def ntype(a):
        if (a.get("type") or "").lower() == "intervention":
            return "intervention"
        return (a.get("concept_category") or "").lower() or "unknown"

    by_type = defaultdict(lambda: {"n": 0, "loose": 0, "strict": 0})
    for a in na.values():
        t = ntype(a)
        nm = a.get("name")
        by_type[t]["n"] += 1
        if loose_hit(nm):
            by_type[t]["loose"] += 1
        if strict_hit(nm):
            by_type[t]["strict"] += 1

    print("\n=== (1) race prevalence by node type (full graph, name-only) ===")
    order = [
        "risk",
        "problem analysis",
        "theoretical insight",
        "design rationale",
        "implementation mechanism",
        "validation evidence",
        "intervention",
    ]
    type_rows = {}
    for t in order + [k for k in by_type if k not in order]:
        if t not in by_type:
            continue
        r = by_type[t]
        lo = 100 * r["loose"] / r["n"] if r["n"] else 0
        st = 100 * r["strict"] / r["n"] if r["n"] else 0
        type_rows[t] = {
            "n": r["n"],
            "loose": r["loose"],
            "loose_pct": round(lo, 2),
            "strict": r["strict"],
            "strict_pct": round(st, 2),
        }
        print(
            f"  {t:26s} n={r['n']:>6}  loose={r['loose']:>4} ({lo:4.2f}%)  strict={r['strict']:>4} ({st:4.2f}%)"
        )

    # (2) EDGE-only paths
    print("\n=== (2) race prevalence across EDGE-only canonical paths ===")
    n_paths = 0
    n_any_loose = n_any_strict = 0
    n_pa_loose = n_pa_strict = 0
    n_paths_with_pa = 0
    with open(PATHS) as f:
        for line in f:
            d = json.loads(line)
            nodes = d["path"]
            n_paths += 1
            any_l = any_s = pa_l = pa_s = False
            has_pa = False
            for i, nid in enumerate(nodes):
                a = na.get(nid) or {}
                nm = a.get("name")
                cc = (a.get("concept_category") or "").lower()
                if loose_hit(nm):
                    any_l = True
                if strict_hit(nm):
                    any_s = True
                if cc == "problem analysis":
                    has_pa = True
                    if loose_hit(nm):
                        pa_l = True
                    if strict_hit(nm):
                        pa_s = True
            n_any_loose += any_l
            n_any_strict += any_s
            n_paths_with_pa += has_pa
            n_pa_loose += pa_l
            n_pa_strict += pa_s

    def pct(x, d):
        return round(100 * x / d, 2) if d else 0

    paths_res = {
        "n_paths": n_paths,
        "paths_with_any_race_node_loose": n_any_loose,
        "pct_loose": pct(n_any_loose, n_paths),
        "paths_with_any_race_node_strict": n_any_strict,
        "pct_strict": pct(n_any_strict, n_paths),
        "n_paths_with_pa": n_paths_with_pa,
        "paths_with_race_PA_loose": n_pa_loose,
        "pct_pa_loose": pct(n_pa_loose, n_paths),
        "paths_with_race_PA_strict": n_pa_strict,
        "pct_pa_strict": pct(n_pa_strict, n_paths),
    }
    print(f"  paths: {n_paths}")
    print(
        f"  with ANY race-worded node:  loose {n_any_loose} ({pct(n_any_loose, n_paths)}%)  strict {n_any_strict} ({pct(n_any_strict, n_paths)}%)"
    )
    print(
        f"  with race-worded PA node:   loose {n_pa_loose} ({pct(n_pa_loose, n_paths)}%)  strict {n_pa_strict} ({pct(n_pa_strict, n_paths)}%)"
    )

    out = {
        "keywords": {"loose": LOOSE, "strict": STRICT.pattern},
        "by_node_type_full_graph": type_rows,
        "edge_only_paths": paths_res,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}\nDONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
