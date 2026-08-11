#!/usr/bin/env python
"""Audit every quantitative claim in paper/paperA_draft_v2.tex against RAW data.

Re-derives each number from the primary source (the Step-1 node PKL and the raw path
jsonl) rather than trusting the receipt JSONs, then compares to the value printed in
the draft. Emits a pass/fail table.

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_paper_claim_audit.py

Output: graph_analysis/phase2_results/experiment_paper_claim_audit.json
"""

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
RAW = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
DEDUP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
OUT = ROOT / "phase2_results/experiment_paper_claim_audit.json"

BODY = [
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]
RACE_STRICT = re.compile(r"\brac(?:e|es|ed|ing)\b|competi|arms.?race", re.I)

results = []


def check(claim, paper_value, recomputed, ok=None, note=""):
    if ok is None:
        ok = paper_value == recomputed
    results.append(
        {
            "claim": claim,
            "in_draft": paper_value,
            "recomputed_from_raw": recomputed,
            "verdict": "PASS" if ok else "FAIL",
            "note": note,
        }
    )


def load_paths(fp, na):
    rows = []
    for line in open(fp, encoding="utf-8"):
        d = json.loads(line)
        nodes = d["path"]
        urls = {na.get(n, {}).get("url") for n in nodes}
        pres = {
            (na.get(n, {}).get("concept_category") or "").lower() for n in nodes[1:-1]
        }
        rows.append(
            {
                "nodes": nodes,
                "n": len(nodes),
                "urls": urls,
                "present": pres & set(BODY),
            }
        )
    return rows


def path_block(rows, label):
    n = len(rows)
    all5 = sum(1 for r in rows if r["present"].issuperset(BODY))
    hist = Counter(r["n"] for r in rows)
    per_paper = Counter(next(iter(r["urls"])) for r in rows)
    ri_nodes = {(r["nodes"][0], r["nodes"][-1]) for r in rows}
    return {
        "label": label,
        "n_paths": n,
        "all5": all5,
        "all5_pct": round(100 * all5 / n, 1),
        "len_hist": dict(sorted(hist.items())),
        "pct_len_eq7": round(100 * hist[7] / n, 1),
        "pct_len_gt7": round(100 * sum(v for k, v in hist.items() if k > 7) / n, 1),
        "pct_len_lt7": round(100 * sum(v for k, v in hist.items() if k < 7) / n, 1),
        "n_source_papers": len(per_paper),
        "max_paths_one_paper": per_paper.most_common(1)[0][1],
        "top5_papers": per_paper.most_common(5),
        "pct_paths_from_top1pct_papers": round(
            100
            * sum(v for _, v in per_paper.most_common(max(1, len(per_paper) // 100)))
            / n,
            1,
        ),
        "distinct_ri_node_pairs": len(ri_nodes),
        "subtype_presence_pct": {
            st: round(100 * sum(1 for r in rows if st in r["present"]) / n, 1)
            for st in BODY
        },
        "pct_single_paper_paths": round(
            100 * sum(1 for r in rows if len(r["urls"]) == 1) / n, 1
        ),
    }


def main():
    na = pickle.load(open(STEP1 / "graph_node_attributes.pkl", "rb"))

    # ---- node inventory ----------------------------------------------------------
    n_total = len(na)
    by_type = Counter((a.get("type") or "").lower() for a in na.values())
    by_cat = Counter(
        (a.get("concept_category") or "").lower()
        for a in na.values()
        if (a.get("type") or "").lower() == "concept"
    )
    urls = {a.get("url") for a in na.values() if a.get("url")}

    check("total nodes", 200525, n_total)
    check("risk nodes", 19096, by_cat.get("risk", 0))
    check("intervention nodes", 36959, by_type.get("intervention", 0))
    for label, val in [
        ("problem analysis", 27748),
        ("theoretical insight", 26361),
        ("design rationale", 27543),
        ("implementation mechanism", 34222),
        ("validation evidence", 28596),
    ]:
        check(f"{label} nodes", val, by_cat.get(label, 0))
    intermediate = sum(by_cat.get(b, 0) for b in BODY)
    check("intermediate concept nodes (sum of 5 subtypes)", 144470, intermediate)
    check(
        "node inventory sums to total",
        n_total,
        by_cat.get("risk", 0) + intermediate + by_type.get("intervention", 0),
    )
    check("corpus documents", 11779, len(urls))

    # ---- maturity ----------------------------------------------------------------
    mat = Counter(
        a.get("intervention_maturity")
        for a in na.values()
        if (a.get("type") or "").lower() == "intervention"
    )
    n_int = sum(mat.values())
    for m, paper_pct in [(1, 27.6), (2, 57.3), (3, 12.7), (4, 2.5)]:
        check(
            f"intervention maturity {m} pct",
            paper_pct,
            round(100 * mat.get(m, 0) / n_int, 1),
        )

    # ---- paths -------------------------------------------------------------------
    raw = path_block(load_paths(RAW, na), "raw v4_edge_only (8954)")
    ded = path_block(load_paths(DEDUP, na), "deduped v4_edge_only (2772)")

    # --- reporting unit: the DE-DUPLICATED set (what the draft leads with) ---------
    check("chain count (reporting unit)", 2772, ded["n_paths"])
    check("pct chains with all 5 stages", 87.4, ded["all5_pct"])
    check("pct chains of length 7", 64.0, ded["pct_len_eq7"])
    check("pct chains longer than 7", 25.8, ded["pct_len_gt7"])
    check("pct chains shorter than 7", 10.2, ded["pct_len_lt7"])
    check("distinct risk->intervention node pairs", 2643, ded["distinct_ri_node_pairs"])
    check("papers yielding a complete chain", 1868, ded["n_source_papers"])
    check("max chains from any one paper (deduped)", 7, ded["max_paths_one_paper"])
    for st, val in [
        ("problem analysis", 99.0),
        ("theoretical insight", 93.9),
        ("design rationale", 96.1),
        ("implementation mechanism", 96.0),
        ("validation evidence", 98.0),
    ]:
        check(f"subtype presence {st}", val, ded["subtype_presence_pct"][st])

    # --- raw set, quoted in the draft as a sensitivity check ----------------------
    check("raw chain count (sensitivity)", 8954, raw["n_paths"])
    check("raw pct with all 5 stages (sensitivity)", 89.0, raw["all5_pct"])
    check("raw pct of length 7 (sensitivity)", 48.0, raw["pct_len_eq7"])
    check(
        "raw distinct R->I node pairs (sensitivity)",
        3222,
        raw["distinct_ri_node_pairs"],
    )
    check(
        "raw: max chains from one paper (disclosed in Methods)",
        700,
        raw["max_paths_one_paper"],
        note="pseudo-replication -- the reason the deduped set is the reporting unit",
    )
    check(
        "raw: same source-paper count as deduped",
        ded["n_source_papers"],
        raw["n_source_papers"],
    )
    check(
        "all extracted chains are single-paper",
        100.0,
        raw["pct_single_paper_paths"],
        note="supports the 'extraction is single-source' claim",
    )

    # ---- race prevalence ---------------------------------------------------------
    pa_nodes = [
        a
        for a in na.values()
        if (a.get("concept_category") or "").lower() == "problem analysis"
    ]
    risk_nodes = [
        a for a in na.values() if (a.get("concept_category") or "").lower() == "risk"
    ]

    def race_pct(nodes):
        hit = sum(1 for a in nodes if RACE_STRICT.search(str(a.get("name") or "")))
        return round(100 * hit / len(nodes), 2)

    check("race framing, pct of problem-analysis nodes", 2.21, race_pct(pa_nodes))
    check("race framing, pct of risk nodes", 1.65, race_pct(risk_nodes))

    # ---- receipt-vs-draft consistency (not re-derivable without FalkorDB) ---------
    rec = json.loads(
        (ROOT / "phase2_results/experiment_merge_vs_simexcl_ec_report.json").read_text(
            encoding="utf-8"
        )
    )
    conds = {c["label"]: c for c in rec["conditions"]}
    m_sim = conds["4 merged(risk), risk<->risk SIM EXCLUDED"]["top10"]
    ratio = m_sim[0]["ec"] / m_sim[1]["ec"]
    check("merged+SIM-excluded EC ratio (90.1x)", 90.1, round(ratio, 1))
    m_full = conds["3 merged(risk), full SIM"]["top10"]
    check(
        "merged+full-SIM EC ratio (33.6x)",
        33.6,
        round(m_full[0]["ec"] / m_full[1]["ec"], 1),
    )
    u_full = conds["1 un-merged, full SIM"]["top10"][0]["ec"]
    u_excl = conds["2 un-merged, risk<->risk SIM EXCLUDED"]["top10"][0]["ec"]
    check("un-merged EC flattening factor (31x)", 31, round(u_full / u_excl))
    check(
        "biggest merged node members",
        4066,
        rec["conditions"][2]["biggest_merged_node_members"],
    )
    check(
        "biggest merged node EDGE degree",
        7777,
        rec["conditions"][2]["biggest_merged_node_edge_degree"],
    )

    # NOTE 2026-08-11: three scope-composition checks (routed-chain count, share
    # unplaceable, share capability-gap-only) were added here and then REMOVED. They
    # read phase2_routing_assignments.jsonl, which is untracked and not re-derivable
    # by a reader at reasonable cost, so the claims they backed were withdrawn from
    # the manuscript. Reinstate only if that file ships with the release.

    out = {
        "audit": "paperA_draft_v2.tex quantitative claims vs raw data",
        "n_claims_checked": len(results),
        "n_pass": sum(1 for r in results if r["verdict"] == "PASS"),
        "n_fail": sum(1 for r in results if r["verdict"] == "FAIL"),
        "claims": results,
        "path_set_comparison": {"raw": raw, "deduped": ded},
    }
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    print(f"{out['n_pass']}/{out['n_claims_checked']} PASS, {out['n_fail']} FAIL\n")
    for r in results:
        if r["verdict"] == "FAIL":
            print(
                f"  FAIL  {r['claim']}: draft={r['in_draft']} raw={r['recomputed_from_raw']}"
            )
    print(
        f"\nRAW      : {raw['n_paths']} paths, {raw['n_source_papers']} papers, "
        f"max {raw['max_paths_one_paper']} from one paper, all5 {raw['all5_pct']}%"
    )
    print(
        f"DEDUPED  : {ded['n_paths']} paths, {ded['n_source_papers']} papers, "
        f"max {ded['max_paths_one_paper']} from one paper, all5 {ded['all5_pct']}%"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
