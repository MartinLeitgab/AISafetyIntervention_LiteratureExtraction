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
    check("un-merged EC flattening factor (30.7x)", 30.7, round(u_full / u_excl, 1))
    u_full_ratio = u_full / conds["1 un-merged, full SIM"]["top10"][1]["ec"]
    u_excl_ratio = (
        u_excl / conds["2 un-merged, risk<->risk SIM EXCLUDED"]["top10"][1]["ec"]
    )
    check("un-merged full-SIM EC1/EC2 (1.00x in tab:ec)", 1.0, round(u_full_ratio, 2))
    check(
        "un-merged SIM-excluded EC1/EC2 (1.03x in tab:ec)", 1.03, round(u_excl_ratio, 2)
    )
    check(
        "un-merged full-SIM EC1 printed as 0.03275",
        0.03275,
        round(u_full, 5),
    )
    check(
        "un-merged SIM-excluded EC1 printed as 0.00107",
        0.00107,
        round(u_excl, 5),
    )
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

    # ---- numbers added in the 2026-08-14 review-response pass --------------------
    # Each reads the receipt written by the named script, so a stale receipt fails the
    # audit instead of silently agreeing with the manuscript.
    def receipt(name):
        p = ROOT / "phase2_results" / name
        if not p.exists():
            raise SystemExit(
                f"FATAL: missing receipt {p}\n"
                "  produced by the matching experiment_review_*.py script.\n"
                "  This audit does NOT skip claims whose receipt is absent."
            )
        return json.loads(p.read_text(encoding="utf-8"))

    ov = receipt("experiment_review_judge_overlap_report.json")["headline"]
    check(
        "judged papers that yield a chain", 12, ov["n_judged_papers_that_yield_a_chain"]
    )
    check(
        "chains covered by the judged sample",
        17,
        ov["n_chains_in_reporting_unit_covered_by_the_judged_sample"],
    )
    check("judged sample size", 100, ov["n_judged_papers"])

    gs = receipt("experiment_review_gate_sensitivity_report.json")
    rep = gs["baseline_reproduction_check"]
    check(
        "gate re-enumeration reproduces the released raw path set",
        True,
        rep["raw_path_multisets_identical"],
    )
    check(
        "gate re-enumeration reproduces the released deduped path set",
        True,
        rep["deduped_path_multisets_identical"],
    )
    grid = gs["gate_grid"]
    for key, chains, papers, yld, all5 in [
        ("conf>=3, maturity>=3", 2772, 1868, 15.9, 87.4),
        ("conf>=3, maturity>=2", 7718, 4108, 34.9, 83.7),
        ("conf>=3, maturity>=1", 8505, 4388, 37.3, 82.0),
        ("conf>=2, maturity>=3", 6188, 3546, 30.1, 89.7),
        ("conf>=2, maturity>=2", 24427, 10166, 86.3, 87.9),
        ("conf>=2, maturity>=1", 30189, 11175, 94.9, 86.7),
        ("conf>=1, maturity>=3", 6285, 3593, 30.5, 89.8),
        ("conf>=1, maturity>=2", 25228, 10489, 89.0, 88.1),
        ("conf>=1, maturity>=1", 31740, 11709, 99.4, 87.1),
    ]:
        g = grid[key]
        check(f"tab:gates {key} chains", chains, g["n_chains"])
        check(f"tab:gates {key} papers", papers, g["n_source_papers"])
        check(f"tab:gates {key} yield pct", yld, g["corpus_yield_pct"])
        check(f"tab:gates {key} all-five pct", all5, g["all5_pct"])
    cont = gs["containment_threshold_sensitivity"]
    for th, n in [("0.60", 2658), ("0.70", 2772), ("0.80", 3356), ("0.90", 5460)]:
        check(f"containment {th} chain count", n, cont[th]["n_chains"])
    loss = gs["containment_losslessness"]
    check("containment: dropped paths", 6182, loss["n_paths_dropped_at_0.70"])
    check(
        "containment: pct dropped paths carrying a novel node",
        78.3,
        loss["pct_dropped_paths_carrying_a_novel_node"],
    )
    check(
        "containment: nodes lost from the chain set",
        1169,
        loss["n_nodes_present_in_a_dropped_path_and_in_no_kept_path"],
    )
    check(
        "containment: pct of chain-set nodes lost",
        6.13,
        loss["pct_of_raw_chain_set_nodes_lost"],
    )
    check(
        "containment: papers affected",
        695,
        loss["n_papers_with_at_least_one_lost_node"],
    )

    ga = receipt("experiment_review_grader_agreement_report.json")[
        "raw_score_agreement"
    ]
    check("grader ICC(2,1) pre-repair", 0.921, ga["pre_repair"]["ICC_2_1"])
    check("grader ICC(2,1) post-repair", 0.151, ga["post_repair"]["ICC_2_1"])
    check("grader ICC(2,k) pre-repair", 0.972, ga["pre_repair"]["ICC_2_k"])
    check("grader ICC(2,k) post-repair", 0.348, ga["post_repair"]["ICC_2_k"])
    check(
        "grader Krippendorff alpha pre-repair",
        0.917,
        ga["pre_repair"]["krippendorff_alpha_interval"],
    )
    check(
        "grader Krippendorff alpha post-repair",
        0.043,
        ga["post_repair"]["krippendorff_alpha_interval"],
    )

    om = receipt("experiment_review_omission_relative_report.json")
    j = om["judge_proposed_additions_over_the_100_audited"]
    g = om["grader_missed_concepts_over_the_43_profiled"]
    check("judged papers: extracted nodes", 1617, j["extracted_nodes_total"])
    check(
        "judge additions as pct of extracted nodes",
        0.6,
        j["omissions_as_pct_of_extracted_nodes"],
    )
    check("judge implied coverage pct", 99.4, j["implied_coverage_pct"])
    check("profiled papers: extracted nodes", 751, g["extracted_nodes_total"])
    check(
        "missed concepts as pct of extracted nodes",
        28.8,
        g["omissions_as_pct_of_extracted_nodes"],
    )
    check("grader implied coverage pct", 77.7, g["implied_coverage_pct"])
    check("missed concepts total", 216, g["omissions_total"])

    sil = receipt("experiment_review_silhouette_report.json")["headline"]
    check(
        "UMAP k=40 silhouette in its own space",
        0.2809,
        sil["umap_k40_silhouette_in_its_own_space"],
    )
    check(
        "UMAP k=40 silhouette in the original space",
        0.0043,
        sil["umap_k40_silhouette_in_original_space"],
    )
    check(
        "direct k=40 silhouette in the original space",
        0.0141,
        sil["direct_k40_silhouette_in_original_space"],
    )

    ir = receipt("experiment_review_intake_recency_report.json")
    fn = ir["intake_funnel"]
    check("reconstructed ARD intake", 13632, fn["ard_records_taken_in_RECONSTRUCTED"])
    check(
        "documents with a parseable extraction",
        11779,
        fn["documents_with_a_parseable_extraction_REDERIVED"],
    )
    check("extraction yield pct", 86.4, fn["extraction_yield_pct"])
    rec = ir["recency"]["corpus_11779_documents"]
    check("documents dated 2024 or later", 0.0, rec["pct_2024_or_later"])
    check("median publication year", 2021, rec["median_year"])
    check("documents carrying a year", 11761, rec["n_documents_with_a_year"])
    check("mean nodes per document", 17.02, ir["nodes_per_document"]["mean"])

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
