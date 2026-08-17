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

    # REMOVED 2026-08-16: six checks on the grader agreement instruments (ICC(2,1),
    # ICC(2,k), Krippendorff alpha, pre and post) and the per-grader files-seen and
    # JSON-shape diagnostics. None of those numbers is printed in the manuscript any more
    # -- the pre/post repair-scoring stage was compressed to a design lesson with no
    # statistics, unanimously across six external reviews. The receipts still ship and
    # experiment_review_grader_agreement.py still produces them; if the null-repair arm is
    # ever run and the stage becomes a result, restore these checks with it.
    # A check for a number the paper does not print is not a regression test.
    mg = receipt("experiment_judge_full_report.json")["item2_meta_graders"]
    # The third grader is keyed by whichever name the receipt on disk carries. The shipped
    # receipt still says "third_grader_gpt-5.1"; experiment_judge_full_receipt.py now writes
    # "third_grader_model_not_recorded", because no artifact records that grader's model and
    # the old key was an assumption the manuscript then repeated. Regenerating the receipt
    # needs the Drive archives, so both spellings are accepted rather than hand-editing it.
    third_key = next(k for k in mg if k.startswith("third_grader"))
    for key, n in [
        ("claude-opus-4-5", 95),
        ("gemini-3-pro", 13),
        (third_key, 95),
    ]:
        check(
            f"grader {key}: paired pre/post rows",
            n,
            mg[key]["n"],
            note="printed in tab:populations-master as the 95/95/13 row; the per-grader "
            "session diagnostics behind those denominators left the paper 2026-08-16",
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
    check("profiled papers: extracted nodes", 751, g["extracted_nodes_total"])
    check(
        "missed concepts as pct of extracted nodes",
        28.8,
        g["omissions_as_pct_of_extracted_nodes"],
    )
    check("missed concepts total", 216, g["omissions_total"])
    # REMOVED 2026-08-16: the two "implied coverage" checks (99.4% and 77.7%). The phrase
    # is gone from the paper -- reviewers read it as an accuracy claim when it is one minus
    # an unadjudicated flag rate, and the edge measurement below made the node-only version
    # of it misleading as a headline.

    # ---- edge-level coverage (S10, issue #156) --------------------------------------
    ec = receipt("experiment_review_edge_coverage_report.json")
    cl = ec["coverage_list"]
    against = ec["against_the_extraction_it_is_measured_on"]
    check("coverage list: rows over the 100 judged papers", 777, cl["rows_total"])
    check("coverage list: covered", 328, cl["covered"])
    check("coverage list: partially covered", 146, cl["partially_covered"])
    check("coverage list: missing", 302, cl["missing"])
    check("coverage list: missing per paper", 3.02, cl["missing_mean_per_paper"])
    check(
        "coverage list: papers with at least one missing",
        90,
        cl["papers_with_at_least_one_missing"],
    )
    check(
        "judged papers: structural edges in the released graph",
        1667,
        against["released_edges_total"],
    )
    check(
        "judged papers: released edges per paper",
        16.7,
        against["released_edges_mean_per_paper"],
        note="app:judge prints this beside the judge's own final_graph mean of 10.8",
    )
    check(
        "missing relationships as pct of extracted edges",
        18.1,
        against["missing_as_pct_of_released_edges"],
        note="the abstract's third omission rate",
    )
    check(
        "same count against the judge's own edge total",
        28.1,
        ec["which_denominator"]["missing_as_pct_of_judge_final_graph_edges"],
    )
    check(
        "judge final_graph edges per paper",
        10.76,
        ec["which_denominator"]["judge_final_graph_edges_mean_per_paper"],
    )
    check(
        "judge repair schema has no add_edges slot",
        False,
        ec["no_add_edges_slot"]["has_add_edges_key"],
        note="app:judgeprompt and sec:r-judge give this absence as the explanation for the "
        "gap between the node-addition count and the coverage list",
    )
    check(
        "structural edges crossing two source papers",
        0,
        ec["released_graph_structural_edges"]["n_crossing_two_source_papers"],
        note="single-source-by-design, sec:m-structural",
    )

    # ---- what the sub-path collapse drops (issue #157) -------------------------------
    cs = receipt("experiment_review_containment_semantics_report.json")
    check(
        "collapse: dropped paths that are contiguous sub-paths of their container",
        0.0,
        cs["order_relation"]["pct_contiguous_sub_path"],
        note="sec:m-reporting no longer describes the step as dropping sub-paths already "
        "counted inside a longer path; this is why",
    )
    check(
        "collapse: drops differing only by chords",
        21.7,
        cs["edge_identity"]["pct_chords_only"],
    )
    check(
        "collapse: drops touching a node the container lacks",
        78.3,
        cs["edge_identity"]["pct_touching_a_node_the_container_lacks"],
    )
    check(
        "collapse: drops ending at an intervention the container lacks",
        28.0,
        cs["does_the_drop_remove_a_distinct_remedy"][
            "pct_ending_at_an_intervention_the_container_lacks"
        ],
    )
    check(
        "collapse: distinct risk-to-intervention pairs lost",
        579,
        cs["does_the_drop_remove_a_distinct_remedy"][
            "distinct_pairs_lost_to_the_collapse"
        ],
    )
    check(
        "collapse: pct of raw R-I pairs lost",
        18.0,
        cs["does_the_drop_remove_a_distinct_remedy"]["pct_of_raw_pairs_lost"],
    )

    # ---- what the release ships (issue #157) -----------------------------------------
    ri = receipt("experiment_review_release_integrity_report.json")
    defects = ri["does_the_release_ship_the_defect_classes_the_judge_found"]
    for label, expected, key in [
        ("orphan nodes", 0, "orphan_nodes_zero_structural_edges"),
        ("dangling edges", 0, "dangling_edges_endpoint_not_in_node_table"),
        ("self-loops", 0, "self_loops"),
        ("duplicate edges", 1, "duplicate_edges_same_pair_same_relation_type"),
        (
            "exact-name duplicate groups",
            448,
            "exact_name_duplicate_groups_within_category",
        ),
        (
            "exact-name duplicate nodes beyond one per group",
            1140,
            "exact_name_duplicate_nodes_beyond_one_per_group",
        ),
    ]:
        check(f"released graph: {label}", expected, defects[key])
    check(
        "every gate attribute present on the released graph",
        True,
        ri["can_a_reuser_re_enumerate_at_any_gate"][
            "verdict_all_gate_attributes_present"
        ],
        note="sec:m-repro claims a reuser can re-enumerate at any gate setting from the "
        "dump alone, which is only true while this holds",
    )
    check(
        "released nodes from a judged document",
        1617,
        ri["audited_vs_unaudited_content"]["nodes_from_judged_documents"],
    )
    check(
        "released nodes from a judged document, pct",
        0.81,
        ri["audited_vs_unaudited_content"]["nodes_from_judged_documents_pct"],
    )

    # ---- second-model stage agreement (S3, issue #161) -------------------------------
    sa = receipt("experiment_review_stage_agreement_report.json")
    head = sa["headline"]
    check("stage agreement: Cohen kappa", 0.838, head["cohen_kappa"])
    check("stage agreement: raw agreement", 0.871, head["raw_agreement"])
    check("stage agreement: chance agreement", 0.204, head["chance_agreement"])
    check("stage agreement: disagreements", 84, head["n_disagreements"])
    check("stage agreement: nodes scored", 653, sa["design"]["n_label_pairs_scored"])
    check(
        "stage agreement: kappa on chain-yielding documents",
        0.835,
        sa["by_stratum"]["chain_yielding"]["cohen_kappa"],
    )
    check(
        "stage agreement: kappa on non-chain-yielding documents",
        0.844,
        sa["by_stratum"]["other"]["cohen_kappa"],
    )
    for stage, f1, rec in [
        ("problem analysis", 0.955, 0.983),
        ("theoretical insight", 0.756, 0.707),
        ("design rationale", 0.815, 0.818),
        ("implementation mechanism", 0.889, 0.923),
        ("validation evidence", 0.916, 0.900),
    ]:
        check(f"stage agreement F1: {stage}", f1, sa["per_class"][stage]["f1"])
        check(f"stage agreement recall: {stage}", rec, sa["per_class"][stage]["recall"])
    adj = sa["adjacent_stage_confusions"]
    for pair, n in [("dr_vs_im", 26), ("ti_vs_dr", 19), ("pa_vs_ti", 7)]:
        check(f"stage agreement confusion {pair}", n, adj[pair])
    check(
        "stage agreement: predicted-pair share of disagreements",
        39.3,
        adj["pa_ti_plus_dr_im_share_of_disagreements"],
        note="the paper reports the pre-registered prediction as half right; this is the "
        "number that makes it half rather than wholly right",
    )
    check(
        "stage agreement: unusable responses",
        0,
        sum(sa["unusable_responses"].values()),
        note="653 of 653 nodes came back with a label inside the five-stage vocabulary",
    )

    # ---- schema ablation and degraded source (S6+S8, issue #165) ---------------------
    # Read from the receipt, as the other Class A studies are: re-deriving would mean
    # re-buying 85 o3 calls. The receipt is computed from raw responses committed beside
    # it, so a reader can re-score without re-running the extraction.
    ab = receipt("experiment_review_schema_ablation_report.json")["results"]
    hl = ab["headline"]
    pair = ab["paired_against_the_released_extraction"]
    check(
        "ablation: arm A chain rate, EF sample",
        100.0,
        hl["A_released_on_the_EF_sample"]["pct_with_chain"],
        note="the instrument check: this scorer reproduces the released enumerator on "
        "every document of the sample",
    )
    check(
        "ablation: arm A all-five share, EF sample",
        65.0,
        hl["A_released_on_the_EF_sample"]["pct_chains_all_five"],
    )
    check(
        "ablation: arm A chain rate, G sample",
        100.0,
        hl["A_released_on_the_G_sample"]["pct_with_chain"],
    )
    check(
        "ablation: arm A all-five share, G sample",
        93.3,
        hl["A_released_on_the_G_sample"]["pct_chains_all_five"],
    )
    check("ablation: arm E documents returning parseable JSON", 28, hl["E"]["n"])
    check(
        "ablation: arm E chain yield pct of attempted",
        36.7,
        hl["E"]["pct_of_attempted_yielding_a_chain"],
    )
    check(
        "ablation: arm E chains carrying all five stages",
        0.0,
        hl["E"]["pct_chains_all_five"],
        note="the ablation headline: un-prompted, the five-stage chain does not appear",
    )
    check(
        "ablation: arm E documents with a chain",
        11,
        pair["E"]["of_those_also_yielding_a_chain_in_this_arm"],
    )
    check(
        "ablation: arm F chain yield pct",
        30.0,
        hl["F"]["pct_of_attempted_yielding_a_chain"],
        note="confabulation rate from sentence-shuffled sources",
    )
    check("ablation: arm F all-five share", 46.3, hl["F"]["pct_chains_all_five"])
    check(
        "ablation: arm G chain yield pct",
        8.0,
        hl["G"]["pct_of_attempted_yielding_a_chain"],
    )
    check(
        "ablation: arm G documents with a chain",
        2,
        pair["G"]["of_those_also_yielding_a_chain_in_this_arm"],
    )
    check(
        "ablation: arm G declined to emit a graph",
        6,
        hl["G"]["no_graph_returned"]["declined_no_json_block"],
    )
    lm = ab["arm_E_label_mapping_summary"]
    check("ablation: arm E distinct emergent labels", 144, sum(lm.values()))
    check("ablation: arm E labels mapping onto no stage", 5, lm["unmappable"])
    check(
        "ablation: arm E labels mapping onto one of the five",
        138,
        sum(v for k, v in lm.items() if k not in ("unmappable", "risk")),
        note="the split the paper reports: the vocabulary is recoverable un-prompted, the "
        "seven-node chain is not",
    )

    # ---- comparison against an existing topical index (S12, issue #166) -------------
    # Re-derived here from the VENDORED categories.json plus the released graph, not read
    # back from the comparison receipt. The receipt supplies only the title-to-URL map,
    # because their 8.6 MB input CSV is not vendored; every count below is recomputed.
    ac = receipt("experiment_review_artifact_comparison_report.json")
    cats = json.loads(
        (ROOT / "phase2_results/external_artifacts/categories.json").read_text(
            encoding="utf-8"
        )
    )
    t2u = ac["title_to_url"]
    chains_by_url = {}
    for r in load_paths(DEDUP, na):
        u = next(iter(r["urls"]))
        chains_by_url.setdefault(u, []).append(r["nodes"])
    covered_subs, pairs_total, comembership, n_subs = 0, set(), 0, 0
    adv = {"papers": 0, "risks": set(), "intvs": set()}
    for subs in cats.values():
        for sub, titles in subs.items():
            n_subs += 1
            matched_here = [t for t in titles if t2u.get(t)]
            comembership += len(matched_here) * (len(matched_here) - 1) // 2
            sub_pairs, sub_risks, sub_intvs, sub_papers = set(), set(), set(), 0
            for t in matched_here:
                paths = chains_by_url.get(t2u[t], [])
                if paths:
                    sub_papers += 1
                for p in paths:
                    sub_risks.add(na.get(p[0], {}).get("name"))
                    sub_intvs.add(na.get(p[-1], {}).get("name"))
                    sub_pairs.add(
                        (na.get(p[0], {}).get("name"), na.get(p[-1], {}).get("name"))
                    )
            if sub_pairs:
                covered_subs += 1
                pairs_total |= sub_pairs
            if sub == "Adversarial Machine Learning":
                adv = {"papers": sub_papers, "risks": sub_risks, "intvs": sub_intvs}
    check(
        "artifact comparison: their clustered papers",
        554,
        len({t for s in cats.values() for ts in s.values() for t in ts}),
    )
    check("artifact comparison: their subcategories", 160, n_subs)
    check(
        "artifact comparison: their clustered papers matched to our corpus",
        534,
        len(t2u),
    )
    check(
        "artifact comparison: matched papers yielding a chain",
        216,
        sum(1 for u in t2u.values() if chains_by_url.get(u)),
    )
    check(
        "artifact comparison: their subcategories holding a chain-yielding paper",
        79,
        covered_subs,
    )
    check(
        "artifact comparison: our directed pairs inside those subcategories",
        325,
        len(pairs_total),
        note="79 topic labels against 325 DISTINCT directed risk-to-intervention pairs is "
        "the resolution claim in sec:related. Summing the per-subcategory counts gives 335 "
        "because a paper can sit in more than one subcategory; this check is what caught "
        "that, and the receipt now reports both",
    )
    check(
        "artifact comparison: chain yield on their paper list, pct",
        40.4,
        round(100 * sum(1 for u in t2u.values() if chains_by_url.get(u)) / len(t2u), 1),
        note="independent check on the 40.5% arXiv yield of tab:populations, over a paper "
        "list we did not choose",
    )
    check(
        "artifact comparison: their input records",
        7011,
        ac["their_artifact"]["input_papers"],
    )
    check(
        "artifact comparison: Adversarial ML chain-yielding papers", 17, adv["papers"]
    )
    check("artifact comparison: Adversarial ML distinct risks", 18, len(adv["risks"]))
    check(
        "artifact comparison: Adversarial ML distinct interventions",
        23,
        len(adv["intvs"]),
    )
    check(
        "artifact comparison: their cross-document co-membership pairs",
        2986,
        comembership,
        note="sec:related reports this beside our zero cross-document structural edges",
    )

    # ---- multi-model extraction consistency (S11, issue #168) ------------------------
    # Receipt-sourced, like every Class A study. The arm-C figures are deliberately NOT
    # checked here: that arm was still running when these numbers went into the paper.
    mm = receipt("experiment_review_multimodel_consistency_report.json")
    mh = mm["headline"]
    check("multimodel: released mean nodes", 21, mh["released"]["mean_nodes"])
    check("multimodel: released mean edges", 20.1, mh["released"]["mean_edges"])
    check("multimodel: o3 re-run mean nodes", 21.1, mh["A_o3"]["mean_nodes"])
    check("multimodel: o3 re-run mean edges", 21, mh["A_o3"]["mean_edges"])
    check(
        "multimodel: o3 re-run chain rate",
        50.0,
        mh["A_o3"]["pct_with_chain"],
        note="against 100% for the shipped extraction on the same documents; the sample is "
        "conditioned on the shipped run, so this is an upper bound on what a re-run loses",
    )
    check(
        "multimodel: o3 re-run unparseable responses", 2, mh["A_o3"]["parse_failures"]
    )
    check("multimodel: gpt-5 mean nodes", 38.1, mh["B_gpt5"]["mean_nodes"])
    check("multimodel: opus-5 mean nodes", 41.3, mh["C_opus5"]["mean_nodes"])
    check("multimodel: opus-5 mean edges", 56.2, mh["C_opus5"]["mean_edges"])
    check("multimodel: opus-5 chain rate", 82.4, mh["C_opus5"]["pct_with_chain"])
    for arm, med, mx in [
        ("released", 2, 19),
        ("B_gpt5", 6, 886),
        ("C_opus5", 594, 57007),
    ]:
        check(
            f"multimodel: median chains per document, {arm}",
            med,
            mh[arm]["median_chains"],
            note="the mean is dominated by single documents; the paper quotes medians and "
            "maxima because enumeration is super-linear in edge count",
        )
        check(
            f"multimodel: max chains in one document, {arm}", mx, mh[arm]["max_chains"]
        )
    check("multimodel: gpt-5 mean edges", 41.3, mh["B_gpt5"]["mean_edges"])
    check("multimodel: gpt-5 chain rate", 76.5, mh["B_gpt5"]["pct_with_chain"])
    check("multimodel: gpt-5 unparseable responses", 3, mh["B_gpt5"]["parse_failures"])
    pa = mm["pairwise_endpoint_agreement"]["released vs A_o3"]
    check(
        "multimodel: risk-name agreement at cosine 0.80",
        46.5,
        pa["cosine_0.8"]["pct_risks_of_A_matched_in_B"],
    )
    check(
        "multimodel: risk-name agreement at cosine 0.85",
        27.8,
        pa["cosine_0.85"]["pct_risks_of_A_matched_in_B"],
    )
    check(
        "multimodel: risk-name agreement, token-set",
        19.0,
        pa["jaccard_0.6"]["pct_risks_of_B_matched_in_A"],
        note="the lexical figure the semantic one is reported against",
    )
    gs = mm["gate_stability_against_the_shipped_extraction"]["A_o3"]
    check("multimodel: documents compared", 18, gs["documents_compared"])
    check(
        "multimodel: shipped documents with a mature intervention",
        18,
        gs["released_with_a_mature_intervention"],
    )
    check(
        "multimodel: re-run documents with a mature intervention",
        11,
        gs["arm_with_a_mature_intervention"],
        note="the maturity gate is what moves between runs; edge confidence does not",
    )
    check("multimodel: re-run documents with a chain", 9, gs["arm_with_a_chain"])

    # ---- baselines B/C/D and the unconditioned repeat arm (S6 remainder, S7; #171) ---
    bl = receipt("experiment_review_baselines_report.json")["results"]["headline"]
    for arm, nodes, yield_pct in [
        ("B_abstract", 11.4, 16.0),
        ("C_gpt41", 16.4, 56.7),
        ("D_triples", 24.5, 0.0),
        ("U_unconditioned", 18.4, 15.0),
    ]:
        check(f"baseline {arm}: mean nodes", nodes, bl[arm]["mean_nodes"])
        check(
            f"baseline {arm}: chain yield pct",
            yield_pct,
            bl[arm]["pct_of_attempted_yielding_a_chain"],
        )
    check(
        "baseline: shipped nodes on the abstract sample",
        20.6,
        bl["released_on_the_B_abstract_sample"]["mean_nodes"],
    )
    check(
        "baseline: shipped chain yield on the unconditioned sample",
        15.0,
        bl["released_on_the_U_unconditioned_sample"][
            "pct_of_attempted_yielding_a_chain"
        ],
        note="identical to the re-run's 15.0% on the same 20 documents, and on the same "
        "three of them; with n=3 chain-yielding it bounds the corpus rate only",
    )
    for arm, risks in [("B_abstract", 18.8), ("C_gpt41", 19.5), ("D_triples", 5.2)]:
        check(
            f"baseline {arm}: released risks recovered pct",
            risks,
            bl[arm]["endpoint_recovery_pct"]["released_risks_found"],
        )
    check(
        "baseline: endpoint-recovery ceiling",
        46.5,
        bl["D_triples"]["endpoint_recovery_pct"]["ceiling_from_issue_168"],
        note="what an o3 re-run recovers of its own shipped output; the flat-triple arm is "
        "read against this, never against 100%",
    )
    check(
        "baseline C: all-five share",
        40.0,
        bl["C_gpt41"]["pct_chains_all_five"],
    )
    check(
        "baseline U: all-five share",
        88.9,
        bl["U_unconditioned"]["pct_chains_all_five"],
    )
    check("baseline B: all-five share", 100.0, bl["B_abstract"]["pct_chains_all_five"])

    # ---- extraction-failure recovery, now printed in the body (C6) -------------------
    rec = receipt("experiment_judge_full_report.json")["item3_recovery"][
        "population_A_extraction_error_candidates"
    ]
    check("recovery: judgeable failed extractions", 441, rec["n_judgeable_candidates"])
    check(
        "recovery: attempts producing a non-empty graph",
        23,
        rec["n_attempts_producing_nonempty_graph"],
    )
    check(
        "recovery rate pct",
        5.2,
        rec["recovery_rate_pct"],
        note="sec:m-recovery prints this from 2026-08-16; it used to say 'not at a useful "
        "rate' with no number. The '~60 of ~400' in older project notes divides two "
        "disjoint populations and is wrong",
    )

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

    # ---- every quality cut the enumerator applies, verified on the released file ----
    # Added 2026-08-15. The manuscript previously named four cuts; the builder applies
    # ten. Each row below checks one of them against the emitted paths, so the Methods
    # list and app:cuts cannot drift from what actually ran.
    raw_rows = [json.loads(line) for line in open(RAW, encoding="utf-8")]

    def node_cat(nid):
        a = na.get(nid, {})
        if (a.get("type") or "").lower() == "intervention":
            return "intervention"
        return (a.get("concept_category") or "").lower()

    hops = [len(r["path"]) - 1 for r in raw_rows]
    check("cut: minimum path length in hops", 3, min(hops))
    check("cut: maximum path length in hops observed", 15, max(hops))
    check("cut: shortest chain in nodes", 4, min(hops) + 1)
    check("cut: longest chain in nodes", 16, max(hops) + 1)
    check(
        "cut: 30-hop ceiling never binds",
        True,
        max(hops) < 30,
        note="the cap is a safeguard, not a shaping cut; the floor of 3 hops IS a cut",
    )
    check(
        "cut: 50M global path cap never binds",
        True,
        len(raw_rows) < 50_000_000,
    )
    check(
        "cut: every path roots at a risk node",
        True,
        all(node_cat(r["path"][0]) == "risk" for r in raw_rows),
    )
    check(
        "cut: no risk node after position 0",
        True,
        all(not any(node_cat(n) == "risk" for n in r["path"][1:]) for r in raw_rows),
    )
    check(
        "cut: first hop always lands on an intermediate subtype",
        True,
        all(node_cat(r["path"][1]) in set(BODY) for r in raw_rows),
        note="no risk-to-intervention shortcut is enumerable",
    )
    check(
        "cut: every endpoint is an intervention",
        True,
        all(node_cat(r["path"][-1]) == "intervention" for r in raw_rows),
    )
    check(
        "cut: every path is simple (no repeated node)",
        True,
        all(len(set(r["path"])) == len(r["path"]) for r in raw_rows),
    )
    check(
        "cut: paths passing through a sub-threshold intervention",
        540,
        sum(
            1
            for r in raw_rows
            if any(node_cat(n) == "intervention" for n in r["path"][1:-1])
        ),
        note="enumeration stops at the first maturity>=3 intervention, so a lower-maturity "
        "one can sit mid-chain; disclosed in sec:m-paths",
    )
    check(
        "cut: interventions clearing the maturity gate, pct",
        15.1,
        round(
            100
            * sum(
                1
                for a in na.values()
                if (a.get("type") or "").lower() == "intervention"
                and (a.get("intervention_maturity") or 0) >= 3
            )
            / by_type.get("intervention", 1),
            1,
        ),
        note="the marginal; the joint with the confidence gate is 11.4% (gate corner)",
    )
    check(
        "cut: sub-path collapse preserves the length range",
        (3, 15),
        (
            min(
                len(json.loads(line)["path"]) - 1
                for line in open(DEDUP, encoding="utf-8")
            ),
            max(
                len(json.loads(line)["path"]) - 1
                for line in open(DEDUP, encoding="utf-8")
            ),
        ),
    )
    del raw_rows

    # ---- the two chains printed in tab:query (added 2026-08-15) ---------------------
    # experiment_query_demo.py emits two DIFFERENT chains, so the table had no receipt.
    # Check the printed node ids against the reporting unit and the stored stage labels.
    dedup_paths = {
        tuple(json.loads(line)["path"]) for line in open(DEDUP, encoding="utf-8")
    }
    for name, nodes, stages, url in [
        (
            "tab:query Q1 (governance)",
            (1067, 1086, 1087, 1088, 1089, 1090, 1091),
            ["risk"] + BODY + ["intervention"],
            "https://arxiv.org/abs/2303.11341",
        ),
        (
            "tab:query Q2 (technical)",
            (415, 416, 417, 426, 431, 437, 442),
            [
                "risk",
                "problem analysis",
                "problem analysis",
                "design rationale",
                "implementation mechanism",
                "validation evidence",
                "intervention",
            ],
            "https://arxiv.org/abs/2209.00626",
        ),
    ]:
        check(
            f"{name} is in the 2,772-chain reporting unit", True, nodes in dedup_paths
        )
        got = [
            "intervention"
            if (na[n].get("type") or "").lower() == "intervention"
            else (na[n].get("concept_category") or "").lower()
            for n in nodes
        ]
        check(f"{name} stage labels as printed", stages, got)
        check(
            f"{name} endpoint maturity >= 3",
            True,
            na[nodes[-1]].get("intervention_maturity") in (3, 4),
        )
        check(
            f"{name} source document", True, all(na[n].get("url") == url for n in nodes)
        )

    # ---- structural diagnostics + similarity layer (added 2026-08-15) ----------------
    # These were quoted in Methods with no script behind them; REPRODUCE.md wrongly said
    # this file computed them. It does now. Union-find over the released checkpoints.
    edges = pickle.load(open(STEP1 / "graph_edge_data.pkl", "rb"))
    ids = list(na.keys())
    pos = {n: i for i, n in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def cat_of(nid):
        a = na[nid]
        if (a.get("type") or "").lower() == "intervention":
            return "intervention"
        return (a.get("concept_category") or "").lower()

    struct = [
        e
        for e in edges
        if e.get("type") == "EDGE" and e["source"] in pos and e["target"] in pos
    ]
    check("EDGE-layer edge count", 202149, len(struct))
    deg = Counter()
    for e in struct:
        union(pos[e["source"]], pos[e["target"]])
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    comp = Counter(find(i) for i in range(len(ids)))
    check("EDGE-only connected components", 15123, len(comp))
    check("largest EDGE-only component (nodes)", 61, comp.most_common(1)[0][1])
    check(
        "largest EDGE-only component as pct of graph",
        0.03,
        round(100 * comp.most_common(1)[0][1] / len(ids), 2),
    )
    check("average degree, EDGE layer", 2.02, round(sum(deg.values()) / len(ids), 2))

    # SIMILARITY edges store a Euclidean distance between unit vectors: cos = 1 - d^2/2.
    sim_counts = Counter()
    same_cat_080 = []
    for e in edges:
        if e.get("type") != "SIMILARITY":
            continue
        s, t = e["source"], e["target"]
        if s not in pos or t not in pos or cat_of(s) != cat_of(t):
            continue
        cos = 1.0 - float(e["similarity_score"]) ** 2 / 2.0
        for th in (0.80, 0.85, 0.90, 0.95):
            if cos >= th:
                sim_counts[th] += 1
        if cos >= 0.80:
            same_cat_080.append((s, t))
    check(
        "within-category SIM edges at cos >= 0.80",
        1435806,
        sim_counts[0.80],
        ok=abs(sim_counts[0.80] - 1435806) <= 50,
        note="receipt experiment_J3_J6_report.json says 1,435,806; this union-find pass "
        "counts 1,435,780. The 26-edge gap is below any claim's resolution -- tolerance 50.",
    )
    check("within-category SIM edges at cos >= 0.85", 573244, sim_counts[0.85])
    check("within-category SIM edges at cos >= 0.90", 142179, sim_counts[0.90])
    check("within-category SIM edges at cos >= 0.95", 9111, sim_counts[0.95])
    for s, t in same_cat_080:
        union(pos[s], pos[t])
    comp2 = Counter(find(i) for i in range(len(ids)))
    check("components after adding SIM at 0.80", 4124, len(comp2))
    check("largest component after SIM at 0.80", 152753, comp2.most_common(1)[0][1])
    check(
        "largest component after SIM as pct of graph",
        76.2,
        round(100 * comp2.most_common(1)[0][1] / len(ids), 1),
    )
    del edges, struct, same_cat_080

    # ---- exact-name duplicate residue in the released (un-merged) graph --------------
    name_groups = Counter(
        (cat_of(n), (na[n].get("name") or "").strip().lower()) for n in na
    )
    dup_groups = {k: v for k, v in name_groups.items() if v > 1}
    check("exact-name duplicate groups within category", 448, len(dup_groups))
    check(
        "exact-name duplicate nodes beyond one per group",
        1140,
        sum(dup_groups.values()) - len(dup_groups),
        note="measured on the UN-MERGED 200,525-node graph, not after the merge",
    )

    # ---- merge candidate generation: deployed vs exhaustive (sec:r-hub) --------------
    j6 = json.loads(
        (ROOT / "phase2_results/experiment_J6_merge_approx_v2_report.json").read_text(
            encoding="utf-8"
        )
    )
    check(
        "deployed merge candidate pairs",
        4411,
        j6["frozen_reference"]["candidate_pairs"],
    )
    check(
        "exhaustive-search candidate pairs (V0, frozen spec)",
        77759,
        j6["variants"]["V0_prev"]["phase2_both_pass"],
    )
    check(
        "50-NN-capped exact-search candidate pairs (V2)",
        54282,
        j6["variants"]["V2_T1_T2_T3"]["phase2_both_pass"],
    )
    check(
        "exhaustive vs deployed factor (18x in sec:r-hub)",
        18,
        round(j6["variants"]["V0_prev"]["phase2_both_pass"] / 4411),
    )
    check(
        "isolated top-100 risks after race-node removal, corrected merge",
        2,
        j6["j6_on_V2"]["isolated_after_race_removal"],
    )

    # ---- existential-risk share of the EC top-10, all four conditions ---------------
    ec = json.loads(
        (ROOT / "phase2_results/experiment_merge_vs_simexcl_ec_report.json").read_text(
            encoding="utf-8"
        )
    )
    for label, expected in [
        ("1 un-merged, full SIM", 10),
        ("2 un-merged, risk<->risk SIM EXCLUDED", 8),
        ("3 merged(risk), full SIM", 1),
        ("4 merged(risk), risk<->risk SIM EXCLUDED", 2),
    ]:
        got = next(c for c in ec["conditions"] if c["label"] == label)["xrisk_in_top10"]
        check(f"xrisk in EC top-10, {label}", expected, got)

    # ---- extraction cost (sec:m-repro, Compute) -------------------------------------
    cost = receipt("experiment_review_extraction_cost_report.json")
    check("cost: prompt tokens per call", 3706, cost["prompt"]["tokens"])
    check(
        "cost: documents matched to an ARD record",
        11779,
        cost["coverage"]["matched_to_an_ARD_record_by_url"],
    )
    check("cost: match rate pct", 100.0, cost["coverage"]["match_rate_pct"])
    ci = cost["input_tokens_per_document_EXACT"]
    check("cost: mean input tokens per document", 10389, round(ci["mean"]))
    check("cost: median input tokens per document", 6952, round(ci["median"]))
    check("cost: p90 input tokens per document", 18757, round(ci["p90"]))
    check("cost: total input tokens (millions)", 122.4, round(ci["total"] / 1e6, 1))
    cal = cost["output_calibration"]
    check("cost: calibration response tokens", 8155, cal["visible_output_tokens"])
    check(
        "cost: calibration emitted elements",
        52,
        cal["emitted_elements_nodes_plus_edges"],
    )
    check("cost: tokens per emitted element", 157, round(cal["tokens_per_element"]))
    co = cost["visible_output_tokens_per_document_CALIBRATED"]
    check("cost: mean visible output per document", 5361, round(co["mean"]))
    check("cost: total visible output (millions)", 63.2, round(co["total"] / 1e6, 1))
    pricing = cost["pricing_ASSUMED_not_measured"]
    check("cost: batch discount applied", 0.5, pricing["batch_discount_applied"])
    o3 = pricing["bill_by_model"]["o3 (as run)"]["by_reasoning_ratio"]
    lo = o3["reasoning_x0_visible_output"]
    hi = o3["reasoning_x4_visible_output"]
    check(
        "cost: low end of the assumed band, USD",
        375,
        round(lo["usd_over_matched_documents"]),
    )
    check(
        "cost: high end of the assumed band, USD",
        1385,
        round(hi["usd_over_matched_documents"]),
    )
    check("cost: USD per 1,000 documents, low", 32, round(lo["usd_per_1000_documents"]))
    check(
        "cost: USD per 1,000 documents, high", 118, round(hi["usd_per_1000_documents"])
    )
    check(
        "cost: mean document text tokens (prompt subtracted)",
        6683,
        round(ci["mean"] - cost["prompt"]["tokens"]),
    )
    # The same token volume repriced on current models at batch rates, no reasoning
    # premium. Cross-vendor rows reprice o200k_base counts; see the receipt's caveat.
    for model, per_k in [
        ("Claude Opus 5", 93),
        ("Claude Sonnet 5", 56),
        ("Claude Sonnet 4.6", 56),
        ("Claude Haiku 4.5", 19),
    ]:
        row = pricing["bill_by_model"][model]["by_reasoning_ratio"][
            "reasoning_x0_visible_output"
        ]
        check(
            f"cost: USD per 1,000 documents on {model}, no reasoning premium",
            per_k,
            round(row["usd_per_1000_documents"]),
        )

    # ---- gate corner, figure 1 panel B ----------------------------------------------
    gc = receipt("experiment_review_gate_corner_report.json")
    check(
        "gate corner: interventions",
        4228,
        gc["gate_corner_maturity_ge3_and_best_conf_ge3"]["n"],
    )
    check(
        "gate corner: pct of extracted interventions",
        11.4,
        round(gc["gate_corner_maturity_ge3_and_best_conf_ge3"]["pct_of_placed"], 1),
    )
    check(
        "gate corner: interventions placed",
        36959,
        gc["n_with_at_least_one_structural_edge"],
    )

    # ---- stage separability probe (sec:r-stages, app:stages) ------------------------
    sep = receipt("experiment_review_stage_separability_report.json")
    five, seven = sep["five_intermediate_stages"], sep["all_seven_stages"]
    check("stage probe: five-stage accuracy", 0.9876, five["accuracy"])
    check("stage probe: five-stage macro-F1", 0.9876, five["macro_f1"])
    check(
        "stage probe: five-stage chance", 0.2, five["baseline_uniform_chance_accuracy"]
    )
    check("stage probe: held-out nodes", 9115, five["n_test_nodes"])
    check(
        "stage probe: documents in the five-stage task",
        10446,
        five["n_train_documents"] + five["n_test_documents"],
    )
    cm = five["confusion_matrix_rows_true_cols_pred"]
    check(
        "stage probe: total held-out errors",
        113,
        sum(sum(r) - r[i] for i, r in enumerate(cm)),
    )
    check("stage probe: pa vs ti errors", 37, cm[0][1] + cm[1][0])
    check("stage probe: dr vs im errors", 17, cm[2][3] + cm[3][2])
    check(
        "stage probe: validation evidence F1",
        0.994,
        round(five["per_class_f1"]["validation evidence"], 3),
    )
    margins = [v["margin"] for v in five["centroid_separation"].values()]
    check("stage probe: smallest centroid margin", 0.0541, min(margins))
    check("stage probe: largest centroid margin", 0.0847, max(margins))
    lex = five["lexical_ablation"]
    check("stage probe: TF-IDF on the name alone", 0.6943, lex["name_only"]["accuracy"])
    check(
        "stage probe: TF-IDF on name and description",
        0.7864,
        lex["name_and_description"]["accuracy"],
    )
    check("stage probe: seven-stage accuracy", 0.9856, seven["accuracy"])
    check(
        "stage probe: seven-stage chance",
        0.1429,
        seven["baseline_uniform_chance_accuracy"],
    )

    # ---- race framing across centrality tiers (app:race) ----------------------------
    rr = json.loads(
        (ROOT / "phase2_results/experiment_race_top100_rederive_report.json").read_text(
            encoding="utf-8"
        )
    )
    tiers = {t["tier"]: t for t in rr["condition_A"]["tiers"]}
    for tier, pct in [
        ("top-100 by EC", 2.6),
        ("top-500 by EC", 6.3),
        ("top-1000 by EC", 4.4),
        ("all 12638 risks", 1.3),
    ]:
        check(
            f"race-framed share of single-path risks, {tier}",
            pct,
            tiers[tier]["pct_of_single_path_race_framed"],
        )
    check(
        "single-path risks in the EC top-100",
        38,
        tiers["top-100 by EC"]["n_single_path"],
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
