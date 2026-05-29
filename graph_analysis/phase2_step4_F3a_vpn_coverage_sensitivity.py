"""
phase2_step4_F3a_vpn_coverage_sensitivity.py [rev8 — Task #7 sensitivity]

For a given F3 cluster_memberships_rev8_<suffix>.pkl, sweep the path retention
criterion: how many R-I pairs (and paths) survive at varying body-node-coverage
thresholds?

Constraint set per scenario:
  - Risk endpoint must be assigned to a non-noise risk cluster (100% strict)
  - Intervention endpoint must be assigned to a non-noise intervention cluster
    AND have intervention_maturity >= 3 (100% strict)
  - Body nodes: at least body_min_coverage fraction must be mapped to non-noise
    body subtype clusters (variable)

Inputs:
  --paths-file      F2v4 hopwise (or BFS-shortest) jsonl
  --memberships-pkl cluster_memberships_rev8_<suffix>.pkl
  --output-suffix   suffix for outputs
  --coverage-thresholds  comma list, default 1.00,0.90,0.80,0.70,0.60,0.50,0.40

Outputs:
  step4_cluster_tables/vpn_coverage_sensitivity_<suffix>.csv
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
STEP1_DIR = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4_DIR = ROOT / "phase2_results/step4_finalanalysis"
OUT_TABLES = STEP4_DIR / "step4_cluster_tables"

BODY_SUBTYPES = {
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
}
CATEGORY_NORMALIZE = {
    "problem analysis": "problem_analysis",
    "theoretical insight": "theoretical_insight",
    "design rationale": "design_rationale",
    "implementation mechanism": "implementation_mechanism",
    "validation evidence": "validation_evidence",
    "risk": "risk",
}


def normalize_category(raw):
    if raw is None:
        return ""
    return CATEGORY_NORMALIZE.get(str(raw).strip(), str(raw).strip())


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-file", required=True)
    ap.add_argument("--memberships-pkl", required=True)
    ap.add_argument("--output-suffix", required=True)
    ap.add_argument(
        "--coverage-thresholds", default="1.00,0.90,0.80,0.70,0.60,0.50,0.40"
    )
    ap.add_argument("--no-consim1", action="store_true")
    ap.add_argument("--sim-threshold", type=float, default=0.9)
    args = ap.parse_args()

    print("Loading PKL artifacts...")
    with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)

    sim_edge_set = set()
    if not args.no_consim1:
        with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
            edge_data = pickle.load(f)
        for e in edge_data:
            if str(e.get("type", "")).upper() != "SIMILARITY":
                continue
            score = e.get("similarity_score")
            if score is None or cos_sim_from_score(score) < args.sim_threshold:
                continue
            try:
                s, t = int(e["source"]), int(e["target"])
                sim_edge_set.add((min(s, t), max(s, t)))
            except (ValueError, TypeError):
                pass
        del edge_data

    def max_consec_sim(path):
        max_run = run = 0
        for i in range(len(path) - 1):
            a, b = int(path[i]), int(path[i + 1])
            if (min(a, b), max(a, b)) in sim_edge_set:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return max_run

    print(f"Loading memberships: {args.memberships_pkl}")
    with open(args.memberships_pkl, "rb") as f:
        cm = pickle.load(f)

    # Build node_id -> (subtype, cluster_id) map
    nid_to_subtype_cid = {}
    for key, members in cm.items():
        # key: (variant, mode, subtype, algo, cluster_id)
        subtype = str(key[2])
        cid = str(key[4])
        for nid in members:
            nid_to_subtype_cid[int(nid)] = (subtype, cid)
    print(f"  {len(nid_to_subtype_cid):,} nodes have a cluster ID")

    coverage_thresholds = [float(x) for x in args.coverage_thresholds.split(",")]
    coverage_thresholds = sorted(coverage_thresholds, reverse=True)

    # Per-scenario tally
    def _empty():
        return {
            "n_paths": 0,
            "ri_pairs": set(),
            "risk_clusters": set(),
            "interv_clusters": set(),
        }

    scenario_stats = {ct: _empty() for ct in coverage_thresholds}
    scenario_stats["risk_only"] = _empty()
    scenario_stats["intervention_only"] = _empty()
    scenario_stats["risk_and_intervention_only"] = _empty()
    # "at most N body-nodes unmapped" scenarios (path-length-aware)
    unmapped_allowances = [0, 1, 2, 3]
    for u in unmapped_allowances:
        scenario_stats[f"unmapped_<={u}"] = _empty()

    n_total = n_qual = 0
    print(f"Reading {args.paths_file} ...")
    with open(args.paths_file) as f:
        for line in f:
            obj = json.loads(line)
            n_total += 1
            path = [int(x) for x in obj["path"]]
            interv_id = path[-1]
            risk_id = path[0]
            mat = node_attrs.get(interv_id, {}).get("intervention_maturity", 0)
            try:
                mat_i = int(mat) if mat is not None else 0
            except Exception:
                mat_i = 0
            if mat_i < 3:
                continue
            if (not args.no_consim1) and max_consec_sim(path) > 1:
                continue
            n_qual += 1

            # Check risk cluster membership
            risk_in_cluster = (
                int(risk_id) in nid_to_subtype_cid
                and nid_to_subtype_cid[int(risk_id)][0] == "risk"
            )
            # Check intervention cluster membership (mat>=3 already filtered)
            interv_in_cluster = (
                int(interv_id) in nid_to_subtype_cid
                and nid_to_subtype_cid[int(interv_id)][0] == "intervention"
            )
            # Body coverage: count body nodes with cluster ID in some body subtype
            body_nodes = path[1:-1]
            body_n_total = len(body_nodes)
            body_n_mapped = 0
            for bnid in body_nodes:
                attrs = node_attrs.get(int(bnid), {})
                cat_norm = normalize_category(attrs.get("concept_category"))
                if cat_norm not in BODY_SUBTYPES:
                    continue
                if (
                    int(bnid) in nid_to_subtype_cid
                    and nid_to_subtype_cid[int(bnid)][0] == cat_norm
                ):
                    body_n_mapped += 1
            body_cov = body_n_mapped / body_n_total if body_n_total else 1.0

            ri_pair = (int(risk_id), int(interv_id))
            risk_cid = (
                nid_to_subtype_cid.get(int(risk_id), (None, None))[1]
                if risk_in_cluster
                else None
            )
            interv_cid = (
                nid_to_subtype_cid.get(int(interv_id), (None, None))[1]
                if interv_in_cluster
                else None
            )

            def _add(scn):
                scenario_stats[scn]["n_paths"] += 1
                scenario_stats[scn]["ri_pairs"].add(ri_pair)
                if risk_cid is not None:
                    scenario_stats[scn]["risk_clusters"].add(risk_cid)
                if interv_cid is not None:
                    scenario_stats[scn]["interv_clusters"].add(interv_cid)

            if risk_in_cluster:
                _add("risk_only")
            if interv_in_cluster:
                _add("intervention_only")
            if risk_in_cluster and interv_in_cluster:
                _add("risk_and_intervention_only")
                # Body coverage thresholds (% rule)
                for ct in coverage_thresholds:
                    if body_cov >= ct - 1e-9:
                        _add(ct)
                # Unmapped-allowance rule (path-length-aware)
                body_n_unmapped = body_n_total - body_n_mapped
                for u in unmapped_allowances:
                    if body_n_unmapped <= u:
                        _add(f"unmapped_<={u}")

    def _row(label, key, body_cov_min=None):
        s = scenario_stats[key]
        return {
            "scenario": label,
            "body_coverage_min": body_cov_min,
            "n_paths_retained": s["n_paths"],
            "n_unique_ri_pairs": len(s["ri_pairs"]),
            "n_unique_risk_clusters": len(s["risk_clusters"]),
            "n_unique_interv_clusters": len(s["interv_clusters"]),
            "path_retention_rate": round(s["n_paths"] / n_qual if n_qual else 0, 4),
        }

    rows = []
    for ct in coverage_thresholds:
        rows.append(_row(f"R+I strict, body_cov>={ct:.2f}", ct, body_cov_min=ct))
    rows.append(_row("risk endpoint only (body N/A)", "risk_only"))
    rows.append(_row("intervention endpoint only (body N/A)", "intervention_only"))
    rows.append(_row("R+I strict, no body requirement", "risk_and_intervention_only"))
    for u in unmapped_allowances:
        rows.append(_row(f"R+I strict, body unmapped <={u}", f"unmapped_<={u}"))

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    out_csv = OUT_TABLES / f"vpn_coverage_sensitivity_{args.output_suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv}")
    print(f"Total qualifying paths (mat>=3 + consim1): {n_qual:,} of {n_total:,}")

    # Emit canonical 100%-strict (R + body + I all in clusters) jsonl
    canonical_jsonl = PATHS_DIR / f"vpn_strict_RIbody_{args.output_suffix}.jsonl"
    n_emitted = 0
    with open(args.paths_file) as fin, open(canonical_jsonl, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            path = [int(x) for x in obj["path"]]
            interv_id = path[-1]
            risk_id = path[0]
            mat = node_attrs.get(interv_id, {}).get("intervention_maturity", 0)
            try:
                mat_i = int(mat) if mat is not None else 0
            except Exception:
                mat_i = 0
            if mat_i < 3:
                continue
            if (not args.no_consim1) and max_consec_sim(path) > 1:
                continue
            r_in = (
                int(risk_id) in nid_to_subtype_cid
                and nid_to_subtype_cid[int(risk_id)][0] == "risk"
            )
            i_in = (
                int(interv_id) in nid_to_subtype_cid
                and nid_to_subtype_cid[int(interv_id)][0] == "intervention"
            )
            if not (r_in and i_in):
                continue
            body_nodes = path[1:-1]
            ok = True
            for bnid in body_nodes:
                cat_norm = normalize_category(
                    node_attrs.get(int(bnid), {}).get("concept_category")
                )
                if cat_norm not in BODY_SUBTYPES:
                    ok = False
                    break
                lbl_pair = nid_to_subtype_cid.get(int(bnid))
                if lbl_pair is None or lbl_pair[0] != cat_norm:
                    ok = False
                    break
            if not ok:
                continue
            fout.write(json.dumps(obj) + "\n")
            n_emitted += 1
    print(f"\nWritten canonical 100%-strict (R+body+I): {canonical_jsonl}")
    print(f"  paths emitted = {n_emitted:,}")


if __name__ == "__main__":
    main()
