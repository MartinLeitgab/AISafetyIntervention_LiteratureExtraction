"""
phase2_step4_F1_rev8_rebuild.py [rev8 — Task #6]

Parameterized analog of phase2_step4_F1_consim1_custom_rebuild.py. Reads:
  - any path JSONL whose schema is {"path": [int,...], "categories": [str,...]}
  - any rev8 cluster_memberships PKL (variant=rev8_vpn_post, algo=hdbscan)

Applies STRICT filter inline (R + every body node + I all in non-noise clusters of
their respective subtypes via the rev8 PKL), builds frozenset signatures, and
emits:
  step4_paths/representative_pathways_<suffix>.jsonl
  step4_cluster_tables/optionB_cooccurrence_families_<suffix>.csv
  step4_connectivity/ri_triplets_<suffix>.csv
  step4_connectivity/<suffix>_summary.txt

Usage:
  python phase2_step4_F1_rev8_rebuild.py \
      --paths graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_sim0.9.jsonl \
      --memberships graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/cluster_memberships_rev8_sim0.9.pkl \
      --output-suffix hopwise_sim0.9
"""

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
STEP1_DIR = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4_DIR = ROOT / "phase2_results/step4_finalanalysis"
OUT_PATHS = STEP4_DIR / "step4_paths"
OUT_TABLES = STEP4_DIR / "step4_cluster_tables"
OUT_CONN = STEP4_DIR / "step4_connectivity"
for d in [OUT_PATHS, OUT_TABLES, OUT_CONN]:
    d.mkdir(parents=True, exist_ok=True)

BODY_SUBTYPES = {
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
}
SUBTYPE_PREFIX = {
    "problem_analysis": "pa",
    "theoretical_insight": "ti",
    "design_rationale": "dr",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="Path JSONL")
    ap.add_argument(
        "--memberships", required=True, help="cluster_memberships_rev8_*.pkl"
    )
    ap.add_argument("--output-suffix", required=True, help="output filename suffix")
    ap.add_argument(
        "--maturity-min",
        type=int,
        default=0,
        help="min intervention_maturity (0=disabled)",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=5,
        help="min n_paths per frozenset to include in CSV (default 5)",
    )
    args = ap.parse_args()

    print("=" * 70)
    print(f"Phase 2 Step 4 rev8 rebuild — suffix={args.output_suffix}")
    print("=" * 70)
    print(f"  paths       = {args.paths}")
    print(f"  memberships = {args.memberships}")
    print(f"  maturity    = >= {args.maturity_min} (0=disabled)")

    # ─── Load cluster memberships + node attrs ────────────────────────────────
    t0 = time.time()
    print("\nLoading cluster_memberships PKL ...")
    with open(args.memberships, "rb") as f:
        cm = pickle.load(f)
    print(f"  {len(cm):,} cluster records ({time.time() - t0:.1f}s)")

    print("Loading graph_node_attributes.pkl ...")
    t1 = time.time()
    with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  {len(node_attrs):,} nodes ({time.time() - t1:.1f}s)")

    # ─── Build node→cluster maps from rev8 PKL schema ─────────────────────────
    node_to_body_cluster = {}  # nid -> (subtype, cid)
    node_to_risk = {}  # nid -> cid
    node_to_interv = {}  # nid -> cid
    for key, members in cm.items():
        subtype = str(key[2])
        cid = str(key[4])
        for nid in members:
            nid_int = int(nid)
            if subtype in BODY_SUBTYPES:
                node_to_body_cluster[nid_int] = (subtype, cid)
            elif subtype == "risk":
                node_to_risk[nid_int] = cid
            elif subtype == "intervention":
                node_to_interv[nid_int] = cid
    print(f"  body-cluster nodes : {len(node_to_body_cluster):,}")
    print(f"  risk-cluster nodes : {len(node_to_risk):,}")
    print(f"  intv-cluster nodes : {len(node_to_interv):,}")

    # ─── Read paths and apply STRICT filter inline ────────────────────────────
    print(f"\nReading {args.paths} ...")
    t2 = time.time()
    qual_paths = []
    total = 0
    n_strict = 0
    n_dropped_endpoints = 0
    n_dropped_body = 0
    n_dropped_maturity = 0
    with open(args.paths) as f:
        for line in f:
            obj = json.loads(line)
            seq = obj.get("path") or obj.get("node_id_sequence")
            if not seq or len(seq) < 3:
                continue
            path = [int(x) for x in seq]
            total += 1
            risk_node, interv_node = path[0], path[-1]
            if risk_node not in node_to_risk or interv_node not in node_to_interv:
                n_dropped_endpoints += 1
                continue
            body = path[1:-1]
            if any(b not in node_to_body_cluster for b in body):
                n_dropped_body += 1
                continue
            if args.maturity_min > 0:
                mat = int(
                    node_attrs.get(interv_node, {}).get("intervention_maturity", 0) or 0
                )
                if mat < args.maturity_min:
                    n_dropped_maturity += 1
                    continue
            n_strict += 1
            qual_paths.append(
                {
                    "path": path,
                    "categories": obj.get("categories", []),
                }
            )
    print(f"  total paths : {total:,}")
    print(f"  strict      : {n_strict:,} ({n_strict / max(total, 1) * 100:.2f}%)")
    print(f"  dropped — endpoints not clustered: {n_dropped_endpoints:,}")
    print(f"  dropped — body not fully clustered: {n_dropped_body:,}")
    print(f"  dropped — maturity filter        : {n_dropped_maturity:,}")
    print(f"  ({time.time() - t2:.1f}s)")

    # ─── Build frozenset signatures + triplet counts ──────────────────────────
    print("\nBuilding frozenset signatures ...")
    t3 = time.time()
    sig_counts = Counter()
    sig_intv_sources = defaultdict(set)  # n_intervention_sources (legacy n_sources)
    sig_risk_sources = defaultdict(set)  # n_risk_sources
    sig_total_sources = defaultdict(set)  # n_total_paper_sources (any node any path)
    sig_RI_pairs = defaultdict(set)  # n_distinct_RI_pairs (R-node, I-node) tuples
    sig_top_subtype_counts = defaultdict(Counter)
    triplet_counts = Counter()  # (risk_cid, sig, interv_cid) -> n
    r2f_counts = Counter()
    f2i_counts = Counter()

    def _url(nid):
        u = str(node_attrs.get(nid, {}).get("url", ""))
        return u if u and u not in {"", "None", "nan"} else None

    for p in qual_paths:
        path = p["path"]
        body = path[1:-1]
        sig = frozenset(node_to_body_cluster[n] for n in body)
        if not sig:
            continue
        sig_counts[sig] += 1
        risk_node = path[0]
        interv_node = path[-1]

        # n_intervention_sources
        u_i = _url(interv_node)
        if u_i:
            sig_intv_sources[sig].add(u_i)
        # n_risk_sources
        u_r = _url(risk_node)
        if u_r:
            sig_risk_sources[sig].add(u_r)
        # n_total_paper_sources — any URL across any node in path
        for nid in path:
            u_n = _url(nid)
            if u_n:
                sig_total_sources[sig].add(u_n)
        # n_distinct_RI_pairs — (R-node, I-node) tuples
        sig_RI_pairs[sig].add((risk_node, interv_node))

        for nid in body:
            cat = (
                str(node_attrs.get(nid, {}).get("concept_category", ""))
                .strip()
                .replace(" ", "_")
            )
            if cat in BODY_SUBTYPES:
                sig_top_subtype_counts[sig][cat] += 1
        risk_cid = node_to_risk[path[0]]
        interv_cid = node_to_interv[path[-1]]
        triplet_counts[(risk_cid, sig, interv_cid)] += 1
        r2f_counts[(risk_cid, sig)] += 1
        f2i_counts[(sig, interv_cid)] += 1

    print(f"  unique frozensets : {len(sig_counts):,}  ({time.time() - t3:.1f}s)")
    large_sigs = sorted(
        [s for s, c in sig_counts.items() if c >= args.min_n],
        key=lambda s: -sig_counts[s],
    )
    print(f"  frozensets w/ n_paths>={args.min_n}: {len(large_sigs):,}")

    sig_to_fam = {sig: i for i, sig in enumerate(large_sigs)}

    def sig_to_str(sig):
        parts = []
        for st, cid in sig:
            prefix = SUBTYPE_PREFIX.get(st, st[:2])
            parts.append(f"{prefix}:{cid}")
        return " & ".join(sorted(parts))

    # ─── Output 1: cooccurrence families CSV ──────────────────────────────────
    fam_rows = []
    for sig in large_sigs:
        fam_rows.append(
            {
                "family_id": sig_to_fam[sig],
                "n_paths": sig_counts[sig],
                "n_distinct_RI_pairs": len(sig_RI_pairs[sig]),
                "n_total_paper_sources": len(sig_total_sources[sig]),
                "n_risk_sources": len(sig_risk_sources[sig]),
                "n_intervention_sources": len(sig_intv_sources[sig]),
                "n_sources": len(
                    sig_intv_sources[sig]
                ),  # legacy column for back-compat
                "signature_size": len(sig),
                "signature_str": sig_to_str(sig),
                "top_subtypes": str(dict(sig_top_subtype_counts[sig].most_common(5))),
            }
        )
    fam_df = pd.DataFrame(fam_rows)
    fam_path = OUT_TABLES / f"optionB_cooccurrence_families_{args.output_suffix}.csv"
    fam_df.to_csv(fam_path, index=False)
    print(f"\nWritten: {fam_path}")

    # ─── Output 2: ri_triplets CSV ────────────────────────────────────────────
    trip_rows = []
    for (risk_cid, sig, interv_cid), n_trip in triplet_counts.items():
        if sig not in sig_to_fam:
            continue
        trip_rows.append(
            {
                "risk_cid": int(risk_cid),
                "bfamily_id": sig_to_fam[sig],
                "n_paths_r2c": r2f_counts[(risk_cid, sig)],
                "interv_cid": int(interv_cid),
                "n_paths_c2i": f2i_counts[(sig, interv_cid)],
                "n_triplet_paths": n_trip,
                "risk_name": "",
                "chain_name": "",
                "interv_name": "",
            }
        )
    trip_df = (
        pd.DataFrame(trip_rows).sort_values("n_triplet_paths", ascending=False)
        if trip_rows
        else pd.DataFrame()
    )
    trip_path = OUT_CONN / f"ri_triplets_{args.output_suffix}.csv"
    trip_df.to_csv(trip_path, index=False)
    print(f"Written: {trip_path} ({len(trip_df):,} rows)")

    # ─── Output 3: representative_pathways_<suffix>.jsonl ─────────────────────
    qpath_file = OUT_PATHS / f"representative_pathways_{args.output_suffix}.jsonl"
    with open(qpath_file, "w") as f:
        for p in qual_paths:
            f.write(
                json.dumps(
                    {
                        "node_id_sequence": p["path"],
                        "categories": p["categories"],
                    }
                )
                + "\n"
            )
    print(f"Written: {qpath_file} ({len(qual_paths):,} paths)")

    # ─── Output 4: summary text ───────────────────────────────────────────────
    summary_path = OUT_CONN / f"{args.output_suffix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"rev8 rebuild summary — suffix={args.output_suffix}\n")
        f.write("=" * 60 + "\n")
        f.write(f"paths_input: {args.paths}\n")
        f.write(f"memberships: {args.memberships}\n")
        f.write(f"maturity_min: {args.maturity_min}\n")
        f.write(f"total_paths: {total}\n")
        f.write(f"strict_paths: {n_strict}\n")
        f.write(f"retention: {n_strict / max(total, 1):.4f}\n")
        f.write(f"node_to_body_cluster: {len(node_to_body_cluster)}\n")
        f.write(f"node_to_risk: {len(node_to_risk)}\n")
        f.write(f"node_to_interv: {len(node_to_interv)}\n")
        f.write(f"unique_frozensets: {len(sig_counts)}\n")
        f.write(f"frozensets_n_ge_5: {len(large_sigs)}\n")
        f.write(f"ri_triplet_rows: {len(trip_df)}\n")
    print(f"Written: {summary_path}")

    # ─── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"SUMMARY (rev8 rebuild — {args.output_suffix})")
    print("=" * 70)
    print(f"  total paths           : {total:,}")
    print(
        f"  strict paths          : {n_strict:,}  ({n_strict / max(total, 1) * 100:.2f}%)"
    )
    print(f"  unique frozensets     : {len(sig_counts):,}")
    print(f"  frozensets n>=5       : {len(large_sigs):,}")
    print(f"  ri_triplet rows       : {len(trip_df):,}")
    if len(fam_df) > 0:
        print("  Top 5 frozensets by n_distinct_RI_pairs:")
        for _, r in (
            fam_df.sort_values("n_distinct_RI_pairs", ascending=False)
            .head(5)
            .iterrows()
        ):
            print(
                f"    fam {int(r['family_id']):>4}  RI={int(r['n_distinct_RI_pairs']):>5}  "
                f"papers={int(r['n_total_paper_sources']):>3}  paths={int(r['n_paths']):>6}  "
                f"sig_size={int(r['signature_size']):>2}  | {r['signature_str'][:60]}"
            )
        print("  Top 5 frozensets by n_total_paper_sources:")
        for _, r in (
            fam_df.sort_values("n_total_paper_sources", ascending=False)
            .head(5)
            .iterrows()
        ):
            print(
                f"    fam {int(r['family_id']):>4}  papers={int(r['n_total_paper_sources']):>3}  "
                f"RI={int(r['n_distinct_RI_pairs']):>5}  paths={int(r['n_paths']):>6}  "
                f"sig_size={int(r['signature_size']):>2}  | {r['signature_str'][:60]}"
            )
        print(
            f"  Frozensets w/ n_total_paper_sources >= 3: "
            f"{(fam_df['n_total_paper_sources'] >= 3).sum():,}"
        )
    print("\nDONE.")


if __name__ == "__main__":
    main()
