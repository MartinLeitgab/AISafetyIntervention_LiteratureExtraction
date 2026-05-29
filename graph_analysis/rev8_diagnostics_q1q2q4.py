"""
rev8_diagnostics_q1q2q4.py — answers user questions Q1, Q2, Q4 from 2026-05-06.

Q1: How many unique R-cluster × I-cluster pairs and how many strict paths
    in the 100%-strict pathway dataset?

Q2: What does n_total_paper_sources >= 3 actually count?

Q4: Per body subtype: cluster size distribution, single-cluster-per-node
    verification, fraction of nodes in pa:1 / ti:0 / ti:1 / dr:69 / pa:111.
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4 = ROOT / "phase2_results/step4_finalanalysis"

PATHS_FILE = STEP4 / "step4_paths/representative_pathways_hopwise_sim0.9.jsonl"
PKL_FILE = STEP1 / "cluster_memberships_rev8_sim0.9.pkl"
ATTR_FILE = STEP1 / "graph_node_attributes.pkl"

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
STOP_WORDS = {
    ("problem_analysis", "1"),
    ("theoretical_insight", "0"),
    ("theoretical_insight", "1"),
    ("design_rationale", "69"),
    ("problem_analysis", "111"),
}


def main():
    print("=" * 70)
    print("rev8 Q1+Q2+Q4 diagnostics")
    print("=" * 70)

    print("\nLoading cluster_memberships PKL ...")
    with open(PKL_FILE, "rb") as f:
        cm = pickle.load(f)
    print(f"  {len(cm):,} cluster records")

    print("Loading graph_node_attributes.pkl ...")
    with open(ATTR_FILE, "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  {len(node_attrs):,} nodes total in attrs")

    # ─── Build per-subtype node→cluster maps ──────────────────────────────────
    subtype_node_to_cluster = defaultdict(dict)  # subtype → {nid: cid}
    cluster_size = defaultdict(Counter)  # subtype → Counter(cid → count)
    multi_cluster_nodes = defaultdict(
        set
    )  # subtype → set of nids that appear in >1 cid
    for key, members in cm.items():
        # PKL key schema: (variant, mode, subtype, algo, cluster_id)
        subtype = str(key[2])
        cid = str(key[4])
        if cid in {"-1", "noise"}:
            continue
        for nid in members:
            nid_int = int(nid)
            prev = subtype_node_to_cluster[subtype].get(nid_int)
            if prev is not None and prev != cid:
                multi_cluster_nodes[subtype].add(nid_int)
            subtype_node_to_cluster[subtype][nid_int] = cid
            cluster_size[subtype][cid] += 1

    # ─── Q4: cluster size distribution per subtype ────────────────────────────
    print("\n" + "=" * 70)
    print("Q4 — Cluster size distribution per subtype (non-noise clusters)")
    print("=" * 70)
    print(
        f"{'subtype':<28} {'#nodes':>8} {'#clusters':>10} {'top1':>10} {'top2':>10} "
        f"{'top3':>10} {'top1%':>8} {'top1+2+3%':>10} {'#multi':>7}"
    )
    for subtype in sorted(subtype_node_to_cluster.keys()):
        sizes = cluster_size[subtype]
        total = sum(sizes.values())
        n_clusters = len(sizes)
        topN = sizes.most_common(3)
        t1 = topN[0][1] if len(topN) > 0 else 0
        t2 = topN[1][1] if len(topN) > 1 else 0
        t3 = topN[2][1] if len(topN) > 2 else 0
        n_multi = len(multi_cluster_nodes[subtype])
        print(
            f"{subtype:<28} {total:>8,} {n_clusters:>10} {t1:>10,} {t2:>10,} "
            f"{t3:>10,} {t1 / total * 100:>7.2f}% {(t1 + t2 + t3) / total * 100:>9.2f}% {n_multi:>7,}"
        )

    print("\nTop-5 cluster IDs per subtype (cid -> size):")
    for subtype in sorted(subtype_node_to_cluster.keys()):
        sizes = cluster_size[subtype]
        prefix = SUBTYPE_PREFIX.get(subtype, subtype[:2])
        top5 = sizes.most_common(5)
        line = ", ".join(f"{prefix}:{cid} ({n})" for cid, n in top5)
        print(f"  {subtype:<28} {line}")

    # ─── Q4: stop-word fraction ───────────────────────────────────────────────
    print("\nStop-word fraction per subtype (frac of subtype nodes in named cluster):")
    for st, cid in sorted(STOP_WORDS):
        sizes = cluster_size[st]
        total = sum(sizes.values())
        n_in = sizes.get(cid, 0)
        prefix = SUBTYPE_PREFIX.get(st, st[:2])
        print(
            f"  {prefix}:{cid:<4} {n_in:>5,} / {total:>6,} = {n_in / total * 100:>5.2f}%"
        )

    # ─── Q1: read strict paths and compute (R, I) cluster pair stats ──────────
    print("\n" + "=" * 70)
    print("Q1 — R-cluster × I-cluster combinations + total strict paths")
    print("=" * 70)
    risk_node_to_cid = subtype_node_to_cluster["risk"]
    interv_node_to_cid = subtype_node_to_cluster["intervention"]

    print(f"  risk-clustered nodes  : {len(risk_node_to_cid):,}")
    print(f"  intv-clustered nodes  : {len(interv_node_to_cid):,}")
    print(f"  unique risk cluster IDs: {len(set(risk_node_to_cid.values())):,}")
    print(f"  unique intv cluster IDs: {len(set(interv_node_to_cid.values())):,}")
    print(
        f"  theoretical max R×I    : "
        f"{len(set(risk_node_to_cid.values())) * len(set(interv_node_to_cid.values())):,}"
    )

    n_paths = 0
    ri_cluster_pairs = set()  # (risk_cid, interv_cid)
    ri_node_pairs = set()  # (risk_node, interv_node)
    risk_cids_seen = set()
    interv_cids_seen = set()
    print(f"\n  Reading {PATHS_FILE.name} ...")
    with open(PATHS_FILE) as f:
        for line in f:
            obj = json.loads(line)
            seq = obj.get("path") or obj.get("node_id_sequence")
            if not seq or len(seq) < 3:
                continue
            r_n = int(seq[0])
            i_n = int(seq[-1])
            r_c = risk_node_to_cid.get(r_n)
            i_c = interv_node_to_cid.get(i_n)
            if r_c is None or i_c is None:
                continue
            n_paths += 1
            ri_cluster_pairs.add((r_c, i_c))
            ri_node_pairs.add((r_n, i_n))
            risk_cids_seen.add(r_c)
            interv_cids_seen.add(i_c)

    print(f"\n  Total strict paths     : {n_paths:,}")
    print(f"  Unique (R-cluster, I-cluster) pairs : {len(ri_cluster_pairs):,}")
    print(f"  Unique (R-node, I-node) pairs       : {len(ri_node_pairs):,}")
    print(f"  Distinct risk clusters seen in strict paths : {len(risk_cids_seen):,}")
    print(f"  Distinct intv clusters seen in strict paths : {len(interv_cids_seen):,}")
    print(
        f"  Avg paths per R-cluster × I-cluster pair    : "
        f"{n_paths / max(len(ri_cluster_pairs), 1):,.1f}"
    )
    print(
        f"  Avg paths per R-node × I-node pair          : "
        f"{n_paths / max(len(ri_node_pairs), 1):,.1f}"
    )

    # ─── Q2: demonstrate n_total_paper_sources semantics ──────────────────────
    print("\n" + "=" * 70)
    print("Q2 — n_total_paper_sources semantics")
    print("=" * 70)
    print(
        "  CODE in F1 rebuild (lines 182-186):\n"
        "      for nid in path:\n"
        "          u_n = _url(nid)\n"
        "          if u_n: sig_total_sources[sig].add(u_n)\n"
        "  → counts UNION of node-URLs across ALL nodes of ALL paths matching that\n"
        "    frozenset signature. NOT 'frozenset appears in N papers'.\n"
        "    A single 6-node path traversing 6 distinct papers via SIM bridges → already 6."
    )

    # Spot-check on top frozenset by n_paths to demonstrate
    print("\n  Empirical check on sim=0.9 hopwise output:")
    fam_csv = (
        STEP4 / "step4_cluster_tables/optionB_cooccurrence_families_hopwise_sim0.9.csv"
    )
    if fam_csv.exists():
        import pandas as pd

        df = pd.read_csv(fam_csv)
        print(f"    rows in CSV                       : {len(df):,}")
        print(
            f"    rows with n_total_paper_sources>=3: {(df['n_total_paper_sources'] >= 3).sum():,}"
        )
        print(
            f"    rows with n_total_paper_sources==1: {(df['n_total_paper_sources'] == 1).sum():,}"
        )
        print(
            f"    rows with n_total_paper_sources==2: {(df['n_total_paper_sources'] == 2).sum():,}"
        )
        print(
            "    rows with n_paths>=5 & n_total_paper_sources>=signature_size+2 (likely "
            "single-paper, all-distinct-paper-bridge case):"
        )
        df["size_p2"] = df["signature_size"] + 2
        print(f"      {(df['n_total_paper_sources'] >= df['size_p2']).sum():,}")
        print("\n  Top 5 by n_total_paper_sources (likely SIM-bridge accumulators):")
        for _, r in (
            df.sort_values("n_total_paper_sources", ascending=False).head(5).iterrows()
        ):
            print(
                f"    fam {int(r['family_id']):>4}  papers={int(r['n_total_paper_sources']):>4}"
                f"  RI_pairs={int(r['n_distinct_RI_pairs']):>4}  "
                f"paths={int(r['n_paths']):>6}  sig_size={int(r['signature_size']):>2}  "
                f"| {str(r['signature_str'])[:58]}"
            )
        print(
            "\n  Frozensets w/ n_paths>=5 and n_total_paper_sources<3 (truly under-sourced):"
        )
        sub = df[df["n_total_paper_sources"] < 3]
        print(f"    count: {len(sub):,}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
