"""
rev8_diagnostics_q1q3_simhops.py — answers Q1+Q3 from 2026-05-06 follow-up.

Q1: SIM-hop distribution per strict path. How many paths and how many distinct
    frozensets fall into 0 / 1 / 2 / >=3 SIM-hop buckets? If we cap SIM hops at
    1 or 2, how many strict paths and frozensets remain?

Q3: Why do stop-word clusters appear in nearly every frozenset if they are
    each only ~3-6% of subtype nodes? Compute empirical inclusion rate and
    contrast with uniform-distribution baseline.
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4 = ROOT / "phase2_results/step4_finalanalysis"

# Source hopwise_v4 file has edge_types (representative_pathways drops them)
PATHS_FILE = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_sim0.9.jsonl"
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
STOP_WORDS = [
    ("problem_analysis", "1"),
    ("theoretical_insight", "0"),
    ("theoretical_insight", "1"),
    ("design_rationale", "69"),
    ("problem_analysis", "111"),
]


def main():
    print("=" * 70)
    print("rev8 Q1+Q3 SIM-hop + stop-word inclusion diagnostics")
    print("=" * 70)

    print("\nLoading PKL ...")
    with open(PKL_FILE, "rb") as f:
        cm = pickle.load(f)
    # subtype-aware node->cluster
    node_to_body = {}
    node_to_risk = {}
    node_to_interv = {}
    subtype_size = defaultdict(Counter)
    for key, members in cm.items():
        subtype = str(key[2])
        cid = str(key[4])
        if cid in {"-1", "noise"}:
            continue
        for nid in members:
            nid_int = int(nid)
            if subtype in BODY_SUBTYPES:
                node_to_body[nid_int] = (subtype, cid)
            elif subtype == "risk":
                node_to_risk[nid_int] = cid
            elif subtype == "intervention":
                node_to_interv[nid_int] = cid
            subtype_size[subtype][cid] += 1
    print(f"  body-clustered nodes: {len(node_to_body):,}")
    print(f"  risk-clustered     : {len(node_to_risk):,}")
    print(f"  interv-clustered   : {len(node_to_interv):,}")

    # ─── Read paths with edge_types, apply strict filter, bucket by SIM count ──
    print(f"\nReading {PATHS_FILE.name} ...")
    sim_count_paths = Counter()  # n_sim_hops -> n_paths
    sim_count_frozensets = defaultdict(set)  # n_sim_hops -> set of frozensets
    stop_word_path_count = Counter()  # (subtype, cid) -> n_strict_paths_containing
    sig_size_by_sim_count = defaultdict(list)
    n_total = 0
    n_strict = 0
    all_strict_frozensets = set()
    body_subtype_path_count = Counter()  # (subtype, cid) -> n_strict_paths
    sig_to_simhop_min = defaultdict(lambda: 999)  # min SIM hops for a frozenset

    with open(PATHS_FILE) as f:
        for line in f:
            obj = json.loads(line)
            seq = obj.get("path") or obj.get("node_id_sequence")
            if not seq or len(seq) < 3:
                continue
            n_total += 1
            path = [int(x) for x in seq]
            r_n, i_n = path[0], path[-1]
            if r_n not in node_to_risk or i_n not in node_to_interv:
                continue
            body = path[1:-1]
            if any(b not in node_to_body for b in body):
                continue
            n_strict += 1

            edge_types = obj.get("edge_types", [])
            n_sim = sum(1 for e in edge_types if e == "SIMILARITY" or e == "SIM")
            sim_count_paths[n_sim] += 1
            sig = frozenset(node_to_body[b] for b in body)
            sim_count_frozensets[n_sim].add(sig)
            sig_size_by_sim_count[n_sim].append(len(sig))
            all_strict_frozensets.add(sig)
            if n_sim < sig_to_simhop_min[sig]:
                sig_to_simhop_min[sig] = n_sim

            seen_in_path = set()
            for b in body:
                tup = node_to_body[b]
                seen_in_path.add(tup)
            for tup in seen_in_path:
                body_subtype_path_count[tup] += 1
            for sw in STOP_WORDS:
                if sw in seen_in_path:
                    stop_word_path_count[sw] += 1

    # ─── Q1: SIM-hop distribution ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Q1 — SIM-hop distribution across 100%-strict paths")
    print("=" * 70)
    print(f"  total paths read     : {n_total:,}")
    print(f"  strict paths         : {n_strict:,}")
    print(f"  unique strict frozensets: {len(all_strict_frozensets):,}")

    print(
        f"\n  {'SIM-hops':<10} {'#paths':>10} {'%paths':>8} "
        f"{'#frozensets':>12} {'%frozensets':>12} {'avg_sig_size':>12}"
    )
    for k in sorted(sim_count_paths.keys()):
        n_p = sim_count_paths[k]
        n_s = len(sim_count_frozensets[k])
        sizes = sig_size_by_sim_count[k]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        print(
            f"  {k:<10} {n_p:>10,} {n_p / n_strict * 100:>7.2f}% "
            f"{n_s:>12,} {n_s / len(all_strict_frozensets) * 100:>11.2f}% "
            f"{avg_size:>12.1f}"
        )

    # ─── Cap analysis: paths and frozensets remaining at SIM<=K ───────────────
    print("\n  Cumulative cap analysis:")
    print(
        f"  {'cap (max SIM)':<14} {'#paths':>10} {'%paths':>8} "
        f"{'#frozensets':>12} {'%frozensets':>12}"
    )
    for cap in [0, 1, 2, 3]:
        n_p = sum(sim_count_paths[k] for k in range(cap + 1))
        sigs = set()
        for k in range(cap + 1):
            sigs |= sim_count_frozensets[k]
        print(
            f"  <={cap:<11} {n_p:>10,} {n_p / n_strict * 100:>7.2f}% "
            f"{len(sigs):>12,} {len(sigs) / len(all_strict_frozensets) * 100:>11.2f}%"
        )

    # ─── Q1 — frozensets achievable with EDGE-only paths ──────────────────────
    edge_only_sigs = sim_count_frozensets[0]
    print(
        f"\n  EDGE-only (zero SIM hops) frozensets: {len(edge_only_sigs):,}  "
        f"({len(edge_only_sigs) / len(all_strict_frozensets) * 100:.2f}% of all)"
    )
    # How many of those are also reachable via SIM-using paths?
    n_edge_only_only = sum(1 for sig, m in sig_to_simhop_min.items() if m == 0)
    print(
        f"  Frozensets with min_SIM_hops==0: {n_edge_only_only:,} "
        f"(some may also appear in SIM-using paths)"
    )

    # ─── Q3: Stop-word inclusion rates ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Q3 — Why stop-words appear in nearly every frozenset")
    print("=" * 70)
    avg_sig_size = (
        sum(sum(sig_size_by_sim_count[k]) for k in sig_size_by_sim_count) / n_strict
    )
    print(f"\n  Avg body length per strict path: {avg_sig_size:.2f}")
    print("\n  Empirical: fraction of STRICT PATHS containing each stop-word")
    print("  (i.e., the path traverses at least one node in that cluster)")
    print(
        f"\n  {'cluster':<10} {'subtype share':>14} {'#strict paths':>15} "
        f"{'%strict paths':>15} {'baseline%(*)':>12}"
    )
    print(
        f"  {'(in subtype)':<10} {'(% of nodes)':>14} {'containing':>15} "
        f"{'containing':>15} {'see below':>12}"
    )
    for sw in STOP_WORDS:
        st, cid = sw
        prefix = SUBTYPE_PREFIX[st]
        n_in_st = subtype_size[st][cid]
        st_total = sum(subtype_size[st].values())
        st_pct = n_in_st / st_total * 100
        n_paths = stop_word_path_count[sw]
        pct_paths = n_paths / n_strict * 100
        # Naive baseline: P(path of length L contains a node from cluster of share p)
        # ~= 1 - (1 - p_subtype)^L_avg  if uniform-routing
        # where p_subtype = p_within_subtype * p(slot_is_this_subtype)
        # Approx avg slot composition: assume body slots are evenly across 5 subtypes
        # so p_slot_in_subtype = 0.2 * p_within_subtype = 0.2 * st_pct/100
        p_slot = 0.2 * (st_pct / 100)
        baseline_pct = (1 - (1 - p_slot) ** avg_sig_size) * 100
        print(
            f"  {prefix}:{cid:<6} {st_pct:>13.2f}% {n_paths:>15,} "
            f"{pct_paths:>14.2f}% {baseline_pct:>11.2f}%"
        )
    print(
        "\n  (*) baseline% = uniform-routing P(at least one slot lands in this "
        "cluster) = 1 - (1 - 0.2*subtype_share)^avg_body_length"
    )
    print("      Compares observed % to a counter-factual where every body slot")
    print("      randomly picks a subtype (1/5) and within subtype picks a cluster")
    print("      proportional to cluster size.")

    print("\n  Top-10 most-included BODY clusters across strict paths (any subtype):")
    print(
        f"  {'cluster':<10} {'size':>6} {'subtype share':>14} "
        f"{'#strict paths':>15} {'%strict paths':>14}"
    )
    top = body_subtype_path_count.most_common(10)
    for (st, cid), n_paths in top:
        prefix = SUBTYPE_PREFIX[st]
        n_in_st = subtype_size[st][cid]
        st_total = sum(subtype_size[st].values())
        st_pct = n_in_st / st_total * 100
        pct_paths = n_paths / n_strict * 100
        print(
            f"  {prefix}:{cid:<6} {n_in_st:>6} {st_pct:>13.2f}% "
            f"{n_paths:>15,} {pct_paths:>13.2f}%"
        )

    print("\nDONE.")


if __name__ == "__main__":
    main()
