"""
rev8_diagnostics_simthresh_and_edgeonly.py — answer Q1+Q2 from 2026-05-06.

Q1: At SIM thresholds 0.80, 0.85, 0.90, how many body-body edges exist? What
    fraction is intra-stop-word, cross-stop-word, non-stop-word? Do lower
    thresholds surface non-stop-word cross-paper connectivity?

Q2: For EDGE-only (zero SIM hops) strict paths, what is the stop-word
    inclusion rate? Do EDGE-only paths give clean non-stop-word frozensets?
    Are HDBSCAN body clusters alone sufficient for cross-paper grouping
    without needing SIM hops?
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"

PKL_FILE = STEP1 / "cluster_memberships_rev8_sim0.9.pkl"
ATTR_FILE = STEP1 / "graph_node_attributes.pkl"
EDGE_FILE = STEP1 / "graph_edge_data.pkl"
PATHS_FILE = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"

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
LOGICAL_ORDER = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]
STOP_WORDS = {
    ("problem_analysis", "1"),
    ("theoretical_insight", "0"),
    ("theoretical_insight", "1"),
    ("design_rationale", "69"),
    ("problem_analysis", "111"),
}


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def sig_to_str_logical(sig):
    """Format frozenset in logical-chain subtype order."""
    by_st = defaultdict(list)
    for st, cid in sig:
        by_st[st].append(cid)
    parts = []
    for st in LOGICAL_ORDER:
        if st in by_st:
            for cid in sorted(
                by_st[st], key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 0
            ):
                parts.append(f"{SUBTYPE_PREFIX[st]}:{cid}")
    return " & ".join(parts)


def main():
    print("=" * 70)
    print("rev8 Q1+Q2 — SIM threshold scan + EDGE-only frozenset analysis")
    print("=" * 70)

    print("\nLoading cluster memberships ...")
    with open(PKL_FILE, "rb") as f:
        cm = pickle.load(f)
    node_to_body = {}
    cluster_size = defaultdict(Counter)
    for key, members in cm.items():
        subtype = str(key[2])
        cid = str(key[4])
        if cid in {"-1", "noise"}:
            continue
        for nid in members:
            nid_int = int(nid)
            if subtype in BODY_SUBTYPES:
                node_to_body[nid_int] = (subtype, cid)
                cluster_size[subtype][cid] += 1
    n_body = len(node_to_body)
    print(f"  body-clustered nodes: {n_body:,}")
    print(
        f"  body cluster IDs by subtype: "
        f"{ {st: len(cluster_size[st]) for st in LOGICAL_ORDER} }"
    )

    print("\nLoading edges (1.7M total)...")
    with open(EDGE_FILE, "rb") as f:
        edge_data = pickle.load(f)

    # Per-threshold tabulation
    thresholds = [0.80, 0.85, 0.90]
    bb_edges_by_t = {t: [] for t in thresholds}
    for e in edge_data:
        if str(e.get("type", "")).upper() != "SIMILARITY":
            continue
        s = e.get("similarity_score")
        if s is None:
            continue
        cos = cos_sim_from_score(s)
        u, v = int(e["source"]), int(e["target"])
        if u not in node_to_body or v not in node_to_body:
            continue
        for t in thresholds:
            if cos >= t:
                bb_edges_by_t[t].append((u, v))

    # ─── Q1: SIM threshold density scan ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Q1 — SIM threshold body-body edge density scan")
    print("=" * 70)
    print(
        f"  {'thresh':<7} {'#bb_edges':>11} {'#nodes_w_nbrs':>15} "
        f"{'%nodes_connected':>18} {'#non_SW_edges':>15} {'#non_SW_nodes':>15}"
    )
    for t in thresholds:
        edges = bb_edges_by_t[t]
        nodes_with_nbr = set()
        non_sw_edges = 0
        non_sw_nodes = set()
        intra_sw = 0
        cross_sw = 0
        non_sw = 0
        sw_to_non = 0
        for u, v in edges:
            cu = node_to_body[u]
            cv = node_to_body[v]
            nodes_with_nbr.add(u)
            nodes_with_nbr.add(v)
            u_is_sw = cu in STOP_WORDS
            v_is_sw = cv in STOP_WORDS
            if u_is_sw and v_is_sw:
                if cu == cv:
                    intra_sw += 1
                else:
                    cross_sw += 1
            elif (not u_is_sw) and (not v_is_sw):
                non_sw += 1
                non_sw_edges += 1
                non_sw_nodes.add(u)
                non_sw_nodes.add(v)
            else:  # one is stop-word, one is not
                sw_to_non += 1
        print(
            f"  {t:<7.2f} {len(edges):>11,} {len(nodes_with_nbr):>15,} "
            f"{len(nodes_with_nbr) / n_body * 100:>17.1f}% "
            f"{non_sw_edges:>15,} {len(non_sw_nodes):>15,}"
        )
        print(
            f"          breakdown: intra-SW={intra_sw:,}  cross-SW={cross_sw:,}  "
            f"SW↔nonSW={sw_to_non:,}  nonSW-nonSW={non_sw:,}"
        )

    # Connected components in non-SW SIM>=0.85 subgraph
    print("\n  Non-SW connected-component analysis (SIM>=0.85, body-body only):")
    for t in [0.80, 0.85, 0.90]:
        non_sw_adj = defaultdict(set)
        for u, v in bb_edges_by_t[t]:
            cu = node_to_body[u]
            cv = node_to_body[v]
            if cu in STOP_WORDS or cv in STOP_WORDS:
                continue
            non_sw_adj[u].add(v)
            non_sw_adj[v].add(u)
        # BFS components
        seen = set()
        components = []
        for start in list(non_sw_adj.keys()):
            if start in seen:
                continue
            stack = [start]
            comp = set()
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                comp.add(n)
                stack.extend(non_sw_adj[n] - seen)
            components.append(comp)
        comp_sizes = sorted([len(c) for c in components], reverse=True)
        print(f"    {t}: {len(components):>4} components,  top sizes: {comp_sizes[:8]}")
        # Cross-cluster (non-SW) pair-edges?
        cross_cluster_edges = 0
        for u, v in bb_edges_by_t[t]:
            cu = node_to_body[u]
            cv = node_to_body[v]
            if cu in STOP_WORDS or cv in STOP_WORDS:
                continue
            if cu != cv:
                cross_cluster_edges += 1
        print(f"        cross-cluster non-SW edges: {cross_cluster_edges}")

    # ─── Q2: EDGE-only path frozenset analysis ────────────────────────────────
    print("\n" + "=" * 70)
    print("Q2 — EDGE-only (zero SIM hops) strict path frozenset analysis")
    print("=" * 70)

    # Build risk and intervention maps
    node_to_risk = {}
    node_to_interv = {}
    for key, members in cm.items():
        subtype = str(key[2])
        cid = str(key[4])
        if cid in {"-1", "noise"}:
            continue
        for nid in members:
            nid_int = int(nid)
            if subtype == "risk":
                node_to_risk[nid_int] = cid
            elif subtype == "intervention":
                node_to_interv[nid_int] = cid

    edge_only_pair_paths = defaultdict(int)  # (rcid, icid) -> n_paths
    edge_only_sigs = Counter()
    sig_to_examples = defaultdict(list)
    sig_to_pairs = defaultdict(set)
    sig_with_sw = 0
    sig_no_sw = []
    n_total = 0
    n_strict = 0

    print(f"  reading {PATHS_FILE.name} (filtering to EDGE-only)...")
    with open(PATHS_FILE) as f:
        for line in f:
            obj = json.loads(line)
            seq = obj.get("path") or obj.get("node_id_sequence")
            if not seq or len(seq) < 3:
                continue
            n_total += 1
            edge_types = obj.get("edge_types", [])
            if any(et == "SIMILARITY" or et == "SIM" for et in edge_types):
                continue
            # EDGE-only path
            path = [int(x) for x in seq]
            r_n, i_n = path[0], path[-1]
            if r_n not in node_to_risk or i_n not in node_to_interv:
                continue
            body = path[1:-1]
            if any(b not in node_to_body for b in body):
                continue
            n_strict += 1
            sig = frozenset(node_to_body[b] for b in body)
            edge_only_sigs[sig] += 1
            edge_only_pair_paths[(node_to_risk[r_n], node_to_interv[i_n])] += 1
            sig_to_pairs[sig].add((node_to_risk[r_n], node_to_interv[i_n]))
            if len(sig_to_examples[sig]) < 1:
                sig_to_examples[sig].append(path)

    print(f"\n  EDGE-only strict paths: {n_strict:,}")
    print(f"  Unique R-I cluster pairs covered: {len(edge_only_pair_paths):,}")
    print(f"  Unique frozensets: {len(edge_only_sigs):,}")
    if edge_only_sigs:
        sizes = [len(s) for s in edge_only_sigs]
        print(
            f"  Frozenset size distribution: "
            f"min={min(sizes)}, median={sorted(sizes)[len(sizes) // 2]}, "
            f"max={max(sizes)}, mean={sum(sizes) / len(sizes):.1f}"
        )
        sig_with_sw = sum(1 for s in edge_only_sigs if any(c in STOP_WORDS for c in s))
        sig_no_sw = [s for s in edge_only_sigs if not any(c in STOP_WORDS for c in s)]
        print(
            f"  Frozensets containing AT LEAST ONE stop-word: {sig_with_sw}/{len(edge_only_sigs)} "
            f"({sig_with_sw / len(edge_only_sigs) * 100:.1f}%)"
        )
        print(f"  Frozensets NOT containing any stop-word: {len(sig_no_sw)}")

        # Stop-word inclusion in PATHS (not just frozensets)
        path_with_sw = 0
        for s, n in edge_only_sigs.items():
            if any(c in STOP_WORDS for c in s):
                path_with_sw += n
        print(
            f"  EDGE-only paths containing stop-word: "
            f"{path_with_sw}/{n_strict} ({path_with_sw / n_strict * 100:.1f}%)"
        )

        # Pair coverage
        n_pairs_with_5plus = sum(1 for p, n in edge_only_pair_paths.items() if n >= 5)
        n_pairs_with_2plus = sum(1 for p, n in edge_only_pair_paths.items() if n >= 2)
        print(f"  R-I cluster pairs with >=5 paths: {n_pairs_with_5plus}")
        print(f"  R-I cluster pairs with >=2 paths: {n_pairs_with_2plus}")
        print(
            f"  R-I cluster pairs with exactly 1 path: "
            f"{sum(1 for p, n in edge_only_pair_paths.items() if n == 1)}"
        )

        print("\n  Top 15 EDGE-only frozensets by path count:")
        print(
            f"  {'#paths':<7} {'#R-I pairs':<11} {'sig_size':<9} {'has_SW':<7} {'sig (logical order)':<60}"
        )
        for sig, n in edge_only_sigs.most_common(15):
            has_sw = any(c in STOP_WORDS for c in sig)
            n_pairs = len(sig_to_pairs[sig])
            print(
                f"  {n:<7} {n_pairs:<11} {len(sig):<9} "
                f"{'YES' if has_sw else 'no':<7} {sig_to_str_logical(sig)[:60]}"
            )

        print("\n  Top 15 EDGE-only NON-STOP-WORD frozensets by path count:")
        print(
            f"  {'#paths':<7} {'#R-I pairs':<11} {'sig_size':<9} {'sig (logical order)':<60}"
        )
        non_sw_sigs = sorted(
            ((s, edge_only_sigs[s]) for s in sig_no_sw), key=lambda x: -x[1]
        )[:15]
        for sig, n in non_sw_sigs:
            n_pairs = len(sig_to_pairs[sig])
            print(
                f"  {n:<7} {n_pairs:<11} {len(sig):<9} {sig_to_str_logical(sig)[:60]}"
            )

    print("\nDONE.")


if __name__ == "__main__":
    main()
