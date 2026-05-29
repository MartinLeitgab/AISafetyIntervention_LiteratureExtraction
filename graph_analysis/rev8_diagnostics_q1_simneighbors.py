"""
rev8_diagnostics_q1_simneighbors.py — answer Q1 from 2026-05-06 follow-up.

For each stop-word cluster, look at all of its nodes, gather their SIM>=0.9
neighbors, and tabulate the cluster-distribution of those neighbors.

Key question: do the 95% paths through pa:111 happen because
  (a) pa:111 is an INTRA-CLUSTER attractor — most SIM>=0.9 neighbors of pa:111
      nodes are themselves in pa:111 (other-paper instances of "goal
      misalignment"), so once any path lands in pa:111 it stays bouncing
      between pa:111 instances via SIM bridges, OR
  (b) pa:111 is a CROSS-CLUSTER routing hub — its nodes have SIM>=0.9 edges
      into many DIFFERENT clusters, so paths from arbitrary other topics get
      pulled in.

If (a), pa:111 dominance is intra-cluster SIM amplification + DFS bouncing.
If (b), pa:111 nodes are genuinely close to many disparate concepts.

We also check the REVERSE: across ALL clustered body nodes, how many have at
least one SIM>=0.9 edge to a stop-word? That tells us how hard pa:111 is to
"avoid" while traversing the SIM>=0.9 graph.
"""

import pickle
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"

PKL_FILE = STEP1 / "cluster_memberships_rev8_sim0.9.pkl"
ATTR_FILE = STEP1 / "graph_node_attributes.pkl"
EDGE_FILE = STEP1 / "graph_edge_data.pkl"

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


def cos_sim_from_score(s):
    """Convert L2 distance score to cosine similarity."""
    return 1.0 - float(s) ** 2 / 2.0


def main():
    print("=" * 70)
    print("rev8 Q1 — SIM>=0.9 neighborhood of stop-word clusters")
    print("=" * 70)

    print("\nLoading cluster memberships ...")
    with open(PKL_FILE, "rb") as f:
        cm = pickle.load(f)

    # Build node -> (subtype, cid) map for body subtypes only
    node_to_body = {}
    cluster_members = defaultdict(list)  # (subtype, cid) -> [nids]
    for key, members in cm.items():
        subtype = str(key[2])
        cid = str(key[4])
        if cid in {"-1", "noise"}:
            continue
        for nid in members:
            nid_int = int(nid)
            if subtype in BODY_SUBTYPES:
                node_to_body[nid_int] = (subtype, cid)
                cluster_members[(subtype, cid)].append(nid_int)
    print(f"  body-clustered nodes: {len(node_to_body):,}")

    print("Loading edges (this is large — ~1.5M SIM edges) ...")
    with open(EDGE_FILE, "rb") as f:
        edge_data = pickle.load(f)
    print(f"  total edges: {len(edge_data):,}")

    # Build SIM>=0.9 undirected adjacency over body-clustered nodes
    sim_adj = defaultdict(list)  # nid -> [neighbor_nid]
    n_sim = 0
    n_sim_09 = 0
    n_sim_09_body = 0
    for e in edge_data:
        if str(e.get("type", "")).upper() != "SIMILARITY":
            continue
        n_sim += 1
        s = e.get("similarity_score")
        if s is None:
            continue
        if cos_sim_from_score(s) < 0.90:
            continue
        n_sim_09 += 1
        u, v = int(e["source"]), int(e["target"])
        if u in node_to_body and v in node_to_body:
            n_sim_09_body += 1
            sim_adj[u].append(v)
            sim_adj[v].append(u)
    print(f"  SIM edges total            : {n_sim:,}")
    print(f"  SIM>=0.9 edges             : {n_sim_09:,}")
    print(f"  SIM>=0.9 edges (both endpts in body cluster): {n_sim_09_body:,}")

    # ─── Per stop-word: cluster distribution of neighbors ─────────────────────
    print("\n" + "=" * 70)
    print("Stop-word neighbor cluster distribution (SIM>=0.9 only)")
    print("=" * 70)
    for sw in STOP_WORDS:
        prefix = SUBTYPE_PREFIX[sw[0]]
        name = f"{prefix}:{sw[1]}"
        members = cluster_members[sw]
        n_members = len(members)
        neighbor_clusters = Counter()
        intra = 0
        inter_clusters = 0
        unique_intra_partners = set()
        unique_inter_partners = set()
        member_with_inter_neighbor = 0
        for nid in members:
            nbrs = sim_adj.get(nid, [])
            has_inter = False
            for nb in nbrs:
                nb_cluster = node_to_body[nb]
                neighbor_clusters[nb_cluster] += 1
                if nb_cluster == sw:
                    intra += 1
                    unique_intra_partners.add(nb)
                else:
                    inter_clusters += 1
                    unique_inter_partners.add(nb)
                    has_inter = True
            if has_inter:
                member_with_inter_neighbor += 1
        total_nbrs = intra + inter_clusters
        avg_nbrs = total_nbrs / max(n_members, 1)
        print(f"\n  {name}  ({n_members} nodes)")
        print(f"    avg SIM>=0.9 neighbors per node: {avg_nbrs:.1f}")
        print(
            f"    intra-cluster edges (both endpts in {name}): {intra:,} "
            f"({intra / max(total_nbrs, 1) * 100:.1f}% of edges)"
        )
        print(
            f"    cross-cluster edges (other endpoint elsewhere): {inter_clusters:,} "
            f"({inter_clusters / max(total_nbrs, 1) * 100:.1f}% of edges)"
        )
        print(
            f"    unique intra-partners: {len(unique_intra_partners):,} of {n_members - 1} possible"
        )
        print(
            f"    unique cross-partners: {len(unique_inter_partners):,} body nodes outside {name}"
        )
        print(
            f"    {name} members with >=1 cross-cluster neighbor: "
            f"{member_with_inter_neighbor}/{n_members}"
        )
        # Top destination clusters
        non_self = [(c, n) for c, n in neighbor_clusters.items() if c != sw]
        non_self.sort(key=lambda x: -x[1])
        print("    Top-10 destination clusters (cross only):")
        for c, n in non_self[:10]:
            cl_name = f"{SUBTYPE_PREFIX.get(c[0], c[0][:2])}:{c[1]}"
            cl_size = len(cluster_members[c])
            unique_partners = len(
                {
                    nb
                    for nid in members
                    for nb in sim_adj.get(nid, [])
                    if node_to_body[nb] == c
                }
            )
            print(
                f"      {cl_name:<10} (size {cl_size:>3}): "
                f"{n:>4} edges, {unique_partners:>3} unique partner nodes"
            )

    # ─── Reverse direction: how many body clusters touch a stop-word? ─────────
    print("\n" + "=" * 70)
    print("Reverse: across ALL body-clustered nodes, who has a stop-word neighbor?")
    print("=" * 70)
    nodes_touching_each = {sw: 0 for sw in STOP_WORDS}
    nodes_touching_any = 0
    nodes_total = len(node_to_body)
    nodes_with_any_sim = 0
    cluster_touching_each = {sw: set() for sw in STOP_WORDS}
    sw_set = set(STOP_WORDS)
    for nid, my_cluster in node_to_body.items():
        nbrs = sim_adj.get(nid, [])
        if nbrs:
            nodes_with_any_sim += 1
        touched_any = False
        for nb in nbrs:
            nb_cluster = node_to_body[nb]
            if nb_cluster in sw_set and nb_cluster != my_cluster:
                nodes_touching_each[nb_cluster] += 1
                cluster_touching_each[nb_cluster].add(my_cluster)
                touched_any = True
        if touched_any:
            nodes_touching_any += 1
    print(f"  Body-clustered nodes total              : {nodes_total:,}")
    print(
        f"  Body nodes with >=1 SIM>=0.9 body neighbor: {nodes_with_any_sim:,} "
        f"({nodes_with_any_sim / nodes_total * 100:.1f}%)"
    )
    print(
        f"  Body nodes touching ANY stop-word        : {nodes_touching_any:,} "
        f"({nodes_touching_any / nodes_total * 100:.1f}% of all body-clustered)"
    )
    for sw in STOP_WORDS:
        prefix = SUBTYPE_PREFIX[sw[0]]
        name = f"{prefix}:{sw[1]}"
        n_n = nodes_touching_each[sw]
        n_c = len(cluster_touching_each[sw])
        print(
            f"  Body nodes outside {name:<10} touching {name:<10} via SIM>=0.9: "
            f"{n_n:>5,} ({n_n / nodes_total * 100:>5.2f}%) — across {n_c:>3} distinct other clusters"
        )

    print("\nDONE.")


if __name__ == "__main__":
    main()
