"""
rev8_hamming_families_and_intersim.py

Q1: Build Hamming-edit-distance mechanism family grouping on the 176 EDGE-only
    strict frozensets. Symmetric-difference distance:
       d(F1, F2) = |F1 XOR F2|
       d=0 same; d=1 one add/remove; d=2 one swap; d>=3 different family
    Two outputs:
       (A) connected-components families: each frozenset in exactly one family
       (B) soft-coverage families: each frozenset reports all family centers
           it lies within distance <=2 of (overlap analysis)

Q2: Inter-cluster cosine similarity distribution. For body clusters appearing
    in EDGE-only strict paths, compute pairwise centroid cosine sim. Report
    full distribution + identify near-duplicate clusters (centroid sim >0.7)
    that should be merged.
"""

import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4 = ROOT / "phase2_results/step4_finalanalysis"

PKL_FILE = STEP1 / "cluster_memberships_rev8_sim0.9.pkl"
ATTR_FILE = STEP1 / "graph_node_attributes.pkl"
COOC_CSV = (
    STEP4 / "step4_cluster_tables/optionB_cooccurrence_families_edge_only_all.csv"
)

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


def parse_signature_str(s):
    """Parse 'pa:65 & ti:31 & dr:53 & im:22 & va:36' -> frozenset."""
    out = set()
    inv = {v: k for k, v in SUBTYPE_PREFIX.items()}
    for part in s.split(" & "):
        if ":" not in part:
            continue
        prefix, cid = part.split(":", 1)
        if prefix in inv:
            out.add((inv[prefix], cid))
    return frozenset(out)


def sig_to_str_logical(sig):
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
    print("rev8 Hamming family grouping + inter-cluster sim distribution")
    print("=" * 70)

    # ─── Q1: Hamming family grouping ──────────────────────────────────────────
    print("\nLoading EDGE-only frozensets ...")
    df = pd.read_csv(COOC_CSV)
    df["sig"] = df["signature_str"].map(parse_signature_str)
    sigs = list(df["sig"])
    n_paths = list(df["n_paths"])
    n_RI = list(df["n_distinct_RI_pairs"])
    n_papers = list(df["n_total_paper_sources"])
    n = len(sigs)
    print(f"  {n} frozensets loaded")

    # Pairwise symmetric-difference distance
    print("\nComputing pairwise symmetric-difference distances ...")
    pair_dists = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = len(sigs[i] ^ sigs[j])
            pair_dists[i, j] = d
            pair_dists[j, i] = d
    flat = pair_dists[np.triu_indices(n, k=1)]
    print(f"  pairs: {len(flat):,}")
    print(f"  d=0 (identical): {(flat == 0).sum():,}")
    print(f"  d=1 (one add/remove): {(flat == 1).sum():,}")
    print(f"  d=2 (one swap): {(flat == 2).sum():,}")
    print(f"  d=3: {(flat == 3).sum():,}")
    print(f"  d=4: {(flat == 4).sum():,}")
    print(f"  d>=5: {(flat >= 5).sum():,}")
    print(
        f"  d distribution: min={flat.min()}, mean={flat.mean():.2f}, "
        f"median={int(np.median(flat))}, max={flat.max()}"
    )

    # (A) Connected components at d <= D_THRESH
    for D_THRESH in [1, 2, 3]:
        print(f"\n--- Connected components at sym-diff distance <= {D_THRESH} ---")
        adj = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                if pair_dists[i, j] <= D_THRESH:
                    adj[i].add(j)
                    adj[j].add(i)
        seen = set()
        comps = []
        for start in range(n):
            if start in seen:
                continue
            stack = [start]
            comp = []
            while stack:
                v = stack.pop()
                if v in seen:
                    continue
                seen.add(v)
                comp.append(v)
                stack.extend(adj[v] - seen)
            comps.append(sorted(comp))
        comps.sort(key=lambda c: -sum(n_paths[i] for i in c))
        print(f"  # families: {len(comps)}")
        print("  family-size distribution (#frozensets per family):")
        sizes = [len(c) for c in comps]
        size_hist = Counter(sizes)
        for sz in sorted(size_hist.keys(), reverse=True)[:10]:
            print(f"    size {sz:>3}: {size_hist[sz]:>4} families")
        non_singleton = sum(1 for s in sizes if s > 1)
        print(f"  multi-frozenset families (size > 1): {non_singleton}")
        print(f"  singleton families: {sum(1 for s in sizes if s == 1)}")
        # Top 10 families
        print("\n  Top 10 families by #paths covered:")
        print(
            f"  {'fam':<4} {'#frozensets':<11} {'#paths':<7} {'#R-I pairs':<11} {'#papers':<8} {'centroid frozenset (most-paths member)':<60}"
        )
        for fam_idx, comp in enumerate(comps[:10]):
            tot_paths = sum(n_paths[i] for i in comp)
            # Use the most-paths member as representative
            best = max(comp, key=lambda i: n_paths[i])
            tot_RI = sum(n_RI[i] for i in comp)  # upper bound
            tot_papers = sum(n_papers[i] for i in comp)  # upper bound
            print(
                f"  {fam_idx:<4} {len(comp):<11} {tot_paths:<7} <={tot_RI:<10} <={tot_papers:<7} "
                f"{sig_to_str_logical(sigs[best])[:60]}"
            )

    # (B) Soft coverage at d <= 2: each frozenset reports all neighbors
    print("\n--- Soft coverage analysis at d <= 2 ---")
    multi_membership_count = 0
    n_neighbors = []
    for i in range(n):
        nbrs = (pair_dists[i] <= 2) & (pair_dists[i] > 0)
        cnt = int(nbrs.sum())
        n_neighbors.append(cnt)
        if cnt > 1:
            multi_membership_count += 1
    print(
        f"  frozensets with 0 d<=2 neighbors (singletons): "
        f"{sum(1 for c in n_neighbors if c == 0)}"
    )
    print(
        f"  frozensets with exactly 1 d<=2 neighbor: "
        f"{sum(1 for c in n_neighbors if c == 1)}"
    )
    print(
        f"  frozensets with 2-4 d<=2 neighbors: "
        f"{sum(1 for c in n_neighbors if 2 <= c <= 4)}"
    )
    print(
        f"  frozensets with 5-9 d<=2 neighbors: "
        f"{sum(1 for c in n_neighbors if 5 <= c <= 9)}"
    )
    print(
        f"  frozensets with 10+ d<=2 neighbors: "
        f"{sum(1 for c in n_neighbors if c >= 10)}"
    )
    print(f"  Mean neighbors per frozenset: {np.mean(n_neighbors):.2f}")

    # Save canonical (d<=2) family table
    print("\nWriting canonical (d<=2) family table ...")
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if pair_dists[i, j] <= 2:
                adj[i].add(j)
                adj[j].add(i)
    seen = set()
    comps = []
    for start in range(n):
        if start in seen:
            continue
        stack = [start]
        comp = []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack.extend(adj[v] - seen)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: -sum(n_paths[i] for i in c))

    rows = []
    for fam_idx, comp in enumerate(comps):
        tot_paths = sum(n_paths[i] for i in comp)
        tot_RI = sum(n_RI[i] for i in comp)
        tot_papers = sum(n_papers[i] for i in comp)
        best = max(comp, key=lambda i: n_paths[i])
        rows.append(
            {
                "family_id": fam_idx,
                "n_frozensets": len(comp),
                "n_paths_total": tot_paths,
                "n_RI_pairs_total_upperbound": tot_RI,
                "n_papers_total_upperbound": tot_papers,
                "core_signature": sig_to_str_logical(sigs[best]),
                "all_signatures": " ;; ".join(
                    sig_to_str_logical(sigs[i]) for i in comp
                ),
            }
        )
    fam_df = pd.DataFrame(rows)
    out_path = STEP4 / "step4_cluster_tables/hamming_families_edge_only_d2.csv"
    fam_df.to_csv(out_path, index=False)
    print(f"  written: {out_path.name} ({len(fam_df)} families)")

    # ─── Q2: inter-cluster centroid cosine similarity for EDGE-only set ──────
    print("\n" + "=" * 70)
    print("Q2 — Inter-cluster cosine sim for body clusters in EDGE-only paths")
    print("=" * 70)

    print("\nLoading PKL + node attrs ...")
    with open(PKL_FILE, "rb") as f:
        cm = pickle.load(f)
    with open(ATTR_FILE, "rb") as f:
        node_attrs = pickle.load(f)

    cluster_members = defaultdict(list)
    for key, members in cm.items():
        subtype = str(key[2])
        cid = str(key[4])
        if cid in {"-1", "noise"}:
            continue
        if subtype in BODY_SUBTYPES:
            cluster_members[(subtype, cid)] = list(members)

    # Identify body clusters appearing in EDGE-only frozensets
    used_clusters = set()
    for sig in sigs:
        used_clusters |= sig
    print(f"  body clusters appearing in EDGE-only frozensets: {len(used_clusters)}")
    print(f"  total body clusters in PKL: {len(cluster_members)}")

    # Compute centroid for each used cluster
    print("\n  Computing centroids ...")

    def parse_emb(emb):
        if emb is None:
            return None
        if isinstance(emb, np.ndarray):
            return emb.astype(np.float32)
        if isinstance(emb, (list, tuple)):
            return np.array(emb, dtype=np.float32)
        if isinstance(emb, str):
            s = emb.strip()
            if s.startswith("<") or s.startswith("'<"):
                s = s.strip("'<>").strip()
                try:
                    return np.array([float(x) for x in s.split(",")], dtype=np.float32)
                except (ValueError, TypeError):
                    return None
        return None

    centroids = {}
    skip_log = Counter()
    for cl in used_clusters:
        nids = cluster_members.get(cl, [])
        embs = []
        for nid in nids:
            emb = node_attrs.get(int(nid), {}).get("embedding")
            arr = parse_emb(emb)
            if arr is None:
                skip_log["null_or_unparsable"] += 1
                continue
            embs.append(arr)
        if embs:
            centroid = np.mean(embs, axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-9)
            centroids[cl] = centroid
    if skip_log:
        print(f"  embedding skips: {dict(skip_log)}")

    print(f"  centroids built for {len(centroids)} of {len(used_clusters)} clusters")

    # Pairwise cosine sim
    cl_list = sorted(centroids.keys(), key=lambda x: (LOGICAL_ORDER.index(x[0]), x[1]))
    arr = np.stack([centroids[cl] for cl in cl_list])
    sim_matrix = arr @ arr.T
    pair_sims = sim_matrix[np.triu_indices(len(cl_list), k=1)]
    print("\n  Pairwise centroid cosine sim distribution:")
    print(f"    n_pairs: {len(pair_sims):,}")
    print(
        f"    min: {pair_sims.min():.4f}, mean: {pair_sims.mean():.4f}, "
        f"median: {np.median(pair_sims):.4f}, max: {pair_sims.max():.4f}"
    )
    print(f"    sim ≥ 0.95 (essentially identical): {(pair_sims >= 0.95).sum()}")
    print(f"    sim ≥ 0.90 (very near-duplicate)    : {(pair_sims >= 0.90).sum()}")
    print(f"    sim ≥ 0.85                          : {(pair_sims >= 0.85).sum()}")
    print(f"    sim ≥ 0.80                          : {(pair_sims >= 0.80).sum()}")
    print(f"    sim ≥ 0.77 (above farthest cutoff)  : {(pair_sims >= 0.77).sum()}")
    print(f"    sim ≥ 0.70                          : {(pair_sims >= 0.70).sum()}")

    # Top-20 closest cluster pairs (likely should-be-merged)
    print(
        "\n  Top-20 closest cluster pairs (highest centroid cos sim) — possibly should be merged:"
    )
    pairs = []
    for i in range(len(cl_list)):
        for j in range(i + 1, len(cl_list)):
            pairs.append((sim_matrix[i, j], cl_list[i], cl_list[j]))
    pairs.sort(key=lambda x: -x[0])
    for sim, cl1, cl2 in pairs[:20]:
        n1 = f"{SUBTYPE_PREFIX[cl1[0]]}:{cl1[1]}"
        n2 = f"{SUBTYPE_PREFIX[cl2[0]]}:{cl2[1]}"
        sz1 = len(cluster_members[cl1])
        sz2 = len(cluster_members[cl2])
        print(
            f"    sim={sim:.4f}  {n1:<8} (size {sz1:>3})  vs  {n2:<8} (size {sz2:>3})"
        )

    print("\nDONE.")


if __name__ == "__main__":
    main()
