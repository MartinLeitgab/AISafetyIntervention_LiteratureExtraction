"""
phase2_step4_phase1_paper_clustering.py

Phase 1 of rev8 paper canonical pipeline (per plan_rev8_paper_canonical_pipeline.md).

Two clustering methods on the 19,073-node EDGE-only VPN (paths_hopwise_v4_edge_only.jsonl
intersected with intervention_maturity >= 3):

  Method A: HDBSCAN on UMAP-2D + 0.80 raw-cosine centroid filter, iterative residual
            recluster, mcs=5, strict per-iter >=5.
  Method B: Louvain on SIM>=0.80 k-NN graph (raw cosine) + 0.80 centroid filter
            (post-hoc), drop <5-member communities, iterative residual recluster.

Two pools:
  Risk pool (~2,464 nodes)         — risks cluster separately (cross-paper).
  NR  pool (~16,609 nodes)         — body + intervention pooled (paper-specific
                                      1:1 custom designs).

Outputs (in phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/):
  cluster_memberships_rev8_paper_methodA.pkl
      dict[("rev8_paper", "umap2d_hdbscan_iter", pool, "hdbscan", str(cid)) -> [nids]]
  cluster_memberships_rev8_paper_methodB.pkl
      dict[("rev8_paper", "louvain_sim080_iter", pool, "louvain", str(cid)) -> [nids]]
  role_of_rev8_paper.pkl
      dict[nid -> "risk" | "intervention" | <body subtype>]
  phase1_coverage_summary.csv
      method × pool rows: n_input, n_clustered, n_clusters, coverage,
                          mean_cluster_size, median_cluster_size, wall_s
"""

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# F3 helpers
from phase2_step4_F3_body_recluster import (
    iterative_residual_recluster,
    apply_centroid_cutoff,
    parse_embedding,
)

ROOT = Path(__file__).parent
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
STEP1_DIR = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
EDGE_ONLY_PATH = PATHS_DIR / "paths_hopwise_v4_edge_only.jsonl"

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
}


def normalize_role(raw):
    if raw is None:
        return ""
    s = str(raw).strip()
    return CATEGORY_NORMALIZE.get(s, s.replace(" ", "_"))


# ─── VPN construction ────────────────────────────────────────────────────────
def build_vpn(node_attrs, pool_mode="pooled"):
    """pool_mode: "pooled" → risk + nr (body+intervention together);
    "per_subtype" → risk + 5 body subtypes + intervention (7 pools)."""
    print(f"Reading {EDGE_ONLY_PATH.name} ...")
    t1 = time.time()
    vpn = set()
    role_of = {}
    n_total = n_kept = 0
    with open(EDGE_ONLY_PATH) as f:
        for line in f:
            obj = json.loads(line)
            n_total += 1
            path = [int(x) for x in obj["path"]]
            cats = obj.get("categories", [])
            interv_id = path[-1]
            mat = node_attrs.get(interv_id, {}).get("intervention_maturity", 0)
            try:
                mat_i = int(mat) if mat is not None else 0
            except Exception:
                mat_i = 0
            if mat_i < 3:
                continue
            n_kept += 1
            for nid, cat in zip(path, cats):
                vpn.add(nid)
                if cat == "risk":
                    role_of[nid] = "risk"
                elif cat == "intervention":
                    role_of[nid] = "intervention"
                else:
                    rc = node_attrs.get(nid, {}).get("concept_category", "")
                    role_of[nid] = normalize_role(rc)
    print(f"  paths total={n_total:,}  qualifying(mat>=3)={n_kept:,}")
    print(f"  VPN: {len(vpn):,} unique nodes  ({time.time() - t1:.1f}s)")

    pool_nodes = defaultdict(list)
    if pool_mode == "pooled":
        for nid in vpn:
            rl = role_of.get(nid, "")
            if rl == "risk":
                pool_nodes["risk"].append(nid)
            elif rl == "intervention" or rl in BODY_SUBTYPES:
                pool_nodes["nr"].append(nid)
        print(
            f"  pool sizes (pooled):  risk={len(pool_nodes['risk']):,}  "
            f"nr={len(pool_nodes['nr']):,}"
        )
    elif pool_mode == "per_subtype":
        for nid in vpn:
            rl = role_of.get(nid, "")
            if rl == "risk":
                pool_nodes["risk"].append(nid)
            elif rl == "intervention":
                pool_nodes["intervention"].append(nid)
            elif rl in BODY_SUBTYPES:
                pool_nodes[rl].append(nid)
        # canonical logical-chain order in the printout (CLAUDE.md rule)
        order = [
            "risk",
            "problem_analysis",
            "theoretical_insight",
            "design_rationale",
            "implementation_mechanism",
            "validation_evidence",
            "intervention",
        ]
        print("  pool sizes (per_subtype):")
        for p in order:
            if p in pool_nodes:
                print(f"    {p:<26} {len(pool_nodes[p]):>5,}")
    else:
        raise ValueError(f"unknown pool_mode: {pool_mode}")
    return vpn, role_of, pool_nodes


def build_embedding_matrix(nids, node_attrs):
    embs, kept = [], []
    for nid in nids:
        emb = node_attrs.get(int(nid), {}).get("embedding")
        arr = parse_embedding(emb)
        if arr is None:
            continue
        embs.append(arr)
        kept.append(int(nid))
    if not embs:
        return None, []
    X = np.stack(embs)
    return X, kept


# ─── Method A: HDBSCAN-2D iterative ──────────────────────────────────────────
def run_method_A(X_raw, kept_nids, cutoff, mcs, strict_min, max_iter):
    print(
        f"  Method A (HDBSCAN on UMAP-2D + cutoff={cutoff}, mcs={mcs}, strict>={strict_min})"
    )
    t0 = time.time()
    labels_final, iter_records, _, _ = iterative_residual_recluster(
        X_raw,
        hdbscan_mcs_iter1=mcs,
        fixed_k_resid=40,
        cutoff=cutoff,
        umap_n_components=2,  # <-- HDBSCAN-2D substrate
        umap_n_neighbors=15,
        umap_min_dist=0.0,
        coverage_target=0.95,
        max_iter=max_iter,
        resid_method="hdbscan",
        resid_mcs=mcs,
        strict_min_size=strict_min,
    )
    wall = time.time() - t0
    n_clustered = int((labels_final != -1).sum())
    n_clusters = len(set(int(c) for c in labels_final if c >= 0))
    coverage = n_clustered / max(len(labels_final), 1)
    print(
        f"    done {wall:.1f}s, iters={len(iter_records)}, "
        f"clusters={n_clusters}, clustered={n_clustered:,} ({coverage * 100:.1f}%)"
    )
    for rec in iter_records:
        print(
            f"      iter {rec['iteration']:>2}: {rec['method']:<25} "
            f"k_in={rec.get('k_input', 0):>3} k_added={rec['n_clusters_added']:>3} "
            f"cum_cov={rec['cumulative_coverage']:.4f}  {rec.get('stopped_reason') or ''}"
        )
    return labels_final, iter_records, wall


# ─── Method B: Louvain on SIM>=0.80 k-NN graph, iterative ────────────────────
def _build_sim_graph(X_raw, sim_threshold, chunk=2000):
    """Build NetworkX graph with edges where raw cosine sim >= sim_threshold.
    Self-loops excluded.
    """
    import networkx as nx

    N = len(X_raw)
    norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    Xn = X_raw / np.where(norms > 0, norms, 1.0)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    n_edges = 0
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        sims = Xn[i0:i1] @ Xn.T  # (chunk, N)
        for li, gi in enumerate(range(i0, i1)):
            row = sims[li]
            # Edges to indices > gi to avoid double-counting; mask out self.
            js = np.where(row >= sim_threshold)[0]
            js = js[js > gi]  # upper triangle only
            if len(js):
                weights = row[js].astype(np.float32)
                G.add_weighted_edges_from(
                    zip([gi] * len(js), js.tolist(), weights.tolist())
                )
                n_edges += len(js)
    return G, n_edges


def _louvain_one_pass(X_sub, sim_threshold, mcs):
    """Run Louvain on a single residual block. Returns labels (len N_sub),
    -1 for nodes not in any community of >=2 (singletons → noise) AND
    pre-cutoff communities of <mcs.
    """
    from networkx.algorithms.community import louvain_communities

    N = len(X_sub)
    G, n_edges = _build_sim_graph(X_sub, sim_threshold)
    print(f"      sim graph: N={N}, edges={n_edges:,}")
    if G.number_of_edges() == 0:
        return np.full(N, -1, dtype=int)
    communities = louvain_communities(G, weight="weight", seed=42, resolution=1.0)
    labels = np.full(N, -1, dtype=int)
    cid = 0
    for comm in communities:
        if len(comm) < mcs:
            continue
        for n in comm:
            labels[n] = cid
        cid += 1
    return labels


def iterative_louvain_recluster(
    X_raw, sim_threshold, cutoff, mcs, strict_min, max_iter, coverage_target
):
    """Iteratively run Louvain + 0.80 centroid filter on residual."""
    N = len(X_raw)
    labels_final = np.full(N, -1, dtype=int)
    next_cid = 0
    remaining = np.arange(N)
    iter_records = []

    for it in range(1, max_iter + 1):
        if len(remaining) < 3 * mcs:
            iter_records.append(
                {
                    "iteration": it,
                    "n_input": int(len(remaining)),
                    "method": "skip",
                    "k_input": 0,
                    "n_clusters_pre_filter": 0,
                    "n_clusters_added": 0,
                    "n_clustered_added": 0,
                    "cumulative_clustered": int((labels_final != -1).sum()),
                    "cumulative_coverage": float((labels_final != -1).sum() / N),
                    "stopped_reason": f"remaining<{3 * mcs}",
                }
            )
            break
        X_sub = X_raw[remaining]
        labels_sub = _louvain_one_pass(X_sub, sim_threshold, mcs)
        k_input = int(np.unique(labels_sub[labels_sub >= 0]).size)
        method_used = f"louvain_sim>={sim_threshold:.2f}_mcs={mcs}"
        labels_post, _refined, _initial = apply_centroid_cutoff(
            X_sub, labels_sub, cutoff
        )
        # Strict per-iter: drop tiny clusters post-filter.
        unique_pre = sorted(set(int(c) for c in labels_post if c >= 0))
        n_pre = len(unique_pre)
        for cid in unique_pre:
            mask = labels_post == cid
            if int(mask.sum()) < strict_min:
                labels_post[mask] = -1
        unique_loc = sorted(set(int(c) for c in labels_post if c >= 0))
        n_added = len(unique_loc)
        n_new_clustered = 0
        for cid in unique_loc:
            local_idx = np.where(labels_post == cid)[0]
            global_idx = remaining[local_idx]
            labels_final[global_idx] = next_cid
            next_cid += 1
            n_new_clustered += len(global_idx)
        new_noise_local = np.where(labels_post == -1)[0]
        new_remaining = remaining[new_noise_local]
        cumulative = int((labels_final != -1).sum())
        cum_cov = cumulative / N
        iter_records.append(
            {
                "iteration": it,
                "n_input": int(len(remaining)),
                "method": method_used,
                "k_input": int(k_input),
                "n_clusters_pre_filter": int(n_pre),
                "n_clusters_added": int(n_added),
                "n_clustered_added": int(n_new_clustered),
                "cumulative_clustered": int(cumulative),
                "cumulative_coverage": round(cum_cov, 4),
                "stopped_reason": None,
            }
        )
        if cum_cov >= coverage_target:
            iter_records[-1]["stopped_reason"] = "coverage_target_reached"
            break
        if n_added == 0:
            iter_records[-1]["stopped_reason"] = "no_strict_clusters_added (converged)"
            break
        remaining = new_remaining
    else:
        iter_records[-1]["stopped_reason"] = "max_iter_reached"
    return labels_final, iter_records


def run_method_B(X_raw, kept_nids, cutoff, sim_threshold, mcs, strict_min, max_iter):
    print(
        f"  Method B (Louvain on SIM>={sim_threshold} k-NN + cutoff={cutoff}, "
        f"mcs={mcs}, strict>={strict_min})"
    )
    t0 = time.time()
    labels_final, iter_records = iterative_louvain_recluster(
        X_raw,
        sim_threshold=sim_threshold,
        cutoff=cutoff,
        mcs=mcs,
        strict_min=strict_min,
        max_iter=max_iter,
        coverage_target=0.95,
    )
    wall = time.time() - t0
    n_clustered = int((labels_final != -1).sum())
    n_clusters = len(set(int(c) for c in labels_final if c >= 0))
    coverage = n_clustered / max(len(labels_final), 1)
    print(
        f"    done {wall:.1f}s, iters={len(iter_records)}, "
        f"clusters={n_clusters}, clustered={n_clustered:,} ({coverage * 100:.1f}%)"
    )
    for rec in iter_records:
        print(
            f"      iter {rec['iteration']:>2}: {rec['method']:<32} "
            f"k_in={rec.get('k_input', 0):>3} k_added={rec['n_clusters_added']:>3} "
            f"cum_cov={rec['cumulative_coverage']:.4f}  {rec.get('stopped_reason') or ''}"
        )
    return labels_final, iter_records, wall


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.80)
    ap.add_argument(
        "--sim-threshold",
        type=float,
        default=0.80,
        help="Method B SIM>= threshold for k-NN edges",
    )
    ap.add_argument("--mcs", type=int, default=5)
    ap.add_argument("--strict-min", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--methods", default="A,B")
    ap.add_argument(
        "--tag",
        default="",
        help="optional output suffix, e.g. '_c70m3' for cutoff=0.7 mcs=3",
    )
    ap.add_argument(
        "--pool-mode",
        choices=["pooled", "per_subtype"],
        default="pooled",
        help="pooled = risk+nr; per_subtype = 7 pools (risk + 5 body + intervention)",
    )
    args = ap.parse_args()

    methods_to_run = set(s.strip() for s in args.methods.split(","))

    print("=" * 76)
    print(
        f"PHASE 1 paper clustering | cutoff={args.cutoff} | mcs={args.mcs} | "
        f"strict>={args.strict_min} | methods={','.join(sorted(methods_to_run))}"
    )
    print("=" * 76)

    # ─── Load attrs + build VPN ──────────────────────────────────────────────
    print("\nLoading node_attrs ...")
    t0 = time.time()
    with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  {len(node_attrs):,} nodes ({time.time() - t0:.1f}s)")

    vpn, role_of, pool_nodes = build_vpn(node_attrs, pool_mode=args.pool_mode)

    # Save role_of
    role_pkl = STEP1_DIR / "role_of_rev8_paper.pkl"
    with open(role_pkl, "wb") as f:
        pickle.dump(role_of, f)
    print(f"\nWrote role_of: {role_pkl.name}")

    # ─── Run methods on each pool ────────────────────────────────────────────
    cm_A = {}
    cm_B = {}
    summary_rows = []
    iter_records_all = {}  # (method, pool) -> list

    if args.pool_mode == "pooled":
        pool_order = ["risk", "nr"]
    else:
        pool_order = [
            "risk",
            "problem_analysis",
            "theoretical_insight",
            "design_rationale",
            "implementation_mechanism",
            "validation_evidence",
            "intervention",
        ]

    for pool in pool_order:
        nids = pool_nodes.get(pool, [])
        print(f"\n{'=' * 76}\nPOOL: {pool}  (N={len(nids)} nodes)\n{'=' * 76}")
        if len(nids) < 3 * args.mcs:
            print("  too few, skipping")
            continue
        X_raw, kept = build_embedding_matrix(nids, node_attrs)
        if X_raw is None:
            print("  no parseable embeddings, skipping")
            continue
        print(f"  X_raw shape: {X_raw.shape}")

        # Method A
        if "A" in methods_to_run:
            labels_A, iters_A, wall_A = run_method_A(
                X_raw,
                kept,
                cutoff=args.cutoff,
                mcs=args.mcs,
                strict_min=args.strict_min,
                max_iter=args.max_iter,
            )
            iter_records_all[("A", pool)] = iters_A
            for cid in sorted(set(int(c) for c in labels_A if c >= 0)):
                members = [kept[i] for i, c in enumerate(labels_A) if c == cid]
                key = ("rev8_paper", "umap2d_hdbscan_iter", pool, "hdbscan", str(cid))
                cm_A[key] = members
            sizes_A = [
                int((labels_A == cid).sum())
                for cid in sorted(set(int(c) for c in labels_A if c >= 0))
            ]
            n_clustered_A = int((labels_A != -1).sum())
            summary_rows.append(
                {
                    "method": "A_hdbscan_2d",
                    "pool": pool,
                    "n_input": len(kept),
                    "n_clustered": n_clustered_A,
                    "coverage": round(n_clustered_A / max(len(kept), 1), 4),
                    "n_clusters": len(sizes_A),
                    "mean_cluster_size": round(float(np.mean(sizes_A)), 1)
                    if sizes_A
                    else 0,
                    "median_cluster_size": int(np.median(sizes_A)) if sizes_A else 0,
                    "wall_s": round(wall_A, 1),
                }
            )

        # Method B
        if "B" in methods_to_run:
            labels_B, iters_B, wall_B = run_method_B(
                X_raw,
                kept,
                cutoff=args.cutoff,
                sim_threshold=args.sim_threshold,
                mcs=args.mcs,
                strict_min=args.strict_min,
                max_iter=args.max_iter,
            )
            iter_records_all[("B", pool)] = iters_B
            for cid in sorted(set(int(c) for c in labels_B if c >= 0)):
                members = [kept[i] for i, c in enumerate(labels_B) if c == cid]
                key = ("rev8_paper", "louvain_sim080_iter", pool, "louvain", str(cid))
                cm_B[key] = members
            sizes_B = [
                int((labels_B == cid).sum())
                for cid in sorted(set(int(c) for c in labels_B if c >= 0))
            ]
            n_clustered_B = int((labels_B != -1).sum())
            summary_rows.append(
                {
                    "method": "B_louvain_sim080",
                    "pool": pool,
                    "n_input": len(kept),
                    "n_clustered": n_clustered_B,
                    "coverage": round(n_clustered_B / max(len(kept), 1), 4),
                    "n_clusters": len(sizes_B),
                    "mean_cluster_size": round(float(np.mean(sizes_B)), 1)
                    if sizes_B
                    else 0,
                    "median_cluster_size": int(np.median(sizes_B)) if sizes_B else 0,
                    "wall_s": round(wall_B, 1),
                }
            )

    # ─── Save outputs ────────────────────────────────────────────────────────
    if "A" in methods_to_run:
        out_A = STEP1_DIR / f"cluster_memberships_rev8_paper_methodA{args.tag}.pkl"
        with open(out_A, "wb") as f:
            pickle.dump(cm_A, f)
        print(f"\nWrote {out_A.name}: {len(cm_A):,} cluster records")

    if "B" in methods_to_run:
        out_B = STEP1_DIR / f"cluster_memberships_rev8_paper_methodB{args.tag}.pkl"
        with open(out_B, "wb") as f:
            pickle.dump(cm_B, f)
        print(f"Wrote {out_B.name}: {len(cm_B):,} cluster records")

    # iteration records (debug)
    iter_pkl = STEP1_DIR / f"phase1_iter_records{args.tag}.pkl"
    with open(iter_pkl, "wb") as f:
        pickle.dump({f"{m}_{p}": rs for (m, p), rs in iter_records_all.items()}, f)
    print(f"Wrote {iter_pkl.name}")

    # Summary CSV
    summary_csv = STEP1_DIR / f"phase1_coverage_summary{args.tag}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv.name}")
    print("\n" + "=" * 76)
    print("PHASE 1 COVERAGE SUMMARY")
    print("=" * 76)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nPhase 1 DONE. Next: phase 1 coverage report (A vs B union/intersect/ARI).")


if __name__ == "__main__":
    main()
