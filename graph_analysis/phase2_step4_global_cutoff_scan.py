"""
phase2_step4_global_cutoff_scan.py

Mini-study: cluster-cutoff scan {0.80, 0.85, 0.90, 0.95} on EDGE-only VPN.

For each cutoff:
  1. Cluster risks (intra-risk, iterative HDBSCAN+UMAP+cutoff, mcs=5, strict>=5)
  2. Cluster interventions (intra-intervention, same params)
  3. Cluster body GLOBALLY (all 5 subtype labels pooled, no subtype filter)
  4. Save cluster_memberships PKL with role_label preserved per node

VPN definition: nodes appearing in paths from paths_hopwise_v4_edge_only.jsonl
that satisfy intervention_maturity >= 3 (EDGE conf>=3 baked in by F2v4 build).

PKL schema: dict[ (variant, mode, group_key, algo, cluster_id) -> [nids] ]
  group_key in {"risk", "intervention", "body_global"}
"""

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import F3 helpers
from phase2_step4_F3_body_recluster import (
    iterative_residual_recluster,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cutoff",
        type=float,
        required=True,
        help="centroid-sim cutoff (intra-cluster member-to-centroid)",
    )
    ap.add_argument(
        "--mcs", type=int, default=5, help="HDBSCAN min_cluster_size (iter1 + residual)"
    )
    ap.add_argument(
        "--strict-min",
        type=int,
        default=5,
        help="strict per-iteration min cluster size",
    )
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--coverage-target", type=float, default=0.95)
    ap.add_argument("--umap-components", type=int, default=15)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    args = ap.parse_args()

    cutoff = args.cutoff
    suffix = f"global_cutoff{cutoff:.2f}"
    out_pkl = STEP1_DIR / f"cluster_memberships_rev8_{suffix}.pkl"

    print("=" * 70)
    print(f"global cutoff scan — cutoff={cutoff:.2f}")
    print("=" * 70)

    # ─── Load node attrs ──────────────────────────────────────────────────────
    print("\nLoading node_attrs ...")
    t0 = time.time()
    with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  {len(node_attrs):,} nodes  ({time.time() - t0:.1f}s)")

    # ─── Build EDGE-only VPN with maturity>=3 ─────────────────────────────────
    print(f"\nReading {EDGE_ONLY_PATH.name} ...")
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
                    # body — get LLM concept_category
                    rc = node_attrs.get(nid, {}).get("concept_category", "")
                    role_of[nid] = normalize_role(rc)
    print(f"  paths: total={n_total:,}  qualifying(mat>=3)={n_kept:,}")
    print(f"  VPN: {len(vpn):,} unique nodes  ({time.time() - t1:.1f}s)")

    # Split by group
    group_nodes = defaultdict(list)
    for nid in vpn:
        rl = role_of.get(nid, "")
        if rl == "risk":
            group_nodes["risk"].append(nid)
        elif rl == "intervention":
            group_nodes["intervention"].append(nid)
        elif rl in BODY_SUBTYPES:
            group_nodes["body_global"].append(nid)
        else:
            # unknown role — skip
            pass
    print("\n  group sizes:")
    for g in ["risk", "body_global", "intervention"]:
        print(f"    {g:<14}: {len(group_nodes[g]):>5,}")

    # ─── Cluster each group ───────────────────────────────────────────────────
    cluster_memberships = {}
    summary = {}
    for group in ["risk", "body_global", "intervention"]:
        nids = group_nodes[group]
        print(f"\n{'-' * 70}")
        print(f"Clustering group: {group}  (N={len(nids)} nodes, cutoff={cutoff:.2f})")
        print(f"{'-' * 70}")
        if len(nids) < 3 * args.mcs:
            print("  too few nodes for clustering, skipping")
            continue
        # Build embedding matrix
        embs = []
        kept_nids = []
        for nid in nids:
            emb = node_attrs.get(int(nid), {}).get("embedding")
            arr = parse_embedding(emb)
            if arr is None:
                continue
            embs.append(arr)
            kept_nids.append(int(nid))
        if not embs:
            print("  no parseable embeddings, skipping")
            continue
        X_raw = np.stack(embs)
        print(f"  X_raw shape: {X_raw.shape}")

        # Iterative cluster
        t2 = time.time()
        labels_final, iter_records, _, _ = iterative_residual_recluster(
            X_raw,
            hdbscan_mcs_iter1=args.mcs,
            fixed_k_resid=40,
            cutoff=cutoff,
            umap_n_components=args.umap_components,
            umap_n_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            coverage_target=args.coverage_target,
            max_iter=args.max_iter,
            resid_method="hdbscan",
            resid_mcs=args.mcs,
            strict_min_size=args.strict_min,
        )
        wall = time.time() - t2
        print(f"  done in {wall:.1f}s, iterations: {len(iter_records)}")
        n_clustered = int((labels_final != -1).sum())
        n_clusters = len(set(int(c) for c in labels_final if c >= 0))
        coverage = n_clustered / max(len(labels_final), 1)
        print(
            f"  clusters: {n_clusters}, clustered nodes: {n_clustered:,} "
            f"({coverage * 100:.1f}% coverage)"
        )
        for rec in iter_records:
            print(
                f"    iter {rec['iteration']:>2}: {rec['method']:<25} "
                f"k_in={rec.get('k_input', 0)} k_added={rec['n_clusters_added']} "
                f"cum_cov={rec['cumulative_coverage']:.4f}  {rec['stopped_reason'] or ''}"
            )

        # Build PKL records
        for cid in sorted(set(int(c) for c in labels_final if c >= 0)):
            members = [kept_nids[i] for i, c in enumerate(labels_final) if c == cid]
            key = (
                "rev8_global_edge_only",
                "umap_hdbscan_cutoff",
                group,
                "hdbscan",
                str(cid),
            )
            cluster_memberships[key] = members
        summary[group] = {
            "n_input": len(kept_nids),
            "n_clustered": n_clustered,
            "n_clusters": n_clusters,
            "coverage": coverage,
            "wall_s": wall,
        }

    # ─── Save PKL ─────────────────────────────────────────────────────────────
    print(f"\nWriting {out_pkl} ...")
    with open(out_pkl, "wb") as f:
        pickle.dump(cluster_memberships, f)
    print(f"  {len(cluster_memberships):,} cluster records")

    # ─── Save role_of map for downstream F1 ───────────────────────────────────
    role_pkl = STEP1_DIR / f"role_of_rev8_{suffix}.pkl"
    with open(role_pkl, "wb") as f:
        pickle.dump(role_of, f)
    print(f"Wrote role_of map: {role_pkl.name}")

    # ─── Console summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — cutoff={cutoff:.2f}")
    print(f"{'=' * 70}")
    for g, s in summary.items():
        print(
            f"  {g:<14}: N_in={s['n_input']:>5,} "
            f"clustered={s['n_clustered']:>5,} ({s['coverage'] * 100:.1f}%) "
            f"K={s['n_clusters']:>4}  wall={s['wall_s']:.1f}s"
        )
    print("\nDONE.")


if __name__ == "__main__":
    main()
