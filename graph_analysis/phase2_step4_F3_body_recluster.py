"""
phase2_step4_F3_body_recluster.py [rev8 — Task #7, BERTopic-style v2]

Body recluster on path-participating nodes (VPN_paperpair) using the
BERTopic-style framework: UMAP dimensionality reduction first, then HDBSCAN
clustering, with quality metrics adapted to single-domain text corpora.

Replaces the prior raw-cosine Agglomerative + Pareto (0.70/0.30) approach,
which proved unattainable in raw embedding space for AI safety content. See
Step 4 Findings Report §19.3b for the methodological pivot rationale.

Pipeline per subtype:
  1. Filter VPN nodes by concept_category match → raw embeddings (1024-dim)
  2. UMAP(15D, cosine) for clustering substrate
  3. UMAP(2D, cosine) for visualization
  4. HDBSCAN scan over min_cluster_size = HDBSCAN_MCS
     - Cluster on UMAP-15D
     - Metrics:
       * silhouette (UMAP-15D, cosine)        target > 0.25
       * coverage = 1 - noise_rate            target > 0.50
       * n_clusters_realized
       * random-baseline z-score (UMAP-15D centroids):
            z_intra = (obs_intra − rand_intra_mean) / rand_intra_std
            z_inter = (rand_inter_mean − obs_inter) / rand_inter_std
            target > 2.0 on both → clusters statistically tighter / more
            separated than random label shuffles
  5. Pareto pass: sil > sil_threshold AND coverage > cov_threshold
     AND z_intra > z_threshold AND z_inter > z_threshold
  6. Per-cluster 3 closest + 3 farthest representatives by RAW-embedding
     cosine sim to cluster centroid (data-driven characterization; NO LLM
     naming load-bearing in this step — see §19.3b)
  7. 2D UMAP scatter plot per subtype with cluster overlay
  8. Save cluster_memberships_rev8_<suffix>.pkl with chosen mcs

Inputs:
  --paths-file  PATH   F2v4 hop-wise (or BFS-shortest) jsonl path file
  --sim-threshold F    cosine sim threshold for consim1 SIM-edge filter
  --output-suffix S    suffix for all outputs (sim0.9, sim0.85, ..., edge_only)

Outputs (under phase2_results/step4_finalanalysis/step4_cluster_tables/):
  body_kscan_metrics_<suffix>.csv               per (subtype, mcs)
  body_kscan_chosen_k_<suffix>.csv              per subtype: chosen mcs + status
  body_kscan_representatives_<suffix>.csv       3 closest + 3 farthest per cluster
  body_kscan_pareto_plot_<subtype>_<suffix>.png 2D UMAP scatter with clusters
  body_kscan_population_summary_<suffix>.csv    n_vpn per subtype

Under step1_load_and_parse_umapwithoutlocalsatellites/:
  cluster_memberships_rev8_<suffix>.pkl
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

try:
    import hdbscan as _hdbscan_mod
except ImportError:
    _hdbscan_mod = None

try:
    import umap as _umap_mod
except ImportError:
    _umap_mod = None

try:
    from sklearn.metrics import adjusted_rand_score as _ari
except ImportError:
    _ari = None

matplotlib.use("Agg")

ROOT = Path(__file__).parent
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
STEP1_DIR = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4_DIR = ROOT / "phase2_results/step4_finalanalysis"
OUT_TABLES = STEP4_DIR / "step4_cluster_tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

BODY_SUBTYPES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]
ALL_NODE_TYPES = BODY_SUBTYPES + ["risk", "intervention"]

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
    s = str(raw).strip()
    return CATEGORY_NORMALIZE.get(s, s)


def _umap_then_hdbscan(X_raw, mcs, umap_n_components, umap_n_neighbors, umap_min_dist):
    """One-pass UMAP+HDBSCAN. Returns (labels, X_umap_for_clustering, X_umap2d)."""
    if _umap_mod is None or _hdbscan_mod is None:
        raise RuntimeError("umap-learn or hdbscan not installed")
    reducer = _umap_mod.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    X_umap = reducer.fit_transform(X_raw)
    reducer2 = _umap_mod.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    X_umap2 = reducer2.fit_transform(X_raw)
    clusterer = _hdbscan_mod.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean")
    labels = clusterer.fit_predict(X_umap)
    return labels, X_umap, X_umap2


def iterative_residual_recluster(
    X_raw,
    hdbscan_mcs_iter1,
    fixed_k_resid,
    cutoff,
    umap_n_components,
    umap_n_neighbors,
    umap_min_dist,
    coverage_target,
    max_iter,
    resid_method="hdbscan",
    resid_mcs=5,
    strict_min_size=5,
):
    """Iteratively cluster residual (post-cutoff noise) until coverage_target
    is reached OR max_iter exhausted OR remaining set too small.

    Iteration 1: full data, UMAP+HDBSCAN(mcs=hdbscan_mcs_iter1)+0.77 cutoff.
    Iteration 2+: take previous iteration's failures (HDBSCAN-noise +
                  cutoff-rejected). Recluster with one of:
                  - resid_method="hdbscan": UMAP+HDBSCAN(mcs=resid_mcs).
                    Adapts to where dense pockets exist in residual.
                  - resid_method="agglo_ward": UMAP+Agglomerative(K=fixed_k_resid,
                    ward+euclidean). Forces K partitions; usually worse because
                    centroids of forced groups don't pass 0.77.
                  Apply 0.77 cutoff. Survivors get new cluster IDs.

    Returns:
      labels_final: int array (N,), -1 for nodes never tight-clustered
      iter_records: per-iteration stats
      X_umap2_first: 2D UMAP from iteration 1 (for visualization)
    """
    N = len(X_raw)
    labels_final = np.full(N, -1, dtype=int)
    next_cid = 0
    remaining = np.arange(N)
    X_umap2_first = None
    iter_records = []
    # Maps each globally-issued cluster ID to its INITIAL centroid (used as
    # filter reference). Members have sim >= cutoff to this centroid by
    # construction; the refined centroid (post-survivor mean) may differ.
    initial_centroids_by_cid = {}

    for it in range(1, max_iter + 1):
        # Min remaining size depends on iter 2+ method
        if it == 1:
            min_remaining = 3 * hdbscan_mcs_iter1
        elif resid_method == "hdbscan":
            min_remaining = 3 * resid_mcs
        else:
            min_remaining = fixed_k_resid
        if len(remaining) < min_remaining:
            iter_records.append(
                {
                    "iteration": it,
                    "n_input": len(remaining),
                    "method": "skip",
                    "k_input": 0,
                    "n_clusters_added": 0,
                    "n_clustered_added": 0,
                    "cumulative_clustered": int((labels_final != -1).sum()),
                    "cumulative_coverage": float((labels_final != -1).sum() / N),
                    "stopped_reason": f"remaining<{min_remaining}",
                }
            )
            break
        X_sub = X_raw[remaining]
        if it == 1:
            labels_sub, _X_umap, X_umap2 = _umap_then_hdbscan(
                X_sub,
                hdbscan_mcs_iter1,
                umap_n_components,
                umap_n_neighbors,
                umap_min_dist,
            )
            X_umap2_first = X_umap2
            method_used = f"hdbscan_mcs={hdbscan_mcs_iter1}"
            k_input = int(np.unique(labels_sub[labels_sub >= 0]).size)
        elif resid_method == "hdbscan":
            labels_sub, _X_umap, _X2 = _umap_then_hdbscan(
                X_sub, resid_mcs, umap_n_components, umap_n_neighbors, umap_min_dist
            )
            method_used = f"hdbscan_mcs={resid_mcs}"
            k_input = int(np.unique(labels_sub[labels_sub >= 0]).size)
        else:
            if _umap_mod is None:
                raise RuntimeError("umap-learn not installed")
            reducer = _umap_mod.UMAP(
                n_components=umap_n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric="cosine",
                random_state=42,
            )
            X_umap_sub = reducer.fit_transform(X_sub)
            from sklearn.cluster import AgglomerativeClustering as _AC

            k_eff = min(fixed_k_resid, len(remaining) - 1)
            clusterer = _AC(n_clusters=k_eff, metric="euclidean", linkage="ward")
            labels_sub = clusterer.fit_predict(X_umap_sub)
            method_used = f"agglo_ward_k={k_eff}"
            k_input = k_eff
        labels_sub_post, _refined, initial_centroids_local = apply_centroid_cutoff(
            X_sub, labels_sub, cutoff
        )
        # Strict per-iteration filter: only keep clusters whose surviving
        # member count is >= strict_min_size. Members of dropped tiny
        # clusters revert to noise (-1) so they are eligible for next iter.
        unique_pre_filter = sorted(set(int(c) for c in labels_sub_post if c >= 0))
        n_clusters_pre_filter = len(unique_pre_filter)
        for cid_local in unique_pre_filter:
            mask = labels_sub_post == cid_local
            if int(mask.sum()) < strict_min_size:
                labels_sub_post[mask] = -1
        unique_local = sorted(set(int(c) for c in labels_sub_post if c >= 0))
        n_clusters_added = len(unique_local)
        n_newly_clustered = 0
        for cid_local in unique_local:
            local_idx = np.where(labels_sub_post == cid_local)[0]
            global_idx = remaining[local_idx]
            labels_final[global_idx] = next_cid
            initial_centroids_by_cid[next_cid] = initial_centroids_local[cid_local]
            next_cid += 1
            n_newly_clustered += len(global_idx)
        new_noise_local = np.where(labels_sub_post == -1)[0]
        new_remaining = remaining[new_noise_local]
        cumulative = int((labels_final != -1).sum())
        cum_cov = cumulative / N
        iter_records.append(
            {
                "iteration": it,
                "n_input": int(len(remaining)),
                "method": method_used,
                "k_input": int(k_input),
                "n_clusters_pre_filter": int(n_clusters_pre_filter),
                "n_clusters_added": int(n_clusters_added),
                "n_clustered_added": int(n_newly_clustered),
                "cumulative_clustered": int(cumulative),
                "cumulative_coverage": round(cum_cov, 4),
                "stopped_reason": None,
            }
        )
        if cum_cov >= coverage_target:
            iter_records[-1]["stopped_reason"] = "coverage_target_reached"
            break
        if n_clusters_added == 0:
            # No cluster met >=strict_min_size + >=cutoff this iteration ->
            # residual unchanged -> next iter would be identical (UMAP+HDBSCAN
            # deterministic on same input). True convergence: no more
            # tight clusters of >= strict_min_size members exist.
            iter_records[-1]["stopped_reason"] = "no_strict_clusters_added (converged)"
            break
        remaining = new_remaining
    else:
        iter_records[-1]["stopped_reason"] = "max_iter_reached"

    return labels_final, iter_records, X_umap2_first, initial_centroids_by_cid


def filter_final_clusters_by_size(labels_final, min_size):
    """Post-iteration cleanup: drop clusters with surviving member count below
    min_size. Their members revert to noise (-1). Returns filtered labels and
    a stats dict (n_dropped_clusters, n_dropped_members).
    """
    labels = labels_final.copy()
    n_dropped_clusters = 0
    n_dropped_members = 0
    for cid in sorted(set(int(c) for c in labels if c >= 0)):
        members = labels == cid
        if int(members.sum()) < min_size:
            labels[members] = -1
            n_dropped_clusters += 1
            n_dropped_members += int(members.sum())
    return labels, {
        "n_dropped_clusters": n_dropped_clusters,
        "n_dropped_members": n_dropped_members,
    }


def apply_centroid_cutoff(X_raw, labels_in, cutoff):
    """Single-pass centroid refinement.

    Returns:
        labels_out: same shape as labels_in, weak members -> -1
        refined_centroids: dict cid -> unit-normalized vector (post-survivor mean)
        initial_centroids: dict cid -> unit-normalized vector (pre-cutoff mean,
                          USED for the sim<cutoff filter; members of labels_out
                          have sim >= cutoff to initial_centroids[cid] by
                          construction)
    """
    norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    Xn = X_raw / np.where(norms > 0, norms, 1.0)
    labels_out = labels_in.copy()
    initial_centroids = {}
    for cid in sorted(set(int(c) for c in labels_in if c >= 0)):
        members = labels_in == cid
        if members.sum() == 0:
            continue
        c = Xn[members].mean(axis=0)
        nrm = np.linalg.norm(c)
        if nrm > 0:
            c = c / nrm
        initial_centroids[cid] = c
    for cid, c in initial_centroids.items():
        members = np.where(labels_in == cid)[0]
        if len(members) == 0:
            continue
        sims = Xn[members] @ c
        weak = members[sims < cutoff]
        labels_out[weak] = -1
    refined_centroids = {}
    for cid in sorted(set(int(c) for c in labels_out if c >= 0)):
        members = labels_out == cid
        if members.sum() == 0:
            continue
        c = Xn[members].mean(axis=0)
        nrm = np.linalg.norm(c)
        if nrm > 0:
            c = c / nrm
        refined_centroids[cid] = c
    return labels_out, refined_centroids, initial_centroids


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def parse_embedding(emb_val):
    if isinstance(emb_val, np.ndarray):
        v = emb_val.astype(np.float32)
    elif emb_val is None:
        return None
    else:
        s = str(emb_val).strip()
        if s.startswith("<") and s.endswith(">"):
            s = s[1:-1]
        try:
            v = np.array([float(x) for x in s.split(",")], dtype=np.float32)
        except Exception:
            return None
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None


def random_baseline_zscore(X_umap, labels, n_shuffles=20, rng=None):
    """Shuffle cluster labels n_shuffles times; compute observed vs random
    intra (mean within-cluster pair distance) and inter (max between-centroid
    sim) and return z-scores.

    X_umap is the UMAP-projected embeddings (already in cluster space).
    Distances/sims computed via cosine on the UMAP-projected vectors.
    Noise points (label == -1) are excluded.
    """
    rng = rng or np.random.default_rng(7)
    valid = labels != -1
    Xv = X_umap[valid]
    Lv = labels[valid]
    if len(np.unique(Lv)) < 2 or len(Xv) < 4:
        return None, None
    # Unit-normalize for cosine sim consistency
    norms = np.linalg.norm(Xv, axis=1, keepdims=True)
    Xn = Xv / np.where(norms > 0, norms, 1.0)

    def _intra_inter(L):
        intra_per_cluster = []
        cluster_centroids = []
        for cid in np.unique(L):
            mask = L == cid
            Ec = Xn[mask]
            if len(Ec) < 2:
                continue
            sims = Ec @ Ec.T
            n = len(Ec)
            mask_diag = ~np.eye(n, dtype=bool)
            intra_per_cluster.append(sims[mask_diag].mean())
            c = Ec.mean(axis=0)
            nrm = np.linalg.norm(c)
            cluster_centroids.append(c / nrm if nrm > 0 else c)
        if len(cluster_centroids) < 2:
            return None, None
        intra = float(np.mean(intra_per_cluster))
        C = np.stack(cluster_centroids)
        cs = C @ C.T
        mc = ~np.eye(len(C), dtype=bool)
        inter = float(cs[mc].max())
        return intra, inter

    obs_intra, obs_inter = _intra_inter(Lv)
    if obs_intra is None:
        return None, None
    rand_intras, rand_inters = [], []
    for _ in range(n_shuffles):
        L_shuffled = Lv.copy()
        rng.shuffle(L_shuffled)
        ri, rii = _intra_inter(L_shuffled)
        if ri is None:
            continue
        rand_intras.append(ri)
        rand_inters.append(rii)
    if not rand_intras:
        return None, None
    ri_mean = float(np.mean(rand_intras))
    ri_std = float(np.std(rand_intras)) or 1e-6
    re_mean = float(np.mean(rand_inters))
    re_std = float(np.std(rand_inters)) or 1e-6
    z_intra = (obs_intra - ri_mean) / ri_std
    z_inter = (re_mean - obs_inter) / re_std  # inverted: smaller obs = better
    return float(z_intra), float(z_inter)


def representatives_per_cluster(
    ids, X_raw, labels, node_attrs, n_each=3, initial_centroids=None
):
    """For each cluster, return 3 closest + 3 farthest members by cosine sim
    in RAW embedding space.

    If initial_centroids is provided (dict cid -> unit-vec), the displayed
    cosine_to_centroid uses the INITIAL (pre-refinement) centroid. By the
    cutoff filter applied during clustering, every member has sim >=cutoff
    to the initial centroid by construction. The refined (post-survivor)
    centroid sim is also reported for transparency.
    If initial_centroids is None, only refined-centroid sim is reported.
    """
    rows = []
    norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    Xn = X_raw / np.where(norms > 0, norms, 1.0)
    for cid in sorted(set(int(c) for c in labels if c >= 0)):
        mask = labels == cid
        Ec = Xn[mask]
        ids_c = [ids[i] for i in range(len(ids)) if mask[i]]
        refined = Ec.mean(axis=0)
        nrm = np.linalg.norm(refined)
        if nrm > 0:
            refined /= nrm
        # Initial centroid: prefer provided dict; fall back to refined if absent
        if initial_centroids is not None and cid in initial_centroids:
            initial = initial_centroids[cid]
        else:
            initial = refined
        sims_initial = Ec @ initial
        sims_refined = Ec @ refined
        order = np.argsort(sims_initial)
        farthest_idx = order[: min(n_each, len(order))]
        closest_idx = order[-min(n_each, len(order)) :][::-1]
        for rank, i in enumerate(closest_idx):
            nid = int(ids_c[i])
            attrs = node_attrs.get(nid, {})
            rows.append(
                {
                    "cluster_id": int(cid),
                    "role": "closest",
                    "rank": int(rank + 1),
                    "node_id": nid,
                    "name": str(attrs.get("name", ""))[:300],
                    "sim_initial_centroid": round(float(sims_initial[i]), 4),
                    "sim_refined_centroid": round(float(sims_refined[i]), 4),
                }
            )
        for rank, i in enumerate(farthest_idx):
            nid = int(ids_c[i])
            attrs = node_attrs.get(nid, {})
            rows.append(
                {
                    "cluster_id": int(cid),
                    "role": "farthest",
                    "rank": int(rank + 1),
                    "node_id": nid,
                    "name": str(attrs.get("name", ""))[:300],
                    "sim_initial_centroid": round(float(sims_initial[i]), 4),
                    "sim_refined_centroid": round(float(sims_refined[i]), 4),
                }
            )
    return rows


def scan_subtype(
    subtype,
    vpn_ids,
    node_attrs,
    hdbscan_mcs,
    umap_n_components,
    umap_n_neighbors,
    umap_min_dist,
    rng,
):
    """BERTopic-style scan for one subtype. Returns rows (one per mcs),
    membership dict per mcs, representatives for chosen mcs (filled in by
    main loop), and the UMAP-2D embedding for plotting.
    """
    embs = []
    ids = []
    skipped = 0
    for nid in vpn_ids:
        attrs = node_attrs.get(nid, {})
        cat_norm = normalize_category(attrs.get("concept_category"))
        if cat_norm != subtype and subtype in BODY_SUBTYPES:
            continue
        if subtype == "risk":
            if str(attrs.get("type", "")) != "concept":
                continue
            if cat_norm != "risk":
                continue
        if subtype == "intervention":
            if str(attrs.get("type", "")) != "intervention":
                continue
            mat = attrs.get("intervention_maturity", 0)
            try:
                mat_i = int(mat) if mat is not None else 0
            except Exception:
                mat_i = 0
            if mat_i < 3:
                continue
        e = parse_embedding(attrs.get("embedding"))
        if e is None:
            skipped += 1
            continue
        embs.append(e)
        ids.append(nid)
    if len(embs) < max(hdbscan_mcs) * 3:
        print(f"  [{subtype}] WARNING: only {len(embs)} embeddable nodes; skipped")
        return [], {}, None, None, len(embs), skipped
    X_raw = np.stack(embs)
    print(
        f"  [{subtype}] {len(ids):,} VPN nodes, raw dim {X_raw.shape[1]} "
        f"(skipped {skipped})"
    )

    if _umap_mod is None:
        raise RuntimeError("umap-learn not installed (required for BERTopic-style)")
    if _hdbscan_mod is None:
        raise RuntimeError("hdbscan not installed")

    t_umap = time.time()
    reducer_15 = _umap_mod.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    X_umap15 = reducer_15.fit_transform(X_raw)
    reducer_2 = _umap_mod.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    X_umap2 = reducer_2.fit_transform(X_raw)
    print(
        f"  [{subtype}] UMAP {umap_n_components}D + 2D done ({time.time() - t_umap:.1f}s)"
    )

    rows = []
    memberships = {}
    for mcs in hdbscan_mcs:
        t0 = time.time()
        clusterer = _hdbscan_mod.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean")
        labels = clusterer.fit_predict(X_umap15)
        n_noise = int((labels == -1).sum())
        unique_labels = sorted(set(int(c) for c in labels if c >= 0))
        n_clusters = len(unique_labels)
        if n_clusters < 2:
            print(f"  [{subtype}/hdbscan] mcs={mcs}: <2 clusters; skipped")
            continue
        memberships[int(mcs)] = (ids, labels, X_umap2)

        coverage = 1.0 - n_noise / len(labels)
        try:
            valid = labels != -1
            sil = float(
                silhouette_score(X_umap15[valid], labels[valid], metric="cosine")
            )
        except Exception:
            sil = None
        z_intra, z_inter = random_baseline_zscore(
            X_umap15, labels, n_shuffles=20, rng=rng
        )
        sizes = [int((labels == c).sum()) for c in unique_labels]
        elapsed = time.time() - t0
        print(
            f"  [{subtype}/hdbscan] mcs={mcs:3d}: k={n_clusters} "
            f"noise={n_noise} cov={coverage:.2f} sil={sil and round(sil, 3)} "
            f"z_intra={z_intra and round(z_intra, 2)} "
            f"z_inter={z_inter and round(z_inter, 2)} "
            f"sizes={min(sizes)}/{int(np.median(sizes))}/{max(sizes)} "
            f"({elapsed:.1f}s)"
        )
        rows.append(
            {
                "subtype": subtype,
                "min_cluster_size": int(mcs),
                "n_nodes": len(ids),
                "n_clusters_realized": int(n_clusters),
                "n_noise": n_noise,
                "coverage": round(coverage, 4),
                "silhouette_umap15": round(sil, 4) if sil is not None else None,
                "z_intra": round(z_intra, 3) if z_intra is not None else None,
                "z_inter": round(z_inter, 3) if z_inter is not None else None,
                "size_min": int(min(sizes)),
                "size_med": int(np.median(sizes)),
                "size_max": int(max(sizes)),
            }
        )

    return rows, memberships, X_raw, X_umap2, len(ids), skipped


def choose_clustering(metrics_df, subtype, sil_threshold, cov_threshold, z_threshold):
    df = metrics_df[metrics_df["subtype"] == subtype].copy()
    if df.empty:
        return None
    pass_mask = (
        (df["silhouette_umap15"].fillna(-1) > sil_threshold)
        & (df["coverage"].fillna(0) > cov_threshold)
        & (df["z_intra"].fillna(-1) > z_threshold)
        & (df["z_inter"].fillna(-1) > z_threshold)
    )
    if pass_mask.any():
        # Prefer smaller mcs (more clusters) among passing
        candidates = df[pass_mask].sort_values("min_cluster_size", ascending=True)
        row = candidates.iloc[0]
        status = "pass"
    else:
        # Best-of: max silhouette × coverage product
        df = df.copy()
        df["score"] = df["silhouette_umap15"].fillna(0) * df["coverage"].fillna(0)
        row = df.sort_values("score", ascending=False).iloc[0]
        status = "fail"
    return {
        "subtype": subtype,
        "min_cluster_size": int(row["min_cluster_size"]),
        "n_clusters_realized": int(row["n_clusters_realized"]),
        "n_noise": int(row["n_noise"]),
        "coverage": float(row["coverage"]),
        "silhouette_umap15": float(row["silhouette_umap15"])
        if row["silhouette_umap15"] is not None
        else None,
        "z_intra": float(row["z_intra"]) if row["z_intra"] is not None else None,
        "z_inter": float(row["z_inter"]) if row["z_inter"] is not None else None,
        "status": status,
    }


def plot_umap2d(subtype, ids, labels, X_umap2, chosen, out_path):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    ax.set_facecolor("white")
    unique = sorted(set(int(c) for c in labels if c >= 0))
    cmap = plt.cm.get_cmap("tab20", max(20, len(unique)))
    # Plot noise first (gray, low alpha)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            X_umap2[noise_mask, 0],
            X_umap2[noise_mask, 1],
            s=6,
            c="lightgray",
            alpha=0.4,
            label=f"noise (n={int(noise_mask.sum())})",
        )
    for i, cid in enumerate(unique):
        m = labels == cid
        ax.scatter(
            X_umap2[m, 0],
            X_umap2[m, 1],
            s=10,
            c=[cmap(i % cmap.N)],
            alpha=0.85,
            label=f"c{cid} (n={int(m.sum())})",
        )
    title = (
        f"{subtype} — UMAP 2D, HDBSCAN mcs={chosen['min_cluster_size']} "
        f"k={chosen['n_clusters_realized']} cov={chosen['coverage']:.2f} "
        f"sil={chosen['silhouette_umap15']:.3f}"
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if len(unique) <= 25:
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--paths-file", default=str(PATHS_DIR / "paths_hopwise_v4_sim0.9.jsonl")
    )
    ap.add_argument("--no-consim1", action="store_true")
    ap.add_argument("--sim-threshold", type=float, default=0.9)
    ap.add_argument("--output-suffix", default="sim0.9")
    ap.add_argument("--hdbscan-mcs", default="5,10,20,50,100")
    ap.add_argument("--umap-n-components", type=int, default=15)
    ap.add_argument("--umap-n-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--sil-threshold", type=float, default=0.25)
    ap.add_argument("--cov-threshold", type=float, default=0.50)
    ap.add_argument("--z-threshold", type=float, default=2.0)
    ap.add_argument(
        "--centroid-sim-cutoff",
        type=float,
        default=0.77,
        help="raw-embedding cosine sim cutoff for cluster membership "
        "(applied after HDBSCAN); single-pass refinement",
    )
    ap.add_argument(
        "--cutoff-sweep",
        default="0.73,0.75,0.77,0.79,0.81",
        help="comma-separated cutoff values for iter-1 robustness sweep",
    )
    ap.add_argument(
        "--iter-coverage-target",
        type=float,
        default=0.90,
        help="iterative residual clustering: stop when this fraction "
        "of nodes is in some post-cutoff cluster (or convergence)",
    )
    ap.add_argument(
        "--iter-max",
        type=int,
        default=50,
        help="max iterations of residual reclustering (safety cap; "
        "true stop = n_clustered_added==0)",
    )
    ap.add_argument(
        "--final-min-cluster-size",
        type=int,
        default=5,
        help="post-iteration filter: drop clusters whose final "
        "surviving member count is below this. Applied AFTER "
        "all iterations converge so transient tiny clusters "
        "don't terminate the loop prematurely.",
    )
    ap.add_argument(
        "--iter-k",
        type=int,
        default=40,
        help="fixed K for AgglomerativeClustering in residual "
        "iterations when --resid-method=agglo_ward",
    )
    ap.add_argument(
        "--resid-method",
        default="hdbscan",
        choices=["hdbscan", "agglo_ward"],
        help="iter 2+ residual reclustering method: 'hdbscan' "
        "(adaptive, default) or 'agglo_ward' (forced K)",
    )
    ap.add_argument(
        "--resid-mcs",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size for residual iterations "
        "(only when --resid-method=hdbscan)",
    )
    args = ap.parse_args()

    hdbscan_mcs = [int(x) for x in args.hdbscan_mcs.split(",")]
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("Phase 2 Step 4 rev8 [Task #7] — body recluster (BERTopic-style v2)")
    print("=" * 70)
    print(f"  paths_file       = {args.paths_file}")
    print(f"  output_suffix    = {args.output_suffix}")
    print(f"  hdbscan_mcs      = {hdbscan_mcs}")
    print(f"  UMAP n_components= {args.umap_n_components}")
    print(f"  UMAP n_neighbors = {args.umap_n_neighbors}")
    print(f"  UMAP min_dist    = {args.umap_min_dist}")
    print(f"  sil_threshold    = {args.sil_threshold}")
    print(f"  cov_threshold    = {args.cov_threshold}")
    print(f"  z_threshold      = {args.z_threshold}")
    print(f"  consim1 filter   = {not args.no_consim1}")
    print()

    # Load PKL
    t0 = time.time()
    print("Loading graph_node_attributes.pkl ...")
    with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  {len(node_attrs):,} nodes  ({time.time() - t0:.1f}s)")

    if not args.no_consim1:
        t1 = time.time()
        print("Loading graph_edge_data.pkl ...")
        with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
            edge_data = pickle.load(f)
        sim_edge_set = set()
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
        print(
            f"  {len(sim_edge_set):,} SIM>={args.sim_threshold} pairs "
            f"({time.time() - t1:.1f}s)"
        )
        del edge_data
    else:
        sim_edge_set = None

    def max_consec_sim(path):
        max_run = run = 0
        for i in range(len(path) - 1):
            a, b = int(path[i]), int(path[i + 1])
            if (min(a, b), max(a, b)) in sim_edge_set:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        return max_run

    # Build VPN
    t2 = time.time()
    print(f"\nReading {args.paths_file} ...")
    vpn_paperpair = set()
    n_total = n_kept = 0
    with open(args.paths_file) as f:
        for line in f:
            obj = json.loads(line)
            n_total += 1
            path = [int(x) for x in obj["path"]]
            interv_id = path[-1]
            mat = node_attrs.get(interv_id, {}).get("intervention_maturity", 0)
            try:
                mat_i = int(mat) if mat is not None else 0
            except Exception:
                mat_i = 0
            if mat_i < 3:
                continue
            if (not args.no_consim1) and max_consec_sim(path) > 1:
                continue
            n_kept += 1
            vpn_paperpair.update(path)
    print(f"  paths total={n_total:,} | qualifying={n_kept:,}")
    print(
        f"  VPN_paperpair = {len(vpn_paperpair):,} unique nodes "
        f"({time.time() - t2:.1f}s)"
    )

    # Per-subtype scan
    print("\n--- BERTopic-style scan ---")
    all_rows = []
    membership_records = {}  # (subtype, mcs) -> (ids, labels, X_umap2)
    raw_per_subtype = {}  # subtype -> X_raw (for representatives)
    pop_summary = []

    for nt in ALL_NODE_TYPES:
        print(f"\n[{nt}]")
        rows, memberships, X_raw, X_umap2, n_nodes, n_skipped = scan_subtype(
            nt,
            vpn_paperpair,
            node_attrs,
            hdbscan_mcs,
            args.umap_n_components,
            args.umap_n_neighbors,
            args.umap_min_dist,
            rng,
        )
        all_rows.extend(rows)
        for mcs, payload in memberships.items():
            membership_records[(nt, mcs)] = payload
        if X_raw is not None:
            raw_per_subtype[nt] = X_raw
        pop_summary.append(
            {"node_type": nt, "n_vpn": n_nodes, "n_skipped_no_emb": n_skipped}
        )

    metrics_df = pd.DataFrame(all_rows)
    metrics_csv = OUT_TABLES / f"body_kscan_metrics_{args.output_suffix}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nWritten: {metrics_csv.name} ({len(metrics_df)} rows)")

    pd.DataFrame(pop_summary).to_csv(
        OUT_TABLES / f"body_kscan_population_summary_{args.output_suffix}.csv",
        index=False,
    )

    # Pareto choice per subtype
    print("\n--- Pareto-frontier choice (BERTopic-style) ---")
    chosen_rows = []
    chosen_by_subtype = {}
    for nt in ALL_NODE_TYPES:
        if metrics_df[metrics_df["subtype"] == nt].empty:
            continue
        chosen = choose_clustering(
            metrics_df, nt, args.sil_threshold, args.cov_threshold, args.z_threshold
        )
        if chosen is None:
            continue
        chosen_by_subtype[nt] = chosen
        chosen_rows.append(
            {
                "node_type": nt,
                "min_cluster_size": chosen["min_cluster_size"],
                "n_clusters_realized": chosen["n_clusters_realized"],
                "n_noise": chosen["n_noise"],
                "coverage": round(chosen["coverage"], 4),
                "silhouette_umap15": round(chosen["silhouette_umap15"], 4)
                if chosen["silhouette_umap15"] is not None
                else None,
                "z_intra": round(chosen["z_intra"], 3)
                if chosen["z_intra"] is not None
                else None,
                "z_inter": round(chosen["z_inter"], 3)
                if chosen["z_inter"] is not None
                else None,
                "status": chosen["status"],
                "sil_threshold": args.sil_threshold,
                "cov_threshold": args.cov_threshold,
                "z_threshold": args.z_threshold,
            }
        )
        verdict = "PASS" if chosen["status"] == "pass" else "FAIL"
        print(
            f"  [{nt}] mcs={chosen['min_cluster_size']} k={chosen['n_clusters_realized']} "
            f"cov={chosen['coverage']:.2f} sil={chosen['silhouette_umap15']:.3f} "
            f"z_intra={chosen['z_intra']:.2f} z_inter={chosen['z_inter']:.2f}  "
            f"** {verdict} **"
        )

    chosen_df = pd.DataFrame(chosen_rows)
    chosen_csv = OUT_TABLES / f"body_kscan_chosen_k_{args.output_suffix}.csv"
    chosen_df.to_csv(chosen_csv, index=False)
    print(f"\nWritten: {chosen_csv.name}")

    # Apply centroid-sim cutoff (single-pass at chosen mcs) — used for SWEEP only
    print("\n--- Centroid-sim cutoff sweep (single-pass at chosen mcs) ---")
    cutoff_values = [float(x) for x in args.cutoff_sweep.split(",")]
    if args.centroid_sim_cutoff not in cutoff_values:
        cutoff_values = sorted(set(cutoff_values + [args.centroid_sim_cutoff]))

    sweep_rows = []
    ari_rows = []

    for nt in ALL_NODE_TYPES:
        if nt not in chosen_by_subtype:
            continue
        chosen = chosen_by_subtype[nt]
        ids, labels_init, X_umap2 = membership_records[(nt, chosen["min_cluster_size"])]
        X_raw = raw_per_subtype[nt]
        per_cutoff_labels = {}
        for cutoff in cutoff_values:
            labels_post, _ref, _init = apply_centroid_cutoff(X_raw, labels_init, cutoff)
            per_cutoff_labels[cutoff] = labels_post
            n_post_clusters = len(set(int(c) for c in labels_post if c >= 0))
            n_noise_post = int((labels_post == -1).sum())
            coverage_post = 1.0 - n_noise_post / len(labels_post)
            sweep_rows.append(
                {
                    "subtype": nt,
                    "min_cluster_size": chosen["min_cluster_size"],
                    "cutoff": cutoff,
                    "n_clusters_post": int(n_post_clusters),
                    "n_noise_post": int(n_noise_post),
                    "coverage_post": round(coverage_post, 4),
                }
            )
        ref_labels = per_cutoff_labels[args.centroid_sim_cutoff]
        if _ari is not None:
            for cutoff in cutoff_values:
                lab = per_cutoff_labels[cutoff]
                mask = (ref_labels != -1) & (lab != -1)
                if (
                    int(mask.sum()) >= 4
                    and len(set(ref_labels[mask])) > 1
                    and len(set(lab[mask])) > 1
                ):
                    ari_val = float(_ari(ref_labels[mask], lab[mask]))
                else:
                    ari_val = None
                ari_rows.append(
                    {
                        "subtype": nt,
                        "ref_cutoff": args.centroid_sim_cutoff,
                        "compare_cutoff": cutoff,
                        "n_intersection": int(mask.sum()),
                        "ari": round(ari_val, 4) if ari_val is not None else None,
                    }
                )

    # Iterative residual reclustering — final labels per subtype
    # Iter 1: HDBSCAN at chosen mcs (variable-k natural-density).
    # Iter 2+: AgglomerativeClustering(K=iter_k) on UMAP-15D — every input
    #          node assigned to one of K buckets; only 0.77 cutoff filters.
    if args.resid_method == "hdbscan":
        resid_desc = f"HDBSCAN(mcs={args.resid_mcs})"
    else:
        resid_desc = f"Agglo_ward(K={args.iter_k})"
    print(
        f"\n--- Iterative residual reclustering (target cov={args.iter_coverage_target}, "
        f"max_iter={args.iter_max}, iter1=HDBSCAN(mcs=chosen), "
        f"iter2+={resid_desc}, cutoff={args.centroid_sim_cutoff}) ---"
    )
    post_labels = {}  # (subtype) -> final labels (post all iterations)
    iter_records_all = []  # per-iteration stats per subtype
    iter_X_umap2 = {}  # (subtype) -> 2D UMAP from iter 1 (for plotting)
    nodeid_to_label_default = {}  # for VPN_strict
    init_centroids_per_subtype = {}  # (subtype) -> {cluster_id: initial_centroid}

    for nt in ALL_NODE_TYPES:
        if nt not in chosen_by_subtype:
            continue
        chosen_mcs_iter1 = int(chosen_by_subtype[nt]["min_cluster_size"])
        ids, _, _ = membership_records[(nt, chosen_mcs_iter1)]
        X_raw = raw_per_subtype[nt]
        labels_final, iter_records, X_umap2_first, init_cents = (
            iterative_residual_recluster(
                X_raw,
                chosen_mcs_iter1,
                args.iter_k,
                args.centroid_sim_cutoff,
                args.umap_n_components,
                args.umap_n_neighbors,
                args.umap_min_dist,
                args.iter_coverage_target,
                args.iter_max,
                resid_method=args.resid_method,
                resid_mcs=args.resid_mcs,
                strict_min_size=args.final_min_cluster_size,
            )
        )
        # Per-iter strict filter is already applied inside the loop, so
        # labels_final is clean. No post-filter needed; defensive sanity-check
        # confirms every cluster has >= final_min_cluster_size members.
        post_labels[nt] = labels_final
        iter_X_umap2[nt] = X_umap2_first
        init_centroids_per_subtype[nt] = init_cents
        n_final_clusters = len(set(int(c) for c in labels_final if c >= 0))
        cov_final = (labels_final != -1).sum() / len(labels_final)
        cluster_sizes = [
            int((labels_final == c).sum())
            for c in sorted(set(int(c) for c in labels_final if c >= 0))
        ]
        smallest = min(cluster_sizes) if cluster_sizes else 0
        print(
            f"  [{nt}] iters={len(iter_records)} "
            f"k_final={n_final_clusters} smallest_cluster={smallest} "
            f"cov_final={cov_final:.2%} stopped={iter_records[-1]['stopped_reason']}"
        )
        for r in iter_records:
            r["subtype"] = nt
            iter_records_all.append(r)
        for i, nid in enumerate(ids):
            nodeid_to_label_default[(nt, int(nid))] = int(labels_final[i])

    pd.DataFrame(iter_records_all).to_csv(
        OUT_TABLES / f"body_kscan_iter_records_{args.output_suffix}.csv", index=False
    )
    print(
        f"Wrote body_kscan_iter_records_{args.output_suffix}.csv "
        f"({len(iter_records_all)} rows)"
    )

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_csv = OUT_TABLES / f"body_kscan_cutoff_sweep_{args.output_suffix}.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"Written: {sweep_csv.name} ({len(sweep_df)} rows)")

    ari_df = pd.DataFrame(ari_rows)
    ari_csv = OUT_TABLES / f"body_kscan_cutoff_ari_{args.output_suffix}.csv"
    ari_df.to_csv(ari_csv, index=False)
    print(f"Written: {ari_csv.name} ({len(ari_df)} rows)")

    # Print compact cutoff sweep summary per subtype
    for nt in ALL_NODE_TYPES:
        if nt not in chosen_by_subtype:
            continue
        sub = sweep_df[sweep_df["subtype"] == nt].sort_values("cutoff")
        line = f"  [{nt}] " + " | ".join(
            f"{c:.2f}:k={int(r)}cov={float(cov):.2f}"
            for c, r, cov in zip(
                sub["cutoff"], sub["n_clusters_post"], sub["coverage_post"]
            )
        )
        print(line)

    # Representatives + plots POST-cutoff (default cutoff)
    print(
        f"\n--- Representatives + 2D UMAP plots (post cutoff={args.centroid_sim_cutoff}) ---"
    )
    rep_rows_all = []
    for nt in ALL_NODE_TYPES:
        if nt not in chosen_by_subtype:
            continue
        chosen = chosen_by_subtype[nt]
        ids, labels_init, X_umap2 = membership_records[(nt, chosen["min_cluster_size"])]
        labels_post = post_labels[nt]
        X_raw = raw_per_subtype[nt]
        rep_rows = representatives_per_cluster(
            ids,
            X_raw,
            labels_post,
            node_attrs,
            n_each=3,
            initial_centroids=init_centroids_per_subtype.get(nt),
        )
        for r in rep_rows:
            r["subtype"] = nt
        rep_rows_all.extend(rep_rows)
        # Update chosen dict with post-cutoff stats for plot title
        n_post = len(set(int(c) for c in labels_post if c >= 0))
        n_noise_post = int((labels_post == -1).sum())
        cov_post = 1.0 - n_noise_post / len(labels_post)
        chosen_for_plot = dict(chosen)
        chosen_for_plot["n_clusters_realized"] = n_post
        chosen_for_plot["coverage"] = cov_post
        plot_path = OUT_TABLES / f"body_kscan_pareto_plot_{nt}_{args.output_suffix}.png"
        plot_umap2d(nt, ids, labels_post, X_umap2, chosen_for_plot, plot_path)
        print(f"  Wrote {plot_path.name} (post k={n_post} cov={cov_post:.2f})")

    rep_csv = OUT_TABLES / f"body_kscan_representatives_{args.output_suffix}.csv"
    pd.DataFrame(rep_rows_all).to_csv(rep_csv, index=False)
    print(f"Wrote {rep_csv.name} ({len(rep_rows_all)} rows)")

    # Build VPN_strict_paperpair: paths where every body node has a non-noise
    # cluster ID after the default cutoff.
    print("\n--- Building VPN_strict from post-cutoff labels ---")
    n_paths_strict = 0
    n_paths_total_qual = 0
    strict_paths_path = PATHS_DIR / f"vpn_strict_paths_{args.output_suffix}.jsonl"
    body_subtype_set = set(BODY_SUBTYPES)
    with open(args.paths_file) as fin, open(strict_paths_path, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            path = [int(x) for x in obj["path"]]
            interv_id = path[-1]
            mat = node_attrs.get(interv_id, {}).get("intervention_maturity", 0)
            try:
                mat_i = int(mat) if mat is not None else 0
            except Exception:
                mat_i = 0
            if mat_i < 3:
                continue
            if (not args.no_consim1) and max_consec_sim(path) > 1:
                continue
            n_paths_total_qual += 1
            # Body nodes are everything except first (risk) and last (intervention)
            body_nodes = path[1:-1]
            all_mapped = True
            for bnid in body_nodes:
                cat_norm = normalize_category(
                    node_attrs.get(int(bnid), {}).get("concept_category")
                )
                if cat_norm not in body_subtype_set:
                    all_mapped = False
                    break
                lbl = nodeid_to_label_default.get((cat_norm, int(bnid)))
                if lbl is None or lbl == -1:
                    all_mapped = False
                    break
            if all_mapped:
                fout.write(json.dumps(obj) + "\n")
                n_paths_strict += 1
    strict_retention = n_paths_strict / n_paths_total_qual if n_paths_total_qual else 0
    print(
        f"  paths qual (pre-cutoff)={n_paths_total_qual:,}  strict={n_paths_strict:,}  "
        f"retention={strict_retention:.2%}"
    )
    print(f"  Written: {strict_paths_path}")

    # Save VPN_strict retention summary
    pd.DataFrame(
        [
            {
                "n_paths_qualifying_pre_cutoff": int(n_paths_total_qual),
                "n_paths_strict_post_cutoff": int(n_paths_strict),
                "retention_rate": round(strict_retention, 4),
                "centroid_sim_cutoff": args.centroid_sim_cutoff,
            }
        ]
    ).to_csv(
        OUT_TABLES / f"vpn_strict_retention_{args.output_suffix}.csv",
        index=False,
    )

    # Save memberships PKL (POST-cutoff)
    print("\n--- Building cluster_memberships_rev8 PKL (post-cutoff) ---")
    new_cm = {}
    for nt in ALL_NODE_TYPES:
        if nt not in chosen_by_subtype:
            continue
        chosen = chosen_by_subtype[nt]
        ids, _, _ = membership_records[(nt, chosen["min_cluster_size"])]
        labels_post = post_labels[nt]
        unique_lbls = sorted(set(int(c) for c in labels_post if c >= 0))
        for cid in unique_lbls:
            members = [int(ids[i]) for i in range(len(ids)) if labels_post[i] == cid]
            if not members:
                continue
            key = ("rev8_vpn_post", "umap_hdbscan_cutoff", nt, "hdbscan", str(int(cid)))
            new_cm[key] = members
    out_pkl = STEP1_DIR / f"cluster_memberships_rev8_{args.output_suffix}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(new_cm, f)
    print(f"  Written: {out_pkl} ({len(new_cm)} cluster records)")

    print("\nDone.")


if __name__ == "__main__":
    main()
