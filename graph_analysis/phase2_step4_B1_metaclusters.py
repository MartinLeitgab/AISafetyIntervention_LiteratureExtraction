"""
Phase 2 Track B1 — Meta-clustering of risk and intervention clusters
=====================================================================
Centroids computed exclusively from VPN_consim1 members.

VPN_consim1 = union of all nodes on any qualifying consim1 path.
A qualifying consim1 path satisfies simultaneously:
  - SIM edges: cos_sim >= 0.9  (baked into paths_unconstrained_sim0.9.jsonl)
  - EDGE edges: confidence >= 3 (baked into path generation via load_graph(min_conf=3))
  - Intervention endpoint: maturity >= 3 (baked into path generation)
  - consim1: max consecutive SIM hops <= 1 (applied here)

Construction:
  Pass 1 — scan sim0.9 file with maturity>=3 check → vpn_unconstrained
  Build sim_edge_set restricted to vpn_unconstrained pairs (SIM>=0.9)
  Pass 2 — scan sim0.9 file, apply consim1 filter → vpn_consim1
  PKL cluster members intersected with vpn_consim1 before centroid computation.

Outputs (all in step4_finalanalysis/):
  step4_metaclusters/risk_meta_assignments.csv
  step4_metaclusters/intervention_meta_assignments.csv
  step4_metaclusters/risk_intercent_sim_matrix.csv
  step4_metaclusters/interv_intercent_sim_matrix.csv
  step4_metaclusters/meta_cluster_ri_connectivity.csv
  step4_metaclusters/risk_sim_heatmap.png
  step4_metaclusters/interv_sim_heatmap.png
  step4_metaclusters/risk_dendrogram.png
  step4_metaclusters/interv_dendrogram.png
  step4_metaclusters/meta_connectivity_network.png
"""

import gc
import json
import os
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP1_DIR = os.path.join(RESULTS_DIR, "step1_load_and_parse_umapwithoutlocalsatellites")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
TABLES_DIR = os.path.join(STEP4_DIR, "step4_cluster_tables")
CONN_DIR = os.path.join(STEP4_DIR, "step4_connectivity")
OUT_DIR = os.path.join(STEP4_DIR, "step4_metaclusters")
NAMING_DIR = os.path.join(RESULTS_DIR, "step5_naming")
RAWPATHS_DIR = os.path.join(ROOT, "phase1_rawpathsfiles")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_embedding(emb_raw):
    if isinstance(emb_raw, np.ndarray):
        return emb_raw.astype(np.float32)
    s = str(emb_raw).strip().strip("<>")
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


# ─── STEP 1: Load PKL files ───────────────────────────────────────────────────
print("Loading cluster_memberships.pkl ...")
t0 = time.time()
with open(os.path.join(STEP1_DIR, "cluster_memberships.pkl"), "rb") as f:
    cm = pickle.load(f)
print(f"  cm: {len(cm)} keys  ({time.time() - t0:.1f}s)")

print("Loading node_attrs.pkl ...")
t1 = time.time()
with open(os.path.join(STEP1_DIR, "graph_node_attributes.pkl"), "rb") as f:
    node_attrs = pickle.load(f)
print(f"  node_attrs: {len(node_attrs)} nodes  ({time.time() - t1:.1f}s)")

print("Loading graph_edge_data.pkl ...")
t2 = time.time()
with open(os.path.join(STEP1_DIR, "graph_edge_data.pkl"), "rb") as f:
    edge_data = pickle.load(f)
print(f"  edge_data: {len(edge_data)} edges  ({time.time() - t2:.1f}s)")

# ─── STEP 2: Build embedding cache ────────────────────────────────────────────
print("Building embedding cache ...")
t3 = time.time()
emb_cache = {}
for nid, attrs in node_attrs.items():
    emb_raw = attrs.get("embedding")
    if emb_raw is not None:
        try:
            emb_cache[int(nid)] = parse_embedding(emb_raw)
        except Exception:
            pass
print(f"  emb_cache: {len(emb_cache)} nodes  ({time.time() - t3:.1f}s)")

# ─── STEP 3: Build VPN_consim1 ────────────────────────────────────────────────
# Pass 1: vpn_unconstrained (maturity>=3, any path in sim0.9 file)
print("\nBuilding VPN_consim1 ...")
sim09_file = os.path.join(RAWPATHS_DIR, "paths_unconstrained_sim0.9.jsonl")

print("  Pass 1: vpn_unconstrained (maturity>=3) ...")
t_p1 = time.time()
vpn_unconstrained = set()
with open(sim09_file) as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            vpn_unconstrained.update(path)
print(
    f"    vpn_unconstrained: {len(vpn_unconstrained)} nodes  ({time.time() - t_p1:.1f}s)"
)

# Build sim_edge_set restricted to vpn_unconstrained
print("  Building sim_edge_set (SIM>=0.9, VPN-restricted) ...")
t_se = time.time()
sim_edge_set = set()
for e in edge_data:
    if str(e.get("type", "")).upper() == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= 0.9:
            try:
                s, tgt = int(e["source"]), int(e["target"])
                if s in vpn_unconstrained and tgt in vpn_unconstrained:
                    sim_edge_set.add((min(s, tgt), max(s, tgt)))
            except (ValueError, TypeError):
                pass
print(f"    sim_edge_set: {len(sim_edge_set)} pairs  ({time.time() - t_se:.1f}s)")


def max_consec_sim(path_ids):
    max_run = run = 0
    for i in range(len(path_ids) - 1):
        a, b = int(path_ids[i]), int(path_ids[i + 1])
        if (min(a, b), max(a, b)) in sim_edge_set:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


# Pass 2: vpn_consim1 (consim1 filter on top of maturity>=3)
print("  Pass 2: vpn_consim1 (consim1 = max consecutive SIM hops <= 1) ...")
t_p2 = time.time()
vpn_consim1 = set()
n_consim1_paths = 0
with open(sim09_file) as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) < 3:
            continue
        if max_consec_sim(path) <= 1:
            vpn_consim1.update(path)
            n_consim1_paths += 1
print(
    f"    vpn_consim1: {len(vpn_consim1)} nodes from {n_consim1_paths} paths  ({time.time() - t_p2:.1f}s)"
)

# Free edge_data — no longer needed
del edge_data
gc.collect()

# ─── STEP 4: Extract PKL cluster assignments ──────────────────────────────────
print("\nExtracting PKL cluster assignments ...")


def get_type_clusters(cm, ec, mode, node_type, algo="agglomerative"):
    result = {}
    try:
        ec_float = float(ec)
    except Exception:
        ec_float = None
    for k, v in cm.items():
        k0 = k[0]
        try:
            match = float(k0) == ec_float
        except Exception:
            match = str(k0) == str(ec)
        if match and str(k[1]) == mode and str(k[2]) == node_type and str(k[3]) == algo:
            result[str(k[4])] = [int(n) for n in v]
    return result


risk_clusters_pkl = get_type_clusters(cm, "0.9", "unconstrained", "risk")
interv_clusters_pkl = get_type_clusters(cm, "0.9", "unconstrained", "intervention")
print(
    f"  Risk PKL clusters: {len(risk_clusters_pkl)}, Intervention PKL clusters: {len(interv_clusters_pkl)}"
)


# ─── STEP 5: Compute VPN_consim1-filtered cluster centroids ───────────────────
def compute_centroids(clusters_dict, emb_cache, vpn):
    """
    Compute L2-normalised centroid from VPN_consim1-filtered member embeddings.
    Only members in vpn are used. Clusters with no qualifying members are skipped.
    Returns: {cid_str: centroid_array}, {cid_str: n_vpn_members}
    """
    centroids = {}
    n_vpn_per_cluster = {}
    for cid_str, member_ids in clusters_dict.items():
        vpn_members = [nid for nid in member_ids if nid in vpn]
        embs = [emb_cache[nid] for nid in vpn_members if nid in emb_cache]
        if len(embs) == 0:
            print(
                f"  WARNING: cluster {cid_str} has no VPN_consim1 members with embeddings — excluded"
            )
            continue
        X = np.stack(embs).astype(np.float32)
        X_norm = normalize_rows(X)
        centroid = X_norm.mean(axis=0)
        centroid = centroid / (norm(centroid) + 1e-8)
        centroids[cid_str] = centroid.astype(np.float32)
        n_vpn_per_cluster[cid_str] = len(vpn_members)
    return centroids, n_vpn_per_cluster


print("Computing risk cluster centroids (VPN_consim1 members only) ...")
risk_centroids, risk_n_vpn = compute_centroids(
    risk_clusters_pkl, emb_cache, vpn_consim1
)
print(f"  {len(risk_centroids)} risk cluster centroids computed")
for cid, n in sorted(risk_n_vpn.items(), key=lambda x: int(x[0])):
    if n < 10:
        print(f"    R{cid}: n_vpn={n}  (small)")

print("Computing intervention cluster centroids (VPN_consim1 members only) ...")
interv_centroids, interv_n_vpn = compute_centroids(
    interv_clusters_pkl, emb_cache, vpn_consim1
)
print(f"  {len(interv_centroids)} intervention cluster centroids computed")

# ─── STEP 6: Load cluster name tables (v2 naming from step5_naming) ───────────
risk_names_df = pd.read_csv(os.path.join(NAMING_DIR, "risk_cluster_names_llm_v2.csv"))
interv_names_df = pd.read_csv(
    os.path.join(NAMING_DIR, "intervention_cluster_names_llm_v2.csv")
)

name_col_r = "final_name" if "final_name" in risk_names_df.columns else "llm_name"
name_col_i = "final_name" if "final_name" in interv_names_df.columns else "llm_name"

risk_name_map = dict(
    zip(risk_names_df["cluster_id"].astype(str), risk_names_df[name_col_r].str[:80])
)
interv_name_map = dict(
    zip(interv_names_df["cluster_id"].astype(str), interv_names_df[name_col_i].str[:80])
)

# n_nodes = VPN_consim1 member count (computed above)
risk_nnodes_map = {cid: n for cid, n in risk_n_vpn.items()}
interv_nnodes_map = {cid: n for cid, n in interv_n_vpn.items()}


# ─── STEP 7: Build inter-centroid similarity matrices ─────────────────────────
def build_sim_matrix(centroids_dict, name_map):
    cids = sorted(centroids_dict.keys(), key=lambda x: int(x))
    labels = [f"C{c}: {name_map.get(c, c)[:45]}" for c in cids]
    short_labels = [f"C{c}" for c in cids]
    C = np.stack([centroids_dict[c] for c in cids]).astype(np.float32)
    sim = C @ C.T  # centroids are already L2-normalised
    return cids, labels, short_labels, sim


print("\nBuilding similarity matrices ...")
risk_cids, risk_labels, risk_short, risk_sim = build_sim_matrix(
    risk_centroids, risk_name_map
)
interv_cids, interv_labels, interv_short, interv_sim = build_sim_matrix(
    interv_centroids, interv_name_map
)
print(
    f"  Risk: {len(risk_cids)}×{len(risk_cids)}, Intervention: {len(interv_cids)}×{len(interv_cids)}"
)

pd.DataFrame(risk_sim, index=risk_cids, columns=risk_cids).to_csv(
    os.path.join(OUT_DIR, "risk_intercent_sim_matrix.csv")
)
pd.DataFrame(interv_sim, index=interv_cids, columns=interv_cids).to_csv(
    os.path.join(OUT_DIR, "interv_intercent_sim_matrix.csv")
)
print("  Saved similarity matrices")

ri = risk_sim[np.triu_indices(len(risk_cids), k=1)]
ii = interv_sim[np.triu_indices(len(interv_cids), k=1)]
print(f"  Risk sim: mean={ri.mean():.3f}, min={ri.min():.3f}, max={ri.max():.3f}")
print(
    f"  Intervention sim: mean={ii.mean():.3f}, min={ii.min():.3f}, max={ii.max():.3f}"
)


# ─── STEP 8: Hierarchical meta-clustering ─────────────────────────────────────
def run_meta_clustering(sim_matrix, cids, k):
    dist = 1.0 - sim_matrix
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, k, criterion="maxclust")
    return labels, Z


print("\nRunning meta-clustering ...")
risk_meta_k10, risk_Z = run_meta_clustering(risk_sim, risk_cids, k=10)
interv_meta_k10, interv_Z = run_meta_clustering(interv_sim, interv_cids, k=10)
risk_meta_k12, _ = run_meta_clustering(risk_sim, risk_cids, k=12)
interv_meta_k12, _ = run_meta_clustering(interv_sim, interv_cids, k=12)
interv_meta_k15, _ = run_meta_clustering(interv_sim, interv_cids, k=15)

print(
    f"  Risk meta-clusters (k=10): sizes = {[int((risk_meta_k10 == m).sum()) for m in range(1, 11)]}"
)
print(
    f"  Interv meta-clusters (k=10): sizes = {[int((interv_meta_k10 == m).sum()) for m in range(1, 11)]}"
)


# ─── STEP 9: Save meta-cluster assignment tables ──────────────────────────────
def save_meta_assignments(
    cids,
    meta_labels_10,
    meta_labels_12,
    name_map,
    nnodes_map,
    cluster_type,
    out_filename,
    meta_labels_15=None,
):
    rows = []
    for i, cid in enumerate(cids):
        row = {
            "cluster_id": int(cid),
            "meta_k10": int(meta_labels_10[i]),
            "meta_k12": int(meta_labels_12[i]),
            "cluster_name": name_map.get(cid, ""),
            "n_nodes": nnodes_map.get(cid, 0),
        }
        if meta_labels_15 is not None:
            row["meta_k15"] = int(meta_labels_15[i])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["meta_k10", "cluster_id"])
    out_path = os.path.join(OUT_DIR, out_filename)
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")

    print(f"\n  === {cluster_type} Meta-clusters (k=10) ===")
    for mc in sorted(df["meta_k10"].unique()):
        sub = df[df["meta_k10"] == mc]
        total_nodes = sub["n_nodes"].sum()
        cluster_list = ", ".join(
            f"C{r['cluster_id']}:{r['cluster_name'][:35]}" for _, r in sub.iterrows()
        )
        print(
            f"  Meta-{mc} ({len(sub)} clusters, {total_nodes} nodes): {cluster_list[:120]}"
        )
    return df


print("\n--- Risk meta-cluster assignments ---")
risk_meta_df = save_meta_assignments(
    risk_cids,
    risk_meta_k10,
    risk_meta_k12,
    risk_name_map,
    risk_nnodes_map,
    "Risk",
    "risk_meta_assignments.csv",
)
print("\n--- Intervention meta-cluster assignments ---")
interv_meta_df = save_meta_assignments(
    interv_cids,
    interv_meta_k10,
    interv_meta_k12,
    interv_name_map,
    interv_nnodes_map,
    "Intervention",
    "intervention_meta_assignments.csv",
    meta_labels_15=interv_meta_k15,
)

# ─── STEP 10: Build meta-cluster connectivity from cross_config_ri_pairs.csv ──
print("\nBuilding meta-cluster R→I connectivity ...")
ri_pairs = pd.read_csv(os.path.join(CONN_DIR, "cross_config_ri_pairs.csv"))
ri_pairs["risk_cid_str"] = ri_pairs["risk_cid"].astype(int).astype(str)
ri_pairs["interv_cid_str"] = ri_pairs["interv_cid"].astype(int).astype(str)

risk_to_meta = dict(
    zip(risk_meta_df["cluster_id"].astype(str), risk_meta_df["meta_k10"])
)
interv_to_meta = dict(
    zip(interv_meta_df["cluster_id"].astype(str), interv_meta_df["meta_k10"])
)

ri_pairs["risk_meta"] = ri_pairs["risk_cid_str"].map(risk_to_meta)
ri_pairs["interv_meta"] = ri_pairs["interv_cid_str"].map(interv_to_meta)

# Drop rows where risk or interv cluster was excluded (no VPN_consim1 members)
ri_pairs = ri_pairs.dropna(subset=["risk_meta", "interv_meta"])

meta_conn = (
    ri_pairs.groupby(["risk_meta", "interv_meta"])
    .agg(
        n_paths_c1=("n_paths_c1", "sum"),
        n_paths_c0=("n_paths_c0", "sum"),
        n_cluster_pairs=("risk_cid", "count"),
    )
    .reset_index()
)
meta_conn = meta_conn.sort_values("n_paths_c1", ascending=False)


def get_meta_name(meta_df, meta_id, name_map, max_len=60):
    sub = meta_df[meta_df["meta_k10"] == meta_id].sort_values(
        "n_nodes", ascending=False
    )
    if len(sub) == 0:
        return f"Meta-{meta_id}"
    top_name = name_map.get(str(int(sub.iloc[0]["cluster_id"])), f"Meta-{meta_id}")
    return top_name[:max_len]


meta_conn["risk_meta_name"] = meta_conn["risk_meta"].apply(
    lambda m: get_meta_name(risk_meta_df, m, risk_name_map)
)
meta_conn["interv_meta_name"] = meta_conn["interv_meta"].apply(
    lambda m: get_meta_name(interv_meta_df, m, interv_name_map)
)

meta_conn_path = os.path.join(OUT_DIR, "meta_cluster_ri_connectivity.csv")
meta_conn.to_csv(meta_conn_path, index=False)
print(f"  Saved {meta_conn_path} ({len(meta_conn)} meta-cluster pairs)")
print(
    f"  Total meta-cluster R→I pairs with n_paths_c1>0: {(meta_conn['n_paths_c1'] > 0).sum()}"
)

# ─── STEP 11: Plots ───────────────────────────────────────────────────────────
CMAP_SIM = "RdYlGn"


def plot_sim_heatmap(sim_matrix, cids, meta_labels, name_map, title, out_path, k=10):
    order = sorted(range(len(cids)), key=lambda i: (meta_labels[i], int(cids[i])))
    sorted_cids = [cids[i] for i in order]
    sorted_meta = [meta_labels[i] for i in order]
    sorted_sim = sim_matrix[np.ix_(order, order)]

    short_labels = [f"C{c}" for c in sorted_cids]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sorted_sim, cmap=CMAP_SIM, vmin=0.6, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Cosine similarity")

    ax.set_xticks(range(len(sorted_cids)))
    ax.set_yticks(range(len(sorted_cids)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)

    colors = plt.cm.tab10(np.linspace(0, 1, k))
    prev_mc = sorted_meta[0]
    block_start = 0
    for i in range(1, len(sorted_cids) + 1):
        mc = sorted_meta[i] if i < len(sorted_cids) else -1
        if mc != prev_mc or i == len(sorted_cids):
            block_end = i
            rect = plt.Rectangle(
                (block_start - 0.5, block_start - 0.5),
                block_end - block_start,
                block_end - block_start,
                linewidth=2,
                edgecolor=colors[prev_mc % k],
                facecolor="none",
            )
            ax.add_patch(rect)
            block_start = block_end
            prev_mc = mc

    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_dendrogram(Z, cids, name_map, title, out_path):
    import textwrap

    labels = [textwrap.fill(f"C{c}: {name_map.get(c, str(c))}", width=50) for c in cids]
    n = len(labels)
    fig_h = max(10, n * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    dendrogram(Z, labels=labels, ax=ax, orientation="left", leaf_font_size=7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Distance (1 - cos sim)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_meta_connectivity(
    meta_conn_df,
    risk_meta_df,
    interv_meta_df,
    risk_name_map,
    interv_name_map,
    out_path,
    k=10,
    top_n=30,
):
    import textwrap as _tw

    def meta_display(meta_df, mc_id, name_map, prefix):
        sub = meta_df[meta_df["meta_k10"] == mc_id].sort_values(
            "n_nodes", ascending=False
        )
        n_clusters = len(sub)
        total_nodes = int(sub["n_nodes"].sum())
        dominant_name = (
            name_map.get(str(int(sub.iloc[0]["cluster_id"])), f"Meta-{mc_id}")
            if len(sub) > 0
            else f"Meta-{mc_id}"
        )
        theme = "\n".join(_tw.wrap(dominant_name, width=28))
        return f"{prefix}{mc_id}\n({n_clusters}cl/{total_nodes}n)\n{theme}"

    risk_mcs = sorted(risk_meta_df["meta_k10"].unique())
    interv_mcs = sorted(interv_meta_df["meta_k10"].unique())
    n_r = len(risk_mcs)
    n_i = len(interv_mcs)

    risk_pos = {mc: (0, (n_r - 1 - idx) * 2.5) for idx, mc in enumerate(risk_mcs)}
    interv_pos = {mc: (5, (n_i - 1 - idx) * 2.5) for idx, mc in enumerate(interv_mcs)}

    top_edges = meta_conn_df.nlargest(top_n, "n_paths_c1")
    max_paths = top_edges["n_paths_c1"].max()

    fig, ax = plt.subplots(figsize=(20, max(n_r, n_i) * 2.0))
    ax.set_xlim(-3.5, 9.5)
    ax.set_ylim(-2, max(n_r, n_i) * 2.5)
    ax.axis("off")

    for _, row in top_edges.iterrows():
        rm, im = row["risk_meta"], row["interv_meta"]
        if rm not in risk_pos or im not in interv_pos:
            continue
        rx, ry = risk_pos[rm]
        ix, iy = interv_pos[im]
        lw = 0.5 + 5.0 * (row["n_paths_c1"] / max_paths)
        alpha = 0.3 + 0.5 * (row["n_paths_c1"] / max_paths)
        ax.plot([rx, ix], [ry, iy], "b-", lw=lw, alpha=alpha, zorder=1)
        if row["n_paths_c1"] > max_paths * 0.1:
            ax.text(
                (rx + ix) / 2,
                (ry + iy) / 2,
                str(int(row["n_paths_c1"])),
                fontsize=5,
                color="blue",
                alpha=0.7,
                ha="center",
                va="center",
            )

    r_colors = plt.cm.Reds(np.linspace(0.4, 0.8, n_r))
    for idx, mc in enumerate(risk_mcs):
        x, y = risk_pos[mc]
        sub = risk_meta_df[risk_meta_df["meta_k10"] == mc]
        size = 300 + sub["n_nodes"].sum() * 0.5
        ax.scatter(
            [x], [y], s=size, c=[r_colors[idx]], zorder=3, edgecolors="black", lw=0.5
        )
        ax.text(
            x - 0.3,
            y,
            meta_display(risk_meta_df, mc, risk_name_map, "R-Meta"),
            fontsize=6,
            ha="right",
            va="center",
            zorder=4,
        )

    i_colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_i))
    for idx, mc in enumerate(interv_mcs):
        x, y = interv_pos[mc]
        sub = interv_meta_df[interv_meta_df["meta_k10"] == mc]
        size = 300 + sub["n_nodes"].sum() * 0.5
        ax.scatter(
            [x], [y], s=size, c=[i_colors[idx]], zorder=3, edgecolors="black", lw=0.5
        )
        ax.text(
            x + 0.3,
            y,
            meta_display(interv_meta_df, mc, interv_name_map, "I-Meta"),
            fontsize=6,
            ha="left",
            va="center",
            zorder=4,
        )

    ax.text(
        0,
        max(n_r, n_i) * 2.5 - 1,
        "RISK META-CLUSTERS",
        fontsize=11,
        fontweight="bold",
        ha="center",
        color="darkred",
    )
    ax.text(
        5,
        max(n_r, n_i) * 2.5 - 1,
        "INTERVENTION META-CLUSTERS",
        fontsize=11,
        fontweight="bold",
        ha="center",
        color="darkblue",
    )
    ax.set_title(
        f"Meta-cluster R→I Connectivity (top-{top_n} edges by consim1 paths)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


print("\nGenerating plots ...")

plot_sim_heatmap(
    risk_sim,
    risk_cids,
    risk_meta_k10,
    risk_name_map,
    f"Risk Cluster Inter-Centroid Similarity ({len(risk_cids)}×{len(risk_cids)}, VPN_consim1, sorted by meta-cluster)",
    os.path.join(OUT_DIR, "risk_sim_heatmap.png"),
    k=10,
)

plot_sim_heatmap(
    interv_sim,
    interv_cids,
    interv_meta_k10,
    interv_name_map,
    f"Intervention Cluster Inter-Centroid Similarity ({len(interv_cids)}×{len(interv_cids)}, VPN_consim1, sorted by meta-cluster)",
    os.path.join(OUT_DIR, "interv_sim_heatmap.png"),
    k=10,
)

plot_dendrogram(
    risk_Z,
    risk_cids,
    risk_name_map,
    "Risk Cluster Dendrogram (average linkage, cosine distance, VPN_consim1 centroids)",
    os.path.join(OUT_DIR, "risk_dendrogram.png"),
)

plot_dendrogram(
    interv_Z,
    interv_cids,
    interv_name_map,
    "Intervention Cluster Dendrogram (average linkage, cosine distance, VPN_consim1 centroids)",
    os.path.join(OUT_DIR, "interv_dendrogram.png"),
)

plot_meta_connectivity(
    meta_conn,
    risk_meta_df,
    interv_meta_df,
    risk_name_map,
    interv_name_map,
    os.path.join(OUT_DIR, "meta_connectivity_network.png"),
    k=10,
    top_n=30,
)

# ─── STEP 12: Intra-centroid coherence stats (meta-cluster coherence) ──────────
print("\nComputing meta-cluster coherence stats ...")


def meta_coherence(sim_matrix, cids, meta_labels, name_map, cluster_type):
    rows = []
    unique_metas = sorted(set(meta_labels))
    for mc in unique_metas:
        mc_indices = [i for i, m in enumerate(meta_labels) if m == mc]
        mc_cids = [cids[i] for i in mc_indices]
        if len(mc_indices) < 2:
            # Single-cluster meta — inter-sim undefined
            rows.append(
                {
                    "cluster_type": cluster_type,
                    "meta_id": mc,
                    "n_clusters": 1,
                    "mean_intra_sim": float(sim_matrix[mc_indices[0], mc_indices[0]]),
                    "min_intra_sim": float(sim_matrix[mc_indices[0], mc_indices[0]]),
                    "member_cids": str(mc_cids),
                }
            )
        else:
            sub = sim_matrix[np.ix_(mc_indices, mc_indices)]
            upper = sub[np.triu_indices(len(mc_indices), k=1)]
            rows.append(
                {
                    "cluster_type": cluster_type,
                    "meta_id": mc,
                    "n_clusters": len(mc_indices),
                    "mean_intra_sim": float(upper.mean()),
                    "min_intra_sim": float(upper.min()),
                    "member_cids": str(mc_cids),
                }
            )
    return rows


coherence_rows = meta_coherence(
    risk_sim, risk_cids, risk_meta_k10, risk_name_map, "risk"
) + meta_coherence(
    interv_sim, interv_cids, interv_meta_k10, interv_name_map, "intervention"
)
coherence_df = pd.DataFrame(coherence_rows)
coherence_path = os.path.join(OUT_DIR, "meta_cluster_coherence.csv")
coherence_df.to_csv(coherence_path, index=False)
print(f"  Saved {coherence_path}")

# ─── STEP 13: Summary statistics ──────────────────────────────────────────────
print("\n=== B1 Summary ===")
print(f"VPN_consim1: {len(vpn_consim1)} nodes from {n_consim1_paths} qualifying paths")
print(f"Risk clusters: {len(risk_cids)}, meta-clusters k=10: {len(set(risk_meta_k10))}")
print(
    f"Intervention clusters: {len(interv_cids)}, meta-clusters k=10: {len(set(interv_meta_k10))}"
)
print(f"Meta R→I pairs (consim1 paths): {len(meta_conn)}")
print("Top-5 meta R→I pairs by consim1 paths:")
for _, row in meta_conn.head(5).iterrows():
    print(
        f"  R-Meta{int(row['risk_meta'])} ({row['risk_meta_name'][:40]}) → "
        f"I-Meta{int(row['interv_meta'])} ({row['interv_meta_name'][:40]}): "
        f"{int(row['n_paths_c1'])} paths"
    )

print("\nInter-centroid similarity stats (VPN_consim1 centroids):")
print(f"  Risk: mean={ri.mean():.3f}, min={ri.min():.3f}, max={ri.max():.3f}")
print(f"  Intervention: mean={ii.mean():.3f}, min={ii.min():.3f}, max={ii.max():.3f}")

print("\nDone — all outputs use VPN_consim1-filtered centroids.")
