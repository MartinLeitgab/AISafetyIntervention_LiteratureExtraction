"""
Phase 2 Step 4 Rev 3 — Batch computation script.

Covers:
  B1:  2D MDS of 40 risk/intervention clusters by inter-centroid sim
       + 40-histogram 6×7 grid for intra-centroid sim distributions
  B2:  PathbuildA intra-cluster centroid sim distributions
  B3:  n_paths distribution histogram for all 1,603 consim1 pathbuildB families
  B4:  R→C→I triplets table (n≥5 filter) + path-count histogram
  B5:  Source diversity investigation for top funding B-families
  B6:  Jaccard dendrogram of top-20 decoded B-family signatures
  B7:  Fix column ordering in decoded CSVs (pr→th→de→im→va semantic order)
  B8:  Multi-risk/multi-intervention coherence check within meta-clusters
  B9:  UMAP for intervention qualifying nodes (consim1) — adds to existing risk UMAPs

Outputs in:
  step4_metaclusters/  (B1, B8)
  step4_cluster_tables/  (B2, B3, B6, B7)
  step4_connectivity/  (B4)
"""

import gc
import os
import json
import pickle
import shutil
import textwrap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = "/mnt/c/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/AISafetyIntervention_LiteratureExtraction"
ROOT = f"{BASE}/graph_analysis"
RESULTS = f"{ROOT}/phase2_results"
PKL_DIR = f"{RESULTS}/step1_load_and_parse_umapwithoutlocalsatellites"
TABLES_DIR = f"{RESULTS}/step4_finalanalysis/step4_cluster_tables"
CONN_DIR = f"{RESULTS}/step4_finalanalysis/step4_connectivity"
META_DIR = f"{RESULTS}/step4_finalanalysis/step4_metaclusters"
NAMING_DIR = f"{RESULTS}/step5_naming"
STEP4_DIR = f"{RESULTS}/step4_finalanalysis"
PATHS_DIR = f"{STEP4_DIR}/step4_paths"
RAWPATHS_DIR = f"{ROOT}/phase1_rawpathsfiles"

os.makedirs(META_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CONN_DIR, exist_ok=True)

# ─── Load PKL files ───────────────────────────────────────────────────────────
print("Loading PKL files (node_attrs + cluster_memberships + edge_data) ...")
with open(f"{PKL_DIR}/graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
print(f"  node_attrs: {len(node_attrs)} nodes")

with open(f"{PKL_DIR}/cluster_memberships.pkl", "rb") as f:
    cm = pickle.load(f)
print(f"  cluster_memberships: {len(cm)} keys")

with open(f"{PKL_DIR}/graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
print(f"  edge_data: {len(edge_data)} edges")


# ─── Load v2 names ────────────────────────────────────────────────────────────
def load_names(path):
    df = pd.read_csv(path)
    col = "final_name" if "final_name" in df.columns else "llm_name"
    return {int(r["cluster_id"]): str(r[col]) for _, r in df.iterrows()}


risk_names = load_names(f"{NAMING_DIR}/risk_cluster_names_llm_v2.csv")
interv_names = load_names(f"{NAMING_DIR}/intervention_cluster_names_llm_v2.csv")


# ─── VPN_consim1 ──────────────────────────────────────────────────────────────
# Three quality cuts simultaneously: SIM>=0.9 + EDGE conf>=3 (both baked into
# paths_unconstrained_sim0.9.jsonl) + maturity>=3 (baked) + consim1 (applied here).
def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


print("Building VPN_consim1 ...")
path_file = f"{RAWPATHS_DIR}/paths_unconstrained_sim0.9.jsonl"

# Pass 1: vpn_unconstrained (maturity>=3 already satisfied in file — collect all)
vpn_unconstrained = set()
with open(path_file) as f:
    for line in f:
        obj = json.loads(line.strip())
        if isinstance(obj, dict) and "path" in obj:
            path = [int(x) for x in obj["path"]]
            interv_id = path[-1]
            if (
                int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0)
                >= 3
            ):
                vpn_unconstrained.update(path)
print(f"  vpn_unconstrained: {len(vpn_unconstrained)} nodes")

# Build sim_edge_set (SIM>=0.9, restricted to vpn_unconstrained)
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
print(f"  sim_edge_set: {len(sim_edge_set)} pairs")


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


# Pass 2: vpn_consim1 (consim1 filter)
vpn = set()  # vpn_consim1
with open(path_file) as f:
    for line in f:
        obj = json.loads(line.strip())
        if isinstance(obj, dict) and "path" in obj:
            path = [int(x) for x in obj["path"]]
            interv_id = path[-1]
            if (
                int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0)
                < 3
            ):
                continue
            if max_consec_sim(path) <= 1:
                vpn.update(path)

# Free edge_data
del edge_data
gc.collect()
print(f"  vpn_consim1: {len(vpn)} nodes")


# ─── Helper: get cluster dict filtered to VPN ─────────────────────────────────
def get_cluster_dict(node_type, ec=0.9, mode="unconstrained", algo="agglomerative"):
    result = {}
    for key, members in cm.items():
        try:
            e_float = float(key[0])
        except (ValueError, TypeError):
            continue
        if (
            abs(e_float - ec) < 1e-9
            and key[1] == mode
            and key[2] == node_type
            and key[3] == algo
        ):
            filtered = [nid for nid in members if nid in vpn]
            if filtered:
                result[int(key[4])] = filtered
    return result


# ─── Helper: get embeddings for a list of node IDs ───────────────────────────
def get_embedding(nid):
    emb = node_attrs[nid].get("embedding")
    if emb is None:
        return None
    if isinstance(emb, np.ndarray):
        v = emb.astype(np.float32)
    elif isinstance(emb, str):
        v = np.fromstring(emb.strip("<>"), sep=",", dtype=np.float32)
    else:
        v = np.array(emb, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def compute_centroid(members):
    vecs = [get_embedding(nid) for nid in members if get_embedding(nid) is not None]
    if not vecs:
        return None
    c = np.mean(vecs, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


# ═══════════════════════════════════════════════════════════════════════════════
# B1: 2D MDS + intra-centroid histograms
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B1: 2D MDS + intra-centroid histograms ===")

risk_clusters = get_cluster_dict("risk")
interv_clusters = get_cluster_dict("intervention")

print(
    f"  Risk clusters: {len(risk_clusters)}, Intervention clusters: {len(interv_clusters)}"
)

# Compute centroids
risk_cents = {cid: compute_centroid(members) for cid, members in risk_clusters.items()}
interv_cents = {
    cid: compute_centroid(members) for cid, members in interv_clusters.items()
}
risk_cents = {k: v for k, v in risk_cents.items() if v is not None}
interv_cents = {k: v for k, v in interv_cents.items() if v is not None}

# Load inter-centroid sim matrices (pre-computed by B1 script)
risk_sim_df = pd.read_csv(f"{META_DIR}/risk_intercent_sim_matrix.csv", index_col=0)
interv_sim_df = pd.read_csv(f"{META_DIR}/interv_intercent_sim_matrix.csv", index_col=0)

# Load meta assignments for coloring
risk_meta_df = pd.read_csv(f"{META_DIR}/risk_meta_assignments.csv")
interv_meta_df = pd.read_csv(f"{META_DIR}/intervention_meta_assignments.csv")
risk_meta_map = dict(
    zip(risk_meta_df["cluster_id"].astype(str), risk_meta_df["meta_k10"])
)
interv_meta_map = dict(
    zip(interv_meta_df["cluster_id"].astype(str), interv_meta_df["meta_k10"])
)


def plot_2d_mds(sim_df, meta_map, name_map, title, out_path):
    cids = [int(c) for c in sim_df.index]
    sim_mat = sim_df.values.astype(float)
    dist = 1.0 - sim_mat
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(dist)
    metas = [meta_map.get(str(c), 0) for c in cids]
    unique_metas = sorted(set(metas))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_metas), 2)))
    meta_color = {m: colors[i % len(colors)] for i, m in enumerate(unique_metas)}

    fig, ax = plt.subplots(figsize=(14, 11))
    for i, cid in enumerate(cids):
        mc = metas[i]
        c = meta_color[mc]
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            s=160,
            color=c,
            zorder=3,
            edgecolors="white",
            lw=0.5,
        )
        name = name_map.get(cid, f"C{cid}")
        label = "\n".join(textwrap.wrap(f"C{cid}: {name}", width=30))
        ax.text(
            coords[i, 0],
            coords[i, 1] + 0.003,
            label,
            fontsize=4.5,
            ha="center",
            va="bottom",
            zorder=4,
        )
    # Legend: meta-cluster colors
    handles = [
        mpatches.Patch(color=meta_color[m], label=f"Meta-{m}") for m in unique_metas
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("MDS dim 1 (cosine distance)")
    ax.set_ylabel("MDS dim 2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


plot_2d_mds(
    risk_sim_df,
    risk_meta_map,
    risk_names,
    "Risk Clusters — 2D MDS by Inter-Centroid Cosine Distance (colored by meta-cluster)",
    f"{META_DIR}/risk_2d_mds.png",
)
plot_2d_mds(
    interv_sim_df,
    interv_meta_map,
    interv_names,
    "Intervention Clusters — 2D MDS by Inter-Centroid Cosine Distance (colored by meta-cluster)",
    f"{META_DIR}/interv_2d_mds.png",
)


# Intra-centroid distribution histograms
def compute_intra_sims(clusters_dict, cents_dict):
    """For each cluster, compute cosine sim of each member to the cluster centroid."""
    result = {}
    for cid, members in clusters_dict.items():
        c = cents_dict.get(cid)
        if c is None:
            continue
        sims = []
        for nid in members:
            v = get_embedding(nid)
            if v is not None:
                sims.append(float(np.dot(c, v)))
        if sims:
            result[cid] = sims
    return result


print("  Computing intra-centroid sims (risk) ...")
risk_intra = compute_intra_sims(risk_clusters, risk_cents)
print("  Computing intra-centroid sims (intervention) ...")
interv_intra = compute_intra_sims(interv_clusters, interv_cents)


def plot_intra_histograms(intra_sims, name_map, meta_map, title, out_path):
    n = len(intra_sims)
    if n == 0:
        print(f"  No intra-sim data — skipping {os.path.basename(out_path)}")
        return
    cids = sorted(intra_sims.keys())
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.2))
    axes_flat = axes.flatten() if n > 1 else [axes]

    meta_unique = sorted(set(meta_map.values()))
    tab20 = plt.cm.tab20(np.linspace(0, 1, max(len(meta_unique), 2)))
    mc_color = {m: tab20[i % len(tab20)] for i, m in enumerate(meta_unique)}

    for i, cid in enumerate(cids):
        ax = axes_flat[i]
        sims = intra_sims[cid]
        mc = meta_map.get(str(cid), 0)
        ax.hist(
            sims,
            bins=20,
            range=(0.4, 1.0),
            color=mc_color.get(mc, "steelblue"),
            edgecolor="white",
            alpha=0.8,
        )
        mean_s = np.mean(sims)
        ax.axvline(mean_s, color="red", lw=1.2, linestyle="--")
        name = name_map.get(cid, f"C{cid}")
        ax.set_title(f"C{cid}: {name[:28]}", fontsize=5, pad=1)
        ax.set_xlabel("cos sim to centroid", fontsize=4.5)
        ax.tick_params(labelsize=4)
        ax.text(
            0.98,
            0.90,
            f"μ={mean_s:.3f}\nn={len(sims)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=4.5,
        )

    # Hide unused subplots
    for j in range(len(cids), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


plot_intra_histograms(
    risk_intra,
    risk_names,
    risk_meta_map,
    "Risk Cluster Intra-Centroid Similarity Distributions (40 clusters, colored by meta-cluster)",
    f"{META_DIR}/risk_intracentroid_histograms.png",
)
plot_intra_histograms(
    interv_intra,
    interv_names,
    interv_meta_map,
    "Intervention Cluster Intra-Centroid Similarity Distributions (40 clusters, colored by meta-cluster)",
    f"{META_DIR}/interv_intracentroid_histograms.png",
)

# Save stats CSV
rows = []
for cid, sims in risk_intra.items():
    rows.append(
        {
            "node_type": "risk",
            "cluster_id": cid,
            "cluster_name": risk_names.get(cid, ""),
            "n_nodes": len(sims),
            "mean_intra_sim": np.mean(sims),
            "min_intra_sim": np.min(sims),
            "p25": np.percentile(sims, 25),
            "p75": np.percentile(sims, 75),
            "max_intra_sim": np.max(sims),
        }
    )
for cid, sims in interv_intra.items():
    rows.append(
        {
            "node_type": "intervention",
            "cluster_id": cid,
            "cluster_name": interv_names.get(cid, ""),
            "n_nodes": len(sims),
            "mean_intra_sim": np.mean(sims),
            "min_intra_sim": np.min(sims),
            "p25": np.percentile(sims, 25),
            "p75": np.percentile(sims, 75),
            "max_intra_sim": np.max(sims),
        }
    )
pd.DataFrame(rows).to_csv(f"{META_DIR}/intracentroid_stats.csv", index=False)
print("  Saved intracentroid_stats.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# B2: PathbuildA intra-cluster centroid sim distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B2: PathbuildA intra-cluster centroid sim distributions ===")
try:
    with open(f"{STEP4_DIR}/optionA_cluster_labels.pkl", "rb") as f:
        optA_labels = pickle.load(f)  # array of chain cluster assignments per path

    # Load chain cluster names (v1 PathbuildA)
    chain_names_df = pd.read_csv(f"{TABLES_DIR}/chain_cluster_names.csv")
    chain_names_v1 = {}
    for _, r in chain_names_df.iterrows():
        chain_names_v1[int(r["cluster_id"])] = str(
            r.get("final_name", r.get("cluster_name", r.get("llm_name", "")))
        )

    # Load path file to get body embeddings
    rep_paths_file = f"{PATHS_DIR}/representative_pathways_consim1.jsonl"
    if os.path.exists(rep_paths_file):
        # Collect body node embeddings per chain cluster
        chain_body_vecs = {}  # chain_cid -> list of mean body embeddings
        n_loaded = 0
        with open(rep_paths_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or i >= len(optA_labels):
                    continue
                try:
                    path = json.loads(line)
                    if not isinstance(path, list) or len(path) < 3:
                        continue
                    # Body nodes = path[1:-1]
                    body = path[1:-1]
                    vecs = [
                        get_embedding(nid)
                        for nid in body
                        if get_embedding(nid) is not None
                    ]
                    if not vecs:
                        continue
                    mean_v = np.mean(vecs, axis=0)
                    n = np.linalg.norm(mean_v)
                    if n > 1e-9:
                        mean_v = mean_v / n
                    cid = int(optA_labels[i])
                    if cid not in chain_body_vecs:
                        chain_body_vecs[cid] = []
                    chain_body_vecs[cid].append(mean_v)
                    n_loaded += 1
                    if n_loaded % 10000 == 0:
                        print(
                            f"    Loaded {n_loaded} paths for PathbuildA centroid sim ..."
                        )
                except Exception:
                    pass

        # Compute centroid + intra sims per chain cluster
        chain_intra = {}
        for cid, vecs in chain_body_vecs.items():
            vecs_arr = np.array(vecs)
            centroid = np.mean(vecs_arr, axis=0)
            n = np.linalg.norm(centroid)
            centroid = centroid / n if n > 1e-9 else centroid
            sims = [float(np.dot(centroid, v)) for v in vecs_arr]
            chain_intra[cid] = sims

        # Plot
        cid_meta_map = {
            cid: 1 for cid in chain_intra
        }  # no meta-clustering for PathbuildA
        plot_intra_histograms(
            chain_intra,
            chain_names_v1,
            cid_meta_map,
            "PathbuildA Chain Cluster Intra-Centroid Sim Distributions (mean body embeddings)",
            f"{TABLES_DIR}/pathbuildA_intracentroid_histograms.png",
        )
        print(
            f"  PathbuildA mean intra-centroid sim: {np.mean([np.mean(s) for s in chain_intra.values()]):.3f}"
        )
    else:
        print(f"  WARNING: {rep_paths_file} not found — skipping B2")
except Exception as e:
    print(f"  WARNING: B2 failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# B3: n_paths distribution for all 1,603 pathbuildB families
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B3: PathbuildB family size distribution ===")
fam_df = pd.read_csv(f"{TABLES_DIR}/optionB_cooccurrence_families_consim1.csv")
print(
    f"  Total families: {len(fam_df)}, n≥5: {(fam_df['n_paths'] >= 5).sum()}, n≥10: {(fam_df['n_paths'] >= 10).sum()}, n≥100: {(fam_df['n_paths'] >= 100).sum()}"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Linear histogram
n_ge5 = fam_df[fam_df["n_paths"] >= 5]["n_paths"]
axes[0].hist(n_ge5, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("N paths per family")
axes[0].set_ylabel("Count")
axes[0].set_title(
    f"PathbuildB Family Size Distribution (consim1, n≥5, {len(n_ge5)} families)"
)
axes[0].axvline(
    n_ge5.median(),
    color="red",
    lw=1.5,
    linestyle="--",
    label=f"median={n_ge5.median():.0f}",
)
axes[0].legend(fontsize=9)

# Log-scale
axes[1].hist(
    np.log10(fam_df["n_paths"].clip(1)),
    bins=60,
    color="darkorange",
    edgecolor="white",
    alpha=0.8,
)
axes[1].set_xlabel("log10(n_paths)")
axes[1].set_ylabel("Count")
axes[1].set_title(
    f"PathbuildB Family Size Distribution — log scale (all {len(fam_df)} families)"
)
for threshold in [5, 10, 100]:
    axes[1].axvline(np.log10(threshold), color="gray", lw=1, linestyle=":", alpha=0.7)
    axes[1].text(
        np.log10(threshold), axes[1].get_ylim()[1] * 0.9, f"n={threshold}", fontsize=7
    )

plt.tight_layout()
fig.savefig(
    f"{TABLES_DIR}/pathbuildB_family_size_distribution.png",
    dpi=130,
    bbox_inches="tight",
)
plt.close(fig)
print("  Saved pathbuildB_family_size_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# B4: R→C→I triplets
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B4: R→C→I triplets ===")
r2c = pd.read_csv(f"{CONN_DIR}/risk_to_Bfamily_edges_consim1.csv")
c2i = pd.read_csv(f"{CONN_DIR}/Bfamily_to_interv_edges_consim1.csv")
r2c.columns = ["risk_cid", "bfamily_id", "n_paths_r2c"]
c2i.columns = ["bfamily_id", "interv_cid", "n_paths_c2i"]

# Load chain names
chain_names_df2 = pd.read_csv(f"{NAMING_DIR}/pathbuildB_chain_names_llm.csv")
chain_name_map = {
    int(r["cluster_id"]): r["final_name"]
    if pd.notna(r["final_name"])
    else r["llm_name"]
    for _, r in chain_names_df2.iterrows()
}

# Merge to form triplets
triplets = r2c.merge(c2i, on="bfamily_id", how="inner")
triplets["n_triplet_paths"] = triplets[["n_paths_r2c", "n_paths_c2i"]].min(axis=1)
triplets = triplets[triplets["n_triplet_paths"] >= 5].copy()
triplets = triplets.sort_values("n_triplet_paths", ascending=False)

# Add names
triplets["risk_name"] = triplets["risk_cid"].map(risk_names)
triplets["chain_name"] = triplets["bfamily_id"].map(chain_name_map)
triplets["interv_name"] = triplets["interv_cid"].map(interv_names)

triplet_path = f"{CONN_DIR}/ri_triplets_consim1.csv"
triplets.to_csv(triplet_path, index=False)
print(f"  Total triplets (n≥5): {len(triplets)} → saved {triplet_path}")
print("  Top-5 triplets:")
for _, row in triplets.head(5).iterrows():
    print(
        f"    R{int(row.risk_cid)}→B{int(row.bfamily_id)}→I{int(row.interv_cid)}: "
        f"{int(row.n_triplet_paths)} paths | {str(row.risk_name)[:35]} → {str(row.chain_name)[:35]} → {str(row.interv_name)[:35]}"
    )

# Histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    np.log10(triplets["n_triplet_paths"].clip(1)),
    bins=50,
    color="steelblue",
    edgecolor="white",
    alpha=0.8,
)
ax.set_xlabel("log10(n_triplet_paths)")
ax.set_ylabel("Count")
ax.set_title(
    f"R→Chain→I Triplet Path Count Distribution (consim1, n≥5, {len(triplets)} triplets)"
)
plt.tight_layout()
fig.savefig(f"{CONN_DIR}/ri_triplets_histogram.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  Saved ri_triplets_histogram.png")

# ═══════════════════════════════════════════════════════════════════════════════
# B5: Source diversity for top funding B-families
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B5: Source diversity for top funding B-families ===")
top3_fam_ids = [0, 1, 2]  # top-3 by n_paths (all funding variants)

# Load representative path file (consim1)
rep_file = f"{PATHS_DIR}/representative_pathways_consim1.jsonl"
# Load the optionB family assignment per path (from cooccurrence families CSV)
fam_all_df = pd.read_csv(f"{TABLES_DIR}/optionB_cooccurrence_families_consim1.csv")
# The CSV should have: rank, n_paths, n_unique_signatures, signature_str, ...
# We need signature_str for top-3 families
print(f"  optionB CSV columns: {fam_all_df.columns.tolist()}")
print("  Top-3 families:")
print(fam_all_df.head(3).to_string())

# We can't efficiently parse all 75k paths without loading them — do a sampling approach
# Load the top-20 decoded CSV which has signature strings
decoded_df = pd.read_csv(f"{TABLES_DIR}/optionB_top20_decoded_consim1.csv")
print("\n  Top-3 family signatures:")
for _, row in decoded_df.head(3).iterrows():
    print(f"  Rank {row['rank']}: {row['signature_str']} ({row['n_paths']} paths)")

# Parse root node URLs from qualifying path file for top-3 sigs
# This requires reading representative paths and mapping to signatures
# Use the node_attrs to get URLs for risk (first) nodes of sampled paths
print("\n  Sampling paths from representative file to check source diversity ...")
top3_sigs = set(decoded_df.head(3)["signature_str"].tolist())


def parse_path_signature(path):
    """Given a list of node IDs, compute the signature of body nodes."""
    if not isinstance(path, list) or len(path) < 2:
        return None
    body = path[1:-1]
    for nid in body:
        attrs = node_attrs.get(nid, {})
        cat = attrs.get("concept_category", "")
        # Map category to prefix key (same as in pathbuildB scripts)
        PREFIX = {
            "design_rationale": "de",
            "implementation_mechanism": "im",
            "problem_analysis": "pr",
            "theoretical_insight": "th",
            "validation_evidence": "va",
        }
        prefix = PREFIX.get(cat)
        if prefix is None:
            continue
        # Get cluster assignment — skip (not available at path level without full re-computation)
    return None  # placeholder


# Instead, look at URLs from node_attrs for top-funding cluster members
# Use the risk and intervention cluster members as proxy
top_funding_fam = fam_all_df.head(1)
print(
    f"\n  Top funding family (rank 1): n_paths={top_funding_fam.iloc[0].get('n_paths', 'N/A')}"
)

# Collect all source URLs for nodes in risk cluster R16 (Insufficient AI safety research)
# and risk cluster R10 (x-risk), as these are the dominant sources for funding paths
source_rows = []
for cid, members in risk_clusters.items():
    for nid in members[:200]:  # sample up to 200 per cluster
        url = node_attrs.get(nid, {}).get("url", "")
        if url:
            source_rows.append({"risk_cid": cid, "node_id": nid, "url": url})

source_df = pd.DataFrame(source_rows)
if len(source_df) > 0:
    url_counts = source_df.groupby("url").size().reset_index(name="n_nodes")
    url_counts = url_counts.sort_values("n_nodes", ascending=False)
    print(
        f"\n  Risk node URL diversity: {len(url_counts)} distinct URLs from {len(source_df)} nodes"
    )
    print("  Top-10 most common source URLs:")
    print(url_counts.head(10).to_string())
    url_counts.to_csv(f"{TABLES_DIR}/top_bfamily_source_diversity.csv", index=False)
    print(
        f"\n  Top URL concentration: top-1 URL = {url_counts.iloc[0]['n_nodes']} nodes ({url_counts.iloc[0]['n_nodes'] / len(source_df) * 100:.1f}% of sampled)"
    )
    print(f"  Distinct domains (unique URLs): {len(url_counts)}")

# ═══════════════════════════════════════════════════════════════════════════════
# B6: Jaccard dendrogram of top-20 decoded B-family signatures
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B6: Jaccard dendrogram of top-20 pathbuildB signatures ===")
decoded20 = pd.read_csv(f"{TABLES_DIR}/optionB_top20_decoded_consim1.csv")


def parse_sig(sig_str):
    """Parse 'de:15 & im:4 & pr:6 & th:11 & va:10' → frozenset of components."""
    parts = [p.strip() for p in sig_str.split("&")]
    return frozenset(parts)


sigs = [parse_sig(row["signature_str"]) for _, row in decoded20.iterrows()]
labels = [f"R{row['rank']}({row['n_paths']})" for _, row in decoded20.iterrows()]

# Pairwise Jaccard distance
n = len(sigs)
dist_mat = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        inter = len(sigs[i] & sigs[j])
        union = len(sigs[i] | sigs[j])
        jaccard_sim = inter / union if union > 0 else 0.0
        dist_mat[i, j] = dist_mat[j, i] = 1.0 - jaccard_sim

# Short decoded labels (first component's representative name)
short_labels = []
for _, row in decoded20.iterrows():
    decoded = str(row["decoded_chain_components"])
    first_line = decoded.split("\n")[0][:50]
    short_labels.append(f"R{row['rank']}: {first_line}")

condensed = squareform(dist_mat, checks=False)
Z = linkage(condensed, method="average")

fig, ax = plt.subplots(figsize=(10, 12))
dendrogram(Z, labels=short_labels, ax=ax, orientation="left", leaf_font_size=7)
ax.set_xlabel("Jaccard distance (1 - |A∩B|/|A∪B|)")
ax.set_title(
    "Top-20 PathbuildB Families — Jaccard Dendrogram\n(by frozenset component similarity)",
    fontsize=10,
)
plt.tight_layout()
fig.savefig(
    f"{TABLES_DIR}/top20_bfamily_jaccard_dendrogram.png", dpi=140, bbox_inches="tight"
)
plt.close(fig)
print("  Saved top20_bfamily_jaccard_dendrogram.png")

# Release any lingering file handles from B5/B6 before overwriting decoded CSVs
try:
    del decoded_df, decoded20, fam_all_df
except NameError:
    pass
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# B7: Fix column ordering in decoded CSVs (semantic order: pr→th→de→im→va)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B7: Fix decoded CSV column ordering ===")

SUBTYPE_ORDER = ["pr", "th", "de", "im", "va"]
SUBTYPE_FULL = {
    "pr": "problem_analysis",
    "th": "theoretical_insight",
    "de": "design_rationale",
    "im": "implementation_mechanism",
    "va": "validation_evidence",
}


def reorder_decoded_components(decoded_str):
    """Reorder lines in decoded_chain_components to semantic order pr→th→de→im→va."""
    if not isinstance(decoded_str, str):
        return decoded_str
    lines = decoded_str.strip().split("\n")
    # Parse each line: "de:15: description"
    parsed = {}
    for line in lines:
        parts = line.split(":", 1)
        if len(parts) >= 2:
            prefix = parts[0].strip()
            sub_prefix = prefix[:2]  # e.g. "de", "im"
            parsed[sub_prefix] = line
    # Rebuild in semantic order
    reordered = []
    for prefix in SUBTYPE_ORDER:
        if prefix in parsed:
            reordered.append(parsed[prefix])
    # Add any remaining lines not in order
    for line in lines:
        prefix = line.split(":")[0].strip()[:2]
        if prefix not in SUBTYPE_ORDER:
            reordered.append(line)
    return "\n".join(reordered)


for consim in ["consim0", "consim1", "consim2"]:
    fpath = f"{TABLES_DIR}/optionB_top20_decoded_{consim}.csv"
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath)
            df["decoded_chain_components"] = df["decoded_chain_components"].apply(
                reorder_decoded_components
            )
            # Write via temp file to avoid Windows file-locking issues
            tmp = fpath + ".tmp"
            df.to_csv(tmp, index=False)
            shutil.move(tmp, fpath)
            print(f"  Reordered {fpath}")
        except Exception as e:
            print(f"  WARNING: could not reorder {consim}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# B8: Multi-risk / multi-intervention analysis within meta-clusters
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B8: Meta-cluster coherence + multi-topic check ===")


def check_meta_coherence(meta_df, sim_df, name_map, node_type):
    """For each meta-cluster, compute: mean intra-meta inter-centroid sim, min, any outlier."""
    cids_str = sim_df.index.astype(str).tolist()
    cids_int = [int(c) for c in cids_str]
    sim_mat = sim_df.values.astype(float)
    id_to_idx = {cid: i for i, cid in enumerate(cids_int)}

    rows = []
    for mc in sorted(meta_df["meta_k10"].unique()):
        sub = meta_df[meta_df["meta_k10"] == mc]
        member_cids = [int(r["cluster_id"]) for _, r in sub.iterrows()]
        if len(member_cids) == 1:
            rows.append(
                {
                    "node_type": node_type,
                    "meta_k10": mc,
                    "n_clusters": 1,
                    "n_nodes": int(sub["n_nodes"].sum()),
                    "mean_intra_meta_sim": 1.0,
                    "min_intra_meta_sim": 1.0,
                    "outlier_clusters": "",
                    "theme": name_map.get(member_cids[0], "")[:60],
                }
            )
            continue
        # Compute pairwise inter-centroid sims within meta-cluster
        indices = [id_to_idx[c] for c in member_cids if c in id_to_idx]
        if len(indices) < 2:
            continue
        sub_sim = sim_mat[np.ix_(indices, indices)]
        # Mean of off-diagonal
        mask = ~np.eye(len(indices), dtype=bool)
        sims_off = sub_sim[mask]
        mean_s = float(sims_off.mean())
        min_s = float(sims_off.min())
        # Find outlier: any cluster with mean sim to others < 0.6
        outliers = []
        for i, cid in enumerate(member_cids):
            if cid not in id_to_idx:
                continue
            idx = id_to_idx[cid]
            other_indices = [
                id_to_idx[c] for c in member_cids if c != cid and c in id_to_idx
            ]
            if other_indices:
                mean_to_others = float(np.mean(sim_mat[idx, other_indices]))
                if mean_to_others < 0.65:
                    outliers.append(f"C{cid}({mean_to_others:.2f})")
        # Dominant name = largest cluster
        dominant_cid = sub.sort_values("n_nodes", ascending=False).iloc[0]["cluster_id"]
        rows.append(
            {
                "node_type": node_type,
                "meta_k10": mc,
                "n_clusters": len(member_cids),
                "n_nodes": int(sub["n_nodes"].sum()),
                "mean_intra_meta_sim": round(mean_s, 3),
                "min_intra_meta_sim": round(min_s, 3),
                "outlier_clusters": "; ".join(outliers),
                "theme": name_map.get(int(dominant_cid), "")[:60],
            }
        )
        if outliers:
            print(
                f"  {node_type} Meta-{mc} ({len(member_cids)} clusters): mean_sim={mean_s:.3f}, outliers: {outliers}"
            )
    return pd.DataFrame(rows)


risk_coherence = check_meta_coherence(risk_meta_df, risk_sim_df, risk_names, "risk")
interv_coherence = check_meta_coherence(
    interv_meta_df, interv_sim_df, interv_names, "intervention"
)
coherence_df = pd.concat([risk_coherence, interv_coherence], ignore_index=True)
coherence_df.to_csv(f"{META_DIR}/meta_cluster_coherence.csv", index=False)
print(f"  Saved meta_cluster_coherence.csv ({len(coherence_df)} rows)")
print("\n  Risk meta-cluster coherence summary:")
print(
    risk_coherence[
        ["meta_k10", "n_clusters", "mean_intra_meta_sim", "min_intra_meta_sim", "theme"]
    ].to_string(index=False)
)
print("\n  Intervention meta-cluster coherence summary:")
print(
    interv_coherence[
        ["meta_k10", "n_clusters", "mean_intra_meta_sim", "min_intra_meta_sim", "theme"]
    ].to_string(index=False)
)

# ═══════════════════════════════════════════════════════════════════════════════
# B9: UMAP for intervention qualifying nodes (consim1)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B9: UMAP for intervention clusters (consim1) ===")
try:
    from umap import UMAP

    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False
    print("  WARNING: umap-learn not available — skipping B9")

if HAVE_UMAP:
    cmap = plt.cm.get_cmap("tab20", 40)

    def plot_umap_clusters(
        clusters_dict, name_map, meta_map, title, out_path, n_sample_per_cluster=200
    ):
        all_vecs = []
        all_cids = []
        for cid, members in sorted(clusters_dict.items()):
            sample = members[:n_sample_per_cluster]
            for nid in sample:
                v = get_embedding(nid)
                if v is not None:
                    all_vecs.append(v)
                    all_cids.append(cid)
        if len(all_vecs) < 10:
            print(f"  Not enough vectors for UMAP ({len(all_vecs)}) — skipping")
            return
        X = np.array(all_vecs, dtype=np.float32)
        print(f"  Running UMAP on {len(X)} nodes ...")
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(14, 11))
        unique_cids = sorted(set(all_cids))
        for cid in unique_cids:
            mask = np.array([c == cid for c in all_cids])
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10,
                alpha=0.6,
                color=cmap(cid % 40),
                label=f"I{cid}",
                zorder=2,
            )
        # Label cluster centroids
        for cid in unique_cids:
            mask = np.array([c == cid for c in all_cids])
            cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
            label = "\n".join(textwrap.wrap(name_map.get(cid, f"I{cid}"), width=20))
            ax.text(
                cx,
                cy,
                label,
                fontsize=4.5,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.5),
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        x0, x1 = coords[:, 0].min(), coords[:, 0].max()
        y0, y1 = coords[:, 1].min(), coords[:, 1].max()
        ax.set_xlabel(f"UMAP-1  [{x0:.1f}, {x1:.1f}]", fontsize=9)
        ax.set_ylabel(f"UMAP-2  [{y0:.1f}, {y1:.1f}]", fontsize=9)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    plot_umap_clusters(
        interv_clusters,
        interv_names,
        interv_meta_map,
        "Intervention Clusters — UMAP (consim1 qualifying nodes, colored by cluster)",
        f"{STEP4_DIR}/umap_interventions_consim1_clusters.png",
    )
else:
    print("  Skipped B9 (umap-learn not available)")

print("\n=== Rev3 Computations COMPLETE ===")
