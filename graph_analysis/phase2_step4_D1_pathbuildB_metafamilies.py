"""
Phase 2 Step 4 D1 — Meta-cluster pathbuildB families via Jaccard similarity.

Algorithm:
  1. Parse all 1,603 consim1 B-family signatures into frozensets of components
  2. Compute pairwise Jaccard distance matrix
  3. Hierarchical agglomerative clustering (average linkage)
  4. Scan k=20..80: pick k that maximises mean intra-meta Jaccard sim (target ≥0.3)
  5. Assign meta-family IDs; name each by dominant component pattern
  6. Aggregate R→C→I edges to meta-family level
  7. Regenerate three-layer network with meta-families as L2 layer

Outputs:
  step4_cluster_tables/pathbuildB_metafamilies_consim1.csv
  step4_connectivity/risk_to_metafamily_edges_consim1.csv
  step4_connectivity/metafamily_to_interv_edges_consim1.csv
  step4_connectivity/three_layer_network_pathbuildB_metafamily_consim1.png
"""

import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = "/mnt/c/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/AISafetyIntervention_LiteratureExtraction"
ROOT = f"{BASE}/graph_analysis"
RESULTS = f"{ROOT}/phase2_results"
TABLES_DIR = f"{RESULTS}/step4_finalanalysis/step4_cluster_tables"
CONN_DIR = f"{RESULTS}/step4_finalanalysis/step4_connectivity"
NAMING_DIR = f"{RESULTS}/step5_naming"

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CONN_DIR, exist_ok=True)

# ─── Load families ─────────────────────────────────────────────────────────────
print("Loading consim1 B-families ...")
fam_df = pd.read_csv(f"{TABLES_DIR}/optionB_cooccurrence_families_consim1.csv")
print(f"  {len(fam_df)} families loaded")


def parse_sig(sig_str):
    """Parse 'de:15 & im:4 & pr:6' → frozenset of components."""
    if not isinstance(sig_str, str) or not sig_str.strip():
        return frozenset()
    return frozenset(p.strip() for p in sig_str.split("&"))


fam_df["components"] = fam_df["signature_str"].apply(parse_sig)
fam_ids = fam_df["family_id"].tolist()
n = len(fam_ids)

# ─── Build binary component matrix ────────────────────────────────────────────
print("Building binary component matrix ...")
all_components = sorted(set(c for fs in fam_df["components"] for c in fs))
comp_idx = {c: i for i, c in enumerate(all_components)}
print(f"  {len(all_components)} unique components across {n} families")

# Binary matrix: rows=families, cols=components
X = np.zeros((n, len(all_components)), dtype=np.uint8)
for i, row in fam_df.iterrows():
    for comp in row["components"]:
        j = comp_idx.get(comp)
        if j is not None:
            X[i, j] = 1

# ─── Pairwise Jaccard distance ────────────────────────────────────────────────
print("Computing pairwise Jaccard distance matrix ...")


def jaccard_batch(X):
    """Efficient Jaccard distance for binary matrix using dot products."""
    # intersection[i,j] = sum over k of (X[i,k] AND X[j,k]) = X @ X.T
    inter = X @ X.T  # (n, n), dtype int
    # |A| for each row
    row_sums = X.sum(axis=1, keepdims=True)  # (n, 1)
    # union[i,j] = |A| + |B| - |A∩B|
    union = row_sums + row_sums.T - inter  # (n, n)
    # Jaccard sim; handle union=0 case (both empty → sim=1)
    sim = np.where(union > 0, inter / union, 1.0)
    np.fill_diagonal(sim, 1.0)
    return 1.0 - sim  # distance matrix


dist_mat = jaccard_batch(X.astype(np.float32))
print(f"  Distance matrix shape: {dist_mat.shape}")
print(f"  Distance range: {dist_mat.min():.3f} – {dist_mat.max():.3f}")

# ─── Hierarchical clustering ──────────────────────────────────────────────────
print("Running hierarchical clustering (average linkage) ...")
# Convert to condensed form
np.fill_diagonal(dist_mat, 0.0)
dist_mat = np.clip(dist_mat, 0, None)
condensed = squareform(dist_mat, checks=False)
Z = linkage(condensed, method="average")

# ─── Scan k to find optimal number of meta-families ──────────────────────────
print("Scanning k=20..80 for optimal meta-family count ...")


def mean_intra_jaccard(labels, dist_mat):
    """Compute mean of intra-cluster pairwise similarities (1 - dist)."""
    unique_labels = np.unique(labels)
    intra_sims = []
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        sub_dist = dist_mat[np.ix_(idx, idx)]
        mask = ~np.eye(len(idx), dtype=bool)
        intra_sims.append((1.0 - sub_dist[mask]).mean())
    return np.mean(intra_sims) if intra_sims else 0.0


scan_results = []
for k in range(20, 81):
    labels = fcluster(Z, t=k, criterion="maxclust")
    mean_sim = mean_intra_jaccard(labels, dist_mat)
    scan_results.append({"k": k, "mean_intra_jaccard": round(mean_sim, 4)})

scan_df = pd.DataFrame(scan_results)
print("\n  k scan results:")
print(
    scan_df[scan_df["mean_intra_jaccard"] >= 0.25][
        ["k", "mean_intra_jaccard"]
    ].to_string(index=False)
)

# Pick k where mean_intra_jaccard first exceeds 0.30, or the best k otherwise
target_k_df = scan_df[scan_df["mean_intra_jaccard"] >= 0.30]
if len(target_k_df) > 0:
    best_k = int(target_k_df.iloc[0]["k"])
else:
    best_k = int(scan_df.loc[scan_df["mean_intra_jaccard"].idxmax(), "k"])

print(
    f"\n  Selected k={best_k} (mean_intra_jaccard={scan_df.loc[scan_df['k'] == best_k, 'mean_intra_jaccard'].values[0]:.4f})"
)

labels_best = fcluster(Z, t=best_k, criterion="maxclust")
fam_df["meta_family_id"] = labels_best

# ─── Compute meta-family summary names ────────────────────────────────────────
print("Computing meta-family summaries ...")

# Load chain names for cross-reference
try:
    chain_names_df = pd.read_csv(f"{NAMING_DIR}/pathbuildB_chain_names_llm.csv")
    chain_name_map = dict(
        zip(
            chain_names_df["cluster_id"].astype(str),
            chain_names_df.get(
                "final_name", chain_names_df.get("llm_name", pd.Series(dtype=str))
            ),
        )
    )
except Exception:
    chain_name_map = {}

meta_summary_rows = []
for mf_id in sorted(fam_df["meta_family_id"].unique()):
    sub = fam_df[fam_df["meta_family_id"] == mf_id].copy()
    n_fam = len(sub)
    n_paths_total = int(sub["n_paths"].sum())
    # Dominant family (highest n_paths)
    dominant = sub.sort_values("n_paths", ascending=False).iloc[0]
    dominant_name = chain_name_map.get(
        str(int(dominant["family_id"])), f"B{int(dominant['family_id'])}"
    )
    # Most common components across all families in this meta-family
    comp_counts = {}
    for fs in sub["components"]:
        for c in fs:
            comp_counts[c] = comp_counts.get(c, 0) + 1
    # Components present in ≥50% of member families
    core_components = sorted(
        [c for c, cnt in comp_counts.items() if cnt >= n_fam * 0.5]
    )
    # Mean intra-meta Jaccard sim
    member_indices = sub.index.tolist()
    if len(member_indices) >= 2:
        sub_dist = dist_mat[np.ix_(member_indices, member_indices)]
        mask = ~np.eye(len(member_indices), dtype=bool)
        mean_intra = float((1.0 - sub_dist[mask]).mean())
    else:
        mean_intra = 1.0
    meta_summary_rows.append(
        {
            "meta_family_id": mf_id,
            "n_families": n_fam,
            "n_paths_total": n_paths_total,
            "dominant_family_id": int(dominant["family_id"]),
            "dominant_family_name": dominant_name,
            "core_components": " & ".join(core_components),
            "mean_intra_jaccard": round(mean_intra, 3),
        }
    )

meta_summary_df = pd.DataFrame(meta_summary_rows).sort_values(
    "n_paths_total", ascending=False
)

# Save full family assignments
fam_out = fam_df[
    ["family_id", "n_paths", "n_sources", "signature_str", "meta_family_id"]
].copy()
fam_out.to_csv(f"{TABLES_DIR}/pathbuildB_metafamilies_consim1.csv", index=False)
print(f"  Saved pathbuildB_metafamilies_consim1.csv ({len(fam_out)} rows)")

meta_summary_df.to_csv(
    f"{TABLES_DIR}/pathbuildB_metafamily_summary_consim1.csv", index=False
)
print(
    f"  Saved pathbuildB_metafamily_summary_consim1.csv ({len(meta_summary_df)} rows)"
)

print("\n  Top-15 meta-families by total paths:")
print(
    meta_summary_df.head(15)[
        [
            "meta_family_id",
            "n_families",
            "n_paths_total",
            "dominant_family_name",
            "mean_intra_jaccard",
        ]
    ].to_string(index=False)
)

# Save k scan results
scan_df.to_csv(f"{TABLES_DIR}/pathbuildB_metafamily_k_scan.csv", index=False)

# ─── Aggregate R→C edge and C→I edges to meta-family level ───────────────────
print("\nAggregating connectivity edges to meta-family level ...")

r2c_df = pd.read_csv(f"{CONN_DIR}/risk_to_Bfamily_edges_consim1.csv")
c2i_df = pd.read_csv(f"{CONN_DIR}/Bfamily_to_interv_edges_consim1.csv")

family_to_meta = dict(zip(fam_df["family_id"], fam_df["meta_family_id"]))

# Risk → meta-family
r2c_df["meta_family_id"] = r2c_df["cluster_b"].map(family_to_meta)
r2meta_df = (
    r2c_df.dropna(subset=["meta_family_id"])
    .groupby(["cluster_a", "meta_family_id"], as_index=False)["n_paths"]
    .sum()
    .rename(columns={"cluster_a": "risk_cluster", "meta_family_id": "meta_family_id"})
)
r2meta_df["meta_family_id"] = r2meta_df["meta_family_id"].astype(int)
r2meta_df.to_csv(f"{CONN_DIR}/risk_to_metafamily_edges_consim1.csv", index=False)
print(f"  Saved risk_to_metafamily_edges_consim1.csv ({len(r2meta_df)} rows)")

# Meta-family → intervention
c2i_df["meta_family_id"] = c2i_df["cluster_a"].map(family_to_meta)
meta2i_df = (
    c2i_df.dropna(subset=["meta_family_id"])
    .groupby(["meta_family_id", "cluster_b"], as_index=False)["n_paths"]
    .sum()
    .rename(columns={"meta_family_id": "meta_family_id", "cluster_b": "interv_cluster"})
)
meta2i_df["meta_family_id"] = meta2i_df["meta_family_id"].astype(int)
meta2i_df.to_csv(f"{CONN_DIR}/metafamily_to_interv_edges_consim1.csv", index=False)
print(f"  Saved metafamily_to_interv_edges_consim1.csv ({len(meta2i_df)} rows)")

# ─── Three-layer network with meta-families ───────────────────────────────────
print("\nGenerating three-layer network with meta-families ...")


def load_names(path):
    try:
        df = pd.read_csv(path)
        col = "final_name" if "final_name" in df.columns else "llm_name"
        return {int(r["cluster_id"]): str(r[col]) for _, r in df.iterrows()}
    except Exception:
        return {}


risk_names = load_names(f"{NAMING_DIR}/risk_cluster_names_llm_v2.csv")
interv_names = load_names(f"{NAMING_DIR}/intervention_cluster_names_llm_v2.csv")
# Meta-family names = dominant family name
meta_names = {
    int(row["meta_family_id"]): row["dominant_family_name"]
    for _, row in meta_summary_df.iterrows()
}

MAX_RISK = 20
MAX_META = 30
MAX_INTERV = 20
MAX_EDGES = 200

# Build r2i totals for node selection
r2i_agg = (
    r2c_df.dropna(subset=["meta_family_id"])
    .merge(
        meta2i_df.rename(columns={"meta_family_id": "mf_id"}),
        left_on=["meta_family_id"],
        right_on=["mf_id"],
        how="inner",
    )
    .groupby(["cluster_a", "interv_cluster"], as_index=False)["n_paths_x"]
    .sum()
    .rename(columns={"cluster_a": "risk_cluster", "n_paths_x": "n_paths"})
)

top_risk = (
    r2meta_df.groupby("risk_cluster")["n_paths"]
    .sum()
    .nlargest(MAX_RISK)
    .index.astype(int)
    .tolist()
)
top_meta = (
    r2meta_df.groupby("meta_family_id")["n_paths"]
    .sum()
    .nlargest(MAX_META)
    .index.astype(int)
    .tolist()
)
top_interv = (
    meta2i_df.groupby("interv_cluster")["n_paths"]
    .sum()
    .nlargest(MAX_INTERV)
    .index.astype(int)
    .tolist()
)


def y_positions(items):
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0]: 0.5}
    return {item: 1.0 - i / (n - 1) for i, item in enumerate(items)}


risk_y = y_positions(top_risk)
meta_y = y_positions(top_meta)
interv_y = y_positions(top_interv)

X_RISK = 0.0
X_META = 0.5
X_INTERV = 1.0

n_r = len(top_risk)
n_m = len(top_meta)
n_i = len(top_interv)

fig, ax = plt.subplots(figsize=(32, max(16, max(n_r, n_m, n_i) * 0.45)))

# Risk → meta edges
sub_r2m = r2meta_df[
    r2meta_df["risk_cluster"].astype(int).isin(top_risk)
    & r2meta_df["meta_family_id"].isin(top_meta)
].head(MAX_EDGES)
if len(sub_r2m) > 0:
    max_lp = np.log1p(sub_r2m["n_paths"].max())
    for _, row in sub_r2m.iterrows():
        rc = int(row["risk_cluster"])
        mc = int(row["meta_family_id"])
        ry = risk_y.get(rc)
        my = meta_y.get(mc)
        if ry is None or my is None:
            continue
        lw = max(0.2, np.log1p(row["n_paths"]) / max_lp * 4.0)
        ax.plot([X_RISK, X_META], [ry, my], color="steelblue", alpha=0.3, linewidth=lw)

# Meta → intervention edges
sub_m2i = meta2i_df[
    meta2i_df["meta_family_id"].isin(top_meta)
    & meta2i_df["interv_cluster"].astype(int).isin(top_interv)
].head(MAX_EDGES)
if len(sub_m2i) > 0:
    max_lp = np.log1p(sub_m2i["n_paths"].max())
    for _, row in sub_m2i.iterrows():
        mc = int(row["meta_family_id"])
        ic = int(row["interv_cluster"])
        my = meta_y.get(mc)
        iy = interv_y.get(ic)
        if my is None or iy is None:
            continue
        lw = max(0.2, np.log1p(row["n_paths"]) / max_lp * 4.0)
        ax.plot(
            [X_META, X_INTERV], [my, iy], color="darkorange", alpha=0.3, linewidth=lw
        )


def wrap(text, width=36):
    return "\n".join(textwrap.wrap(text, width))


# Risk nodes
for cid, y in risk_y.items():
    ax.scatter(X_RISK, y, s=180, c="steelblue", zorder=4)
    label = wrap(risk_names.get(cid, f"R{cid}"))
    ax.text(
        X_RISK - 0.03,
        y,
        label,
        ha="right",
        va="center",
        fontsize=5.5,
        color="steelblue",
    )

# Meta-family nodes
for mf_id, y in meta_y.items():
    n_fam_here = (
        int(
            meta_summary_df.loc[
                meta_summary_df["meta_family_id"] == mf_id, "n_families"
            ].values[0]
        )
        if mf_id in meta_summary_df["meta_family_id"].values
        else 1
    )
    ax.scatter(X_META, y, s=140, c="seagreen", zorder=4)
    label = (
        wrap(meta_names.get(mf_id, f"MF{mf_id}"), width=30) + f"\n({n_fam_here} fam)"
    )
    ax.text(
        X_META,
        y,
        label,
        ha="center",
        va="center",
        fontsize=4.5,
        color="seagreen",
        rotation=40,
    )

# Intervention nodes
for cid, y in interv_y.items():
    ax.scatter(X_INTERV, y, s=180, c="darkorange", zorder=4)
    label = wrap(interv_names.get(cid, f"I{cid}"))
    ax.text(
        X_INTERV + 0.03,
        y,
        label,
        ha="left",
        va="center",
        fontsize=5.5,
        color="darkorange",
    )

# Headers
ax.text(
    X_RISK,
    1.05,
    "RISK\nClusters",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color="steelblue",
)
ax.text(
    X_META,
    1.05,
    f"META-CHAIN\n(PathbuildB Meta-Families, k={best_k})",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color="seagreen",
)
ax.text(
    X_INTERV,
    1.05,
    "INTERVENTION\nClusters",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color="darkorange",
)

legend_handles = [
    mpatches.Patch(color="steelblue", label="Risk → Meta-family"),
    mpatches.Patch(color="darkorange", label="Meta-family → Intervention"),
]
ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=2,
    fontsize=8,
)

total_r2i = int(r2meta_df["n_paths"].sum())
ax.set_title(
    f"Three-Layer Network — consim1_pathbuildB_metafamilies (k={best_k})\n"
    f"(top-{MAX_RISK} risk, top-{MAX_META} meta-families from {best_k}, top-{MAX_INTERV} interv; "
    f"total r→meta paths: {total_r2i:,})",
    fontsize=11,
    pad=20,
)
ax.set_xlim(-0.75, 1.75)
ax.set_ylim(-0.15, 1.2)
ax.axis("off")
plt.tight_layout()

out_png = f"{CONN_DIR}/three_layer_network_pathbuildB_metafamily_consim1.png"
plt.savefig(out_png, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved {out_png}")

# ─── Dendrogram of top-30 meta-families ──────────────────────────────────────
print("\nGenerating meta-family dendrogram ...")
top30_meta_ids = meta_summary_df.head(30)["meta_family_id"].tolist()
# Compute meta-family centroids (mean of member family binary vectors)
meta_vecs = {}
for mf_id in top30_meta_ids:
    member_idx = fam_df[fam_df["meta_family_id"] == mf_id].index.tolist()
    if member_idx:
        meta_vecs[mf_id] = X[member_idx].mean(axis=0).astype(np.float32)

if len(meta_vecs) >= 2:
    mf_ids = sorted(meta_vecs.keys())
    V = np.stack([meta_vecs[m] for m in mf_ids])
    # Jaccard between centroid vectors (soft Jaccard via min/max)
    n_mf = len(mf_ids)
    meta_dist = np.zeros((n_mf, n_mf))
    for i in range(n_mf):
        for j in range(i + 1, n_mf):
            inter = np.minimum(V[i], V[j]).sum()
            union = np.maximum(V[i], V[j]).sum()
            d = 1.0 - (inter / union if union > 0 else 1.0)
            meta_dist[i, j] = meta_dist[j, i] = d

    np.fill_diagonal(meta_dist, 0.0)
    meta_cond = squareform(meta_dist, checks=False)
    Z_meta = linkage(meta_cond, method="average")

    meta_labels = [
        textwrap.fill(f"MF{mf_id}: {meta_names.get(mf_id, '')[:40]}", width=48)
        for mf_id in mf_ids
    ]
    n_mf = len(meta_labels)
    fig2, ax2 = plt.subplots(figsize=(10, max(8, n_mf * 0.32)))
    dendrogram(Z_meta, labels=meta_labels, ax=ax2, orientation="left", leaf_font_size=7)
    ax2.set_xlabel("Jaccard distance (centroid-based)")
    ax2.set_title(
        f"PathbuildB Meta-Family Dendrogram (top-30 by n_paths, k={best_k})",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    out_dendro = f"{TABLES_DIR}/pathbuildB_metafamily_dendrogram.png"
    fig2.savefig(out_dendro, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {out_dendro}")

print("\n=== D1: PathbuildB meta-family clustering COMPLETE ===")
print(f"  Best k = {best_k}")
print(f"  {len(meta_summary_df)} meta-families covering {len(fam_df)} B-families")
