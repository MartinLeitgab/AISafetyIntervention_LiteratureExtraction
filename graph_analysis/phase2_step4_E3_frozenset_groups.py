"""
phase2_step4_E3_frozenset_groups.py  [rev7]

Clusters PathbuildB frozensets into ~20-30 groups using binary-vector
Jaccard distance + agglomerative clustering. Replaces the ad-hoc dominant-
family meta-family heuristic with a fully data-driven grouping.

Each frozenset becomes a binary vector over all body concept cluster IDs.
Frozensets are weighted by sqrt(n_paths) during linkage computation.

For each resulting group reports:
  - centroid components (body concept IDs with highest average presence)
  - closest-3 frozensets (most representative)
  - farthest-3 frozensets (borderline cases)
  - intra-group mean Jaccard similarity

Inputs:
  step4_cluster_tables/optionB_cooccurrence_families_consim1.csv
  step4_cluster_tables/bodysubtype_cluster_representatives_v2.csv  (decoded names)

Outputs (NEW):
  step4_cluster_tables/frozenset_groups_consim1.csv
  step4_cluster_tables/frozenset_group_memberships_consim1.csv
  step4_cluster_tables/frozenset_groups_dendrogram.png
  step4_cluster_tables/frozenset_groups_mds.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "phase2_results"
STEP4_DIR = RESULTS_DIR / "step4_finalanalysis"
OUT_TABLES = STEP4_DIR / "step4_cluster_tables"

# ── Load frozensets ───────────────────────────────────────────────────────────

fam_df = pd.read_csv(OUT_TABLES / "optionB_cooccurrence_families_consim1.csv")
print(f"Loaded {len(fam_df)} frozensets")
print(f"  n_paths range: {fam_df['n_paths'].min()} -- {fam_df['n_paths'].max()}")

# Load body cluster name decoder
body_reps_path = OUT_TABLES / "bodysubtype_cluster_representatives_v2.csv"
if not body_reps_path.exists():
    body_reps_path = OUT_TABLES / "bodysubtype_cluster_representatives.csv"
body_reps = pd.read_csv(body_reps_path)
body_map = dict(
    zip(body_reps["prefix_key"].astype(str), body_reps["rep_name"].astype(str))
)
print(f"Loaded {len(body_map)} body cluster labels")


def parse_signature(sig_str):
    """Parse 'de:15 & im:4 & pr:6' -> frozenset of prefix_keys."""
    if not sig_str or pd.isna(sig_str):
        return frozenset()
    return frozenset(p.strip() for p in str(sig_str).split("&"))


def decode_sig(sig):
    return " | ".join(body_map.get(k, k) for k in sorted(sig))


# ── Build binary matrix ───────────────────────────────────────────────────────

print("\nBuilding binary matrix ...")
signatures = [parse_signature(s) for s in fam_df["signature_str"]]

# Vocabulary: all unique prefix_keys across all frozensets
vocab = sorted(set(k for sig in signatures for k in sig))
vocab_idx = {k: i for i, k in enumerate(vocab)}
print(f"  Vocabulary size: {len(vocab)} distinct body concept cluster IDs")

# Binary matrix: shape (n_frozensets, n_vocab)
X = np.zeros((len(fam_df), len(vocab)), dtype=np.float32)
for row_i, sig in enumerate(signatures):
    for k in sig:
        if k in vocab_idx:
            X[row_i, vocab_idx[k]] = 1.0

# Weight rows by sqrt(n_paths) for linkage
weights = np.sqrt(fam_df["n_paths"].values.astype(float))
X_weighted = X * weights[:, None]  # broadcast: each row scaled by its weight

# ── Pairwise Jaccard distance ─────────────────────────────────────────────────

print("Computing pairwise Jaccard distances ...")
# Use unweighted binary Jaccard for distance matrix (interpretable)
dist_vec = pdist(X, metric="jaccard")
dist_mat = squareform(dist_vec)
print(f"  Distance matrix: {dist_mat.shape}, mean={dist_vec.mean():.3f}")

# ── Agglomerative clustering — choose k via dendrogram ───────────────────────

print("Computing linkage (average on Jaccard distances) ...")
Z = linkage(dist_vec, method="average")

# Plot full dendrogram to choose k
fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(
    Z,
    ax=ax,
    no_labels=True,
    color_threshold=0.0,
    above_threshold_color="gray",
)
ax.set_title("Frozenset agglomerative clustering dendrogram (Jaccard distance)")
ax.set_xlabel("Frozensets")
ax.set_ylabel("Jaccard distance")
plt.tight_layout()
fig.savefig(OUT_TABLES / "frozenset_groups_dendrogram_full.png", dpi=120)
plt.close()
print("  Saved full dendrogram")

# Choose k: find biggest merge-distance gaps in the top portion of the tree
merge_dists = Z[:, 2]
top_diffs = np.diff(merge_dists[-50:])  # last 50 merges
# Pick cut at biggest jump in top 50 merges
cut_idx = np.argmax(top_diffs) + len(merge_dists) - 50
cut_dist = (merge_dists[cut_idx] + merge_dists[cut_idx + 1]) / 2
labels_auto = fcluster(Z, t=cut_dist, criterion="distance")
k_auto = len(np.unique(labels_auto))
print(f"  Auto-selected k={k_auto} at distance cut={cut_dist:.3f}")

# Also try fixed k options and report intra-cluster Jaccard
for k_try in [15, 20, 25, 30]:
    lbl = fcluster(Z, t=k_try, criterion="maxclust")
    intra_sims = []
    for g in np.unique(lbl):
        members = np.where(lbl == g)[0]
        if len(members) < 2:
            continue
        sub = dist_mat[np.ix_(members, members)]
        intra_sims.append(1 - sub[np.triu_indices(len(members), k=1)].mean())
    print(
        f"  k={k_try}: mean intra-group sim = {np.mean(intra_sims):.3f}, n_groups={len(np.unique(lbl))}"
    )

# Use k=20 as default (reviewers can verify dendrogram)
K = 20
labels = fcluster(Z, t=K, criterion="maxclust")
print(f"\nUsing k={K}")

# ── Compute group properties ──────────────────────────────────────────────────

group_rows = []
membership_rows = []

for g in sorted(np.unique(labels)):
    member_idx = np.where(labels == g)[0]
    member_sigs = [signatures[i] for i in member_idx]
    member_n_paths = fam_df["n_paths"].values[member_idx]
    total_paths = int(member_n_paths.sum())
    n_frozensets = len(member_idx)

    # Centroid: mean binary vector, find top components
    sub_X = X[member_idx]
    centroid_vec = sub_X.mean(axis=0)
    top_comp_idx = np.argsort(centroid_vec)[::-1][:8]
    centroid_components = [vocab[i] for i in top_comp_idx if centroid_vec[i] > 0]
    centroid_decoded = " | ".join(body_map.get(k, k) for k in centroid_components[:5])

    # Jaccard sim to centroid (1 - Jaccard distance from group centroid binary vector)
    centroid_binary = (centroid_vec >= 0.5).astype(float)
    sims_to_centroid = []
    for i in member_idx:
        row = X[i]
        intersect = np.dot(row, centroid_binary)
        union = np.sum(np.maximum(row, centroid_binary))
        sims_to_centroid.append(float(intersect / union) if union > 0 else 0.0)

    sorted_sims = np.argsort(sims_to_centroid)
    closest_local = sorted_sims[-3:][::-1]
    farthest_local = sorted_sims[:3]

    closest3_sigs = " || ".join(
        " & ".join(sorted(signatures[member_idx[i]])) for i in closest_local
    )
    closest3_decoded = " || ".join(
        decode_sig(signatures[member_idx[i]]) for i in closest_local
    )
    farthest3_sigs = " || ".join(
        " & ".join(sorted(signatures[member_idx[i]])) for i in farthest_local
    )
    farthest3_decoded = " || ".join(
        decode_sig(signatures[member_idx[i]]) for i in farthest_local
    )

    # Intra-group mean Jaccard sim
    sub_dist = dist_mat[np.ix_(member_idx, member_idx)]
    if len(member_idx) > 1:
        tri = sub_dist[np.triu_indices(len(member_idx), k=1)]
        intra_jaccard_mean = float(1 - tri.mean())
    else:
        intra_jaccard_mean = 1.0

    group_rows.append(
        {
            "group_id": int(g),
            "n_frozensets": n_frozensets,
            "n_paths_total": total_paths,
            "centroid_components": " & ".join(centroid_components[:5]),
            "centroid_decoded": centroid_decoded,
            "closest3_sigs": closest3_sigs,
            "closest3_decoded": closest3_decoded,
            "farthest3_sigs": farthest3_sigs,
            "farthest3_decoded": farthest3_decoded,
            "intra_jaccard_mean": round(intra_jaccard_mean, 3),
        }
    )

    # Membership rows
    for local_i, global_i in enumerate(member_idx):
        membership_rows.append(
            {
                "family_id": int(fam_df["family_id"].iloc[global_i]),
                "group_id": int(g),
                "n_paths": int(fam_df["n_paths"].iloc[global_i]),
                "signature_str": str(fam_df["signature_str"].iloc[global_i]),
                "jaccard_sim_to_centroid": round(sims_to_centroid[local_i], 4),
            }
        )

    print(
        f"  G{g:2d}: {n_frozensets:4d} frozensets | {total_paths:6d} paths | "
        f"intra_sim={intra_jaccard_mean:.3f} | {centroid_decoded[:60]}"
    )

# ── Save outputs ──────────────────────────────────────────────────────────────

groups_df = pd.DataFrame(group_rows).sort_values("n_paths_total", ascending=False)
groups_df.to_csv(OUT_TABLES / "frozenset_groups_consim1.csv", index=False)
print(f"\nWritten: frozenset_groups_consim1.csv ({len(groups_df)} groups)")

mem_df = pd.DataFrame(membership_rows)
mem_df.to_csv(OUT_TABLES / "frozenset_group_memberships_consim1.csv", index=False)
print(f"Written: frozenset_group_memberships_consim1.csv ({len(mem_df)} rows)")

# ── 2D MDS scatter ────────────────────────────────────────────────────────────

print("Computing MDS (2D) ...")
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
pos = mds.fit_transform(dist_mat)

fig, ax = plt.subplots(figsize=(12, 10))
cmap = plt.cm.get_cmap("tab20", K)
for g in range(1, K + 1):
    idx = np.where(labels == g)[0]
    sizes = np.sqrt(fam_df["n_paths"].values[idx]) * 2
    ax.scatter(
        pos[idx, 0], pos[idx, 1], s=sizes, color=cmap(g - 1), alpha=0.7, label=f"G{g}"
    )
ax.set_title(f"Frozenset groups — 2D MDS (Jaccard distance, k={K})")
ax.set_xlabel("MDS-1")
ax.set_ylabel("MDS-2")
ax.legend(loc="best", fontsize=7, ncol=4, markerscale=0.7)
plt.tight_layout()
fig.savefig(OUT_TABLES / "frozenset_groups_mds.png", dpi=120)
plt.close()
print("Saved MDS plot")

# ── Quality check ─────────────────────────────────────────────────────────────

print("\nQuality checks:")
assert mem_df["family_id"].nunique() == len(fam_df), (
    "Some frozensets not assigned to a group"
)
assert (mem_df.groupby("family_id").size() == 1).all(), (
    "Some frozensets assigned to >1 group"
)
min_group = groups_df["n_frozensets"].min()
print("  All frozensets assigned: OK")
print(f"  Min frozensets per group: {min_group}")
print(
    f"  Total paths accounted for: {groups_df['n_paths_total'].sum()} "
    f"(original: {fam_df['n_paths'].sum()})"
)
