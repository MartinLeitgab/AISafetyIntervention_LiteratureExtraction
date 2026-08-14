"""
B2 — PathbuildA intra-cluster centroid similarity distributions.

Loads the archived PathbuildA KMeans model and cluster labels, computes
the mean body embedding per path (matching what KMeans was trained on),
then plots a 40-histogram canvas showing each cluster's member-to-centroid
cosine similarity distribution.

Expected finding: wider/lower distributions than risk/intervention clusters
because PathbuildA body embeddings span multiple concept subtypes.

Outputs:
  step4_metaclusters/pathbuildA_intracentroid_histograms.png
  step4_metaclusters/pathbuildA_intracentroid_stats.csv
"""

import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
PROJECT_ROOT = BASE.parent
STEP1_DIR = (
    PROJECT_ROOT
    / "graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
)
ARCHIVE = BASE / "phase2_results/step4_finalanalysis/archive_rev3"
META_OUT = BASE / "phase2_results/step4_finalanalysis/step4_metaclusters"
META_OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load PathbuildA PKLs from archive ──────────────────────────────────
print("Loading optionA_kmeans_model.pkl …", flush=True)
with open(ARCHIVE / "optionA_kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)
centroids = kmeans.cluster_centers_  # (40, 1536)
print(f"  {kmeans.n_clusters} clusters, centroid shape: {centroids.shape}", flush=True)

# Normalize centroids for cosine sim
centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

print("Loading optionA_cluster_labels.pkl …", flush=True)
with open(ARCHIVE / "optionA_cluster_labels.pkl", "rb") as f:
    data = pickle.load(f)
labels = data["labels"]  # (N,) cluster assignment per path
records = data["records"]  # list of (body_node_ids, full_path_ids) tuples
print(f"  {len(labels):,} paths, {len(records):,} records", flush=True)

# ── 2. Load node embeddings ────────────────────────────────────────────────
print("Loading graph_node_attributes.pkl …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs: dict = pickle.load(f)
print(f"  {len(node_attrs):,} nodes  ({time.time() - t0:.1f}s)", flush=True)


def parse_embedding(emb) -> np.ndarray | None:
    if emb is None:
        return None
    if isinstance(emb, np.ndarray):
        return emb.astype(np.float32)
    if isinstance(emb, str):
        return np.fromstring(emb.strip("<>"), sep=", ").astype(np.float32)
    return np.array(emb, dtype=np.float32)


# ── 3. Compute mean body embedding per path → cosine sim to centroid ───────
print("Computing member-to-centroid cosine sims …", flush=True)
t0 = time.time()

# records[i][0] = body_node_ids list for path i
# records[i][1] = full_path_ids list for path i
# We use body_node_ids (index 0) — these are the interior nodes KMeans was run on

cluster_sims: dict[int, list[float]] = {c: [] for c in range(kmeans.n_clusters)}
n_skipped = 0

for i, (record, label) in enumerate(zip(records, labels)):
    body_nids = record[0]  # list of body node IDs
    if not body_nids:
        n_skipped += 1
        continue

    # Collect valid embeddings
    embs = [
        parse_embedding(node_attrs.get(nid, {}).get("embedding")) for nid in body_nids
    ]
    embs = [e for e in embs if e is not None and len(e) > 0]
    if not embs:
        n_skipped += 1
        continue

    mean_emb = np.mean(embs, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_emb)
    if norm < 1e-12:
        n_skipped += 1
        continue
    mean_emb_norm = mean_emb / norm

    # Cosine sim to assigned cluster centroid
    cos_sim = float(np.dot(mean_emb_norm, centroids_norm[label]))
    cluster_sims[label].append(cos_sim)

print(f"  Done in {time.time() - t0:.1f}s  (skipped {n_skipped:,} paths)", flush=True)
for c in range(kmeans.n_clusters):
    print(f"  Cluster {c:2d}: {len(cluster_sims[c]):,} paths", flush=True)

# ── 4. Stats CSV ───────────────────────────────────────────────────────────
rows = []
for c in range(kmeans.n_clusters):
    sims = cluster_sims[c]
    if sims:
        rows.append(
            {
                "cluster_id": c,
                "n_paths": len(sims),
                "mean_cosine_sim": float(np.mean(sims)),
                "median_cosine_sim": float(np.median(sims)),
                "std_cosine_sim": float(np.std(sims)),
                "min_cosine_sim": float(np.min(sims)),
                "max_cosine_sim": float(np.max(sims)),
            }
        )
    else:
        rows.append(
            {
                "cluster_id": c,
                "n_paths": 0,
                "mean_cosine_sim": np.nan,
                "median_cosine_sim": np.nan,
                "std_cosine_sim": np.nan,
                "min_cosine_sim": np.nan,
                "max_cosine_sim": np.nan,
            }
        )

stats_df = pd.DataFrame(rows)
stats_path = META_OUT / "pathbuildA_intracentroid_stats.csv"
stats_df.to_csv(stats_path, index=False)
print(f"\nStats saved: {stats_path}")
print(f"Overall mean cosine sim: {stats_df['mean_cosine_sim'].mean():.4f}")
print(f"Overall median cosine sim: {stats_df['median_cosine_sim'].mean():.4f}")

# ── 5. 40-histogram canvas (8×5 grid) ─────────────────────────────────────
print("Plotting 40-histogram canvas …", flush=True)
NCOLS = 8
NROWS = 5
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(24, 14))
axes_flat = axes.flatten()

overall_mean = stats_df["mean_cosine_sim"].mean()

for c in range(40):
    ax = axes_flat[c]
    sims = cluster_sims[c]
    n = len(sims)
    if n == 0:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title(f"C{c}", fontsize=8)
        continue

    mean_s = float(np.mean(sims))
    ax.hist(sims, bins=30, color="coral", edgecolor="none", alpha=0.85)
    ax.axvline(mean_s, color="darkred", linewidth=1.2, linestyle="--")
    ax.set_title(f"C{c}  μ={mean_s:.3f}  n={n}", fontsize=7.5)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("cos sim to centroid", fontsize=6)
    ax.set_ylabel("count", fontsize=6)

# Hide any unused panels (shouldn't be any for 40 clusters in 8×5)
for i in range(40, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle(
    f"PathbuildA Chain Clusters — Intra-centroid Cosine Similarity Distributions\n"
    f"(40 KMeans clusters on mean chain-body embeddings; overall mean={overall_mean:.3f})\n"
    f"Note: wider/lower distributions expected — body spans multiple concept subtypes",
    fontsize=11,
    y=1.01,
)
plt.tight_layout()
out_path = META_OUT / "pathbuildA_intracentroid_histograms.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}", flush=True)
print("Done — B2 PathbuildA intra-centroid complete.", flush=True)
