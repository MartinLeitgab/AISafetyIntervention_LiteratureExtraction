"""
phase2_step4_umap_plots.py

Produce UMAP 2D projection plots for valid_pathway_nodes-filtered
risk and intervention cluster members (edge_config=0.9, mode=unconstrained,
algo=agglomerative).

Outputs:
  graph_analysis/phase2_results/step4_finalanalysis/umap_risks.png
  graph_analysis/phase2_results/step4_finalanalysis/umap_interventions.png
"""

import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP1_DIR = (
    PROJECT_ROOT
    / "graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
)
PATHS_FILE = (
    PROJECT_ROOT
    / "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
)
OUT_DIR = PROJECT_ROOT / "graph_analysis/phase2_results/step4_finalanalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load PKL files
# ---------------------------------------------------------------------------
print("Loading graph_node_attributes.pkl …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs: dict = pickle.load(f)
print(f"  Loaded {len(node_attrs):,} nodes in {time.time() - t0:.1f}s", flush=True)

print("Loading cluster_memberships.pkl …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "cluster_memberships.pkl", "rb") as f:
    cluster_memberships: dict = pickle.load(f)
print(
    f"  Loaded {len(cluster_memberships):,} cluster records in {time.time() - t0:.1f}s",
    flush=True,
)

# ---------------------------------------------------------------------------
# 2. Build valid_pathway_nodes from paths_unconstrained_sim0.9.jsonl
# ---------------------------------------------------------------------------
print("Building valid_pathway_nodes …", flush=True)
t0 = time.time()
valid_pathway_nodes: set = set()
with open(PATHS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for node_id in record["path"]:
            valid_pathway_nodes.add(node_id)
print(
    f"  {len(valid_pathway_nodes):,} unique pathway nodes in {time.time() - t0:.1f}s",
    flush=True,
)


# ---------------------------------------------------------------------------
# Helper: parse embedding string if needed
# ---------------------------------------------------------------------------
def parse_embedding(emb) -> np.ndarray:
    if isinstance(emb, np.ndarray):
        return emb.astype(np.float32)
    if isinstance(emb, str):
        return np.fromstring(emb.strip("<>"), sep=", ").astype(np.float32)
    # fallback: try converting
    return np.array(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: build color list cycling through tab20 + tab20b for 40 clusters
# ---------------------------------------------------------------------------
def build_color_map(cluster_ids: list) -> dict:
    tab20 = plt.cm.tab20.colors  # 20 colours
    tab20b = plt.cm.tab20b.colors  # 20 colours
    palette = list(tab20) + list(tab20b)  # 40 colours total
    unique_sorted = sorted(set(cluster_ids))
    return {cid: palette[i % len(palette)] for i, cid in enumerate(unique_sorted)}


# ---------------------------------------------------------------------------
# 3 & 4. Collect filtered cluster members for risk and intervention
# ---------------------------------------------------------------------------
EDGE_CONFIG = 0.9
MODE = "unconstrained"
ALGO = "agglomerative"


def collect_filtered_members(node_type: str):
    """Return (node_ids, cluster_labels) for valid_pathway_nodes-filtered members."""
    node_ids = []
    labels = []
    for (ec, mode, nt, algo, cluster_id), members in cluster_memberships.items():
        if ec == EDGE_CONFIG and mode == MODE and nt == node_type and algo == ALGO:
            for nid in members:
                if nid in valid_pathway_nodes:
                    node_ids.append(nid)
                    labels.append(cluster_id)
    return node_ids, labels


print("\nCollecting filtered risk cluster members …", flush=True)
risk_ids, risk_labels = collect_filtered_members("risk")
print(f"  {len(risk_ids):,} risk nodes pass valid_pathway_nodes filter", flush=True)

print("Collecting filtered intervention cluster members …", flush=True)
interv_ids, interv_labels = collect_filtered_members("intervention")
print(
    f"  {len(interv_ids):,} intervention nodes pass valid_pathway_nodes filter",
    flush=True,
)


# ---------------------------------------------------------------------------
# 5. Build embedding matrices
# ---------------------------------------------------------------------------
def build_embedding_matrix(node_ids: list) -> np.ndarray:
    embs = []
    missing = 0
    for nid in node_ids:
        attrs = node_attrs.get(nid)
        if attrs is None or attrs.get("embedding") is None:
            missing += 1
            embs.append(None)
            continue
        embs.append(parse_embedding(attrs["embedding"]))
    if missing:
        print(
            f"    WARNING: {missing} nodes have no embedding — they will be dropped",
            flush=True,
        )
    dim = next((e.shape[0] for e in embs if e is not None), None)
    if dim is None:
        raise ValueError("No valid embeddings found")
    matrix = np.array(
        [e if e is not None else np.zeros(dim, dtype=np.float32) for e in embs],
        dtype=np.float32,
    )
    return matrix, missing


print("\nBuilding risk embedding matrix …", flush=True)
t0 = time.time()
risk_matrix, risk_missing = build_embedding_matrix(risk_ids)
print(
    f"  Shape: {risk_matrix.shape}, missing: {risk_missing}, time: {time.time() - t0:.1f}s",
    flush=True,
)

print("Building intervention embedding matrix …", flush=True)
t0 = time.time()
interv_matrix, interv_missing = build_embedding_matrix(interv_ids)
print(
    f"  Shape: {interv_matrix.shape}, missing: {interv_missing}, time: {time.time() - t0:.1f}s",
    flush=True,
)


# ---------------------------------------------------------------------------
# 6. Run UMAP
# ---------------------------------------------------------------------------
UMAP_PARAMS = dict(
    n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
)

print("\nRunning UMAP for risk nodes …", flush=True)
t0 = time.time()
reducer_risk = umap.UMAP(**UMAP_PARAMS)
risk_2d = reducer_risk.fit_transform(risk_matrix)
print(
    f"  UMAP done in {time.time() - t0:.1f}s, output shape: {risk_2d.shape}", flush=True
)

print("Running UMAP for intervention nodes …", flush=True)
t0 = time.time()
reducer_interv = umap.UMAP(**UMAP_PARAMS)
interv_2d = reducer_interv.fit_transform(interv_matrix)
print(
    f"  UMAP done in {time.time() - t0:.1f}s, output shape: {interv_2d.shape}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 7. Plot and save
# ---------------------------------------------------------------------------
def make_umap_plot(coords_2d, labels, title, out_path):
    color_map = build_color_map(labels)
    colors = [color_map[lb] for lb in labels]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=colors,
        alpha=0.5,
        s=3,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.tick_params(labelsize=8)

    # Legend (one patch per cluster)
    unique_labels = sorted(set(labels))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[cid],
            markersize=6,
            label=f"Cluster {cid}",
        )
        for cid in unique_labels
    ]
    ax.legend(
        handles=handles,
        title="Cluster",
        title_fontsize=7,
        fontsize=6,
        loc="upper right",
        ncol=2,
        framealpha=0.7,
        markerscale=1.2,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


n_risk_plotted = risk_matrix.shape[0]
n_interv_plotted = interv_matrix.shape[0]

print("\nPlotting risk UMAP …", flush=True)
make_umap_plot(
    risk_2d,
    risk_labels,
    f"Risk Clusters — UMAP 2D (valid_pathway_nodes, n={n_risk_plotted:,})",
    OUT_DIR / "umap_risks.png",
)

print("Plotting intervention UMAP …", flush=True)
make_umap_plot(
    interv_2d,
    interv_labels,
    f"Intervention Clusters — UMAP 2D (valid_pathway_nodes, n={n_interv_plotted:,})",
    OUT_DIR / "umap_interventions.png",
)

print("\nDone.", flush=True)
