"""
phase2_step4_umap_plots.py

Produce UMAP 2D projection plots for valid_pathway_nodes-filtered
risk and intervention cluster members (edge_config=0.9, mode=unconstrained,
algo=agglomerative).

Three consim configs are produced:
  consim0 — edge-only paths (paths_unconstrained_edge_only.jsonl)
  consim1 — sim0.9 paths filtered to max_consec_sim <= 1
  consim2 — sim0.9 paths filtered to max_consec_sim <= 2

For intervention nodes a maturity>=3 filter is applied in addition to
valid_pathway_nodes membership.

Outputs (in graph_analysis/phase2_results/step4_finalanalysis/):
  umap_risks.png              — original unconstrained (unchanged)
  umap_interventions.png      — original unconstrained (unchanged)
  umap_risks_consim0.png
  umap_interventions_consim0.png
  umap_risks_consim1.png
  umap_interventions_consim1.png
  umap_risks_consim2.png
  umap_interventions_consim2.png
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
PATHS_SIM09_FILE = (
    PROJECT_ROOT
    / "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
)
PATHS_EDGE_ONLY_FILE = (
    PROJECT_ROOT
    / "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_edge_only.jsonl"
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

print("Loading graph_edge_data.pkl (needed for sim_edge_set) …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data: list = pickle.load(f)
print(
    f"  Loaded {len(edge_data):,} edges in {time.time() - t0:.1f}s",
    flush=True,
)


# ---------------------------------------------------------------------------
# 2. Build unconstrained VPN first (needed to restrict sim_edge_set)
#    maturity>=3 endpoint filter — path gen used ALL_INTERVENTION_IDS
# ---------------------------------------------------------------------------
def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


print(
    "Building unconstrained valid_pathway_nodes for sim_edge_set restriction …",
    flush=True,
)
t0 = time.time()
_vpn_for_sim: set = set()
with open(PATHS_SIM09_FILE) as _f:
    for _line in _f:
        _line = _line.strip()
        if not _line:
            continue
        _record = json.loads(_line)
        _path_ids = _record["path"]
        _interv_id = int(_path_ids[-1])
        if (
            int(node_attrs.get(_interv_id, {}).get("intervention_maturity", 0) or 0)
            >= 3
        ):
            _vpn_for_sim.update(int(x) for x in _path_ids)
print(
    f"  unconstrained VPN for sim restriction: {len(_vpn_for_sim):,} nodes in {time.time() - t0:.1f}s",
    flush=True,
)

# ---------------------------------------------------------------------------
# Build sim_edge_set (cos_sim >= 0.9, restricted to VPN pairs)
# ---------------------------------------------------------------------------
print("Building sim_edge_set (cos_sim >= 0.9, VPN-restricted) …", flush=True)
t0 = time.time()
sim_edge_set: set = set()
for e in edge_data:
    if str(e.get("type", "")).upper() == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= 0.9:
            try:
                s2, t2 = int(e["source"]), int(e["target"])
                if s2 in _vpn_for_sim and t2 in _vpn_for_sim:
                    sim_edge_set.add((min(s2, t2), max(s2, t2)))
            except (ValueError, TypeError):
                pass
print(
    f"  {len(sim_edge_set):,} sim edges at cos_sim>=0.9 in {time.time() - t0:.1f}s",
    flush=True,
)
# edge_data no longer needed — free memory
del edge_data
del _vpn_for_sim


# ---------------------------------------------------------------------------
# 3. Helper: max consecutive SIM hops in a path
# ---------------------------------------------------------------------------
def max_consec_sim(path_ids, sim_set):
    max_run = run = 0
    for i in range(len(path_ids) - 1):
        a, b = int(path_ids[i]), int(path_ids[i + 1])
        if (min(a, b), max(a, b)) in sim_set:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


# ---------------------------------------------------------------------------
# 4. Build valid_pathway_nodes per consim config
# ---------------------------------------------------------------------------
def load_pathway_nodes_edge_only(path_file: Path) -> set:
    """Load node IDs from an edge-only JSONL path file.
    maturity>=3 endpoint filter — path gen used ALL_INTERVENTION_IDS
    """
    nodes: set = set()
    with open(path_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            path_ids = record["path"]
            interv_id = int(path_ids[-1])
            if (
                int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0)
                >= 3
            ):
                nodes.update(int(x) for x in path_ids)
    return nodes


def load_pathway_nodes_sim09_filtered(path_file: Path, max_consec: int) -> set:
    """Load node IDs from sim0.9 JSONL keeping only paths with max_consec_sim <= max_consec.
    maturity>=3 endpoint filter — path gen used ALL_INTERVENTION_IDS
    """
    nodes: set = set()
    with open(path_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            path_ids = record["path"]
            interv_id = int(path_ids[-1])
            if (
                int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0)
                >= 3
                and max_consec_sim(path_ids, sim_edge_set) <= max_consec
            ):
                nodes.update(int(x) for x in path_ids)
    return nodes


print("\nBuilding valid_pathway_nodes for consim0 (edge-only) …", flush=True)
t0 = time.time()
vpn_consim0 = load_pathway_nodes_edge_only(PATHS_EDGE_ONLY_FILE)
print(f"  consim0: {len(vpn_consim0):,} nodes in {time.time() - t0:.1f}s", flush=True)

print("Building valid_pathway_nodes for consim1 (max_consec_sim<=1) …", flush=True)
t0 = time.time()
vpn_consim1 = load_pathway_nodes_sim09_filtered(PATHS_SIM09_FILE, max_consec=1)
print(f"  consim1: {len(vpn_consim1):,} nodes in {time.time() - t0:.1f}s", flush=True)

print("Building valid_pathway_nodes for consim2 (max_consec_sim<=2) …", flush=True)
t0 = time.time()
vpn_consim2 = load_pathway_nodes_sim09_filtered(PATHS_SIM09_FILE, max_consec=2)
print(f"  consim2: {len(vpn_consim2):,} nodes in {time.time() - t0:.1f}s", flush=True)

# Also keep unconstrained (all maturity>=3-endpoint paths in sim0.9 file) for original plots
# maturity>=3 endpoint filter — path gen used ALL_INTERVENTION_IDS
print(
    "Building valid_pathway_nodes for unconstrained (all sim0.9 paths, maturity>=3) …",
    flush=True,
)
t0 = time.time()
vpn_unconstrained: set = set()
with open(PATHS_SIM09_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        path_ids = record["path"]
        interv_id = int(path_ids[-1])
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            vpn_unconstrained.update(int(x) for x in path_ids)
print(
    f"  unconstrained: {len(vpn_unconstrained):,} nodes in {time.time() - t0:.1f}s",
    flush=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_embedding(emb) -> np.ndarray:
    if isinstance(emb, np.ndarray):
        return emb.astype(np.float32)
    if isinstance(emb, str):
        return np.fromstring(emb.strip("<>"), sep=", ").astype(np.float32)
    return np.array(emb, dtype=np.float32)


def build_color_map(cluster_ids: list) -> dict:
    tab20 = plt.cm.tab20.colors
    tab20b = plt.cm.tab20b.colors
    palette = list(tab20) + list(tab20b)
    unique_sorted = sorted(set(cluster_ids))
    return {cid: palette[i % len(palette)] for i, cid in enumerate(unique_sorted)}


EDGE_CONFIG = 0.9
MODE = "unconstrained"
ALGO = "agglomerative"


def collect_filtered_members(
    node_type: str, valid_pathway_nodes: set, maturity_filter: bool = False
):
    """Return (node_ids, cluster_labels) applying valid_pathway_nodes filter.

    For intervention nodes pass maturity_filter=True to additionally require
    intervention_maturity >= 3.
    """
    node_ids = []
    labels = []
    for (ec, mode, nt, algo, cluster_id), members in cluster_memberships.items():
        if ec == EDGE_CONFIG and mode == MODE and nt == node_type and algo == ALGO:
            for nid in members:
                if nid not in valid_pathway_nodes:
                    continue
                if maturity_filter:
                    mat = int(
                        node_attrs.get(nid, {}).get("intervention_maturity", 0) or 0
                    )
                    if mat < 3:
                        continue
                node_ids.append(nid)
                labels.append(cluster_id)
    return node_ids, labels


def build_embedding_matrix(node_ids: list):
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


UMAP_PARAMS = dict(
    n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
)


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


def run_umap_and_plot(valid_pathway_nodes: set, label: str, suffix: str):
    """Collect members, run UMAP, and save plots for one consim config."""
    print(f"\n--- {label} ---", flush=True)

    print("  Collecting risk members …", flush=True)
    risk_ids, risk_labels = collect_filtered_members(
        "risk", valid_pathway_nodes, maturity_filter=False
    )
    print(f"    {len(risk_ids):,} risk nodes", flush=True)

    print("  Collecting intervention members (maturity>=3) …", flush=True)
    interv_ids, interv_labels = collect_filtered_members(
        "intervention", valid_pathway_nodes, maturity_filter=True
    )
    print(f"    {len(interv_ids):,} intervention nodes", flush=True)

    print("  Building risk embedding matrix …", flush=True)
    t0 = time.time()
    risk_matrix, risk_missing = build_embedding_matrix(risk_ids)
    print(
        f"    Shape: {risk_matrix.shape}, missing: {risk_missing}, time: {time.time() - t0:.1f}s",
        flush=True,
    )

    print("  Building intervention embedding matrix …", flush=True)
    t0 = time.time()
    interv_matrix, interv_missing = build_embedding_matrix(interv_ids)
    print(
        f"    Shape: {interv_matrix.shape}, missing: {interv_missing}, time: {time.time() - t0:.1f}s",
        flush=True,
    )

    print("  Running UMAP for risk …", flush=True)
    t0 = time.time()
    risk_2d = umap.UMAP(**UMAP_PARAMS).fit_transform(risk_matrix)
    print(f"    Done in {time.time() - t0:.1f}s", flush=True)

    print("  Running UMAP for interventions …", flush=True)
    t0 = time.time()
    interv_2d = umap.UMAP(**UMAP_PARAMS).fit_transform(interv_matrix)
    print(f"    Done in {time.time() - t0:.1f}s", flush=True)

    make_umap_plot(
        risk_2d,
        risk_labels,
        f"Risk Clusters — UMAP 2D ({label}, n={risk_matrix.shape[0]:,})",
        OUT_DIR / f"umap_risks{suffix}.png",
    )
    make_umap_plot(
        interv_2d,
        interv_labels,
        f"Intervention Clusters — UMAP 2D ({label}, maturity>=3, n={interv_matrix.shape[0]:,})",
        OUT_DIR / f"umap_interventions{suffix}.png",
    )


# ---------------------------------------------------------------------------
# 5. Original unconstrained plots (umap_risks.png / umap_interventions.png)
#    maturity>=3 endpoint filter applied for consistency.
# ---------------------------------------------------------------------------
print("\n=== Original unconstrained plots (maturity>=3 filter) ===", flush=True)
print("  Collecting risk members (unconstrained) …", flush=True)
risk_ids_unc, risk_labels_unc = collect_filtered_members(
    "risk", vpn_unconstrained, maturity_filter=False
)
print(f"    {len(risk_ids_unc):,} risk nodes", flush=True)

print(
    "  Collecting intervention members (unconstrained, maturity>=3) …",
    flush=True,
)
interv_ids_unc, interv_labels_unc = collect_filtered_members(
    "intervention", vpn_unconstrained, maturity_filter=True
)
print(f"    {len(interv_ids_unc):,} intervention nodes", flush=True)

t0 = time.time()
risk_matrix_unc, _ = build_embedding_matrix(risk_ids_unc)
print(f"  Risk matrix {risk_matrix_unc.shape} in {time.time() - t0:.1f}s", flush=True)

t0 = time.time()
interv_matrix_unc, _ = build_embedding_matrix(interv_ids_unc)
print(
    f"  Intervention matrix {interv_matrix_unc.shape} in {time.time() - t0:.1f}s",
    flush=True,
)

print("  Running UMAP for risk (unconstrained) …", flush=True)
t0 = time.time()
risk_2d_unc = umap.UMAP(**UMAP_PARAMS).fit_transform(risk_matrix_unc)
print(f"    Done in {time.time() - t0:.1f}s", flush=True)

print("  Running UMAP for interventions (unconstrained) …", flush=True)
t0 = time.time()
interv_2d_unc = umap.UMAP(**UMAP_PARAMS).fit_transform(interv_matrix_unc)
print(f"    Done in {time.time() - t0:.1f}s", flush=True)

make_umap_plot(
    risk_2d_unc,
    risk_labels_unc,
    f"Risk Clusters — UMAP 2D (valid_pathway_nodes, n={risk_matrix_unc.shape[0]:,})",
    OUT_DIR / "umap_risks.png",
)
make_umap_plot(
    interv_2d_unc,
    interv_labels_unc,
    f"Intervention Clusters — UMAP 2D (valid_pathway_nodes, n={interv_matrix_unc.shape[0]:,})",
    OUT_DIR / "umap_interventions.png",
)

# ---------------------------------------------------------------------------
# 6. Per-consim plots
# ---------------------------------------------------------------------------
run_umap_and_plot(vpn_consim0, "consim0 (edge-only paths)", "_consim0")
run_umap_and_plot(vpn_consim1, "consim1 (max_consec_sim<=1)", "_consim1")
run_umap_and_plot(vpn_consim2, "consim2 (max_consec_sim<=2)", "_consim2")

print("\nDone. All 8 plots written.", flush=True)
