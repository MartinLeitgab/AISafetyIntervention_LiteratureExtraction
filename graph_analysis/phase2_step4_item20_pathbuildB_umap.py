"""
Item 20: PathbuildB body-node UMAP (consim1).

Collects all interior nodes from consim1 qualifying paths, assigns each
to its dominant pathbuildB meta-family (highest path-count frequency),
and produces a UMAP 2D projection colored by meta-family.

Top-20 meta-families are labeled in the legend; all other nodes are grey.

Output:
  step4_finalanalysis/umap_pathbuildB_body_nodes_consim1.png
"""

import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap as umap_lib

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
PROJECT_ROOT = BASE.parent
STEP1_DIR = (
    PROJECT_ROOT
    / "graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
)
PATHS_FILE = (
    PROJECT_ROOT
    / "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
)
TABLES = BASE / "phase2_results/step4_finalanalysis/step4_cluster_tables"
NAMING = BASE / "phase2_results/step5_naming"
OUT_DIR = BASE / "phase2_results/step4_finalanalysis"

EDGE_CONFIG = 0.9
MODE = "unconstrained"
ALGO = "agglomerative"
MAX_NODES = 30_000  # subsample cap for UMAP speed

SUBTYPE_PREFIXES = {
    "problem_analysis": "pr",
    "theoretical_insight": "th",
    "design_rationale": "de",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
}
BODY_NODE_TYPES = set(SUBTYPE_PREFIXES.keys())

# ── 1. Load PKL checkpoints ────────────────────────────────────────────────
print("Loading graph_node_attributes.pkl …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs: dict = pickle.load(f)
print(f"  {len(node_attrs):,} nodes  ({time.time() - t0:.1f}s)", flush=True)

print("Loading cluster_memberships.pkl …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "cluster_memberships.pkl", "rb") as f:
    cluster_memberships: dict = pickle.load(f)
print(f"  {len(cluster_memberships):,} records  ({time.time() - t0:.1f}s)", flush=True)

print("Loading graph_edge_data.pkl …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data: list = pickle.load(f)
print(f"  {len(edge_data):,} edges  ({time.time() - t0:.1f}s)", flush=True)

# ── 2. Build node → component string lookup for body node types ────────────
print("Building node→component map …", flush=True)
node_to_comp: dict[int, str] = {}
for (ec, mode, nt, algo, cid), members in cluster_memberships.items():
    if ec == EDGE_CONFIG and mode == MODE and algo == ALGO and nt in BODY_NODE_TYPES:
        comp = f"{SUBTYPE_PREFIXES[nt]}:{cid}"
        for nid in members:
            node_to_comp[nid] = comp
print(f"  {len(node_to_comp):,} body nodes mapped", flush=True)


# ── 3. Build sim_edge_set for consim1 filtering ────────────────────────────
def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


print("Building unconstrained VPN for sim_edge_set restriction …", flush=True)
t0 = time.time()
vpn_for_sim: set = set()
with open(PATHS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        path_ids = rec["path"]
        if (
            int(
                node_attrs.get(int(path_ids[-1]), {}).get("intervention_maturity", 0)
                or 0
            )
            >= 3
        ):
            vpn_for_sim.update(int(x) for x in path_ids)
print(f"  {len(vpn_for_sim):,} nodes  ({time.time() - t0:.1f}s)", flush=True)

print("Building sim_edge_set (cos_sim>=0.9, VPN-restricted) …", flush=True)
t0 = time.time()
sim_edge_set: set = set()
for e in edge_data:
    if str(e.get("type", "")).upper() == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= 0.9:
            try:
                s2, t2 = int(e["source"]), int(e["target"])
                if s2 in vpn_for_sim and t2 in vpn_for_sim:
                    sim_edge_set.add((min(s2, t2), max(s2, t2)))
            except (ValueError, TypeError):
                pass
print(f"  {len(sim_edge_set):,} sim edges  ({time.time() - t0:.1f}s)", flush=True)
del edge_data, vpn_for_sim


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


# ── 4. Load B-family meta assignments ─────────────────────────────────────
fam_df = pd.read_csv(TABLES / "pathbuildB_metafamilies_consim1.csv")
sig_to_meta: dict[str, int] = dict(
    zip(fam_df["signature_str"], fam_df["meta_family_id"].astype(int))
)

mf_summary = pd.read_csv(TABLES / "pathbuildB_metafamily_summary_consim1.csv")
mf_names = dict(zip(mf_summary["meta_family_id"], mf_summary["dominant_family_name"]))
mf_paths_total = dict(zip(mf_summary["meta_family_id"], mf_summary["n_paths_total"]))

# ── 5. Scan consim1 paths → map body nodes to meta-families ───────────────
print("Scanning consim1 paths to assign body nodes to meta-families …", flush=True)
t0 = time.time()
node_mf_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
n_scanned = n_matched = 0

with open(PATHS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        path_ids = rec["path"]
        interv_id = int(path_ids[-1])
        # maturity filter
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) < 3:
            continue
        # consim1 filter
        if max_consec_sim(path_ids, sim_edge_set) > 1:
            continue
        n_scanned += 1

        body_nids = [int(x) for x in path_ids[1:-1]]
        if not body_nids:
            continue

        # Build frozenset signature (same format as families CSV)
        comps = sorted({node_to_comp[nid] for nid in body_nids if nid in node_to_comp})
        sig = " & ".join(comps)
        meta_fam_id = sig_to_meta.get(sig)
        if meta_fam_id is None:
            continue
        n_matched += 1

        for nid in body_nids:
            if nid in node_to_comp:
                node_mf_counts[nid][meta_fam_id] += 1

print(
    f"  Scanned {n_scanned:,} consim1 paths, matched {n_matched:,}  ({time.time() - t0:.1f}s)",
    flush=True,
)
print(f"  {len(node_mf_counts):,} body nodes with meta-family assignments", flush=True)

# Assign each node to dominant meta-family
node_dominant_mf = {
    nid: max(counts, key=counts.get) for nid, counts in node_mf_counts.items()
}


# ── 6. Filter to nodes with valid embeddings ───────────────────────────────
def parse_embedding(emb) -> np.ndarray:
    if isinstance(emb, np.ndarray):
        return emb.astype(np.float32)
    if isinstance(emb, str):
        return np.fromstring(emb.strip("<>"), sep=", ").astype(np.float32)
    return np.array(emb, dtype=np.float32)


print("Filtering to nodes with valid embeddings …", flush=True)
valid_nids, valid_labels = [], []
for nid, mf in node_dominant_mf.items():
    attrs = node_attrs.get(nid)
    if attrs is not None and attrs.get("embedding") is not None:
        valid_nids.append(nid)
        valid_labels.append(mf)
print(f"  {len(valid_nids):,} nodes with embeddings", flush=True)

# Subsample if needed
if len(valid_nids) > MAX_NODES:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(valid_nids), MAX_NODES, replace=False)
    valid_nids = [valid_nids[i] for i in idx]
    valid_labels = [valid_labels[i] for i in idx]
    print(f"  Subsampled to {MAX_NODES:,}", flush=True)

# ── 7. Build embedding matrix ──────────────────────────────────────────────
print("Building embedding matrix …", flush=True)
t0 = time.time()
matrix = np.array(
    [parse_embedding(node_attrs[nid]["embedding"]) for nid in valid_nids],
    dtype=np.float32,
)
print(f"  Shape: {matrix.shape}  ({time.time() - t0:.1f}s)", flush=True)

# ── 8. UMAP ────────────────────────────────────────────────────────────────
print("Running UMAP …", flush=True)
t0 = time.time()
coords_2d = umap_lib.UMAP(
    n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
).fit_transform(matrix)
print(f"  Done in {time.time() - t0:.1f}s", flush=True)

# ── 9. Plot ────────────────────────────────────────────────────────────────
print("Plotting …", flush=True)

# Top-20 meta-families by total path count
top20_mf = sorted(mf_paths_total, key=mf_paths_total.get, reverse=True)[:20]
tab20 = plt.cm.tab20.colors
tab20b = plt.cm.tab20b.colors
palette = list(tab20) + list(tab20b)
top20_color_map = {mf: palette[i % len(palette)] for i, mf in enumerate(top20_mf)}
other_color = (0.72, 0.72, 0.72)

fig, ax = plt.subplots(figsize=(16, 12))

# Plot "other" nodes first (background)
other_idx = [i for i, lb in enumerate(valid_labels) if lb not in top20_color_map]
if other_idx:
    ax.scatter(
        coords_2d[other_idx, 0],
        coords_2d[other_idx, 1],
        c=[other_color] * len(other_idx),
        alpha=0.25,
        s=5,
        linewidths=0,
        rasterized=True,
    )

# Plot top-20 meta-families
handles = []
for mf in top20_mf:
    mf_idx = [i for i, lb in enumerate(valid_labels) if lb == mf]
    if not mf_idx:
        continue
    c = top20_color_map[mf]
    short = str(mf_names.get(mf, f"MF{mf}"))[:55]
    ax.scatter(
        coords_2d[mf_idx, 0],
        coords_2d[mf_idx, 1],
        c=[c] * len(mf_idx),
        alpha=0.65,
        s=8,
        linewidths=0,
        rasterized=True,
    )
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=7,
            label=f"MF{mf}: {short}",
        )
    )

other_handle = plt.Line2D(
    [0],
    [0],
    marker="o",
    color="w",
    markerfacecolor=other_color,
    markersize=7,
    label="Other meta-families",
)
ax.legend(
    handles=handles + [other_handle],
    title="PathbuildB meta-family",
    title_fontsize=8,
    fontsize=6.5,
    loc="upper right",
    ncol=1,
    framealpha=0.8,
)

n_total = len(valid_nids)
n_other = len(other_idx)
ax.set_title(
    f"PathbuildB Body Nodes — UMAP 2D (consim1)\n"
    f"{n_total:,} nodes colored by dominant meta-family "
    f"(top-20 of 32 shown; {n_other:,} grey = other)",
    fontsize=12,
)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

plt.tight_layout()
out_path = OUT_DIR / "umap_pathbuildB_body_nodes_consim1.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}", flush=True)
print("Done — Item 20 PathbuildB UMAP complete.", flush=True)
