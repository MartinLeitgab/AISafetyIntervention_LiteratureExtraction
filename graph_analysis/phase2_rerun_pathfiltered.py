"""
Re-run Step 3 Sections F, D, D2 with CORRECT node restriction.

Prior re-run (phase2_step3_rerun_betweenness_sectionf.py) was wrong:
it built valid_nodes from PKL (all concept nodes + maturity-filtered interventions)
giving 169,156 nodes. Correct valid_nodes = only nodes appearing in at least one
complete risk→intervention path file, which is ~21,585 nodes.

All path files in phase1_rawpathsfiles/paths_*.jsonl were generated with
load_graph(min_conf=3) so every EDGE edge in every path has conf>=3 and every
intervention endpoint has maturity>=3. The union of nodes across all path files
is the valid-pathway node set.

Naming:
  _unfiltered backups already exist for Section D main files and Section F.
  Section D2 files have no _unfiltered backup yet — backed up here before overwrite.

Run from graph_analysis/:
    python -u phase2_rerun_pathfiltered.py > /tmp/betweenness_pathfiltered.log 2>&1 &
    echo PID: $!
"""

import json
import pickle
import shutil
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

STEP1_DIR = Path("phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")
STEP3_DIR = Path("phase2_results/step3_validation_and_selection")
PATHS_DIR = Path("phase1_rawpathsfiles")
STEP3_DIR.mkdir(exist_ok=True)

SIM_THRESH = 0.90
MIN_CONF = 3


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def backup_if_no_backup_exists(path: Path, suffix: str = "_unfiltered"):
    """Only creates backup if one doesn't already exist — avoids overwriting known-good backups."""
    bak = path.with_name(path.stem + suffix + path.suffix)
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backed up {path.name} → {bak.name}")
    elif bak.exists():
        print(f"  Backup already exists: {bak.name} — not overwritten")


# ─── LOAD PKLs ────────────────────────────────────────────────────────────────

print("=" * 70)
print("Loading PKLs …")
print("=" * 70)

with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
with open(STEP1_DIR / "cluster_memberships.pkl", "rb") as f:
    cluster_memberships = pickle.load(f)

print(f"  node_attrs:          {len(node_attrs):,} nodes")
print(f"  edge_data:           {len(edge_data):,} edges")
print(f"  cluster_memberships: {len(cluster_memberships):,} records")

# ─── BUILD VALID-NODE SET FROM PATH FILES ─────────────────────────────────────

print("\n" + "=" * 70)
print("Building valid-node set from path files …")
print("=" * 70)

valid_nodes: set = set()
path_files = sorted(PATHS_DIR.glob("paths_*.jsonl"))
print(f"  Reading {len(path_files)} path files …")
for pf in path_files:
    n_before = len(valid_nodes)
    with open(pf) as f:
        for line in f:
            rec = json.loads(line)
            path = rec.get("path", [])
            if isinstance(path, str):
                path = json.loads(path)
            valid_nodes.update(str(n) for n in path)
    print(
        f"  {pf.name:55s}: +{len(valid_nodes) - n_before:,} new  cumulative {len(valid_nodes):,}"
    )

print(f"\n  TOTAL valid-pathway nodes: {len(valid_nodes):,}")
print(
    f"  (Total PKL nodes: {len(node_attrs):,} — {len(node_attrs) - len(valid_nodes):,} excluded)"
)

# ─── SECTION F — EDGE Subgraph Stats ─────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION F — EDGE Subgraph (valid-pathway nodes, conf>=3)")
print("=" * 70)

# _unfiltered backups already exist from prior run — don't overwrite them
backup_if_no_backup_exists(STEP3_DIR / "edge_subgraph_stats.csv")
backup_if_no_backup_exists(STEP3_DIR / "edge_degree_distribution.png")

G_edge = nx.DiGraph()
G_edge.add_nodes_from(valid_nodes)

n_kept, n_low_conf, n_invalid_node = 0, 0, 0
for e in edge_data:
    if str(e.get("type", "")).upper() != "EDGE":
        continue
    src, tgt = str(e.get("source", "")), str(e.get("target", ""))
    if src not in valid_nodes or tgt not in valid_nodes:
        n_invalid_node += 1
        continue
    conf = e.get("confidence")
    try:
        if float(conf) < MIN_CONF:
            n_low_conf += 1
            continue
    except (TypeError, ValueError):
        n_low_conf += 1
        continue
    G_edge.add_edge(src, tgt)
    n_kept += 1

print(f"  EDGE edges kept (conf>={MIN_CONF}, both endpoints valid): {n_kept:,}")
print(f"  Dropped — low confidence:    {n_low_conf:,}")
print(f"  Dropped — non-valid node:    {n_invalid_node:,}")
print(
    f"  Graph nodes: {G_edge.number_of_nodes():,}  edges: {G_edge.number_of_edges():,}"
)

wccs = list(nx.weakly_connected_components(G_edge))
wcc_sizes = sorted([len(c) for c in wccs], reverse=True)
print(f"  WCC: {len(wccs):,} components")
print(
    f"  Largest WCC: {wcc_sizes[0]} nodes  2nd: {wcc_sizes[1] if len(wcc_sizes) > 1 else 0}"
)

# Approximate diameter via BFS on largest WCC
largest_wcc_nodes = max(wccs, key=len)
G_sub = G_edge.subgraph(largest_wcc_nodes).to_undirected()
sample_sources = list(largest_wcc_nodes)[: min(50, len(largest_wcc_nodes))]
max_sp = 0
for s in sample_sources:
    lengths = nx.single_source_shortest_path_length(G_sub, s)
    if lengths:
        max_sp = max(max_sp, max(lengths.values()))
approx_diameter = max_sp
print(f"  Approx diameter (BFS sample): {approx_diameter}")

degrees = [d for _, d in G_edge.degree()]
mean_deg = np.mean(degrees) if degrees else 0.0
deg_ge2 = sum(1 for d in degrees if d >= 2) / len(degrees) * 100 if degrees else 0.0
print(f"  Mean degree: {mean_deg:.2f}")
print(f"  Nodes with degree >= 2: {deg_ge2:.1f}%")

# Check Step2b top-25 betweenness nodes (from prior correct analyses)
old_btw_file = STEP3_DIR / "betweenness_sim09_unfiltered.csv"
if old_btw_file.exists():
    df_old = pd.read_csv(old_btw_file)
    old_top25 = (
        set(df_old.head(25)["node_id"].astype(str).tolist())
        if "node_id" in df_old.columns
        else set()
    )
    if old_top25:
        present = sum(1 for n in old_top25 if n in G_edge)
        print(
            f"  Top-25 unfiltered betweenness nodes in valid EDGE subgraph: {present}/25"
        )

stats = {
    "metric": [
        "Nodes (valid-pathway set)",
        "EDGE edges (conf>=3, valid endpoints)",
        "Weakly connected components",
        "Largest WCC",
        "2nd largest WCC",
        "Approx diameter",
        "Mean degree",
        "Nodes with degree >= 2 (%)",
    ],
    "value": [
        G_edge.number_of_nodes(),
        G_edge.number_of_edges(),
        len(wccs),
        wcc_sizes[0],
        wcc_sizes[1] if len(wcc_sizes) > 1 else 0,
        approx_diameter,
        round(mean_deg, 2),
        round(deg_ge2, 1),
    ],
}
pd.DataFrame(stats).to_csv(STEP3_DIR / "edge_subgraph_stats.csv", index=False)
print("  Saved edge_subgraph_stats.csv")

# Degree distribution plot
fig, ax = plt.subplots(figsize=(8, 5))
deg_vals = [d for _, d in G_edge.degree() if d > 0]
bins = np.arange(0, min(max(deg_vals) + 2, 30)) if deg_vals else [0, 1]
ax.hist(deg_vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
ax.set_xlabel("Node degree (EDGE edges, valid-pathway nodes, conf≥3)")
ax.set_ylabel("Count")
ax.set_title(
    f"EDGE Subgraph Degree Distribution\n"
    f"(valid-pathway nodes only, conf≥3 | {len(wccs):,} WCCs, "
    f"mean degree {mean_deg:.2f})"
)
plt.tight_layout()
plt.savefig(STEP3_DIR / "edge_degree_distribution.png", dpi=120)
plt.close()
print("  Saved edge_degree_distribution.png")

# ─── HEARTBEAT THREAD ─────────────────────────────────────────────────────────

_stop_heartbeat = threading.Event()


def _heartbeat(label, interval=600):
    t0 = time.time()
    while not _stop_heartbeat.is_set():
        time.sleep(interval)
        if not _stop_heartbeat.is_set():
            elapsed = (time.time() - t0) / 3600
            print(f"  [heartbeat] {label} running — {elapsed:.1f}h elapsed", flush=True)


# ─── SECTION D — Full-Graph Betweenness ───────────────────────────────────────

print("\n" + "=" * 70)
print(
    "SECTION D — Full-graph betweenness (valid-pathway nodes, SIM>=0.9 + conf>=3 EDGE)"
)
print("=" * 70)

# _unfiltered backups already exist — don't overwrite
for fname in [
    "betweenness_sim09.csv",
    "betweenness_bridge_clusters.csv",
    "betweenness_comparison.png",
]:
    backup_if_no_backup_exists(STEP3_DIR / fname)

print("  Building graph on valid-pathway nodes …")
G = nx.DiGraph()
G.add_nodes_from(valid_nodes)

n_sim, n_edge_kept, n_edge_conf, n_edge_node = 0, 0, 0, 0
for e in edge_data:
    src, tgt = str(e.get("source", "")), str(e.get("target", ""))
    if src not in valid_nodes or tgt not in valid_nodes:
        continue
    etype = str(e.get("type", "")).upper()
    if etype == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= SIM_THRESH:
            G.add_edge(src, tgt)
            n_sim += 1
    elif etype == "EDGE":
        conf = e.get("confidence")
        try:
            if float(conf) >= MIN_CONF:
                G.add_edge(src, tgt)
                n_edge_kept += 1
            else:
                n_edge_conf += 1
        except (TypeError, ValueError):
            n_edge_conf += 1

# Remove isolates (no edges)
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

print(f"  SIM>=0.9 edges added:      {n_sim:,}")
print(f"  EDGE conf>=3 edges added:  {n_edge_kept:,}")
print(f"  EDGE conf<3 skipped:       {n_edge_conf:,}")
print(f"  Nodes after removing isolates: {G.number_of_nodes():,}")
print(f"  Total edges: {G.number_of_edges():,}")
print(f"\n  Computing EXACT betweenness ({G.number_of_nodes():,} nodes) …", flush=True)

hb = threading.Thread(target=_heartbeat, args=("Section D betweenness",), daemon=True)
hb.start()
t0 = time.time()

betweenness = nx.betweenness_centrality(G, normalized=True)

_stop_heartbeat.set()
elapsed_h = (time.time() - t0) / 3600
print(f"  Betweenness complete in {elapsed_h:.2f}h")
_stop_heartbeat.clear()

# Save checkpoint
ckpt = STEP3_DIR / "betweenness_raw_checkpoint_pathfiltered.pkl"
with open(ckpt, "wb") as f:
    pickle.dump(betweenness, f)
print(f"  Checkpoint saved: {ckpt.name}")

# Top-100 with node attributes
rows = []
node_attrs_str = {str(k): v for k, v in node_attrs.items()}
for nid, bval in sorted(betweenness.items(), key=lambda x: -x[1])[:100]:
    attrs = node_attrs_str.get(str(nid), {})
    rows.append(
        {
            "node_id": nid,
            "name": attrs.get("name", "")[:80],
            "category": attrs.get("concept_category") or attrs.get("type", ""),
            "betweenness": round(bval, 6),
            "url": attrs.get("url", "")[:80],
        }
    )
df_btw = pd.DataFrame(rows)
df_btw.to_csv(STEP3_DIR / "betweenness_sim09.csv", index=False)
print(f"  Saved betweenness_sim09.csv ({len(df_btw)} rows)")

# Compare to unfiltered top-100
old_csv = STEP3_DIR / "betweenness_sim09_unfiltered.csv"
if old_csv.exists():
    df_old = pd.read_csv(old_csv)
    old_top = (
        set(df_old.head(100)["node_id"].astype(str).tolist())
        if "node_id" in df_old.columns
        else set()
    )
    new_top = set(df_btw.head(100)["node_id"].astype(str).tolist())
    overlap = len(old_top & new_top)
    print(f"  Top-100 overlap with unfiltered run: {overlap}/100")

# Bridge clusters (k=12 Agglomerative on top-50)
print("  Computing bridge cluster assignments (k=12) …")
top50_ids = df_btw.head(50)["node_id"].astype(str).tolist()
emb_matrix, valid_ids = [], []
for nid in top50_ids:
    attrs = node_attrs_str.get(nid, {})
    emb = attrs.get("embedding")
    if emb is None:
        continue
    if isinstance(emb, str):
        try:
            emb = np.array(
                [float(x) for x in emb.strip("<>").split(",")], dtype=np.float32
            )
        except Exception:
            continue
    emb = np.array(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb_matrix.append(emb / norm)
        valid_ids.append(nid)

cluster_labels = [-1] * len(valid_ids)
if len(valid_ids) >= 12:
    clustering = AgglomerativeClustering(
        n_clusters=12, metric="cosine", linkage="average"
    )
    cluster_labels = clustering.fit_predict(np.array(emb_matrix))

cluster_rows = []
for nid, clabel in zip(valid_ids, cluster_labels):
    attrs = node_attrs_str.get(nid, {})
    cluster_rows.append(
        {
            "node_id": nid,
            "cluster_id": int(clabel),
            "name": attrs.get("name", "")[:80],
            "category": attrs.get("concept_category") or attrs.get("type", ""),
            "betweenness": round(betweenness.get(nid, 0), 6),
        }
    )
df_clusters = pd.DataFrame(cluster_rows).sort_values(
    ["cluster_id", "betweenness"], ascending=[True, False]
)
df_clusters.to_csv(STEP3_DIR / "betweenness_bridge_clusters.csv", index=False)
print("  Saved betweenness_bridge_clusters.csv")

# Comparison plot: new vs old betweenness ranks
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top20_new = df_btw.head(20)
axes[0].barh(
    range(len(top20_new)), top20_new["betweenness"].values[::-1], color="steelblue"
)
axes[0].set_yticks(range(len(top20_new)))
axes[0].set_yticklabels([n[:35] for n in top20_new["name"].values[::-1]], fontsize=7)
axes[0].set_title("Top-20 Bridge Nodes\n(path-file restricted, valid-pathway only)")
axes[0].set_xlabel("Betweenness centrality")

if old_csv.exists():
    df_old_20 = pd.read_csv(old_csv).head(20)
    if "betweenness" in df_old_20.columns and "name" in df_old_20.columns:
        axes[1].barh(
            range(len(df_old_20)), df_old_20["betweenness"].values[::-1], color="coral"
        )
        axes[1].set_yticks(range(len(df_old_20)))
        axes[1].set_yticklabels(
            [n[:35] for n in df_old_20["name"].values[::-1]], fontsize=7
        )
        axes[1].set_title("Top-20 Bridge Nodes\n(unfiltered — all PKL nodes)")
        axes[1].set_xlabel("Betweenness centrality")
    else:
        axes[1].set_visible(False)
else:
    axes[1].set_visible(False)

plt.suptitle("Betweenness Comparison: Path-file restricted vs Unfiltered", fontsize=11)
plt.tight_layout()
plt.savefig(STEP3_DIR / "betweenness_comparison.png", dpi=120)
plt.close()
print("  Saved betweenness_comparison.png")

# ─── SECTION D2 — Both-Mode Betweenness ───────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION D2 — Both-mode betweenness (both-mode cluster nodes)")
print("=" * 70)

# Backup both09 files BEFORE overwriting (no _unfiltered backup exists yet)
for fname in [
    "betweenness_both09.csv",
    "betweenness_both09_bridge_clusters.csv",
    "betweenness_both09_comparison.png",
    "betweenness_both09_raw_checkpoint.pkl",
]:
    backup_if_no_backup_exists(STEP3_DIR / fname)

# Get both-mode ec=0.9 agglomerative cluster nodes
both_nodes: set = set()
for key, members in cluster_memberships.items():
    if (
        len(key) == 5
        and str(key[0]) == "0.9"
        and key[1] == "both"
        and key[3] == "agglomerative"
    ):
        both_nodes.update(str(m) for m in members)

# Intersect with valid_nodes (should already be a subset, but be explicit)
both_nodes = both_nodes & valid_nodes
print(
    f"  Both-mode cluster nodes: {len(both_nodes):,} (intersection with valid-pathway: {len(both_nodes):,})"
)

print("  Building both-mode subgraph …")
G2 = nx.DiGraph()
G2.add_nodes_from(both_nodes)

n_sim2, n_edge2 = 0, 0
for e in edge_data:
    src, tgt = str(e.get("source", "")), str(e.get("target", ""))
    if src not in both_nodes or tgt not in both_nodes:
        continue
    etype = str(e.get("type", "")).upper()
    if etype == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= SIM_THRESH:
            G2.add_edge(src, tgt)
            n_sim2 += 1
    elif etype == "EDGE":
        conf = e.get("confidence")
        try:
            if float(conf) >= MIN_CONF:
                G2.add_edge(src, tgt)
                n_edge2 += 1
        except (TypeError, ValueError):
            pass

isolates2 = list(nx.isolates(G2))
G2.remove_nodes_from(isolates2)

print(f"  SIM>=0.9 edges: {n_sim2:,}  EDGE conf>=3: {n_edge2:,}")
print(
    f"  Both-mode graph: {G2.number_of_nodes():,} nodes, {G2.number_of_edges():,} edges"
)
print(f"\n  Computing EXACT betweenness ({G2.number_of_nodes():,} nodes) …", flush=True)

_stop_heartbeat.clear()
hb2 = threading.Thread(target=_heartbeat, args=("Section D2 betweenness",), daemon=True)
hb2.start()
t1 = time.time()

betweenness2 = nx.betweenness_centrality(G2, normalized=True)

_stop_heartbeat.set()
elapsed2 = (time.time() - t1) / 3600
print(f"  Both-mode betweenness complete in {elapsed2:.2f}h")
_stop_heartbeat.clear()

ckpt2 = STEP3_DIR / "betweenness_both09_raw_checkpoint_pathfiltered.pkl"
with open(ckpt2, "wb") as f:
    pickle.dump(betweenness2, f)
print(f"  Checkpoint saved: {ckpt2.name}")

# Top-100 with attributes
rows2 = []
for nid, bval in sorted(betweenness2.items(), key=lambda x: -x[1])[:100]:
    attrs = node_attrs_str.get(str(nid), {})
    rows2.append(
        {
            "node_id": nid,
            "name": attrs.get("name", "")[:80],
            "category": attrs.get("concept_category") or attrs.get("type", ""),
            "betweenness_both09": round(bval, 6),
            "url": attrs.get("url", "")[:80],
        }
    )
df_btw2 = pd.DataFrame(rows2)
df_btw2.to_csv(STEP3_DIR / "betweenness_both09.csv", index=False)
print("  Saved betweenness_both09.csv")

# Bridge clusters
top50_ids2 = df_btw2.head(50)["node_id"].astype(str).tolist()
emb_matrix2, valid_ids2 = [], []
for nid in top50_ids2:
    attrs = node_attrs_str.get(nid, {})
    emb = attrs.get("embedding")
    if emb is None:
        continue
    if isinstance(emb, str):
        try:
            emb = np.array(
                [float(x) for x in emb.strip("<>").split(",")], dtype=np.float32
            )
        except Exception:
            continue
    emb = np.array(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb_matrix2.append(emb / norm)
        valid_ids2.append(nid)

cluster_labels2 = [-1] * len(valid_ids2)
if len(valid_ids2) >= 12:
    clustering2 = AgglomerativeClustering(
        n_clusters=12, metric="cosine", linkage="average"
    )
    cluster_labels2 = clustering2.fit_predict(np.array(emb_matrix2))

cluster_rows2 = []
for nid, clabel in zip(valid_ids2, cluster_labels2):
    attrs = node_attrs_str.get(nid, {})
    cluster_rows2.append(
        {
            "node_id": nid,
            "cluster_id": int(clabel),
            "name": attrs.get("name", "")[:80],
            "category": attrs.get("concept_category") or attrs.get("type", ""),
            "betweenness_both09": round(betweenness2.get(nid, 0), 6),
        }
    )
df_clusters2 = pd.DataFrame(cluster_rows2).sort_values(
    ["cluster_id", "betweenness_both09"], ascending=[True, False]
)
df_clusters2.to_csv(STEP3_DIR / "betweenness_both09_bridge_clusters.csv", index=False)
print("  Saved betweenness_both09_bridge_clusters.csv")

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top20_2 = df_btw2.head(20)
axes[0].barh(
    range(len(top20_2)), top20_2["betweenness_both09"].values[::-1], color="steelblue"
)
axes[0].set_yticks(range(len(top20_2)))
axes[0].set_yticklabels([n[:35] for n in top20_2["name"].values[::-1]], fontsize=7)
axes[0].set_title("Top-20 Both-Mode Bridge Nodes\n(path-file restricted)")
axes[0].set_xlabel("Betweenness centrality")

old_csv2 = STEP3_DIR / "betweenness_both09_unfiltered.csv"
if old_csv2.exists():
    df_old2 = pd.read_csv(old_csv2).head(20)
    if "betweenness_both09" in df_old2.columns and "name" in df_old2.columns:
        axes[1].barh(
            range(len(df_old2)),
            df_old2["betweenness_both09"].values[::-1],
            color="coral",
        )
        axes[1].set_yticks(range(len(df_old2)))
        axes[1].set_yticklabels(
            [n[:35] for n in df_old2["name"].values[::-1]], fontsize=7
        )
        axes[1].set_title("Top-20 Both-Mode Bridge Nodes\n(unfiltered — original run)")
        axes[1].set_xlabel("Betweenness centrality")
    else:
        axes[1].set_visible(False)
else:
    axes[1].set_visible(False)

plt.suptitle(
    "Both-Mode Betweenness Comparison: Path-file restricted vs Unfiltered", fontsize=11
)
plt.tight_layout()
plt.savefig(STEP3_DIR / "betweenness_both09_comparison.png", dpi=120)
plt.close()
print("  Saved betweenness_both09_comparison.png")

print("\n" + "=" * 70)
print("ALL SECTIONS COMPLETE")
print("=" * 70)
print("  Section F:  edge_subgraph_stats.csv, edge_degree_distribution.png")
print(
    "  Section D:  betweenness_sim09.csv, betweenness_bridge_clusters.csv, betweenness_comparison.png"
)
print(
    "  Section D2: betweenness_both09.csv, betweenness_both09_bridge_clusters.csv, betweenness_both09_comparison.png"
)
