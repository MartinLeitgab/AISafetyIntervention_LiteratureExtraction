"""
Re-run Step 3 Section F (EDGE subgraph stats) and Section D betweenness
(unconstrained full-graph + both-mode subgraph) with conf>=3 and
intervention_maturity>=3 filters applied consistently.

Overwrites the affected output files in step3_validation_and_selection/.
Old files are backed up with _unfiltered suffix before overwriting.

Run from graph_analysis/:
    uv run phase2_step3_rerun_betweenness_sectionf.py
"""

import pickle
import sys
import threading
import shutil
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

STEP1_DIR = Path("phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")
STEP2_DIR = Path("phase2_results/step2_metrics_and_stability")
STEP3_DIR = Path("phase2_results/step3_validation_and_selection")
STEP3_DIR.mkdir(exist_ok=True)

MIN_CONF = 3
MIN_MATURITY = 3
SIM_THRESH = 0.90


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def backup(path: Path):
    bak = path.with_name(path.stem + "_unfiltered" + path.suffix)
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backed up {path.name} → {bak.name}")


# ─── LOAD ────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Loading PKLs …")
print("=" * 70)

with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
with open(STEP1_DIR / "cluster_memberships.pkl", "rb") as f:
    cluster_memberships = pickle.load(f)

print(f"  node_attrs: {len(node_attrs):,} nodes")
print(f"  edge_data:  {len(edge_data):,} edges")
print(f"  cluster_memberships: {len(cluster_memberships):,} records")

# Node attrs keyed by both str and int for fast lookup
node_attrs_str = {str(k): v for k, v in node_attrs.items()}

# ─── BUILD VALID-NODE SET ─────────────────────────────────────────────────────
# Intervention nodes: only maturity >= 3
# All other nodes: unconditionally valid

print("\nBuilding valid-node set (intervention_maturity>=3 for interventions) …")
valid_nodes = set()
n_int_total, n_int_kept = 0, 0
for nid, attrs in node_attrs.items():
    nid_str = str(nid)
    if attrs.get("type") == "intervention":
        n_int_total += 1
        mat = attrs.get("intervention_maturity", 0)
        try:
            if float(mat) >= MIN_MATURITY:
                valid_nodes.add(nid_str)
                n_int_kept += 1
        except (TypeError, ValueError):
            pass
    else:
        valid_nodes.add(nid_str)

print(
    f"  Intervention nodes: {n_int_kept:,}/{n_int_total:,} kept (maturity>={MIN_MATURITY})"
)
print(f"  Total valid nodes: {len(valid_nodes):,}/{len(node_attrs):,}")


# ─── HELPER: GET CATEGORY ────────────────────────────────────────────────────


def get_category(nid):
    attrs = node_attrs_str.get(str(nid), {})
    return attrs.get("concept_category") or attrs.get("type", "")


# ─── SECTION F — EDGE SUBGRAPH STATS ─────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION F — EDGE Subgraph Stats (conf>=3, maturity>=3)")
print("=" * 70)

for fname in ["edge_subgraph_stats.csv", "edge_degree_distribution.png"]:
    backup(STEP3_DIR / fname)

G_edge = nx.DiGraph()
n_conf_kept, n_conf_skipped, n_node_filtered = 0, 0, 0
for e in edge_data:
    if str(e.get("type", "")).upper() != "EDGE":
        continue
    conf = e.get("confidence")
    try:
        if float(conf) < MIN_CONF:
            n_conf_skipped += 1
            continue
    except (TypeError, ValueError):
        n_conf_skipped += 1
        continue
    src, tgt = str(e.get("source", "")), str(e.get("target", ""))
    if not src or not tgt:
        continue
    if src not in valid_nodes or tgt not in valid_nodes:
        n_node_filtered += 1
        continue
    G_edge.add_edge(src, tgt)
    n_conf_kept += 1

print(f"  EDGE edges kept (conf>={MIN_CONF}): {n_conf_kept:,}")
print(f"  EDGE edges dropped (low conf): {n_conf_skipped:,}")
print(f"  EDGE edges dropped (low-maturity node): {n_node_filtered:,}")
print(
    f"  Graph: {G_edge.number_of_nodes():,} nodes, {G_edge.number_of_edges():,} edges"
)

# WCC
wccs = list(nx.weakly_connected_components(G_edge))
wccs_sorted = sorted(wccs, key=len, reverse=True)
largest_wcc = len(wccs_sorted[0]) if wccs_sorted else 0
largest_wcc_pct = (
    largest_wcc / G_edge.number_of_nodes() if G_edge.number_of_nodes() else 0
)
print(
    f"  WCC: {len(wccs):,} components, largest={largest_wcc:,} ({largest_wcc_pct:.1%})"
)

# Approximate diameter on largest WCC
try:
    import random

    largest_subgraph = G_edge.subgraph(wccs_sorted[0]).copy()
    sample_nodes = random.sample(
        list(largest_subgraph.nodes()), min(200, largest_subgraph.number_of_nodes())
    )
    max_dist = 0
    for n in sample_nodes:
        lengths = nx.single_source_shortest_path_length(largest_subgraph, n, cutoff=20)
        if lengths:
            max_dist = max(max_dist, max(lengths.values()))
    approx_diameter = max_dist
except Exception as ex:
    approx_diameter = f"error: {ex}"
print(f"  Approx diameter (BFS sample): {approx_diameter}")

# Degree distribution
degrees = [d for _, d in G_edge.degree()]
mean_deg = np.mean(degrees) if degrees else 0
print(f"  Mean degree: {mean_deg:.2f}")

# Top-25 betweenness nodes overlap
old_btw_top25 = set()
old_btw_file = STEP2_DIR / "mechanism_transfer_betweenness.csv"
if old_btw_file.exists():
    df_old_btw = pd.read_csv(old_btw_file)
    id_col = "node_id" if "node_id" in df_old_btw.columns else df_old_btw.columns[0]
    old_btw_top25 = set(str(x) for x in df_old_btw.head(25)[id_col].tolist())
    overlap_count = sum(1 for n in old_btw_top25 if n in G_edge.nodes())
    print(
        f"  Top-25 old betweenness nodes present in filtered EDGE subgraph: {overlap_count}/25"
    )

stats = dict(
    n_nodes=G_edge.number_of_nodes(),
    n_edges=G_edge.number_of_edges(),
    n_wcc=len(wccs),
    largest_wcc_nodes=largest_wcc,
    largest_wcc_pct=round(largest_wcc_pct, 4),
    approx_diameter=approx_diameter,
    mean_degree=round(mean_deg, 3),
    filter_conf_min=MIN_CONF,
    filter_maturity_min=MIN_MATURITY,
)
pd.DataFrame([stats]).to_csv(STEP3_DIR / "edge_subgraph_stats.csv", index=False)
print("  Saved edge_subgraph_stats.csv")

# Degree distribution plot
fig, ax = plt.subplots(figsize=(8, 5))
deg_counter = defaultdict(int)
for d in degrees:
    deg_counter[d] += 1
xs = sorted(deg_counter.keys())
ys = [deg_counter[x] for x in xs]
ax.loglog(xs, ys, "o", markersize=3, alpha=0.6)
ax.set_xlabel("Degree")
ax.set_ylabel("Count")
ax.set_title(
    f"EDGE subgraph degree distribution\n(conf>={MIN_CONF}, maturity>={MIN_MATURITY})"
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(STEP3_DIR / "edge_degree_distribution.png", dpi=150)
plt.close(fig)
print("  Saved edge_degree_distribution.png")
del G_edge


# ─── HELPER: BUILD SIM0.9+EDGE FILTERED GRAPH ────────────────────────────────


def build_sim09_edge_graph(node_set=None):
    """
    Build undirected SIM>=0.9 + EDGE (conf>=3, maturity>=3) graph.
    If node_set is provided, restrict to edges where BOTH endpoints are in node_set.
    """
    G = nx.Graph()
    n_sim, n_edge_kept, n_edge_low_conf, n_edge_low_mat = 0, 0, 0, 0
    for e in edge_data:
        src, tgt = str(e.get("source", "")), str(e.get("target", ""))
        if not src or not tgt:
            continue
        # Node validity filter
        if src not in valid_nodes or tgt not in valid_nodes:
            n_edge_low_mat += 1
            continue
        # Optional subset filter
        if node_set is not None and (src not in node_set or tgt not in node_set):
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
                if float(conf) < MIN_CONF:
                    n_edge_low_conf += 1
                    continue
            except (TypeError, ValueError):
                n_edge_low_conf += 1
                continue
            G.add_edge(src, tgt)
            n_edge_kept += 1
    print(
        f"  SIM09={n_sim:,}  EDGE_kept={n_edge_kept:,}  "
        f"EDGE_low_conf={n_edge_low_conf:,}  low_mat_dropped={n_edge_low_mat:,}"
    )
    return G


# ─── HELPER: TOP-50 BRIDGE CLUSTERING ────────────────────────────────────────


def cluster_bridges(top100_rows, node_attrs_str, n_clusters=12):
    top50_ids = [r["node_id"] for r in top100_rows[:50]]
    embeddings, valid_ids = [], []
    for nid in top50_ids:
        attrs = node_attrs_str.get(str(nid), {})
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
            embeddings.append(emb / norm)
            valid_ids.append(nid)
    bridge_rows = []
    if len(embeddings) >= 10:
        X = np.stack(embeddings)
        labels = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(embeddings))
        ).fit_predict(X)
        for nid, cid in zip(valid_ids, labels):
            attrs = node_attrs_str.get(str(nid), {})
            bridge_rows.append(
                dict(
                    node_id=nid,
                    name=attrs.get("name", ""),
                    category=get_category(nid),
                    cluster_id=int(cid),
                )
            )
    return pd.DataFrame(bridge_rows)


# ─── SECTION D — FULL-GRAPH BETWEENNESS (UNCONSTRAINED) ──────────────────────

print("\n" + "=" * 70)
print("SECTION D — Full-graph betweenness (SIM>=0.9 + EDGE conf>=3, maturity>=3)")
print("=" * 70)

for fname in [
    "betweenness_sim09.csv",
    "betweenness_bridge_clusters.csv",
    "betweenness_comparison.png",
]:
    backup(STEP3_DIR / fname)

print("  Building filtered graph …")
G_full = build_sim09_edge_graph(node_set=None)
print(
    f"  Full graph: {G_full.number_of_nodes():,} nodes, {G_full.number_of_edges():,} edges"
)
sys.stdout.flush()

_done_full = threading.Event()


def _hb_full():
    elapsed = 0
    while not _done_full.wait(timeout=1800):
        elapsed += 1800
        print(
            f"  [heartbeat] full-graph betweenness running — "
            f"{elapsed // 3600}h {(elapsed % 3600) // 60}m elapsed",
            flush=True,
        )


threading.Thread(target=_hb_full, daemon=True).start()
print(
    f"  Computing EXACT betweenness ({G_full.number_of_nodes():,} nodes) …", flush=True
)

btw_full = nx.betweenness_centrality(G_full, normalized=True)
_done_full.set()

# Checkpoint
ckpt_full = STEP3_DIR / "betweenness_raw_checkpoint_filtered.pkl"
with open(ckpt_full, "wb") as f:
    pickle.dump(btw_full, f, protocol=4)
print(f"  Checkpoint: {ckpt_full} ({len(btw_full):,} nodes)")

# Top-100 CSV
old_rank = {}
if old_btw_file.exists():
    df_old = pd.read_csv(old_btw_file)
    id_col = "node_id" if "node_id" in df_old.columns else df_old.columns[0]
    for rank, (_, row) in enumerate(df_old.iterrows(), 1):
        old_rank[str(row[id_col])] = rank

top100_full = sorted(btw_full.items(), key=lambda x: x[1], reverse=True)[:100]
rows_full = []
for rank, (nid, btw) in enumerate(top100_full, 1):
    attrs = node_attrs_str.get(str(nid), {})
    rows_full.append(
        dict(
            node_id=nid,
            name=attrs.get("name", ""),
            category=get_category(nid),
            betweenness_sim09=btw,
            rank_sim09=rank,
            rank_sim08=old_rank.get(str(nid), -1),
        )
    )

df_btw_full = pd.DataFrame(rows_full)
df_btw_full.to_csv(STEP3_DIR / "betweenness_sim09.csv", index=False)
print(f"  Saved betweenness_sim09.csv (top {len(df_btw_full)})")

df_bridge_full = cluster_bridges(rows_full, node_attrs_str)
df_bridge_full.to_csv(STEP3_DIR / "betweenness_bridge_clusters.csv", index=False)
print(f"  Saved betweenness_bridge_clusters.csv ({len(df_bridge_full)} nodes)")

# Comparison plot
fig, ax = plt.subplots(figsize=(10, 8))
colors = {"concept": "#3498db", "intervention": "#e74c3c"}
for _, row in df_btw_full.iterrows():
    c = colors.get(str(row["category"]).lower(), "#95a5a6")
    if row["rank_sim08"] > 0:
        ax.scatter(
            1.0 / max(row["rank_sim08"], 1),
            row["betweenness_sim09"],
            color=c,
            alpha=0.7,
            s=40,
        )
for _, row in df_btw_full.head(20).iterrows():
    if row["rank_sim08"] > 0:
        ax.annotate(
            str(row["name"])[:30],
            (1.0 / max(row["rank_sim08"], 1), row["betweenness_sim09"]),
            fontsize=6,
            alpha=0.8,
        )
ax.set_xlabel("SIM>=0.8 betweenness (1/rank proxy)")
ax.set_ylabel(f"SIM>=0.9 + EDGE conf>={MIN_CONF} betweenness")
ax.set_title(f"Betweenness comparison (conf>={MIN_CONF}, maturity>={MIN_MATURITY})")
handles = [
    plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=lb
    )
    for lb, c in colors.items()
]
ax.legend(handles=handles)
plt.tight_layout()
fig.savefig(STEP3_DIR / "betweenness_comparison.png", dpi=150)
plt.close(fig)
print("  Saved betweenness_comparison.png")
del G_full, btw_full


# ─── SECTION D — BOTH-MODE BETWEENNESS ───────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION D — Both-mode betweenness (ec=0.9, mode=both, conf>=3, maturity>=3)")
print("=" * 70)

for fname in [
    "betweenness_both09.csv",
    "betweenness_both09_bridge_clusters.csv",
    "betweenness_both09_comparison.png",
]:
    backup(STEP3_DIR / fname)

# Collect both-mode nodes
both_nodes = set()
for key, members in cluster_memberships.items():
    if len(key) == 5:
        ec, mode, node_type, algo, cluster_id = key
        if str(ec) == "0.9" and str(mode) == "both" and str(algo) == "agglomerative":
            both_nodes.update(str(m) for m in members)

# Intersect with valid_nodes (maturity>=3 filter)
both_nodes_filtered = both_nodes & valid_nodes
print(
    f"  Both-mode nodes (ec=0.9): {len(both_nodes):,}, "
    f"after maturity filter: {len(both_nodes_filtered):,}"
)

print("  Building filtered subgraph …")
G_both = build_sim09_edge_graph(node_set=both_nodes_filtered)
print(
    f"  Both-mode subgraph: {G_both.number_of_nodes():,} nodes, "
    f"{G_both.number_of_edges():,} edges"
)
sys.stdout.flush()

_done_both = threading.Event()


def _hb_both():
    elapsed = 0
    while not _done_both.wait(timeout=1800):
        elapsed += 1800
        print(
            f"  [heartbeat] both-mode betweenness running — "
            f"{elapsed // 3600}h {(elapsed % 3600) // 60}m elapsed",
            flush=True,
        )


threading.Thread(target=_hb_both, daemon=True).start()
print(
    f"  Computing EXACT betweenness ({G_both.number_of_nodes():,} nodes) …", flush=True
)

btw_both = nx.betweenness_centrality(G_both, normalized=True)
_done_both.set()

ckpt_both = STEP3_DIR / "betweenness_both09_raw_checkpoint_filtered.pkl"
with open(ckpt_both, "wb") as f:
    pickle.dump(btw_both, f, protocol=4)
print(f"  Checkpoint: {ckpt_both} ({len(btw_both):,} nodes)")

top100_both = sorted(btw_both.items(), key=lambda x: x[1], reverse=True)[:100]
rows_both = []
old_rank_both = {}
if old_btw_file.exists():
    old_rank_both = old_rank  # reuse same old reference
for rank, (nid, btw) in enumerate(top100_both, 1):
    attrs = node_attrs_str.get(str(nid), {})
    rows_both.append(
        dict(
            node_id=nid,
            name=attrs.get("name", ""),
            category=get_category(nid),
            betweenness_both09=btw,
            rank_both09=rank,
            rank_sim08=old_rank_both.get(str(nid), -1),
        )
    )

df_btw_both = pd.DataFrame(rows_both)
df_btw_both.to_csv(STEP3_DIR / "betweenness_both09.csv", index=False)
print(f"  Saved betweenness_both09.csv (top {len(df_btw_both)})")

df_bridge_both = cluster_bridges(rows_both, node_attrs_str)
df_bridge_both.to_csv(STEP3_DIR / "betweenness_both09_bridge_clusters.csv", index=False)
print(f"  Saved betweenness_both09_bridge_clusters.csv ({len(df_bridge_both)} nodes)")

fig, ax = plt.subplots(figsize=(10, 8))
for _, row in df_btw_both.iterrows():
    c = colors.get(str(row["category"]).lower(), "#95a5a6")
    if row["rank_sim08"] > 0:
        ax.scatter(
            1.0 / max(row["rank_sim08"], 1),
            row["betweenness_both09"],
            color=c,
            alpha=0.7,
            s=40,
        )
for _, row in df_btw_both.head(20).iterrows():
    if row["rank_sim08"] > 0:
        ax.annotate(
            str(row["name"])[:30],
            (1.0 / max(row["rank_sim08"], 1), row["betweenness_both09"]),
            fontsize=6,
            alpha=0.8,
        )
ax.set_xlabel("SIM>=0.8 betweenness (1/rank proxy)")
ax.set_ylabel(f"Both-mode SIM>=0.9+EDGE conf>={MIN_CONF} betweenness")
ax.set_title(
    f"Both-mode betweenness comparison (conf>={MIN_CONF}, maturity>={MIN_MATURITY})"
)
ax.legend(handles=handles)
plt.tight_layout()
fig.savefig(STEP3_DIR / "betweenness_both09_comparison.png", dpi=150)
plt.close(fig)
print("  Saved betweenness_both09_comparison.png")
del G_both, btw_both

print("\n" + "=" * 70)
print("ALL DONE — Section F + Section D (full + both-mode) re-run complete")
print("Filters applied: EDGE conf>=3, intervention_maturity>=3")
print("=" * 70)
