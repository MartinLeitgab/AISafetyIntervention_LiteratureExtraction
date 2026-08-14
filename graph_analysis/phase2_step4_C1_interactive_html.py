"""
C1 — Interactive Plotly HTML visualizations for Step 4.

Generates interactive .html files alongside the existing PNGs:

1. risk_sim_heatmap.html         — inter-centroid heatmap (risk, 40×40)
2. interv_sim_heatmap.html       — inter-centroid heatmap (intervention, 40×40)
3. risk_dendrogram.html          — hierarchical dendrogram (risk meta-clusters)
4. interv_dendrogram.html        — hierarchical dendrogram (interv meta-clusters)
5. meta_connectivity_network.html — R-meta ↔ I-meta bipartite graph
6. three_layer_network_pathbuildB_metafamily_consim1.html — three-layer network

All saved in same directories as the corresponding PNGs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
STEP4 = BASE / "phase2_results/step4_finalanalysis"
META_DIR = STEP4 / "step4_metaclusters"
CONN_DIR = STEP4 / "step4_connectivity"
TABLE_DIR = STEP4 / "step4_cluster_tables"

print("=== C1: Interactive Plotly HTML generation ===\n")


# ── Helper ─────────────────────────────────────────────────────────────────────
def truncate(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ══════════════════════════════════════════════════════════════════════════════
# 1 & 2  Inter-centroid similarity heatmaps
# ══════════════════════════════════════════════════════════════════════════════
def make_heatmap_html(
    sim_csv: Path,
    meta_assignments_csv: Path,
    out_path: Path,
    title: str,
) -> None:
    sim_df = pd.read_csv(sim_csv, index_col=0)
    # Normalize index and columns to int
    sim_df.index = sim_df.index.astype(int)
    sim_df.columns = sim_df.columns.astype(int)

    meta_df = pd.read_csv(meta_assignments_csv)
    meta_df["cluster_id"] = meta_df["cluster_id"].astype(int)
    meta_map = dict(zip(meta_df["cluster_id"], meta_df["meta_k10"]))
    name_map = dict(zip(meta_df["cluster_id"], meta_df["cluster_name"]))
    n_map = dict(zip(meta_df["cluster_id"], meta_df["n_nodes"]))

    # Order clusters by meta-group
    ids = sorted(sim_df.index, key=lambda c: (meta_map.get(c, 99), c))
    # Use integer column access (columns were set to int above)
    sim_ordered = sim_df.loc[ids, ids].values.astype(float)

    labels = [
        f"{'R' if 'risk' in str(sim_csv) else 'I'}{c}: {truncate(name_map.get(c, '?'), 55)}"
        for c in ids
    ]
    hover = [
        [
            f"<b>{labels[i]}</b><br>"
            f"vs <b>{labels[j]}</b><br>"
            f"cos-sim: {sim_ordered[i, j]:.4f}<br>"
            f"n_nodes_row: {n_map.get(ids[i], '?')}, n_nodes_col: {n_map.get(ids[j], '?')}"
            for j in range(len(ids))
        ]
        for i in range(len(ids))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=sim_ordered,
            x=labels,
            y=labels,
            colorscale="RdYlGn",
            zmin=0.4,
            zmax=1.0,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="cos-sim"),
        )
    )

    # Meta-cluster boundary rectangles
    meta_vals = [meta_map.get(c, 99) for c in ids]
    n = len(ids)
    boundaries = []
    prev_meta = meta_vals[0]
    start = 0
    for k in range(1, n + 1):
        cur_meta = meta_vals[k] if k < n else None
        if cur_meta != prev_meta:
            boundaries.append((start, k - 1, prev_meta))
            start = k
            prev_meta = cur_meta

    shapes = []
    for s, e, _ in boundaries:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=s - 0.5,
                x1=e + 0.5,
                y0=s - 0.5,
                y1=e + 0.5,
                line=dict(color="black", width=1.5),
                fillcolor="rgba(0,0,0,0)",
            )
        )

    fig.update_layout(
        title=title,
        width=1000,
        height=1000,
        shapes=shapes,
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
        margin=dict(l=250, r=50, t=80, b=250),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Saved: {out_path.name}")


print("1/6  Risk inter-centroid heatmap …")
make_heatmap_html(
    META_DIR / "risk_intercent_sim_matrix.csv",
    META_DIR / "risk_meta_assignments.csv",
    META_DIR / "risk_sim_heatmap.html",
    "Risk Cluster Inter-Centroid Cosine Similarity (40×40, ordered by meta-cluster)",
)

print("2/6  Intervention inter-centroid heatmap …")
make_heatmap_html(
    META_DIR / "interv_intercent_sim_matrix.csv",
    META_DIR / "intervention_meta_assignments.csv",
    META_DIR / "interv_sim_heatmap.html",
    "Intervention Cluster Inter-Centroid Cosine Similarity (40×40, ordered by meta-cluster)",
)


# ══════════════════════════════════════════════════════════════════════════════
# 3 & 4  Dendrograms (interactive Plotly)
# ══════════════════════════════════════════════════════════════════════════════
def make_dendrogram_html(
    sim_csv: Path,
    meta_assignments_csv: Path,
    out_path: Path,
    title: str,
    prefix: str,
) -> None:
    sim_df = pd.read_csv(sim_csv, index_col=0)
    sim_df.index = sim_df.index.astype(int)
    sim_df.columns = sim_df.columns.astype(int)

    meta_df = pd.read_csv(meta_assignments_csv)
    meta_df["cluster_id"] = meta_df["cluster_id"].astype(int)
    name_map = dict(zip(meta_df["cluster_id"], meta_df["cluster_name"]))
    n_map = dict(zip(meta_df["cluster_id"], meta_df["n_nodes"]))
    meta_map = dict(zip(meta_df["cluster_id"], meta_df["meta_k10"]))

    ids = sorted(sim_df.index)
    sim_mat = sim_df.loc[ids, ids].values.astype(float)
    np.fill_diagonal(sim_mat, 1.0)
    dist_mat = 1.0 - sim_mat
    dist_mat = np.clip(dist_mat, 0, None)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")

    # Build dendrogram layout via scipy (just for leaf order + merges)
    dend = dendrogram(Z, no_plot=True)
    leaf_order = dend["leaves"]  # cluster index positions in ids[]
    icoord = dend["icoord"]  # x positions of each merge (list of 4-tuples)
    dcoord = dend["dcoord"]  # y (distance) positions

    # Build scatter trace for dendrogram lines
    xs, ys = [], []
    for xi, yi in zip(icoord, dcoord):
        xs += list(xi) + [None]
        ys += list(yi) + [None]

    # Leaf labels
    leaf_ids = [ids[i] for i in leaf_order]
    leaf_names = [
        f"{prefix}{c}: {truncate(name_map.get(c, '?'), 70)}" for c in leaf_ids
    ]
    leaf_x = list(range(5, 5 * (len(leaf_ids) + 1), 5))  # dendrogram x-positions
    leaf_hover = [
        f"{prefix}{c}: {name_map.get(c, '?')}<br>n_nodes: {n_map.get(c, '?')}<br>meta_k10: {meta_map.get(c, '?')}"
        for c in leaf_ids
    ]
    leaf_colors = [meta_map.get(c, 0) for c in leaf_ids]

    # Color palette for meta clusters
    import plotly.colors as pc

    palette = pc.qualitative.Plotly + pc.qualitative.D3 + pc.qualitative.T10
    color_seq = [palette[m % len(palette)] for m in leaf_colors]

    fig = go.Figure()
    # Dendrogram lines
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="gray", width=1),
            hoverinfo="skip",
            name="linkage",
        )
    )
    # Leaf dots colored by meta-cluster
    fig.add_trace(
        go.Scatter(
            x=leaf_x,
            y=[0] * len(leaf_ids),
            mode="markers",
            marker=dict(color=color_seq, size=10, line=dict(width=1, color="black")),
            text=leaf_hover,
            hovertemplate="%{text}<extra></extra>",
            name="clusters",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=leaf_x,
            ticktext=leaf_names,
            tickangle=-45,
            tickfont=dict(size=8),
            title="",
        ),
        yaxis=dict(title="1 − cosine similarity (linkage distance)"),
        width=1400,
        height=700,
        margin=dict(l=80, r=50, t=80, b=260),
        showlegend=False,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Saved: {out_path.name}")


print("3/6  Risk dendrogram …")
make_dendrogram_html(
    META_DIR / "risk_intercent_sim_matrix.csv",
    META_DIR / "risk_meta_assignments.csv",
    META_DIR / "risk_dendrogram.html",
    "Risk Cluster Dendrogram (average linkage on 1-cosine-sim)",
    "R",
)

print("4/6  Intervention dendrogram …")
make_dendrogram_html(
    META_DIR / "interv_intercent_sim_matrix.csv",
    META_DIR / "intervention_meta_assignments.csv",
    META_DIR / "interv_dendrogram.html",
    "Intervention Cluster Dendrogram (average linkage on 1-cosine-sim)",
    "I",
)


# ══════════════════════════════════════════════════════════════════════════════
# 5  Meta-connectivity bipartite network
# ══════════════════════════════════════════════════════════════════════════════
print("5/6  Meta-connectivity network …")

ri_conn = pd.read_csv(META_DIR / "meta_cluster_ri_connectivity.csv")
risk_meta_df = pd.read_csv(META_DIR / "risk_meta_assignments.csv")
interv_meta_df = pd.read_csv(META_DIR / "intervention_meta_assignments.csv")

# Build meta-level label maps
risk_meta_names = {}
risk_meta_n = {}
for _, row in risk_meta_df.iterrows():
    m = int(row["meta_k10"])
    if m not in risk_meta_names:
        risk_meta_names[m] = row["cluster_name"]
        risk_meta_n[m] = 0
    risk_meta_n[m] += int(row["n_nodes"])

interv_meta_names = {}
interv_meta_n = {}
for _, row in interv_meta_df.iterrows():
    m = int(row["meta_k10"])
    if m not in interv_meta_names:
        interv_meta_names[m] = row["cluster_name"]
        interv_meta_n[m] = 0
    interv_meta_n[m] += int(row["n_nodes"])

# Unique meta ids
risk_metas = sorted(risk_meta_names.keys())
interv_metas = sorted(interv_meta_names.keys())


# Node positions: risk on left (x=0), interv on right (x=1)
def evenly_spaced(n):
    return [i / max(n - 1, 1) for i in range(n)]


rx = [0.0] * len(risk_metas)
ry = evenly_spaced(len(risk_metas))
ix = [1.0] * len(interv_metas)
iy = evenly_spaced(len(interv_metas))

r_pos = {m: (0.0, ry[i]) for i, m in enumerate(risk_metas)}
i_pos = {m: (1.0, iy[i]) for i, m in enumerate(interv_metas)}

# Max paths for edge width scaling
max_paths = ri_conn["n_paths_c1"].max()

# Build edge traces (one per connection, colored by weight)
edge_traces = []
for _, row in ri_conn.iterrows():
    rm = int(row["risk_meta"])
    im = int(row["interv_meta"])
    n = int(row["n_paths_c1"])
    if rm not in r_pos or im not in i_pos:
        continue
    x0, y0 = r_pos[rm]
    x1, y1 = i_pos[im]
    width = 0.5 + 4.0 * (n / max_paths) ** 0.4
    rname = truncate(risk_meta_names.get(rm, f"RM{rm}"), 60)
    iname = truncate(interv_meta_names.get(im, f"IM{im}"), 60)
    hover_txt = (
        f"<b>Risk meta {rm}:</b> {rname}<br>"
        f"<b>Interv meta {im}:</b> {iname}<br>"
        f"n_paths (consim1): {n:,}<br>"
        f"n_cluster_pairs: {int(row.get('n_cluster_pairs', 0)):,}"
    )
    edge_traces.append(
        go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(
                width=width,
                color=f"rgba(100,100,200,{min(0.9, 0.1 + 0.8 * (n / max_paths) ** 0.3)})",
            ),
            hoverinfo="text",
            text=[hover_txt, hover_txt, None],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

# Node traces
risk_hover = [
    f"<b>Risk meta {m}</b><br>{risk_meta_names.get(m, '?')}<br>n_nodes_total: {risk_meta_n.get(m, '?')}"
    for m in risk_metas
]
interv_hover = [
    f"<b>Interv meta {m}</b><br>{interv_meta_names.get(m, '?')}<br>n_nodes_total: {interv_meta_n.get(m, '?')}"
    for m in interv_metas
]

risk_node = go.Scatter(
    x=[r_pos[m][0] for m in risk_metas],
    y=[r_pos[m][1] for m in risk_metas],
    mode="markers+text",
    marker=dict(size=16, color="tomato", line=dict(width=1, color="black")),
    text=[f"RM{m}" for m in risk_metas],
    textposition="middle left",
    hovertext=risk_hover,
    hovertemplate="%{hovertext}<extra></extra>",
    name="Risk meta-clusters",
)
interv_node = go.Scatter(
    x=[i_pos[m][0] for m in interv_metas],
    y=[i_pos[m][1] for m in interv_metas],
    mode="markers+text",
    marker=dict(size=16, color="mediumseagreen", line=dict(width=1, color="black")),
    text=[f"IM{m}" for m in interv_metas],
    textposition="middle right",
    hovertext=interv_hover,
    hovertemplate="%{hovertext}<extra></extra>",
    name="Intervention meta-clusters",
)

fig = go.Figure(data=edge_traces + [risk_node, interv_node])
fig.update_layout(
    title="Meta-Cluster R↔I Connectivity Network (consim1, hover for details)",
    width=1100,
    height=900,
    xaxis=dict(visible=False, range=[-0.15, 1.15]),
    yaxis=dict(visible=False),
    margin=dict(l=100, r=100, t=80, b=40),
    hovermode="closest",
    plot_bgcolor="white",
)
fig.write_html(str(META_DIR / "meta_connectivity_network.html"), include_plotlyjs="cdn")
print("  Saved: meta_connectivity_network.html")


# ══════════════════════════════════════════════════════════════════════════════
# 6  Three-layer network: Risk → PathbuildB meta-families → Intervention
# ══════════════════════════════════════════════════════════════════════════════
print("6/6  Three-layer network (pathbuildB meta-families, consim1) …")

# Load data
r2m = pd.read_csv(CONN_DIR / "risk_to_metafamily_edges_consim1.csv")
m2i = pd.read_csv(CONN_DIR / "metafamily_to_interv_edges_consim1.csv")
mf_summary = pd.read_csv(TABLE_DIR / "pathbuildB_metafamily_summary_consim1.csv")
risk_clusters = pd.read_csv(TABLE_DIR / "risk_clusters_consim1.csv")
interv_clusters = pd.read_csv(TABLE_DIR / "intervention_clusters_consim1.csv")

# Build lookup maps
risk_name = dict(zip(risk_clusters["cluster_id"], risk_clusters["top_node_name"]))
interv_name = dict(zip(interv_clusters["cluster_id"], interv_clusters["top_node_name"]))
mf_name = dict(zip(mf_summary["meta_family_id"], mf_summary["dominant_family_name"]))
mf_paths = dict(zip(mf_summary["meta_family_id"], mf_summary["n_paths_total"]))

# Unique IDs
risk_ids = sorted(r2m["risk_cluster"].unique())
mf_ids = sorted(r2m["meta_family_id"].unique())
interv_ids = sorted(m2i["interv_cluster"].unique())


# Positions: x=0 (risk), x=0.5 (meta-family), x=1 (interv)
def ys(n):
    return [i / max(n - 1, 1) for i in range(n)]


r_pos = {c: (0.0, ys(len(risk_ids))[i]) for i, c in enumerate(risk_ids)}
m_pos = {c: (0.5, ys(len(mf_ids))[i]) for i, c in enumerate(mf_ids)}
i_pos = {c: (1.0, ys(len(interv_ids))[i]) for i, c in enumerate(interv_ids)}

max_r2m = r2m["n_paths"].max()
max_m2i = m2i["n_paths"].max()

edge_traces = []

# Risk → meta-family edges
for _, row in r2m.iterrows():
    rc, mf, n = (
        int(row["risk_cluster"]),
        int(row["meta_family_id"]),
        int(row["n_paths"]),
    )
    if rc not in r_pos or mf not in m_pos:
        continue
    x0, y0 = r_pos[rc]
    x1, y1 = m_pos[mf]
    w = 0.3 + 3.0 * (n / max_r2m) ** 0.35
    hover = (
        f"<b>Risk R{rc}:</b> {truncate(risk_name.get(rc, '?'), 60)}<br>"
        f"→ <b>MF{mf}:</b> {truncate(mf_name.get(mf, '?'), 60)}<br>"
        f"n_paths: {n:,}"
    )
    edge_traces.append(
        go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=w, color="rgba(220,80,80,0.35)"),
            hoverinfo="text",
            text=[hover, hover, None],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

# Meta-family → intervention edges
for _, row in m2i.iterrows():
    mf, ic, n = (
        int(row["meta_family_id"]),
        int(row["interv_cluster"]),
        int(row["n_paths"]),
    )
    if mf not in m_pos or ic not in i_pos:
        continue
    x0, y0 = m_pos[mf]
    x1, y1 = i_pos[ic]
    w = 0.3 + 3.0 * (n / max_m2i) ** 0.35
    hover = (
        f"<b>MF{mf}:</b> {truncate(mf_name.get(mf, '?'), 60)}<br>"
        f"→ <b>Interv I{ic}:</b> {truncate(interv_name.get(ic, '?'), 60)}<br>"
        f"n_paths: {n:,}"
    )
    edge_traces.append(
        go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=w, color="rgba(80,160,80,0.35)"),
            hoverinfo="text",
            text=[hover, hover, None],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

# Node traces
risk_hover = [
    f"<b>Risk R{c}</b><br>{risk_name.get(c, '?')}<br>n_nodes: {risk_clusters.set_index('cluster_id')['n_nodes'].get(c, '?') if c in risk_clusters['cluster_id'].values else '?'}"
    for c in risk_ids
]
mf_hover = [
    f"<b>MF{c}</b><br>{mf_name.get(c, '?')}<br>n_paths_total: {mf_paths.get(c, '?'):,}"
    for c in mf_ids
]
interv_hover = [f"<b>Interv I{c}</b><br>{interv_name.get(c, '?')}" for c in interv_ids]

risk_node_tr = go.Scatter(
    x=[r_pos[c][0] for c in risk_ids],
    y=[r_pos[c][1] for c in risk_ids],
    mode="markers+text",
    marker=dict(size=10, color="tomato", line=dict(width=1, color="darkred")),
    text=[f"R{c}" for c in risk_ids],
    textposition="middle left",
    hovertext=risk_hover,
    hovertemplate="%{hovertext}<extra></extra>",
    name="Risk clusters",
)
mf_node_tr = go.Scatter(
    x=[m_pos[c][0] for c in mf_ids],
    y=[m_pos[c][1] for c in mf_ids],
    mode="markers+text",
    marker=dict(size=10, color="limegreen", line=dict(width=1, color="darkgreen")),
    text=[f"MF{c}" for c in mf_ids],
    textposition="middle center",
    hovertext=mf_hover,
    hovertemplate="%{hovertext}<extra></extra>",
    name="PathbuildB meta-families",
)
interv_node_tr = go.Scatter(
    x=[i_pos[c][0] for c in interv_ids],
    y=[i_pos[c][1] for c in interv_ids],
    mode="markers+text",
    marker=dict(size=10, color="cornflowerblue", line=dict(width=1, color="darkblue")),
    text=[f"I{c}" for c in interv_ids],
    textposition="middle right",
    hovertext=interv_hover,
    hovertemplate="%{hovertext}<extra></extra>",
    name="Intervention clusters",
)

fig = go.Figure(data=edge_traces + [risk_node_tr, mf_node_tr, interv_node_tr])
fig.update_layout(
    title="Three-Layer Network: Risk → PathbuildB Meta-Families → Intervention (consim1)<br>"
    "<sup>Hover nodes/edges for full names and path counts</sup>",
    width=1400,
    height=1400,
    xaxis=dict(visible=False, range=[-0.12, 1.12]),
    yaxis=dict(visible=False),
    margin=dict(l=60, r=60, t=100, b=40),
    hovermode="closest",
    plot_bgcolor="white",
    legend=dict(x=0.35, y=1.01, orientation="h"),
)
out_three = CONN_DIR / "three_layer_network_pathbuildB_metafamily_consim1.html"
fig.write_html(str(out_three), include_plotlyjs="cdn")
print("  Saved: three_layer_network_pathbuildB_metafamily_consim1.html")

print("\n=== C1 complete: 6 interactive HTML files generated ===")
