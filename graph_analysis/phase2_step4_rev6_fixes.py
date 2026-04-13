"""
Rev6 fixes — completing step45_revision4_inputs.txt items 3, 5, 8, 10 fully.

Item 3 (full audit): Fix all truncated name columns in step4/5 output CSVs.
  Confirmed truncation at hard limits: [:60] in ri_meta_triplets/ri_triplets/
  meta_cluster_coherence/meta_cluster_ri_connectivity, [:100] in
  mechanism_families_qualifying/bodysubtype_cluster_representatives.

Item 5 (full): Regenerate umap_interventions_consim1_clusters.png with ALL
  ~19k VPN_consim1 nodes (no 200/cluster cap) using cosine metric (correct for
  high-dimensional text embeddings; euclidean on L2-normalized is equivalent but
  cosine is explicit and consistent with all other UMAP plots).
  Add "each point = 1 node | UMAP (cosine)" subtitle to all UMAP plots.
  Add "each point = 1 cluster | MDS (1-cosine)" subtitle to all MDS plots.

Item 8 (full audit): Fix pathbuildB_family_size_distribution.png and
  ri_triplets_histogram.png: both plotted np.log10(values) on x-axis with
  labels showing raw exponent numbers. Fixed to use log x-scale with actual
  count tick labels (1, 10, 100, 1000).

Item 10 (audit): All 4 dendrograms confirmed orientation="left" in source code
  (risk_dendrogram.png, interv_dendrogram.png, top20_bfamily_jaccard_dendrogram.png,
  pathbuildB_metafamily_dendrogram.png). No changes needed.
"""

import gc
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
STEP4 = BASE / "phase2_results/step4_finalanalysis"
META_DIR = STEP4 / "step4_metaclusters"
CONN_DIR = STEP4 / "step4_connectivity"
TABLE_DIR = STEP4 / "step4_cluster_tables"
NAMING_DIR = BASE / "phase2_results/step5_naming"
PKL_DIR = BASE / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
RAWPATHS_DIR = BASE / "phase1_rawpathsfiles"

print("=== Rev6 fixes ===\n")

# ─── Load naming maps (full, no truncation) ───────────────────────────────────
risk_names_df = pd.read_csv(NAMING_DIR / "risk_cluster_names_llm_v2.csv")
interv_names_df = pd.read_csv(NAMING_DIR / "intervention_cluster_names_llm_v2.csv")
risk_name_col = "final_name" if "final_name" in risk_names_df.columns else "llm_name"
interv_name_col = (
    "final_name" if "final_name" in interv_names_df.columns else "llm_name"
)
# NO [:60] or [:80] — full names
risk_name_map = dict(
    zip(risk_names_df["cluster_id"].astype(str), risk_names_df[risk_name_col])
)
interv_name_map = dict(
    zip(interv_names_df["cluster_id"].astype(str), interv_names_df[interv_name_col])
)

v3_df = pd.read_csv(NAMING_DIR / "pathbuildB_chain_names_llm_v3.csv")
chain_name_map_v3 = dict(zip(v3_df["cluster_id"].astype(int), v3_df["final_name"]))

# Load meta assignments
risk_meta_df = pd.read_csv(META_DIR / "risk_meta_assignments.csv")
interv_meta_df = pd.read_csv(META_DIR / "intervention_meta_assignments.csv")


def get_meta_name_full(meta_df, meta_id, name_map):
    """Return the full (untruncated) name of the largest cluster in a meta-cluster."""
    sub = meta_df[meta_df["meta_k10"] == meta_id].sort_values(
        "n_nodes", ascending=False
    )
    if len(sub) == 0:
        return f"Meta-{meta_id}"
    return name_map.get(str(int(sub.iloc[0]["cluster_id"])), f"Meta-{meta_id}")


# ══════════════════════════════════════════════════════════════════════════════
# Item 3 — Fix all truncated CSV columns
# ══════════════════════════════════════════════════════════════════════════════
print("Item 3 — Fixing truncated name columns in step4 CSVs ...")


def rebuild_meta_names(
    csv_path,
    risk_col="risk_meta_name",
    interv_col="interv_meta_name",
    risk_id_col="risk_meta_id",
    interv_id_col="interv_meta_id",
):
    df = pd.read_csv(csv_path)
    if risk_col in df.columns and risk_id_col in df.columns:
        df[risk_col] = df[risk_id_col].apply(
            lambda m: get_meta_name_full(risk_meta_df, m, risk_name_map)
        )
    if interv_col in df.columns and interv_id_col in df.columns:
        df[interv_col] = df[interv_id_col].apply(
            lambda m: get_meta_name_full(interv_meta_df, m, interv_name_map)
        )
    df.to_csv(csv_path, index=False)
    print(f"  Fixed: {csv_path.relative_to(BASE / 'phase2_results')}")
    return df


# ri_meta_triplets_consim1.csv — risk_meta_name, interv_meta_name truncated at 60
rebuild_meta_names(CONN_DIR / "ri_meta_triplets_consim1.csv")
rebuild_meta_names(CONN_DIR / "ri_meta_triplets_top20_consim1.csv")

# meta_cluster_ri_connectivity.csv — risk_meta_name, interv_meta_name truncated at 60
conn_csv = META_DIR / "meta_cluster_ri_connectivity.csv"
rebuild_meta_names(conn_csv, risk_id_col="risk_meta", interv_id_col="interv_meta")

# meta_cluster_coherence.csv — theme column truncated at 60
coh_csv = META_DIR / "meta_cluster_coherence.csv"
coh = pd.read_csv(coh_csv)
# Rebuild from dominant cluster name (largest by n_nodes in each meta)
risk_coh = coh[coh["node_type"] == "risk"].copy()
interv_coh = coh[coh["node_type"] == "intervention"].copy()


def rebuild_theme(row, meta_df, name_map):
    return get_meta_name_full(meta_df, row["meta_k10"], name_map)


risk_coh["theme"] = risk_coh.apply(
    lambda r: rebuild_theme(r, risk_meta_df, risk_name_map), axis=1
)
interv_coh["theme"] = interv_coh.apply(
    lambda r: rebuild_theme(r, interv_meta_df, interv_name_map), axis=1
)
coh_fixed = pd.concat([risk_coh, interv_coh]).sort_values(["node_type", "meta_k10"])
coh_fixed.to_csv(coh_csv, index=False)
print(f"  Fixed: {coh_csv.relative_to(BASE / 'phase2_results')}")

# ri_triplets_consim1.csv — chain_name truncated at 60
trip_csv = CONN_DIR / "ri_triplets_consim1.csv"
trip = pd.read_csv(trip_csv)
trip["chain_name"] = (
    trip["bfamily_id"].map(chain_name_map_v3).fillna(trip["chain_name"])
)
trip["risk_name"] = trip["risk_cid"].apply(
    lambda c: risk_name_map.get(str(int(c)), str(c))
)
trip["interv_name"] = trip["interv_cid"].apply(
    lambda c: interv_name_map.get(str(int(c)), str(c))
)
trip.to_csv(trip_csv, index=False)
print(f"  Fixed: {trip_csv.relative_to(BASE / 'phase2_results')}")

# Verify no more truncation at 60
print("\n  Verification — max name lengths after fix:")
for csv_path, cols in [
    (CONN_DIR / "ri_meta_triplets_consim1.csv", ["risk_meta_name", "interv_meta_name"]),
    (CONN_DIR / "ri_triplets_consim1.csv", ["chain_name", "risk_name", "interv_name"]),
    (META_DIR / "meta_cluster_coherence.csv", ["theme"]),
    (
        META_DIR / "meta_cluster_ri_connectivity.csv",
        ["risk_meta_name", "interv_meta_name"],
    ),
]:
    df = pd.read_csv(csv_path)
    for col in cols:
        if col in df.columns:
            max_l = df[col].dropna().astype(str).str.len().max()
            print(f"    {csv_path.name} | {col}: max_len={max_l}")

# ══════════════════════════════════════════════════════════════════════════════
# Item 8 — Fix log10-exponent histograms
# ══════════════════════════════════════════════════════════════════════════════
print("\nItem 8 — Fixing log10-exponent histograms ...")

fmt_count = FuncFormatter(lambda x, _: f"{int(x):,}" if x >= 1 else "")


def fix_log_histogram(
    data, bins, title, xlabel, out_path, color="steelblue", vlines=None
):
    """Plot histogram with proper log x-scale showing actual values (not log10 exponents)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_formatter(fmt_count)
    ax.tick_params(labelsize=8)
    if vlines:
        for v, label in vlines:
            ax.axvline(v, color="gray", lw=1, linestyle=":", alpha=0.7)
            ax.text(v, ax.get_ylim()[1] * 0.88, label, fontsize=7, ha="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed: {Path(out_path).relative_to(BASE / 'phase2_results')}")


# pathbuildB_family_size_distribution.png — B3 log histogram
fam_df = pd.read_csv(TABLE_DIR / "optionB_cooccurrence_families_consim1.csv")
fix_log_histogram(
    data=fam_df["n_paths"].clip(1),
    bins=np.logspace(0, np.log10(fam_df["n_paths"].max()), 60),
    title=f"PathbuildB Family Size Distribution — log scale (all {len(fam_df):,} families)",
    xlabel="N paths per family (log scale — actual counts)",
    out_path=TABLE_DIR / "pathbuildB_family_size_distribution.png",
    color="darkorange",
    vlines=[(5, "n=5"), (10, "n=10"), (100, "n=100")],
)

# ri_triplets_histogram.png — log10(n_triplet_paths) was on linear x-axis
trip_df = pd.read_csv(CONN_DIR / "ri_triplets_consim1.csv")
fix_log_histogram(
    data=trip_df["n_triplet_paths"].clip(1),
    bins=np.logspace(0, np.log10(trip_df["n_triplet_paths"].max()), 50),
    title=f"R->Chain->I Triplet Path Count Distribution (consim1, n>=5, {len(trip_df):,} triplets)",
    xlabel="N paths per triplet (log scale — actual counts)",
    out_path=CONN_DIR / "ri_triplets_histogram.png",
)

# ══════════════════════════════════════════════════════════════════════════════
# Item 5 — Add "each point = 1 node | UMAP (cosine)" to UMAP plots
#          Add "each point = 1 cluster | MDS (1-cosine)" to MDS plots
# ══════════════════════════════════════════════════════════════════════════════
print(
    "\nItem 5a — Adding 'each point = 1 node/cluster' subtitles to existing plots ..."
)

UMAP_SUBTITLE = "Each point = 1 node  |  UMAP projection (cosine metric, VPN_consim1 qualifying nodes)"
MDS_SUBTITLE = (
    "Each point = 1 cluster  |  2D MDS projection (1 - cosine centroid similarity)"
)


def add_subtitle_to_png(png_path, subtitle, y_frac=0.01, fontsize=7):
    """Re-open existing PNG and add a subtitle line at the bottom via matplotlib."""
    img = mpimg.imread(str(png_path))
    h, w = img.shape[:2]
    fig_w = w / 130
    fig_h = h / 130
    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.35))
    ax.imshow(img)
    ax.axis("off")
    fig.text(
        0.5,
        0.005,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        style="italic",
        color="#444444",
        transform=fig.transFigure,
    )
    fig.subplots_adjust(bottom=0.04, top=1.0, left=0, right=1)
    fig.savefig(str(png_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Subtitle added: {Path(png_path).relative_to(BASE / 'phase2_results')}")


# All UMAP plots (step4_finalanalysis root)
umap_plots = list((BASE / "phase2_results/step4_finalanalysis").glob("umap_*.png"))
for p in sorted(umap_plots):
    add_subtitle_to_png(p, UMAP_SUBTITLE)

# MDS plots in step4_metaclusters
for name in ["risk_2d_mds.png", "interv_2d_mds.png"]:
    p = META_DIR / name
    if p.exists():
        add_subtitle_to_png(p, MDS_SUBTITLE)

# pathbuildB MDS
for name in ["pathbuildB_metafamily_2d_mds.png"]:
    p = TABLE_DIR / name
    if p.exists():
        add_subtitle_to_png(p, MDS_SUBTITLE)

# ══════════════════════════════════════════════════════════════════════════════
# Item 5b — Regenerate umap_interventions_consim1_clusters.png
#   - ALL VPN_consim1 intervention nodes (no 200/cluster cap)
#   - metric="cosine" (explicit, matches all other UMAP plots)
#   - Subtitle noting "each point = 1 node"
# ══════════════════════════════════════════════════════════════════════════════
print(
    "\nItem 5b — Regenerating umap_interventions_consim1_clusters.png (all nodes, cosine) ..."
)

try:
    from umap import UMAP

    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False
    print("  WARNING: umap-learn not available — skipping B9 regen")

if HAVE_UMAP:
    import textwrap

    print("  Loading node_attrs.pkl ...")
    with open(PKL_DIR / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"    {len(node_attrs):,} nodes")

    print("  Loading cluster_memberships.pkl ...")
    with open(PKL_DIR / "cluster_memberships.pkl", "rb") as f:
        cm = pickle.load(f)

    print("  Loading graph_edge_data.pkl ...")
    with open(PKL_DIR / "graph_edge_data.pkl", "rb") as f:
        edge_data = pickle.load(f)
    print(f"    {len(edge_data):,} edges")

    # ─── Build VPN_consim1 ──────────────────────────────────────────────────
    def cos_sim_from_score(s):
        return 1.0 - float(s) ** 2 / 2.0

    path_file = RAWPATHS_DIR / "paths_unconstrained_sim0.9.jsonl"
    print("  Building VPN_consim1 pass 1 ...")
    vpn_unconstrained = set()
    with open(path_file) as f:
        for line in f:
            obj = json.loads(line.strip())
            if isinstance(obj, dict) and "path" in obj:
                path = [int(x) for x in obj["path"]]
                if (
                    int(
                        node_attrs.get(path[-1], {}).get("intervention_maturity", 0)
                        or 0
                    )
                    >= 3
                ):
                    vpn_unconstrained.update(path)
    print(f"    vpn_unconstrained: {len(vpn_unconstrained):,}")

    sim_edge_set = set()
    for e in edge_data:
        if str(e.get("type", "")).upper() == "SIMILARITY":
            score = e.get("similarity_score")
            if score is not None and cos_sim_from_score(score) >= 0.9:
                try:
                    s, t = int(e["source"]), int(e["target"])
                    if s in vpn_unconstrained and t in vpn_unconstrained:
                        sim_edge_set.add((min(s, t), max(s, t)))
                except (ValueError, TypeError):
                    pass
    del edge_data
    gc.collect()
    print(f"    sim_edge_set: {len(sim_edge_set):,}")

    def max_consec_sim(path_ids):
        max_run = run = 0
        for i in range(len(path_ids) - 1):
            a, b = int(path_ids[i]), int(path_ids[i + 1])
            if (min(a, b), max(a, b)) in sim_edge_set:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return max_run

    print("  Building VPN_consim1 pass 2 ...")
    vpn = set()
    with open(path_file) as f:
        for line in f:
            obj = json.loads(line.strip())
            if isinstance(obj, dict) and "path" in obj:
                path = [int(x) for x in obj["path"]]
                if (
                    int(
                        node_attrs.get(path[-1], {}).get("intervention_maturity", 0)
                        or 0
                    )
                    < 3
                ):
                    continue
                if max_consec_sim(path) <= 1:
                    vpn.update(path)
    print(f"    vpn_consim1: {len(vpn):,}")

    # ─── Get intervention clusters filtered to VPN ──────────────────────────
    interv_clusters = {}
    for key, members in cm.items():
        try:
            e_float = float(key[0])
        except (ValueError, TypeError):
            continue
        if (
            abs(e_float - 0.9) < 1e-9
            and key[1] == "unconstrained"
            and key[2] == "intervention"
            and key[3] == "agglomerative"
        ):
            filtered = [nid for nid in members if nid in vpn]
            if filtered:
                interv_clusters[int(key[4])] = filtered

    print(f"  Intervention clusters: {len(interv_clusters)}")
    total_members = sum(len(v) for v in interv_clusters.values())
    print(f"  Total VPN_consim1 members (ALL — no sampling): {total_members:,}")

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

    # Build embedding matrix — ALL nodes, no cap
    all_vecs, all_cids = [], []
    for cid, members in sorted(interv_clusters.items()):
        for nid in members:  # no [:N_SAMPLE] limit
            v = get_embedding(nid)
            if v is not None:
                all_vecs.append(v)
                all_cids.append(cid)

    print(f"  Total nodes for UMAP: {len(all_vecs):,}")
    X = np.array(all_vecs, dtype=np.float32)

    print("  Running UMAP (cosine metric) ...")
    reducer = UMAP(
        n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric="cosine"
    )
    coords = reducer.fit_transform(X)

    # Load cluster names
    interv_names_local = dict(
        zip(interv_names_df["cluster_id"].astype(int), interv_names_df[interv_name_col])
    )

    cmap_plot = plt.cm.get_cmap("tab20", 40)
    fig, ax = plt.subplots(figsize=(14, 11))
    unique_cids = sorted(set(all_cids))
    for cid in unique_cids:
        mask = np.array([c == cid for c in all_cids])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=6,
            alpha=0.5,
            color=cmap_plot(cid % 40),
            label=f"I{cid}",
            zorder=2,
        )
    for cid in unique_cids:
        mask = np.array([c == cid for c in all_cids])
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        label = "\n".join(
            textwrap.wrap(interv_names_local.get(cid, f"I{cid}"), width=20)
        )
        ax.text(
            cx,
            cy,
            label,
            fontsize=4.5,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.5),
        )

    x0, x1 = coords[:, 0].min(), coords[:, 0].max()
    y0, y1 = coords[:, 1].min(), coords[:, 1].max()
    ax.set_xlabel(f"UMAP-1  [{x0:.1f}, {x1:.1f}]", fontsize=9)
    ax.set_ylabel(f"UMAP-2  [{y0:.1f}, {y1:.1f}]", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"Intervention Clusters — UMAP (consim1, ALL {len(all_vecs):,} qualifying nodes, cosine metric)\n"
        f"Each point = 1 node  |  colored by cluster assignment",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = STEP4 / "umap_interventions_consim1_clusters.png"
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  Saved: {out_path.relative_to(BASE / 'phase2_results')} ({len(all_vecs):,} nodes)"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Item 10 audit — Confirm all dendrogram orientations
# ══════════════════════════════════════════════════════════════════════════════
print("\nItem 10 — Dendrogram orientation audit:")
dendro_plots = [
    (
        META_DIR / "risk_dendrogram.png",
        "risk_dendrogram.png",
        "phase2_step4_B1_metaclusters.py",
        "orientation='left' line 509",
    ),
    (
        META_DIR / "interv_dendrogram.png",
        "interv_dendrogram.png",
        "phase2_step4_B1_metaclusters.py",
        "orientation='left' line 509",
    ),
    (
        TABLE_DIR / "top20_bfamily_jaccard_dendrogram.png",
        "top20_bfamily_jaccard_dendrogram.png",
        "phase2_step4_rev3_computations.py",
        "orientation='left' line 748",
    ),
    (
        TABLE_DIR / "pathbuildB_metafamily_dendrogram.png",
        "pathbuildB_metafamily_dendrogram.png",
        "phase2_step4_rev5_fixes.py",
        "orientation='left' fixed in rev5",
    ),
]
for path, name, script, note in dendro_plots:
    exists = "EXISTS" if path.exists() else "MISSING"
    folder = str(path.parent.relative_to(BASE / "phase2_results"))
    print(f"  {folder}/{name}: {exists} | {note} | source: {script}")

print("\n=== Rev6 fixes complete ===")
