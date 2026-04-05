"""
Fix hub_quality_metrics.csv: recompute ALL degree columns restricting
SIM-edge partner nodes to the valid-pathway node set.

Prior version (hub_quality_metrics.csv after phase2_fix_hub_quality_degree.py):
  - degree_structural: already fixed (conf>=3, both endpoints valid-pathway)
  - degree_sim_*: still counted ALL SIM partners regardless of validity
  - n_sources_at_config_thr: still counted partner URLs from non-valid-pathway nodes
  - degree_total_0.80, sim_ratio_90_80, is_high_thr_hub: derived — also wrong

This script fixes all SIM-based columns:
  For each SIM edge, only count if BOTH endpoints are in valid-pathway node set.

Inputs:
  phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl
  phase2_results/step2_metrics_and_stability/hub_quality_metrics.csv
  phase1_rawpathsfiles/paths_*.jsonl  (defines valid-pathway node set)
  phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_node_attributes.pkl
    (for hub name lookups in bar chart)

Outputs:
  Backup existing:
    hub_quality_metrics.csv          → hub_quality_metrics_structonly.csv
    hub_quality_scatter.png          → hub_quality_scatter_structonly.png
    hub_quality_scatter_v2.png       → hub_quality_scatter_v2_structonly.png
    hub_quality_bar_v2.png           → hub_quality_bar_v2_structonly.png
  Write corrected:
    hub_quality_metrics.csv          (all degree columns fixed)
    hub_quality_scatter.png
    hub_quality_scatter_v2.png
    hub_quality_bar_v2.png

Run from graph_analysis/:
    python phase2_fix_hub_quality_sim_degrees.py > /tmp/hub_quality_sim_fix.log 2>&1
"""

import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

STEP1_DIR = Path("phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")
STEP2_DIR = Path("phase2_results/step2_metrics_and_stability")
PATHS_DIR = Path("phase1_rawpathsfiles")
MIN_CONF = 3

SIM_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

# Primary config for plots (matches step2c)
PRIMARY_EDGE_CONFIG = "0.9"
PRIMARY_MODE = "both"
NODE_TYPES_ORDER = [
    "risk",
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
    "intervention",
    "all_concepts",
]
NODE_TYPE_COLORS = [
    "#d62728",
    "#ff7f0e",
    "#2ca02c",
    "#1f77b4",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]
NODE_TYPE_LABELS = {
    "risk": "Risk",
    "problem_analysis": "Problem Analysis",
    "theoretical_insight": "Theoretical Insight",
    "design_rationale": "Design Rationale",
    "implementation_mechanism": "Implementation Mechanism",
    "validation_evidence": "Validation Evidence",
    "intervention": "Intervention",
    "all_concepts": "All Concepts",
}
DPI = 120


def backup(src: Path, suffix: str):
    dst = src.with_name(src.stem + suffix + src.suffix)
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Backed up {src.name} → {dst.name}")
    elif dst.exists():
        print(f"  Backup already exists: {dst.name}")


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


# ─── STEP 1: build valid-pathway node set ────────────────────────────────────

print("Building valid-pathway node set from path files …")
valid_nodes: set = set()
for pf in sorted(PATHS_DIR.glob("paths_*.jsonl")):
    n_before = len(valid_nodes)
    with open(pf) as f:
        for line in f:
            rec = json.loads(line)
            path = rec.get("path", [])
            if isinstance(path, str):
                path = json.loads(path)
            valid_nodes.update(str(n) for n in path)
    print(
        f"  {pf.name:<55}: +{len(valid_nodes) - n_before:,} new  cumulative {len(valid_nodes):,}"
    )
print(f"  Total valid-pathway nodes: {len(valid_nodes):,}")


# ─── STEP 2: load edge_data, build per-hub degree index ─────────────────────

print("\nLoading edge_data.pkl …")
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
print(f"  Loaded {len(edge_data):,} edges")

print("Computing degree index restricted to valid-pathway nodes …")

# Index structure: {node_id: {structural, sim_0.80, ..., partner_urls_*}}
idx: dict = defaultdict(
    lambda: {
        "structural": 0,
        "sim_0.80": 0,
        "sim_0.85": 0,
        "sim_0.90": 0,
        "sim_0.95": 0,
        "total_0.80": 0,
        "partner_urls_0.80": set(),
        "partner_urls_0.85": set(),
        "partner_urls_0.90": set(),
        "partner_urls_0.95": set(),
    }
)

n_struct_kept, n_struct_conf, n_struct_node = 0, 0, 0
n_sim_kept, n_sim_node = 0, 0

# We need node_attrs for partner URL lookups
print("Loading graph_node_attributes.pkl …")
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
print(f"  Loaded {len(node_attrs):,} node attributes")

for e in edge_data:
    src_raw = e.get("source")
    tgt_raw = e.get("target")
    src = str(src_raw) if src_raw is not None else ""
    tgt = str(tgt_raw) if tgt_raw is not None else ""
    etype = str(e.get("type", "")).upper()

    # Both endpoints must be in valid-pathway set
    if src not in valid_nodes or tgt not in valid_nodes:
        if etype == "EDGE":
            n_struct_node += 1
        else:
            n_sim_node += 1
        continue

    if etype == "EDGE":
        conf = e.get("confidence")
        try:
            if float(conf) >= MIN_CONF:
                idx[src]["structural"] += 1
                idx[tgt]["structural"] += 1
                idx[src]["total_0.80"] += 1
                idx[tgt]["total_0.80"] += 1
                n_struct_kept += 1
            else:
                n_struct_conf += 1
        except (TypeError, ValueError):
            n_struct_conf += 1

    elif etype == "SIMILARITY":
        score = e.get("similarity_score")
        if score is None:
            continue
        cos_sim = cos_sim_from_score(score)
        # node_attrs uses integer keys — use raw (int) values for lookup
        partner_url_src = node_attrs.get(tgt_raw, {}).get("url", "")
        partner_url_tgt = node_attrs.get(src_raw, {}).get("url", "")
        for nid, partner_url in ((src, partner_url_src), (tgt, partner_url_tgt)):
            for thr in SIM_THRESHOLDS:
                if cos_sim >= thr:
                    key = f"sim_{thr:.2f}"
                    idx[nid][key] += 1
                    if partner_url:
                        idx[nid][f"partner_urls_{thr:.2f}"].add(partner_url)
        # total_0.80 counts SIM edges at any threshold (all SIM are >= 0.80)
        if cos_sim >= 0.80:
            idx[src]["total_0.80"] += 1
            idx[tgt]["total_0.80"] += 1
            n_sim_kept += 1

print(f"  EDGE kept (conf>={MIN_CONF}, both valid): {n_struct_kept:,}")
print(
    f"  EDGE dropped conf<{MIN_CONF}: {n_struct_conf:,}  non-valid endpoint: {n_struct_node:,}"
)
print(
    f"  SIM kept (both valid, cos>=0.80): {n_sim_kept:,}  non-valid endpoint: {n_sim_node:,}"
)


# ─── STEP 3: backup and update hub_quality_metrics.csv ───────────────────────

hub_csv = STEP2_DIR / "hub_quality_metrics.csv"
backup(hub_csv, "_structonly")

print(f"\nLoading {hub_csv.name} …")
df = pd.read_csv(hub_csv, dtype={"hub_node_id": str})
print(f"  {len(df):,} rows, columns: {list(df.columns)}")

config_to_url_key = {
    "0.8": "partner_urls_0.80",
    "0.80": "partner_urls_0.80",
    "0.85": "partner_urls_0.85",
    "0.9": "partner_urls_0.90",
    "0.90": "partner_urls_0.90",
    "0.95": "partner_urls_0.95",
    "EDGE": "partner_urls_0.80",  # EDGE-config rows: use 0.80 for n_sources
}

old_means = {}
new_means = {}
for col in [
    "degree_structural",
    "degree_sim_0.80",
    "degree_sim_0.85",
    "degree_sim_0.90",
    "degree_sim_0.95",
    "degree_total_0.80",
    "n_sources_at_config_thr",
    "sim_ratio_90_80",
]:
    if col in df.columns:
        old_means[col] = df[col].mean()


# Recompute all columns
def get_n_sources(nid, ec):
    url_key = config_to_url_key.get(str(ec), "partner_urls_0.80")
    return len(idx[str(nid)].get(url_key, set()))


df["degree_structural"] = df["hub_node_id"].apply(lambda n: idx[str(n)]["structural"])
df["degree_sim_0.80"] = df["hub_node_id"].apply(lambda n: idx[str(n)]["sim_0.80"])
df["degree_sim_0.85"] = df["hub_node_id"].apply(lambda n: idx[str(n)]["sim_0.85"])
df["degree_sim_0.90"] = df["hub_node_id"].apply(lambda n: idx[str(n)]["sim_0.90"])
df["degree_sim_0.95"] = df["hub_node_id"].apply(lambda n: idx[str(n)]["sim_0.95"])
df["degree_total_0.80"] = df["hub_node_id"].apply(lambda n: idx[str(n)]["total_0.80"])
df["n_sources_at_config_thr"] = df.apply(
    lambda row: get_n_sources(row["hub_node_id"], row["edge_config"]), axis=1
)
df["sim_ratio_90_80"] = df.apply(
    lambda row: round(row["degree_sim_0.90"] / row["degree_sim_0.80"], 3)
    if row["degree_sim_0.80"] > 0
    else 0.0,
    axis=1,
)
df["is_high_thr_hub"] = df["degree_sim_0.90"] >= 50

for col in [
    "degree_structural",
    "degree_sim_0.80",
    "degree_sim_0.85",
    "degree_sim_0.90",
    "degree_sim_0.95",
    "degree_total_0.80",
    "n_sources_at_config_thr",
    "sim_ratio_90_80",
]:
    if col in df.columns:
        new_means[col] = df[col].mean()
        old_m = old_means.get(col, float("nan"))
        print(
            f"  {col:<30}: old mean={old_m:.2f}  new mean={new_means[col]:.2f}  "
            f"old max={df[col].max():.0f}"
        )

df.to_csv(hub_csv, index=False)
print(f"\n  Saved fully corrected {hub_csv.name}")


# ─── STEP 4: regenerate scatter plots ────────────────────────────────────────

for plot_name in ["hub_quality_scatter.png", "hub_quality_scatter_v2.png"]:
    backup(STEP2_DIR / plot_name, "_structonly")

print("\nRegenerating hub quality scatter plots …")

df_plot = (
    df[df["edge_config"] == PRIMARY_EDGE_CONFIG].copy()
    if "edge_config" in df.columns
    else df.copy()
)
if df_plot.empty:
    df_plot = df.copy()
df_plot = (
    df_plot[df_plot["mode"] == PRIMARY_MODE] if "mode" in df_plot.columns else df_plot
)
if df_plot.empty:
    df_plot = (
        df[df["edge_config"] == PRIMARY_EDGE_CONFIG].copy()
        if not df.empty
        else df.copy()
    )

# Scatter v2
fig, ax = plt.subplots(figsize=(10, 7))
if all(
    c in df_plot.columns
    for c in [
        "degree_sim_0.90",
        "n_sources_at_config_thr",
        "degree_total_0.80",
        "degree_structural",
    ]
):
    x = df_plot["degree_sim_0.90"].values
    y = df_plot["n_sources_at_config_thr"].values
    sizes = np.clip(df_plot["degree_total_0.80"].values / 10, 10, 300)
    colors = df_plot["degree_structural"].values

    sc = ax.scatter(
        x, y, s=sizes, c=colors, cmap="viridis", alpha=0.7, edgecolors="none"
    )
    plt.colorbar(
        sc, ax=ax, label="Structural degree (EDGE conf≥3, valid-pathway both endpoints)"
    )

    ax.axhline(10, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(50, color="blue", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("SIM≥0.9 degree (valid-pathway partner nodes only)")
    ax.set_ylabel("n_sources (distinct partner papers, valid-pathway only)")
    ax.set_title(
        "Hub Quality — SIM degree vs Source Diversity\n"
        f"(color=structural degree conf≥3 | config=SIM≥0.9 | n={len(df_plot):,})\n"
        "All counts restricted to valid-pathway node set"
    )

    legend_els = [
        mlines.Line2D([0], [0], color="red", linestyle="--", label="n_sources=10"),
        mlines.Line2D([0], [0], color="blue", linestyle="--", label="SIM degree=50"),
    ]
    ax.legend(handles=legend_els, fontsize=8)
else:
    ax.text(
        0.5,
        0.5,
        "Required columns not found",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

plt.tight_layout()
plt.savefig(STEP2_DIR / "hub_quality_scatter_v2.png", dpi=DPI)
plt.close()
print("  Saved hub_quality_scatter_v2.png")

# Scatter v1
fig, ax = plt.subplots(figsize=(10, 7))
if all(
    c in df_plot.columns
    for c in ["degree_sim_0.90", "n_sources_at_config_thr", "degree_structural"]
):
    sc = ax.scatter(
        df_plot["degree_sim_0.90"].values,
        df_plot["n_sources_at_config_thr"].values,
        c=df_plot["degree_structural"].values,
        cmap="plasma",
        alpha=0.6,
        s=40,
    )
    plt.colorbar(sc, ax=ax, label="Structural degree (conf≥3, valid-pathway)")
    ax.set_xlabel("SIM≥0.9 degree (valid-pathway partners only)")
    ax.set_ylabel("n_sources (valid-pathway partners only)")
    ax.set_title(
        "Hub Quality Scatter (all degrees restricted to valid-pathway node set)"
    )
plt.tight_layout()
plt.savefig(STEP2_DIR / "hub_quality_scatter.png", dpi=DPI)
plt.close()
print("  Saved hub_quality_scatter.png")


# ─── STEP 5: regenerate hub_quality_bar_v2.png ───────────────────────────────

backup(STEP2_DIR / "hub_quality_bar_v2.png", "_structonly")

print("\nRegenerating hub_quality_bar_v2.png …")
df_bar = df[
    (df["edge_config"].astype(str) == PRIMARY_EDGE_CONFIG)
    & (df["mode"] == PRIMARY_MODE)
].copy()

if df_bar.empty:
    print("  ⚠ No data for primary config — skipping bar chart")
else:
    corr = df_bar["degree_sim_0.90"].corr(df_bar["n_sources_at_config_thr"])
    n_equal = (df_bar["degree_sim_0.90"] == df_bar["n_sources_at_config_thr"]).sum()
    print(
        f"  degree_sim_0.90 == n_sources: {n_equal}/{len(df_bar)} cases (corr={corr:.4f})"
    )

    hub1_deg = int(df_bar["degree_sim_0.90"].max()) if not df_bar.empty else 0
    hub100_row = (
        df_bar.nlargest(100, "degree_sim_0.90").iloc[-1] if len(df_bar) >= 100 else None
    )
    hub100_deg = int(hub100_row["degree_sim_0.90"]) if hub100_row is not None else None

    df_bar = df_bar[df_bar["degree_sim_0.90"] > 0]
    print(f"  After dropping zero-degree: {len(df_bar):,} rows")

    TOP_N = 20
    plot_node_types = [
        nt for nt in NODE_TYPES_ORDER if nt in df_bar["node_type"].unique()
    ]
    n_types = len(plot_node_types)

    if n_types == 0:
        print("  ⚠ No node types found — skipping bar chart")
    else:
        fig, axes = plt.subplots(
            n_types, 1, figsize=(14, max(4, n_types * 3)), constrained_layout=True
        )
        if n_types == 1:
            axes = [axes]

        for ax, (node_type, color) in zip(axes, zip(plot_node_types, NODE_TYPE_COLORS)):
            nt_df = df_bar[df_bar["node_type"] == node_type].nlargest(
                TOP_N, "degree_sim_0.90"
            )
            if nt_df.empty:
                ax.set_visible(False)
                continue

            names = [str(n)[:70] for n in nt_df["hub_name"].values]
            degrees = nt_df["degree_sim_0.90"].values.astype(int)
            y_pos = np.arange(len(names))

            bars = ax.barh(
                y_pos,
                degrees,
                color=color,
                alpha=0.82,
                edgecolor="black",
                linewidth=0.4,
            )
            for bar, deg in zip(bars, degrees):
                ax.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{deg:,}",
                    va="center",
                    ha="left",
                    fontsize=8,
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("SIM≥0.9 Degree (valid-pathway partners only)", fontsize=9)
            ax.set_title(
                f"{NODE_TYPE_LABELS.get(node_type, node_type)}  —  top-{len(names)} hubs",
                fontsize=10,
                fontweight="bold",
                color=color,
                loc="left",
            )
            ax.axvline(
                50,
                color="green",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label="Strong hub (50)",
            )
            ax.grid(True, axis="x", alpha=0.25, linestyle=":")

        hub100_str = f"  |  Hub #100: {hub100_deg:,}" if hub100_deg else ""
        fig.suptitle(
            f"Hub Quality — Primary Analysis Cut: SIM≥0.9 + 'both' mode\n"
            f"Top-{TOP_N} hubs per node type by SIM≥0.9 degree  |  Hub #1: {hub1_deg:,}{hub100_str}\n"
            "All SIM degrees restricted to valid-pathway node partners",
            fontsize=11,
            fontweight="bold",
        )

        plt.savefig(STEP2_DIR / "hub_quality_bar_v2.png", dpi=DPI, bbox_inches="tight")
        plt.close()
        print("  Saved hub_quality_bar_v2.png")

print(
    "\nDone. hub_quality_metrics.csv all degree columns corrected (valid-pathway partner restriction)."
)
print("Backups: *_structonly.csv / *_structonly.png")
