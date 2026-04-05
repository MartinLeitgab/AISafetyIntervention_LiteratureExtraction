"""
Fix hub_quality_metrics.csv: recompute degree_structural using only
conf>=3 EDGE edges where BOTH endpoints are in the valid-pathway node set.

Prior version counted ALL EDGE edges from edge_data.pkl regardless of
confidence, overcounting structural degree.

Inputs:
  phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl
  phase2_results/step2_metrics_and_stability/hub_quality_metrics.csv
  phase1_rawpathsfiles/paths_*.jsonl  (defines valid-pathway node set)

Outputs:
  Backup existing:
    hub_quality_metrics.csv          → hub_quality_metrics_allconf.csv
    hub_quality_scatter.png          → hub_quality_scatter_allconf.png
    hub_quality_scatter_v2.png       → hub_quality_scatter_v2_allconf.png
  Write corrected:
    hub_quality_metrics.csv          (degree_structural column fixed)
    hub_quality_scatter_v2.png       (regenerated with corrected colors)

Run from graph_analysis/:
    python phase2_fix_hub_quality_degree.py > /tmp/hub_quality_fix.log 2>&1
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


def backup(src: Path, suffix: str):
    dst = src.with_name(src.stem + suffix + src.suffix)
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Backed up {src.name} → {dst.name}")
    elif dst.exists():
        print(f"  Backup already exists: {dst.name}")


# ─── STEP 1: build valid-pathway node set ────────────────────────────────────

print("Building valid-pathway node set from path files …")
valid_nodes: set = set()
for pf in sorted(PATHS_DIR.glob("paths_*.jsonl")):
    with open(pf) as f:
        for line in f:
            rec = json.loads(line)
            path = rec.get("path", [])
            if isinstance(path, str):
                path = json.loads(path)
            valid_nodes.update(str(n) for n in path)
print(f"  Valid-pathway nodes: {len(valid_nodes):,}")


# ─── STEP 2: load edge_data, build conf>=3 structural degree index ────────────

print("Loading edge_data.pkl …")
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
print(f"  Loaded {len(edge_data):,} edges")

print("Computing conf>=3 structural degree for valid-pathway nodes …")
deg_structural_conf3: dict = defaultdict(int)
n_kept, n_low_conf, n_invalid = 0, 0, 0
for e in edge_data:
    if str(e.get("type", "")).upper() != "EDGE":
        continue
    src, tgt = str(e.get("source", "")), str(e.get("target", ""))
    if src not in valid_nodes or tgt not in valid_nodes:
        n_invalid += 1
        continue
    conf = e.get("confidence")
    try:
        if float(conf) >= MIN_CONF:
            deg_structural_conf3[src] += 1
            deg_structural_conf3[tgt] += 1
            n_kept += 1
        else:
            n_low_conf += 1
    except (TypeError, ValueError):
        n_low_conf += 1

print(f"  EDGE edges kept (conf>={MIN_CONF}, both valid): {n_kept:,}")
print(f"  Dropped low-conf: {n_low_conf:,}  non-valid endpoint: {n_invalid:,}")


# ─── STEP 3: backup and update hub_quality_metrics.csv ───────────────────────

hub_csv = STEP2_DIR / "hub_quality_metrics.csv"
backup(hub_csv, "_allconf")

print(f"Loading {hub_csv.name} …")
df = pd.read_csv(hub_csv, dtype={"hub_node_id": str})
print(f"  {len(df):,} rows, columns: {list(df.columns)}")

if "degree_structural" not in df.columns:
    print("  WARNING: degree_structural column not found — nothing to fix")
else:
    old_vals = df["degree_structural"].copy()
    df["degree_structural"] = df["hub_node_id"].apply(
        lambda nid: deg_structural_conf3.get(str(nid), 0)
    )
    changed = (df["degree_structural"] != old_vals).sum()
    print(f"  degree_structural updated for {changed:,}/{len(df):,} rows")
    print(
        f"  Old mean: {old_vals.mean():.1f}  New mean: {df['degree_structural'].mean():.1f}"
    )
    print(
        f"  Old max:  {old_vals.max():.0f}  New max:  {df['degree_structural'].max():.0f}"
    )

df.to_csv(hub_csv, index=False)
print(f"  Saved corrected {hub_csv.name}")


# ─── STEP 4: regenerate scatter plots ────────────────────────────────────────

for plot_name in ["hub_quality_scatter.png", "hub_quality_scatter_v2.png"]:
    backup(STEP2_DIR / plot_name, "_allconf")

print("Regenerating hub quality scatter plots …")

# Filter to config = SIM>=0.9 (the primary config) for cleaner plot
df_plot = (
    df[df["edge_config"] == "0.9"].copy() if "edge_config" in df.columns else df.copy()
)
if df_plot.empty:
    df_plot = df.copy()

# Scatter v2: degree_sim_0.90 vs n_sources, size=degree_total_0.80, color=degree_structural
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
        sc, ax=ax, label="Structural degree (EDGE conf>=3, valid-pathway only)"
    )

    ax.axhline(10, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(50, color="blue", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("SIM>=0.9 degree (cross-paper similarity connections)")
    ax.set_ylabel("n_sources (distinct partner papers at config threshold)")
    ax.set_title(
        "Hub Quality — SIM degree vs Source Diversity\n"
        f"(color=structural degree conf>=3 | config=SIM>=0.9 | n={len(df_plot):,})"
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
plt.savefig(STEP2_DIR / "hub_quality_scatter_v2.png", dpi=120)
plt.close()
print("  Saved hub_quality_scatter_v2.png")

# Also regenerate simple scatter (v1 style: degree_sim_0.80 vs n_sources)
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
    plt.colorbar(sc, ax=ax, label="Structural degree (conf>=3)")
    ax.set_xlabel("SIM>=0.9 degree")
    ax.set_ylabel("n_sources")
    ax.set_title("Hub Quality Scatter (conf>=3 corrected structural degree)")
plt.tight_layout()
plt.savefig(STEP2_DIR / "hub_quality_scatter.png", dpi=120)
plt.close()
print("  Saved hub_quality_scatter.png")

print("\nDone. hub_quality_metrics.csv degree_structural column corrected.")
print(
    "Backups: hub_quality_metrics_allconf.csv, hub_quality_scatter_allconf.png, hub_quality_scatter_v2_allconf.png"
)
