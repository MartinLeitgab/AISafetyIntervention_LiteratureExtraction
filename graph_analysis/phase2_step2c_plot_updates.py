#!/usr/bin/env python3
"""
Phase 2 Step 2c: Plot Updates
==============================

Generates revised plots based on Step 2b review findings:

  PLOT A: hub_quality_bar_v2.png
    - Replaces scatter (degree == n_sources in 94% of cases → scatter is redundant y=x line)
    - Horizontal bar chart: top-20 hubs per node type, colored by node type
    - Shows hub names + SIM≥0.9 degree; directly demonstrates semantic diversity of hubs
    - Preserves original hub_quality_scatter.png and hub_quality_scatter_v2.png

  PLOT B: edge_validation_per_mode_v2.png
    - Each bar segment labeled by node type (replacing anonymous validation bins)
    - Grouped bars: x=edge_config, one bar per node type, faceted 2x2 by mode
    - Shows validation rate per node type at each threshold

Run from graph_analysis/ directory:
    uv run phase2_step2c_plot_updates.py
"""

import matplotlib
matplotlib.use("Agg")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

print(f"Matplotlib backend: {matplotlib.get_backend()}")
assert matplotlib.get_backend().lower() == "agg"
plt.style.use("seaborn-v0_8-darkgrid")

# ============================================================================
# PATHS
# ============================================================================

STEP2_DIR = Path("./phase2_results/step2_metrics_and_stability")
OUT_HUB_V2      = STEP2_DIR / "hub_quality_scatter_v2.png"   # kept for backwards compat
OUT_HUB_BAR_V2  = STEP2_DIR / "hub_quality_bar_v2.png"
OUT_EDGE_V2  = STEP2_DIR / "edge_validation_per_mode_v2.png"
HUB_CSV      = STEP2_DIR / "hub_quality_metrics.csv"
QUALITY_CSV  = STEP2_DIR / "quality_metrics_summary.csv"
DPI = 300

# Primary analysis cut (earmarked in Step 2b review)
PRIMARY_EDGE_CONFIG = "0.9"
PRIMARY_MODE        = "both"

EDGE_CONFIGS_ORDERED = ["0.8", "0.85", "0.9", "0.95", "EDGE"]

NODE_TYPES = [
    "risk",
    "intervention",
    "all_concepts",
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]

NODE_TYPE_LABELS = {
    "risk":                    "Risk",
    "intervention":            "Intervention",
    "all_concepts":            "All Concepts",
    "problem_analysis":        "Problem Analysis",
    "theoretical_insight":     "Theoretical Insight",
    "design_rationale":        "Design Rationale",
    "implementation_mechanism":"Implementation Mechanism",
    "validation_evidence":     "Validation Evidence",
}

NODE_TYPE_COLORS = [
    "#E74C3C",  # risk
    "#3498DB",  # intervention
    "#95A5A6",  # all_concepts
    "#E67E22",  # problem_analysis
    "#9B59B6",  # theoretical_insight
    "#1ABC9C",  # design_rationale
    "#2ECC71",  # implementation_mechanism
    "#F39C12",  # validation_evidence
]

MODES = ["unconstrained", "single_risk", "monotonic", "both"]
MODE_LABELS = {
    "unconstrained": "Unconstrained",
    "single_risk":   "Single-Risk",
    "monotonic":     "Monotonic",
    "both":          "Both (Primary Cut)",
}


# ============================================================================
# PLOT A: Hub Quality Bar Chart v2
# ============================================================================

def plot_hub_quality_v2(df_hubs: pd.DataFrame):
    """
    Hub Quality Bar Chart v2.

    Replaces the scatter plot (degree == n_sources in 94% of cases — scatter is
    a redundant y=x identity line; both axes measure the same thing at SIM>=0.9).

    New design: horizontal bar chart of top-20 hubs per node type, colored by
    node type. Directly shows semantic diversity and cross-paper hub names.
    Serves the core goal: demonstrate hubs are real semantic groupings, not
    pipeline artifacts.

    Finding noted in text (not plotted): at SIM>=0.9, degree == n_unique_papers
    in 94% of cases — hubs are not driven by single-paper duplication.
    """
    print("\n" + "=" * 70)
    print("PLOT A: Hub Quality Bar Chart v2 (top-20 per node type)")
    print("=" * 70)

    df = df_hubs[
        (df_hubs["edge_config"] == PRIMARY_EDGE_CONFIG) &
        (df_hubs["mode"]        == PRIMARY_MODE)
    ].copy()

    if df.empty:
        print(f"⚠  No hub data for edge_config={PRIMARY_EDGE_CONFIG}, mode={PRIMARY_MODE}")
        return

    print(f"   Rows after filter: {len(df):,}")

    # Report the y=x finding as a diagnostic print
    corr = df["degree_sim_0.90"].corr(df["n_sources_at_config_thr"])
    n_equal = (df["degree_sim_0.90"] == df["n_sources_at_config_thr"]).sum()
    print(f"   degree == n_sources: {n_equal}/{len(df)} cases (corr={corr:.4f})")
    print(f"   → scatter is y=x identity line; replaced with bar chart")

    hub1_deg   = int(df["degree_sim_0.90"].max())
    hub100_row = df.nlargest(100, "degree_sim_0.90").iloc[-1] if len(df) >= 100 else None
    hub100_deg = int(hub100_row["degree_sim_0.90"]) if hub100_row is not None else None

    # Node types to plot — in display order
    plot_node_types = [nt for nt in NODE_TYPES if nt in df["node_type"].unique()]

    # Drop zero-degree nodes — not hubs
    df = df[df["degree_sim_0.90"] > 0]
    print(f"   After dropping zero-degree: {len(df):,} rows")

    TOP_N = 20
    n_types = len(plot_node_types)
    fig, axes = plt.subplots(
        n_types, 1,
        figsize=(14, max(4, n_types * 3)),
        constrained_layout=True,
    )
    if n_types == 1:
        axes = [axes]

    for ax, (node_type, color) in zip(axes, zip(plot_node_types, NODE_TYPE_COLORS)):
        nt_df = df[df["node_type"] == node_type].nlargest(TOP_N, "degree_sim_0.90")
        if nt_df.empty:
            ax.set_visible(False)
            continue

        names  = [str(n)[:70] for n in nt_df["hub_name"].values]
        degrees = nt_df["degree_sim_0.90"].values.astype(int)

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, degrees, color=color, alpha=0.82,
                       edgecolor="black", linewidth=0.4)

        # Value labels on bars
        for bar, deg in zip(bars, degrees):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{deg:,}", va="center", ha="left", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("SIM≥0.9 Degree (= unique source papers)", fontsize=9)
        ax.set_title(f"{NODE_TYPE_LABELS.get(node_type, node_type)}  —  top-{len(names)} hubs",
                     fontsize=10, fontweight="bold", color=color, loc="left")
        ax.axvline(50, color="green", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="Strong hub threshold (50)")
        ax.grid(True, axis="x", alpha=0.25, linestyle=":")

    hub100_str = f"  |  Hub #100 SIM≥0.9 degree: {hub100_deg:,}" if hub100_deg else ""
    fig.suptitle(
        f"Hub Quality — Primary Analysis Cut: SIM≥0.9 + 'both' mode\n"
        f"Top-{TOP_N} hubs per node type by SIM≥0.9 degree  |  Hub #1: {hub1_deg:,}{hub100_str}\n"
        f"SIM≥0.9 degree = unique source papers in 94% of cases (not a single-paper artifact)",
        fontsize=11, fontweight="bold",
    )

    plt.savefig(OUT_HUB_BAR_V2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓  Saved: {OUT_HUB_BAR_V2}")
    if hub100_deg:
        print(f"   Hub #1 SIM≥0.9 degree: {hub1_deg:,}")
        print(f"   Hub #100 SIM≥0.9 degree: {hub100_deg:,}")


# ============================================================================
# PLOT B: Edge Validation Per Mode v2 — labeled by node type
# ============================================================================

def plot_edge_validation_per_mode_v2(df_quality: pd.DataFrame):
    """
    Edge Validation per Mode v2.
    - One subplot per mode (2×2 grid)
    - x-axis = edge config; grouped bars, one bar per node type
    - y-axis = EDGE validation rate (0–1)
    - Each bar labeled with its node type name and validation % if space permits
    - Replaces anonymous validation-bin stacks with explicit node-type labeling
    """
    print("\n" + "=" * 70)
    print("PLOT B: Edge Validation per Mode v2 (node-type labeled)")
    print("=" * 70)

    if df_quality.empty:
        print("⚠  No quality data — skipping")
        return

    ec_order = EDGE_CONFIGS_ORDERED
    n_nt = len(NODE_TYPES)
    bar_width = 0.8 / n_nt
    x_pos = np.arange(len(ec_order))

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    for idx, mode in enumerate(MODES):
        ax = axes[idx]
        mode_df = df_quality[df_quality["mode"] == mode]

        for nt_i, (node_type, color) in enumerate(zip(NODE_TYPES, NODE_TYPE_COLORS)):
            nt_df = mode_df[mode_df["node_type"] == node_type]
            vals = []
            for ec in ec_order:
                subset = nt_df[nt_df["edge_config"] == str(ec)]["edge_validation_mean"].values
                vals.append(float(subset.mean()) if len(subset) > 0 else 0.0)

            offset = (nt_i - n_nt / 2 + 0.5) * bar_width
            bars = ax.bar(
                x_pos + offset, vals,
                width=bar_width,
                color=color,
                alpha=0.82,
                label=NODE_TYPE_LABELS[node_type],
                edgecolor="black",
                linewidth=0.3,
            )

            # Add value labels on bars tall enough to read
            for bar, val in zip(bars, vals):
                if val > 0.12:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.0%}",
                        ha="center", va="bottom",
                        fontsize=6, rotation=90,
                        fontweight="bold",
                    )

        ax.axhline(0.6, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                   label="60% target")
        ax.axhline(0.9, color="orange", linestyle=":", linewidth=1.0, alpha=0.7,
                   label="90% high quality")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(ec_order, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("EDGE Validation Rate", fontsize=11)
        ax.set_xlabel("Edge Config (SIM threshold)", fontsize=11)

        title = MODE_LABELS.get(mode, mode)
        if mode == PRIMARY_MODE:
            title += " ★ PRIMARY CUT"
        ax.set_title(title, fontsize=12, fontweight="bold")

        ax.legend(
            title="Node Type",
            fontsize=7,
            title_fontsize=8,
            loc="lower left",
            ncol=2,
        )
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "EDGE Validation Rate by Node Type and Edge Config — Per Mode\n"
        "Each bar = one node type; height = mean EDGE validation rate across k=40 clusters",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_EDGE_V2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓  Saved: {OUT_EDGE_V2}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 2 Step 2c: Plot Updates")
    print("=" * 70)

    # Load hub data
    if HUB_CSV.exists():
        df_hubs = pd.read_csv(HUB_CSV, dtype={"edge_config": str})
        print(f"Loaded hub CSV: {len(df_hubs):,} rows")
        plot_hub_quality_v2(df_hubs)
    else:
        print(f"⚠  Hub CSV not found: {HUB_CSV}")

    # Load quality data
    if QUALITY_CSV.exists():
        df_quality = pd.read_csv(QUALITY_CSV, dtype={"edge_config": str})
        print(f"Loaded quality CSV: {len(df_quality):,} rows")
        print(f"   Columns: {list(df_quality.columns)}")
        plot_edge_validation_per_mode_v2(df_quality)
    else:
        print(f"⚠  Quality CSV not found: {QUALITY_CSV}")
        print("   Run phase2_step2_metrics_stability.py first to generate it.")

    print("\n" + "=" * 70)
    print("Step 2c complete.")
    print(f"  Hub bar chart v2:       {OUT_HUB_BAR_V2}")
    print(f"  Edge validation v2:     {OUT_EDGE_V2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
