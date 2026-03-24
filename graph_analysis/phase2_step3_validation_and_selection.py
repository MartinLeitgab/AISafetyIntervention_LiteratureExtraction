#!/usr/bin/env python3
"""
Phase 2 Step 3: Validation & Config Selection
=============================================

Sections:
  A  — #23 Multi-criteria scoring (CRITICAL, CSV-only, ~2 min)
  B  — #21 Edge threshold sensitivity (ESSENTIAL, CSV-only, ~1 min)
  C  — #22 EDGE-only baseline validation (ESSENTIAL, PKL, ~5 min)
  D  — Betweenness on SIM>=0.9 graph (ENRICHMENT, PKL, ~20 min, optional)
  E  — #24 Held-out validation (ENRICHMENT, PKL, ~3 min)
  F  — #25 EDGE subgraph consistency (ENRICHMENT, PKL, ~3 min)

Run from graph_analysis/ directory:
    uv run phase2_step3_validation_and_selection.py [--skip-betweenness]
"""

import matplotlib

matplotlib.use("Agg")

import argparse
import json
import pickle
import random
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(".")
STEP1_DIR = (
    SCRIPT_DIR / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
)
STEP2_DIR = SCRIPT_DIR / "phase2_results/step2_metrics_and_stability"
STEP3_DIR = SCRIPT_DIR / "phase2_results/step3_validation_and_selection"
PATHS_DIR = SCRIPT_DIR / "phase1_rawpathsfiles"
STEP3_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
# Ascending selectivity order (low→high)
SELECTIVITY_ORDER = ["0.8", "0.85", "0.9", "0.95", "EDGE"]
MODES = ["unconstrained", "single_risk", "monotonic", "both"]
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
PRIMARY_CUT_EC = "0.9"
PRIMARY_CUT_MODE = "both"

INTERPRETABILITY = {
    "risk": 1.0,
    "intervention": 1.0,
    "design_rationale": 1.0,
    "implementation_mechanism": 1.0,
    "problem_analysis": 1.0,
    "theoretical_insight": 1.0,
    "validation_evidence": 1.0,
    "all_concepts": 0.5,
}

plt.style.use("seaborn-v0_8-darkgrid")


# ─── Helpers ──────────────────────────────────────────────────────────────────


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def normalize_series(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0 + 1.0
    return (s - mn) / (mx - mn)


def triangular_cluster_score(k):
    """Peak 1.0 at k=40–50, linear decay to 0 at k<=20 or k>=80."""
    if k <= 20 or k >= 80:
        return 0.0
    if k <= 40:
        return (k - 20) / 20.0
    if k <= 50:
        return 1.0
    return (80 - k) / 30.0


def get_pairwise_ari(df_pairwise, node_type, mode, t1, t2):
    """Look up ARI between two thresholds, regardless of column order."""
    t1s, t2s = str(t1), str(t2)
    mask = (
        (df_pairwise["node_type"] == node_type)
        & (df_pairwise["mode"] == mode)
        & (
            (
                (df_pairwise["threshold_1"].astype(str) == t1s)
                & (df_pairwise["threshold_2"].astype(str) == t2s)
            )
            | (
                (df_pairwise["threshold_1"].astype(str) == t2s)
                & (df_pairwise["threshold_2"].astype(str) == t1s)
            )
        )
    )
    rows = df_pairwise[mask]
    return rows["ari"].iloc[0] if len(rows) > 0 else np.nan


def get_ari_high(df_pairwise, node_type, mode, edge_config):
    """Mean ARI from edge_config to all configs with higher selectivity."""
    ec = str(edge_config)
    idx = SELECTIVITY_ORDER.index(ec) if ec in SELECTIVITY_ORDER else -1
    if idx < 0:
        return np.nan
    higher = SELECTIVITY_ORDER[idx + 1 :]
    if not higher:
        # EDGE is most selective — use nearest neighbor
        higher = [SELECTIVITY_ORDER[-2]]
    aris = [get_pairwise_ari(df_pairwise, node_type, mode, ec, t) for t in higher]
    aris = [a for a in aris if not np.isnan(a)]
    return float(np.mean(aris)) if aris else np.nan


def load_pkl(path):
    print(f"  Loading {path.name} ...", flush=True)
    with open(path, "rb") as f:
        return pickle.load(f)


def get_assignments(cluster_memberships, ec, mode, node_type, algo="agglomerative"):
    """Return {node_id: cluster_id} for a specific config."""
    assign = {}
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            k_ec, k_mode, k_nt, k_algo, cid = key
            if (
                str(k_ec) == str(ec)
                and k_mode == mode
                and k_nt == node_type
                and k_algo == algo
            ):
                for nid in members:
                    assign[nid] = cid
    return assign


# ─── SECTION A: Multi-Criteria Scoring (#23) ─────────────────────────────────


def run_section_a(df_quality, df_purity, df_pairwise):
    print("\n" + "=" * 70)
    print("SECTION A — #23 Multi-Criteria Scoring")
    print("=" * 70)

    # Gold purity fraction per config
    gold_pct = (
        df_purity.groupby(["edge_config", "mode", "node_type"])["is_gold_standard"]
        .mean()
        .reset_index()
        .rename(columns={"is_gold_standard": "gold_pct"})
    )

    rows = []
    for _, row in df_quality.iterrows():
        ec = str(row["edge_config"])
        mode = row["mode"]
        nt = row["node_type"]

        silhouette = row["silhouette_mean"]
        edge_pct = row["edge_validation_mean"]
        n_clusters = row["n_clusters"]
        cluster_count_score = triangular_cluster_score(n_clusters)
        ari_high = get_ari_high(df_pairwise, nt, mode, ec)
        interp = INTERPRETABILITY.get(nt, 0.5)

        gp_mask = (
            (gold_pct["edge_config"] == ec)
            & (gold_pct["mode"] == mode)
            & (gold_pct["node_type"] == nt)
        )
        gold_pct_val = gold_pct[gp_mask]["gold_pct"].values
        gold_pct_raw = float(gold_pct_val[0]) if len(gold_pct_val) > 0 else np.nan

        rows.append(
            dict(
                edge_config=ec,
                mode=mode,
                node_type=nt,
                n_clusters=n_clusters,
                silhouette=silhouette,
                edge_pct=edge_pct,
                cluster_count_score=cluster_count_score,
                ari_high=ari_high,
                gold_pct=gold_pct_raw,
                interpretability=interp,
            )
        )

    df = pd.DataFrame(rows)

    # Normalize per node_type, compute composite
    df["sil_norm"] = np.nan
    df["edge_norm"] = np.nan
    df["cc_norm"] = np.nan
    df["ari_norm"] = np.nan
    df["gold_norm"] = np.nan

    for nt in df["node_type"].unique():
        mask = df["node_type"] == nt
        df.loc[mask, "sil_norm"] = normalize_series(df.loc[mask, "silhouette"])
        df.loc[mask, "edge_norm"] = normalize_series(df.loc[mask, "edge_pct"])
        df.loc[mask, "cc_norm"] = normalize_series(df.loc[mask, "cluster_count_score"])
        df.loc[mask, "ari_norm"] = normalize_series(df.loc[mask, "ari_high"])
        df.loc[mask, "gold_norm"] = normalize_series(df.loc[mask, "gold_pct"])

    df["composite"] = (
        0.25 * df["sil_norm"]
        + 0.30 * df["edge_norm"]
        + 0.20 * df["cc_norm"]
        + 0.15 * df["ari_norm"]
        + 0.10 * df["gold_norm"]
    )

    # Rank within node_type
    df["rank"] = (
        df.groupby("node_type")["composite"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    df_sorted = df.sort_values(["node_type", "rank"])
    df_sorted.to_csv(STEP3_DIR / "optimal_configs_ranked.csv", index=False)
    print(f"  Saved optimal_configs_ranked.csv ({len(df_sorted)} rows)")

    # Winner per node_type
    winners = df_sorted[df_sorted["rank"] == 1].copy()
    winners.to_csv(STEP3_DIR / "optimal_configs_final.csv", index=False)
    print(f"  Saved optimal_configs_final.csv ({len(winners)} rows)")

    # Print results
    print("\n  Winner per node_type:")
    for _, w in winners.iterrows():
        flag = ""
        if w["node_type"] in ("risk", "intervention"):
            expected = (
                w["edge_config"] == PRIMARY_CUT_EC and w["mode"] == PRIMARY_CUT_MODE
            )
            flag = (
                " ✓ CONFIRMED"
                if expected
                else f" ✗ DIFFERS from earmarked {PRIMARY_CUT_EC}+{PRIMARY_CUT_MODE}"
            )
        print(
            f"    {w['node_type']:30s}  edge_config={w['edge_config']:5s}  mode={w['mode']:15s}"
            f"  composite={w['composite']:.3f}  rank=1{flag}"
        )

    # Parallel coordinates plot
    _plot_parallel_coords(df_sorted)

    # Selection justification
    _write_selection_justification(winners, df_sorted)

    return df_sorted, winners


def _plot_parallel_coords(df):
    metric_cols = ["sil_norm", "edge_norm", "cc_norm", "ari_norm", "gold_norm"]
    metric_labels = [
        "Silhouette\n(0.25)",
        "EDGE%\n(0.30)",
        "Cluster\nCount\n(0.20)",
        "ARI\nHigh\n(0.15)",
        "Gold\nPurity\n(0.10)",
    ]

    focus_node_types = ["risk", "intervention", "all_concepts", "design_rationale"]
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(focus_node_types)))

    fig, axes = plt.subplots(1, len(focus_node_types), figsize=(20, 6))
    if len(focus_node_types) == 1:
        axes = [axes]

    for ax, nt, color in zip(axes, focus_node_types, colors):
        sub = df[df["node_type"] == nt]
        top10 = sub.nsmallest(10, "rank")

        for _, row in sub.iterrows():
            vals = [row[c] for c in metric_cols]
            ax.plot(range(len(metric_cols)), vals, color="lightgrey", alpha=0.4, lw=0.8)

        for _, row in top10.iterrows():
            vals = [row[c] for c in metric_cols]
            alpha = 1.0 if row["rank"] == 1 else 0.6
            lw = 2.5 if row["rank"] == 1 else 1.2
            label = f"{row['edge_config']}+{row['mode']}" if row["rank"] == 1 else None
            ax.plot(
                range(len(metric_cols)),
                vals,
                color=color,
                alpha=alpha,
                lw=lw,
                label=label,
            )

        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels(metric_labels, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{nt}\n(n=160 configs)", fontsize=10)
        ax.set_ylabel("Normalized [0,1]" if ax == axes[0] else "")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Multi-Criteria Config Selection — Parallel Coordinates\n(grey=all 160, colored=top-10 per node_type)",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(STEP3_DIR / "multi_criteria_parallel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved multi_criteria_parallel.png")


def _write_selection_justification(winners, df_full):
    lines = [
        "# Config Selection Justification",
        "",
        "**Method:** 5-criteria weighted composite score applied to all 160 configurations.",
        "**Weights:** EDGE validation 30%, Silhouette 25%, Cluster count 20%, ARI stability 15%, Gold purity (via interpretability proxy) 10%.",
        "**Normalization:** Per node_type min-max (not global), ensuring fair comparison within each node category.",
        "",
        "## Winner Per Node Type",
        "",
        "| node_type | edge_config | mode | composite | rank | earmarked? |",
        "|-----------|-------------|------|-----------|------|------------|",
    ]
    for _, w in winners.iterrows():
        earmarked = "n/a"
        if w["node_type"] in ("risk", "intervention"):
            earmarked = (
                "CONFIRMED"
                if (
                    w["edge_config"] == PRIMARY_CUT_EC and w["mode"] == PRIMARY_CUT_MODE
                )
                else "UPDATED"
            )
        lines.append(
            f"| {w['node_type']} | {w['edge_config']} | {w['mode']} | {w['composite']:.3f} | 1 | {earmarked} |"
        )

    lines += [
        "",
        "## Primary Analysis Cut Decision",
        "",
    ]

    risk_winner = winners[winners["node_type"] == "risk"]
    int_winner = winners[winners["node_type"] == "intervention"]

    if len(risk_winner) > 0 and len(int_winner) > 0:
        rw = risk_winner.iloc[0]
        iw = int_winner.iloc[0]
        if rw["edge_config"] == PRIMARY_CUT_EC and iw["edge_config"] == PRIMARY_CUT_EC:
            lines.append(
                f"The earmarked primary cut (**SIM≥0.9, mode=both**) is **CONFIRMED** for both risk and intervention "
                f"node types (composite scores: risk={rw['composite']:.3f}, intervention={iw['composite']:.3f})."
            )
        else:
            lines.append(
                "The earmarked primary cut (SIM≥0.9, mode=both) is **UPDATED** for one or more node types."
            )
            lines.append(
                f"Risk winner: edge_config={rw['edge_config']}, mode={rw['mode']} (composite={rw['composite']:.3f})"
            )
            lines.append(
                f"Intervention winner: edge_config={iw['edge_config']}, mode={iw['mode']} (composite={iw['composite']:.3f})"
            )

    lines += [
        "",
        "## Top-3 Per Node Type (risk and intervention)",
        "",
        "| node_type | rank | edge_config | mode | composite | silhouette | edge_pct | ari_high |",
        "|-----------|------|-------------|------|-----------|------------|----------|----------|",
    ]
    for nt in ["risk", "intervention"]:
        top3 = df_full[df_full["node_type"] == nt].nsmallest(3, "rank")
        for _, r in top3.iterrows():
            lines.append(
                f"| {r['node_type']} | {r['rank']} | {r['edge_config']} | {r['mode']} "
                f"| {r['composite']:.3f} | {r['silhouette']:.3f} | {r['edge_pct']:.3f} | {r['ari_high']:.3f} |"
            )

    lines += [
        "",
        "## Workshop Paper Methods Text",
        "",
        "Configuration selection was determined by a 5-criteria weighted composite score "
        "(EDGE validation 30%, silhouette 25%, cluster count 20%, ARI stability 15%, gold purity 10%) "
        "applied to all 160 configurations (5 edge thresholds × 4 modes × 8 node types). "
        "See `optimal_configs_ranked.csv` for full rankings and `multi_criteria_parallel.png` for visualization.",
    ]

    out = STEP3_DIR / "selection_justification.md"
    out.write_text("\n".join(lines))
    print("  Saved selection_justification.md")


# ─── SECTION B: Threshold Sensitivity (#21) ──────────────────────────────────


def run_section_b(df_quality, df_purity, df_pairwise, df_centroid):
    print("\n" + "=" * 70)
    print("SECTION B — #21 Edge Threshold Sensitivity")
    print("=" * 70)

    # Gold pct per config
    gold_cfg = (
        df_purity.groupby(["edge_config", "mode", "node_type"])["is_gold_standard"]
        .mean()
        .reset_index()
        .rename(columns={"is_gold_standard": "gold_pct"})
    )

    adjacent_pairs = [
        ("0.8", "0.85"),
        ("0.85", "0.9"),
        ("0.9", "0.95"),
        ("0.95", "EDGE"),
    ]

    results = []
    for nt in NODE_TYPES:
        for mode in MODES:
            # stability_score per threshold: mean ARI to all higher thresholds
            for ec in SELECTIVITY_ORDER:
                stability = get_ari_high(df_pairwise, nt, mode, ec)
                results.append(
                    dict(
                        node_type=nt,
                        mode=mode,
                        edge_config=ec,
                        metric="stability_score",
                        value=stability,
                        threshold_pair="N/A",
                    )
                )

            # Δ metrics for adjacent pairs
            for t1, t2 in adjacent_pairs:
                pair_label = f"{t1}→{t2}"

                delta_ari = get_pairwise_ari(df_pairwise, nt, mode, t1, t2)

                def _sil(ec):
                    r = df_quality[
                        (df_quality["edge_config"] == ec)
                        & (df_quality["mode"] == mode)
                        & (df_quality["node_type"] == nt)
                    ]
                    return r["silhouette_mean"].values[0] if len(r) > 0 else np.nan

                def _edge_pct(ec):
                    r = df_quality[
                        (df_quality["edge_config"] == ec)
                        & (df_quality["mode"] == mode)
                        & (df_quality["node_type"] == nt)
                    ]
                    return r["edge_validation_mean"].values[0] if len(r) > 0 else np.nan

                def _gold(ec):
                    r = gold_cfg[
                        (gold_cfg["edge_config"] == ec)
                        & (gold_cfg["mode"] == mode)
                        & (gold_cfg["node_type"] == nt)
                    ]
                    return r["gold_pct"].values[0] if len(r) > 0 else np.nan

                def _centroid(t_from, t_to):
                    r = df_centroid[
                        (df_centroid["node_type"] == nt)
                        & (df_centroid["mode"] == mode)
                        & (df_centroid["threshold_from"].astype(str) == str(t_from))
                        & (df_centroid["threshold_to"].astype(str) == str(t_to))
                    ]
                    return r["centroid_sim_mean"].values[0] if len(r) > 0 else np.nan

                delta_sil = _sil(t2) - _sil(t1)
                delta_edge = _edge_pct(t2) - _edge_pct(t1)
                delta_gold = _gold(t2) - _gold(t1)
                # Centroid sim at this transition
                cent_sim = _centroid(t1, t2)
                if np.isnan(cent_sim):
                    cent_sim = _centroid(t2, t1)

                results.append(
                    dict(
                        node_type=nt,
                        mode=mode,
                        edge_config=t1,
                        metric="delta_ari",
                        value=delta_ari,
                        threshold_pair=pair_label,
                    )
                )
                results.append(
                    dict(
                        node_type=nt,
                        mode=mode,
                        edge_config=t1,
                        metric="delta_silhouette",
                        value=delta_sil,
                        threshold_pair=pair_label,
                    )
                )
                results.append(
                    dict(
                        node_type=nt,
                        mode=mode,
                        edge_config=t1,
                        metric="delta_edge_pct",
                        value=delta_edge,
                        threshold_pair=pair_label,
                    )
                )
                results.append(
                    dict(
                        node_type=nt,
                        mode=mode,
                        edge_config=t1,
                        metric="delta_gold_purity",
                        value=delta_gold,
                        threshold_pair=pair_label,
                    )
                )
                results.append(
                    dict(
                        node_type=nt,
                        mode=mode,
                        edge_config=t1,
                        metric="centroid_sim",
                        value=cent_sim,
                        threshold_pair=pair_label,
                    )
                )

    df_sens = pd.DataFrame(results)
    df_sens.to_csv(STEP3_DIR / "threshold_sensitivity_analysis.csv", index=False)
    print(f"  Saved threshold_sensitivity_analysis.csv ({len(df_sens)} rows)")

    _plot_threshold_sensitivity(df_quality, df_pairwise, df_sens)
    return df_sens


def _plot_threshold_sensitivity(df_quality, df_pairwise, df_sens):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    focus_nts = ["risk", "intervention"]
    focus_mode = "both"
    colors_nt = {"risk": "#e74c3c", "intervention": "#3498db"}

    adj_pairs = [("0.8", "0.85"), ("0.85", "0.9"), ("0.9", "0.95"), ("0.95", "EDGE")]
    pair_labels = ["0.8→0.85", "0.85→0.9", "0.9→0.95", "0.95→EDGE"]

    # Panel A: metric line plots vs threshold
    ax = axes[0]
    for nt in focus_nts:
        sub = df_quality[
            (df_quality["mode"] == focus_mode) & (df_quality["node_type"] == nt)
        ]
        sil_vals = [
            sub[sub["edge_config"] == ec]["silhouette_mean"].values[0]
            if len(sub[sub["edge_config"] == ec]) > 0
            else np.nan
            for ec in SELECTIVITY_ORDER
        ]
        edge_vals = [
            sub[sub["edge_config"] == ec]["edge_validation_mean"].values[0]
            if len(sub[sub["edge_config"] == ec]) > 0
            else np.nan
            for ec in SELECTIVITY_ORDER
        ]
        ax.plot(
            range(5),
            sil_vals,
            "o-",
            color=colors_nt[nt],
            label=f"{nt} silhouette",
            lw=2,
        )
        ax.plot(
            range(5),
            edge_vals,
            "s--",
            color=colors_nt[nt],
            label=f"{nt} EDGE%",
            lw=1.5,
            alpha=0.7,
        )

    ax.axvspan(2, 4, alpha=0.12, color="green", label="Stable regime (0.9–EDGE)")
    ax.set_xticks(range(5))
    ax.set_xticklabels(SELECTIVITY_ORDER)
    ax.set_xlabel("Edge config (ascending selectivity)")
    ax.set_ylabel("Value")
    ax.set_title("Panel A: Metric profiles vs threshold (mode=both)")
    ax.legend(fontsize=8, ncol=2)

    # Panel B: Δ metric bar chart per adjacent pair
    ax = axes[1]
    x = np.arange(len(adj_pairs))
    width = 0.35
    for i, nt in enumerate(focus_nts):
        delta_aris = []
        for t1, t2 in adj_pairs:
            val = get_pairwise_ari(df_pairwise, nt, focus_mode, t1, t2)
            delta_aris.append(val if not np.isnan(val) else 0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, delta_aris, width, label=nt, color=colors_nt[nt], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=15)
    ax.set_ylabel("ARI")
    ax.set_title(
        "Panel B: ARI per adjacent threshold pair (mode=both)\n(higher = more stable transition)"
    )
    ax.axvline(1.5, color="green", linestyle="--", alpha=0.6, label="0.9 boundary")
    ax.legend(fontsize=9)

    # Panel C: stability_score per threshold
    ax = axes[2]
    for nt in focus_nts:
        scores = []
        for ec in SELECTIVITY_ORDER:
            val = get_ari_high(df_pairwise, nt, focus_mode, ec)
            scores.append(val if not np.isnan(val) else 0)
        ax.plot(range(5), scores, "o-", color=colors_nt[nt], label=nt, lw=2)

    ax.axvspan(2, 4, alpha=0.12, color="green")
    ax.set_xticks(range(5))
    ax.set_xticklabels(SELECTIVITY_ORDER)
    ax.set_xlabel("Edge config")
    ax.set_ylabel("Stability score (mean ARI to higher thresholds)")
    ax.set_title("Panel C: Stability score per threshold")
    ax.legend()

    # Panel D: ARI heatmap for risk + intervention combined
    ax = axes[3]
    # Build 5x5 ARI matrix for risk/both
    nt_heat = "risk"
    mode_heat = "both"
    ari_mat = np.full((5, 5), np.nan)
    for i, t1 in enumerate(SELECTIVITY_ORDER):
        for j, t2 in enumerate(SELECTIVITY_ORDER):
            if i != j:
                ari_mat[i, j] = get_pairwise_ari(
                    df_pairwise, nt_heat, mode_heat, t1, t2
                )
    im = ax.imshow(ari_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(SELECTIVITY_ORDER)
    ax.set_yticklabels(SELECTIVITY_ORDER)
    plt.colorbar(im, ax=ax)
    for i in range(5):
        for j in range(5):
            if not np.isnan(ari_mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{ari_mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if ari_mat[i, j] > 0.4 else "white",
                )
    ax.set_title(f"Panel D: ARI heatmap (node_type={nt_heat}, mode={mode_heat})")

    plt.suptitle("Edge Threshold Sensitivity Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(
        STEP3_DIR / "threshold_sensitivity_profile.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved threshold_sensitivity_profile.png")


# ─── SECTION C: EDGE-Only Baseline Validation (#22) ──────────────────────────


def run_section_c(
    df_quality, df_purity, df_pairwise, cluster_memberships, node_attrs, edge_data
):
    print("\n" + "=" * 70)
    print("SECTION C — #22 EDGE-Only Baseline Validation")
    print("=" * 70)

    # Test 6: ARI between EDGE-only and SIM>=0.9 (from CSV, no PKL)
    print("\n  Test 6: ARI overlap EDGE↔SIM0.9 ...")
    test6_rows = []
    for nt in NODE_TYPES:
        for mode in MODES:
            ari = get_pairwise_ari(df_pairwise, nt, mode, "EDGE", "0.9")
            test6_rows.append(
                dict(
                    node_type=nt,
                    mode=mode,
                    ari_edge_vs_09=ari,
                    meets_target=ari > 0.5 if not np.isnan(ari) else False,
                )
            )
    df_t6 = pd.DataFrame(test6_rows)
    print(
        f"    risk/both ARI(EDGE↔0.9): {df_t6[(df_t6['node_type'] == 'risk') & (df_t6['mode'] == 'both')]['ari_edge_vs_09'].values}"
    )

    # Test 7: EDGE-only pathway sampling
    print("\n  Test 7: Sampling 100 EDGE-only pathways ...")
    test_set = _sample_edge_pathways(node_attrs, cluster_memberships)
    jsonl_path = STEP3_DIR / "edge_only_test_set.jsonl"
    with open(jsonl_path, "w") as f:
        for record in test_set:
            f.write(json.dumps(record) + "\n")
    print(f"    Saved {len(test_set)} pathways to edge_only_test_set.jsonl")

    # Test 8: SIM-only node coverage
    print("\n  Test 8: SIM-only node coverage ...")
    t8_results = _analyze_sim_coverage(
        node_attrs, edge_data, cluster_memberships, df_purity
    )

    # Build comparison table
    _build_comparison_table(df_quality, df_purity, df_t6, t8_results, df_pairwise)

    return df_t6, t8_results


def _sample_edge_pathways(node_attrs, cluster_memberships):
    """Sample 100 EDGE-only pathways stratified by intervention lifecycle stage."""
    # Get nodes in EDGE-only configs
    edge_nodes = set()
    for key, members in cluster_memberships.items():
        if len(key) == 5 and str(key[0]) == "EDGE" and key[2] == "intervention":
            edge_nodes.update(members)

    # Load path file
    path_file = PATHS_DIR / "paths_both_edge_only.jsonl"
    if not path_file.exists():
        path_file = PATHS_DIR / "paths_unconstrained_edge_only.jsonl"

    pathways_by_lifecycle = defaultdict(list)
    if path_file.exists():
        with open(path_file) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    path_ids = rec.get("path", [])
                    if isinstance(path_ids, str):
                        path_ids = json.loads(path_ids)
                    # Find lifecycle of first intervention node in path
                    lifecycle = None
                    for nid in path_ids:
                        nid_str = str(nid)
                        attrs = node_attrs.get(nid_str, node_attrs.get(nid, {}))
                        lc = attrs.get("intervention_lifecycle")
                        if lc is not None:
                            lifecycle = int(lc)
                            break
                    if lifecycle is None:
                        lifecycle = 0
                    # Lifecycle stage: 1-2=design, 3-4=training, 5-6=deployment
                    stage = (
                        "design"
                        if lifecycle <= 2
                        else ("training" if lifecycle <= 4 else "deployment")
                    )
                    if lifecycle == 0:
                        stage = "other"
                    pathways_by_lifecycle[stage].append(
                        (path_ids, rec.get("categories", []))
                    )
                except Exception:
                    continue

    # Stratified sampling
    target = {"design": 33, "training": 33, "deployment": 34}
    sampled = []
    cluster_counts = defaultdict(int)

    for stage, target_n in target.items():
        pool = pathways_by_lifecycle.get(stage, [])
        random.shuffle(pool)
        added = 0
        for path_ids, cats in pool:
            if added >= target_n:
                break
            # Get cluster of first node
            first_nid = str(path_ids[0]) if path_ids else ""
            cluster_id = None
            for key, members in cluster_memberships.items():
                if (
                    len(key) == 5
                    and str(key[0]) == "EDGE"
                    and first_nid in [str(m) for m in members]
                ):
                    cluster_id = key[4]
                    break
            if cluster_counts[cluster_id] >= 3:
                continue

            # Build record
            node_names, node_types_list, source_urls = [], [], []
            for nid in path_ids:
                nid_str = str(nid)
                attrs = node_attrs.get(nid_str, node_attrs.get(nid, {}))
                node_names.append(attrs.get("name", f"node_{nid}"))
                node_types_list.append(attrs.get("type", "unknown"))
                source_urls.append(attrs.get("url", ""))

            sampled.append(
                dict(
                    node_ids=[str(n) for n in path_ids],
                    node_names=node_names,
                    node_types=node_types_list,
                    source_urls=source_urls,
                    cluster_id=str(cluster_id),
                    lifecycle_stage=stage,
                    path_length=len(path_ids),
                    categories=cats,
                )
            )
            cluster_counts[cluster_id] += 1
            added += 1

    # Fill remaining from "other" if under 100
    remaining = 100 - len(sampled)
    if remaining > 0:
        pool = pathways_by_lifecycle.get("other", [])
        random.shuffle(pool)
        for path_ids, cats in pool[:remaining]:
            node_names = []
            node_types_list = []
            source_urls = []
            for nid in path_ids:
                nid_str = str(nid)
                attrs = node_attrs.get(nid_str, node_attrs.get(nid, {}))
                node_names.append(attrs.get("name", f"node_{nid}"))
                node_types_list.append(attrs.get("type", "unknown"))
                source_urls.append(attrs.get("url", ""))
            sampled.append(
                dict(
                    node_ids=[str(n) for n in path_ids],
                    node_names=node_names,
                    node_types=node_types_list,
                    source_urls=source_urls,
                    cluster_id="unknown",
                    lifecycle_stage="other",
                    path_length=len(path_ids),
                    categories=cats,
                )
            )

    return sampled[:100]


def _analyze_sim_coverage(node_attrs, edge_data, cluster_memberships, df_purity):
    """Classify nodes as anchored (in EDGE paths) vs SIM-only."""
    print("    Building EDGE-only node set ...")
    edge_nodes = set()
    for key, members in cluster_memberships.items():
        if len(key) == 5 and str(key[0]) == "EDGE":
            edge_nodes.update(str(m) for m in members)

    print("    Building SIM>=0.9 node set ...")
    sim09_nodes = set()
    for key, members in cluster_memberships.items():
        if len(key) == 5 and str(key[0]) == "0.9":
            sim09_nodes.update(str(m) for m in members)

    anchored = edge_nodes & sim09_nodes
    sim_only = sim09_nodes - edge_nodes

    print(f"    Anchored nodes (in both): {len(anchored):,}")
    print(f"    SIM-only nodes: {len(sim_only):,}")

    # Degree of SIM-only nodes in SIM>=0.9 graph
    print("    Computing SIM-only degree ...")
    sim09_degree = defaultdict(int)
    for e in edge_data:
        if str(e.get("type", "")).upper() == "SIMILARITY":
            score = e.get("similarity_score")
            if score is not None and cos_sim_from_score(score) >= 0.90:
                src, tgt = str(e.get("source", "")), str(e.get("target", ""))
                if src in sim_only:
                    sim09_degree[src] += 1
                if tgt in sim_only:
                    sim09_degree[tgt] += 1

    # Gold purity per cluster for SIM>=0.9/both
    gold_purity_map = {}
    sub_purity = df_purity[
        (df_purity["edge_config"] == "0.9") & (df_purity["mode"] == "both")
    ]
    for _, r in sub_purity.iterrows():
        gold_purity_map[str(r["cluster_id"])] = r["edge_purity"]

    # Node-to-cluster mapping for SIM>=0.9/both/agglomerative
    node_to_cluster = {}
    for key, members in cluster_memberships.items():
        if (
            len(key) == 5
            and str(key[0]) == "0.9"
            and key[1] == "both"
            and key[3] == "agglomerative"
        ):
            for nid in members:
                node_to_cluster[str(nid)] = str(key[4])

    # Classify SIM-only nodes
    n_foundational, n_niche = 0, 0
    for nid in sim_only:
        degree = sim09_degree.get(nid, 0)
        cid = node_to_cluster.get(nid)
        purity = gold_purity_map.get(cid, 0) if cid else 0
        if degree >= 10 and purity >= 0.8:
            n_foundational += 1
        else:
            n_niche += 1

    print(f"    Foundational SIM-only: {n_foundational:,}  Niche: {n_niche:,}")

    return dict(
        n_anchored=len(anchored),
        n_sim_only=len(sim_only),
        n_sim09_total=len(sim09_nodes),
        n_foundational=n_foundational,
        n_niche=n_niche,
        mean_sim_only_degree=float(np.mean(list(sim09_degree.values())))
        if sim09_degree
        else 0,
    )


def _build_comparison_table(df_quality, df_purity, df_t6, t8, df_pairwise):
    """Build EDGE vs SIM>=0.9 comparison CSV and plot."""
    gold_cfg = (
        df_purity.groupby(["edge_config", "mode", "node_type"])["is_gold_standard"]
        .mean()
        .reset_index()
        .rename(columns={"is_gold_standard": "gold_pct"})
    )

    focus_nts = ["risk", "intervention"]
    rows = []
    for nt in focus_nts:
        for ec in ["EDGE", "0.9"]:
            q = df_quality[
                (df_quality["edge_config"] == ec)
                & (df_quality["mode"] == "both")
                & (df_quality["node_type"] == nt)
            ]
            if len(q) == 0:
                continue
            q = q.iloc[0]
            gp = gold_cfg[
                (gold_cfg["edge_config"] == ec)
                & (gold_cfg["mode"] == "both")
                & (gold_cfg["node_type"] == nt)
            ]
            gp_val = gp["gold_pct"].values[0] if len(gp) > 0 else np.nan
            ari_val = (
                get_pairwise_ari(df_pairwise, nt, "both", "EDGE", "0.9")
                if ec == "EDGE"
                else np.nan
            )

            rows.append(
                dict(
                    node_type=nt,
                    edge_config=ec,
                    mode="both",
                    silhouette=q["silhouette_mean"],
                    edge_pct=q["edge_validation_mean"],
                    n_clusters=q["n_clusters"],
                    gold_purity_pct=gp_val,
                    n_nodes=q["n_embeddings"],
                    ari_vs_sim09=ari_val,
                )
            )

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(STEP3_DIR / "edge_only_comparison.csv", index=False)
    print("  Saved edge_only_comparison.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: grouped quality bars EDGE vs SIM>=0.9
    metrics = ["silhouette", "edge_pct", "gold_purity_pct"]
    metric_labels = ["Silhouette", "EDGE%", "Gold Purity"]
    x = np.arange(len(metrics))
    width = 0.18
    nt_colors = {"risk": "#e74c3c", "intervention": "#3498db"}
    hatches = {"EDGE": "//", "0.9": ""}

    ax = axes[0]
    offset_map = {
        ("risk", "EDGE"): -0.27,
        ("risk", "0.9"): -0.09,
        ("intervention", "EDGE"): 0.09,
        ("intervention", "0.9"): 0.27,
    }
    for nt in focus_nts:
        for ec in ["EDGE", "0.9"]:
            sub = df_cmp[(df_cmp["node_type"] == nt) & (df_cmp["edge_config"] == ec)]
            if len(sub) == 0:
                continue
            vals = [
                sub[m].values[0] if not np.isnan(sub[m].values[0]) else 0
                for m in metrics
            ]
            offset = offset_map.get((nt, ec), 0)
            label = f"{nt}/{ec}"
            ax.bar(
                x + offset,
                vals,
                width,
                label=label,
                color=nt_colors[nt],
                hatch=hatches[ec],
                alpha=0.8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(
        "Panel A: EDGE-only vs SIM≥0.9 quality comparison\n(hatch=EDGE, solid=SIM≥0.9)"
    )
    ax.legend(fontsize=8, ncol=2)

    # Panel B: SIM-only node classification
    ax = axes[1]
    categories = ["Anchored\n(both)", "SIM-only\nFoundational", "SIM-only\nNiche"]
    vals = [t8["n_anchored"], t8["n_foundational"], t8["n_niche"]]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(categories, vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}",
            ha="center",
            fontsize=10,
        )
    ax.set_ylabel("Node count")
    ax.set_title(
        "Panel B: SIM-only node classification\n(SIM≥0.9 configs vs EDGE-only baseline)"
    )
    ax.set_yscale("log")

    plt.suptitle("EDGE-Only vs SIM≥0.9 Baseline Validation", fontsize=13)
    plt.tight_layout()
    fig.savefig(STEP3_DIR / "edge_vs_sim_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved edge_vs_sim_coverage.png")


# ─── SECTION D: Betweenness on SIM>=0.9 Graph ────────────────────────────────


def run_section_d(node_attrs, edge_data):
    print("\n" + "=" * 70)
    print("SECTION D — Betweenness on SIM>=0.9 Graph")
    print("=" * 70)
    print("  Building filtered graph (SIM>=0.9 + structural EDGE) ...")

    G = nx.Graph()
    n_sim09, n_struct = 0, 0
    for e in edge_data:
        etype = str(e.get("type", "")).upper()
        src, tgt = str(e.get("source", "")), str(e.get("target", ""))
        if not src or not tgt:
            continue
        if etype == "SIMILARITY":
            score = e.get("similarity_score")
            if score is not None and cos_sim_from_score(score) >= 0.90:
                G.add_edge(src, tgt)
                n_sim09 += 1
        elif etype == "EDGE":
            G.add_edge(src, tgt)
            n_struct += 1

    print(
        f"  Full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
        f"(SIM09={n_sim09:,}, STRUCT={n_struct:,})"
    )

    # Compute exact betweenness on the full graph so that shortest paths can
    # traverse all nodes including degree-1/2 intermediates.  Restricting to a
    # degree>=3 induced subgraph severs those paths and fragments the graph into
    # 16K+ tiny components, producing biased scores.
    import sys
    import threading

    print(
        f"  Computing EXACT betweenness on full graph "
        f"({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges) "
        f"— estimated 15-30 hours ..."
    )
    sys.stdout.flush()

    _btw_done = threading.Event()

    def _heartbeat():
        interval = 1800  # 30 min
        elapsed = 0
        while not _btw_done.wait(timeout=interval):
            elapsed += interval
            print(
                f"  [heartbeat] betweenness still running — {elapsed // 3600}h {(elapsed % 3600) // 60}m elapsed"
            )
            sys.stdout.flush()

    _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()

    betweenness = nx.betweenness_centrality(G, normalized=True)
    _btw_done.set()

    # Checkpoint raw betweenness immediately before any post-processing that could fail
    checkpoint_path = STEP3_DIR / "betweenness_raw_checkpoint.pkl"
    import pickle as _pickle

    with open(checkpoint_path, "wb") as _f:
        _pickle.dump(betweenness, _f, protocol=4)
    print(f"  Checkpoint saved: {checkpoint_path} ({len(betweenness):,} nodes)")
    sys.stdout.flush()

    # Top 100
    top100 = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:100]

    # Load old betweenness for comparison
    old_btw_file = STEP2_DIR / "mechanism_transfer_betweenness.csv"
    old_rank = {}
    if old_btw_file.exists():
        df_old = pd.read_csv(old_btw_file)
        id_col = "node_id" if "node_id" in df_old.columns else df_old.columns[0]
        for rank, (_, row) in enumerate(df_old.iterrows(), 1):
            old_rank[str(row[id_col])] = rank

    rows = []
    for rank, (nid, btw) in enumerate(top100, 1):
        attrs = node_attrs.get(
            nid, node_attrs.get(int(nid) if nid.isdigit() else nid, {})
        )
        rows.append(
            dict(
                node_id=nid,
                name=attrs.get("name", ""),
                category=attrs.get("concept_category") or attrs.get("type", ""),
                betweenness_sim09=btw,
                rank_sim09=rank,
                rank_sim08=old_rank.get(nid, -1),
            )
        )

    df_btw = pd.DataFrame(rows)
    df_btw.to_csv(STEP3_DIR / "betweenness_sim09.csv", index=False)
    print(f"  Saved betweenness_sim09.csv (top {len(df_btw)} nodes)")

    # Cluster top-50 by embedding
    top50_ids = [r["node_id"] for r in rows[:50]]
    embeddings_50 = []
    valid_ids = []
    for nid in top50_ids:
        attrs = node_attrs.get(
            nid, node_attrs.get(int(nid) if nid.isdigit() else nid, {})
        )
        emb = attrs.get("embedding")
        if emb is not None:
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
                embeddings_50.append(emb / norm)
                valid_ids.append(nid)

    bridge_clusters_rows = []
    if len(embeddings_50) >= 10:
        X = np.stack(embeddings_50)
        clust = AgglomerativeClustering(n_clusters=min(12, len(embeddings_50)))
        labels = clust.fit_predict(X)
        for nid, cid in zip(valid_ids, labels):
            attrs = node_attrs.get(
                nid, node_attrs.get(int(nid) if nid.isdigit() else nid, {})
            )
            bridge_clusters_rows.append(
                dict(
                    node_id=nid,
                    name=attrs.get("name", ""),
                    category=attrs.get("concept_category") or attrs.get("type", ""),
                    cluster_id=int(cid),
                )
            )
    else:
        for nid in valid_ids:
            attrs = node_attrs.get(
                nid, node_attrs.get(int(nid) if nid.isdigit() else nid, {})
            )
            bridge_clusters_rows.append(
                dict(
                    node_id=nid,
                    name=attrs.get("name", ""),
                    category=attrs.get("concept_category") or attrs.get("type", ""),
                    cluster_id=0,
                )
            )

    df_bridge = pd.DataFrame(bridge_clusters_rows)
    df_bridge.to_csv(STEP3_DIR / "betweenness_bridge_clusters.csv", index=False)
    print(f"  Saved betweenness_bridge_clusters.csv ({len(df_bridge)} bridge nodes)")

    # Plot
    _plot_betweenness_comparison(df_btw, old_rank)
    return df_btw


def _plot_betweenness_comparison(df_btw, old_rank):
    fig, ax = plt.subplots(figsize=(10, 8))

    has_old = df_btw["rank_sim08"] > 0
    colors = {"concept": "#3498db", "intervention": "#e74c3c", "": "#95a5a6"}

    for _, row in df_btw.iterrows():
        c = colors.get(str(row["category"]).lower(), "#95a5a6")
        if has_old[row.name] and row["rank_sim08"] > 0:
            # Normalize old rank to [0,1] for x-axis (lower rank = higher betweenness)
            old_btw_approx = 1.0 / max(row["rank_sim08"], 1)
            ax.scatter(
                old_btw_approx, row["betweenness_sim09"], color=c, alpha=0.7, s=40
            )

    # Label top-20
    for _, row in df_btw.head(20).iterrows():
        c = colors.get(str(row["category"]).lower(), "#95a5a6")
        if row["rank_sim08"] > 0:
            old_btw_approx = 1.0 / max(row["rank_sim08"], 1)
            ax.annotate(
                str(row["name"])[:30],
                (old_btw_approx, row["betweenness_sim09"]),
                fontsize=6,
                alpha=0.8,
            )

    from matplotlib.patches import Patch

    legend_els = [Patch(color=v, label=k) for k, v in colors.items() if k]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_xlabel("1/rank_sim08 (proxy for old betweenness)")
    ax.set_ylabel("Betweenness in SIM>=0.9 graph")
    ax.set_title(
        "Betweenness comparison: SIM>=0.8 (rank proxy) vs SIM>=0.9 (recomputed)\nTop-100 nodes, colored by category"
    )

    plt.tight_layout()
    fig.savefig(STEP3_DIR / "betweenness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved betweenness_comparison.png")


# ─── FIX CATEGORIES: relabel existing full-graph betweenness CSVs ────────────


def fix_betweenness_categories(node_attrs):
    """Reload betweenness_sim09.csv and betweenness_bridge_clusters.csv,
    replace generic 'concept'/'intervention' category with concept_category
    where available, and resave in-place."""
    import sys

    node_attrs_str = {str(k): v for k, v in node_attrs.items()}

    def _get_category(nid):
        attrs = node_attrs_str.get(str(nid), {})
        return attrs.get("concept_category") or attrs.get("type", "")

    btw_path = STEP3_DIR / "betweenness_sim09.csv"
    if btw_path.exists():
        df = pd.read_csv(btw_path)
        df["category"] = df["node_id"].apply(lambda nid: _get_category(nid))
        df.to_csv(btw_path, index=False)
        print(f"  Fixed betweenness_sim09.csv ({len(df)} rows)")
        print(f"  Category distribution:\n{df['category'].value_counts().to_string()}")
        sys.stdout.flush()
    else:
        print("  betweenness_sim09.csv not found — skipping")

    bc_path = STEP3_DIR / "betweenness_bridge_clusters.csv"
    if bc_path.exists():
        df = pd.read_csv(bc_path)
        df["category"] = df["node_id"].apply(lambda nid: _get_category(nid))
        df.to_csv(bc_path, index=False)
        print(f"  Fixed betweenness_bridge_clusters.csv ({len(df)} rows)")
        print(f"  Category distribution:\n{df['category'].value_counts().to_string()}")
        sys.stdout.flush()
    else:
        print("  betweenness_bridge_clusters.csv not found — skipping")

    # Reload old_rank for plot regeneration
    old_btw_file = STEP2_DIR / "mechanism_transfer_betweenness.csv"
    old_rank = {}
    if old_btw_file.exists():
        df_old = pd.read_csv(old_btw_file)
        id_col = "node_id" if "node_id" in df_old.columns else df_old.columns[0]
        for rank, (_, row) in enumerate(df_old.iterrows(), 1):
            old_rank[str(row[id_col])] = rank

    df_btw = pd.read_csv(btw_path)
    _plot_betweenness_comparison(df_btw, old_rank)
    print("  Regenerated betweenness_comparison.png")


# ─── BETWEENNESS ON BOTH-MODE SUBGRAPH ───────────────────────────────────────


def run_betweenness_both(node_attrs, edge_data, cluster_memberships):
    """Exact betweenness on the SIM>=0.9+EDGE induced subgraph of all nodes
    assigned to any cluster under mode=both, ec=0.9, agglomerative.
    Saves to betweenness_both09.csv / betweenness_both09_bridge_clusters.csv /
    betweenness_both09_comparison.png — does NOT touch full-graph outputs."""
    import sys
    import threading
    import pickle as _pickle

    print("\n" + "=" * 70)
    print("BETWEENNESS — Both-mode SIM>=0.9 subgraph")
    print("=" * 70)

    # Collect all nodes in both-mode ec=0.9 agglomerative clusters
    both_nodes = set()
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            ec, mode, node_type, algo, cluster_id = key
            if (
                str(ec) == "0.9"
                and str(mode) == "both"
                and str(algo) == "agglomerative"
            ):
                both_nodes.update(str(m) for m in members)

    print(f"  Both-mode nodes (ec=0.9, all node_types): {len(both_nodes):,}")

    # Build induced subgraph: SIM>=0.9 + structural EDGE edges between both_nodes
    G = nx.Graph()
    n_sim, n_edge = 0, 0
    for e in edge_data:
        src, tgt = str(e.get("source", "")), str(e.get("target", ""))
        if src not in both_nodes or tgt not in both_nodes:
            continue
        etype = str(e.get("type", "")).upper()
        if etype == "SIMILARITY":
            score = e.get("similarity_score")
            if score is not None and cos_sim_from_score(score) >= 0.90:
                G.add_edge(src, tgt)
                n_sim += 1
        elif etype == "EDGE":
            G.add_edge(src, tgt)
            n_edge += 1

    print(
        f"  Subgraph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
        f"(SIM09={n_sim:,}, EDGE={n_edge:,})"
    )
    sys.stdout.flush()

    _btw_done = threading.Event()

    def _heartbeat():
        interval = 1800
        elapsed = 0
        while not _btw_done.wait(timeout=interval):
            elapsed += interval
            print(
                f"  [heartbeat] both-mode betweenness still running — "
                f"{elapsed // 3600}h {(elapsed % 3600) // 60}m elapsed"
            )
            sys.stdout.flush()

    _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()

    print("  Computing EXACT betweenness on both-mode subgraph ...")
    sys.stdout.flush()
    betweenness = nx.betweenness_centrality(G, normalized=True)
    _btw_done.set()

    # Checkpoint immediately
    ckpt = STEP3_DIR / "betweenness_both09_raw_checkpoint.pkl"
    with open(ckpt, "wb") as _f:
        _pickle.dump(betweenness, _f, protocol=4)
    print(f"  Checkpoint saved: {ckpt} ({len(betweenness):,} nodes)")
    sys.stdout.flush()

    node_attrs_str = {str(k): v for k, v in node_attrs.items()}

    def _get_category(nid):
        attrs = node_attrs_str.get(str(nid), {})
        return attrs.get("concept_category") or attrs.get("type", "")

    # Top 100
    top100 = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:100]

    old_btw_file = STEP2_DIR / "mechanism_transfer_betweenness.csv"
    old_rank = {}
    if old_btw_file.exists():
        df_old = pd.read_csv(old_btw_file)
        id_col = "node_id" if "node_id" in df_old.columns else df_old.columns[0]
        for rank, (_, row) in enumerate(df_old.iterrows(), 1):
            old_rank[str(row[id_col])] = rank

    rows = []
    for rank, (nid, btw) in enumerate(top100, 1):
        attrs = node_attrs_str.get(str(nid), {})
        rows.append(
            dict(
                node_id=nid,
                name=attrs.get("name", ""),
                category=_get_category(nid),
                betweenness_both09=btw,
                rank_both09=rank,
                rank_sim08=old_rank.get(nid, -1),
            )
        )

    df_btw = pd.DataFrame(rows)
    df_btw.to_csv(STEP3_DIR / "betweenness_both09.csv", index=False)
    print(f"  Saved betweenness_both09.csv (top {len(df_btw)} nodes)")

    # Cluster top-50 by embedding
    top50_ids = [r["node_id"] for r in rows[:50]]
    embeddings_50, valid_ids = [], []
    for nid in top50_ids:
        attrs = node_attrs_str.get(str(nid), {})
        emb = attrs.get("embedding")
        if emb is not None:
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
                embeddings_50.append(emb / norm)
                valid_ids.append(nid)

    bridge_rows = []
    if len(embeddings_50) >= 10:
        X = np.stack(embeddings_50)
        clust = AgglomerativeClustering(n_clusters=min(12, len(embeddings_50)))
        labels = clust.fit_predict(X)
        for nid, cid in zip(valid_ids, labels):
            attrs = node_attrs_str.get(str(nid), {})
            bridge_rows.append(
                dict(
                    node_id=nid,
                    name=attrs.get("name", ""),
                    category=_get_category(nid),
                    cluster_id=int(cid),
                )
            )

    df_bridge = pd.DataFrame(bridge_rows)
    df_bridge.to_csv(STEP3_DIR / "betweenness_both09_bridge_clusters.csv", index=False)
    print(f"  Saved betweenness_both09_bridge_clusters.csv ({len(df_bridge)} nodes)")

    # Plot — reuse comparison plot with both09 column name
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"risk": "#e74c3c", "intervention": "#2ecc71", "concept": "#3498db"}
    for _, row in df_btw.iterrows():
        c = colors.get(str(row["category"]).lower(), "#95a5a6")
        if row["rank_sim08"] > 0:
            old_btw_approx = 1.0 / max(row["rank_sim08"], 1)
            ax.scatter(
                old_btw_approx, row["betweenness_both09"], color=c, alpha=0.7, s=40
            )
    ax.set_xlabel("SIM>=0.8 betweenness (approx, 1/rank proxy)")
    ax.set_ylabel("Both-mode SIM>=0.9 betweenness (exact)")
    ax.set_title("Betweenness comparison: SIM>=0.8 approx vs Both-mode SIM>=0.9 exact")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=lab
        )
        for lab, c in colors.items()
    ]
    ax.legend(handles=handles)
    plt.tight_layout()
    fig.savefig(
        STEP3_DIR / "betweenness_both09_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved betweenness_both09_comparison.png")
    sys.stdout.flush()
    return df_btw


# ─── SECTION E: Held-Out Validation (#24) ────────────────────────────────────


def run_section_e(node_attrs, cluster_memberships):
    print("\n" + "=" * 70)
    print("SECTION E — #24 Held-Out Validation (leave-20%-out)")
    print("=" * 70)

    results = []
    # Use primary cut: SIM=0.9, mode=both, agglomerative
    clusters = defaultdict(list)
    for key, members in cluster_memberships.items():
        if (
            len(key) == 5
            and str(key[0]) == PRIMARY_CUT_EC
            and key[1] == PRIMARY_CUT_MODE
            and key[3] == "agglomerative"
        ):
            nt = key[2]
            cid = key[4]
            clusters[(nt, cid)].extend(str(m) for m in members)

    print(f"  Found {len(clusters)} cluster×node_type groups in primary cut")

    for (nt, cid), members in clusters.items():
        if len(members) < 5:
            continue

        # Get embeddings
        embs = {}
        for nid in members:
            attrs = node_attrs.get(
                nid, node_attrs.get(int(nid) if nid.isdigit() else nid, {})
            )
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
                embs[nid] = emb / norm

        valid_members = list(embs.keys())
        if len(valid_members) < 5:
            continue

        n_withhold = max(1, len(valid_members) // 5)
        random.seed(42)
        withheld = random.sample(valid_members, n_withhold)
        train = [m for m in valid_members if m not in withheld]

        centroid = np.mean([embs[m] for m in train], axis=0)
        results.append(
            dict(
                node_type=nt,
                cluster_id=cid,
                n_members=len(valid_members),
                n_withheld=n_withhold,
                centroid=centroid,
                withheld_embs=[embs[m] for m in withheld],
            )
        )

    if not results:
        print("  No clusters with sufficient embeddings found.")
        return {}

    # Group by node_type for within-type comparison (correct: don't compare risk to intervention centroids)
    by_nt = defaultdict(list)
    for i, r in enumerate(results):
        by_nt[r["node_type"]].append(r)

    # Evaluate each cluster's withheld nodes against same-node_type centroids only
    n_correct_total, n_total = 0, 0
    eval_rows = []
    for nt, nt_results in by_nt.items():
        nt_centroids = np.array([r["centroid"] for r in nt_results])
        for i, r in enumerate(nt_results):
            correct = 0
            for emb in r["withheld_embs"]:
                sims = nt_centroids.dot(emb)
                if np.argmax(sims) == i:
                    correct += 1
            acc = correct / len(r["withheld_embs"]) if r["withheld_embs"] else 0
            eval_rows.append(
                dict(
                    node_type=r["node_type"],
                    cluster_id=r["cluster_id"],
                    n_members=r["n_members"],
                    n_withheld=r["n_withheld"],
                    n_correct=correct,
                    accuracy=acc,
                )
            )
            n_correct_total += correct
            n_total += len(r["withheld_embs"])

    mean_acc = n_correct_total / n_total if n_total > 0 else 0
    print(
        f"  Mean leave-20%-out accuracy: {mean_acc:.3f} ({n_correct_total}/{n_total} correct)"
    )

    df_held = pd.DataFrame(eval_rows)
    df_held.to_csv(STEP3_DIR / "held_out_validation.csv", index=False)
    print(f"  Saved held_out_validation.csv ({len(df_held)} clusters)")

    return {
        "mean_accuracy": mean_acc,
        "n_clusters": len(df_held),
        "n_correct": n_correct_total,
        "n_total": n_total,
    }


# ─── SECTION F: EDGE Subgraph Consistency (#25) ──────────────────────────────


def run_section_f(node_attrs, edge_data):
    print("\n" + "=" * 70)
    print("SECTION F — #25 EDGE Subgraph Consistency")
    print("=" * 70)

    print("  Building EDGE-only directed subgraph ...")
    G_edge = nx.DiGraph()
    for e in edge_data:
        if str(e.get("type", "")).upper() == "EDGE":
            src, tgt = str(e.get("source", "")), str(e.get("target", ""))
            if src and tgt:
                G_edge.add_edge(src, tgt)

    n_nodes = G_edge.number_of_nodes()
    n_edges = G_edge.number_of_edges()
    print(f"  EDGE subgraph: {n_nodes:,} nodes, {n_edges:,} edges")

    # WCC analysis
    wccs = list(nx.weakly_connected_components(G_edge))
    wccs_sorted = sorted(wccs, key=len, reverse=True)
    largest_wcc = len(wccs_sorted[0]) if wccs_sorted else 0
    largest_wcc_pct = largest_wcc / n_nodes if n_nodes > 0 else 0

    print(
        f"  WCC: {len(wccs):,} components, largest={largest_wcc:,} ({largest_wcc_pct:.1%} of nodes)"
    )

    # Approximate diameter via BFS sample on largest WCC
    approx_diameter = "N/A"
    try:
        giant = G_edge.subgraph(wccs_sorted[0]).copy()
        # Sample 20 random sources, find max eccentricity approx
        sample_nodes = random.sample(
            list(giant.nodes()), min(20, giant.number_of_nodes())
        )
        max_path = 0
        for src in sample_nodes:
            lengths = nx.single_source_shortest_path_length(giant, src, cutoff=20)
            if lengths:
                max_path = max(max_path, max(lengths.values()))
        approx_diameter = max_path
        print(f"  Approximate diameter (BFS sample n=20): {approx_diameter}")
    except Exception as e:
        print(f"  Diameter computation failed: {e}")

    # Degree distribution
    degrees = [d for _, d in G_edge.degree()]
    mean_degree = float(np.mean(degrees)) if degrees else 0
    pct_degree_gt2 = sum(1 for d in degrees if d >= 2) / len(degrees) if degrees else 0

    # Overlap with betweenness top-25
    btw_file = STEP2_DIR / "mechanism_transfer_betweenness.csv"
    btw_overlap_pct = None
    if btw_file.exists():
        df_btw = pd.read_csv(btw_file)
        id_col = "node_id" if "node_id" in df_btw.columns else df_btw.columns[0]
        top25_ids = set(str(v) for v in df_btw[id_col].head(25))
        edge_node_ids = set(G_edge.nodes())
        overlap = len(top25_ids & edge_node_ids)
        btw_overlap_pct = overlap / 25
        print(
            f"  Top-25 betweenness nodes in EDGE subgraph: {overlap}/25 ({btw_overlap_pct:.1%})"
        )

    stats = dict(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_wcc=len(wccs),
        largest_wcc_size=largest_wcc,
        largest_wcc_pct=round(largest_wcc_pct, 4),
        approx_diameter=str(approx_diameter),
        mean_degree=round(mean_degree, 3),
        pct_nodes_degree_gte2=round(pct_degree_gt2, 4),
        btw_top25_in_edge_pct=round(btw_overlap_pct, 4)
        if btw_overlap_pct is not None
        else None,
    )

    df_stats = pd.DataFrame([stats])
    df_stats.to_csv(STEP3_DIR / "edge_subgraph_stats.csv", index=False)
    print("  Saved edge_subgraph_stats.csv")

    # Log-log degree distribution plot
    _plot_edge_degree_distribution(degrees, stats)
    return stats


def _plot_edge_degree_distribution(degrees, stats):
    from collections import Counter

    deg_counts = Counter(degrees)
    x = np.array(sorted(deg_counts.keys()))
    y = np.array([deg_counts[d] for d in x])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=15, alpha=0.6, color="#2c3e50")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (in+out)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"EDGE-only Subgraph Degree Distribution (log-log)\n"
        f"N={stats['n_nodes']:,} nodes, mean degree={stats['mean_degree']:.1f}, "
        f"{stats['largest_wcc_pct']:.1%} in largest WCC"
    )

    # Annotate mean
    ax.axvline(
        stats["mean_degree"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"mean degree={stats['mean_degree']:.1f}",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(
        STEP3_DIR / "edge_degree_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved edge_degree_distribution.png")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-betweenness",
        action="store_true",
        help="Skip Section D (betweenness, ~20 min)",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Run only Sections A and B (no PKL loading)",
    )
    parser.add_argument(
        "--betweenness-only",
        action="store_true",
        help="Run only Section D (betweenness) — skips A-C, E-F",
    )
    parser.add_argument(
        "--fix-categories",
        action="store_true",
        help="Relabel category column in existing full-graph betweenness CSVs using concept_category",
    )
    parser.add_argument(
        "--betweenness-both",
        action="store_true",
        help="Run exact betweenness on both-mode SIM>=0.9 subgraph; saves to betweenness_both09.* files",
    )
    args = parser.parse_args()

    import time

    t0 = time.time()

    print("=" * 70)
    print("Phase 2 Step 3: Validation & Config Selection")
    print("=" * 70)
    print(f"Output dir: {STEP3_DIR}")

    # ── Load CSV inputs ──────────────────────────────────────────────────────
    print("\nLoading Step 2 CSV inputs ...")
    df_quality = pd.read_csv(
        STEP2_DIR / "quality_metrics_summary.csv", dtype={"edge_config": str}
    )
    df_purity = pd.read_csv(
        STEP2_DIR / "cluster_edge_purity.csv", dtype={"edge_config": str}
    )
    df_pairwise = pd.read_csv(
        STEP2_DIR / "stability_ari_pairwise.csv",
        dtype={"threshold_1": str, "threshold_2": str},
    )
    df_centroid = pd.read_csv(
        STEP2_DIR / "cluster_centroid_similarity.csv",
        dtype={"threshold_from": str, "threshold_to": str},
    )
    print(f"  quality_metrics_summary: {len(df_quality)} rows")
    print(f"  cluster_edge_purity: {len(df_purity)} rows")
    print(f"  stability_ari_pairwise: {len(df_pairwise)} rows")
    print(f"  cluster_centroid_similarity: {len(df_centroid)} rows")

    # ── Section A & B (CSV-only) ─────────────────────────────────────────────
    df_ranked, winners = run_section_a(df_quality, df_purity, df_pairwise)
    run_section_b(df_quality, df_purity, df_pairwise, df_centroid)

    if args.csv_only:
        print(f"\nCSV-only mode: Sections A+B complete in {time.time() - t0:.1f}s")
        return

    # ── Load PKL checkpoints ─────────────────────────────────────────────────
    print("\nLoading PKL checkpoints ...")
    node_attrs = load_pkl(STEP1_DIR / "graph_node_attributes.pkl")
    edge_data = load_pkl(STEP1_DIR / "graph_edge_data.pkl")

    if args.fix_categories:
        print(f"  node_attrs: {len(node_attrs):,} nodes")
        fix_betweenness_categories(node_attrs)
        elapsed = time.time() - t0
        print(f"\n--fix-categories complete in {elapsed:.1f}s")
        return

    if args.betweenness_only:
        print(f"  node_attrs: {len(node_attrs):,} nodes")
        print(f"  edge_data: {len(edge_data):,} edges")
        run_section_d(node_attrs, edge_data)
        elapsed = time.time() - t0
        print(f"\nBetweenness-only mode complete in {elapsed / 60:.1f} min")
        return

    cluster_memberships = load_pkl(STEP1_DIR / "cluster_memberships.pkl")

    if args.betweenness_both:
        print(f"  node_attrs: {len(node_attrs):,} nodes")
        print(f"  edge_data: {len(edge_data):,} edges")
        print(f"  cluster_memberships: {len(cluster_memberships):,} keys")
        run_betweenness_both(node_attrs, edge_data, cluster_memberships)
        elapsed = time.time() - t0
        print(f"\n--betweenness-both complete in {elapsed / 60:.1f} min")
        return
    print(f"  node_attrs: {len(node_attrs):,} nodes")
    print(f"  edge_data: {len(edge_data):,} edges")
    print(f"  cluster_memberships: {len(cluster_memberships):,} keys")

    # ── Section C ────────────────────────────────────────────────────────────
    df_t6, t8_results = run_section_c(
        df_quality, df_purity, df_pairwise, cluster_memberships, node_attrs, edge_data
    )

    # ── Section E ────────────────────────────────────────────────────────────
    held_out_results = run_section_e(node_attrs, cluster_memberships)

    # ── Section F ────────────────────────────────────────────────────────────
    edge_stats = run_section_f(node_attrs, edge_data)

    # ── Section D (optional) ─────────────────────────────────────────────────
    if not args.skip_betweenness:
        run_section_d(node_attrs, edge_data)
    else:
        print("\nSection D (betweenness) SKIPPED (--skip-betweenness flag)")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"STEP 3 COMPLETE in {elapsed / 60:.1f} min")
    print("=" * 70)
    print("\nOutputs in", STEP3_DIR)

    output_files = list(STEP3_DIR.glob("*"))
    for f in sorted(output_files):
        print(f"  {f.name}")

    # Quick summary of key findings
    risk_winner = winners[winners["node_type"] == "risk"]
    if len(risk_winner) > 0:
        rw = risk_winner.iloc[0]
        confirmed = (
            rw["edge_config"] == PRIMARY_CUT_EC and rw["mode"] == PRIMARY_CUT_MODE
        )
        print(
            f"\n  Primary cut (risk): edge_config={rw['edge_config']}, mode={rw['mode']} "
            f"({'CONFIRMED' if confirmed else 'UPDATED'})"
        )

    if held_out_results:
        print(
            f"  Held-out accuracy: {held_out_results.get('mean_accuracy', 0):.3f} "
            f"({held_out_results.get('n_clusters', 0)} clusters)"
        )

    if edge_stats:
        print(
            f"  EDGE subgraph: {edge_stats['largest_wcc_pct']:.1%} in largest WCC, "
            f"diameter≈{edge_stats['approx_diameter']}"
        )


if __name__ == "__main__":
    main()
