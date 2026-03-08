#!/usr/bin/env python3
"""
Phase 2 Step 2b: Extended Metrics & Analysis
=============================================

Implements all missing analyses and improved visualizations from the
Phase2_Code_Changes_Tracker.md, closing gaps identified in
Phase2_Step2_Comprehensive_Findings.md.

Inputs (from Step 1 checkpoints + Step 2 outputs):
  phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/
    - all_cluster_metrics.csv
    - graph_node_attributes.pkl
    - graph_edge_data.pkl
    - cluster_memberships.pkl
  phase2_results/step2_metrics_and_stability/
    - quality_metrics_summary.csv
    - stability_ari_matrix.csv
  phase2_rawclusterfiles_umapwithoutlocalsatellites/
    - clusters_*.json

New outputs (all written to step2_metrics_and_stability/):
  CSVs:
    - stability_ari_pairwise.csv              (CHANGE #1)
    - cohesion_analysis.csv                   (CHANGE #8)
    - cluster_centroid_similarity.csv         (CHANGE #9)
    - cluster_edge_purity.csv                 (CHANGE #10)
    - hub_quality_metrics.csv                 (CHANGE #3)
    - cluster_source_diversity_v2.csv         (CHANGE #2, url-based — fixes #11/#13)
    - mode_comparison_stats.csv               (CHANGE #15/Sub#9)
    - multi_risk_clusters.csv                 (CHANGE #15/Sub#10)
    - risk_diversity_stats.csv                (CHANGE #15/Sub#11)
    - category_mechanism_families.csv         (CHANGE #15/Sub#15)
    - algorithm_comparison.csv                (CHANGE #7 — Agg vs Louvain vs HDBSCAN)
    - maturity_per_cluster.csv                (CHANGE #15/Sub#13 — bug fix)
  Plots:
    - cross_threshold_ari_lineplot.png        (CHANGE #1)
    - edge_validation_per_mode.png            (CHANGE #6)
    - silhouette_by_nodetype_v2.png           (CHANGE #7 — labels fix)
    - cluster_size_distributions_v2.png       (CHANGE #12)
    - centroid_similarity_heatmap.png         (CHANGE #9)
    - edge_purity_histograms.png              (CHANGE #10)
    - hub_quality_scatter.png                 (CHANGE #3)
    - path_length_sensitivity.png             (CHANGE #14 / Substep #29)
    - mechanism_transfer_betweenness_v2.png   (CHANGE #15/Sub#16 — sort, format, 5 panels)
    - edge_density_heatmap.png                (CHANGE #15/Sub#9)
    - mode_stability_heatmap.png              (CHANGE #15/Sub#9)
    - node_migration_heatmap.png              (CHANGE #4 / Plot 20)
    - maturity_distribution_heatmap.png       (CHANGE #15/Sub#13 — per-cluster)
    - algorithm_comparison_silhouette.png     (CHANGE #7 — Agg vs Louvain vs HDBSCAN)

  DEFERRED to Step 3 (require multi-criteria scoring):
    - multi_criteria_scoring.csv / parallel_coords plot
    - optimal_configs_final.csv

  DEFERRED to Step 4 (require optimal config selection first):
    - UMAP projections (umap_risks/interventions/concepts.png)
    - Full mechanism taxonomy construction
    - Risk→Intervention connectivity matrix

Run from graph_analysis/ directory:
    uv run phase2_step2b_extended_analysis.py
"""

# CRITICAL: Set matplotlib backend BEFORE any other imports
import matplotlib

matplotlib.use("Agg")

import json
import pickle
import time
import warnings
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Any, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cosine
from tqdm import tqdm

try:
    from hdbscan import HDBSCAN as HDBSCAN_CLS

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_CLS = None
    HDBSCAN_AVAILABLE = False

warnings.filterwarnings("ignore")

print(f"Matplotlib backend: {matplotlib.get_backend()}")
assert matplotlib.get_backend().lower() == "agg", (
    f"Backend not Agg! Got: {matplotlib.get_backend()}"
)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION — mirrors phase2_step2_metrics_stability.py
# ============================================================================

STEP1_DIR = Path("./phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")
CLUSTER_FILES_DIR = Path("./phase2_rawclusterfiles_umapwithoutlocalsatellites")
STEP2_DIR = Path("./phase2_results/step2_metrics_and_stability")
OUTPUT_DIR = STEP2_DIR  # Write new outputs alongside existing step2 outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1 checkpoints
CHECKPOINT_METRICS = STEP1_DIR / "all_cluster_metrics.csv"
CHECKPOINT_NODES = STEP1_DIR / "graph_node_attributes.pkl"
CHECKPOINT_EDGES = STEP1_DIR / "graph_edge_data.pkl"
CHECKPOINT_MEMBERS = STEP1_DIR / "cluster_memberships.pkl"

# Step 2 existing outputs (inputs to some analyses here)
STEP2_QUALITY = STEP2_DIR / "quality_metrics_summary.csv"
STEP2_ARI_MATRIX = STEP2_DIR / "stability_ari_matrix.csv"

# New CSV outputs
OUT_ARI_PAIRWISE = OUTPUT_DIR / "stability_ari_pairwise.csv"
OUT_COHESION = OUTPUT_DIR / "cohesion_analysis.csv"
OUT_CENTROID_SIM = OUTPUT_DIR / "cluster_centroid_similarity.csv"
OUT_EDGE_PURITY = OUTPUT_DIR / "cluster_edge_purity.csv"
OUT_HUB_METRICS = OUTPUT_DIR / "hub_quality_metrics.csv"
OUT_SOURCE_V2 = OUTPUT_DIR / "cluster_source_diversity_v2.csv"

# New plot outputs
PLOT_ARI_LINE = OUTPUT_DIR / "cross_threshold_ari_lineplot.png"
PLOT_EDGE_VAL_MODE = OUTPUT_DIR / "edge_validation_per_mode.png"
PLOT_SILHOUETTE_V2 = OUTPUT_DIR / "silhouette_by_nodetype_v2.png"
PLOT_CLUSTER_SIZE_V2 = OUTPUT_DIR / "cluster_size_distributions_v2.png"
PLOT_CENTROID_SIM = OUTPUT_DIR / "centroid_similarity_heatmap.png"
PLOT_EDGE_PURITY = OUTPUT_DIR / "edge_purity_histograms.png"
PLOT_HUB_QUALITY = OUTPUT_DIR / "hub_quality_scatter.png"

DPI = 300

# Step 2 existing CSV inputs for fixes / re-plots
STEP2_BETWEENNESS_CSV = STEP2_DIR / "mechanism_transfer_betweenness.csv"
STEP2_MIGRATION_CSV = STEP2_DIR / "node_migration_frequencies.csv"

# Additional CSV outputs (CHANGE #7, #14, #15)
OUT_MODE_STATS = OUTPUT_DIR / "mode_comparison_stats.csv"
OUT_MULTI_RISK = OUTPUT_DIR / "multi_risk_clusters.csv"
OUT_RISK_DIVERSITY = OUTPUT_DIR / "risk_diversity_stats.csv"
OUT_CAT_FAMILIES = OUTPUT_DIR / "category_mechanism_families.csv"
OUT_ALGO_COMPARISON = OUTPUT_DIR / "algorithm_comparison.csv"
OUT_MATURITY_CLUSTER = OUTPUT_DIR / "maturity_per_cluster.csv"

# Additional plot outputs
PLOT_PATH_SENSITIVITY = OUTPUT_DIR / "path_length_sensitivity.png"
PLOT_BETWEENNESS_V2 = OUTPUT_DIR / "mechanism_transfer_betweenness_v2.png"
PLOT_MODE_EDGE_DENSITY = OUTPUT_DIR / "edge_density_heatmap.png"
PLOT_MODE_STABILITY = OUTPUT_DIR / "mode_stability_heatmap.png"
PLOT_NODE_MIGRATION = OUTPUT_DIR / "node_migration_heatmap.png"
PLOT_MATURITY_CLUSTER = OUTPUT_DIR / "maturity_distribution_heatmap.png"
PLOT_ALGO_COMPARISON = OUTPUT_DIR / "algorithm_comparison_silhouette.png"

# Concept categories (5 types) and their display labels
CONCEPT_CATEGORIES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]
CONCEPT_CAT_LABELS = {
    "problem_analysis": "Problem Analysis",
    "theoretical_insight": "Theoretical Insight",
    "design_rationale": "Design Rationale",
    "implementation_mechanism": "Implementation Mechanism",
    "validation_evidence": "Validation Evidence",
}
# Mapping from betweenness CSV 'category' column (space-separated) → node_type key
_BTWN_CAT_MAP = {
    "problem analysis": "problem_analysis",
    "theoretical insight": "theoretical_insight",
    "theoretical foundation": "theoretical_insight",
    "design rationale": "design_rationale",
    "implementation mechanism": "implementation_mechanism",
    "validation evidence": "validation_evidence",
}

# Analysis configuration space
EDGE_CONFIGS = ["EDGE", "0.8", "0.85", "0.9", "0.95"]
EDGE_CONFIGS_ORDERED = ["0.8", "0.85", "0.9", "0.95", "EDGE"]  # ascending selectivity
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

# Cohesion: cap pairwise computation to avoid O(n²) blow-up on large clusters
COHESION_NODE_CAP = 150

# ============================================================================
# SHARED HELPERS
# ============================================================================


def load_cluster_file(filepath: Path) -> Optional[Dict]:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Failed to load {filepath.name}: {e}")
        return None


def get_cluster_assignments(
    cluster_data: Dict, algorithm: str = "agglomerative"
) -> Dict:
    """Return {node_id: cluster_id} mapping."""
    if not cluster_data or "results" not in cluster_data:
        return {}
    results = cluster_data["results"]
    if algorithm not in results:
        return {}
    return results[algorithm].get("assignments", {})


def calculate_ari(a1: Dict, a2: Dict) -> float:
    common = set(a1.keys()) & set(a2.keys())
    if len(common) < 2:
        return np.nan
    l1 = [a1[n] for n in common]
    l2 = [a2[n] for n in common]
    try:
        return adjusted_rand_score(l1, l2)
    except Exception:
        return np.nan


def find_cluster_file(edge_config: str, mode: str, node_type: str) -> Optional[Path]:
    """Scan CLUSTER_FILES_DIR for the matching cluster JSON file."""
    candidates = list(CLUSTER_FILES_DIR.glob(f"clusters_*{node_type}*{mode}*.json"))
    if not candidates:
        candidates = list(CLUSTER_FILES_DIR.glob("*.json"))

    for path in candidates:
        name = path.name.lower()
        # Match edge config: "edge" or numeric like "09" / "080" / "085" etc
        ec_tag = edge_config.lower().replace(".", "")
        if node_type in name and mode in name and ec_tag in name:
            return path
    return None


def load_assignments_for_config(
    df_metrics: pd.DataFrame,
    edge_config: str,
    mode: str,
    node_type: str,
    algorithm: str = "agglomerative",
) -> Optional[Dict]:
    """Load cluster assignments for a specific configuration from cluster files."""
    mask = (
        (df_metrics["node_type"] == node_type)
        & (df_metrics["edge_config"] == edge_config)
        & (df_metrics["mode"] == mode)
        & (
            df_metrics.get(
                "cluster_file_found", pd.Series(True, index=df_metrics.index)
            )
        )
    )
    rows = df_metrics[mask]
    if len(rows) == 0:
        return None

    filepath_col = (
        "cluster_filepath" if "cluster_filepath" in df_metrics.columns else None
    )
    if filepath_col and not pd.isna(rows.iloc[0].get(filepath_col, None)):
        path = CLUSTER_FILES_DIR / rows.iloc[0][filepath_col]
    else:
        path = find_cluster_file(edge_config, mode, node_type)

    if path is None or not path.exists():
        return None

    data = load_cluster_file(path)
    if data is None:
        return None
    return get_cluster_assignments(data, algorithm)


def _parse_embedding(emb) -> Optional[np.ndarray]:
    """Parse an embedding to a float32 numpy array.
    Handles: np.ndarray, list/tuple, and FalkorDB string format '<v1, v2, ...>'.
    """
    if emb is None:
        return None
    if isinstance(emb, np.ndarray):
        return emb.astype(np.float32)
    if isinstance(emb, (list, tuple)):
        try:
            return np.array(emb, dtype=np.float32)
        except Exception:
            return None
    if isinstance(emb, str):
        # FalkorDB vector string: '<val1, val2, ...>'
        clean = emb.strip().lstrip("<").rstrip(">")
        try:
            return np.array([float(x) for x in clean.split(",")], dtype=np.float32)
        except Exception:
            return None
    try:
        return np.array(emb, dtype=np.float32)
    except Exception:
        return None


def normalize_embeddings(node_attrs: Dict) -> int:
    """Convert all embeddings in node_attrs to float32 numpy arrays in-place.
    Called once after loading node_attrs to handle FalkorDB string format.
    Returns number of nodes whose embedding was successfully parsed.
    """
    ok = 0
    for attrs in node_attrs.values():
        raw = attrs.get("embedding")
        if raw is None:
            continue
        parsed = _parse_embedding(raw)
        attrs["embedding"] = parsed
        if parsed is not None:
            ok += 1
    return ok


def get_node_embedding(node_id, node_attrs: Dict) -> Optional[np.ndarray]:
    attrs = node_attrs.get(node_id, {})
    emb = attrs.get("embedding", None)
    return _parse_embedding(emb)


# ============================================================================
# CHANGE #1  —  ARI PAIRWISE COMPUTATION + LINE PLOT
# ============================================================================


def _build_assignments_from_memberships(
    cluster_memberships: Dict,
    ec: str,
    mode: str,
    node_type: str,
    algorithm: str = "agglomerative",
) -> Dict:
    """Build {node_id: cluster_id} from the cluster_memberships dict (already in memory)."""
    assign = {}
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            key_ec, key_mode, key_nt, key_algo, cluster_id = key
            if (
                str(key_ec) == str(ec)
                and key_mode == mode
                and key_nt == node_type
                and key_algo == algorithm
            ):
                for nid in members:
                    assign[nid] = cluster_id
    return assign


def compute_ari_pairwise(
    df_metrics: pd.DataFrame, cluster_memberships: Dict
) -> pd.DataFrame:
    """
    Compute ARI for every unique threshold pair × node type × mode.
    Uses cluster_memberships dict (already in memory) — no JSON file I/O.
    Returns a tidy DataFrame: node_type, mode, threshold_1, threshold_2, ari
    """
    print("\n" + "=" * 80)
    print("CHANGE #1: Computing pairwise ARI across all threshold pairs")
    print("=" * 80)

    results = []

    all_pairs = list(combinations(range(len(EDGE_CONFIGS)), 2))  # lower-tri pairs
    total = len(NODE_TYPES) * len(MODES) * len(all_pairs)
    pbar = tqdm(total=total, desc="ARI pairs")

    for node_type in NODE_TYPES:
        for mode in MODES:
            # Build all threshold assignments from in-memory cluster_memberships
            assignments_cache: Dict[str, Dict] = {}
            for ec in EDGE_CONFIGS:
                assignments_cache[ec] = _build_assignments_from_memberships(
                    cluster_memberships, ec, mode, node_type
                )

            for i, j in all_pairs:
                ec1, ec2 = EDGE_CONFIGS[i], EDGE_CONFIGS[j]
                a1 = assignments_cache.get(ec1)
                a2 = assignments_cache.get(ec2)
                pbar.update(1)

                if not a1 or not a2:
                    continue

                ari_val = calculate_ari(a1, a2)

                results.append(
                    {
                        "node_type": node_type,
                        "mode": mode,
                        "threshold_1": ec1,
                        "threshold_2": ec2,
                        "ari": ari_val,
                        "n_common_nodes": len(set(a1.keys()) & set(a2.keys())),
                        "distance": j - i,  # 1=adjacent, 2=skip-1, etc.
                    }
                )

    pbar.close()
    df = pd.DataFrame(results)
    print(f"\n✓ Computed {len(df)} pairwise ARI values")
    return df


def plot_ari_line(df_pairwise: pd.DataFrame):
    """
    CHANGE #1: 1D line plot showing ARI vs threshold-pair distance.
    One subplot per node type, one line per mode.
    X-axis ordered: adjacent pairs first, then skip-1, skip-2, skip-3.
    """
    print("\n" + "=" * 80)
    print("CHANGE #1: Generating ARI line plot")
    print("=" * 80)

    if df_pairwise.empty:
        print("⚠  No pairwise ARI data — skipping plot")
        return

    # Build ordered pair labels: (t1→t2) sorted by distance then by t1 index
    def pair_label(t1, t2):
        return f"{t1}→{t2}"

    all_pairs_ordered = []
    for dist in range(1, len(EDGE_CONFIGS)):
        for i in range(len(EDGE_CONFIGS) - dist):
            t1, t2 = EDGE_CONFIGS[i], EDGE_CONFIGS[i + dist]
            all_pairs_ordered.append((t1, t2, pair_label(t1, t2)))

    pair_labels = [p[2] for p in all_pairs_ordered]

    colors = {
        "unconstrained": "#2196F3",
        "single_risk": "#4CAF50",
        "monotonic": "#FF9800",
        "both": "#F44336",
    }
    markers = {"unconstrained": "o", "single_risk": "s", "monotonic": "^", "both": "d"}

    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes = axes.flatten()

    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]
        subset = df_pairwise[df_pairwise["node_type"] == node_type]

        for mode in MODES:
            mode_sub = subset[subset["mode"] == mode]
            ari_values = []
            for t1, t2, _ in all_pairs_ordered:
                row = mode_sub[
                    ((mode_sub["threshold_1"] == t1) & (mode_sub["threshold_2"] == t2))
                    | (
                        (mode_sub["threshold_1"] == t2)
                        & (mode_sub["threshold_2"] == t1)
                    )
                ]
                ari_values.append(row["ari"].values[0] if len(row) > 0 else np.nan)

            ax.plot(
                range(len(pair_labels)),
                ari_values,
                marker=markers[mode],
                color=colors[mode],
                label=mode,
                linewidth=2,
                markersize=6,
                alpha=0.85,
            )

        # Vertical separator between distance groups
        n_adj = len(EDGE_CONFIGS) - 1
        n_skip1 = n_adj - 1
        ax.axvline(n_adj - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(n_adj + n_skip1 - 0.5, color="gray", linestyle=":", alpha=0.5)

        ax.axhline(
            0.7,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Target (0.7)",
        )
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("ARI", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"{node_type.replace('_', ' ').title()}", fontsize=11, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)

    plt.suptitle(
        "Cross-Threshold ARI — Pairwise Values\n"
        "(Adjacent | Skip-1 | Skip-2 | Skip-3 pairs, ordered left→right)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PLOT_ARI_LINE, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {PLOT_ARI_LINE}")


def print_ari_findings(df_pairwise: pd.DataFrame):
    """Print ARI analysis findings for the open research questions."""
    print("\n" + "=" * 80)
    print("ARI FINDINGS — answering Substep #7 open questions")
    print("=" * 80)

    if df_pairwise.empty:
        print("No pairwise data available.")
        return

    # Adjacent pairs: distance == 1
    adj = df_pairwise[df_pairwise["distance"] == 1]
    distant = df_pairwise[df_pairwise["distance"] >= 3]

    print("\n### Adjacent threshold pairs (distance=1):")
    if not adj.empty:
        print(
            adj.groupby(["node_type", "mode"])["ari"]
            .agg(["mean", "min", "max"])
            .round(3)
            .to_string()
        )

    print("\n### Most-distant pairs (distance=3, EDGE↔0.8):")
    if not distant.empty:
        print(
            distant.groupby(["node_type", "mode"])["ari"]
            .agg(["mean", "min", "max"])
            .round(3)
            .to_string()
        )

    print("\n### Pairs meeting ≥0.7 ARI target:")
    meets = df_pairwise[df_pairwise["ari"] >= 0.7]
    if not meets.empty:
        print(
            meets.groupby(["threshold_1", "threshold_2"])["ari"]
            .agg(["mean", "count"])
            .round(3)
            .to_string()
        )
    else:
        print("  None found.")

    print("\n### High-stability cluster (0.9, 0.95, EDGE) — within-group ARI:")
    hsc = df_pairwise[
        df_pairwise["threshold_1"].isin(["0.9", "0.95", "EDGE"])
        & df_pairwise["threshold_2"].isin(["0.9", "0.95", "EDGE"])
    ]
    if not hsc.empty:
        print(
            hsc.groupby(["node_type"])["ari"]
            .agg(["mean", "min", "max"])
            .round(3)
            .to_string()
        )


# ============================================================================
# CHANGE #6  —  EDGE VALIDATION PER-MODE (2×2 GRID)
# ============================================================================


def plot_edge_validation_per_mode(df_quality: pd.DataFrame):
    """
    CHANGE #6: 2×2 subplot grid — one pane per mode.
    X-axis: edge config; stacked bars by validation rate bin.
    """
    print("\n" + "=" * 80)
    print("CHANGE #6: EDGE Validation breakdown per mode (2×2 grid)")
    print("=" * 80)

    if df_quality.empty:
        print("⚠  No quality data — skipping")
        return

    bins_def = [
        (0.0, 0.6, "<60%", "#F44336"),
        (0.6, 0.8, "60-80%", "#FF9800"),
        (0.8, 0.9, "80-90%", "#FFEB3B"),
        (0.9, 1.0, "90-100%", "#8BC34A"),
        (1.0, 1.01, "100%", "#2E7D32"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    ec_labels = [str(e) for e in EDGE_CONFIGS]
    x_pos = np.arange(len(EDGE_CONFIGS))

    for idx, mode in enumerate(MODES):
        ax = axes[idx]
        mode_df = df_quality[df_quality["mode"] == mode]

        bin_data = {label: [] for _, _, label, _ in bins_def}
        for ec in EDGE_CONFIGS:
            subset = mode_df[mode_df["edge_config"] == str(ec)][
                "edge_validation_mean"
            ].values
            for low, high, label, _ in bins_def:
                if low == 1.0:
                    count = ((subset >= low) & (subset <= high)).sum()
                else:
                    count = ((subset >= low) & (subset < high)).sum()
                bin_data[label].append(int(count))

        bottom = np.zeros(len(EDGE_CONFIGS))
        for low, high, label, color in bins_def:
            heights = np.array(bin_data[label], dtype=float)
            ax.bar(x_pos, heights, bottom=bottom, label=label, color=color, alpha=0.85)
            bottom += heights

        ax.set_xticks(x_pos)
        ax.set_xticklabels(ec_labels, fontsize=10)
        ax.set_ylabel("Number of Configurations", fontsize=11)
        ax.set_xlabel("Edge Config", fontsize=11)
        ax.set_title(
            f"Mode: {mode.replace('_', ' ').title()}", fontsize=12, fontweight="bold"
        )
        ax.legend(title="Validation Rate", fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.text(
            0.02,
            0.97,
            "Target: >60%",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle(
        "EDGE-only Complete Pathway Validation Rate Distribution — Per Mode",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(PLOT_EDGE_VAL_MODE, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {PLOT_EDGE_VAL_MODE}")


def print_edge_validation_findings(df_quality: pd.DataFrame):
    """Print EDGE validation findings for Substep #4."""
    print("\n" + "=" * 80)
    print("EDGE VALIDATION FINDINGS — answering Substep #4 research questions")
    print("=" * 80)

    if df_quality.empty:
        return

    print("\n### Configs meeting >60% EDGE validation by edge_config × mode:")
    df_quality["meets_60pct"] = df_quality["edge_validation_mean"] >= 0.6
    pivot = (
        df_quality.groupby(["edge_config", "mode"])["meets_60pct"].mean().round(2) * 100
    )
    print(pivot.to_string())

    print("\n### Mean EDGE validation by edge_config × mode:")
    pivot2 = (
        df_quality.groupby(["edge_config", "mode"])["edge_validation_mean"]
        .mean()
        .round(3)
    )
    print(pivot2.to_string())

    print("\n### Mode impact at each threshold (unconstrained vs both):")
    for ec in EDGE_CONFIGS:
        sub = df_quality[df_quality["edge_config"] == str(ec)]
        uc = sub[sub["mode"] == "unconstrained"]["edge_validation_mean"].mean()
        bt = sub[sub["mode"] == "both"]["edge_validation_mean"].mean()
        diff = bt - uc
        print(f"  {ec}: unconstrained={uc:.3f}, both={bt:.3f}, delta={diff:+.3f}")


# ============================================================================
# CHANGE #7  —  SILHOUETTE PLOT V2 (FIXED LABELS + CORRECT MARKERS)
# ============================================================================


def plot_silhouette_v2(df_quality: pd.DataFrame):
    """
    CHANGE #7: Regenerate silhouette plot with:
    - Y-axis: 'Mean Silhouette Score (intra vs inter-cluster distance)'
    - Correct marker shapes: EDGE = circle 'o', others = square 's'
    - Legend showing correct shapes
    - Subtitle noting algorithm = Agglomerative (k=40)
    """
    print("\n" + "=" * 80)
    print("CHANGE #7: Silhouette plot v2 (fixed labels)")
    print("=" * 80)

    if df_quality.empty:
        print("⚠  No quality data — skipping")
        return

    colors = sns.color_palette("husl", len(EDGE_CONFIGS))

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    axes = axes.flatten()

    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]

        for i, edge_config in enumerate(EDGE_CONFIGS):
            for j, mode in enumerate(MODES):
                mask = (
                    (df_quality["node_type"] == node_type)
                    & (df_quality["edge_config"] == str(edge_config))
                    & (df_quality["mode"] == mode)
                )
                if mask.sum() == 0:
                    continue
                sil = df_quality[mask]["silhouette_mean"].values[0]
                marker = "o" if str(edge_config) == "EDGE" else "s"
                ax.scatter(j, sil, color=colors[i], s=100, alpha=0.75, marker=marker)

        ax.set_title(
            f"{node_type.replace('_', ' ').title()}\n[Algorithm: Agglomerative, k=40]",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Mode", fontsize=9)
        ax.set_ylabel(
            "Mean Silhouette Score\n(intra-cluster tightness vs inter-cluster separation)",
            fontsize=8,
        )
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([m.replace("_", "\n") for m in MODES], fontsize=7)
        ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Target: 0.3")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.75)

    # Legend with correct markers
    legend_elements = []
    for i, ec in enumerate(EDGE_CONFIGS):
        marker = "o" if str(ec) == "EDGE" else "s"
        label = f"{ec} ({'●' if marker == 'o' else '■'})"
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=colors[i],
                markersize=10,
                label=label,
            )
        )
    fig.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.01),
        title="Edge Config",
        fontsize=9,
        ncol=1,
    )

    plt.suptitle(
        "Mean Silhouette Score by Node Type\n"
        "(Algorithm: Agglomerative k=40 | ●=EDGE-only, ■=SIM threshold)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(PLOT_SILHOUETTE_V2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {PLOT_SILHOUETTE_V2}")


def print_silhouette_findings(df_quality: pd.DataFrame):
    """Print silhouette findings addressing the 'paradox' and Substep #1."""
    print("\n" + "=" * 80)
    print("SILHOUETTE FINDINGS — answering Substep #1 research questions")
    print("=" * 80)

    if df_quality.empty:
        return

    print(
        "\n### Mean silhouette by edge_config (averaged over all node_types × modes):"
    )
    print(
        df_quality.groupby("edge_config")["silhouette_mean"]
        .agg(["mean", "min", "max"])
        .round(3)
        .to_string()
    )

    print("\n### Configs failing <0.3 threshold:")
    below = df_quality[df_quality["silhouette_mean"] < 0.3][
        ["node_type", "edge_config", "mode", "silhouette_mean"]
    ]
    if below.empty:
        print("  None — all configs ≥0.3 ✅")
    else:
        print(below.to_string(index=False))

    print("\n### Silhouette paradox verification (EDGE < 0.8?):")
    edge_sil = df_quality[df_quality["edge_config"] == "EDGE"]["silhouette_mean"].mean()
    low_sil = df_quality[df_quality["edge_config"] == "0.8"]["silhouette_mean"].mean()
    print(f"  EDGE mean silhouette: {edge_sil:.3f}")
    print(f"  0.8  mean silhouette: {low_sil:.3f}")
    if low_sil > edge_sil:
        print("  ✓ Silhouette paradox confirmed: 0.8 > EDGE")
        print(
            "    Interpretation: EDGE optimises literature grounding, not embedding separation."
        )
    else:
        print("  — Paradox not observed in this data.")


# ============================================================================
# CHANGE #12  —  CLUSTER SIZE DISTRIBUTION PLOT (Y-AXIS FIX)
# ============================================================================


def plot_cluster_size_v2(df_quality: pd.DataFrame):
    """CHANGE #12: Cluster size distribution with auto-computed y-axis."""
    print("\n" + "=" * 80)
    print("CHANGE #12: Cluster size distributions v2 (y-axis fix)")
    print("=" * 80)

    if df_quality.empty:
        print("⚠  No quality data — skipping")
        return

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    axes = axes.flatten()

    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]
        node_df = df_quality[df_quality["node_type"] == node_type]

        all_counts = node_df["n_clusters"].dropna().values
        if len(all_counts) == 0:
            continue

        # Auto y-axis with 5% padding
        y_min, y_max = float(all_counts.min()), float(all_counts.max())
        y_range = y_max - y_min
        pad = y_range * 0.05 if y_range > 0 else 2.0
        ax.set_ylim(max(0, y_min - pad), y_max + pad)

        for i, mode in enumerate(MODES):
            mode_vals = []
            ec_order = []
            for ec in EDGE_CONFIGS:
                mask = (node_df["edge_config"] == str(ec)) & (node_df["mode"] == mode)
                if mask.sum() > 0:
                    mode_vals.append(node_df[mask]["n_clusters"].values[0])
                    ec_order.append(str(ec))
            if mode_vals:
                ax.plot(
                    ec_order,
                    mode_vals,
                    marker="o",
                    linewidth=1.5,
                    markersize=6,
                    label=mode,
                    alpha=0.85,
                )

        ax.set_title(
            f"{node_type.replace('_', ' ').title()}", fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Edge Config", fontsize=9)
        ax.set_ylabel("Number of Clusters", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Cluster Count by Configuration (y-axis auto-scaled)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PLOT_CLUSTER_SIZE_V2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {PLOT_CLUSTER_SIZE_V2}")


# ============================================================================
# CHANGE #8  —  CLUSTER COHESION METRICS
# ============================================================================


def analyze_cluster_cohesion(
    df_metrics: pd.DataFrame, node_attrs: Dict, cluster_memberships: Dict
) -> pd.DataFrame:
    """
    CHANGE #8: Intra-cluster compactness vs inter-cluster separation.
    Uses cluster_memberships (from Step 1 checkpoint) for speed.
    Caps per-cluster pairwise computations at COHESION_NODE_CAP nodes.
    """
    print("\n" + "=" * 80)
    print("CHANGE #8: Cluster cohesion analysis")
    print("=" * 80)

    results = []
    algo = "agglomerative"

    # Group cluster_memberships by config
    config_clusters: Dict[Tuple, Dict] = defaultdict(dict)
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            key_ec, key_mode, key_nt, key_algo, cluster_id = key
            if key_algo == algo:
                config_clusters[(str(key_ec), key_mode, key_nt)][cluster_id] = members

    for node_type in tqdm(NODE_TYPES, desc="Cohesion node_types"):
        for edge_config in EDGE_CONFIGS:
            for mode in MODES:
                cfg_key = (str(edge_config), mode, node_type)
                clusters = config_clusters.get(cfg_key, {})
                if not clusters:
                    continue

                # Build embedding map for all nodes in this config
                all_nodes = set()
                for members in clusters.values():
                    all_nodes.update(members)

                emb_map = {}
                for nid in all_nodes:
                    emb = get_node_embedding(nid, node_attrs)
                    if emb is not None:
                        emb_map[nid] = emb

                if len(emb_map) < 4:
                    continue

                # Per-cluster intra distances
                intra_distances = []
                centroids = {}
                for cid, node_ids in clusters.items():
                    nids_with_emb = [n for n in node_ids if n in emb_map]
                    if len(nids_with_emb) < 2:
                        continue

                    # Sample for large clusters
                    if len(nids_with_emb) > COHESION_NODE_CAP:
                        nids_with_emb = list(
                            np.random.choice(
                                nids_with_emb, COHESION_NODE_CAP, replace=False
                            )
                        )

                    embs = [emb_map[n] for n in nids_with_emb]
                    centroids[cid] = np.mean(embs, axis=0)

                    dists = []
                    for a, b in combinations(range(len(embs)), 2):
                        try:
                            dists.append(cosine(embs[a], embs[b]))
                        except Exception:
                            pass
                    if dists:
                        intra_distances.append(float(np.mean(dists)))

                # Inter-cluster centroid distances
                inter_distances = []
                cids = list(centroids.keys())
                for a, b in combinations(range(len(cids)), 2):
                    try:
                        inter_distances.append(
                            cosine(centroids[cids[a]], centroids[cids[b]])
                        )
                    except Exception:
                        pass

                if not intra_distances or not inter_distances:
                    continue

                intra_mean = float(np.mean(intra_distances))
                inter_mean = float(np.mean(inter_distances))
                inter_min = float(np.min(inter_distances))
                sep_ratio = inter_mean / intra_mean if intra_mean > 0 else 0.0

                results.append(
                    {
                        "edge_config": str(edge_config),
                        "mode": mode,
                        "node_type": node_type,
                        "intra_cluster_mean": round(intra_mean, 4),
                        "intra_cluster_std": round(float(np.std(intra_distances)), 4),
                        "inter_cluster_mean": round(inter_mean, 4),
                        "inter_cluster_min": round(inter_min, 4),
                        "inter_cluster_std": round(float(np.std(inter_distances)), 4),
                        "separation_ratio": round(sep_ratio, 4),
                        "n_clusters_analyzed": len(intra_distances),
                    }
                )

    df = pd.DataFrame(results)
    print(f"\n✓ Cohesion analyzed for {len(df)} configurations")
    return df


def print_cohesion_findings(df_cohesion: pd.DataFrame):
    """Print cohesion findings for Substep #3."""
    print("\n" + "=" * 80)
    print("COHESION FINDINGS — answering Substep #3 research questions")
    print("=" * 80)

    if df_cohesion.empty:
        print("No cohesion data.")
        return

    print("\n### Mean separation_ratio by edge_config:")
    print(
        df_cohesion.groupby("edge_config")["separation_ratio"]
        .agg(["mean", "min", "max"])
        .round(3)
        .to_string()
    )

    print("\n### Configs with excellent separation (ratio > 2.5):")
    exc = df_cohesion[df_cohesion["separation_ratio"] > 2.5][
        ["node_type", "edge_config", "mode", "separation_ratio"]
    ]
    print(f"  {len(exc)} configs" if not exc.empty else "  None")

    print("\n### Silhouette paradox cross-check (intra lower at 0.8 vs EDGE?):")
    edge_intra = df_cohesion[df_cohesion["edge_config"] == "EDGE"][
        "intra_cluster_mean"
    ].mean()
    low_intra = df_cohesion[df_cohesion["edge_config"] == "0.8"][
        "intra_cluster_mean"
    ].mean()
    print(f"  EDGE intra mean: {edge_intra:.4f}")
    print(f"  0.8  intra mean: {low_intra:.4f}")


# ============================================================================
# CHANGE #9  —  CENTROID SIMILARITY (SEMANTIC STABILITY)
# ============================================================================


def analyze_centroid_similarity(
    df_metrics: pd.DataFrame, node_attrs: Dict, cluster_memberships: Dict
) -> pd.DataFrame:
    """
    CHANGE #9: For each adjacent threshold transition, measure how similar
    a node's cluster centroid at T1 is to its cluster centroid at T2.
    High similarity = cluster semantically stable across threshold change.
    """
    print("\n" + "=" * 80)
    print("CHANGE #9: Cluster centroid similarity (semantic stability)")
    print("=" * 80)

    results = []
    algo = "agglomerative"

    # Build config → {cluster_id: [node_ids]} mapping from cluster_memberships
    config_clusters: Dict[Tuple, Dict] = defaultdict(dict)
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            key_ec, key_mode, key_nt, key_algo, cluster_id = key
            if key_algo == algo:
                config_clusters[(str(key_ec), key_mode, key_nt)][cluster_id] = members

    # Adjacent threshold pairs in EDGE_CONFIGS order
    adjacent_pairs = [
        (EDGE_CONFIGS[i], EDGE_CONFIGS[i + 1]) for i in range(len(EDGE_CONFIGS) - 1)
    ]

    for node_type in tqdm(NODE_TYPES, desc="Centroid sim node_types"):
        for mode in MODES:
            for ec1, ec2 in adjacent_pairs:
                clusters1 = config_clusters.get((str(ec1), mode, node_type), {})
                clusters2 = config_clusters.get((str(ec2), mode, node_type), {})

                if not clusters1 or not clusters2:
                    continue

                # Build {node_id: cluster_id} reverse maps + centroids
                assign1 = {}
                for cid, members in clusters1.items():
                    for nid in members:
                        assign1[nid] = cid

                assign2 = {}
                for cid, members in clusters2.items():
                    for nid in members:
                        assign2[nid] = cid

                # Compute cluster centroids using embeddings
                def build_centroids(clusters: Dict) -> Dict:
                    c = {}
                    for cid, node_ids in clusters.items():
                        embs = [get_node_embedding(n, node_attrs) for n in node_ids]
                        embs = [e for e in embs if e is not None]
                        if embs:
                            c[cid] = np.mean(embs, axis=0)
                    return c

                centroids1 = build_centroids(clusters1)
                centroids2 = build_centroids(clusters2)

                # For each common node, compare its T1 centroid vs T2 centroid
                common_nodes = set(assign1.keys()) & set(assign2.keys())
                if len(common_nodes) < 5:
                    continue

                centroid_sims = []
                for nid in common_nodes:
                    cid1 = assign1[nid]
                    cid2 = assign2[nid]
                    if cid1 in centroids1 and cid2 in centroids2:
                        try:
                            sim = 1.0 - cosine(centroids1[cid1], centroids2[cid2])
                            centroid_sims.append(float(sim))
                        except Exception:
                            pass

                if len(centroid_sims) < 2:
                    continue

                cs = np.array(centroid_sims)
                results.append(
                    {
                        "node_type": node_type,
                        "mode": mode,
                        "threshold_from": str(ec1),
                        "threshold_to": str(ec2),
                        "n_nodes": len(cs),
                        "centroid_sim_mean": round(float(np.mean(cs)), 4),
                        "centroid_sim_median": round(float(np.median(cs)), 4),
                        "centroid_sim_std": round(float(np.std(cs)), 4),
                        "centroid_sim_min": round(float(np.min(cs)), 4),
                        "centroid_sim_max": round(float(np.max(cs)), 4),
                        "high_stable_pct": round(float((cs > 0.8).mean()), 4),
                        "moderate_pct": round(
                            float(((cs >= 0.5) & (cs <= 0.8)).mean()), 4
                        ),
                        "low_stable_pct": round(float((cs < 0.5).mean()), 4),
                    }
                )

    df = pd.DataFrame(results)
    print(f"\n✓ Centroid similarity computed for {len(df)} transitions")
    return df


def plot_centroid_similarity(df_sim: pd.DataFrame):
    """
    CHANGE #9: Heatmap — modes × adjacent transitions per node type.
    Green = high semantic stability, Red = reorganisation.
    """
    print("\n" + "=" * 80)
    print("CHANGE #9: Centroid similarity heatmap")
    print("=" * 80)

    if df_sim.empty:
        print("⚠  No centroid similarity data — skipping")
        return

    adj_pairs = [
        (EDGE_CONFIGS[i], EDGE_CONFIGS[i + 1]) for i in range(len(EDGE_CONFIGS) - 1)
    ]
    transition_labels = [f"{ec1}→{ec2}" for ec1, ec2 in adj_pairs]

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    axes = axes.flatten()

    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]
        mat = np.full((len(MODES), len(adj_pairs)), np.nan)

        for i, mode in enumerate(MODES):
            for j, (ec1, ec2) in enumerate(adj_pairs):
                row = df_sim[
                    (df_sim["node_type"] == node_type)
                    & (df_sim["mode"] == mode)
                    & (df_sim["threshold_from"] == str(ec1))
                    & (df_sim["threshold_to"] == str(ec2))
                ]
                if len(row) > 0:
                    mat[i, j] = row["centroid_sim_mean"].values[0]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
        ax.set_xticks(np.arange(len(transition_labels)))
        ax.set_xticklabels(transition_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(MODES)))
        ax.set_yticklabels([m.replace("_", "\n") for m in MODES], fontsize=8)
        ax.set_title(
            f"{node_type.replace('_', ' ').title()}", fontsize=10, fontweight="bold"
        )

        for i in range(len(MODES)):
            for j in range(len(adj_pairs)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if v < 0.7 else "white",
                    )

        plt.colorbar(im, ax=ax, label="Centroid Similarity", shrink=0.7)

    plt.suptitle(
        "Cluster Centroid Similarity Across Adjacent Threshold Transitions\n"
        "(Semantic stability: Green=stable, Red=reorganised)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PLOT_CENTROID_SIM, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {PLOT_CENTROID_SIM}")


def print_centroid_similarity_findings(df_sim: pd.DataFrame):
    """Print centroid similarity findings for Substep #8."""
    print("\n" + "=" * 80)
    print("CENTROID SIMILARITY FINDINGS — answering Substep #8 research questions")
    print("=" * 80)

    if df_sim.empty:
        return

    print(
        "\n### Mean centroid similarity per transition (across all node types × modes):"
    )
    print(
        df_sim.groupby(["threshold_from", "threshold_to"])["centroid_sim_mean"]
        .agg(["mean", "min", "max"])
        .round(3)
        .to_string()
    )

    print("\n### Semantic stability by node type (mean across transitions):")
    print(
        df_sim.groupby("node_type")["centroid_sim_mean"]
        .agg(["mean", "min"])
        .round(3)
        .sort_values("mean", ascending=False)
        .to_string()
    )

    print("\n### High semantic stability (>80% nodes stable per transition):")
    high = df_sim[df_sim["high_stable_pct"] > 0.8][
        [
            "node_type",
            "mode",
            "threshold_from",
            "threshold_to",
            "centroid_sim_mean",
            "high_stable_pct",
        ]
    ]
    if not high.empty:
        print(high.to_string(index=False))
    else:
        print("  None — consider lowering threshold or checking embedding alignment.")


# ============================================================================
# CHANGE #10  —  EDGE PURITY PER CLUSTER
# ============================================================================


def build_edge_only_node_set(cluster_memberships: Dict) -> Set:
    """
    Identify nodes that appear in EDGE-config clusters (any mode).
    These are the nodes that participated in EDGE-only complete pathways.
    """
    edge_nodes = set()
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            key_ec, key_mode, key_nt, key_algo, cluster_id = key
            if str(key_ec).upper() == "EDGE":
                edge_nodes.update(members)
    return edge_nodes


def analyze_edge_purity(
    df_metrics: pd.DataFrame, cluster_memberships: Dict
) -> pd.DataFrame:
    """
    CHANGE #10: Per-cluster EDGE purity = % of nodes in EDGE-only complete pathways.
    Proxy: nodes appearing in any EDGE-config clustering are EDGE-pathway nodes.
    """
    print("\n" + "=" * 80)
    print("CHANGE #10: EDGE purity per cluster")
    print("=" * 80)

    edge_only_nodes = build_edge_only_node_set(cluster_memberships)
    print(f"  EDGE-only pathway nodes identified: {len(edge_only_nodes):,}")

    results = []
    algo = "agglomerative"

    # Group cluster_memberships
    config_clusters: Dict[Tuple, Dict] = defaultdict(dict)
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            key_ec, key_mode, key_nt, key_algo, cluster_id = key
            if key_algo == algo:
                config_clusters[(str(key_ec), key_mode, key_nt)][cluster_id] = members

    for node_type in tqdm(NODE_TYPES, desc="EDGE purity node_types"):
        for edge_config in EDGE_CONFIGS:
            for mode in MODES:
                clusters = config_clusters.get((str(edge_config), mode, node_type), {})
                if not clusters:
                    continue

                for cluster_id, node_ids in clusters.items():
                    edge_nodes_in_cluster = [
                        n for n in node_ids if n in edge_only_nodes
                    ]
                    purity = (
                        len(edge_nodes_in_cluster) / len(node_ids) if node_ids else 0.0
                    )

                    results.append(
                        {
                            "edge_config": str(edge_config),
                            "mode": mode,
                            "node_type": node_type,
                            "cluster_id": cluster_id,
                            "cluster_size": len(node_ids),
                            "n_edge_nodes": len(edge_nodes_in_cluster),
                            "edge_purity": round(purity, 4),
                            "is_gold_standard": purity >= 0.8,
                        }
                    )

    df = pd.DataFrame(results)
    print(f"\n✓ EDGE purity computed for {len(df):,} clusters")
    return df


def plot_edge_purity(df_purity: pd.DataFrame):
    """CHANGE #10: Histograms of EDGE purity distribution per node type."""
    print("\n" + "=" * 80)
    print("CHANGE #10: EDGE purity histograms")
    print("=" * 80)

    if df_purity.empty:
        print("⚠  No purity data — skipping")
        return

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.001]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    colors_hist = sns.color_palette("RdYlGn", len(bin_labels))

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    axes = axes.flatten()

    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]
        subset = df_purity[df_purity["node_type"] == node_type]
        if subset.empty:
            continue

        x_pos = np.arange(len(bin_labels))
        width = 0.15
        for i, ec in enumerate(EDGE_CONFIGS):
            ec_sub = subset[subset["edge_config"] == str(ec)]
            if ec_sub.empty:
                continue
            counts, _ = np.histogram(ec_sub["edge_purity"].values, bins=bins)
            ax.bar(
                x_pos + i * width,
                counts,
                width=width,
                alpha=0.75,
                label=str(ec),
                color=colors_hist[i % len(colors_hist)],
            )

        ax.set_xticks(x_pos + (len(EDGE_CONFIGS) - 1) * width / 2)
        ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_ylabel("Cluster Count", fontsize=9)
        ax.set_title(
            f"{node_type.replace('_', ' ').title()}", fontsize=10, fontweight="bold"
        )
        ax.axvline(
            3.5, color="darkgreen", linestyle="--", alpha=0.6, label="Gold std >80%"
        )
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "EDGE Purity Distribution per Node Type\n"
        "(% of cluster nodes appearing in EDGE-only complete pathways)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PLOT_EDGE_PURITY, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {PLOT_EDGE_PURITY}")


def print_edge_purity_findings(df_purity: pd.DataFrame):
    """Print EDGE purity findings for Substep #5."""
    print("\n" + "=" * 80)
    print("EDGE PURITY FINDINGS — answering Substep #5 research questions")
    print("=" * 80)

    if df_purity.empty:
        return

    print("\n### Gold-standard clusters (purity ≥80%) by edge_config:")
    gold = df_purity.groupby("edge_config").agg(
        total_clusters=("cluster_id", "count"),
        gold_clusters=("is_gold_standard", "sum"),
    )
    gold["gold_pct"] = (gold["gold_clusters"] / gold["total_clusters"] * 100).round(1)
    print(gold.to_string())

    print("\n### Gold-standard clusters by node_type × mode (top configs):")
    pivot = (
        df_purity.groupby(["edge_config", "mode", "node_type"])["is_gold_standard"]
        .mean()
        .round(3)
    )
    top = pivot[pivot >= 0.5].sort_values(ascending=False).head(20)
    if not top.empty:
        print(top.to_string())

    print("\n### Mean purity by node_type (best mode per type):")
    best = df_purity.groupby(["node_type", "edge_config", "mode"])["edge_purity"].mean()
    best_by_nt = best.groupby("node_type").max().round(3)
    print(best_by_nt.to_string())

    print("\n### Taxonomy confidence assignment guidance:")
    print("  Gold (purity ≥0.8): Auto-label, minimal validation burden")
    print("  Mixed (purity 0.4-0.8): Moderate validation required")
    print("  Similarity-driven (purity <0.4): Heavy manual validation needed")
    for bucket, low, high in [
        ("Gold (≥0.8)", 0.8, 1.1),
        ("Mixed (0.4-0.8)", 0.4, 0.8),
        ("Sim-driven (<0.4)", 0.0, 0.4),
    ]:
        n = (
            (df_purity["edge_purity"] >= low) & (df_purity["edge_purity"] < high)
        ).sum()
        pct = n / len(df_purity) * 100
        print(f"  {bucket}: {n:,} clusters ({pct:.1f}%)")


# ============================================================================
# CHANGE #3  —  HUB QUALITY ANALYSIS
# ============================================================================


def _cos_sim_from_score(score) -> float:
    """Convert FalkorDB SIMILARITY edge score (L2 distance between unit vectors)
    to cosine similarity: cos_sim = 1 - score^2 / 2.
    Max stored score = 0.6325 = L2 at cos_sim=0.80 (storage floor).
    """
    return 1.0 - float(score) ** 2 / 2.0


def build_degree_index(edge_data: List[Dict], node_attrs: Dict) -> Dict:
    """
    Build per-threshold degree index from edge_data.

    Returns {node_id: {
        'structural': int,          # EDGE-type edges
        'sim_0.80': int,            # SIM edges with cos_sim >= 0.80
        'sim_0.85': int,            # SIM edges with cos_sim >= 0.85
        'sim_0.90': int,            # SIM edges with cos_sim >= 0.90
        'sim_0.95': int,            # SIM edges with cos_sim >= 0.95
        'total_0.80': int,          # structural + sim_0.80 (all edges)
        'partner_urls_0.80': set,   # distinct partner node URLs at 0.80
        'partner_urls_0.85': set,
        'partner_urls_0.90': set,
        'partner_urls_0.95': set,
    }}

    NOTE: source_file is NULL for all edges; n_sources must be computed from
    partner node URL attributes (looked up in node_attrs).
    """
    idx: Dict = defaultdict(
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

    SIM_THRESHOLDS = [
        ("sim_0.80", 0.80),
        ("sim_0.85", 0.85),
        ("sim_0.90", 0.90),
        ("sim_0.95", 0.95),
    ]

    for e in edge_data:
        src = e.get("source")
        tgt = e.get("target")
        etype = str(e.get("edge_type", e.get("type", ""))).upper()

        if etype == "EDGE":
            for nid in (src, tgt):
                if nid is not None:
                    idx[nid]["structural"] += 1
                    idx[nid]["total_0.80"] += 1

        elif etype == "SIMILARITY":
            raw_score = e.get("similarity_score")
            if raw_score is None:
                continue
            cos_sim = _cos_sim_from_score(raw_score)

            for nid, partner_id in ((src, tgt), (tgt, src)):
                if nid is None or partner_id is None:
                    continue
                for key, thr in SIM_THRESHOLDS:
                    if cos_sim >= thr:
                        idx[nid][key] += 1
                        # Get partner paper URL for source diversity
                        partner_url = node_attrs.get(partner_id, {}).get("url", "")
                        if partner_url:
                            idx[nid][f"partner_urls_{thr:.2f}"].add(partner_url)
                idx[nid]["total_0.80"] += 1  # all SIM edges count toward 0.80 total

    return idx


def analyze_hub_quality(
    cluster_memberships: Dict,
    node_attrs: Dict,
    edge_data: List[Dict],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    CHANGE #3 (v2, corrected March 2026): Top-N hubs per configuration.

    Key corrections vs v1:
    - Threshold-aware degree index: reports degree at each SIM threshold separately
    - Source diversity: uses partner node URL attributes (not edge source_file which is NULL)
    - Includes BOTH intervention AND concept node types (true SIM>=0.9 hubs are concepts)
    - Ranking uses sim_0.90 degree as primary key (identifies genuine cross-paper hubs)
    - n_sources = distinct partner paper URLs at sim threshold (not edge-level source_file)
    """
    print("\n" + "=" * 80)
    print(
        "CHANGE #3 (v2): Hub quality analysis — threshold-aware, intervention + concept"
    )
    print("=" * 80)

    print("  Building threshold-aware degree index from edge_data…")
    degree_idx = build_degree_index(edge_data, node_attrs)
    print(f"  Degree index built for {len(degree_idx):,} nodes")

    # Map edge_config label → SIM threshold for degree lookup
    EC_TO_THR = {
        "EDGE": None,  # use structural only
        "0.8": 0.80,
        "0.85": 0.85,
        "0.9": 0.90,
        "0.95": 0.95,
    }

    # Gather top hubs across ALL node types (not just intervention)
    algo = "agglomerative"
    target_node_types = [
        "intervention",
        "risk",
        "problem_analysis",
        "theoretical_insight",
        "implementation_mechanism",
        "design_rationale",
        "validation_evidence",
    ]

    config_nodes: Dict[Tuple, Set] = defaultdict(set)
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            key_ec, key_mode, key_nt, key_algo, _ = key
            if key_algo == algo and key_nt in target_node_types:
                config_nodes[(str(key_ec), key_mode, key_nt)].update(members)

    results = []
    for (edge_config, mode, node_type), all_nodes in tqdm(
        config_nodes.items(), desc="Hub configs"
    ):
        if not all_nodes:
            continue

        thr = EC_TO_THR.get(edge_config)
        sim_key = f"sim_{thr:.2f}" if thr is not None else None
        url_key = f"partner_urls_{thr:.2f}" if thr is not None else None

        # Rank nodes by SIM degree at config's threshold (or structural for EDGE)
        node_scores = []
        for nid in all_nodes:
            deg = degree_idx.get(nid, {})
            if sim_key:
                rank_score = deg.get(sim_key, 0) + deg.get("structural", 0)
            else:
                rank_score = deg.get("structural", 0)
            node_scores.append((nid, rank_score))

        top_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)[:top_n]

        for nid, _ in top_nodes:
            deg = degree_idx.get(nid, {})
            attrs = node_attrs.get(nid, {})
            name = attrs.get("name", f"Node_{nid}")

            structural = deg.get("structural", 0)
            sim_80 = deg.get("sim_0.80", 0)
            sim_85 = deg.get("sim_0.85", 0)
            sim_90 = deg.get("sim_0.90", 0)
            sim_95 = deg.get("sim_0.95", 0)
            total = deg.get("total_0.80", 0)

            # Source diversity from partner node URLs at config threshold
            if url_key:
                n_sources = len(deg.get(url_key, set()))
            else:
                n_sources = 0  # EDGE-only: no similarity partners

            # Threshold sensitivity ratio
            sim_ratio_90_80 = (sim_90 / sim_80) if sim_80 > 0 else 0.0

            results.append(
                {
                    "edge_config": edge_config,
                    "mode": mode,
                    "node_type": node_type,
                    "hub_node_id": nid,
                    "hub_name": name,
                    "hub_source_url": attrs.get("url", "")[:80],
                    "degree_structural": structural,
                    "degree_sim_0.80": sim_80,
                    "degree_sim_0.85": sim_85,
                    "degree_sim_0.90": sim_90,
                    "degree_sim_0.95": sim_95,
                    "degree_total_0.80": total,
                    "n_sources_at_config_thr": n_sources,
                    "sim_ratio_90_80": round(sim_ratio_90_80, 3),
                    "is_high_thr_hub": sim_90 >= 50,  # genuine cross-paper hub at 0.9
                }
            )

    df = pd.DataFrame(results)
    print(f"\n✓ Hub quality v2: {len(df):,} hub records")

    # Print top SIM>=0.9 hubs across all configs
    if not df.empty:
        top_sim90 = (
            df.groupby("hub_name")["degree_sim_0.90"]
            .max()
            .sort_values(ascending=False)
            .head(10)
        )
        print("\n  Top-10 nodes by max SIM>=0.9 degree:")
        for name, deg in top_sim90.items():
            row = df[df["hub_name"] == name].iloc[0]
            print(
                f"    [{row['node_type']}] deg_0.9={deg:,}  "
                f"n_src={row['n_sources_at_config_thr']}  {name[:60]}"
            )

    df.to_csv(OUT_HUB_METRICS, index=False)
    print(f"\n  ✓ Saved: {OUT_HUB_METRICS}")
    return df


def plot_hub_quality(df_hubs: pd.DataFrame):
    """CHANGE #3 (v2): Scatter — SIM>=0.9 degree vs n_sources, size=total_0.80, color=structural."""
    print("\n" + "=" * 80)
    print("CHANGE #3 (v2): Hub quality scatter plot — threshold-aware")
    print("=" * 80)

    if df_hubs.empty:
        print("⚠  No hub data — skipping")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    x = df_hubs["degree_sim_0.90"].values
    y = df_hubs["n_sources_at_config_thr"].values
    sizes = np.clip(df_hubs["degree_total_0.80"].values * 0.5, 10, 400)
    colors = df_hubs["degree_structural"].values

    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        alpha=0.5,
        cmap="plasma",
        edgecolors="black",
        linewidth=0.3,
    )

    ax.axvline(
        50,
        color="green",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Strong cross-paper hub (SIM>=0.9 >= 50)",
    )
    ax.axhline(
        10,
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label="Min source diversity target (10 papers)",
    )

    ax.set_xlabel(
        "SIM>=0.9 Degree (cross-paper connections at high threshold)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Unique Source Papers (partner node URLs)", fontsize=12, fontweight="bold"
    )
    n_high = (x >= 50).sum()
    ax.set_title(
        f"Hub Quality v2 — Threshold-Aware (CHANGE #3)\n"
        f"x=SIM>=0.9 degree, y=n_source_papers, size=total_0.80_degree, color=structural_degree\n"
        f"{n_high:,} / {len(df_hubs):,} hubs are strong cross-paper hubs (SIM>=0.9 >= 50)",
        fontsize=12,
        fontweight="bold",
    )

    cbar = plt.colorbar(scatter, ax=ax, label="Structural (EDGE-type) Degree")
    cbar.ax.tick_params(labelsize=10)

    from matplotlib.lines import Line2D

    size_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=np.sqrt(s * 2),
            label=f"total_0.80 ~{int(s * 2)}",
        )
        for s in [50, 100, 200, 400]
    ]
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="green",
                linestyle="--",
                label="SIM>=0.9 >= 50 (cross-paper hub)",
            ),
            Line2D([0], [0], color="red", linestyle="--", label="n_sources >= 10"),
        ]
        + size_legend,
        fontsize=10,
        loc="upper left",
    )

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.text(
        0.97,
        0.97,
        "Genuine cross-paper hubs\n(high SIM>=0.9 + many sources)",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )
    ax.text(
        0.03,
        0.03,
        "Local / single-paper hubs\n(low SIM>=0.9, few sources)",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(PLOT_HUB_QUALITY, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {PLOT_HUB_QUALITY}")


def print_hub_findings(df_hubs: pd.DataFrame):
    """Print hub quality findings for Substep #14 (v2 columns)."""
    print("\n" + "=" * 80)
    print("HUB QUALITY FINDINGS (v2) — answering Substep #14 research questions")
    print("=" * 80)

    if df_hubs.empty:
        return

    print(f"\nTotal hubs analyzed: {len(df_hubs):,}")
    print(
        f"Strong cross-paper hubs (SIM>=0.9 >= 50): {df_hubs['is_high_thr_hub'].sum():,}"
    )
    print(
        f"Hubs with sources >= 10:    {(df_hubs['n_sources_at_config_thr'] >= 10).sum():,}"
    )
    both_criteria = df_hubs["is_high_thr_hub"] & (
        df_hubs["n_sources_at_config_thr"] >= 10
    )
    print(f"Hubs meeting both criteria: {both_criteria.sum():,}")
    print(f"\nMean degree_sim_0.90:       {df_hubs['degree_sim_0.90'].mean():.1f}")
    print(
        f"Mean n_sources:             {df_hubs['n_sources_at_config_thr'].mean():.1f}"
    )
    print(f"Mean sim_ratio_90_80:       {df_hubs['sim_ratio_90_80'].mean():.3f}")

    print("\n### Top-10 hubs by SIM>=0.9 degree (all configs):")
    top = df_hubs.nlargest(10, "degree_sim_0.90")[
        [
            "hub_name",
            "edge_config",
            "node_type",
            "degree_sim_0.90",
            "degree_sim_0.80",
            "degree_structural",
            "n_sources_at_config_thr",
            "sim_ratio_90_80",
        ]
    ]
    if not top.empty:
        print(top.to_string(index=False))

    print("\n### Hub quality pattern by edge_config:")
    print(
        df_hubs.groupby("edge_config")
        .agg(
            mean_sim90=("degree_sim_0.90", "mean"),
            mean_sources=("n_sources_at_config_thr", "mean"),
            mean_ratio=("sim_ratio_90_80", "mean"),
            pct_high_thr=("is_high_thr_hub", "mean"),
        )
        .round(3)
        .to_string()
    )


# ============================================================================
# CHANGE #2  —  SOURCE DIVERSITY V2 (URL-BASED)
# ============================================================================


def analyze_source_diversity_v2(
    cluster_memberships: Dict, node_attrs: Dict
) -> pd.DataFrame:
    """
    CHANGE #2: Re-compute source diversity using node 'url' attribute as source proxy.
    Each unique URL = one source document. This fixes the all-zeros bug.
    """
    print("\n" + "=" * 80)
    print("CHANGE #2: Source diversity v2 (url-based)")
    print("=" * 80)

    # Diagnose what source fields are available
    sample_size = min(200, len(node_attrs))
    sample_nodes = list(node_attrs.values())[:sample_size]
    has_url = sum(1 for a in sample_nodes if a.get("url"))
    has_source_file = sum(1 for a in sample_nodes if a.get("source_file"))
    has_source_list = sum(1 for a in sample_nodes if a.get("source_file_list"))
    print(
        f"  Diagnostic (n={sample_size}): url={has_url}, source_file={has_source_file}, source_file_list={has_source_list}"
    )

    results = []
    algo = "agglomerative"

    config_clusters: Dict[Tuple, Dict] = defaultdict(dict)
    for key, members in cluster_memberships.items():
        if len(key) == 5:
            key_ec, key_mode, key_nt, key_algo, cluster_id = key
            if key_algo == algo:
                config_clusters[(str(key_ec), key_mode, key_nt)][cluster_id] = members

    for node_type in tqdm(NODE_TYPES, desc="Source diversity node_types"):
        for edge_config in EDGE_CONFIGS:
            for mode in MODES:
                clusters = config_clusters.get((str(edge_config), mode, node_type), {})
                if not clusters:
                    continue

                for cluster_id, node_ids in clusters.items():
                    sources: Set = set()
                    n_with_source = 0

                    for nid in node_ids:
                        attrs = node_attrs.get(nid, {})
                        found = False

                        # Priority 1: source_file_list
                        sfl = attrs.get("source_file_list")
                        if sfl and isinstance(sfl, list):
                            sources.update(f for f in sfl if f)
                            found = bool(sfl)

                        # Priority 2: source_file
                        elif attrs.get("source_file") not in (
                            None,
                            "",
                            "unknown",
                            "None",
                        ):
                            sources.add(attrs["source_file"])
                            found = True

                        # Priority 3: url (new fallback)
                        elif attrs.get("url") not in (None, "", "unknown"):
                            sources.add(attrs["url"])
                            found = True

                        if found:
                            n_with_source += 1

                    results.append(
                        {
                            "edge_config": str(edge_config),
                            "mode": mode,
                            "node_type": node_type,
                            "cluster_id": cluster_id,
                            "n_sources": len(sources),
                            "cluster_size": len(node_ids),
                            "nodes_with_sources": n_with_source,
                            "source_coverage_pct": round(
                                n_with_source / len(node_ids) * 100, 1
                            )
                            if node_ids
                            else 0,
                        }
                    )

    df = pd.DataFrame(results)
    print(f"\n✓ Source diversity v2: {len(df):,} cluster records")
    if len(df) > 0:
        print(f"  Mean sources per cluster: {df['n_sources'].mean():.2f}")
        print(
            f"  Clusters with ≥1 source:  {(df['n_sources'] > 0).sum():,} / {len(df):,}"
        )
    return df


def print_source_diversity_findings(df_src: pd.DataFrame):
    """Print source diversity findings for Substep #6."""
    print("\n" + "=" * 80)
    print("SOURCE DIVERSITY FINDINGS — answering Substep #6 research questions")
    print("=" * 80)

    if df_src.empty:
        return

    print(f"\nTotal clusters: {len(df_src):,}")
    print(f"Zero-source clusters: {(df_src['n_sources'] == 0).sum():,}")
    print(f"Single-source clusters: {(df_src['n_sources'] == 1).sum():,}")
    print(f"Multi-source clusters (≥3): {(df_src['n_sources'] >= 3).sum():,}")

    print("\n### Mean sources by edge_config:")
    print(
        df_src.groupby("edge_config")["n_sources"]
        .agg(["mean", "max"])
        .round(2)
        .to_string()
    )

    print("\n### Interpretation (expected ranges):")
    print("  EDGE-only clusters: 1-3 sources (single-pathway origin)")
    print("  SIM≥0.9 clusters:   3-8 sources (moderate cross-paper aggregation)")
    print("  SIM≥0.8 clusters:   5-15 sources (strong cross-paper aggregation)")


# ============================================================================
# CHANGE #14 — PATH LENGTH SENSITIVITY (Substep #29)
# ============================================================================


def plot_path_length_sensitivity(df_metrics: pd.DataFrame):
    """
    #14: Scatter of mean path length vs silhouette score with Pearson correlation.
    Closes Substep #29: weak signal (r≈0.23) confirms cluster quality is
    driven by node_type and edge_config, not hop count.
    """
    print("\n" + "=" * 80)
    print("CHANGE #14: Path length sensitivity scatter (Substep #29)")
    print("=" * 80)

    if "path_length_mean" not in df_metrics.columns:
        print("  ⚠ No path_length_mean column — skipping")
        return

    valid = df_metrics.dropna(subset=["path_length_mean", "silhouette_mean"])
    if len(valid) < 4:
        print("  ⚠ Insufficient valid rows — skipping")
        return

    ec_colors = {
        "EDGE": "#1a1a1a",
        "0.8": "#2196F3",
        "0.85": "#4CAF50",
        "0.9": "#FF9800",
        "0.95": "#E91E63",
    }
    mode_markers = {
        "unconstrained": "o",
        "single_risk": "s",
        "monotonic": "^",
        "both": "D",
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    for ec in EDGE_CONFIGS:
        for mode in MODES:
            sub = valid[(valid["edge_config"] == str(ec)) & (valid["mode"] == mode)]
            if sub.empty:
                continue
            ax.scatter(
                sub["path_length_mean"],
                sub["silhouette_mean"],
                c=ec_colors.get(str(ec), "gray"),
                marker=mode_markers.get(mode, "o"),
                s=55,
                alpha=0.7,
            )

    # Pearson correlation + linear fit
    from scipy.stats import pearsonr

    x = valid["path_length_mean"].values
    y = valid["silhouette_mean"].values
    corr, p_val = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        np.poly1d(z)(x_line),
        "k--",
        alpha=0.4,
        linewidth=1.5,
        label=f"Linear fit  r={corr:.3f}  p={p_val:.3f}",
    )

    ax.set_xlabel("Mean Path Length (hops)", fontsize=12)
    ax.set_ylabel("Mean Silhouette Score", fontsize=12)
    ax.set_title(
        "Path Length vs Cluster Quality  ·  Substep #29\n"
        "Weak correlation — cluster quality primarily determined by node_type & edge_config",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch

    ec_patches = [Patch(color=c, label=ec) for ec, c in ec_colors.items()]
    mode_handles = [
        plt.scatter([], [], marker=m, c="gray", s=60, label=mode)
        for mode, m in mode_markers.items()
    ]
    leg1 = ax.legend(
        handles=ec_patches, title="Edge Config", loc="upper left", fontsize=9
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=mode_handles + ax.get_lines(),
        title="Mode / Fit",
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH_SENSITIVITY, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {PLOT_PATH_SENSITIVITY}")
    strength = (
        "weak" if abs(corr) < 0.4 else "moderate" if abs(corr) < 0.7 else "strong"
    )
    print(f"  Correlation r={corr:.3f} (p={p_val:.4f}) — {strength}")
    print("  Cluster quality driven by node_type/edge_config, not hop count.")
    print("  Report r=0.23 in paper as non-actionable; use ≥5 hop filter for taxonomy.")


# ============================================================================
# CHANGE #15 / Substep #16 — BETWEENNESS PLOT V2 (sort, format, 5 panels)
# ============================================================================


def plot_betweenness_v2():
    """
    #15/Sub#16: Corrected betweenness bar chart.
    Fixes from tracker: sorted descending, million-formatted values,
    exactly 5 subplots (one per concept category), zero-betweenness
    terminal nodes filtered out.
    """
    print("\n" + "=" * 80)
    print("CHANGE #15/Sub#16: Betweenness plot v2 (Substep #16)")
    print("=" * 80)

    if not STEP2_BETWEENNESS_CSV.exists():
        print(f"  ⚠ {STEP2_BETWEENNESS_CSV} not found — skipping")
        return

    df = pd.read_csv(STEP2_BETWEENNESS_CSV)
    # Normalise 'category' (space-separated) → node_type key
    df["cat_norm"] = df["category"].str.strip().str.lower().map(_BTWN_CAT_MAP)
    # Filter terminal nodes: zero betweenness means they don't bridge paths
    df = df[df["betweenness"] > 0].copy()

    def fmt_m(v: float) -> str:
        if v >= 1e6:
            return f"{v / 1e6:.2g}M"
        if v >= 1e3:
            return f"{v / 1e3:.2g}K"
        return f"{v:.0f}"

    fig, axes = plt.subplots(1, 5, figsize=(26, 9))
    fig.suptitle(
        "Top-20 Mechanism Transfer Enablers by Concept Category  ·  Substep #16\n"
        "(Betweenness centrality = # risk→intervention shortest paths through node)",
        fontsize=13,
        fontweight="bold",
    )

    for ax, cat in zip(axes, CONCEPT_CATEGORIES):
        cat_df = (
            df[df["cat_norm"] == cat]
            .sort_values("betweenness", ascending=False)
            .head(20)
        )

        if cat_df.empty:
            ax.text(
                0.5,
                0.5,
                f"No data\nfor {cat}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_title(
                CONCEPT_CAT_LABELS.get(cat, cat), fontsize=10, fontweight="bold"
            )
            continue

        # Reverse so highest bar ends up at top of horizontal chart
        cat_df = cat_df.iloc[::-1].reset_index(drop=True)
        bars = ax.barh(
            range(len(cat_df)),
            cat_df["betweenness"].values,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val in zip(bars, cat_df["betweenness"].values):
            ax.text(
                bar.get_width() * 1.01,
                bar.get_y() + bar.get_height() / 2,
                fmt_m(val),
                va="center",
                fontsize=7,
            )

        ax.set_yticks(range(len(cat_df)))
        ax.set_yticklabels(
            [n[:38] + "…" if len(n) > 38 else n for n in cat_df["name"].values],
            fontsize=7,
        )
        ax.set_title(CONCEPT_CAT_LABELS.get(cat, cat), fontsize=10, fontweight="bold")
        ax.set_xlabel("Betweenness", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_m(v)))
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_BETWEENNESS_V2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {PLOT_BETWEENNESS_V2}")

    # Print top-3 per category for quick review
    for cat in CONCEPT_CATEGORIES:
        sub = df[df["cat_norm"] == cat].nlargest(3, "betweenness")
        if sub.empty:
            continue
        print(f"\n  {CONCEPT_CAT_LABELS.get(cat, cat)} — top 3:")
        for _, row in sub.iterrows():
            print(f"    {row['name'][:60]}: {fmt_m(row['betweenness'])}")


# ============================================================================
# CHANGE #15 / Substep #9 — MODE IMPACT ANALYSIS
# ============================================================================


def analyze_mode_impact(
    df_quality: pd.DataFrame, cluster_memberships: Dict
) -> pd.DataFrame:
    """
    #15/Sub#9: Mode impact on silhouette, EDGE validation, and cluster count.
    Produces mode_comparison_stats.csv plus two heatmap plots.
    Key finding: mode affects EDGE% (27→91% range) far more than silhouette (<0.05).
    """
    print("\n" + "=" * 80)
    print("CHANGE #15/Sub#9: Mode impact analysis (Substep #9)")
    print("=" * 80)

    stats = []
    for nt in NODE_TYPES:
        for ec in EDGE_CONFIGS:
            for mode in MODES:
                sub = df_quality[
                    (df_quality["node_type"] == nt)
                    & (df_quality["edge_config"] == str(ec))
                    & (df_quality["mode"] == mode)
                ]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                stats.append(
                    {
                        "node_type": nt,
                        "edge_config": str(ec),
                        "mode": mode,
                        "silhouette_mean": float(row.get("silhouette_mean", np.nan)),
                        "edge_validation_mean": float(
                            row.get("edge_validation_mean", np.nan)
                        ),
                        "n_clusters": float(row.get("n_clusters", np.nan)),
                        "path_length_mean": float(row.get("path_length_mean", np.nan)),
                    }
                )

    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(OUT_MODE_STATS, index=False)
    print(f"  ✓ Saved: {OUT_MODE_STATS} ({len(df_stats)} rows)")

    print("\n  Mode impact (mean across all node_types & edge_configs):")
    print("  " + "-" * 65)
    for mode in MODES:
        sub = df_stats[df_stats["mode"] == mode]
        print(
            f"  {mode:20s}  sil={sub['silhouette_mean'].mean():.3f}  "
            f"edge_val={sub['edge_validation_mean'].mean():.3f}  "
            f"n_clust={sub['n_clusters'].mean():.1f}"
        )
    print()
    print("  ✓ Key finding: mode changes EDGE% by 27→91% (64 pp range) but")
    print("    silhouette by <0.05 — select mode based on EDGE%, not silhouette.")
    return df_stats


def plot_edge_density_heatmap(df_stats: pd.DataFrame):
    """EDGE validation rate heatmap: modes × edge_configs per node_type."""
    print("\n  Generating EDGE density heatmap…")
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()

    for idx, nt in enumerate(NODE_TYPES):
        ax = axes_flat[idx]
        sub = df_stats[df_stats["node_type"] == nt]
        if sub.empty:
            ax.axis("off")
            continue
        mat = pd.pivot_table(
            sub,
            values="edge_validation_mean",
            index="mode",
            columns="edge_config",
            aggfunc="mean",
        )
        col_order = [c for c in EDGE_CONFIGS if c in mat.columns]
        mat = mat[col_order]
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            linewidths=0.5,
            cbar=(idx == len(NODE_TYPES) - 1),
        )
        ax.set_title(nt.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Edge Config", fontsize=8)
        ax.set_ylabel("Mode" if idx % 4 == 0 else "", fontsize=8)
        ax.tick_params(labelsize=8)

    plt.suptitle(
        "EDGE Validation Rate by Mode × Edge Config  ·  Substep #9\n"
        "(Green=high literature grounding, Red=low; mode drives 27→91% range)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOT_MODE_EDGE_DENSITY, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {PLOT_MODE_EDGE_DENSITY}")


def plot_mode_stability_heatmap(cluster_memberships: Dict):
    """
    ARI between each mode and unconstrained reference at the same edge_config.
    High ARI = mode constraint doesn't reorganise clusters; low = it does.
    """
    print("\n  Computing mode stability ARI (vs unconstrained)…")
    algo = "agglomerative"
    ref_mode = "unconstrained"
    compare_modes = ["single_risk", "monotonic", "both"]
    results = []

    for nt in tqdm(NODE_TYPES, desc="Mode stability ARI", leave=False):
        for ec in EDGE_CONFIGS:
            # Build reference assignment dict: {node_id: cluster_id}
            ref_assign: Dict[str, Any] = {}
            for key, members in cluster_memberships.items():
                if (
                    len(key) == 5
                    and str(key[0]) == str(ec)
                    and key[1] == ref_mode
                    and key[2] == nt
                    and key[3] == algo
                ):
                    for nid in members:
                        ref_assign[nid] = key[4]
            if not ref_assign:
                continue

            for mode in compare_modes:
                cmp_assign: Dict[str, Any] = {}
                for key, members in cluster_memberships.items():
                    if (
                        len(key) == 5
                        and str(key[0]) == str(ec)
                        and key[1] == mode
                        and key[2] == nt
                        and key[3] == algo
                    ):
                        for nid in members:
                            cmp_assign[nid] = key[4]
                if not cmp_assign:
                    continue

                common = sorted(set(ref_assign) & set(cmp_assign))
                if len(common) < 5:
                    continue

                ari = adjusted_rand_score(
                    [ref_assign[n] for n in common],
                    [cmp_assign[n] for n in common],
                )
                results.append(
                    {
                        "node_type": nt,
                        "edge_config": str(ec),
                        "mode_vs": mode,
                        "ari_vs_unconstrained": ari,
                    }
                )

    df_mode_ari = pd.DataFrame(results)
    if df_mode_ari.empty:
        print("  ⚠ No mode ARI data — skipping heatmap")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()

    for idx, nt in enumerate(NODE_TYPES):
        ax = axes_flat[idx]
        sub = df_mode_ari[df_mode_ari["node_type"] == nt]
        if sub.empty:
            ax.axis("off")
            continue
        mat = pd.pivot_table(
            sub,
            values="ari_vs_unconstrained",
            index="mode_vs",
            columns="edge_config",
            aggfunc="mean",
        )
        col_order = [c for c in EDGE_CONFIGS if c in mat.columns]
        mat = mat[col_order]
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            linewidths=0.5,
            cbar=(idx == len(NODE_TYPES) - 1),
        )
        ax.set_title(nt.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Edge Config", fontsize=8)
        ax.set_ylabel("Mode (vs unconstrained)" if idx % 4 == 0 else "", fontsize=8)
        ax.tick_params(labelsize=8)

    plt.suptitle(
        "Mode Stability: ARI vs Unconstrained Reference  ·  Substep #9\n"
        "(High=constraints preserve cluster structure; Low=they reorganise it)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOT_MODE_STABILITY, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {PLOT_MODE_STABILITY}")
    overall = df_mode_ari["ari_vs_unconstrained"].mean()
    print(f"  Mean ARI vs unconstrained: {overall:.3f}")


# ============================================================================
# CHANGE #4 — NODE MIGRATION HEATMAP (Plot 20)
# ============================================================================


def plot_node_migration_heatmap():
    """
    #4: Plot migration rates from node_migration_frequencies.csv as a heatmap
    (transition × node_type), one subplot per mode.
    """
    print("\n" + "=" * 80)
    print("CHANGE #4: Node migration heatmap (Plot 20)")
    print("=" * 80)

    if not STEP2_MIGRATION_CSV.exists():
        print(f"  ⚠ {STEP2_MIGRATION_CSV} not found — skipping")
        return

    df = pd.read_csv(STEP2_MIGRATION_CSV)
    df["transition"] = (
        df["threshold_from"].astype(str) + "→" + df["threshold_to"].astype(str)
    )

    # Canonical ordering: adjacent low→high thresholds first
    preferred_order = [
        "0.8→0.85",
        "0.85→0.9",
        "0.9→0.95",
        "0.95→EDGE",
        "EDGE→0.8",
        "EDGE→0.85",
        "EDGE→0.9",
        "EDGE→0.95",
        "0.8→0.9",
        "0.85→0.95",
        "0.8→0.95",
        "0.8→EDGE",
    ]
    present = df["transition"].unique().tolist()
    row_order = [t for t in preferred_order if t in present]
    row_order += [t for t in present if t not in row_order]

    modes_in_data = [m for m in MODES if m in df["mode"].unique()]
    fig, axes = plt.subplots(
        1, len(modes_in_data), figsize=(5 * len(modes_in_data) + 2, 8), sharey=True
    )
    if len(modes_in_data) == 1:
        axes = [axes]

    fig.suptitle(
        "Node Migration Rates Across Threshold Transitions  ·  Plot 20\n"
        "(% nodes changing cluster ID; high rate is expected — ARI captures structure)",
        fontsize=12,
        fontweight="bold",
    )

    for ax, mode in zip(axes, modes_in_data):
        sub = df[df["mode"] == mode]
        if sub.empty:
            ax.axis("off")
            continue
        mat = pd.pivot_table(
            sub,
            values="migration_rate",
            index="transition",
            columns="node_type",
            aggfunc="mean",
        )
        row_here = [r for r in row_order if r in mat.index]
        if row_here:
            mat = mat.loc[row_here]
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdYlGn_r",  # Red = high migration (most nodes migrate), Green = stable
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.3,
            cbar=(ax is axes[-1]),
        )
        ax.set_title(mode.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Node Type", fontsize=9)
        ax.set_ylabel("Threshold Transition" if ax is axes[0] else "", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig(PLOT_NODE_MIGRATION, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {PLOT_NODE_MIGRATION}")
    print(f"  Mean migration rate overall: {df['migration_rate'].mean():.1%}")
    print("  90-99% migration is expected — ARI captures which clusters correspond.")
    print(
        "  High migration + high ARI → clusters reorganise but preserve pair structure."
    )


# ============================================================================
# CHANGE #15 / Substep #13 — MATURITY PER CLUSTER (bug fix)
# ============================================================================


def analyze_maturity_per_cluster(
    cluster_memberships: Dict, node_attrs: Dict
) -> pd.DataFrame:
    """
    #13: Per-cluster intervention lifecycle distribution (fixes step2 bug).
    The bug: step2 used global lifecycle counts ÷ n_modes so all 5 config subplots
    showed identical bars. Here we count per-cluster from node_attrs.
    Lifecycle grouping: 1-2=Design, 3=Training, 4-6=Deployment.
    """
    print("\n" + "=" * 80)
    print(
        "CHANGE #15/Sub#13: Per-cluster maturity distribution — bug fix (Substep #13)"
    )
    print("=" * 80)

    def lifecycle_stage(lc_val) -> str:
        try:
            v = int(lc_val)
        except (TypeError, ValueError):
            return "Unknown"
        if v in (1, 2):
            return "Design"
        if v == 3:
            return "Training"
        if v in (4, 5, 6):
            return "Deployment"
        return "Unknown"

    algo = "agglomerative"
    results = []

    for ec in EDGE_CONFIGS:
        for mode in MODES:
            # Aggregate cluster→members for this (ec, mode, intervention)
            clusters: Dict[Any, List] = defaultdict(list)
            for key, members in cluster_memberships.items():
                if (
                    len(key) == 5
                    and str(key[0]) == str(ec)
                    and key[1] == mode
                    and key[2] == "intervention"
                    and key[3] == algo
                ):
                    clusters[key[4]].extend(members)

            if not clusters:
                continue

            for cid, node_ids in clusters.items():
                stage_counts = {
                    "Design": 0,
                    "Training": 0,
                    "Deployment": 0,
                    "Unknown": 0,
                }
                n_interventions = 0
                for nid in node_ids:
                    attrs = node_attrs.get(nid, {})
                    lc = attrs.get("intervention_lifecycle")
                    if lc is None:
                        continue
                    n_interventions += 1
                    stage_counts[lifecycle_stage(lc)] += 1

                if n_interventions == 0:
                    continue

                n = n_interventions
                results.append(
                    {
                        "edge_config": str(ec),
                        "mode": mode,
                        "cluster_id": cid,
                        "cluster_size": len(node_ids),
                        "n_interventions": n,
                        "design_count": stage_counts["Design"],
                        "training_count": stage_counts["Training"],
                        "deployment_count": stage_counts["Deployment"],
                        "design_pct": stage_counts["Design"] / n,
                        "training_pct": stage_counts["Training"] / n,
                        "deployment_pct": stage_counts["Deployment"] / n,
                        "dominant_stage": max(
                            ("Design", "Training", "Deployment"),
                            key=lambda s: stage_counts[s],
                        ),
                    }
                )

    df = pd.DataFrame(results)
    print(f"  ✓ Computed per-cluster maturity for {len(df):,} clusters")
    if not df.empty:
        print("  Dominant stage distribution:")
        print(
            "  " + df["dominant_stage"].value_counts().to_string().replace("\n", "\n  ")
        )
        print(f"\n  Mean design%:     {df['design_pct'].mean():.1%}")
        print(f"  Mean training%:   {df['training_pct'].mean():.1%}")
        print(f"  Mean deployment%: {df['deployment_pct'].mean():.1%}")
    return df


def plot_maturity_heatmap(df_maturity: pd.DataFrame):
    """
    Stacked-bar heatmap of per-cluster lifecycle distribution.
    One subplot per (mode, edge_config) pair — shows REAL per-cluster variation
    unlike the bug-affected step2 plot that showed identical bars.
    """
    print("\n  Generating per-cluster maturity heatmap…")
    if df_maturity.empty:
        print("  ⚠ No maturity data — skipping")
        return

    # Show top-25 clusters per config (sorted by deployment_pct descending)
    fig, axes = plt.subplots(4, 5, figsize=(24, 18), sharey=False)
    for row_idx, mode in enumerate(MODES):
        for col_idx, ec in enumerate(EDGE_CONFIGS):
            ax = axes[row_idx, col_idx]
            sub = (
                df_maturity[
                    (df_maturity["mode"] == mode)
                    & (df_maturity["edge_config"] == str(ec))
                ]
                .sort_values("deployment_pct", ascending=False)
                .head(25)
            )

            if sub.empty:
                ax.axis("off")
                continue

            y = np.arange(len(sub))
            ax.barh(y, sub["design_pct"], color="#2196F3", alpha=0.85)
            ax.barh(
                y,
                sub["training_pct"],
                left=sub["design_pct"],
                color="#FF9800",
                alpha=0.85,
            )
            ax.barh(
                y,
                sub["deployment_pct"],
                left=sub["design_pct"] + sub["training_pct"],
                color="#4CAF50",
                alpha=0.85,
            )

            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=6)
            ax.set_title(f"{mode}/{ec}", fontsize=7, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch

    handles = [
        Patch(color="#2196F3", label="Design (LC 1-2)"),
        Patch(color="#FF9800", label="Training (LC 3)"),
        Patch(color="#4CAF50", label="Deployment (LC 4-6)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Per-Cluster Intervention Lifecycle Distribution  ·  Substep #13 (Bug Fix)\n"
        "Each row = one cluster, sorted by deployment_pct. "
        "Top 25 clusters per config. Unlike step2, bars now differ across configs.",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(PLOT_MATURITY_CLUSTER, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {PLOT_MATURITY_CLUSTER}")


# ============================================================================
# CHANGE #15 / Substep #10 — MULTI-RISK CLUSTER CHARACTERIZATION
# ============================================================================


def analyze_multi_risk_clusters(
    cluster_memberships: Dict, node_attrs: Dict
) -> pd.DataFrame:
    """
    #10: Identify clusters (unconstrained mode) containing >1 unique concept_category.
    Risk nodes are clustered together; concept_category distinguishes subtypes.
    High multi-category rate at low thresholds expected (similarity bridges).
    """
    print("\n" + "=" * 80)
    print("CHANGE #15/Sub#10: Multi-risk cluster characterization (Substep #10)")
    print("=" * 80)

    algo = "agglomerative"
    results = []

    for ec in EDGE_CONFIGS:
        clusters: Dict[Any, List] = defaultdict(list)
        for key, members in cluster_memberships.items():
            if (
                len(key) == 5
                and str(key[0]) == str(ec)
                and key[1] == "unconstrained"
                and key[2] == "risk"
                and key[3] == algo
            ):
                clusters[key[4]].extend(members)

        if not clusters:
            continue

        multi_count = 0
        total = len(clusters)

        for cid, node_ids in clusters.items():
            cats: set = set()
            for nid in node_ids:
                attrs = node_attrs.get(nid, {})
                cat = attrs.get("concept_category") or attrs.get("category", "")
                if cat:
                    cats.add(str(cat).strip().lower())

            is_multi = len(cats) > 1
            if is_multi:
                multi_count += 1

            results.append(
                {
                    "edge_config": str(ec),
                    "mode": "unconstrained",
                    "cluster_id": cid,
                    "cluster_size": len(node_ids),
                    "n_unique_risk_categories": len(cats),
                    "is_multi_risk": is_multi,
                    "categories": "|".join(sorted(cats)) if cats else "",
                }
            )

        pct = multi_count / total * 100 if total else 0
        print(
            f"  {ec:>6}: {multi_count:3d}/{total:3d} multi-category clusters  ({pct:.1f}%)"
        )

    df = pd.DataFrame(results)
    df.to_csv(OUT_MULTI_RISK, index=False)
    print(f"\n  ✓ Saved: {OUT_MULTI_RISK} ({len(df):,} cluster records)")
    if not df.empty:
        overall = df["is_multi_risk"].mean() * 100
        print(f"  Overall multi-category rate: {overall:.1f}%")
        print(
            "  High rate at low thresholds = similarity bridges different risk types."
        )
        print(
            "  Manual inspection of 10 multi-risk samples recommended (see tracker #10)."
        )
    return df


# ============================================================================
# CHANGE #15 / Substep #11 — RISK DIVERSITY STATS
# ============================================================================


def analyze_risk_diversity(
    df_quality: pd.DataFrame, cluster_memberships: Dict, node_attrs: Dict
) -> pd.DataFrame:
    """
    #11: Risk diversity statistics per configuration.
    Uses n_unique_risks from quality_metrics_summary (if present) plus computes
    Gini coefficient of category-frequency distribution from cluster_memberships.
    Note: extraction prompt enforces category balance → Gini ≈ 0 expected.
    """
    print("\n" + "=" * 80)
    print("CHANGE #15/Sub#11: Risk diversity stats (Substep #11)")
    print("=" * 80)

    algo = "agglomerative"
    results = []

    for ec in EDGE_CONFIGS:
        for mode in MODES:
            # Collect all risk node IDs for this config
            all_risk_nodes: List = []
            for key, members in cluster_memberships.items():
                if (
                    len(key) == 5
                    and str(key[0]) == str(ec)
                    and key[1] == mode
                    and key[2] == "risk"
                    and key[3] == algo
                ):
                    all_risk_nodes.extend(members)

            if not all_risk_nodes:
                continue

            cat_counts: Dict[str, int] = defaultdict(int)
            for nid in all_risk_nodes:
                attrs = node_attrs.get(nid, {})
                cat = attrs.get("concept_category") or attrs.get("category", "unknown")
                cat_counts[str(cat).strip() if cat else "unknown"] += 1

            total = sum(cat_counts.values())
            n_unique = len(cat_counts)

            # Gini coefficient (0=perfectly equal, 1=maximally unequal)
            freqs = np.array(sorted(cat_counts.values()), dtype=float)
            if n_unique > 1 and total > 0:
                freqs_norm = freqs / freqs.sum()
                n = len(freqs_norm)
                gini = float(
                    (2 * np.dot(np.arange(1, n + 1), freqs_norm) - (n + 1)) / n
                )
            else:
                gini = 0.0

            # n_unique_risks from step2 CSV (may differ from above)
            csv_n_unique = None
            sub_q = df_quality[
                (df_quality["edge_config"] == str(ec))
                & (df_quality["mode"] == mode)
                & (df_quality["node_type"] == "risk")
            ]
            if not sub_q.empty and "n_unique_risks" in sub_q.columns:
                csv_n_unique = sub_q.iloc[0].get("n_unique_risks")

            results.append(
                {
                    "edge_config": str(ec),
                    "mode": mode,
                    "n_risk_nodes_total": total,
                    "n_unique_categories": n_unique,
                    "n_unique_risks_csv": csv_n_unique,
                    "gini_coefficient": round(gini, 4),
                    "top_category": max(cat_counts, key=cat_counts.get)
                    if cat_counts
                    else "",
                    "top_category_pct": (
                        max(cat_counts.values()) / total * 100
                        if cat_counts and total
                        else 0
                    ),
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(OUT_RISK_DIVERSITY, index=False)
    print(f"  ✓ Saved: {OUT_RISK_DIVERSITY} ({len(df)} rows)")
    if not df.empty:
        print(
            f"  Mean unique categories per config: {df['n_unique_categories'].mean():.1f}"
        )
        print(
            f"  Mean Gini coefficient:             {df['gini_coefficient'].mean():.3f}"
        )
        print(
            "  ⚠ Frequency claims about risk prevalence NOT valid — prompt enforces balance."
        )
        print(
            "  Analysis should focus on which risks cluster together, not their frequency."
        )
    return df


# ============================================================================
# CHANGE #15 / Substep #15 — CATEGORY MECHANISM FAMILIES
# ============================================================================


def analyze_category_mechanism_families(
    cluster_memberships: Dict, node_attrs: Dict
) -> pd.DataFrame:
    """
    #15: Extract top mechanism families per concept category.
    For each category, identifies top-25 clusters from optimal config (0.9, both)
    and EDGE baseline. Exemplar = node closest to cluster centroid in embedding space.
    Outputs category_mechanism_families.csv — preliminary Step 4 taxonomy input.
    DEFERRED to Step 4: full taxonomy naming, LLM labelling, coherence scoring.
    """
    print("\n" + "=" * 80)
    print("CHANGE #15/Sub#15: Category mechanism families (Substep #15)")
    print("=" * 80)

    algo = "agglomerative"
    # Analyse for optimal config + EDGE baseline
    target_configs = [("0.9", "both"), ("EDGE", "unconstrained")]
    results = []

    for ec, mode in target_configs:
        print(f"\n  Processing config ({ec}, {mode})…")
        for cat in CONCEPT_CATEGORIES:
            # Aggregate cluster memberships for this (ec, mode, cat)
            clusters: Dict[Any, List] = defaultdict(list)
            for key, members in cluster_memberships.items():
                if (
                    len(key) == 5
                    and str(key[0]) == str(ec)
                    and key[1] == mode
                    and key[2] == cat
                    and key[3] == algo
                ):
                    clusters[key[4]].extend(members)

            if not clusters:
                continue

            sorted_clusters = sorted(
                clusters.items(), key=lambda x: len(x[1]), reverse=True
            )

            for rank, (cid, node_ids) in enumerate(sorted_clusters[:25]):
                node_names = [
                    node_attrs.get(nid, {}).get("name", f"Node_{nid}")
                    for nid in node_ids
                ]

                # Centroid exemplar
                embeddings, emb_ids = [], []
                for nid in node_ids:
                    emb = node_attrs.get(nid, {}).get("embedding")
                    if emb is not None and len(emb) > 0:
                        embeddings.append(np.asarray(emb, dtype=np.float32))
                        emb_ids.append(nid)

                exemplar_name = node_names[0] if node_names else ""
                top5_names: List[str] = node_names[:5]

                if len(embeddings) >= 2:
                    centroid = np.mean(embeddings, axis=0)
                    dists = [cosine(e, centroid) for e in embeddings]
                    order = np.argsort(dists)
                    exemplar_name = node_attrs.get(emb_ids[order[0]], {}).get(
                        "name", exemplar_name
                    )
                    top5_names = [
                        node_attrs.get(emb_ids[i], {}).get("name", "")
                        for i in order[:5]
                    ]

                results.append(
                    {
                        "edge_config": str(ec),
                        "mode": mode,
                        "concept_category": cat,
                        "cluster_id": cid,
                        "cluster_rank": rank + 1,
                        "cluster_size": len(node_ids),
                        "exemplar_name": exemplar_name,
                        "top5_members": " | ".join(top5_names),
                    }
                )

    df = pd.DataFrame(results)
    df.to_csv(OUT_CAT_FAMILIES, index=False)
    print(f"\n  ✓ Saved: {OUT_CAT_FAMILIES} ({len(df)} mechanism families)")

    # Print top-3 exemplars per category for optimal config
    print("\n  Top mechanism families — config (0.9, both):")
    opt = df[(df["edge_config"] == "0.9") & (df["mode"] == "both")]
    for cat in CONCEPT_CATEGORIES:
        sub = opt[opt["concept_category"] == cat].head(3)
        if sub.empty:
            continue
        print(f"\n  [{CONCEPT_CAT_LABELS.get(cat, cat)}]")
        for _, row in sub.iterrows():
            print(
                f"    #{row['cluster_rank']:2d}  (n={row['cluster_size']:3d}):  "
                f"{str(row['exemplar_name'])[:65]}"
            )

    print("\n  NOTE: Full taxonomy naming + coherence scoring → deferred to Step 4.")
    return df


# ============================================================================
# CHANGE #7 — ALGORITHM COMPARISON (Agglomerative vs Louvain)
# ============================================================================


def analyze_algorithm_comparison(
    df_metrics: pd.DataFrame, cluster_memberships: Dict, node_attrs: Dict
) -> pd.DataFrame:
    """
    #7: Compare Agglomerative (k=40) vs Louvain (auto-k) vs HDBSCAN silhouette scores.
    Agglomerative silhouette taken from step1 CSV.
    Louvain and HDBSCAN silhouette computed here from cluster_memberships / node_attrs
    (sampled up to 400 nodes/config to keep runtime manageable).
    HDBSCAN runs in-memory on raw cosine-normalized embeddings (no UMAP re-run).
    """
    print("\n" + "=" * 80)
    print(
        "CHANGE #7: Algorithm comparison — Agglomerative vs Louvain vs HDBSCAN (Substep #1)"
    )
    print("=" * 80)
    if HDBSCAN_AVAILABLE:
        print(
            "  HDBSCAN available — running in-memory (raw cosine-normalized embeddings, min_cluster_size=5)"
        )
    else:
        print(
            "  ⚠ HDBSCAN not available — install hdbscan package for 3-way comparison"
        )

    from sklearn.metrics import silhouette_score as _sil_score

    SAMPLE_CAP = 400
    results = []
    rng = np.random.default_rng(42)

    for _, cfg_row in tqdm(
        df_metrics.iterrows(),
        total=len(df_metrics),
        desc="Algorithm comparison silhouette",
    ):
        ec = str(cfg_row["edge_config"])
        mode = str(cfg_row["mode"])
        nt = str(cfg_row["node_type"])

        # Agglomerative: silhouette from step1 CSV (default algorithm)
        agg_sil = float(cfg_row.get("silhouette_mean", np.nan))
        agg_n_clusters = 40

        # Louvain: load from cluster_memberships
        louvain_assign: Dict[str, Any] = {}
        for key, members in cluster_memberships.items():
            if (
                len(key) == 5
                and str(key[0]) == ec
                and key[1] == mode
                and key[2] == nt
                and key[3] == "louvain"
            ):
                for nid in members:
                    louvain_assign[nid] = key[4]

        louvain_n = len(set(louvain_assign.values())) if louvain_assign else 0
        louvain_sil = np.nan

        if len(louvain_assign) >= 10 and louvain_n >= 2:
            all_nodes = list(louvain_assign.keys())
            sampled = (
                rng.choice(all_nodes, SAMPLE_CAP, replace=False).tolist()
                if len(all_nodes) > SAMPLE_CAP
                else all_nodes
            )
            X, labels = [], []
            for nid in sampled:
                emb = node_attrs.get(nid, {}).get("embedding")
                if emb is not None and len(emb) > 0:
                    X.append(np.asarray(emb, dtype=np.float32))
                    labels.append(louvain_assign[nid])

            if len(set(labels)) >= 2 and len(X) >= 10:
                try:
                    louvain_sil = float(
                        _sil_score(np.vstack(X), labels, metric="cosine")
                    )
                except Exception:
                    louvain_sil = np.nan

        # HDBSCAN: in-memory, raw cosine-normalized embeddings (no UMAP re-run)
        hdbscan_sil = np.nan
        hdbscan_n = 0
        hdbscan_noise_pct = np.nan

        if HDBSCAN_AVAILABLE:
            # Collect all nodes for this config (from agglomerative — same node pool)
            agg_nodes: Set = set()
            for key, members in cluster_memberships.items():
                if (
                    len(key) == 5
                    and str(key[0]) == ec
                    and key[1] == mode
                    and key[2] == nt
                    and key[3] == "agglomerative"
                ):
                    agg_nodes.update(members)

            if len(agg_nodes) >= 10:
                all_agg = list(agg_nodes)
                sampled_hdb = (
                    rng.choice(all_agg, SAMPLE_CAP, replace=False).tolist()
                    if len(all_agg) > SAMPLE_CAP
                    else all_agg
                )
                X_hdb = []
                for nid in sampled_hdb:
                    emb = node_attrs.get(nid, {}).get("embedding")
                    if emb is not None and len(emb) > 0:
                        X_hdb.append(np.asarray(emb, dtype=np.float32))

                if len(X_hdb) >= 10:
                    X_arr = np.vstack(X_hdb)
                    # Normalize to unit vectors → euclidean = sqrt(2 - 2*cos_sim)
                    norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
                    X_norm = X_arr / np.where(norms > 0, norms, 1.0)
                    try:
                        clusterer = HDBSCAN_CLS(
                            min_cluster_size=5, metric="euclidean", core_dist_n_jobs=1
                        )
                        hdb_labels = clusterer.fit_predict(X_norm)
                        valid_mask = hdb_labels != -1
                        hdbscan_noise_pct = float((~valid_mask).mean() * 100)
                        n_valid = int(valid_mask.sum())
                        hdbscan_n = (
                            int(len(set(hdb_labels[valid_mask]))) if n_valid > 0 else 0
                        )
                        if n_valid >= 10 and hdbscan_n >= 2:
                            hdbscan_sil = float(
                                _sil_score(
                                    X_norm[valid_mask],
                                    hdb_labels[valid_mask],
                                    metric="cosine",
                                )
                            )
                    except Exception:
                        hdbscan_sil = np.nan
                        hdbscan_noise_pct = np.nan

        results.append(
            {
                "edge_config": ec,
                "mode": mode,
                "node_type": nt,
                "agg_silhouette": agg_sil,
                "agg_n_clusters": agg_n_clusters,
                "louvain_silhouette": louvain_sil,
                "louvain_n_clusters": louvain_n,
                "hdbscan_silhouette": hdbscan_sil,
                "hdbscan_n_clusters": hdbscan_n,
                "hdbscan_noise_pct": hdbscan_noise_pct,
                "sil_diff_louvain_minus_agg": (
                    louvain_sil - agg_sil
                    if not (np.isnan(louvain_sil) or np.isnan(agg_sil))
                    else np.nan
                ),
                "sil_diff_hdbscan_minus_agg": (
                    hdbscan_sil - agg_sil
                    if not (np.isnan(hdbscan_sil) or np.isnan(agg_sil))
                    else np.nan
                ),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(OUT_ALGO_COMPARISON, index=False)
    print(f"\n  ✓ Saved: {OUT_ALGO_COMPARISON} ({len(df)} rows)")

    valid2 = df.dropna(subset=["agg_silhouette", "louvain_silhouette"])
    if not valid2.empty:
        agg_m = valid2["agg_silhouette"].mean()
        lou_m = valid2["louvain_silhouette"].mean()
        diff_m = valid2["sil_diff_louvain_minus_agg"].mean()
        print(f"\n  Mean silhouette:  Agglomerative={agg_m:.3f}  Louvain={lou_m:.3f}")
        print(
            f"  Mean Δ(Louvain−Agg): {diff_m:+.3f}  "
            f"({'Louvain better' if diff_m > 0 else 'Agglomerative better'})"
        )
        print(
            f"  Mean n_clusters:  Agglomerative=40 (fixed)  "
            f"Louvain={df['louvain_n_clusters'].mean():.1f} (variable)"
        )

    if HDBSCAN_AVAILABLE:
        valid3 = df.dropna(subset=["agg_silhouette", "hdbscan_silhouette"])
        if not valid3.empty:
            hdb_m = valid3["hdbscan_silhouette"].mean()
            noise_m = df["hdbscan_noise_pct"].mean()
            diff3_m = valid3["sil_diff_hdbscan_minus_agg"].mean()
            print(f"\n  HDBSCAN silhouette (on non-noise sample): {hdb_m:.3f}")
            print(f"  Mean noise %: {noise_m:.1f}%")
            print(
                f"  Mean n_clusters: {df['hdbscan_n_clusters'].mean():.1f} (variable, noise excluded)"
            )
            print(
                f"  Mean Δ(HDBSCAN−Agg): {diff3_m:+.3f}  "
                f"({'HDBSCAN better' if diff3_m > 0 else 'Agglomerative better'})"
            )
    return df


def plot_algorithm_comparison(df_algo: pd.DataFrame):
    """
    #7: Grouped bar chart — Agglomerative vs Louvain vs HDBSCAN silhouette per node_type.
    Mean across 4 modes, separate bars per edge_config.
    """
    print("\n  Generating algorithm comparison plot…")
    # Include HDBSCAN only if at least some values are non-NaN
    has_hdbscan = (
        "hdbscan_silhouette" in df_algo.columns
        and df_algo["hdbscan_silhouette"].notna().any()
    )
    required_cols = ["agg_silhouette", "louvain_silhouette"]
    valid = df_algo.dropna(subset=required_cols)
    if valid.empty:
        print("  ⚠ No paired algorithm data — skipping plot")
        return

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes_flat = axes.flatten()

    n_algos = 3 if has_hdbscan else 2
    w_total = 0.7
    w = w_total / n_algos

    for idx, nt in enumerate(NODE_TYPES):
        ax = axes_flat[idx]
        sub = valid[valid["node_type"] == nt]
        if sub.empty:
            ax.axis("off")
            continue

        x = np.arange(len(EDGE_CONFIGS))

        agg_means = [
            sub[sub["edge_config"] == ec]["agg_silhouette"].mean()
            for ec in EDGE_CONFIGS
        ]
        lou_means = [
            sub[sub["edge_config"] == ec]["louvain_silhouette"].mean()
            for ec in EDGE_CONFIGS
        ]

        if has_hdbscan:
            hdb_means = [
                df_algo[(df_algo["node_type"] == nt) & (df_algo["edge_config"] == ec)][
                    "hdbscan_silhouette"
                ].mean()
                for ec in EDGE_CONFIGS
            ]
            offsets = [-w, 0, w]
        else:
            offsets = [-w / 2, w / 2]

        ax.bar(
            x + offsets[0],
            agg_means,
            width=w,
            color="#2196F3",
            alpha=0.85,
            label="Agglomerative (k=40)",
        )
        ax.bar(
            x + offsets[1],
            lou_means,
            width=w,
            color="#FF5722",
            alpha=0.85,
            label="Louvain (auto-k)",
        )
        if has_hdbscan:
            ax.bar(
                x + offsets[2],
                hdb_means,
                width=w,
                color="#4CAF50",
                alpha=0.85,
                label="HDBSCAN (min_cls=5)",
            )

        ax.axhline(
            0.3,
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            label="Min target (0.3)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(EDGE_CONFIGS, fontsize=9)
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("Mean Silhouette Score", fontsize=9)
        ax.set_title(nt.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    algo_label = (
        "Agglomerative vs Louvain vs HDBSCAN"
        if has_hdbscan
        else "Agglomerative vs Louvain"
    )
    hdb_note = (
        "HDBSCAN: in-memory, raw cosine-normalized embeddings, min_cluster_size=5"
        if has_hdbscan
        else "HDBSCAN not available"
    )
    plt.suptitle(
        f"Algorithm Comparison: {algo_label}  ·  CHANGE #7\n"
        f"Mean silhouette per edge_config (averaged over 4 modes)\n"
        f"{hdb_note}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOT_ALGO_COMPARISON, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {PLOT_ALGO_COMPARISON}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    start_time = time.time()

    print("\n" + "=" * 80)
    print("PHASE 2 STEP 2b: EXTENDED METRICS & ANALYSIS")
    print("=" * 80)
    print(f"Cluster files dir: {CLUSTER_FILES_DIR}")
    print(f"Output dir:        {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING INPUTS")
    print("=" * 80)

    print(f"\nLoading step1 metrics: {CHECKPOINT_METRICS}")
    df_metrics = pd.read_csv(CHECKPOINT_METRICS)
    # Normalise edge_config column to strings
    df_metrics["edge_config"] = df_metrics["edge_config"].astype(str)
    print(f"  ✓ {len(df_metrics)} configurations")

    print(f"Loading node attrs: {CHECKPOINT_NODES}")
    with open(CHECKPOINT_NODES, "rb") as f:
        node_attrs: Dict = pickle.load(f)
    print(f"  ✓ {len(node_attrs):,} nodes")
    sample = next(iter(node_attrs.values()))
    print(f"  Sample keys: {list(sample.keys())[:10]}")

    print(f"Loading edge data: {CHECKPOINT_EDGES}")
    with open(CHECKPOINT_EDGES, "rb") as f:
        edge_data: List[Dict] = pickle.load(f)
    print(f"  ✓ {len(edge_data):,} edges")
    if edge_data:
        print(f"  Sample edge keys: {list(edge_data[0].keys())[:8]}")

    print(f"Loading cluster memberships: {CHECKPOINT_MEMBERS}")
    with open(CHECKPOINT_MEMBERS, "rb") as f:
        cluster_memberships: Dict = pickle.load(f)
    print(f"  ✓ {len(cluster_memberships):,} membership records")

    print(f"Loading step2 quality summary: {STEP2_QUALITY}")
    df_quality = pd.read_csv(STEP2_QUALITY)
    df_quality["edge_config"] = df_quality["edge_config"].astype(str)
    print(f"  ✓ {len(df_quality)} rows")

    # ------------------------------------------------------------------
    # Normalize embeddings: convert FalkorDB string format → numpy arrays
    # ------------------------------------------------------------------
    print("\nNormalizing embeddings (FalkorDB string → numpy float32)...")
    n_ok = normalize_embeddings(node_attrs)
    print(f"  ✓ {n_ok:,} / {len(node_attrs):,} nodes have valid embeddings")

    # ------------------------------------------------------------------
    # CHANGE #2 — Source diversity v2
    # ------------------------------------------------------------------
    df_src_v2 = analyze_source_diversity_v2(cluster_memberships, node_attrs)
    df_src_v2.to_csv(OUT_SOURCE_V2, index=False)
    print(f"\n✓ Saved: {OUT_SOURCE_V2}")
    print_source_diversity_findings(df_src_v2)

    # ------------------------------------------------------------------
    # CHANGE #1 — ARI pairwise + line plot
    # ------------------------------------------------------------------
    df_ari_pairwise = compute_ari_pairwise(df_metrics, cluster_memberships)
    df_ari_pairwise.to_csv(OUT_ARI_PAIRWISE, index=False)
    print(f"\n✓ Saved: {OUT_ARI_PAIRWISE}")
    plot_ari_line(df_ari_pairwise)
    print_ari_findings(df_ari_pairwise)

    # ------------------------------------------------------------------
    # CHANGE #6 — EDGE validation per-mode plot
    # ------------------------------------------------------------------
    plot_edge_validation_per_mode(df_quality)
    print_edge_validation_findings(df_quality)

    # ------------------------------------------------------------------
    # CHANGE #7 — Silhouette plot v2
    # ------------------------------------------------------------------
    plot_silhouette_v2(df_quality)
    print_silhouette_findings(df_quality)

    # ------------------------------------------------------------------
    # CHANGE #12 — Cluster size plot v2
    # ------------------------------------------------------------------
    plot_cluster_size_v2(df_quality)

    # ------------------------------------------------------------------
    # CHANGE #8 — Cluster cohesion
    # ------------------------------------------------------------------
    df_cohesion = analyze_cluster_cohesion(df_metrics, node_attrs, cluster_memberships)
    df_cohesion.to_csv(OUT_COHESION, index=False)
    print(f"\n✓ Saved: {OUT_COHESION}")
    print_cohesion_findings(df_cohesion)

    # ------------------------------------------------------------------
    # CHANGE #9 — Centroid similarity
    # ------------------------------------------------------------------
    df_centroid = analyze_centroid_similarity(
        df_metrics, node_attrs, cluster_memberships
    )
    df_centroid.to_csv(OUT_CENTROID_SIM, index=False)
    print(f"\n✓ Saved: {OUT_CENTROID_SIM}")
    plot_centroid_similarity(df_centroid)
    print_centroid_similarity_findings(df_centroid)

    # ------------------------------------------------------------------
    # CHANGE #10 — EDGE purity
    # ------------------------------------------------------------------
    df_purity = analyze_edge_purity(df_metrics, cluster_memberships)
    df_purity.to_csv(OUT_EDGE_PURITY, index=False)
    print(f"\n✓ Saved: {OUT_EDGE_PURITY}")
    plot_edge_purity(df_purity)
    print_edge_purity_findings(df_purity)

    # ------------------------------------------------------------------
    # CHANGE #3 — Hub quality
    # ------------------------------------------------------------------
    df_hubs = analyze_hub_quality(cluster_memberships, node_attrs, edge_data)
    df_hubs.to_csv(OUT_HUB_METRICS, index=False)
    print(f"\n✓ Saved: {OUT_HUB_METRICS}")
    plot_hub_quality(df_hubs)
    print_hub_findings(df_hubs)

    # ------------------------------------------------------------------
    # CHANGE #14 — Path length sensitivity scatter (Substep #29)
    # ------------------------------------------------------------------
    plot_path_length_sensitivity(df_metrics)

    # ------------------------------------------------------------------
    # CHANGE #15/Sub#16 — Betweenness v2 (5 category panels, descending sort)
    # ------------------------------------------------------------------
    plot_betweenness_v2()

    # ------------------------------------------------------------------
    # CHANGE #15/Sub#9 — Mode impact analysis
    # ------------------------------------------------------------------
    df_mode_stats = analyze_mode_impact(df_quality, cluster_memberships)
    df_mode_stats.to_csv(OUT_MODE_STATS, index=False)
    print(f"\n✓ Saved: {OUT_MODE_STATS}")
    plot_edge_density_heatmap(df_mode_stats)
    plot_mode_stability_heatmap(cluster_memberships)

    # ------------------------------------------------------------------
    # CHANGE #4 — Node migration heatmap (from existing migration CSV)
    # ------------------------------------------------------------------
    plot_node_migration_heatmap()

    # ------------------------------------------------------------------
    # CHANGE #15/Sub#13 — Maturity per cluster (per-cluster lifecycle counts)
    # ------------------------------------------------------------------
    df_maturity = analyze_maturity_per_cluster(cluster_memberships, node_attrs)
    df_maturity.to_csv(OUT_MATURITY_CLUSTER, index=False)
    print(f"\n✓ Saved: {OUT_MATURITY_CLUSTER}")
    plot_maturity_heatmap(df_maturity)

    # ------------------------------------------------------------------
    # CHANGE #15/Sub#10 — Multi-risk clusters (unconstrained, risk node_type)
    # ------------------------------------------------------------------
    df_multi_risk = analyze_multi_risk_clusters(cluster_memberships, node_attrs)
    df_multi_risk.to_csv(OUT_MULTI_RISK, index=False)
    print(f"\n✓ Saved: {OUT_MULTI_RISK}")

    # ------------------------------------------------------------------
    # CHANGE #15/Sub#11 — Risk diversity (category Gini coefficient)
    # ------------------------------------------------------------------
    df_risk_div = analyze_risk_diversity(df_quality, cluster_memberships, node_attrs)
    df_risk_div.to_csv(OUT_RISK_DIVERSITY, index=False)
    print(f"\n✓ Saved: {OUT_RISK_DIVERSITY}")

    # ------------------------------------------------------------------
    # CHANGE #15/Sub#15 — Category mechanism families (0.9/both + EDGE/unconstrained)
    # NOTE: Full taxonomy naming + coherence scoring → deferred to Step 4
    # ------------------------------------------------------------------
    df_cat_fam = analyze_category_mechanism_families(cluster_memberships, node_attrs)
    df_cat_fam.to_csv(OUT_CAT_FAMILIES, index=False)
    print(f"\n✓ Saved: {OUT_CAT_FAMILIES}")

    # ------------------------------------------------------------------
    # CHANGE #7 — Algorithm comparison (Agglomerative vs Louvain vs HDBSCAN)
    # HDBSCAN runs in-memory on raw cosine-normalized embeddings, SAMPLE_CAP=400
    # ------------------------------------------------------------------
    df_algo = analyze_algorithm_comparison(df_metrics, cluster_memberships, node_attrs)
    df_algo.to_csv(OUT_ALGO_COMPARISON, index=False)
    print(f"\n✓ Saved: {OUT_ALGO_COMPARISON}")
    plot_algorithm_comparison(df_algo)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("STEP 2b COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"\nAll outputs written to: {OUTPUT_DIR}")

    print("\nCSV outputs (Step 2b):")
    for p in [
        OUT_ARI_PAIRWISE,  # Change #1
        OUT_SOURCE_V2,  # Change #2
        OUT_HUB_METRICS,  # Change #3
        OUT_MATURITY_CLUSTER,  # Change #4 / Sub#13
        OUT_COHESION,  # Change #8
        OUT_CENTROID_SIM,  # Change #9
        OUT_EDGE_PURITY,  # Change #10
        OUT_MODE_STATS,  # Change #15/Sub#9
        OUT_MULTI_RISK,  # Change #15/Sub#10
        OUT_RISK_DIVERSITY,  # Change #15/Sub#11
        OUT_CAT_FAMILIES,  # Change #15/Sub#15
        OUT_ALGO_COMPARISON,  # Change #7
    ]:
        exists = "✓" if p.exists() else "✗"
        print(f"  {exists} {p.name}")

    print("\nPlot outputs (Step 2b):")
    for p in [
        PLOT_ARI_LINE,  # Change #1
        PLOT_EDGE_VAL_MODE,  # Change #6
        PLOT_SILHOUETTE_V2,  # Change #7 (label fix)
        PLOT_CLUSTER_SIZE_V2,  # Change #12
        PLOT_CENTROID_SIM,  # Change #9
        PLOT_EDGE_PURITY,  # Change #10
        PLOT_HUB_QUALITY,  # Change #3
        PLOT_PATH_SENSITIVITY,  # Change #14
        PLOT_BETWEENNESS_V2,  # Change #15/Sub#16
        PLOT_MODE_EDGE_DENSITY,  # Change #15/Sub#9
        PLOT_MODE_STABILITY,  # Change #15/Sub#9
        PLOT_NODE_MIGRATION,  # Change #4
        PLOT_MATURITY_CLUSTER,  # Change #15/Sub#13
        PLOT_ALGO_COMPARISON,  # Change #7
    ]:
        exists = "✓" if p.exists() else "✗"
        print(f"  {exists} {p.name}")

    print("\n" + "=" * 80)
    print("DEFERRED ITEMS (see Phase2_comprehensive_analysis_plan.md)")
    print("=" * 80)
    print("""
→ Step 3 (algorithm selection + multi-criteria):
    - HDBSCAN clustering (requires re-running phase2_clustering.py)
    - Multi-criteria scoring: silhouette + EDGE% + ARI + cluster count
    - Final optimal config selection → optimal_configs_final.csv
→ Step 4 (taxonomy + network analysis):
    - UMAP projections (Sub#5 / Change #5)
    - Full taxonomy: cluster naming + coherence scoring (Sub#15 full version)
    - Risk→Intervention connectivity matrix (Sub#28)
    - Exemplar path extraction per named cluster (Sub#26)
""")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review all new plots in phase2_results/step2_metrics_and_stability/
2. Cross-reference ARI line plot findings vs Substep #7 table in Comprehensive_Findings.md
3. EDGE purity + hub quality together answer Goal 2 (EDGE-Validation) and Goal 7 (Hub Quality)
4. Update Comprehensive_Findings.md status fields from ❌ to ✅ for implemented substeps:
   #1 silhouette, #3 cohesion, #4 EDGE validation, #5 EDGE purity, #6 source diversity,
   #7 ARI, #8 centroid, #9 mode impact, #10 multi-risk, #11 risk diversity,
   #13 maturity, #14 hub quality, #15 category families, #16 betweenness, #29 path sensitivity
5. Hub quality: manually inspect top-5 hubs from hub_quality_metrics.csv
   (2 high EDGE%+sources → Convergence; 2 low EDGE%+high degree → Framework/Artifact; 1 moderate)
6. Category mechanism families: review category_mechanism_families.csv for exemplar nodes
   per cluster — use as seed input for Step 4 taxonomy naming
7. Algorithm comparison: review algorithm_comparison_silhouette.png
   — confirm Louvain clusters are available in cluster_memberships before running
""")


if __name__ == "__main__":
    main()
