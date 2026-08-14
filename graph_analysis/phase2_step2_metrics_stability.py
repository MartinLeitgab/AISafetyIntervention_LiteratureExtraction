#!/usr/bin/env python3
"""
Phase 2 Step 2: Core Metrics & Stability Analysis
==================================================

Comprehensive analysis of clustering quality, stability, and validation metrics
across all 160 configurations (5 edge configs Ã— 4 modes Ã— 8 node types).

Inputs:
- phase2_results/step1_load_and_parse*/all_cluster_metrics.csv
- phase2_results/step1_load_and_parse*/graph_node_attributes.pkl
- phase2_results/step1_load_and_parse*/graph_edge_data.pkl
- phase2_results/step1_load_and_parse*/cluster_memberships.pkl
- phase2_rawclusterfiles_*/clusters_*.json (for detailed cluster data)

Outputs:
- quality_metrics_summary.csv
- stability_ari_matrix.csv
- node_migration_frequencies.csv
- cluster_source_diversity.csv
- cluster_temporal_coverage.csv
- All visualization plots (1, 2, 5, 6, 8, 9, 10)

Author: Phase 2 Analysis Pipeline
Date: January 2026
"""

# CRITICAL: Set matplotlib backend BEFORE any other imports
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend - must be first!

import json
import pickle
import time
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Verify backend is set correctly
print(f"Matplotlib backend: {matplotlib.get_backend()}")
assert matplotlib.get_backend() == "agg", "Matplotlib backend not set to Agg!"

# Set style for publication-quality figures
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input directories
STEP1_DIR = Path("./phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")
CLUSTER_FILES_DIR = Path("./phase2_rawclusterfiles_umapwithoutlocalsatellites")
OUTPUT_DIR = Path("./phase2_results/step2_metrics_and_stability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input checkpoints from Step 1
CHECKPOINT_METRICS = STEP1_DIR / "all_cluster_metrics.csv"
CHECKPOINT_NODES = STEP1_DIR / "graph_node_attributes.pkl"
CHECKPOINT_EDGES = STEP1_DIR / "graph_edge_data.pkl"
CHECKPOINT_MEMBERS = STEP1_DIR / "cluster_memberships.pkl"

# Output files
OUT_QUALITY_METRICS = OUTPUT_DIR / "quality_metrics_summary.csv"
OUT_STABILITY_ARI = OUTPUT_DIR / "stability_ari_matrix.csv"
OUT_NODE_MIGRATION = OUTPUT_DIR / "node_migration_frequencies.csv"
OUT_SOURCE_DIVERSITY = OUTPUT_DIR / "cluster_source_diversity.csv"
OUT_TEMPORAL_COVERAGE = OUTPUT_DIR / "cluster_temporal_coverage.csv"
OUT_BETWEENNESS = OUTPUT_DIR / "mechanism_transfer_betweenness.csv"

# Visualization outputs
PLOT_SILHOUETTE = OUTPUT_DIR / "silhouette_by_nodetype.png"
PLOT_ARI = OUTPUT_DIR / "cross_threshold_ari.png"
PLOT_CLUSTER_SIZE = OUTPUT_DIR / "cluster_size_distributions.png"
PLOT_EDGE_VALIDATION = OUTPUT_DIR / "edge_validation_breakdown.png"
PLOT_LIFECYCLE = OUTPUT_DIR / "lifecycle_distribution.png"
PLOT_TEMPORAL = OUTPUT_DIR / "temporal_coverage.png"
PLOT_BETWEENNESS = OUTPUT_DIR / "mechanism_transfer_betweenness.png"

# Configuration space
EDGE_CONFIGS = ["EDGE", 0.80, 0.85, 0.90, 0.95]
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
CONCEPT_CATEGORIES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]

# DPI for publication-quality figures
DPI = 300


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_cluster_file(filepath: Path) -> Dict:
    """Load a single cluster JSON file"""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def get_cluster_assignments(
    cluster_data: Dict, algorithm: str = "agglomerative"
) -> Dict[int, int]:
    """Extract node_id -> cluster_id mapping from cluster data"""
    if not cluster_data or "results" not in cluster_data:
        return {}

    results = cluster_data["results"]
    if algorithm not in results:
        return {}

    return results[algorithm].get("assignments", {})


def calculate_ari(assignments1: Dict[int, int], assignments2: Dict[int, int]) -> float:
    """Calculate Adjusted Rand Index between two clustering assignments"""
    # Find common nodes
    common_nodes = set(assignments1.keys()) & set(assignments2.keys())

    if len(common_nodes) < 2:
        return np.nan

    # Get labels for common nodes
    labels1 = [assignments1[node] for node in common_nodes]
    labels2 = [assignments2[node] for node in common_nodes]

    try:
        return adjusted_rand_score(labels1, labels2)
    except Exception:
        return np.nan


# ============================================================================
# SECTION I: QUALITY METRICS ANALYSIS
# ============================================================================


def analyze_quality_metrics(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive quality metrics analysis

    Calculates:
    - Silhouette score statistics by node_type, edge_config, mode
    - Cluster size distributions
    - EDGE validation rates
    - Algorithm comparison
    """
    print("\n" + "=" * 80)
    print("SECTION I: QUALITY METRICS ANALYSIS")
    print("=" * 80)

    # Diagnostic: Check what we have in df_metrics
    print("\n### Input Data Diagnostics")
    print(f"Total rows in df_metrics: {len(df_metrics)}")
    print(
        f"Unique node_types: {df_metrics['node_type'].nunique()} - {df_metrics['node_type'].unique().tolist()}"
    )
    print(
        f"Unique edge_configs: {df_metrics['edge_config'].nunique()} - {df_metrics['edge_config'].unique().tolist()}"
    )
    print(
        f"Unique modes: {df_metrics['mode'].nunique()} - {df_metrics['mode'].unique().tolist()}"
    )
    print(
        f"Cluster files found: {df_metrics['cluster_file_found'].sum()} / {len(df_metrics)}"
    )

    quality_metrics = []

    # Group by node_type, edge_config, mode
    for node_type in NODE_TYPES:
        for edge_config in EDGE_CONFIGS:
            for mode in MODES:
                # Filter to this configuration - convert edge_config to string for comparison
                mask = (
                    (df_metrics["node_type"] == node_type)
                    & (df_metrics["edge_config"] == str(edge_config))
                    & (df_metrics["mode"] == mode)
                )
                config_data = df_metrics[mask]

                if len(config_data) == 0:
                    continue

                row = config_data.iloc[0]

                # Extract metrics
                metrics = {
                    "node_type": node_type,
                    "edge_config": str(edge_config),
                    "mode": mode,
                    "n_clusters": row["n_clusters"],
                    "n_embeddings": row["n_embeddings"],
                    "n_edge_only": row["n_edge_only"],
                    "cluster_size_mean": row["cluster_size_mean"],
                    "cluster_size_median": row["cluster_size_median"],
                    "cluster_size_std": row["cluster_size_std"],
                    "cluster_size_min": row["cluster_size_min"],
                    "cluster_size_max": row["cluster_size_max"],
                    "silhouette_mean": row["silhouette_mean"],
                    "silhouette_median": row["silhouette_median"],
                    "silhouette_min": row["silhouette_min"],
                    "silhouette_max": row["silhouette_max"],
                    "edge_validation_mean": row["edge_validation_mean"],
                    "edge_validation_min": row["edge_validation_min"],
                    "edge_validation_max": row["edge_validation_max"],
                    "n_pathways": row["n_pathways"],
                    "path_length_mean": row["path_length_mean"],
                    "path_length_median": row["path_length_median"],
                }

                quality_metrics.append(metrics)

    df_quality = pd.DataFrame(quality_metrics)

    # Save summary
    df_quality.to_csv(OUT_QUALITY_METRICS, index=False)
    print(f"\nâœ“ Saved: {OUT_QUALITY_METRICS}")

    # Print summary statistics
    print("\n### Quality Metrics Summary")
    print(f"Total configurations: {len(df_quality)}")

    if len(df_quality) > 0:
        print("\nSilhouette scores:")
        print(f"  Mean: {df_quality['silhouette_mean'].mean():.3f}")
        print(f"  Median: {df_quality['silhouette_mean'].median():.3f}")
        print(
            f"  Range: [{df_quality['silhouette_mean'].min():.3f}, {df_quality['silhouette_mean'].max():.3f}]"
        )
        print("\nEDGE validation rates:")
        print(f"  Mean: {df_quality['edge_validation_mean'].mean():.3f}")
        print(
            f"  Configs with 100% EDGE validation: {(df_quality['edge_validation_mean'] == 1.0).sum()}"
        )
        print("\nCluster counts:")
        print(f"  Mean: {df_quality['n_clusters'].mean():.1f}")
        print(
            f"  Target range (40-60): {((df_quality['n_clusters'] >= 40) & (df_quality['n_clusters'] <= 60)).sum()} configs"
        )
    else:
        print("  âš  No quality metrics data found")
        print("  This may indicate:")
        print("    - Missing cluster files")
        print("    - Data format mismatch")
        print("    - Incorrect edge_config values in df_metrics")

    return df_quality


# ============================================================================
# SECTION II: STABILITY ANALYSIS (ARI)
# ============================================================================


def compute_cross_threshold_ari(
    df_metrics: pd.DataFrame, node_type: str, mode: str
) -> pd.DataFrame:
    """
    Compute ARI matrix across similarity thresholds for a given node_type and mode

    Returns 5x5 matrix comparing EDGE, 0.8, 0.85, 0.9, 0.95
    """
    ari_matrix = np.zeros((len(EDGE_CONFIGS), len(EDGE_CONFIGS)))

    # Load cluster assignments for all edge configs
    assignments = {}
    missing_reasons = {}

    for i, edge_config in enumerate(EDGE_CONFIGS):
        # Convert to string for comparison
        edge_config_str = str(edge_config)

        mask = (
            (df_metrics["node_type"] == node_type)
            & (df_metrics["edge_config"] == edge_config_str)
            & (df_metrics["mode"] == mode)
            & df_metrics["cluster_file_found"]
        )

        if mask.sum() == 0:
            missing_reasons[edge_config] = "No matching row in df_metrics"
            continue

        filepath = df_metrics[mask].iloc[0]["cluster_filepath"]

        # Check if file exists
        if not Path(filepath).exists():
            missing_reasons[edge_config] = f"File not found: {filepath}"
            continue

        cluster_data = load_cluster_file(Path(filepath))

        if not cluster_data:
            missing_reasons[edge_config] = "Failed to load JSON"
            continue

        assign = get_cluster_assignments(cluster_data, "agglomerative")

        if len(assign) == 0:
            missing_reasons[edge_config] = "No agglomerative assignments in file"
            continue

        assignments[edge_config] = assign

    # Only print diagnostics if some assignments are missing
    if len(assignments) < len(EDGE_CONFIGS):
        # Only print once per node_type (not for every mode)
        if mode == "unconstrained":  # Print only for first mode
            print(
                f"\n  âš  {node_type}: {len(assignments)}/{len(EDGE_CONFIGS)} edge configs have assignments"
            )
            for edge_config, reason in missing_reasons.items():
                print(f"    Missing {edge_config}: {reason}")

    # Compute pairwise ARI
    for i, edge1 in enumerate(EDGE_CONFIGS):
        for j, edge2 in enumerate(EDGE_CONFIGS):
            if i == j:
                ari_matrix[i, j] = 1.0
            elif edge1 in assignments and edge2 in assignments:
                ari_matrix[i, j] = calculate_ari(assignments[edge1], assignments[edge2])
            else:
                ari_matrix[i, j] = np.nan

    # Create DataFrame
    df_ari = pd.DataFrame(
        ari_matrix,
        index=[str(e) for e in EDGE_CONFIGS],
        columns=[str(e) for e in EDGE_CONFIGS],
    )

    return df_ari


def analyze_stability(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-threshold stability analysis using Adjusted Rand Index

    Computes ARI for all node_type Ã— mode combinations across edge configs
    """
    print("\n" + "=" * 80)
    print("SECTION II: STABILITY ANALYSIS")
    print("=" * 80)

    stability_results = []

    print("\nComputing cross-threshold ARI matrices...")

    # Diagnostic: Track which configs have data
    configs_with_data = 0
    configs_without_data = 0

    for node_type in tqdm(NODE_TYPES, desc="Node types"):
        for mode in MODES:
            # Compute ARI matrix
            df_ari = compute_cross_threshold_ari(df_metrics, node_type, mode)

            # Extract summary statistics
            # Get off-diagonal elements (exclude self-comparisons)
            off_diag = []
            for i in range(len(EDGE_CONFIGS)):
                for j in range(i + 1, len(EDGE_CONFIGS)):
                    val = df_ari.iloc[i, j]
                    if not np.isnan(val):
                        off_diag.append(val)

            if len(off_diag) > 0:
                stability_results.append(
                    {
                        "node_type": node_type,
                        "mode": mode,
                        "ari_mean": np.mean(off_diag),
                        "ari_median": np.median(off_diag),
                        "ari_min": np.min(off_diag),
                        "ari_max": np.max(off_diag),
                        "ari_std": np.std(off_diag),
                        "n_comparisons": len(off_diag),
                    }
                )
                configs_with_data += 1
            else:
                configs_without_data += 1

    df_stability = pd.DataFrame(stability_results)

    if len(df_stability) == 0:
        print("\nâš  Warning: No stability results generated")
        print("  This may indicate missing cluster files or assignment data")
        # Create empty DataFrame with expected columns
        df_stability = pd.DataFrame(
            columns=[
                "node_type",
                "mode",
                "ari_mean",
                "ari_median",
                "ari_min",
                "ari_max",
                "ari_std",
                "n_comparisons",
            ]
        )

    df_stability.to_csv(OUT_STABILITY_ARI, index=False)
    print(f"\nâœ“ Saved: {OUT_STABILITY_ARI}")

    # Print summary
    if len(df_stability) > 0:
        print("\n### Stability Summary")
        print(f"Total configurations analyzed: {len(df_stability)}")
        print(f"  Configs with ARI data: {configs_with_data}")
        print(f"  Configs without ARI data: {configs_without_data}")
        print(f"Mean ARI across all comparisons: {df_stability['ari_mean'].mean():.3f}")
        print(
            f"High stability configs (ARI > 0.7): {(df_stability['ari_mean'] > 0.7).sum()}"
        )
        print(
            f"Medium stability configs (0.5 < ARI â‰¤ 0.7): {((df_stability['ari_mean'] > 0.5) & (df_stability['ari_mean'] <= 0.7)).sum()}"
        )
        print(
            f"Low stability configs (ARI â‰¤ 0.5): {(df_stability['ari_mean'] <= 0.5).sum()}"
        )

        if configs_without_data > 0:
            print(f"\nâš  Note: {configs_without_data} configs had no ARI data")
            print("  Possible reasons:")
            print("    - Missing cluster assignment data in JSON files")
            print("    - Cluster files don't contain 'agglomerative' algorithm results")
            print("    - No common nodes between threshold pairs")
    else:
        print("\n### Stability Summary")
        print("No stability data available")

    return df_stability


# ============================================================================
# SECTION III: NODE MIGRATION ANALYSIS
# ============================================================================


def analyze_node_migration(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Track how nodes migrate between clusters as similarity threshold changes
    """
    print("\n" + "=" * 80)
    print("SECTION III: NODE MIGRATION ANALYSIS")
    print("=" * 80)

    migration_results = []

    print("\nTracking node migrations across thresholds...")
    for node_type in tqdm(NODE_TYPES, desc="Node types"):
        for mode in MODES:
            # Load assignments for adjacent thresholds
            for i in range(len(EDGE_CONFIGS) - 1):
                edge1 = EDGE_CONFIGS[i]
                edge2 = EDGE_CONFIGS[i + 1]

                # Convert to strings for comparison
                edge1_str = str(edge1)
                edge2_str = str(edge2)

                # Get cluster assignments
                mask1 = (
                    (df_metrics["node_type"] == node_type)
                    & (df_metrics["edge_config"] == edge1_str)
                    & (df_metrics["mode"] == mode)
                    & df_metrics["cluster_file_found"]
                )
                mask2 = (
                    (df_metrics["node_type"] == node_type)
                    & (df_metrics["edge_config"] == edge2_str)
                    & (df_metrics["mode"] == mode)
                    & df_metrics["cluster_file_found"]
                )

                if mask1.sum() == 0 or mask2.sum() == 0:
                    continue

                filepath1 = df_metrics[mask1].iloc[0]["cluster_filepath"]
                filepath2 = df_metrics[mask2].iloc[0]["cluster_filepath"]

                # Check if files exist
                if not Path(filepath1).exists() or not Path(filepath2).exists():
                    continue

                cluster_data1 = load_cluster_file(Path(filepath1))
                cluster_data2 = load_cluster_file(Path(filepath2))

                if not cluster_data1 or not cluster_data2:
                    continue

                assign1 = get_cluster_assignments(cluster_data1, "agglomerative")
                assign2 = get_cluster_assignments(cluster_data2, "agglomerative")

                # Count migrations
                common_nodes = set(assign1.keys()) & set(assign2.keys())
                if len(common_nodes) == 0:
                    continue

                migrations = sum(
                    1 for node in common_nodes if assign1[node] != assign2[node]
                )
                migration_rate = migrations / len(common_nodes)

                migration_results.append(
                    {
                        "node_type": node_type,
                        "mode": mode,
                        "threshold_from": edge1_str,
                        "threshold_to": edge2_str,
                        "n_common_nodes": len(common_nodes),
                        "n_migrations": migrations,
                        "migration_rate": migration_rate,
                    }
                )

    df_migration = pd.DataFrame(migration_results)

    if len(df_migration) == 0:
        print("\nâš  Warning: No migration results generated")
        df_migration = pd.DataFrame(
            columns=[
                "node_type",
                "mode",
                "threshold_from",
                "threshold_to",
                "n_common_nodes",
                "n_migrations",
                "migration_rate",
            ]
        )

    df_migration.to_csv(OUT_NODE_MIGRATION, index=False)
    print(f"\nâœ“ Saved: {OUT_NODE_MIGRATION}")

    # Print summary
    if len(df_migration) > 0:
        print("\n### Migration Summary")
        print(f"Total transitions analyzed: {len(df_migration)}")
        print(f"Average migration rate: {df_migration['migration_rate'].mean():.3f}")
        print(
            f"Migration rate range: [{df_migration['migration_rate'].min():.3f}, {df_migration['migration_rate'].max():.3f}]"
        )
    else:
        print("\n### Migration Summary")
        print("No migration data available")

    return df_migration


# ============================================================================
# SECTION IV: SOURCE DIVERSITY & TEMPORAL COVERAGE
# ============================================================================


def analyze_cluster_attributes(
    df_metrics: pd.DataFrame, node_attrs: Dict, cluster_memberships: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze source diversity and temporal coverage for each cluster
    """
    print("\n" + "=" * 80)
    print("SECTION IV: SOURCE DIVERSITY & TEMPORAL COVERAGE")
    print("=" * 80)

    # Diagnostic: Check what attributes are present
    if len(node_attrs) > 0:
        sample_node = next(iter(node_attrs.values()))
        print("\n### Node Attribute Diagnostics")
        print(f"Sample node attributes: {list(sample_node.keys())}")
        print(f"Has 'source_file_list': {'source_file_list' in sample_node}")
        print(f"Has 'source_diversity': {'source_diversity' in sample_node}")
        print(f"Has 'first_published': {'first_published' in sample_node}")
        print(f"Has 'concept_category': {'concept_category' in sample_node}")
        print(f"Has 'betweenness': {'betweenness' in sample_node}")

        # Check actual values
        n_with_source_list = sum(
            1 for a in node_attrs.values() if a.get("source_file_list")
        )
        n_with_source_file = sum(1 for a in node_attrs.values() if a.get("source_file"))
        n_with_pub_year = sum(
            1 for a in node_attrs.values() if a.get("first_published") is not None
        )

        print("\nActual data availability:")
        print(f"  Nodes with non-empty source_file_list: {n_with_source_list:,}")
        print(f"  Nodes with non-empty source_file: {n_with_source_file:,}")
        print(f"  Nodes with first_published: {n_with_pub_year:,}")

        if n_with_pub_year > 0:
            years = [
                a["first_published"]
                for a in node_attrs.values()
                if a.get("first_published") is not None
            ]
            print(f"  Publication year range: {min(years)} - {max(years)}")

    source_diversity_results = []
    temporal_results = []

    print("\nAnalyzing cluster attributes...")

    # Diagnostic: Check cluster membership keys
    if len(cluster_memberships) > 0:
        sample_keys = list(cluster_memberships.keys())[:5]
        print("\n### Cluster Membership Diagnostics")
        print(f"Total membership keys: {len(cluster_memberships)}")
        print(f"Sample keys: {sample_keys}")
        print(
            "Expected key format: (edge_config_str, mode, node_type, 'agglomerative')"
        )

    # Iterate through all configurations
    configs_processed = 0
    configs_with_members = 0

    for _, row in tqdm(df_metrics.iterrows(), total=len(df_metrics), desc="Configs"):
        edge_config = row["edge_config"]
        mode = row["mode"]
        node_type = row["node_type"]

        configs_processed += 1

        # Collect all clusters for this config across all cluster_ids
        # Key format in cluster_memberships: (edge_config, mode, node_type, algo_key, cluster_id)
        config_clusters = {}
        algo_name = "agglomerative"

        # Find all keys matching this config and algorithm
        for key, members in cluster_memberships.items():
            if len(key) == 5:  # (edge_config, mode, node_type, algo_key, cluster_id)
                key_edge, key_mode, key_node_type, key_algo, cluster_id = key

                if (
                    str(key_edge) == str(edge_config)
                    and key_mode == mode
                    and key_node_type == node_type
                    and key_algo == algo_name
                ):
                    config_clusters[cluster_id] = members

        if len(config_clusters) == 0:
            continue

        configs_with_members += 1

        # Analyze each cluster
        for cluster_id, node_ids in config_clusters.items():
            # Source diversity - use source_file_list if available
            sources = set()
            n_nodes_with_sources = 0

            for node_id in node_ids:
                if node_id in node_attrs:
                    attrs = node_attrs[node_id]

                    # Try source_file_list first (multiple sources)
                    if "source_file_list" in attrs and attrs["source_file_list"]:
                        file_list = attrs["source_file_list"]
                        # Handle different list types
                        if isinstance(file_list, list) and len(file_list) > 0:
                            sources.update(f for f in file_list if f)
                            n_nodes_with_sources += 1
                    # Fallback to single source_file
                    elif "source_file" in attrs and attrs["source_file"]:
                        source_file = attrs["source_file"]
                        if (
                            source_file
                            and source_file != "unknown"
                            and source_file != "None"
                        ):
                            sources.add(source_file)
                            n_nodes_with_sources += 1

            # Record for ALL clusters (even if 0 sources) to track coverage
            source_diversity_results.append(
                {
                    "edge_config": str(edge_config),
                    "mode": mode,
                    "node_type": node_type,
                    "cluster_id": cluster_id,
                    "n_sources": len(sources),
                    "cluster_size": len(node_ids),
                    "nodes_with_sources": n_nodes_with_sources,
                }
            )

            # Temporal coverage
            years = []
            for node_id in node_ids:
                if node_id in node_attrs:
                    attrs = node_attrs[node_id]
                    if "first_published" in attrs:
                        year = attrs["first_published"]
                        if year is not None and not (
                            isinstance(year, float) and np.isnan(year)
                        ):
                            try:
                                years.append(int(year))
                            except (ValueError, TypeError):
                                pass

            # Record for ALL clusters to track coverage
            if len(years) > 0:
                temporal_results.append(
                    {
                        "edge_config": str(edge_config),
                        "mode": mode,
                        "node_type": node_type,
                        "cluster_id": cluster_id,
                        "year_min": min(years),
                        "year_max": max(years),
                        "year_range": max(years) - min(years),
                        "year_mean": np.mean(years),
                        "year_median": np.median(years),
                        "n_nodes_with_year": len(years),
                        "cluster_size": len(node_ids),
                    }
                )

    # Save results
    df_source = pd.DataFrame(source_diversity_results)
    df_temporal = pd.DataFrame(temporal_results)

    print("\n### Processing Summary")
    print(f"Configs processed: {configs_processed}")
    print(f"Configs with cluster members: {configs_with_members}")
    print(f"Source diversity records created: {len(df_source)}")
    print(f"Temporal coverage records created: {len(df_temporal)}")

    # Handle empty DataFrames
    if len(df_source) == 0:
        df_source = pd.DataFrame(
            columns=[
                "edge_config",
                "mode",
                "node_type",
                "cluster_id",
                "n_sources",
                "cluster_size",
                "nodes_with_sources",
            ]
        )

    if len(df_temporal) == 0:
        df_temporal = pd.DataFrame(
            columns=[
                "edge_config",
                "mode",
                "node_type",
                "cluster_id",
                "year_min",
                "year_max",
                "year_range",
                "year_mean",
                "year_median",
                "n_nodes_with_year",
                "cluster_size",
            ]
        )

    df_source.to_csv(OUT_SOURCE_DIVERSITY, index=False)
    df_temporal.to_csv(OUT_TEMPORAL_COVERAGE, index=False)

    print(f"\nâœ“ Saved: {OUT_SOURCE_DIVERSITY}")
    print(f"âœ“ Saved: {OUT_TEMPORAL_COVERAGE}")

    # Print summaries
    print("\n### Source Diversity Summary")
    if len(df_source) > 0:
        print(f"Total clusters analyzed: {len(df_source):,}")
        print(f"Mean sources per cluster: {df_source['n_sources'].mean():.1f}")
        print(f"Single-source clusters: {(df_source['n_sources'] == 1).sum()}")
        print(f"Multi-source clusters (â‰¥3): {(df_source['n_sources'] >= 3).sum()}")
    else:
        print("No source diversity data available")
        print("  Possible reasons:")
        print(
            "    - Node attributes missing 'source_file_list' or 'source_file' fields"
        )
        print("    - All source fields are None or 'unknown'")

    print("\n### Temporal Coverage Summary")
    if len(df_temporal) > 0:
        print(f"Total clusters with temporal data: {len(df_temporal):,}")
        print(
            f"Publication year range: {int(df_temporal['year_min'].min())} - {int(df_temporal['year_max'].max())}"
        )
        print(f"Mean cluster year range: {df_temporal['year_range'].mean():.1f} years")
    else:
        print("No temporal coverage data available")
        print("  Possible reasons:")
        print("    - Node attributes missing 'first_published' field")
        print("    - All publication years are None or NaN")

    return df_source, df_temporal


# ============================================================================
# SECTION V: MECHANISM TRANSFER BETWEENNESS
# ============================================================================


def analyze_betweenness(node_attrs: Dict, edge_data: List[Dict]) -> pd.DataFrame:
    """
    Analyze betweenness centrality for concept nodes (mechanism transfer enablers)
    """
    print("\n" + "=" * 80)
    print("SECTION V: MECHANISM TRANSFER BETWEENNESS")
    print("=" * 80)

    betweenness_results = []

    # Filter to nodes with betweenness scores
    # NOTE: Category names in actual data have SPACES not underscores
    concept_categories = [
        "problem analysis",
        "theoretical insight",
        "design rationale",
        "implementation mechanism",
        "validation evidence",
    ]

    # Also check with underscores for compatibility
    concept_categories_alt = [
        "problem_analysis",
        "theoretical_insight",
        "design_rationale",
        "implementation_mechanism",
        "validation_evidence",
    ]

    # Diagnostic: Check node type distribution
    type_counts = Counter()
    category_counts = Counter()
    betweenness_count = 0

    for attrs in node_attrs.values():
        node_type = attrs.get("type", "unknown")
        category = attrs.get("concept_category", "unknown")
        type_counts[node_type] += 1
        category_counts[category] += 1
        if attrs.get("betweenness") is not None and attrs.get("betweenness") > 0:
            betweenness_count += 1

    print("\n### Node Type Diagnostics")
    print(f"Total nodes: {len(node_attrs):,}")
    print("Node type distribution (top 5):")
    for ntype, count in type_counts.most_common(5):
        print(f"  {ntype}: {count:,}")
    print("\nConcept category distribution (top 10):")
    for cat, count in category_counts.most_common(10):
        print(f"  {cat}: {count:,}")
    print(f"\nNodes with betweenness > 0: {betweenness_count:,}")

    print("\nExtracting betweenness scores for concept nodes...")
    for node_id, attrs in tqdm(node_attrs.items(), desc="Nodes"):
        # Check both 'concept_category' and 'type' fields
        category = attrs.get("concept_category", None)
        node_type = attrs.get("type", None)

        # Include if category matches (with spaces OR underscores)
        is_concept = False
        final_category = None

        if category:
            # Check both space and underscore versions
            if category in concept_categories or category in concept_categories_alt:
                is_concept = True
                final_category = category

        if not is_concept and node_type == "concept" and category:
            # Sometimes it's stored as type='concept' with concept_category set
            if category in concept_categories or category in concept_categories_alt:
                is_concept = True
                final_category = category

        if is_concept:
            betweenness = attrs.get("betweenness", 0)
            pagerank = attrs.get("pagerank", 0)

            # Handle None values
            if betweenness is None:
                betweenness = 0
            if pagerank is None:
                pagerank = 0

            betweenness_results.append(
                {
                    "node_id": node_id,
                    "name": attrs.get("name", "unknown"),
                    "category": final_category,
                    "betweenness": float(betweenness),
                    "pagerank": float(pagerank),
                    "degree_in": attrs.get("in_degree", 0) or 0,
                    "degree_out": attrs.get("out_degree", 0) or 0,
                    "degree_total": (attrs.get("in_degree", 0) or 0)
                    + (attrs.get("out_degree", 0) or 0),
                }
            )

    df_betweenness = pd.DataFrame(betweenness_results)

    # Handle empty DataFrame
    if len(df_betweenness) == 0:
        df_betweenness = pd.DataFrame(
            columns=[
                "node_id",
                "name",
                "category",
                "betweenness",
                "pagerank",
                "degree_in",
                "degree_out",
                "degree_total",
            ]
        )
        print(f"\nâœ“ Saved: {OUT_BETWEENNESS}")
        print("\n### Betweenness Summary")
        print("No betweenness data available (no concept nodes found)")
        print("  Possible reasons:")
        print("    - Node 'concept_category' field doesn't match expected categories")
        print("    - No nodes with betweenness scores")
        print("    - Nodes might be categorized differently in the graph")
        df_betweenness.to_csv(OUT_BETWEENNESS, index=False)
        return df_betweenness

    # Sort by betweenness
    df_betweenness = df_betweenness.sort_values("betweenness", ascending=False)

    df_betweenness.to_csv(OUT_BETWEENNESS, index=False)
    print(f"\nâœ“ Saved: {OUT_BETWEENNESS}")

    # Print summary
    print("\n### Betweenness Summary")
    print(f"Total concept nodes: {len(df_betweenness):,}")

    # Count by category (check both space and underscore versions)
    all_categories = set(concept_categories + concept_categories_alt)
    for category in sorted(all_categories):
        n = (df_betweenness["category"] == category).sum()
        if n > 0:
            print(f"  {category}: {n:,}")

    # Show distribution of betweenness scores
    non_zero = (df_betweenness["betweenness"] > 0).sum()
    print(
        f"\nNodes with betweenness > 0: {non_zero:,} ({100 * non_zero / len(df_betweenness):.1f}%)"
    )

    if len(df_betweenness) > 0 and non_zero > 0:
        print("\nTop 10 transfer enablers (by betweenness):")
        for i, row in df_betweenness.head(10).iterrows():
            if row["betweenness"] > 0:
                print(
                    f"  {row['name'][:50]:50s} | {row['category']:30s} | BC={row['betweenness']:.4f}"
                )

    return df_betweenness


# ============================================================================
# VISUALIZATION: PLOT 1 - Silhouette by Node Type
# ============================================================================


def plot_silhouette_by_nodetype(df_quality: pd.DataFrame):
    """
    Plot 1: Cluster quality by node type
    8 subplots (one per node type)
    Each shows: 5 edge configs Ã— 4 modes = 20 lines
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT 1: Silhouette by Node Type")
    print("=" * 80)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    colors = sns.color_palette("husl", len(EDGE_CONFIGS))

    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]

        for i, edge_config in enumerate(EDGE_CONFIGS):
            for j, mode in enumerate(MODES):
                mask = (
                    (df_quality["node_type"] == node_type)
                    & (df_quality["edge_config"] == str(edge_config))
                    & (df_quality["mode"] == mode)
                )

                if mask.sum() > 0:
                    sil = df_quality[mask]["silhouette_mean"].values[0]
                    ax.scatter(
                        j,
                        sil,
                        color=colors[i],
                        s=100,
                        alpha=0.7,
                        marker="o" if edge_config == "EDGE" else "s",
                    )

        # Format
        ax.set_title(
            f"{node_type.replace('_', ' ').title()}", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Mode", fontsize=10)
        ax.set_ylabel("Silhouette Score", fontsize=10)
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([m.replace("_", "\n") for m in MODES], fontsize=8)
        ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Target: 0.3")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.7)

    # Legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=10,
            label=str(e),
        )
        for i, e in enumerate(EDGE_CONFIGS)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        title="Edge Config",
        fontsize=10,
        ncol=1,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(PLOT_SILHOUETTE, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"âœ“ Saved: {PLOT_SILHOUETTE}")


# ============================================================================
# VISUALIZATION: PLOT 2 - Cross-Threshold ARI
# ============================================================================


def plot_cross_threshold_ari(df_metrics: pd.DataFrame):
    """
    Plot 2: Cross-threshold stability ARI
    8 subplots (one per node type)
    Each shows: 5Ã—5 heatmap of ARI values
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT 2: Cross-Threshold ARI")
    print("=" * 80)

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    axes = axes.flatten()

    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]

        # Compute average ARI across all modes
        ari_sum = np.zeros((len(EDGE_CONFIGS), len(EDGE_CONFIGS)))
        ari_count = np.zeros((len(EDGE_CONFIGS), len(EDGE_CONFIGS)))

        for mode in MODES:
            df_ari = compute_cross_threshold_ari(df_metrics, node_type, mode)

            for i in range(len(EDGE_CONFIGS)):
                for j in range(len(EDGE_CONFIGS)):
                    val = df_ari.iloc[i, j]
                    if not np.isnan(val):
                        ari_sum[i, j] += val
                        ari_count[i, j] += 1

        # Average
        ari_avg = np.divide(
            ari_sum, ari_count, where=ari_count > 0, out=np.full_like(ari_sum, np.nan)
        )

        # Plot heatmap
        im = ax.imshow(ari_avg, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        # Add values
        for i in range(len(EDGE_CONFIGS)):
            for j in range(len(EDGE_CONFIGS)):
                val = ari_avg[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

        # Format
        ax.set_title(
            f"{node_type.replace('_', ' ').title()}", fontsize=12, fontweight="bold"
        )
        ax.set_xticks(range(len(EDGE_CONFIGS)))
        ax.set_yticks(range(len(EDGE_CONFIGS)))
        ax.set_xticklabels([str(e) for e in EDGE_CONFIGS], fontsize=9)
        ax.set_yticklabels([str(e) for e in EDGE_CONFIGS], fontsize=9)
        ax.set_xlabel("Edge Config", fontsize=10)
        ax.set_ylabel("Edge Config", fontsize=10)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("ARI", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_ARI, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"âœ“ Saved: {PLOT_ARI}")


# ============================================================================
# VISUALIZATION: PLOT 5 - Cluster Size Distributions
# ============================================================================


def plot_cluster_size_distributions(df_quality: pd.DataFrame):
    """
    Plot 5: Cluster size distributions
    8 subplots (one per node type)
    Violin plots showing distribution across all configs
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT 5: Cluster Size Distributions")
    print("=" * 80)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]

        # Get data for this node type
        data = df_quality[df_quality["node_type"] == node_type]

        if len(data) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Create violin plot by edge config
        plot_data = []
        labels = []

        for edge_config in EDGE_CONFIGS:
            values = data[data["edge_config"] == str(edge_config)]["n_clusters"].values
            if len(values) > 0:
                plot_data.append(values)
                labels.append(str(edge_config))

        if len(plot_data) > 0:
            parts = ax.violinplot(
                plot_data,
                positions=range(len(plot_data)),
                showmeans=True,
                showmedians=True,
            )

            # Color violins
            colors = sns.color_palette("husl", len(plot_data))
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

        # Target range
        ax.axhspan(40, 60, alpha=0.2, color="green", label="Target: 40-60")

        # Format
        ax.set_title(
            f"{node_type.replace('_', ' ').title()}", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Edge Config", fontsize=10)
        ax.set_ylabel("Number of Clusters", fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_CLUSTER_SIZE, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"âœ“ Saved: {PLOT_CLUSTER_SIZE}")


# ============================================================================
# VISUALIZATION: PLOT 6 - EDGE Validation Breakdown
# ============================================================================


def plot_edge_validation_breakdown(df_quality: pd.DataFrame):
    """
    Plot 6: EDGE validation breakdown
    Stacked bar chart showing validation rates by edge config
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT 6: EDGE Validation Breakdown")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Categories for validation rate
    bins = [0, 0.6, 0.8, 0.9, 1.0, 1.01]  # Last bin catches exactly 1.0
    labels = ["<60%", "60-80%", "80-90%", "90-100%", "100%"]

    # Count configs in each bin for each edge config
    data = []
    for edge_config in EDGE_CONFIGS:
        mask = df_quality["edge_config"] == str(edge_config)
        values = df_quality[mask]["edge_validation_mean"].values

        counts = []
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:  # Last bin
                count = ((values >= bins[i]) & (values <= bins[i + 1])).sum()
            else:
                count = ((values >= bins[i]) & (values < bins[i + 1])).sum()
            counts.append(count)

        data.append(counts)

    # Plot stacked bars
    x = np.arange(len(EDGE_CONFIGS))
    width = 0.6
    colors = sns.color_palette("RdYlGn", len(labels))

    bottom = np.zeros(len(EDGE_CONFIGS))
    for i, label in enumerate(labels):
        values = [data[j][i] for j in range(len(EDGE_CONFIGS))]
        ax.bar(x, values, width, bottom=bottom, label=label, color=colors[i])
        bottom += values

    # Format
    ax.set_xlabel("Edge Config", fontsize=12)
    ax.set_ylabel("Number of Configurations", fontsize=12)
    ax.set_title("EDGE Validation Rate Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in EDGE_CONFIGS], fontsize=11)
    ax.legend(title="Validation Rate", loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOT_EDGE_VALIDATION, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"âœ“ Saved: {PLOT_EDGE_VALIDATION}")


# ============================================================================
# VISUALIZATION: PLOT 8 - Lifecycle Distribution
# ============================================================================


def plot_lifecycle_distribution(df_quality: pd.DataFrame, node_attrs: Dict):
    """
    Plot 8: Intervention lifecycle distribution
    5 subplots (one per edge config)
    Stacked bar showing Design/Training/Deployment per mode
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT 8: Intervention Lifecycle Distribution")
    print("=" * 80)

    # This requires loading intervention lifecycle data from node attributes
    # For now, create placeholder based on maturity levels

    # Count interventions by lifecycle stage
    lifecycle_counts = defaultdict(lambda: defaultdict(int))

    for node_id, attrs in node_attrs.items():
        if attrs.get("type") == "intervention":
            lifecycle = attrs.get("intervention_lifecycle", 0)

            # Map lifecycle value to stage name
            if lifecycle == 1:
                stage = "Design"
            elif lifecycle == 2:
                stage = "Training"
            elif lifecycle == 3:
                stage = "Deployment"
            else:
                continue  # Skip if no lifecycle data

            lifecycle_counts[stage]["all"] += 1

    # Create plot
    fig, axes = plt.subplots(1, len(EDGE_CONFIGS), figsize=(20, 5))

    stages = ["Design", "Training", "Deployment"]
    colors = sns.color_palette("Set2", len(stages))

    for idx, edge_config in enumerate(EDGE_CONFIGS):
        ax = axes[idx]

        # Get counts for this edge config across modes
        mode_data = {stage: [] for stage in stages}

        for mode in MODES:
            for stage in stages:
                # Simplified: use overall counts
                mode_data[stage].append(lifecycle_counts[stage]["all"] / len(MODES))

        # Plot stacked bars
        x = np.arange(len(MODES))
        width = 0.6
        bottom = np.zeros(len(MODES))

        for i, stage in enumerate(stages):
            ax.bar(
                x, mode_data[stage], width, bottom=bottom, label=stage, color=colors[i]
            )
            bottom += mode_data[stage]

        # Format
        ax.set_title(f"Edge Config: {edge_config}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mode", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Number of Interventions", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in MODES], fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        if idx == len(EDGE_CONFIGS) - 1:
            ax.legend(title="Lifecycle", loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_LIFECYCLE, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"âœ“ Saved: {PLOT_LIFECYCLE}")


# ============================================================================
# VISUALIZATION: PLOT 9 - Temporal Coverage
# ============================================================================


def plot_temporal_coverage(df_temporal: pd.DataFrame):
    """
    Plot 9: Temporal coverage
    Timeline with violin plots showing cluster publication date distributions
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT 9: Temporal Coverage")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Group by node type and get year distributions
    node_type_colors = sns.color_palette("husl", len(NODE_TYPES))

    positions = []
    data_to_plot = []
    labels = []
    colors_list = []

    for idx, node_type in enumerate(NODE_TYPES):
        subset = df_temporal[df_temporal["node_type"] == node_type]

        if len(subset) > 0:
            # Collect all years for this node type
            years = []
            for _, row in subset.iterrows():
                # Use year range to show temporal spread
                year_min = row["year_min"]
                year_max = row["year_max"]
                n_samples = max(1, int(row["n_nodes_with_year"] / 10))

                if year_max > year_min:
                    import numpy as np

                    year_samples = np.linspace(year_min, year_max, n_samples)
                else:
                    year_samples = [year_min] * n_samples
                years.extend(year_samples)

            if len(years) > 0:
                positions.append(idx)
                data_to_plot.append(years)
                labels.append(node_type.replace("_", "\n"))
                colors_list.append(node_type_colors[idx])

    # Create violin plot
    if len(data_to_plot) > 0:
        parts = ax.violinplot(
            data_to_plot,
            positions=positions,
            vert=True,
            showmeans=True,
            showmedians=True,
        )

        # Color violins
        for pc, color in zip(parts["bodies"], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

    # Format
    ax.set_ylabel("Publication Year", fontsize=12)
    ax.set_xlabel("Node Type", fontsize=12)
    ax.set_title(
        "Temporal Coverage: Cluster Publication Date Distributions",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(1970, 2025)

    # Add decade markers
    for decade in range(1970, 2030, 10):
        ax.axhline(decade, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.text(-0.5, decade, f"{decade}s", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(PLOT_TEMPORAL, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"âœ“ Saved: {PLOT_TEMPORAL}")


# ============================================================================
# VISUALIZATION: PLOT 10 - Mechanism Transfer Betweenness
# ============================================================================


def plot_betweenness(df_betweenness: pd.DataFrame):
    """
    Plot 10: Mechanism transfer betweenness
    6 subplots (one per concept category)
    Bar chart of top-20 transfer enablers by betweenness centrality
    """
    print("\n" + "=" * 80)
    print("GENERATING PLOT 10: Mechanism Transfer Betweenness")
    print("=" * 80)

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    axes = axes.flatten()

    # Use space-separated names to match actual data in CSV
    concept_categories = [
        "problem analysis",
        "theoretical insight",
        "design rationale",
        "implementation mechanism",
        "validation evidence",
    ]

    for idx, category in enumerate(concept_categories):
        ax = axes[idx]

        # Get top 20 nodes for this category
        subset = df_betweenness[df_betweenness["category"] == category]
        top20 = subset.nlargest(20, "betweenness")

        if len(top20) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Plot horizontal bar chart
        y_pos = np.arange(len(top20))
        values = top20["betweenness"].values

        bars = ax.barh(y_pos, values, color=sns.color_palette("viridis", len(top20)))

        # Format
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:40] for name in top20["name"].values], fontsize=7)
        ax.set_xlabel("Betweenness Centrality", fontsize=10)
        ax.set_title(
            f"{category.replace('_', ' ').title()}", fontsize=11, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                ax.text(val, i, f"{val:.4f}", va="center", fontsize=6)

    # Hide unused subplot
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(PLOT_BETWEENNESS, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"âœ“ Saved: {PLOT_BETWEENNESS}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function"""
    start_time = time.time()

    print("=" * 80)
    print("PHASE 2 STEP 2: CORE METRICS & STABILITY ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")

    # Check for required inputs
    if not CHECKPOINT_METRICS.exists():
        print(f"\nâœ— Error: Missing checkpoint file: {CHECKPOINT_METRICS}")
        print("  Please run phase2_step1_loadandparse.py first")
        return

    # Load checkpoints
    print("\n" + "=" * 80)
    print("LOADING CHECKPOINTS FROM STEP 1")
    print("=" * 80)

    print(f"Loading: {CHECKPOINT_METRICS}")
    df_metrics = pd.read_csv(CHECKPOINT_METRICS)
    print(f"  âœ“ Loaded {len(df_metrics)} configurations")

    print(f"Loading: {CHECKPOINT_NODES}")
    with open(CHECKPOINT_NODES, "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  âœ“ Loaded {len(node_attrs):,} nodes")

    # temp After line: node_attrs = pickle.load(f)
    print("\n### Pickle Load Verification")
    sample = next(iter(node_attrs.values()))
    print(f"Sample node keys: {list(sample.keys())}")
    print(f"Has 'url': {'url' in sample}")
    if "url" in sample:
        print(f"Sample url value: {sample['url']}")
    urls_count = sum(1 for a in node_attrs.values() if a.get("url"))
    print(f"Nodes with url: {urls_count:,}")

    print(f"Loading: {CHECKPOINT_EDGES}")
    with open(CHECKPOINT_EDGES, "rb") as f:
        edge_data = pickle.load(f)
    print(f"  âœ“ Loaded {len(edge_data):,} edges")

    print(f"Loading: {CHECKPOINT_MEMBERS}")
    with open(CHECKPOINT_MEMBERS, "rb") as f:
        cluster_memberships = pickle.load(f)
    print(
        f"  âœ“ Loaded cluster memberships for {len(cluster_memberships)} configurations"
    )

    # ========================================================================
    # SECTION I: QUALITY METRICS
    # ========================================================================

    df_quality = analyze_quality_metrics(df_metrics)

    # ========================================================================
    # SECTION II: STABILITY ANALYSIS
    # ========================================================================

    df_stability = analyze_stability(df_metrics)

    # ========================================================================
    # SECTION III: NODE MIGRATION
    # ========================================================================

    analyze_node_migration(df_metrics)

    # ========================================================================
    # SECTION IV: CLUSTER ATTRIBUTES
    # ========================================================================

    df_source, df_temporal = analyze_cluster_attributes(
        df_metrics, node_attrs, cluster_memberships
    )

    # ========================================================================
    # SECTION V: BETWEENNESS
    # ========================================================================

    df_betweenness = analyze_betweenness(node_attrs, edge_data)

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Generate plots (skip if data is missing)
    if len(df_quality) > 0:
        plot_silhouette_by_nodetype(df_quality)
    else:
        print("âš  Skipping Plot 1 (Silhouette): No quality data")

    if len(df_stability) > 0:
        plot_cross_threshold_ari(df_metrics)
    else:
        print("âš  Skipping Plot 2 (ARI): No stability data")

    if len(df_quality) > 0:
        plot_cluster_size_distributions(df_quality)
        plot_edge_validation_breakdown(df_quality)
    else:
        print("âš  Skipping Plots 5-6: No quality data")

    if len(node_attrs) > 0 and len(df_quality) > 0:
        plot_lifecycle_distribution(df_quality, node_attrs)
    else:
        print("âš  Skipping Plot 8 (Lifecycle): Missing data")

    if len(df_temporal) > 0:
        plot_temporal_coverage(df_temporal)
    else:
        print("âš  Skipping Plot 9 (Temporal): No temporal data")

    if len(df_betweenness) > 0:
        plot_betweenness(df_betweenness)
    else:
        print("âš  Skipping Plot 10 (Betweenness): No betweenness data")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - {OUT_QUALITY_METRICS.name}")
    print(f"  - {OUT_STABILITY_ARI.name}")
    print(f"  - {OUT_NODE_MIGRATION.name}")
    print(f"  - {OUT_SOURCE_DIVERSITY.name}")
    print(f"  - {OUT_TEMPORAL_COVERAGE.name}")
    print(f"  - {OUT_BETWEENNESS.name}")
    print("  - 7 visualization plots")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
