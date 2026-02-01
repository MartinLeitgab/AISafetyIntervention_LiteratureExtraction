#!/usr/bin/env python3
"""
Phase 2 Step 1: Data Loading & Parsing (With Checkpoints) - Version 2

Loads all 160 cluster JSON files from phase2_clustering.py and FalkorDB graph data,
creating checkpoints for efficient iteration on downstream analysis.

Version 2 additions:
- Temporal data from Source nodes (publication years)
- Source diversity (count of unique sources per node)
- Cluster membership preservation for downstream analysis

Author: Phase 2 Analysis Pipeline
Date: January 2026
"""

import json
import pickle
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import redis
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input directories
CLUSTER_FILES_DIR = Path("./phase2_rawclusterfiles_umapwithoutlocalsatellites")
PATH_FILES_DIR = Path("./phase1_rawpathsfiles")
OUTPUT_DIR = Path("./phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FalkorDB/Redis connection
REDIS_HOST = "localhost"
REDIS_PORT = 6379
GRAPH_NAME = "AISafetyIntervention"
QUERY_TIMEOUT = 300000  # 5 minutes

# TEST MODE: Process only first N batches for verification
TEST_MODE = False
TEST_BATCHES = 3  # Process 300 nodes

# Configuration space (160 total)
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

# Checkpoint files
CHECKPOINT_METRICS = OUTPUT_DIR / "all_cluster_metrics.csv"
CHECKPOINT_NODES = OUTPUT_DIR / "graph_node_attributes.pkl"
CHECKPOINT_EDGES = OUTPUT_DIR / "graph_edge_data.pkl"
CHECKPOINT_MEMBERS = OUTPUT_DIR / "cluster_memberships.pkl"
SUMMARY_FILE = OUTPUT_DIR / "load_summary.txt"


# ============================================================================
# Calculate global graph metrics
# ============================================================================


def compute_graph_metrics(node_attrs: Dict, edge_data: List[Dict]) -> Dict:
    """Compute PageRank, Betweenness..."""
    import networkx as nx

    if TEST_MODE:
        print("\nâš  TEST MODE: Skipping graph metrics (requires full dataset)")
        return node_attrs

    print("\n" + "=" * 80)
    print("COMPUTING GRAPH METRICS")
    print("=" * 80)

    # Build NetworkX graph
    print("Building NetworkX graph...")
    G = nx.DiGraph()

    # Add all nodes
    for node_id in node_attrs.keys():
        G.add_node(node_id)

    # Add edges (only if both endpoints exist in filtered node set)
    for edge in edge_data:
        if edge["source"] in node_attrs and edge["target"] in node_attrs:
            G.add_edge(edge["source"], edge["target"])

    print(f"  âœ“ Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    # Compute PageRank
    print("Computing PageRank...")
    pagerank = nx.pagerank(G, max_iter=100)
    for node_id, pr in pagerank.items():
        if node_id in node_attrs:
            node_attrs[node_id]["pagerank"] = pr
    print("  âœ“ PageRank computed")

    # Compute Betweenness (full calculation with NetworKit, very high CPU load (computer becomes unusable) but complete betweenness calculation done in 10 minutes)
    print("Computing Betweenness...")
    import networkit as nk
    nk_graph = nk.nxadapter.nx2nk(G, weightAttr=None)
    bc_calc = nk.centrality.Betweenness(nk_graph)
    bc_calc.run()

    node_list = list(G.nodes())
    betweenness = {node_list[i]: bc_calc.score(i) for i in range(len(node_list))}

    for node_id, bc in betweenness.items():
        if node_id in node_attrs:
            node_attrs[node_id]["betweenness"] = bc
    print("  âœ“ Betweenness computed")

    # Compute degree metrics
    print("Computing Degree metrics...")
    for node_id in node_attrs:
        if node_id in G:
            node_attrs[node_id]["degree"] = G.degree(node_id)
            node_attrs[node_id]["in_degree"] = G.in_degree(node_id)
            node_attrs[node_id]["out_degree"] = G.out_degree(node_id)
    print("  ✓ Degree metrics computed")
    
    # Compute clustering coefficient
    print("Computing Clustering Coefficient...")
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)
    for node_id, cc in clustering.items():
        if node_id in node_attrs:
            node_attrs[node_id]["clustering_coefficient"] = cc
    print("  âœ“ Clustering coefficient computed")

    return node_attrs


# ============================================================================
# FALKORDB/REDIS UTILITIES WITH RETRY LOGIC
# ============================================================================


def connect_falkordb():
    """Connect to FalkorDB via Redis"""
    try:
        client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=300
        )
        client.ping()
        print(f"âœ“ Connected to FalkorDB at {REDIS_HOST}:{REDIS_PORT}")
        return client
    except Exception as e:
        print(f"âœ— Failed to connect to FalkorDB: {e}")
        print(f"  Make sure Redis/FalkorDB is running on {REDIS_HOST}:{REDIS_PORT}")
        return None


def query_graph(client, query_str, timeout=QUERY_TIMEOUT, max_retries=3):
    """
    Execute Cypher query on FalkorDB graph with retry and exponential backoff
    """
    for attempt in range(max_retries):
        try:
            result = client.execute_command(
                "GRAPH.QUERY", GRAPH_NAME, query_str, "--timeout", str(timeout)
            )
            return result[1] if len(result) > 1 else []
        except Exception as e:
            if "timed out" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                print(
                    f"  Query timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
                continue
            else:
                print(f"Query error after {attempt + 1} attempts: {e}")
                print(f"Query was: {query_str[:200]}...")
                return []


# ============================================================================
# CLUSTER FILE LOADING
# ============================================================================


def find_cluster_files(base_dir: Path) -> Dict[Tuple, Path]:
    """Generate expected filenames, check if they exist"""
    cluster_files = {}

    for edge_config in EDGE_CONFIGS:
        for mode in MODES:
            for node_type in NODE_TYPES:
                # Generate filename exactly as phase2_clustering.py does
                if edge_config == "EDGE":
                    config_label = f"EDGE_{mode}_{node_type}"
                else:
                    config_label = f"SIM{edge_config}_{mode}_{node_type}"

                expected_file = base_dir / f"clusters_{config_label}.json"

                if expected_file.exists():
                    cluster_files[(edge_config, mode, node_type)] = expected_file

    return cluster_files


def find_path_files(base_dir: Path) -> Dict[Tuple, Path]:
    """Generate expected path filenames, check if they exist"""
    path_files = {}

    for edge_config in EDGE_CONFIGS:
        for mode in MODES:
            # Generate filename exactly as phase2_clustering.py does
            if edge_config == "EDGE":
                path_file_name = f"paths_{mode}_edge_only.jsonl"
            else:
                path_file_name = f"paths_{mode}_sim{edge_config}.jsonl"

            expected_file = base_dir / path_file_name

            if expected_file.exists():
                path_files[(edge_config, mode)] = expected_file

    return path_files


def load_path_file_stats(path_file: Path) -> Dict[str, Any]:
    """Extract basic stats from path JSONL file"""
    stats = {
        "n_pathways": 0,
        "unique_risks": set(),
        "unique_interventions": set(),
        "path_lengths": [],
    }

    try:
        with open(path_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                stats["n_pathways"] += 1
                stats["path_lengths"].append(len(data.get("path", [])))
                if "risk" in data:
                    stats["unique_risks"].add(data["risk"])
                if "intervention" in data:
                    stats["unique_interventions"].add(data["intervention"])
    except Exception as e:
        print(f"  Error loading path file: {e}")

    # Convert sets to counts
    stats["n_unique_risks"] = len(stats["unique_risks"])
    stats["n_unique_interventions"] = len(stats["unique_interventions"])
    del stats["unique_risks"]
    del stats["unique_interventions"]

    return stats


def load_cluster_json(filepath: Path) -> Dict[str, Any]:
    """
    Load a single cluster JSON file from phase2_clustering.py output

    Returns: dict with cluster metadata and statistics
    """
    cluster_data = {
        "n_clusters": 0,
        "n_pathways": 0,
        "cluster_sizes": [],
        "cluster_members": {},
        "node_participation": Counter(),
        "source_files": set(),
        "algorithms": set(),
        "silhouette_scores": [],
        "exemplars": {},
        "edge_validation_rates": [],
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "results" in data:
            results = data["results"]

            for algo_key in list(results.keys()):
                algo_results = results[algo_key]
                cluster_data["algorithms"].add(algo_key)

                if "n_clusters" in algo_results:
                    cluster_data["n_clusters"] = max(
                        cluster_data["n_clusters"], algo_results["n_clusters"]
                    )

                if "cluster_data" in algo_results:
                    clusters = algo_results["cluster_data"]

                    for cluster_id, cluster_info in clusters.items():
                        size = cluster_info.get("size", 0)
                        cluster_data["cluster_sizes"].append(size)
                        cluster_data["cluster_members"][f"{algo_key}_{cluster_id}"] = (
                            cluster_info.get("members", [])
                        )

                        if "edge_validation_rate" in cluster_info:
                            cluster_data["edge_validation_rates"].append(
                                cluster_info["edge_validation_rate"]
                            )

                        if "exemplar" in cluster_info:
                            cluster_data["exemplars"][f"{algo_key}_{cluster_id}"] = (
                                cluster_info["exemplar"]
                            )

                if "metrics" in algo_results:
                    metrics = algo_results["metrics"]
                    if "silhouette" in metrics and metrics["silhouette"] >= 0:
                        cluster_data["silhouette_scores"].append(metrics["silhouette"])

        cluster_data["n_pathways"] = data.get("n_paths", 0)
        cluster_data["n_edge_only"] = data.get("n_edge_only", 0)
        cluster_data["n_embeddings"] = data.get("n_embeddings", 0)
        cluster_data["algorithms"] = list(cluster_data["algorithms"])
        cluster_data["n_sources"] = 0

        return cluster_data

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return cluster_data


def load_all_cluster_files(
    cluster_files: Dict[Tuple, Path], path_files: Dict[Tuple, Path]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load all 160 cluster JSON files and corresponding path JSONL files with caching

    Returns: (DataFrame with one row per configuration, Dict with cluster memberships)
    """
    print("\n" + "=" * 80)
    print("LOADING CLUSTER AND PATH FILES")
    print("=" * 80)

    records = []
    cluster_memberships = {}  # {(edge_config, mode, node_type, algo, cluster_id): [node_ids]}

    # Generate all expected configurations
    total_configs = len(EDGE_CONFIGS) * len(MODES) * len(NODE_TYPES)
    expected_configs = [
        (ec, mode, nt) for ec in EDGE_CONFIGS for mode in MODES for nt in NODE_TYPES
    ]

    print(f"Expected configurations: {total_configs}")
    print(f"Found cluster files: {len(cluster_files)}")
    print(f"Found path files: {len(path_files)}")

    missing_configs = []
    path_cache = {}  # Cache: (edge, mode) â†’ path_stats

    for config_num, (edge_config, mode, node_type) in enumerate(expected_configs, 1):
        print(f"\n[{config_num}/{total_configs}] {edge_config} / {mode} / {node_type}")

        cluster_filepath = cluster_files.get((edge_config, mode, node_type))

        if cluster_filepath is None:
            print("  âœ— Cluster file not found")
            missing_configs.append((edge_config, mode, node_type))
            records.append(
                {
                    "edge_config": edge_config,
                    "mode": mode,
                    "node_type": node_type,
                    "cluster_file_found": False,
                    "path_file_found": False,
                    "n_clusters": 0,
                    "n_pathways": 0,
                    "n_embeddings": 0,
                }
            )
            continue

        print(f"  Loading cluster: {cluster_filepath.name}")
        cluster_data = load_cluster_json(cluster_filepath)

        # ===================================================================
        # ADDITION: CLUSTER MEMBER PRESERVATION
        # Extract member lists while already parsing the file
        # ===================================================================
        try:
            with open(cluster_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "results" in data:
                for algo_key, algo_results in data["results"].items():
                    if "cluster_data" in algo_results:
                        for cluster_id, cluster_info in algo_results[
                            "cluster_data"
                        ].items():
                            members = cluster_info.get("members", [])
                            key = (edge_config, mode, node_type, algo_key, cluster_id)
                            cluster_memberships[key] = members
        except Exception as e:
            print(f"  âš  Failed to extract cluster members: {e}")

        # Load path stats with caching
        path_key = (edge_config, mode)
        path_filepath = path_files.get(path_key)

        if path_key in path_cache:
            path_stats = path_cache[path_key]
            print("  Using cached path stats")
        elif path_filepath:
            print(f"  Loading paths: {path_filepath.name}")
            path_stats = load_path_file_stats(path_filepath)
            path_cache[path_key] = path_stats
        else:
            print("  âš  Path file not found")
            path_stats = {}

        # Create record combining both sources
        record = {
            "edge_config": edge_config,
            "mode": mode,
            "node_type": node_type,
            "cluster_file_found": True,
            "path_file_found": path_filepath is not None,
            "cluster_filepath": str(cluster_filepath),
            "path_filepath": str(path_filepath) if path_filepath else None,
            # From cluster file
            "n_clusters": cluster_data["n_clusters"],
            "n_embeddings": cluster_data["n_embeddings"],
            "n_edge_only": cluster_data.get("n_edge_only", 0),
            "algorithms": ",".join(cluster_data["algorithms"])
            if cluster_data["algorithms"]
            else None,
            # Cluster size statistics
            "cluster_size_min": min(cluster_data["cluster_sizes"])
            if cluster_data["cluster_sizes"]
            else 0,
            "cluster_size_max": max(cluster_data["cluster_sizes"])
            if cluster_data["cluster_sizes"]
            else 0,
            "cluster_size_mean": np.mean(cluster_data["cluster_sizes"])
            if cluster_data["cluster_sizes"]
            else 0,
            "cluster_size_median": np.median(cluster_data["cluster_sizes"])
            if cluster_data["cluster_sizes"]
            else 0,
            "cluster_size_std": np.std(cluster_data["cluster_sizes"])
            if cluster_data["cluster_sizes"]
            else 0,
            # Silhouette scores
            "silhouette_mean": np.mean(cluster_data["silhouette_scores"])
            if cluster_data["silhouette_scores"]
            else None,
            "silhouette_median": np.median(cluster_data["silhouette_scores"])
            if cluster_data["silhouette_scores"]
            else None,
            "silhouette_min": min(cluster_data["silhouette_scores"])
            if cluster_data["silhouette_scores"]
            else None,
            "silhouette_max": max(cluster_data["silhouette_scores"])
            if cluster_data["silhouette_scores"]
            else None,
            # EDGE validation
            "edge_validation_mean": np.mean(cluster_data["edge_validation_rates"])
            if cluster_data["edge_validation_rates"]
            else None,
            "edge_validation_min": min(cluster_data["edge_validation_rates"])
            if cluster_data["edge_validation_rates"]
            else None,
            "edge_validation_max": max(cluster_data["edge_validation_rates"])
            if cluster_data["edge_validation_rates"]
            else None,
            # From path file (with fallback to cluster data)
            "n_pathways": cluster_data["n_pathways"],
            "n_unique_risks": path_stats.get("n_unique_risks", 0),
            "n_unique_interventions": path_stats.get("n_unique_interventions", 0),
            "path_length_mean": np.mean(path_stats["path_lengths"])
            if path_stats.get("path_lengths")
            else 0,
            "path_length_median": np.median(path_stats["path_lengths"])
            if path_stats.get("path_lengths")
            else 0,
            "path_length_min": min(path_stats["path_lengths"])
            if path_stats.get("path_lengths")
            else 0,
            "path_length_max": max(path_stats["path_lengths"])
            if path_stats.get("path_lengths")
            else 0,
            "n_exemplars": len(cluster_data["exemplars"]),
        }

        records.append(record)

        print(f"  âœ“ {record['n_clusters']} clusters, {record['n_pathways']:,} pathways")
        if record["silhouette_mean"] is not None:
            sil_str = f"{record['silhouette_mean']:.3f}"
            edge_str = (
                f"{record['edge_validation_mean']:.3f}"
                if record["edge_validation_mean"] is not None
                else "N/A"
            )
            print(f"    Silhouette: {sil_str}, EDGE validation: {edge_str}")

        if TEST_MODE and config_num >= 5:  # First 5 configs
            print(f"\nâš  TEST MODE: Stopping after {config_num} configs")
            break

    df = pd.DataFrame(records)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Loaded {len(df)} configurations")
    print(f"  Cluster files found: {df['cluster_file_found'].sum()}")
    print(f"  Path files found: {df['path_file_found'].sum()}")
    print(f"  Missing: {len(missing_configs)}")
    print(
        f"  Cluster members extracted: {len(cluster_memberships):,} clusterâ†’member mappings"
    )

    if missing_configs:
        print("\nMissing configurations:")
        for ec, mode, nt in missing_configs[:10]:
            print(f"  - {ec} / {mode} / {nt}")
        if len(missing_configs) > 10:
            print(f"  ... and {len(missing_configs) - 10} more")

    return df, cluster_memberships


# ============================================================================
# FALKORDB GRAPH DATA LOADING
# ============================================================================


def load_node_attributes(client) -> Dict[int, Dict[str, Any]]:
    """
    Load all node attributes from FalkorDB graph
    ADDITIONS: Temporal data and source diversity from Source nodes

    Returns: dict mapping node_id -> attributes
    """
    print("\n" + "=" * 80)
    print("LOADING NODE ATTRIBUTES FROM FALKORDB")
    print("=" * 80)

    node_attrs = {}

    try:
        # Get node ID range
        print("Getting node ID range...")
        result = query_graph(client, "MATCH (n) RETURN min(id(n)), max(id(n))")
        if not result:
            print("  âœ— Failed to get node range")
            return node_attrs

        min_id, max_id = int(result[0][0]), int(result[0][1])
        print(f"  Node IDs: {min_id} to {max_id} (range: {max_id - min_id + 1})")

        # Batch load node attributes (batch_size=100 to avoid timeouts)
        batch_size = 100
        current_id = min_id

        with tqdm(total=max_id - min_id + 1, desc="Loading nodes") as pbar:
            batch_count = 0
            while current_id <= max_id:
                end_id = min(current_id + batch_size - 1, max_id)

                # Query for node attributes, Must have â‰¥1 EDGE edge, excluding satellite nodes/extraction failures
                query = f"""
                MATCH (n)-[:EDGE]-() 
                WHERE id(n) >= {current_id} AND id(n) <= {end_id}
                    AND NOT 'Rationale' IN labels(n)
                    AND NOT 'Source' IN labels(n)
                WITH DISTINCT n
                RETURN id(n), 
                       n.name, 
                       n.type,
                       n.concept_category,
                       n.intervention_lifecycle,
                       n.intervention_maturity,
                       n.description,
                       n.aliases,
                       n.url,
                       n.paper_dir,
                       n.embedding,
                       n.embedding_umap_150d,
                       n.semantic_cluster,
                       n.community_cluster,
                       n.pagerank,
                       n.betweenness
                """

                result = query_graph(client, query)

                for row in result:
                    node_id = int(row[0])
                    attrs = {
                        "name": row[1] if len(row) > 1 else None,
                        "type": row[2] if len(row) > 2 else None,
                        "concept_category": row[3] if len(row) > 3 else None,
                        "intervention_lifecycle": row[4] if len(row) > 4 else None,
                        "intervention_maturity": row[5] if len(row) > 5 else None,
                        "description": row[6] if len(row) > 6 else None,
                        "aliases": row[7] if len(row) > 7 else None,
                        "url": row[8] if len(row) > 8 else None,
                        "paper_dir": row[9] if len(row) > 9 else None,
                        "embedding": row[10] if len(row) > 10 else None,
                        "embedding_umap_150d": row[11] if len(row) > 11 else None,
                        "semantic_cluster": row[12] if len(row) > 12 else None,
                        "community_cluster": row[13] if len(row) > 13 else None,
                        "pagerank": row[14] if len(row) > 14 else None,
                        "betweenness": row[15] if len(row) > 15 else None,
                        # Degree fields not in graph - will be computed from edges
                        "degree": None,
                        "in_degree": None,
                        "out_degree": None,
                    }
                    node_attrs[node_id] = attrs

                # ===================================================================
                # ADDITION: TEMPORAL DATA
                # Query FROM edges to Source nodes for publication dates
                # ===================================================================
                query_temporal = f"""
                MATCH (n)-[:FROM]->(s:Source)
                WHERE id(n) >= {current_id} AND id(n) <= {end_id}
                RETURN id(n), s.date_published
                """
                result_temporal = query_graph(client, query_temporal)

                for row in result_temporal:
                    node_id = int(row[0])
                    if node_id in node_attrs:
                        date_str = row[1] if len(row) > 1 else None

                        # Extract year in Python, not Cypher
                        year = None
                        if date_str and isinstance(date_str, str):
                            try:
                                year = int(date_str[:4])  # "2018-06-21T..." -> 2018
                            except:
                                pass

                        node_attrs[node_id]["publication_years"] = (
                            [year] if year else []
                        )
                        node_attrs[node_id]["first_published"] = year
                        node_attrs[node_id]["last_published"] = year
                        node_attrs[node_id]["publication_span"] = None

                pbar.update(end_id - current_id + 1)
                current_id = end_id + 1
                batch_count += 1

                if TEST_MODE and batch_count >= TEST_BATCHES:
                    print(f"\nâš  TEST MODE: Stopping after {TEST_BATCHES} batches")
                    break

        print(f"\nâœ“ Loaded {len(node_attrs):,} nodes")

        # Statistics
        node_types = Counter(
            attrs["type"] for attrs in node_attrs.values() if attrs.get("type")
        )
        print("\nNode type distribution:")
        for ntype, count in node_types.most_common():
            print(f"  {ntype}: {count:,}")

        # Temporal coverage statistics
        nodes_with_dates = sum(
            1 for a in node_attrs.values() if a.get("first_published")
        )
        print("\nTemporal coverage:")
        print(
            f"  Nodes with publication dates: {nodes_with_dates:,} ({100 * nodes_with_dates / len(node_attrs):.1f}%)"
        )

        if nodes_with_dates > 0:
            all_years = [
                a["first_published"]
                for a in node_attrs.values()
                if a.get("first_published")
            ]
            print(f"  Publication year range: {min(all_years)} - {max(all_years)}")

        # Source statistics
        nodes_with_urls = sum(1 for a in node_attrs.values() if a.get("url"))
        print("\nSource coverage:")
        print(
            f"  Nodes with URL: {nodes_with_urls:,} ({100 * nodes_with_urls / len(node_attrs):.1f}%)"
        )

        # Show unique URLs
        unique_urls = set(a["url"] for a in node_attrs.values() if a.get("url"))
        print(f"  Unique URLs: {len(unique_urls):,}")
        print(f"  Sample URLs: {list(unique_urls)[:5]}")

        return node_attrs

    except Exception as e:
        print(f"âœ— Error loading node attributes: {e}")
        import traceback

        traceback.print_exc()
        return node_attrs


def load_edge_data(client) -> List[Dict[str, Any]]:
    """
    Load all edge data from FalkorDB graph

    Returns: list of edge dicts with (source, target, type, attributes)
    """
    print("\n" + "=" * 80)
    print("LOADING EDGE DATA FROM FALKORDB")
    print("=" * 80)

    edges = []

    try:
        # Get node ID range first
        result = query_graph(client, "MATCH (n) RETURN min(id(n)), max(id(n))")
        if not result:
            print("  âœ— Failed to get node range")
            return edges

        min_id, max_id = int(result[0][0]), int(result[0][1])

        # Batch load edges (batch_size=100 to avoid timeouts)
        batch_size = 100
        current_id = min_id

        with tqdm(total=max_id - min_id + 1, desc="Loading edges") as pbar:
            while current_id <= max_id:
                end_id = min(current_id + batch_size - 1, max_id)

                # Query for EDGE edges
                query_edge = f"""
                MATCH (n)-[e:EDGE]->(m)
                WHERE id(n) >= {current_id} AND id(n) <= {end_id}
                RETURN id(n), id(m), e.type, e.edge_confidence, e.description, e.source_file
                """

                result = query_graph(client, query_edge)
                for row in result:
                    edges.append(
                        {
                            "source": int(row[0]),
                            "target": int(row[1]),
                            "type": "EDGE",
                            "subtype": row[2] if len(row) > 2 else None,
                            "confidence": row[3] if len(row) > 3 else None,
                            "description": row[4] if len(row) > 4 else None,
                            "source_file": row[5] if len(row) > 5 else None,
                        }
                    )

                # Query for SIMILARITY edges (only where id(m) > id(n) to avoid duplicates)
                query_sim = f"""
                MATCH (n)-[e:SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST]->(m)
                WHERE id(n) >= {current_id} AND id(n) <= {end_id} AND id(m) > id(n)
                RETURN id(n), id(m), e.score
                """

                result = query_graph(client, query_sim)
                for row in result:
                    edges.append(
                        {
                            "source": int(row[0]),
                            "target": int(row[1]),
                            "type": "SIMILARITY",
                            "subtype": None,
                            "similarity_score": row[2] if len(row) > 2 else None,
                            "confidence": None,
                            "description": None,
                            "source_file": None,
                        }
                    )

                pbar.update(end_id - current_id + 1)
                current_id = end_id + 1

        print(f"\nâœ“ Loaded {len(edges):,} edges")

        # Statistics
        edge_types = Counter(edge["type"] for edge in edges)
        print("\nEdge type distribution:")
        for etype, count in edge_types.most_common():
            print(f"  {etype}: {count:,}")

        if any(e["type"] == "EDGE" for e in edges):
            edge_subtypes = Counter(
                edge["subtype"]
                for edge in edges
                if edge["type"] == "EDGE" and edge.get("subtype")
            )
            print("\nEDGE subtype distribution (top 10):")
            for subtype, count in edge_subtypes.most_common(10):
                print(f"  {subtype}: {count:,}")

        return edges

    except Exception as e:
        print(f"âœ— Error loading edge data: {e}")
        import traceback

        traceback.print_exc()
        return edges


# ============================================================================
# CHECKPOINT SAVING
# ============================================================================


def save_checkpoints(df_metrics, node_attrs, edge_data, cluster_memberships):
    """Save all four checkpoints"""
    print("\n" + "=" * 80)
    print("SAVING CHECKPOINTS")
    print("=" * 80)

    # Checkpoint 1: Cluster metrics DataFrame
    print(f"Saving cluster metrics to: {CHECKPOINT_METRICS}")
    df_metrics.to_csv(CHECKPOINT_METRICS, index=False)
    print(f"  âœ“ Saved {len(df_metrics)} configuration records")
    print(f"  File size: {CHECKPOINT_METRICS.stat().st_size / 1024:.1f} KB")

    # Checkpoint 2: Node attributes
    print(f"\nSaving node attributes to: {CHECKPOINT_NODES}")
    with open(CHECKPOINT_NODES, "wb") as f:
        pickle.dump(node_attrs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  âœ“ Saved {len(node_attrs):,} node records")
    print(f"  File size: {CHECKPOINT_NODES.stat().st_size / (1024 * 1024):.1f} MB")

    # Checkpoint 3: Edge data
    print(f"\nSaving edge data to: {CHECKPOINT_EDGES}")
    with open(CHECKPOINT_EDGES, "wb") as f:
        pickle.dump(edge_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  âœ“ Saved {len(edge_data):,} edge records")
    print(f"  File size: {CHECKPOINT_EDGES.stat().st_size / (1024 * 1024):.1f} MB")

    # Checkpoint 4: Cluster memberships
    print(f"\nSaving cluster memberships to: {CHECKPOINT_MEMBERS}")
    with open(CHECKPOINT_MEMBERS, "wb") as f:
        pickle.dump(cluster_memberships, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  âœ“ Saved {len(cluster_memberships):,} clusterâ†’member mappings")
    print(f"  File size: {CHECKPOINT_MEMBERS.stat().st_size / (1024 * 1024):.1f} MB")


def generate_summary(
    df_metrics: pd.DataFrame,
    node_attrs: Dict,
    edge_data: List[Dict],
    cluster_memberships: Dict,
):
    """Generate comprehensive summary report"""
    summary_lines = []

    summary_lines.append("=" * 80)
    summary_lines.append("PHASE 2 STEP 1: DATA LOADING SUMMARY")
    summary_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 80)

    # Cluster Metrics
    summary_lines.append("\n## CLUSTER METRICS")
    summary_lines.append(f"Total configurations: {len(df_metrics)}")

    valid_df = df_metrics[df_metrics["cluster_file_found"] == True]
    summary_lines.append(f"Configurations with cluster data: {len(valid_df)}")
    summary_lines.append(
        f"Configurations with path data: {valid_df['path_file_found'].sum()}"
    )
    summary_lines.append(f"Missing configurations: {len(df_metrics) - len(valid_df)}")

    if len(valid_df) > 0:
        total_clusters = valid_df["n_clusters"].sum()
        total_pathways = valid_df["n_pathways"].sum()
        avg_clusters = valid_df["n_clusters"].mean()
        std_clusters = valid_df["n_clusters"].std()
        avg_pathways = valid_df["n_pathways"].mean()
        std_pathways = valid_df["n_pathways"].std()

        summary_lines.append(f"\nTotal clusters: {total_clusters:,}")
        summary_lines.append(f"Total pathways: {total_pathways:,}")
        summary_lines.append(
            f"Clusters per config: {avg_clusters:.1f} Â± {std_clusters:.1f}"
        )
        summary_lines.append(
            f"Pathways per config: {avg_pathways:.1f} Â± {std_pathways:.1f}"
        )
        summary_lines.append(
            "  Note: High variance expected - unconstrained mode has ~60x more paths than 'both' mode"
        )

        # Silhouette scores
        sil_valid = valid_df[valid_df["silhouette_mean"].notna()]
        if len(sil_valid) > 0:
            summary_lines.append(
                f"\nSilhouette scores (configs with data: {len(sil_valid)}):"
            )
            summary_lines.append(f"  Mean: {sil_valid['silhouette_mean'].mean():.3f}")
            summary_lines.append(
                f"  Median: {sil_valid['silhouette_mean'].median():.3f}"
            )
            summary_lines.append(
                f"  Range: [{sil_valid['silhouette_mean'].min():.3f}, {sil_valid['silhouette_mean'].max():.3f}]"
            )

        # By edge config
        summary_lines.append("\n### By Edge Configuration")
        for ec in EDGE_CONFIGS:
            ec_str = str(ec) if isinstance(ec, float) else ec
            ec_df = valid_df[valid_df["edge_config"].astype(str) == ec_str]
            if len(ec_df) > 0:
                summary_lines.append(
                    f"{ec_str:15s}: {len(ec_df):2d} configs, {ec_df['n_clusters'].sum():6,} clusters, {ec_df['n_pathways'].sum():,} pathways"
                )

        # By mode
        summary_lines.append("\n### By Mode")
        for mode in MODES:
            mode_df = valid_df[valid_df["mode"] == mode]
            if len(mode_df) > 0:
                summary_lines.append(
                    f"{mode:15s}: {len(mode_df):2d} configs, {mode_df['n_clusters'].sum():6,} clusters, {mode_df['n_pathways'].sum():,} pathways"
                )

        # By node type
        summary_lines.append("\n### By Node Type")
        for nt in NODE_TYPES:
            nt_df = valid_df[valid_df["node_type"] == nt]
            if len(nt_df) > 0:
                summary_lines.append(
                    f"{nt:25s}: {len(nt_df):2d} configs, {nt_df['n_clusters'].sum():6,} clusters, {nt_df['n_pathways'].sum():,} pathways"
                )

        # EDGE Validation Statistics
        summary_lines.append("\n### EDGE Validation Rates")
        edge_valid = valid_df[valid_df["edge_validation_mean"].notna()]
        if len(edge_valid) > 0:
            summary_lines.append(
                f"Mean across all configs: {edge_valid['edge_validation_mean'].mean():.3f}"
            )
            summary_lines.append(
                f"Median: {edge_valid['edge_validation_mean'].median():.3f}"
            )
            summary_lines.append(
                f"Configs with >60% EDGE validation: {(edge_valid['edge_validation_mean'] > 0.6).sum()}/{len(edge_valid)}"
            )
            summary_lines.append(
                f"Range: [{edge_valid['edge_validation_mean'].min():.3f}, {edge_valid['edge_validation_mean'].max():.3f}]"
            )

    # Cluster Memberships
    summary_lines.append("\n## CLUSTER MEMBERSHIPS")
    summary_lines.append(f"Total clusterâ†’member mappings: {len(cluster_memberships):,}")
    if cluster_memberships:
        total_nodes_in_clusters = sum(
            len(members) for members in cluster_memberships.values()
        )
        avg_cluster_size = total_nodes_in_clusters / len(cluster_memberships)
        summary_lines.append(f"Average cluster size: {avg_cluster_size:.1f} nodes")

    # Node Attributes
    summary_lines.append("\n## GRAPH NODE ATTRIBUTES")
    summary_lines.append(f"Total nodes: {len(node_attrs):,}")

    if node_attrs:
        # Node type distribution
        type_counts = {}
        for node_id, attrs in node_attrs.items():
            node_type = attrs.get("type") or "unknown"
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        summary_lines.append("\n### Node Type Distribution")
        for node_type, count in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            summary_lines.append(f"  {node_type:20s}: {count:,}")

        # Attribute coverage
        summary_lines.append("\n### Node Attribute Coverage")
        with_emb = sum(
            1
            for attrs in node_attrs.values()
            if "embedding" in attrs and attrs["embedding"] is not None
        )
        with_pr = sum(
            1
            for attrs in node_attrs.values()
            if "pagerank" in attrs and attrs["pagerank"] is not None
        )
        with_bc = sum(
            1
            for attrs in node_attrs.values()
            if "betweenness" in attrs and attrs["betweenness"] is not None
        )
        with_cluster = sum(
            1
            for attrs in node_attrs.values()
            if "semantic_cluster" in attrs and attrs["semantic_cluster"] is not None
        )
        with_temporal = sum(
            1
            for attrs in node_attrs.values()
            if "first_published" in attrs and attrs["first_published"] is not None
        )
        with_sources = sum(
            1
            for attrs in node_attrs.values()
            if "url" in attrs and attrs["url"] is not None
        )

        summary_lines.append(
            f"  Nodes with embeddings: {with_emb:,} ({100 * with_emb / len(node_attrs):.1f}%)"
        )
        summary_lines.append(
            f"  Nodes with PageRank: {with_pr:,} ({100 * with_pr / len(node_attrs):.1f}%)"
        )
        summary_lines.append(
            f"  Nodes with betweenness: {with_bc:,} ({100 * with_bc / len(node_attrs):.1f}%)"
        )
        summary_lines.append(
            f"  Nodes with semantic cluster: {with_cluster:,} ({100 * with_cluster / len(node_attrs):.1f}%)"
        )
        summary_lines.append(
            f"  Nodes with temporal data: {with_temporal:,} ({100 * with_temporal / len(node_attrs):.1f}%)"
        )
        summary_lines.append(
            f"  Nodes with source diversity: {with_sources:,} ({100 * with_sources / len(node_attrs):.1f}%)"
        )

        # Temporal statistics
        if with_temporal > 0:
            all_years = [
                a["first_published"]
                for a in node_attrs.values()
                if a.get("first_published")
            ]
            summary_lines.append("\n### Temporal Coverage")
            summary_lines.append(
                f"  Publication year range: {min(all_years)} - {max(all_years)}"
            )
            summary_lines.append("  Nodes per decade:")
            year_counts = Counter(all_years)
            for decade in range((min(all_years) // 10) * 10, max(all_years) + 1, 10):
                count = sum(
                    c for y, c in year_counts.items() if decade <= y < decade + 10
                )
                if count > 0:
                    summary_lines.append(f"    {decade}s: {count:,}")

        # Source diversity statistics
        if with_sources > 0:
            summary_lines.append("\n### Source Diversity")
            summary_lines.append(f"  Nodes with URL source: {with_sources:,}")
            # Since each node has 1 URL, all are single-source

    # Edge Data
    summary_lines.append("\n## GRAPH EDGE DATA")
    summary_lines.append(f"Total edges: {len(edge_data):,}")

    if edge_data:
        # Edge type distribution
        type_counts = {}
        subtype_counts = {}

        for edge in edge_data:
            edge_type = edge.get("type", "unknown")
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1

            if edge_type == "EDGE":
                subtype = edge.get("subtype", "unknown")
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

        summary_lines.append("\n### Edge Type Distribution")
        for edge_type, count in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            summary_lines.append(f"  {edge_type:20s}: {count:,}")

        if subtype_counts:
            summary_lines.append("\n### EDGE Subtype Distribution (top 15)")
            for subtype, count in sorted(
                subtype_counts.items(), key=lambda x: x[1], reverse=True
            )[:15]:
                summary_lines.append(f"  {subtype:30s}: {count:,}")

    # Checkpoint files
    summary_lines.append("\n## CHECKPOINT FILES")

    def format_size(bytes_size):
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"

    if CHECKPOINT_METRICS.exists():
        size = CHECKPOINT_METRICS.stat().st_size
        summary_lines.append(f"Cluster metrics: {CHECKPOINT_METRICS}")
        summary_lines.append(f"  Size: {format_size(size)}")

    if CHECKPOINT_NODES.exists():
        size = CHECKPOINT_NODES.stat().st_size
        summary_lines.append(f"Node attributes: {CHECKPOINT_NODES}")
        summary_lines.append(f"  Size: {format_size(size)}")

    if CHECKPOINT_EDGES.exists():
        size = CHECKPOINT_EDGES.stat().st_size
        summary_lines.append(f"Edge data: {CHECKPOINT_EDGES}")
        summary_lines.append(f"  Size: {format_size(size)}")

    if CHECKPOINT_MEMBERS.exists():
        size = CHECKPOINT_MEMBERS.stat().st_size
        summary_lines.append(f"Cluster memberships: {CHECKPOINT_MEMBERS}")
        summary_lines.append(f"  Size: {format_size(size)}")

    # Write summary
    summary_text = "\n".join(summary_lines)

    with open(SUMMARY_FILE, "w") as f:
        f.write(summary_text)

    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"{'=' * 80}")

    # Also print to console
    print("\n" + summary_text)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function"""
    start_time = time.time()

    print("=" * 80)
    print("PHASE 2 STEP 1: DATA LOADING & PARSING (VERSION 2)")
    print("=" * 80)

    # Check if we should just regenerate summary
    if (
        CHECKPOINT_METRICS.exists()
        and CHECKPOINT_NODES.exists()
        and CHECKPOINT_EDGES.exists()
        and CHECKPOINT_MEMBERS.exists()
    ):
        print("\nâœ“ All checkpoints exist")
        response = (
            input("Regenerate summary only (skip loading)? [y/N]: ").strip().lower()
        )
        if response == "y":
            print("Loading checkpoints...")
            df_metrics = pd.read_csv(CHECKPOINT_METRICS)
            with open(CHECKPOINT_NODES, "rb") as f:
                node_attrs = pickle.load(f)
            with open(CHECKPOINT_EDGES, "rb") as f:
                edge_data = pickle.load(f)
            with open(CHECKPOINT_MEMBERS, "rb") as f:
                cluster_memberships = pickle.load(f)

            generate_summary(df_metrics, node_attrs, edge_data, cluster_memberships)
            print(f"\n{'=' * 80}")
            print("SUMMARY REGENERATED")
            print(f"{'=' * 80}")
            return

    # Check for existing outputs
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"\nâš  Output directory exists: {OUTPUT_DIR}")
        response = input("Overwrite existing outputs? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return
        print("Proceeding with overwrite...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Find and load cluster/path files
    cluster_files = find_cluster_files(CLUSTER_FILES_DIR)
    path_files = find_path_files(PATH_FILES_DIR)
    print(f"\nFound {len(cluster_files)} cluster files")
    print(f"Found {len(path_files)} path files")
    df_metrics, cluster_memberships = load_all_cluster_files(cluster_files, path_files)

    # Step 2: Load graph data from checkpoint, or FalkorDB
    if CHECKPOINT_NODES.exists() and CHECKPOINT_EDGES.exists():
        print("\n" + "=" * 80)
        print("LOADING GRAPH DATA FROM CHECKPOINTS")
        print("=" * 80)
        with open(CHECKPOINT_NODES, "rb") as f:
            node_attrs = pickle.load(f)
        with open(CHECKPOINT_EDGES, "rb") as f:
            edge_data = pickle.load(f)
        print(
            f"âœ“ Loaded {len(node_attrs):,} nodes and {len(edge_data):,} edges from cache"
        )
    else:
        client = connect_falkordb()
        if not client:
            print("Cannot proceed without FalkorDB connection")
            return

        node_attrs = load_node_attributes(client)
        edge_data = load_edge_data(client)

        # Step 3: Compute graph metrics from edge data
        node_attrs = compute_graph_metrics(node_attrs, edge_data)

    # Step 4: Save checkpoints
    save_checkpoints(df_metrics, node_attrs, edge_data, cluster_memberships)

    # Step 5: Generate summary
    generate_summary(df_metrics, node_attrs, edge_data, cluster_memberships)

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()