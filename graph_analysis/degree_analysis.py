import redis
from collections import Counter
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend to avoid Qt errors
import matplotlib.pyplot as plt
from scipy import stats

# different edge names depending on embedding creation before rdb dump created- narrow and wide
embeddings_type = "wide"  # "narrow" or "wide"
if embeddings_type == "narrow":
    similarity_edge_name = "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
else:
    similarity_edge_name = "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"


class GraphEdgeAnalyzer:
    def __init__(self, host="localhost", port=6379, graph_name="AISafetyIntervention"):
        """
        Initialize connection to FalkorDB

        Args:
            host: Redis host
            port: Redis port
            graph_name: Name of the graph in FalkorDB
        """
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph_name = graph_name

    def get_node_type_counts(self):
        """
        Get counts of each node type in the graph

        Returns:
            dict: Node type to count mapping
        """
        print("\nCounting node types...")

        # Try to get node labels/types
        try:
            query = "CALL db.labels()"
            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            labels = []
            if len(result) > 1:
                for row in result[1]:
                    labels.append(row[0])

            print(f"Found {len(labels)} node types/labels: {labels}")

            # Count each type
            type_counts = {}
            for label in labels:
                count_query = f"MATCH (n:{label}) RETURN count(n)"
                count_result = self.client.execute_command(
                    "GRAPH.QUERY", self.graph_name, count_query
                )
                if len(count_result) > 1 and len(count_result[1]) > 0:
                    type_counts[label] = int(count_result[1][0][0])

            return type_counts

        except Exception as e:
            print(f"Error getting node types: {e}")
            return {}

    def discover_relationship_types(self):
        """
        Discover all relationship types in the graph

        Returns:
            list: List of relationship type names
        """
        print("\nDiscovering relationship types in the graph...")

        # Query to get all relationship types
        query = "CALL db.relationshipTypes()"

        try:
            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            rel_types = []
            if len(result) > 1:
                for row in result[1]:
                    rel_types.append(row[0])

            print(f"Found {len(rel_types)} relationship types:")
            for rel_type in rel_types:
                print(f"  - {rel_type}")

            return rel_types
        except Exception as e:
            print(f"Error discovering relationship types: {e}")
            print("Trying alternative method...")

            # Alternative: sample edges and get their types
            query_alt = "MATCH ()-[r]->() RETURN DISTINCT type(r) LIMIT 100"
            result = self.client.execute_command(
                "GRAPH.QUERY", self.graph_name, query_alt
            )

            rel_types = []
            if len(result) > 1:
                for row in result[1]:
                    rel_types.append(row[0])

            print(f"Found {len(rel_types)} relationship types:")
            for rel_type in rel_types:
                print(f"  - {rel_type}")

            return rel_types

    def get_node_degrees_by_edge_type_batched(
        self,
        within_doc_rel,
        cross_doc_rel,
        node_types=["Concept", "Intervention"],
        batch_size=5000,
    ):
        """
        Batched version for large graphs using ID-based partitioning
        Does TWO separate passes - one for each edge type
        Filters to only specified node types

        Args:
            within_doc_rel: Relationship type name for within-document edges
            cross_doc_rel: Relationship type name for cross-document edges
            node_types: List of node type labels to include (e.g., ['Concept', 'Intervention'])
            batch_size: Number of nodes per batch

        Returns:
            dict: Dictionary with three degree lists
        """
        # Build WHERE clause for node types
        if node_types:
            type_conditions = " OR ".join([f"'{nt}' IN labels(n)" for nt in node_types])
            where_clause_full = f"WHERE ({type_conditions})"
            where_clause_and = f"AND ({type_conditions})"
        else:
            where_clause_full = ""
            where_clause_and = ""

        # Get total node count for these types
        count_query = f"""
        MATCH (n)
        {where_clause_full}
        RETURN count(n) as total
        """
        count_result = self.client.execute_command(
            "GRAPH.QUERY", self.graph_name, count_query
        )

        total_nodes = 0
        if len(count_result) > 1 and len(count_result[1]) > 0:
            total_nodes = int(count_result[1][0][0])

        print(f"Total nodes of types {node_types}: {total_nodes:,}")

        # Get node ID range for these types
        id_query = f"""
        MATCH (n)
        {where_clause_full}
        RETURN min(id(n)) as min_id, max(id(n)) as max_id
        """
        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, id_query)

        if len(result) > 1 and len(result[1]) > 0:
            min_id = int(result[1][0][0])
            max_id = int(result[1][0][1])
            print(f"Node ID range: {min_id} to {max_id}")
        else:
            raise Exception("Could not determine node ID range")

        # PASS 1: Get within-document edge degrees
        print(f"\nPass 1: Extracting '{within_doc_rel}' degrees in batches...")
        within_doc_degrees = {}
        current_id = min_id
        batch_num = 1

        while current_id <= max_id:
            query = f"""
            MATCH (n)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size} {where_clause_and}
            RETURN id(n) as node_id, SIZE((n)-[:{within_doc_rel}]-()) as degree
            """

            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            if len(result) > 1 and len(result[1]) > 0:
                for row in result[1]:
                    node_id = int(row[0])
                    degree = int(row[1])
                    within_doc_degrees[node_id] = degree

                progress = len(within_doc_degrees)
                progress_pct = 100 * progress / total_nodes if total_nodes > 0 else 0
                print(
                    f"  Batch {batch_num}: {progress:,}/{total_nodes:,} nodes ({progress_pct:.1f}%)"
                )
                batch_num += 1

            current_id += batch_size

        print(f"✓ Pass 1 complete: {len(within_doc_degrees):,} nodes")

        # PASS 2: Get cross-document similarity degrees
        print(f"\nPass 2: Extracting '{cross_doc_rel}' degrees in batches...")
        cross_doc_degrees = {}
        current_id = min_id
        batch_num = 1

        while current_id <= max_id:
            query = f"""
            MATCH (n)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size} {where_clause_and}
            RETURN id(n) as node_id, SIZE((n)-[:{cross_doc_rel}]-()) as degree
            """

            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            if len(result) > 1 and len(result[1]) > 0:
                for row in result[1]:
                    node_id = int(row[0])
                    degree = int(row[1])
                    cross_doc_degrees[node_id] = degree

                progress = len(cross_doc_degrees)
                progress_pct = 100 * progress / total_nodes if total_nodes > 0 else 0
                print(
                    f"  Batch {batch_num}: {progress:,}/{total_nodes:,} nodes ({progress_pct:.1f}%)"
                )
                batch_num += 1

            current_id += batch_size

        print(f"✓ Pass 2 complete: {len(cross_doc_degrees):,} nodes")

        # Combine results - ensure all node IDs are covered
        all_node_ids = sorted(
            set(within_doc_degrees.keys()) | set(cross_doc_degrees.keys())
        )

        within_doc_list = []
        cross_doc_list = []
        combined_list = []

        for node_id in all_node_ids:
            within_deg = within_doc_degrees.get(node_id, 0)
            cross_deg = cross_doc_degrees.get(node_id, 0)

            within_doc_list.append(within_deg)
            cross_doc_list.append(cross_deg)
            combined_list.append(within_deg + cross_deg)

        print(f"\n✓ Extracted degrees for {len(combined_list):,} nodes")

        return {
            "within_document": within_doc_list,
            "cross_document": cross_doc_list,
            "combined": combined_list,
        }

    def plot_degree_comparison(self, degrees_dict, save_path="degree_comparison.png"):
        """
        Plot three distributions overlaid: within-doc, cross-doc, and combined
        Creates 3 subplots: lin-lin, log-log, and CCDF

        Args:
            degrees_dict: Dictionary with 'within_document', 'cross_document', 'combined' degree lists
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        save_path = save_path.replace(
            "degree_",
            f"degree_analysis_degreedistribution_embeddings-{embeddings_type}_",
        )

        # Colors and labels for the three distributions
        configs = [
            ("combined", "Combined (Local + Similarity)", "#2E86AB", "o"),
            ("within_document", "Within-document (Local only)", "#A23B72", "s"),
            ("cross_document", "Cross-document (Similarity only)", "#F18F01", "^"),
        ]

        # 1. Linear-Linear Histogram (limited to degree 50)
        ax1 = axes[0]
        for key, label, color, marker in configs:
            degrees = degrees_dict[key]
            # Filter to degrees <= 50 for better resolution
            degrees_filtered = [d for d in degrees if d <= 50]
            ax1.hist(
                degrees_filtered,
                bins=50,
                range=(0, 50),
                alpha=0.5,
                label=label,
                color=color,
                edgecolor="black",
            )

        ax1.set_xlabel("Degree", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title(
            "Degree Distribution (Linear Scale, 0-50)", fontsize=14, fontweight="bold"
        )
        ax1.set_xlim(0, 50)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Log-Log Scatter
        ax2 = axes[1]
        for key, label, color, marker in configs:
            degrees = degrees_dict[key]
            degree_counts = Counter(degrees)
            unique_degrees = sorted([d for d in degree_counts.keys() if d > 0])
            frequencies = [degree_counts[d] for d in unique_degrees]

            if len(unique_degrees) > 0:
                ax2.scatter(
                    unique_degrees,
                    frequencies,
                    alpha=0.6,
                    s=50,
                    label=label,
                    color=color,
                    marker=marker,
                )

                # Fit power law ONLY for cross-document and combined (not for within-document)
                if key in ["cross_document"] and len(unique_degrees) > 5:
                    log_degrees = np.log10(unique_degrees)
                    log_freqs = np.log10(frequencies)
                    # Create mask and apply to both arrays
                    mask = log_degrees <= 2
                    log_degrees = log_degrees[mask]
                    log_freqs = log_freqs[mask]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        log_degrees, log_freqs
                    )

                    # Plot fitted line
                    fit_x = np.array(unique_degrees)
                    fit_y = 10 ** (intercept) * fit_x**slope
                    ax2.plot(
                        fit_x,
                        fit_y,
                        "--",
                        linewidth=2,
                        color=color,
                        alpha=0.7,
                        label=f"{label} fit: γ={-slope:.2f}, R²={r_value**2:.3f}",
                    )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Degree (log scale)", fontsize=12)
        ax2.set_ylabel("Frequency (log scale)", fontsize=12)
        ax2.set_title(
            "Degree Distribution (Log-Log Plot)", fontsize=14, fontweight="bold"
        )
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, which="both")

        # 3. CCDF (Complementary Cumulative Distribution Function)
        ax3 = axes[2]
        for key, label, color, marker in configs:
            degrees = degrees_dict[key]
            if len(degrees) == 0:
                continue

            sorted_degrees = np.sort(degrees)
            # Remove zeros for log scale
            sorted_degrees = sorted_degrees[sorted_degrees > 0]

            if len(sorted_degrees) > 0:
                ccdf = 1 - np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
                ax3.scatter(
                    sorted_degrees,
                    ccdf,
                    alpha=0.5,
                    s=20,
                    label=label,
                    color=color,
                    marker=marker,
                )

        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel("Degree (log scale)", fontsize=12)
        ax3.set_ylabel("P(Degree ≥ k)", fontsize=12)
        ax3.set_title(
            "CCDF - Complementary Cumulative Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
        plt.close()

    def analyze_similarity_link_diversity_stepwise(
        self,
        cross_doc_rel=similarity_edge_name,
        node_types=["Concept", "Intervention"],
        sample_size=50,
        min_degree=100,
    ):
        """
        Analyze what fraction of similarity connections go to unique data sources.

        For each node:
        - Count total similarity connections (degree)
        - Count unique data sources among those connections
        - Calculate ratio: unique_sources / total_connections

        A high ratio (close to 1.0) means each connection is to a different source.
        A low ratio means multiple connections to nodes from the same sources (concentration).

        Only reports nodes with diversity ratio < 1.0 (non-perfect diversity).

        Args:
            cross_doc_rel: Relationship type name for cross-document edges
            node_types: List of node type labels to check
            sample_size: Number of nodes to analyze (will attempt to find this many with >min_degree)
            min_degree: Minimum similarity degree to include in sample

        Returns:
            dict: Statistics about link diversity and concentration
        """
        print("\n" + "=" * 80)
        print("SIMILARITY LINK DIVERSITY ANALYSIS")
        print("=" * 80)
        print(
            f"\nFinding nodes with >{min_degree} similarity edges (may take a moment)...\n"
        )

        # Build WHERE clause for node types
        if node_types:
            type_conditions = " OR ".join([f"'{nt}' IN labels(n)" for nt in node_types])
            where_type = f"AND ({type_conditions})"
        else:
            where_type = ""

        # STEP 1: Get candidate node IDs (simple query, no counting yet)
        print("Step 1: Getting candidate node IDs...")

        # Get a large pool of candidates - we'll filter by degree in the per-node queries
        id_query = f"""
        MATCH (n)-[:{cross_doc_rel}]-()
        WHERE id(n) >= 0 {where_type}
        RETURN DISTINCT id(n) as node_id
        LIMIT {sample_size * 5}
        """

        try:
            result = self.client.execute_command(
                "GRAPH.QUERY", self.graph_name, id_query
            )

            if len(result) <= 1 or len(result[1]) == 0:
                print("⚠ No nodes found with similarity edges")
                return None

            candidate_ids = [int(row[0]) for row in result[1]]
            print(f"✓ Got {len(candidate_ids)} candidate nodes")

        except Exception as e:
            print(f"✗ Error getting node IDs: {e}")
            return None

        # STEP 2: For each candidate, check degree and analyze if it qualifies
        print(f"\nStep 2: Filtering and analyzing nodes with >{min_degree} edges...")

        nodes_analyzed = []
        nodes_checked = 0

        for node_id in candidate_ids:
            if len(nodes_analyzed) >= sample_size:
                break

            try:
                # Build node type filter for neighbors
                if node_types:
                    type_conditions_m = " OR ".join(
                        [f"'{nt}' IN labels(m)" for nt in node_types]
                    )
                    where_type_m = f"AND ({type_conditions_m})"
                else:
                    where_type_m = ""

                # Simple query: count connections and check if >= min_degree
                # DIAGNOSTIC: Check for unexpected edge patterns
                node_query = f"""
                MATCH (n)
                WHERE id(n) = {node_id}
                MATCH (n)-[:FROM]->(s:Source)
                WITH n, s.url as source_url
                MATCH (n)-[r:{cross_doc_rel}]-(m)
                WHERE m <> n {where_type_m}
                WITH n, source_url, m, COUNT(r) as edge_count
                MATCH (m)-[:FROM]->(target:Source)
                WHERE target.url <> source_url
                WITH n, source_url, m, edge_count, COLLECT(target.url) as target_urls
                WITH n, source_url, m, edge_count, target_urls[0] as target_url
                WITH n.name as node_name,
                    source_url,
                    COUNT(DISTINCT m) as total_connections,
                    SUM(edge_count) as total_edges,
                    COLLECT(DISTINCT target_url) as unique_target_sources
                WHERE total_connections > {min_degree}
                RETURN node_name, 
                    source_url, 
                    total_connections,
                    total_edges,
                    SIZE(unique_target_sources) as unique_sources,
                    unique_target_sources
                """

                node_result = self.client.execute_command(
                    "GRAPH.QUERY", self.graph_name, node_query
                )
                nodes_checked += 1

                if len(node_result) > 1 and len(node_result[1]) > 0:
                    row = node_result[1][0]
                    node_name = row[0] if row[0] else "No name"
                    source_url = row[1] if row[1] else "Unknown"
                    total_connections = int(row[2]) if len(row) > 2 else 0
                    total_edges = int(row[3]) if len(row) > 3 else 0
                    unique_sources = int(row[4]) if len(row) > 4 else 0
                    target_sources = row[5] if len(row) > 5 else []

                    # DIAGNOSTIC: Check if total_edges != total_connections
                    if total_edges != total_connections:
                        print(
                            f"  ⚠ WARNING: Node {node_id} has {total_edges} edges but {total_connections} unique neighbors!"
                        )
                        print(
                            "    This suggests multiple edges between same node pairs or bidirectional edges."
                        )

                    if total_connections > 0 and unique_sources > 0:
                        diversity_ratio = unique_sources / total_connections

                        # Only include nodes with non-perfect diversity
                        if diversity_ratio < 1.0:
                            nodes_analyzed.append(
                                {
                                    "node_id": node_id,
                                    "name": node_name,
                                    "source": source_url,
                                    "total_connections": total_connections,
                                    "unique_sources": unique_sources,
                                    "diversity_ratio": diversity_ratio,
                                    "target_sources": target_sources,
                                }
                            )

                # Progress update
                if nodes_checked % 10 == 0:
                    print(
                        f"  Checked {nodes_checked} nodes, found {len(nodes_analyzed)} qualifying nodes with concentration..."
                    )

            except Exception as e:
                print(f"  Warning: Query failed for node {node_id}: {e}")
                continue

        print(f"\n✓ Checked {nodes_checked} nodes total")

        if len(nodes_analyzed) == 0:
            print(
                f"✓ All analyzed nodes with >{min_degree} edges have perfect diversity (100% unique sources)!"
            )
            print(
                "  No concentration detected - each similarity connection goes to a different source."
            )
            return {
                "nodes_sampled": nodes_checked,
                "nodes_with_concentration": 0,
                "mean_diversity_ratio": 1.0,
                "message": "All nodes have perfect diversity",
            }

        print(
            f"\n✓ Found {len(nodes_analyzed)} nodes with concentration (< 100% diversity)"
        )
        print(
            f"  ({nodes_checked - len(nodes_analyzed)} nodes had perfect diversity or didn't qualify)"
        )

        # Calculate statistics
        diversity_ratios = [n["diversity_ratio"] for n in nodes_analyzed]
        total_connections_list = [n["total_connections"] for n in nodes_analyzed]
        unique_sources_list = [n["unique_sources"] for n in nodes_analyzed]

        print("\n" + "=" * 80)
        print("DIVERSITY RATIO STATISTICS (Unique Sources / Total Connections)")
        print("=" * 80)
        print(f"  Nodes checked: {nodes_checked}")
        print(
            f"  Nodes with >{min_degree} edges and concentration: {len(nodes_analyzed)} ({100 * len(nodes_analyzed) / nodes_checked:.1f}%)"
        )
        print("\nAmong nodes with concentration:")
        print(f"  Mean ratio: {np.mean(diversity_ratios):.3f}")
        print(f"  Median ratio: {np.median(diversity_ratios):.3f}")
        print(f"  Std Dev: {np.std(diversity_ratios):.3f}")
        print(f"  Min ratio: {np.min(diversity_ratios):.3f}")
        print(f"  Max ratio: {np.max(diversity_ratios):.3f}")

        print("\nInterpretation:")
        print(
            "  Ratio = 1.0 means all connections go to different sources (maximum diversity)"
        )
        print(
            "  Ratio < 0.5 means high concentration (>2 connections per source on average)"
        )
        print(
            "  Ratio < 0.2 means extreme concentration (>5 connections per source on average)"
        )

        print("\n" + "=" * 80)
        print("CONNECTION STATISTICS (Nodes with Concentration Only)")
        print("=" * 80)
        print("Total Connections Per Node:")
        print(f"  Mean: {np.mean(total_connections_list):.1f}")
        print(f"  Median: {np.median(total_connections_list):.1f}")
        print(f"  Max: {np.max(total_connections_list)}")

        print("\nUnique Sources Per Node:")
        print(f"  Mean: {np.mean(unique_sources_list):.1f}")
        print(f"  Median: {np.median(unique_sources_list):.1f}")
        print(f"  Max: {np.max(unique_sources_list)}")

        # Distribution of diversity ratios
        from collections import Counter

        # Bin diversity ratios
        ratio_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        binned_ratios = []
        for ratio in diversity_ratios:
            for i in range(len(ratio_bins) - 1):
                if ratio <= ratio_bins[i + 1]:
                    binned_ratios.append(f"{ratio_bins[i]:.1f}-{ratio_bins[i + 1]:.1f}")
                    break

        distribution = Counter(binned_ratios)
        print("\nDiversity Ratio Distribution:")
        for bin_range in [
            f"{ratio_bins[i]:.1f}-{ratio_bins[i + 1]:.1f}"
            for i in range(len(ratio_bins) - 1)
        ]:
            count = distribution[bin_range]
            pct = 100 * count / len(nodes_analyzed)
            print(f"  {bin_range}: {count} nodes ({pct:.1f}%)")

        # Find nodes with high concentration (low diversity ratio)
        nodes_high_concentration = sorted(
            nodes_analyzed, key=lambda x: x["diversity_ratio"]
        )[:5]

        print("\n" + "=" * 80)
        print("TOP 5 NODES WITH HIGHEST CONCENTRATION (Lowest Diversity Ratio)")
        print("=" * 80)
        for i, node in enumerate(nodes_high_concentration, 1):
            avg_connections_per_source = (
                node["total_connections"] / node["unique_sources"]
            )
            print(f"\n  {i}. Node {node['node_id']}")
            print(f"     Name: {node['name'][:80]}...")
            print(f"     Total connections: {node['total_connections']}")
            print(f"     Unique sources: {node['unique_sources']}")
            print(f"     Diversity ratio: {node['diversity_ratio']:.3f}")
            print(f"     Avg connections per source: {avg_connections_per_source:.1f}")
            if node["source"] != "Unknown":
                print(f"     From: {node['source'][:60]}...")

        return {
            "nodes_sampled": nodes_checked,
            "nodes_with_concentration": len(nodes_analyzed),
            "concentration_percentage": 100 * len(nodes_analyzed) / nodes_checked,
            "mean_diversity_ratio": np.mean(diversity_ratios),
            "median_diversity_ratio": np.median(diversity_ratios),
            "mean_total_connections": np.mean(total_connections_list),
            "mean_unique_sources": np.mean(unique_sources_list),
            "ratio_distribution": dict(distribution),
        }

    def inspect_node_edges(self, node_id, rel_type=similarity_edge_name):
        """
        Diagnostic function to inspect all edges of a specific node

        Args:
            node_id: The node ID to inspect
            rel_type: Relationship type to inspect
        """
        print("\n" + "=" * 80)
        print(f"INSPECTING NODE {node_id} EDGES")
        print("=" * 80)

        # Check FROM relationships
        from_query = f"""
        MATCH (n)-[r:FROM]->(s)
        WHERE id(n) = {node_id}
        RETURN type(r) as rel_type, id(s) as source_id, s.url as source_url, n.url as node_url
        """

        print(f"\nFROM relationships for node {node_id}:")
        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, from_query)
        if len(result) > 1 and len(result[1]) > 0:
            for i, row in enumerate(result[1]):
                print(
                    f"  {i + 1}. {row[0]} -> Source ID {row[1]}, url {row[2]}; node.url: {row[3]}"
                )
        else:
            print("  None found")

        # Get all similarity edges with details
        edge_query = f"""
        MATCH (n)-[r:{rel_type}]-(m)
        WHERE id(n) = {node_id}
        RETURN id(m) as target_id,
               type(r) as rel_type,
               startNode(r) = n as is_outgoing,
               labels(m) as target_labels,
               m.name as target_name,
               r.score as similarity_score,
               n.name as node_name,
               n.description as source_desc,
               n.concept_category as source_category,
               n.aliases as source_aliases,
               m.description as target_desc,
               m.concept_category as target_category,
               m.aliases as target_aliases
        ORDER BY target_id
        """

        print(f"\nAll {rel_type} edges for node {node_id}:")
        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, edge_query)

        if len(result) > 1 and len(result[1]) > 0:
            print(f"  Total edge matches: {len(result[1])}")

            # Count by target
            from collections import Counter

            target_counts = Counter([int(row[0]) for row in result[1]])
            unique_targets = len(target_counts)

            print(f"  Unique target nodes: {unique_targets}")
            print(f"  Edges per target ratio: {len(result[1]) / unique_targets:.2f}")

            # Find duplicates
            duplicates = {
                tid: count for tid, count in target_counts.items() if count > 1
            }
            if duplicates:
                print(f"\n  Found {len(duplicates)} targets with multiple edges:")
                for tid, count in sorted(duplicates.items(), key=lambda x: -x[1])[:10]:
                    print(f"    Target {tid}: {count} edges")
                    # Show details for this target
                    for row in result[1]:
                        if int(row[0]) == tid:
                            direction = (
                                "outgoing (n->m)" if row[2] else "incoming (m->n)"
                            )
                            print(
                                f"      - {direction}, labels: {row[3]}, name: {row[4]}"
                            )
            else:
                print("  No duplicates - each target appears exactly once")

            # Sample first 100 edges sorted by highest cosine similarity
            print(
                f"\n  Sample of 100 highest similarity edges of node {node_id} with name '{result[1][0][6]}', description '{result[1][0][7]}', concept_category '{result[1][0][8]}' and ",
                end="",
            )
            target_aliases_str = result[1][0][9]

            if target_aliases_str and target_aliases_str != "[]":
                # Strip brackets and split by comma
                aliases_list = [
                    x.strip() for x in target_aliases_str.strip("[]").split(", ")
                ]

                first_alias = aliases_list[0] if len(aliases_list) > 0 else None
                second_alias = aliases_list[1] if len(aliases_list) > 1 else None
                third_alias = aliases_list[2] if len(aliases_list) > 2 else None

                print(f"aliases '{first_alias}' and '{second_alias}'", end="")
                if third_alias:
                    print(f" and '{third_alias}':")
                else:
                    print(":")

                # Output: aliases 'AI-caused human extinction' and 'Existential risk from misaligned AI'
            else:
                print("No aliases")

            sorted_edges = sorted(
                result[1][:100],
                key=lambda row: 1 - (float(row[5]) ** 2) / 2
                if row[5] is not None
                else float("-inf"),
                reverse=True,
            )

            for i, row in enumerate(sorted_edges):
                direction = "n->m" if row[2] else "m->n"
                eucl_score = f"{float(row[5]):.3g}" if row[5] is not None else "None"
                cosine_similarity = (
                    f"{(1 - (float(row[5]) ** 2) / 2):.3g}"
                    if row[5] is not None
                    else "None"
                )
                print(
                    f"    {i + 1}. Target {row[0]} ({direction}), labels: {row[3]}, name: {row[4] if row[4] else 'None'}, Cosine sim: {cosine_similarity}, Eucl score: {eucl_score},  description '{row[10]}', concept_category '{row[11]}' and ",
                    end="",
                )
                target_aliases_str = row[12]

                if target_aliases_str and target_aliases_str != "[]":
                    # Strip brackets and split by comma
                    aliases_list = [
                        x.strip() for x in target_aliases_str.strip("[]").split(", ")
                    ]

                    first_alias = aliases_list[0] if len(aliases_list) > 0 else None
                    second_alias = aliases_list[1] if len(aliases_list) > 1 else None

                    print(f"aliases '{first_alias}' and '{second_alias}'")
                    # Output: aliases 'AI-caused human extinction' and 'Existential risk from misaligned AI'
                else:
                    print("No aliases")
        else:
            print("  No edges found")

    def verify_cross_document_edges(
        self,
        cross_doc_rel=similarity_edge_name,
        node_types=["Concept", "Intervention"],
        sample_size=1000,
    ):
        """
        Verify that similarity edges only connect nodes from different data sources
        Uses the 'url' property from Source nodes to distinguish data sources

        Args:
            cross_doc_rel: Relationship type name for cross-document edges
            node_types: List of node type labels to check
            sample_size: Number of edges to sample for verification

        Returns:
            dict: Statistics about same-source vs different-source connections
        """
        print("\n" + "=" * 80)
        print("CROSS-DOCUMENT EDGE VERIFICATION")
        print("=" * 80)
        print(
            f"\nChecking if '{cross_doc_rel}' edges connect nodes from different data sources..."
        )
        print(f"Sampling up to {sample_size} edges for verification...\n")

        # Build WHERE clause for node types
        if node_types:
            type_conditions_n = " OR ".join(
                [f"'{nt}' IN labels(n)" for nt in node_types]
            )
            type_conditions_m = " OR ".join(
                [f"'{nt}' IN labels(m)" for nt in node_types]
            )
            where_clause_both = f"AND ({type_conditions_n}) AND ({type_conditions_m})"
        else:
            where_clause_both = ""

        # Check if edges connect different sources
        query = f"""
        MATCH (n)-[:{cross_doc_rel}]-(m)
        WHERE id(n) < id(m) {where_clause_both}
        MATCH (n)-[:FROM]->(s1:Source)
        MATCH (m)-[:FROM]->(s2:Source)
        RETURN id(n), id(m), s1.url, s2.url
        LIMIT {sample_size}
        """

        try:
            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            if len(result) <= 1 or len(result[1]) == 0:
                print(
                    "⚠ WARNING: No edges found or nodes don't have Source relationships"
                )
                print("Cannot verify cross-document property.")
                return None

            # Analyze results
            same_source_count = 0
            different_source_count = 0
            same_source_examples = []

            for row in result[1]:
                node1_id = int(row[0])
                node2_id = int(row[1])
                source1_url = row[2] if len(row) > 2 else None
                source2_url = row[3] if len(row) > 3 else None

                if source1_url and source2_url:
                    if source1_url == source2_url:
                        same_source_count += 1
                        if len(same_source_examples) < 5:
                            same_source_examples.append(
                                {
                                    "node1": node1_id,
                                    "node2": node2_id,
                                    "source": source1_url,
                                }
                            )
                    else:
                        different_source_count += 1

            total_edges = same_source_count + different_source_count

            # Print results
            print(f"Total edges sampled: {total_edges:,}")
            print(
                f"Edges connecting DIFFERENT sources: {different_source_count:,} ({100 * different_source_count / total_edges:.1f}%)"
            )
            print(
                f"Edges connecting SAME source: {same_source_count:,} ({100 * same_source_count / total_edges:.1f}%)"
            )

            if same_source_count > 0:
                print(
                    f"\n⚠ WARNING: Found {same_source_count} similarity edges connecting nodes within the SAME document!"
                )
                print(
                    "This suggests the similarity threshold may be creating intra-document connections."
                )
                print("\nExamples of same-source connections:")
                for i, example in enumerate(same_source_examples, 1):
                    print(f"  {i}. Nodes {example['node1']} ↔ {example['node2']}")
                    print(f"     Source: {example['source'][:80]}...")
            else:
                print(
                    "\n✓ VERIFIED: All sampled similarity edges connect nodes from DIFFERENT sources"
                )

            return {
                "total_sampled": total_edges,
                "different_source": different_source_count,
                "same_source": same_source_count,
                "different_source_pct": 100 * different_source_count / total_edges
                if total_edges > 0
                else 0,
            }

        except Exception as e:
            print(f"✗ Error during verification: {e}")
            print("This might be due to:")
            print("  - Nodes not having FROM relationships to Source nodes")
            print("  - Source nodes not having 'url' property")
            print("  - Different property name (check your schema)")
            return None

    def print_statistics(self, degrees_dict):
        """
        Print summary statistics for all three degree distributions
        """
        print("\n" + "=" * 80)
        print("DEGREE DISTRIBUTION STATISTICS")
        print("=" * 80)

        for key, label in [
            ("within_document", "Within-Document (Local)"),
            ("cross_document", "Cross-Document (Similarity)"),
            ("combined", "Combined (Local + Similarity)"),
        ]:
            degrees = degrees_dict[key]

            if len(degrees) == 0:
                print(f"\n{label}: No data")
                continue

            # Calculate statistics
            zero_count = sum(1 for d in degrees if d == 0)
            non_zero_degrees = [d for d in degrees if d > 0]

            print(f"\n{label}:")
            print(f"  Total nodes: {len(degrees):,}")

            # Only show mean/median for within-document (peaked distribution)
            # Skip for power law distributions (cross-document and combined)
            if key == "within_document":
                print(f"  Mean degree (all nodes): {np.mean(degrees):.2f}")
                if non_zero_degrees:
                    print(
                        f"  Mean degree (non-zero only): {np.mean(non_zero_degrees):.2f}"
                    )
                print(f"  Median degree: {np.median(degrees):.2f}")

            print(f"  Std deviation: {np.std(degrees):.2f}")
            print(f"  Min degree: {np.min(degrees)}")
            print(f"  Max degree: {np.max(degrees)}")
            print(
                f"  Nodes with degree 0: {zero_count:,} ({100 * zero_count / len(degrees):.1f}%)"
            )
            if non_zero_degrees:
                print(
                    f"  Nodes with degree > 0: {len(non_zero_degrees):,} ({100 * len(non_zero_degrees) / len(degrees):.1f}%)"
                )

            # Most common degrees
            degree_counts = Counter(degrees)
            most_common = degree_counts.most_common(5)
            print("  Most common degrees:")
            for degree, count in most_common:
                print(
                    f"    Degree {degree}: {count:,} nodes ({100 * count / len(degrees):.1f}%)"
                )

            # Power law fit - only for cross-document
            if key in ["cross_document"] and len(non_zero_degrees) > 5:
                degree_counts_nz = Counter(non_zero_degrees)
                unique_degrees = sorted(degree_counts_nz.keys())
                frequencies = [degree_counts_nz[d] for d in unique_degrees]

                log_degrees = np.log10(unique_degrees)
                # limit fit to low noise range with sufficient statistics
                log_freqs = np.log10(frequencies)
                # Create mask and apply to both arrays
                mask = log_degrees <= 2
                log_degrees = log_degrees[mask]
                log_freqs = log_freqs[mask]
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_degrees, log_freqs
                )

                print("  Power law fit (non-zero degrees):")
                print(f"    Exponent (γ): {-slope:.3f}")
                print(f"    R² value: {r_value**2:.3f}")
                print(f"    P-value: {p_value:.6f}")
                # print(f" degree distribution:{log_degrees}  ")
                # Interpretation
                if -slope < 2:
                    print(
                        "    Note: γ < 2 indicates EXTREME hub concentration (very heavy-tailed)"
                    )
                elif -slope < 3:
                    print("    Note: 2 < γ < 3 is typical scale-free behavior")
                else:
                    print("    Note: γ > 3 indicates more uniform distribution")


def main():
    # Initialize analyzer
    analyzer = GraphEdgeAnalyzer(
        host="localhost", port=6379, graph_name="AISafetyIntervention"
    )

    print("Connecting to FalkorDB...")

    try:
        # Test connection
        analyzer.client.ping()
        print("✓ Connected to FalkorDB")

        # Get node type counts
        type_counts = analyzer.get_node_type_counts()
        if type_counts:
            print("\nNode type counts:")
            for node_type, count in sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {node_type}: {count:,} nodes")

        # Discover relationship types
        rel_types = analyzer.discover_relationship_types()

        if len(rel_types) == 0:
            print("\n✗ No relationship types found in the graph!")
            return

        # Use specified relationship types
        within_doc_rel = "EDGE"
        cross_doc_rel = similarity_edge_name

        # Verify these relationship types exist
        if within_doc_rel not in rel_types:
            print(
                f"\n✗ ERROR: Relationship type '{within_doc_rel}' not found in graph!"
            )
            print("Available types:", rel_types)
            return

        if cross_doc_rel not in rel_types:
            print(f"\n✗ ERROR: Relationship type '{cross_doc_rel}' not found in graph!")
            print("Available types:", rel_types)
            return

        print("\n✓ Using relationship types:")
        print(f"  Within-document edges: '{within_doc_rel}'")
        print(f"  Cross-document edges: '{cross_doc_rel}'")

        # Specify node types to analyze (Concept and Intervention only)
        node_types_to_analyze = ["Concept", "Intervention"]
        print(f"\n✓ Filtering to node types: {node_types_to_analyze}")

        # Use batched approach for large graph
        print("\nUsing batched approach for large graph...")
        degrees_dict = analyzer.get_node_degrees_by_edge_type_batched(
            within_doc_rel, cross_doc_rel, node_types=node_types_to_analyze
        )

        # Check if we actually got data
        if all(max(degrees_dict[key]) == 0 for key in degrees_dict):
            print("\n✗ WARNING: All degrees are 0!")
            print("This suggests the relationship types or node types are incorrect.")
            print("Please check the relationship type names and node type labels.")
            return

        # Verify cross-document edges connect different sources
        _verification_result = analyzer.verify_cross_document_edges(
            cross_doc_rel=cross_doc_rel,
            node_types=node_types_to_analyze,
            sample_size=1000,
        )

        # DIAGNOSTIC: Inspect a specific high-degree node
        print("\n" + "=" * 80)
        print("DIAGNOSTIC: Inspecting node 384 to understand edge counting")
        print("=" * 80)
        analyzer.inspect_node_edges(node_id=384, rel_type=cross_doc_rel)

        # Analyze similarity link diversity (how many unique papers each node connects to)
        # Using step-wise approach with tiny queries, filtering to high-degree nodes
        _diversity_result = analyzer.analyze_similarity_link_diversity_stepwise(
            cross_doc_rel=cross_doc_rel,
            node_types=node_types_to_analyze,
            sample_size=50,  # Analyze 50 nodes
            min_degree=100,  # Only nodes with >100 similarity edges
        )

        # Print statistics
        analyzer.print_statistics(degrees_dict)

        # Create comparison plot
        print("\n" + "=" * 80)
        print("CREATING COMPARISON PLOTS")
        print("=" * 80)
        analyzer.plot_degree_comparison(
            degrees_dict, "degree_comparison_by_edge_type.png"
        )

        print("\n✓ Analysis complete!")

    except redis.exceptions.ConnectionError as e:
        print(f"✗ Connection error: {e}")
        print("Make sure FalkorDB is running on localhost:6379")
    except redis.exceptions.ResponseError as e:
        print(f"✗ Query error: {e}")
        print("Make sure the graph name is correct and edge types exist")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
