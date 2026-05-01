"""
Cluster Integration Utility

Loads semantic clustering graph and provides mapping between
FalkorDB node IDs and semantic cluster assignments.

Usage:
    cluster_map = ClusterMapper('graph_stage2_umap_k40.pkl')
    cluster_id = cluster_map.get_cluster(falkordb_node_id)
    cluster_nodes = cluster_map.get_cluster_members(cluster_id)
"""

import pickle
import redis
from collections import defaultdict, Counter


class ClusterMapper:
    def __init__(
        self,
        graph_pickle_path,
        falkordb_host="localhost",
        falkordb_port=6379,
        graph_name="AISafetyIntervention",
    ):
        """
        Initialize cluster mapper

        Args:
            graph_pickle_path: Path to graph_stage2_umap_k40.pkl
            falkordb_host: FalkorDB host
            falkordb_port: FalkorDB port
            graph_name: FalkorDB graph name
        """
        print("\n" + "=" * 80)
        print("LOADING SEMANTIC CLUSTERING DATA")
        print("=" * 80)

        # Load NetworkX graph with clustering
        print(f"\nLoading clustering graph from {graph_pickle_path}...")
        with open(graph_pickle_path, "rb") as f:
            self.nx_graph = pickle.load(f)

        print(
            f"✓ Loaded graph: {len(self.nx_graph.nodes())} nodes, "
            f"{len(self.nx_graph.edges())} edges"
        )

        # Connect to FalkorDB
        self.client = redis.Redis(
            host=falkordb_host, port=falkordb_port, decode_responses=True
        )
        self.graph_name = graph_name
        # Bug fix 2026-04-30 (CF-5): bump RESULTSET_SIZE to 10M so queries do
        # not silently truncate at the default 10k row limit. Defensive even
        # though queries below are now batched.
        try:
            self.client.execute_command(
                "GRAPH.CONFIG", "SET", "RESULTSET_SIZE", "10000000"
            )
        except Exception as e:
            print(f"  WARN: could not bump RESULTSET_SIZE: {e}")

        # Build mapping: FalkorDB node_id → semantic_cluster
        print("\nBuilding FalkorDB → cluster mapping...")
        self.node_to_cluster = self._build_mapping()

        print(f"✓ Mapped {len(self.node_to_cluster)} FalkorDB nodes to clusters")

        # Build reverse mapping: cluster_id → [FalkorDB node_ids]
        self.cluster_to_nodes = defaultdict(list)
        for node_id, cluster_id in self.node_to_cluster.items():
            self.cluster_to_nodes[cluster_id].append(node_id)

        # Cluster statistics
        cluster_counts = Counter(self.node_to_cluster.values())
        self.num_clusters = len(cluster_counts)

        print("\nCluster distribution:")
        print(f"  Total clusters: {self.num_clusters}")
        print(
            f"  Nodes per cluster (mean): {len(self.node_to_cluster) / self.num_clusters:.1f}"
        )
        print(
            f"  Largest cluster: {max(cluster_counts.values())} nodes (Cluster {max(cluster_counts, key=cluster_counts.get)})"
        )
        print(
            f"  Smallest cluster: {min(cluster_counts.values())} nodes (Cluster {min(cluster_counts, key=cluster_counts.get)})"
        )

    def _build_mapping(self):
        """
        Build mapping between FalkorDB node IDs and semantic clusters

        Strategy:
        1. Get all node names from FalkorDB
        2. Match to NetworkX graph nodes by name
        3. Extract semantic_cluster attribute

        Returns:
            dict: {falkordb_node_id: semantic_cluster}
        """
        # Build name → cluster lookup from NetworkX graph
        name_to_cluster = {}
        for nx_node_id in self.nx_graph.nodes():
            attrs = self.nx_graph.nodes[nx_node_id]
            name = attrs.get("name", "")
            cluster = attrs.get("semantic_cluster")

            if name and cluster is not None:
                name_to_cluster[name] = cluster

        print(f"  NetworkX nodes with clusters: {len(name_to_cluster)}")

        # Query FalkorDB for all Concept/Intervention nodes, batched by id-range.
        # Bug fix 2026-04-30 (CF-5): the original single-shot query was silently
        # truncated at 10,000 rows by FalkorDB's default RESULTSET_SIZE limit.
        # With ~200k Concept+Intervention nodes in the corpus, ~95% of mappings
        # were silently lost. See graph_analysis/phase2_results/rev8_active_state.md
        # Bug Audit (B-1).
        id_q = (
            "MATCH (n) "
            "WHERE 'Concept' IN labels(n) OR 'Intervention' IN labels(n) "
            "RETURN min(id(n)), max(id(n))"
        )
        id_res = self.client.execute_command("GRAPH.QUERY", self.graph_name, id_q)

        mapping = {}
        matched = 0
        unmatched = 0

        if len(id_res) > 1 and id_res[1]:
            min_id = int(id_res[1][0][0])
            max_id = int(id_res[1][0][1])
            cur = min_id
            batch_size = 5000
            all_rows = []
            while cur <= max_id:
                q = (
                    f"MATCH (n) "
                    f"WHERE id(n) >= {cur} AND id(n) < {cur + batch_size} "
                    f"AND ('Concept' IN labels(n) OR 'Intervention' IN labels(n)) "
                    f"RETURN id(n) as node_id, n.name as name"
                )
                res = self.client.execute_command("GRAPH.QUERY", self.graph_name, q)
                if len(res) > 1:
                    all_rows.extend(res[1])
                cur += batch_size

            for row in all_rows:
                falkordb_id = int(row[0])
                name = row[1] if row[1] else ""

                # Try exact match
                if name in name_to_cluster:
                    mapping[falkordb_id] = name_to_cluster[name]
                    matched += 1
                else:
                    # Try case-insensitive match
                    name_lower = name.lower()
                    found = False
                    for nx_name, cluster in name_to_cluster.items():
                        if nx_name.lower() == name_lower:
                            mapping[falkordb_id] = cluster
                            matched += 1
                            found = True
                            break

                    if not found:
                        unmatched += 1

        print(
            f"  Matched: {matched} nodes ({100 * matched / (matched + unmatched):.1f}%)"
        )
        if unmatched > 0:
            print(f"  Unmatched: {unmatched} nodes (likely new or renamed)")

        return mapping

    def get_cluster(self, falkordb_node_id):
        """
        Get semantic cluster for a FalkorDB node ID

        Args:
            falkordb_node_id: Node ID from FalkorDB

        Returns:
            int: Cluster ID (0-39) or None if not mapped
        """
        return self.node_to_cluster.get(falkordb_node_id)

    def get_cluster_members(self, cluster_id):
        """
        Get all FalkorDB node IDs in a cluster

        Args:
            cluster_id: Cluster ID (0-39)

        Returns:
            list: FalkorDB node IDs in this cluster
        """
        return self.cluster_to_nodes.get(cluster_id, [])

    def get_cluster_name(self, cluster_id):
        """
        Get descriptive name for a cluster

        Heuristic: Most common words in node names

        Args:
            cluster_id: Cluster ID

        Returns:
            str: Descriptive cluster name
        """
        node_ids = self.get_cluster_members(cluster_id)

        if len(node_ids) == 0:
            return f"Cluster {cluster_id}"

        # Get node names from FalkorDB
        id_list = ",".join(map(str, node_ids[:100]))  # Sample first 100
        query = f"""
        MATCH (n)
        WHERE id(n) IN [{id_list}]
        RETURN n.name
        """

        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

        names = []
        if len(result) > 1:
            names = [row[0] for row in result[1] if row[0]]

        if len(names) == 0:
            return f"Cluster {cluster_id}"

        # Extract common words (simple heuristic)
        from collections import Counter

        words = []
        for name in names:
            words.extend(name.lower().split())

        # Filter common words
        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "for",
            "to",
            "and",
            "or",
            "with",
            "by",
        }
        words = [w for w in words if w not in stop_words and len(w) > 3]

        if len(words) == 0:
            return f"Cluster {cluster_id}"

        word_counts = Counter(words)
        top_words = [w for w, c in word_counts.most_common(3)]

        return f"Cluster {cluster_id}: {' '.join(top_words).title()}"

    def get_cluster_distribution(self):
        """
        Get cluster size distribution

        Returns:
            dict: {cluster_id: node_count}
        """
        return {
            cluster_id: len(nodes)
            for cluster_id, nodes in self.cluster_to_nodes.items()
        }

    def get_major_clusters(self, top_n=10):
        """
        Get largest N clusters

        Args:
            top_n: Number of clusters to return

        Returns:
            list: [(cluster_id, node_count, cluster_name), ...]
        """
        distribution = self.get_cluster_distribution()
        sorted_clusters = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

        return [
            (cluster_id, count, self.get_cluster_name(cluster_id))
            for cluster_id, count in sorted_clusters[:top_n]
        ]

    def stratify_nodes_by_cluster(self, node_ids, samples_per_cluster=2):
        """
        Stratify a list of nodes by cluster

        Args:
            node_ids: List of FalkorDB node IDs
            samples_per_cluster: Target samples per cluster

        Returns:
            dict: {cluster_id: [sampled_node_ids]}
        """
        # Group by cluster
        clusters = defaultdict(list)
        for node_id in node_ids:
            cluster = self.get_cluster(node_id)
            if cluster is not None:
                clusters[cluster].append(node_id)

        # Sample from each cluster
        import random

        stratified = {}
        for cluster_id, cluster_nodes in clusters.items():
            if len(cluster_nodes) <= samples_per_cluster:
                stratified[cluster_id] = cluster_nodes
            else:
                stratified[cluster_id] = random.sample(
                    cluster_nodes, samples_per_cluster
                )

        return stratified


def test_cluster_mapper():
    """Test cluster mapper functionality"""
    print("\n" + "=" * 80)
    print("TESTING CLUSTER MAPPER")
    print("=" * 80)

    mapper = ClusterMapper("graph_stage2_umap_k40.pkl")

    # Test major clusters
    print("\nTop 10 largest clusters:")
    major_clusters = mapper.get_major_clusters(top_n=10)
    for cluster_id, count, name in major_clusters:
        print(f"  {name}: {count} nodes")

    # Test node lookup
    print("\nTesting node → cluster lookup:")
    sample_nodes = list(mapper.node_to_cluster.keys())[:5]
    for node_id in sample_nodes:
        cluster = mapper.get_cluster(node_id)
        cluster_name = mapper.get_cluster_name(cluster)
        print(f"  Node {node_id} → {cluster_name}")

    print("\n✓ Cluster mapper test complete")


if __name__ == "__main__":
    test_cluster_mapper()
