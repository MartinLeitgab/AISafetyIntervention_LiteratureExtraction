import redis
import networkx as nx
from collections import Counter
import numpy as np


class GraphDiagnostics:
    def __init__(self, host="localhost", port=6379, graph_name="AISafetyIntervention"):
        """
        Initialize connection to FalkorDB for diagnostic tests
        """
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph_name = graph_name

    def get_node_info_batch(self, node_ids):
        """
        Get name/description/label information for a batch of nodes
        """
        node_info = {}

        # Process in batches to avoid huge queries
        batch_size = 100
        for i in range(0, len(node_ids), batch_size):
            batch = node_ids[i : i + batch_size]
            id_list = ",".join(map(str, batch))

            query = f"""
            MATCH (n)
            WHERE id(n) IN [{id_list}]
            RETURN id(n), labels(n), n.name, n.description
            """

            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            if len(result) > 1:
                for row in result[1]:
                    node_id = int(row[0])
                    labels = row[1] if len(row) > 1 else []
                    name = (
                        row[2] if (len(row) > 2 and row[2] is not None) else "No name"
                    )
                    description = (
                        row[3]
                        if (len(row) > 3 and row[3] is not None)
                        else "No description"
                    )
                    node_info[node_id] = {
                        "labels": labels,
                        "name": name,
                        "description": description,
                    }

        return node_info

    def get_node_neighbors(self, node_id, relationship_type, limit=10):
        """
        Get neighbors of a specific node for a specific relationship type
        """
        query = f"""
        MATCH (n)-[:{relationship_type}]-(m)
        WHERE id(n) = {node_id}
        RETURN id(m), labels(m), m.name, m.description, m.url
        LIMIT {limit}
        """

        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

        neighbors = []
        if len(result) > 1:
            for row in result[1]:
                neighbors.append(
                    {
                        "id": int(row[0]),
                        "labels": row[1] if len(row) > 1 else [],
                        "name": row[2]
                        if (len(row) > 2 and row[2] is not None)
                        else "No name",
                        "description": row[3]
                        if (len(row) > 3 and row[3] is not None)
                        else "No description",
                        "url": row[4]
                        if (len(row) > 4 and row[4] is not None)
                        else "No description",
                    }
                )

        return neighbors

    def get_node_document_source(self, node_id):
        """
        Get the source document URL for a given node
        """
        query = f"""
        MATCH (n)-[:FROM]->(s:Source)
        WHERE id(n) = {node_id}
        RETURN s.url
        LIMIT 1
        """

        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

        if len(result) > 1 and len(result[1]) > 0:
            source = result[1][0][0]
            return source if source is not None else "Unknown source"
        return "Unknown source"

    def hub_quality_test(
        self,
        within_doc_rel="EDGE",
        cross_doc_rel="SIMILARITY_ABOVE_POINT_EIGHT_FOR_REAL",
        node_types=["Concept", "Intervention"],
        top_n=20,
        batch_size=10000,
    ):
        """
        Test 1: Hub Quality Analysis
        Identify and inspect top hubs for both within-document and cross-document edges
        Uses batching to avoid timeouts on large graphs
        """
        print("\n" + "=" * 80)
        print("HUB QUALITY TEST")
        print("=" * 80)

        # Build WHERE clause for node types
        if node_types:
            type_conditions = " OR ".join([f"'{nt}' IN labels(n)" for nt in node_types])
            where_clause_and = f"AND ({type_conditions})"
        else:
            where_clause_and = ""

        # Get node ID range
        print("\nDetermining node ID range...")
        id_query = f"""
        MATCH (n)
        WHERE id(n) >= 0 {where_clause_and}
        RETURN min(id(n)) as min_id, max(id(n)) as max_id
        """
        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, id_query)

        if len(result) > 1 and len(result[1]) > 0:
            min_id = int(result[1][0][0])
            max_id = int(result[1][0][1])
            print(f"Node ID range: {min_id} to {max_id}")
        else:
            print("Could not determine node ID range")
            return

        # ===== WITHIN-DOCUMENT HUBS =====
        print(
            f"\n--- Finding Top {top_n} WITHIN-DOCUMENT hubs ('{within_doc_rel}') ---"
        )
        print("Processing in batches to avoid timeout...")

        within_degrees = []
        current_id = min_id
        batch_num = 1

        while current_id <= max_id:
            query = f"""
            MATCH (n)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size} {where_clause_and}
            WITH n, SIZE([(n)-[:{within_doc_rel}]-(m) WHERE n.url = m.url | 1]) as degree
            WHERE degree > 0
            RETURN id(n), degree
            """

            # print(f"{query}")
            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            if len(result) > 1:
                for row in result[1]:
                    within_degrees.append((int(row[0]), int(row[1])))
                print(f"  Batch {batch_num}: Found {len(result[1])} nodes with edges")
                batch_num += 1

            current_id += batch_size

        print(f"✓ Collected {len(within_degrees)} nodes with within-document edges")

        # Sort and get top N
        within_degrees.sort(key=lambda x: x[1], reverse=True)
        within_hubs = within_degrees[:top_n]

        # Display within-document hubs
        if within_hubs:
            print(f"\nTop {len(within_hubs)} within-document hubs:\n")
            node_ids = [node_id for node_id, _ in within_hubs]
            node_info = self.get_node_info_batch(node_ids)

            for i, (node_id, degree) in enumerate(within_hubs, 1):
                info = node_info.get(node_id, {})
                labels = info.get("labels", [])
                name = info.get("name", "No name")
                description = info.get("description", "No description")
                source = self.get_node_document_source(node_id)

                print(f"{i}. Node ID: {node_id}, Degree: {degree}")
                print(f"   Labels: {labels}")
                print(f"   Name: {name}")
                print(f"   Description: {description}")
                print(f"   Source: {source}")

                # Show sample neighbors
                # neighbors = self.get_node_neighbors(node_id, within_doc_rel, limit=5)
                neighbors = self.get_node_neighbors(node_id, within_doc_rel)
                print(f"   Sample neighbors ({len(neighbors)}):")
                for j, neighbor in enumerate(neighbors, 1):
                    print(
                        f"     {j}. {neighbor['name']}, {neighbor['id']}, {neighbor['labels']}, {neighbor['url']}"
                    )
                print()

        ## disregard below code, not the right approach to assess under/overconnection
        return
        # ===== CROSS-DOCUMENT HUBS =====
        print(f"\n--- Finding Top {top_n} CROSS-DOCUMENT hubs ('{cross_doc_rel}') ---")
        print("Processing in batches to avoid timeout...")

        cross_degrees = []
        current_id = min_id
        batch_num = 1

        while current_id <= max_id:
            query = f"""
            MATCH (n)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size} {where_clause_and}
            WITH n, SIZE((n)-[:{cross_doc_rel}]-()) as degree
            WHERE degree > 0
            RETURN id(n), degree
            """

            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            if len(result) > 1:
                for row in result[1]:
                    cross_degrees.append((int(row[0]), int(row[1])))
                print(f"  Batch {batch_num}: Found {len(result[1])} nodes with edges")
                batch_num += 1

            current_id += batch_size

        print(f"✓ Collected {len(cross_degrees)} nodes with cross-document edges")

        # Sort and get top N
        cross_degrees.sort(key=lambda x: x[1], reverse=True)
        cross_hubs = cross_degrees[:top_n]

        # Display cross-document hubs
        if cross_hubs:
            print(f"\nTop {len(cross_hubs)} cross-document hubs:\n")
            node_ids = [node_id for node_id, _ in cross_hubs]
            node_info = self.get_node_info_batch(node_ids)

            for i, (node_id, degree) in enumerate(cross_hubs, 1):
                info = node_info.get(node_id, {})
                labels = info.get("labels", [])
                name = info.get("name", "No name")
                description = info.get("description", "No description")
                source = self.get_node_document_source(node_id)

                print(f"{i}. Node ID: {node_id}, Degree: {degree}")
                print(f"   Labels: {labels}")
                print(f"   Name: {name}")
                print(f"   Description: {description}")
                print(f"   Source: {source}")

                # Count unique documents connected to (with LIMIT 1 to avoid Cartesian product)
                unique_docs_query = f"""
                MATCH (n)-[:{cross_doc_rel}]-(m)
                WHERE id(n) = {node_id}
                WITH DISTINCT m
                MATCH (m)-[:FROM]->(s:Source)
                WITH m, COLLECT(s.url) as source_urls
                WITH m, source_urls[0] as first_source_url
                RETURN COUNT(DISTINCT first_source_url) as unique_sources
                """
                try:
                    doc_result = self.client.execute_command(
                        "GRAPH.QUERY", self.graph_name, unique_docs_query
                    )
                    if len(doc_result) > 1 and len(doc_result[1]) > 0:
                        unique_docs = int(doc_result[1][0][0])
                        print(f"   Connects to {unique_docs} unique documents")
                except Exception as e:
                    print(f"   Could not determine unique document count: {e}")

                # Show sample neighbors from different documents
                neighbors = self.get_node_neighbors(node_id, cross_doc_rel, limit=5)
                print(f"   Sample cross-document neighbors ({len(neighbors)}):")
                for j, neighbor in enumerate(neighbors, 1):
                    print(
                        f"     {j}. {neighbor['name']}, {neighbor['id']}, {neighbor['labels']}, {neighbor['url']} "
                    )
                print()

        # Summary categorization prompt
        print("\n" + "-" * 80)
        print("MANUAL CATEGORIZATION TASK:")
        print("-" * 80)
        print("\nFor cross-document hubs, categorize each as:")
        print("  [G] Generic/Vague (e.g., 'AI system', 'safety', 'model')")
        print(
            "  [F] Foundational Concept (e.g., 'RLHF', 'mesa-optimization', 'alignment')"
        )
        print(
            "  [I] Specific Intervention (e.g., 'Constitutional AI', 'debate', 'ELK')"
        )
        print("\nIf >50% are [G] → Consider increasing similarity cutoff to 0.85-0.9")
        print("If >70% are [F] or [I] → Current cutoff (0.8) is appropriate")

    def query_performance_test(
        self,
        within_doc_rel="EDGE",
        cross_doc_rel="SIMILARITY_ABOVE_POINT_EIGHT_FOR_REAL",
        node_types=["Concept", "Intervention"],
        num_queries=10,
    ):
        """
        Test 2: Query Performance
        Test paths from interventions to top-level risks
        """
        print("\n" + "=" * 80)
        print("QUERY PERFORMANCE TEST")
        print("=" * 80)

        # Build WHERE clause for node types
        if node_types:
            type_conditions = " OR ".join([f"'{nt}' IN labels(n)" for nt in node_types])
            where_type = f"AND ({type_conditions})"
        else:
            where_type = ""

        # Find intervention nodes (containing keywords)
        print("\nIdentifying intervention nodes...")
        intervention_keywords = [
            "RLHF",
            "interpretability",
            "constitutional",
            "debate",
            "oversight",
            "amplification",
            "distillation",
            "fine-tuning",
        ]

        intervention_nodes = []
        for keyword in intervention_keywords:
            query = f"""
            MATCH (n)
            WHERE (toLower(n.name) CONTAINS toLower('{keyword}') OR toLower(n.description) CONTAINS toLower('{keyword}')) {where_type}
            RETURN id(n), n.name
            LIMIT 5
            """
            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)
            if len(result) > 1:
                for row in result[1]:
                    name = (
                        row[1] if (len(row) > 1 and row[1] is not None) else "No name"
                    )
                    intervention_nodes.append((int(row[0]), name))

        # Find risk nodes
        print("Identifying risk nodes...")
        risk_keywords = [
            "existential risk",
            "x-risk",
            "misalignment",
            "deception",
            "power-seeking",
            "catastrophic",
            "unsafe",
        ]

        risk_nodes = []
        for keyword in risk_keywords:
            query = f"""
            MATCH (n)
            WHERE (toLower(n.name) CONTAINS toLower('{keyword}') OR toLower(n.description) CONTAINS toLower('{keyword}')) {where_type}
            RETURN id(n), n.name
            LIMIT 5
            """
            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)
            if len(result) > 1:
                for row in result[1]:
                    name = (
                        row[1] if (len(row) > 1 and row[1] is not None) else "No name"
                    )
                    risk_nodes.append((int(row[0]), name))

        print(f"\nFound {len(intervention_nodes)} intervention nodes")
        print(f"Found {len(risk_nodes)} risk nodes")

        if len(intervention_nodes) == 0 or len(risk_nodes) == 0:
            print("\n⚠ WARNING: Could not find intervention or risk nodes.")
            print("This might be due to text field name or keyword matching.")
            return

        # Test paths
        print(
            f"\n--- Testing {min(num_queries, len(intervention_nodes))} Intervention → Risk Paths ---\n"
        )

        # Sample intervention-risk pairs
        test_pairs = []
        for i in range(min(num_queries, len(intervention_nodes))):
            intervention = intervention_nodes[i % len(intervention_nodes)]
            risk = risk_nodes[i % len(risk_nodes)]
            test_pairs.append((intervention, risk))

        for i, ((int_id, int_text), (risk_id, risk_text)) in enumerate(test_pairs, 1):
            print(f"\nQuery {i}:")
            print(f"  From: {int_text}...")
            print(f"  To: {risk_text}...")

            # Try to find shortest path
            path_query = f"""
            MATCH path = shortestPath((n)-[*..10]-(m))
            WHERE id(n) = {int_id} AND id(m) = {risk_id}
            RETURN [node in nodes(path) | node.name] as path_names, length(path) as path_length
            """

            try:
                result = self.client.execute_command(
                    "GRAPH.QUERY", self.graph_name, path_query
                )

                if len(result) > 1 and len(result[1]) > 0:
                    path_names = result[1][0][0]
                    path_length = int(result[1][0][1])

                    print(f"  ✓ Path found! Length: {path_length} hops")
                    print("  Path preview:")
                    for j, name in enumerate(path_names[:5]):
                        print(f"    {j}. {name}...")
                    if len(path_names) > 5:
                        print(f"    ... ({len(path_names) - 5} more nodes)")
                else:
                    print("  ✗ No path found within 10 hops")
            except Exception as e:
                print(f"  ✗ Error finding path: {e}")

        print("\n" + "-" * 80)
        print("EVALUATION:")
        print("-" * 80)
        print("Manually assess:")
        print("  1. Do the paths make semantic sense?")
        print("  2. Are intermediate nodes meaningful connections?")
        print("  3. Are path lengths reasonable (4-8 hops ideal)?")
        print("  4. Are there false connections or obvious gaps?")

    def path_length_analysis(
        self,
        within_doc_rel="EDGE",
        cross_doc_rel="SIMILARITY_ABOVE_POINT_EIGHT_FOR_REAL",
        node_types=["Concept", "Intervention"],
        sample_size=100,
    ):
        """
        Test 3: Path Length Distribution
        Analyze path lengths between random node pairs
        """
        print("\n" + "=" * 80)
        print("PATH LENGTH ANALYSIS")
        print("=" * 80)

        # Get sample of nodes
        if node_types:
            type_conditions = " OR ".join([f"'{nt}' IN labels(n)" for nt in node_types])
            where_clause = f"WHERE {type_conditions}"
        else:
            where_clause = ""

        print(f"\nSampling {sample_size} random node pairs...")

        # Get random node IDs
        sample_query = f"""
        MATCH (n)
        {where_clause}
        RETURN id(n)
        ORDER BY rand()
        LIMIT {sample_size}
        """

        result = self.client.execute_command(
            "GRAPH.QUERY", self.graph_name, sample_query
        )

        node_ids = []
        if len(result) > 1:
            node_ids = [int(row[0]) for row in result[1]]

        if len(node_ids) < 10:
            print("⚠ Not enough nodes sampled")
            return

        # Test paths between random pairs
        path_lengths = []
        no_path_count = 0

        print(f"Testing paths between {min(50, len(node_ids) // 2)} random pairs...")

        for i in range(min(50, len(node_ids) // 2)):
            node1 = node_ids[i * 2]
            node2 = node_ids[i * 2 + 1]

            path_query = f"""
            MATCH path = shortestPath((n)-[*..15]-(m))
            WHERE id(n) = {node1} AND id(m) = {node2}
            RETURN length(path) as path_length
            """

            try:
                result = self.client.execute_command(
                    "GRAPH.QUERY", self.graph_name, path_query
                )

                if len(result) > 1 and len(result[1]) > 0:
                    path_length = int(result[1][0][0])
                    path_lengths.append(path_length)
                else:
                    no_path_count += 1
            except Exception:
                no_path_count += 1

        # Statistics
        total_pairs = len(path_lengths) + no_path_count
        connectivity_pct = (
            100 * len(path_lengths) / total_pairs if total_pairs > 0 else 0
        )

        print("\n--- Results ---")
        print(f"Pairs tested: {total_pairs}")
        print(f"Paths found: {len(path_lengths)} ({connectivity_pct:.1f}%)")
        print(f"No path: {no_path_count} ({100 - connectivity_pct:.1f}%)")

        if path_lengths:
            print("\nPath length statistics:")
            print(f"  Mean: {np.mean(path_lengths):.2f} hops")
            print(f"  Median: {np.median(path_lengths):.0f} hops")
            print(f"  Min: {np.min(path_lengths)} hops")
            print(f"  Max: {np.max(path_lengths)} hops")
            print(f"  Std dev: {np.std(path_lengths):.2f}")

            # Distribution
            length_counts = Counter(path_lengths)
            print("\nPath length distribution:")
            for length in sorted(length_counts.keys()):
                count = length_counts[length]
                pct = 100 * count / len(path_lengths)
                print(f"  {length} hops: {count} paths ({pct:.1f}%)")

        print("\n" + "-" * 80)
        print("INTERPRETATION:")
        print("-" * 80)
        print(
            f"Connectivity: {connectivity_pct:.1f}% (target: 50-80% for nascent field)"
        )
        if path_lengths:
            mean_len = np.mean(path_lengths)
            if mean_len < 4:
                print(f"Mean path length: {mean_len:.1f} (might be TOO CONNECTED)")
            elif mean_len <= 8:
                print(f"Mean path length: {mean_len:.1f} (GOOD RANGE)")
            else:
                print(f"Mean path length: {mean_len:.1f} (might be FRAGMENTED)")

    def component_analysis(
        self,
        within_doc_rel="EDGE",
        cross_doc_rel="SIMILARITY_ABOVE_POINT_EIGHT_FOR_REAL",
        node_types=["Concept", "Intervention"],
        batch_size=10000,
    ):
        """
        Test 4: Connected Components Analysis
        Identify and characterize disconnected components
        Uses batching to avoid timeouts
        """
        print("\n" + "=" * 80)
        print("COMPONENT ANALYSIS")
        print("=" * 80)

        print("\nNote: This analysis requires loading the graph into NetworkX.")
        print("For large graphs (>50k nodes), this may take several minutes...")

        # Build graph in NetworkX
        print("\nBuilding NetworkX graph using batched queries...")
        G = nx.Graph()

        # Get all nodes of specified types
        if node_types:
            type_conditions = " OR ".join([f"'{nt}' IN labels(n)" for nt in node_types])
            where_clause_and = f"AND ({type_conditions})"
        else:
            where_clause_and = ""

        # Get node ID range
        id_query = f"""
        MATCH (n)
        WHERE id(n) >= 0 {where_clause_and}
        RETURN min(id(n)), max(id(n)), count(n)
        """
        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, id_query)

        if len(result) > 1 and len(result[1]) > 0:
            min_id = int(result[1][0][0])
            max_id = int(result[1][0][1])
            total_nodes = int(result[1][0][2])
            print(f"Node ID range: {min_id} to {max_id}, Total: {total_nodes:,}")
        else:
            print("Could not determine node range")
            return

        # Add nodes in batches
        print("Adding nodes in batches...")
        current_id = min_id
        node_count = 0
        batch_num = 1

        while current_id <= max_id:
            node_query = f"""
            MATCH (n)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size} {where_clause_and}
            RETURN id(n), n.name
            """

            result = self.client.execute_command(
                "GRAPH.QUERY", self.graph_name, node_query
            )

            if len(result) > 1:
                for row in result[1]:
                    node_id = int(row[0])
                    name = row[1] if (len(row) > 1 and row[1] is not None) else ""
                    G.add_node(node_id, name=name)
                    node_count += 1

                print(f"  Batch {batch_num}: {node_count:,}/{total_nodes:,} nodes")
                batch_num += 1

            current_id += batch_size

        print(f"✓ Added {node_count:,} nodes")

        # Add edges in batches (within-document)
        print("\nAdding within-document edges in batches...")
        current_id = min_id
        within_edges = 0
        batch_num = 1

        while current_id <= max_id:
            edge_query = f"""
            MATCH (n)-[:{within_doc_rel}]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size} 
                  AND id(n) < id(m) {where_clause_and}
            RETURN id(n), id(m)
            """

            result = self.client.execute_command(
                "GRAPH.QUERY", self.graph_name, edge_query
            )

            if len(result) > 1:
                for row in result[1]:
                    G.add_edge(int(row[0]), int(row[1]), type="within")
                    within_edges += 1

                print(f"  Batch {batch_num}: {within_edges:,} edges")
                batch_num += 1

            current_id += batch_size

        print(f"✓ Added {within_edges:,} within-document edges")

        # Add cross-document edges in batches
        print("\nAdding cross-document edges in batches...")
        current_id = min_id
        cross_edges = 0
        batch_num = 1

        while current_id <= max_id:
            edge_query = f"""
            MATCH (n)-[:{cross_doc_rel}]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
                  AND id(n) < id(m) {where_clause_and}
            RETURN id(n), id(m)
            """

            result = self.client.execute_command(
                "GRAPH.QUERY", self.graph_name, edge_query
            )

            if len(result) > 1:
                for row in result[1]:
                    G.add_edge(int(row[0]), int(row[1]), type="cross")
                    cross_edges += 1

                print(f"  Batch {batch_num}: {cross_edges:,} edges")
                batch_num += 1

            current_id += batch_size

        print(f"✓ Added {cross_edges:,} cross-document edges")

        # Analyze components
        print("\nAnalyzing connected components...")
        components = list(nx.connected_components(G))
        components = sorted(components, key=len, reverse=True)

        print("\n--- Component Statistics ---")
        print(f"Total components: {len(components)}")
        print(
            f"Largest component: {len(components[0]):,} nodes ({100 * len(components[0]) / len(G):.1f}%)"
        )

        # Size distribution
        size_bins = [1, 10, 50, 100, 500, 1000, 5000, float("inf")]
        size_labels = [
            "1 (isolated)",
            "2-9",
            "10-49",
            "50-99",
            "100-499",
            "500-999",
            "1000-4999",
            "5000+",
        ]
        size_counts = [0] * (len(size_bins) - 1)

        for comp in components:
            size = len(comp)
            for i in range(len(size_bins) - 1):
                if size_bins[i] <= size < size_bins[i + 1]:
                    size_counts[i] += 1
                    break

        print("\nComponent size distribution:")
        for label, count in zip(size_labels, size_counts):
            if count > 0:
                print(f"  {label} nodes: {count} components")

        # Show top 10 components
        print("\n--- Top 10 Largest Components ---")
        for i, comp in enumerate(components[:10], 1):
            # Sample nodes from this component
            sample_nodes = list(comp)[:20]
            names = [G.nodes[n].get("name", "") for n in sample_nodes]

            print(f"\n{i}. Size: {len(comp):,} nodes")
            print("   Sample nodes:")
            for j, name in enumerate(names, 1):
                print(f"     {j}. {name}")

        # Modularity (if not too large)
        if len(G) < 50000:
            print("\nCalculating modularity (this may take a moment)...")
            try:
                communities = nx.community.greedy_modularity_communities(G)
                modularity = nx.community.modularity(G, communities)
                print(f"Modularity: {modularity:.3f}")
                print(f"Number of communities: {len(communities)}")

                if modularity > 0.3:
                    print("  → Good community structure detected")
                elif modularity > 0.1:
                    print("  → Moderate community structure")
                else:
                    print("  → Weak community structure")
            except Exception as e:
                print(f"Could not calculate modularity: {e}")
        else:
            print(
                f"\nSkipping modularity calculation (graph has {len(G):,} nodes, > 50k threshold)"
            )

        print("\n" + "-" * 80)
        print("INTERPRETATION:")
        print("-" * 80)
        print("For AI Safety literature (2023/2024):")
        print("  • Multiple large components (5-20) → Different research areas")
        print("  • One giant component (>80%) → Highly connected field")
        print("  • Many small components → Fragmented/nascent field")
        print(
            f"\nYour network: {len([c for c in components if len(c) >= 100])} components with 100+ nodes"
        )


def main():
    diagnostics = GraphDiagnostics(
        host="localhost", port=6379, graph_name="AISafetyIntervention"
    )

    print("=" * 80)
    print("KNOWLEDGE GRAPH DIAGNOSTIC SUITE")
    print("=" * 80)

    try:
        diagnostics.client.ping()
        print("✓ Connected to FalkorDB\n")

        # Test 1: Local Hub Check
        # print("\nRunning Test 1: Local Hub Check...\n")
        # diagnostics.hub_quality_test(top_n=20)

        # Test 2: Query Performance
        # print("\nRunning Test 2: Query Performance...\n")
        # diagnostics.query_performance_test(num_queries=10)

        # Test 3: Path Length Analysis
        # print("\nRunning Test 3: Path Length Analysis...\n")
        # diagnostics.path_length_analysis(sample_size=100)

        # Test 4: Component Analysis
        print("\nRunning Test 4: Component Analysis...\n")
        diagnostics.component_analysis()

        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUITE COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
