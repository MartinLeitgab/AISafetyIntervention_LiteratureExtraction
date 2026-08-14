"""
Risk & Intervention Hierarchy Analysis (Quality-Filtered, Full Graph)
Extracts complete subgraphs showing how risks/interventions connect to each other.
Tests across 4 thresholds: 0.8, 0.85, 0.9, 0.95
Quality filters: maturity≥3, edge confidence≥4
Processes ALL nodes - no sampling limits
Runtime: ~30-60 min depending on graph size
"""

import redis
import json
import numpy as np
import networkx as nx

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class HierarchyAnalyzer:
    def __init__(self, host="localhost", port=6379, graph="AISafetyIntervention"):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph = graph

    def euclidean_from_cosine(self, cosine):
        return np.sqrt(2 * (1 - cosine))

    def query(self, cypher, timeout=30000):
        result = self.client.execute_command(
            "GRAPH.QUERY", self.graph, cypher, "--timeout", str(timeout)
        )
        return result[1] if len(result) > 1 else []

    def query_all(self, base_query, timeout=120000):
        """Query all results by batching, continuing past timeouts"""
        all_results = []
        batch_size = 2000  # Reduced from 5000
        skip = 0
        consecutive_failures = 0

        while consecutive_failures < 10:  # Increased from 5
            query = f"{base_query} SKIP {skip} LIMIT {batch_size}"
            try:
                batch = self.query(query, timeout)
                consecutive_failures = 0

                if not batch:
                    break
                all_results.extend(batch)
                if len(batch) < batch_size:
                    break

            except Exception:
                consecutive_failures += 1
                print(
                    f"    Timeout at skip={skip}, continuing... ({consecutive_failures}/10)"
                )

            skip += batch_size
            if len(all_results) % 50000 == 0 and len(all_results) > 0:
                print(f"    Retrieved {len(all_results)} rows...")

        return all_results

    def extract_risk_hierarchy(self, cosine_threshold):
        """Extract risk→risk connections via EDGE and SIMILARITY"""
        euclidean = self.euclidean_from_cosine(cosine_threshold)

        print("  Extracting all risk nodes...")
        base_query = """
        MATCH (r:Concept)
        WHERE r.concept_category = 'risk'
        RETURN id(r), r.name
        """
        risk_rows = self.query_all(base_query)
        risks = [(int(row[0]), row[1]) for row in risk_rows]
        print(f"  Found {len(risks)} risk nodes")

        print("  Extracting EDGE connections...")
        base_edge_query = """
        MATCH (r1:Concept)-[e:EDGE]-(r2:Concept)
        WHERE r1.concept_category = 'risk' 
          AND r2.concept_category = 'risk'
          AND e.edge_confidence >= 4
        RETURN id(r1), id(r2), e.type, e.edge_confidence
        """
        edge_rows = self.query_all(base_edge_query)
        print(f"  Found {len(edge_rows)} EDGE connections")

        # Get SIMILARITY connections - ID-based batching (avoids SKIP timeouts)
        print("  Extracting SIMILARITY connections...")

        # Get risk node ID range
        id_range_query = """
        MATCH (r:Concept)
        WHERE r.concept_category = 'risk'
        RETURN min(id(r)) as min_id, max(id(r)) as max_id
        """
        id_result = self.query(id_range_query)

        if not id_result or not id_result[0]:
            print("  No risks found")
            return risks, edge_rows, []

        min_id = int(id_result[0][0])
        max_id = int(id_result[0][1])
        print(f"  Risk ID range: {min_id} to {max_id}")

        sim_edges = []
        current_id = min_id
        batch_size = 2000
        batch_num = 1

        while current_id <= max_id:
            sim_query = f"""
            MATCH (r1:Concept)
            WHERE id(r1) >= {current_id} AND id(r1) < {current_id + batch_size}
              AND r1.concept_category = 'risk'
            WITH r1
            MATCH (r1)-[s:{SIMILARITY_EDGE}]-(r2:Concept)
            WHERE r2.concept_category = 'risk'
              AND id(r2) > id(r1)
              AND s.score < {euclidean}
            RETURN id(r1), id(r2), s.score
            """

            try:
                batch = self.query(sim_query, timeout=60000)
                for row in batch:
                    sim_edges.append((int(row[0]), int(row[1]), float(row[2])))

                if batch_num % 10 == 0:
                    print(f"    Batch {batch_num}: {len(sim_edges)} edges total")
                batch_num += 1

            except Exception as e:
                print(f"    Batch {batch_num} failed: {e}")
                batch_num += 1

            current_id += batch_size

        print(f"  Found {len(sim_edges)} SIMILARITY connections")
        return risks, edge_rows, sim_edges

    def extract_intervention_hierarchy(self, cosine_threshold):
        """Extract intervention→intervention connections"""
        euclidean = self.euclidean_from_cosine(cosine_threshold)

        print("  Extracting all intervention nodes...")
        base_query = """
        MATCH (i:Intervention)
        WHERE i.intervention_maturity >= 3
        RETURN id(i), i.name, i.intervention_maturity, i.intervention_lifecycle
        """
        int_rows = self.query_all(base_query)
        interventions = [(int(row[0]), row[1], row[2], row[3]) for row in int_rows]
        print(f"  Found {len(interventions)} intervention nodes")

        print("  Extracting EDGE connections...")
        base_edge_query = """
        MATCH (i1:Intervention)-[e:EDGE]-(i2:Intervention)
        WHERE i1.intervention_maturity >= 3 AND i2.intervention_maturity >= 3
          AND e.edge_confidence >= 4
        RETURN id(i1), id(i2), e.type, e.edge_confidence
        """
        edge_rows = self.query_all(base_edge_query)
        print(f"  Found {len(edge_rows)} EDGE connections")

        # Get SIMILARITY connections - ID-based batching
        print("  Extracting SIMILARITY connections...")

        # Get intervention ID range
        int_ids = [i[0] for i in interventions]
        if not int_ids:
            return interventions, edge_rows, []

        min_id = min(int_ids)
        max_id = max(int_ids)
        print(f"  Intervention ID range: {min_id} to {max_id}")

        sim_edges = []
        current_id = min_id
        batch_size = 2000
        batch_num = 1

        while current_id <= max_id:
            sim_query = f"""
            MATCH (i1:Intervention)
            WHERE id(i1) >= {current_id} AND id(i1) < {current_id + batch_size}
              AND i1.intervention_maturity >= 3
            WITH i1
            MATCH (i1)-[s:{SIMILARITY_EDGE}]-(i2:Intervention)
            WHERE i2.intervention_maturity >= 3
              AND id(i2) > id(i1)
              AND s.score < {euclidean}
            RETURN id(i1), id(i2), s.score
            """

            try:
                batch = self.query(sim_query, timeout=60000)
                for row in batch:
                    sim_edges.append((int(row[0]), int(row[1]), float(row[2])))

                if batch_num % 10 == 0:
                    print(f"    Batch {batch_num}: {len(sim_edges)} edges total")
                batch_num += 1

            except Exception as e:
                print(f"    Batch {batch_num} failed: {e}")
                batch_num += 1

            current_id += batch_size

        print(f"  Found {len(sim_edges)} SIMILARITY connections")
        return interventions, edge_rows, sim_edges

    def analyze_graph_properties(self, nodes, edge_rows, sim_edges, label):
        """Compute graph statistics distinguishing EDGE vs SIMILARITY"""
        G_total = nx.Graph()
        G_edge = nx.Graph()
        G_sim = nx.Graph()

        # Add nodes
        for node in nodes:
            G_total.add_node(node[0])
            G_edge.add_node(node[0])
            G_sim.add_node(node[0])

        # Add EDGE edges
        for edge in edge_rows:
            G_total.add_edge(int(edge[0]), int(edge[1]))
            G_edge.add_edge(int(edge[0]), int(edge[1]))

        # Add SIMILARITY edges
        for edge in sim_edges:
            G_total.add_edge(edge[0], edge[1])
            G_sim.add_edge(edge[0], edge[1])

        # Statistics
        if len(G_total.nodes()) == 0:
            print(f"  {label}: No nodes")
            return {}

        components_total = list(nx.connected_components(G_total))
        largest_cc = max(components_total, key=len) if components_total else set()

        degrees_total = dict(G_total.degree())
        degrees_edge = dict(G_edge.degree())
        degrees_sim = dict(G_sim.degree())

        stats = {
            "nodes": len(G_total.nodes()),
            "edges_total": len(G_total.edges()),
            "edges_EDGE": len(G_edge.edges()),
            "edges_SIMILARITY": len(G_sim.edges()),
            "components": len(components_total),
            "largest_component": len(largest_cc),
            "avg_degree_total": np.mean(list(degrees_total.values()))
            if degrees_total
            else 0,
            "avg_degree_EDGE": np.mean(list(degrees_edge.values()))
            if degrees_edge
            else 0,
            "avg_degree_SIMILARITY": np.mean(list(degrees_sim.values()))
            if degrees_sim
            else 0,
            "max_degree": max(degrees_total.values()) if degrees_total else 0,
        }

        print(f"  {label}: {stats['nodes']} nodes")
        print(
            f"    EDGE edges: {stats['edges_EDGE']}, avg degree {stats['avg_degree_EDGE']:.1f}"
        )
        print(
            f"    SIMILARITY edges: {stats['edges_SIMILARITY']}, avg degree {stats['avg_degree_SIMILARITY']:.1f}"
        )
        print(
            f"    Total: {stats['edges_total']} edges, {stats['components']} components"
        )

        return stats

    def run_analysis(self, thresholds=[0.8, 0.85, 0.9, 0.95]):
        print("=" * 80)
        print("RISK & INTERVENTION HIERARCHY ANALYSIS")
        print("=" * 80)

        all_results = {}

        for threshold in thresholds:
            print(f"\n{'=' * 80}")
            print(f"THRESHOLD ≥{threshold}")
            print(f"{'=' * 80}")

            # Risk hierarchy
            print("\nRisk Hierarchy:")
            risks, risk_edges, risk_sims = self.extract_risk_hierarchy(threshold)
            risk_stats = self.analyze_graph_properties(
                risks, risk_edges, risk_sims, "Risks"
            )

            # Intervention hierarchy
            print("\nIntervention Hierarchy:")
            interventions, int_edges, int_sims = self.extract_intervention_hierarchy(
                threshold
            )
            int_stats = self.analyze_graph_properties(
                interventions, int_edges, int_sims, "Interventions"
            )

            all_results[threshold] = {"risk": risk_stats, "intervention": int_stats}

        # Summary table
        print(f"\n{'=' * 80}")
        print("SUMMARY ACROSS THRESHOLDS")
        print(f"{'=' * 80}")

        print("\nRisk Hierarchies:")
        print(
            f"{'Threshold':<12} {'Nodes':<8} {'EDGE':<8} {'SIM':<8} {'Total':<8} {'Components':<12}"
        )
        print("-" * 70)
        for t in thresholds:
            r = all_results[t]["risk"]
            print(
                f"≥{t:<11} {r.get('nodes', 0):<8} {r.get('edges_EDGE', 0):<8} "
                f"{r.get('edges_SIMILARITY', 0):<8} {r.get('edges_total', 0):<8} {r.get('components', 0):<12}"
            )

        print("\nIntervention Hierarchies:")
        print(
            f"{'Threshold':<12} {'Nodes':<8} {'EDGE':<8} {'SIM':<8} {'Total':<8} {'Components':<12}"
        )
        print("-" * 70)
        for t in thresholds:
            r = all_results[t]["intervention"]
            print(
                f"≥{t:<11} {r.get('nodes', 0):<8} {r.get('edges_EDGE', 0):<8} "
                f"{r.get('edges_SIMILARITY', 0):<8} {r.get('edges_total', 0):<8} {r.get('components', 0):<12}"
            )

        # Save
        with open("hierarchy_analysis.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n✓ Saved hierarchy_analysis.json")

        return all_results


def main():
    analyzer = HierarchyAnalyzer()
    analyzer.client.ping()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
