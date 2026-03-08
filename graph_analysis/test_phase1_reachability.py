"""
Phase 1: Hub-Aware Reachability Filter
Identifies hub vs non-hub interventions and reports stratified connectivity.
"""

import redis
import numpy as np
from collections import deque

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE_NAME = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class HubAwareReachabilityTest:
    def __init__(self, host="localhost", port=6379, graph_name="AISafetyIntervention"):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph_name = graph_name
        self.neighbor_cache = {}

    def euclidean_from_cosine(self, cosine):
        return np.sqrt(2 * (1 - cosine))

    def get_neighbors_cached(self, node_id, euclidean_threshold=None):
        cache_key = (node_id, euclidean_threshold)
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]

        if euclidean_threshold:
            query = f"""
            MATCH (n)-[r]-(m)
            WHERE id(n) = {node_id}
              AND (type(r) = 'EDGE' OR (type(r) = '{SIMILARITY_EDGE_NAME}' AND r.score < {euclidean_threshold}))
            RETURN DISTINCT id(m) LIMIT 100
            """
        else:
            query = f"""
            MATCH (n)-[r:EDGE|{SIMILARITY_EDGE_NAME}]-(m)
            WHERE id(n) = {node_id}
            RETURN DISTINCT id(m) LIMIT 100
            """

        try:
            result = self.client.execute_command(
                "GRAPH.QUERY", self.graph_name, query, "--timeout", "3000"
            )
            neighbors = [int(row[0]) for row in result[1]] if len(result) > 1 else []
            self.neighbor_cache[cache_key] = neighbors
            return neighbors
        except Exception:
            self.neighbor_cache[cache_key] = []
            return []

    def get_node_degree(self, node_id, euclidean_threshold):
        """Get out-degree for a node"""
        return len(self.get_neighbors_cached(node_id, euclidean_threshold))

    def classify_hubs(self, interventions, threshold=0.85, hub_percentile=90):
        """Classify interventions as hub vs non-hub by degree"""
        print(f"\nClassifying hubs (top {100 - hub_percentile}% by degree)...")

        euclidean_threshold = self.euclidean_from_cosine(threshold)

        degrees = {}
        for i, intervention in enumerate(interventions):
            if (i + 1) % 100 == 0:
                print(f"  Measured {i + 1}/{len(interventions)} degrees")
            degree = self.get_node_degree(intervention["node_id"], euclidean_threshold)
            degrees[intervention["node_id"]] = degree

        degree_values = list(degrees.values())
        cutoff = np.percentile(degree_values, hub_percentile)

        hubs = [i for i in interventions if degrees[i["node_id"]] >= cutoff]
        non_hubs = [i for i in interventions if degrees[i["node_id"]] < cutoff]

        print(f"\n  Hub cutoff: degree ≥{cutoff:.0f}")
        print(f"  Hubs: {len(hubs)} ({100 * len(hubs) / len(interventions):.1f}%)")
        print(
            f"  Non-hubs: {len(non_hubs)} ({100 * len(non_hubs) / len(interventions):.1f}%)"
        )
        print(f"  Degree range: {min(degree_values):.0f} - {max(degree_values):.0f}")

        return hubs, non_hubs, degrees

    def bidirectional_reachable(
        self, start_id, end_id, max_hops=6, euclidean_threshold=None
    ):
        forward_visited = {start_id: 0}
        forward_queue = deque([start_id])
        backward_visited = {end_id: 0}
        backward_queue = deque([end_id])

        max_depth_per_side = max_hops // 2 + 1
        forward_depth = 0
        backward_depth = 0

        while (forward_queue or backward_queue) and (
            forward_depth + backward_depth
        ) <= max_hops:
            if forward_queue and forward_depth < max_depth_per_side:
                level_size = len(forward_queue)
                for _ in range(level_size):
                    node = forward_queue.popleft()
                    if node in backward_visited:
                        return True
                    neighbors = self.get_neighbors_cached(node, euclidean_threshold)
                    for neighbor in neighbors:
                        if neighbor not in forward_visited:
                            forward_visited[neighbor] = forward_depth + 1
                            forward_queue.append(neighbor)
                forward_depth += 1

            if backward_queue and backward_depth < max_depth_per_side:
                level_size = len(backward_queue)
                for _ in range(level_size):
                    node = backward_queue.popleft()
                    if node in forward_visited:
                        return True
                    neighbors = self.get_neighbors_cached(node, euclidean_threshold)
                    for neighbor in neighbors:
                        if neighbor not in backward_visited:
                            backward_visited[neighbor] = backward_depth + 1
                            backward_queue.append(neighbor)
                backward_depth += 1

        return False

    def find_sample_nodes(self):
        query = "MATCH (n:Intervention) RETURN id(n), n.name LIMIT 500"
        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)
        interventions = (
            [
                {"node_id": int(row[0]), "name": row[1] if row[1] else f"I{row[0]}"}
                for row in result[1]
            ]
            if len(result) > 1
            else []
        )

        risk_keywords = [
            "existential risk",
            "misalignment",
            "deceptive alignment",
            "power-seeking",
            "x-risk",
            "catastrophic",
            "reward hacking",
            "mesa-optimization",
        ]
        risks = []
        for keyword in risk_keywords:
            query = f"""
            MATCH (n:Concept)
            WHERE toLower(n.name) CONTAINS '{keyword}' OR toLower(n.description) CONTAINS '{keyword}'
            RETURN id(n), n.name LIMIT 20
            """
            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)
            if len(result) > 1:
                for row in result[1]:
                    node_id = int(row[0])
                    if node_id not in [r["node_id"] for r in risks]:
                        risks.append(
                            {
                                "node_id": node_id,
                                "name": row[1] if row[1] else f"R{row[0]}",
                            }
                        )

        return interventions, risks

    def run_stratified_test(self, interventions, risks, threshold, target_pairs=30):
        """Single run on intervention list"""
        import random

        euclidean_threshold = self.euclidean_from_cosine(threshold)

        test_interventions = interventions[:]
        test_risks = risks[:]
        random.shuffle(test_interventions)
        random.shuffle(test_risks)

        connected = []
        tests = 0

        for i in test_interventions:
            if len(connected) >= target_pairs:
                break
            for r in test_risks:
                if self.bidirectional_reachable(
                    i["node_id"],
                    r["node_id"],
                    max_hops=6,
                    euclidean_threshold=euclidean_threshold,
                ):
                    connected.append((i["node_id"], r["node_id"]))
                    if len(connected) >= target_pairs:
                        break
                tests += 1
                if tests >= 500:  # Cap
                    break
            if len(connected) >= target_pairs or tests >= 500:
                break

        connectivity = len(connected) / tests if tests > 0 else 0
        return connected, connectivity, tests

    def run_test(self, thresholds=[0.8, 0.85, 0.9, 0.95], num_runs=5):
        print("=" * 80)
        print("PHASE 1: HUB-AWARE MULTI-THRESHOLD REACHABILITY")
        print("=" * 80)

        interventions, risks = self.find_sample_nodes()
        print(f"Sample: {len(interventions)} interventions, {len(risks)} risks")

        all_results = {}

        for threshold in thresholds:
            print(f"\n{'=' * 80}")
            print(f"THRESHOLD ≥{threshold}")
            print(f"{'=' * 80}")

            hubs, non_hubs, degrees = self.classify_hubs(interventions, threshold)

            # Test hubs
            print(f"\nHubs ({len(hubs)} interventions):")
            hub_connectivities = []
            for run in range(num_runs):
                self.neighbor_cache = {}
                _, conn, tests = self.run_stratified_test(
                    hubs, risks, threshold, target_pairs=30
                )
                hub_connectivities.append(conn)
                print(f"  Run {run + 1}: {conn * 100:.1f}% ({tests} tests)")

            # Test non-hubs
            print(f"\nNon-hubs ({len(non_hubs)} interventions):")
            nonhub_connectivities = []
            for run in range(num_runs):
                self.neighbor_cache = {}
                _, conn, tests = self.run_stratified_test(
                    non_hubs, risks, threshold, target_pairs=30
                )
                nonhub_connectivities.append(conn)
                print(f"  Run {run + 1}: {conn * 100:.1f}% ({tests} tests)")

            # Statistics
            hub_mean = np.mean(hub_connectivities)
            hub_std = np.std(hub_connectivities)
            nonhub_mean = np.mean(nonhub_connectivities)
            nonhub_std = np.std(nonhub_connectivities)
            hub_weight = len(hubs) / len(interventions)
            overall_mean = hub_mean * hub_weight + nonhub_mean * (1 - hub_weight)

            all_results[threshold] = {
                "hub_mean": hub_mean,
                "hub_std": hub_std,
                "nonhub_mean": nonhub_mean,
                "nonhub_std": nonhub_std,
                "hub_weight": hub_weight,
                "overall_mean": overall_mean,
            }

            print(f"\nHub: {hub_mean * 100:.1f}% ± {hub_std * 100:.1f}%")
            print(f"Non-hub: {nonhub_mean * 100:.1f}% ± {nonhub_std * 100:.1f}%")
            print(f"Overall: {overall_mean * 100:.1f}%")

        # Summary table
        print(f"\n{'=' * 80}")
        print("THRESHOLD COMPARISON")
        print(f"{'=' * 80}")
        print(
            f"\n{'Threshold':<12} {'Hub %':<12} {'Non-hub %':<12} {'Overall %':<12} {'Extrap (M)':<12}"
        )
        print("-" * 60)

        for threshold in thresholds:
            r = all_results[threshold]
            hub_pairs = (37000 * r["hub_weight"]) * 20000
            nonhub_pairs = (37000 * (1 - r["hub_weight"])) * 20000
            connected = (
                hub_pairs * r["hub_mean"] + nonhub_pairs * r["nonhub_mean"]
            ) / 1e6

            print(
                f"≥{threshold:<11} {r['hub_mean'] * 100:<11.1f} "
                f"{r['nonhub_mean'] * 100:<11.1f} {r['overall_mean'] * 100:<11.1f} {connected:<11.0f}"
            )

        # Save best threshold results
        import json

        with open("threshold_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n✓ Saved results to threshold_results.json")


def main():
    tester = HubAwareReachabilityTest()
    tester.client.ping()
    tester.run_test(thresholds=[0.8, 0.85, 0.9, 0.95], num_runs=5)


if __name__ == "__main__":
    main()
