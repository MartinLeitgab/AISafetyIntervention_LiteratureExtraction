"""
Phase 1 (Optimized): All-Pairs Shortest Path with Pre-loaded Graph
Loads entire filtered graph into memory once, then runs fast in-memory BFS.
"""

import matplotlib

matplotlib.use("Agg")

import redis
import numpy as np
import json
from collections import deque, defaultdict
import scipy.sparse as sp
import time

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class FastPathfinder:
    def __init__(self, host="localhost", port=6379, graph="AISafetyIntervention"):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph = graph

    def euclidean_from_cosine(self, cosine):
        return np.sqrt(2 * (1 - cosine))

    def query(self, cypher, timeout=120000):
        result = self.client.execute_command(
            "GRAPH.QUERY", self.graph, cypher, "--timeout", str(timeout)
        )
        return result[1] if len(result) > 1 else []

    def load_interventions_and_risks(self):
        """Load all interventions (maturity>=3) and risks"""
        print("\nLoading interventions (maturity>=3)...")

        # Get ID range for interventions
        id_query = """
        MATCH (i:Intervention)
        WHERE i.intervention_maturity >= 3
        RETURN min(id(i)), max(id(i))
        """
        id_result = self.query(id_query)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        interventions = {}
        current_id = min_id
        batch_size = 5000

        while current_id <= max_id:
            query = f"""
            MATCH (i:Intervention)
            WHERE id(i) >= {current_id} AND id(i) < {current_id + batch_size}
              AND i.intervention_maturity >= 3
            RETURN id(i), i.name
            """
            for row in self.query(query):
                interventions[int(row[0])] = row[1]
            current_id += batch_size

        print(f"  Loaded {len(interventions)} interventions")

        # Load risks
        print("Loading risks...")
        id_query = """
        MATCH (r:Concept)
        WHERE r.concept_category = 'risk'
        RETURN min(id(r)), max(id(r))
        """
        id_result = self.query(id_query)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        risks = {}
        current_id = min_id

        while current_id <= max_id:
            query = f"""
            MATCH (r:Concept)
            WHERE id(r) >= {current_id} AND id(r) < {current_id + batch_size}
              AND r.concept_category = 'risk'
            RETURN id(r), r.name
            """
            for row in self.query(query):
                risks[int(row[0])] = row[1]
            current_id += batch_size

        print(f"  Loaded {len(risks)} risks")
        return interventions, risks

    def load_graph_edges(self, threshold):
        """Load all filtered edges into memory adjacency list"""
        euclidean = self.euclidean_from_cosine(threshold)

        print(f"\nLoading graph edges (confidence>=4, similarity<{euclidean:.4f})...")
        adj_list = defaultdict(set)

        # Load EDGE edges with confidence>=4 using ID-based batching
        print("  Loading EDGE edges...")

        # Get node ID range
        all_nodes_query = "MATCH (n) RETURN min(id(n)), max(id(n))"
        id_result = self.query(all_nodes_query)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        edge_count = 0
        current_id = min_id
        batch_size = 2000
        batch_num = 0

        while current_id <= max_id:
            edge_query = f"""
            MATCH (n)-[e:EDGE]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
              AND id(m) > id(n)
              AND e.edge_confidence >= 4
            RETURN id(n), id(m)
            """
            batch = self.query(edge_query)
            for row in batch:
                n1, n2 = int(row[0]), int(row[1])
                adj_list[n1].add(n2)
                adj_list[n2].add(n1)
                edge_count += 1

            if batch_num % 20 == 0 and edge_count > 0:
                print(f"    Batch {batch_num}: {edge_count} EDGE edges")

            current_id += batch_size
            batch_num += 1

        print(f"    Loaded {edge_count} EDGE edges")

        # Load SIMILARITY edges below threshold (ALL node type combinations)
        print("  Loading SIMILARITY edges (all node types)...")

        sim_count = 0
        current_id = min_id
        batch_num = 0

        while current_id <= max_id:
            sim_query = f"""
            MATCH (n)-[s:{SIMILARITY_EDGE}]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
              AND id(m) > id(n)
              AND s.score < {euclidean}
            RETURN id(n), id(m)
            """
            batch = self.query(sim_query)
            for row in batch:
                n1, n2 = int(row[0]), int(row[1])
                adj_list[n1].add(n2)
                adj_list[n2].add(n1)
                sim_count += 1

            if batch_num % 20 == 0 and sim_count > 0:
                print(f"    Batch {batch_num}: {sim_count} SIMILARITY edges")

            current_id += batch_size
            batch_num += 1

        print(f"    Loaded {sim_count} SIMILARITY edges (all types)")
        print(f"  Total nodes in graph: {len(adj_list)}")
        print("  (Only nodes with ≥1 filtered edge included)")

        return adj_list

    def bfs_shortest_paths(self, start_id, target_ids, adj_list, max_hops=50):
        """Fast in-memory BFS"""
        visited = {start_id: 0}
        queue = deque([start_id])
        found_targets = {}
        target_set = set(target_ids)

        while queue and len(found_targets) < len(target_ids):
            node_id = queue.popleft()
            current_dist = visited[node_id]

            if current_dist >= max_hops:
                break

            for neighbor_id in adj_list.get(node_id, []):
                if neighbor_id not in visited:
                    visited[neighbor_id] = current_dist + 1
                    queue.append(neighbor_id)

                    if neighbor_id in target_set:
                        found_targets[neighbor_id] = current_dist + 1

        return found_targets

    def run_all_pairs(self, threshold, interventions, risks, adj_list):
        """Run in-memory BFS from all interventions to all risks"""
        print(f"\n{'=' * 80}")
        print(f"THRESHOLD ≥{threshold}")
        print(f"{'=' * 80}")
        print(
            f"Testing {len(interventions)} interventions → {len(risks)} risks = {len(interventions) * len(risks):,} pairs"
        )

        int_ids = sorted(interventions.keys())
        risk_ids = sorted(risks.keys())
        risk_id_set = set(risk_ids)

        int_id_to_idx = {iid: idx for idx, iid in enumerate(int_ids)}
        risk_id_to_idx = {rid: idx for idx, rid in enumerate(risk_ids)}

        path_lengths = sp.lil_matrix((len(int_ids), len(risk_ids)), dtype=np.int16)

        reachable_int_ids = set()
        reachable_risk_ids = set()

        start_time = time.time()

        for i, int_id in enumerate(int_ids):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = len(int_ids) - (i + 1)
                eta_minutes = (remaining / rate) / 60

                print(
                    f"  Progress: {i + 1}/{len(int_ids)} ({100 * (i + 1) / len(int_ids):.1f}%) | "
                    f"Rate: {rate:.1f} int/sec | ETA: {eta_minutes:.1f} min | "
                    f"Reachable: {len(reachable_int_ids)} int, {len(reachable_risk_ids)} risks"
                )

            paths = self.bfs_shortest_paths(int_id, risk_id_set, adj_list, max_hops=50)

            if paths:
                reachable_int_ids.add(int_id)
                for risk_id, length in paths.items():
                    reachable_risk_ids.add(risk_id)
                    i_idx = int_id_to_idx[int_id]
                    r_idx = risk_id_to_idx[risk_id]
                    path_lengths[i_idx, r_idx] = length

        path_lengths = path_lengths.tocsr()

        total_time = time.time() - start_time
        total_pairs = len(int_ids) * len(risk_ids)
        connected_pairs = path_lengths.nnz

        print(f"\n  Results (completed in {total_time / 60:.1f} min):")
        print(f"    Total pairs: {total_pairs:,}")
        print(
            f"    Connected: {connected_pairs:,} ({100 * connected_pairs / total_pairs:.2f}%)"
        )
        print(
            f"    Reachable interventions: {len(reachable_int_ids)}/{len(int_ids)} ({100 * len(reachable_int_ids) / len(int_ids):.1f}%)"
        )
        print(
            f"    Reachable risks: {len(reachable_risk_ids)}/{len(risks)} ({100 * len(reachable_risk_ids) / len(risks):.1f}%)"
        )

        if connected_pairs > 0:
            lengths = path_lengths.data
            print(
                f"    Path length: min={np.min(lengths)}, median={np.median(lengths):.1f}, max={np.max(lengths)}"
            )

        return (
            path_lengths,
            int_id_to_idx,
            risk_id_to_idx,
            reachable_int_ids,
            reachable_risk_ids,
        )

    def save_results(
        self,
        threshold,
        path_lengths,
        int_idx,
        risk_idx,
        reach_ints,
        reach_risks,
        interventions,
        risks,
    ):
        """Save all results to disk"""
        sp.save_npz(f"reachability_matrix_{threshold}.npz", path_lengths)

        reachable_ints = [
            {"id": iid, "name": interventions[iid], "index": int_idx[iid]}
            for iid in sorted(reach_ints)
        ]
        with open(f"reachable_interventions_{threshold}.json", "w") as f:
            json.dump(reachable_ints, f, indent=2)

        reachable_rsks = [
            {"id": rid, "name": risks[rid], "index": risk_idx[rid]}
            for rid in sorted(reach_risks)
        ]
        with open(f"reachable_risks_{threshold}.json", "w") as f:
            json.dump(reachable_rsks, f, indent=2)

        print(f"\n  ✓ Saved results for threshold {threshold}")


def main():
    finder = FastPathfinder()
    finder.client.ping()

    print("=" * 80)
    print("PHASE 1 (OPTIMIZED): ALL-PAIRS SHORTEST PATH")
    print("=" * 80)

    interventions, risks = finder.load_interventions_and_risks()

    for threshold in [0.8, 0.85, 0.9, 0.95]:
        adj_list = finder.load_graph_edges(threshold)

        path_lengths, int_idx, risk_idx, reach_ints, reach_risks = finder.run_all_pairs(
            threshold, interventions, risks, adj_list
        )

        finder.save_results(
            threshold,
            path_lengths,
            int_idx,
            risk_idx,
            reach_ints,
            reach_risks,
            interventions,
            risks,
        )

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
