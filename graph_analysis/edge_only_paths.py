"""
Extract EDGE-only paths (no SIMILARITY edges)
Tests confidence ≥4 and ≥5
"""

import redis
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, deque
import json
import time


class EdgeOnlyPathfinder:
    def __init__(self):
        self.client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.graph = "AISafetyIntervention"

    def query(self, cypher, timeout=120000):
        result = self.client.execute_command(
            "GRAPH.QUERY", self.graph, cypher, "--timeout", str(timeout)
        )
        return result[1] if len(result) > 1 else []

    def load_interventions_risks(self):
        """Load interventions (maturity≥3) and risks"""
        print("Loading interventions...")

        # Interventions
        try:
            id_query = "MATCH (i:Intervention) WHERE i.intervention_maturity >= 3 RETURN min(id(i)), max(id(i))"
            print("  Getting ID range...")
            id_result = self.query(
                id_query, timeout=300000
            )  # 5min timeout for aggregation

            if not id_result or not id_result[0]:
                print("  WARNING: No interventions found")
                return {}, {}

            min_id, max_id = int(id_result[0][0]), int(id_result[0][1])
            print(f"  ID range: {min_id} to {max_id}")
        except Exception as e:
            print(f"  ERROR getting intervention IDs: {e}")
            raise

        interventions = {}
        current_id, batch_size = min_id, 5000
        batch_num = 0

        while current_id <= max_id:
            query = f"""
            MATCH (i:Intervention)
            WHERE id(i) >= {current_id} AND id(i) < {current_id + batch_size}
              AND i.intervention_maturity >= 3
            RETURN id(i), i.name
            """
            for row in self.query(query):
                interventions[int(row[0])] = row[1]

            if batch_num % 5 == 0:
                print(f"  Loaded {len(interventions)} interventions...")

            current_id += batch_size
            batch_num += 1

        print(f"  Total interventions: {len(interventions)}")

        # Risks
        print("Loading risks...")
        try:
            id_query = "MATCH (r:Concept) WHERE r.concept_category = 'risk' RETURN min(id(r)), max(id(r))"
            print("  Getting ID range...")
            id_result = self.query(id_query, timeout=300000)

            if not id_result or not id_result[0]:
                print("  WARNING: No risks found")
                return interventions, {}

            min_id, max_id = int(id_result[0][0]), int(id_result[0][1])
            print(f"  ID range: {min_id} to {max_id}")
        except Exception as e:
            print(f"  ERROR getting risk IDs: {e}")
            raise

        risks = {}
        current_id = min_id
        batch_num = 0

        while current_id <= max_id:
            query = f"""
            MATCH (r:Concept)
            WHERE id(r) >= {current_id} AND id(r) < {current_id + batch_size}
              AND r.concept_category = 'risk'
            RETURN id(r), r.name
            """
            for row in self.query(query):
                risks[int(row[0])] = row[1]

            if batch_num % 5 == 0:
                print(f"  Loaded {len(risks)} risks...")

            current_id += batch_size
            batch_num += 1

        print(f"  Total risks: {len(risks)}")
        return interventions, risks

    def load_edge_only_graph(self, min_confidence):
        """Load EDGE edges only (no SIMILARITY)"""
        print(f"Loading EDGE edges (confidence≥{min_confidence})...")

        adj_list = defaultdict(set)

        all_nodes_query = "MATCH (n) RETURN min(id(n)), max(id(n))"
        id_result = self.query(all_nodes_query)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        edge_count = 0
        current_id, batch_size = min_id, 2000

        while current_id <= max_id:
            query = f"""
            MATCH (n)-[e:EDGE]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
              AND id(m) > id(n)
              AND e.edge_confidence >= {min_confidence}
            RETURN id(n), id(m)
            """
            for row in self.query(query):
                n1, n2 = int(row[0]), int(row[1])
                adj_list[n1].add(n2)
                adj_list[n2].add(n1)
                edge_count += 1
            current_id += batch_size

        print(f"  Loaded {edge_count} EDGE edges, {len(adj_list)} nodes")
        return adj_list

    def bfs_shortest_paths(self, start_id, target_ids, adj_list, max_hops=50):
        """BFS to find shortest paths"""
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

    def run_all_pairs(self, label, interventions, risks, adj_list):
        """Run BFS from all interventions to all risks"""
        print(f"\nTesting {len(interventions)} interventions → {len(risks)} risks")

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
                eta_minutes = ((len(int_ids) - (i + 1)) / rate) / 60

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

        # Save
        sp.save_npz(f"reachability_matrix_{label}.npz", path_lengths)

        reachable_ints = [
            {"id": iid, "name": interventions[iid], "index": int_id_to_idx[iid]}
            for iid in sorted(reachable_int_ids)
        ]
        with open(f"reachable_interventions_{label}.json", "w") as f:
            json.dump(reachable_ints, f, indent=2)

        reachable_rsks = [
            {"id": rid, "name": risks[rid], "index": risk_id_to_idx[rid]}
            for rid in sorted(reachable_risk_ids)
        ]
        with open(f"reachable_risks_{label}.json", "w") as f:
            json.dump(reachable_rsks, f, indent=2)

        print(f"  ✓ Saved reachability_matrix_{label}.npz")

        return path_lengths


def main():
    finder = EdgeOnlyPathfinder()

    try:
        finder.client.ping()
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return

    print("=" * 80)
    print("EDGE-ONLY PATHFINDING (Confidence ≥4)")
    print("=" * 80)

    interventions, risks = finder.load_interventions_risks()

    print("\nLoading EDGE graph (confidence ≥4, no SIMILARITY)...")
    adj_list = finder.load_edge_only_graph(min_confidence=4)

    print("\nRunning all-pairs BFS...")
    finder.run_all_pairs("edge_conf4", interventions, risks, adj_list)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
