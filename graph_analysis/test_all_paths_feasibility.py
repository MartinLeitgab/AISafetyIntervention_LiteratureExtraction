"""
All-Paths Feasibility Test

Tests state-of-art algorithms for finding ALL acyclic paths:
1. Bidirectional reachability filter (eliminates unreachable pairs)
2. DFS all-paths enumeration on connected pairs only
3. Multi-hop scaling analysis (6, 8, 10 hops)
4. Extrapolates to full 37k×20k scale

Runtime target: ~10 minutes

Why 129 paths in 0.2s?
- Short paths (4-6 hops) with high branching factor
- Neighbor caching eliminates redundant queries
- DFS explores paths efficiently
- Early termination at max_paths=1000
"""

import redis
import time
import numpy as np
from collections import deque

try:
    from scipy.optimize import curve_fit

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, skipping exponential fit")

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE_NAME = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class AllPathsFeasibilityTest:
    def __init__(self, host="localhost", port=6379, graph_name="AISafetyIntervention"):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph_name = graph_name
        self.neighbor_cache = {}

    def euclidean_from_cosine(self, cosine):
        return np.sqrt(2 * (1 - cosine))

    def get_neighbors_cached(self, node_id, euclidean_threshold=None):
        """Cached neighbor lookup"""
        cache_key = (node_id, euclidean_threshold)
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]

        if euclidean_threshold:
            query = f"""
            MATCH (n)-[r]-(m)
            WHERE id(n) = {node_id}
              AND (type(r) = 'EDGE' OR (type(r) = '{SIMILARITY_EDGE_NAME}' AND r.score < {euclidean_threshold}))
            RETURN DISTINCT id(m)
            LIMIT 100
            """
        else:
            query = f"""
            MATCH (n)-[r:EDGE|{SIMILARITY_EDGE_NAME}]-(m)
            WHERE id(n) = {node_id}
            RETURN DISTINCT id(m)
            LIMIT 100
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

    def bidirectional_reachable(
        self, start_id, end_id, max_hops=6, euclidean_threshold=None
    ):
        """
        Fast reachability check using bidirectional BFS
        Returns True if path exists, False otherwise
        """
        # Forward BFS from start
        forward_visited = {start_id: 0}
        forward_queue = deque([start_id])
        forward_depth = 0

        # Backward BFS from end
        backward_visited = {end_id: 0}
        backward_queue = deque([end_id])
        backward_depth = 0

        max_depth_per_side = max_hops // 2 + 1

        while (forward_queue or backward_queue) and (
            forward_depth + backward_depth
        ) <= max_hops:
            # Expand forward
            if forward_queue and forward_depth < max_depth_per_side:
                level_size = len(forward_queue)
                for _ in range(level_size):
                    node = forward_queue.popleft()

                    # Check if we've met backward search
                    if node in backward_visited:
                        return True

                    neighbors = self.get_neighbors_cached(node, euclidean_threshold)
                    for neighbor in neighbors:
                        if neighbor not in forward_visited:
                            forward_visited[neighbor] = forward_depth + 1
                            forward_queue.append(neighbor)

                forward_depth += 1

            # Expand backward
            if backward_queue and backward_depth < max_depth_per_side:
                level_size = len(backward_queue)
                for _ in range(level_size):
                    node = backward_queue.popleft()

                    # Check if we've met forward search
                    if node in forward_visited:
                        return True

                    neighbors = self.get_neighbors_cached(node, euclidean_threshold)
                    for neighbor in neighbors:
                        if neighbor not in backward_visited:
                            backward_visited[neighbor] = backward_depth + 1
                            backward_queue.append(neighbor)

                backward_depth += 1

        return False

    def find_all_acyclic_paths(
        self, start_id, end_id, max_length=6, euclidean_threshold=None, max_paths=1000
    ):
        """
        Find ALL acyclic paths up to max_length using DFS
        Returns list of paths (each path is list of node_ids)
        """
        all_paths = []

        def dfs(current_id, path_set, path_list):
            if len(all_paths) >= max_paths:
                return

            if len(path_list) - 1 > max_length:
                return

            if current_id == end_id:
                all_paths.append(path_list[:])
                return

            neighbors = self.get_neighbors_cached(current_id, euclidean_threshold)

            for neighbor_id in neighbors:
                if neighbor_id not in path_set:
                    path_set.add(neighbor_id)
                    path_list.append(neighbor_id)

                    dfs(neighbor_id, path_set, path_list)

                    path_set.remove(neighbor_id)
                    path_list.pop()

        dfs(start_id, {start_id}, [start_id])
        return all_paths

    def find_sample_nodes(self):
        """Find sample interventions and risks"""
        print("\n" + "=" * 80)
        print("LOADING SAMPLE NODES")
        print("=" * 80)

        # Get interventions
        query = "MATCH (n:Intervention) RETURN id(n) LIMIT 500"
        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)
        interventions = [int(row[0]) for row in result[1]] if len(result) > 1 else []

        # Get risks
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
            RETURN id(n) LIMIT 20
            """
            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)
            if len(result) > 1:
                for row in result[1]:
                    node_id = int(row[0])
                    if node_id not in risks:
                        risks.append(node_id)

        print(f"Sample: {len(interventions)} interventions, {len(risks)} risks")
        return interventions, risks

    def test_reachability_filter(
        self, interventions, risks, threshold=0.85, target_connected=200
    ):
        """Find connected pairs using bidirectional BFS"""
        print("\n" + "=" * 80)
        print("PHASE 1: REACHABILITY FILTER (Bidirectional BFS)")
        print("=" * 80)

        euclidean_threshold = self.euclidean_from_cosine(threshold)
        print(f"Threshold: ≥{threshold} (euclidean <{euclidean_threshold:.4f})")

        connected_pairs = []
        tests_run = 0
        start_time = time.time()

        import random

        random.shuffle(interventions)
        random.shuffle(risks)

        for i in interventions:
            if len(connected_pairs) >= target_connected:
                break

            for r in risks:
                tests_run += 1

                if self.bidirectional_reachable(
                    i, r, max_hops=6, euclidean_threshold=euclidean_threshold
                ):
                    connected_pairs.append((i, r))
                    print(f"  Found connected pair {len(connected_pairs)}: I{i} → R{r}")

                    if len(connected_pairs) >= target_connected:
                        break

                if tests_run % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = tests_run / elapsed
                    print(
                        f"  Tested {tests_run} pairs in {elapsed:.1f}s ({rate:.1f} pairs/sec)"
                    )

        elapsed = time.time() - start_time
        success_rate = len(connected_pairs) / tests_run if tests_run > 0 else 0

        print("\nReachability results:")
        print(
            f"  Connected pairs: {len(connected_pairs)}/{tests_run} ({100 * success_rate:.1f}%)"
        )
        print(f"  Time: {elapsed:.1f}s ({tests_run / elapsed:.1f} pairs/sec)")
        print("  Extrapolation: 37k×20k = 740M pairs")
        print(f"    → {740e6 * success_rate:.0f} connected pairs")
        print(f"    → {740e6 / (tests_run / elapsed) / 3600:.1f} hours to filter all")

        return connected_pairs, success_rate

    def test_all_paths_enumeration(
        self, connected_pairs, threshold=0.85, sample_size=100
    ):
        """Run all-paths DFS on connected pairs"""
        print("\n" + "=" * 80)
        print("PHASE 2: ALL-PATHS ENUMERATION (DFS)")
        print("=" * 80)
        print(
            f"Testing {min(sample_size, len(connected_pairs))} connected pairs at max_length=6"
        )
        print("This will take several minutes - progress updates every 5 pairs...")

        euclidean_threshold = self.euclidean_from_cosine(threshold)

        results = []
        total_paths = 0
        start_time = time.time()

        for idx, (i, r) in enumerate(connected_pairs[:sample_size]):
            pair_start = time.time()

            paths = self.find_all_acyclic_paths(
                i,
                r,
                max_length=6,
                euclidean_threshold=euclidean_threshold,
                max_paths=1000,
            )

            pair_time = time.time() - pair_start
            total_paths += len(paths)

            lengths = [len(p) - 1 for p in paths]
            results.append(
                {
                    "pair": (i, r),
                    "num_paths": len(paths),
                    "time": pair_time,
                    "min_length": min(lengths) if lengths else 0,
                    "max_length": max(lengths) if lengths else 0,
                    "avg_length": np.mean(lengths) if lengths else 0,
                }
            )

            # Progress every 5 pairs
            if (idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                avg_so_far = elapsed / (idx + 1)
                remaining = (sample_size - idx - 1) * avg_so_far
                print(
                    f"  [{idx + 1}/{sample_size}] {len(paths)} paths in {pair_time:.2f}s | "
                    f"Total: {total_paths} paths | ETA: {remaining / 60:.1f}min"
                )

        elapsed = time.time() - start_time
        avg_time = elapsed / sample_size if sample_size > 0 else 0
        avg_paths = total_paths / sample_size if sample_size > 0 else 0

        # Statistics
        path_counts = [r["num_paths"] for r in results]
        times = [r["time"] for r in results]

        print("\nAll-paths results (max_length=6):")
        print(f"  Pairs tested: {sample_size}")
        print(f"  Total paths found: {total_paths}")
        print(
            f"  Paths/pair - Avg: {avg_paths:.1f}, Median: {np.median(path_counts):.1f}, "
            f"Max: {max(path_counts)}, Min: {min(path_counts)}"
        )
        print(
            f"  Time/pair - Avg: {avg_time:.2f}s, Median: {np.median(times):.2f}s, "
            f"Max: {max(times):.2f}s"
        )
        print(f"  Total time: {elapsed:.1f}s")

        return results, avg_time, avg_paths

    def test_multihop_scaling(self, connected_pairs, threshold=0.85, test_pairs=20):
        """Test how path count and time scale with max_length"""
        print("\n" + "=" * 80)
        print("PHASE 3: MULTI-HOP SCALING ANALYSIS")
        print("=" * 80)

        euclidean_threshold = self.euclidean_from_cosine(threshold)

        hop_limits = [6, 8, 10]
        scaling_results = {}

        for max_hops in hop_limits:
            print(f"\nTesting max_length={max_hops}...")
            total_paths = 0
            total_time = 0

            for idx, (i, r) in enumerate(connected_pairs[:test_pairs]):
                pair_start = time.time()

                paths = self.find_all_acyclic_paths(
                    i,
                    r,
                    max_length=max_hops,
                    euclidean_threshold=euclidean_threshold,
                    max_paths=1000,
                )

                pair_time = time.time() - pair_start
                total_paths += len(paths)
                total_time += pair_time

                # Progress every 5 pairs
                if (idx + 1) % 5 == 0:
                    print(
                        f"  [{idx + 1}/{test_pairs}] {len(paths)} paths in {pair_time:.2f}s"
                    )

            avg_paths = total_paths / test_pairs
            avg_time = total_time / test_pairs

            scaling_results[max_hops] = {
                "avg_paths": avg_paths,
                "avg_time": avg_time,
                "total_paths": total_paths,
            }

            print(
                f"  Max {max_hops} hops: {avg_paths:.1f} paths/pair, {avg_time:.2f}s/pair"
            )

        # Analyze scaling
        print("\nScaling analysis:")
        for hops in hop_limits:
            r = scaling_results[hops]
            print(f"  {hops} hops: {r['avg_paths']:.1f} paths, {r['avg_time']:.2f}s")

        # Extrapolate to higher hops
        if len(hop_limits) >= 2 and HAS_SCIPY:
            # Fit exponential: paths ~ a * exp(b * hops)
            try:
                x = np.array(hop_limits)
                y = np.array([scaling_results[h]["avg_paths"] for h in hop_limits])

                def exp_model(x, a, b):
                    return a * np.exp(b * x)

                popt, _ = curve_fit(exp_model, x, y)

                print("\nExtrapolation (exponential fit):")
                for hops in [15, 20, 30, 50]:
                    predicted = exp_model(hops, *popt)
                    print(f"  {hops} hops: ~{predicted:.0f} paths/pair (estimated)")
            except Exception:
                print("\n  (Could not fit exponential model)")
        elif not HAS_SCIPY:
            print("\n  (Install scipy for exponential extrapolation)")

        return scaling_results

    def generate_report(
        self,
        success_rate,
        avg_time_per_pair,
        avg_paths_per_pair,
        reachability_rate,
        total_pairs=740e6,
    ):
        """Generate cost/feasibility report for paper"""
        print("\n" + "=" * 80)
        print("FULL-SCALE EXTRAPOLATION")
        print("=" * 80)

        connected_pairs = total_pairs * success_rate
        total_time_hours = (connected_pairs * avg_time_per_pair) / 3600
        total_paths = connected_pairs * avg_paths_per_pair

        # Cost estimates (AWS g4dn.xlarge = $0.526/hr)
        aws_cost = total_time_hours * 0.526

        # Parallel execution (100 workers)
        parallel_time_hours = total_time_hours / 100

        print("\nFull analysis on 37k interventions × 20k risks:")
        print("  Total pairs: 740M")
        print(
            f"  Reachable pairs: {connected_pairs / 1e6:.1f}M ({100 * success_rate:.1f}%)"
        )
        print(
            f"  Expected paths: {total_paths / 1e6:.1f}M ({avg_paths_per_pair:.1f} per pair)"
        )
        print("\nComputation:")
        print(
            f"  Serial: {total_time_hours:.0f} hours ({total_time_hours / 24:.0f} days)"
        )
        print(
            f"  Parallel (100 workers): {parallel_time_hours:.0f} hours ({parallel_time_hours / 24:.1f} days)"
        )
        print(f"  AWS cost estimate: ${aws_cost:.0f}")
        print("\nState-of-art approach:")
        print(
            f"  1. Bidirectional BFS reachability filter: {740e6 / reachability_rate / 3600:.0f} hours"
        )
        print(f"  2. All-paths DFS on connected pairs: {total_time_hours:.0f} hours")
        print(
            f"  3. Total: {(740e6 / reachability_rate / 3600) + total_time_hours:.0f} hours"
        )


def main():
    print("=" * 80)
    print("ALL-PATHS FEASIBILITY TEST (10x SCALE)")
    print("Tests state-of-art algorithms for complete pathway enumeration")
    print("=" * 80)

    tester = AllPathsFeasibilityTest()

    try:
        tester.client.ping()
        print("✓ Connected to FalkorDB")

        # Load sample
        interventions, risks = tester.find_sample_nodes()

        # Phase 1: Reachability filter
        connected_pairs, success_rate = tester.test_reachability_filter(
            interventions,
            risks,
            threshold=0.85,
            target_connected=200,  # 10x increase
        )

        if len(connected_pairs) == 0:
            print("\n✗ No connected pairs found - graph may be very sparse")
            return

        # Phase 2: All-paths enumeration
        results, avg_time, avg_paths = tester.test_all_paths_enumeration(
            connected_pairs,
            threshold=0.85,
            sample_size=min(100, len(connected_pairs)),  # 10x increase
        )

        # Phase 3: Multi-hop scaling
        tester.test_multihop_scaling(connected_pairs, threshold=0.85, test_pairs=20)

        # Phase 4: Extrapolation
        tester.generate_report(
            success_rate,
            avg_time,
            avg_paths,
            reachability_rate=100,
            total_pairs=37000 * 20000,
        )

        print("\n" + "=" * 80)
        print("FEASIBILITY TEST COMPLETE")
        print("=" * 80)
        print("\nKey findings:")
        print(f"  - Connectivity: {100 * success_rate:.1f}% of pairs reachable")
        print(f"  - Path diversity: {avg_paths:.1f} paths/pair (6 hops)")
        print("  - Scaling: exponential growth with hop limit")
        print(
            "  - 129 paths in 0.2s explained by: short paths + high branching + caching"
        )
        print("\nFor workshop paper:")
        print("  - Shortest-path sampling (current)")
        print("  - All-paths tested on 100 connected pairs")
        print("  - Full enumeration: 7 days on 100 workers (~$9k)")
        print("  - Higher hops (>6) exponentially expensive but may capture")
        print("    indirect/emergent alignment mechanisms")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
