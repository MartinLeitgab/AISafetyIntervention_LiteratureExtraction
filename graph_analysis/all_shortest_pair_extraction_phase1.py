"""
Phase 1: All-Pairs Shortest Path + Reachable Subset Extraction
Tests all intervention→risk pairs with confidence≥4 filter across thresholds.
Extracts reachable nodes and computes degree distributions on subset.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import redis
import numpy as np
import json
from collections import Counter, deque
import scipy.sparse as sp

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class PathfindingAnalyzer:
    def __init__(self, host="localhost", port=6379, graph="AISafetyIntervention"):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph = graph
        # Bug fix 2026-04-30 (CF-5): bump RESULTSET_SIZE to 10M so queries do
        # not silently truncate at the default 10k row limit.
        try:
            self.client.execute_command(
                "GRAPH.CONFIG", "SET", "RESULTSET_SIZE", "10000000"
            )
        except Exception as e:
            print(f"WARN: could not bump RESULTSET_SIZE: {e}")

    def euclidean_from_cosine(self, cosine):
        return np.sqrt(2 * (1 - cosine))

    def query(self, cypher, timeout=60000):
        result = self.client.execute_command(
            "GRAPH.QUERY", self.graph, cypher, "--timeout", str(timeout)
        )
        return result[1] if len(result) > 1 else []

    def get_all_interventions(self):
        """Get all interventions with maturity>=3, batched by id-range.

        Bug fix 2026-04-30 (CF-5): the original single-shot query was silently
        truncated at 10,000 rows by FalkorDB's default RESULTSET_SIZE limit.
        With ~22k mature interventions in the corpus, this lost up to 12k.
        See `graph_analysis/phase2_results/rev8_active_state.md` Bug Audit.
        """
        # Get id range
        id_result = self.query(
            "MATCH (i:Intervention) WHERE i.intervention_maturity >= 3 "
            "RETURN min(id(i)), max(id(i))"
        )
        if not id_result or not id_result[0]:
            return {}
        min_id = int(id_result[0][0])
        max_id = int(id_result[0][1])

        interventions = {}
        cur = min_id
        batch_size = 5000
        while cur <= max_id:
            q = (
                f"MATCH (i:Intervention) "
                f"WHERE id(i) >= {cur} AND id(i) < {cur + batch_size} "
                f"AND i.intervention_maturity >= 3 "
                f"RETURN id(i), i.name"
            )
            for row in self.query(q):
                interventions[int(row[0])] = row[1]
            cur += batch_size
        return interventions

    def get_all_risks(self):
        """Get all risk concepts using ID-based batching"""
        # Get ID range
        id_query = """
        MATCH (r:Concept)
        WHERE r.concept_category = 'risk'
        RETURN min(id(r)), max(id(r))
        """
        id_result = self.query(id_query)

        if not id_result or not id_result[0]:
            return {}

        min_id = int(id_result[0][0])
        max_id = int(id_result[0][1])

        risks = {}
        current_id = min_id
        batch_size = 5000

        while current_id <= max_id:
            query = f"""
            MATCH (r:Concept)
            WHERE id(r) >= {current_id} AND id(r) < {current_id + batch_size}
              AND r.concept_category = 'risk'
            RETURN id(r), r.name
            """
            batch = self.query(query)
            for row in batch:
                risks[int(row[0])] = row[1]
            current_id += batch_size

        return risks

    def get_neighbors(self, node_id, euclidean_threshold):
        """Get neighbors via confidence≥4 edges ONLY"""
        query = f"""
        MATCH (n)-[e]-(m)
        WHERE id(n) = {node_id}
          AND ((type(e) = 'EDGE' AND e.edge_confidence >= 4)
            OR (type(e) = '{SIMILARITY_EDGE}' AND e.score < {euclidean_threshold}))
        RETURN DISTINCT id(m)
        """
        rows = self.query(query)
        return [int(row[0]) for row in rows]

    def bfs_shortest_paths(
        self, start_id, target_ids, euclidean_threshold, max_hops=50
    ):
        """BFS to find shortest paths to all targets"""
        visited = {start_id: 0}
        queue = deque([start_id])
        found_targets = {}

        while queue and len(found_targets) < len(target_ids):
            node_id = queue.popleft()
            current_dist = visited[node_id]

            if current_dist >= max_hops:
                break

            neighbors = self.get_neighbors(node_id, euclidean_threshold)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited[neighbor_id] = current_dist + 1
                    queue.append(neighbor_id)

                    if neighbor_id in target_ids:
                        found_targets[neighbor_id] = current_dist + 1

        return found_targets

    def run_all_pairs(self, threshold, interventions, risks):
        """Run BFS from all interventions to all risks"""
        import time

        euclidean = self.euclidean_from_cosine(threshold)
        risk_ids = set(risks.keys())

        print(f"\n{'=' * 80}")
        print(f"THRESHOLD ≥{threshold}")
        print(f"{'=' * 80}")
        print(
            f"Testing {len(interventions)} interventions → {len(risks)} risks = {len(interventions) * len(risks):,} pairs"
        )
        print("Quality filters: maturity≥3, edge confidence≥4, ≤50 hops")

        # Sparse matrix to store path lengths
        int_ids = sorted(interventions.keys())
        risk_ids_sorted = sorted(risks.keys())

        int_id_to_idx = {iid: idx for idx, iid in enumerate(int_ids)}
        risk_id_to_idx = {rid: idx for idx, rid in enumerate(risk_ids_sorted)}

        path_lengths = sp.lil_matrix(
            (len(int_ids), len(risk_ids_sorted)), dtype=np.int16
        )

        reachable_int_ids = set()
        reachable_risk_ids = set()

        start_time = time.time()
        last_report_time = start_time

        for i, int_id in enumerate(int_ids):
            current_time = time.time()

            # Report every 50 interventions or every 30 seconds
            if (i + 1) % 50 == 0 or (current_time - last_report_time) > 30:
                elapsed = current_time - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = len(int_ids) - (i + 1)
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60

                print(
                    f"  Progress: {i + 1}/{len(int_ids)} interventions ({100 * (i + 1) / len(int_ids):.1f}%)"
                )
                print(
                    f"    Rate: {rate:.1f} interventions/sec, ETA: {eta_minutes:.1f} min"
                )
                print(
                    f"    Reachable so far: {len(reachable_int_ids)} interventions, {len(reachable_risk_ids)} risks"
                )

                last_report_time = current_time

            paths = self.bfs_shortest_paths(int_id, risk_ids, euclidean, max_hops=50)

            if paths:
                reachable_int_ids.add(int_id)
                for risk_id, length in paths.items():
                    reachable_risk_ids.add(risk_id)
                    i_idx = int_id_to_idx[int_id]
                    r_idx = risk_id_to_idx[risk_id]
                    path_lengths[i_idx, r_idx] = length

        # Convert to CSR for efficient storage
        path_lengths = path_lengths.tocsr()

        # Statistics
        total_time = time.time() - start_time
        total_pairs = len(int_ids) * len(risk_ids_sorted)
        connected_pairs = path_lengths.nnz

        print(f"\n  Results (completed in {total_time / 60:.1f} min):")
        print(f"    Total pairs tested: {total_pairs:,}")
        print(
            f"    Connected pairs: {connected_pairs:,} ({100 * connected_pairs / total_pairs:.2f}%)"
        )
        print(
            f"    Reachable interventions: {len(reachable_int_ids)}/{len(int_ids)} ({100 * len(reachable_int_ids) / len(int_ids):.1f}%)"
        )
        print(
            f"    Reachable risks: {len(reachable_risk_ids)}/{len(risks)} ({100 * len(reachable_risk_ids) / len(risks):.1f}%)"
        )

        # Path length distribution
        if connected_pairs > 0:
            lengths = path_lengths.data
            print(
                f"    Path length: min={np.min(lengths)}, max={np.max(lengths)}, median={np.median(lengths):.1f}, mean={np.mean(lengths):.1f}"
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
        int_id_to_idx,
        risk_id_to_idx,
        reachable_int_ids,
        reachable_risk_ids,
        interventions,
        risks,
    ):
        """Save reachability matrix and reachable node lists"""
        # Save sparse matrix
        sp.save_npz(f"reachability_matrix_{threshold}.npz", path_lengths)

        # Save reachable interventions with names
        reachable_ints = [
            {"id": iid, "name": interventions[iid], "index": int_id_to_idx[iid]}
            for iid in sorted(reachable_int_ids)
        ]
        with open(f"reachable_interventions_{threshold}.json", "w") as f:
            json.dump(reachable_ints, f, indent=2)

        # Save reachable risks with names
        reachable_rsks = [
            {"id": rid, "name": risks[rid], "index": risk_id_to_idx[rid]}
            for rid in sorted(reachable_risk_ids)
        ]
        with open(f"reachable_risks_{threshold}.json", "w") as f:
            json.dump(reachable_rsks, f, indent=2)

        print(f"\n  ✓ Saved reachability_matrix_{threshold}.npz")
        print(f"  ✓ Saved reachable_interventions_{threshold}.json")
        print(f"  ✓ Saved reachable_risks_{threshold}.json")

    def compute_reachable_degrees(
        self, threshold, reachable_int_ids, reachable_risk_ids
    ):
        """Compute degree distributions for reachable nodes only"""
        euclidean = self.euclidean_from_cosine(threshold)

        print("\n  Computing degrees for reachable subset...")

        # Intervention degrees
        int_degrees = []
        for int_id in sorted(reachable_int_ids):
            query = f"""
            MATCH (i:Intervention)-[e]-(m:Intervention)
            WHERE id(i) = {int_id}
              AND m.intervention_maturity >= 3
              AND ((type(e) = 'EDGE' AND e.edge_confidence >= 4)
                OR (type(e) = '{SIMILARITY_EDGE}' AND e.score < {euclidean}))
            RETURN count(DISTINCT m)
            """
            rows = self.query(query)
            int_degrees.append(int(rows[0][0]) if rows else 0)

        # Risk degrees
        risk_degrees = []
        for i, risk_id in enumerate(sorted(reachable_risk_ids)):
            if (i + 1) % 500 == 0:
                print(f"    Computing risk degrees: {i + 1}/{len(reachable_risk_ids)}")
            query = f"""
            MATCH (r:Concept)-[e]-(m:Concept)
            WHERE id(r) = {risk_id}
              AND m.concept_category = 'risk'
              AND ((type(e) = 'EDGE' AND e.edge_confidence >= 4)
                OR (type(e) = '{SIMILARITY_EDGE}' AND e.score < {euclidean}))
            RETURN count(DISTINCT m)
            """
            rows = self.query(query)
            risk_degrees.append(int(rows[0][0]) if rows else 0)

        return int_degrees, risk_degrees

    def plot_reachable_degrees(self, threshold, int_degrees, risk_degrees):
        """Plot degree distributions for reachable nodes"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Interventions log-log
        if int_degrees and max(int_degrees) > 0:
            degree_counts = Counter(int_degrees)
            unique_degrees = sorted([d for d in degree_counts.keys() if d > 0])
            frequencies = [degree_counts[d] for d in unique_degrees]

            axes[0, 0].scatter(
                unique_degrees,
                frequencies,
                alpha=0.6,
                s=60,
                color="#3498DB",
                marker="o",
                edgecolors="black",
                linewidth=0.5,
            )
            axes[0, 0].set_xscale("log")
            axes[0, 0].set_yscale("log")
            axes[0, 0].set_xlabel("Degree (log scale)", fontsize=12, fontweight="bold")
            axes[0, 0].set_ylabel(
                "Frequency (log scale)", fontsize=12, fontweight="bold"
            )
            axes[0, 0].set_title(
                f"Reachable Interventions (≥{threshold})\n{len(int_degrees)} nodes",
                fontsize=14,
                fontweight="bold",
            )
            axes[0, 0].grid(True, alpha=0.3, which="both")

        # Interventions CCDF
        if int_degrees:
            sorted_degrees = np.sort([d for d in int_degrees if d > 0])
            if len(sorted_degrees) > 0:
                ccdf = 1 - np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
                axes[0, 1].scatter(
                    sorted_degrees,
                    ccdf,
                    alpha=0.6,
                    s=40,
                    color="#3498DB",
                    marker="o",
                    edgecolors="black",
                    linewidth=0.5,
                )
                axes[0, 1].set_xscale("log")
                axes[0, 1].set_yscale("log")
                axes[0, 1].set_xlabel(
                    "Degree (log scale)", fontsize=12, fontweight="bold"
                )
                axes[0, 1].set_ylabel("P(Degree ≥ k)", fontsize=12, fontweight="bold")
                axes[0, 1].set_title(
                    "Interventions CCDF", fontsize=14, fontweight="bold"
                )
                axes[0, 1].grid(True, alpha=0.3, which="both")

        # Risks log-log
        if risk_degrees and max(risk_degrees) > 0:
            degree_counts = Counter(risk_degrees)
            unique_degrees = sorted([d for d in degree_counts.keys() if d > 0])
            frequencies = [degree_counts[d] for d in unique_degrees]

            axes[1, 0].scatter(
                unique_degrees,
                frequencies,
                alpha=0.6,
                s=60,
                color="#E74C3C",
                marker="s",
                edgecolors="black",
                linewidth=0.5,
            )
            axes[1, 0].set_xscale("log")
            axes[1, 0].set_yscale("log")
            axes[1, 0].set_xlabel("Degree (log scale)", fontsize=12, fontweight="bold")
            axes[1, 0].set_ylabel(
                "Frequency (log scale)", fontsize=12, fontweight="bold"
            )
            axes[1, 0].set_title(
                f"Reachable Risks (≥{threshold})\n{len(risk_degrees)} nodes",
                fontsize=14,
                fontweight="bold",
            )
            axes[1, 0].grid(True, alpha=0.3, which="both")

        # Risks CCDF
        if risk_degrees:
            sorted_degrees = np.sort([d for d in risk_degrees if d > 0])
            if len(sorted_degrees) > 0:
                ccdf = 1 - np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
                axes[1, 1].scatter(
                    sorted_degrees,
                    ccdf,
                    alpha=0.6,
                    s=40,
                    color="#E74C3C",
                    marker="s",
                    edgecolors="black",
                    linewidth=0.5,
                )
                axes[1, 1].set_xscale("log")
                axes[1, 1].set_yscale("log")
                axes[1, 1].set_xlabel(
                    "Degree (log scale)", fontsize=12, fontweight="bold"
                )
                axes[1, 1].set_ylabel("P(Degree ≥ k)", fontsize=12, fontweight="bold")
                axes[1, 1].set_title("Risks CCDF", fontsize=14, fontweight="bold")
                axes[1, 1].grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(
            f"reachable_degree_distributions_{threshold}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  ✓ Saved reachable_degree_distributions_{threshold}.png")
        plt.close()


def main():
    analyzer = PathfindingAnalyzer()
    analyzer.client.ping()

    print("=" * 80)
    print("PHASE 1: ALL-PAIRS SHORTEST PATH + REACHABLE SUBSET")
    print("=" * 80)

    # Load all interventions and risks
    print("\nLoading interventions and risks...")
    interventions = analyzer.get_all_interventions()
    risks = analyzer.get_all_risks()
    print(f"  Interventions (maturity≥3): {len(interventions)}")
    print(f"  Risks: {len(risks)}")

    thresholds = [0.8, 0.85, 0.9, 0.95]

    for threshold in thresholds:
        # Run all-pairs BFS
        path_lengths, int_idx, risk_idx, reach_ints, reach_risks = (
            analyzer.run_all_pairs(threshold, interventions, risks)
        )

        # Save results
        analyzer.save_results(
            threshold,
            path_lengths,
            int_idx,
            risk_idx,
            reach_ints,
            reach_risks,
            interventions,
            risks,
        )

        # Compute degrees for reachable subset
        int_degrees, risk_degrees = analyzer.compute_reachable_degrees(
            threshold, reach_ints, reach_risks
        )

        # Plot
        analyzer.plot_reachable_degrees(threshold, int_degrees, risk_degrees)

        print("\n  Degree statistics (reachable subset):")
        if int_degrees:
            print(
                f"    Interventions: mean={np.mean(int_degrees):.1f}, max={max(int_degrees)}"
            )
        if risk_degrees:
            print(
                f"    Risks: mean={np.mean(risk_degrees):.1f}, max={max(risk_degrees)}"
            )

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
