"""
Analyze fraction of low-confidence EDGE edges per path
Shows how many paths excluded by each confidence cutoff
"""

import redis
import numpy as np
from collections import defaultdict, deque
import json

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class PathConfidenceAnalyzer:
    def __init__(self):
        self.client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.graph = "AISafetyIntervention"

    def query(self, cypher, timeout=120000):
        result = self.client.execute_command(
            "GRAPH.QUERY", self.graph, cypher, "--timeout", str(timeout)
        )
        return result[1] if len(result) > 1 else []

    def euclidean_from_cosine(self, cosine):
        return np.sqrt(2 * (1 - cosine))

    def load_adjacency_with_metadata(self, threshold):
        """Load graph with edge types and confidences"""
        euclidean = self.euclidean_from_cosine(threshold)
        print(f"Loading graph with metadata (threshold {threshold})...")

        # Structure: adj[node_id] = {neighbor_id: {'type': 'EDGE'/'SIM', 'confidence': int}}
        adj = defaultdict(dict)

        all_nodes_query = "MATCH (n) RETURN min(id(n)), max(id(n))"
        id_result = self.query(all_nodes_query)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        # EDGE edges
        current_id, batch_size = min_id, 2000
        edge_count = 0

        while current_id <= max_id:
            query = f"""
            MATCH (n)-[e:EDGE]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
              AND e.edge_confidence >= 4
            RETURN id(n), id(m), e.edge_confidence
            """
            for row in self.query(query):
                n1, n2, conf = int(row[0]), int(row[1]), int(row[2])
                adj[n1][n2] = {"type": "EDGE", "confidence": conf}
                adj[n2][n1] = {"type": "EDGE", "confidence": conf}
                edge_count += 1
            current_id += batch_size

        print(f"  Loaded {edge_count} EDGE edges")

        # SIMILARITY edges
        current_id = min_id
        sim_count = 0

        while current_id <= max_id:
            query = f"""
            MATCH (n)-[s:{SIMILARITY_EDGE}]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
              AND id(m) > id(n)
              AND s.score < {euclidean}
            RETURN id(n), id(m)
            """
            for row in self.query(query):
                n1, n2 = int(row[0]), int(row[1])
                adj[n1][n2] = {"type": "SIM", "confidence": None}
                adj[n2][n1] = {"type": "SIM", "confidence": None}
                sim_count += 1
            current_id += batch_size

        print(f"  Loaded {sim_count} SIMILARITY edges")
        return adj

    def bfs_reconstruct_path(self, start_id, target_id, adj):
        """BFS that returns actual path"""
        visited = {start_id: None}
        queue = deque([start_id])

        while queue:
            node_id = queue.popleft()

            if node_id == target_id:
                # Reconstruct path
                path = []
                current = target_id
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return list(reversed(path))

            for neighbor_id in adj.get(node_id, {}):
                if neighbor_id not in visited:
                    visited[neighbor_id] = node_id
                    queue.append(neighbor_id)

        return None

    def analyze_path_confidences(self, path, adj):
        """Analyze EDGE confidences in path"""
        edge_count = 0
        edge_confidences = []
        sim_count = 0

        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            edge_meta = adj[n1][n2]

            if edge_meta["type"] == "EDGE":
                edge_count += 1
                edge_confidences.append(edge_meta["confidence"])
            else:
                sim_count += 1

        if edge_count == 0:
            return None  # No EDGE edges

        # Calculate fractions below each threshold
        fractions = {}
        for cutoff in [1, 2, 3, 4, 5]:
            below = sum(1 for c in edge_confidences if c < cutoff)
            fractions[f"<{cutoff}"] = below / edge_count

        return {
            "total_edges": len(path) - 1,
            "edge_edges": edge_count,
            "sim_edges": sim_count,
            "confidences": edge_confidences,
            "fractions": fractions,
        }

    def load_edge_only_adjacency_with_confidence(self):
        """Load EDGE edges only with confidence metadata"""
        print("Loading EDGE edges (all confidences)...")

        # Structure: adj[node_id] = {neighbor_id: confidence}
        adj = defaultdict(dict)

        all_nodes_query = "MATCH (n) RETURN min(id(n)), max(id(n))"
        id_result = self.query(all_nodes_query)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        current_id, batch_size = min_id, 2000
        edge_count = 0

        while current_id <= max_id:
            query = f"""
            MATCH (n)-[e:EDGE]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
              AND id(m) > id(n)
            RETURN id(n), id(m), e.edge_confidence
            """
            for row in self.query(query):
                n1, n2, conf = int(row[0]), int(row[1]), int(row[2])
                adj[n1][n2] = conf
                adj[n2][n1] = conf
                edge_count += 1
            current_id += batch_size

        print(f"  Loaded {edge_count} EDGE edges, {len(adj)} nodes")
        return adj

    def bfs_reconstruct_path(self, start_id, target_id, adj):  # noqa: F811
        """BFS that returns actual path"""
        visited = {start_id: None}
        queue = deque([start_id])

        while queue:
            node_id = queue.popleft()

            if node_id == target_id:
                # Reconstruct path
                path = []
                current = target_id
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return list(reversed(path))

            for neighbor_id in adj.get(node_id, {}):
                if neighbor_id not in visited:
                    visited[neighbor_id] = node_id
                    queue.append(neighbor_id)

        return None

    def analyze_edge_only_path_confidences(self, path, adj):
        """Analyze EDGE confidences in EDGE-only path"""
        confidences = []

        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            conf = adj[n1].get(n2)
            if conf is None:
                return None  # Not an EDGE-only path
            confidences.append(conf)

        # Calculate fractions below each threshold
        n_edges = len(confidences)
        fractions = {}
        for cutoff in [1, 2, 3, 4, 5]:
            below = sum(1 for c in confidences if c < cutoff)
            fractions[f"<{cutoff}"] = below / n_edges

        return {
            "hops": len(path) - 1,
            "confidences": confidences,
            "fractions": fractions,
        }

    def analyze_all_edge_only_paths(self):
        """Analyze all EDGE-only paths"""
        print("\nAnalyzing ALL EDGE-only paths...")

        # Load interventions and risks
        print("Loading interventions and risks...")
        id_query = "MATCH (i:Intervention) WHERE i.intervention_maturity >= 3 RETURN min(id(i)), max(id(i))"
        id_result = self.query(id_query, timeout=300000)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        interventions = {}
        current_id, batch_size = min_id, 5000
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

        id_query = "MATCH (r:Concept) WHERE r.concept_category = 'risk' RETURN min(id(r)), max(id(r))"
        id_result = self.query(id_query, timeout=300000)
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

        print(f"  {len(interventions)} interventions, {len(risks)} risks")

        # Load EDGE-only graph
        adj = self.load_edge_only_adjacency_with_confidence()

        # Test all intervention-risk pairs
        print("  Finding and analyzing EDGE-only paths...")

        int_ids = list(interventions.keys())
        risk_ids = list(risks.keys())

        path_stats = []
        import time

        start_time = time.time()

        for i_idx, int_id in enumerate(int_ids):
            if (i_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i_idx + 1) / elapsed
                eta_min = (len(int_ids) - (i_idx + 1)) / rate / 60

                print(
                    f"    Progress: {i_idx + 1}/{len(int_ids)} interventions ({100 * (i_idx + 1) / len(int_ids):.1f}%) | "
                    f"Rate: {rate:.0f} int/sec | ETA: {eta_min:.1f} min | "
                    f"Paths found: {len(path_stats):,}"
                )

            for risk_id in risk_ids:
                path = self.bfs_reconstruct_path(int_id, risk_id, adj)
                if not path:
                    continue

                stats = self.analyze_edge_only_path_confidences(path, adj)
                if stats is None:
                    continue

                path_stats.append(stats)

        print(f"\n  Total EDGE-only paths found: {len(path_stats):,}")

        if len(path_stats) == 0:
            print("  No EDGE-only paths found!")
            return

        # Aggregate statistics
        print("\n" + "=" * 80)
        print("EDGE-ONLY PATH CONFIDENCE ANALYSIS")
        print("=" * 80)

        # Paths with at least one edge below each cutoff
        for cutoff in [5, 4, 3, 2, 1]:
            has_below = sum(1 for s in path_stats if s["fractions"][f"<{cutoff}"] > 0)
            pct = 100 * has_below / len(path_stats)
            print(
                f"Paths with ≥1 EDGE edge confidence <{cutoff}: {has_below:,}/{len(path_stats):,} ({pct:.1f}%)"
            )

        print("\n" + "=" * 80)
        print("FRACTION OF EDGES BELOW CUTOFF (per path)")
        print("=" * 80)

        for cutoff in [5, 4, 3, 2, 1]:
            fractions = [s["fractions"][f"<{cutoff}"] for s in path_stats]
            print(f"\nConfidence <{cutoff}:")
            print(f"  Mean: {np.mean(fractions):.3f}")
            print(f"  Median: {np.median(fractions):.3f}")
            print(f"  Min: {np.min(fractions):.3f}, Max: {np.max(fractions):.3f}")

        # Path length distribution
        hops = [s["hops"] for s in path_stats]
        print("\n" + "=" * 80)
        print("PATH LENGTH DISTRIBUTION")
        print("=" * 80)
        print(f"Mean: {np.mean(hops):.1f} hops")
        print(f"Median: {np.median(hops):.1f} hops")
        print(f"Min: {np.min(hops)}, Max: {np.max(hops)}")

        # Save results
        output = {
            "total_edge_only_paths": len(path_stats),
            "paths_with_edge_below": {
                f"<{c}": sum(1 for s in path_stats if s["fractions"][f"<{c}"] > 0)
                for c in [5, 4, 3, 2, 1]
            },
            "fraction_below_cutoff": {
                f"<{c}": {
                    "mean": float(
                        np.mean([s["fractions"][f"<{c}"] for s in path_stats])
                    ),
                    "median": float(
                        np.median([s["fractions"][f"<{c}"] for s in path_stats])
                    ),
                }
                for c in [5, 4, 3, 2, 1]
            },
            "path_length": {
                "mean": float(np.mean(hops)),
                "median": float(np.median(hops)),
                "min": int(np.min(hops)),
                "max": int(np.max(hops)),
            },
        }

        with open("edge_only_path_confidence_analysis.json", "w") as f:
            json.dump(output, f, indent=2)

        print("\n✓ Saved edge_only_path_confidence_analysis.json")

    def extract_paths_at_confidence(self, min_confidence):
        """Extract all paths with EDGE edges >= min_confidence"""
        # Load interventions and risks
        print("Loading nodes...")
        id_query = "MATCH (i:Intervention) WHERE i.intervention_maturity >= 3 RETURN min(id(i)), max(id(i))"
        id_result = self.query(id_query, timeout=300000)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        interventions = {}
        current_id, batch_size = min_id, 5000
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

        id_query = "MATCH (r:Concept) WHERE r.concept_category = 'risk' RETURN min(id(r)), max(id(r))"
        id_result = self.query(id_query, timeout=300000)
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

        # Load EDGE graph with confidence filter
        print(f"Loading EDGE edges (confidence≥{min_confidence})...")
        adj = defaultdict(set)

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
                adj[n1].add(n2)
                adj[n2].add(n1)
                edge_count += 1
            current_id += batch_size

        print(f"  Loaded {edge_count} edges")

        # Test all pairs
        print("Finding paths...")
        int_ids = list(interventions.keys())
        risk_ids = list(risks.keys())

        path_count = 0
        path_lengths = []

        import time

        start_time = time.time()

        for i_idx, int_id in enumerate(int_ids):
            if (i_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i_idx + 1) / elapsed
                eta_min = (len(int_ids) - (i_idx + 1)) / rate / 60

                print(
                    f"  Progress: {i_idx + 1}/{len(int_ids)} ({100 * (i_idx + 1) / len(int_ids):.1f}%) | "
                    f"ETA: {eta_min:.1f} min | Paths: {path_count:,}"
                )

            # BFS to all risks
            visited = {int_id: 0}
            queue = deque([int_id])

            while queue:
                node_id = queue.popleft()
                current_dist = visited[node_id]

                if current_dist >= 50:
                    break

                for neighbor_id in adj.get(node_id, []):
                    if neighbor_id not in visited:
                        visited[neighbor_id] = current_dist + 1
                        queue.append(neighbor_id)

                        if neighbor_id in risk_ids:
                            path_count += 1
                            path_lengths.append(current_dist + 1)

        result = {
            "min_confidence": min_confidence,
            "total_paths": path_count,
            "path_length": {
                "mean": float(np.mean(path_lengths)) if path_lengths else 0,
                "median": float(np.median(path_lengths)) if path_lengths else 0,
                "min": int(np.min(path_lengths)) if path_lengths else 0,
                "max": int(np.max(path_lengths)) if path_lengths else 0,
            },
        }

        print(f"\n  Total paths: {path_count:,}")
        if path_lengths:
            print(
                f"  Path length: mean={np.mean(path_lengths):.1f}, median={np.median(path_lengths):.1f}"
            )

        return result


def main():
    analyzer = PathConfidenceAnalyzer()
    analyzer.client.ping()

    print("=" * 80)
    print("EDGE-ONLY PATH EXTRACTION AT MULTIPLE CONFIDENCE THRESHOLDS")
    print("=" * 80)

    # Extract paths at each confidence threshold
    results = {}

    for min_conf in [1, 2, 3, 4, 5]:
        print(f"\n{'=' * 80}")
        print(f"EXTRACTING PATHS: EDGE confidence ≥{min_conf}")
        print(f"{'=' * 80}")

        result = analyzer.extract_paths_at_confidence(min_conf)
        results[min_conf] = result

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: PATHS AT EACH CONFIDENCE THRESHOLD")
    print("=" * 80)
    for conf in [1, 2, 3, 4, 5]:
        count = results[conf]["total_paths"]
        print(f"Confidence ≥{conf}: {count:,} paths")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
