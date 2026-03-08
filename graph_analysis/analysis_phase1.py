"""
Phase 1 Analysis: Hop Distributions, Reachable Degrees, Component Analysis
Analyzes saved reachability matrices from Phase 1.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import redis
import numpy as np
import json
import scipy.sparse as sp
from collections import Counter, defaultdict, deque

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class Phase1Analyzer:
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

    def load_reachability_data(self, threshold):
        """Load saved reachability matrix and node lists"""
        matrix = sp.load_npz(f"reachability_matrix_{threshold}.npz")

        with open(f"reachable_interventions_{threshold}.json") as f:
            interventions = json.load(f)

        with open(f"reachable_risks_{threshold}.json") as f:
            risks = json.load(f)

        return matrix, interventions, risks

    def plot_hop_distributions(self, all_matrices):
        """Plot hop length distributions across thresholds"""
        print("  Generating hop length distribution plots...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors = [
            "#2ECC71",
            "#9B59B6",
            "#3498DB",
            "#F39C12",
            "#E74C3C",
        ]  # Added green for EDGE
        markers = ["X", "o", "s", "^", "D"]  # X for EDGE

        # Left: Log-log scatter
        ax1 = axes[0]
        print("    Processing log-log scatter...")

        # Sort keys: EDGE first, then thresholds
        sorted_keys = []
        if "EDGE≥4" in all_matrices:
            sorted_keys.append("EDGE≥4")
        sorted_keys.extend(sorted([k for k in all_matrices.keys() if k != "EDGE≥4"]))

        for i, key in enumerate(sorted_keys):
            print(f"      {key}...")
            matrix = all_matrices[key]
            lengths = matrix.data[matrix.data > 0]

            if len(lengths) == 0:
                continue

            length_counts = Counter(lengths)
            unique_lengths = sorted(length_counts.keys())
            frequencies = [length_counts[ln] for ln in unique_lengths]

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            label = key if key == "EDGE≥4" else f"Cosine ≥{key}"

            ax1.scatter(
                unique_lengths,
                frequencies,
                alpha=0.6,
                s=60,
                label=label,
                color=color,
                marker=marker,
                edgecolors="black",
                linewidth=0.5,
            )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Path Length (hops, log scale)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Frequency (log scale)", fontsize=14, fontweight="bold")
        ax1.set_title(
            "Intervention→Risk Path Length Distribution", fontsize=16, fontweight="bold"
        )
        ax1.legend(fontsize=11, loc="best")
        ax1.grid(True, alpha=0.3, which="both")

        # Right: CCDF
        ax2 = axes[1]
        print("    Processing CCDF...")

        for i, key in enumerate(sorted_keys):
            print(f"      {key}...")
            matrix = all_matrices[key]
            lengths = matrix.data[matrix.data > 0]

            if len(lengths) == 0:
                continue

            sorted_lengths = np.sort(lengths)
            ccdf = 1 - np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            label = key if key == "EDGE≥4" else f"Cosine ≥{key}"

            ax2.scatter(
                sorted_lengths,
                ccdf,
                alpha=0.6,
                s=40,
                label=label,
                color=color,
                marker=marker,
                edgecolors="black",
                linewidth=0.5,
            )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Path Length (hops, log scale)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("P(Length ≥ k)", fontsize=14, fontweight="bold")
        ax2.set_title("CCDF - Path Length Distribution", fontsize=16, fontweight="bold")
        ax2.legend(fontsize=11, loc="best")
        ax2.grid(True, alpha=0.3, which="both")

        print("    Saving plot...")
        plt.tight_layout()
        plt.savefig("path_length_distributions.png", dpi=300, bbox_inches="tight")
        print("✓ Saved path_length_distributions.png")
        plt.close()

    def load_adjacency_for_degrees(self, threshold):
        """Load adjacency list for degree computation"""
        euclidean = self.euclidean_from_cosine(threshold)

        print("  Loading adjacency list...")
        adj_list = defaultdict(set)

        # EDGE edges
        all_nodes_query = "MATCH (n) RETURN min(id(n)), max(id(n))"
        id_result = self.query(all_nodes_query)
        min_id, max_id = int(id_result[0][0]), int(id_result[0][1])

        current_id = min_id
        batch_size = 2000

        while current_id <= max_id:
            edge_query = f"""
            MATCH (n)-[e:EDGE]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
              AND id(m) > id(n)
              AND e.edge_confidence >= 4
            RETURN id(n), id(m)
            """
            for row in self.query(edge_query):
                n1, n2 = int(row[0]), int(row[1])
                adj_list[n1].add(n2)
                adj_list[n2].add(n1)
            current_id += batch_size

        # SIMILARITY edges
        current_id = min_id
        while current_id <= max_id:
            sim_query = f"""
            MATCH (n)-[s:{SIMILARITY_EDGE}]-(m)
            WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
              AND id(m) > id(n)
              AND s.score < {euclidean}
            RETURN id(n), id(m)
            """
            for row in self.query(sim_query):
                n1, n2 = int(row[0]), int(row[1])
                adj_list[n1].add(n2)
                adj_list[n2].add(n1)
            current_id += batch_size

        print(f"    Loaded adjacency for {len(adj_list)} nodes")
        return adj_list

    def compute_reachable_degrees(
        self, adj_list, reachable_int_ids, reachable_risk_ids
    ):
        """Compute degrees from adjacency list"""
        print("  Computing degrees from adjacency list...")

        int_degrees = {iid: len(adj_list[iid]) for iid in reachable_int_ids}
        risk_degrees = {rid: len(adj_list[rid]) for rid in reachable_risk_ids}

        return int_degrees, risk_degrees

    def find_components_from_adj(self, adj_list, reachable_ids):
        """Find connected components from adjacency list"""
        visited = set()
        components = []

        for start_id in reachable_ids:
            if start_id in visited:
                continue

            component = set()
            queue = deque([start_id])

            while queue:
                node_id = queue.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)
                component.add(node_id)

                for neighbor in adj_list[node_id]:
                    if neighbor not in visited and neighbor in reachable_ids:
                        queue.append(neighbor)

            components.append(component)

        return components

    def plot_reachable_degrees(self, all_degrees):
        """Plot degree distributions for reachable subsets"""
        for node_type in ["interventions", "risks"]:
            print(f"  Generating {node_type} degree distribution plots...")
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            colors = ["#9B59B6", "#3498DB", "#F39C12", "#E74C3C"]
            markers = ["o", "s", "^", "D"]

            # Left: Log-log
            ax1 = axes[0]
            print("    Processing log-log scatter...")

            for i, threshold in enumerate(sorted(all_degrees.keys())):
                print(f"      Threshold {threshold}...")
                degrees = list(all_degrees[threshold][node_type].values())

                if not degrees or max(degrees) == 0:
                    continue

                degree_counts = Counter(degrees)
                unique_degrees = sorted([d for d in degree_counts.keys() if d > 0])
                frequencies = [degree_counts[d] for d in unique_degrees]

                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]

                ax1.scatter(
                    unique_degrees,
                    frequencies,
                    alpha=0.6,
                    s=60,
                    label=f"Cosine ≥{threshold}",
                    color=color,
                    marker=marker,
                    edgecolors="black",
                    linewidth=0.5,
                )

            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_xlabel("Degree (log scale)", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Frequency (log scale)", fontsize=12, fontweight="bold")
            ax1.set_title(
                f"Reachable {node_type.capitalize()} Degree Distribution",
                fontsize=14,
                fontweight="bold",
            )
            ax1.legend(fontsize=10, loc="best")
            ax1.grid(True, alpha=0.3, which="both")

            # Right: CCDF
            ax2 = axes[1]
            print("    Processing CCDF...")

            for i, threshold in enumerate(sorted(all_degrees.keys())):
                print(f"      Threshold {threshold}...")
                degrees = list(all_degrees[threshold][node_type].values())

                if not degrees:
                    continue

                sorted_degrees = np.sort([d for d in degrees if d > 0])

                if len(sorted_degrees) > 0:
                    ccdf = 1 - np.arange(1, len(sorted_degrees) + 1) / len(
                        sorted_degrees
                    )
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]

                    ax2.scatter(
                        sorted_degrees,
                        ccdf,
                        alpha=0.6,
                        s=40,
                        label=f"Cosine ≥{threshold}",
                        color=color,
                        marker=marker,
                        edgecolors="black",
                        linewidth=0.5,
                    )

            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel("Degree (log scale)", fontsize=12, fontweight="bold")
            ax2.set_ylabel("P(Degree ≥ k)", fontsize=12, fontweight="bold")
            ax2.set_title(
                f"{node_type.capitalize()} CCDF", fontsize=14, fontweight="bold"
            )
            ax2.legend(fontsize=10, loc="best")
            ax2.grid(True, alpha=0.3, which="both")

            print("    Saving plot...")
            plt.tight_layout()
            plt.savefig(
                f"reachable_{node_type}_degrees.png", dpi=300, bbox_inches="tight"
            )
            print(f"✓ Saved reachable_{node_type}_degrees.png")
            plt.close()

    def save_component_analysis(
        self,
        threshold,
        int_components,
        risk_components,
        interventions,
        risks,
        int_degrees,
        risk_degrees,
    ):
        """Save component analysis with hub identification"""

        # Intervention components
        int_comp_data = []
        for comp in sorted(int_components, key=len, reverse=True)[:20]:  # Top 20
            # Find hub (highest degree)
            hub_id = max(comp, key=lambda x: int_degrees.get(x, 0))
            hub_name = next(
                (i["name"] for i in interventions if i["id"] == hub_id), "Unknown"
            )
            hub_degree = int_degrees.get(hub_id, 0)

            int_comp_data.append(
                {
                    "size": len(comp),
                    "hub_id": hub_id,
                    "hub_name": hub_name,
                    "hub_degree": hub_degree,
                    "member_ids": list(comp),
                }
            )

        # Risk components
        risk_comp_data = []
        for comp in sorted(risk_components, key=len, reverse=True)[:20]:  # Top 20
            hub_id = max(comp, key=lambda x: risk_degrees.get(x, 0))
            hub_name = next((r["name"] for r in risks if r["id"] == hub_id), "Unknown")
            hub_degree = risk_degrees.get(hub_id, 0)

            risk_comp_data.append(
                {
                    "size": len(comp),
                    "hub_id": hub_id,
                    "hub_name": hub_name,
                    "hub_degree": hub_degree,
                    "member_ids": list(comp),
                }
            )

        output = {
            "threshold": threshold,
            "intervention_components": {
                "total": len(int_components),
                "top_20": int_comp_data,
            },
            "risk_components": {
                "total": len(risk_components),
                "top_20": risk_comp_data,
            },
        }

        with open(f"components_{threshold}.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"  ✓ Saved components_{threshold}.json")
        print(f"    Interventions: {len(int_components)} components")
        print(f"    Risks: {len(risk_components)} components")


def main():
    analyzer = Phase1Analyzer()
    analyzer.client.ping()

    print("=" * 80)
    print("PHASE 1 ANALYSIS: HOP DISTRIBUTIONS, DEGREES, COMPONENTS")
    print("=" * 80)

    thresholds = [0.8, 0.85, 0.9, 0.95]
    all_matrices = {}
    all_degrees = {}

    # Load EDGE-only results
    print("\nLoading EDGE-only results...")
    try:
        edge_matrix = sp.load_npz("reachability_matrix_edge_conf4.npz")
        all_matrices["EDGE≥4"] = edge_matrix
        print(f"  ✓ Loaded EDGE-only: {edge_matrix.nnz:,} paths")
    except FileNotFoundError:
        print("  ⚠ EDGE-only results not found, skipping")

    for threshold in thresholds:
        print(f"\n{'=' * 80}")
        print(f"THRESHOLD ≥{threshold}")
        print(f"{'=' * 80}")

        # Load data
        matrix, interventions, risks = analyzer.load_reachability_data(threshold)
        all_matrices[threshold] = matrix

        int_ids = {i["id"] for i in interventions}
        risk_ids = {r["id"] for r in risks}

        # Load adjacency and compute degrees
        adj_list = analyzer.load_adjacency_for_degrees(threshold)
        int_degrees, risk_degrees = analyzer.compute_reachable_degrees(
            adj_list, int_ids, risk_ids
        )
        all_degrees[threshold] = {"interventions": int_degrees, "risks": risk_degrees}

        # Components (reuse adj_list)
        int_components = analyzer.find_components_from_adj(adj_list, int_ids)
        risk_components = analyzer.find_components_from_adj(adj_list, risk_ids)

        # Save
        analyzer.save_component_analysis(
            threshold,
            int_components,
            risk_components,
            interventions,
            risks,
            int_degrees,
            risk_degrees,
        )

    # Plot hop distributions
    print(f"\n{'=' * 80}")
    print("GENERATING PLOTS")
    print("=" * 80)
    analyzer.plot_hop_distributions(all_matrices)
    analyzer.plot_reachable_degrees(all_degrees)

    print("\n" + "=" * 80)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
