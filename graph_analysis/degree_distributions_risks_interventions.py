"""
Node Degree Distribution Analysis
Histograms of intervention and risk connectivity across thresholds.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import redis
import numpy as np
import json
from collections import Counter
from scipy.optimize import curve_fit

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class DegreeAnalyzer:
    def __init__(self, host="localhost", port=6379, graph="AISafetyIntervention"):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph = graph

    def euclidean_from_cosine(self, cosine):
        return np.sqrt(2 * (1 - cosine))

    def query(self, cypher):
        result = self.client.execute_command("GRAPH.QUERY", self.graph, cypher)
        return result[1] if len(result) > 1 else []

    def get_node_degrees(self, node_type, threshold, category=None):
        """Get degree distribution for nodes using ID-based batching"""
        euclidean = self.euclidean_from_cosine(threshold)

        # Get ID range
        if node_type == "Intervention":
            id_query = """
            MATCH (n:Intervention)
            WHERE n.intervention_maturity >= 3
            RETURN min(id(n)), max(id(n))
            """
        else:  # Concept with category
            id_query = f"""
            MATCH (n:Concept)
            WHERE n.concept_category = '{category}'
            RETURN min(id(n)), max(id(n))
            """

        id_result = self.query(id_query)
        if not id_result or not id_result[0]:
            return []

        min_id = int(id_result[0][0])
        max_id = int(id_result[0][1])

        # Batch by ID range
        degrees = []
        current_id = min_id
        batch_size = 2000

        while current_id <= max_id:
            if node_type == "Intervention":
                query = f"""
                MATCH (n:Intervention)
                WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
                  AND n.intervention_maturity >= 3
                WITH n
                OPTIONAL MATCH (n)-[e:EDGE|{SIMILARITY_EDGE}]-(m:Intervention)
                WHERE m.intervention_maturity >= 3
                  AND ((type(e) = 'EDGE' AND e.edge_confidence >= 4)
                    OR (type(e) = '{SIMILARITY_EDGE}' AND e.score < {euclidean}))
                RETURN id(n), count(DISTINCT m) as degree
                """
            else:  # Concept
                query = f"""
                MATCH (n:Concept)
                WHERE id(n) >= {current_id} AND id(n) < {current_id + batch_size}
                  AND n.concept_category = '{category}'
                WITH n
                OPTIONAL MATCH (n)-[e:EDGE|{SIMILARITY_EDGE}]-(m:Concept)
                WHERE m.concept_category = '{category}'
                  AND ((type(e) = 'EDGE' AND e.edge_confidence >= 4)
                    OR (type(e) = '{SIMILARITY_EDGE}' AND e.score < {euclidean}))
                RETURN id(n), count(DISTINCT m) as degree
                """

            batch = self.query(query)
            for row in batch:
                degrees.append(int(row[1]))

            current_id += batch_size

        return degrees

    def plot_distributions(self, all_results):
        """Create degree distribution plots matching reference format"""
        thresholds = sorted(all_results.keys())

        # Separate plots for interventions and risks
        for node_type in ["interventions", "risks"]:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            colors = ["#9B59B6", "#3498DB", "#F39C12", "#E74C3C"]
            markers = ["o", "s", "^", "D"]

            # Left: Log-Log Scatter with Power Law Fits
            ax1 = axes[0]

            for i, threshold in enumerate(thresholds):
                degrees = all_results[threshold][node_type]
                if not degrees or max(degrees) == 0:
                    continue

                degree_counts = Counter(degrees)
                unique_degrees = sorted([d for d in degree_counts.keys() if d > 0])
                frequencies = [degree_counts[d] for d in unique_degrees]

                if len(unique_degrees) > 0:
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

                    # Fit power law
                    fit_result = self.fit_power_law(unique_degrees, frequencies)
                    if fit_result is not None:
                        ax1.plot(
                            fit_result["fit_x"],
                            fit_result["fit_y"],
                            "--",
                            linewidth=2.5,
                            color=color,
                            alpha=0.8,
                            label=f"≥{threshold} fit: γ={fit_result['gamma']:.2f}±{fit_result['gamma_err']:.2f}, R²={fit_result['r_squared']:.3f}",
                        )

            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_xlabel("Degree (log scale)", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Frequency (log scale)", fontsize=14, fontweight="bold")
            ax1.set_title(
                f"{node_type.capitalize()} Degree Distribution (Log-Log)",
                fontsize=16,
                fontweight="bold",
            )
            ax1.legend(fontsize=9, loc="best")
            ax1.grid(True, alpha=0.3, which="both")

            # Right: CCDF
            ax2 = axes[1]

            for i, threshold in enumerate(thresholds):
                degrees = all_results[threshold][node_type]
                if not degrees:
                    continue

                sorted_degrees = np.sort([d for d in degrees if d > 0])

                if len(sorted_degrees) > 0:
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]
                    ccdf = 1 - np.arange(1, len(sorted_degrees) + 1) / len(
                        sorted_degrees
                    )
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
            ax2.set_xlabel("Degree (log scale)", fontsize=14, fontweight="bold")
            ax2.set_ylabel("P(Degree ≥ k)", fontsize=14, fontweight="bold")
            ax2.set_title(
                "CCDF - Complementary Cumulative Distribution",
                fontsize=16,
                fontweight="bold",
            )
            ax2.legend(fontsize=10, loc="best")
            ax2.grid(True, alpha=0.3, which="both")

            plt.tight_layout()
            filename = f"degree_distributions_{node_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✓ Saved {filename}")
            plt.close()

    def fit_power_law(self, unique_degrees, frequencies, degree_min=2, degree_max=100):
        """Fit power law with uncertainties"""

        if len(unique_degrees) < 5:
            return None

        log_degrees = np.log10(unique_degrees)
        log_freqs = np.log10(frequencies)
        mask = (log_degrees >= np.log10(degree_min)) & (
            log_degrees <= np.log10(degree_max)
        )

        if np.sum(mask) < 5:
            return None

        weights = np.array(frequencies)

        def power_law(x, slope, intercept):
            return intercept + slope * x

        try:
            popt, pcov = curve_fit(
                power_law,
                log_degrees[mask],
                log_freqs[mask],
                sigma=1 / np.sqrt(weights[mask]),
                absolute_sigma=True,
            )
            slope, intercept = popt
            slope_err = np.sqrt(np.diag(pcov))[0]

            residuals = log_freqs[mask] - power_law(log_degrees[mask], slope, intercept)
            ss_res = np.sum(weights[mask] * residuals**2)
            ss_tot = np.sum(
                weights[mask] * (log_freqs[mask] - np.mean(log_freqs[mask])) ** 2
            )
            r_squared = 1 - (ss_res / ss_tot)

            fit_x = np.array(unique_degrees)[mask]
            fit_y = 10 ** (intercept) * fit_x**slope

            return {
                "slope": slope,
                "gamma": -slope,
                "gamma_err": slope_err,
                "r_squared": r_squared,
                "fit_x": fit_x,
                "fit_y": fit_y,
            }
        except Exception:
            return None

    def run_analysis(self, thresholds=[0.8, 0.85, 0.9, 0.95]):
        print("=" * 80)
        print("DEGREE DISTRIBUTION ANALYSIS")
        print("=" * 80)

        all_results = {}

        for threshold in thresholds:
            print(f"\nThreshold ≥{threshold}:")

            print("  Computing intervention degrees...")
            int_degrees = self.get_node_degrees("Intervention", threshold)

            print("  Computing risk degrees...")
            risk_degrees = self.get_node_degrees("Concept", threshold, category="risk")

            all_results[threshold] = {
                "interventions": int_degrees,
                "risks": risk_degrees,
            }

            print(
                f"    Interventions: {len(int_degrees)} nodes, "
                f"isolated: {int_degrees.count(0)}, "
                f"max degree: {max(int_degrees) if int_degrees else 0}"
            )
            print(
                f"    Risks: {len(risk_degrees)} nodes, "
                f"isolated: {risk_degrees.count(0) if risk_degrees else 0}, "
                f"max degree: {max(risk_degrees) if risk_degrees else 0}"
            )

        self.plot_distributions(all_results)

        with open("degree_distributions.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n✓ Saved degree_distributions.json")


def main():
    analyzer = DegreeAnalyzer()
    analyzer.client.ping()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
