"""
Attribute Distribution Analysis (All Data)
Analyzes intervention maturity, lifecycle, concept categories, edge confidence.
No quality filters - shows complete distributions.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import redis
from collections import Counter

EMBEDDINGS_TYPE = "wide"
SIMILARITY_EDGE = (
    "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"
    if EMBEDDINGS_TYPE == "wide"
    else "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
)


class AttributeAnalyzer:
    def __init__(self, host="localhost", port=6379, graph="AISafetyIntervention"):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph = graph

    def query(self, cypher):
        result = self.client.execute_command("GRAPH.QUERY", self.graph, cypher)
        return result[1] if len(result) > 1 else []

    def query_all(self, base_query):
        """Query all results by batching"""
        all_results = []
        batch_size = 10000
        skip = 0

        print("  Batching query...")
        while True:
            query = f"{base_query} SKIP {skip} LIMIT {batch_size}"
            batch = self.query(query)
            if not batch:
                break
            all_results.extend(batch)
            skip += batch_size
            print(f"    Retrieved {len(all_results)} rows...")
            if len(batch) < batch_size:
                break

        return all_results

    def intervention_distributions(self):
        """Get intervention maturity and lifecycle distributions"""
        base_query = """
        MATCH (i:Intervention)
        RETURN i.intervention_maturity, i.intervention_lifecycle, i.name
        """
        rows = self.query_all(base_query)

        maturities = []
        lifecycles = []
        for row in rows:
            if row[0] is not None:
                maturities.append(int(row[0]))
            if row[1] is not None:
                lifecycles.append(int(row[1]))

        print(f"Interventions: {len(rows)}")
        print(f"  Maturity: {dict(Counter(maturities))}")
        print(f"  Lifecycle: {dict(Counter(lifecycles))}")

        return maturities, lifecycles

    def concept_categories(self):
        """Get concept category distribution"""
        base_query = """
        MATCH (c:Concept)
        RETURN c.concept_category
        """
        rows = self.query_all(base_query)
        categories = [row[0] for row in rows if row[0]]

        print(f"\nConcept categories: {len(categories)} total")
        print(f"  Distribution: {dict(Counter(categories))}")

        return categories

    def edge_confidence(self):
        """Get edge confidence distribution (all)"""
        base_query = """
        MATCH ()-[e:EDGE]->()
        RETURN e.edge_confidence
        """
        rows = self.query_all(base_query)
        confidences = [int(row[0]) for row in rows if row[0] is not None]

        print(f"\nEDGE edges: {len(confidences)}")
        print(f"  Distribution: {dict(Counter(confidences))}")

        return confidences

    def plot_distributions(self, maturities, lifecycles, categories, confidences):
        """Create 4-panel distribution plot"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Maturity
        mat_counts = Counter(maturities)
        mat_labels = [
            "Foundational/\nTheoretical",
            "Experimental/\nProof-of-Concept",
            "Prototype/\nPilot Studies",
            "Operational/\nDeployment",
        ]
        mat_values = [mat_counts.get(i, 0) for i in range(1, 5)]
        axes[0, 0].bar(range(4), mat_values, color="#3498DB", edgecolor="black")
        axes[0, 0].set_xticks(range(4))
        axes[0, 0].set_xticklabels(mat_labels, rotation=45, ha="right", fontsize=9)
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title(f"Intervention Maturity\nTotal: {len(maturities)}")
        axes[0, 0].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

        # Lifecycle
        lc_counts = Counter(lifecycles)
        lc_labels = [
            "Model\nDesign",
            "Pre-\nTraining",
            "Fine-Tuning/\nRL",
            "Pre-Deploy\nTesting",
            "Deploy-\nment",
            "Other",
        ]
        lc_values = [lc_counts.get(i, 0) for i in range(1, 7)]
        axes[0, 1].bar(range(6), lc_values, color="#E74C3C", edgecolor="black")
        axes[0, 1].set_xticks(range(6))
        axes[0, 1].set_xticklabels(lc_labels, rotation=45, ha="right", fontsize=9)
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title(f"Intervention Lifecycle\nTotal: {len(lifecycles)}")
        axes[0, 1].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

        # Categories
        cat_counts = Counter(categories)
        cat_names = [
            "Risk",
            "Problem\nAnalysis",
            "Theoretical\nInsight",
            "Design\nRationale",
            "Implementation\nMechanism",
            "Validation\nEvidence",
        ]
        cat_values = [
            cat_counts.get(name.replace("\n", " ").lower(), 0)
            for name in [
                "Risk",
                "Problem Analysis",
                "Theoretical Insight",
                "Design Rationale",
                "Implementation Mechanism",
                "Validation Evidence",
            ]
        ]

        axes[1, 0].bar(
            range(len(cat_names)), cat_values, color="#2ECC71", edgecolor="black"
        )
        axes[1, 0].set_xticks(range(len(cat_names)))
        axes[1, 0].set_xticklabels(cat_names, rotation=45, ha="right", fontsize=9)
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title(f"Concept Categories\nTotal: {len(categories)}")
        axes[1, 0].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

        # Edge confidence
        conf_counts = Counter(confidences)
        conf_labels = ["Speculative", "Weak", "Medium", "Strong", "Very Strong"]
        conf_values = [conf_counts.get(i, 0) for i in range(1, 6)]
        axes[1, 1].bar(range(5), conf_values, color="#F39C12", edgecolor="black")
        axes[1, 1].set_xticks(range(5))
        axes[1, 1].set_xticklabels(conf_labels, rotation=45, ha="right", fontsize=9)
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title(f"EDGE Confidence\nTotal: {len(confidences)}")
        axes[1, 1].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.savefig("attribute_distributions.png", dpi=300, bbox_inches="tight")
        print("\n✓ Saved attribute_distributions.png")
        plt.close()


def main():
    analyzer = AttributeAnalyzer()
    analyzer.client.ping()

    print("=" * 80)
    print("ATTRIBUTE DISTRIBUTIONS (ALL DATA)")
    print("=" * 80)

    maturities, lifecycles = analyzer.intervention_distributions()
    categories = analyzer.concept_categories()
    confidences = analyzer.edge_confidence()

    analyzer.plot_distributions(maturities, lifecycles, categories, confidences)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
