"""
Pathway Sampling with BFS Pathfinding (Workshop Validation Item 2)

Uses Python BFS instead of Cypher variable-length patterns to avoid timeouts.
"""

import redis
import json
import random
from collections import deque
import numpy as np

# Configuration
EMBEDDINGS_TYPE = "wide"
if EMBEDDINGS_TYPE == "narrow":
    SIMILARITY_EDGE_NAME = "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
else:
    SIMILARITY_EDGE_NAME = "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"


class PathwaySampler:
    def __init__(self, host="localhost", port=6379, graph_name="AISafetyIntervention"):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph_name = graph_name

    def euclidean_from_cosine(self, cosine_threshold):
        return np.sqrt(2 * (1 - cosine_threshold))

    def find_intervention_nodes(self, limit=200):
        print("\n" + "=" * 80)
        print("FINDING INTERVENTION NODES")
        print("=" * 80)

        query = f"""
        MATCH (n:Intervention)
        RETURN id(n) as node_id,
               n.name as name,
               n.description as description
        LIMIT {limit}
        """

        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

        interventions = []
        if len(result) > 1:
            for row in result[1]:
                interventions.append(
                    {
                        "node_id": int(row[0]),
                        "name": row[1] if row[1] else "Unnamed intervention",
                        "description": row[2] if row[2] else "No description",
                    }
                )

        print(f"Found {len(interventions)} intervention nodes")
        return interventions

    def find_risk_nodes(self, limit=100):
        print("\n" + "=" * 80)
        print("FINDING RISK NODES")
        print("=" * 80)

        risk_keywords = [
            "existential risk",
            "x-risk",
            "catastrophic",
            "extinction",
            "misalignment",
            "deceptive alignment",
            "power-seeking",
            "treacherous turn",
            "reward hacking",
            "goal misgeneralization",
            "mesa-optimization",
        ]

        risks = []
        for keyword in risk_keywords:
            query = f"""
            MATCH (n:Concept)
            WHERE toLower(n.name) CONTAINS toLower('{keyword}')
               OR toLower(n.description) CONTAINS toLower('{keyword}')
            RETURN id(n) as node_id,
                   n.name as name,
                   n.description as description
            LIMIT 10
            """

            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            if len(result) > 1:
                for row in result[1]:
                    node_id = int(row[0])
                    if node_id not in [r["node_id"] for r in risks]:
                        risks.append(
                            {
                                "node_id": node_id,
                                "name": row[1] if row[1] else "Unnamed risk",
                                "description": row[2] if row[2] else "No description",
                            }
                        )

        print(f"Found {len(risks)} risk nodes")
        return risks

    def _get_neighbors(self, node_id, euclidean_threshold=None):
        """Get neighbor node IDs via EDGE or SIMILARITY edges"""
        if euclidean_threshold is not None:
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

            if len(result) > 1:
                return [int(row[0]) for row in result[1]]
        except Exception:
            pass

        return []

    def find_path(self, start_id, end_id, max_length=6, euclidean_threshold=None):
        """
        Find shortest path using BFS in Python

        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            max_length: Maximum path length to search
            euclidean_threshold: Optional similarity score filter

        Returns:
            dict with node_ids and length, or None
        """
        # BFS
        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()

            # Check if we've exceeded max length
            if len(path) - 1 > max_length:
                continue

            # Found target
            if current_id == end_id:
                return {"node_ids": path, "length": len(path) - 1}

            # Get neighbors
            neighbors = self._get_neighbors(current_id, euclidean_threshold)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_node_details(self, node_id):
        """Get node details"""
        query = f"""
        MATCH (n)
        WHERE id(n) = {node_id}
        RETURN id(n) as node_id,
               labels(n) as labels,
               n.name as name,
               n.description as description,
               n.concept_category as category
        """

        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

        if len(result) > 1 and len(result[1]) > 0:
            row = result[1][0]
            return {
                "node_id": int(row[0]),
                "labels": row[1] if row[1] else [],
                "name": row[2] if row[2] else "No name",
                "description": row[3] if row[3] else "No description",
                "category": row[4] if len(row) > 4 else None,
            }
        return None

    def get_source_url(self, node_id):
        """Get source URL"""
        query = f"""
        MATCH (n)-[:FROM]->(s:Source)
        WHERE id(n) = {node_id}
        RETURN s.url
        LIMIT 1
        """

        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

        if len(result) > 1 and len(result[1]) > 0:
            return result[1][0][0]
        return "Unknown source"

    def extract_pathway_context(self, pathway):
        """Extract full context for pathway"""
        nodes_detail = []

        for node_id in pathway["node_ids"]:
            node_info = self.get_node_details(node_id)
            if node_info:
                source_url = self.get_source_url(node_id)
                node_info["source_url"] = source_url
                nodes_detail.append(node_info)

        pathway["nodes"] = nodes_detail
        return pathway

    def stratified_sample_pathways(
        self,
        interventions,
        risks,
        cosine_thresholds=[0.8, 0.85, 0.9, 0.95],
        target_short=10,
        target_medium=15,
        target_long=5,
        max_attempts=300,
    ):
        """
        Generate stratified sample using BFS pathfinding

        Reduced max_attempts since BFS is slower than Cypher
        """
        print("\n" + "=" * 80)
        print("STRATIFIED PATHWAY SAMPLING")
        print("=" * 80)
        print(
            f"Target: {target_short} short, {target_medium} medium, {target_long} long pathways"
        )
        print(
            f"Testing {len(interventions)} interventions × {len(risks)} risks × {len(cosine_thresholds)} thresholds"
        )

        euclidean_thresholds = {
            cos_t: self.euclidean_from_cosine(cos_t) for cos_t in cosine_thresholds
        }

        pathways = {"short": [], "medium": [], "long": []}

        attempts = 0

        while attempts < max_attempts:
            # Random selection
            intervention = random.choice(interventions)
            risk = random.choice(risks)
            cos_threshold = random.choice(cosine_thresholds)
            eucl_threshold = euclidean_thresholds[cos_threshold]

            # Try to find path
            path_result = self.find_path(
                intervention["node_id"],
                risk["node_id"],
                max_length=6,  # Keep reasonable for BFS
                euclidean_threshold=eucl_threshold,
            )

            attempts += 1

            if path_result is None:
                if attempts % 50 == 0:
                    total = sum(len(pathways[cat]) for cat in pathways)
                    print(f"  Attempts: {attempts}, Found: {total} pathways")
                continue

            path_length = path_result["length"]

            # Categorize
            if path_length <= 3:
                category = "short"
                target = target_short
            elif path_length <= 6:
                category = "medium"
                target = target_medium
            else:
                category = "long"
                target = target_long

            if len(pathways[category]) >= target:
                continue

            # Add pathway
            pathway = {
                "pathway_id": sum(len(pathways[cat]) for cat in pathways),
                "intervention": intervention,
                "risk": risk,
                "threshold_cosine": cos_threshold,
                "threshold_euclidean": eucl_threshold,
                "length": path_length,
                "length_category": category,
                "node_ids": path_result["node_ids"],
            }

            pathways[category].append(pathway)

            print(
                f"  Found {category} pathway ({path_length} hops) at threshold {cos_threshold}"
            )

            # Check if done
            total = sum(len(pathways[cat]) for cat in pathways)
            target_total = target_short + target_medium + target_long
            if total >= target_total:
                break

        all_pathways = pathways["short"] + pathways["medium"] + pathways["long"]

        print(f"\n{'=' * 80}")
        print("SAMPLING RESULTS")
        print(f"{'=' * 80}")
        print(f"Short pathways (≤3): {len(pathways['short'])}/{target_short}")
        print(f"Medium pathways (4-6): {len(pathways['medium'])}/{target_medium}")
        print(f"Long pathways (7+): {len(pathways['long'])}/{target_long}")
        print(
            f"Total: {len(all_pathways)}/{target_short + target_medium + target_long}"
        )
        print(f"Attempts: {attempts}")

        return all_pathways

    def save_pathways_for_annotation(
        self, pathways, output_file="pathways_for_annotation.json"
    ):
        """Save pathways with annotation template"""
        print(f"\n{'=' * 80}")
        print("EXTRACTING PATHWAY CONTEXTS")
        print(f"{'=' * 80}")

        annotated_pathways = []

        for i, pathway in enumerate(pathways):
            print(f"  Extracting context for pathway {i + 1}/{len(pathways)}...")

            pathway_with_context = self.extract_pathway_context(pathway)

            pathway_with_context["annotation"] = {
                "judgment": None,  # 'correct' | 'incorrect' | 'uncertain'
                "notes": "",
                "annotator": "",
            }

            annotated_pathways.append(pathway_with_context)

        with open(output_file, "w") as f:
            json.dump(annotated_pathways, f, indent=2)

        print(f"\n✓ Saved {len(annotated_pathways)} pathways to {output_file}")


def main():
    sampler = PathwaySampler()

    print("=" * 80)
    print("PATHWAY SAMPLING (BFS VERSION)")
    print("=" * 80)

    try:
        sampler.client.ping()
        print("✓ Connected to FalkorDB")

        # Find nodes
        interventions = sampler.find_intervention_nodes(limit=200)
        risks = sampler.find_risk_nodes(limit=75)

        if len(interventions) == 0 or len(risks) == 0:
            print("\n✗ ERROR: Could not find intervention or risk nodes")
            return

        # Generate sample
        pathways = sampler.stratified_sample_pathways(
            interventions,
            risks,
            cosine_thresholds=[0.8, 0.85, 0.9, 0.95],
            target_short=10,
            target_medium=15,
            target_long=5,
            max_attempts=300,  # Reduced since BFS is slower
        )

        if len(pathways) == 0:
            print("\n✗ ERROR: No pathways found")
            print("  Suggestion: Network may be sparse. Try:")
            print("  - Increase max_attempts to 500")
            print("  - Reduce target counts")
            print("  - Check if EDGE and SIMILARITY edges exist")
            return

        # Save for annotation
        sampler.save_pathways_for_annotation(pathways)

        print("\n" + "=" * 80)
        print("PATHWAY SAMPLING COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review pathways_for_annotation.json")
        print("  2. Manually annotate pathways:")
        print("     - Set judgment: 'correct' | 'incorrect' | 'uncertain'")
        print("     - Add notes if helpful")
        print("  3. Two annotators annotate same 15 pathways for Cohen's κ")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
