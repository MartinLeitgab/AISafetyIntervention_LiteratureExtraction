"""
Pathway Sampling for Manual Validation (Workshop Validation Item 2)

Architecture:
1. Connect to FalkorDB and query for intervention/risk nodes
2. Find intervention→risk paths using similarity thresholds
3. Stratify sample by:
   - Cluster assignment (from semantic clustering)
   - Path length (short ≤3, medium 4-6, long 7+)
   - Similarity threshold (0.8, 0.85, 0.9, 0.95)
4. Extract context snippets from source papers
5. Save to JSON for manual annotation

Dependencies: redis, json
"""

import redis
import json
import random
import numpy as np

# Configuration
EMBEDDINGS_TYPE = "wide"  # "narrow" or "wide"
if EMBEDDINGS_TYPE == "narrow":
    SIMILARITY_EDGE_NAME = "SIMILARITY_ABOVE_POINT_EIGHT_1300_NEAREST"
else:
    SIMILARITY_EDGE_NAME = "SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST"


class PathwaySampler:
    def __init__(self, host="localhost", port=6379, graph_name="AISafetyIntervention"):
        """Initialize FalkorDB connection"""
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.graph_name = graph_name

    def euclidean_from_cosine(self, cosine_threshold):
        """Convert cosine similarity to Euclidean distance"""
        return np.sqrt(2 * (1 - cosine_threshold))

    def find_intervention_nodes(self, limit=100):
        """
        Find intervention nodes across the graph
        Returns: List of (node_id, name, description, cluster_id)
        """
        print("\n" + "=" * 80)
        print("FINDING INTERVENTION NODES")
        print("=" * 80)

        query = """
        MATCH (n:Intervention)
        RETURN id(n) as node_id, 
               n.name as name, 
               n.description as description,
               n.cluster_id as cluster_id
        LIMIT {limit}
        """.format(limit=limit)

        result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

        interventions = []
        if len(result) > 1:
            for row in result[1]:
                node_id = int(row[0])
                name = row[1] if row[1] else "Unnamed intervention"
                description = row[2] if row[2] else "No description"
                cluster_id = (
                    int(row[3]) if len(row) > 3 and row[3] is not None else None
                )

                interventions.append(
                    {
                        "node_id": node_id,
                        "name": name,
                        "description": description,
                        "cluster_id": cluster_id,
                    }
                )

        print(f"Found {len(interventions)} intervention nodes")
        return interventions

    def find_risk_nodes(self, limit=50):
        """
        Find top-level risk nodes (containing risk keywords)
        Returns: List of (node_id, name, description, concept_category)
        """
        print("\n" + "=" * 80)
        print("FINDING RISK NODES")
        print("=" * 80)

        # Keywords indicating top-level risks
        risk_keywords = [
            "existential risk",
            "x-risk",
            "catastrophic",
            "extinction",
            "misalignment",
            "deceptive alignment",
            "power-seeking",
            "treacherous turn",
        ]

        risks = []
        for keyword in risk_keywords:
            query = f"""
            MATCH (n:Concept)
            WHERE (toLower(n.name) CONTAINS toLower('{keyword}') 
                   OR toLower(n.description) CONTAINS toLower('{keyword}'))
                  AND n.concept_category = 'risk'
            RETURN id(n) as node_id,
                   n.name as name,
                   n.description as description,
                   n.concept_category as category
            LIMIT 10
            """

            result = self.client.execute_command("GRAPH.QUERY", self.graph_name, query)

            if len(result) > 1:
                for row in result[1]:
                    node_id = int(row[0])
                    # Avoid duplicates
                    if node_id not in [r["node_id"] for r in risks]:
                        risks.append(
                            {
                                "node_id": node_id,
                                "name": row[1] if row[1] else "Unnamed risk",
                                "description": row[2] if row[2] else "No description",
                                "category": row[3] if len(row) > 3 else "risk",
                            }
                        )

        print(f"Found {len(risks)} risk nodes")
        return risks

    def find_path(self, start_id, end_id, max_length=6, euclidean_threshold=None):
        """
        Find shortest path between two nodes

        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            max_length: Maximum path length to search
            euclidean_threshold: Optional similarity score filter

        Returns: List of node IDs in path, or None if no path found
        """
        """Reduced max_length to 6 for performance"""

        query = f"""
            MATCH (start), (end)
            WHERE id(start) = {start_id} AND id(end) = {end_id}
            MATCH path = (start)-[:EDGE|{SIMILARITY_EDGE_NAME}*1..{max_length}]-(end)
            RETURN [node IN nodes(path) | id(node)] as node_ids,
                length(path) as path_length
            ORDER BY length(path)
            LIMIT 1
            """

        try:
            result = self.client.execute_command(
                "GRAPH.QUERY", self.graph_name, query, "--timeout", "5000"
            )

            if len(result) > 1 and len(result[1]) > 0:
                node_ids = [int(nid) for nid in result[1][0][0]]
                path_length = int(result[1][0][1])

                # Filter by threshold post-query if needed
                if euclidean_threshold is not None:
                    if not self._path_meets_threshold(node_ids, euclidean_threshold):
                        return None

                return {"node_ids": node_ids, "length": path_length}
        except Exception as e:
            print(f"  Error finding path: {e}")
            return None

    def get_node_details(self, node_id):
        """Get detailed information about a node"""
        query = f"""
        MATCH (n)
        WHERE id(n) = {node_id}
        RETURN id(n) as node_id,
               labels(n) as labels,
               n.name as name,
               n.description as description,
               n.concept_category as category,
               n.cluster_id as cluster_id
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
                "cluster_id": int(row[5])
                if len(row) > 5 and row[5] is not None
                else None,
            }
        return None

    def get_source_url(self, node_id):
        """Get source document URL for a node"""
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
        """
        Extract context for each node in pathway

        Args:
            pathway: Dict with 'node_ids', 'intervention', 'risk', 'length', 'threshold'

        Returns: Enhanced pathway dict with node details and sources
        """
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
        target_total=30,
        target_short=10,
        target_medium=15,
        target_long=5,
    ):
        """
        Generate stratified sample of intervention→risk pathways

        Stratification:
        - Path length: short (≤3), medium (4-6), long (7+)
        - Similarity threshold: 0.8, 0.85, 0.9, 0.95
        - Cluster: representative coverage

        Args:
            interventions: List of intervention nodes
            risks: List of risk nodes
            cosine_thresholds: Thresholds to test
            target_total: Total pathways to collect
            target_short/medium/long: Target counts per length category

        Returns: List of pathway dicts
        """
        print("\n" + "=" * 80)
        print("STRATIFIED PATHWAY SAMPLING")
        print("=" * 80)

        euclidean_thresholds = {
            cos_t: self.euclidean_from_cosine(cos_t) for cos_t in cosine_thresholds
        }

        pathways = {
            "short": [],  # ≤3 hops
            "medium": [],  # 4-6 hops
            "long": [],  # 7+ hops
        }

        # Shuffle to get random sampling
        random.shuffle(interventions)
        random.shuffle(risks)

        attempts = 0
        max_attempts = 500

        print(
            f"\nTarget: {target_short} short, {target_medium} medium, {target_long} long pathways"
        )
        print(
            f"Testing {len(interventions)} interventions × {len(risks)} risks × {len(cosine_thresholds)} thresholds"
        )

        for intervention in interventions:
            if attempts >= max_attempts:
                break

            for risk in risks:
                if attempts >= max_attempts:
                    break

                # Try each threshold
                for cos_threshold in cosine_thresholds:
                    eucl_threshold = euclidean_thresholds[cos_threshold]

                    # Find path
                    path_result = self.find_path(
                        intervention["node_id"],
                        risk["node_id"],
                        max_length=6,
                        euclidean_threshold=eucl_threshold,
                    )

                    attempts += 1

                    if path_result is None:
                        continue

                    path_length = path_result["length"]

                    # Categorize by length
                    if path_length <= 3:
                        category = "short"
                        target = target_short
                    elif path_length <= 6:
                        category = "medium"
                        target = target_medium
                    else:
                        category = "long"
                        target = target_long

                    # Check if we need more of this category
                    if len(pathways[category]) >= target:
                        continue

                    # Add pathway
                    pathway = {
                        "pathway_id": len(pathways["short"])
                        + len(pathways["medium"])
                        + len(pathways["long"]),
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
                        f"  Found {category} pathway ({path_length} hops) at threshold {cos_threshold}: "
                        f"{intervention['name'][:30]}... → {risk['name'][:30]}..."
                    )

                    # Check if we're done
                    total = sum(len(pathways[cat]) for cat in pathways)
                    if total >= target_total:
                        break

                total = sum(len(pathways[cat]) for cat in pathways)
                if total >= target_total:
                    break

        # Combine all pathways
        all_pathways = pathways["short"] + pathways["medium"] + pathways["long"]

        print(f"\n{'=' * 80}")
        print("SAMPLING RESULTS")
        print(f"{'=' * 80}")
        print(f"Short pathways (≤3 hops): {len(pathways['short'])}/{target_short}")
        print(f"Medium pathways (4-6 hops): {len(pathways['medium'])}/{target_medium}")
        print(f"Long pathways (7+ hops): {len(pathways['long'])}/{target_long}")
        print(f"Total: {len(all_pathways)}/{target_total}")
        print(f"Attempts: {attempts}")

        # Threshold distribution
        threshold_counts = {}
        for pathway in all_pathways:
            t = pathway["threshold_cosine"]
            threshold_counts[t] = threshold_counts.get(t, 0) + 1

        print("\nThreshold distribution:")
        for threshold in sorted(threshold_counts.keys()):
            print(f"  ≥{threshold}: {threshold_counts[threshold]} pathways")

        return all_pathways

    def save_pathways_for_annotation(
        self, pathways, output_file="pathways_for_annotation.json"
    ):
        """
        Save pathways to JSON with context for manual annotation

        Format:
        {
            "pathway_id": 0,
            "intervention": {...},
            "risk": {...},
            "threshold": 0.85,
            "length": 5,
            "nodes": [
                {"node_id": 123, "name": "...", "description": "...", "source_url": "..."},
                ...
            ],
            "annotation": {
                "judgment": null,  # To be filled: "correct" | "incorrect" | "uncertain"
                "notes": "",
                "annotator": ""
            }
        }
        """
        print(f"\n{'=' * 80}")
        print("EXTRACTING PATHWAY CONTEXTS")
        print(f"{'=' * 80}")

        annotated_pathways = []

        for i, pathway in enumerate(pathways):
            print(f"  Extracting context for pathway {i + 1}/{len(pathways)}...")

            # Get detailed node information
            pathway_with_context = self.extract_pathway_context(pathway)

            # Add annotation template
            pathway_with_context["annotation"] = {
                "judgment": None,
                "notes": "",
                "annotator": "",
            }

            annotated_pathways.append(pathway_with_context)

        # Save to JSON
        with open(output_file, "w") as f:
            json.dump(annotated_pathways, f, indent=2)

        print(f"\n✓ Saved {len(annotated_pathways)} pathways to {output_file}")
        print("\nAnnotation instructions:")
        print(f"  1. Open {output_file}")
        print(
            "  2. For each pathway, set 'judgment' to: 'correct' | 'incorrect' | 'uncertain'"
        )
        print("  3. Add optional 'notes' for interesting observations")
        print("  4. Set 'annotator' to your name/ID")


def main():
    sampler = PathwaySampler()

    print("=" * 80)
    print("PATHWAY SAMPLING FOR MANUAL VALIDATION")
    print("=" * 80)

    try:
        sampler.client.ping()
        print("✓ Connected to FalkorDB")

        # Step 1: Find interventions and risks
        interventions = sampler.find_intervention_nodes(limit=200)
        risks = sampler.find_risk_nodes(limit=50)

        if len(interventions) == 0 or len(risks) == 0:
            print("\n✗ ERROR: Could not find intervention or risk nodes")
            print("Check node labels and properties in your graph")
            return

        # Step 2: Generate stratified sample
        pathways = sampler.stratified_sample_pathways(
            interventions,
            risks,
            cosine_thresholds=[0.8, 0.85, 0.9, 0.95],
            target_total=30,
            target_short=10,
            target_medium=15,
            target_long=5,
        )

        if len(pathways) == 0:
            print("\n✗ ERROR: No pathways found")
            print("Check that similarity edges exist in the graph")
            return

        # Step 3: Save for annotation
        sampler.save_pathways_for_annotation(pathways)

        print("\n" + "=" * 80)
        print("PATHWAY SAMPLING COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review pathways_for_annotation.json")
        print("  2. Manually annotate each pathway")
        print("  3. Calculate inter-annotator agreement (Cohen's κ)")
        print("  4. Analyze correctness by length/threshold/cluster")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
