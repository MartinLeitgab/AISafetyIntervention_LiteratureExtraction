"""
Pathways per Source Histogram
Uses validated risk→intervention BFS, stops at intervention nodes
Compares: all pathways (conf≥1, mat≥1) vs high quality (conf≥3, mat≥3)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import redis
import numpy as np
from collections import defaultdict, deque, Counter
import json

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
graph = "AISafetyIntervention"


def query(q, timeout=120000):
    result = client.execute_command("GRAPH.QUERY", graph, q, "--timeout", str(timeout))
    return result[1] if len(result) > 1 else []


def load_graph(min_conf):
    """Load adjacency list with EDGE edges only"""
    adj = defaultdict(set)
    q = "MATCH (n) RETURN min(id(n)), max(id(n))"
    min_id, max_id = int(query(q)[0][0]), int(query(q)[0][1])

    cur, batch = min_id, 2000
    while cur <= max_id:
        q = f"MATCH (n)-[e:EDGE]-(m) WHERE id(n)>={cur} AND id(n)<{cur + batch} AND id(m)>id(n) AND e.edge_confidence>={min_conf} RETURN id(n),id(m)"
        for row in query(q):
            n1, n2 = int(row[0]), int(row[1])
            adj[n1].add(n2)
            adj[n2].add(n1)
        cur += batch
    return adj


def count_pathways_per_source(min_conf, min_mat):
    """
    Count pathways per source using validated method:
    - Start at risks
    - Stop when reaching interventions (allow intervention→intervention)
    Returns: pathway counts AND full pathway data for high-count sources
    """
    print(f"\nExtracting pathways (conf≥{min_conf}, mat≥{min_mat})...")

    # Load graph
    adj, edge_map = load_graph_with_edges(min_conf)

    # Get source ID range
    q = "MATCH (s:Source) RETURN min(id(s)), max(id(s))"
    min_id, max_id = int(query(q)[0][0]), int(query(q)[0][1])
    print(f"  Source ID range: {min_id} to {max_id}")

    # Load interventions and risks per source using RANGE SCANNING
    source_data = {}
    cur, batch = min_id, 500

    while cur <= max_id:
        # Interventions
        if min_mat >= 3:
            q = f"MATCH (s:Source)<-[:FROM]-(i:Intervention) WHERE id(s)>={cur} AND id(s)<{cur + batch} AND i.intervention_maturity>=3 RETURN id(s), id(i)"
        else:
            q = f"MATCH (s:Source)<-[:FROM]-(i:Intervention) WHERE id(s)>={cur} AND id(s)<{cur + batch} RETURN id(s), id(i)"

        for row in query(q):
            sid = int(row[0])
            if sid not in source_data:
                source_data[sid] = {"ints": set(), "risks": set()}
            source_data[sid]["ints"].add(int(row[1]))

        # Risks
        q = f"MATCH (s:Source)<-[:FROM]-(r:Concept) WHERE id(s)>={cur} AND id(s)<{cur + batch} AND r.concept_category='risk' RETURN id(s), id(r)"
        for row in query(q):
            sid = int(row[0])
            if sid not in source_data:
                source_data[sid] = {"ints": set(), "risks": set()}
            source_data[sid]["risks"].add(int(row[1]))

        cur += batch

    # Filter: only sources with both interventions and risks
    valid_sources = {
        sid: data
        for sid, data in source_data.items()
        if len(data["ints"]) > 0 and len(data["risks"]) > 0
    }

    print(f"  Sources with int+risk: {len(valid_sources)}")

    # Extract pathways
    pathway_counts = {}
    all_pathways = {}  # Store actual pathways for high-count sources

    for i, (sid, data) in enumerate(valid_sources.items()):
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(valid_sources)} sources...")

        ints = data["ints"]
        risks = data["risks"]

        pathways = []

        # Start from each RISK
        for risk_id in risks:
            visited = {risk_id: 0}
            queue = deque([risk_id])
            parent = {risk_id: None}
            found_ints = set()

            while queue:
                node = queue.popleft()
                dist = visited[node]

                # If we hit an intervention, record path and STOP
                if node in ints and node != risk_id:
                    if node not in found_ints:
                        found_ints.add(node)
                        # Reconstruct path
                        path, edges = [], []
                        curr = node
                        while curr is not None:
                            path.append(curr)
                            if parent.get(curr) is not None:
                                p = parent[curr]
                                edges.append(
                                    edge_map.get(
                                        (min(p, curr), max(p, curr)), "unknown"
                                    )
                                )
                            curr = parent.get(curr)
                        path = path[::-1]
                        edges = edges[::-1]
                        pathways.append({"path": path, "edges": edges})
                    continue

                # Expand neighbors
                for nb in adj.get(node, []):
                    if nb not in visited:
                        visited[nb] = dist + 1
                        parent[nb] = node
                        queue.append(nb)

        pathway_counts[sid] = len(pathways)

        # Store pathways if count > 19
        if len(pathways) > 19:
            all_pathways[sid] = pathways

    return pathway_counts, all_pathways


def load_graph_with_edges(min_conf):
    """Load adjacency list with edge types"""
    adj, edge_map = defaultdict(set), {}
    q = "MATCH (n) RETURN min(id(n)), max(id(n))"
    min_id, max_id = int(query(q)[0][0]), int(query(q)[0][1])

    cur, batch = min_id, 2000
    while cur <= max_id:
        q = f"MATCH (n)-[e:EDGE]-(m) WHERE id(n)>={cur} AND id(n)<{cur + batch} AND id(m)>id(n) AND e.edge_confidence>={min_conf} RETURN id(n),id(m),e.type"
        for row in query(q):
            n1, n2 = int(row[0]), int(row[1])
            edge_type = row[2] if row[2] else "EDGE"
            adj[n1].add(n2)
            adj[n2].add(n1)
            edge_map[(n1, n2)] = edge_type
        cur += batch
    return adj, edge_map


print("=" * 80)
print("PATHWAYS PER SOURCE HISTOGRAM")
print("=" * 80)

# Extract both conditions
counts_all, pathways_all = count_pathways_per_source(min_conf=1, min_mat=1)
counts_quality, pathways_quality = count_pathways_per_source(min_conf=3, min_mat=3)

# Print detailed pathways for sources with >19 pathways (quality cuts)
if pathways_quality:
    print("\n" + "=" * 80)
    print("DETAILED PATHWAYS (Sources with >19 pathways, quality cuts)")
    print("=" * 80)

    # Load node names and categories
    all_node_ids = set()
    for pathways in pathways_quality.values():
        for p in pathways:
            all_node_ids.update(p["path"])

    print(f"\nLoading {len(all_node_ids)} node names...")
    node_data = {}
    node_ids_list = list(all_node_ids)

    for i in range(0, len(node_ids_list), 100):
        batch = node_ids_list[i : i + 100]
        ids_str = ",".join(map(str, batch))

        # Concepts
        q = f"MATCH (c:Concept) WHERE id(c) IN [{ids_str}] RETURN id(c), c.name, c.concept_category"
        for row in query(q):
            node_data[int(row[0])] = {"name": row[1], "category": row[2]}

        # Interventions
        q = f"MATCH (i:Intervention) WHERE id(i) IN [{ids_str}] RETURN id(i), i.name"
        for row in query(q):
            node_data[int(row[0])] = {"name": row[1], "category": "intervention"}

    # Get source info and counts
    source_info = {}
    for sid in pathways_quality.keys():
        q = f"MATCH (s:Source) WHERE id(s)={sid} RETURN s.arxiv_id, s.title, s.url"
        result = query(q)
        if result:
            source_info[sid] = {
                "arxiv_id": result[0][0],
                "title": result[0][1],
                "url": result[0][2] if len(result[0]) > 2 else "unknown",
            }

        # Count interventions and risks
        q = f"MATCH (s:Source)<-[:FROM]-(i:Intervention) WHERE id(s)={sid} AND i.intervention_maturity>=3 RETURN count(i)"
        int_count = query(q)[0][0] if query(q) else 0

        q = f"MATCH (s:Source)<-[:FROM]-(r:Concept) WHERE id(s)={sid} AND r.concept_category='risk' RETURN count(r)"
        risk_count = query(q)[0][0] if query(q) else 0

        source_info[sid]["int_count"] = int_count
        source_info[sid]["risk_count"] = risk_count

    # Print pathways
    for sid, pathways in sorted(
        pathways_quality.items(), key=lambda x: len(x[1]), reverse=True
    ):
        info = source_info.get(sid, {})
        arxiv = info.get("arxiv_id", "unknown")
        title = info.get("title", "unknown")
        url = info.get("url", "unknown")
        int_count = info.get("int_count", 0)
        risk_count = info.get("risk_count", 0)

        print(f"\n{'=' * 80}")
        print(f"Source {sid} ({arxiv})")
        print(f"Title: {title}")
        print(f"URL: {url}")
        print(f"Interventions: {int_count}, Risks: {risk_count}")
        print(f"Pathways: {len(pathways)}")
        print(f"{'=' * 80}")

        for i, p in enumerate(pathways[:50], 1):  # Limit to 50 pathways
            print(f"\nPathway {i} ({len(p['path']) - 1} hops):")
            for j, nid in enumerate(p["path"]):
                data = node_data.get(
                    nid, {"name": f"Unknown({nid})", "category": "unknown"}
                )
                cat = data["category"]
                name = data["name"]

                # Color code by category
                if cat == "risk":
                    print(f"  [{cat.upper()}] {name}")
                elif cat == "intervention":
                    print(f"  [{cat.upper()}] {name}")
                else:
                    print(f"  [{cat}] {name}")

                # Print edge
                if j < len(p["edges"]):
                    print(f"     --[{p['edges'][j]}]-->")

# Build distributions
dist_all = Counter(counts_all.values())
dist_quality = Counter(counts_quality.values())

# Remove zero counts
dist_all = {k: v for k, v in dist_all.items() if k > 0}
dist_quality = {k: v for k, v in dist_quality.items() if k > 0}

print(f"\nAll pathways: {sum(counts_all.values()):,} total")
print(f"Quality pathways: {sum(counts_quality.values()):,} total")

# Create combined histogram
fig, ax = plt.subplots(figsize=(12, 6))

# Get bin edges (linear bins for x-axis)
all_pathway_counts = set(dist_all.keys()) | set(dist_quality.keys())
max_count = max(all_pathway_counts)
bins = np.linspace(1, max_count, 50)

# Histogram data
pathway_vals_all = []
for k, v in dist_all.items():
    pathway_vals_all.extend([k] * v)

pathway_vals_quality = []
for k, v in dist_quality.items():
    pathway_vals_quality.extend([k] * v)

# Plot
ax.hist(
    pathway_vals_all,
    bins=bins,
    alpha=0.5,
    color="gray",
    label="All mat, conf≥1",
    edgecolor="black",
    linewidth=0.5,
)
ax.hist(
    pathway_vals_quality,
    bins=bins,
    alpha=0.7,
    color="orange",
    label="Mat≥3, conf≥3",
    edgecolor="black",
    linewidth=0.5,
)

# Log y-axis only
ax.set_yscale("log")
ax.set_xlabel("Pathways per Source", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Sources", fontsize=13, fontweight="bold")
ax.set_title("Pathways per Source Distribution", fontsize=15, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("pathways_per_source_histogram.png", dpi=300, bbox_inches="tight")
print("\n✓ pathways_per_source_histogram.png")

# Save data
output = {
    "all": {
        "total_pathways": sum(counts_all.values()),
        "sources_with_pathways": len([c for c in counts_all.values() if c > 0]),
        "distribution": {int(k): int(v) for k, v in dist_all.items()},
    },
    "quality": {
        "total_pathways": sum(counts_quality.values()),
        "sources_with_pathways": len([c for c in counts_quality.values() if c > 0]),
        "distribution": {int(k): int(v) for k, v in dist_quality.items()},
    },
}

with open("pathways_per_source_data.json", "w") as f:
    json.dump(output, f, indent=2)

print("✓ pathways_per_source_data.json")
