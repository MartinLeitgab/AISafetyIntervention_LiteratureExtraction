"""
Debug SIMILARITY edge loading
"""

import redis
import json
from collections import defaultdict

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
graph = "AISafetyIntervention"


def query(q, timeout=120000):
    result = client.execute_command("GRAPH.QUERY", graph, q, "--timeout", str(timeout))
    return result[1] if len(result) > 1 else []


# Load source data
with open("source_pathways_final.json") as f:
    source_data = json.load(f)

# Get mature sources
all_mature = list(
    set(int(sid) for sid in source_data["mature"]["conf>=1"]["per_source"].keys())
)
print(f"Total mature sources: {len(all_mature)}")

# Check SIMILARITY edge name
print("\nChecking SIMILARITY edge type...")
q = "MATCH ()-[e:SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST]->() RETURN type(e) LIMIT 1"
res = query(q)
print(f"SIMILARITY edge type: {res[0][0] if res else 'NOT FOUND'}")

# Get node ID range
q = "MATCH (n) RETURN min(id(n)), max(id(n))"
min_id, max_id = int(query(q)[0][0]), int(query(q)[0][1])
print(f"\nNode ID range: {min_id} to {max_id}")

# Count SIMILARITY edges - convert cosine threshold to euclidean
print("\nCounting SIMILARITY edges (pre-filtered ≥0.8 at creation)...")
import numpy as np  # noqa: E402

# Relationship already filtered at 0.8, but can apply stricter thresholds
for cos_thresh in [0.80, 0.85, 0.90, 0.95]:
    eucl_thresh = np.sqrt(2 * (1 - cos_thresh))
    current, batch = min_id, 2000
    count = 0
    while current <= max_id:
        q = f"MATCH (n)-[e:SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST]-(m) WHERE id(n)>={current} AND id(n)<{current + batch} AND id(m)>id(n) AND e.score<{eucl_thresh} RETURN count(*)"
        res = query(q)
        count += res[0][0] if res else 0
        current += batch
    print(f"  Cosine ≥{cos_thresh} (Euclidean <{eucl_thresh:.4f}): {count:,} edges")

# Load EDGE graph (conf≥3)
print("\nLoading EDGE graph (conf≥3)...")
adj_edge = defaultdict(set)

current, batch = min_id, 2000
edge_count = 0
while current <= max_id:
    q = f"MATCH (n)-[e:EDGE]-(m) WHERE id(n)>={current} AND id(n)<{current + batch} AND id(m)>id(n) AND e.edge_confidence>=3 RETURN id(n),id(m)"
    for row in query(q):
        adj_edge[int(row[0])].add(int(row[1]))
        adj_edge[int(row[1])].add(int(row[0]))
        edge_count += 1
    current += batch
print(f"  EDGE edges (conf≥3): {edge_count:,}")

# Load EDGE + SIMILARITY graph
print("\nLoading EDGE + SIMILARITY graph (conf≥3, cosine≥0.8)...")
adj_combined = defaultdict(set)
for n, nbs in adj_edge.items():
    adj_combined[n] = nbs.copy()

import numpy as np  # noqa: E402

cos_thresh = 0.80
eucl_thresh = np.sqrt(2 * (1 - cos_thresh))
current = min_id
sim_count = 0
while current <= max_id:
    q = f"MATCH (n)-[e:SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST]-(m) WHERE id(n)>={current} AND id(n)<{current + batch} AND id(m)>id(n) AND e.score<{eucl_thresh} RETURN id(n),id(m)"
    for row in query(q):
        n1, n2 = int(row[0]), int(row[1])
        adj_combined[n1].add(n2)
        adj_combined[n2].add(n1)
        sim_count += 1
    current += batch
print(f"  SIMILARITY edges (≥0.80): {sim_count:,}")

# Check if any mature sources gain connectivity
print("\nChecking mature source connectivity...")
sources_with_edge = 0
sources_with_sim = 0

for sid in all_mature[:100]:  # Sample 100 sources
    # Check interventions
    q = f"MATCH (s:Source)<-[:FROM]-(i:Intervention) WHERE id(s)={sid} AND i.intervention_maturity>=3 RETURN id(i)"
    ints = [int(r[0]) for r in query(q)]

    # Check risks
    q = f"MATCH (s:Source)<-[:FROM]-(r:Concept) WHERE id(s)={sid} AND r.concept_category='risk' RETURN id(r)"
    risks = [int(r[0]) for r in query(q)]

    if not ints or not risks:
        continue

    # Check if connected via EDGE
    has_edge_path = False
    for int_id in ints:
        if int_id in adj_edge and any(r in adj_edge for r in risks):
            has_edge_path = True
            break

    # Check if connected via EDGE+SIM
    has_sim_path = False
    for int_id in ints:
        if int_id in adj_combined and any(r in adj_combined for r in risks):
            has_sim_path = True
            break

    if has_edge_path:
        sources_with_edge += 1
    if has_sim_path:
        sources_with_sim += 1

print("Sample of 100 sources:")
print(f"  Connected via EDGE only: {sources_with_edge}")
print(f"  Connected via EDGE+SIM: {sources_with_sim}")
print(f"  New connections from SIM: {sources_with_sim - sources_with_edge}")
