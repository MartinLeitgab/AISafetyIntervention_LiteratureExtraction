"""
Check how FROM edges distribute across Source nodes
"""

import redis

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
graph = "AISafetyIntervention"


def query(cypher):
    result = client.execute_command("GRAPH.QUERY", graph, cypher, "--timeout", "120000")
    return result[1] if len(result) > 1 else []


# Count edges per source
print("FROM edges per Source (top 20):")
q = """
MATCH ()-[f:FROM]->(s:Source)
RETURN id(s), count(f) as edge_count
ORDER BY edge_count DESC
LIMIT 20
"""
results = query(q)
for row in results:
    source_id = int(row[0])
    count = int(row[1])
    print(f"  Source {source_id}: {count} edges")

# Distribution summary
print("\nDistribution of edges per source:")
q = """
MATCH ()-[f:FROM]->(s:Source)
WITH s, count(f) as edge_count
RETURN edge_count, count(s) as num_sources
ORDER BY edge_count DESC
LIMIT 10
"""
results = query(q)
for row in results:
    edges = int(row[0])
    sources = int(row[1])
    print(f"  {edges} edges: {sources} sources")

# Check sources with zero edges
print("\nSources with NO incoming FROM edges:")
q = """
MATCH (s:Source)
WHERE NOT (s)<-[:FROM]-()
RETURN count(s)
"""
zero_count = int(query(q)[0][0])
print(f"  {zero_count} sources have no nodes")

# Average nodes per source
print("\nAverage nodes per source:")
q = """
MATCH ()-[f:FROM]->(s:Source)
WITH s, count(f) as edge_count
RETURN avg(edge_count), min(edge_count), max(edge_count)
"""
results = query(q)
if results:
    avg, minval, maxval = float(results[0][0]), int(results[0][1]), int(results[0][2])
    print(f"  Avg: {avg:.1f}, Min: {minval}, Max: {maxval}")
