"""
Complete source-level pathway analysis
Uses individual queries per source (proven working)
"""

import redis
from collections import defaultdict, deque, Counter
import json
import time

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
graph = "AISafetyIntervention"


def query(q, timeout=120000):
    result = client.execute_command("GRAPH.QUERY", graph, q, "--timeout", str(timeout))
    return result[1] if len(result) > 1 else []


def load_edges(min_conf):
    adj = defaultdict(set)
    q = "MATCH (n) RETURN min(id(n)), max(id(n))"
    min_id, max_id = int(query(q)[0][0]), int(query(q)[0][1])

    current, batch = min_id, 2000
    while current <= max_id:
        q = f"MATCH (n)-[e:EDGE]-(m) WHERE id(n)>={current} AND id(n)<{current + batch} AND id(m)>id(n) AND e.edge_confidence>={min_conf} RETURN id(n),id(m)"
        for row in query(q):
            n1, n2 = int(row[0]), int(row[1])
            adj[n1].add(n2)
            adj[n2].add(n1)
        current += batch
    return adj


def count_paths(starts, targets, adj):
    target_set = set(targets)
    found = 0
    for start in starts:
        visited, queue = {start}, deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    if nb in target_set:
                        found += 1
    return found


print("=" * 80)
print("COMPLETE SOURCE PATHWAY ANALYSIS")
print("=" * 80)

# Get ALL source IDs using batching
print("\nFinding all sources...")
q = "MATCH (s:Source) RETURN min(id(s)), max(id(s))"
result = query(q)
min_id, max_id = int(result[0][0]), int(result[0][1])

all_sources = []
current_id, batch_size = min_id, 1000

while current_id <= max_id:
    q = f"MATCH (s:Source) WHERE id(s) >= {current_id} AND id(s) < {current_id + batch_size} RETURN id(s)"
    all_sources.extend([int(r[0]) for r in query(q)])
    current_id += batch_size

print(f"  Total sources: {len(all_sources)}")

# Load per-source data
print("\nLoading source node mappings...")
source_data = {"mature": {}, "all": {}}

start_time = time.time()
for i, sid in enumerate(all_sources):
    if (i + 1) % 500 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(all_sources) - (i + 1)) / rate / 60
        print(
            f"  {i + 1}/{len(all_sources)} sources ({rate:.0f}/sec, ETA {eta:.1f}min)"
        )

    # Get mature interventions
    q = f"MATCH (s:Source)<-[:FROM]-(i:Intervention) WHERE id(s)={sid} AND i.intervention_maturity>=3 RETURN id(i)"
    ints_mature = [int(r[0]) for r in query(q)]

    # Get all interventions
    q = f"MATCH (s:Source)<-[:FROM]-(i:Intervention) WHERE id(s)={sid} RETURN id(i)"
    ints_all = [int(r[0]) for r in query(q)]

    # Get risks
    q = f"MATCH (s:Source)<-[:FROM]-(r:Concept) WHERE id(s)={sid} AND r.concept_category='risk' RETURN id(r)"
    risks = [int(r[0]) for r in query(q)]

    if ints_mature and risks:
        source_data["mature"][sid] = {"ints": ints_mature, "risks": risks}

    if ints_all and risks:
        source_data["all"][sid] = {"ints": ints_all, "risks": risks}

print(f"\n  Mature sources with int+risk: {len(source_data['mature'])}")
print(f"  All sources with int+risk: {len(source_data['all'])}")

# Analyze paths at each confidence
results = {}

for mat_label in ["mature", "all"]:
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: MATURITY {'>=3' if mat_label == 'mature' else 'ALL'}")
    print(f"{'=' * 80}")

    results[mat_label] = {}
    sources = source_data[mat_label]

    for min_conf in [1, 2, 3, 4, 5]:
        print(f"\nConfidence ≥{min_conf}:")

        start = time.time()
        adj = load_edges(min_conf)
        print(f"  Loaded graph ({time.time() - start:.1f}s)")

        path_counts = {}
        total = 0

        start = time.time()
        for i, (sid, data) in enumerate(sources.items()):
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - start)
                eta = (len(sources) - (i + 1)) / rate / 60
                print(f"    {i + 1}/{len(sources)} ({rate:.0f}/sec, ETA {eta:.1f}min)")

            paths = count_paths(data["ints"], data["risks"], adj)
            path_counts[sid] = paths
            total += paths

        dist = Counter(path_counts.values())

        results[mat_label][f"conf>={min_conf}"] = {
            "total_paths": total,
            "sources_with_paths": sum(1 for c in path_counts.values() if c > 0),
            "distribution": dict(dist),
            "per_source": path_counts,
        }

        print(f"  Total paths: {total:,}")
        print(
            f"  Sources with ≥1 path: {sum(1 for c in path_counts.values() if c > 0)}/{len(sources)}"
        )
        print(f"  Top path counts: {sorted(dist.items(), reverse=True)[:5]}")

# Summary
print(f"\n{'=' * 80}")
print("EXCLUSION BY CONFIDENCE")
print(f"{'=' * 80}")

for mat_label in ["mature", "all"]:
    print(f"\n{'MATURITY ≥3' if mat_label == 'mature' else 'ALL MATURITIES'}:")
    baseline = results[mat_label]["conf>=1"]["total_paths"]
    print(f"  Baseline (conf≥1): {baseline:,}")

    for min_conf in [2, 3, 4, 5]:
        count = results[mat_label][f"conf>={min_conf}"]["total_paths"]
        excl = baseline - count
        pct = 100 * excl / baseline if baseline > 0 else 0
        print(f"  Conf≥{min_conf}: {count:,} ({excl:,} excluded, {pct:.1f}%)")

with open("source_pathways_final.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Saved source_pathways_final.json")
