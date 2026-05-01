"""
phase2_step4_F2v4_hopwise_falkordb.py [rev8]

v4 of hop-wise path enumeration. Queries FalkorDB live (instead of PKL) so the
graph snapshot is current. Quality filters applied in Cypher (no truncation):
  - EDGE edges: edge_confidence >= 3
  - SIM edges: score < 0.6325 (= cos_sim >= 0.9)
  - Intervention endpoint: intervention_maturity >= 3

Path-level caps (data-faithful — last-resort safeguards, tracked in summary):
  - Min path length: 3 (intentional quality cut, not a sampling cap)
  - Max path length: 50 (raised from v3's 20 to recover ~4,400 length>20 R-I pairs)
  - Global cap: 10,000,000 paths (raised from v3's 1M)

If the global cap fires, the summary records the risk index reached and the
fact of incomplete enumeration. Future re-run with higher cap or downstream
cluster-level enumeration (post Task #7 body recluster) recovers full data.

The 10,752 R-I pairs missing from F2v3 at length 3-20 vs custom-BFS are likely
due to graph-data drift between the PKL snapshot and live FalkorDB. v4 queries
FalkorDB directly to eliminate that source of discrepancy.

Outputs:
  graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_sim0.9.jsonl
  graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity/hopwise_v4_summary.txt
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import redis

ROOT = Path(__file__).parent
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
OUT_DIR = ROOT / "phase2_results/step4_finalanalysis/step4_connectivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = PATHS_DIR / "paths_hopwise_v4_sim0.9.jsonl"
SUMMARY = OUT_DIR / "hopwise_v4_summary.txt"

SIM_THRESHOLD = 0.9  # cos_sim >= 0.9 → euclidean score < sqrt(2*(1-0.9)) = 0.6325
SIM_SCORE_MAX = (2 * (1 - SIM_THRESHOLD)) ** 0.5  # = 0.6325
EDGE_CONFIDENCE_MIN = 3
INTERVENTION_MATURITY_MIN = 3
MIN_PATH_LENGTH = 3
MAX_PATH_LENGTH = 50  # raised from 20
MAX_TOTAL_PATHS = 10_000_000  # raised from 1M

BODY_SUBTYPES = {
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
}

GRAPH = "AISafetyIntervention"


def query(client, cypher, timeout_ms=600000):
    return client.execute_command(
        "GRAPH.QUERY", GRAPH, cypher, "--timeout", str(timeout_ms)
    )


print("=" * 70)
print("Phase 2 Step 4 rev8 — F2v4 hop-wise from FalkorDB live")
print("=" * 70)
print(
    f"  SIM threshold: cos_sim >= {SIM_THRESHOLD} (euclidean score < {SIM_SCORE_MAX:.4f})"
)
print(f"  EDGE confidence: >= {EDGE_CONFIDENCE_MIN}")
print(f"  Intervention maturity: >= {INTERVENTION_MATURITY_MIN}")
print(f"  Path length: {MIN_PATH_LENGTH} - {MAX_PATH_LENGTH}")
print(f"  Global path cap: {MAX_TOTAL_PATHS:,}")

t0 = time.time()
client = redis.Redis(host="localhost", port=6379, decode_responses=True)
print(f"\nFalkorDB ping: {client.execute_command('PING')}")

# ─── Categorise nodes (queries FalkorDB live) ────────────────────────────────
print("\nCategorising nodes from FalkorDB...")
t1 = time.time()


def query_node_ids(cypher_filter):
    res = query(client, cypher_filter)
    rows = res[1] if len(res) > 1 else []
    return {int(row[0]) for row in rows}


risk_nodes = query_node_ids(
    "MATCH (n:Concept) WHERE n.concept_category = 'risk' RETURN id(n)"
)
print(f"  risk_nodes: {len(risk_nodes):,}")

body_nodes = set()
for st in BODY_SUBTYPES:
    body_set = query_node_ids(
        f"MATCH (n:Concept) WHERE n.concept_category = '{st}' RETURN id(n)"
    )
    body_nodes |= body_set
print(f"  body_nodes (5 subtypes): {len(body_nodes):,}")

intervention_nodes = query_node_ids("MATCH (n:Intervention) RETURN id(n)")
print(f"  intervention_nodes (any maturity): {len(intervention_nodes):,}")

maturity3_interventions = query_node_ids(
    f"MATCH (n:Intervention) WHERE n.intervention_maturity >= {INTERVENTION_MATURITY_MIN} RETURN id(n)"
)
print(f"  intervention_nodes (maturity>=3): {len(maturity3_interventions):,}")
print(f"  ({time.time() - t1:.1f}s)")

# ─── Build adjacency from FalkorDB queries ───────────────────────────────────
print("\nBuilding adjacency from FalkorDB...")
t2 = time.time()
adj_edge = defaultdict(set)
adj_sim = defaultdict(set)

# EDGE edges with conf >= 3 (batched by source-id range)
print("  Loading EDGE edges (conf>=3)...")
n_min = query(client, "MATCH (n) RETURN min(id(n))")
n_max = query(client, "MATCH (n) RETURN max(id(n))")
min_id = int(n_min[1][0][0])
max_id = int(n_max[1][0][0])
print(f"    node id range: [{min_id}, {max_id}]")

batch = 5000
n_edge = 0
cur = min_id
while cur <= max_id:
    cy = (
        f"MATCH (n)-[e:EDGE]-(m) "
        f"WHERE id(n) >= {cur} AND id(n) < {cur + batch} AND id(m) > id(n) "
        f"AND e.edge_confidence >= {EDGE_CONFIDENCE_MIN} "
        f"RETURN id(n), id(m)"
    )
    res = query(client, cy)
    rows = res[1] if len(res) > 1 else []
    for row in rows:
        s, t = int(row[0]), int(row[1])
        adj_edge[s].add(t)
        adj_edge[t].add(s)
        n_edge += 1
    cur += batch
print(f"    EDGE edges (undirected pairs): {n_edge:,}  ({time.time() - t2:.1f}s)")

# SIM edges (2150_NEAREST, score < 0.6325 = cos_sim >= 0.9)
print("  Loading SIM edges (cos_sim>=0.9)...")
t3 = time.time()
n_sim = 0
cur = min_id
while cur <= max_id:
    cy = (
        f"MATCH (n)-[e:SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST]-(m) "
        f"WHERE id(n) >= {cur} AND id(n) < {cur + batch} AND id(m) > id(n) "
        f"AND e.score < {SIM_SCORE_MAX} "
        f"RETURN id(n), id(m)"
    )
    res = query(client, cy)
    rows = res[1] if len(res) > 1 else []
    for row in rows:
        s, t = int(row[0]), int(row[1])
        adj_sim[s].add(t)
        adj_sim[t].add(s)
        n_sim += 1
    cur += batch
print(f"    SIM edges (undirected pairs): {n_sim:,}  ({time.time() - t3:.1f}s)")

# ─── DFS enumeration ──────────────────────────────────────────────────────────
t4 = time.time()
print(f"\nEnumerating paths from {len(risk_nodes):,} risk nodes...")
sys.setrecursionlimit(50000)

risk_nodes_list = sorted(risk_nodes)
total_paths = 0
ri_pairs_emitted = set()
length_histogram = Counter()
hit_global_cap = False
risks_processed = 0
out_f = open(OUTPUT, "w")
progress_step = max(1, len(risk_nodes_list) // 50)


def emit_path(path, edge_types):
    global total_paths, hit_global_cap
    L = len(path) - 1
    if L < MIN_PATH_LENGTH:
        return False
    cats = []
    for nid in path:
        if nid in risk_nodes:
            cats.append("risk")
        elif nid in intervention_nodes:
            cats.append("intervention")
        elif nid in body_nodes:
            cats.append("body")
        else:
            cats.append("?")
    out_f.write(
        json.dumps(
            {
                "path": path,
                "edge_types": edge_types,
                "categories": cats,
                "length": L,
            }
        )
        + "\n"
    )
    total_paths += 1
    length_histogram[L] += 1
    ri_pairs_emitted.add((path[0], path[-1]))
    if total_paths >= MAX_TOTAL_PATHS:
        hit_global_cap = True
    return True


for r_idx, R in enumerate(risk_nodes_list):
    if hit_global_cap:
        print(
            f"\n  HIT GLOBAL CAP {MAX_TOTAL_PATHS:,} at risk index {r_idx} of {len(risk_nodes_list):,}",
            flush=True,
        )
        break
    risks_processed = r_idx + 1

    if r_idx % progress_step == 0:
        elapsed = time.time() - t4
        rate = (r_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(risk_nodes_list) - r_idx - 1) / rate if rate > 0 else 0
        print(
            f"  [{r_idx + 1}/{len(risk_nodes_list)}] R={R} | total_paths={total_paths:,} "
            f"| rate={rate:.1f}/s | ETA={eta / 60:.1f}min",
            flush=True,
        )

    visited = {R}
    parent = {R: None}
    parent_edge_type = {R: None}

    # First hop: EDGE OR SIM, target body subtype only
    first_hop_targets = []
    for nb in adj_edge.get(R, ()):
        if nb in visited:
            continue
        if nb in body_nodes:
            first_hop_targets.append((nb, "EDGE"))
    for nb in adj_sim.get(R, ()):
        if nb in visited:
            continue
        if nb in body_nodes:
            first_hop_targets.append((nb, "SIM"))

    def dfs(curr, prev_was_sim, depth):
        if hit_global_cap:
            return
        if curr in maturity3_interventions:
            path = []
            edge_types = []
            cur = curr
            while cur is not None:
                path.append(cur)
                if parent_edge_type.get(cur) is not None:
                    edge_types.append(parent_edge_type[cur])
                cur = parent.get(cur)
            path = path[::-1]
            edge_types = edge_types[::-1]
            emit_path(path, edge_types)
            return
        if depth >= MAX_PATH_LENGTH:
            return
        for nb in adj_edge.get(curr, ()):
            if nb in visited:
                continue
            if nb in risk_nodes:
                continue
            visited.add(nb)
            parent[nb] = curr
            parent_edge_type[nb] = "EDGE"
            dfs(nb, False, depth + 1)
            visited.discard(nb)
            parent.pop(nb, None)
            parent_edge_type.pop(nb, None)
            if hit_global_cap:
                return
        if not prev_was_sim:
            for nb in adj_sim.get(curr, ()):
                if nb in visited:
                    continue
                if nb in risk_nodes:
                    continue
                visited.add(nb)
                parent[nb] = curr
                parent_edge_type[nb] = "SIM"
                dfs(nb, True, depth + 1)
                visited.discard(nb)
                parent.pop(nb, None)
                parent_edge_type.pop(nb, None)
                if hit_global_cap:
                    return

    for body_nb, first_etype in first_hop_targets:
        if hit_global_cap:
            break
        visited.add(body_nb)
        parent[body_nb] = R
        parent_edge_type[body_nb] = first_etype
        dfs(body_nb, prev_was_sim=(first_etype == "SIM"), depth=1)
        visited.discard(body_nb)
        parent.pop(body_nb, None)
        parent_edge_type.pop(body_nb, None)

out_f.close()

print(
    f"\nDFS complete: {total_paths:,} paths, {len(ri_pairs_emitted):,} unique (R, I) pairs"
    f"\n  hit_global_cap={hit_global_cap}, risks_processed={risks_processed}/{len(risk_nodes_list)}"
    f"\n  ({time.time() - t4:.1f}s, total run {time.time() - t0:.1f}s)"
)
print(f"\nWritten: {OUTPUT}")

with open(SUMMARY, "w") as f:
    f.write("rev8 hop-wise v4 (FalkorDB live) summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"sim_threshold: {SIM_THRESHOLD}\n")
    f.write(f"sim_score_max (euclidean): {SIM_SCORE_MAX:.6f}\n")
    f.write(f"edge_confidence_min: {EDGE_CONFIDENCE_MIN}\n")
    f.write(f"intervention_maturity_min: {INTERVENTION_MATURITY_MIN}\n")
    f.write(f"min_path_length: {MIN_PATH_LENGTH}\n")
    f.write(f"max_path_length: {MAX_PATH_LENGTH}\n")
    f.write(f"max_total_paths: {MAX_TOTAL_PATHS}\n")
    f.write(f"hit_global_cap: {hit_global_cap}\n")
    f.write(f"risks_processed: {risks_processed}/{len(risk_nodes_list)}\n")
    f.write("first_hop_constraint: EDGE or SIM, target body-subtype only\n")
    f.write(f"graph_source: FalkorDB live ({GRAPH}, container 'reverent_faraday')\n\n")
    f.write(f"risk_nodes_in_graph: {len(risk_nodes)}\n")
    f.write(f"body_nodes_in_graph: {len(body_nodes)}\n")
    f.write(f"intervention_nodes (any maturity): {len(intervention_nodes)}\n")
    f.write(f"maturity3_interventions: {len(maturity3_interventions)}\n")
    f.write(f"edge_edges_after_filter: {n_edge}\n")
    f.write(f"sim_edges_after_filter: {n_sim}\n")
    f.write(f"total_paths_emitted: {total_paths}\n")
    f.write(f"unique_ri_pairs: {len(ri_pairs_emitted)}\n")
    f.write(f"runtime_seconds: {time.time() - t0:.1f}\n")
    f.write("\nPath length histogram (length -> n_paths):\n")
    for L in sorted(length_histogram.keys()):
        f.write(f"  L={L:2d}: {length_histogram[L]:>10,}\n")

print(f"Summary written: {SUMMARY}")
print("\nDONE.")
