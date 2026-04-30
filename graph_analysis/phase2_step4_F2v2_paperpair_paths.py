"""
phase2_step4_F2v2_paperpair_paths.py [rev8]

v2 of paper-pair enumeration. Removes vpn_custom restriction (which was
BFS-derived) and enforces first-hop and minimum-length constraints from
the user's spec.

Changes vs F2 v1:
  - No vpn_custom pre-restriction. Runs on full graph with edge filters.
  - First hop from a risk must be EDGE (not SIM).
  - First hop target must be a body subtype (not risk, not intervention).
  - Reject paths shorter than 3 hops (length>=3, ≥4 nodes total).

Constraint set:
  Edge filters:
    - EDGE edges: edge_confidence >= 3
    - SIM edges: cos_sim >= 0.9

  Endpoint filter:
    - Intervention endpoint: intervention_maturity >= 3
    - Path[0]: any risk node in the graph
    - Min path length: 3 hops (4 nodes: R -> B1 -> B2 -> I or longer)

  Path-level constraints:
    - Simple paths (no node repeats within a path; paths can share nodes
      across different (R, I) pairs or alternate routes)
    - consim1: no two consecutive SIM edges
    - First hop: EDGE only, target must be body subtype
    - Custom single-risk: no risk nodes other than path[0]
    - Custom single-intervention: stop at first maturity>=3 intervention

Inputs:
  step1_load_and_parse.../graph_node_attributes.pkl
  step1_load_and_parse.../graph_edge_data.pkl

Outputs:
  phase1_rawpathsfiles/paths_paperpair_v2_sim0.9.jsonl
  phase2_results/step4_finalanalysis/step4_connectivity/paperpair_v2_summary.txt
"""

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
STEP1_DIR = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
OUT_DIR = ROOT / "phase2_results/step4_finalanalysis/step4_connectivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = PATHS_DIR / "paths_paperpair_v2_sim0.9.jsonl"
SUMMARY = OUT_DIR / "paperpair_v2_summary.txt"

SIM_THRESHOLD = 0.9
EDGE_CONFIDENCE_MIN = 3
INTERVENTION_MATURITY_MIN = 3
MIN_PATH_LENGTH = 3  # >=3 hops = >=4 nodes = R->B1->B2->I or longer
MAX_PATH_LENGTH = 20  # cap at 20 hops to prevent combinatorial explosion
MAX_TOTAL_PATHS = 1_000_000  # global stop after 1M paths to keep run bounded

# node_attrs concept_category uses space form (different from cluster_memberships
# pkl key which uses underscore form). Match space form for node_attrs lookup.
BODY_SUBTYPES = {
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
}


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


print("=" * 70)
print("Phase 2 Step 4 rev8 — paper-pair path enumeration v2 (full graph)")
print("=" * 70)

t0 = time.time()
print("Loading PKL files...")
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
print(
    f"  Loaded {len(node_attrs):,} nodes, {len(edge_data):,} edges  ({time.time() - t0:.1f}s)"
)


# ─── Categorise nodes ────────────────────────────────────────────────────────
t1 = time.time()
print("\nCategorising nodes (full graph)...")
risk_nodes = set()
intervention_nodes = set()
body_nodes = set()  # nodes with concept_category in BODY_SUBTYPES
maturity3_interventions = set()

for nid, attrs in node_attrs.items():
    if attrs.get("type") == "intervention":
        intervention_nodes.add(int(nid))
        if int(attrs.get("intervention_maturity", 0) or 0) >= INTERVENTION_MATURITY_MIN:
            maturity3_interventions.add(int(nid))
    else:
        cat = str(attrs.get("concept_category", ""))
        if cat == "risk":
            risk_nodes.add(int(nid))
        elif cat in BODY_SUBTYPES:
            body_nodes.add(int(nid))

print(
    f"  risk nodes: {len(risk_nodes):,}"
    f"\n  body nodes: {len(body_nodes):,}"
    f"\n  intervention nodes (any maturity): {len(intervention_nodes):,}"
    f"\n  intervention nodes (maturity>=3): {len(maturity3_interventions):,}"
    f"  ({time.time() - t1:.1f}s)"
)

# ─── Build adjacency on FULL graph (no VPN restriction) ──────────────────────
t2 = time.time()
print("\nBuilding adjacency on full graph (EDGE conf>=3, SIM cos_sim>=0.9)...")
adj_edge = defaultdict(set)
adj_sim = defaultdict(set)
n_edge = 0
n_sim = 0

for e in edge_data:
    try:
        s = int(e["source"])
        t = int(e["target"])
    except (ValueError, TypeError, KeyError):
        continue
    etype = str(e.get("type", "")).upper()
    if etype == "EDGE":
        conf = e.get("confidence")
        try:
            if conf is None or int(conf) < EDGE_CONFIDENCE_MIN:
                continue
        except (ValueError, TypeError):
            continue
        adj_edge[s].add(t)
        adj_edge[t].add(s)
        n_edge += 1
    elif etype == "SIMILARITY":
        score = e.get("similarity_score")
        if score is None or cos_sim_from_score(score) < SIM_THRESHOLD:
            continue
        adj_sim[s].add(t)
        adj_sim[t].add(s)
        n_sim += 1

print(
    f"  EDGE edges (undirected, counted once each direction): {n_edge:,}"
    f"\n  SIM edges (undirected, counted once each direction): {n_sim:,}"
    f"\n  ({time.time() - t2:.1f}s)"
)

# ─── DFS enumeration ──────────────────────────────────────────────────────────
t3 = time.time()
print(f"\nEnumerating paths from {len(risk_nodes):,} risk nodes...")
sys.setrecursionlimit(50000)

risk_nodes_list = sorted(risk_nodes)
total_paths = 0
ri_pairs_emitted = set()
length_histogram = Counter()
hit_global_cap = False
out_f = open(OUTPUT, "w")
progress_step = max(1, len(risk_nodes_list) // 50)


def emit_path(path, edge_types):
    """Write a path to output if it satisfies min-length constraint."""
    global total_paths, hit_global_cap
    L = len(path) - 1
    if L < MIN_PATH_LENGTH:
        return False
    cats = []
    for nid in path:
        a = node_attrs.get(int(nid), {})
        if a.get("type") == "intervention":
            cats.append("intervention")
        else:
            cats.append(str(a.get("concept_category", "")))
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
            f"\n  HIT GLOBAL CAP of {MAX_TOTAL_PATHS:,} paths at risk index {r_idx} of {len(risk_nodes_list)}. Stopping.",
            flush=True,
        )
        break

    if r_idx % progress_step == 0:
        elapsed = time.time() - t3
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

    first_hop_targets = []
    for nb in adj_edge.get(R, ()):
        if nb in visited:
            continue
        if nb in body_nodes:
            first_hop_targets.append(nb)

    def dfs(curr, prev_was_sim, depth):
        # global stop
        if hit_global_cap:
            return
        # max length cap: depth = number of edges in path so far (= len(path)-1).
        # If depth >= MAX_PATH_LENGTH, can't extend further; only emit if curr
        # is an intervention.
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
            return  # stop expanding; this path arm cannot reach an interv at depth<=L
        # EDGE neighbors
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
        # SIM neighbors (only if prev wasn't SIM)
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

    for body_nb in first_hop_targets:
        if hit_global_cap:
            break
        visited.add(body_nb)
        parent[body_nb] = R
        parent_edge_type[body_nb] = "EDGE"
        dfs(body_nb, False, depth=1)
        visited.discard(body_nb)
        parent.pop(body_nb, None)
        parent_edge_type.pop(body_nb, None)

out_f.close()

print(
    f"\nDFS complete: {total_paths:,} total paths, {len(ri_pairs_emitted):,} unique (R, I) pairs"
    f"\n  ({time.time() - t3:.1f}s)"
)
print(f"\nWritten: {OUTPUT}")

with open(SUMMARY, "w") as f:
    f.write("rev8 paper-pair v2 path enumeration summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"sim_threshold: {SIM_THRESHOLD}\n")
    f.write(f"edge_confidence_min: {EDGE_CONFIDENCE_MIN}\n")
    f.write(f"intervention_maturity_min: {INTERVENTION_MATURITY_MIN}\n")
    f.write(f"min_path_length: {MIN_PATH_LENGTH}\n")
    f.write(f"max_path_length: {MAX_PATH_LENGTH}\n")
    f.write(f"max_total_paths: {MAX_TOTAL_PATHS}\n")
    f.write(f"hit_global_cap: {hit_global_cap}\n")
    f.write("VPN_restriction: NONE (full graph, edge-filtered)\n")
    f.write("first_hop_constraint: EDGE only, target body-subtype only\n")
    f.write("\n")
    f.write(f"risk_nodes_in_full_graph: {len(risk_nodes)}\n")
    f.write(f"body_nodes_in_full_graph: {len(body_nodes)}\n")
    f.write(f"intervention_nodes (any maturity): {len(intervention_nodes)}\n")
    f.write(f"maturity3_interventions: {len(maturity3_interventions)}\n")
    f.write(f"edge_edges_after_filter: {n_edge // 2}\n")
    f.write(f"sim_edges_after_filter: {n_sim // 2}\n")
    f.write(f"total_paths_emitted: {total_paths}\n")
    f.write(f"unique_ri_pairs: {len(ri_pairs_emitted)}\n")
    f.write(f"runtime_seconds: {time.time() - t0:.1f}\n")
    f.write("\n")
    f.write("Path length histogram (length -> n_paths):\n")
    for L in sorted(length_histogram.keys()):
        f.write(f"  L={L:2d}: {length_histogram[L]:>10,}\n")

print("\nPath length histogram:")
for L in sorted(length_histogram.keys()):
    print(f"  L={L:2d}: {length_histogram[L]:>10,}")

print(f"Summary written: {SUMMARY}")
print("\nDONE.")
