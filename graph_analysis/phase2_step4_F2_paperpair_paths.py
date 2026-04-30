"""
phase2_step4_F2_paperpair_paths.py [rev8]

Replaces BFS-shortest-path enumeration with full DFS-based enumeration of all
simple paths under the rev8 constraint set:

  Edge filters:
    - EDGE edges: edge_confidence >= 3
    - SIM edges: cos_sim >= 0.9
    - Both endpoints in vpn_custom (= nodes on at least one custom-mode path)

  Endpoint filter:
    - Intervention endpoint: intervention_maturity >= 3

  Path-level constraints:
    - Simple paths (no node repeats — DFS visited set)
    - consim1: no two consecutive SIM edges
    - Custom single-risk: path[0] is risk, no other risks anywhere
    - Custom single-intervention: path[-1] is FIRST intervention reached;
      DFS does not traverse past interventions

  No depth bound. No safety cap. All valid paths emitted.

Inputs:
  step1_load_and_parse.../graph_node_attributes.pkl
  step1_load_and_parse.../graph_edge_data.pkl
  phase1_rawpathsfiles/paths_custom_sim0.9.jsonl  (for vpn_custom construction)

Outputs:
  phase1_rawpathsfiles/paths_paperpair_sim0.9.jsonl
  phase2_results/step4_finalanalysis/step4_connectivity/paperpair_summary.txt
"""

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
STEP1_DIR = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
OUT_DIR = ROOT / "phase2_results/step4_finalanalysis/step4_connectivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_PATHS = PATHS_DIR / "paths_custom_sim0.9.jsonl"
OUTPUT = PATHS_DIR / "paths_paperpair_sim0.9.jsonl"
SUMMARY = OUT_DIR / "paperpair_summary.txt"

SIM_THRESHOLD = 0.9  # cos_sim >= 0.9
EDGE_CONFIDENCE_MIN = 3
INTERVENTION_MATURITY_MIN = 3


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


# ─── Load PKLs ────────────────────────────────────────────────────────────────
print("=" * 70)
print("Phase 2 Step 4 rev8 — paper-pair path enumeration (DFS)")
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

# ─── Build vpn_custom (same as F1: nodes on any custom-mode path) ─────────────
t1 = time.time()
print(f"\nBuilding vpn_custom from {CUSTOM_PATHS.name} ...")
vpn_custom = set()
with open(CUSTOM_PATHS) as f:
    for line in f:
        obj = json.loads(line)
        for nid in obj["path"]:
            vpn_custom.add(int(nid))
print(f"  vpn_custom: {len(vpn_custom):,} nodes  ({time.time() - t1:.1f}s)")


# ─── Build category lookup ────────────────────────────────────────────────────
def cat_of(nid):
    a = node_attrs.get(int(nid), {})
    if a.get("type") == "intervention":
        return "intervention"
    return str(a.get("concept_category", ""))


risk_nodes = {nid for nid in vpn_custom if cat_of(nid) == "risk"}
all_intervention_nodes = {nid for nid in vpn_custom if cat_of(nid) == "intervention"}
maturity3_interventions = {
    nid
    for nid in all_intervention_nodes
    if int(node_attrs.get(nid, {}).get("intervention_maturity", 0) or 0)
    >= INTERVENTION_MATURITY_MIN
}
print(
    f"\n  risk nodes in VPN: {len(risk_nodes):,}"
    f"\n  intervention nodes in VPN: {len(all_intervention_nodes):,}"
    f"\n  intervention nodes with maturity>=3: {len(maturity3_interventions):,}"
)

# ─── Build edge adjacency: EDGE (conf>=3) and SIM (cos_sim>=0.9), both within VPN ──
t2 = time.time()
print("\nBuilding adjacency (EDGE conf>=3, SIM cos_sim>=0.9, both in VPN)...")
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
    if s not in vpn_custom or t not in vpn_custom:
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
    f"  EDGE edges (undirected pairs counted twice): {n_edge:,}"
    f"\n  SIM edges (undirected pairs counted twice): {n_sim:,}"
    f"\n  ({time.time() - t2:.1f}s)"
)

# ─── DFS from each risk node ──────────────────────────────────────────────────
t3 = time.time()
print(f"\nEnumerating paths from {len(risk_nodes):,} risk nodes...")
sys.setrecursionlimit(50000)


def neighbors_of(curr):
    """Yield (neighbor, edge_type) where edge_type in {'EDGE', 'SIM'}."""
    for nb in adj_edge.get(curr, ()):
        yield nb, "EDGE"
    for nb in adj_sim.get(curr, ()):
        yield nb, "SIM"


risk_nodes_list = sorted(risk_nodes)
total_paths = 0
ri_pairs_emitted = set()
out_f = open(OUTPUT, "w")

risk_progress_step = max(1, len(risk_nodes_list) // 50)

for r_idx, R in enumerate(risk_nodes_list):
    if r_idx % risk_progress_step == 0:
        elapsed = time.time() - t3
        rate = (r_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(risk_nodes_list) - r_idx - 1) / rate if rate > 0 else 0
        print(
            f"  [{r_idx + 1}/{len(risk_nodes_list)}] R={R} | total_paths_so_far={total_paths:,} | rate={rate:.1f}/s | ETA={eta / 60:.1f}min",
            flush=True,
        )

    visited = {R}
    parent = {R: None}
    parent_edge_type = {R: None}

    def dfs(curr, prev_was_sim):
        global total_paths
        # If curr is an intervention with maturity>=3: emit path and stop
        if curr in maturity3_interventions:
            # Reconstruct path
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
                        "length": len(path) - 1,
                    }
                )
                + "\n"
            )
            total_paths += 1
            ri_pairs_emitted.add((path[0], path[-1]))
            return  # do not recurse past intervention

        for nb, etype in neighbors_of(curr):
            if nb in visited:
                continue
            if nb in risk_nodes:
                continue  # custom rule: skip risk neighbors (start risk already in visited)
            if etype == "SIM" and prev_was_sim:
                continue  # consim1
            visited.add(nb)
            parent[nb] = curr
            parent_edge_type[nb] = etype
            dfs(nb, etype == "SIM")
            visited.discard(nb)
            parent.pop(nb, None)
            parent_edge_type.pop(nb, None)

    dfs(R, prev_was_sim=False)

out_f.close()
print(
    f"\nDFS complete: {total_paths:,} total paths, {len(ri_pairs_emitted):,} unique (R, I) pairs"
    f"\n  ({time.time() - t3:.1f}s)"
)
print(f"\nWritten: {OUTPUT}")

# ─── Summary file ─────────────────────────────────────────────────────────────
with open(SUMMARY, "w") as f:
    f.write("rev8 paper-pair path enumeration summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"sim_threshold: {SIM_THRESHOLD}\n")
    f.write(f"edge_confidence_min: {EDGE_CONFIDENCE_MIN}\n")
    f.write(f"intervention_maturity_min: {INTERVENTION_MATURITY_MIN}\n")
    f.write(f"vpn_custom_nodes: {len(vpn_custom)}\n")
    f.write(f"risk_nodes_in_vpn: {len(risk_nodes)}\n")
    f.write(f"intervention_nodes_in_vpn: {len(all_intervention_nodes)}\n")
    f.write(f"maturity3_interventions: {len(maturity3_interventions)}\n")
    f.write(f"edge_edges: {n_edge // 2}\n")  # undirected -> div 2
    f.write(f"sim_edges: {n_sim // 2}\n")
    f.write(f"total_paths_emitted: {total_paths}\n")
    f.write(f"unique_ri_pairs: {len(ri_pairs_emitted)}\n")
    f.write(f"runtime_seconds: {time.time() - t0:.1f}\n")

print(f"Summary written: {SUMMARY}")
print("\nDONE.")
