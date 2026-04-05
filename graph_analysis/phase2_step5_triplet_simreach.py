"""
Phase 2 Step 5 — Triplet SIM Reach Analysis
For each (risk, chain, intervention) triplet from the top-15 prevalent pathways,
compute the union of distinct partner paper URLs reachable via SIM>=0.9 edges
from any node belonging to any of the 3 clusters.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

import csv

BASE = Path("graph_analysis/phase2_results")
STEP1 = BASE / "step1_load_and_parse_umapwithoutlocalsatellites"
STEP5E = BASE / "step5_examples"
STEP5N = BASE / "step5_naming"
PATHS_UNCONSTRAINED = Path(
    "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
)

# ── 1. Load top-15 triplets ──────────────────────────────────────────────────
print("Loading top-15 triplet data …")
with open(STEP5E / "pathway_examples_prevalent.json") as f:
    prev_data = json.load(f)

triplets = []
for item in prev_data["top15_risk_to_intervention"]:
    triplets.append(
        {
            "risk_id": item["risk_cluster"]["id"],
            "risk_name": item["risk_cluster"]["name"],
            "chain_id": item["dominant_chain_cluster"]["id"],
            "chain_name": item["dominant_chain_cluster"]["name"],
            "interv_id": item["intervention_cluster"]["id"],
            "interv_name": item["intervention_cluster"]["name"],
            "n_paths": item["n_paths_total"],
        }
    )
print(f"  {len(triplets)} triplets loaded")

# ── 2. Load PKL files ────────────────────────────────────────────────────────
print("Loading PKL files …")
with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
print(f"  node_attrs: {len(node_attrs):,} nodes")

with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
print(f"  edge_data: {len(edge_data):,} edges")

with open(STEP1 / "cluster_memberships.pkl", "rb") as f:
    cluster_memberships = pickle.load(f)
print(f"  cluster_memberships: {len(cluster_memberships):,} keys")

with open(BASE / "step4_finalanalysis" / "optionA_cluster_labels.pkl", "rb") as f:
    chain_labels = pickle.load(f)
print(f"  chain_labels keys: {list(chain_labels.keys())[:3]}")

# ── 2b. Load valid_pathway_nodes (Gap 5b fix) ────────────────────────────────
print("Building valid_pathway_nodes …")
valid_pathway_nodes = set()
with open(PATHS_UNCONSTRAINED) as _f:
    for _line in _f:
        _obj = json.loads(_line)
        _path = _obj.get("path") or _obj.get("node_id_sequence") or []
        for _nid in _path:
            valid_pathway_nodes.add(int(_nid))
print(f"  valid_pathway_nodes: {len(valid_pathway_nodes):,} nodes")

# ── 3. Build node → risk/intervention cluster dicts ─────────────────────────
print("Building cluster membership dicts …")

# Risk clusters: edge_config=0.9, mode=unconstrained, node_type=risk, algo=agglomerative
# Gap 5b fix: filter to valid_pathway_nodes (holistic qualifying universe)
node_to_risk = {}
node_to_interv = {}

for key, node_list in cluster_memberships.items():
    ec, mode, ntype, algo, cid = key
    if ec != 0.9 or mode != "unconstrained" or algo != "agglomerative":
        continue
    if ntype == "risk":
        for nid in node_list:
            if nid in valid_pathway_nodes:
                node_to_risk[nid] = int(cid)
    elif ntype == "intervention":
        for nid in node_list:
            if nid in valid_pathway_nodes:
                node_to_interv[nid] = int(cid)

print(f"  node_to_risk: {len(node_to_risk):,} nodes")
print(f"  node_to_interv: {len(node_to_interv):,} nodes")

# Chain clusters from optionA_cluster_labels
# Structure: {"labels": np.array, "records": [(body_ids, full_path_ids), ...]}
labels_arr = chain_labels["labels"]
records = chain_labels["records"]

node_to_chain = {}
for path_idx, (body_ids, full_path_ids) in enumerate(records):
    clabel = int(labels_arr[path_idx])
    for nid in body_ids:
        # body_ids may be stored as node IDs; take majority label per node
        if nid not in node_to_chain:
            node_to_chain[nid] = clabel
print(f"  node_to_chain: {len(node_to_chain):,} nodes")

# ── 4. Build cluster → node sets ────────────────────────────────────────────
print("Building cluster → node set dicts …")

risk_cluster_nodes = defaultdict(set)
for nid, cid in node_to_risk.items():
    risk_cluster_nodes[cid].add(nid)

interv_cluster_nodes = defaultdict(set)
for nid, cid in node_to_interv.items():
    interv_cluster_nodes[cid].add(nid)

chain_cluster_nodes = defaultdict(set)
for nid, cid in node_to_chain.items():
    chain_cluster_nodes[cid].add(nid)

# ── 5. Build SIM>=0.9 edge lookup: node → set of partner node IDs ───────────
print("Building SIM>=0.9 partner lookup …")

SIM_THRESHOLD = 0.9
SCORE_CUTOFF = np.sqrt(2 * (1 - SIM_THRESHOLD))  # ~0.4472

node_sim_partners = defaultdict(set)
n_sim = 0
for e in edge_data:
    if str(e.get("type", "")).upper() != "SIMILARITY":
        continue
    score = e.get("similarity_score")
    if score is None:
        continue
    cos = 1.0 - float(score) ** 2 / 2.0
    if cos < SIM_THRESHOLD:
        continue
    src = e["source"]
    tgt = e["target"]
    node_sim_partners[src].add(tgt)
    node_sim_partners[tgt].add(src)
    n_sim += 1

print(
    f"  SIM>=0.9 edges: {n_sim:,}, nodes with SIM partners: {len(node_sim_partners):,}"
)


# ── 6. Helper: cluster SIM reach (distinct partner paper URLs) ───────────────
def cluster_sim_reach(node_set):
    """Return set of distinct partner paper URLs reachable via SIM>=0.9 from node_set."""
    urls = set()
    for nid in node_set:
        for pid in node_sim_partners.get(nid, set()):
            url = node_attrs.get(pid, {}).get("url", "")
            if url:
                urls.add(url)
    return urls


# ── 7. Per-triplet combined SIM reach ────────────────────────────────────────
print("\nComputing triplet SIM reach …")

results = []
for t in triplets:
    rid = t["risk_id"]
    cid = t["chain_id"]
    iid = t["interv_id"]

    r_nodes = risk_cluster_nodes.get(rid, set())
    c_nodes = chain_cluster_nodes.get(cid, set())
    i_nodes = interv_cluster_nodes.get(iid, set())

    r_urls = cluster_sim_reach(r_nodes)
    c_urls = cluster_sim_reach(c_nodes)
    i_urls = cluster_sim_reach(i_nodes)

    union_urls = r_urls | c_urls | i_urls
    intersect_rc = r_urls & c_urls
    intersect_ri = r_urls & i_urls
    intersect_ci = c_urls & i_urls
    triplet_core = r_urls & c_urls & i_urls

    results.append(
        {
            **t,
            "r_nodes": len(r_nodes),
            "c_nodes": len(c_nodes),
            "i_nodes": len(i_nodes),
            "r_reach": len(r_urls),
            "c_reach": len(c_urls),
            "i_reach": len(i_urls),
            "union_reach": len(union_urls),
            "r_c_intersect": len(intersect_rc),
            "r_i_intersect": len(intersect_ri),
            "c_i_intersect": len(intersect_ci),
            "triplet_core": len(triplet_core),
        }
    )
    print(
        f"  R{rid}→C{cid}→I{iid}: union={len(union_urls)}, core={len(triplet_core)}, n_paths={t['n_paths']}"
    )

# ── 8. Sort by union reach ───────────────────────────────────────────────────
results.sort(key=lambda x: x["union_reach"], reverse=True)

print("\n── Final ranking by union SIM reach ──")
for i, r in enumerate(results, 1):
    print(
        f"{i:2d}. R{r['risk_id']}→C{r['chain_id']}→I{r['interv_id']}  "
        f"union={r['union_reach']:,}  core={r['triplet_core']:,}  "
        f"n_paths={r['n_paths']:,}  "
        f"  | R:{r['r_reach']} C:{r['c_reach']} I:{r['i_reach']}"
    )

# Also sort by triplet_core (papers discussing all 3 simultaneously)
results_by_core = sorted(results, key=lambda x: x["triplet_core"], reverse=True)
print("\n── Final ranking by triplet CORE (papers in all 3 clusters) ──")
for i, r in enumerate(results_by_core, 1):
    print(
        f"{i:2d}. R{r['risk_id']}→C{r['chain_id']}→I{r['interv_id']}  "
        f"core={r['triplet_core']:,}  union={r['union_reach']:,}  "
        f"n_paths={r['n_paths']:,}"
    )

# ── 9. Save results ──────────────────────────────────────────────────────────

out_path = BASE / "step5_naming" / "triplet_simreach.csv"
fieldnames = [
    "risk_id",
    "risk_name",
    "chain_id",
    "chain_name",
    "interv_id",
    "interv_name",
    "n_paths",
    "r_nodes",
    "c_nodes",
    "i_nodes",
    "r_reach",
    "c_reach",
    "i_reach",
    "union_reach",
    "r_c_intersect",
    "r_i_intersect",
    "c_i_intersect",
    "triplet_core",
]
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in results:
        w.writerow({k: r[k] for k in fieldnames})
print(f"\nSaved: {out_path}")
print("Done.")
