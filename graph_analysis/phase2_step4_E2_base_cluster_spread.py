"""
phase2_step4_E2_base_cluster_spread.py  [rev7]

Adds closest-3 and farthest-3 nodes (by cosine sim to centroid) to the
L1 risk and L3 intervention cluster tables.

VPN filter: uses vpn_unconstrained (superset of vpn_consim1) to avoid
loading the heavy edge_data PKL. The centroid computed from the superset
differs negligibly from the consim1-filtered centroid.

Inputs:
  cluster_memberships.pkl
  graph_node_attributes.pkl
  phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl  -- VPN construction
  step4_cluster_tables/risk_clusters_consim1.csv
  step4_cluster_tables/intervention_clusters_consim1.csv

Outputs (NEW, do not overwrite old):
  step4_cluster_tables/risk_clusters_consim1_v2.csv
  step4_cluster_tables/intervention_clusters_consim1_v2.csv
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "phase2_results"
STEP1_DIR = RESULTS_DIR / "step1_load_and_parse_umapwithoutlocalsatellites"
STEP4_DIR = RESULTS_DIR / "step4_finalanalysis"
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
OUT_TABLES = STEP4_DIR / "step4_cluster_tables"

# ── Load PKL files ────────────────────────────────────────────────────────────

print("Loading cluster_memberships.pkl ...")
t0 = time.time()
with open(STEP1_DIR / "cluster_memberships.pkl", "rb") as f:
    cm = pickle.load(f)
print(f"  {len(cm)} keys  ({time.time() - t0:.1f}s)")

print("Loading graph_node_attributes.pkl ...")
t1 = time.time()
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
print(f"  {len(node_attrs)} nodes  ({time.time() - t1:.1f}s)")

# ── Build VPN_unconstrained (superset of vpn_consim1) ────────────────────────

print("Building vpn_unconstrained (maturity>=3 filter) ...")
t2 = time.time()
vpn_unconstrained = set()
vp_file = PATHS_DIR / "paths_unconstrained_sim0.9.jsonl"
with open(vp_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        if isinstance(obj, dict) and "path" in obj:
            path = [int(x) for x in obj["path"]]
            interv_id = path[-1]
            if (
                int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0)
                >= 3
            ):
                vpn_unconstrained.update(path)
print(
    f"  {len(vpn_unconstrained):,} vpn_unconstrained nodes  ({time.time() - t2:.1f}s)"
)


def get_cluster_dict(node_type, ec=0.9, mode="unconstrained", algo="agglomerative"):
    result = {}
    for key, members in cm.items():
        try:
            e_float = float(key[0])
        except (ValueError, TypeError):
            continue
        if (
            abs(e_float - ec) < 1e-9
            and key[1] == mode
            and key[2] == node_type
            and key[3] == algo
        ):
            filtered = [int(nid) for nid in members if int(nid) in vpn_unconstrained]
            if filtered:
                result[int(key[4])] = filtered
    return result


def parse_embedding(emb_val):
    if isinstance(emb_val, np.ndarray):
        arr = emb_val.astype(np.float32)
    elif emb_val is None:
        return None
    else:
        s = str(emb_val).strip()
        if s.startswith("<") and s.endswith(">"):
            s = s[1:-1]
        try:
            arr = np.array([float(x) for x in s.split(",")], dtype=np.float32)
        except Exception:
            return None
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def get_spread(member_ids, top_n=3):
    vecs, ids_with_emb = [], []
    for nid in member_ids:
        emb = parse_embedding(node_attrs.get(nid, {}).get("embedding"))
        if emb is not None:
            vecs.append(emb)
            ids_with_emb.append(nid)
    if len(vecs) < 2:
        return [], [], None, None
    vecs_arr = np.stack(vecs)
    centroid = vecs_arr.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm
    sims = [float(np.dot(centroid, v)) for v in vecs]
    idx = np.argsort(sims)
    farthest_ids = [ids_with_emb[i] for i in idx[:top_n]]
    closest_ids = [ids_with_emb[i] for i in idx[-top_n:][::-1]]

    def name(nid):
        return str(node_attrs.get(nid, {}).get("name", f"node_{nid}"))[:120]

    closest = " | ".join(name(n) for n in closest_ids)
    farthest = " | ".join(name(n) for n in farthest_ids)
    return closest, farthest, min(sims), max(sims)


# ── Process risk and intervention clusters ────────────────────────────────────

for node_type, in_file, out_file in [
    ("risk", "risk_clusters_consim1.csv", "risk_clusters_consim1_v2.csv"),
    (
        "intervention",
        "intervention_clusters_consim1.csv",
        "intervention_clusters_consim1_v2.csv",
    ),
]:
    print(f"\nProcessing {node_type} clusters ...")
    existing = pd.read_csv(OUT_TABLES / in_file)
    clusters = get_cluster_dict(node_type)
    print(f"  Found {len(clusters)} {node_type} clusters in pkl")

    rows = []
    for _, row in existing.iterrows():
        cid = int(row["cluster_id"])
        member_ids = clusters.get(cid, [])
        closest, farthest, sim_min, sim_max = get_spread(member_ids)
        row = row.to_dict()
        row["closest3_names"] = closest
        row["farthest3_names"] = farthest
        row["centroid_sim_min"] = round(sim_min, 4) if sim_min is not None else None
        row["centroid_sim_max"] = round(sim_max, 4) if sim_max is not None else None
        rows.append(row)
        print(
            f"  C{cid:2d}: {len(member_ids)} members | sim [{sim_min:.3f}, {sim_max:.3f}]"
        )

    out_df = pd.DataFrame(rows)
    orig_cols = list(existing.columns)
    new_cols = [
        "closest3_names",
        "farthest3_names",
        "centroid_sim_min",
        "centroid_sim_max",
    ]
    out_df = out_df[orig_cols + new_cols]
    out_path = OUT_TABLES / out_file
    out_df.to_csv(out_path, index=False)
    print(f"  Written: {out_path} ({len(out_df)} rows)")

print("\nE2 complete.")
