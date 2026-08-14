"""
phase2_step4_E1_body_cluster_spread.py  [rev7]

Adds closest-3 and farthest-3 nodes (by cosine similarity to centroid) to
bodysubtype_cluster_representatives, giving reviewers a verifiable quality
signal: if farthest-3 are thematically consistent with closest-3, the cluster
is tight; otherwise the LLM name based on that cluster is flagged as partial.

Inputs:
  cluster_memberships.pkl         -- body node cluster assignments
  graph_node_attributes.pkl       -- node embeddings and names
  phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl  -- VPN construction

Output (NEW, does not overwrite old):
  step4_cluster_tables/bodysubtype_cluster_representatives_v2.csv
  New cols: closest3_names, farthest3_names, centroid_sim_min, centroid_sim_max
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

BODY_SUBTYPES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]

SUBTYPE_PREFIX = {
    "problem_analysis": "pr",
    "theoretical_insight": "th",
    "design_rationale": "de",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
}

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

# ── Build VPN (same as original pathbuildB script) ───────────────────────────

print("Building valid_pathway_nodes ...")
t2 = time.time()
valid_pathway_nodes = set()
vp_file = PATHS_DIR / "paths_unconstrained_sim0.9.jsonl"
with open(vp_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        for nid in obj["path"]:
            valid_pathway_nodes.add(int(nid))
print(f"  {len(valid_pathway_nodes):,} VPN nodes  ({time.time() - t2:.1f}s)")


def get_clusters(edge_config, mode, node_type, algo="agglomerative"):
    result = {}
    try:
        ec_float = float(edge_config)
    except Exception:
        ec_float = None
    for k, v in cm.items():
        k0 = k[0]
        try:
            match = float(k0) == ec_float
        except Exception:
            match = str(k0) == str(edge_config)
        if match and str(k[1]) == mode and str(k[2]) == node_type and str(k[3]) == algo:
            result[str(k[4])] = [int(n) for n in v]
    return result


def parse_embedding(emb_val):
    """Parse FalkorDB embedding string '<v1, v2, ...>' → np.float32 array."""
    if isinstance(emb_val, np.ndarray):
        return emb_val.astype(np.float32)
    if emb_val is None:
        return None
    s = str(emb_val).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    try:
        arr = np.array([float(x) for x in s.split(",")], dtype=np.float32)
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr
    except Exception:
        return None


def cosine_sim(a, b):
    return float(np.dot(a, b))  # both unit-normalized


def get_spread(member_ids, node_attrs, top_n=3):
    """
    Given a list of node IDs, compute centroid and return:
    centroid embedding, closest-N names, farthest-N names,
    min cosine sim, max cosine sim.
    """
    vecs = []
    ids_with_emb = []
    for nid in member_ids:
        attrs = node_attrs.get(nid, {})
        emb = parse_embedding(attrs.get("embedding"))
        if emb is not None and len(emb) > 0:
            vecs.append(emb)
            ids_with_emb.append(nid)

    if len(vecs) < 2:
        return None, [], [], None, None

    vecs_arr = np.stack(vecs)
    centroid = vecs_arr.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    sims = [cosine_sim(centroid, v) for v in vecs]
    sorted_idx = np.argsort(sims)

    farthest_ids = [ids_with_emb[i] for i in sorted_idx[:top_n]]
    closest_ids = [ids_with_emb[i] for i in sorted_idx[-top_n:][::-1]]

    def get_name(nid):
        return str(node_attrs.get(nid, {}).get("name", f"node_{nid}"))[:120]

    closest_names = " | ".join(get_name(nid) for nid in closest_ids)
    farthest_names = " | ".join(get_name(nid) for nid in farthest_ids)

    return centroid, closest_names, farthest_names, min(sims), max(sims)


# ── Main loop ─────────────────────────────────────────────────────────────────

existing = pd.read_csv(OUT_TABLES / "bodysubtype_cluster_representatives.csv")
print(f"\nLoaded existing reps: {len(existing)} rows")

results = []

for subtype in BODY_SUBTYPES:
    prefix = SUBTYPE_PREFIX[subtype]
    clusters = get_clusters("0.9", "unconstrained", subtype)
    print(f"\n  Subtype '{prefix}' ({subtype}): {len(clusters)} clusters")

    for cid_str, node_ids in clusters.items():
        # Filter to VPN
        vpn_ids = [n for n in node_ids if n in valid_pathway_nodes]
        prefix_key = f"{prefix}:{cid_str}"

        _, closest_names, farthest_names, sim_min, sim_max = get_spread(
            vpn_ids, node_attrs
        )

        # Get existing row
        row_mask = existing["prefix_key"] == prefix_key
        if row_mask.sum() == 0:
            print(f"    WARNING: {prefix_key} not found in existing reps")
            continue

        row = existing[row_mask].iloc[0].to_dict()
        row["closest3_names"] = closest_names
        row["farthest3_names"] = farthest_names
        row["centroid_sim_min"] = round(sim_min, 4) if sim_min is not None else None
        row["centroid_sim_max"] = round(sim_max, 4) if sim_max is not None else None
        results.append(row)

    print(f"    Done: {len([r for r in results if r['subtype'] == subtype])} entries")

out_df = pd.DataFrame(results)
# Preserve original column order + new cols
orig_cols = list(existing.columns)
new_cols = ["closest3_names", "farthest3_names", "centroid_sim_min", "centroid_sim_max"]
out_df = out_df[orig_cols + new_cols]

out_path = OUT_TABLES / "bodysubtype_cluster_representatives_v2.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWritten: {out_path} ({len(out_df)} rows)")

# ── Quality check ─────────────────────────────────────────────────────────────
n_missing_closest = out_df["closest3_names"].isna().sum()
n_missing_farthest = out_df["farthest3_names"].isna().sum()
print(
    f"Quality: missing closest3={n_missing_closest}, missing farthest3={n_missing_farthest}"
)
print(
    f"Centroid sim range overall: {out_df['centroid_sim_min'].min():.3f} -- {out_df['centroid_sim_max'].max():.3f}"
)
