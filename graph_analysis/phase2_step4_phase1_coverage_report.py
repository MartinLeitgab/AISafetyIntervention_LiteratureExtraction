"""
phase2_step4_phase1_coverage_report.py

After Phase 1 (HDBSCAN-2D + Louvain) on 19k EDGE-only VPN, report:
  - Coverage_A, Coverage_B, Coverage_union(A,B), Coverage_intersect (A∩B)
  - ARI(A, B) on intersect — method-agreement signal
  - Per-pool breakdown

Usage:
  python phase2_step4_phase1_coverage_report.py            # default 0.80/mcs=5
  python phase2_step4_phase1_coverage_report.py --tag _c70m3  # 0.70/mcs=3
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
args = ap.parse_args()
TAG = args.tag

ROOT = Path(__file__).parent
STEP1_DIR = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"

with open(STEP1_DIR / f"cluster_memberships_rev8_paper_methodA{TAG}.pkl", "rb") as f:
    cm_A = pickle.load(f)
with open(STEP1_DIR / f"cluster_memberships_rev8_paper_methodB{TAG}.pkl", "rb") as f:
    cm_B = pickle.load(f)
with open(STEP1_DIR / "role_of_rev8_paper.pkl", "rb") as f:
    role_of = pickle.load(f)

# Pool membership (from VPN)
BODY_SUBTYPES = {
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
}
pool_of_node = {}
for nid, rl in role_of.items():
    if rl == "risk":
        pool_of_node[nid] = "risk"
    elif rl == "intervention" or rl in BODY_SUBTYPES:
        pool_of_node[nid] = "nr"


# Build (pool, nid) -> cluster_id maps for A and B
def labels_per_pool(cm):
    """Return dict[pool] -> dict[nid] -> cluster_label."""
    out = defaultdict(dict)
    for key, members in cm.items():
        # key = (variant, mode, pool, algo, cid)
        pool = key[2]
        cid = key[4]
        for nid in members:
            out[pool][int(nid)] = cid
    return out


A_per_pool = labels_per_pool(cm_A)
B_per_pool = labels_per_pool(cm_B)

# Pool universe = all nodes in role_of with that pool
pool_universe = defaultdict(set)
for nid, rl in role_of.items():
    if rl == "risk":
        pool_universe["risk"].add(int(nid))
    elif rl == "intervention" or rl in BODY_SUBTYPES:
        pool_universe["nr"].add(int(nid))

rows = []
for pool in ["risk", "nr"]:
    universe = pool_universe[pool]
    A_set = set(A_per_pool[pool].keys())
    B_set = set(B_per_pool[pool].keys())
    n_input = len(universe)
    cov_A = len(A_set) / n_input
    cov_B = len(B_set) / n_input
    union = A_set | B_set
    intersect = A_set & B_set
    cov_union = len(union) / n_input
    cov_intersect = len(intersect) / n_input
    only_A = A_set - B_set
    only_B = B_set - A_set
    # ARI on intersect (method agreement)
    if len(intersect) >= 2:
        common = sorted(intersect)
        L_A = [A_per_pool[pool][n] for n in common]
        L_B = [B_per_pool[pool][n] for n in common]
        ari = adjusted_rand_score(L_A, L_B)
    else:
        ari = float("nan")
    rows.append(
        {
            "pool": pool,
            "n_input": n_input,
            "cov_A_hdbscan2d": round(cov_A, 4),
            "cov_B_louvain": round(cov_B, 4),
            "cov_union_AB": round(cov_union, 4),
            "cov_intersect_AB": round(cov_intersect, 4),
            "n_clustered_A": len(A_set),
            "n_clustered_B": len(B_set),
            "n_union": len(union),
            "n_intersect": len(intersect),
            "n_only_A": len(only_A),
            "n_only_B": len(only_B),
            "n_residual": n_input - len(union),
            "ARI_on_intersect": round(ari, 4) if not np.isnan(ari) else None,
        }
    )

# Combined (risk + nr)
total_input = sum(len(pool_universe[p]) for p in ["risk", "nr"])
total_A = sum(len(A_per_pool[p]) for p in ["risk", "nr"])
total_B = sum(len(B_per_pool[p]) for p in ["risk", "nr"])
total_union = sum(
    len(set(A_per_pool[p].keys()) | set(B_per_pool[p].keys())) for p in ["risk", "nr"]
)
total_intersect = sum(
    len(set(A_per_pool[p].keys()) & set(B_per_pool[p].keys())) for p in ["risk", "nr"]
)
rows.append(
    {
        "pool": "TOTAL",
        "n_input": total_input,
        "cov_A_hdbscan2d": round(total_A / total_input, 4),
        "cov_B_louvain": round(total_B / total_input, 4),
        "cov_union_AB": round(total_union / total_input, 4),
        "cov_intersect_AB": round(total_intersect / total_input, 4),
        "n_clustered_A": total_A,
        "n_clustered_B": total_B,
        "n_union": total_union,
        "n_intersect": total_intersect,
        "n_only_A": total_A - total_intersect,
        "n_only_B": total_B - total_intersect,
        "n_residual": total_input - total_union,
        "ARI_on_intersect": None,
    }
)

df = pd.DataFrame(rows)
out_csv = STEP1_DIR / f"phase1_union_intersect_report{TAG}.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote {out_csv.name}")
print()
print("=" * 100)
print("PHASE 1 UNION/INTERSECT REPORT")
print("=" * 100)
print(df.to_string(index=False))
print()
print("Notes:")
print(
    "  - ARI_on_intersect: high (>0.7) = methods agree on cluster assignments where both"
)
print(
    "    cluster the node; low (<0.3) = methods disagree, treat as different signals."
)
print("  - n_residual = nodes neither method clustered → goes to Phase 2 LLM thematic.")
