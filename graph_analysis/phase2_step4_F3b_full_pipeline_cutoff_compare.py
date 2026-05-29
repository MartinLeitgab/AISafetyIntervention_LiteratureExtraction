"""
phase2_step4_F3b_full_pipeline_cutoff_compare.py [rev8 — Task #7 robustness]

Compares full-pipeline F3 outputs across multiple cutoff values to show
that small variations around 0.77 don't drastically change clusters or yield.

Inputs (one cluster_memberships PKL per cutoff):
  --pkl-pattern   path pattern with {cutoff} placeholder, e.g.
                  "graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/cluster_memberships_rev8_edge_only_cutoff{cutoff}.pkl"
  --cutoffs       comma-separated cutoff values (e.g. 0.73,0.75,0.77,0.79,0.81)
  --reference-cutoff  e.g. 0.77 (used as ARI reference)
  --output-suffix  e.g. edge_only

For each cutoff:
  - Load cluster_memberships PKL
  - Per node-type: count clusters, count members, build node->cluster map
  - Compute ARI vs reference cutoff (per node-type, on intersection of clustered nodes)

Outputs:
  step4_cluster_tables/full_cutoff_compare_<output-suffix>.csv
    rows: subtype × cutoff → n_clusters, n_clustered, ARI_vs_ref
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

try:
    from sklearn.metrics import adjusted_rand_score as _ari
except ImportError:
    _ari = None

ROOT = Path(__file__).parent
STEP4_DIR = ROOT / "phase2_results/step4_finalanalysis"
OUT_TABLES = STEP4_DIR / "step4_cluster_tables"


def load_pkl_to_subtype_node_label(pkl_path):
    """Returns {subtype: {node_id: cluster_id}}."""
    with open(pkl_path, "rb") as f:
        cm = pickle.load(f)
    subtype_node_lbl = {}
    for key, members in cm.items():
        # key: (variant, mode, subtype, algo, cluster_id)
        subtype = str(key[2])
        cid = str(key[4])
        if subtype not in subtype_node_lbl:
            subtype_node_lbl[subtype] = {}
        for nid in members:
            subtype_node_lbl[subtype][int(nid)] = cid
    return subtype_node_lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pkl-pattern",
        required=True,
        help="format string with {cutoff}, e.g. .../cluster_memberships_rev8_edge_only_cutoff{cutoff}.pkl",
    )
    ap.add_argument("--cutoffs", default="0.73,0.75,0.77,0.79,0.81")
    ap.add_argument("--reference-cutoff", default="0.77")
    ap.add_argument("--output-suffix", default="edge_only")
    args = ap.parse_args()

    cutoffs = [c.strip() for c in args.cutoffs.split(",")]
    ref = args.reference_cutoff
    print(f"Loading PKLs for cutoffs: {cutoffs} (reference={ref})")
    subtype_lbls = {}
    for c in cutoffs:
        path = args.pkl_pattern.format(cutoff=c)
        if not Path(path).exists():
            print(f"  [{c}] MISSING: {path}; skipping")
            continue
        subtype_lbls[c] = load_pkl_to_subtype_node_label(path)
        n_total = sum(len(v) for v in subtype_lbls[c].values())
        print(
            f"  [{c}] loaded {n_total:,} clustered nodes across {len(subtype_lbls[c])} subtypes"
        )

    if ref not in subtype_lbls:
        raise SystemExit(f"reference cutoff {ref} not loaded; cannot compute ARI")

    rows = []
    all_subtypes = sorted(set(s for d in subtype_lbls.values() for s in d.keys()))

    for st in all_subtypes:
        ref_map = subtype_lbls[ref].get(st, {})
        ref_cluster_count = len(set(ref_map.values()))
        ref_n = len(ref_map)
        for c in cutoffs:
            if c not in subtype_lbls:
                continue
            cur_map = subtype_lbls[c].get(st, {})
            cluster_count = len(set(cur_map.values()))
            n_members = len(cur_map)
            # ARI on intersection of clustered nodes
            common = set(ref_map.keys()) & set(cur_map.keys())
            ari_val = None
            if _ari is not None and len(common) > 4:
                ref_arr = [ref_map[n] for n in sorted(common)]
                cur_arr = [cur_map[n] for n in sorted(common)]
                if len(set(ref_arr)) > 1 and len(set(cur_arr)) > 1:
                    ari_val = float(_ari(ref_arr, cur_arr))
            rows.append(
                {
                    "subtype": st,
                    "cutoff": c,
                    "n_clusters": cluster_count,
                    "n_clustered_members": n_members,
                    "delta_clusters_vs_ref": cluster_count - ref_cluster_count,
                    "delta_members_vs_ref": n_members - ref_n,
                    "ari_vs_ref_on_intersection": round(ari_val, 4)
                    if ari_val is not None
                    else None,
                    "intersection_size": len(common),
                }
            )

    df = pd.DataFrame(rows)
    out_path = OUT_TABLES / f"full_cutoff_compare_{args.output_suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    print()
    # Pretty-print summary per subtype
    print(df.pivot(index="subtype", columns="cutoff", values="n_clusters").to_string())
    print("\nARI vs reference (clusters identity stability):")
    print(
        df.pivot(
            index="subtype", columns="cutoff", values="ari_vs_ref_on_intersection"
        ).to_string()
    )


if __name__ == "__main__":
    main()
