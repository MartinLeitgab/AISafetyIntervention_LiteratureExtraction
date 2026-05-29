"""Decode cluster IDs in top family signatures into centroid representative names.

For cutoff=0.80, finds the node closest to centroid for each cluster_id appearing
in top-5 strict_tuple and top-5 semantic_only families. Outputs:
  - Decoded family table CSV with representative names appended
  - Per-cluster representative table (cluster_id -> closest member name)
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4 = ROOT / "phase2_results/step4_finalanalysis"

PKL_FILE = STEP1 / "cluster_memberships_rev8_global_cutoff0.80.pkl"
ATTR_FILE = STEP1 / "graph_node_attributes.pkl"

CUTOFF = 0.80
SUFFIX = f"global_cutoff{CUTOFF:.2f}"
TOP_FAMILIES_CSV = (
    STEP4 / f"step4_cluster_tables/global_cutoff_top_families_{SUFFIX}.csv"
)


def parse_emb(emb):
    if emb is None:
        return None
    if isinstance(emb, np.ndarray):
        return emb.astype(np.float32)
    if isinstance(emb, (list, tuple)):
        return np.array(emb, dtype=np.float32)
    if isinstance(emb, str):
        s = emb.strip().strip("'<>").strip()
        try:
            return np.array([float(x) for x in s.split(",")], dtype=np.float32)
        except (ValueError, TypeError):
            return None
    return None


def main():
    print("Loading PKL + attrs ...")
    with open(PKL_FILE, "rb") as f:
        cm = pickle.load(f)
    with open(ATTR_FILE, "rb") as f:
        node_attrs = pickle.load(f)

    # Group cluster members by (group, cid)
    cluster_members = defaultdict(list)
    for key, members in cm.items():
        group = str(key[2])
        cid = str(key[4])
        cluster_members[(group, cid)] = list(members)
    print(f"  {len(cluster_members)} clusters total")

    # Build closest-to-centroid representative for each cluster
    closest_rep = {}  # (group, cid) -> {"name": str, "nid": int, "sim_to_centroid": float, "n_members": int}
    for (group, cid), members in cluster_members.items():
        embs = []
        nids_with_emb = []
        for nid in members:
            emb = node_attrs.get(int(nid), {}).get("embedding")
            arr = parse_emb(emb)
            if arr is None:
                continue
            embs.append(arr)
            nids_with_emb.append(int(nid))
        if not embs:
            continue
        E = np.stack(embs)
        # unit-normalize each row, then compute mean → centroid → unit-normalize
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / np.where(norms > 0, norms, 1.0)
        centroid = E_norm.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        sims = E_norm @ centroid
        best_idx = int(np.argmax(sims))
        closest_rep[(group, cid)] = {
            "nid": nids_with_emb[best_idx],
            "name": str(node_attrs[nids_with_emb[best_idx]].get("name", ""))[:100],
            "concept_category": str(
                node_attrs[nids_with_emb[best_idx]].get("concept_category", "")
            ),
            "sim_to_centroid": float(sims[best_idx]),
            "n_members": len(members),
        }

    print(f"  built representatives for {len(closest_rep)} clusters")

    # Save full representative table
    rep_rows = []
    for (group, cid), info in closest_rep.items():
        rep_rows.append(
            {
                "group": group,
                "cluster_id": cid,
                "n_members": info["n_members"],
                "rep_node_id": info["nid"],
                "rep_name": info["name"],
                "rep_concept_category": info["concept_category"],
                "rep_sim_to_centroid": round(info["sim_to_centroid"], 4),
            }
        )
    rep_df = pd.DataFrame(rep_rows).sort_values(["group", "cluster_id"])
    rep_csv = STEP4 / f"step4_cluster_tables/cluster_representatives_{SUFFIX}.csv"
    rep_df.to_csv(rep_csv, index=False)
    print(f"  wrote: {rep_csv.name}")

    # ─── Decode top families ──────────────────────────────────────────────────
    fams = pd.read_csv(TOP_FAMILIES_CSV)
    print(
        f"\nLoaded top families: {len(fams)} rows across "
        f"{fams['level'].nunique()} levels"
    )

    def decode_signature(sig_str, level):
        """Parse 'pa:141 & ti:253 & ...' or 'sem:141 & sem:253 & ...' or
        'pa=1 ti=1 dr=1 im=1 va=1' → list of (label, body_global_cid_or_None, rep_name)."""
        out = []
        if level == "role_pattern":
            return [(p, None, None) for p in sig_str.split()]
        for part in sig_str.split(" & "):
            if ":" not in part:
                continue
            prefix, cid = part.split(":", 1)
            cid = cid.strip()
            # Both strict_tuple and semantic_only use body_global namespace
            rep_info = closest_rep.get(("body_global", cid))
            if rep_info:
                out.append((part, cid, rep_info["name"]))
            else:
                out.append((part, cid, "(no centroid info)"))
        return out

    # Add decoded representation per family
    decoded_rows = []
    for _, row in fams.iterrows():
        sig = row["core_signature"]
        level = row["level"]
        decoded = decode_signature(sig, level)
        if level == "role_pattern":
            decoded_str = sig
        else:
            decoded_str = "  |  ".join(f"{lbl} ({nm[:60]})" for lbl, _, nm in decoded)
        decoded_rows.append(
            {
                **row.to_dict(),
                "core_signature_decoded": decoded_str,
            }
        )
    decoded_df = pd.DataFrame(decoded_rows)
    decoded_csv = (
        STEP4 / f"step4_cluster_tables/global_cutoff_top_families_{SUFFIX}_DECODED.csv"
    )
    decoded_df.to_csv(decoded_csv, index=False)
    print(f"  wrote: {decoded_csv.name}")

    # ─── Print decoded top-5 strict_tuple and top-5 semantic_only ─────────────
    print("\n" + "=" * 80)
    print("TOP 5 STRICT_TUPLE FAMILIES (cutoff=0.80)")
    print("=" * 80)
    strict5 = decoded_df[decoded_df["level"] == "strict_tuple"].head(5)
    for _, r in strict5.iterrows():
        print(
            f"\n  fam {int(r['family_id']):>2}  #frozensets={int(r['n_frozensets']):>2}  "
            f"#paths={int(r['n_paths_total']):>2}  #R-I_pairs={int(r['n_RI_pairs_total']):>2}"
        )
        for lbl, _, nm in decode_signature(r["core_signature"], "strict_tuple"):
            print(f"    {lbl:<10} {nm[:90]}")

    print("\n" + "=" * 80)
    print("TOP 5 SEMANTIC_ONLY FAMILIES (cutoff=0.80)")
    print("=" * 80)
    sem5 = decoded_df[decoded_df["level"] == "semantic_only"].head(5)
    for _, r in sem5.iterrows():
        print(
            f"\n  fam {int(r['family_id']):>2}  #frozensets={int(r['n_frozensets']):>2}  "
            f"#paths={int(r['n_paths_total']):>2}  #R-I_pairs={int(r['n_RI_pairs_total']):>2}"
        )
        for lbl, _, nm in decode_signature(r["core_signature"], "semantic_only"):
            print(f"    {lbl:<10} {nm[:90]}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
