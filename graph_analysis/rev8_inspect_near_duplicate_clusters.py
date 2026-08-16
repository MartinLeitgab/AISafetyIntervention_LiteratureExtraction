"""
rev8_inspect_near_duplicate_clusters.py

Inspect the top near-duplicate cluster pairs (centroid sim >= 0.85) and show
their actual member names. Diagnose whether these are:
 (a) truly different concepts that happen to embed similarly, or
 (b) the same concept fragmented by HDBSCAN density separation, or
 (c) the same concept assigned different LLM subtype labels.
"""

import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"

PKL_FILE = STEP1 / "cluster_memberships_rev8_sim0.9.pkl"
ATTR_FILE = STEP1 / "graph_node_attributes.pkl"

BODY_SUBTYPES = {
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
}
SUBTYPE_PREFIX = {
    "problem_analysis": "pa",
    "theoretical_insight": "ti",
    "design_rationale": "dr",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
}

# Pairs to inspect (from previous diagnostic, top near-duplicates)
PAIRS = [
    (("problem_analysis", "136"), ("problem_analysis", "64")),
    (("problem_analysis", "1"), ("theoretical_insight", "1")),
    (("theoretical_insight", "31"), ("design_rationale", "53")),
    (("theoretical_insight", "40"), ("design_rationale", "69")),
    (("design_rationale", "69"), ("implementation_mechanism", "70")),
    (("theoretical_insight", "17"), ("design_rationale", "67")),
    (("problem_analysis", "107"), ("problem_analysis", "151")),
]


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
    print("=" * 70)
    print("rev8 — inspect near-duplicate cluster pairs")
    print("=" * 70)

    with open(PKL_FILE, "rb") as f:
        cm = pickle.load(f)
    with open(ATTR_FILE, "rb") as f:
        node_attrs = pickle.load(f)

    cluster_members = defaultdict(list)
    for key, members in cm.items():
        subtype = str(key[2])
        cid = str(key[4])
        if cid in {"-1", "noise"}:
            continue
        if subtype in BODY_SUBTYPES:
            cluster_members[(subtype, cid)] = list(members)

    for cl1, cl2 in PAIRS:
        n1 = f"{SUBTYPE_PREFIX[cl1[0]]}:{cl1[1]}"
        n2 = f"{SUBTYPE_PREFIX[cl2[0]]}:{cl2[1]}"
        m1 = cluster_members.get(cl1, [])
        m2 = cluster_members.get(cl2, [])
        # centroid sim
        embs1 = [parse_emb(node_attrs[int(n)].get("embedding")) for n in m1]
        embs1 = [e for e in embs1 if e is not None]
        embs2 = [parse_emb(node_attrs[int(n)].get("embedding")) for n in m2]
        embs2 = [e for e in embs2 if e is not None]
        c1 = np.mean(embs1, axis=0)
        c1 /= max(np.linalg.norm(c1), 1e-9)
        c2 = np.mean(embs2, axis=0)
        c2 /= max(np.linalg.norm(c2), 1e-9)
        cent_sim = float(c1 @ c2)

        # within-cluster member sim distribution
        def within_sims(embs, cent):
            return np.array(
                [float(e @ cent / (np.linalg.norm(e) + 1e-9)) for e in embs]
            )

        w1 = within_sims(embs1, c1)
        w2 = within_sims(embs2, c2)

        # cross-cluster member sims (each member of cl1 vs centroid of cl2 etc.)
        cross_1to2 = within_sims(embs1, c2)
        cross_2to1 = within_sims(embs2, c1)

        # LLM-extracted subtype labels of members (cross-check labeling consistency)
        labs1 = Counter(
            str(node_attrs[int(n)].get("type", "?"))
            + "/"
            + str(node_attrs[int(n)].get("concept_category", "?"))
            for n in m1
        )
        labs2 = Counter(
            str(node_attrs[int(n)].get("type", "?"))
            + "/"
            + str(node_attrs[int(n)].get("concept_category", "?"))
            for n in m2
        )

        print(f"\n{'=' * 70}")
        print(
            f"PAIR: {n1} (size {len(m1)}) vs {n2} (size {len(m2)})  centroid_sim={cent_sim:.4f}"
        )
        print(f"{'=' * 70}")
        print(
            f"  {n1} member sim-to-own-centroid: min={w1.min():.3f} mean={w1.mean():.3f} max={w1.max():.3f}"
        )
        print(
            f"  {n1} member sim-to-{n2}-centroid: min={cross_1to2.min():.3f} mean={cross_1to2.mean():.3f} max={cross_1to2.max():.3f}"
        )
        print(
            f"  {n2} member sim-to-own-centroid: min={w2.min():.3f} mean={w2.mean():.3f} max={w2.max():.3f}"
        )
        print(
            f"  {n2} member sim-to-{n1}-centroid: min={cross_2to1.min():.3f} mean={cross_2to1.mean():.3f} max={cross_2to1.max():.3f}"
        )
        print(f"  {n1} subtype labels of members: {dict(labs1)}")
        print(f"  {n2} subtype labels of members: {dict(labs2)}")
        print(f"\n  Sample {n1} member names (up to 8):")
        for nid in m1[:8]:
            nm = node_attrs[int(nid)].get("name", "")
            print(f"    [{nid}] {str(nm)[:90]}")
        print(f"\n  Sample {n2} member names (up to 8):")
        for nid in m2[:8]:
            nm = node_attrs[int(nid)].get("name", "")
            print(f"    [{nid}] {str(nm)[:90]}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
